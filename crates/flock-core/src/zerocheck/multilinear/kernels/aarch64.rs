use crate::field::F128;

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
/// Fused fold-and-message pass for the rounds-3+ tail, entirely in q
/// registers. The incumbent shape made two passes per worker chunk -- fold
/// `a_in`/`b_in` into `a_out`/`b_out`, then RE-READ the multi-megabyte output
/// chunk to build the message, shuttling every value through F128 structs in
/// general registers on the way. Here each output pair is folded, stored once
/// (the required write), and consumed for the message while still in vector
/// registers: same PMULL count, one memory pass instead of two, no
/// register-file boundary crossings.
///
/// Contract matches the generic branch of `fold_and_compute_round_pair_into`:
/// `a_in.len() == 4 * eq_lo.len()`, `a_out.len() == 2 * eq_lo.len()`, fold at
/// `r_fold` (out = even + r * (even + odd)), returns the REDUCED
/// `(sum eq*g1, sum eq*g_inf)` for the caller's `eq_hi` fold.
///
/// # Safety
/// Requires the `aes` target feature (PMULL); slices per the contract above.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_and_message_neon(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::{WideNeon, mul_q, wide_mul_unreduced_q};
    use core::arch::aarch64::*;

    let lo_size = eq_lo.len();
    debug_assert_eq!(a_in.len(), 4 * lo_size);
    debug_assert_eq!(a_out.len(), 2 * lo_size);

    // SAFETY: F128 is repr(C, align(16)); all offsets below stay within the
    // slice lengths asserted above.
    unsafe {
        let r_arr = [r_fold.lo, r_fold.hi];
        let r_q = vld1q_u64(r_arr.as_ptr());
        let ap = a_in.as_ptr() as *const u64;
        let bp = b_in.as_ptr() as *const u64;
        let aop = a_out.as_mut_ptr() as *mut u64;
        let bop = b_out.as_mut_ptr() as *mut u64;
        let eqp = eq_lo.as_ptr() as *const u64;

        let mut p1_nacc = WideNeon::zero();
        let mut pinf_nacc = WideNeon::zero();

        // fold: even + r * (even ^ odd), all operands resident in q registers.
        #[inline(always)]
        unsafe fn fold_pair_q(
            e: core::arch::aarch64::uint64x2_t,
            o: core::arch::aarch64::uint64x2_t,
            r: core::arch::aarch64::uint64x2_t,
        ) -> core::arch::aarch64::uint64x2_t {
            use core::arch::aarch64::*;
            unsafe { veorq_u64(e, mul_q(r, veorq_u64(e, o))) }
        }

        for x in 0..lo_size {
            let i = 4 * x;
            let a0 = fold_pair_q(vld1q_u64(ap.add(2 * i)), vld1q_u64(ap.add(2 * i + 2)), r_q);
            let a1 = fold_pair_q(vld1q_u64(ap.add(2 * i + 4)), vld1q_u64(ap.add(2 * i + 6)), r_q);
            let b0 = fold_pair_q(vld1q_u64(bp.add(2 * i)), vld1q_u64(bp.add(2 * i + 2)), r_q);
            let b1 = fold_pair_q(vld1q_u64(bp.add(2 * i + 4)), vld1q_u64(bp.add(2 * i + 6)), r_q);

            let o = 2 * x;
            vst1q_u64(aop.add(2 * o), a0);
            vst1q_u64(aop.add(2 * o + 2), a1);
            vst1q_u64(bop.add(2 * o), b0);
            vst1q_u64(bop.add(2 * o + 2), b1);

            let g1 = mul_q(a1, b1);
            let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            let eq_q = vld1q_u64(eqp.add(2 * x));
            p1_nacc.xor_assign(wide_mul_unreduced_q(eq_q, g1));
            pinf_nacc.xor_assign(wide_mul_unreduced_q(eq_q, g_inf));
        }
        (p1_nacc.reduce(), pinf_nacc.reduce())
    }
}

/// [`fold_one_row_neon_unchecked_8`] without the final lane extraction: the
/// XOR-accumulated row stays in a q register for callers that keep computing
/// on it (the round-2 message chain). Same safety contract.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_q_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        // One u64 load + in-register extracts instead of eight byte loads:
        // the bytes only feed gather addresses, so the extraction rides the
        // integer side and the freed load slots go to the table gathers.
        // Same mechanism as the round-1 prep word-extract (-5.1% there).
        let w = u64::from_le((bytes_ptr as *const u64).read_unaligned());
        let mut acc = vld1q_u8(table_data.add((w & 0xff) as usize * 16));
        for j in 1..8usize {
            acc = veorq_u8(
                acc,
                vld1q_u8(
                    table_data.add(j * STRIDE + ((w >> (8 * j)) & 0xff) as usize * 16),
                ),
            );
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// Production callers moved to the q-returning variant; this remains as the
// extraction-included form the NEON-vs-scalar test exercises.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}

/// One eq-hi chunk of a lookahead pass (fold + the 8 lookahead product sums),
/// entirely in q registers — the NEON analog of [`fold_and_message_neon`]
/// for the cascade tail. `PER_U` selects the pass width: 16 = fold TWO
/// pending variables per output (`rhos.0` then `rhos.1`, the steady-state
/// 4→1 pass), 8 = fold ONE (`rhos.0`; `rhos.1` unused, the entry pass).
///
/// Per eq-lo position `u` this folds four output values per array, stores
/// them once (the required write), and consumes them for the eight
/// lookahead products while still in vector registers — no F128 struct
/// crossings, one memory pass. Product indices match
/// [`super::super::lookahead_products`] exactly; the caller applies its
/// `eq_hi` weight to the returned reduced sums.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `a`/`b` must hold at least
/// `(base_u + lo_size) * PER_U` elements; `ao`/`bo` at least `4 * lo_size`;
/// `eq_lo` at least `lo_size`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn lookahead_chunk_neon<const PER_U: usize>(
    a: &[F128],
    b: &[F128],
    ao: &mut [F128],
    bo: &mut [F128],
    base_u: usize,
    lo_size: usize,
    rhos: (F128, F128),
    eq_lo: &[F128],
) -> [F128; 8] {
    use crate::field::gf2_128::aarch64::{WideNeon, mul_q, wide_mul_unreduced_q};
    use core::arch::aarch64::*;

    debug_assert!(PER_U == 8 || PER_U == 16);
    debug_assert!(a.len() >= (base_u + lo_size) * PER_U);
    debug_assert!(ao.len() >= 4 * lo_size);
    debug_assert_eq!(eq_lo.len(), lo_size);

    // SAFETY: F128 is repr(C, align(16)); all offsets stay within the bounds
    // asserted above.
    unsafe {
        let r1_arr = [rhos.0.lo, rhos.0.hi];
        let r1_q = vld1q_u64(r1_arr.as_ptr());
        let r2_arr = [rhos.1.lo, rhos.1.hi];
        let r2_q = vld1q_u64(r2_arr.as_ptr());
        let ap = a.as_ptr() as *const u64;
        let bp = b.as_ptr() as *const u64;
        let aop = ao.as_mut_ptr() as *mut u64;
        let bop = bo.as_mut_ptr() as *mut u64;
        let eqp = eq_lo.as_ptr() as *const u64;

        #[inline(always)]
        unsafe fn fold_pair_q(
            e: core::arch::aarch64::uint64x2_t,
            o: core::arch::aarch64::uint64x2_t,
            r: core::arch::aarch64::uint64x2_t,
        ) -> core::arch::aarch64::uint64x2_t {
            use core::arch::aarch64::*;
            unsafe { veorq_u64(e, mul_q(r, veorq_u64(e, o))) }
        }

        // One folded output value: PER_U/4 consecutive inputs starting at
        // element index `base_elem`.
        #[inline(always)]
        unsafe fn fold_group_q<const PER_U: usize>(
            p: *const u64,
            base_elem: usize,
            r1: core::arch::aarch64::uint64x2_t,
            r2: core::arch::aarch64::uint64x2_t,
        ) -> core::arch::aarch64::uint64x2_t {
            use core::arch::aarch64::*;
            unsafe {
                if PER_U == 16 {
                    let x0 = fold_pair_q(
                        vld1q_u64(p.add(2 * base_elem)),
                        vld1q_u64(p.add(2 * base_elem + 2)),
                        r1,
                    );
                    let x1 = fold_pair_q(
                        vld1q_u64(p.add(2 * base_elem + 4)),
                        vld1q_u64(p.add(2 * base_elem + 6)),
                        r1,
                    );
                    fold_pair_q(x0, x1, r2)
                } else {
                    fold_pair_q(
                        vld1q_u64(p.add(2 * base_elem)),
                        vld1q_u64(p.add(2 * base_elem + 2)),
                        r1,
                    )
                }
            }
        }

        let mut acc = [
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
            WideNeon::zero(),
        ];

        for u_lo in 0..lo_size {
            let u = base_u + u_lo;
            let step = PER_U / 4;
            let ga0 = fold_group_q::<PER_U>(ap, u * PER_U, r1_q, r2_q);
            let ga1 = fold_group_q::<PER_U>(ap, u * PER_U + step, r1_q, r2_q);
            let ga2 = fold_group_q::<PER_U>(ap, u * PER_U + 2 * step, r1_q, r2_q);
            let ga3 = fold_group_q::<PER_U>(ap, u * PER_U + 3 * step, r1_q, r2_q);
            let gb0 = fold_group_q::<PER_U>(bp, u * PER_U, r1_q, r2_q);
            let gb1 = fold_group_q::<PER_U>(bp, u * PER_U + step, r1_q, r2_q);
            let gb2 = fold_group_q::<PER_U>(bp, u * PER_U + 2 * step, r1_q, r2_q);
            let gb3 = fold_group_q::<PER_U>(bp, u * PER_U + 3 * step, r1_q, r2_q);

            let o = 4 * u_lo;
            vst1q_u64(aop.add(2 * o), ga0);
            vst1q_u64(aop.add(2 * o + 2), ga1);
            vst1q_u64(aop.add(2 * o + 4), ga2);
            vst1q_u64(aop.add(2 * o + 6), ga3);
            vst1q_u64(bop.add(2 * o), gb0);
            vst1q_u64(bop.add(2 * o + 2), gb1);
            vst1q_u64(bop.add(2 * o + 4), gb2);
            vst1q_u64(bop.add(2 * o + 6), gb3);

            // The 8 lookahead products, index-matched to lookahead_products:
            // ga = [g(0,0), g(1,0), g(0,1), g(1,1)] at positions [0,1,2,3].
            let sxa0 = veorq_u64(ga0, ga1);
            let sxb0 = veorq_u64(gb0, gb1);
            let sxa1 = veorq_u64(ga2, ga3);
            let sxb1 = veorq_u64(gb2, gb3);
            let dca = veorq_u64(ga0, ga2);
            let dcb = veorq_u64(gb0, gb2);
            let dsa = veorq_u64(sxa0, sxa1);
            let dsb = veorq_u64(sxb0, sxb1);

            let eq_q = vld1q_u64(eqp.add(2 * u_lo));
            acc[0].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(ga1, gb1)));
            acc[1].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(sxa0, sxb0)));
            acc[2].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(ga2, gb2)));
            acc[3].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(ga3, gb3)));
            acc[4].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(sxa1, sxb1)));
            acc[5].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(dca, dcb)));
            acc[6].xor_assign(wide_mul_unreduced_q(
                eq_q,
                mul_q(veorq_u64(dca, dsa), veorq_u64(dcb, dsb)),
            ));
            acc[7].xor_assign(wide_mul_unreduced_q(eq_q, mul_q(dsa, dsb)));
        }

        [
            acc[0].reduce(),
            acc[1].reduce(),
            acc[2].reduce(),
            acc[3].reduce(),
            acc[4].reduce(),
            acc[5].reduce(),
            acc[6].reduce(),
            acc[7].reduce(),
        ]
    }
}
