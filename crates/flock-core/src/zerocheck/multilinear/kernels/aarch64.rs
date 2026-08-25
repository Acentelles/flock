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
        macro_rules! row {
            ($j:expr) => {
                vld1q_u8(table_data.add($j * STRIDE + ((w >> (8 * $j)) & 0xff) as usize * 16))
            };
        }
        let t0 = vld1q_u8(table_data.add((w & 0xff) as usize * 16));
        let t1 = row!(1);
        let t2 = row!(2);
        let t3 = row!(3);
        let t4 = row!(4);
        let t5 = row!(5);
        let t6 = row!(6);
        let t7 = row!(7);
        // Tree reduction instead of a serial 7-XOR chain: the round-2 loop is
        // latency-bound (measured: manual unrolling regresses, address loads
        // are latency-hidden), so the row's critical path is what matters.
        // Depth drops from 7 XORs to 2 EOR3 levels at the same op count.
        #[cfg(target_feature = "sha3")]
        {
            veor3q_u8(
                veor3q_u8(t0, t1, t2),
                veor3q_u8(t3, t4, t5),
                veorq_u8(t6, t7),
            )
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            veorq_u8(
                veorq_u8(veorq_u8(t0, t1), veorq_u8(t2, t3)),
                veorq_u8(veorq_u8(t4, t5), veorq_u8(t6, t7)),
            )
        }
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
