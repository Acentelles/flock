use crate::field::F128;

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

// ---------------------------------------------------------------------------
// Interleaved-path kernels (the PCS-commit NTT). Previously these dispatched
// to the portable scalar butterflies on aarch64 -- per lane, a binius multiply
// (6 PMULLs) plus full struct/GPR round trips. Here: q-resident throughout,
// and because the twiddle is constant per call, Karatsuba with a precomputed
// half-sum needs 3 PMULLs per multiply, with the vec2-style vectorized
// reduction shared across each lane pair.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull_local(a: u64, b: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    // SAFETY: aes target feature carried by the attribute.
    unsafe { core::mem::transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
}

/// Multiply two lanes (q registers) by one broadcast twiddle given as
/// precomputed halves `(t_lo, t_hi, t_lo ^ t_hi)`: 3 PMULLs per lane,
/// lane-paired vectorized reduction, q-resident in and out.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn mul_tw2(
    t_lo: u64,
    t_hi: u64,
    t_sum: u64,
    v0: core::arch::aarch64::uint64x2_t,
    v1: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    // SAFETY: caller is aes-gated.
    unsafe {
        let v0_lo = vgetq_lane_u64::<0>(v0);
        let v0_hi = vgetq_lane_u64::<1>(v0);
        let v1_lo = vgetq_lane_u64::<0>(v1);
        let v1_hi = vgetq_lane_u64::<1>(v1);
        // Karatsuba per lane: ll, hh, and one product of sums.
        let ll0 = pmull_local(t_lo, v0_lo);
        let hh0 = pmull_local(t_hi, v0_hi);
        let mid0 = pmull_local(t_sum, v0_lo ^ v0_hi);
        let c0 = veorq_u64(veorq_u64(mid0, ll0), hh0);
        let ll1 = pmull_local(t_lo, v1_lo);
        let hh1 = pmull_local(t_hi, v1_hi);
        let mid1 = pmull_local(t_sum, v1_lo ^ v1_hi);
        let c1 = veorq_u64(veorq_u64(mid1, ll1), hh1);

        // Lane-paired words, then the vec2 vectorized reduction.
        let r0 = vzip1q_u64(ll0, ll1);
        let r1 = veorq_u64(vzip2q_u64(ll0, ll1), vzip1q_u64(c0, c1));
        let r2 = veorq_u64(vzip1q_u64(hh0, hh1), vzip2q_u64(c0, c1));
        let r3 = vzip2q_u64(hh0, hh1);

        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo2 = veorq_u64(veorq_u64(r2, s1_lo), veorq_u64(s2_lo, s7_lo));
        let t_hi2 = veorq_u64(veorq_u64(r3, s1_hi), veorq_u64(s2_hi, s7_hi));
        let ov = veorq_u64(
            veorq_u64(vshrq_n_u64::<63>(r3), vshrq_n_u64::<62>(r3)),
            vshrq_n_u64::<57>(r3),
        );
        let corr = veorq_u64(
            veorq_u64(ov, vshlq_n_u64::<1>(ov)),
            veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
        );
        let final_lo = veorq_u64(veorq_u64(r0, t_lo2), corr);
        let final_hi = veorq_u64(r1, t_hi2);

        (
            vzip1q_u64(final_lo, final_hi),
            vzip2q_u64(final_lo, final_hi),
        )
    }
}

/// q-resident broadcast-twiddle row butterfly:
/// `top[l] = top[l] + t*bot[l]; bot[l] += top[l]` across all lanes.
///
/// # Safety
/// Requires the `aes` target feature; `top.len() == bot.len()`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_row_pair_neon(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    use core::arch::aarch64::*;
    debug_assert_eq!(top.len(), bot.len());
    let (t_lo, t_hi, t_sum) = (twiddle.lo, twiddle.hi, twiddle.lo ^ twiddle.hi);
    let n = top.len();
    // SAFETY: F128 is repr(C, align(16)); pointers stay within the slices.
    unsafe {
        let tp = top.as_mut_ptr() as *mut u64;
        let bp = bot.as_mut_ptr() as *mut u64;
        let mut l = 0usize;
        while l + 2 <= n {
            let v0 = vld1q_u64(bp.add(2 * l));
            let v1 = vld1q_u64(bp.add(2 * l + 2));
            let u0 = vld1q_u64(tp.add(2 * l));
            let u1 = vld1q_u64(tp.add(2 * l + 2));
            let (p0, p1) = mul_tw2(t_lo, t_hi, t_sum, v0, v1);
            let nu0 = veorq_u64(u0, p0);
            let nu1 = veorq_u64(u1, p1);
            vst1q_u64(tp.add(2 * l), nu0);
            vst1q_u64(tp.add(2 * l + 2), nu1);
            vst1q_u64(bp.add(2 * l), veorq_u64(v0, nu0));
            vst1q_u64(bp.add(2 * l + 2), veorq_u64(v1, nu1));
            l += 2;
        }
        while l < n {
            let v = bot[l];
            let new_u = top[l] + v * twiddle;
            top[l] = new_u;
            bot[l] = v + new_u;
            l += 1;
        }
    }
}

/// q-resident fused two-layer butterfly (same math as
/// `portable::butterfly_fused_2layer`), two lanes per iteration, all three
/// twiddles' half-sums hoisted.
///
/// # Safety
/// Requires the `aes` target feature; all four rows the same length.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_2layer_neon(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::aarch64::*;
    let n = a.len();
    let (to_l, to_h, to_s) = (t_outer.lo, t_outer.hi, t_outer.lo ^ t_outer.hi);
    let (ta_l, ta_h, ta_s) = (t_inner_a.lo, t_inner_a.hi, t_inner_a.lo ^ t_inner_a.hi);
    let (tb_l, tb_h, tb_s) = (t_inner_b.lo, t_inner_b.hi, t_inner_b.lo ^ t_inner_b.hi);
    // SAFETY: F128 is repr(C, align(16)); pointers stay within the slices.
    unsafe {
        let ap = a.as_mut_ptr() as *mut u64;
        let bp = b.as_mut_ptr() as *mut u64;
        let cp = c.as_mut_ptr() as *mut u64;
        let dp = d.as_mut_ptr() as *mut u64;
        let mut l = 0usize;
        while l + 2 <= n {
            let xa0 = vld1q_u64(ap.add(2 * l));
            let xa1 = vld1q_u64(ap.add(2 * l + 2));
            let xb0 = vld1q_u64(bp.add(2 * l));
            let xb1 = vld1q_u64(bp.add(2 * l + 2));
            let xc0 = vld1q_u64(cp.add(2 * l));
            let xc1 = vld1q_u64(cp.add(2 * l + 2));
            let xd0 = vld1q_u64(dp.add(2 * l));
            let xd1 = vld1q_u64(dp.add(2 * l + 2));

            // Layer L: (a,c) and (b,d) at t_outer.
            let (p0, p1) = mul_tw2(to_l, to_h, to_s, xc0, xc1);
            let na0 = veorq_u64(xa0, p0);
            let na1 = veorq_u64(xa1, p1);
            let nc0 = veorq_u64(xc0, na0);
            let nc1 = veorq_u64(xc1, na1);
            let (q0, q1) = mul_tw2(to_l, to_h, to_s, xd0, xd1);
            let nb0 = veorq_u64(xb0, q0);
            let nb1 = veorq_u64(xb1, q1);
            let nd0 = veorq_u64(xd0, nb0);
            let nd1 = veorq_u64(xd1, nb1);

            // Layer L+1: (a,b) at t_inner_a; (c,d) at t_inner_b.
            let (r0, r1) = mul_tw2(ta_l, ta_h, ta_s, nb0, nb1);
            let na2_0 = veorq_u64(na0, r0);
            let na2_1 = veorq_u64(na1, r1);
            let nb2_0 = veorq_u64(nb0, na2_0);
            let nb2_1 = veorq_u64(nb1, na2_1);
            let (s0, s1) = mul_tw2(tb_l, tb_h, tb_s, nd0, nd1);
            let nc2_0 = veorq_u64(nc0, s0);
            let nc2_1 = veorq_u64(nc1, s1);
            let nd2_0 = veorq_u64(nd0, nc2_0);
            let nd2_1 = veorq_u64(nd1, nc2_1);

            vst1q_u64(ap.add(2 * l), na2_0);
            vst1q_u64(ap.add(2 * l + 2), na2_1);
            vst1q_u64(bp.add(2 * l), nb2_0);
            vst1q_u64(bp.add(2 * l + 2), nb2_1);
            vst1q_u64(cp.add(2 * l), nc2_0);
            vst1q_u64(cp.add(2 * l + 2), nc2_1);
            vst1q_u64(dp.add(2 * l), nd2_0);
            vst1q_u64(dp.add(2 * l + 2), nd2_1);
            l += 2;
        }
        while l < n {
            let xa = a[l];
            let xb = b[l];
            let xc = c[l];
            let xd = d[l];
            let na = xa + xc * t_outer;
            let nc = xc + na;
            let nb = xb + xd * t_outer;
            let nd = xd + nb;
            let na2 = na + nb * t_inner_a;
            let nb2 = nb + na2;
            let nc2 = nc + nd * t_inner_b;
            let nd2 = nd + nc2;
            a[l] = na2;
            b[l] = nb2;
            c[l] = nc2;
            d[l] = nd2;
            l += 1;
        }
    }
}
