//! Folded values feed their message before being stored. Message products use
//! the existing paired NEON multiply, while eq products retain the original
//! deferred-reduction convention until the end of the chunk.
use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;
use crate::field::{F128, F256Unreduced};

#[inline(always)]
fn mul_pair(a: [F128; 2], b: [F128; 2]) -> [F128; 2] {
    // SAFETY: this module is compiled only for aarch64 with aes enabled.
    unsafe { ghash_mul_vec2_neon(a, b) }
}

#[inline(always)]
fn fold_two(source: &[F128], offset: usize, r: F128) -> [F128; 2] {
    let first = source[offset];
    let second = source[offset + 2];
    let products = mul_pair(
        [r, r],
        [first + source[offset + 1], second + source[offset + 3]],
    );
    [first + products[0], second + products[1]]
}

/// Four source values and two outputs per eq entry. Arbitrary numbers of
/// entries, including odd and zero, are accepted. Returned sums are canonical;
/// the caller applies eq_hi and the Convention A factor afterwards.
pub(super) fn fold_and_message_fused(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(4));
    assert_eq!(a.len() / 4, eq_lo.len());
    assert_eq!(a_out.len(), a.len() / 2);
    assert_eq!(b_out.len(), a_out.len());
    let mut p1 = F256Unreduced::ZERO;
    let mut pinf = F256Unreduced::ZERO;
    let mut pair = 0;
    while pair + 1 < eq_lo.len() {
        let input = 4 * pair;
        let output = 2 * pair;
        let a01 = fold_two(a, input, r);
        let b01 = fold_two(b, input, r);
        let a23 = fold_two(a, input + 4, r);
        let b23 = fold_two(b, input + 4, r);
        let g1 = mul_pair([a01[1], a23[1]], [b01[1], b23[1]]);
        let ginf = mul_pair(
            [a01[0] + a01[1], a23[0] + a23[1]],
            [b01[0] + b01[1], b23[0] + b23[1]],
        );
        p1 ^= eq_lo[pair].mul_unreduced(g1[0]);
        p1 ^= eq_lo[pair + 1].mul_unreduced(g1[1]);
        pinf ^= eq_lo[pair].mul_unreduced(ginf[0]);
        pinf ^= eq_lo[pair + 1].mul_unreduced(ginf[1]);
        a_out[output..output + 2].copy_from_slice(&a01);
        a_out[output + 2..output + 4].copy_from_slice(&a23);
        b_out[output..output + 2].copy_from_slice(&b01);
        b_out[output + 2..output + 4].copy_from_slice(&b23);
        pair += 2;
    }
    if pair < eq_lo.len() {
        let input = 4 * pair;
        let output = 2 * pair;
        let av = fold_two(a, input, r);
        let bv = fold_two(b, input, r);
        let products = mul_pair([av[1], av[0] + av[1]], [bv[1], bv[0] + bv[1]]);
        p1 ^= eq_lo[pair].mul_unreduced(products[0]);
        pinf ^= eq_lo[pair].mul_unreduced(products[1]);
        a_out[output..output + 2].copy_from_slice(&av);
        b_out[output..output + 2].copy_from_slice(&bv);
    }
    (p1.reduce(), pinf.reduce())
}
