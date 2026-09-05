//! Exact coefficient accumulation for the thirteen-factor scatter.
use flock_core::field::F128;

#[inline]
fn mul22(a: [F128; 2], b: [F128; 2]) -> [F128; 3] {
    let p = a[0] * b[0];
    let q = a[1] * b[1];
    [p, (a[0] + a[1]) * (b[0] + b[1]) + p + q, q]
}

#[inline]
fn mul32(a: [F128; 3], b: [F128; 2]) -> [F128; 4] {
    let p = a[0] * b[0];
    let q0 = a[1] * b[1];
    let q1 = a[2] * b[1];
    let s = b[0] + b[1];
    let z0 = (a[0] + a[1]) * s;
    let z1 = a[2] * s;
    [p, z0 + p + q0, z1 + q1 + q0, q1]
}

#[inline]
fn mul33(a: [F128; 3], b: [F128; 3]) -> [F128; 5] {
    let p0 = a[0] * b[0];
    let p1 = a[1] * b[1];
    let p2 = a[2] * b[2];
    [
        p0,
        (a[0] + a[1]) * (b[0] + b[1]) + p0 + p1,
        (a[0] + a[2]) * (b[0] + b[2]) + p0 + p1 + p2,
        (a[1] + a[2]) * (b[1] + b[2]) + p1 + p2,
        p2,
    ]
}

#[inline]
fn combine<const O: usize>(lo: &[F128], hi: &[F128], cross: &[F128], k: usize) -> [F128; O] {
    let mut out = [F128::ZERO; O];
    for (i, &v) in lo.iter().enumerate() {
        out[i] += v;
        out[i + k] += v;
    }
    for (i, &v) in hi.iter().enumerate() {
        out[i + 2 * k] += v;
        out[i + k] += v;
    }
    for (i, &v) in cross.iter().enumerate() {
        out[i + k] += v;
    }
    out
}

#[inline]
fn mul44(a: [F128; 4], b: [F128; 4]) -> [F128; 7] {
    combine(
        &mul22([a[0], a[1]], [b[0], b[1]]),
        &mul22([a[2], a[3]], [b[2], b[3]]),
        &mul22([a[0] + a[2], a[1] + a[3]], [b[0] + b[2], b[1] + b[3]]),
        2,
    )
}

#[inline]
fn mul54(a: [F128; 5], b: [F128; 4]) -> [F128; 8] {
    combine(
        &mul22([a[0], a[1]], [b[0], b[1]]),
        &mul32([a[2], a[3], a[4]], [b[2], b[3]]),
        &mul32([a[0] + a[2], a[1] + a[3], a[4]], [b[0] + b[2], b[1] + b[3]]),
        2,
    )
}

#[inline]
fn mul55(a: [F128; 5], b: [F128; 5]) -> [F128; 9] {
    combine(
        &mul22([a[0], a[1]], [b[0], b[1]]),
        &mul33([a[2], a[3], a[4]], [b[2], b[3], b[4]]),
        &mul33(
            [a[0] + a[2], a[1] + a[3], a[4]],
            [b[0] + b[2], b[1] + b[3], b[4]],
        ),
        2,
    )
}

#[inline]
fn mul95(a: [F128; 9], b: [F128; 5]) -> [F128; 13] {
    let hi = [
        a[4] * b[4],
        a[5] * b[4],
        a[6] * b[4],
        a[7] * b[4],
        a[8] * b[4],
    ];
    combine(
        &mul44([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]),
        &hi,
        &mul54(
            [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7], a[8]],
            [b[0] + b[4], b[1], b[2], b[3]],
        ),
        4,
    )
}

/// Coefficients of thirteen affine factors, in ascending monomial order.
/// Karatsuba products are exact in characteristic two. No interpolation
/// challenges or field encodings are changed.
#[inline]
fn product13_full(a: &[[F128; 2]; 13]) -> [F128; 14] {
    let q0 = mul33(mul22(a[0], a[1]), mul22(a[2], a[3]));
    let q1 = mul33(mul22(a[4], a[5]), mul22(a[6], a[7]));
    let q2 = mul33(mul22(a[8], a[9]), mul22(a[10], a[11]));
    let p = mul95(mul55(q0, q1), q2);
    let mut out = [F128::ZERO; 14];
    for i in 0..13 {
        out[i] += p[i] * a[12][0];
        out[i + 1] += p[i] * a[12][1];
    }
    out
}

#[inline]
fn four(a: &[[F128; 2]]) -> [F128; 5] {
    mul33(mul22(a[0], a[1]), mul22(a[2], a[3]))
}

#[inline]
fn eight(a: &[[F128; 2]]) -> [F128; 9] {
    mul55(four(a), four(&a[4..]))
}

#[inline]
fn naive_into(a: &[F128], b: &[F128], out: &mut [F128]) {
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
}

/// Strip constant factors before selecting a smaller polynomial product.
/// Coefficients above the actual degree remain explicitly zero.
#[inline]
pub(crate) fn product13(a: &[[F128; 2]; 13]) -> [F128; 14] {
    let mut variable = [[F128::ZERO; 2]; 13];
    let mut count = 0;
    let mut constant = F128::ONE;
    for &factor in a {
        if factor[1] == F128::ZERO {
            if factor[0] == F128::ZERO {
                return [F128::ZERO; 14];
            }
            if factor[0] != F128::ONE {
                constant *= factor[0];
            }
        } else {
            variable[count] = factor;
            count += 1;
        }
    }
    if count == 13 {
        return product13_full(a);
    }
    let a = &variable;
    let mut out = [F128::ZERO; 14];
    match count {
        0 => out[0] = F128::ONE,
        1 => out[..2].copy_from_slice(&a[0]),
        2 => out[..3].copy_from_slice(&mul22(a[0], a[1])),
        3 => out[..4].copy_from_slice(&mul32(mul22(a[0], a[1]), a[2])),
        4 => out[..5].copy_from_slice(&four(a)),
        5 => naive_into(&four(a), &a[4], &mut out),
        6 => naive_into(&four(a), &mul22(a[4], a[5]), &mut out),
        7 => out[..8].copy_from_slice(&mul54(four(a), mul32(mul22(a[4], a[5]), a[6]))),
        8 => out[..9].copy_from_slice(&eight(a)),
        9 => naive_into(&eight(a), &a[8], &mut out),
        10 => naive_into(&eight(a), &mul22(a[8], a[9]), &mut out),
        11 => naive_into(&eight(a), &mul32(mul22(a[8], a[9]), a[10]), &mut out),
        12 => out[..13].copy_from_slice(&mul95(eight(a), four(&a[8..]))),
        _ => unreachable!(),
    }
    if constant != F128::ONE {
        for c in &mut out[..=count] {
            *c *= constant;
        }
    }
    out
}

pub(super) fn round(factors: &[Vec<F128>]) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(factors.len(), 13);
    let half = factors[0].len() / 2;
    let coeffs = (0..half)
        .into_par_iter()
        .fold(
            || [F128::ZERO; 14],
            |mut acc, i| {
                let mut affine = [[F128::ZERO; 2]; 13];
                for (a, factor) in affine.iter_mut().zip(factors) {
                    let lo = factor[i];
                    let hi = factor[half + i];
                    if lo == F128::ZERO && hi == F128::ZERO {
                        return acc;
                    }
                    *a = [lo, lo + hi];
                }
                for (a, p) in acc.iter_mut().zip(product13_full(&affine)) {
                    *a += p;
                }
                acc
            },
        )
        .reduce(
            || [F128::ZERO; 14],
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += b;
                }
                a
            },
        );
    (0..14)
        .map(|i| {
            let x = F128 { lo: i, hi: 0 };
            coeffs.iter().rev().fold(F128::ZERO, |acc, &c| acc * x + c)
        })
        .collect()
}
