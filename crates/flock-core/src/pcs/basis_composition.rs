//! Compose a fixed field multiplication with the bit-linear ring-switch map.
//! This changes only basis assembly; every resulting field element is exact.

use super::ring_switch::fold_one_slot;
use crate::field::{F128, mul_by_x};

// Native paired measurements cover block sizes 4096 and 8192. Smaller blocks
// keep their existing arithmetic because table setup can outweigh savings.
pub(super) const MIN_BLOCK_CELLS: usize = 4096;

pub(super) fn compose_and_fold(
    low: &[F128],
    high: F128,
    original: &[F128],
    output: &mut [F128],
    composed: &mut [F128],
    add: bool,
) {
    debug_assert_eq!(low.len(), output.len());
    compose_table(high, original, composed);
    if add {
        for (slot, &value) in output.iter_mut().zip(low) {
            *slot += fold_one_slot(value, composed);
        }
    } else {
        for (slot, &value) in output.iter_mut().zip(low) {
            *slot = fold_one_slot(value, composed);
        }
    }
}

// For the existing GF(2)-linear map T, construct U(e_i) = T(high * e_i).
// Then U(x) = T(high * x) for every field element x, including high = 0.
// This uses GF(2)-linearity only; T need not be GF(2^128)-linear.
fn compose_table(high: F128, original: &[F128], composed: &mut [F128]) {
    assert_eq!(composed.len(), 4096);
    let mut basis = high;
    for byte in composed.chunks_exact_mut(256) {
        byte[0] = F128::ZERO;
        for bit in 0..8 {
            let image = fold_one_slot(basis, original);
            basis = mul_by_x(basis);
            let half = 1 << bit;
            for value in 0..half {
                byte[half + value] = byte[value] + image;
            }
        }
    }
}
