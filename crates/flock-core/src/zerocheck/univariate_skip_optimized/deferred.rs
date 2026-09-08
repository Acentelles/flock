//! Exact worker-local reduction deferral; no transcript or field change.
use super::{ELL, kernels};
use crate::field::{F128, F256Unreduced};

pub(super) trait PartialAccumulator: Copy + Send {
    const ZERO: Self;
    fn reduced(self) -> F128;
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        ab: &[[u8; 64]; 16],
        c: &[[u8; 64]; 16],
        n: usize,
        convert: &[F128],
        weight: F128,
        partial_ab: &mut [Self; ELL],
        c0: &mut [Self; ELL],
        c1: &mut [Self; ELL],
    );
}
impl PartialAccumulator for F128 {
    const ZERO: Self = F128::ZERO;
    #[inline]
    fn reduced(self) -> F128 {
        self
    }
    #[inline]
    fn accumulate(
        ab: &[[u8; 64]; 16],
        c: &[[u8; 64]; 16],
        n: usize,
        convert: &[F128],
        weight: F128,
        partial_ab: &mut [Self; ELL],
        c0: &mut [Self; ELL],
        c1: &mut [Self; ELL],
    ) {
        kernels::accumulate_convert_with_s_hat_v(ab, c, n, convert, weight, partial_ab, c0, c1);
    }
}
impl PartialAccumulator for F256Unreduced {
    const ZERO: Self = F256Unreduced::ZERO;
    #[inline]
    fn reduced(self) -> F128 {
        self.reduce()
    }
    #[inline]
    fn accumulate(
        ab: &[[u8; 64]; 16],
        c: &[[u8; 64]; 16],
        n: usize,
        convert: &[F128],
        weight: F128,
        partial_ab: &mut [Self; ELL],
        c0: &mut [Self; ELL],
        c1: &mut [Self; ELL],
    ) {
        assert!(n <= 16);
        assert!(convert.len() >= n * 256);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: arrays cover all lanes and the table bound is checked above.
        unsafe {
            kernels::aarch64::accumulate_convert_deferred(
                ab, c, n, convert, weight, partial_ab, c0, c1,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        for lane in 0..ELL {
            let mut values = [F128::ZERO; 3];
            for medium in 0..n {
                let base = medium * 256;
                let byte = c[medium][lane] as usize;
                values[0] += convert[base + ab[medium][lane] as usize];
                values[1] += convert[base + (byte & 0x55)];
                values[2] += convert[base + (byte & 0xaa)];
            }
            partial_ab[lane] ^= values[0].mul_unreduced(weight);
            c0[lane] ^= values[1].mul_unreduced(weight);
            c1[lane] ^= values[2].mul_unreduced(weight);
        }
    }
}

#[cfg(test)]
mod tests;
