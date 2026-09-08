//! Isolated implementation choice for the existing multilinear tail message.
use crate::field::F128;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

/// Same output arrays and message as `fold_and_compute_round_pair_into`.
/// The ARM kernel consumes folded values before storing them; all challenges,
/// eq weights, padding behavior and outer reductions remain caller-owned.
pub fn fold_and_compute_round_pair_fused_tail_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        super::fold_and_compute_round_pair_into(a, b, a_out, b_out, r_fold, r_next)
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        use crate::zerocheck::univariate_skip::SplitEqGhash;
        use rayon::prelude::*;

        let n = a.len();
        assert_eq!(b.len(), n);
        assert!(n.is_power_of_two() && n >= 8);
        assert_eq!(a_out.len(), n / 2);
        assert_eq!(b_out.len(), n / 2);
        assert_eq!(r_next.len(), n.trailing_zeros() as usize - 1);
        let eq = SplitEqGhash::new(&r_next[1..]);
        let lo_size = eq.lo.len();
        assert!(lo_size >= 2);
        assert_eq!(lo_size * eq.hi.len() * 2, n / 2);
        let (sum1, sum_inf) = a_out
            .par_chunks_mut(2 * lo_size)
            .zip(b_out.par_chunks_mut(2 * lo_size))
            .enumerate()
            .map(|(hi, (af, bf))| {
                let start = hi * 4 * lo_size;
                let end = start + 4 * lo_size;
                let (p1, pinf) = aarch64::fold_and_message_fused(
                    &a[start..end],
                    &b[start..end],
                    af,
                    bf,
                    r_fold,
                    &eq.lo,
                );
                (eq.hi[hi] * p1, eq.hi[hi] * pinf)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a1, ai), (b1, bi)| (a1 + b1, ai + bi),
            );
        (r_next[0] * sum1, sum_inf)
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "aes"))]
mod tests;
