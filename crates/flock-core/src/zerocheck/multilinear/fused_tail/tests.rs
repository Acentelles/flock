use super::aarch64::fold_and_message_fused;
use super::*;
use crate::challenger::{Challenger, FsChallenger};
use crate::field::F256Unreduced;
use crate::zerocheck::multilinear::fold_and_compute_round_pair_into;
use crate::zerocheck::univariate_skip::SplitEqGhash;

fn reference(a: &[F128], b: &[F128], r: F128, eq: &[F128]) -> (Vec<F128>, Vec<F128>, (F128, F128)) {
    let mut af = vec![F128::ZERO; a.len() / 2];
    let mut bf = vec![F128::ZERO; b.len() / 2];
    crate::field::f128_slice::fold_pairs(a, 0, &mut af, r);
    crate::field::f128_slice::fold_pairs(b, 0, &mut bf, r);
    let mut p1 = F256Unreduced::ZERO;
    let mut pinf = F256Unreduced::ZERO;
    for (pair, &weight) in eq.iter().enumerate() {
        let i = 2 * pair;
        p1 ^= weight.mul_unreduced(af[i + 1] * bf[i + 1]);
        pinf ^= weight.mul_unreduced((af[i] + af[i + 1]) * (bf[i] + bf[i + 1]));
    }
    (af, bf, (p1.reduce(), pinf.reduce()))
}

#[test]
fn tail_fused_arbitrary_fields_odd_pairs_and_offsets_match_reference() {
    let mut rng = FsChallenger::new(b"tail-fused-arbitrary-v1");
    for pairs in [0, 1, 2, 3, 7, 16, 31, 127, 129, 1023, 1024, 1025] {
        for offset in [0, 1, 3] {
            let a = rng.sample_f128_vec(4 * pairs + offset + 1);
            let b = rng.sample_f128_vec(a.len());
            let a = &a[offset..offset + 4 * pairs];
            let b = &b[offset..offset + 4 * pairs];
            let eq = rng.sample_f128_vec(pairs);
            for r in [F128::ZERO, F128::ONE, rng.sample_f128()] {
                let expected = reference(a, b, r, &eq);
                let mut af = vec![F128::ONE; 2 * pairs];
                let mut bf = vec![F128::ONE; 2 * pairs];
                let sums = fold_and_message_fused(a, b, &mut af, &mut bf, r, &eq);
                assert_eq!((af.clone(), bf.clone(), sums), expected);
                // Independent fold formula checks both endpoints and every lane.
                for i in 0..af.len() {
                    assert_eq!(af[i], (F128::ONE + r) * a[2 * i] + r * a[2 * i + 1]);
                    assert_eq!(bf[i], (F128::ONE + r) * b[2 * i] + r * b[2 * i + 1]);
                }
            }
        }
    }
}

#[test]
fn tail_fused_chunk_boundaries_preserve_complete_round_message() {
    let mut rng = FsChallenger::new(b"tail-fused-round-v1");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for log_n in [10usize, 11, 12, 15] {
        let a = rng.sample_f128_vec(1 << log_n);
        let b = rng.sample_f128_vec(a.len());
        let r = rng.sample_f128();
        let next = rng.sample_f128_vec(log_n - 1);
        let mut expected_a = vec![F128::ZERO; a.len() / 2];
        let mut expected_b = vec![F128::ZERO; b.len() / 2];
        let expected =
            fold_and_compute_round_pair_into(&a, &b, &mut expected_a, &mut expected_b, r, &next);
        let mut af = vec![F128::ONE; a.len() / 2];
        let mut bf = vec![F128::ONE; b.len() / 2];
        let candidate = pool.install(|| {
            fold_and_compute_round_pair_fused_tail_into(&a, &b, &mut af, &mut bf, r, &next)
        });
        assert_eq!(
            (af, bf, candidate),
            (expected_a.clone(), expected_b.clone(), expected)
        );
        let eq = SplitEqGhash::new(&next[1..]);
        let lo = eq.lo.len();
        // Split each original outer tile at awkward pair boundaries to expose
        // dropped tails or state shared between output chunks.
        for chunk_pairs in [1, 3, 7, lo] {
            let mut af = vec![F128::ONE; a.len() / 2];
            let mut bf = vec![F128::ONE; b.len() / 2];
            let mut total = (F128::ZERO, F128::ZERO);
            for (hi, &weight) in eq.hi.iter().enumerate() {
                let mut local = (F128::ZERO, F128::ZERO);
                for start in (0..lo).step_by(chunk_pairs) {
                    let end = (start + chunk_pairs).min(lo);
                    let global_start = hi * lo + start;
                    let global_end = hi * lo + end;
                    let sums = fold_and_message_fused(
                        &a[4 * global_start..4 * global_end],
                        &b[4 * global_start..4 * global_end],
                        &mut af[2 * global_start..2 * global_end],
                        &mut bf[2 * global_start..2 * global_end],
                        r,
                        &eq.lo[start..end],
                    );
                    local.0 += sums.0;
                    local.1 += sums.1;
                }
                total.0 += weight * local.0;
                total.1 += weight * local.1;
            }
            assert_eq!((af, bf), (expected_a.clone(), expected_b.clone()));
            assert_eq!((next[0] * total.0, total.1), expected);
        }
    }
}

#[test]
fn tail_fused_high_bits_and_cancellation_match_reference() {
    let full = F128::new(u64::MAX, u64::MAX);
    for bit in 0..128 {
        let basis = if bit < 64 {
            F128::new(1 << bit, 0)
        } else {
            F128::new(0, 1 << (bit - 64))
        };
        let a = [basis, full, F128::ONE, basis, basis, full, F128::ONE, basis];
        let b = [
            full,
            basis,
            basis,
            F128::ZERO,
            full,
            basis,
            basis,
            F128::ZERO,
        ];
        let eq = [full, full];
        let mut af = [F128::ZERO; 4];
        let mut bf = [F128::ZERO; 4];
        let actual = fold_and_message_fused(&a, &b, &mut af, &mut bf, basis, &eq);
        assert_eq!(actual, (F128::ZERO, F128::ZERO));
        assert_eq!(
            (af.to_vec(), bf.to_vec(), actual),
            reference(&a, &b, basis, &eq)
        );
    }
}
