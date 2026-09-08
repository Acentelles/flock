use super::*;
use flock_core::challenger::FsChallenger;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn packed_scatter_rows_and_complete_rounds_match_arbitrary_scalar_bits() {
    for record_vars in [2, 5] {
        for special in [false, true] {
            let mut source = FsChallenger::new(b"packed-scatter-arbitrary-source-v1");
            let words = source.sample_f128_vec(1 << (record_vars + slots::K_LOG - 7));
            let bit_at = |address: usize| {
                let w = words[address / 128];
                ((if address % 128 < 64 { w.lo } else { w.hi }) >> (address % 64)) & 1 != 0
            };
            let reads = AtomicUsize::new(0);
            let word_at = |address: usize| {
                assert_eq!(address % 64, 0);
                reads.fetch_add(1, Ordering::SeqCst);
                let w = words[address / 128];
                if address.is_multiple_of(128) {
                    w.lo
                } else {
                    w.hi
                }
            };
            let beta = if special {
                F128::ONE
            } else {
                source.sample_f128()
            };
            let gamma = source.sample_f128();
            let delta = if special {
                F128::ZERO
            } else {
                source.sample_f128()
            };
            let scalar = Factors::new(bit_at, record_vars, beta, gamma, delta);
            let packed = Factors::new_packed(word_at, record_vars, beta, gamma, delta);
            assert_eq!(scalar.rows, packed.rows);
            assert_eq!(
                reads.load(Ordering::SeqCst),
                (1 << record_vars) * SLOTS.div_ceil(64) * 43
            );
            let mut reference = FsChallenger::new(b"packed-scatter-proof-identity-v1");
            let mut candidate = FsChallenger::new(b"packed-scatter-proof-identity-v1");
            let (expected, expected_point) = scalar.prove(&mut reference);
            let (actual, actual_point) = packed.prove(&mut candidate);
            assert_eq!(expected.claim, actual.claim);
            assert_eq!(expected.rounds, actual.rounds);
            assert_eq!(expected_point, actual_point);
            assert_eq!(reference.sample_f128(), candidate.sample_f128());
        }
    }
}
