use super::PackedInputs;
use crate::challenger::{Challenger, FsChallenger};
use crate::field::F128;
use crate::zerocheck::{self, PaddingSpec, ProverOptions, Round1Mode};

fn words(n: usize, mut seed: u64) -> Vec<F128> {
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    (0..n)
        .map(|_| F128 {
            lo: next(),
            hi: next(),
        })
        .collect()
}

fn bytes(words: &[F128]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|word| {
            word.lo
                .to_ne_bytes()
                .into_iter()
                .chain(word.hi.to_ne_bytes())
        })
        .collect()
}

fn clear_padding(words: &mut [F128], m: usize, padding: PaddingSpec) {
    for bit in 0..(1 << m) {
        if bit % (1 << padding.k_log) >= padding.useful_bits_per_block {
            let word = &mut words[bit / 128];
            if bit % 128 < 64 {
                word.lo &= !(1u64 << (bit % 64));
            } else {
                word.hi &= !(1u64 << (bit % 64));
            }
        }
    }
}

#[test]
fn owned_zerocheck_reuses_exact_allocations_at_fused_boundary() {
    for n_in in [128, 512, 1024, 2048, 8192] {
        let a = words(n_in / 2, 7);
        let b = words(n_in / 2, 11);
        let pointers = (a.as_ptr(), b.as_ptr());
        let capacities = (a.capacity(), b.capacity());
        let expected = (bytes(&a), bytes(&b));
        let inputs = PackedInputs::Owned(a, b);
        assert_eq!(
            inputs.bytes(),
            (expected.0.as_slice(), expected.1.as_slice())
        );
        let (a, b) = inputs.into_tail_scratch(n_in);
        if n_in >= 1024 {
            assert_eq!((a.as_ptr(), b.as_ptr()), pointers);
            assert_eq!((a.capacity(), b.capacity()), capacities);
            assert_eq!((a.len(), b.len()), (n_in / 2, n_in / 2));
            assert_eq!((bytes(&a), bytes(&b)), expected);
        } else {
            assert!(a.is_empty() && b.is_empty());
        }
    }
}

#[test]
fn owned_zerocheck_proof_claim_capture_and_transcript_match() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for (m, useful, honest_padding) in [
                (13, 8192, true),
                (15, 65, true),
                (16, 8191, true),
                (17, 511, true),
                (21, 8191, true),
                (16, 65, false),
            ] {
                let padding = PaddingSpec {
                    k_log: 13,
                    useful_bits_per_block: useful,
                };
                let mut a = words(1 << (m - 7), 0x1234);
                let mut b = words(a.len(), 0x8765);
                if honest_padding {
                    clear_padding(&mut a, m, padding);
                    clear_padding(&mut b, m, padding);
                }
                let c: Vec<_> = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| F128 {
                        lo: a.lo & b.lo,
                        hi: a.hi & b.hi,
                    })
                    .collect();
                let (ab, bb, cb) = (bytes(&a), bytes(&b), bytes(&c));
                for options in [
                    ProverOptions::default(),
                    ProverOptions {
                        round1: Round1Mode::Deferred,
                        fused_tail: true,
                    },
                ] {
                    let mut reference_ch = FsChallenger::new(b"owned-zerocheck-parity");
                    let expected = zerocheck::prove_packed_padded_capture_s_hat_v_c_with_options(
                        &ab,
                        &bb,
                        &cb,
                        m,
                        &padding,
                        options,
                        &mut reference_ch,
                    );
                    let mut owned_ch = FsChallenger::new(b"owned-zerocheck-parity");
                    let actual = zerocheck::prove_packed_padded_capture_s_hat_v_c_owned(
                        a.clone(),
                        b.clone(),
                        &cb,
                        m,
                        &padding,
                        options,
                        &mut owned_ch,
                    );
                    assert_eq!(
                        actual, expected,
                        "workers={workers}, m={m}, useful={useful}"
                    );
                    let continuation = owned_ch.sample_f128_vec(4);
                    assert_eq!(continuation, reference_ch.sample_f128_vec(4));
                    if honest_padding {
                        let mut verify_ch = FsChallenger::new(b"owned-zerocheck-parity");
                        assert_eq!(
                            zerocheck::verify(m, &actual.0, &mut verify_ch).unwrap(),
                            actual.1
                        );
                        assert_eq!(verify_ch.sample_f128_vec(4), continuation);
                        let mut bad = actual.0;
                        bad.round1_ab[0] += F128::ONE;
                        let mut bad_ch = FsChallenger::new(b"owned-zerocheck-parity");
                        assert!(zerocheck::verify(m, &bad, &mut bad_ch).is_err());
                    }
                }
            }
        });
    }
}

#[test]
fn owned_zerocheck_rejects_odd_and_mismatched_shapes_before_challenges() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for which in 0..3 {
        for extra in [-1isize, 1] {
            let m = 13;
            let mut a = words(1 << (m - 7), 13);
            let mut b = words(a.len(), 17);
            let mut c = bytes(&a);
            match which {
                0 => a.resize(a.len().checked_add_signed(extra).unwrap(), F128::ZERO),
                1 => b.resize(b.len().checked_add_signed(extra).unwrap(), F128::ZERO),
                _ => c.resize(c.len().checked_add_signed(extra).unwrap(), 0),
            }
            let (ab, bb) = (bytes(&a), bytes(&b));
            let mut reference_ch = FsChallenger::new(b"owned-zerocheck-shape");
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    zerocheck::prove_packed_padded_capture_s_hat_v_c_with_options(
                        &ab,
                        &bb,
                        &c,
                        m,
                        &PaddingSpec::dense(m),
                        ProverOptions::default(),
                        &mut reference_ch,
                    )
                }))
                .is_err()
            );
            let mut owned_ch = FsChallenger::new(b"owned-zerocheck-shape");
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    zerocheck::prove_packed_padded_capture_s_hat_v_c_owned(
                        a,
                        b,
                        &c,
                        m,
                        &PaddingSpec::dense(m),
                        ProverOptions::default(),
                        &mut owned_ch,
                    )
                }))
                .is_err()
            );
            let mut untouched = FsChallenger::new(b"owned-zerocheck-shape");
            let expected = untouched.sample_f128_vec(4);
            assert_eq!(reference_ch.sample_f128_vec(4), expected);
            assert_eq!(owned_ch.sample_f128_vec(4), expected);
        }
    }
}
