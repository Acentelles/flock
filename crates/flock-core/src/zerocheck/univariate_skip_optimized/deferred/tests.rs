use super::super::{
    K_SKIP, medium_challenges_ghash,
    round1_shift_reduce_extract_c_packed_padded_with_s_hat_v as reduced,
    round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_deferred as deferred,
    small_challenges_ghash,
};
use super::*;
use crate::challenger::{Challenger, FsChallenger};
use crate::field::F8;
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use crate::zerocheck::{self, PaddingSpec, ProverOptions, Round1Mode};

fn random_bytes(ch: &mut FsChallenger, n: usize) -> Vec<u8> {
    ch.sample_f128_vec(n.div_ceil(16))
        .into_iter()
        .flat_map(|v| {
            let mut bytes = [0; 16];
            bytes[..8].copy_from_slice(&v.lo.to_le_bytes());
            bytes[8..].copy_from_slice(&v.hi.to_le_bytes());
            bytes
        })
        .take(n)
        .collect()
}

#[test]
fn round1_deferred_arbitrary_conversions_banks_and_cancellation() {
    let mut ch = FsChallenger::new(b"round1-deferred-accumulator-v1");
    // Arbitrary field entries, not only the protocol convert subspace, test
    // reduction linearity independently of the optimized witness transform.
    let convert = ch.sample_f128_vec(16 * 256);
    let mut expected = [[F128::ZERO; ELL]; 3];
    let mut actual = [[F256Unreduced::ZERO; ELL]; 3];
    for i in 0..37 {
        let ab: [[u8; 64]; 16] =
            std::array::from_fn(|_| random_bytes(&mut ch, 64).try_into().unwrap());
        let mut c: [[u8; 64]; 16] =
            std::array::from_fn(|_| random_bytes(&mut ch, 64).try_into().unwrap());
        // Exercise both individual banks and their union on every lane.
        for row in &mut c {
            for (lane, byte) in row.iter_mut().enumerate() {
                match (lane + i) % 4 {
                    0 => *byte &= 0x55,
                    1 => *byte &= 0xaa,
                    2 => *byte = 0xff,
                    _ => {}
                }
            }
        }
        let weight = match i % 4 {
            0 => F128::ZERO,
            1 => F128::ONE,
            2 => F128::new(u64::MAX, u64::MAX),
            _ => ch.sample_f128(),
        };
        let n = [0, 1, 3, 7, 15, 16][i % 6];
        // Repeating identical products twice must cancel in characteristic 2.
        for _ in 0..if i % 3 == 0 { 2 } else { 1 } {
            let [ea, ec0, ec1] = &mut expected;
            F128::accumulate(&ab, &c, n, &convert, weight, ea, ec0, ec1);
            let [aa, ac0, ac1] = &mut actual;
            F256Unreduced::accumulate(&ab, &c, n, &convert, weight, aa, ac0, ac1);
        }
        for bank in 0..3 {
            for lane in 0..ELL {
                assert_eq!(
                    actual[bank][lane].reduce(),
                    expected[bank][lane],
                    "step={i}, bank={bank}, lane={lane}"
                );
            }
        }
    }
}

#[test]
fn round1_deferred_arbitrary_witness_and_padding_boundaries() {
    let mut ch = FsChallenger::new(b"round1-deferred-padding-v1");
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1 << K_SKIP));
    let inv = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    // Include empty/full/partial medium windows, partial bytes, and m>20:
    // there are then multiple eq_lo contributions before each final reduction.
    for (m, k_log, useful) in [
        (13, 13, 8192),
        (14, 13, 0),
        (14, 13, 1),
        (14, 13, 511),
        (14, 13, 512),
        (15, 14, 513),
        (17, 16, 8193),
        (21, 19, 522624),
        (23, 19, 522624),
        (21, 19, 524287),
    ] {
        let n = (1 << m) / 8;
        let mut a = random_bytes(&mut ch, n);
        let mut b = random_bytes(&mut ch, n);
        let mut c = random_bytes(&mut ch, n);
        let mut r = ch.sample_f128_vec(m);
        r[6..9].copy_from_slice(&small_challenges_ghash());
        r[9..13].copy_from_slice(&medium_challenges_ghash());
        let pad = PaddingSpec {
            k_log,
            useful_bits_per_block: useful,
        };
        // Even malformed nonzero declared padding must behave identically;
        // this optimization must not introduce any new input assumptions.
        assert_eq!(
            reduced(&a, &b, &c, m, K_SKIP, &r, &inv, &pad),
            deferred(&a, &b, &c, m, K_SKIP, &r, &inv, &pad)
        );
        for data in [&mut a, &mut b, &mut c] {
            for block in data.chunks_mut((1 << k_log) / 8) {
                for bit in useful..1 << k_log {
                    block[bit / 8] &= !(1 << (bit % 8));
                }
            }
        }
        let oracle = reduced(&a, &b, &c, m, K_SKIP, &r, &inv, &PaddingSpec::dense(m));
        assert_eq!(
            oracle,
            deferred(&a, &b, &c, m, K_SKIP, &r, &inv, &pad),
            "m={m}, k_log={k_log}, useful={useful}"
        );
    }
}

#[test]
fn round1_deferred_complete_zerocheck_and_transcript_identity() {
    let m = 21;
    let mut rng = FsChallenger::new(b"round1-deferred-proof-input-v1");
    let a = random_bytes(&mut rng, (1 << m) / 8);
    let b = random_bytes(&mut rng, a.len());
    let c: Vec<_> = a.iter().zip(&b).map(|(a, b)| a & b).collect();
    let run = |mode| {
        let mut ch = FsChallenger::new(b"round1-deferred-proof-v1");
        let (proof, claim, bank) = zerocheck::prove_packed_padded_capture_s_hat_v_c_with_options(
            &a,
            &b,
            &c,
            m,
            &PaddingSpec::dense(m),
            ProverOptions {
                round1: mode,
                ..ProverOptions::default()
            },
            &mut ch,
        );
        let next = ch.sample_f128();
        let mut verifier = FsChallenger::new(b"round1-deferred-proof-v1");
        assert_eq!(zerocheck::verify(m, &proof, &mut verifier).unwrap(), claim);
        assert_eq!(verifier.sample_f128(), next);
        (proof, claim, bank, next)
    };
    let expected = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(Round1Mode::Reduced));
    let actual = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap()
        .install(|| run(Round1Mode::Deferred));
    assert_eq!(actual, expected);
    for category in 0..3 {
        let mut bad = actual.0.clone();
        match category {
            0 => bad.round1_ab[0] += F128::ONE,
            1 => bad.round1_c[0] += F128::ONE,
            _ => bad.final_a_eval += F128::ONE,
        }
        let mut ch = FsChallenger::new(b"round1-deferred-proof-v1");
        assert!(zerocheck::verify(m, &bad, &mut ch).is_err());
    }
}
