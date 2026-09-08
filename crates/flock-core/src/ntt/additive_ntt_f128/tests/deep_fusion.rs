//! Exact-reference coverage for the optional cache-resident layer fusion.
use super::*;

#[test]
fn c59_deep_ntt_fused_quarters_match_two_scalar_layers() {
    let mut rng = Rng::new(0xC590);
    for stride in [1, 2, 8, 64, 513] {
        for twiddles in [
            [F128::ZERO; 3],
            [F128::ONE; 3],
            [rng.f128(), rng.f128(), rng.f128()],
        ] {
            let input = rand_vec(&mut rng, 4 * stride);
            let mut expected = input.clone();
            // Independent index-based definition of the original two layers.
            for i in 0..2 * stride {
                let v = expected[i + 2 * stride];
                expected[i] += v * twiddles[0];
                expected[i + 2 * stride] = v + expected[i];
            }
            for half in 0..2 {
                let base = half * 2 * stride;
                for i in 0..stride {
                    let v = expected[base + i + stride];
                    expected[base + i] += v * twiddles[half + 1];
                    expected[base + i + stride] = v + expected[base + i];
                }
            }
            let mut actual = input;
            butterfly_interleaved_fused_2layer(&mut actual, twiddles[0], twiddles[1], twiddles[2]);
            assert_eq!(actual, expected, "quarter stride={stride}");
        }
    }
}

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[test]
fn c59_deep_ntt_all_start_layers_match_scalar_and_unfused() {
    let mut rng = Rng::new(0xC591);
    for workers in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            // Below the cache threshold, at the parallelism floor, and above
            // the cache threshold. The last case has several nonzero sub_idx.
            for (log_d, lanes) in [(8, 1), (12, 2), (12, 8), (12, 32), (12, 64)] {
                // A high-half-only independent basis checks that fusion never
                // assumes standard-basis or low-limb twiddle values. A larger
                // ambient basis also checks log_d != log_domain_size.
                let basis: Vec<_> = (0..log_d + 1)
                    .map(|bit| F128::new(0, 1u64 << bit))
                    .collect();
                let ntt = AdditiveNttF128::new(&basis);
                let input = rand_vec(&mut rng, (1 << log_d) * lanes);
                for start in 0..=log_d {
                    let mut expected = input.clone();
                    ntt.forward_transform_interleaved_scalar_from_layer(&mut expected, lanes, start);
                    for enabled in [false, true] {
                        let mut actual = input.clone();
                        ntt.forward_transform_interleaved_parallel_from_layer_impl(
                            &mut actual, lanes, start, enabled,
                        );
                        assert_eq!(actual, expected,
                            "workers={workers} log_d={log_d} lanes={lanes} start={start} enabled={enabled}");
                    }
                }
            }
        });
    }
}

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[test]
fn c59_deep_ntt_replicated_codeword_and_full_merkle_tree_match() {
    use crate::{
        hash::HashKind,
        merkle,
        pcs::{PcsParams, ligerito::LigeritoProfile},
    };
    let mut rng = Rng::new(0xC592);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        for (m, rate, batch) in [(20, 1, 1), (22, 1, 6), (21, 2, 3)] {
            let params = PcsParams {
                m,
                log_inv_rate: rate,
                log_batch_size: batch,
                profile: if rate == 1 {
                    LigeritoProfile::Fast
                } else {
                    LigeritoProfile::Slim
                },
                merkle_hash: Default::default(),
            };
            assert!(m >= 7 + batch && rate >= 1);
            let message = rand_vec(&mut rng, 1 << params.log_msg_len());
            let ntt = AdditiveNttF128::standard(params.k_code());
            let mut expected = vec![F128::ZERO; params.codeword_len_f128()];
            expected[..message.len()].copy_from_slice(&message);
            ntt.forward_transform_interleaved_scalar(&mut expected, params.num_ntts());
            let mut candidate = vec![F128::ZERO; expected.len()];
            for replica in candidate.chunks_exact_mut(message.len()) {
                replica.copy_from_slice(&message);
            }
            ntt.forward_transform_interleaved_parallel_from_layer_impl(
                &mut candidate,
                params.num_ntts(),
                rate,
                true,
            );
            assert_eq!(
                candidate, expected,
                "replicate skip m={m} rate={rate} batch={batch}"
            );
            // Explicit wire-order conversion is independent of the production
            // zero-copy cast. Compare all nodes, not merely the final root.
            let bytes = |values: &[F128]| -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|value| {
                        value
                            .lo
                            .to_le_bytes()
                            .into_iter()
                            .chain(value.hi.to_le_bytes())
                    })
                    .collect()
            };
            let expected_bytes = bytes(&expected);
            let candidate_bytes = bytes(&candidate);
            for hash in [HashKind::Sha256, HashKind::Blake3] {
                assert_eq!(
                    merkle::merkle_tree(&candidate_bytes, params.n_leaves(), hash),
                    merkle::merkle_tree_sequential(&expected_bytes, params.n_leaves(), hash),
                );
            }
        }
    });
}
