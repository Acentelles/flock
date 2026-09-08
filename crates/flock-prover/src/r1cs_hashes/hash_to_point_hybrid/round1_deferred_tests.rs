use super::*;
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::zerocheck::{ProverOptions, Round1Mode};

#[test]
fn hybrid_zerocheck_options_complete_proof_and_transcript_identity() {
    let inputs = tests::fixtures(32);
    let setup = Setup::new(inputs.len());
    let publics: Vec<_> = inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let run = |options| {
        let w = tests::direct_witness(&setup, &inputs);
        let prepared = commit(&setup, w);
        let mut ch = FsChallenger::new(b"hybrid-round1-deferred-v1");
        let core = prove_core_with_zerocheck_options(&setup, prepared, true, options, &mut ch);
        let proof = open(&setup, core, &mut ch);
        let binding = ch.sample_f128();
        let mut verifier = FsChallenger::new(b"hybrid-round1-deferred-v1");
        let core = verify_core(&setup, &publics, &proof, &mut verifier).unwrap();
        verify_open(&setup, &proof, core, &mut verifier).unwrap();
        assert_eq!(verifier.sample_f128(), binding);
        (proof, binding)
    };
    let (control, control_binding) = run(ProverOptions::default());
    let control_bytes = bincode::serialize(&control).unwrap();
    for options in [
        ProverOptions {
            round1: Round1Mode::Deferred,
            fused_tail: false,
        },
        ProverOptions {
            round1: Round1Mode::Reduced,
            fused_tail: true,
        },
        ProverOptions {
            round1: Round1Mode::Deferred,
            fused_tail: true,
        },
    ] {
        let (candidate, candidate_binding) = run(options);
        assert_eq!(
            control_bytes,
            bincode::serialize(&candidate).unwrap(),
            "options={options:?}"
        );
        assert_eq!(control_binding, candidate_binding, "options={options:?}");
        let mut bad = candidate;
        bad.fingerprint += F128::ONE;
        let mut verifier = FsChallenger::new(b"hybrid-round1-deferred-v1");
        let result = verify_core(&setup, &publics, &bad, &mut verifier)
            .and_then(|core| verify_open(&setup, &bad, core, &mut verifier));
        assert!(result.is_err());
    }
}
