use super::*;
use flock_core::challenger::{Challenger, FsChallenger};

/// The validation driver compares this complete-proof digest across separate
/// default/reuse feature processes. No environment mutation or timed caching.
#[test]
fn hybrid_owned_zerocheck_complete_proof_identity_receipt() {
    let inputs = tests::fixtures(8);
    let setup = Setup::new(inputs.len());
    let publics: Vec<_> = inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let witness = tests::direct_witness(&setup, &inputs);
    let mut ch = FsChallenger::new(b"hybrid-owned-zerocheck-v1");
    let core = prove_core_packed_record(&setup, commit(&setup, witness), &mut ch);
    let proof = open(&setup, core, &mut ch);
    let binding = ch.sample_f128_vec(4);
    let mut verifier = FsChallenger::new(b"hybrid-owned-zerocheck-v1");
    let core = verify_core(&setup, &publics, &proof, &mut verifier).unwrap();
    verify_open(&setup, &proof, core, &mut verifier).unwrap();
    assert_eq!(verifier.sample_f128_vec(4), binding);
    let bytes = bincode::serialize(&proof).unwrap();
    let mut digest = blake3::Hasher::new();
    digest.update(&bytes);
    digest.update(&bincode::serialize(&binding).unwrap());
    println!(
        "C64_IDENTITY bytes={} digest={}",
        bytes.len(),
        digest.finalize().to_hex()
    );
    let mut bad = proof;
    bad.fingerprint += F128::ONE;
    let mut verifier = FsChallenger::new(b"hybrid-owned-zerocheck-v1");
    let result = verify_core(&setup, &publics, &bad, &mut verifier)
        .and_then(|core| verify_open(&setup, &bad, core, &mut verifier));
    assert!(result.is_err());
}
