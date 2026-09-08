use super::super::hash_to_point_slots as slots;
use super::*;
use flock_core::challenger::FsChallenger;

#[test]
fn hybrid_packed_record_preserves_complete_proof_and_transcript() {
    let inputs: Vec<_> = (0..32)
        .map(|i| sponge::SpongeRecord {
            salt: [i as u8; 40],
            hpk: [i as u8 + 1; 64],
            message: vec![i as u8; [31, 32, 165][i % 3]],
        })
        .collect();
    let publics: Vec<_> = inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let setup = Setup::new(inputs.len());
    let run = |packed| {
        let sponge = sponge::sponge_witness_without_lincheck(
            &sponge::SpongeSetup::new(inputs.len()),
            &inputs,
        );
        let blocks: Vec<[u16; slots::SLOTS]> = sponge
            .all_words
            .iter()
            .map(|w| w.as_slice().try_into().unwrap())
            .collect();
        let record =
            record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
        let prepared = commit(&setup, assemble(&setup, sponge, record.z_packed));
        let mut ch = FsChallenger::new(b"hybrid-packed-record-proof-identity-v1");
        let core = if packed {
            prove_core_packed_record(&setup, prepared, &mut ch)
        } else {
            prove_core(&setup, prepared, &mut ch)
        };
        let proof = open(&setup, core, &mut ch);
        let mut verifier = FsChallenger::new(b"hybrid-packed-record-proof-identity-v1");
        let core = verify_core(&setup, &publics, &proof, &mut verifier).unwrap();
        verify_open(&setup, &proof, core, &mut verifier).unwrap();
        (bincode::serialize(&proof).unwrap(), ch.sample_f128())
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn hybrid_packed_record_access_matches_arbitrary_scalar_relation() {
    let record_vars = 3;
    let mut source = FsChallenger::new(b"hybrid-packed-record-access-v1");
    let z = source.sample_f128_vec(1 << (record_vars + layout::K_LOG - 7));
    let read_bit = |p: usize| {
        layout::record_position(p % slots::K)
            .is_some_and(|q| layout::bit(&z, (p / slots::K) * layout::K + q))
    };
    let read_word = |p: usize| {
        assert_eq!(p % 64, 0);
        layout::record_position(p % slots::K).map_or(0, |q| {
            assert_eq!(q % 64, 0);
            let bit = (p / slots::K) * layout::K + q;
            let w = z[bit / 128];
            if bit.is_multiple_of(128) { w.lo } else { w.hi }
        })
    };
    let evaluate = |points: &[Vec<F128>]| evaluations(&z, &flatten(points, record_fragments)).0;
    let mut reference = FsChallenger::new(b"hybrid-packed-record-arbitrary-v1");
    let mut candidate = FsChallenger::new(b"hybrid-packed-record-arbitrary-v1");
    let expected = record::prove_record_relation(record_vars, read_bit, evaluate, &mut reference);
    let actual =
        record::prove_record_relation_packed(record_vars, read_word, evaluate, &mut candidate);
    assert_eq!(expected.scatter.claim, actual.scatter.claim);
    assert_eq!(expected.scatter.rounds, actual.scatter.rounds);
    assert_eq!(expected.points, actual.points);
    assert_eq!(expected.values, actual.values);
    assert_eq!(expected.r_fp, actual.r_fp);
    assert_eq!(expected.fingerprint, actual.fingerprint);
    assert_eq!(reference.sample_f128(), candidate.sample_f128());
}
