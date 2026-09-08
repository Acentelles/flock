use super::super::hash_to_point_slots as slots;
use super::*;
use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::LincheckCircuit,
};

pub(super) fn fixtures(n: usize) -> Vec<sponge::SpongeRecord> {
    (0..n)
        .map(|i| sponge::SpongeRecord {
            salt: [i as u8; 40],
            hpk: [(i as u8).wrapping_add(1); 64],
            message: vec![i as u8; [31, 32, 165][i % 3]],
        })
        .collect()
}
fn witness(setup: &Setup, inputs: &[sponge::SpongeRecord]) -> Witness {
    let sp =
        sponge::sponge_witness_without_lincheck(&sponge::SpongeSetup::new(inputs.len()), inputs);
    let blocks: Vec<[u16; slots::SLOTS]> = sp
        .all_words
        .iter()
        .map(|w| w.as_slice().try_into().unwrap())
        .collect();
    let rp = record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
    assemble(setup, sp, rp.z_packed)
}

#[test]
fn hybrid_content_point_hook_preserves_full_proof_and_transcript() {
    let inputs = fixtures(32);
    let setup = Setup::new(32);
    let publics: Vec<_> = inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let mut reference_ch = FsChallenger::new(b"hybrid-content-point-hook");
    let reference_core = prove_core(
        &setup,
        commit(&setup, witness(&setup, &inputs)),
        &mut reference_ch,
    );
    let expected_point = reference_core.r_fp.clone();
    let reference = open(&setup, reference_core, &mut reference_ch);
    let reference_bytes = bincode::serialize(&reference).unwrap();
    let expected_next = reference_ch.sample_f128();
    for packed_record in [false, true] {
        let mut ch = FsChallenger::new(b"hybrid-content-point-hook");
        let mut calls = 0;
        let mut point_seen = Vec::new();
        let core = prove_core_with_point_hook(
            &setup,
            commit(&setup, witness(&setup, &inputs)),
            packed_record,
            Default::default(),
            &mut ch,
            |point| {
                calls += 1;
                point_seen = point.to_vec();
            },
        );
        assert_eq!(calls, 1);
        assert_eq!(point_seen, expected_point);
        assert_eq!(point_seen, core.r_fp);
        let proof = open(&setup, core, &mut ch);
        assert_eq!(bincode::serialize(&proof).unwrap(), reference_bytes);
        assert_eq!(ch.sample_f128(), expected_next);
        let mut verifier = FsChallenger::new(b"hybrid-content-point-hook");
        let core = verify_core(&setup, &publics, &proof, &mut verifier).unwrap();
        verify_open(&setup, &proof, core, &mut verifier).unwrap();
        assert_eq!(verifier.sample_f128(), expected_next);
    }
}

pub(super) fn direct_witness(setup: &Setup, inputs: &[sponge::SpongeRecord]) -> Witness {
    let sp = compact_sponge_witness(setup, inputs);
    let blocks: Vec<[u16; slots::SLOTS]> = sp
        .all_words
        .iter()
        .map(|w| w.as_slice().try_into().unwrap())
        .collect();
    let rp = record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
    assemble_direct(setup, sp, rp.z_packed)
}

#[test]
fn hybrid_rows_and_word_copy_constraints() {
    let setup = Setup::new(8);
    let w = witness(&setup, &fixtures(8));
    for ((a, b), z) in w.a.iter().zip(&w.b).zip(&w.z) {
        assert_eq!(a.lo & b.lo, z.lo);
        assert_eq!(a.hi & b.hi, z.hi);
    }
    for slot in 0..slots::SLOTS {
        for bit in 0..16 {
            let row = layout::record_position(slots::word_position(slot, bit)).unwrap();
            assert_eq!(
                layout::bit(&w.a, row),
                layout::bit(&w.z, layout::word_source(slot, bit))
            );
            assert!(layout::bit(&w.b, row));
            // Flipping just the record input breaks the copy row itself.
            assert_ne!(
                layout::bit(&w.a, row) & layout::bit(&w.b, row),
                !layout::bit(&w.z, row)
            );
        }
    }
    let mut ch = FsChallenger::new(b"hybrid-walker-differential-v1");
    let alpha = ch.sample_f128();
    let weights = ch.sample_f128_vec(layout::K);
    let columns = setup.fold_alpha_batched(alpha, &weights);
    let left = weights.iter().enumerate().fold(F128::ZERO, |s, (i, &r)| {
        s + if layout::bit(&w.a, i) {
            alpha * r
        } else {
            F128::ZERO
        } + if layout::bit(&w.b, i) { r } else { F128::ZERO }
    });
    let right = columns.iter().enumerate().fold(F128::ZERO, |s, (i, &r)| {
        s + if layout::bit(&w.z, i) { r } else { F128::ZERO }
    });
    assert_eq!(left, right);
}

#[test]
fn hybrid_complete_relation_and_rejections() {
    let inputs = fixtures(32);
    let publics: Vec<_> = inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let setup = Setup::new(32);
    let w = witness(&setup, &inputs);
    let prepared = commit(&setup, w);
    let mut ch = FsChallenger::new(b"hybrid-full-relation-test-v1");
    let core = prove_core(&setup, prepared, &mut ch);
    #[cfg(all(feature = "aligned-hybrid", feature = "compact-fingerprint"))]
    {
        let mut closure_ch = FsChallenger::new(b"hybrid-opening-count-v2");
        let closed =
            super::super::face_closure::close_faces(&core.points, &core.values, &mut closure_ch)
                .unwrap();
        assert_eq!(closed.len() + 2, 27);
    }
    let r_fp = core.r_fp.clone();
    let proof = open(&setup, core, &mut ch);
    let check = |p: &Proof, publics: &[sponge::SpongePublic]| {
        let mut ch = FsChallenger::new(b"hybrid-full-relation-test-v1");
        let core = verify_core(&setup, publics, p, &mut ch)?;
        assert_eq!(core.r_fp, r_fp);
        verify_open(&setup, p, core, &mut ch)
    };
    check(&proof, &publics).unwrap();
    for category in 0..5 {
        let mut bad = proof.clone();
        match category {
            0 => bad.sponge_values[0] += F128::ONE,
            1 => bad.record_values[0] += F128::ONE,
            2 => bad.fingerprint += F128::ONE,
            3 => bad.fragment_values[0] += F128::ONE,
            _ => {
                bad.fragment_values.pop();
            }
        }
        assert!(check(&bad, &publics).is_err());
    }
    let mut bad_publics = publics.clone();
    bad_publics[0].hpk[0] ^= 1;
    assert!(check(&proof, &bad_publics).is_err());
    let mut incompatible = Setup::new(32);
    incompatible.descriptor[0] ^= 1;
    let mut ch = FsChallenger::new(b"hybrid-full-relation-test-v1");
    assert!(verify_core(&incompatible, &publics, &proof, &mut ch).is_err());
}

#[test]
fn hybrid_relocation_matches_original_witness() {
    let inputs = fixtures(8);
    let setup = Setup::new(8);
    let sp = sponge::sponge_witness(&sponge::SpongeSetup::new(8), &inputs);
    let blocks: Vec<[u16; slots::SLOTS]> = sp
        .all_words
        .iter()
        .map(|w| w.as_slice().try_into().unwrap())
        .collect();
    let rp = record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
    let old_record = rp.z_packed[..slots::K / 128].to_vec();
    let old_sponge = sp.z_packed[..layout::K / 128].to_vec();
    let w = assemble(&setup, sp, rp.z_packed);
    let mut ch = FsChallenger::new(b"hybrid-relocation-differential-v1");
    let points = vec![
        ch.sample_f128_vec(17),
        vec![F128::ZERO; 17],
        vec![F128::ONE; 17],
    ];
    let (relocated, _, _) =
        evaluations(&w.z[..layout::K / 128], &flatten(&points, record_fragments));
    assert_eq!(relocated, sponge::gather_eval_many(&old_record, &points));
    let points = sponge::sponge_relation_points(0, &mut ch);
    let (relocated, _, _) =
        evaluations(&w.z[..layout::K / 128], &flatten(&points, sponge_fragments));
    assert_eq!(relocated, sponge::gather_eval_many(&old_sponge, &points));
}

#[test]
fn hybrid_optimizations_preserve_complete_proof() {
    let inputs = fixtures(32);
    let setup = Setup::new(inputs.len());
    let run = |mode| {
        let reference = mode == 0;
        let w = if reference {
            let sp = sponge::sponge_witness(&sponge::SpongeSetup::new(inputs.len()), &inputs);
            let blocks: Vec<[u16; slots::SLOTS]> = sp
                .all_words
                .iter()
                .map(|w| w.as_slice().try_into().unwrap())
                .collect();
            let rp =
                record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
            circuit::assemble_reference(&setup, sp, rp.z_packed)
        } else if mode == 1 {
            witness(&setup, &inputs)
        } else {
            direct_witness(&setup, &inputs)
        };
        let prepared = commit(&setup, w);
        let mut ch = FsChallenger::new(b"hybrid-stripes-proof-identity-v1");
        let core = prove_core_with_packer_and_record(
            &setup,
            prepared,
            &mut ch,
            |z, m, k_log| {
                if reference {
                    // Independent logical Boolean oracle, including every
                    // padding bit. The complete proof and replay state must
                    // remain identical, beyond just the local byte layout.
                    let bits: Vec<_> = (0..1 << m).map(|p| layout::bit(z, p)).collect();
                    lincheck::pack_z_lincheck(&bits, m, k_log)
                } else {
                    lincheck::pack_z_lincheck_from_packed(z, m, k_log)
                }
            },
            mode == 3,
        );
        let proof = open(&setup, core, &mut ch);
        (bincode::serialize(&proof).unwrap(), ch.sample_f128())
    };
    let expected = run(0);
    assert_eq!(run(1), expected);
    assert_eq!(run(2), expected);
    assert_eq!(run(3), expected);
}

#[test]
fn hybrid_direct_generation_matches_reference_for_all_message_lengths() {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let n = 256;
    let setup = Setup::new(n);
    let mut inputs = fixtures(n);
    let mut ch = FsChallenger::new(b"hybrid-direct-sponge-differential-v1");
    for (i, input) in inputs.iter_mut().enumerate() {
        let bytes: Vec<_> = ch
            .sample_f128_vec(18)
            .iter()
            .flat_map(|word| {
                word.lo
                    .to_le_bytes()
                    .into_iter()
                    .chain(word.hi.to_le_bytes())
            })
            .collect();
        input.salt.copy_from_slice(&bytes[..40]);
        input.hpk.copy_from_slice(&bytes[40..104]);
        // Cover every permitted length and second-block padding boundary.
        input.message = bytes[104..104 + 31 + i % 135].to_vec();
    }
    let original = sponge::sponge_witness_without_lincheck(&sponge::SpongeSetup::new(n), &inputs);
    let direct = compact_sponge_witness(&setup, &inputs);
    assert_eq!(direct.all_words, original.all_words);
    for (input, words) in inputs.iter().zip(&direct.all_words) {
        let mut shake = sha3::Shake256::default();
        shake.update(&input.salt);
        shake.update(&input.hpk);
        shake.update(&[0, 0]);
        shake.update(&input.message);
        let mut expected = [0u8; slots::SLOTS * 2];
        shake.finalize_xof().read(&mut expected);
        let expected: Vec<_> = expected
            .chunks_exact(2)
            .map(|w| u16::from_be_bytes([w[0], w[1]]))
            .collect();
        assert_eq!(*words, expected);
    }
    let blocks: Vec<[u16; slots::SLOTS]> = direct
        .all_words
        .iter()
        .map(|w| w.as_slice().try_into().unwrap())
        .collect();
    let rp = record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
    let expected = circuit::assemble_reference(&setup, original, rp.z_packed.clone());
    let got = assemble_direct(&setup, direct, rp.z_packed);
    assert_eq!(got.z, expected.z);
    assert_eq!(got.a, expected.a);
    assert_eq!(got.b, expected.b);
}

#[test]
fn hybrid_direct_record_assembly_preserves_arbitrary_inputs_and_fallback() {
    for (n, irregular) in [(8, false), (32, false), (8, true)] {
        let mut setup = Setup::new(n);
        if irregular {
            setup.slots.r1cs.a_0.rows = (0..slots::K)
                .map(|row| vec![(row * 40503 + 17) % slots::K])
                .collect();
        }
        let inputs = fixtures(n);
        let original =
            sponge::sponge_witness_without_lincheck(&sponge::SpongeSetup::new(n), &inputs);
        let direct = compact_sponge_witness(&setup, &inputs);
        let mut ch = FsChallenger::new(b"hybrid-direct-record-arbitrary-v1");
        // Include inconsistent SHAKE/record words, non-one constants and
        // nonzero high/padding bits; do not assume a satisfying record witness.
        let record = ch.sample_f128_vec(n * slots::K / 128);
        let expected = circuit::assemble_reference(&setup, original, record.clone());
        let got = assemble_direct(&setup, direct, record);
        assert_eq!(got.z, expected.z);
        assert_eq!(got.a, expected.a);
        assert_eq!(got.b, expected.b);
    }
}

#[test]
fn hybrid_direct_generation_rejects_invalid_message_lengths() {
    let setup = Setup::new(8);
    let sp_setup = sponge::SpongeSetup::new(8);
    for length in [0, 30, 166] {
        let mut inputs = fixtures(8);
        inputs[0].message = vec![0; length];
        assert!(
            std::panic::catch_unwind(|| {
                sponge::sponge_witness_without_lincheck(&sp_setup, &inputs)
            })
            .is_err()
        );
        assert!(std::panic::catch_unwind(|| compact_sponge_witness(&setup, &inputs)).is_err());
    }
}

#[test]
fn hybrid_optional_stripes_and_inplace_assembly_preserve_witness() {
    for n in [8, 32] {
        let inputs = fixtures(n);
        let setup = Setup::new(n);
        let sp_setup = sponge::SpongeSetup::new(n);
        let original = sponge::sponge_witness(&sp_setup, &inputs);
        let compact = sponge::sponge_witness_without_lincheck(&sp_setup, &inputs);
        assert!(!original.z_lincheck.is_empty());
        assert!(compact.z_lincheck.is_empty());
        assert_eq!(compact.z_packed, original.z_packed);
        assert_eq!(compact.a_packed, original.a_packed);
        assert_eq!(compact.b_packed, original.b_packed);
        assert_eq!(compact.all_words, original.all_words);
        let addresses = (
            compact.z_packed.as_ptr(),
            compact.a_packed.as_ptr(),
            compact.b_packed.as_ptr(),
        );
        let blocks: Vec<[u16; slots::SLOTS]> = original
            .all_words
            .iter()
            .map(|w| w.as_slice().try_into().unwrap())
            .collect();
        let rp = record::record_witness(&setup.slots, &blocks, &[[false; 128]; slots::MASK_REPS]);
        let expected = circuit::assemble_reference(&setup, original, rp.z_packed.clone());
        let got = assemble(&setup, compact, rp.z_packed);
        assert_eq!((got.z.as_ptr(), got.a.as_ptr(), got.b.as_ptr()), addresses);
        assert_eq!(got.z, expected.z);
        assert_eq!(got.a, expected.a);
        assert_eq!(got.b, expected.b);
    }
}

#[test]
fn hybrid_inplace_assembly_matches_reference_on_arbitrary_bits() {
    for (records, irregular) in [(8, false), (32, false), (8, true)] {
        let mut setup = Setup::new(records);
        if irregular {
            // Keep the general matrix fallback correct, without relying on an
            // honest record or an all-one constant wire.
            setup.slots.r1cs.a_0.rows = (0..slots::K)
                .map(|row| vec![(row * 40503 + 17) % slots::K])
                .collect();
            assert!(flock_core::r1cs::word_apply::Program::new(&setup.slots.r1cs.a_0).is_none());
        }
        assert_eq!(
            flock_core::r1cs::word_apply::Program::new(&setup.slots.r1cs.a_0).is_some()
                && flock_core::r1cs::word_apply::Program::new(&setup.slots.r1cs.b_0).is_some(),
            !irregular
        );
        let mut ch = FsChallenger::new(b"hybrid-arbitrary-relocation-v1");
        let sp = sponge::SpongeWitness {
            z_packed: ch.sample_f128_vec(records * layout::K / 128),
            a_packed: ch.sample_f128_vec(records * layout::K / 128),
            b_packed: ch.sample_f128_vec(records * layout::K / 128),
            z_lincheck: vec![5; 64],
            all_words: vec![],
        };
        let copied = sponge::SpongeWitness {
            z_packed: sp.z_packed.clone(),
            a_packed: sp.a_packed.clone(),
            b_packed: sp.b_packed.clone(),
            z_lincheck: vec![],
            all_words: vec![],
        };
        let record = ch.sample_f128_vec(records * slots::K / 128);
        // Arbitrary high and padding bits exercise every copied limb. Word-copy
        // A rows must still come from SHAKE even when slot input bits disagree.
        let expected = circuit::assemble_reference(&setup, sp, record.clone());
        let got = assemble(&setup, copied, record);
        assert_eq!(got.z, expected.z);
        assert_eq!(got.a, expected.a);
        assert_eq!(got.b, expected.b);
    }
}

#[test]
fn hybrid_packed_word_rows_preserve_endianness_boundaries_and_padding() {
    let mut ch = FsChallenger::new(b"hybrid-packed-copy-rows-v1");
    let initial = ch.sample_f128_vec(layout::K / 128);
    let check = |z: &[F128]| {
        let mut expected = initial.clone();
        for slot in 0..slots::SLOTS {
            for bit in 0..16 {
                layout::set_bit(
                    &mut expected,
                    layout::record_position(slots::word_position(slot, bit)).unwrap(),
                    layout::bit(z, layout::word_source(slot, bit)),
                );
            }
        }
        let mut actual = initial.clone();
        circuit::copy_word_rows(z, &mut actual);
        assert_eq!(actual, expected);
    };
    check(&ch.sample_f128_vec(layout::K / 128));
    // Basis bits exercise byte order, packet and rate-block transitions,
    // compact record segment transitions, and the final four-word packet.
    for slot in [0, 3, 7, 8, 63, 64, 67, 68, 255, 256, 511, 512, 608, 611] {
        for bit in 0..16 {
            let mut z = vec![F128::ZERO; layout::K / 128];
            layout::set_bit(&mut z, layout::word_source(slot, bit), true);
            check(&z);
        }
    }
}
