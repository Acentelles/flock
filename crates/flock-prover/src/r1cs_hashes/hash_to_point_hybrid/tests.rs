use super::super::hash_to_point_slots as slots;
use super::*;
use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::LincheckCircuit,
};

fn fixtures(n: usize) -> Vec<sponge::SpongeRecord> {
    (0..n)
        .map(|i| sponge::SpongeRecord {
            salt: [i as u8; 40],
            hpk: [i as u8 + 1; 64],
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
    assert_eq!(core.sponge_values.len(), 22);
    let mut closure_ch = FsChallenger::new(b"hybrid-opening-count-v2");
    let closed =
        super::super::face_closure::close_faces(&core.points, &core.values, &mut closure_ch)
            .unwrap();
    eprintln!(
        "hybrid aligned claims: fragments={}, PCS={}",
        core.points.len(),
        closed.len() + 2
    );
    assert!(closed.len() + 2 < 123);
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
fn hybrid_aligned_layout_retains_live_rows_and_disjoint_columns() {
    use super::super::keccak3;
    let setup = Setup::new(8);
    // An omitted record row must have been exactly 0 * 0 = z_row;
    // no retained gate may reference an omitted column.
    for matrix in [&setup.slots.r1cs.a_0, &setup.slots.r1cs.b_0] {
        for (row, columns) in matrix.rows.iter().enumerate() {
            if layout::record_position(row).is_none() {
                assert!(columns.is_empty(), "omitted live row {row}");
            }
            for &column in columns {
                assert!(layout::record_position(column).is_some());
            }
        }
    }
    let mut occupied = vec![false; layout::K];
    occupied[layout::CONST] = true;
    for old in 1..slots::K {
        if let Some(new) = layout::record_position(old) {
            assert!(!occupied[new], "record alias at {new}");
            occupied[new] = true;
        }
    }
    for block in 0..4 {
        for old in 0..keccak3::K {
            if old == keccak3::Z_CONST {
                assert_eq!(layout::sponge_position(block, old), Some(layout::CONST));
            } else if let Some(new) = layout::sponge_position(block, old) {
                assert!(!occupied[new], "sponge alias at {new}");
                occupied[new] = true;
            }
        }
    }
    // New state padding must be unoccupied by every row and column map.
    for state in 0..20 {
        let base = layout::STATES + state * layout::STATE_STRIDE;
        assert_eq!(base % 2048, 0);
        assert!(occupied[base..base + 1600].iter().all(|&b| b));
        assert!(occupied[base + 1600..base + 2048].iter().all(|&b| !b));
    }
    assert!(occupied[layout::END..].iter().all(|&b| !b));
}

#[test]
fn hybrid_aligned_state_queries_close_to_four_claims() {
    let mut ch = FsChallenger::new(b"hybrid-aligned-state-faces-v2");
    let points = sponge::sponge_relation_points(14, &mut ch);
    let groups = flatten(&points, sponge_fragments);
    assert_eq!(groups.len(), 22);
    assert!(groups.iter().all(|group| group.len() == 1));
    assert!(groups.iter().flatten().all(|f| f.weight == F128::ONE));
    let points: Vec<_> = groups.iter().flatten().map(|f| f.point.clone()).collect();
    let closed =
        super::super::face_closure::close_faces(&points, &vec![F128::ZERO; points.len()], &mut ch)
            .unwrap();
    // Two complete faces cover the twenty states, plus two salt subcubes.
    assert_eq!(closed.len(), 4);
}

#[test]
fn hybrid_aligned_fragments_match_arbitrary_retained_values() {
    let setup = Setup::new(8);
    let mut ch = FsChallenger::new(b"hybrid-aligned-arbitrary-mle-v2");
    let mut old_record = ch.sample_f128_vec(8 * slots::K / 128);
    for record in 0..8 {
        for old in 0..slots::K {
            if layout::record_position(old).is_none() || (1..64).contains(&old) {
                layout::set_bit(&mut old_record, record * slots::K + old, false);
            }
        }
    }
    let mut old_sponge = ch.sample_f128_vec(8 * layout::K / 128);
    for record in 0..8 {
        for block in 0..4 {
            for slot in 0..6 {
                for bit in 1600..2048 {
                    layout::set_bit(
                        &mut old_sponge,
                        record * layout::K + block * slots::K + slot * 2048 + bit,
                        false,
                    );
                }
            }
        }
    }
    let sp = sponge::SpongeWitness {
        z_packed: old_sponge.clone(),
        a_packed: vec![F128::ZERO; old_sponge.len()],
        b_packed: vec![F128::ZERO; old_sponge.len()],
        z_lincheck: vec![],
        all_words: vec![],
    };
    let w = assemble(&setup, sp, old_record.clone());
    // Nonzero top target bits and inactive-but-retained slot bits are kept.
    // Evaluate the full multi-record MLE at independent field coordinates.
    let points = vec![ch.sample_f128_vec(20), ch.sample_f128_vec(20)];
    let (relocated, _, _) = evaluations(&w.z, &flatten(&points, record_fragments));
    assert_eq!(relocated, sponge::gather_eval_many(&old_record, &points));
    let points = sponge::sponge_relation_points(3, &mut ch);
    let (relocated, _, _) = evaluations(&w.z, &flatten(&points, sponge_fragments));
    assert_eq!(relocated, sponge::gather_eval_many(&old_sponge, &points));
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
    let run = |reference| {
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
        } else {
            witness(&setup, &inputs)
        };
        let prepared = commit(&setup, w);
        let mut ch = FsChallenger::new(b"hybrid-stripes-proof-identity-v1");
        let core = prove_core_with_packer(&setup, prepared, &mut ch, |z, m, k_log| {
            if reference {
                // Independent logical Boolean oracle, including every
                // padding bit. The complete proof and replay state must
                // remain identical, beyond just the local byte layout.
                let bits: Vec<_> = (0..1 << m).map(|p| layout::bit(z, p)).collect();
                lincheck::pack_z_lincheck(&bits, m, k_log)
            } else {
                lincheck::pack_z_lincheck_from_packed(z, m, k_log)
            }
        });
        let proof = open(&setup, core, &mut ch);
        (bincode::serialize(&proof).unwrap(), ch.sample_f128())
    };
    assert_eq!(run(false), run(true));
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
    let setup = Setup::new(8);
    let mut ch = FsChallenger::new(b"hybrid-arbitrary-relocation-v1");
    let sp = sponge::SpongeWitness {
        z_packed: ch.sample_f128_vec(8 * layout::K / 128),
        a_packed: ch.sample_f128_vec(8 * layout::K / 128),
        b_packed: ch.sample_f128_vec(8 * layout::K / 128),
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
    let record = ch.sample_f128_vec(8 * slots::K / 128);
    // Arbitrary high and padding bits exercise every copied limb. Word-copy
    // A rows must still come from SHAKE even when slot input bits disagree.
    let expected = circuit::assemble_reference(&setup, sp, record.clone());
    let got = assemble(&setup, copied, record);
    assert_eq!(got.z, expected.z);
    assert_eq!(got.a, expected.a);
    assert_eq!(got.b, expected.b);
}
