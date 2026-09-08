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
    let sp = sponge::sponge_witness(&sponge::SpongeSetup::new(inputs.len()), inputs);
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
