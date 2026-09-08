//! Aligned-state-v2 layout and direct-generation regressions. The layout
//! tests retain the C50 independent row/column and MLE checks.
use super::super::hash_to_point_slots as slots;
use super::*;
use flock_core::challenger::{Challenger, FsChallenger};

#[test]
fn hybrid_aligned_matches_frozen_c50_descriptor_and_maps() {
    use super::super::keccak3;
    assert_eq!(layout::PROFILE, "aligned-state-v2");
    assert_eq!(layout::K, 524288);
    assert_eq!(layout::END, 523264);
    let setup = Setup::new(8);
    let mut h = blake3::Hasher::new();
    h.update(b"aerie/hybrid-keccak-f1600-24-slot-v2/256-256-128-slot-segments/aligned-state-subcubes/copy-big-endian-words");
    h.update(&setup.r1cs.statement_digest());
    h.update(&setup.slots.r1cs.statement_digest());
    for old in 0..slots::K {
        let expected = c50::c50_record_position(old);
        assert_eq!(layout::record_position(old), expected);
        h.update(&(expected.unwrap_or(usize::MAX) as u64).to_le_bytes());
    }
    for block in 0..4 {
        for old in 0..keccak3::K {
            let expected = c50::c50_sponge_position(block, old);
            assert_eq!(layout::sponge_position(block, old), expected);
            h.update(&(expected.unwrap_or(usize::MAX) as u64).to_le_bytes());
        }
    }
    for slot in 0..slots::SLOTS {
        for bit in 0..16 {
            let expected = c50::c50_word_source(slot, bit);
            assert_eq!(layout::word_source(slot, bit), expected);
            h.update(&(expected as u64).to_le_bytes());
        }
    }
    assert_eq!(setup.descriptor, *h.finalize().as_bytes());
    let mut reference = FsChallenger::new(b"hybrid-aligned-c50-binding");
    reference.observe_label(b"aerie-hybrid-compact-circuit-v2");
    reference.observe_bytes(h.finalize().as_bytes());
    let mut actual = FsChallenger::new(b"hybrid-aligned-c50-binding");
    setup.bind(&mut actual);
    assert_eq!(actual.sample_f128(), reference.sample_f128());
}

#[test]
fn hybrid_aligned_direct_preserves_state_padding_and_record_boundaries() {
    let inputs = super::tests::fixtures(8);
    let setup = Setup::new(inputs.len());
    let original =
        sponge::sponge_witness_without_lincheck(&sponge::SpongeSetup::new(inputs.len()), &inputs);
    let direct = compact_sponge_witness(&setup, &inputs);
    let mut ch = FsChallenger::new(b"hybrid-aligned-direct-arbitrary-padding");
    // Every retained inactive slot and high target bit is arbitrary. The
    // direct builder may clear only omitted cells and SHAKE state padding.
    let record = ch.sample_f128_vec(inputs.len() * slots::K / 128);
    let expected = circuit::assemble_reference(&setup, original, record.clone());
    let actual = assemble_direct(&setup, direct, record.clone());
    assert_eq!(actual.z, expected.z);
    assert_eq!(actual.a, expected.a);
    assert_eq!(actual.b, expected.b);
    for rec in 0..inputs.len() {
        for state in 0..20 {
            let base = rec * layout::K + 98304 + state * 2048;
            for bit in 1600..2048 {
                assert!(!layout::bit(&actual.z, base + bit));
                assert!(!layout::bit(&actual.a, base + bit));
                assert!(!layout::bit(&actual.b, base + bit));
            }
        }
        for plane in [0, 15, 64, 96, 111, 112, 119] {
            for slot in [255, 256, 511, 512, 611, 612, 639, 640, 767, 768, 1023] {
                let old = plane * 1024 + slot;
                if let Some(new) = c50::c50_record_position(old) {
                    assert_eq!(
                        layout::bit(&actual.z, rec * layout::K + new),
                        layout::bit(&record, rec * slots::K + old),
                    );
                }
            }
        }
    }
}

#[test]
fn hybrid_aligned_nonzero_state_padding_is_rejected() {
    let inputs = super::tests::fixtures(8);
    let setup = Setup::new(inputs.len());
    let witness = super::tests::direct_witness(&setup, &inputs);
    let bytes = |packed: &[F128]| -> Vec<u8> {
        packed
            .iter()
            .flat_map(|w| w.lo.to_le_bytes().into_iter().chain(w.hi.to_le_bytes()))
            .collect()
    };
    let a = bytes(&witness.a);
    let b = bytes(&witness.b);
    let mut c = bytes(&witness.z);
    let run = |c: &[u8]| {
        let mut prover = FsChallenger::new(b"hybrid-aligned-padding-rejection");
        let (proof, _) = zerocheck::prove_packed_padded(
            &a,
            &b,
            c,
            setup.r1cs.m,
            &setup.r1cs.padding_spec(),
            &mut prover,
        );
        let mut verifier = FsChallenger::new(b"hybrid-aligned-padding-rejection");
        zerocheck::verify(setup.r1cs.m, &proof, &mut verifier)
    };
    run(&c).unwrap();
    // This retained state-padding cell lies inside END, so the actual prover
    // padding optimization must not omit its C = z contribution.
    let bit = 7 * layout::K + 98304 + 19 * 2048 + 2047;
    assert_eq!(a[bit / 8] >> (bit % 8) & 1, 0);
    assert_eq!(b[bit / 8] >> (bit % 8) & 1, 0);
    c[bit / 8] ^= 1 << (bit % 8);
    assert!(run(&c).is_err());
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

// Address formulas copied from published Flock C50 654996687a2b5047.
// Keep the literal reference independent of the new shared position helpers.
mod c50 {
    use super::super::super::{hash_to_point_slots as slots, keccak3};
    const TARGET: usize = 81920;
    const CONST: usize = 90112;
    const STATES: usize = 98304;
    const STATE_STRIDE: usize = 2048;
    const T_BASE: usize = 139264;

    pub(super) fn c50_record_position(old: usize) -> Option<usize> {
        if old == slots::Z_CONST_POS {
            return Some(CONST);
        }
        let (plane, slot) = (old / 1024, old % 1024);
        if plane < 112 && slot < 640 {
            let (base, width, offset) = if slot < 512 {
                ((slot / 256) * 32768, 256, slot % 256)
            } else {
                (2 * 32768, 128, slot - 512)
            };
            Some(base + plane * width + offset)
        } else if (112..120).contains(&plane) {
            Some(TARGET + (plane - 112) * 1024 + slot)
        } else {
            None
        }
    }

    pub(super) fn c50_sponge_position(block: usize, old: usize) -> Option<usize> {
        if old == keccak3::Z_CONST {
            return Some(CONST);
        }
        if old < 6 * 2048 {
            let slot = old / 2048;
            let permutation = 4 * (slot / 2) + block;
            let bit = old % 2048;
            (permutation < 10 && bit < 1600)
                .then_some(STATES + (2 * permutation + slot % 2) * STATE_STRIDE + bit)
        } else if (keccak3::T_PACKED_BIT_BASE..keccak3::USEFUL_BITS).contains(&old) {
            let offset = old - keccak3::T_PACKED_BIT_BASE;
            let permutation = 4 * (offset / 38400) + block;
            (permutation < 10).then_some(T_BASE + permutation * 38400 + offset % 38400)
        } else {
            None
        }
    }

    pub(super) fn c50_word_source(slot: usize, bit: usize) -> usize {
        assert!(slot < slots::SLOTS && bit < 16);
        STATES + (2 * (1 + slot / 68) + 1) * STATE_STRIDE + 16 * (slot % 68) + (bit ^ 8)
    }
}
