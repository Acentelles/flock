use super::*;

#[test]
fn weighted_tail_has_exact_coordinate_weights() {
    let (tail, scale) = fingerprint_tail();
    for coordinate in 0..16 {
        let weight = tail.iter().enumerate().fold(scale, |acc, (j, &p)| {
            acc * if coordinate & (1 << (3 - j)) == 0 {
                F128::ONE + p
            } else {
                p
            }
        });
        assert_eq!(
            weight,
            F128 {
                lo: 1 << coordinate,
                hi: 0
            }
        );
    }
}

#[test]
fn fingerprint_matches_reference_with_nonzero_padding() {
    use flock_core::challenger::FsChallenger;
    let mut ch = FsChallenger::new(b"record-fingerprint-independent-layout-test");
    // Arbitrary packed bits, including coordinate fifteen in the Z_H region.
    // No relation-positive witness or honest zero-padding assumption is used.
    let mut packed: Vec<_> = (0..(1 << (slots::K_LOG - 7)))
        .map(|_| ch.sample_f128())
        .collect();
    for leaf in 0..512 {
        let address = slots::zh_position(leaf, 15);
        if address % 128 < 64 {
            packed[address / 128].lo |= 1 << (address % 64);
        } else {
            packed[address / 128].hi |= 1 << (address % 64);
        }
    }
    for r in [
        vec![F128::ZERO; 9],
        vec![F128::ONE; 9],
        ch.sample_f128_vec(9),
    ] {
        let reference = super::super::hash_to_point_sponge::gather_eval_many_reference(
            &packed,
            &zh_fingerprint_reference_points(0, &r),
        );
        let expected = reference
            .iter()
            .enumerate()
            .fold(F128::ZERO, |sum, (i, &v)| {
                sum + F128 { lo: 1 << i, hi: 0 } * v
            });
        let points = zh_fingerprint_points(0, &r);
        let values = super::super::packed_mle::evaluate_packed(&packed, &points);
        assert_eq!(values.len(), FINGERPRINT_CLAIMS);
        assert_eq!(zh_fingerprint_value(&values), expected);
        if cfg!(feature = "compact-fingerprint") {
            assert_ne!(values[1], F128::ZERO, "padding contribution is exercised");
            assert_ne!(fingerprint_tail().1 * values[0], expected);
        }
    }
}

#[test]
fn compact_profile_is_bound_before_fingerprint_challenges() {
    use flock_core::challenger::FsChallenger;
    let mut selected = FsChallenger::new(b"record-profile-separation");
    let mut original = selected.clone();
    bind_fingerprint_profile(&mut selected);
    let selected = selected.sample_f128();
    let original = original.sample_f128();
    assert_eq!(selected == original, !cfg!(feature = "compact-fingerprint"));
}
