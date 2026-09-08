use super::super::hash_to_point_slots::SlotSetup;
use flock_core::{field::F128, r1cs::word_apply::Program};

#[test]
fn hybrid_sorted_record_program_matches_complete_slot_matrices() {
    let setup = SlotSetup::new(8);
    for matrix in [&setup.r1cs.a_0, &setup.r1cs.b_0] {
        let original = Program::new_with_sorted_scratch(matrix, false).unwrap();
        let candidate = Program::new_with_sorted_scratch(matrix, true).unwrap();
        // This compares every term, coefficient mask and term order for the
        // actual public circuit, not only one honest witness's evaluations.
        assert_eq!(candidate, original);
        let width = matrix.num_rows / 128;
        let mut state = 0xc61e_f001_babe_a118u64;
        let input: Vec<_> = (0..3 * width)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                F128 {
                    lo: state,
                    hi: !state.rotate_left(19),
                }
            })
            .collect();
        let mut expected = vec![F128::ZERO; input.len()];
        let mut actual = expected.clone();
        original.apply(&input, &mut expected);
        candidate.apply(&input, &mut actual);
        assert_eq!(actual, expected);
    }
}
