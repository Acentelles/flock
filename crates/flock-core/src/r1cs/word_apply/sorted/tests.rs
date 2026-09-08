use super::*;
use crate::field::F128;

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn check(matrix: &SparseBinaryMatrix, seed: u64) {
    let maps = Program::new_with_sorted_scratch(matrix, false);
    let sorted = Program::new_with_sorted_scratch(matrix, true);
    assert_eq!(sorted, maps, "complete compiled terms must match");
    if let Some(program) = sorted {
        let width = matrix.num_cols / 128;
        if width == 0 {
            return;
        }
        let mut state = seed;
        let input: Vec<_> = (0..3 * width)
            .map(|_| F128 {
                lo: next(&mut state),
                hi: next(&mut state),
            })
            .collect();
        let mut actual = vec![F128::ZERO; input.len()];
        program.apply(&input, &mut actual);
        let mut expected = actual.clone();
        expected.fill(F128::ZERO);
        // Independent per-edge GF(2) evaluation, including duplicate entries.
        for (input, output) in input
            .chunks_exact(width)
            .zip(expected.chunks_exact_mut(width))
        {
            for (row, edges) in matrix.rows.iter().enumerate() {
                let mut parity = 0;
                for &column in edges {
                    let word = input[column / 128];
                    let limb = if column % 128 < 64 { word.lo } else { word.hi };
                    parity ^= (limb >> (column % 64)) & 1;
                }
                let out = &mut output[row / 128];
                if row % 128 < 64 {
                    out.lo |= parity << (row % 64);
                } else {
                    out.hi |= parity << (row % 64);
                }
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn sorted_record_program_complete_terms_and_arbitrary_inputs_match() {
    let mut state = 0xc61a_7001_f00d_8011;
    for k in [128, 256, 1024, 4096] {
        for shift in [0, 1, 63, 64, 65, 127, 128, k - 1] {
            let rows = (0..k)
                .map(|row| {
                    if row % 29 == 0 {
                        return vec![];
                    }
                    let cancelled = next(&mut state) as usize % k;
                    let mut edges = vec![(row + shift) % k, (row + 3) % k, 7, cancelled, cancelled];
                    if row % 2 == 0 {
                        edges.reverse();
                    }
                    edges
                })
                .collect();
            check(
                &SparseBinaryMatrix {
                    num_rows: k,
                    num_cols: k,
                    rows,
                },
                state,
            );
        }
    }
}

#[test]
fn sorted_record_program_broadcast_threshold_and_signed_windows_match() {
    let k = 512;
    let mut matrix = SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k)
            .map(|row| vec![(row + 1) % k, (row + k - 1) % k])
            .collect(),
    };
    // Exactly three vs four surviving occurrences, with input-order noise
    // and duplicates that must disappear before the broadcast decision.
    for row in 20..23 {
        matrix.rows[row].extend([300, 300, 300]);
    }
    for row in 30..34 {
        matrix.rows[row].extend([400, 301, 400]);
    }
    let program = Program::new_with_sorted_scratch(&matrix, true).unwrap();
    assert!(program.rows[0].iter().any(|term| matches!(term,
        Term::Broadcast { word: 2, bit: 45, mask } if *mask == (15u128 << 30))));
    assert!(!program.rows[0].iter().any(|term| matches!(
        term,
        Term::Broadcast {
            word: 2,
            bit: 44,
            ..
        }
    )));
    assert!(program.rows[0].iter().any(|term| matches!(
        term,
        Term::Window {
            word: -1,
            shift: 127,
            ..
        }
    )));
    check(&matrix, 0xc61b_0001);
}

#[test]
fn sorted_record_program_fallback_and_invalid_shapes_match() {
    for k in [0, 1, 127, 128, 256, 8192] {
        let empty = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: vec![vec![]; k],
        };
        check(&empty, 0xc61c_0011);
        let mut non_square = empty.clone();
        non_square.num_cols += 1;
        check(&non_square, 0xc61c_0012);
        let mut missing_row = empty.clone();
        missing_row.rows.push(vec![]);
        check(&missing_row, 0xc61c_0013);
        if k > 0 {
            let mut invalid = empty;
            invalid.rows[k - 1].push(k);
            check(&invalid, 0xc61c_0014);
        }
    }
    let k = 8192;
    for edge_count in [4095, 4096, 4097, 8192] {
        let mut state = 0xc61c_1111;
        let matrix = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k)
                .map(|row| {
                    if row < edge_count {
                        vec![next(&mut state) as usize % k]
                    } else {
                        vec![]
                    }
                })
                .collect(),
        };
        assert!(Program::new_with_sorted_scratch(&matrix, false).is_none());
        check(&matrix, state);
    }
}

#[test]
fn sorted_record_program_reuses_scratch_without_retaining_previous_block() {
    let k = 1024;
    let mut matrix = SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k)
            .map(|row| match row / 128 {
                0 | 4 => (0..32)
                    .flat_map(|column| [column, column])
                    .chain([7, (row + 1) % k])
                    .collect(),
                1 | 5 => vec![],
                2 | 6 => vec![row],
                _ => vec![k - 1, (row + k - 1) % k],
            })
            .collect(),
    };
    check(&matrix, 0xc61d_1111);
    // Clear formerly dense blocks, then compile a new matrix through the same
    // entry point. Scratch is proof-local, so no prior wiring can survive.
    matrix.rows[..128].iter_mut().for_each(Vec::clear);
    check(&matrix, 0xc61d_2222);
}
