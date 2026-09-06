//! Compile a sparse binary matrix into masked packed-word operations.
//!
//! This is an exact GF(2) regrouping for arbitrary witnesses. Repeated
//! columns become broadcasts of the actual input bit (never assumed one).
//! Translated columns become aligned or shifted windows. The output mask
//! records exactly which rows contain each edge, including padding rows.

use super::SparseBinaryMatrix;
use crate::field::F128;
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug)]
enum Term {
    Aligned { word: usize, mask: u128 },
    Window { word: isize, shift: u32, mask: u128 },
    Broadcast { word: usize, bit: u32, mask: u128 },
}

impl Term {
    fn evaluate(&self, input: &[F128]) -> u128 {
        let word = |i: usize| {
            input
                .get(i)
                .map_or(0, |x| u128::from(x.lo) | (u128::from(x.hi) << 64))
        };
        match *self {
            Self::Aligned { word: i, mask } => word(i) & mask,
            Self::Window {
                word: i,
                shift,
                mask,
            } => {
                // An edge at the first/last word can straddle the block.
                // Out-of-block bits are zero and never selected by the mask.
                ((word(i as usize) >> shift) | (word((i + 1) as usize) << (128 - shift))) & mask
            }
            Self::Broadcast { word: i, bit, mask } => {
                mask & 0u128.wrapping_sub((word(i) >> bit) & 1)
            }
        }
    }
}

pub(super) struct Program {
    rows: Vec<Vec<Term>>,
}

impl Program {
    pub(super) fn new(matrix: &SparseBinaryMatrix) -> Option<Self> {
        if matrix.num_rows != matrix.num_cols
            || matrix.rows.len() != matrix.num_rows
            || !matrix.num_rows.is_multiple_of(128)
        {
            return None;
        }
        let mut nnz = 0usize;
        let mut operations = 0usize;
        let mut rows = Vec::with_capacity(matrix.num_rows / 128);
        for block in matrix.rows.chunks_exact(128) {
            let mut columns = BTreeMap::<usize, u128>::new();
            for (bit, row) in block.iter().enumerate() {
                nnz += row.len();
                for &column in row {
                    if column >= matrix.num_cols {
                        return None;
                    }
                    // Duplicate entries cancel exactly as in the CSR evaluator.
                    *columns.entry(column).or_default() ^= 1u128 << bit;
                }
            }
            let mut windows = BTreeMap::<isize, u128>::new();
            let mut terms = Vec::new();
            for (column, mut mask) in columns {
                if mask.count_ones() >= 4 {
                    terms.push(Term::Broadcast {
                        word: column / 128,
                        bit: (column % 128) as u32,
                        mask,
                    });
                } else {
                    while mask != 0 {
                        let bit = mask.trailing_zeros();
                        mask &= mask - 1;
                        *windows.entry(column as isize - bit as isize).or_default() ^= 1u128 << bit;
                    }
                }
            }
            for (start, mask) in windows {
                let word = start.div_euclid(128);
                let shift = start.rem_euclid(128) as u32;
                if shift == 0 {
                    terms.push(Term::Aligned {
                        word: word as usize,
                        mask,
                    });
                } else {
                    terms.push(Term::Window { word, shift, mask });
                }
            }
            operations += terms.len();
            rows.push(terms);
            // Irregular matrices keep the existing transposed CSR kernel.
            // Stop compilation early once enough edges expose that shape.
            if nnz >= 4096 && operations * 4 > nnz {
                return None;
            }
        }
        if operations * 4 > nnz {
            return None;
        }
        Some(Self { rows })
    }

    pub(super) fn apply(&self, input: &[F128], output: &mut [F128]) {
        assert_eq!(input.len(), output.len());
        let width = self.rows.len();
        assert!(width != 0 && input.len().is_multiple_of(width));
        output
            .par_chunks_mut(width)
            .zip(input.par_chunks(width))
            .for_each(|(out, input)| {
                for (out, terms) in out.iter_mut().zip(&self.rows) {
                    let value = terms
                        .iter()
                        .fold(0u128, |sum, term| sum ^ term.evaluate(input));
                    *out = F128 {
                        lo: value as u64,
                        hi: (value >> 64) as u64,
                    };
                }
            });
    }

    #[cfg(test)]
    pub(super) fn operations(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(matrix: &SparseBinaryMatrix, input: &[F128]) -> Vec<F128> {
        let width = matrix.num_rows / 128;
        let mut out = vec![F128::ZERO; input.len()];
        for (input, out) in input.chunks(width).zip(out.chunks_mut(width)) {
            for (row, columns) in matrix.rows.iter().enumerate() {
                let value = columns.iter().fold(0, |value, &column| {
                    let word = input[column / 128];
                    let limb = if column % 128 < 64 { word.lo } else { word.hi };
                    value ^ ((limb >> (column % 64)) & 1)
                });
                if row % 128 < 64 {
                    out[row / 128].lo |= value << (row % 64);
                } else {
                    out[row / 128].hi |= value << (row % 64);
                }
            }
        }
        out
    }

    #[test]
    fn word_operations_match_scalar_oracle_on_arbitrary_bits() {
        let k = 1024;
        for shift in [0, 1, 63, 64, 65, 127, 128, 129, 1023] {
            let matrix = SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: (0..k)
                    .map(|i| {
                        if i % 29 == 0 {
                            return vec![];
                        }
                        // Duplicate edges cancel; repeated input is not a constant-one assumption.
                        vec![(i + shift) % k, (i + 3) % k, 7, (i + 17) % k, (i + 17) % k]
                    })
                    .collect(),
            };
            let program = Program::new(&matrix).expect("translated matrix compresses");
            for blocks in [1, 3, 8, 65] {
                let mut state = 0x9e37_79b9_7f4a_7c15u128;
                let input: Vec<_> = (0..blocks * k / 128)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        F128 {
                            lo: state as u64,
                            hi: (state >> 64) as u64,
                        }
                    })
                    .collect();
                let mut actual = vec![F128::ZERO; input.len()];
                program.apply(&input, &mut actual);
                assert_eq!(
                    actual,
                    reference(&matrix, &input),
                    "shift={shift}, blocks={blocks}"
                );
            }
        }
    }

    #[test]
    fn boundary_windows_and_broadcasts_preserve_every_basis_bit() {
        let k = 256;
        let matrix = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k)
                .map(|i| vec![(i + 1) % k, (i + k - 1) % k, 127, 255])
                .collect(),
        };
        let program = Program::new(&matrix).unwrap();
        for bit in 0..k {
            let mut input = vec![F128::ZERO; k / 128];
            if bit % 128 < 64 {
                input[bit / 128].lo = 1 << (bit % 64);
            } else {
                input[bit / 128].hi = 1 << (bit % 64);
            }
            let mut out = vec![F128::ZERO; input.len()];
            program.apply(&input, &mut out);
            assert_eq!(out, reference(&matrix, &input), "basis bit {bit}");
        }
    }

    #[test]
    fn empty_and_irregular_matrices_have_explicit_paths() {
        let k = 128;
        let empty = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: vec![vec![]; k],
        };
        let program = Program::new(&empty).unwrap();
        let input = vec![
            F128 {
                lo: u64::MAX,
                hi: u64::MAX
            };
            8
        ];
        let mut out = input.clone();
        program.apply(&input, &mut out);
        assert!(out.iter().all(|x| *x == F128::ZERO));
        let mut permutation: Vec<_> = (0..k).collect();
        let mut state = 0x936c_afe1_u64;
        for i in (1..k).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            permutation.swap(i, state as usize % (i + 1));
        }
        let random = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: permutation.into_iter().map(|i| vec![i]).collect(),
        };
        assert!(Program::new(&random).is_none());
        let mut invalid = empty;
        invalid.rows[0].push(k);
        assert!(Program::new(&invalid).is_none());
    }
}
