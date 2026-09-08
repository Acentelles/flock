//! Allocation-light equivalent of the ordered-map compiler.
//!
//! Two scratch vectors retain their capacity across all 128-row blocks.
//! Sorting and XOR reduction reproduce ascending column order, duplicate
//! cancellation, then ascending signed-window order exactly. Final term
//! vectors remain owned by the program; no matrix or witness is cached.

use super::{Program, SparseBinaryMatrix, Term};

pub(super) fn compile(matrix: &SparseBinaryMatrix) -> Option<Program> {
    if matrix.num_rows != matrix.num_cols
        || matrix.rows.len() != matrix.num_rows
        || !matrix.num_rows.is_multiple_of(128)
    {
        return None;
    }
    let mut nnz = 0usize;
    let mut operations = 0usize;
    let mut rows = Vec::with_capacity(matrix.num_rows / 128);
    let mut columns = Vec::<(usize, u8)>::new();
    let mut windows = Vec::<(isize, u128)>::new();
    for block in matrix.rows.chunks_exact(128) {
        columns.clear();
        windows.clear();
        for (bit, row) in block.iter().enumerate() {
            nnz += row.len();
            for &column in row {
                if column >= matrix.num_cols {
                    return None;
                }
                columns.push((column, bit as u8));
            }
        }
        columns.sort_unstable_by_key(|&(column, _)| column);
        let mut terms = Vec::new();
        let mut at = 0;
        while at < columns.len() {
            let column = columns[at].0;
            let mut mask = 0u128;
            while at < columns.len() && columns[at].0 == column {
                mask ^= 1u128 << columns[at].1;
                at += 1;
            }
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
                    windows.push((column as isize - bit as isize, 1u128 << bit));
                }
            }
        }
        windows.sort_unstable_by_key(|&(start, _)| start);
        let mut at = 0;
        while at < windows.len() {
            let start = windows[at].0;
            let mut mask = 0u128;
            while at < windows.len() && windows[at].0 == start {
                mask ^= windows[at].1;
                at += 1;
            }
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
        // Preserve the original exact fallback boundary, including its
        // per-block early exit after the cumulative edge count reaches 4096.
        if nnz >= 4096 && operations * 4 > nnz {
            return None;
        }
    }
    if operations * 4 > nnz {
        return None;
    }
    Some(Program { rows })
}

#[cfg(test)]
mod tests;
