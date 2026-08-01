//! **Fancy jagged** — the §6 variant of Hemo–Jue–Rabinovich–Roh–Rothblum
//! ("Jagged Polynomial Commitments", ePrint 2025/917, EUROCRYPT 2026), for the
//! case where columns are grouped into **tables**: runs of columns that share a
//! height. The verifier's evaluation of the sparse→dense weight then costs
//! `O(#tables)` instead of `O(#columns)`.
//!
//! Standalone kernel, deliberately not wired into the commitment path — same
//! posture as [`super::jagged`] ("the packing-agnostic kernel … does not wire
//! into ring-switch / ligerito"). See "Adoption cost" below for why wiring is
//! not a small change.
//!
//! ## Why grouping columns needs a different layout
//!
//! Basic jagged flattens column-major: within column `y`, dense index
//! `i = t_{y−1} + row`. Grouping columns into a table would then put column
//! `col` of table `y` at `i = t_{y−1} + col·h_y + row` — and `h_y` is an
//! arbitrary height, so `col·h_y` is a general multiplication, which no
//! small branching program can check.
//!
//! Fancy jagged flattens **row-major within a table**:
//!
//! ```text
//!   i = t_{y−1} + row·2^{c_y} + col          (table y is 2^{c_y} columns wide)
//! ```
//!
//! Now the stride is a power of two, so `row·2^{c_y}` is a *shift* by the
//! constant `c_y` — and the whole relation is checkable bit-by-bit. **That is
//! the entire trick**: the layout change is what buys the per-table verifier
//! cost, which is also why table widths must be powers of two.
//!
//! ## The width-6 branching program
//!
//! `g_u(row, col, i, t_prev, t_next) = [ i = t_prev + row·2^u + col ∧ i < t_next ]`
//!
//! read once, LSB→MSB. Against basic jagged's width-4 program it differs in
//! exactly the two ways §6 names:
//!
//! * it adds **three** bits per layer (`t_prev`, the shifted `row`, and `col`)
//!   rather than two, so the carry reaches 2 and the register is a **trit**;
//! * the `2^u` factor is just an index shift, `u` being a per-table constant.
//!
//! With the one inequality bit that is `3 × 2 = 6` states. Branching-program
//! evaluation is quadratic in width, so each evaluation costs `(6/4)² = 2.25×`
//! a basic-jagged one — the trade §6 calls a mild caveat, against a factor of
//! `#columns / #tables`.
//!
//! ## Non-power-of-two widths
//!
//! Do not pad. [`FancyJaggedParams::from_tables`] decomposes a width-`W` table
//! into the `popcount(W)` power-of-two tables named by `W`'s set bits (§6's
//! "a table of width 9 … one of width 8 and one of width 1"), so the verifier
//! cost is `Σ_T log₂(#col(T))`.
//!
//! **This re-indexes the polynomial.** After decomposition, `tab` enumerates
//! *physical* sub-tables and `col` is an offset within one, so a claim stated
//! against the original `(table, row, col)` coordinates is not a claim against
//! these. Callers own that translation.
//!
//! ## Adoption cost, for flock specifically
//!
//! Two things, one small and one not:
//!
//! * The registry already *has* the table structure — a slot is `k_t`
//!   consecutive columns of height `n_t`, which is exactly a table — and
//!   `jagged::assist_boundaries` currently flattens it away into per-column
//!   boundary pairs. So the configuration is free.
//! * But the dense stack is column-major (`UnionInstance::compact_witness`),
//!   and fancy jagged needs row-major-within-table. That changes what `q` is,
//!   hence the commitment.
//!
//! The prize is not a faster assist but **no assist**: §6 notes the verifier
//! "can compute the latter on its own, *or* employ a similar jagged assist" —
//! the assist is the fallback for when per-column evaluation is too expensive
//! for the verifier, and per-table it is not. Composed with the Frobenius
//! decomposition, `Ŵ(ρ) = Σ_j c_j·f̂(z^{2^j}, ρ)` becomes directly evaluable,
//! so the 128·K-statement delegation — 88% of the measured prove gap — has
//! nothing left to do. Note §6 itself claims only a verifier improvement; the
//! prover's own weight pass stays `O(area)`, "similarly to Section 3".

use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::pcs::jagged::{int_bit, point_bit};

/// One physical table: `2^log_width` columns, `height` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    pub log_width: u32,
    pub height: u64,
}

/// Fancy-jagged configuration — §6's `(t_y, c_y)_y`, plus the variable counts.
///
/// Table `y` occupies dense indices `[table_prefix_sums[y], table_prefix_sums[y+1])`,
/// is `2^{log_widths[y]}` columns wide, and holds `heights[y]` rows.
#[derive(Clone, Debug)]
pub struct FancyJaggedParams {
    /// `log2` bound on rows per table (`row` variable count).
    pub n: usize,
    /// `log2` of the number of tables (`tab` variable count).
    pub k: usize,
    /// `log2` bound on table width (`col` variable count) = `max_y c_y`.
    pub c: usize,
    /// `log2` of the dense area bound: `Σ_y 2^{c_y}·h_y ≤ 2^m`.
    pub m: usize,
    pub log_widths: Vec<u32>,
    pub heights: Vec<u64>,
    /// Length `2^k + 1`; `[y]` is table `y`'s start, `[y+1]` its end.
    pub table_prefix_sums: Vec<u64>,
}

impl FancyJaggedParams {
    /// Build from physical tables, zero-padding the table count to `2^k`.
    /// Every width must already be a power of two — use
    /// [`Self::from_tables`] to get there from arbitrary widths.
    pub fn new(tables: &[Table], n: usize, m: usize) -> Self {
        assert!(!tables.is_empty(), "need at least one table");
        let k = tables.len().next_power_of_two().trailing_zeros() as usize;
        let n_tab = 1usize << k;
        let c = tables.iter().map(|t| t.log_width as usize).max().unwrap();
        let mut log_widths = vec![0u32; n_tab];
        let mut heights = vec![0u64; n_tab];
        let mut table_prefix_sums = Vec::with_capacity(n_tab + 1);
        let mut acc = 0u64;
        table_prefix_sums.push(0);
        for (y, slot) in log_widths.iter_mut().enumerate() {
            let (lw, h) = match tables.get(y) {
                Some(t) => (t.log_width, t.height),
                None => (0, 0), // padding table: width 1, height 0, empty
            };
            assert!(h <= 1u64 << n, "table {y} height {h} exceeds 2^{n}");
            *slot = lw;
            heights[y] = h;
            acc += (1u64 << lw) * h;
            table_prefix_sums.push(acc);
        }
        assert!(acc <= 1u64 << m, "dense area {acc} exceeds 2^{m}");
        Self {
            n,
            k,
            c,
            m,
            log_widths,
            heights,
            table_prefix_sums,
        }
    }

    /// Build from **logical** `(width, height)` tables of arbitrary width,
    /// decomposing each into the power-of-two tables named by its set bits,
    /// widest first (§6: width 9 → 8 then 1).
    ///
    /// Re-indexes the polynomial — see the module docs.
    pub fn from_tables(logical: &[(u64, u64)], n: usize, m: usize) -> Self {
        let mut phys = Vec::new();
        for &(width, height) in logical {
            assert!(width > 0, "table width must be positive");
            for bit in (0..u64::BITS).rev() {
                if (width >> bit) & 1 == 1 {
                    phys.push(Table {
                        log_width: bit,
                        height,
                    });
                }
            }
        }
        Self::new(&phys, n, m)
    }

    /// Total nonzero entries `Σ_y 2^{c_y}·h_y`.
    pub fn area(&self) -> u64 {
        *self.table_prefix_sums.last().unwrap()
    }

    /// Number of physical tables actually carrying data — the quantity the
    /// verifier's cost is proportional to, against basic jagged's column count.
    pub fn live_tables(&self) -> usize {
        self.heights.iter().filter(|&&h| h > 0).count()
    }

    /// The bijection `i ↦ (tab, row, col)` for `i < area()`, i.e.
    /// `i = t_{tab} + row·2^{c_tab} + col`.
    pub fn unrank(&self, i: u64) -> (usize, u64, u64) {
        debug_assert!(i < self.area());
        let tab = self.table_prefix_sums.partition_point(|&t| t <= i) - 1;
        let off = i - self.table_prefix_sums[tab];
        let w = 1u64 << self.log_widths[tab];
        (tab, off / w, off % w)
    }

    /// Inverse of [`Self::unrank`]; `None` if out of the table's extent.
    pub fn rank(&self, tab: usize, row: u64, col: u64) -> Option<u64> {
        if tab >= 1usize << self.k
            || row >= self.heights[tab]
            || col >= 1u64 << self.log_widths[tab]
        {
            return None;
        }
        Some(self.table_prefix_sums[tab] + (row << self.log_widths[tab]) + col)
    }
}

// ---------------------------------------------------------------------------
// The width-6 branching program
// ---------------------------------------------------------------------------

/// `state = carry + 3·less_than`, carry ∈ {0,1,2}.
const STATE_INITIAL: usize = 0; // carry 0, not-yet-less
const STATE_SUCCESS: usize = 3; // carry 0, less-than established
const N_STATES: usize = 6;

/// One layer of `g_u(row, col, i, t_prev, t_next) = [i = t_prev + row·2^u + col
/// ∧ i < t_next]`, reading LSB→MSB. `row` is the already-shifted row bit (bit
/// `layer − u` of the row). Returns the next state, or `None` on the rejecting
/// sink (the addition disagrees with `i` at this bit).
#[inline]
fn transition(
    row: bool,
    col: bool,
    index: bool,
    prev: bool,
    next: bool,
    state: usize,
) -> Option<usize> {
    let carry = state % 3;
    let less = state / 3;
    // Three addends plus the carry: max 1+1+1+2 = 5, so the carry out is ≤ 2 —
    // this is the trit §6 calls for.
    let sum = row as usize + col as usize + prev as usize + carry;
    if (index as usize) != (sum & 1) {
        return None;
    }
    let new_carry = sum >> 1;
    debug_assert!(new_carry <= 2);
    // i < t_next, decided LSB→MSB: equal bits defer to the lower decision,
    // differing bits let the higher one rule.
    let new_less = if index == next { less } else { next as usize };
    Some(new_carry + 3 * new_less)
}

/// Bit `layer` of a row point shifted up by `u` — i.e. bit `layer − u` of the
/// row, which is how the `·2^u` factor enters.
#[inline]
fn shifted_row_bit(z_row: &[F128], layer: usize, u: usize) -> F128 {
    if layer >= u {
        point_bit(z_row, layer - u)
    } else {
        F128::ZERO
    }
}

/// Multilinear extension `ĝ_u(z_row, z_col, z_index, t_prev, t_next)` by the
/// Holmgren–Rothblum layer DP over the 6 reachable states, with the boundary
/// coordinates supplied per layer by `pn`. `O(m)` field ops.
fn g_hat_cd(
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
    m: usize,
    u: usize,
    pn: impl Fn(usize) -> (F128, F128),
) -> F128 {
    // dp[s] = weight of reaching the accepting sink from state `s` over the
    // layers already processed. Seed the accepting state, peel MSB→LSB, read
    // off the initial state.
    let mut dp = [F128::ZERO; N_STATES];
    dp[STATE_SUCCESS] = F128::ONE;
    for layer in (0..=m).rev() {
        let (prev, next) = pn(layer);
        let eq = build_eq_table(&[
            shifted_row_bit(z_row, layer, u),
            point_bit(z_col, layer),
            point_bit(z_index, layer),
            prev,
            next,
        ]);
        let mut new_dp = [F128::ZERO; N_STATES];
        for (s, slot) in new_dp.iter_mut().enumerate() {
            let mut acc = F128::ZERO;
            for (idx, &w) in eq.iter().enumerate() {
                let row = idx & 1 != 0;
                let col = (idx >> 1) & 1 != 0;
                let index = (idx >> 2) & 1 != 0;
                let prev_b = (idx >> 3) & 1 != 0;
                let next_b = (idx >> 4) & 1 != 0;
                // `col < 2^u`: this table is only `2^u` wide, so col's bits at
                // positions ≥ u must be zero. Without this the addition would
                // alias — col = 2^u looks exactly like row+1, col = 0 — and the
                // map would stop being a bijection. Dropping the branch (rather
                // than pinning the coordinate) keeps the correct `(1 + z_col_j)`
                // factor on the surviving bit-0 branch, which is what the MLE of
                // a function vanishing on `col_j = 1` requires.
                if col && layer >= u {
                    continue;
                }
                if let Some(out) = transition(row, col, index, prev_b, next_b, s) {
                    acc += w * dp[out];
                }
            }
            *slot = acc;
        }
        dp = new_dp;
    }
    dp[STATE_INITIAL]
}

/// [`g_hat_cd`] at boolean table boundaries.
fn g_hat(
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
    m: usize,
    u: usize,
    t_prev: u64,
    t_next: u64,
) -> F128 {
    g_hat_cd(z_row, z_col, z_index, m, u, |layer| {
        (int_bit(t_prev, layer), int_bit(t_next, layer))
    })
}

/// Evaluate `f̂_{t,c}(z_tab, z_row, z_col, z_index)` at an arbitrary field
/// point — §6's
///
/// ```text
///   f_{t,c} = Σ_y eq(z_tab, y) · Σ_u eq(u, c_y) · ĝ_u(z_row, z_col, i, t_y, t_{y−1})
/// ```
///
/// The configuration is public, so `c_y` is a known constant per table and the
/// inner `Σ_u` collapses to its single live term `u = c_y`.
///
/// Cost `O(#live_tables · m)` — against basic jagged's `O(#columns · m)`, which
/// is the point of the whole variant.
pub fn f_hat_fancy(
    params: &FancyJaggedParams,
    z_tab: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
) -> F128 {
    assert_eq!(z_tab.len(), params.k, "z_tab must span the table vars");
    assert_eq!(z_row.len(), params.n, "z_row must span the row vars");
    assert_eq!(z_col.len(), params.c, "z_col must span the col vars");
    assert_eq!(z_index.len(), params.m, "z_index must span the dense vars");
    let eq_tab = build_eq_table(z_tab);
    let mut acc = F128::ZERO;
    for y in 0..1usize << params.k {
        // Empty tables contribute nothing: t_prev == t_next makes `i < t_next`
        // unsatisfiable for any `i ≥ t_prev`.
        if params.heights[y] == 0 {
            continue;
        }
        acc += eq_tab[y]
            * g_hat(
                z_row,
                z_col,
                z_index,
                params.m,
                params.log_widths[y] as usize,
                params.table_prefix_sums[y],
                params.table_prefix_sums[y + 1],
            );
    }
    acc
}

// ---------------------------------------------------------------------------
// Aligned tables: §6 adapted to a global column index
// ---------------------------------------------------------------------------

/// A table pinned to an **aligned block** of the global column index.
///
/// §6 gives tables their own `tab ∈ {0,1}^k` variable. Flock's claims don't:
/// the ring-switch claim point splits as `[row | chunk]` over the padded
/// witness's BatchMajor address, so `z_col` is a single `k_cols`-bit field over
/// the *global* chunk index and there is no separate table variable.
///
/// The fix is the union layer's own trick — aligned addressing instead of an
/// extra variable. A table occupying the aligned block
/// `[col_offset, col_offset + 2^log_width)` splits the global column as
/// `global = (col_offset >> log_width)·2^log_width + within`, so
///
/// ```text
///   eq(z_col, global) = eq(z_col[..lw], within) · eq(z_col[lw..], col_offset >> lw)
/// ```
///
/// — the low coordinates feed the branching program's `col` slot and the high
/// ones become a known scalar selecting the table. Tables of differing widths
/// coexist, exactly as registry slots of differing `κ_t` do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedTable {
    pub log_width: u32,
    pub height: u64,
    /// First global column; must be a multiple of `2^log_width`.
    pub col_offset: u64,
}

/// Aligned-table configuration over a global column index of `k_cols` bits.
#[derive(Clone, Debug)]
pub struct AlignedParams {
    pub n: usize,
    pub k_cols: usize,
    pub m: usize,
    pub tables: Vec<AlignedTable>,
    /// Dense start of each table, length `tables.len() + 1`.
    pub prefix_sums: Vec<u64>,
}

impl AlignedParams {
    pub fn new(tables: Vec<AlignedTable>, n: usize, k_cols: usize, m: usize) -> Self {
        let mut prefix_sums = Vec::with_capacity(tables.len() + 1);
        let mut acc = 0u64;
        prefix_sums.push(0);
        for t in &tables {
            assert!(
                t.col_offset.is_multiple_of(1u64 << t.log_width),
                "table at {} is not {}-aligned",
                t.col_offset,
                1u64 << t.log_width
            );
            assert!(
                t.col_offset + (1u64 << t.log_width) <= 1u64 << k_cols,
                "table exceeds the global column space"
            );
            assert!(t.height <= 1u64 << n, "table height exceeds 2^n");
            acc += (1u64 << t.log_width) * t.height;
            prefix_sums.push(acc);
        }
        assert!(acc <= 1u64 << m, "dense area {acc} exceeds 2^{m}");
        Self {
            n,
            k_cols,
            m,
            tables,
            prefix_sums,
        }
    }

    pub fn area(&self) -> u64 {
        *self.prefix_sums.last().unwrap()
    }

    /// `i ↦ (global_col, row)` — the **row-major-within-table** bijection that
    /// `UnionInstance::compact_witness_row_major` writes.
    pub fn unrank(&self, i: u64) -> (u64, u64) {
        debug_assert!(i < self.area());
        let t = self.prefix_sums.partition_point(|&s| s <= i) - 1;
        let tab = &self.tables[t];
        let off = i - self.prefix_sums[t];
        let w = 1u64 << tab.log_width;
        (tab.col_offset + (off % w), off / w)
    }
}

/// `f̂(z_row, z_col, z_index)` for an aligned-table configuration:
///
/// ```text
///   Σ_tables eq(z_col[lw..], col_offset >> lw)
///            · ĝ_{lw}(z_row, z_col[..lw], z_index, t_prev, t_next)
/// ```
///
/// Cost `O(#tables · m)` against basic jagged's `O(#columns · m)`.
pub fn f_hat_aligned(
    params: &AlignedParams,
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
) -> F128 {
    assert_eq!(z_row.len(), params.n, "z_row must span the row vars");
    assert_eq!(
        z_col.len(),
        params.k_cols,
        "z_col must span the column vars"
    );
    assert_eq!(z_index.len(), params.m, "z_index must span the dense vars");
    let mut acc = F128::ZERO;
    for (t, tab) in params.tables.iter().enumerate() {
        if tab.height == 0 {
            continue;
        }
        let lw = tab.log_width as usize;
        // High coordinates select this aligned block; a known scalar.
        let mut sel = F128::ONE;
        let prefix = tab.col_offset >> lw;
        for (b, &zc) in z_col[lw..].iter().enumerate() {
            sel *= if (prefix >> b) & 1 == 1 {
                zc
            } else {
                F128::ONE + zc
            };
        }
        if sel.is_zero() {
            continue;
        }
        acc += sel
            * g_hat(
                z_row,
                &z_col[..lw],
                z_index,
                params.m,
                lw,
                params.prefix_sums[t],
                params.prefix_sums[t + 1],
            );
    }
    acc
}

/// The untwisted weight over the dense cube for the **row-major** stack —
/// `W[d] = eq(z_row, row(d))·eq(z_col, col(d))`, zero past the area.
///
/// The row-major mirror of `jagged::build_merged_weight_and_prime`'s inner
/// loop: that one hoists `eq_col` and walks rows, this hoists `eq_row` and
/// walks columns, because a run of `2^lw` consecutive dense indices now shares
/// a row instead of a column.
pub fn build_weight_row_major(
    params: &AlignedParams,
    z_row: &[F128],
    z_col: &[F128],
    out: &mut [F128],
) {
    assert_eq!(out.len(), 1usize << params.m);
    let eq_r = build_eq_table(z_row);
    let eq_c = build_eq_table(z_col);
    out.fill(F128::ZERO);
    for (t, tab) in params.tables.iter().enumerate() {
        let w = 1usize << tab.log_width;
        let base = params.prefix_sums[t] as usize;
        for row in 0..tab.height as usize {
            let r = eq_r[row];
            let dst = &mut out[base + row * w..base + (row + 1) * w];
            let cols = &eq_c[tab.col_offset as usize..tab.col_offset as usize + w];
            for (slot, &c) in dst.iter_mut().zip(cols) {
                *slot = r * c;
            }
        }
    }
}

/// `Ŵ(ρ)` for the aligned-table configuration: the Φ-twisted weight the merged
/// reduction leaves the verifier owing, evaluated directly via the Frobenius
/// decomposition instead of delegated to a `128·K`-statement assist.
///
/// ```text
///   Ŵ(ρ) = Σ_i Σ_j c_{i,j} · f̂(z_row_i^{2^j}, z_col_i^{2^j}, ρ)
/// ```
///
/// Same coefficient/power walk as `jagged::frobenius_statements`: use, then
/// square. Cost `O(128·K · #tables · m)`.
pub fn twisted_weight_aligned(
    params: &AlignedParams,
    claims: &[crate::pcs::jagged::FrobeniusClaim<'_>],
    rho: &[F128],
) -> F128 {
    let mut acc = F128::ZERO;
    for claim in claims {
        assert_eq!(claim.coeffs.len(), 128);
        let mut zr = claim.z_row.to_vec();
        let mut zc = claim.z_col.to_vec();
        for &c in claim.coeffs.iter() {
            if !c.is_zero() {
                acc += c * f_hat_aligned(params, &zr, &zc, rho);
            }
            for x in zr.iter_mut() {
                *x = *x * *x;
            }
            for x in zc.iter_mut() {
                *x = *x * *x;
            }
        }
    }
    acc
}

/// The **Φ-twisted** weight over the dense cube for the row-major stack, plus
/// the merged sumcheck's round-0 prime — the row-major counterpart of
/// `jagged::build_merged_weight_and_prime`.
///
/// `W[d] = Σ_i Φ_i(eq(z_row_i, row(d))·eq(z_col_i, col(d)))`, zero past the
/// area. Where the column-major builder hoists `eq_col` per column segment and
/// streams rows, this hoists `eq_row` per row and streams the table's columns,
/// because a run of `2^lw` consecutive dense indices now shares a row.
///
/// The prime is taken in a second pass rather than fused into the fill: under
/// row-major a width-1 table's rows are single words, so element pairs
/// `(2t, 2t+1)` straddle segment boundaries and per-segment fusion would be
/// wrong. One extra streaming pass is the cheap, safe trade.
pub fn build_weight_row_major_twisted(
    params: &AlignedParams,
    claims: &[(&[F128], &[F128], &[F128])],
    q: &[F128],
) -> (Vec<F128>, (F128, F128)) {
    use rayon::prelude::*;

    let n_total = 1usize << params.m;
    assert_eq!(q.len(), n_total, "q must be the committed stack");
    let tabs: Vec<(Vec<F128>, Vec<F128>, &[F128])> = claims
        .iter()
        .map(|&(zr, zc, t)| (build_eq_table(zr), build_eq_table(zc), t))
        .collect();
    // Pooled and dirty: the per-table fill writes every word of the area (the
    // first claim assigns, later claims accumulate) and the tail is filled
    // explicitly below.
    let mut w = crate::scratch::take_f128(n_total);

    let mut rest: &mut [F128] = &mut w;
    for (t, tab) in params.tables.iter().enumerate() {
        let width = 1usize << tab.log_width;
        let span = width * tab.height as usize;
        let (seg, tail) = rest.split_at_mut(span);
        rest = tail;
        debug_assert_eq!(
            params.prefix_sums[t + 1] - params.prefix_sums[t],
            span as u64
        );
        if span == 0 {
            continue;
        }
        let c0 = tab.col_offset as usize;
        seg.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out)| {
                let mut first = true;
                for (eq_r, eq_c, fold) in tabs.iter() {
                    let r = eq_r[row];
                    let cols = &eq_c[c0..c0 + width];
                    if first {
                        for (slot, &c) in out.iter_mut().zip(cols) {
                            *slot = crate::pcs::ring_switch::fold_one_slot(r * c, fold);
                        }
                        first = false;
                    } else {
                        for (slot, &c) in out.iter_mut().zip(cols) {
                            *slot += crate::pcs::ring_switch::fold_one_slot(r * c, fold);
                        }
                    }
                }
            });
    }
    // The dead tail past the jagged area contributes zero on both sides.
    rest.par_chunks_mut(1 << 16)
        .for_each(|c| c.fill(F128::ZERO));

    let prime = q
        .par_chunks(1 << 14)
        .zip(w.par_chunks(1 << 14))
        .map(|(qc, wc)| {
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for (qp, wp) in qc
                .as_chunks::<2>()
                .0
                .iter()
                .zip(wc.as_chunks::<2>().0.iter())
            {
                u0 += qp[0] * wp[0];
                u2 += (qp[0] + qp[1]) * (wp[0] + wp[1]);
            }
            (u0, u2)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d));
    (w, prime)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic field elements for tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> F128 {
            let mut out = [0u64; 2];
            for w in out.iter_mut() {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                *w = z ^ (z >> 31);
            }
            F128 {
                lo: out[0],
                hi: out[1],
            }
        }
        fn vec(&mut self, len: usize) -> Vec<F128> {
            (0..len).map(|_| self.next()).collect()
        }
    }

    fn sample() -> FancyJaggedParams {
        // 2 tables: (2 cols × 3 rows) then (1 col × 2 rows) = area 8.
        FancyJaggedParams::new(
            &[
                Table {
                    log_width: 1,
                    height: 3,
                },
                Table {
                    log_width: 0,
                    height: 2,
                },
            ],
            2,
            4,
        )
    }

    /// The bijection round-trips and tiles `[0, area)` exactly once.
    #[test]
    fn bijection_is_a_bijection() {
        let p = sample();
        assert_eq!(p.area(), 2 * 3 + 1 * 2);
        let mut seen = vec![false; p.area() as usize];
        for tab in 0..1usize << p.k {
            for row in 0..p.heights[tab] {
                for col in 0..1u64 << p.log_widths[tab] {
                    let i = p.rank(tab, row, col).expect("in extent");
                    assert!(!seen[i as usize], "index {i} hit twice");
                    seen[i as usize] = true;
                    assert_eq!(p.unrank(i), (tab, row, col));
                }
            }
        }
        assert!(seen.into_iter().all(|b| b), "some dense index unmapped");
    }

    /// At boolean points the branching program IS the indicator of the
    /// bijection — the defining property of `f_{t,c}` (§6, Eq. 8).
    #[test]
    fn boolean_points_give_the_indicator() {
        let p = sample();
        let bits = |v: u64, len: usize| -> Vec<F128> {
            (0..len)
                .map(|b| {
                    if (v >> b) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect()
        };
        for tab in 0..1u64 << p.k {
            for row in 0..1u64 << p.n {
                for col in 0..1u64 << p.c {
                    for i in 0..1u64 << p.m {
                        let got = f_hat_fancy(
                            &p,
                            &bits(tab, p.k),
                            &bits(row, p.n),
                            &bits(col, p.c),
                            &bits(i, p.m),
                        );
                        let want = if p.rank(tab as usize, row, col) == Some(i) {
                            F128::ONE
                        } else {
                            F128::ZERO
                        };
                        assert_eq!(
                            got, want,
                            "indicator mismatch at tab={tab} row={row} col={col} i={i}"
                        );
                    }
                }
            }
        }
    }

    /// **The decisive test.** At a random FIELD point the branching-program
    /// assembly must equal the honest multilinear extension of that indicator,
    /// computed by brute force over the whole cube.
    #[test]
    fn field_point_matches_brute_force_mle() {
        let p = sample();
        let mut rng = Rng(0x_FA5C_1A66);
        for trial in 0..4 {
            let z_tab = rng.vec(p.k);
            let z_row = rng.vec(p.n);
            let z_col = rng.vec(p.c);
            let z_idx = rng.vec(p.m);

            let got = f_hat_fancy(&p, &z_tab, &z_row, &z_col, &z_idx);

            // Σ over every boolean (tab,row,col,i) of eq(·)·indicator.
            let (et, er, ec, ei) = (
                build_eq_table(&z_tab),
                build_eq_table(&z_row),
                build_eq_table(&z_col),
                build_eq_table(&z_idx),
            );
            let mut want = F128::ZERO;
            for tab in 0..1usize << p.k {
                for row in 0..1u64 << p.n {
                    for col in 0..1u64 << p.c {
                        if let Some(i) = p.rank(tab, row, col) {
                            want += et[tab] * er[row as usize] * ec[col as usize] * ei[i as usize];
                        }
                    }
                }
            }
            assert_eq!(got, want, "trial {trial}: BP assembly != brute-force MLE");
        }
    }

    /// Eq. 8: `p̂(z_tab,z_row,z_col) = Σ_i q(i)·f̂_{t,c}(…,i)`, with `f̂` taken
    /// at a random field point in the index slot and the dense `q` random.
    #[test]
    fn eq8_reduction_holds() {
        let p = sample();
        let mut rng = Rng(0x_E98_0008);
        let q: Vec<F128> = rng.vec(1usize << p.m);
        let z_tab = rng.vec(p.k);
        let z_row = rng.vec(p.n);
        let z_col = rng.vec(p.c);

        // Left: p̂ at the claim point, p defined from q through the bijection.
        let (et, er, ec) = (
            build_eq_table(&z_tab),
            build_eq_table(&z_row),
            build_eq_table(&z_col),
        );
        let mut lhs = F128::ZERO;
        for tab in 0..1usize << p.k {
            for row in 0..1u64 << p.n {
                for col in 0..1u64 << p.c {
                    if let Some(i) = p.rank(tab, row, col) {
                        lhs += et[tab] * er[row as usize] * ec[col as usize] * q[i as usize];
                    }
                }
            }
        }

        // Right: Σ_i q(i) · f̂ at boolean i.
        let bits = |v: u64, len: usize| -> Vec<F128> {
            (0..len)
                .map(|b| {
                    if (v >> b) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect()
        };
        let mut rhs = F128::ZERO;
        for i in 0..1u64 << p.m {
            rhs += q[i as usize] * f_hat_fancy(&p, &z_tab, &z_row, &z_col, &bits(i, p.m));
        }
        assert_eq!(lhs, rhs, "Eq. 8 reduction failed");
    }

    /// Decomposition: a width-9 table becomes widths 8 and 1 (§6's example),
    /// tiles the same area, and the whole config still passes the field-point
    /// MLE check.
    #[test]
    fn decomposition_of_non_power_of_two_width() {
        let p = FancyJaggedParams::from_tables(&[(9, 2)], 1, 6);
        assert_eq!(p.log_widths[..2], [3, 0], "9 = 8 + 1, widest first");
        assert_eq!(p.heights[..2], [2, 2]);
        assert_eq!(p.area(), 9 * 2);
        assert_eq!(p.live_tables(), 2);

        let mut rng = Rng(0x_D3C0_9009);
        let z_tab = rng.vec(p.k);
        let z_row = rng.vec(p.n);
        let z_col = rng.vec(p.c);
        let z_idx = rng.vec(p.m);
        let got = f_hat_fancy(&p, &z_tab, &z_row, &z_col, &z_idx);

        let (et, er, ec, ei) = (
            build_eq_table(&z_tab),
            build_eq_table(&z_row),
            build_eq_table(&z_col),
            build_eq_table(&z_idx),
        );
        let mut want = F128::ZERO;
        for tab in 0..1usize << p.k {
            for row in 0..1u64 << p.n {
                for col in 0..1u64 << p.c {
                    if let Some(i) = p.rank(tab, row, col) {
                        want += et[tab] * er[row as usize] * ec[col as usize] * ei[i as usize];
                    }
                }
            }
        }
        assert_eq!(got, want);
    }

    /// The whole point: cost is proportional to TABLES, not columns. A registry
    /// slot of 3,325 equal-height columns is 9 tables after decomposition, with
    /// `Σ log₂(width) = 48` against 3,325 columns for basic jagged.
    #[test]
    fn table_count_replaces_column_count() {
        let p = FancyJaggedParams::from_tables(&[(3_325, 1 << 10)], 10, 32);
        assert_eq!(p.live_tables(), 3_325u64.count_ones() as usize);
        assert_eq!(p.live_tables(), 9);
        let sum_log: u32 = p.log_widths[..p.live_tables()].iter().sum();
        assert_eq!(sum_log, 48);
        assert_eq!(p.area(), 3_325 * (1 << 10));
    }

    // -----------------------------------------------------------------------
    // Aligned tables, and the tie-in to `compact_witness_row_major`
    // -----------------------------------------------------------------------

    /// `f̂_aligned` at a random field point must equal the honest MLE of the
    /// row-major bijection's indicator, brute-forced over the whole cube.
    #[test]
    fn aligned_field_point_matches_brute_force() {
        // Two aligned tables of differing width over a 3-bit column index:
        // [0,4) at height 3, then [4,6) at height 2. Area 4·3 + 2·2 = 16.
        let p = AlignedParams::new(
            vec![
                AlignedTable {
                    log_width: 2,
                    height: 3,
                    col_offset: 0,
                },
                AlignedTable {
                    log_width: 1,
                    height: 2,
                    col_offset: 4,
                },
            ],
            2,
            3,
            5,
        );
        assert_eq!(p.area(), 16);

        let mut rng = Rng(0x_A116_9ED0);
        for trial in 0..4 {
            let z_row = rng.vec(p.n);
            let z_col = rng.vec(p.k_cols);
            let z_idx = rng.vec(p.m);
            let got = f_hat_aligned(&p, &z_row, &z_col, &z_idx);

            let (er, ec, ei) = (
                build_eq_table(&z_row),
                build_eq_table(&z_col),
                build_eq_table(&z_idx),
            );
            let mut want = F128::ZERO;
            for i in 0..p.area() {
                let (col, row) = p.unrank(i);
                want += er[row as usize] * ec[col as usize] * ei[i as usize];
            }
            assert_eq!(got, want, "trial {trial}");
        }
    }

    /// `build_weight_row_major` must be the boolean table whose MLE
    /// `f̂_aligned` computes — i.e. folding it at `ρ` gives `f̂_aligned(…, ρ)`.
    #[test]
    fn row_major_weight_folds_to_f_hat() {
        let p = AlignedParams::new(
            vec![
                AlignedTable {
                    log_width: 2,
                    height: 3,
                    col_offset: 0,
                },
                AlignedTable {
                    log_width: 1,
                    height: 2,
                    col_offset: 4,
                },
            ],
            2,
            3,
            5,
        );
        let mut rng = Rng(0x_0DD_F01D);
        let z_row = rng.vec(p.n);
        let z_col = rng.vec(p.k_cols);
        let rho = rng.vec(p.m);

        let mut w = vec![F128::ZERO; 1usize << p.m];
        build_weight_row_major(&p, &z_row, &z_col, &mut w);

        // Σ_d eq(ρ,d)·W[d] must equal f̂_aligned at ρ.
        let eq_rho = build_eq_table(&rho);
        let mut folded = F128::ZERO;
        for (d, &wd) in w.iter().enumerate() {
            folded += eq_rho[d] * wd;
        }
        assert_eq!(folded, f_hat_aligned(&p, &z_row, &z_col, &rho));
    }

    /// **The tie-in.** With `q` produced by
    /// `UnionInstance::compact_witness_row_major`, the aligned-table
    /// configuration must satisfy the jagged reduction against the real union
    /// geometry: `Σ_d q[d]·W[d] = p̂(z_row, z_col)`, where `p̂` is evaluated
    /// directly on the padded witness. This is what says the kernel and the
    /// compaction agree on the same bijection.
    #[test]
    fn reduction_holds_against_compact_witness_row_major() {
        use crate::r1cs::SparseBinaryMatrix;
        use crate::schedule::{Registry, TableClass, TableType};
        use crate::union::UnionInstance;

        let stub = || SparseBinaryMatrix {
            num_rows: 0,
            num_cols: 0,
            rows: Vec::new(),
        };
        let ty = |k_log: usize, useful_bits: usize| TableType {
            k_log,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
            class: TableClass::Boolean,
        };
        let nu = 3usize;
        // ONE slot, its columns padded to the full power-of-two width so the
        // whole slot is a single aligned table — free here, since the dense
        // area already rounds up to that.
        let reg = Registry::new(vec![ty(10, 8 * 128)], nu);
        let union = UnionInstance::new(&reg, vec![1 << nu]);
        let n_cols = 1usize << (10 - 7); // 8 chunk-columns
        let nt = 1u64 << nu;

        let mut rng = Rng(0x_C0FF_EE01);
        let mut padded = vec![F128::ZERO; union.packed_len()];
        for col in 0..n_cols {
            for row in 0..nt as usize {
                padded[(col << nu) + row] = rng.next();
            }
        }
        let q = union.compact_witness_row_major(&padded);
        let dense_log = union.committed_words().trailing_zeros() as usize;

        let p = AlignedParams::new(
            vec![AlignedTable {
                log_width: (10 - 7) as u32,
                height: nt,
                col_offset: 0,
            }],
            nu,
            10 - 7,
            dense_log,
        );
        assert_eq!(p.area(), (n_cols as u64) * nt);

        let z_row = rng.vec(nu);
        let z_col = rng.vec(10 - 7);

        // Left: p̂ read straight off the padded witness at (z_row, z_col).
        let (er, ec) = (build_eq_table(&z_row), build_eq_table(&z_col));
        let mut lhs = F128::ZERO;
        for col in 0..n_cols {
            for row in 0..nt as usize {
                lhs += er[row] * ec[col] * padded[(col << nu) + row];
            }
        }

        // Right: Σ_d q[d]·W[d] on the row-major stack.
        let mut w = vec![F128::ZERO; 1usize << dense_log];
        build_weight_row_major(&p, &z_row, &z_col, &mut w);
        let mut rhs = F128::ZERO;
        for (d, &wd) in w.iter().enumerate() {
            rhs += q[d] * wd;
        }
        assert_eq!(lhs, rhs, "row-major jagged reduction failed");
    }

    /// **Decision probe.** Does evaluating Ŵ(ρ) directly through the aligned
    /// tables actually beat the (already eq-hoisted) Frobenius assist on the
    /// VERIFIER, at the real depth-26 geometry and the real 128·K batch width?
    ///
    /// The paper's per-table win is stated against per-COLUMN direct
    /// evaluation. Flock's assist is not that — it is a heavily optimized
    /// sumcheck — and `f_hat_aligned` runs once per statement, so the 256×
    /// batch could swamp the 3,326 -> 9 table saving. Measure before wiring.
    ///
    /// `cargo test -p flock-core --release --lib
    ///  pcs::jagged_fancy::tests::aligned_vs_assist_verifier_cost
    ///  -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn aligned_vs_assist_verifier_cost() {
        use crate::challenger::FsChallenger;
        use crate::pcs::jagged::{
            FrobeniusClaim, JaggedParams, prove_frobenius_assist, verify_frobenius_assist,
        };
        use crate::union::UnionInstance;

        let _ = crate::init_perf_thread_pool();
        // depth-26 Merkle single-slot geometry: 3,325 used columns of height
        // 2^10, dense_log 22, k_cols 12.
        let (nu, k_cols, dense_log, used) = (10usize, 12usize, 22usize, 3_325usize);
        let widths = UnionInstance::subtable_widths(used);
        let mut tables = Vec::new();
        let mut off = 0u64;
        for w in &widths {
            tables.push(AlignedTable {
                log_width: w.trailing_zeros(),
                height: 1u64 << nu,
                col_offset: off,
            });
            off += *w as u64;
        }
        let ap = AlignedParams::new(tables, nu, k_cols, dense_log);
        assert_eq!(ap.tables.len(), 9);

        // Basic-jagged params over the same grid, for the assist baseline.
        let mut heights = vec![0u64; 1usize << k_cols];
        for h in &mut heights[..used] {
            *h = 1u64 << nu;
        }
        let jp = JaggedParams::from_heights(&heights, nu, dense_log);

        let mut rng = Rng(0x_A11_A5515);
        let claims_data: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
            .map(|_| (rng.vec(nu), rng.vec(k_cols), rng.vec(128)))
            .collect();
        let claims: Vec<FrobeniusClaim<'_>> = claims_data
            .iter()
            .map(|(zr, zc, c)| FrobeniusClaim {
                z_row: zr,
                z_col: zc,
                coeffs: c,
            })
            .collect();
        let rho = rng.vec(dense_log);

        let mut ch = FsChallenger::new(b"flock-aligned-probe");
        let proof = prove_frobenius_assist(&jp, &claims, &rho, &mut ch);

        let time = |f: &dyn Fn()| -> f64 {
            f();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        let t_assist = time(&|| {
            let mut c = FsChallenger::new(b"flock-aligned-probe");
            let v = verify_frobenius_assist(&jp, &claims, &rho, &proof, &mut c);
            std::hint::black_box(&v);
        });
        let t_direct = time(&|| {
            let v = twisted_weight_aligned(&ap, &claims, &rho);
            std::hint::black_box(&v);
        });
        eprintln!(
            "  128·K = {} statements, {} aligned tables (from {} columns)\n  \
             assist verify (eq-hoisted): {:8.2} ms\n  \
             direct via aligned tables : {:8.2} ms   ({:.2}x)",
            128 * claims.len(),
            ap.tables.len(),
            used,
            t_assist,
            t_direct,
            t_direct / t_assist
        );
    }

    /// **The prover-side number.** Does the row-major twisted weight build cost
    /// the same as the column-major one it replaces? The `-258 ms` from
    /// dropping the assist is only a net win if this pass does not eat it.
    ///
    /// `cargo test -p flock-core --release --lib
    ///  pcs::jagged_fancy::tests::row_major_vs_column_major_weight_build
    ///  -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn row_major_vs_column_major_weight_build() {
        use crate::pcs::jagged::{JaggedParams, build_merged_weight_and_prime};
        use crate::union::UnionInstance;

        let _ = crate::init_perf_thread_pool();
        let (nu, k_cols, dense_log, used) = (10usize, 12usize, 22usize, 3_325usize);

        let widths = UnionInstance::subtable_widths(used);
        let mut tables = Vec::new();
        let mut off = 0u64;
        for w in &widths {
            tables.push(AlignedTable {
                log_width: w.trailing_zeros(),
                height: 1u64 << nu,
                col_offset: off,
            });
            off += *w as u64;
        }
        let ap = AlignedParams::new(tables, nu, k_cols, dense_log);

        let mut heights = vec![0u64; 1usize << k_cols];
        for h in &mut heights[..used] {
            *h = 1u64 << nu;
        }
        let jp = JaggedParams::from_heights(&heights, nu, dense_log);
        assert_eq!(ap.area(), jp.area(), "both must cover the same dense area");

        let mut rng = Rng(0x_B011_D000);
        let q: Vec<F128> = (0..1usize << dense_log).map(|_| rng.next()).collect();
        // Two claims, each with a real 16x256 byte-fold table, as the merged
        // open has.
        let data: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
            .map(|_| {
                (
                    rng.vec(nu),
                    rng.vec(k_cols),
                    crate::pcs::ring_switch::build_fold_byte_table(&rng.vec(128)),
                )
            })
            .collect();
        let claims: Vec<(&[F128], &[F128], &[F128])> = data
            .iter()
            .map(|(zr, zc, t)| (zr.as_slice(), zc.as_slice(), t.as_slice()))
            .collect();

        let time = |f: &dyn Fn()| -> f64 {
            f();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        let t_col = time(&|| {
            let r = build_merged_weight_and_prime(&jp, &claims, &q);
            std::hint::black_box(&r.1);
            crate::scratch::give_f128(r.0);
        });
        let t_row = time(&|| {
            let r = build_weight_row_major_twisted(&ap, &claims, &q);
            std::hint::black_box(&r.1);
            crate::scratch::give_f128(r.0);
        });
        eprintln!(
            "  dense 2^{dense_log} words, {} claims, {} aligned tables\n  \
             column-major W + prime (today): {:8.2} ms\n  \
             row-major    W + prime        : {:8.2} ms   ({:.2}x)",
            claims.len(),
            ap.tables.len(),
            t_col,
            t_row,
            t_row / t_col
        );
    }

    /// **Prover/verifier agreement.** The twisted row-major weight, folded at
    /// ρ, must equal `twisted_weight_aligned` at ρ — i.e. the prover's
    /// `w_eval` from the merged sumcheck and the verifier's own evaluation of
    /// `Ŵ(ρ)` are the same value. That identity is the whole protocol; both
    /// sides are tied through the SAME Φ by deriving the verifier's
    /// coefficients from the prover's fold table.
    #[test]
    fn twisted_row_major_folds_to_twisted_weight_aligned() {
        use crate::pcs::jagged::FrobeniusClaim;
        use crate::pcs::ring_switch::{build_fold_byte_table, linearized_coefficients};

        let (nu, k_cols, dense_log) = (2usize, 3usize, 5usize);
        // Two aligned tables of differing width: [0,4) then [4,6).
        let p = AlignedParams::new(
            vec![
                AlignedTable {
                    log_width: 2,
                    height: 3,
                    col_offset: 0,
                },
                AlignedTable {
                    log_width: 1,
                    height: 2,
                    col_offset: 4,
                },
            ],
            nu,
            k_cols,
            dense_log,
        );

        let mut rng = Rng(0x_7015_7ED0);
        let q: Vec<F128> = (0..1usize << dense_log).map(|_| rng.next()).collect();
        // A LEGITIMATE fold table: `build_fold_byte_table` makes
        // `fold_one_slot` 𝔽₂-linear, which is what the Frobenius
        // decomposition needs. A random 16×256 table is not any linear map's
        // byte decomposition, so `linearized_coefficients` would not describe
        // it and the identity below would (correctly) fail.
        let data: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
            .map(|_| {
                (
                    rng.vec(nu),
                    rng.vec(k_cols),
                    build_fold_byte_table(&rng.vec(128)),
                )
            })
            .collect();
        let claims: Vec<(&[F128], &[F128], &[F128])> = data
            .iter()
            .map(|(zr, zc, t)| (zr.as_slice(), zc.as_slice(), t.as_slice()))
            .collect();

        let (w, _) = build_weight_row_major_twisted(&p, &claims, &q);
        let rho = rng.vec(dense_log);
        let eq_rho = build_eq_table(&rho);
        let mut folded = F128::ZERO;
        for (d, &wd) in w.iter().enumerate() {
            folded += eq_rho[d] * wd;
        }
        crate::scratch::give_f128(w);

        // Verifier side: the same Φ, decomposed into Frobenius powers.
        let coeffs: Vec<Vec<F128>> = data
            .iter()
            .map(|(_, _, t)| linearized_coefficients(t))
            .collect();
        let fclaims: Vec<FrobeniusClaim<'_>> = data
            .iter()
            .zip(&coeffs)
            .map(|((zr, zc, _), c)| FrobeniusClaim {
                z_row: zr,
                z_col: zc,
                coeffs: c,
            })
            .collect();
        assert_eq!(
            folded,
            twisted_weight_aligned(&p, &fclaims, &rho),
            "prover's folded W(ρ) != verifier's Ŵ(ρ)"
        );
    }
}
