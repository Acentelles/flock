//! Basic-jagged formulation — the pieces the merged transport still uses.
//!
//! A *jagged function* `p : {0,1}^n × {0,1}^k → F` is a `2^n × 2^k` table in
//! which column `y` is nonzero only below its height `h_y`. Its nonzero entries
//! flatten, in column-major order, into a single *dense* multilinear
//! `q : {0,1}^m → F` (`2^m ≥ Σ_y h_y`). With cumulative heights
//! `t_y = h_0 + … + h_y`,
//!
//! ```text
//!   p̂(z_r, z_c) = Σ_{i ∈ {0,1}^m} q(i) · f̂_t(z_r, z_c, i)          (2025/917, Eq. 3)
//!   f̂_t(z_r, z_c, i) = eq(row_t(i), z_r) · eq(col_t(i), z_c)        (Eq. 4, boolean i only)
//! ```
//!
//! That is the reduction the PCS runs. **It no longer lives here.** The shipped
//! transport is the *merged* one: the jagged weight is folded into the
//! ring-switch sumcheck rather than proved by a standalone jagged sumcheck, and
//! `f̂` is evaluated per-TABLE by the fancy (§6) branching program in
//! [`super::jagged_fancy`] instead of per-column by the basic (§3.1) width-4
//! one. What that left behind, and what this file now is, is four things:
//!
//! * [`JaggedParams`] — the cumulative-height geometry, in the basic (flat,
//!   per-column) parameterization. The live path uses
//!   [`jagged_fancy::AlignedParams`](super::jagged_fancy::AlignedParams); this
//!   survives as the equivalent flat description the cross-check test builds
//!   the same area from.
//! * `point_bit` / `int_bit` — the LSB-first bit accessors both branching
//!   programs read their inputs through.
//! * [`MergedWeightClaim`] + `build_merged_weight_and_prime` — the merged
//!   reduction's weight-table build, column-major. The production path builds
//!   it row-major (`jagged_fancy::build_weight_row_major_twisted`); this is the
//!   reference the row-major kernel is measured and checked against.
//! * [`FrobeniusClaim`] and the product-sumcheck fold kernels
//!   (`fold_round_claim`, `fold_oop_par`, `fold_and_round_oop_par`) —
//!   both on the live merged path, in `pcs::open_batch_merged` /
//!   `verify_batch_merged`.
//!
//! ## What was removed, and why it is not coming back
//!
//! The basic jagged sumcheck (`prove`/`verify`), the width-4 `f̂_t` evaluator,
//! the *jagged assist* of §5 (which delegated the verifier's `2^k`
//! branching-program DPs to the prover as a sumcheck over the `2(m+1)`
//! cumulative-height variables, with the Lemma 4.6 prefix/suffix streaming
//! prover and its blocked run-tree), the *batched Frobenius assist* (the same
//! thing over `128·K` Φ-twisted statements), the multipoint-twisted transport,
//! and the inverse-Frobenius/Moore-matrix machinery those needed — all gone.
//!
//! Two independent changes killed them. Upstream replaced the standalone
//! jagged transport with the merged one, which orphaned `prove`/`verify` and
//! everything reachable only from them. Separately, §6's per-table branching
//! program dropped the verifier's direct-evaluation cost from 3,326 column DPs
//! to 9 table DPs at the depth-26 geometry — and, with the ρ-side hoisted out
//! of the `128·K` statements, direct evaluation beat the (already
//! eq-hoisted) assist outright. So the delegation had nothing left to delegate:
//! the paper's own §6 note that the verifier "can compute the latter on its
//! own, or employ a similar jagged assist" resolves, per-table, to the former.
//!
//! The Frobenius *decomposition* is untouched by this and is still
//! load-bearing: `ring_switch::linearized_coefficients` builds the `c_{i,j}`,
//! [`FrobeniusClaim`] carries them, and
//! `jagged_fancy::twisted_weight_aligned_batched` walks
//! `Ŵ(ρ) = Σ_i Σ_j c_{i,j}·f̂(z_i^{2^j}, ρ)`. Only the sumcheck that used to
//! prove that value on the prover's behalf is gone.

use crate::field::F128;
use crate::lincheck::build_eq_table;

/// Configuration of a jagged function: the (zero-padded to `2^k`) column
/// heights, summarized as the cumulative-height prefix sums.
#[derive(Clone, Debug)]
pub struct JaggedParams {
    /// `log2` of the height bound (number of row variables of `p̂`).
    pub n: usize,
    /// `log2` of the number of columns (column variables of `p̂`).
    pub k: usize,
    /// `log2` of the dense area: `q` has `2^m` entries, `Σ_y h_y ≤ 2^m`.
    pub m: usize,
    /// Cumulative heights `[t_{-1}=0, t_0, t_1, …, t_{2^k-1}=area]`, length
    /// `2^k + 1`. Column `c` occupies dense indices `[col_prefix_sums[c],
    /// col_prefix_sums[c+1])`.
    pub col_prefix_sums: Vec<u64>,
}

impl JaggedParams {
    /// Build params from per-column heights. `heights.len()` must be `2^k`
    /// (zero-pad empty columns up to a power of two yourself). Requires each
    /// height `≤ 2^n` and total area `≤ 2^m`.
    pub fn from_heights(heights: &[u64], n: usize, m: usize) -> Self {
        assert!(
            heights.len().is_power_of_two(),
            "number of columns must be a power of two (zero-pad)"
        );
        let k = heights.len().trailing_zeros() as usize;
        let mut col_prefix_sums = Vec::with_capacity(heights.len() + 1);
        let mut acc: u64 = 0;
        col_prefix_sums.push(0);
        for &h in heights {
            assert!(h <= (1u64 << n), "column height exceeds 2^n");
            acc += h;
            col_prefix_sums.push(acc);
        }
        assert!(acc <= (1u64 << m), "total area exceeds 2^m");
        JaggedParams {
            n,
            k,
            m,
            col_prefix_sums,
        }
    }

    /// Total number of nonzero entries `Σ_y h_y`.
    pub fn area(&self) -> u64 {
        *self.col_prefix_sums.last().unwrap()
    }

    /// The bijection `i ↦ (row_t(i), col_t(i))` for a dense index `i < area`:
    /// `col` is the column whose range contains `i`, `row = i - t_{col-1}`.
    /// This is the *column-major* dense order; the shipped layout is
    /// row-major-within-table, so the live path unranks through
    /// [`jagged_fancy::AlignedParams`](super::jagged_fancy::AlignedParams)
    /// instead. Retained for `union`'s column-major compaction tests.
    pub fn unrank(&self, i: u64) -> (usize, usize) {
        debug_assert!(i < self.area());
        // First prefix-sum strictly greater than `i`, minus one, is the column.
        let col = self.col_prefix_sums.partition_point(|&t| t <= i) - 1;
        let row = i - self.col_prefix_sums[col];
        (row as usize, col)
    }
}

/// Bit `layer` of the field "point" `z`: the coordinate `z[layer]` if present,
/// else `ZERO` (the variable is pinned to 0 — i.e. zero-padded).
#[inline]
pub(crate) fn point_bit(z: &[F128], layer: usize) -> F128 {
    if layer < z.len() {
        z[layer]
    } else {
        F128::ZERO
    }
}

/// Bit `layer` of the integer `t`, as a field element.
#[inline]
pub(crate) fn int_bit(t: u64, layer: usize) -> F128 {
    if (t >> layer) & 1 == 1 {
        F128::ONE
    } else {
        F128::ZERO
    }
}

/// One claim's (or claim group's) contribution to the merged weight.
pub enum MergedWeightClaim<'a> {
    /// A ring-switched claim: its F₂-linear fold table applied to
    /// `eq_row ⊗ eq_col` — additive but not F128-homogeneous, so it cannot
    /// join a scalar group.
    Folded {
        z_row: &'a [F128],
        z_col: &'a [F128],
        table: &'a [F128],
    },
    /// A GROUP of γ-scaled (F128-linear) packed-direct claims sharing one
    /// row point: `Σᵢ γᵢ·eq_rowᵢ(row)·eq_colᵢ(col) =
    /// eq_row(row)·(Σᵢ γᵢ·eq_colᵢ(col))`, so the whole group costs ONE
    /// multiply-sweep against the precombined (already γ-summed) column
    /// table. Exact — field multiplication distributes and the sums
    /// reassociate — so the produced `W` is bit-identical to per-claim
    /// fold-table sweeps. This is what keeps the Φ-pass from scaling with
    /// the circuit path's gather-claim count (~2^c claims, one shared
    /// ρ_row).
    Scalar { z_row: &'a [F128], cols: Vec<F128> },
}

/// Materialize the merged reduction's twisted weight over the dense cube:
/// `W[d] = Σ_i fold_one_slot(eq_row_i[row(d)]·eq_col_i[col(d)], table_i)`
/// for `d < area`, ZERO on the power-of-two tail — the definitional
/// zero-extension (`q`'s committed tail is zero, and the branching program
/// computes exactly this extension via its comparison state, so prover table
/// and verifier evaluation agree by construction). `claims` = `(z_row, z_col,
/// γ-baked fold table)` views.
///
/// Retained as the column-major reference: the merged path builds the weight
/// row-major (`jagged_fancy::build_weight_row_major_twisted`), and
/// `jagged_fancy::tests::row_major_vs_column_major_weight_build` checks and
/// times the two against each other.
#[allow(dead_code)]
pub(crate) fn build_merged_weight_and_prime(
    params: &JaggedParams,
    claims: &[MergedWeightClaim<'_>],
    q: &[F128],
) -> (Vec<F128>, (F128, F128)) {
    use rayon::prelude::*;
    let area = params.area() as usize;
    let n_total = 1usize << params.m;
    enum ColSide<'a> {
        Fold(Vec<F128>, &'a [F128]),
        Combined(&'a [F128]),
    }
    let tabs: Vec<(Vec<F128>, ColSide<'_>)> = claims
        .iter()
        .map(|c| match c {
            MergedWeightClaim::Folded {
                z_row,
                z_col,
                table,
            } => (
                build_eq_table(z_row),
                ColSide::Fold(build_eq_table(z_col), *table),
            ),
            MergedWeightClaim::Scalar { z_row, cols } => {
                (build_eq_table(z_row), ColSide::Combined(cols.as_slice()))
            }
        })
        .collect();
    assert_eq!(q.len(), n_total);
    let mut w = crate::scratch::take_f128(n_total);
    // Segmented fill (the JaggedWeight lesson): per chunk, ONE cursor into
    // `col_prefix_sums`, then per column segment a claim-OUTER sweep — the
    // column factor hoisted, rows read sequentially, and one claim's 64 KB
    // fold table hot per sweep. The per-element unrank variant measured
    // ~2.5x slower at M = 30.
    // The merged sumcheck's round-0 prime `(u0, u2)` is fused into the same
    // pass (CHUNK is even, so element pairs never straddle chunks); the
    // dead tail past the area contributes zero on both sides.
    const CHUNK: usize = 1 << 14;
    let ps = &params.col_prefix_sums;
    let prime = w
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, out)| {
            let base = (ci * CHUNK) as u64;
            let end = base + out.len() as u64;
            if base >= area as u64 {
                out.fill(F128::ZERO);
                return (F128::ZERO, F128::ZERO);
            }
            let live_end = end.min(area as u64);
            // Zero the dead tail of this chunk (past the jagged area).
            out[(live_end - base) as usize..].fill(F128::ZERO);
            let mut first_claim = true;
            for (eq_r, side) in tabs.iter() {
                let mut col = ps.partition_point(|&t| t <= base) - 1;
                let mut e = base;
                while e < live_end {
                    while ps[col + 1] <= e {
                        col += 1;
                    }
                    let seg_end = ps[col + 1].min(live_end);
                    let row0 = (e - ps[col]) as usize;
                    let dst = &mut out[(e - base) as usize..(seg_end - base) as usize];
                    let rows = &eq_r[row0..row0 + dst.len()];
                    match side {
                        ColSide::Fold(eq_c, tab) => {
                            let c_hoist = eq_c[col];
                            if first_claim {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot =
                                        crate::pcs::ring_switch::fold_one_slot(r * c_hoist, tab);
                                }
                            } else {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot +=
                                        crate::pcs::ring_switch::fold_one_slot(r * c_hoist, tab);
                                }
                            }
                        }
                        ColSide::Combined(cols) => {
                            let c_hoist = cols[col];
                            if first_claim {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot = r * c_hoist;
                                }
                            } else {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot += r * c_hoist;
                                }
                            }
                        }
                    }
                    e = seg_end;
                }
                first_claim = false;
            }
            let qc = &q[base as usize..end as usize];
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for (qp, wp) in qc
                .as_chunks::<2>()
                .0
                .iter()
                .zip(out.as_chunks::<2>().0.iter())
            {
                u0 += qp[0] * wp[0];
                u2 += (qp[0] + qp[1]) * (wp[0] + wp[1]);
            }
            (u0, u2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    (w, prime)
}

/// One ring-switch claim's inputs to the Φ-twisted weight evaluation: the
/// word-level row/column point split and the 128 linearized coefficients
/// (γ-baked) of its fold map. Consumed by
/// `jagged_fancy::twisted_weight_aligned_batched`.
pub struct FrobeniusClaim<'a> {
    pub z_row: &'a [F128],
    pub z_col: &'a [F128],
    pub coeffs: &'a [F128],
}

/// Reduce the running sumcheck claim through one round. The degree-2 round
/// polynomial `G` is given by `G(1) = g_one`, leading coeff `G(∞) = g_inf`, and
/// `G(0) = claim + G(1)` (since `claim = G(0) + G(1)`). Returns `G(r)`.
#[inline]
pub(crate) fn fold_round_claim(claim: F128, g_one: F128, g_inf: F128, r: F128) -> F128 {
    let g0 = claim + g_one; // char-2: G(0) = claim - G(1)
    // G(X) = g0 + (G(1) + g0 + g_inf)·X + g_inf·X²
    g0 + (g_one + g0 + g_inf) * r + g_inf * (r * r)
}

/// Parallel out-of-place fold (no message), `ao/bo` length `a.len()/2`. Used for
/// the final round (size 2 → 1), where there is no successor message.
pub(crate) fn fold_oop_par(a: &[F128], b: &[F128], r: F128, ao: &mut [F128], bo: &mut [F128]) {
    use rayon::prelude::*;
    ao.par_iter_mut()
        .zip(bo.par_iter_mut())
        .enumerate()
        .for_each(|(x, (oa, ob))| {
            *oa = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
            *ob = b[2 * x] + r * (b[2 * x + 1] + b[2 * x]);
        });
}

/// Parallel **fused** round: out-of-place fold at `r` + the next round's message
/// in one pass. Requires `a.len() >= 4`. This is the production kernel — in the
/// bandwidth-bound parallel regime the halved pass count is a ~1.4× win (the
/// serial penalty from the fold→message dependency is hidden across cores).
/// Drives the merged reduction's sumcheck in `pcs::open_batch_merged`.
pub(crate) fn fold_and_round_oop_par(
    a: &[F128],
    b: &[F128],
    r: F128,
    ao: &mut [F128],
    bo: &mut [F128],
) -> (F128, F128) {
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), 2 * ao.len());
    debug_assert!(a.len() >= 4);
    // Output chunk of `CO`; the aligned input chunk is `2*CO` (output is half
    // the input). Slice/`chunks_exact` iteration — no per-element bounds checks —
    // so the reduction scales like the fold (~6× vs ~2.6× for indexed access).
    const CO: usize = 1 << 13;
    ao.par_chunks_mut(CO)
        .zip(bo.par_chunks_mut(CO))
        .zip(a.par_chunks(2 * CO))
        .zip(b.par_chunks(2 * CO))
        .map(|(((oa, ob), ain), bin)| {
            let mut g1 = F128::ZERO;
            let mut gi = F128::ZERO;
            for (((op, opb), aq), bq) in oa
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(ob.as_chunks_mut::<2>().0.iter_mut())
                .zip(ain.as_chunks::<4>().0.iter())
                .zip(bin.as_chunks::<4>().0.iter())
            {
                let na0 = aq[0] + r * (aq[1] + aq[0]);
                let na1 = aq[2] + r * (aq[3] + aq[2]);
                let nb0 = bq[0] + r * (bq[1] + bq[0]);
                let nb1 = bq[2] + r * (bq[3] + bq[2]);
                op[0] = na0;
                op[1] = na1;
                opb[0] = nb0;
                opb[1] = nb1;
                g1 += na1 * nb1;
                gi += (na0 + na1) * (nb0 + nb1);
            }
            (g1, gi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
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

    /// Serial degree-2 round message `(G(1), G(∞))` for `Σ_{x'} a(X,x')·b(X,x')`
    /// with the low bit bound: `a(0,x') = a[2x']`, `a(1,x') = a[2x'+1]`, so
    /// `a(X,x') = a0 + (a1+a0)·X` and the leading coeff is `(a1+a0)(b1+b0)`.
    fn round_msg_ref(a: &[F128], b: &[F128]) -> (F128, F128) {
        let mut g_one = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for x in 0..a.len() / 2 {
            let (a0, a1) = (a[2 * x], a[2 * x + 1]);
            let (b0, b1) = (b[2 * x], b[2 * x + 1]);
            g_one += a1 * b1;
            g_inf += (a0 + a1) * (b0 + b1);
        }
        (g_one, g_inf)
    }

    fn dot(a: &[F128], b: &[F128]) -> F128 {
        a.iter()
            .zip(b)
            .fold(F128::ZERO, |acc, (x, y)| acc + *x * *y)
    }

    /// The invariant the merged reduction rests on, exercised end to end over
    /// every round: the claim folded by [`fold_round_claim`] equals the folded
    /// halves' inner product, and the message [`fold_and_round_oop_par`]
    /// produces in the same pass is the next round's message.
    ///
    /// This is the only correctness check on these three kernels inside this
    /// file — the benchmarks that used to walk them were removed with the
    /// standalone jagged transport.
    #[test]
    fn round_folding_preserves_the_sumcheck_claim() {
        let mut rng = Rng(0x_F01D_C1A1_9);
        let n = 1usize << 6;
        let mut a = rng.vec(n);
        let mut b = rng.vec(n);

        let mut claim = dot(&a, &b);
        let (mut g_one, mut g_inf) = round_msg_ref(&a, &b);

        let mut cur = n;
        while cur > 1 {
            let half = cur / 2;
            let r = rng.next();
            let mut ao = vec![F128::ZERO; half];
            let mut bo = vec![F128::ZERO; half];

            let fused = if cur >= 4 {
                Some(fold_and_round_oop_par(
                    &a[..cur],
                    &b[..cur],
                    r,
                    &mut ao,
                    &mut bo,
                ))
            } else {
                fold_oop_par(&a[..cur], &b[..cur], r, &mut ao, &mut bo);
                None
            };

            claim = fold_round_claim(claim, g_one, g_inf, r);
            assert_eq!(
                claim,
                dot(&ao, &bo),
                "folded claim must equal Σ a'·b' at size {half}"
            );

            if let Some(msg) = fused {
                let want = round_msg_ref(&ao, &bo);
                assert_eq!(
                    msg, want,
                    "fused message must match the reference at {half}"
                );
                (g_one, g_inf) = msg;
            }

            a = ao;
            b = bo;
            cur = half;
        }
        assert_eq!(cur, 1);
    }

    /// [`fold_round_claim`]'s encoding, spelled out: the degree-2 polynomial it
    /// reconstructs from `(claim, G(1), G(∞))` really does hit `G(0)`, `G(1)`,
    /// and the stated leading coefficient. Char-2 specific — `G(0) = claim +
    /// G(1)` uses `−1 = 1`.
    #[test]
    fn fold_round_claim_reconstructs_the_round_polynomial() {
        let mut rng = Rng(0x_C0DE_2);
        for _ in 0..8 {
            let (g_one, g_inf) = (rng.next(), rng.next());
            let g0 = rng.next();
            let claim = g0 + g_one; // the sumcheck's G(0) + G(1)
            assert_eq!(fold_round_claim(claim, g_one, g_inf, F128::ZERO), g0);
            assert_eq!(fold_round_claim(claim, g_one, g_inf, F128::ONE), g_one);
            // Leading coeff: G(r) + G(0) + r·(linear coeff) = g_inf·r².
            let r = rng.next();
            let lin = g_one + g0 + g_inf;
            assert_eq!(
                fold_round_claim(claim, g_one, g_inf, r),
                g0 + lin * r + g_inf * (r * r)
            );
        }
    }

    /// `from_heights` builds the cumulative-height prefix sums the merged
    /// weight build indexes columns through, and zero-height columns produce
    /// repeated boundaries (empty ranges) rather than being dropped.
    #[test]
    fn from_heights_prefix_sums_and_area() {
        let p = JaggedParams::from_heights(&[3, 0, 5, 2], 3, 6);
        assert_eq!(p.k, 2);
        assert_eq!(p.n, 3);
        assert_eq!(p.m, 6);
        assert_eq!(p.col_prefix_sums, vec![0, 3, 3, 8, 10]);
        assert_eq!(p.area(), 10);
    }

    #[test]
    fn bit_accessors_are_lsb_first_and_zero_pad() {
        let z = vec![F128::ONE, F128::ZERO, F128::ONE];
        assert_eq!(point_bit(&z, 0), F128::ONE);
        assert_eq!(point_bit(&z, 1), F128::ZERO);
        assert_eq!(point_bit(&z, 2), F128::ONE);
        // Past the end: the variable is pinned to 0.
        assert_eq!(point_bit(&z, 3), F128::ZERO);
        assert_eq!(int_bit(0b101, 0), F128::ONE);
        assert_eq!(int_bit(0b101, 1), F128::ZERO);
        assert_eq!(int_bit(0b101, 2), F128::ONE);
        assert_eq!(int_bit(0b101, 63), F128::ZERO);
    }
}
