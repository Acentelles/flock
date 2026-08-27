//! Polynomial commitment scheme for the bit-MLE witness `ẑ` over GF(2).
//!
//! Construction: Binius-style PCS with F_{2^128} packing.
//!
//! - **Commit**: pack the 2^m Boolean witness into 2^(m−7) F_{2^128} elements
//!   (one bit per polynomial-basis coordinate of F_{2^128}), batch RS-encode
//!   via additive NTT, Merkle-commit the codeword.
//! - **Open**: at a QuirkyPoint (z_skip, x_outer) from the zerocheck/lincheck:
//!   1. [`ring_switch::prove`] sends 128 partial-evaluations `s_hat_v` and
//!      produces a sumcheck target `(rs_eq_ind, sumcheck_claim)`.
//!   2. [`ligerito::recursive_prover_with_basis`] discharges the combined
//!      claim `⟨packed_witness, b_combined⟩ = target_combined` via the
//!      recursive Ligerito argument, reusing the commit-time codeword and
//!      Merkle tree as Ligerito's L0 commitment.
//! - **Verify**: the verifier replays ring-switching succinctly, then drives
//!   the succinct recursive Ligerito verifier, evaluating the combined basis
//!   at the residual point (see [`verify_opening_batch_ligerito_mixed`]).
//!
//! See [DP24](https://eprint.iacr.org/2024/504) (ring-switching) and the
//! ligerito module docs for the recursion.

pub mod commit;
pub mod jagged;
pub mod ligerito;
pub mod pack;
pub mod ring_switch;
pub mod tensor_algebra;

/// TEMP probe (open campaign): isolated variants of the combine sweep for
/// `benches/open_combine_probe.rs`. Strip when the open campaign closes.
#[doc(hidden)]
pub mod combine_probe {
    use super::ring_switch;
    use crate::field::F128;
    use rayon::prelude::*;

    pub const FOLD_TABLE_LEN: usize = ring_switch::FOLD_TABLE_LEN;
    type Claims = [(Vec<F128>, Vec<F128>, Vec<F128>)];

    /// Production shape: per block, compose each claim's table then sweep.
    fn composed(claims: &Claims, out: &mut [F128]) -> F128 {
        let b = claims[0].0.len();
        out.par_chunks_mut(b)
            .enumerate()
            .map_init(
                || vec![F128::ZERO; FOLD_TABLE_LEN],
                |ctable, (hi, out_block)| {
                    for (ci, (eq_lo, eq_hi, table)) in claims.iter().enumerate() {
                        ring_switch::compose_fold_byte_table_into(eq_hi[hi], table, ctable);
                        if ci == 0 {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot = ring_switch::fold_one_slot(lo, ctable);
                            }
                        } else {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot += ring_switch::fold_one_slot(lo, ctable);
                            }
                        }
                    }
                    out_block[0]
                },
            )
            .reduce(|| F128::ZERO, |a, x| a + x)
    }

    /// Compose hoisted out (stale table — wrong values, isolates build cost).
    fn composed_no_build(claims: &Claims, out: &mut [F128]) -> F128 {
        let b = claims[0].0.len();
        out.par_chunks_mut(b)
            .enumerate()
            .map_init(
                || {
                    let mut ct = vec![F128::ZERO; FOLD_TABLE_LEN];
                    ring_switch::compose_fold_byte_table_into(
                        claims[0].1[0],
                        &claims[0].2,
                        &mut ct,
                    );
                    ct
                },
                |ctable, (_hi, out_block)| {
                    for (ci, (eq_lo, _, _)) in claims.iter().enumerate() {
                        if ci == 0 {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot = ring_switch::fold_one_slot(lo, ctable);
                            }
                        } else {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot += ring_switch::fold_one_slot(lo, ctable);
                            }
                        }
                    }
                    out_block[0]
                },
            )
            .reduce(|| F128::ZERO, |a, x| a + x)
    }

    /// Pre-composed-port baseline: per-slot multiply into the base table.
    fn slot_mul(claims: &Claims, out: &mut [F128]) -> F128 {
        let b = claims[0].0.len();
        out.par_chunks_mut(b)
            .enumerate()
            .map(|(hi, out_block)| {
                for (ci, (eq_lo, eq_hi, table)) in claims.iter().enumerate() {
                    let e_hi = eq_hi[hi];
                    if ci == 0 {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    } else {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    }
                }
                out_block[0]
            })
            .reduce(|| F128::ZERO, |a, x| a + x)
    }

    /// Both claims fused into one sweep: two composed tables live (128 KiB),
    /// one store per slot, no intermediate read-back.
    fn fused_claims(claims: &Claims, out: &mut [F128]) -> F128 {
        let b = claims[0].0.len();
        out.par_chunks_mut(b)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![F128::ZERO; FOLD_TABLE_LEN],
                        vec![F128::ZERO; FOLD_TABLE_LEN],
                    )
                },
                |(ct0, ct1), (hi, out_block)| {
                    ring_switch::compose_fold_byte_table_into(claims[0].1[hi], &claims[0].2, ct0);
                    ring_switch::compose_fold_byte_table_into(claims[1].1[hi], &claims[1].2, ct1);
                    for ((slot, &lo0), &lo1) in out_block
                        .iter_mut()
                        .zip(claims[0].0.iter())
                        .zip(claims[1].0.iter())
                    {
                        *slot = ring_switch::fold_one_slot(lo0, ct0)
                            + ring_switch::fold_one_slot(lo1, ct1);
                    }
                    out_block[0]
                },
            )
            .reduce(|| F128::ZERO, |a, x| a + x)
    }

    pub const VARIANTS: &[(&str, fn(&Claims, &mut [F128]) -> F128)] = &[
        ("composed (production)", composed),
        ("composed, build hoisted (timing only)", composed_no_build),
        ("slot-multiply (old baseline)", slot_mul),
        ("fused claims, one pass", fused_claims),
    ];
}

pub use commit::{
    Commitment, PcsParams, ProverData, commit, commit_encode, commit_encode_into, commit_into,
    commit_leaf_pipeline_shape, commit_merkle, prefault_codeword_during,
};

/// A/B kill switch for the direct (basis-free) opening, DEFAULT ON —
/// certified 2026-08-26: m=32 open 97.4→64.1 MT (3/3 disjoint) and
/// 614→210 ST, end-to-end best 484.25 ms (campaign record), transcript
/// byte-identical by test. `FLOCK_NO_OPEN_DIRECT=1` is the production
/// kill switch; the AtomicBool exists for paired within-process A/B and
/// the proof-identity test.
pub static OPEN_DIRECT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub use pack::{LOG_PACKING, pack_witness, unpack_witness};
pub use ring_switch::{RingSwitchProof, SparseEqTensor};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::PaddingSpec;
use serde::{Deserialize, Serialize};

/// Batched opening proof: ring-switching frontend + Ligerito backend.
/// The combined `b_combined` + target_combined feed
/// [`ligerito::recursive_prover_with_basis`] (see ligerito module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOpeningProofLigerito {
    pub ring_switches: Vec<RingSwitchProof>,
    pub ligerito: ligerito::LigeritoProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    RingSwitch(ring_switch::VerifyError),
    /// The Ligerito recursive verifier rejected the proof.
    Ligerito,
}

/// `eq_ind` representation for a packed-direct claim. The contributed value at
/// scattered index `j` is the tensor entry — for the dense variant the index
/// is the array offset; for the sparse variant it's reconstructed via
/// [`SparseEqTensor::scatter_idx`].
#[derive(Clone, Debug)]
pub enum DirectEqInd {
    /// Fully-materialized `eq_ind(point)` of length `2^L`.
    Dense(Vec<F128>),
    /// Sparse representation — non-zero entries at scattered indices.
    /// Built from a claim point with one or more exactly-zero coords via
    /// [`ring_switch::build_eq_sparse`].
    Sparse(SparseEqTensor),
}

/// A packed-MLE evaluation claim: `ẑ_packed(point) = value`. Unlike a
/// ring-switched claim, this is opened directly without going through the
/// bit-MLE ↔ packed-MLE bridge (no `s_hat_v`, no φ_8 weighting).
///
/// Use case: protocols whose sumcheck output is naturally a packed-MLE
/// evaluation (e.g. the chain shift sumcheck operating on packed columns
/// instead of bit-folded scalars). Skips the ring-switch step for this claim,
/// saving the `fold_1b_rows` + per-opening-tail work at the prover and the
/// ring-switch verify + φ_8 reconstruction at the verifier.
///
/// The claim-combine step adds `γ_k · eq_ind(point)` to `b_combined` and
/// `γ_k · value` to the target; the verifier's residual check contributes
/// `γ_k · eq_eval(point, residual_challenges)`.
#[derive(Clone, Debug)]
pub struct PackedDirectClaim {
    /// Multilinear point of length `L = m − 7`.
    pub point: Vec<F128>,
    /// Claimed `ẑ_packed(point)` value.
    pub value: F128,
    /// `eq_ind(point)` in dense or sparse form. Caller responsibility to
    /// match the claim's `point` — the contribution to `b_combined` is read
    /// directly from this tensor.
    pub eq_ind: DirectEqInd,
}

/// Mixed-claim batched open: supports both **ring-switched** claims (bit-MLE
/// openings reduced via `ring_switch::prove_batched`, with optional per-claim
/// precomputed `s_hat_v`) and **packed-direct** claims (packed-MLE openings
/// that skip ring-switch). Runs the ring_switch + b_combined computation, then
/// routes to [`ligerito::recursive_prover_with_basis`] using the existing
/// `prover_data`'s codeword + tree as Ligerito's L0 commit (no L0 re-commit).
///
/// `lig_config.initial_k` must equal `commitment.params.log_batch_size` so that
/// `prover_data`'s codeword/tree shape matches what Ligerito expects for L0.
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_with_precomputed_s_hat_v<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    open_batch_mixed_ligerito_with_precomputed_s_hat_v_banked(
        packed_witness,
        prover_data,
        commitment,
        x_outers,
        precomputed_s_hat_v,
        &[],
        packed_direct,
        padding,
        lig_config,
        challenger,
    )
}

/// Like [`open_batch_mixed_ligerito_with_precomputed_s_hat_v`], additionally
/// accepting per-claim BANKED statistics for the direct (basis-free) opening.
/// Claims without a supplied bank fall back to the reference scan when the
/// direct gate fires (tests / exotic callers); production provers supply
/// banks captured for free in lincheck (AB) and zerocheck round 1 (C).
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_with_precomputed_s_hat_v_banked<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    banked_s_hat_v: &[Option<&ring_switch::BankedShatV>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t_total = std::time::Instant::now();

    assert_eq!(
        lig_config.initial_k, commitment.params.log_batch_size,
        "ligerito initial_k ({}) must match PcsParams.log_batch_size ({}) for L0 reuse",
        lig_config.initial_k, commitment.params.log_batch_size,
    );
    assert_eq!(
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
        "ligerito log_inv_rates[0] ({}) must match PcsParams.log_inv_rate ({}) for L0 reuse",
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
    );

    // Direct (basis-free) opening gate: opt-in while it certifies. The
    // banked region must span exactly the lane folds, so c = initial_k.
    let direct_on = !(std::env::var_os("FLOCK_NO_OPEN_DIRECT").is_some()
        || OPEN_DIRECT_DISABLE.load(std::sync::atomic::Ordering::Relaxed));
    let direct_c = if direct_on && packed_direct.is_empty() {
        lig_config.initial_k
    } else {
        0
    };
    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        banked_s_hat_v,
        packed_direct,
        direct_c,
        padding,
        challenger,
        trace,
    );

    let t = std::time::Instant::now();
    let ligerito_proof = if let Some(dl0) = combined.direct {
        ligerito::recursive_prover_direct(
            lig_config,
            packed_witness,
            dl0,
            combined.target_combined,
            &prover_data.codeword,
            &prover_data.merkle_tree,
            challenger,
        )
    } else {
        ligerito::recursive_prover_with_basis_precomputed_round0(
            lig_config,
            packed_witness,
            combined.b_combined,
            combined.target_combined,
            &prover_data.codeword,
            &prover_data.merkle_tree,
            combined.round0_prime,
            combined.round1_lookahead,
            challenger,
        )
    };
    if trace {
        eprintln!(
            "  [open_batch] ligerito::recursive_prover_with_basis: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_batch] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }

    BatchOpeningProofLigerito {
        ring_switches: combined.ring_switches,
        ligerito: ligerito_proof,
    }
}

/// What ring_switch + claim-combination produces, fed to the Ligerito backend.
struct CombinedClaim {
    ring_switches: Vec<RingSwitchProof>,
    b_combined: Vec<F128>,
    target_combined: F128,
    /// Round-0 sumcheck `(u_0, u_2)` prime over `packed_witness · b_combined`,
    /// consumed by `recursive_prover_with_basis_precomputed_round0`.
    round0_prime: (F128, F128),
    /// Quadratic coefficients of Ligerito's ROUND-1 message in the round-0
    /// fold challenge, accumulated in the same combine pass (fast path with
    /// no packed-direct claims only). Lets the recursive prover's first lane
    /// fold be an O(1) skip round — see [`ligerito::FoldLookahead`].
    round1_lookahead: Option<ligerito::FoldLookahead>,
    /// Direct (basis-free) opening payload: when `Some`, `b_combined` is
    /// empty and the recursion runs [`ligerito::recursive_prover_direct`].
    direct: Option<ligerito::DirectL0>,
}

/// Runs ring_switch over RS claims, observes packed-direct claim values +
/// samples their gammas, then builds `b_combined` (the γ-weighted linear
/// combination of all `rs_eq_ind`s and `eq_ind`s) and `target_combined`.
/// Also computes the round-0 prime as a side effect (cheap since it shares
/// the b_combined pass).
#[allow(clippy::too_many_arguments)]
fn compute_combined_basis_and_target<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    banked_in: &[Option<&ring_switch::BankedShatV>],
    packed_direct: &[PackedDirectClaim],
    direct_c: usize,
    padding: &PaddingSpec,
    challenger: &mut Ch,
    trace: bool,
) -> CombinedClaim {
    let n_rs = x_outers.len();
    let n_pd = packed_direct.len();
    assert!(n_rs + n_pd > 0, "open_batch_mixed: need at least one claim");
    assert!(
        precomputed_s_hat_v.is_empty() || precomputed_s_hat_v.len() == n_rs,
        "precomputed_s_hat_v: must be empty or length {n_rs}, got {}",
        precomputed_s_hat_v.len(),
    );

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // Direct (basis-free) opening: applies when every RS claim has a dense,
    // long-enough suffix and there are no packed-direct claims. v1 builds the
    // banked statistics here with the reference scan (producer plumbing from
    // lincheck/zerocheck is the follow-up optimization); the transcript is
    // unaffected either way.
    let use_direct = direct_c > 0
        && n_pd == 0
        && n_rs > 0
        && x_outers.iter().all(|x| {
            let suffix = &x[1..];
            suffix.len() > direct_c
                && suffix.iter().filter(|&&c| c == F128::ZERO).count()
                    < ring_switch::SPARSE_ZERO_THRESHOLD
        });
    // Prefer caller-supplied banks (captured for free upstream); fall back
    // to the reference scan per claim only where absent.
    let banked: Vec<Option<ring_switch::BankedShatV>> = if use_direct {
        x_outers
            .iter()
            .enumerate()
            .map(|(k, x)| match banked_in.get(k).copied().flatten() {
                Some(_) => None, // borrowed below
                None => Some(ring_switch::banked_s_hat_v_naive(
                    packed_witness,
                    &x[1..],
                    direct_c,
                )),
            })
            .collect()
    } else {
        Vec::new()
    };
    let banked_refs: Vec<Option<&ring_switch::BankedShatV>> = if use_direct {
        (0..x_outers.len())
            .map(|k| {
                banked_in
                    .get(k)
                    .copied()
                    .flatten()
                    .or_else(|| banked[k].as_ref())
            })
            .collect()
    } else {
        Vec::new()
    };

    // 1. Ring-switching for all x_outers.
    let t = std::time::Instant::now();
    let (rs_results, gammas_rs): (
        Vec<(RingSwitchProof, ring_switch::RingSwitchBatchOutput)>,
        Vec<F128>,
    ) = if n_rs > 0 {
        if use_direct {
            ring_switch::prove_batched_padded_direct(
                packed_witness,
                x_outers,
                precomputed_s_hat_v,
                &banked_refs,
                direct_c,
                padding,
                challenger,
            )
        } else {
            ring_switch::prove_batched_padded_with_precomputed(
                packed_witness,
                x_outers,
                precomputed_s_hat_v,
                padding,
                challenger,
            )
        }
    } else {
        (Vec::new(), Vec::new())
    };
    if trace {
        eprintln!(
            "  [open_batch] ring_switch::prove_batched ×{}: {:6.2} ms",
            n_rs,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2. Observe packed-direct claim values + sample γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    let t = std::time::Instant::now();
    use rayon::prelude::*;

    let l = if let Some((_, out)) = rs_results.first() {
        out.rs_eq_ind.len()
    } else {
        1usize << packed_direct[0].point.len()
    };
    debug_assert!(rs_results.iter().all(|(_, o)| o.rs_eq_ind.len() == l));
    debug_assert!(
        packed_direct.iter().all(|pd| 1usize << pd.point.len() == l),
        "all packed-direct claims must share L (= packed witness length)"
    );

    let mut target_combined = F128::ZERO;
    for ((_, output), g) in rs_results.iter().zip(gammas_rs.iter()) {
        target_combined += *g * output.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    if use_direct {
        // No basis is ever materialized: hand the banked claim bundles to
        // the direct recursion entry.
        let mut ring_switches = Vec::with_capacity(rs_results.len());
        let mut bundles = Vec::with_capacity(rs_results.len());
        for (proof, out) in rs_results {
            ring_switches.push(proof);
            match out.rs_eq_ind {
                ring_switch::RsEqInd::Direct(b) => bundles.push(*b),
                _ => unreachable!("direct gate produced a non-direct claim"),
            }
        }
        return CombinedClaim {
            ring_switches,
            b_combined: Vec::new(),
            target_combined,
            round0_prime: (F128::ZERO, F128::ZERO),
            round1_lookahead: None,
            direct: Some(ligerito::DirectL0 { bundles }),
        };
    }

    let rs_baked: Vec<&[F128]> = rs_results
        .iter()
        .filter_map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::Dense(v) => Some(v.as_slice()),
            _ => None,
        })
        .collect();
    // Deferred-dense claims (fused fast path): the per-claim `γ_k·B_k` buffer
    // was never materialized — fold each slot on the fly below and accumulate
    // straight into `b_combined`, saving a 2^(m-7) materialize + readback per
    // claim. Carries (eq_lo, eq_hi, γ-baked table, log₂ B).
    let rs_deferred: Vec<(&[F128], &[F128], &[F128], usize)> = rs_results
        .iter()
        .filter_map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::DeferredDense {
                eq_lo,
                eq_hi,
                table,
            } => Some((
                eq_lo.as_slice(),
                eq_hi.as_slice(),
                table.as_slice(),
                eq_lo.len().trailing_zeros() as usize,
            )),
            _ => None,
        })
        .collect();
    let pd_dense: Vec<(&[F128], F128)> = packed_direct
        .iter()
        .zip(gammas_pd.iter())
        .filter_map(|(pd, g)| match &pd.eq_ind {
            DirectEqInd::Dense(v) => Some((v.as_slice(), *g)),
            _ => None,
        })
        .collect();

    // ---- Build b_combined (γ-weighted sum of all rs_eq_ind + eq_ind) and the
    //      round-0 prime (u_0, u_2 over packed_witness · b_combined).
    let mut b_combined: Vec<F128> = crate::scratch::take_f128(l);

    // Fast path (compression-proof open: claims ab, c; also chain/merkle): every
    // RS claim is a fused DeferredDense fold and no DENSE packed-direct claim
    // needs the per-element combine. Fold all claims block-by-block straight into
    // b_combined — each claim's `e_hi` hoisted once per block, exactly as in
    // `fold_b128_elems_split` — and fuse the round-0 prime in the same pass.
    // Neither the per-claim `γ_k·B_k` buffer nor a combine readback is ever
    // materialized (saves ~2·L writes + 2·L reads of the 2^(m-7) basis).
    //
    // SPARSE packed-direct claims (the chain/merkle I/O claim) do NOT disable
    // this path: they're scatter-added onto b_combined after the fold (with an
    // incremental round-0 prime adjustment), so they only require
    // `pd_dense.is_empty()`, not `packed_direct.is_empty()`. This keeps the two
    // big ab/c claims on the fused fold instead of materializing them.
    let use_fast =
        !rs_deferred.is_empty() && rs_deferred.len() == rs_results.len() && pd_dense.is_empty();

    // The combine is compute-bound (open_combine_probe: ~4.3 ms traffic floor
    // vs ~18 ms total at m=30 on 4 P-threads), and its flat block-parallel
    // shape drains cleanly around slow cores — run it on the all-core (P+E)
    // pool (−29% on the probe on 4P+4E; a wash-to-slight-loss on 10P+4E, so
    // gated on [`crate::ecore_rich_topology`], `FLOCK_ALLCORE=1` overrides).
    // PCS_COMBINE_PCORES_ONLY=1 keeps it on the caller's pool (A/B toggle).
    // Thread count never changes the output bits: every slot is written
    // deterministically and the prime is an XOR reduction (associative +
    // commutative, exact).
    let combine_all_cores =
        std::env::var("PCS_COMBINE_PCORES_ONLY").is_err() && crate::ecore_rich_topology();
    // With no packed-direct claims nothing is scatter-added after the fast
    // path, so the block tail can also accumulate Ligerito's round-1 message
    // coefficients (groups of 4; +1 unreduced mul per slot) — the round-0
    // prime falls out of the same accumulators. See `CombinedClaim`.
    let want_lookahead = use_fast && packed_direct.is_empty();
    let mut round1_lookahead: Option<ligerito::FoldLookahead> = None;
    let b_combined_ref = &mut b_combined;
    let la_ref = &mut round1_lookahead;
    let mut combine = || {
        if use_fast {
            use crate::field::F256Unreduced;
            let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
            debug_assert!(b >= 2 && b.is_multiple_of(2));
            debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
            // Composed-table sweep: `fold_one_slot(·, T)` is F₂-linear, so the
            // per-slot map `lo ↦ fold_one_slot(lo·e_hi, T)` collapses into one
            // composed byte table per claim per block (see
            // `compose_fold_byte_table_into`) — deleting the per-slot field
            // multiply from the L-sized sweep for a per-block table build
            // amortized over `b` slots. Below ~2^12 slots the build doesn't
            // amortize; tiny shapes keep the direct slot-multiply sweep.
            const COMPOSE_MIN_BLOCK: usize = 1 << 12;
            let composed = b >= COMPOSE_MIN_BLOCK;
            let init_ctable = move || {
                if composed {
                    vec![F128::ZERO; ring_switch::FOLD_TABLE_LEN]
                } else {
                    Vec::new()
                }
            };
            let fold_block = |ctable: &mut Vec<F128>, hi: usize, out_block: &mut [F128]| {
                // Accumulate each claim's block: first claim writes, rest add.
                // `e_hi` is read once per claim per block (composed into the
                // byte table, or multiplied per slot), then swept over eq_lo.
                for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred.iter().enumerate() {
                    let e_hi = eq_hi[hi];
                    if composed {
                        ring_switch::compose_fold_byte_table_into(e_hi, table, ctable);
                        if ci == 0 {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot = ring_switch::fold_one_slot(lo, ctable);
                            }
                        } else {
                            for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                                *slot += ring_switch::fold_one_slot(lo, ctable);
                            }
                        }
                    } else if ci == 0 {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    } else {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    }
                }
            };
            if want_lookahead && b.is_multiple_of(4) {
                // Fused prime + round-1 lookahead tail (groups of 4).
                let acc = b_combined_ref
                    .par_chunks_mut(b)
                    .enumerate()
                    .map_init(init_ctable, |ctable, (hi, out_block)| {
                        fold_block(ctable, hi, out_block);
                        let base = hi * b;
                        let mut acc = [F256Unreduced::ZERO; 8];
                        for g in 0..(b / 4) {
                            let i = 4 * g;
                            let fq = [
                                packed_witness[base + i],
                                packed_witness[base + i + 1],
                                packed_witness[base + i + 2],
                                packed_witness[base + i + 3],
                            ];
                            let bq = [
                                out_block[i],
                                out_block[i + 1],
                                out_block[i + 2],
                                out_block[i + 3],
                            ];
                            ligerito::lookahead_accum_group(&fq, &bq, &mut acc);
                        }
                        acc
                    })
                    .reduce(|| [F256Unreduced::ZERO; 8], ligerito::xor_acc8);
                let (msg, la) = ligerito::lookahead_finish(acc);
                *la_ref = Some(la);
                (msg.u_0, msg.u_2)
            } else {
                let (u0, u2) = b_combined_ref
                    .par_chunks_mut(b)
                    .enumerate()
                    .map_init(init_ctable, |ctable, (hi, out_block)| {
                        fold_block(ctable, hi, out_block);
                        // Round-0 prime over this block's pairs (b is even, base is
                        // even). Unreduced 256-bit accumulation, one reduction at
                        // the very end (XOR-linear, bit-identical to reducing per
                        // term).
                        let base = hi * b;
                        let mut u0 = F256Unreduced::ZERO;
                        let mut u2 = F256Unreduced::ZERO;
                        for t in 0..(b / 2) {
                            let s0 = out_block[2 * t];
                            let s1 = out_block[2 * t + 1];
                            let a0 = packed_witness[base + 2 * t];
                            let a1 = packed_witness[base + 2 * t + 1];
                            u0 ^= a0.mul_unreduced(s0);
                            u2 ^= (a0 + a1).mul_unreduced(s0 + s1);
                        }
                        (u0, u2)
                    })
                    .reduce(
                        || (F256Unreduced::ZERO, F256Unreduced::ZERO),
                        |(x0, x2), (y0, y2)| (x0 ^ y0, x2 ^ y2),
                    );
                (u0.reduce(), u2.reduce())
            }
        } else {
            // General path (mixed / sparse / packed-direct): materialize any
            // deferred-dense claims (parallel block fold), then the per-element
            // combine over all dense buffers + packed-direct, matching the
            // original behavior.
            let materialized: Vec<Vec<F128>> = rs_results
                .iter()
                .filter_map(|(_, o)| match &o.rs_eq_ind {
                    ring_switch::RsEqInd::DeferredDense {
                        eq_lo,
                        eq_hi,
                        table,
                    } => Some(ring_switch::fold_b128_from_table(eq_lo, eq_hi, table)),
                    _ => None,
                })
                .collect();
            let mut rs_dense_all: Vec<&[F128]> = rs_baked.clone();
            rs_dense_all.extend(materialized.iter().map(|v| v.as_slice()));
            let prime = b_combined_ref
                .par_chunks_mut(2)
                .enumerate()
                .map(|(i, chunk)| {
                    let mut b0 = F128::ZERO;
                    let mut b1 = F128::ZERO;
                    for v in rs_dense_all.iter() {
                        b0 += v[2 * i];
                        b1 += v[2 * i + 1];
                    }
                    for (v, g) in pd_dense.iter() {
                        b0 += *g * v[2 * i];
                        b1 += *g * v[2 * i + 1];
                    }
                    chunk[0] = b0;
                    chunk[1] = b1;
                    let a0 = packed_witness[2 * i];
                    let a1 = packed_witness[2 * i + 1];
                    (a0 * b0, (a0 + a1) * (b0 + b1))
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                );
            for v in materialized {
                crate::scratch::give_f128(v);
            }
            prime
        }
    };
    let (mut round0_u0, mut round0_u2) = if combine_all_cores {
        crate::all_core_pool().install(combine)
    } else {
        combine()
    };
    let mut adjust_prime_for_delta = |idx: usize, delta: F128| {
        let pair = idx / 2;
        let a0 = packed_witness[2 * pair];
        let a1 = packed_witness[2 * pair + 1];
        if idx & 1 == 0 {
            round0_u0 += a0 * delta;
        }
        round0_u2 += (a0 + a1) * delta;
    };
    for (_, output) in rs_results.iter() {
        if let ring_switch::RsEqInd::Sparse { entries, .. } = &output.rs_eq_ind {
            // Post-combine mutation of b_combined: the round-1 lookahead
            // coefficients (if any) are now stale. Unreachable when they are
            // emitted (fast path = no sparse RS claims), but keep the
            // invalidation in case the emission condition ever widens.
            if !entries.is_empty() {
                round1_lookahead = None;
            }
            for &(idx, val) in entries {
                b_combined[idx] += val;
                adjust_prime_for_delta(idx, val);
            }
        }
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        if let DirectEqInd::Sparse(eq) = &pd.eq_ind {
            round1_lookahead = None; // see above — pd claims never emit coeffs
            // Scatter-add the sparse claim and fold its round-0 prime
            // contribution in the SAME pass (O(live positions)), instead of a
            // full O(L) re-pass over b_combined. The prime is linear in
            // b_combined, so the delta from scattering `g·eq` equals
            // Σ adjust_prime_for_delta(idx, g·val) over the live positions.
            let (du0, du2) = sparse_scatter_add_parallel(&mut b_combined, packed_witness, eq, *g);
            round0_u0 += du0;
            round0_u2 += du2;
        }
    }
    if trace {
        eprintln!(
            "  [open_batch] combine rs_eq_ind (L={}, rs×{}, pd×{}, b={}): {:6.2} ms",
            l,
            n_rs,
            n_pd,
            rs_deferred.first().map_or(0, |d| d.0.len()),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    CombinedClaim {
        ring_switches: rs_results
            .into_iter()
            .map(|(p, o)| {
                // The per-claim rs_eq_ind (L F128s) dies here — recycle it.
                if let ring_switch::RsEqInd::Dense(v) = o.rs_eq_ind {
                    crate::scratch::give_f128(v);
                }
                p
            })
            .collect(),
        b_combined,
        target_combined,
        round0_prime: (round0_u0, round0_u2),
        direct: None,
        round1_lookahead,
    }
}

/// Parallel sparse scatter-add: `b_combined[scatter_idx(c)] += gamma * eq.live_tensor[c]`
/// for every `c`. Partitions `c`-space across rayon threads; since
/// [`SparseEqTensor::scatter_idx`] is monotonic in `c` (live_positions sorted
/// ascending), each thread's scattered indices fall in a contiguous, disjoint
/// range of `b_combined`. Splits `b_combined` at the chunk boundaries via
/// `split_at_mut`, then writes scatter-adds into the disjoint mutable slices —
/// safe rust, no atomics.
/// Scatter-add `gamma · eq` into `b_combined` and return the resulting
/// round-0 prime delta `(Δu0, Δu2)`. Because the prime is linear in
/// `b_combined`, adding `delta = gamma·val` at index `idx` changes the prime by
/// `Δu0 += a0·delta` (if `idx` even) and `Δu2 += (a0+a1)·delta`, where
/// `a0 = packed_witness[2·pair]`, `a1 = packed_witness[2·pair+1]`,
/// `pair = idx/2`. Computing it here (O(live positions)) avoids a full O(L)
/// re-pass over `b_combined` at the call site.
fn sparse_scatter_add_parallel(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    eq: &SparseEqTensor,
    gamma: F128,
) -> (F128, F128) {
    use rayon::prelude::*;

    let c_total = eq.live_tensor.len();
    if c_total == 0 {
        return (F128::ZERO, F128::ZERO);
    }
    let n_threads = rayon::current_num_threads().max(1);
    let c_per_chunk = c_total.div_ceil(n_threads).max(1);
    let actual_n_chunks = c_total.div_ceil(c_per_chunk);

    // Boundaries in `b_combined` index space. `b_boundaries[i]` is where chunk
    // `i` starts. `b_boundaries[i+1] − b_boundaries[i]` is chunk `i`'s slice
    // length. The last chunk extends to `b_combined.len()` to absorb any tail
    // positions beyond the maximum scatter idx (those contain only dense
    // contributions from the parallel pass).
    let b_boundaries: Vec<usize> = (0..=actual_n_chunks)
        .map(|i| {
            if i == 0 {
                0
            } else if i == actual_n_chunks {
                b_combined.len()
            } else {
                eq.scatter_idx(i * c_per_chunk)
            }
        })
        .collect();
    debug_assert!(b_boundaries.windows(2).all(|w| w[0] <= w[1]));

    // Disjoint mutable slices via repeated split_at_mut.
    let mut remaining: &mut [F128] = b_combined;
    let mut slices: Vec<&mut [F128]> = Vec::with_capacity(actual_n_chunks);
    for i in 1..actual_n_chunks {
        let split_at = b_boundaries[i] - b_boundaries[i - 1];
        let (left, right) = remaining.split_at_mut(split_at);
        slices.push(left);
        remaining = right;
    }
    slices.push(remaining);
    debug_assert_eq!(slices.len(), actual_n_chunks);

    slices
        .into_par_iter()
        .enumerate()
        .map(|(t, slice)| {
            let c_lo = t * c_per_chunk;
            let c_hi = ((t + 1) * c_per_chunk).min(c_total);
            let b_lo = b_boundaries[t];
            let mut du0 = F128::ZERO;
            let mut du2 = F128::ZERO;
            for c in c_lo..c_hi {
                let val = eq.live_tensor[c];
                let idx = eq.scatter_idx(c);
                let delta = gamma * val;
                slice[idx - b_lo] += delta;
                // Round-0 prime delta for this scattered position.
                let pair = idx / 2;
                let a0 = packed_witness[2 * pair];
                let a1 = packed_witness[2 * pair + 1];
                if idx & 1 == 0 {
                    du0 += a0 * delta;
                }
                du2 += (a0 + a1) * delta;
            }
            (du0, du2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
}

/// Verifier reference to a packed-direct claim: the multilinear point at
/// which `ẑ_packed` was claimed equal to `value`. The verifier owns the data
/// (it appears in the public statement of whatever produced the claim, e.g.
/// the chain shift sumcheck output).
#[derive(Clone, Copy, Debug)]
pub struct PackedDirectClaimRef<'a> {
    pub point: &'a [F128],
    pub value: F128,
}

/// Verify a mixed-claim batched opening (mirror of
/// [`open_batch_mixed_ligerito_with_precomputed_s_hat_v`]). Uses
/// `ring_switch::verify_succinct` per claim (no dense `rs_eq_ind`
/// materialization), then drives the succinct recursive Ligerito verifier,
/// evaluating the combined basis only at the residual point.
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_ligerito_mixed<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    skip_weights: &[&[F128]],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    proof: &BatchOpeningProofLigerito,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyError> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(skip_weights.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    assert_eq!(proof.ring_switches.len(), n_rs);
    assert!(n_rs + n_pd > 0);

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1. Ring-switch SUCCINCT verify per claim — gets sumcheck_claim and a
    //    length-128 `eq_r_dprime` instead of the dense `rs_eq_ind`. Saves
    //    ~16 MB allocation at m=29.
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let out = ring_switch::verify_succinct(
            claims[i],
            skip_weights[i],
            x_outers[i],
            &proof.ring_switches[i],
            challenger,
        )
        .map_err(VerifyError::RingSwitch)?;
        rs_outputs.push(out);
    }
    let gammas_rs: Vec<F128> = (0..n_rs).map(|_| challenger.sample_f128()).collect();

    // 2. PD claim values + γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    // 3. target_combined from succinct rs claims + PD values.
    let mut target_combined = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(gammas_rs.iter()) {
        target_combined += *g * out.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    // 4. Batch evaluator: returns b_combined at all yr positions in one call.
    //    For RS claims, precompute the ring_switch tensor PREFIX once (over
    //    the ris part) and only re-do the yr_log_n-step suffix per y.
    //    For PD claims, precompute eq prefix factors over ris and finish per y.
    //    For BLAKE3 m=30: ris is 19 dims, yr is 4 dims → 19× prefix reuse.
    let log_n = commitment.params.m - LOG_PACKING;
    let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
        use crate::zerocheck::multilinear::eq_eval;
        let yr_len = 1usize << yr_log_n;
        let prefix_len = ris.len();

        // ---- RS claim prefixes ----
        let rs_prefixes: Vec<crate::pcs::tensor_algebra::TensorAlgebra> = rs_outputs
            .iter()
            .zip(x_outers.iter())
            .map(|(_out, x_outer)| {
                // x_outer[1..] has length log_n; we feed only the ris prefix.
                ring_switch::eval_rs_eq_prefix(&x_outer[1..1 + prefix_len], ris)
            })
            .collect();

        // ---- PD claim prefix scalars ----
        // eq(pd.point, point) factors over coordinates; precompute the prefix product.
        let pd_prefix_scalars: Vec<F128> = packed_direct
            .iter()
            .map(|pd| eq_eval(&pd.point[..prefix_len], ris))
            .collect();

        // ---- Per-y assembly (parallel over yr positions; each y is independent).
        //      y_suffix is binary (bits of y), so we use the binary-query
        //      specializations of eval_rs_eq_finish / eq_eval — each suffix
        //      step collapses to a single scale_vertical / scalar product.
        use rayon::prelude::*;
        debug_assert!(yr_log_n <= 32, "yr_log_n > 32 not supported by binary path");
        (0..yr_len)
            .into_par_iter()
            .map(|y| {
                let y_bits = y as u32;
                let mut sum = F128::ZERO;
                for (((out, g), x_outer), prefix) in rs_outputs
                    .iter()
                    .zip(gammas_rs.iter())
                    .zip(x_outers.iter())
                    .zip(rs_prefixes.iter())
                {
                    sum += *g
                        * ring_switch::eval_rs_eq_finish_from_prefix_binary_q(
                            prefix,
                            &x_outer[1 + prefix_len..],
                            y_bits,
                            &out.eq_r_dprime,
                        );
                }
                for ((pd, g), prefix_scalar) in packed_direct
                    .iter()
                    .zip(gammas_pd.iter())
                    .zip(pd_prefix_scalars.iter())
                {
                    sum += *g
                        * *prefix_scalar
                        * crate::zerocheck::multilinear::eq_eval_binary_x(
                            &pd.point[prefix_len..],
                            y_bits,
                        );
                }
                sum
            })
            .collect()
    };

    // 5. Drive ligerito SUCCINCT verifier — eval_b_residual is called ONCE
    //    at the residual check (returns all yr_len values in one batch).
    let ok = ligerito::recursive_verifier_with_basis_succinct(
        lig_config,
        &proof.ligerito,
        log_n,
        target_combined,
        &commitment.root,
        eval_b_residual,
        challenger,
    );
    if !ok {
        return Err(VerifyError::Ligerito);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::zerocheck::multilinear::lagrange_weights_naive;
    use crate::zerocheck::univariate_skip::build_eq;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn zhat_skip_reference(z: &[bool], m: usize, z_skip: F128, x_outer: &[F128]) -> F128 {
        const K_SKIP: usize = 6;
        let ell = 1usize << K_SKIP;
        let lambda = lagrange_weights_naive(K_SKIP, z_skip);
        let eq_outer = build_eq(x_outer);
        let mut acc = F128::ZERO;
        for i_outer in 0..(1usize << (m - K_SKIP)) {
            let base = i_outer * ell;
            let mut inner = F128::ZERO;
            for i_skip in 0..ell {
                if z[base + i_skip] {
                    inner += lambda[i_skip];
                }
            }
            acc += eq_outer[i_outer] * inner;
        }
        acc
    }

    /// End-to-end Ligerito backend roundtrip through pcs::open_batch_mixed_ligerito
    /// and verify_opening_batch_ligerito_mixed. Single ring-switched claim
    /// (no PD — PD path is task #11).
    #[test]
    #[ignore] // Heavier — ~50-100 ms; run with `cargo test pcs_ligerito_roundtrip -- --ignored --nocapture`
    fn pcs_ligerito_backend_roundtrip() {
        let m = 22usize;
        let mut rng = Rng::new(0x11_6E_2170);
        let z = rng.bits(1 << m);
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

        // PcsParams MUST set log_batch_size = ligerito_initial_k for L0 reuse.
        let initial_k = 6;
        let params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: initial_k,
            profile: Default::default(),
            merkle_hash: Default::default(),
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        let recursive_ks = vec![3usize, 3, 3];
        let log_inv_rates = vec![1usize, 3, 4, 6];
        let queries: Vec<usize> = log_inv_rates
            .iter()
            .map(|&r| crate::pcs::ligerito::udr_queries(r))
            .collect();
        let grinding_bits = vec![0usize; log_inv_rates.len()];
        let n_levels = log_inv_rates.len();
        let lig_p_cfg = crate::pcs::ligerito::ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };
        let lig_v_cfg = crate::pcs::ligerito::VerifierConfig {
            log_inv_rates,
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };

        let mut ch_p = FsChallenger::new(b"flock-test-lig-v0");
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
            z_packed.clone(),
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m),
            &lig_p_cfg,
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[rs_claim],
            &[&lagrange_weights_naive(6, z_skip)],
            &[x_outer.as_slice()],
            &[],
            &proof,
            &lig_v_cfg,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));
    }

    /// The direct (basis-free) opening produces a byte-identical proof to the
    /// incumbent basis path and verifies. Production config shape (grinding
    /// zeroed for speed); both arms in-process via [`OPEN_DIRECT_FORCE`].
    /// (The neighboring hand-rolled-config roundtrip test has pre-existing
    /// config rot — this one derives its configs from `prover_config_for`.)
    #[test]
    #[ignore] // Heavier; run with `cargo test pcs_direct_open -- --ignored`
    fn pcs_direct_open_proof_identical() {
        use std::sync::atomic::Ordering;
        let m = 22usize;
        let initial_k = 6usize;
        let mut rng = Rng::new(0xD12EC70u64);
        let z = rng.bits(1 << m);
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

        let mut lig_p_cfg = crate::pcs::ligerito::prover_config_for(
            m - LOG_PACKING,
            initial_k,
            Default::default(),
        )
        .expect("production prover config");
        let mut lig_v_cfg = crate::pcs::ligerito::verifier_config_for(
            m - LOG_PACKING,
            initial_k,
            Default::default(),
        )
        .expect("production verifier config");
        // Zero the PoW grinding on both sides (test speed; they must match).
        for b in lig_p_cfg.grinding_bits.iter_mut() {
            *b = 0;
        }
        for b in lig_p_cfg.fold_grinding_bits.iter_mut() {
            *b = 0;
        }
        for b in lig_v_cfg.grinding_bits.iter_mut() {
            *b = 0;
        }
        for b in lig_v_cfg.fold_grinding_bits.iter_mut() {
            *b = 0;
        }

        let params = PcsParams {
            m,
            log_inv_rate: lig_p_cfg.log_inv_rates[0],
            log_batch_size: initial_k,
            profile: Default::default(),
            merkle_hash: Default::default(),
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        let open = |direct: bool| -> BatchOpeningProofLigerito {
            OPEN_DIRECT_DISABLE.store(!direct, Ordering::Relaxed);
            let mut ch = FsChallenger::new(b"flock-test-lig-v0");
            let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
                z_packed.clone(),
                &prover_data,
                &commitment,
                &[x_outer.as_slice()],
                &[],
                &[],
                &PaddingSpec::dense(m),
                &lig_p_cfg,
                &mut ch,
            );
            OPEN_DIRECT_DISABLE.store(false, Ordering::Relaxed);
            proof
        };
        let proof_dense = open(false);
        let proof_direct = open(true);
        assert_eq!(proof_direct.ring_switches, proof_dense.ring_switches, "ring_switches");
        let (a, b) = (&proof_direct.ligerito, &proof_dense.ligerito);
        for (i, (x, y)) in a
            .sumcheck_transcript
            .iter()
            .zip(b.sumcheck_transcript.iter())
            .enumerate()
        {
            assert_eq!(x, y, "sumcheck message {i} diverges");
        }
        assert_eq!(
            a.sumcheck_transcript.len(),
            b.sumcheck_transcript.len(),
            "transcript length"
        );
        assert_eq!(a.initial_root, b.initial_root, "initial_root");
        assert_eq!(a.recursive_roots, b.recursive_roots, "recursive_roots");
        assert_eq!(
            proof_direct, proof_dense,
            "direct open must be byte-identical to the basis path"
        );

        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[rs_claim],
            &[&lagrange_weights_naive(6, z_skip)],
            &[x_outer.as_slice()],
            &[],
            &proof_direct,
            &lig_v_cfg,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("ligerito verify rejected direct-open proof: {e:?}"));
    }
}
