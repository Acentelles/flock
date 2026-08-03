//! Top-level R1CS verifier: walks the challenger in lockstep with the
//! prover, runs `zerocheck::verify` and `lincheck::verify`, derives the two
//! ZClaims, and verifies the PCS openings at those points against the
//! witness commitment.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck;
use crate::pcs::{self, Commitment};
use crate::proof::{R1csClaim, R1csProofLigerito, ZClaim};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    PcsAb(pcs::VerifyError),
    PcsC(pcs::VerifyError),
    /// The jagged-path batched opening rejected (see [`verify_ligerito_jagged`]).
    PcsOpen(pcs::VerifyErrorOpen),
    /// The element-region PIOP rejected.
    Element(crate::element_r1cs::union::VerifyError),
    /// A mixed-class proof carries a class sub-proof the registry has no type
    /// for, or omits one it does — the statement and the proof disagree on
    /// which PIOPs ran.
    ClassMismatch,
}

/// Per-phase wall-clock timings (seconds) of a verify, for benchmark cost
/// breakdowns. Produced by [`verify_ligerito_timed`] (direct).
/// Benchmark-only.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifyPhaseTimings {
    /// `zerocheck::verify` — the zerocheck PIOP replay.
    pub zerocheck_s: f64,
    /// The full lincheck verify (`lincheck::verify` / `lincheck::verify_union`)
    /// including the per-type comb construction below.
    pub lincheck_s: f64,
    /// UNION only: the per-type α-batched comb construction inside the union
    /// lincheck verify — the `O(Σ_t nnz_t)` `fold_alpha_batched` over every
    /// slot, i.e. the multi-slot circuit replay. `0.0` on the direct path,
    /// whose single-table lincheck folds the comb in lockstep with the
    /// sumcheck (no separable phase).
    pub lincheck_comb_s: f64,
    /// The batched PCS opening verify (`verify_claims_ligerito`).
    pub open_s: f64,
}

/// Dedicated single-thread rayon pool that the verifier runs inside.
///
/// The verifier is intentionally single-threaded — matching the convention of
/// comparable provers (binius64, plonky3, hashcaster all ship serial
/// verifiers) and keeping reported verify times honest single-core numbers.
/// The verify path shares several `par_*` helpers with the (multi-threaded)
/// prover — e.g. `lincheck::fold_alpha_batched`, `sumcheck_bind_top_in_place_par`,
/// and the Ligerito residual eval — so rather than fork every shared helper, the
/// reusable verify cores (`verify_core`, `verify_claims_ligerito`)
/// run their body via `verifier_pool().install(..)`. Any `par_iter` reached from
/// there uses this 1-thread pool and collapses onto a single worker, without
/// touching the prover's use of the global pool.
/// Thread count for [`verifier_pool`]. **1 in production** — the override
/// (`FLOCK_VERIFY_THREADS`) exists only so a benchmark can ask what the verify
/// would cost with the prover's parallelism, since the pool below is otherwise
/// the one place that decision is made. Read once per process, so a single run
/// measures a single configuration.
fn verifier_threads() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FLOCK_VERIFY_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    })
}

fn verifier_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(verifier_threads())
            // The whole verify body runs on this worker — including the deep
            // recursive Ligerito verifier — so give it an ample stack. A rayon
            // worker otherwise defaults to ~2 MiB (vs the 8 MiB main thread),
            // which the recursion overflows.
            .stack_size(64 * 1024 * 1024)
            .thread_name(|_| "flock-verify".to_string())
            .build()
            .expect("build single-thread verifier pool")
    })
}

/// Verify an R1CS proof: replay zerocheck + lincheck → the two base z-claims,
/// then verify the batched Ligerito PCS opening covering both.
pub fn verify_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    let (ab, c) = verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )?;
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsAb)?;
    Ok(R1csClaim { ab, c })
}

/// [`verify_ligerito`] with per-phase timers. Splits the verify into
/// zerocheck-verify / lincheck-verify / opening-verify (the single-table
/// lincheck has no separable comb phase, so `lincheck_comb_s == 0`). Kept in
/// lockstep with `verify_ligerito`; benchmark-only, production path
/// undisturbed.
pub fn verify_ligerito_timed<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(R1csClaim, VerifyPhaseTimings), VerifyError> {
    use std::time::Instant;
    // PIOP replay (bind + zerocheck + lincheck) on the 1-thread pool.
    let (ab, c, zerocheck_s, lincheck_s) =
        verifier_pool().install(|| -> Result<(ZClaim, ZClaim, f64, f64), VerifyError> {
            crate::proof::bind_statement(challenger, r1cs, commitment);
            let t0 = Instant::now();
            let zc_claim = zerocheck::verify(r1cs.m, &proof.zerocheck, challenger)
                .map_err(VerifyError::Zerocheck)?;
            let zerocheck_s = t0.elapsed().as_secs_f64();
            let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
            let t0 = Instant::now();
            let lc_claim = lincheck::verify(
                r1cs.m,
                r1cs.k_log,
                r1cs.k_skip,
                lincheck_circuit,
                &x_ab,
                zc_claim.a_eval,
                zc_claim.b_eval,
                &proof.lincheck,
                challenger,
            )
            .map_err(VerifyError::Lincheck)?;
            let lincheck_s = t0.elapsed().as_secs_f64();
            let ab = ZClaim {
                point: r1cs.ab_claim_point(
                    lc_claim.r_inner_skip,
                    &lc_claim.r_inner_rest,
                    &x_ab.x_outer,
                ),
                value: lc_claim.w,
            };
            let c = ZClaim {
                point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
                value: zc_claim.c_eval,
            };
            Ok((ab, c, zerocheck_s, lincheck_s))
        })?;

    let t0 = std::time::Instant::now();
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsAb)?;
    let open_s = t0.elapsed().as_secs_f64();

    let t = VerifyPhaseTimings {
        zerocheck_s,
        lincheck_s,
        lincheck_comb_s: 0.0,
        open_s,
    };
    Ok((R1csClaim { ab, c }, t))
}

/// Statement-binding selector for the union verify path. One variant on
/// this branch (`recursion_circuit` adds `Circuit`); kept as an enum so the
/// binding dispatch below stays a match (mirror of the prove-side enum in
/// `flock_prover::prover`).
enum UnionVerifyBinding {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment root
    /// ([`crate::union::UnionInstance::bind_statement`]).
    Mixed,
}

/// The MERGED-transport union verifier (wire v6) — the Mixed protocol's
/// verify entry for BOOLEAN-only registries: a thin wrapper over
/// [`verify_ligerito_union_mixed_class`] (the one shared
/// body). Handles both lane-major and power-of-two commitments (dispatched
/// on `commitment.params.num_lanes`, which the shared body's
/// params-equality check pins to the count-derived value).
pub fn verify_ligerito_union<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMergedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    // Mirror of the prove-side guard: this entry consumes `R1csClaim` —
    // structurally boolean-only.
    assert!(
        !union.has_element(),
        "this entry is boolean-only; element registries go through \
         verify_ligerito_union_mixed_class"
    );
    // Repackage as a boolean-only mixed-class proof and run the one shared
    // verify body (the two-body split died with the jagged transport). The
    // clone is a few hundred KB against a multi-ms verify.
    let mixed = crate::proof::R1csProofMixedClassMerged {
        boolean: Some(crate::proof::BooleanPiopProof {
            zerocheck: proof.zerocheck.clone(),
            lincheck: proof.lincheck.clone(),
        }),
        element: None,
        pcs_open: proof.pcs_open.clone(),
    };
    let claims = verify_ligerito_union_mixed_class(
        union, circuits, commitment, &mixed, pcs_params, challenger,
    )?;
    Ok(claims.boolean.expect("asserted boolean-only above"))
}

/// [`verify_ligerito_jagged_union_mixed_class`] over the MERGED transport.
///
/// Same statement, same PIOP replay; only the opening differs. The element
/// class's two claims ride as packed-direct claims, which the merged
/// transport carries by expressing each weight as the `F₂`-linear map
/// `x ↦ γ·x` — indistinguishable, to its per-claim weight builder, from a
/// ring-switched claim's Φ-fold.
pub fn verify_ligerito_union_mixed_class<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMixedClassMerged,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<crate::proof::UnionClassClaims, VerifyError> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        pcs_params,
        challenger,
    )?;

    // Same construction as the boolean-only merged verifier: the PCS point
    // is `x_inner_rest ‖ x_outer`, with the skip coordinate carried
    // separately in `z_skip`.
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<F128> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let log_n = pcs_params.m - pcs::LOG_PACKING;
    let lig_v_config = crate::pcs::ligerito::verifier_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| {
            pcs::verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsOpen)?;
    Ok(claims)
}

/// Shared PIOP replay for both union verify shapes: statement binding, the
/// boolean class's zerocheck + lincheck over the `M_bool` prefix subcube, then
/// the element region's PIOP. Returns the per-class claims and the element
/// class's `(point, value)` pairs for the packed-direct intake.
///
/// Runs on the 1-thread verifier pool, like every other verify core.
#[allow(clippy::too_many_arguments)]
fn verify_union_piops<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    binding: UnionVerifyBinding,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    boolean: Option<&crate::proof::BooleanPiopProof>,
    element: Option<&crate::element_r1cs::union::Proof>,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(crate::proof::UnionClassClaims, Vec<(Vec<F128>, F128)>), VerifyError> {
    // The commitment is to the DENSE stack q (M4/M5): PcsParams.m is the
    // dense variable count — count-dependent under height-n_t stacking,
    // derived from the declared counts — while the PIOP and the
    // virtual-opening sumcheck run over the M-variable padded address space.
    assert_eq!(
        pcs_params.m,
        union.dense_m(),
        "PcsParams.m must equal the union's dense_m (committed stack size)"
    );
    // The proof carries `commitment.params`, and the opening reads its
    // `num_ntts()` for the L0 leaf width and the lane-grid rotation — but the
    // transcript binds only the commitment ROOT, so those params are
    // ATTACKER-CONTROLLED. The honest lane count is count-derived
    // (`UnionInstance::commit_lanes`, like `dense_m`), so require the
    // commitment to carry exactly it; a mismatch is a rejection, not a panic.
    if commitment.params.m != pcs_params.m
        || commitment.params.log_batch_size != pcs_params.log_batch_size
        || commitment.params.log_inv_rate != pcs_params.log_inv_rate
        || commitment.params.num_ntts() != pcs_params.num_ntts()
    {
        return Err(VerifyError::PcsOpen(crate::pcs::VerifyErrorOpen::Ligerito));
    }
    // Verification is single-threaded; run the PIOP replay on the dedicated
    // 1-thread pool, like the opening verify after it.
    type PiopOut = (crate::proof::UnionClassClaims, Vec<(Vec<F128>, F128)>);
    verifier_pool().install(|| -> Result<PiopOut, VerifyError> {
        match binding {
            UnionVerifyBinding::Mixed => union.bind_statement(challenger, commitment),
        }

        let bool_claim = match boolean {
            Some(piop) => {
                // The boolean PIOP runs over the BOOLEAN REGION only — the
                // prefix subcube `[0, 2^M_bool)`, `M_bool = M` for a
                // boolean-only registry. (The element region cannot join this
                // sum: `c = z` there.)
                let zc_claim = zerocheck::verify(union.m_bool(), &piop.zerocheck, challenger)
                    .map_err(VerifyError::Zerocheck)?;
                let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
                // The union-column lincheck (one circuit per BOOLEAN slot, in
                // slot order); the declared counts additionally bind through
                // the per-type const-pin target terms.
                let lc_claim = lincheck::verify_union(
                    union,
                    circuits,
                    &x_ab,
                    zc_claim.a_eval,
                    zc_claim.b_eval,
                    &piop.lincheck,
                    challenger,
                )
                .map_err(VerifyError::Lincheck)?;
                Some(R1csClaim {
                    ab: ZClaim {
                        point: union.ab_claim_point(
                            lc_claim.r_inner_skip,
                            &lc_claim.r_inner_rest,
                            &x_ab.x_outer,
                        ),
                        value: lc_claim.w,
                    },
                    c: ZClaim {
                        point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
                        value: zc_claim.c_eval,
                    },
                })
            }
            None => None,
        };

        let el_claim = match element {
            Some(p) => Some(
                crate::element_r1cs::union::verify(union, p, challenger)
                    .map_err(VerifyError::Element)?,
            ),
            None => None,
        };
        let packed_direct = el_claim
            .as_ref()
            .map(|c: &crate::element_r1cs::union::Claims| {
                vec![
                    (c.c_point.clone(), c.c_value),
                    (c.lc_point.clone(), c.lc_value),
                ]
            })
            .unwrap_or_default();

        Ok((
            crate::proof::UnionClassClaims {
                boolean: bool_claim,
                element: el_claim,
            },
            packed_direct,
        ))
    })
}

/// Verify a batched PCS opening over an arbitrary list of `ẑ`-claims — the
/// mirror of `flock_prover::prover::open_claims_with_precomputed_ligerito`.
/// Relation wrappers (e.g. the hash chain) reuse this with their own appended
/// claims. Must run at the same transcript position as the prover's open.
pub fn verify_claims_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProofLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_claims_ligerito_inner(commitment, claims, pcs_open, pcs_params, challenger)
    })
}

fn verify_claims_ligerito_inner<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProofLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    let z_skips: Vec<F128> = claims.iter().map(|c| c.point.z_skip).collect();
    let values: Vec<F128> = claims.iter().map(|c| c.value).collect();
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    pcs::verify_opening_batch_ligerito_mixed(
        commitment,
        &values,
        &z_skips,
        &x_refs,
        &[],
        pcs_open,
        &lig_v_config,
        challenger,
    )
}

/// Replay bind → zerocheck → lincheck and reconstruct the two base z-claims
/// (`ab`, `c`), stopping before the PCS open. Mirror of
/// `flock_prover::prover::prove_fast_core`; relation wrappers reuse this then call
/// [`verify_claims_ligerito`] over `[ab, c, …]`.
pub fn verify_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &zerocheck::ZerocheckProof,
    lincheck_proof: &lincheck::LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), VerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_core_inner(
            r1cs,
            zerocheck_proof,
            lincheck_proof,
            commitment,
            lincheck_circuit,
            challenger,
        )
    })
}

fn verify_core_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &zerocheck::ZerocheckProof,
    lincheck_proof: &lincheck::LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), VerifyError> {
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // ---- Bind FS transcript to the statement (mirrors prover::prove).
    let t = std::time::Instant::now();
    crate::proof::bind_statement(challenger, r1cs, commitment);
    if trace {
        eprintln!(
            "      [vco] bind_statement: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Zerocheck.
    let t = std::time::Instant::now();
    let zc_claim =
        zerocheck::verify(r1cs.m, zerocheck_proof, challenger).map_err(VerifyError::Zerocheck)?;
    if trace {
        eprintln!(
            "      [vco] zerocheck::verify: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build lincheck's shared quirky point from the zerocheck output
    // (layout-aware: the mlv challenges are address-ordered).
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // ---- Lincheck. v_a, v_b come from the zerocheck's final â, b̂ evals.
    let t = std::time::Instant::now();
    let lc_claim = lincheck::verify(
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        lincheck_circuit,
        &x_ab,
        zc_claim.a_eval,
        zc_claim.b_eval,
        lincheck_proof,
        challenger,
    )
    .map_err(VerifyError::Lincheck)?;
    if trace {
        eprintln!(
            "      [vco] lincheck::verify: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build the two z-claims (must match what `prove` returned).
    // Layout-aware: the ZClaim points are address-ordered for the PCS.
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    // c-claim is already a z-claim since `C = I` ⇒ ĉ = ẑ.
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    Ok((ab, c))
}

#[cfg(test)]
mod tests {
    /// The verifier is intentionally single-threaded: every `par_*` reached
    /// from a verify core must collapse onto the one-thread `verifier_pool`.
    /// Guard the invariant so a future `ThreadPoolBuilder` tweak can't silently
    /// re-parallelize verification.
    ///
    /// (The end-to-end prove → verify roundtrip and tamper-rejection tests live
    /// in `flock-prover`'s `tests/verifier_roundtrip.rs`, since they need the
    /// prove path.)
    #[test]
    fn verifier_pool_is_single_threaded() {
        let n = super::verifier_pool().install(rayon::current_num_threads);
        assert_eq!(n, 1, "verifier_pool must have exactly one worker thread");
    }
}
