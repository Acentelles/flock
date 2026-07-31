//! Top-level R1CS verifier: walks the challenger in lockstep with the
//! prover, runs `zerocheck::verify` and `lincheck::verify`, derives the two
//! ZClaims, and verifies the PCS openings at those points against the
//! witness commitment.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck;
use crate::pcs::{self, Commitment};
use crate::proof::{R1csClaim, R1csProofJaggedLigerito, R1csProofLigerito, ZClaim};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    PcsAb(pcs::VerifyError),
    PcsC(pcs::VerifyError),
    /// The jagged-path batched opening rejected (see [`verify_ligerito_jagged`]).
    PcsJagged(pcs::VerifyErrorJagged),
    /// The element-region PIOP rejected.
    Element(crate::element_r1cs::union::VerifyError),
    /// A mixed-class proof carries a class sub-proof the registry has no type
    /// for, or omits one it does — the statement and the proof disagree on
    /// which PIOPs ran.
    ClassMismatch,
    /// The wiring (copy-constraint) argument rejected.
    Wiring(crate::circuit::WiringError),
    /// The circuit and the union instance are not the same statement: a
    /// different registry, or gate counts that are not the union's declared
    /// counts. A rejection, not a panic — both come from the caller.
    CircuitMismatch,
}

/// Per-phase wall-clock timings (seconds) of a verify, for benchmark cost
/// breakdowns. Produced by [`verify_ligerito_timed`] (direct) and
/// [`verify_ligerito_jagged_union_timed`] (union). Benchmark-only.
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
    /// The batched PCS opening verify (`verify_claims_ligerito` /
    /// `verify_claims_jagged_ligerito`).
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

/// [`verify_ligerito`] with per-phase timers — the direct-path counterpart
/// of [`verify_ligerito_jagged_union_timed`]. Splits the verify into
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

/// Verify an R1CS proof whose opening went through the **jagged transport**:
/// replay zerocheck + lincheck → the two base z-claims (identical to
/// [`verify_ligerito`] — the PIOP is shared), then verify the jagged-path
/// batched opening covering both. Mirror of
/// `flock_prover::prover::prove_fast_ligerito_jagged_from_witness`.
pub fn verify_ligerito_jagged<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
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
    verify_claims_jagged_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &[],
        &r1cs.jagged_heights(),
        r1cs.n_log(),
        r1cs.m,
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;
    Ok(R1csClaim { ab, c })
}

/// Statement-binding selector for the union verify path. Private: the two
/// public entries below fix the variant (mirror of the prove-side enum in
/// `flock_prover::prover`).
enum UnionVerifyBinding<'a> {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment root
    /// ([`crate::union::UnionInstance::bind_statement`]).
    Mixed,
    /// The circuit binding: [`UnionVerifyBinding::Mixed`] plus the circuit
    /// digest and the public words.
    Circuit {
        circuit: &'a crate::circuit::Circuit,
        public: &'a [F128],
    },
    /// The M1/M2 differential-harness binding: the slot's single-table
    /// `BlockR1cs` statement digest. Single-type registries only; not a
    /// protocol mode.
    SingleTypeHarness(&'a BlockR1cs),
}

/// Verify a proof produced by the **union prove entry**
/// (`flock_prover::prover::prove_fast_ligerito_jagged_union`): bind the
/// statement as `flock-mixed-v1` (registry digest + counts vector +
/// commitment root, [`crate::union::UnionInstance::bind_statement`]),
/// replay zerocheck + the union-column lincheck over the union address
/// space with the claim points derived from the
/// [`crate::union::UnionInstance`], then verify the jagged-path batched
/// opening against the union's heights. The counts bind in the transcript
/// (before any challenge) and additionally enter through the heights and
/// the lincheck's const-pin target terms.
///
/// Since wire v6 the shipped Mixed protocol uses the MERGED transport
/// ([`verify_ligerito_jagged_union_merged`]); this jagged-transport entry
/// remains as the differential/regression oracle's verifier — not a wire
/// mode.
///
/// `circuits` are the per-type lincheck circuits, one per registry type,
/// **in slot order** (the registry's order — capacity area descending).
pub fn verify_ligerito_jagged_union<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    let (claim, _) = verify_union_with_binding(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof,
        pcs_params,
        false,
        challenger,
    )?;
    Ok(claim)
}

/// [`verify_ligerito_jagged_union`] with the matrix work left undischarged —
/// the "succinct verify" of the accumulation route. Reads no base matrix, so
/// it is what a recursion circuit replays and what
/// [`crate::aggregate`] batches. **Conditional on the returned assertion**:
/// callers not accumulating must use [`verify_ligerito_jagged_union`].
pub fn verify_ligerito_jagged_union_deferred<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(R1csClaim, lincheck::MatrixAssertion), VerifyError> {
    let (claim, matrix) = verify_union_with_binding(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof,
        pcs_params,
        true,
        challenger,
    )?;
    Ok((claim, matrix.expect("deferred")))
}

/// [`verify_ligerito_jagged_union`] (the protocol `flock-mixed-v1` binding)
/// with per-phase timers — the union counterpart of [`verify_ligerito_timed`].
/// Splits the verify into zerocheck-verify / lincheck-verify (with the
/// per-type comb construction, i.e. the multi-slot circuit replay, timed
/// separately as `lincheck_comb_s`) / opening-verify. Kept in lockstep with
/// `verify_ligerito_jagged_union`; benchmark-only, production path undisturbed.
pub fn verify_ligerito_jagged_union_timed<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(R1csClaim, VerifyPhaseTimings), VerifyError> {
    use std::time::Instant;
    assert_eq!(
        pcs_params.m,
        union.dense_m(),
        "PcsParams.m must equal the union's dense_m (committed stack size)"
    );
    // PIOP replay (bind + zerocheck + union lincheck) on the 1-thread pool.
    let (ab, c, zerocheck_s, lincheck_s, lincheck_comb_s) =
        verifier_pool().install(|| -> Result<(ZClaim, ZClaim, f64, f64, f64), VerifyError> {
            union.bind_statement(challenger, commitment);
            let t0 = Instant::now();
            let zc_claim = zerocheck::verify(union.m_bool(), &proof.zerocheck, challenger)
                .map_err(VerifyError::Zerocheck)?;
            let zerocheck_s = t0.elapsed().as_secs_f64();
            let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
            let t0 = Instant::now();
            let (lc_claim, comb_s) = lincheck::verify_union_timed(
                union,
                circuits,
                &x_ab,
                zc_claim.a_eval,
                zc_claim.b_eval,
                &proof.lincheck,
                challenger,
            )
            .map_err(VerifyError::Lincheck)?;
            let lincheck_s = t0.elapsed().as_secs_f64();
            let ab = ZClaim {
                point: union.ab_claim_point(
                    lc_claim.r_inner_skip,
                    &lc_claim.r_inner_rest,
                    &x_ab.x_outer,
                ),
                value: lc_claim.w,
            };
            let c = ZClaim {
                point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
                value: zc_claim.c_eval,
            };
            Ok((ab, c, zerocheck_s, lincheck_s, comb_s))
        })?;

    let t0 = Instant::now();
    verify_claims_jagged_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &[],
        &union.jagged_heights(),
        union.n_log(),
        union.m_total(),
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;
    let open_s = t0.elapsed().as_secs_f64();

    let t = VerifyPhaseTimings {
        zerocheck_s,
        lincheck_s,
        lincheck_comb_s,
        open_s,
    };
    Ok((R1csClaim { ab, c }, t))
}

/// [`verify_ligerito_jagged_union`] under the M1/M2 **harness** binding
/// (the slot's single-table `BlockR1cs` statement digest) — the mirror of
/// `flock_prover::prover::prove_fast_ligerito_jagged_union_harness`.
/// Single-type registries only; on those, acceptance is equivalent to
/// [`verify_ligerito_jagged`] with the slot's `BlockR1cs` at full
/// utilization — the transcript walk is byte-identical.
/// Test/differential harness only — not a protocol mode.
pub fn verify_ligerito_jagged_union_harness<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    slot_r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    let (claim, _) = verify_union_with_binding(
        union,
        UnionVerifyBinding::SingleTypeHarness(slot_r1cs),
        &[lincheck_circuit],
        commitment,
        proof,
        pcs_params,
        false,
        challenger,
    )?;
    Ok(claim)
}

/// The MERGED-transport union verifier (wire v6) — the Mixed protocol's
/// verify entry, kept in lockstep with [`verify_union_with_binding`]
/// (identical binding + PIOP replay; only the PCS verification differs:
/// `pcs::verify_batch_merged`). Mixed binding only; handles both
/// lane-major and power-of-two commitments (dispatched on
/// `commitment.params.num_lanes`, which the params-equality check below
/// pins to the count-derived value).
pub fn verify_ligerito_jagged_union_merged<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMergedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    // Mirror of the prove-side guard: the merged transport has no
    // packed-direct intake, so it cannot carry element claims yet.
    assert!(
        !union.has_element(),
        "the merged transport does not carry element claims yet — \
         use verify_ligerito_jagged_union"
    );
    assert_eq!(
        pcs_params.m,
        union.dense_m(),
        "PcsParams.m must equal the union's dense_m (committed stack size)"
    );
    // Same params-equality rejection as the jagged path (the transcript
    // binds only the root; the opening reads the leaf width / lane count
    // from the commitment's params, so they must equal the count-derived
    // ones).
    if commitment.params.m != pcs_params.m
        || commitment.params.log_batch_size != pcs_params.log_batch_size
        || commitment.params.log_inv_rate != pcs_params.log_inv_rate
        || commitment.params.num_ntts() != pcs_params.num_ntts()
    {
        return Err(VerifyError::PcsJagged(
            crate::pcs::VerifyErrorJagged::Ligerito,
        ));
    }
    let defer_merged = false;
    // `VERIFY_TRACE` phase breakdown, as on the direct path
    // (`verify_core_inner`). Off by default.
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };
    type MergedPiop = (ZClaim, ZClaim, lincheck::MatrixAssertion);
    let (ab, c, matrix) = verifier_pool().install(|| -> Result<MergedPiop, VerifyError> {
        let t = std::time::Instant::now();
        union.bind_statement(challenger, commitment);
        if trace {
            eprintln!(
                "      [vum] bind_statement: {}",
                fmt(t.elapsed().as_secs_f64())
            );
        }
        let t = std::time::Instant::now();
        let zc_claim = zerocheck::verify(union.m_total(), &proof.zerocheck, challenger)
            .map_err(VerifyError::Zerocheck)?;
        if trace {
            eprintln!(
                "      [vum] zerocheck::verify (m_total={}): {}",
                union.m_total(),
                fmt(t.elapsed().as_secs_f64())
            );
        }
        let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
        // DEFERRED — discharged by the wrapper below (or accumulated).
        let t = std::time::Instant::now();
        let (lc_claim, matrix) = lincheck::verify_union_deferred(
            union,
            circuits,
            &x_ab,
            zc_claim.a_eval,
            zc_claim.b_eval,
            &proof.lincheck,
            challenger,
        )
        .map_err(VerifyError::Lincheck)?;
        if trace {
            eprintln!(
                "      [vum] lincheck::verify_union: {}",
                fmt(t.elapsed().as_secs_f64())
            );
        }
        let ab = ZClaim {
            point: union.ab_claim_point(
                lc_claim.r_inner_skip,
                &lc_claim.r_inner_rest,
                &x_ab.x_outer,
            ),
            value: lc_claim.w,
        };
        let c = ZClaim {
            point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
            value: zc_claim.c_eval,
        };
        if !defer_merged {
            matrix
                .check(union, circuits)
                .map_err(VerifyError::Lincheck)?;
        }
        Ok((ab, c, matrix))
    })?;
    let heights = union.jagged_heights();
    let claims = [ab.clone(), c.clone()];
    let t_open = std::time::Instant::now();
    let open_result = verifier_pool().install(|| {
        let z_skips: Vec<F128> = claims.iter().map(|cl| cl.point.z_skip).collect();
        let values: Vec<F128> = claims.iter().map(|cl| cl.value).collect();
        let x_fulls: Vec<Vec<F128>> = claims
            .iter()
            .map(|cl| {
                let mut v = cl.point.x_inner_rest.clone();
                v.extend_from_slice(&cl.point.x_outer);
                v
            })
            .collect();
        let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
        let log_n = pcs_params.m - pcs::LOG_PACKING;
        let lig_v_config = crate::pcs::ligerito::verifier_config_for(
            log_n,
            pcs_params.log_batch_size,
            pcs_params.profile,
        )
        .expect("Ligerito default verifier config");
        pcs::verify_batch_merged(
            commitment,
            &values,
            &z_skips,
            &x_refs,
            &[],
            &heights,
            union.n_log(),
            &proof.pcs_open,
            &lig_v_config,
            challenger,
        )
    });
    if trace {
        eprintln!(
            "      [vum] pcs::verify_batch_merged: {}",
            fmt(t_open.elapsed().as_secs_f64())
        );
    }
    open_result.map_err(VerifyError::PcsJagged)?;
    let _ = matrix;
    Ok(R1csClaim { ab, c })
}

/// [`verify_ligerito_jagged_union`] for a **mixed-class** proof — the mirror of
/// `flock_prover::prover::prove_fast_ligerito_jagged_union_mixed_class`.
///
/// Replays each class's PIOP over its own region in the prover's Fiat–Shamir
/// order (boolean zerocheck + lincheck, then the element region's zerocheck +
/// lincheck), then verifies the single jagged opening with the boolean AB/C
/// claims ring-switched and the element C/LC claims packed-direct.
///
/// A sub-proof must be present exactly when the registry has a type of that
/// class; a mismatch is a rejection, not a panic. `circuits` are the
/// per-BOOLEAN-type lincheck circuits, in slot order.
pub fn verify_ligerito_jagged_union_mixed_class<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMixedClassLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<crate::proof::UnionClassClaims, VerifyError> {
    let (claims, matrix) = mixed_class_inner(
        union, circuits, commitment, proof, pcs_params, false, challenger,
    )?;
    debug_assert!(matrix.is_none(), "non-deferred: discharged internally");
    Ok(claims)
}

/// [`verify_ligerito_jagged_union_mixed_class`] with the matrix work left
/// undischarged: everything else is verified, and the boolean lincheck's
/// [`lincheck::MatrixAssertion`] comes back for the caller to discharge or
/// accumulate ([`crate::aggregate`]).
///
/// This is the "succinct verify" of the accumulation route — no base matrix
/// is read anywhere in it, which is what lets a recursion circuit replay it.
/// **The returned claims are conditional on the assertion**: a proof whose
/// lincheck is simply wrong still returns `Ok` here, so a caller that is not
/// accumulating must use [`verify_ligerito_jagged_union_mixed_class`].
pub fn verify_ligerito_jagged_union_mixed_class_deferred<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMixedClassLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<
    (
        crate::proof::UnionClassClaims,
        Option<lincheck::MatrixAssertion>,
    ),
    VerifyError,
> {
    mixed_class_inner(
        union, circuits, commitment, proof, pcs_params, true, challenger,
    )
}

fn mixed_class_inner<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofMixedClassLigerito,
    pcs_params: &crate::pcs::PcsParams,
    defer: bool,
    challenger: &mut Ch,
) -> Result<
    (
        crate::proof::UnionClassClaims,
        Option<lincheck::MatrixAssertion>,
    ),
    VerifyError,
> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    // Per-phase field-op attribution, `--features mul-count` + `MUL_TRACE=1`.
    // The sibling of `VERIFY_TRACE`, for arithmetic rather than wall clock —
    // and the two rank phases differently, which is the point.
    #[cfg(feature = "mul-count")]
    let mul_trace = std::env::var("MUL_TRACE").is_ok();
    #[cfg(feature = "mul-count")]
    let at_start = crate::field::gf2_128::op_count::snapshot();

    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        None,
        pcs_params,
        challenger,
    )?;

    #[cfg(feature = "mul-count")]
    let after_piops = crate::field::gf2_128::op_count::snapshot();

    let matrix = if defer {
        matrix
    } else {
        if let Some(a) = matrix {
            a.check(union, circuits).map_err(VerifyError::Lincheck)?;
        }
        if let Some(a) = &el_matrix {
            a.check_reported(union).map_err(VerifyError::Element)?;
        }
        None
    };
    let el_matrix = if defer { el_matrix } else { None };
    let _ = &el_matrix;
    let z_claims: Vec<ZClaim> = claims
        .boolean
        .as_ref()
        .map(|c| vec![c.ab.clone(), c.c.clone()])
        .unwrap_or_default();
    let refs: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    verify_claims_jagged_ligerito(
        commitment,
        &z_claims,
        &refs,
        &union.jagged_heights(),
        union.n_log(),
        union.m_total(),
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;

    #[cfg(feature = "mul-count")]
    if mul_trace {
        let end = crate::field::gf2_128::op_count::snapshot();
        let cc = |a: &crate::field::gf2_128::op_count::Snapshot,
                  b: &crate::field::gf2_128::op_count::Snapshot| {
            let muls = (b.native_muls - a.native_muls)
                .saturating_sub((b.invs - a.invs) * crate::field::gf2_128::op_count::MULS_PER_INV);
            let invs = b.invs - a.invs;
            (muls, invs, muls + invs)
        };
        let (pm, pi, pc) = cc(&at_start, &after_piops);
        let (om, oi, oc) = cc(&after_piops, &end);
        let total = pc + oc;
        println!(
            "  [mul] PIOP replay (zerocheck+lincheck, both classes): \
             {pm:>8} muls {pi:>5} invs = {pc:>8} constraints ({:>5.1}%)",
            100.0 * pc as f64 / total as f64
        );
        println!(
            "  [mul] opening (jagged assist + Ligerito):             \
             {om:>8} muls {oi:>5} invs = {oc:>8} constraints ({:>5.1}%)",
            100.0 * oc as f64 / total as f64
        );
    }

    Ok((claims, matrix))
}

/// The **circuit** verify entry over the MERGED transport — the production
/// shape, and the mirror of
/// `flock_prover::prover::prove_fast_ligerito_union_circuit_merged`.
///
/// Same replay as the jagged variant: both class PIOPs, then the wiring
/// argument over the circuit's cell space (σ-aware GKR plus the
/// recombination and `f_eval == g_eval` bindings). Only the opening differs
/// — the wiring's gather claims are packed-direct, which the merged
/// transport carries the same way it carries the element class's.
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit_merged<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuit: &crate::circuit::Circuit,
    public: &[F128],
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofCircuitMerged,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<crate::proof::UnionClassClaims, VerifyError> {
    if !circuit.check_instance(union) || public.len() != circuit.num_public() {
        return Err(VerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        Some(&proof.wiring),
        pcs_params,
        challenger,
    )?;
    if let Some(a) = matrix {
        a.check(union, circuits).map_err(VerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(VerifyError::Element)?;
    }
    verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
}

/// [`verify_ligerito_union_circuit_merged`] with the matrix work left
/// undischarged — what a merge node runs on each child proof.
///
/// Everything else is verified: both class PIOPs, the wiring argument, and
/// the single merged opening. What comes back alongside the claims is the two
/// classes' [`DeferredMatrixWork`], for the caller to fold into an
/// accumulator ([`crate::aggregate`]) rather than evaluate.
///
/// No base matrix is read anywhere in it — that is what lets a recursion
/// circuit replay it. There is deliberately NO jagged counterpart: the merged
/// transport is the production path, and building deferred machinery on the
/// legacy one would be work aimed at something being retired.
///
/// **The claims are conditional on the returned work**: a proof whose
/// lincheck is simply wrong still returns `Ok` here. Callers that are not
/// accumulating must use [`verify_ligerito_union_circuit_merged`].
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit_merged_deferred<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuit: &crate::circuit::Circuit,
    public: &[F128],
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofCircuitMerged,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(crate::proof::UnionClassClaims, DeferredMatrixWork), VerifyError> {
    if !circuit.check_instance(union) || public.len() != circuit.num_public() {
        return Err(VerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        Some(&proof.wiring),
        pcs_params,
        challenger,
    )?;
    let claims = verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
    )?;
    Ok((
        claims,
        DeferredMatrixWork {
            boolean: matrix,
            element: el_matrix,
        },
    ))
}

/// The merged transport's verification, shared by the mixed-class and circuit
/// entries: the boolean pair ring-switched, everything else packed-direct.
fn verify_merged_opening<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    commitment: &Commitment,
    claims: &crate::proof::UnionClassClaims,
    packed_direct_points: &[(Vec<F128>, F128)],
    pcs_open: &crate::pcs::MergedOpenProof,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<crate::proof::UnionClassClaims, VerifyError> {
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
                pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsJagged)?;
    Ok(claims.clone())
}

/// [`verify_ligerito_jagged_union_mixed_class`] over the MERGED transport.
///
/// Same statement, same PIOP replay; only the opening differs. The element
/// class's two claims ride as packed-direct claims, which the merged
/// transport carries by expressing each weight as the `F₂`-linear map
/// `x ↦ γ·x` — indistinguishable, to its per-claim weight builder, from a
/// ring-switched claim's Φ-fold.
pub fn verify_ligerito_jagged_union_mixed_class_merged<Ch: Challenger>(
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
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        None,
        pcs_params,
        challenger,
    )?;
    // Both classes' matrix work comes back undischarged from the PIOP
    // replay; this is a non-deferred entry, so discharge here — after the
    // replay and BEFORE the opening, as the sibling entries do, so an
    // inconsistent lincheck is rejected as Lincheck and before the expensive
    // PCS work.
    if let Some(a) = matrix {
        a.check(union, circuits).map_err(VerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(VerifyError::Element)?;
    }

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
        .map_err(VerifyError::PcsJagged)?;
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
    binding: UnionVerifyBinding<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    boolean: Option<&crate::proof::BooleanPiopProof>,
    element: Option<&crate::element_r1cs::union::Proof>,
    wiring: Option<&crate::circuit::WiringProof>,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<UnionPiopOut, VerifyError> {
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
        return Err(VerifyError::PcsJagged(
            crate::pcs::VerifyErrorJagged::Ligerito,
        ));
    }
    // Verification is single-threaded; run the PIOP replay on the dedicated
    // 1-thread pool (verify_claims_jagged_ligerito installs it itself).
    verifier_pool().install(|| -> Result<UnionPiopOut, VerifyError> {
        match binding {
            UnionVerifyBinding::Mixed => union.bind_statement(challenger, commitment),
            UnionVerifyBinding::Circuit { circuit, public } => {
                union.bind_statement_circuit(challenger, commitment, &circuit.digest(), public)
            }
            UnionVerifyBinding::SingleTypeHarness(slot_r1cs) => {
                union.expect_single_type_slot(slot_r1cs);
                union.bind_statement_single_type(challenger, slot_r1cs, commitment);
            }
        }

        let mut matrix: Option<lincheck::MatrixAssertion> = None;
        let mut el_matrix: Option<crate::element_r1cs::union::ElementAssertion> = None;
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
                // DEFERRED: the matrix work leaves as an assertion instead of
                // being discharged here. Callers that are not accumulating get
                // it discharged for them by the wrappers below.
                let (lc_claim, assertion) = lincheck::verify_union_deferred(
                    union,
                    circuits,
                    &x_ab,
                    zc_claim.a_eval,
                    zc_claim.b_eval,
                    &piop.lincheck,
                    challenger,
                )
                .map_err(VerifyError::Lincheck)?;
                matrix = Some(assertion);
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

        // DEFERRED on this side too: the element class's matrix work leaves
        // as its own assertion rather than being evaluated here, so a
        // `*_deferred` entry really does defer BOTH classes.
        let el_claim = match element {
            Some(p) => {
                let (c, a) = crate::element_r1cs::union::verify_deferred(union, p, challenger)
                    .map_err(VerifyError::Element)?;
                el_matrix = Some(a);
                Some(c)
            }
            None => None,
        };
        let mut packed_direct = el_claim
            .as_ref()
            .map(|c: &crate::element_r1cs::union::Claims| {
                vec![
                    (c.c_point.clone(), c.c_value),
                    (c.lc_point.clone(), c.lc_value),
                ]
            })
            .unwrap_or_default();

        // The wiring argument replays AFTER both classes' PIOPs, at the
        // prover's transcript position; its gather claims join the same
        // packed-direct intake the element claims ride.
        if let UnionVerifyBinding::Circuit { circuit, public } = binding {
            let proof = wiring.ok_or(VerifyError::CircuitMismatch)?;
            #[cfg(feature = "mul-count")]
            let wiring_start = crate::field::gf2_128::op_count::snapshot();
            let gather = crate::circuit::verify_wiring(circuit, public, proof, challenger)
                .map_err(VerifyError::Wiring)?;
            #[cfg(feature = "mul-count")]
            if std::env::var("MUL_TRACE").is_ok() {
                let e = crate::field::gf2_128::op_count::snapshot();
                let invs = e.invs - wiring_start.invs;
                let muls = (e.native_muls - wiring_start.native_muls)
                    .saturating_sub(invs * crate::field::gf2_128::op_count::MULS_PER_INV);
                println!(
                    "  [mul] wiring GKR (grand product + sigma):             \
                     {muls:>8} muls {invs:>5} invs = {:>8} constraints",
                    muls + invs
                );
            }
            packed_direct.extend(gather);
        }

        Ok((
            crate::proof::UnionClassClaims {
                boolean: bool_claim,
                element: el_claim,
            },
            packed_direct,
            matrix,
            el_matrix,
        ))
    })
}

/// What the union PIOP replay yields: the per-class claims, the element
/// class's packed-direct claims for the opening, and — when a boolean PIOP
/// ran — the [`lincheck::MatrixAssertion`] carrying its undischarged matrix
/// work.
type UnionPiopOut = (
    crate::proof::UnionClassClaims,
    Vec<(Vec<F128>, F128)>,
    Option<lincheck::MatrixAssertion>,
    Option<crate::element_r1cs::union::ElementAssertion>,
);

/// Both classes' undischarged matrix work, as a `*_deferred` entry returns
/// it. Either half is `None` when that class has no types in the registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredMatrixWork {
    pub boolean: Option<lincheck::MatrixAssertion>,
    pub element: Option<crate::element_r1cs::union::ElementAssertion>,
}

/// The **circuit** verify entry — the mirror of
/// `flock_prover::prover::prove_fast_ligerito_jagged_union_circuit`.
///
/// Replays both class PIOPs, then the wiring argument over the circuit's cell
/// space (σ-aware GKR + the recombination and `f_eval == g_eval` bindings),
/// then verifies the single jagged opening carrying the class claims AND the
/// wiring's gather claims.
///
/// The circuit is verifier-known (v1): it holds the full description, rebuilds
/// σ and evaluates `ŝ_σ(ρ)` itself in `O(2^μ)`. `public` are the statement's
/// public words, bound into the transcript alongside the circuit digest.
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_jagged_union_circuit<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    circuit: &crate::circuit::Circuit,
    public: &[F128],
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &crate::proof::R1csProofCircuitLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<crate::proof::UnionClassClaims, VerifyError> {
    // The circuit's gate counts ARE the union's declared counts, and its
    // registry is the union's — the same statement, or no statement at all.
    if !circuit.check_instance(union) || public.len() != circuit.num_public() {
        return Err(VerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        Some(&proof.wiring),
        pcs_params,
        challenger,
    )?;
    // Both classes' matrix work is deferred by the PIOP replay; this entry
    // discharges it, here rather than after the opening so an inconsistent
    // lincheck is rejected as such and before the expensive PCS work. A
    // circuit proof that wants to ACCUMULATE it instead goes through
    // `verify_ligerito_jagged_union_circuit_deferred`.
    if let Some(a) = matrix {
        a.check(union, circuits).map_err(VerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(VerifyError::Element)?;
    }
    let z_claims: Vec<ZClaim> = claims
        .boolean
        .as_ref()
        .map(|c| vec![c.ab.clone(), c.c.clone()])
        .unwrap_or_default();
    let refs: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    verify_claims_jagged_ligerito(
        commitment,
        &z_claims,
        &refs,
        &union.jagged_heights(),
        union.n_log(),
        union.m_total(),
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;
    Ok(claims)
}

/// Shared body of the jagged-transport union verify entries; `binding`
/// selects the statement binding, everything else is identical.
fn verify_union_with_binding<Ch: Challenger>(
    union: &crate::union::UnionInstance<'_>,
    binding: UnionVerifyBinding<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofJaggedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    defer: bool,
    challenger: &mut Ch,
) -> Result<(R1csClaim, Option<lincheck::MatrixAssertion>), VerifyError> {
    assert!(
        !union.has_element(),
        "this entry consumes R1csProofJaggedLigerito (boolean classes only); \
         element registries go through verify_ligerito_jagged_union_mixed_class"
    );
    let piop = crate::proof::BooleanPiopProof {
        zerocheck: proof.zerocheck.clone(),
        lincheck: proof.lincheck.clone(),
    };
    let (claims, packed_direct, matrix, _el) = verify_union_piops(
        union,
        binding,
        circuits,
        commitment,
        Some(&piop),
        None,
        None,
        pcs_params,
        challenger,
    )?;
    debug_assert!(packed_direct.is_empty());
    let claim = claims.boolean.expect("boolean sub-proof was supplied");
    let matrix = matrix.expect("a boolean PIOP ran, so it left an assertion");
    // Discharge here, not after the opening: a wrong-count or otherwise
    // inconsistent lincheck must be rejected as Lincheck, and before the
    // expensive PCS work.
    let matrix = if defer {
        Some(matrix)
    } else {
        matrix
            .check(union, circuits)
            .map_err(VerifyError::Lincheck)?;
        None
    };
    verify_claims_jagged_ligerito(
        commitment,
        &[claim.ab.clone(), claim.c.clone()],
        &[],
        &union.jagged_heights(),
        union.n_log(),
        union.m_total(),
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;
    Ok((claim, matrix))
}

/// Verify a jagged-path batched PCS opening over an arbitrary list of
/// `ẑ`-claims — the jagged counterpart of [`verify_claims_ligerito`], and the
/// mirror of the prover's `pcs::open_batch_jagged_ligerito` call. `heights` /
/// `n_log` describe the committed jagged grid (see
/// [`BlockR1cs::jagged_heights`]; the union heights — and hence the dense
/// size — are count-dependent under height-`n_t` stacking); `virtual_m` is
/// the bit-variable count of the VIRTUAL (padded) polynomial the PIOP ran
/// over (`= pcs_params.m` on the single-table paths;
/// `= UnionInstance::m_total` under the dense-stack commit, where
/// `pcs_params.m` is the smaller dense size). Both sides derive all three
/// from the statement, never from the proof. Must run at the same
/// transcript position as the prover's open.
pub fn verify_claims_jagged_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    packed_direct: &[pcs::PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    virtual_m: usize,
    pcs_open: &pcs::BatchOpeningProofJaggedLigerito,
    pcs_params: &crate::pcs::PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyErrorJagged> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
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
        let log_n = pcs_params.m - pcs::LOG_PACKING;
        let lig_v_config = crate::pcs::ligerito::verifier_config_for(
            log_n,
            pcs_params.log_batch_size,
            pcs_params.profile,
        )
        .expect("Ligerito default verifier config");
        pcs::verify_opening_batch_jagged_ligerito(
            commitment,
            &values,
            &z_skips,
            &x_refs,
            packed_direct,
            heights,
            n_log,
            virtual_m - pcs::LOG_PACKING,
            pcs_open,
            &lig_v_config,
            challenger,
        )
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
