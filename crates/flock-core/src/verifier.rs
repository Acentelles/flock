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
    verify_union_with_binding(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof,
        pcs_params,
        challenger,
    )
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
            let zc_claim = zerocheck::verify(union.m_total(), &proof.zerocheck, challenger)
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
    verify_union_with_binding(
        union,
        UnionVerifyBinding::SingleTypeHarness(slot_r1cs),
        &[lincheck_circuit],
        commitment,
        proof,
        pcs_params,
        challenger,
    )
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
    let (ab, c) = verifier_pool().install(|| -> Result<(ZClaim, ZClaim), VerifyError> {
        union.bind_statement(challenger, commitment);
        let zc_claim = zerocheck::verify(union.m_total(), &proof.zerocheck, challenger)
            .map_err(VerifyError::Zerocheck)?;
        let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
        let lc_claim = lincheck::verify_union(
            union,
            circuits,
            &x_ab,
            zc_claim.a_eval,
            zc_claim.b_eval,
            &proof.lincheck,
            challenger,
        )
        .map_err(VerifyError::Lincheck)?;
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
        Ok((ab, c))
    })?;
    let heights = union.jagged_heights();
    let claims = [ab.clone(), c.clone()];
    verifier_pool()
        .install(|| {
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
                &heights,
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsJagged)?;
    Ok(R1csClaim { ab, c })
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
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    // The commitment is to the DENSE stack q (M4/M5): PcsParams.m is the
    // dense variable count — count-dependent under height-n_t stacking,
    // derived from the declared counts — while the PIOP and the
    // virtual-opening sumcheck run over the M-variable padded address
    // space.
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
    let (ab, c) = verifier_pool().install(|| -> Result<(ZClaim, ZClaim), VerifyError> {
        match binding {
            UnionVerifyBinding::Mixed => union.bind_statement(challenger, commitment),
            UnionVerifyBinding::SingleTypeHarness(slot_r1cs) => {
                union.expect_single_type_slot(slot_r1cs);
                union.bind_statement_single_type(challenger, slot_r1cs, commitment);
            }
        }

        let zc_claim = zerocheck::verify(union.m_total(), &proof.zerocheck, challenger)
            .map_err(VerifyError::Zerocheck)?;
        let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
        // The union-column lincheck (one circuit per slot, in slot order);
        // the declared counts additionally bind through the per-type
        // const-pin target terms.
        let lc_claim = lincheck::verify_union(
            union,
            circuits,
            &x_ab,
            zc_claim.a_eval,
            zc_claim.b_eval,
            &proof.lincheck,
            challenger,
        )
        .map_err(VerifyError::Lincheck)?;

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
        Ok((ab, c))
    })?;
    verify_claims_jagged_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &union.jagged_heights(),
        union.n_log(),
        union.m_total(),
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsJagged)?;
    Ok(R1csClaim { ab, c })
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
            &[],
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
    let log_n = pcs_params.m - pcs::LOG_PACKING;
    let lig_v_config = crate::pcs::ligerito::verifier_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
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
