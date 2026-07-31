//! Aggregating many proofs' matrix work into one accumulator.
//!
//! This is the driver that composes the two halves of the accumulation
//! route: [`lincheck::verify_union_deferred`] replays a proof succinctly and
//! hands back a [`MatrixAssertion`] instead of reading the base matrices,
//! and [`crate::matrix_fold`] folds those assertions' claims into a running
//! [`Accumulator`]. Nobody reads a matrix until somebody discharges the
//! accumulator — once, at the end.
//!
//! **This does not make native verification faster** — measured, batching is
//! several times SLOWER. Folding `k` claims costs `k · nnz` (the row phase
//! builds `g_i` per claim), which is exactly what checking those `k` claims
//! directly costs, plus the final discharge on top. No random-linear-
//! combination avoids it: the combined weight is a sum of `k` rank-1 terms,
//! so every nonzero still needs `k` multiplications. `k` claims at `k`
//! distinct points cost `k` passes over the matrix, full stop.
//!
//! What the fold buys is an ASYMMETRY, and it is worth only one thing:
//! recursion. The fold's PROVER pays that `k · nnz`; the fold's VERIFIER
//! pays `O(κ)` and reads no matrix at all. That moves the matrix work from
//! inside a circuit — where it is nnz-preserving and the fixed point cannot
//! close — to a native prover, where it is ordinary. So this module is the
//! thing a recursion circuit arithmetises.
//! [`verify_aggregate`] touches no matrix, so it is exactly what a merge
//! circuit replays: verify the children succinctly, fold their claims plus
//! the accumulators they carried, output one accumulator. The proof that
//! comes out has the same shape as the ones that went in — a proof plus an
//! accumulator — which is what lets the recursion close.
//!
//! ## What folds with what
//!
//! One accumulator per `(boolean type, matrix)`: `A₀` and `B₀` never mix,
//! because only their α-combination appears in a proof's target and α is
//! per-proof, so a claim about `α·A₀ + B₀` names a different polynomial in
//! every proof. Within one accumulator the fold takes
//!
//! * the claim each verified proof emitted (one per proof), plus
//! * the one claim carried in by `prior`, if this is not a leaf.
//!
//! So a leaf over two proofs folds `2 → 1`, and a `2 → 1` merge of two
//! recursive proofs folds `4 → 1` (two inherited, two fresh).

use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{MatrixAssertion, VerifyError};
use crate::matrix_fold::{self, FoldProof, MatrixClaim};
use crate::r1cs::SparseBinaryMatrix;
use crate::schedule::Registry;

const DOMAIN: &[u8] = b"flock-aggregate-v0";

/// The base matrices of one boolean type, `(A₀, B₀)`.
pub type TypeMatrices<'a> = (&'a SparseBinaryMatrix, &'a SparseBinaryMatrix);

/// Accumulated matrix claims: one `(A₀, B₀)` pair per boolean type, in slot
/// order, tied to the registry they are about.
///
/// The digest is load-bearing rather than decorative: claims fold only if
/// they name the same matrices, so an accumulator from a different registry
/// must be rejected, not silently folded into one it does not belong to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accumulator {
    pub registry_digest: [u8; 32],
    pub per_type: Vec<(MatrixClaim, MatrixClaim)>,
}

impl Accumulator {
    /// Discharge through each type's tuned column-marginal kernel.
    ///
    /// A claim's bilinear form is `Σ_c col(c)·comb(c)` with
    /// `comb = Σ_r row(r)·M[r,·]` — a column marginal, which is what
    /// `fold_split` computes with the type's own walker/CSC path.
    ///
    /// **Measured SLOWER than [`Self::discharge`]** (17.3 ms vs 15.3 ms on
    /// the N=4 BLAKE3 batch), and the reason is worth keeping: `fold_split`
    /// returns both matrices' marginals, but the accumulated A and B claims
    /// fold under separate transcripts and so carry *different* row points —
    /// each call therefore throws half its work away. The fold's own k·nnz
    /// pass does not have this problem, because there A and B share a row
    /// weight and one call serves both. Kept for callers without the raw
    /// matrices; prefer `discharge`.
    pub fn discharge_with_circuits(
        &self,
        circuits: &[&dyn crate::lincheck::LincheckCircuit],
    ) -> bool {
        if self.per_type.len() != circuits.len() {
            return false;
        }
        let dot = |a: &[F128], b: &[F128]| {
            a.iter()
                .zip(b)
                .fold(F128::ZERO, |acc, (x, y)| acc + *x * *y)
        };
        self.per_type.iter().zip(circuits).all(|((ca, cb), circ)| {
            // A and B accumulate under separate folds, so their row points
            // differ; each needs its own marginal.
            let (xa, _) = circ.fold_split(&ca.row.materialize());
            let (_, xb) = circ.fold_split(&cb.row.materialize());
            dot(&ca.col.materialize(), &xa) == ca.value
                && dot(&cb.col.materialize(), &xb) == cb.value
        })
    }

    /// Discharge every accumulated claim against the raw matrices — the
    /// generic `O(Σ_t nnz_t)` root check, for callers without circuits.
    pub fn discharge(&self, mats: &[TypeMatrices<'_>]) -> bool {
        self.per_type.len() == mats.len()
            && self
                .per_type
                .iter()
                .zip(mats)
                .all(|((ca, cb), (a, b))| ca.check_direct(a) && cb.check_direct(b))
    }
}

/// Per boolean type, the two folds `(A₀, B₀)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateProof {
    pub folds: Vec<(FoldProof, FoldProof)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregateError {
    /// Nothing to fold.
    Empty,
    /// A prior accumulator names a different registry, or has the wrong
    /// number of per-type claims.
    RegistryMismatch,
    /// The proof is not shaped for this registry.
    Malformed,
    /// An assertion's reported matrix evaluations do not reproduce its
    /// target (`MatrixAssertion::check_reported`).
    Reported(VerifyError),
    /// A fold did not verify.
    Fold(matrix_fold::FoldError),
    /// The accumulated claims did not hold against the real matrices.
    Discharge,
}

/// Claims to fold for one type, in a fixed order: the prior accumulator's
/// first (if any), then one per assertion. Prover and verifier build this
/// the same way, so the fold transcripts line up.
fn gather(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    prior: Option<&Accumulator>,
    t: usize,
) -> (Vec<MatrixClaim>, Vec<MatrixClaim>) {
    let mut a = Vec::with_capacity(assertions.len() + 1);
    let mut b = Vec::with_capacity(assertions.len() + 1);
    if let Some(p) = prior {
        a.push(p.per_type[t].0.clone());
        b.push(p.per_type[t].1.clone());
    }
    for assertion in assertions {
        let (ca, cb) = assertion.claims(registry).swap_remove(t);
        a.push(ca);
        b.push(cb);
    }
    (a, b)
}

fn bind<Ch: Challenger>(registry: &Registry, prior: Option<&Accumulator>, ch: &mut Ch) {
    ch.observe_label(DOMAIN);
    ch.observe_bytes(&registry.digest());
    ch.observe_bytes(&[u8::from(prior.is_some())]);
}

/// Fold `assertions` (and `prior`, if this is not a leaf) into one
/// accumulator. `O(k · Σ_t nnz_t)` — the matrices are read here, natively,
/// so that no circuit ever has to.
pub fn prove_aggregate<Ch: Challenger>(
    registry: &Registry,
    mats: &[TypeMatrices<'_>],
    circuits: &[&dyn crate::lincheck::LincheckCircuit],
    assertions: &[MatrixAssertion],
    prior: Option<&Accumulator>,
    ch: &mut Ch,
) -> Result<(AggregateProof, Accumulator), AggregateError> {
    if assertions.is_empty() && prior.is_none() {
        return Err(AggregateError::Empty);
    }
    if mats.len() != registry.num_boolean() {
        return Err(AggregateError::Malformed);
    }
    if let Some(p) = prior {
        if p.registry_digest != registry.digest() || p.per_type.len() != registry.num_boolean() {
            return Err(AggregateError::RegistryMismatch);
        }
    }

    bind(registry, prior, ch);
    let mut folds = Vec::with_capacity(registry.num_boolean());
    let mut per_type = Vec::with_capacity(registry.num_boolean());
    for (t, (ma, mb)) in mats.iter().enumerate() {
        let (ca, cb) = gather(registry, assertions, prior, t);
        // The k·nnz work. ONE `fold_split` per claim yields the column
        // marginals for BOTH matrices, so the A- and B-folds share it — and
        // it runs on the type's tuned kernel rather than a generic sparse
        // walk. Claims share their row weight (A and B are reported at the
        // same point), so `ca` and `cb` agree here row-wise.
        let n_cols = 1usize << registry.boolean_types()[t].k_log;
        let mut combs_a = Vec::with_capacity(ca.len());
        let mut combs_b = Vec::with_capacity(cb.len());
        for (qa, qb) in ca.iter().zip(&cb) {
            let (xa, xb) = if qa.row == qb.row {
                circuits[t].fold_split(&qa.row.materialize())
            } else {
                (
                    matrix_fold::col_marginal(ma, &qa.row.materialize(), n_cols),
                    matrix_fold::col_marginal(mb, &qb.row.materialize(), n_cols),
                )
            };
            combs_a.push(xa);
            combs_b.push(xb);
        }
        let (pa, out_a) = matrix_fold::prove_fold(ma, &combs_a, &ca, ch);
        let (pb, out_b) = matrix_fold::prove_fold(mb, &combs_b, &cb, ch);
        folds.push((pa, pb));
        per_type.push((out_a, out_b));
    }

    Ok((
        AggregateProof { folds },
        Accumulator {
            registry_digest: registry.digest(),
            per_type,
        },
    ))
}

/// Fold a batch of assertions and discharge them.
///
/// The caller verifies each proof with a `*_deferred` entry (each proof has
/// its own union instance, commitment and challenger, so there is nothing
/// useful to abstract there) and passes the assertions here.
///
/// Use this to exercise or test the route end to end, NOT to speed up native
/// verification — see the module docs: folding `k` claims costs `k · nnz`,
/// so this is strictly more work than discharging each assertion directly.
/// Its value is that the *verifier* half is matrix-free.
pub fn fold_and_discharge(
    registry: &Registry,
    mats: &[TypeMatrices<'_>],
    circuits: &[&dyn crate::lincheck::LincheckCircuit],
    assertions: &[MatrixAssertion],
) -> Result<(), AggregateError> {
    let mut chp = crate::challenger::FsChallenger::new(DOMAIN);
    let (proof, _) = prove_aggregate(registry, mats, circuits, assertions, None, &mut chp)?;
    let mut chv = crate::challenger::FsChallenger::new(DOMAIN);
    let acc = verify_aggregate(registry, assertions, None, &proof, &mut chv)?;
    if acc.discharge(mats) {
        Ok(())
    } else {
        Err(AggregateError::Discharge)
    }
}

/// Replay an aggregation. **Reads no matrix** — this is the half a merge
/// circuit arithmetises.
///
/// It also checks each assertion's reported evaluations against its target
/// ([`MatrixAssertion::check_reported`]), so a caller cannot forget the one
/// step that ties a proof's reported matrix work to the proof itself.
///
/// The accumulator it returns is conditional, like everything else on this
/// path: it is true only if the inputs were, and something must eventually
/// call [`Accumulator::discharge`].
pub fn verify_aggregate<Ch: Challenger>(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    prior: Option<&Accumulator>,
    proof: &AggregateProof,
    ch: &mut Ch,
) -> Result<Accumulator, AggregateError> {
    if assertions.is_empty() && prior.is_none() {
        return Err(AggregateError::Empty);
    }
    if proof.folds.len() != registry.num_boolean() {
        return Err(AggregateError::Malformed);
    }
    if let Some(p) = prior {
        if p.registry_digest != registry.digest() || p.per_type.len() != registry.num_boolean() {
            return Err(AggregateError::RegistryMismatch);
        }
    }
    for assertion in assertions {
        assertion
            .check_reported(registry)
            .map_err(AggregateError::Reported)?;
    }

    bind(registry, prior, ch);
    let mut per_type = Vec::with_capacity(registry.num_boolean());
    for (t, (pa, pb)) in proof.folds.iter().enumerate() {
        let (ca, cb) = gather(registry, assertions, prior, t);
        let out_a = matrix_fold::verify_fold(&ca, pa, ch).map_err(AggregateError::Fold)?;
        let out_b = matrix_fold::verify_fold(&cb, pb, ch).map_err(AggregateError::Fold)?;
        per_type.push((out_a, out_b));
    }

    Ok(Accumulator {
        registry_digest: registry.digest(),
        per_type,
    })
}
