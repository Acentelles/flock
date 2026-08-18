//! Shared R1CS proof types and the Fiat-Shamir statement binding.
//!
//! These live in a backend-neutral module (rather than in `prover`) so the
//! verifier can name them without depending on the prove path. The prover
//! produces these structs; the verifier consumes them.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{self, QuirkyPoint};
use crate::pcs::{self, Commitment};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;
use serde::{Deserialize, Serialize};

/// Top-level R1CS proof: zerocheck + lincheck transcripts, plus one batched
/// Ligerito PCS opening covering both the `ab` and `c` z-claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

/// Top-level R1CS proof with the **AG-skip** zerocheck + Ligerito PCS backend.
/// Identical downstream of the zerocheck (lincheck + the same standard
/// ring-switch open on the std pack); only round 1 of the zerocheck differs
/// (the genus-95 AG multiplication code replaces the RS additive-NTT skip), so
/// `ag` carries the AG round messages instead of a `ZerocheckProof`. The skip
/// stays in the packing prefix `[skip 6 | bit6]`, so both `(ab, c)` claims open
/// via the unchanged RS path with AG base-code skip weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigeritoAg {
    pub ag: zerocheck::ag_skip::AgProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

/// [`R1csProofLigerito`] with the MERGED jagged/ring-switch opening
/// (design doc §"Capacity-free ring-switching") — the PIOP sub-proofs are
/// identical; only the transport differs. The Mixed flavor's wire payload
/// since v6.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMergedLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::MergedOpenProof,
}

/// A **mixed-class** union proof over the MERGED (Frobenius) transport: the
/// boolean PIOP, the element-region PIOP, and ONE merged opening covering
/// all four claims (boolean AB + C ring-switched, element C + LC
/// packed-direct — each expressed, to the weight builder, as the
/// `F₂`-linear map `x ↦ γ·x`, indistinguishable from a ring-switched
/// claim's Φ-fold).
///
/// Each class's sub-proof is `Option`: a boolean-only registry has no element
/// half, an element-only one no boolean half.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMixedClassMerged {
    /// Boolean zerocheck + lincheck over the `M_bool` prefix subcube.
    pub boolean: Option<BooleanPiopProof>,
    /// The element-region zerocheck + lincheck.
    pub element: Option<crate::element_r1cs::union::Proof>,
    pub pcs_open: pcs::MergedOpenProof,
}

/// A **circuit** proof over the MERGED (Frobenius) transport — a
/// mixed-class union proof plus the wiring argument over the circuit's cell
/// space. What it attests, in one proof: every gate row satisfies its
/// table's relation, the circuit's wiring equalities hold, and the
/// designated cells equal the statement's public words.
///
/// The wiring argument's gather claims are packed-direct, which the merged
/// transport carries by expressing each weight as the `F₂`-linear map
/// `x ↦ γ·x` — the same intake the element class's claims use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofCircuitMerged {
    pub boolean: Option<BooleanPiopProof>,
    pub element: Option<crate::element_r1cs::union::Proof>,
    pub wiring: crate::circuit::WiringProof,
    pub pcs_open: pcs::MergedOpenProof,
}

/// The boolean class's two PIOP sub-proofs, as they appear inside
/// [`R1csProofMixedClassMerged`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanPiopProof {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
}

/// [`BooleanPiopProof`] with the **AG-skip** zerocheck: the genus-95 AG
/// multiplication code replaces the RS additive-NTT round 1; the lincheck and
/// every claim downstream are unchanged (the skip point rides as
/// [`lincheck::SkipPoint::Ag`]). PADDING CONTRACT: the AG round-1 sum reads
/// the full `2^m_bool` boolean region, so the witness must be built in
/// honest-zero padding mode — the run-list-gated dirty-padding mode the RS
/// kernels tolerate is UNSOUND here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanPiopProofAg {
    pub ag: zerocheck::ag_skip::AgProof,
    pub lincheck: lincheck::LincheckProof,
}

/// [`R1csProofMergedLigerito`] with the **AG-skip** boolean zerocheck — the
/// boolean-only union proof over the MERGED transport, AG flavor. Same
/// transport, same lincheck, same merged opening; only the zerocheck's round
/// 1 differs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMergedLigeritoAg {
    pub boolean: BooleanPiopProofAg,
    pub pcs_open: pcs::MergedOpenProof,
}

/// The claims a verified mixed-class union proof leaves behind, per class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionClassClaims {
    /// Boolean AB + C — `None` when the registry has no boolean types.
    pub boolean: Option<R1csClaim>,
    /// Element C + LC, in union word coordinates — `None` when the registry
    /// has no element types.
    pub element: Option<crate::element_r1cs::union::Claims>,
}

/// A claim of the form `ẑ(point) = value` for the witness `z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZClaim {
    pub point: QuirkyPoint,
    pub value: F128,
}

/// Two MLE evaluation claims on `z` that the PCS layer must verify.
///
/// Both `point.x_outer` parts differ; both `point.z_skip` and
/// `point.x_inner_rest` shapes match (one univariate-skip coord + multilinear
/// inner-rest), so this is "two quirky-shaped openings of `z`."
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csClaim {
    /// From lincheck: `ẑ(ab.point) = ab.value` — covers both `â` and `b̂` at
    /// the same point (their lincheck claims collapsed to a shared z-claim
    /// at a fresh quirky inner point).
    pub ab: ZClaim,
    /// From the zerocheck's extract_c interpolation: `ẑ(c.point) = c.value`.
    /// Bypasses lincheck because `C = I` ⇒ ĉ-claim is a direct z-claim.
    pub c: ZClaim,
}

/// Bind the Fiat-Shamir transcript to the statement: the R1CS instance digest
/// + the PCS commitment cap. Call once at the top of every R1CS prove/verify
/// path, before any sub-protocol challenge is drawn. RandomChallenger ignores
/// these observations; FsChallenger uses them to defeat statement substitution.
pub fn bind_statement<Ch: Challenger>(
    challenger: &mut Ch,
    r1cs: &BlockR1cs,
    commitment: &Commitment,
) {
    challenger.observe_label(b"flock-r1cs-v0");
    challenger.observe_bytes(&r1cs.statement_digest());
    challenger.observe_bytes(commitment.cap.as_flattened());
}
