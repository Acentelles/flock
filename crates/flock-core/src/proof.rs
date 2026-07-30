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

/// [`R1csProofLigerito`] with the opening routed through the **jagged
/// transport** (`pcs::open_batch_jagged_ligerito`) instead of the direct
/// mixed Ligerito open. The PIOP part (zerocheck + lincheck) is identical —
/// on the same statement and witness the two proofs share a byte-identical
/// transcript prefix up to the opening stage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofJaggedLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofJaggedLigerito,
}

/// [`R1csProofJaggedLigerito`] with the MERGED jagged/ring-switch opening
/// (design doc §"Capacity-free ring-switching") — the PIOP sub-proofs are
/// identical; only the transport differs. The Mixed flavor's wire payload
/// since v6; the jagged variant remains as the differential oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMergedLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::MergedOpenProof,
}

/// A **mixed-class** union proof: the boolean PIOP, the element-region PIOP,
/// and ONE jagged-transport opening covering all four claims (boolean AB + C
/// ring-switched, element C + LC packed-direct).
///
/// Each class's sub-proof is `Option`: a boolean-only registry has no element
/// half, an element-only one no boolean half. Deliberately a NEW type rather
/// than extra fields on [`R1csProofJaggedLigerito`] — that struct's serialized
/// bytes are pinned by the `union_m6_fixtures` anchors, and boolean-only proofs
/// must keep going out through it unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMixedClassLigerito {
    /// Boolean zerocheck + lincheck over the `M_bool` prefix subcube.
    pub boolean: Option<BooleanPiopProof>,
    /// The element-region zerocheck + lincheck.
    pub element: Option<crate::element_r1cs::union::Proof>,
    pub pcs_open: pcs::BatchOpeningProofJaggedLigerito,
}

/// A **circuit** proof: a mixed-class union proof plus the wiring argument
/// over the circuit's cell space. What it attests, in one proof: every gate row
/// satisfies its table's relation, the circuit's wiring equalities hold, and
/// the designated cells equal the statement's public words.
///
/// The gather claims ride the SAME opening as the class claims (they are
/// packed-direct claims on the unmerged jagged path), so `pcs_open` covers all
/// of them; only the wiring transcript and the gather VALUES are extra (the
/// claim points are transcript-derived).
///
/// A separate type rather than an `Option` field on
/// [`R1csProofMixedClassLigerito`], whose serialized bytes are pinned by the
/// `union_element` anchor — non-circuit proofs must keep going out unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofCircuitLigerito {
    pub boolean: Option<BooleanPiopProof>,
    pub element: Option<crate::element_r1cs::union::Proof>,
    pub wiring: crate::circuit::WiringProof,
    pub pcs_open: pcs::BatchOpeningProofJaggedLigerito,
}

/// The boolean class's two PIOP sub-proofs, as they appear inside
/// [`R1csProofMixedClassLigerito`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanPiopProof {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
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
/// + the PCS commitment root. Call once at the top of every R1CS prove/verify
/// path, before any sub-protocol challenge is drawn. RandomChallenger ignores
/// these observations; FsChallenger uses them to defeat statement substitution.
pub fn bind_statement<Ch: Challenger>(
    challenger: &mut Ch,
    r1cs: &BlockR1cs,
    commitment: &Commitment,
) {
    challenger.observe_label(b"flock-r1cs-v0");
    challenger.observe_bytes(&r1cs.statement_digest());
    challenger.observe_bytes(&commitment.root);
}
