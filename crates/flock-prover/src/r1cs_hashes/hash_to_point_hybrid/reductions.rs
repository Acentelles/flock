//! Borrowed reductions and canonical pending-claim ledgers.
//!
//! These helpers can run sequentially on the legacy challenger. No fork or
//! new protocol profile is selected here. In particular, a Merkle root alone
//! is not a claim that the current Johnson/OOD PCS extracts a fixed witness
//! before its later list-binding opening challenges.

use super::*;
use flock_core::pcs::PcsParams;

#[derive(Clone, Copy, PartialEq, Eq)]
struct WitnessSource {
    address: usize,
    len: usize,
    descriptor: [u8; 32],
    m: usize,
}
impl WitnessSource {
    fn new(setup: &Setup, z: &[F128]) -> Self {
        Self {
            address: z.as_ptr() as usize,
            len: z.len(),
            descriptor: setup.descriptor,
            m: setup.r1cs.m,
        }
    }
    fn matches(&self, z: &[F128]) -> bool {
        self.address == z.as_ptr() as usize && self.len == z.len()
    }
}

#[derive(Clone)]
struct RelationLedger {
    source_points: Vec<Vec<F128>>,
    groups: Vec<Vec<Fragment>>,
    originals: Vec<F128>,
    points: Vec<Vec<F128>>,
    values: Vec<F128>,
}

impl RelationLedger {
    fn evaluate(z: &[F128], source: &[Vec<F128>], relocate: fn(&[F128]) -> Vec<Fragment>) -> Self {
        let groups = flatten(source, relocate);
        let (originals, points, values) = evaluations(z, &groups);
        Self {
            source_points: source.to_vec(),
            groups,
            originals,
            points,
            values,
        }
    }

    fn checked(
        source_points: Vec<Vec<F128>>,
        relocate: fn(&[F128]) -> Vec<Fragment>,
        originals: &[F128],
        values: &[F128],
    ) -> Result<Self, &'static str> {
        let groups = flatten(&source_points, relocate);
        let points = check_fragments(&groups, values, originals)?;
        Ok(Self {
            source_points,
            groups,
            originals: originals.to_vec(),
            points,
            values: values.to_vec(),
        })
    }

    fn append(&mut self, other: Self) {
        self.source_points.extend(other.source_points);
        self.groups.extend(other.groups);
        self.originals.extend(other.originals);
        self.points.extend(other.points);
        self.values.extend(other.values);
    }

    fn absorb<Ch: Challenger>(&self, ch: &mut Ch) {
        assert_eq!(self.source_points.len(), self.groups.len());
        assert_eq!(self.source_points.len(), self.originals.len());
        ch.observe_label(b"aerie/hybrid/relation-ledger-v1");
        count(ch, self.source_points.len());
        count(ch, self.values.len());
        let mut offset = 0;
        for ((point, group), &value) in self
            .source_points
            .iter()
            .zip(&self.groups)
            .zip(&self.originals)
        {
            coordinates(ch, point);
            ch.observe_f128(value);
            count(ch, group.len());
            for fragment in group {
                coordinates(ch, &fragment.point);
                ch.observe_f128(fragment.weight);
                ch.observe_f128(self.values[offset]);
                offset += 1;
            }
        }
        debug_assert_eq!(offset, self.values.len());
    }
}

fn count<Ch: Challenger>(ch: &mut Ch, n: usize) {
    ch.observe_bytes(
        &u64::try_from(n)
            .expect("ledger count fits u64")
            .to_le_bytes(),
    );
}
fn coordinates<Ch: Challenger>(ch: &mut Ch, point: &[F128]) {
    count(ch, point.len());
    ch.observe_f128_slice(point);
}
fn quirky<Ch: Challenger>(ch: &mut Ch, claim: &ZClaim) {
    ch.observe_label(b"quirky-binding-v1");
    ch.observe_f128(claim.point.z_skip);
    coordinates(ch, &claim.point.x_inner_rest);
    coordinates(ch, &claim.point.x_outer);
    ch.observe_f128(claim.value);
}

/// Reproduce only the legacy setup/R1CS/root prefix, with no new observations.
/// This does not bind every PCS parameter or assert early PCS extraction.
pub fn bind_root<Ch: Challenger>(setup: &Setup, commitment: &Commitment, ch: &mut Ch) {
    setup.bind(ch);
    flock_core::proof::bind_statement(ch, &setup.r1cs, commitment);
}

/// Optional complete statement ledger for a future explicitly reviewed profile.
/// Unused by legacy proving/verifying. Validate parameters before using any
/// commitment-supplied geometry; all dimensions come from the trusted setup.
pub fn absorb_statement_ledger<Ch: Challenger>(
    setup: &Setup,
    commitment: &Commitment,
    ch: &mut Ch,
) -> Result<(), &'static str> {
    let params = &commitment.params;
    let expected = &setup.params;
    if params.m != expected.m
        || params.log_inv_rate != expected.log_inv_rate
        || params.log_batch_size != expected.log_batch_size
        || params.profile != expected.profile
        || params.merkle_hash != expected.merkle_hash
    {
        return Err("hybrid commitment parameters");
    }
    ch.observe_label(b"aerie/hybrid/statement-ledger-v1");
    ch.observe_bytes(layout::DESCRIPTOR_DOMAIN);
    ch.observe_bytes(&setup.descriptor);
    ch.observe_bytes(&setup.r1cs.statement_digest());
    ch.observe_bytes(&commitment.root);
    absorb_params(ch, params);
    Ok(())
}

fn absorb_params<Ch: Challenger>(ch: &mut Ch, params: &PcsParams) {
    count(ch, params.m);
    count(ch, params.log_inv_rate);
    count(ch, params.log_batch_size);
    ch.observe_bytes(match params.profile {
        pcs::ligerito::LigeritoProfile::Fast => b"fast",
        pcs::ligerito::LigeritoProfile::Slim => b"slim",
        pcs::ligerito::LigeritoProfile::Secure => b"secure",
        pcs::ligerito::LigeritoProfile::Grind => b"grind",
    });
    ch.observe_bytes(params.merkle_hash.as_str().as_bytes());
}

/// Circuit and sponge claims still pending the single binary PCS opening.
pub struct ProverCircuitReduction {
    source: WitnessSource,
    fast: crate::prover::ProveCoreReduction,
    sponge: RelationLedger,
}
impl ProverCircuitReduction {
    /// Bind every derived claim, including returned values absent from the
    /// reduction transcript. This changes the transcript only when called.
    pub fn absorb_ledger<Ch: Challenger>(&self, ch: &mut Ch) {
        circuit_ledger(&self.fast.ab, &self.fast.c, &self.sponge, ch);
    }
}

/// Record claims still pending the single binary PCS opening.
pub struct ProverRecordReduction {
    source: WitnessSource,
    scatter: scatter::ScatterProof,
    record: RelationLedger,
    r_fp: Vec<F128>,
    fingerprint: F128,
}
impl ProverRecordReduction {
    pub fn r_fp(&self) -> &[F128] {
        &self.r_fp
    }
    pub fn fingerprint(&self) -> F128 {
        self.fingerprint
    }
    pub fn absorb_ledger<Ch: Challenger>(&self, ch: &mut Ch) {
        record_ledger(&self.record, &self.r_fp, self.fingerprint, ch);
    }
}

fn circuit_ledger<Ch: Challenger>(ab: &ZClaim, c: &ZClaim, sponge: &RelationLedger, ch: &mut Ch) {
    ch.observe_label(b"aerie/hybrid/circuit-claim-ledger-v1");
    count(ch, 2);
    quirky(ch, ab);
    quirky(ch, c);
    sponge.absorb(ch);
}
fn record_ledger<Ch: Challenger>(
    record: &RelationLedger,
    r_fp: &[F128],
    fingerprint: F128,
    ch: &mut Ch,
) {
    ch.observe_label(b"aerie/hybrid/record-claim-ledger-v1");
    coordinates(ch, r_fp);
    ch.observe_f128(fingerprint);
    record.absorb(ch);
}

/// Borrow z while consuming/recycling the old a/b/stripe buffers. Caller must
/// first bind the statement, e.g. with [`bind_root`]. No new schedule is implied.
pub fn prove_circuit_after_root<Ch: Challenger>(
    setup: &Setup,
    z: &[F128],
    a: Vec<F128>,
    b: Vec<F128>,
    options: zerocheck::ProverOptions,
    ch: &mut Ch,
) -> ProverCircuitReduction {
    prove_circuit_with_packer(
        setup,
        z,
        a,
        b,
        options,
        ch,
        lincheck::pack_z_lincheck_from_packed,
    )
}

pub(super) fn prove_circuit_with_packer<Ch: Challenger>(
    setup: &Setup,
    z: &[F128],
    a: Vec<F128>,
    b: Vec<F128>,
    options: zerocheck::ProverOptions,
    ch: &mut Ch,
    pack: impl FnOnce(&[F128], usize, usize) -> Vec<u8>,
) -> ProverCircuitReduction {
    let stripe_span = tracing::info_span!("hybrid.lincheck_stripes").entered();
    let stripes = pack(z, setup.r1cs.m, layout::K_LOG);
    drop(stripe_span);
    let fast_span = tracing::info_span!("hybrid.zerocheck_lincheck").entered();
    let fast = crate::prover::prove_fast_core_reduction_after_statement(
        &setup.r1cs,
        z,
        a,
        b,
        stripes,
        setup,
        options,
        ch,
    );
    drop(fast_span);
    let points = sponge::sponge_relation_points(setup.record_vars(), ch);
    let _span = tracing::info_span!("hybrid.sponge_relation").entered();
    let sponge = RelationLedger::evaluate(z, &points, sponge_fragments);
    ProverCircuitReduction {
        source: WitnessSource::new(setup, z),
        fast,
        sponge,
    }
}

/// Borrow exactly the committed z used by the circuit reduction. The callback
/// is deterministic preparation only; no authenticated fingerprint is issued.
pub fn prove_record<Ch: Challenger>(
    setup: &Setup,
    z: &[F128],
    packed_record: bool,
    ch: &mut Ch,
    on_point: impl FnOnce(&[F128]),
) -> ProverRecordReduction {
    let _span = tracing::info_span!("hybrid.record_relation").entered();
    // The existing relation evaluates scatter claims and fingerprint claims
    // separately; retain both batches in their existing order.
    let ledger = std::cell::RefCell::new(None::<RelationLedger>);
    let evaluate = |points: &[Vec<F128>]| {
        let batch = RelationLedger::evaluate(z, points, record_fragments);
        let values = batch.originals.clone();
        let mut ledger = ledger.borrow_mut();
        if let Some(previous) = ledger.as_mut() {
            previous.append(batch);
        } else {
            *ledger = Some(batch);
        }
        values
    };
    let relation = if packed_record {
        record::prove_record_relation_packed_with_point_hook(
            setup.record_vars(),
            |p| {
                debug_assert_eq!(p % 64, 0);
                layout::record_position(p % super::super::hash_to_point_slots::K).map_or(0, |q| {
                    debug_assert_eq!(q % 64, 0);
                    let bit = (p / super::super::hash_to_point_slots::K) * layout::K + q;
                    let word = z[bit / 128];
                    if bit.is_multiple_of(128) {
                        word.lo
                    } else {
                        word.hi
                    }
                })
            },
            evaluate,
            ch,
            on_point,
        )
    } else {
        record::prove_record_relation_with_point_hook(
            setup.record_vars(),
            |p| {
                layout::record_position(p % super::super::hash_to_point_slots::K).is_some_and(|q| {
                    layout::bit(
                        z,
                        (p / super::super::hash_to_point_slots::K) * layout::K + q,
                    )
                })
            },
            evaluate,
            ch,
            on_point,
        )
    };
    let record = ledger.into_inner().expect("record evaluation batches");
    debug_assert_eq!(record.source_points, relation.points);
    debug_assert_eq!(record.originals, relation.values);
    ProverRecordReduction {
        source: WitnessSource::new(setup, z),
        scatter: relation.scatter,
        record,
        r_fp: relation.r_fp,
        fingerprint: relation.fingerprint,
    }
}

/// Reattach the same witness/commitment, preserving sponge-before-record order.
/// The allocation guard detects accidental copies or cross-setup results; it
/// does not hash the witness or authenticate commitment/prover-data identity.
/// The coordinator must retain the original Prepared owner and leave z
/// immutable through recombination, with its original commitment and data.
pub fn recombine(
    z: Vec<F128>,
    commitment: Commitment,
    data: pcs::ProverData,
    circuit: ProverCircuitReduction,
    record: ProverRecordReduction,
) -> Result<Core, &'static str> {
    if circuit.source != record.source
        || !circuit.source.matches(&z)
        || circuit.source.m != commitment.params.m
    {
        return Err("hybrid reduction witness source");
    }
    let ProverCircuitReduction { fast, sponge, .. } = circuit;
    let mut points = sponge.points;
    points.extend(record.record.points);
    let mut values = sponge.values;
    values.extend(record.record.values);
    Ok(Core {
        fast: fast.with_witness(z, commitment, Some(data)),
        scatter: record.scatter,
        sponge_values: sponge.originals,
        record_values: record.record.originals,
        fingerprint: record.fingerprint,
        r_fp: record.r_fp,
        points,
        values,
    })
}

/// Replayed circuit claims, not a verified opening or standalone acceptance.
pub struct VerifierCircuitReduction<'proof> {
    proof: &'proof Proof,
    ab: ZClaim,
    c: ZClaim,
    sponge: RelationLedger,
}
impl VerifierCircuitReduction<'_> {
    pub fn absorb_ledger<Ch: Challenger>(&self, ch: &mut Ch) {
        circuit_ledger(&self.ab, &self.c, &self.sponge, ch);
    }
}
/// Replayed record claims, not a verified opening or standalone acceptance.
pub struct VerifierRecordReduction<'proof> {
    proof: &'proof Proof,
    record: RelationLedger,
    r_fp: Vec<F128>,
    fingerprint: F128,
}
impl VerifierRecordReduction<'_> {
    pub fn r_fp(&self) -> &[F128] {
        &self.r_fp
    }
    pub fn fingerprint(&self) -> F128 {
        self.fingerprint
    }
    pub fn absorb_ledger<Ch: Challenger>(&self, ch: &mut Ch) {
        record_ledger(&self.record, &self.r_fp, self.fingerprint, ch);
    }
}

/// Replay pending claims after root binding. A new-profile coordinator must
/// call the checked [`absorb_statement_ledger`] before either reduction.
pub fn verify_circuit_after_root<'proof, Ch: Challenger>(
    setup: &Setup,
    publics: &[sponge::SpongePublic],
    proof: &'proof Proof,
    ch: &mut Ch,
) -> Result<VerifierCircuitReduction<'proof>, &'static str> {
    let (ab, c) = flock_core::verifier::verify_core_after_statement(
        &setup.r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        setup,
        ch,
    )
    .map_err(|_| "hybrid binary circuit")?;
    let points =
        sponge::verify_sponge_relation(setup.record_vars(), publics, &proof.sponge_values, ch)?;
    let n = flatten(&points, sponge_fragments)
        .iter()
        .map(Vec::len)
        .sum();
    let values = proof
        .fragment_values
        .get(..n)
        .ok_or("hybrid relocation shape")?;
    let sponge = RelationLedger::checked(points, sponge_fragments, &proof.sponge_values, values)?;
    Ok(VerifierCircuitReduction {
        proof,
        ab,
        c,
        sponge,
    })
}

/// Replay the record suffix of the proof's fragment ledger. As with the
/// circuit half, a new-profile coordinator validates the full root first.
pub fn verify_record<'proof, Ch: Challenger>(
    setup: &Setup,
    proof: &'proof Proof,
    ch: &mut Ch,
) -> Result<VerifierRecordReduction<'proof>, &'static str> {
    let (points, r_fp) = record::verify_record_relation(
        setup.record_vars(),
        &proof.scatter,
        &proof.record_values,
        proof.fingerprint,
        ch,
    )?;
    let n: usize = flatten(&points, record_fragments)
        .iter()
        .map(Vec::len)
        .sum();
    let offset = proof
        .fragment_values
        .len()
        .checked_sub(n)
        .ok_or("hybrid relocation shape")?;
    let record = RelationLedger::checked(
        points,
        record_fragments,
        &proof.record_values,
        &proof.fragment_values[offset..],
    )?;
    Ok(VerifierRecordReduction {
        proof,
        record,
        r_fp,
        fingerprint: proof.fingerprint,
    })
}

/// Reject overlapping, missing or surplus fragment ledgers before opening.
pub fn recombine_verifier(
    proof: &Proof,
    circuit: VerifierCircuitReduction<'_>,
    record: VerifierRecordReduction<'_>,
) -> Result<VerifyCore, &'static str> {
    if !std::ptr::eq(proof, circuit.proof) || !std::ptr::eq(proof, record.proof) {
        return Err("hybrid reduction proof source");
    }
    if circuit
        .sponge
        .values
        .len()
        .checked_add(record.record.values.len())
        != Some(proof.fragment_values.len())
    {
        return Err("hybrid relocation shape");
    }
    let mut points = circuit.sponge.points;
    points.extend(record.record.points);
    Ok(VerifyCore {
        r_fp: record.r_fp,
        ab: circuit.ab,
        c: circuit.c,
        points,
    })
}

#[cfg(test)]
mod tests;
