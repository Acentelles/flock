//! The cross-lane word linkage: the slot blocks' candidate words ARE the
//! sponge lane's squeeze stream.
//!
//! For every record, squeeze block `q` (nine of them), word `w` (68 per
//! block), and bit `b`, the claim is
//! `x_b(rec, 68q + w) = state24bit(instance 16 rec + 1 + q, 16w + (b xor 8))`
//! (the xor is the big-endian byte swap: one address-bit complement).
//! With post-commitment challenges `(delta, nu, mu, gamma)`, both sides
//! collapse to transparent-weighted sums
//!
//! ```text
//! S = sum delta^rec nu^q mu^w gamma^b * bit
//! ```
//!
//! and every term is a weighted SUB-CUBE OPENING: the x-planes sit at a
//! sixteen-aligned base so the `gamma^b` combination is one tensor over
//! the four low plane bits; the non-power-of-two ranges (`s` in
//! `[68q, 68q+68)`, instance offsets `1..=10`, words `w < 68`) decompose
//! into aligned pieces with per-piece base coefficients. No sumchecks;
//! the equality of the two sums binds every cell pairwise with error
//! about `(records + 93) / 2^128` per challenge tuple.
//!
//! The linkage opens each lane's commitment once more (its claims need
//! challenges sampled after BOTH commitments); folding these claims into
//! the lanes' main batches is a known optimization.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::pcs::{self, Commitment};

use super::hash_to_point_record::{self as record, RecordProof};
use super::hash_to_point_slots as slots;
use super::hash_to_point_slots::SlotSetup;
use super::hash_to_point_sponge::{
    self as sponge, LaneArtifacts, SpongeProof, SpongePublic, SpongeRecord, SpongeSetup,
};
use super::keccak::K_LOG as KECCAK_K_LOG;

/// Greedy aligned decomposition of `[start, end)` into `(base, k)` pieces
/// with `base % 2^k == 0` and `base + 2^k <= end`.
pub fn aligned_pieces(start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut base = start;
    while base < end {
        let align = if base == 0 {
            usize::MAX.count_ones() as usize
        } else {
            base.trailing_zeros() as usize
        };
        let mut k = align.min(usize::BITS as usize - 1);
        while (1_usize << k) > end - base {
            k -= 1;
        }
        pieces.push((base, k));
        base += 1 << k;
    }
    pieces
}

fn pow(base: F128, mut exponent: usize) -> F128 {
    let mut acc = F128::ONE;
    let mut cur = base;
    while exponent != 0 {
        if exponent & 1 == 1 {
            acc *= cur;
        }
        cur *= cur;
        exponent >>= 1;
    }
    acc
}

/// A coordinate binding an address bit with weights `(w0, w1)` for the
/// bit being 0/1: the opening uses `w1 / (w0 + w1)` and the claim carries
/// the scale `w0 + w1`. None when the weights cancel.
fn weighted_coord(w0: F128, w1: F128, scale: &mut F128) -> Option<F128> {
    let sum = w0 + w1;
    if sum == F128::ZERO {
        return None;
    }
    *scale *= sum;
    Some(w1 * sum.inv())
}

/// `delta`-power record coordinates (MSB-first) plus their scale.
fn record_coords(record_vars: usize, delta: F128) -> Option<(Vec<F128>, F128)> {
    let mut scale = F128::ONE;
    let mut coords = Vec::with_capacity(record_vars);
    for i in 0..record_vars {
        let weight = pow(delta, 1 << (record_vars - 1 - i));
        coords.push(weighted_coord(F128::ONE, weight, &mut scale)?);
    }
    Some((coords, scale))
}

/// One weighted claim: an opening point and the public coefficient its
/// value carries in the linkage sum.
pub struct LinkClaim {
    pub point: Vec<F128>,
    pub coefficient: F128,
}

/// The slot-side claims: for each squeeze block `q`, the aligned pieces
/// of `s in [68q, 68q + 68)`, each one opening the sixteen x-planes with
/// a `gamma` tensor over the four low plane bits.
pub fn slot_link_claims(
    record_vars: usize,
    delta: F128,
    nu: F128,
    mu: F128,
    gamma: F128,
) -> Option<Vec<LinkClaim>> {
    let (rec_coords, rec_scale) = record_coords(record_vars, delta)?;
    // Plane bits (7, MSB-first) for planes 16 + b: (0, 0, 1, gamma-tensor).
    let mut plane_scale = F128::ONE;
    let mut plane_coords = vec![F128::ZERO, F128::ZERO, F128::ONE];
    for j in 0..4 {
        plane_coords.push(weighted_coord(
            F128::ONE,
            pow(gamma, 1 << (3 - j)),
            &mut plane_scale,
        )?);
    }
    let mut claims = Vec::new();
    for q in 0..9_usize {
        for (base, k) in aligned_pieces(68 * q, 68 * (q + 1)) {
            let mut scale = rec_scale * plane_scale;
            let mut point = rec_coords.clone();
            point.extend_from_slice(&plane_coords);
            for j in (k..10).rev() {
                point.push(if (base >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            for j in (0..k).rev() {
                point.push(weighted_coord(F128::ONE, pow(mu, 1 << j), &mut scale)?);
            }
            let coefficient = pow(nu, q) * pow(mu, base - 68 * q) * scale;
            claims.push(LinkClaim { point, coefficient });
        }
    }
    Some(claims)
}

/// The Keccak-side claims: instance offsets `1..=10` and words `w < 68`
/// decompose into aligned pieces; the four position bits carry the
/// byte-swapped `gamma` weights (bit 3 complemented).
pub fn keccak_link_claims(
    record_vars: usize,
    delta: F128,
    nu: F128,
    mu: F128,
    gamma: F128,
) -> Option<Vec<LinkClaim>> {
    let (rec_coords, rec_scale) = record_coords(record_vars, delta)?;
    // Position coords (4, MSB-first p3..p0): p3 weights (gamma^8, 1).
    let mut pos_scale = F128::ONE;
    let mut pos_coords = Vec::with_capacity(4);
    pos_coords.push(weighted_coord(pow(gamma, 8), F128::ONE, &mut pos_scale)?);
    for j in (0..3).rev() {
        pos_coords.push(weighted_coord(
            F128::ONE,
            pow(gamma, 1 << j),
            &mut pos_scale,
        )?);
    }
    let mut claims = Vec::new();
    // Squeeze block q reads the OUTPUT of instance 1 + q for q in 0..9,
    // so the source instances are 1..=9.
    for (inst_base, ik) in aligned_pieces(1, 10) {
        for (w_base, wk) in aligned_pieces(0, 68) {
            let mut scale = rec_scale * pos_scale;
            let mut point = rec_coords.clone();
            // Instance low-4 bits: fixed high, nu-tensor free.
            for j in (ik..4).rev() {
                point.push(if (inst_base >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            for j in (0..ik).rev() {
                point.push(weighted_coord(F128::ONE, pow(nu, 1 << j), &mut scale)?);
            }
            // Block offset top five: the state_24 slot.
            point.extend_from_slice(&[F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO, F128::ONE]);
            // Word bits (offset bits 10..4): fixed high, mu-tensor free.
            for j in (wk..7).rev() {
                point.push(if (w_base >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            for j in (0..wk).rev() {
                point.push(weighted_coord(F128::ONE, pow(mu, 1 << j), &mut scale)?);
            }
            point.extend_from_slice(&pos_coords);
            let coefficient = pow(nu, inst_base - 1) * pow(mu, w_base) * scale;
            claims.push(LinkClaim { point, coefficient });
        }
    }
    Some(claims)
}

/// The linkage proof: claimed values plus one extra batched opening per
/// lane commitment.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LinkProof {
    pub slot_values: Vec<F128>,
    pub keccak_values: Vec<F128>,
    pub slot_open: pcs::BatchOpeningProofLigerito,
    pub keccak_open: pcs::BatchOpeningProofLigerito,
}

fn weighted_sum(claims: &[LinkClaim], values: &[F128]) -> F128 {
    claims
        .iter()
        .zip(values)
        .fold(F128::ZERO, |sum, (claim, &value)| {
            sum + claim.coefficient * value
        })
}

pub fn prove_link<Ch: Challenger>(
    slot_setup: &SlotSetup,
    sponge_setup: &SpongeSetup,
    slot_artifacts: &LaneArtifacts,
    keccak_artifacts: &LaneArtifacts,
    challenger: &mut Ch,
) -> LinkProof {
    challenger.observe_label(b"aerie-word-link-v0");
    let delta = challenger.sample_f128();
    let nu = challenger.sample_f128();
    let mu = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 4;

    let slot_claims =
        slot_link_claims(slot_record_vars, delta, nu, mu, gamma).expect("nondegenerate");
    let keccak_claims =
        keccak_link_claims(keccak_record_vars, delta, nu, mu, gamma).expect("nondegenerate");
    let slot_values: Vec<F128> = slot_claims
        .iter()
        .map(|claim| sponge::gather_eval(&slot_artifacts.z_packed, &claim.point))
        .collect();
    let keccak_values: Vec<F128> = keccak_claims
        .iter()
        .map(|claim| sponge::gather_eval(&keccak_artifacts.z_packed, &claim.point))
        .collect();

    let slot_points: Vec<Vec<F128>> = slot_claims.into_iter().map(|c| c.point).collect();
    let keccak_points: Vec<Vec<F128>> = keccak_claims.into_iter().map(|c| c.point).collect();
    let slot_open = record::open_multilinear(
        slot_artifacts.z_packed.clone(),
        &slot_artifacts.prover_data,
        &slot_artifacts.commitment,
        &slot_points,
        &slot_setup.r1cs.padding_spec(),
        &slot_setup.pcs_params,
        challenger,
    );
    let keccak_open = record::open_multilinear(
        keccak_artifacts.z_packed.clone(),
        &keccak_artifacts.prover_data,
        &keccak_artifacts.commitment,
        &keccak_points,
        &sponge_setup.keccak.r1cs.padding_spec(),
        &sponge_setup.keccak.pcs_params,
        challenger,
    );
    LinkProof {
        slot_values,
        keccak_values,
        slot_open,
        keccak_open,
    }
}

pub fn verify_link<Ch: Challenger>(
    slot_setup: &SlotSetup,
    sponge_setup: &SpongeSetup,
    slot_commitment: &Commitment,
    keccak_commitment: &Commitment,
    proof: &LinkProof,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    challenger.observe_label(b"aerie-word-link-v0");
    let delta = challenger.sample_f128();
    let nu = challenger.sample_f128();
    let mu = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 4;

    let slot_claims =
        slot_link_claims(slot_record_vars, delta, nu, mu, gamma).ok_or("degenerate")?;
    let keccak_claims =
        keccak_link_claims(keccak_record_vars, delta, nu, mu, gamma).ok_or("degenerate")?;
    if proof.slot_values.len() != slot_claims.len()
        || proof.keccak_values.len() != keccak_claims.len()
    {
        return Err("wrong linkage claim count");
    }
    if weighted_sum(&slot_claims, &proof.slot_values)
        != weighted_sum(&keccak_claims, &proof.keccak_values)
    {
        return Err("the word linkage does not hold");
    }
    let slot_points: Vec<Vec<F128>> = slot_claims.into_iter().map(|c| c.point).collect();
    let keccak_points: Vec<Vec<F128>> = keccak_claims.into_iter().map(|c| c.point).collect();
    record::verify_multilinear(
        slot_commitment,
        &slot_points,
        &proof.slot_values,
        &proof.slot_open,
        &slot_setup.pcs_params,
        challenger,
    )
    .map_err(|_| "slot-side linkage opening failed")?;
    record::verify_multilinear(
        keccak_commitment,
        &keccak_points,
        &proof.keccak_values,
        &proof.keccak_open,
        &sponge_setup.keccak.pcs_params,
        challenger,
    )
    .map_err(|_| "keccak-side linkage opening failed")
}

/// The complete private-salt HashToPoint proof: both lanes plus the
/// linkage. COVERAGE: with the linkage, the committed candidate words
/// are the genuine XOF stream of the framed (private) salt, so the full
/// Section 3.2 relation holds; the remaining aerie-side work is the
/// dual-commitment fingerprint against the Akita lane.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HashToPointProof {
    pub sponge: SpongeProof,
    pub record: RecordProof,
    pub link: LinkProof,
}

pub fn prove_hash_to_point<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    records: &[SpongeRecord],
    challenger: &mut Ch,
) -> HashToPointProof {
    let (sponge_proof, words, keccak_artifacts) =
        sponge::prove_sponge(sponge_setup, records, challenger);
    let blocks: Vec<[u16; slots::SLOTS]> = words
        .iter()
        .map(|w| {
            let mut block = [0_u16; slots::SLOTS];
            block.copy_from_slice(w);
            block
        })
        .collect();
    let (record_proof, slot_artifacts) = record::prove_record(slot_setup, &blocks, challenger);
    let link = prove_link(
        slot_setup,
        sponge_setup,
        &slot_artifacts,
        &keccak_artifacts,
        challenger,
    );
    HashToPointProof {
        sponge: sponge_proof,
        record: record_proof,
        link,
    }
}

pub fn verify_hash_to_point<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    publics: &[SpongePublic],
    proof: &HashToPointProof,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    sponge::verify_sponge(sponge_setup, publics, &proof.sponge, challenger)?;
    record::verify_record(slot_setup, &proof.record, challenger)?;
    verify_link(
        slot_setup,
        sponge_setup,
        &proof.record.commitment,
        &proof.sponge.commitment,
        &proof.link,
        challenger,
    )
}

#[cfg(test)]
mod tests {
    use flock_core::challenger::FsChallenger;

    use super::*;

    fn small(v: u64) -> F128 {
        F128 { lo: v, hi: 0 }
    }

    fn test_records(n: usize) -> Vec<SpongeRecord> {
        (0..n as u8)
            .map(|seed| SpongeRecord {
                salt: [seed.wrapping_add(1); 40],
                hpk: [seed.wrapping_mul(3).wrapping_add(7); 64],
                message: (0..33).map(|i| seed.wrapping_add(i)).collect(),
            })
            .collect()
    }

    #[test]
    fn aligned_pieces_tile_their_ranges() {
        for (start, end) in [(0, 68), (68, 136), (476, 544), (1, 11), (0, 1024)] {
            let pieces = aligned_pieces(start, end);
            let mut cursor = start;
            for (base, k) in pieces {
                assert_eq!(base, cursor);
                assert_eq!(base % (1 << k), 0);
                cursor = base + (1 << k);
            }
            assert_eq!(cursor, end);
        }
    }

    #[test]
    fn the_linkage_identity_holds_on_honest_witnesses() {
        // No proving: build both witnesses, evaluate every claim by
        // gathering, and check the two weighted sums agree, plus that a
        // tampered word breaks them. This pins all the linkage
        // bookkeeping cheaply.
        let records = 2;
        let sponge_setup = SpongeSetup::new(records);
        let slot_setup = SlotSetup::new(records);
        let inputs = test_records(records);

        let zero_state = [false; super::super::keccak::STATE_BITS];
        let mut initial_states = Vec::new();
        let mut blocks = Vec::new();
        for record in &inputs {
            let (states, words) = sponge::sponge_trace(record);
            initial_states.extend_from_slice(&states);
            initial_states.extend(std::iter::repeat_n(
                zero_state,
                sponge::PERM_SLOTS - sponge::LIVE_PERMS,
            ));
            let mut block = [0_u16; slots::SLOTS];
            block.copy_from_slice(&words);
            blocks.push(block);
        }
        while initial_states.len() < sponge_setup.keccak.n_keccak_slots() {
            initial_states.push(zero_state);
        }
        let (keccak_z, _a, _b, _l) =
            super::super::keccak::generate_witness_with_ab_packed_and_lincheck(
                &initial_states,
                sponge_setup.keccak.n_keccaks_log(),
            );
        let slot_z_bools = slot_setup.generate_witness(&blocks);
        let slot_z = flock_core::pcs::pack_witness(&slot_z_bools, slot_setup.r1cs.m);

        let (delta, nu, mu, gamma) = (small(0x1111), small(0x2323), small(0x4545), small(0x6767));
        let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
        let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 4;
        let slot_claims =
            slot_link_claims(slot_record_vars, delta, nu, mu, gamma).expect("defined");
        let keccak_claims =
            keccak_link_claims(keccak_record_vars, delta, nu, mu, gamma).expect("defined");

        let slot_sum = slot_claims.iter().fold(F128::ZERO, |sum, claim| {
            sum + claim.coefficient * sponge::gather_eval(&slot_z, &claim.point)
        });
        let keccak_sum = keccak_claims.iter().fold(F128::ZERO, |sum, claim| {
            sum + claim.coefficient * sponge::gather_eval(&keccak_z, &claim.point)
        });
        assert_eq!(slot_sum, keccak_sum, "the honest linkage identity");

        // A tampered slot word breaks the identity.
        let mut tampered = blocks.clone();
        tampered[1][100] ^= 4;
        let tampered_bools = slot_setup.generate_witness(&tampered);
        let tampered_z = flock_core::pcs::pack_witness(&tampered_bools, slot_setup.r1cs.m);
        let tampered_sum = slot_claims.iter().fold(F128::ZERO, |sum, claim| {
            sum + claim.coefficient * sponge::gather_eval(&tampered_z, &claim.point)
        });
        assert_ne!(tampered_sum, keccak_sum);
    }

    #[test]
    fn full_hash_to_point_roundtrips() {
        // The COMPLETE private-salt HashToPoint relation: sponge lane,
        // record lane, and the word linkage, at the smallest legal size.
        let records = 32;
        let sponge_setup = SpongeSetup::new(records);
        let slot_setup = SlotSetup::new(records);
        let inputs = test_records(records);
        let publics: Vec<SpongePublic> = inputs
            .iter()
            .map(|r| SpongePublic {
                hpk: r.hpk,
                message: r.message.clone(),
            })
            .collect();

        let mut prover = FsChallenger::new(b"aerie-hash-to-point");
        let proof = prove_hash_to_point(&sponge_setup, &slot_setup, &inputs, &mut prover);

        let mut verifier = FsChallenger::new(b"aerie-hash-to-point");
        verify_hash_to_point(&sponge_setup, &slot_setup, &publics, &proof, &mut verifier)
            .expect("the full HashToPoint proof verifies");

        // A tampered linkage value rejects.
        let mut wrong = proof.clone();
        wrong.link.slot_values[3] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point");
        assert!(
            verify_hash_to_point(&sponge_setup, &slot_setup, &publics, &wrong, &mut fresh).is_err()
        );

        // A tampered public message rejects (the sponge pinning term moves).
        let mut wrong_publics = publics.clone();
        wrong_publics[5].message[0] ^= 1;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point");
        assert!(
            verify_hash_to_point(
                &sponge_setup,
                &slot_setup,
                &wrong_publics,
                &proof,
                &mut fresh
            )
            .is_err()
        );
    }
}
