//! The cross-lane word linkage: the slot blocks' candidate words ARE the
//! sponge lane's squeeze stream.
//!
//! For every record, squeeze block `q` (nine of them), word `w` (68 per
//! block), and bit `b`, the claim is
//! `x_b(rec, 68q + w) = state24bit(perm 1 + q of rec, 16w + (b xor 8))`
//! (the xor is the big-endian byte swap: one address-bit complement),
//! where permutation `e` of a record sits at keccak3 block `e % 4`,
//! sub-keccak `e / 4`, state_24 slot `2 (e / 4) + 1`.
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
//! `[68q, 68q+68)`, source permutations `1..=9`, words `w < 68`)
//! decompose into aligned pieces with per-piece base coefficients. No sumchecks;
//! the equality of the two sums binds every cell pairwise with error
//! about `(records + 93) / 2^128` per challenge tuple.
//!
//! The linkage claims need challenges sampled after BOTH commitments, so
//! the composed driver runs both lanes' cores first, samples the linkage,
//! and folds each side's claims into that lane's single batched opening:
//! two Ligerito openings total for the complete relation.

use flock_core::challenger::Challenger;
use flock_core::field::F128;

use super::hash_to_point_record::{self as record, RecordProof};
use super::hash_to_point_slots as slots;
use super::hash_to_point_slots::SlotSetup;
use super::hash_to_point_sponge::{
    self as sponge, SpongeProof, SpongePublic, SpongeRecord, SpongeSetup,
};
use super::keccak3::K_LOG as KECCAK_K_LOG;

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

/// The Keccak-side claims: the source permutations `e in 1..=9` sit at
/// keccak3 block `e % 4`, sub-keccak `e / 4`, so `nu^(e-1)` factors as a
/// per-piece base coefficient times a `nu` tensor over the free block
/// bits within four fixed-sub pieces; words `w < 68` decompose into
/// aligned pieces; the four position bits carry the byte-swapped `gamma`
/// weights (bit 3 complemented).
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
    // Squeeze block q reads the OUTPUT of permutation 1 + q for q in
    // 0..9. Under e = 4 sub + blk the source set 1..=9 tiles into
    // (sub, block base, free low block bits) pieces:
    let pieces: [(usize, usize, usize); 4] = [
        (0, 1, 0), // e = 1
        (0, 2, 1), // e = 2..=3
        (1, 0, 2), // e = 4..=7
        (2, 0, 1), // e = 8..=9
    ];
    let mut claims = Vec::new();
    for (sub, blk_base, bk) in pieces {
        for (w_base, wk) in aligned_pieces(0, 68) {
            let mut scale = rec_scale * pos_scale;
            let mut point = rec_coords.clone();
            // Block bits (2, MSB-first): fixed high, nu-tensor free.
            for j in (bk..2).rev() {
                point.push(if (blk_base >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            for j in (0..bk).rev() {
                point.push(weighted_coord(F128::ONE, pow(nu, 1 << j), &mut scale)?);
            }
            // Slot bits (6, MSB-first): the sub-keccak's state_24 slot.
            let slot = 2 * sub + 1;
            for j in (0..6).rev() {
                point.push(if (slot >> j) & 1 == 1 { F128::ONE } else { F128::ZERO });
            }
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
            let coefficient = pow(nu, 4 * sub + blk_base - 1) * pow(mu, w_base) * scale;
            claims.push(LinkClaim { point, coefficient });
        }
    }
    Some(claims)
}

/// The linkage proof: the claimed values on both sides. The opening
/// proofs live inside the lanes' single batched openings; the linkage
/// claims are folded in as extra points.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LinkProof {
    pub slot_values: Vec<F128>,
    pub keccak_values: Vec<F128>,
}

fn weighted_sum(claims: &[LinkClaim], values: &[F128]) -> F128 {
    claims
        .iter()
        .zip(values)
        .fold(F128::ZERO, |sum, (claim, &value)| {
            sum + claim.coefficient * value
        })
}

/// Sample the linkage challenges (post both commitments), build both
/// claim sets, and evaluate their values on the packed witnesses. The
/// caller folds the returned points into each lane's batched opening.
pub fn prove_link_claims<Ch: Challenger>(
    slot_setup: &SlotSetup,
    sponge_setup: &SpongeSetup,
    slot_z_packed: &[F128],
    keccak_z_packed: &[F128],
    challenger: &mut Ch,
) -> (LinkProof, Vec<Vec<F128>>, Vec<Vec<F128>>) {
    challenger.observe_label(b"aerie-word-link-v0");
    let delta = challenger.sample_f128();
    let nu = challenger.sample_f128();
    let mu = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 2;

    let slot_claims =
        slot_link_claims(slot_record_vars, delta, nu, mu, gamma).expect("nondegenerate");
    let keccak_claims =
        keccak_link_claims(keccak_record_vars, delta, nu, mu, gamma).expect("nondegenerate");
    let (slot_values, keccak_values): (Vec<F128>, Vec<F128>) = rayon::join(
        || {
            let points: Vec<Vec<F128>> =
                slot_claims.iter().map(|claim| claim.point.clone()).collect();
            sponge::gather_eval_many(slot_z_packed, &points)
        },
        || {
            let points: Vec<Vec<F128>> = keccak_claims
                .iter()
                .map(|claim| claim.point.clone())
                .collect();
            sponge::gather_eval_many(keccak_z_packed, &points)
        },
    );

    let slot_points: Vec<Vec<F128>> = slot_claims.into_iter().map(|c| c.point).collect();
    let keccak_points: Vec<Vec<F128>> = keccak_claims.into_iter().map(|c| c.point).collect();
    (
        LinkProof {
            slot_values,
            keccak_values,
        },
        slot_points,
        keccak_points,
    )
}

/// The masked packed-MLE fingerprint (spec Section 6.1): per repetition
/// `k`, the shared public tag
///
/// ```text
/// tag_k = gamma_k * MLE_K(Z_H, r_k) + mask_H,k
/// ```
///
/// with `(r_k, gamma_k)` sampled after every commitment and `mask_H,k`
/// committed inside `C_H` (the record-lane commitment) before any
/// challenge. The sixteen claims per repetition (fifteen `Z_H` plane
/// sub-cubes plus the theta-weighted mask sub-cube) fold into the record
/// lane's batched opening; the claimed values travel here so the
/// verifier can check the tag equation and the opening batch together.
/// The Akita lane must prove the same `tag_k` against `C_A` (spec 6.2).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsistencyProof {
    /// Per repetition: the fifteen `Z_H` plane values at `r_k`, then the
    /// mask sub-cube value.
    pub values: Vec<Vec<F128>>,
    /// Per repetition: the shared public tag.
    pub tags: Vec<F128>,
}

/// Sample the per-repetition consistency challenges: `r_k` over the
/// packed-leaf variables and the multiplier `gamma_k`. Public so the
/// composed Section 7 driver samples ONCE from the joint transcript and
/// hands the same values to the Akita bridge (spec 6.1: both lanes must
/// prove the same tags at the same challenges).
pub fn consistency_challenges<Ch: Challenger>(
    record_vars: usize,
    challenger: &mut Ch,
) -> Vec<(Vec<F128>, F128)> {
    challenger.observe_label(b"aerie-consistency-v0");
    (0..slots::MASK_REPS)
        .map(|_| {
            let r = challenger.sample_f128_vec(record_vars + 9);
            let gamma = challenger.sample_f128();
            (r, gamma)
        })
        .collect()
}

/// The mask sub-cube claim for repetition `rep`: record coordinates
/// pinned to block 0, plane 15, the repetition's 128-slot range, and the
/// theta-tensor weights `(1, x^(2^j))` over the seven low slot bits, so
/// `scale * value = sum_h x^h * mask_bit_h = mask_K`.
fn mask_claim_point(record_vars: usize, rep: usize) -> (Vec<F128>, F128) {
    let mut point = vec![F128::ZERO; record_vars];
    for j in 0..7 {
        point.push(if (slots::MASK_PLANE >> (6 - j)) & 1 == 1 {
            F128::ONE
        } else {
            F128::ZERO
        });
    }
    // Slot top three bits: the repetition index (rep < 8 shapes fit).
    for j in 0..3 {
        point.push(if (rep >> (2 - j)) & 1 == 1 {
            F128::ONE
        } else {
            F128::ZERO
        });
    }
    // x^(2^j) by repeated squaring; x has odd multiplicative order, so
    // the weights never cancel and every coordinate is defined.
    let mut x_pows = [F128::ZERO; 7];
    let mut current = F128 { lo: 2, hi: 0 };
    for slot in x_pows.iter_mut() {
        *slot = current;
        current *= current;
    }
    let mut scale = F128::ONE;
    for j in (0..7).rev() {
        point.push(weighted_coord(F128::ONE, x_pows[j], &mut scale).expect("odd order"));
    }
    (point, scale)
}

/// Build the consistency claims and tags on the prover side. Returns the
/// proof material and the claim points to fold into the record opening.
fn prove_consistency<Ch: Challenger>(
    record_vars: usize,
    z_packed: &[F128],
    challenger: &mut Ch,
) -> (ConsistencyProof, Vec<Vec<F128>>) {
    let challenges = consistency_challenges(record_vars, challenger);
    prove_consistency_with(record_vars, z_packed, &challenges)
}

/// [`prove_consistency`] against caller-sampled challenges (the composed
/// driver's path; the challenges must come from the joint transcript
/// after every commitment).
pub fn prove_consistency_with(
    record_vars: usize,
    z_packed: &[F128],
    challenges: &[(Vec<F128>, F128)],
) -> (ConsistencyProof, Vec<Vec<F128>>) {
    let mut points = Vec::with_capacity(slots::MASK_REPS * 16);
    let mut values = Vec::with_capacity(slots::MASK_REPS);
    let mut tags = Vec::with_capacity(slots::MASK_REPS);
    for (rep, (r, gamma)) in challenges.iter().enumerate() {
        let mut rep_points = record::zh_fingerprint_points(record_vars, r);
        let (mask_point, mask_scale) = mask_claim_point(record_vars, rep);
        rep_points.push(mask_point);
        let rep_values = sponge::gather_eval_many(z_packed, &rep_points);
        let v = record::zh_fingerprint_value(&rep_values[..15]);
        tags.push(*gamma * v + mask_scale * rep_values[15]);
        points.extend(rep_points);
        values.push(rep_values);
    }
    (ConsistencyProof { values, tags }, points)
}

/// Verify-side counterpart: re-sample the challenges, check the tag
/// equations against the claimed values, and return the claim points and
/// flattened values for the record lane's batched opening verify.
fn verify_consistency<Ch: Challenger>(
    record_vars: usize,
    proof: &ConsistencyProof,
    challenger: &mut Ch,
) -> Result<(Vec<Vec<F128>>, Vec<F128>), &'static str> {
    let challenges = consistency_challenges(record_vars, challenger);
    verify_consistency_with(record_vars, proof, &challenges)
}

/// [`verify_consistency`] against caller-sampled challenges.
pub fn verify_consistency_with(
    record_vars: usize,
    proof: &ConsistencyProof,
    challenges: &[(Vec<F128>, F128)],
) -> Result<(Vec<Vec<F128>>, Vec<F128>), &'static str> {
    if proof.values.len() != slots::MASK_REPS || proof.tags.len() != slots::MASK_REPS {
        return Err("wrong consistency repetition count");
    }
    let mut points = Vec::with_capacity(slots::MASK_REPS * 16);
    let mut flat_values = Vec::with_capacity(slots::MASK_REPS * 16);
    for (rep, (r, gamma)) in challenges.iter().enumerate() {
        let rep_values = &proof.values[rep];
        if rep_values.len() != 16 {
            return Err("wrong consistency claim count");
        }
        let (mask_point, mask_scale) = mask_claim_point(record_vars, rep);
        let v = record::zh_fingerprint_value(&rep_values[..15]);
        if proof.tags[rep] != *gamma * v + mask_scale * rep_values[15] {
            return Err("a consistency tag equation does not hold");
        }
        points.extend(record::zh_fingerprint_points(record_vars, r));
        points.push(mask_point);
        flat_values.extend_from_slice(rep_values);
    }
    Ok((points, flat_values))
}

/// Verify the linkage identity (the two weighted sums agree) and return
/// both claim point lists for the lanes' batched opening verifies.
pub fn verify_link_claims<Ch: Challenger>(
    slot_setup: &SlotSetup,
    sponge_setup: &SpongeSetup,
    proof: &LinkProof,
    challenger: &mut Ch,
) -> Result<(Vec<Vec<F128>>, Vec<Vec<F128>>), &'static str> {
    challenger.observe_label(b"aerie-word-link-v0");
    let delta = challenger.sample_f128();
    let nu = challenger.sample_f128();
    let mu = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 2;

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
    Ok((slot_points, keccak_points))
}

/// The complete private-salt HashToPoint proof: both lanes, the linkage,
/// and the masked consistency tags. COVERAGE: with the linkage, the
/// committed candidate words are the genuine XOF stream of the framed
/// (private) salt, so the full Section 3.2 relation holds and the tags
/// bind `MLE_K(Z_H, r_k)`; the remaining aerie-side work is proving the
/// SAME tags against `C_A` (spec 6.2) and the block-L target derivation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HashToPointProof {
    pub sponge: SpongeProof,
    pub record: RecordProof,
    pub link: LinkProof,
    pub consistency: ConsistencyProof,
}

/// Transcript order: sponge core, record core, linkage challenges and
/// values, consistency challenges and tags, then the two batched
/// openings (sponge first) with the linkage and consistency claims
/// folded in as extra points. Two openings total. In the composed
/// Section 7 flow the challenger is the joint aerie transcript, seeded
/// after `C_A` and the public descriptors, so every consistency
/// challenge already follows both lanes' commitments.
pub fn prove_hash_to_point<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    records: &[SpongeRecord],
    masks: &[[bool; 128]; slots::MASK_REPS],
    challenger: &mut Ch,
) -> HashToPointProof {
    let trace = std::env::var("FLOCK_TRACE").is_ok();
    let mut stage = std::time::Instant::now();
    let mut lap = |label: &str| {
        if trace {
            eprintln!(
                "[hash_to_point] {label}: {:8.1} ms",
                stage.elapsed().as_secs_f64() * 1e3
            );
        }
        stage = std::time::Instant::now();
    };
    let sponge_core = sponge::prove_sponge_core(sponge_setup, records, challenger);
    lap("sponge core (keccak lane)");
    let blocks: Vec<[u16; slots::SLOTS]> = sponge_core
        .all_words
        .iter()
        .map(|w| {
            let mut block = [0_u16; slots::SLOTS];
            block.copy_from_slice(w);
            block
        })
        .collect();
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let record_core = record::prove_record_core_with_masks(slot_setup, &blocks, masks, challenger);
    lap("record core (slot lane)");
    let (link, slot_points, keccak_points) = prove_link_claims(
        slot_setup,
        sponge_setup,
        &record_core.z_packed,
        &sponge_core.fast.z_packed,
        challenger,
    );
    lap("word linkage claims");
    let (consistency, consistency_points) =
        prove_consistency(slot_record_vars, &record_core.z_packed, challenger);
    lap("consistency claims");
    let mut record_extra = slot_points;
    let mut record_extra_values = link.slot_values.clone();
    record_extra.extend(consistency_points);
    for rep_values in &consistency.values {
        record_extra_values.extend_from_slice(rep_values);
    }
    let (sponge_proof, _) =
        sponge::open_sponge(
            sponge_setup,
            sponge_core,
            &keccak_points,
            &link.keccak_values,
            challenger,
        );
    lap("sponge open (keccak PCS)");
    let (record_proof, _) =
        record::open_record(slot_setup, record_core, &record_extra, &record_extra_values, challenger);
    lap("record open (slot PCS)");
    HashToPointProof {
        sponge: sponge_proof,
        record: record_proof,
        link,
        consistency,
    }
}

pub fn verify_hash_to_point<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    publics: &[SpongePublic],
    proof: &HashToPointProof,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    let sponge_core = sponge::verify_sponge_core(sponge_setup, publics, &proof.sponge, challenger)?;
    let record_core = record::verify_record_core(slot_setup, &proof.record, challenger)?;
    let (slot_points, keccak_points) =
        verify_link_claims(slot_setup, sponge_setup, &proof.link, challenger)?;
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let (consistency_points, consistency_values) =
        verify_consistency(slot_record_vars, &proof.consistency, challenger)?;
    let mut record_extra_points = slot_points;
    record_extra_points.extend(consistency_points);
    let mut record_extra_values = proof.link.slot_values.clone();
    record_extra_values.extend(consistency_values);
    sponge::verify_sponge_open(
        sponge_setup,
        &proof.sponge,
        sponge_core,
        &keccak_points,
        &proof.link.keccak_values,
        challenger,
    )?;
    record::verify_record_open(
        slot_setup,
        &proof.record,
        record_core,
        &record_extra_points,
        &record_extra_values,
        challenger,
    )
}

/// The SINGLE-ROOT composed proof (S4): ONE commitment `C_H` over the
/// concatenation of both lanes' packed witnesses, both lane PIOPs bound
/// to it, and ONE batched Ligerito opening carrying every claim from
/// both lanes, lifted to the concatenated domain.
///
/// Witness layout: `m_H = max(m_sponge, m_record) + 1` address bits; the
/// TOP address bit is the lane (sponge = 0, record = 1); each lane's
/// witness sits at offset 0 of its half, zero-padded above. Lifting a
/// lane claim appends `m_H - 1 - m_lane` zero coordinates plus the lane
/// coordinate at the END of its fold suffix (the top of the address
/// domain), which restricts the concatenated MLE exactly to the lane
/// block, so every lane claim value and every precomputed `s_hat_v`
/// carries over unchanged.
///
/// WIRE: this is a different transcript and proof shape from
/// [`HashToPointProof`] (one commitment and one opening instead of two);
/// the per-claim ring-switch wire cost stays (flagged debt; the spec-7.3
/// closure is the eventual fix once its prover regression is closed).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HashToPointSingleRootProof {
    pub commitment: flock_core::pcs::Commitment,
    pub sponge_zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub sponge_lincheck: flock_core::lincheck::LincheckProof,
    pub sponge_opening_values: Vec<F128>,
    pub record_zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub record_lincheck: flock_core::lincheck::LincheckProof,
    pub record_scatter: super::hash_to_point_scatter::ScatterProof,
    pub record_opening_values: Vec<F128>,
    pub record_fingerprint_value: F128,
    pub link: LinkProof,
    pub consistency: ConsistencyProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProofLigerito,
}

/// PCS parameters for the concatenated single root: one variable above
/// the larger lane, with the lanes' (asserted-equal) rate, batch size,
/// profile, and Merkle hash.
pub fn single_root_params(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
) -> flock_core::pcs::PcsParams {
    let sp = &sponge_setup.keccak.pcs_params;
    let rp = &slot_setup.pcs_params;
    assert_eq!(
        sp.log_inv_rate, rp.log_inv_rate,
        "lane PCS rates must agree for the single root"
    );
    assert_eq!(
        sp.log_batch_size, rp.log_batch_size,
        "lane PCS batch sizes must agree for the single root"
    );
    flock_core::pcs::PcsParams {
        m: sp.m.max(rp.m) + 1,
        log_inv_rate: sp.log_inv_rate,
        log_batch_size: sp.log_batch_size,
        profile: sp.profile,
        merkle_hash: sp.merkle_hash,
    }
}

/// Lift one lane fold suffix to the concatenated domain: append the
/// zero pad coordinates and the lane coordinate at the END (the top of
/// the address domain). Exact for Boolean appended coordinates: the
/// lifted eq tensor is the lane tensor supported on the lane block.
fn lift_to_root(mut x_full: Vec<F128>, lane_m: usize, root_m: usize, lane: bool) -> Vec<F128> {
    x_full.reserve(root_m - lane_m);
    x_full.extend(std::iter::repeat_n(F128::ZERO, root_m - 1 - lane_m));
    x_full.push(if lane { F128::ONE } else { F128::ZERO });
    x_full
}

/// Single-root composed prover. Same PIOP content and claim set as
/// [`prove_hash_to_point`], but ONE commitment before both lane PIOPs
/// and ONE batched opening after the linkage and consistency claims.
/// Transcript order: `C_H`, sponge bind + PIOP, record bind + PIOP,
/// linkage, consistency, the single opening (sponge claims first).
pub fn prove_hash_to_point_single_root<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    records: &[SpongeRecord],
    masks: &[[bool; 128]; slots::MASK_REPS],
    challenger: &mut Ch,
) -> HashToPointSingleRootProof {
    use flock_core::pcs;
    let trace = std::env::var("FLOCK_TRACE").is_ok();
    let mut stage = std::time::Instant::now();
    let mut lap = |label: &str| {
        if trace {
            eprintln!(
                "[hash_to_point 1root] {label}: {:8.1} ms",
                stage.elapsed().as_secs_f64() * 1e3
            );
        }
        stage = std::time::Instant::now();
    };
    let m_s = sponge_setup.keccak.r1cs.m;
    let m_r = slot_setup.r1cs.m;
    let params_h = single_root_params(sponge_setup, slot_setup);

    // Both lanes' witnesses BEFORE any transcript interaction (the
    // record blocks are the sponge's squeeze words).
    let sponge_wit = sponge::sponge_witness(sponge_setup, records);
    let blocks: Vec<[u16; slots::SLOTS]> = sponge_wit
        .all_words
        .iter()
        .map(|w| {
            let mut block = [0_u16; slots::SLOTS];
            block.copy_from_slice(w);
            block
        })
        .collect();
    let record_wit = record::record_witness(slot_setup, &blocks, masks);
    lap("witness generation (both lanes)");

    // The concatenated root: sponge half then record half, each lane at
    // offset 0 of its half, zero-padded above.
    let half_words = 1_usize << (params_h.m - 8);
    let mut z_h = vec![F128::ZERO; 2 * half_words];
    z_h[..sponge_wit.z_packed.len()].copy_from_slice(&sponge_wit.z_packed);
    z_h[half_words..half_words + record_wit.z_packed.len()]
        .copy_from_slice(&record_wit.z_packed);
    let (commitment, prover_data) = pcs::commit(&z_h, &params_h);
    lap("single-root commit");

    let sponge_core =
        sponge::prove_sponge_core_bound(sponge_setup, sponge_wit, commitment.clone(), None, challenger);
    lap("sponge core (keccak lane)");
    let record_core =
        record::prove_record_core_bound(slot_setup, record_wit, commitment.clone(), None, challenger);
    lap("record core (slot lane)");
    let (link, slot_points, keccak_points) = prove_link_claims(
        slot_setup,
        sponge_setup,
        &record_core.z_packed,
        &sponge_core.fast.z_packed,
        challenger,
    );
    lap("word linkage claims");
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let (consistency, consistency_points) =
        prove_consistency(slot_record_vars, &record_core.z_packed, challenger);
    lap("consistency claims");

    // The single claim list, lane-grouped: sponge [ab, c, multis…]
    // lifted at lane 0, then record [ab, c, multis…] lifted at lane 1.
    // Multilinear order per lane matches the two-root extra sets:
    // sponge core points, keccak link points; record core points, slot
    // link points, consistency points.
    let mut x_fulls: Vec<Vec<F128>> = Vec::new();
    let mut precomputed: Vec<Option<&[F128]>> = Vec::new();
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&sponge_core.fast.ab.point),
        m_s,
        params_h.m,
        false,
    ));
    precomputed.push(sponge_core.fast.s_hat_v_ab.as_deref());
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&sponge_core.fast.c.point),
        m_s,
        params_h.m,
        false,
    ));
    precomputed.push(Some(sponge_core.fast.s_hat_v_c.as_slice()));
    for point in sponge_core.points.iter().chain(&keccak_points) {
        let (_low, x_outer) = record::flock_claim_shape(point);
        x_fulls.push(lift_to_root(x_outer, m_s, params_h.m, false));
        precomputed.push(None);
    }
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&record_core.ab.point),
        m_r,
        params_h.m,
        true,
    ));
    precomputed.push(Some(record_core.s_hat_v_ab.as_slice()));
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&record_core.c.point),
        m_r,
        params_h.m,
        true,
    ));
    precomputed.push(Some(record_core.s_hat_v_c.as_slice()));
    for point in record_core
        .points
        .iter()
        .chain(&slot_points)
        .chain(&consistency_points)
    {
        let (_low, x_outer) = record::flock_claim_shape(point);
        x_fulls.push(lift_to_root(x_outer, m_r, params_h.m, true));
        precomputed.push(None);
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_config = params_h
        .ligerito_prover_config()
        .expect("Ligerito config for the merged domain");
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_h,
        &prover_data,
        &commitment,
        &x_refs,
        &precomputed,
        &[],
        &flock_core::zerocheck::PaddingSpec::dense(params_h.m),
        &lig_config,
        challenger,
    );
    lap("single batched open");
    drop(precomputed);

    HashToPointSingleRootProof {
        commitment,
        sponge_zerocheck: sponge_core.fast.zc_proof,
        sponge_lincheck: sponge_core.fast.lc_proof,
        sponge_opening_values: sponge_core.opening_values,
        record_zerocheck: record_core.zc_proof,
        record_lincheck: record_core.lc_proof,
        record_scatter: record_core.scatter_proof,
        record_opening_values: record_core.opening_values,
        record_fingerprint_value: record_core.fingerprint_value,
        link,
        consistency,
        pcs_open,
    }
}

pub fn verify_hash_to_point_single_root<Ch: Challenger>(
    sponge_setup: &SpongeSetup,
    slot_setup: &SlotSetup,
    publics: &[SpongePublic],
    proof: &HashToPointSingleRootProof,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    use flock_core::pcs::{self, LowBinding};
    let m_s = sponge_setup.keccak.r1cs.m;
    let m_r = slot_setup.r1cs.m;
    let params_h = single_root_params(sponge_setup, slot_setup);

    let sponge_core = sponge::verify_sponge_core_parts(
        sponge_setup,
        publics,
        &proof.commitment,
        &proof.sponge_zerocheck,
        &proof.sponge_lincheck,
        &proof.sponge_opening_values,
        challenger,
    )?;
    let record_core = record::verify_record_core_parts(
        slot_setup,
        &proof.commitment,
        &proof.record_zerocheck,
        &proof.record_lincheck,
        &proof.record_scatter,
        &proof.record_opening_values,
        proof.record_fingerprint_value,
        challenger,
    )?;
    let (slot_points, keccak_points) =
        verify_link_claims(slot_setup, sponge_setup, &proof.link, challenger)?;
    let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
    let (consistency_points, consistency_values) =
        verify_consistency(slot_record_vars, &proof.consistency, challenger)?;

    // The prover's claim order: sponge [ab, c, multis…], record
    // [ab, c, multis…].
    let mut claim_values: Vec<F128> = Vec::new();
    let mut bindings: Vec<LowBinding> = Vec::new();
    let mut x_fulls: Vec<Vec<F128>> = Vec::new();
    claim_values.push(sponge_core.ab.value);
    bindings.push(LowBinding::Quirky {
        z_skip: sponge_core.ab.point.z_skip,
    });
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&sponge_core.ab.point),
        m_s,
        params_h.m,
        false,
    ));
    claim_values.push(sponge_core.c.value);
    bindings.push(LowBinding::Quirky {
        z_skip: sponge_core.c.point.z_skip,
    });
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&sponge_core.c.point),
        m_s,
        params_h.m,
        false,
    ));
    claim_values.extend_from_slice(&proof.sponge_opening_values);
    claim_values.extend_from_slice(&proof.link.keccak_values);
    for point in sponge_core.points.iter().chain(&keccak_points) {
        let (x_low, x_outer) = record::flock_claim_shape(point);
        bindings.push(LowBinding::Multilinear { x_low });
        x_fulls.push(lift_to_root(x_outer, m_s, params_h.m, false));
    }
    claim_values.push(record_core.ab.value);
    bindings.push(LowBinding::Quirky {
        z_skip: record_core.ab.point.z_skip,
    });
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&record_core.ab.point),
        m_r,
        params_h.m,
        true,
    ));
    claim_values.push(record_core.c.value);
    bindings.push(LowBinding::Quirky {
        z_skip: record_core.c.point.z_skip,
    });
    x_fulls.push(lift_to_root(
        crate::prover::quirky_x_outer_full(&record_core.c.point),
        m_r,
        params_h.m,
        true,
    ));
    claim_values.extend_from_slice(&proof.record_opening_values);
    claim_values.extend_from_slice(&proof.link.slot_values);
    claim_values.extend_from_slice(&consistency_values);
    for point in record_core
        .points
        .iter()
        .chain(&slot_points)
        .chain(&consistency_points)
    {
        let (x_low, x_outer) = record::flock_claim_shape(point);
        bindings.push(LowBinding::Multilinear { x_low });
        x_fulls.push(lift_to_root(x_outer, m_r, params_h.m, true));
    }
    if claim_values.len() != x_fulls.len() || bindings.len() != x_fulls.len() {
        return Err("single-root claim bookkeeping disagrees");
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_config = params_h
        .ligerito_verifier_config()
        .expect("verifier config for the merged domain");
    pcs::verify_opening_batch_ligerito_mixed_bound(
        &proof.commitment,
        &claim_values,
        &bindings,
        &x_refs,
        &[],
        &proof.pcs_open,
        &lig_config,
        challenger,
    )
    .map_err(|_| "single-root batched opening verification failed")
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

        let mut state_lists = Vec::new();
        let mut blocks = Vec::new();
        for record in &inputs {
            let (states, words) = sponge::sponge_trace(record);
            state_lists.push(states);
            let mut block = [0_u16; slots::SLOTS];
            block.copy_from_slice(&words);
            blocks.push(block);
        }
        let initial_states = sponge::sponge_initial_states(&state_lists);
        let (keccak_z, _a, _b, _l) =
            super::super::keccak3::generate_witness_with_ab_packed_and_lincheck(
                &initial_states,
                sponge_setup.keccak.n_blocks_log(),
            );
        let slot_z_bools = slot_setup.generate_witness(&blocks);
        let slot_z = flock_core::pcs::pack_witness(&slot_z_bools, slot_setup.r1cs.m);

        let (delta, nu, mu, gamma) = (small(0x1111), small(0x2323), small(0x4545), small(0x6767));
        let slot_record_vars = slot_setup.r1cs.m - slots::K_LOG;
        let keccak_record_vars = sponge_setup.keccak.r1cs.m - KECCAK_K_LOG - 2;
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

        // Nonzero masks so the mask sub-cube openings carry real values.
        let masks: [[bool; 128]; slots::MASK_REPS] =
            std::array::from_fn(|rep| std::array::from_fn(|bit| (rep + bit) % 3 == 0));
        let mut prover = FsChallenger::new(b"aerie-hash-to-point");
        let proof = prove_hash_to_point(&sponge_setup, &slot_setup, &inputs, &masks, &mut prover);

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

        // A tampered consistency tag rejects at the tag equation.
        let mut wrong = proof.clone();
        wrong.consistency.tags[1] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point");
        assert!(
            verify_hash_to_point(&sponge_setup, &slot_setup, &publics, &wrong, &mut fresh).is_err()
        );

        // A tampered consistency claim value rejects: either the tag
        // equation breaks, or (were the tag forged to match) the batched
        // opening rejects the value — the same authenticated path the
        // linkage-value tamper above exercises.
        let mut wrong = proof.clone();
        wrong.consistency.values[0][7] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point");
        assert!(
            verify_hash_to_point(&sponge_setup, &slot_setup, &publics, &wrong, &mut fresh).is_err()
        );
    }

    #[test]
    fn single_root_hash_to_point_roundtrips() {
        // S4: the complete relation under ONE concatenated commitment and
        // ONE batched opening. Same claim content as the two-root path.
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
        let masks: [[bool; 128]; slots::MASK_REPS] =
            std::array::from_fn(|rep| std::array::from_fn(|bit| (rep + bit) % 3 == 0));

        let mut prover = FsChallenger::new(b"aerie-hash-to-point-1root");
        let proof = prove_hash_to_point_single_root(
            &sponge_setup,
            &slot_setup,
            &inputs,
            &masks,
            &mut prover,
        );

        let mut verifier = FsChallenger::new(b"aerie-hash-to-point-1root");
        verify_hash_to_point_single_root(
            &sponge_setup,
            &slot_setup,
            &publics,
            &proof,
            &mut verifier,
        )
        .expect("the single-root HashToPoint proof verifies");

        // A tampered linkage value rejects through the single opening.
        let mut wrong = proof.clone();
        wrong.link.slot_values[3] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point-1root");
        assert!(
            verify_hash_to_point_single_root(
                &sponge_setup,
                &slot_setup,
                &publics,
                &wrong,
                &mut fresh
            )
            .is_err()
        );

        // A tampered public message rejects (the sponge pinning term moves).
        let mut wrong_publics = publics.clone();
        wrong_publics[5].message[0] ^= 1;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point-1root");
        assert!(
            verify_hash_to_point_single_root(
                &sponge_setup,
                &slot_setup,
                &wrong_publics,
                &proof,
                &mut fresh
            )
            .is_err()
        );

        // A tampered record opening value rejects.
        let mut wrong = proof.clone();
        wrong.record_opening_values[10] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point-1root");
        assert!(
            verify_hash_to_point_single_root(
                &sponge_setup,
                &slot_setup,
                &publics,
                &wrong,
                &mut fresh
            )
            .is_err()
        );

        // A tampered sponge opening value rejects (either a chain edge
        // or the single opening).
        let mut wrong = proof.clone();
        wrong.sponge_opening_values[4] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-hash-to-point-1root");
        assert!(
            verify_hash_to_point_single_root(
                &sponge_setup,
                &slot_setup,
                &publics,
                &wrong,
                &mut fresh
            )
            .is_err()
        );
    }
}
