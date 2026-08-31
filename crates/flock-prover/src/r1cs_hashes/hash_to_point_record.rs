//! Record assembly for the aerie private-salt HashToPoint lane.
//!
//! Composes, against REAL commitments and openings, the pieces built in
//! `hash_to_point_slots` and `hash_to_point_scatter`: the slot R1CS proof,
//! the dense `Z_H` commitment, the scatter sumcheck with its terminal
//! discharge, and the aerie fingerprint opening. The multilinear opening
//! points (plane sub-cubes at random slot coordinates) ride the standard
//! batched ring-switch/Ligerito opening through the `LowBinding` seam:
//! the `s_hat_v` message is binding-agnostic, and the verifier weights it
//! with plain equality weights over the seven low address bits.
//!
//! Point conventions: this module's callers use MOST significant variable
//! first (the aerie fold order); flock's `build_eq` binds `r[i]` to index
//! bit `i` (least significant first). [`flock_claim_shape`] converts.
//!
//! COVERAGE: the proof binds the slot relation (accept, decomposition,
//! centering, counter, gate), the compaction scatter, and the `Z_H` table.
//! The Keccak sponge lane (that the slot words are the XOF stream of the
//! framed salt input) is NOT yet included; see `AERIE-ADAPTER.md` work
//! item 2. Any benchmark of this proof must say so.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck;
use flock_core::pcs::{self, Commitment, LowBinding, PcsParams};
use flock_core::proof::{R1csClaim, ZClaim, bind_statement};
use flock_core::zerocheck::{self, PaddingSpec};

use super::hash_to_point_scatter as scatter;
use super::hash_to_point_slots as slots;
use super::hash_to_point_slots::{SLOTS, SlotSetup};

/// Slot-domain variables (one plane) and the record-block size.
const SLOT_VARS: usize = slots::SLOT_VARS;
const COUNTER_BITS: usize = slots::COUNTER_BITS;

/// Convert one MSB-first `m`-variable opening point into flock's claim
/// shape: the seven low-address-bit coordinates plus the `x_outer` vector
/// (`[bit-6 coordinate, word suffix in flock's LSB-first order]`).
pub fn flock_claim_shape(point_msb: &[F128]) -> ([F128; 7], Vec<F128>) {
    let m = point_msb.len();
    assert!(m > 7);
    // q[i] binds address bit i.
    let q: Vec<F128> = point_msb.iter().rev().copied().collect();
    let x_low: [F128; 7] = q[..7].try_into().expect("seven low coordinates");
    let mut x_outer = Vec::with_capacity(m - 6);
    x_outer.push(q[6]);
    x_outer.extend_from_slice(&q[7..]);
    (x_low, x_outer)
}

/// Bit-MLE evaluation that folds Boolean coordinates by index selection
/// (a gather over the sub-cube) and only the non-Boolean coordinates
/// arithmetically. The record-lane claims all fix their plane bits, so
/// this is orders of magnitude cheaper than the dense fold.
pub fn bit_mle_fast(bits: &[bool], point_msb: &[F128]) -> F128 {
    let vars = point_msb.len();
    assert_eq!(bits.len(), 1 << vars);
    // Partition coordinates: var i (MSB-first) controls address bit
    // (vars - 1 - i). Boolean coords contribute a fixed address offset;
    // the rest form the gathered sub-cube, order preserved.
    let mut fixed_offset = 0_usize;
    let mut free: Vec<(usize, F128)> = Vec::new();
    for (i, &coord) in point_msb.iter().enumerate() {
        let bit = vars - 1 - i;
        if coord == F128::ZERO {
            // contributes nothing
        } else if coord == F128::ONE {
            fixed_offset |= 1 << bit;
        } else {
            free.push((bit, coord));
        }
    }
    let free_vars = free.len();
    let mut layer = vec![F128::ZERO; 1 << free_vars];
    for (index, slot) in layer.iter_mut().enumerate() {
        // Sub-cube index bits map to the free address bits, MSB-first.
        let mut address = fixed_offset;
        for (j, &(bit, _)) in free.iter().enumerate() {
            if (index >> (free_vars - 1 - j)) & 1 == 1 {
                address |= 1 << bit;
            }
        }
        *slot = if bits[address] { F128::ONE } else { F128::ZERO };
    }
    for &(_, r) in &free {
        let half = layer.len() / 2;
        for i in 0..half {
            let low = layer[i];
            layer[i] = low + r * (low + layer[half + i]);
        }
        layer.truncate(half);
    }
    layer[0]
}

/// Direct MSB-first bit-MLE evaluation of a boolean table (reference and
/// prover-side claim values).
pub fn bit_mle(bits: &[bool], point_msb: &[F128]) -> F128 {
    assert_eq!(bits.len(), 1 << point_msb.len());
    let mut layer: Vec<F128> = bits
        .iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect();
    for &r in point_msb {
        let half = layer.len() / 2;
        for i in 0..half {
            let low = layer[i];
            layer[i] = low + r * (low + layer[half + i]);
        }
        layer.truncate(half);
    }
    layer[0]
}

/// Open a committed boolean witness at arbitrary MSB-first multilinear
/// points, over the standard batched ring-switch/Ligerito path.
pub fn open_multilinear<Ch: Challenger>(
    z_packed: Vec<F128>,
    prover_data: &pcs::ProverData,
    commitment: &Commitment,
    points_msb: &[Vec<F128>],
    padding: &PaddingSpec,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> pcs::BatchOpeningProofLigerito {
    let shapes: Vec<(_, Vec<F128>)> = points_msb.iter().map(|p| flock_claim_shape(p)).collect();
    let x_refs: Vec<&[F128]> = shapes.iter().map(|(_, x)| x.as_slice()).collect();
    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito config; bump m for tiny instances");
    pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        &[],
        &[],
        padding,
        &lig_config,
        challenger,
    )
}

/// Verify [`open_multilinear`]'s proof against claimed values.
pub fn verify_multilinear<Ch: Challenger>(
    commitment: &Commitment,
    points_msb: &[Vec<F128>],
    values: &[F128],
    proof: &pcs::BatchOpeningProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    assert_eq!(points_msb.len(), values.len());
    let shapes: Vec<([F128; 7], Vec<F128>)> =
        points_msb.iter().map(|p| flock_claim_shape(p)).collect();
    let bindings: Vec<LowBinding> = shapes
        .iter()
        .map(|(x_low, _)| LowBinding::Multilinear { x_low: *x_low })
        .collect();
    let x_refs: Vec<&[F128]> = shapes.iter().map(|(_, x)| x.as_slice()).collect();
    let lig_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito verifier config");
    pcs::verify_opening_batch_ligerito_mixed_bound(
        commitment,
        values,
        &bindings,
        &x_refs,
        &[],
        proof,
        &lig_config,
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

    fn record_batch(n: usize) -> Vec<[u16; SLOTS]> {
        (0..n)
            .map(|block| {
                let mut words = [0_u16; SLOTS];
                for (slot, word) in words.iter_mut().enumerate() {
                    *word = ((block * SLOTS + slot) as u16)
                        .wrapping_mul(9_973)
                        .wrapping_add(211);
                }
                words
            })
            .collect()
    }

    #[test]
    fn fast_bit_mle_matches_the_dense_fold() {
        let bits: Vec<bool> = (0..1_u64 << 12)
            .map(|i| i.wrapping_mul(0x9E37_79B9) & 64 != 0)
            .collect();
        // Mixed Boolean and field coordinates in several patterns.
        for seed in 0..4_u64 {
            let point: Vec<F128> = (0..12_u64)
                .map(|i| match (i + seed) % 3 {
                    0 => F128::ZERO,
                    1 => F128::ONE,
                    _ => small(i * 977 + seed * 131 + 5),
                })
                .collect();
            assert_eq!(bit_mle_fast(&bits, &point), bit_mle(&bits, &point));
        }
    }

    #[test]
    fn record_proof_roundtrips_and_rejects_tampering() {
        // The complete record lane against real commitments: slot R1CS,
        // scatter, discharge, Z_H binding, one batched opening. Coverage
        // note: the Keccak sponge lane is NOT part of this proof.
        let n_records = 32;
        let setup = SlotSetup::new(n_records);
        let blocks = record_batch(n_records);
        let mut prover_challenger = FsChallenger::new(b"aerie-record-proof");
        let (proof, _artifacts) = prove_record(&setup, &blocks, &mut prover_challenger);

        let mut verifier_challenger = FsChallenger::new(b"aerie-record-proof");
        verify_record(&setup, &proof, &mut verifier_challenger).expect("record proof verifies");

        // A tampered Z_H opening value breaks the binding identity.
        let mut wrong = proof.clone();
        wrong.opening_values[43] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-record-proof");
        assert!(verify_record(&setup, &wrong, &mut fresh).is_err());

        // A tampered counter-plane opening breaks the terminal.
        let mut wrong = proof.clone();
        wrong.opening_values[35] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-record-proof");
        assert!(verify_record(&setup, &wrong, &mut fresh).is_err());

        // A tampered gate opening breaks the terminal reconstruction.
        let mut wrong = proof.clone();
        wrong.opening_values[0] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-record-proof");
        assert!(verify_record(&setup, &wrong, &mut fresh).is_err());
    }

    #[test]
    fn multilinear_openings_verify_against_the_real_commitment() {
        // Milestone: arbitrary-point bit-MLE openings of the committed slot
        // witness through the standard batched PCS path, using the
        // multilinear low binding. This is the seam every scatter terminal
        // rides.
        let n_records = 32; // m = 22, the smallest default-Ligerito size.
        let setup = SlotSetup::new(n_records);
        let blocks = record_batch(n_records);
        let z = setup.generate_witness(&blocks);
        let m = setup.r1cs.m;
        let z_packed = flock_core::pcs::pack_witness(&z, m);
        let (commitment, prover_data) = pcs::commit(&z_packed, &setup.pcs_params);

        // Three points: two plane sub-cubes at random slot/record
        // coordinates (the discharge shape) and one fully random point.
        let record_vars = m - slots::K_LOG;
        let plane = slots::plane_of(slots::gate_position(0));
        let mut plane_point: Vec<F128> = (0..record_vars as u64)
            .map(|i| small(i * 913 + 7))
            .collect();
        for j in 0..7 {
            plane_point.push(small(((plane >> (6 - j)) & 1) as u64));
        }
        plane_point.extend((0..10u64).map(|i| small(i * 331 + 11)));

        let plane2 = slots::plane_of(slots::centering_position(0));
        let mut plane_point2 = plane_point.clone();
        for (j, coord) in plane_point2[record_vars..record_vars + 7]
            .iter_mut()
            .enumerate()
        {
            *coord = small(((plane2 >> (6 - j)) & 1) as u64);
        }

        let random_point: Vec<F128> = (0..m as u64).map(|i| small(i * 7919 + 3)).collect();

        let points = vec![plane_point, plane_point2, random_point];
        let values: Vec<F128> = points.iter().map(|p| bit_mle(&z, p)).collect();

        let padding = setup.r1cs.padding_spec();
        let mut prover_challenger = FsChallenger::new(b"aerie-record-open");
        let proof = open_multilinear(
            z_packed,
            &prover_data,
            &commitment,
            &points,
            &padding,
            &setup.pcs_params,
            &mut prover_challenger,
        );

        let mut verifier_challenger = FsChallenger::new(b"aerie-record-open");
        verify_multilinear(
            &commitment,
            &points,
            &values,
            &proof,
            &setup.pcs_params,
            &mut verifier_challenger,
        )
        .expect("honest multilinear openings verify");

        // A wrong claimed value must fail the ring-switch claim check.
        let mut wrong = values.clone();
        wrong[1] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-record-open");
        assert!(
            verify_multilinear(
                &commitment,
                &points,
                &wrong,
                &proof,
                &setup.pcs_params,
                &mut fresh,
            )
            .is_err()
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-record scatter: the (record, slot) domain with a transparent
// delta-power record factor, bound to the Z_H region of the SAME witness.
// ---------------------------------------------------------------------------

fn frobenius(base: F128, bits: usize) -> Vec<F128> {
    let mut powers = Vec::with_capacity(bits);
    let mut current = base;
    for _ in 0..bits {
        powers.push(current);
        current = current * current;
    }
    powers
}

/// Per-slot value bits from the committed wires: `a_c = x_c ^ M_c ^ b_c`
/// with `M` from the quotient wires and `b` the borrow prefix parity.
fn slot_residue_bits(z: &[bool], base: usize, slot: usize) -> [bool; 14] {
    let word = |i: usize| z[base + slots::word_position(slot, i)];
    let quotient = |i: usize| z[base + slots::quotient_position(slot, i)];
    let borrow_product = |i: usize| z[base + slots::borrow_position(slot, i)];
    let mut bits = [false; 14];
    let mut borrow = false;
    for (c, bit) in bits.iter_mut().enumerate() {
        let m_c = match c {
            0..=2 => quotient(c),
            12 => quotient(0),
            13 => quotient(0) ^ quotient(1),
            _ => false,
        };
        *bit = word(c) ^ m_c ^ borrow;
        borrow ^= borrow_product(c);
    }
    bits
}

/// The thirteen factor tables over the `(record, slot)` domain, from the
/// full witness: the transparent delta-power record factor, the gate, ten
/// counter factors (per-record prefix parity of the accept flags), and
/// the gamma-combined value.
fn record_factor_tables(
    z: &[bool],
    record_vars: usize,
    beta: F128,
    gamma: F128,
    delta: F128,
) -> Vec<Vec<F128>> {
    let records = 1 << record_vars;
    let domain = records << SLOT_VARS;
    let beta_powers = frobenius(beta, COUNTER_BITS);
    let delta_powers = frobenius(delta, record_vars);
    let gamma_powers: Vec<F128> = {
        let mut powers = Vec::with_capacity(15);
        let mut current = F128::ONE;
        for _ in 0..15 {
            powers.push(current);
            current *= gamma;
        }
        powers
    };

    let mut delta_factor = vec![F128::ONE; domain];
    let mut gate = vec![F128::ZERO; domain];
    let mut counter_factors = vec![vec![F128::ONE; domain]; COUNTER_BITS];
    let mut value = vec![F128::ZERO; domain];
    for record in 0..records {
        let base = record << slots::K_LOG;
        let mut delta_power = F128::ONE;
        for (bit, &power) in delta_powers.iter().enumerate() {
            if (record >> bit) & 1 == 1 {
                delta_power *= power;
            }
        }
        for slot in 0..(1 << SLOT_VARS) {
            let index = (record << SLOT_VARS) | slot;
            delta_factor[index] = delta_power;
            if slot < SLOTS {
                // Counter factors from the materialized counter-AFTER
                // wires: the plane formula everywhere, so the factor
                // tables ARE the verifier's opening reconstruction. The
                // beta^(-1) shift moves to the binding identity.
                for (bit, factor) in counter_factors.iter_mut().enumerate() {
                    if z[base + slots::counter_position(slot, bit)] {
                        factor[index] = beta_powers[bit];
                    }
                }
                if z[base + slots::gate_position(slot)] {
                    gate[index] = F128::ONE;
                }
                let bits = slot_residue_bits(z, base, slot);
                let mut val = F128::ZERO;
                for (c, &bit) in bits.iter().enumerate() {
                    if bit {
                        val += gamma_powers[c];
                    }
                }
                if z[base + slots::centering_position(slot)] {
                    val += gamma_powers[14];
                }
                value[index] = val;
            }
        }
    }

    let mut factors = Vec::with_capacity(3 + COUNTER_BITS);
    factors.push(delta_factor);
    factors.push(gate);
    factors.extend(counter_factors);
    factors.push(value);
    factors
}

/// One derived power coordinate `w / (1 + w)`; the caller accumulates the
/// `(1 + w)` scale. None when `w = 1`.
fn derived_coord(weight: F128, scale: &mut F128) -> Option<F128> {
    let denominator = F128::ONE + weight;
    if denominator == F128::ZERO {
        return None;
    }
    *scale *= denominator;
    Some(weight * denominator.inv())
}

/// The Z_H sub-cube opening point and its public scale: MSB-first over the
/// full witness variables, `[delta-derived record coords, the four fixed
/// top plane bits of the Z region, the thirteen (j, c) power coords]`.
fn zh_power_point(
    record_vars: usize,
    beta: F128,
    gamma: F128,
    delta: F128,
) -> Option<(Vec<F128>, F128)> {
    let mut scale = F128::ONE;
    let mut point = Vec::with_capacity(record_vars + slots::K_LOG);
    // Record coords, MSB first: coord i binds record bit (R - 1 - i).
    let delta_powers = frobenius(delta, record_vars);
    for i in 0..record_vars {
        point.push(derived_coord(
            delta_powers[record_vars - 1 - i],
            &mut scale,
        )?);
    }
    // The Z region's fixed top-four plane bits: Z_BASE >> 3 = 13 = 0b1101.
    let top = slots::Z_BASE >> 3;
    for j in 0..4 {
        point.push(if (top >> (3 - j)) & 1 == 1 {
            F128::ONE
        } else {
            F128::ZERO
        });
    }
    // The thirteen (j, c) index coords, MSB first: index bit t has weight
    // gamma^(2^t) for t < 4 and beta^(2^(t-4)) above.
    let beta_powers = frobenius(beta, 9);
    let gamma_powers = frobenius(gamma, 4);
    for t in (0..13).rev() {
        let weight = if t < 4 {
            gamma_powers[t]
        } else {
            beta_powers[t - 4]
        };
        point.push(derived_coord(weight, &mut scale)?);
    }
    Some((point, scale))
}

/// The complete record-lane proof: the slot R1CS, the scatter binding the
/// Z_H region to the gated slot outputs, its discharge, and ONE batched
/// opening carrying the base `[ab, c]` claims plus every multilinear
/// sub-cube claim.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordProof {
    pub commitment: Commitment,
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub scatter: scatter::ScatterProof,
    /// Claimed values of the multilinear openings, in the fixed order
    /// produced by [`multilinear_points`].
    pub opening_values: Vec<F128>,
    /// `MLE_K(Z_H, r)` over the packed leaves at the fingerprint point:
    /// the theta-weighted sum of the fifteen fingerprint openings. The
    /// aerie tag equation consumes this.
    pub fingerprint_value: F128,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

/// The fixed order of the multilinear opening claims: 33 slot-witness
/// planes at the scatter point, ten counter-source planes at the
/// discharge point, and the Z_H power sub-cube.
fn multilinear_points(
    record_vars: usize,
    rs: &[F128],
    r_fp: &[F128],
    beta: F128,
    gamma: F128,
    delta: F128,
) -> Option<Vec<Vec<F128>>> {
    let plane_point = |plane: usize, point: &[F128]| -> Vec<F128> {
        let (rec, slot) = point.split_at(record_vars);
        let mut full = rec.to_vec();
        for j in 0..7 {
            full.push(if (plane >> (6 - j)) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        full.extend_from_slice(slot);
        full
    };
    let mut points = Vec::with_capacity(44);
    points.push(plane_point(slots::plane_of(slots::gate_position(0)), rs));
    for c in 0..14 {
        points.push(plane_point(slots::plane_of(slots::word_position(0, c)), rs));
    }
    for c in 0..3 {
        points.push(plane_point(
            slots::plane_of(slots::quotient_position(0, c)),
            rs,
        ));
    }
    for i in 0..14 {
        points.push(plane_point(
            slots::plane_of(slots::borrow_position(0, i)),
            rs,
        ));
    }
    points.push(plane_point(
        slots::plane_of(slots::centering_position(0)),
        rs,
    ));
    for bit in 0..COUNTER_BITS {
        points.push(plane_point(
            slots::plane_of(slots::counter_position(0, bit)),
            rs,
        ));
    }
    let (zh, _scale) = zh_power_point(record_vars, beta, gamma, delta)?;
    points.push(zh);
    // The aerie fingerprint: fifteen Z-region sub-cubes at the external
    // leaf point, one per live coordinate plane; their theta-weighted sum
    // is MLE_K(Z_H, r) over the packed leaves.
    let top = slots::Z_BASE >> 3;
    for c in 0..15 {
        let mut point = Vec::with_capacity(record_vars + slots::K_LOG);
        point.extend_from_slice(&r_fp[..record_vars]);
        for j in 0..4 {
            point.push(if (top >> (3 - j)) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        point.extend_from_slice(&r_fp[record_vars..]);
        for j in 0..4 {
            point.push(if (c >> (3 - j)) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        points.push(point);
    }
    Some(points)
}

/// Reconstruct the scatter terminal from the claimed opening values (the
/// shared verifier/prover-consistency logic). `values` follows
/// [`multilinear_points`]' order; returns the product of the thirteen
/// factor MLEs at `rs`.
fn reconstruct_terminal(
    record_vars: usize,
    rs: &[F128],
    values: &[F128],
    beta: F128,
    gamma: F128,
    delta: F128,
) -> F128 {
    // Transparent delta factor: product of per-record-bit affine forms;
    // rs record coord i binds record bit (R - 1 - i).
    let delta_powers = frobenius(delta, record_vars);
    let mut product = F128::ONE;
    for (bit, &power) in delta_powers.iter().enumerate() {
        let coord = rs[record_vars - 1 - bit];
        product *= F128::ONE + coord * (power + F128::ONE);
    }
    let gate = values[0];
    product *= gate;
    let beta_powers = frobenius(beta, COUNTER_BITS);
    for bit in 0..COUNTER_BITS {
        product *= F128::ONE + (beta_powers[bit] + F128::ONE) * values[33 + bit];
    }
    // Value factor from the x/q/borrow/centering openings.
    let x = &values[1..15];
    let q = &values[15..18];
    let g = &values[18..32];
    let u = values[32];
    let mut gamma_power = F128::ONE;
    let mut val = F128::ZERO;
    let mut borrow = F128::ZERO;
    for c in 0..14 {
        let m_c = match c {
            0..=2 => q[c],
            12 => q[0],
            13 => q[0] + q[1],
            _ => F128::ZERO,
        };
        val += gamma_power * (x[c] + m_c + borrow);
        borrow += g[c];
        gamma_power *= gamma;
    }
    val += gamma_power * u;
    product * val
}

/// Prove the record lane for `setup.n_blocks` records: slot R1CS, the
/// scatter binding `Z_H` to the gated outputs, the discharge, and one
/// batched opening for everything.
pub fn prove_record<Ch: Challenger>(
    setup: &SlotSetup,
    blocks: &[[u16; SLOTS]],
    challenger: &mut Ch,
) -> (RecordProof, super::hash_to_point_sponge::LaneArtifacts) {
    let r1cs = &setup.r1cs;
    let record_vars = r1cs.m - slots::K_LOG;
    let trace = std::env::var("RECORD_TRACE").is_ok();
    let mut stage = std::time::Instant::now();
    let lap = |label: &str, stage: &mut std::time::Instant| {
        if trace {
            eprintln!(
                "  [prove_record] {label}: {:7.1} ms",
                stage.elapsed().as_secs_f64() * 1e3
            );
        }
        *stage = std::time::Instant::now();
    };
    let z = setup.generate_witness(blocks);
    lap("witness generation", &mut stage);
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    lap("pack_witness", &mut stage);
    let lig_config = setup
        .pcs_params
        .ligerito_prover_config()
        .expect("Ligerito config");

    let (commitment, prover_data) = pcs::commit(&z_packed, &setup.pcs_params);
    bind_statement(challenger, r1cs, &commitment);
    lap("commit", &mut stage);

    // Base R1CS: zerocheck + lincheck, exactly the generic prover's path.
    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    lap("apply a/b", &mut stage);
    let z_packed_lincheck = lincheck::pack_z_lincheck_from_packed(&z_packed, r1cs.m, r1cs.k_log);
    let padding = r1cs.padding_spec();
    let (zc_proof, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        cast(&a_packed_f128),
        cast(&b_packed_f128),
        cast(&z_packed),
        r1cs.m,
        &padding,
        challenger,
    );
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
    let lc_circuit =
        lincheck::SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &lc_circuit,
        &x_ab,
        challenger,
    );
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };
    let s_hat_v_ab = pcs::ring_switch::s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]);
    lap("zerocheck + lincheck", &mut stage);

    // Scatter challenges, post-commitment and post-R1CS.
    challenger.observe_label(b"aerie-record-scatter-challenges-v0");
    let beta = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let delta = challenger.sample_f128();
    // The aerie fingerprint leaf point. In production the challenger IS
    // the joint aerie transcript (seeded through observe_bytes), so this
    // sampling happens after C_H and the aerie-side commitments.
    challenger.observe_label(b"aerie-record-fingerprint-v0");
    let r_fp = challenger.sample_f128_vec(record_vars + 9);

    let factors = record_factor_tables(&z, record_vars, beta, gamma, delta);
    lap("factor tables", &mut stage);
    let (scatter_proof, rs) = scatter::prove(factors, challenger);
    lap("scatter sumcheck", &mut stage);

    let points = multilinear_points(record_vars, &rs, &r_fp, beta, gamma, delta)
        .expect("derived power coordinates defined");
    let opening_values: Vec<F128> = {
        use rayon::prelude::*;
        points.par_iter().map(|p| bit_mle_fast(&z, p)).collect()
    };
    lap("opening values", &mut stage);
    let mut fingerprint_value = F128::ZERO;
    for c in 0..15 {
        fingerprint_value += F128 { lo: 1 << c, hi: 0 } * opening_values[44 + c];
    }

    // The single batched opening: [ab, c] plus every multilinear claim.
    let mut x_fulls: Vec<Vec<F128>> = vec![
        {
            let mut v = ab.point.x_inner_rest.clone();
            v.extend_from_slice(&ab.point.x_outer);
            v
        },
        {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        },
    ];
    for p in &points {
        let (_low, x_outer) = flock_claim_shape(p);
        x_fulls.push(x_outer);
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pre_ab: Option<&[F128]> = Some(s_hat_v_ab.as_slice());
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let mut precomputed: Vec<Option<&[F128]>> = vec![pre_ab, pre_c];
    precomputed.extend(std::iter::repeat_n(None, points.len()));
    let z_packed_kept = z_packed.clone();
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &x_refs,
        &precomputed,
        &[],
        &padding,
        &lig_config,
        challenger,
    );

    lap("batched open", &mut stage);

    let artifacts = super::hash_to_point_sponge::LaneArtifacts {
        z_packed: z_packed_kept,
        prover_data,
        commitment: commitment.clone(),
    };
    (
        RecordProof {
            commitment,
            zerocheck: zc_proof,
            lincheck: lc_proof,
            scatter: scatter_proof,
            opening_values,
            fingerprint_value,
            pcs_open,
        },
        artifacts,
    )
}

/// Verify a [`RecordProof`]: base R1CS replay, the scatter and its
/// discharge, the Z_H binding identity, and the single batched opening.
pub fn verify_record<Ch: Challenger>(
    setup: &SlotSetup,
    proof: &RecordProof,
    challenger: &mut Ch,
) -> Result<(R1csClaim, Vec<F128>, F128), &'static str> {
    let r1cs = &setup.r1cs;
    let record_vars = r1cs.m - slots::K_LOG;
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        r1cs.csc_lincheck_circuit(),
        challenger,
    )
    .map_err(|_| "base R1CS verification failed")?;

    challenger.observe_label(b"aerie-record-scatter-challenges-v0");
    let beta = challenger.sample_f128();
    let gamma = challenger.sample_f128();
    let delta = challenger.sample_f128();
    challenger.observe_label(b"aerie-record-fingerprint-v0");
    let r_fp = challenger.sample_f128_vec(record_vars + 9);

    let factor_count = 3 + COUNTER_BITS;
    let (rs, terminal) = scatter::verify(
        &proof.scatter,
        record_vars + SLOT_VARS,
        factor_count,
        challenger,
    )?;

    // Claimed opening values, in the fixed order.
    if proof.opening_values.len() != 59 {
        return Err("wrong opening value count");
    }
    let values = &proof.opening_values;
    let mut fingerprint_value = F128::ZERO;
    for c in 0..15 {
        fingerprint_value += F128 { lo: 1 << c, hi: 0 } * values[44 + c];
    }
    if fingerprint_value != proof.fingerprint_value {
        return Err("the fingerprint value does not match its openings");
    }

    // Scatter terminal from the claimed openings.
    if terminal != reconstruct_terminal(record_vars, &rs, values, beta, gamma, delta) {
        return Err("scatter terminal does not match the openings");
    }

    // The Z_H binding identity: the slot side carries beta^(count-after)
    // = beta * beta^(write index) on gated slots, so the scatter claim
    // equals beta times the scaled sub-cube opening of the Z region.
    let (_zh_point, zh_scale) =
        zh_power_point(record_vars, beta, gamma, delta).ok_or("degenerate power point")?;
    if proof.scatter.claim != beta * zh_scale * values[43] {
        return Err("the Z_H power sum does not match the scatter claim");
    }

    // The single batched opening: [ab, c] quirky plus the multilinear set.
    let points = multilinear_points(record_vars, &rs, &r_fp, beta, gamma, delta)
        .ok_or("degenerate power point")?;
    let mut claim_values = vec![ab.value, c.value];
    claim_values.extend_from_slice(values);
    let mut bindings = vec![
        LowBinding::Quirky {
            z_skip: ab.point.z_skip,
        },
        LowBinding::Quirky {
            z_skip: c.point.z_skip,
        },
    ];
    let mut x_fulls: Vec<Vec<F128>> = vec![
        {
            let mut v = ab.point.x_inner_rest.clone();
            v.extend_from_slice(&ab.point.x_outer);
            v
        },
        {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        },
    ];
    for p in &points {
        let (x_low, x_outer) = flock_claim_shape(p);
        bindings.push(LowBinding::Multilinear { x_low });
        x_fulls.push(x_outer);
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_config = setup
        .pcs_params
        .ligerito_verifier_config()
        .expect("verifier config");
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
    .map_err(|_| "batched opening verification failed")?;
    Ok((R1csClaim { ab, c }, r_fp, fingerprint_value))
}
