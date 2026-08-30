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
use flock_core::pcs::{self, Commitment, LowBinding, PcsParams};
use flock_core::zerocheck::PaddingSpec;

use super::hash_to_point_slots as slots;
use super::hash_to_point_slots::{SLOTS, SlotSetup};

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
