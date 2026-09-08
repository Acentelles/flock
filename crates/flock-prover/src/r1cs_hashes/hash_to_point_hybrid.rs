//! Experimental complete private-salt HashToPoint relation in one compact
//! circuit, one commitment, one zerocheck/lincheck and one Flock opening.
//! The SHAKE/record word link is enforced by circuit copy rows.
mod circuit;
pub mod layout;
use super::{
    hash_to_point_record as record, hash_to_point_scatter as scatter,
    hash_to_point_sponge as sponge,
};
pub use circuit::{Setup, Witness, assemble};
use flock_core::{
    challenger::Challenger,
    field::F128,
    lincheck,
    pcs::{self, Commitment, LowBinding},
    proof::ZClaim,
    zerocheck,
};
use layout::{Fragment, record_fragments, sponge_fragments};

pub struct Prepared {
    pub witness: Witness,
    pub commitment: Commitment,
    pub data: pcs::ProverData,
}
pub fn commit(setup: &Setup, witness: Witness) -> Prepared {
    let (commitment, data) = pcs::commit(&witness.z, &setup.params);
    Prepared {
        witness,
        commitment,
        data,
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Proof {
    pub commitment: Commitment,
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub scatter: scatter::ScatterProof,
    pub sponge_values: Vec<F128>,
    pub record_values: Vec<F128>,
    pub fingerprint: F128,
    pub fragment_values: Vec<F128>,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

pub struct Core {
    pub fast: crate::prover::ProveCore,
    pub scatter: scatter::ScatterProof,
    pub sponge_values: Vec<F128>,
    pub record_values: Vec<F128>,
    pub fingerprint: F128,
    pub r_fp: Vec<F128>,
    points: Vec<Vec<F128>>,
    values: Vec<F128>,
}

fn flatten(points: &[Vec<F128>], relocate: fn(&[F128]) -> Vec<Fragment>) -> Vec<Vec<Fragment>> {
    points.iter().map(|p| relocate(p)).collect()
}
fn evaluations(z: &[F128], groups: &[Vec<Fragment>]) -> (Vec<F128>, Vec<Vec<F128>>, Vec<F128>) {
    let points: Vec<_> = groups.iter().flatten().map(|f| f.point.clone()).collect();
    let values = sponge::gather_eval_many(z, &points);
    let mut i = 0;
    let original = groups
        .iter()
        .map(|group| {
            group.iter().fold(F128::ZERO, |sum, f| {
                let v = values[i];
                i += 1;
                sum + f.weight * v
            })
        })
        .collect();
    (original, points, values)
}

pub fn prove_core<Ch: Challenger>(setup: &Setup, prepared: Prepared, ch: &mut Ch) -> Core {
    setup.bind(ch);
    let Prepared {
        witness: Witness { z, a, b },
        commitment,
        data,
    } = prepared;
    let stripes = lincheck::pack_z_lincheck_from_packed(&z, setup.r1cs.m, layout::K_LOG);
    let fast = crate::prover::prove_fast_core_bound(
        &setup.r1cs,
        z,
        a,
        b,
        stripes,
        setup,
        commitment,
        Some(data),
        ch,
    );
    let sp = sponge::sponge_relation_points(setup.record_vars(), ch);
    let (sponge_values, points, values) =
        evaluations(&fast.z_packed, &flatten(&sp, sponge_fragments));
    let claims = std::cell::RefCell::new((points, values));
    let relation = record::prove_record_relation(
        setup.record_vars(),
        |p| {
            layout::record_position(p % super::hash_to_point_slots::K).is_some_and(|q| {
                layout::bit(
                    &fast.z_packed,
                    (p / super::hash_to_point_slots::K) * layout::K + q,
                )
            })
        },
        |points| {
            let (values, new_points, fragments) =
                evaluations(&fast.z_packed, &flatten(points, record_fragments));
            let mut claims = claims.borrow_mut();
            claims.0.extend(new_points);
            claims.1.extend(fragments);
            values
        },
        ch,
    );
    let (points, values) = claims.into_inner();
    Core {
        fast,
        scatter: relation.scatter,
        sponge_values,
        record_values: relation.values,
        fingerprint: relation.fingerprint,
        r_fp: relation.r_fp,
        points,
        values,
    }
}

fn quirky_suffix(claim: &ZClaim) -> Vec<F128> {
    claim
        .point
        .x_inner_rest
        .iter()
        .chain(&claim.point.x_outer)
        .copied()
        .collect()
}

pub fn open<Ch: Challenger>(setup: &Setup, core: Core, ch: &mut Ch) -> Proof {
    let closed = super::face_closure::close_faces(&core.points, &core.values, ch)
        .expect("hybrid face closure");
    let mut suffix = vec![quirky_suffix(&core.fast.ab), quirky_suffix(&core.fast.c)];
    suffix.extend(closed.iter().map(|c| record::flock_claim_shape(&c.point).1));
    let refs: Vec<_> = suffix.iter().map(Vec::as_slice).collect();
    let mut pre = vec![
        core.fast.s_hat_v_ab.as_deref(),
        Some(core.fast.s_hat_v_c.as_slice()),
    ];
    pre.extend(std::iter::repeat_n(None, closed.len()));
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        core.fast.z_packed,
        core.fast.prover_data.as_ref().unwrap(),
        &core.fast.commitment,
        &refs,
        &pre,
        &[],
        &setup.r1cs.padding_spec(),
        &setup.params.ligerito_prover_config().unwrap(),
        ch,
    );
    Proof {
        commitment: core.fast.commitment,
        zerocheck: core.fast.zc_proof,
        lincheck: core.fast.lc_proof,
        scatter: core.scatter,
        sponge_values: core.sponge_values,
        record_values: core.record_values,
        fingerprint: core.fingerprint,
        fragment_values: core.values,
        pcs_open,
    }
}

pub struct VerifyCore {
    pub r_fp: Vec<F128>,
    ab: ZClaim,
    c: ZClaim,
    points: Vec<Vec<F128>>,
}

fn check_fragments(
    groups: &[Vec<Fragment>],
    values: &[F128],
    originals: &[F128],
) -> Result<Vec<Vec<F128>>, &'static str> {
    if groups.len() != originals.len() || groups.iter().map(Vec::len).sum::<usize>() != values.len()
    {
        return Err("hybrid relocation shape");
    }
    let mut i = 0;
    for (group, &expected) in groups.iter().zip(originals) {
        let got = group.iter().fold(F128::ZERO, |sum, f| {
            let v = values[i];
            i += 1;
            sum + f.weight * v
        });
        if got != expected {
            return Err("hybrid relocation identity");
        }
    }
    Ok(groups.iter().flatten().map(|f| f.point.clone()).collect())
}

pub fn verify_core<Ch: Challenger>(
    setup: &Setup,
    publics: &[sponge::SpongePublic],
    proof: &Proof,
    ch: &mut Ch,
) -> Result<VerifyCore, &'static str> {
    setup.bind(ch);
    let (ab, c) = flock_core::verifier::verify_core(
        &setup.r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        setup,
        ch,
    )
    .map_err(|_| "hybrid binary circuit")?;
    let sp =
        sponge::verify_sponge_relation(setup.record_vars(), publics, &proof.sponge_values, ch)?;
    let (rp, r_fp) = record::verify_record_relation(
        setup.record_vars(),
        &proof.scatter,
        &proof.record_values,
        proof.fingerprint,
        ch,
    )?;
    let groups: Vec<_> = flatten(&sp, sponge_fragments)
        .into_iter()
        .chain(flatten(&rp, record_fragments))
        .collect();
    let originals: Vec<_> = proof
        .sponge_values
        .iter()
        .chain(&proof.record_values)
        .copied()
        .collect();
    let points = check_fragments(&groups, &proof.fragment_values, &originals)?;
    Ok(VerifyCore {
        r_fp,
        ab,
        c,
        points,
    })
}

pub fn verify_open<Ch: Challenger>(
    setup: &Setup,
    proof: &Proof,
    core: VerifyCore,
    ch: &mut Ch,
) -> Result<(), &'static str> {
    let closed = super::face_closure::close_faces(&core.points, &proof.fragment_values, ch)?;
    let mut suffix = vec![quirky_suffix(&core.ab), quirky_suffix(&core.c)];
    let mut values = vec![core.ab.value, core.c.value];
    let mut bindings = vec![
        LowBinding::Quirky {
            z_skip: core.ab.point.z_skip,
        },
        LowBinding::Quirky {
            z_skip: core.c.point.z_skip,
        },
    ];
    for claim in closed {
        let (x_low, x_outer) = record::flock_claim_shape(&claim.point);
        suffix.push(x_outer);
        values.push(claim.value);
        bindings.push(LowBinding::Multilinear { x_low });
    }
    if proof.pcs_open.ring_switches.len() != values.len() {
        return Err("hybrid opening shape");
    }
    let refs: Vec<_> = suffix.iter().map(Vec::as_slice).collect();
    pcs::verify_opening_batch_ligerito_mixed_bound(
        &proof.commitment,
        &values,
        &bindings,
        &refs,
        &[],
        &proof.pcs_open,
        &setup.params.ligerito_verifier_config().unwrap(),
        ch,
    )
    .map_err(|_| "hybrid PCS opening")
}

#[cfg(test)]
mod tests;
