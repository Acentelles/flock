//! Experimental complete private-salt HashToPoint relation in one compact
//! circuit, one commitment, one zerocheck/lincheck and one Flock opening.
//! The SHAKE/record word link is enforced by circuit copy rows.
#[cfg(all(test, feature = "aligned-hybrid"))]
mod aligned_tests;
mod circuit;
mod direct;
pub mod layout;
use super::{
    hash_to_point_record as record, hash_to_point_scatter as scatter,
    hash_to_point_sponge as sponge,
};
pub use circuit::{Setup, Witness, assemble};
pub use direct::{CompactSpongeWitness, assemble_direct, compact_sponge_witness};
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
    let _span = tracing::info_span!("hybrid.commit").entered();
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
    let _span = tracing::info_span!("hybrid.fragment_evaluations", claims = points.len()).entered();
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
    prove_core_with_packer(setup, prepared, ch, lincheck::pack_z_lincheck_from_packed)
}

/// The identical core proof with packed extraction of the record scatter rows.
/// This only changes witness access; the scalar extraction remains available
/// through `prove_core` for a runtime-controlled comparison.
pub fn prove_core_packed_record<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    ch: &mut Ch,
) -> Core {
    prove_core_with_packer_and_record(
        setup,
        prepared,
        ch,
        lincheck::pack_z_lincheck_from_packed,
        true,
    )
}

fn prove_core_with_packer<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    ch: &mut Ch,
    pack: impl FnOnce(&[F128], usize, usize) -> Vec<u8>,
) -> Core {
    prove_core_with_packer_and_record(setup, prepared, ch, pack, false)
}

fn prove_core_with_packer_and_record<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    ch: &mut Ch,
    pack: impl FnOnce(&[F128], usize, usize) -> Vec<u8>,
    packed_record: bool,
) -> Core {
    prove_core_with_options(
        setup,
        prepared,
        ch,
        pack,
        packed_record,
        flock_core::zerocheck::ProverOptions::default(),
        |_| {},
    )
}

/// Preserve the complete hybrid proof while selecting zerocheck arithmetic.
pub fn prove_core_with_zerocheck_options<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    packed_record: bool,
    options: flock_core::zerocheck::ProverOptions,
    ch: &mut Ch,
) -> Core {
    prove_core_with_options(
        setup,
        prepared,
        ch,
        lincheck::pack_z_lincheck_from_packed,
        packed_record,
        options,
        |_| {},
    )
}

/// Same core and transcript, with a read-only callback at the sampled content point.
/// Callers may start deterministic preparation; proof challenges still require
/// the completed core transcript returned through `ch`.
pub fn prove_core_with_point_hook<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    packed_record: bool,
    options: flock_core::zerocheck::ProverOptions,
    ch: &mut Ch,
    on_point: impl FnOnce(&[F128]),
) -> Core {
    prove_core_with_options(
        setup,
        prepared,
        ch,
        lincheck::pack_z_lincheck_from_packed,
        packed_record,
        options,
        on_point,
    )
}

fn prove_core_with_options<Ch: Challenger>(
    setup: &Setup,
    prepared: Prepared,
    ch: &mut Ch,
    pack: impl FnOnce(&[F128], usize, usize) -> Vec<u8>,
    packed_record: bool,
    options: flock_core::zerocheck::ProverOptions,
    on_point: impl FnOnce(&[F128]),
) -> Core {
    setup.bind(ch);
    let Prepared {
        witness: Witness { z, a, b },
        commitment,
        data,
    } = prepared;
    let stripe_span = tracing::info_span!("hybrid.lincheck_stripes").entered();
    let stripes = pack(&z, setup.r1cs.m, layout::K_LOG);
    drop(stripe_span);
    let fast_span = tracing::info_span!("hybrid.zerocheck_lincheck").entered();
    let fast = crate::prover::prove_fast_core_bound_with_zerocheck_options(
        &setup.r1cs,
        z,
        a,
        b,
        stripes,
        setup,
        commitment,
        Some(data),
        options,
        ch,
    );
    drop(fast_span);
    let sp = sponge::sponge_relation_points(setup.record_vars(), ch);
    let sponge_span = tracing::info_span!("hybrid.sponge_relation").entered();
    let (sponge_values, points, values) =
        evaluations(&fast.z_packed, &flatten(&sp, sponge_fragments));
    drop(sponge_span);
    let claims = std::cell::RefCell::new((points, values));
    let record_span = tracing::info_span!("hybrid.record_relation").entered();
    let evaluate = |points: &[Vec<F128>]| {
        let (values, new_points, fragments) =
            evaluations(&fast.z_packed, &flatten(points, record_fragments));
        let mut claims = claims.borrow_mut();
        claims.0.extend(new_points);
        claims.1.extend(fragments);
        values
    };
    let relation = if packed_record {
        record::prove_record_relation_packed_with_point_hook(
            setup.record_vars(),
            |p| {
                debug_assert_eq!(p % 64, 0);
                layout::record_position(p % super::hash_to_point_slots::K).map_or(0, |q| {
                    debug_assert_eq!(q % 64, 0);
                    let bit = (p / super::hash_to_point_slots::K) * layout::K + q;
                    let word = fast.z_packed[bit / 128];
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
                layout::record_position(p % super::hash_to_point_slots::K).is_some_and(|q| {
                    layout::bit(
                        &fast.z_packed,
                        (p / super::hash_to_point_slots::K) * layout::K + q,
                    )
                })
            },
            evaluate,
            ch,
            on_point,
        )
    };
    drop(record_span);
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
    let face_span =
        tracing::info_span!("hybrid.face_closure", claims = core.points.len()).entered();
    let closed = super::face_closure::close_faces(&core.points, &core.values, ch)
        .expect("hybrid face closure");
    drop(face_span);
    let _span = tracing::info_span!("hybrid.pcs_open", claims = closed.len() + 2).entered();
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

#[cfg(test)]
mod packed_record_tests;

#[cfg(test)]
mod round1_deferred_tests;
