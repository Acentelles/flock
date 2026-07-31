//! The Fiat–Shamir transcript's SHAPE, recorded off real proofs.
//!
//! The recursive verifier replays this transcript inside a fixed-topology
//! circuit, so the circuit is generated from the schedule
//! [`RecordingChallenger`] observes while the *actual* verifier runs. Two
//! obligations follow, and this file is where they are discharged:
//!
//! 1. **The shape must not depend on data.** Same config, different counts and
//!    different witnesses must give the identical op sequence. If it ever does
//!    not, there is a second `sample_distinct_queries` hiding somewhere and no
//!    fixed-topology circuit exists until it is found. The failure names the
//!    op index rather than just asserting.
//!
//! 2. **Prover and verifier must agree.** They share one transcript by
//!    construction, so their recorded shapes must be equal — a free
//!    differential over the whole FS order.
//!
//! The pinned digest is the third guard: any protocol change that moves the FS
//! shape fails here loudly and gets a deliberate re-pin, the same discipline
//! the proof-byte fixtures use.
//!
//! Recording is done against **honest, accepted** proofs on purpose. The
//! verifier early-returns on rejection, so a rejected proof would yield a
//! silently truncated schedule — a circuit constraining a prefix of the
//! transcript would look perfectly healthy. Every case here asserts the verify
//! accepted before touching the shape.

use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::transcript_record::{RecordingChallenger, TranscriptOp, TranscriptShape};
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::PcsParams;
use flock_prover::prover::{self, UnionElementSlotInput};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier;
use std::sync::Arc;

const DOMAIN: &[u8] = b"flock-union-element-v0";

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128::new(self.next_u64(), self.next_u64())
    }
}

/// Same element gate block `union_element.rs` uses: two free wires, a product,
/// a linear pin, zero padding above.
fn gate_block(kappa: usize, w0: F128, w1: F128) -> Arc<ElementTableType> {
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0)
        .free_wire(1)
        .mult(2, 0, 1)
        .linear(3, &[(0, w0), (1, w1)]);
    Arc::new(b.build().expect("gate block is valid"))
}

fn gate_witness(
    ty: &ElementTableType,
    nu: usize,
    n: usize,
    w0: F128,
    w1: F128,
    rng: &mut Rng,
) -> Vec<F128> {
    let at = |c: usize, j: usize| (c << nu) + j;
    let mut z = vec![F128::ZERO; ty.width() << nu];
    for j in 0..n {
        let (a, b) = (rng.f128(), rng.f128());
        z[at(0, j)] = a;
        z[at(1, j)] = b;
        z[at(2, j)] = a * b;
        z[at(3, j)] = w0 * a + w1 * b;
    }
    assert!(ty.satisfies(&z, nu, n), "generated witness must satisfy");
    z
}

fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    }
}

/// What the FS chain would cost if a squeeze's output were **not re-absorbed**.
///
/// The challenge is a deterministic function of the state, so feeding it back
/// adds no information — the state advance that squeezes actually need comes
/// from the squeeze *header*, which is absorbed first. Modelled here rather
/// than in the library because it is a hypothesis about the encoding, not the
/// encoding.
fn inventory_without_reabsorb(shape: &TranscriptShape, domain_len: usize) -> (usize, [usize; 5]) {
    let mut offset = 16 + domain_len.div_ceil(16) * 16;
    let (mut fin_blocks, mut fin_parents, mut xof) = (0usize, 0usize, 0usize);
    let complete = |o: usize| o.saturating_sub(1) / 1024;
    for op in shape.ops() {
        match op {
            TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => {
                offset += 16; // header only; the output does not come back
                fin_blocks += 1;
                fin_parents += complete(offset).count_ones() as usize;
                xof += op.squeezed_bytes().div_ceil(64).saturating_sub(1);
            }
            TranscriptOp::Pow { .. } => {
                fin_blocks += 1;
                fin_parents += complete(offset).count_ones() as usize;
                offset += op.absorbed_bytes(); // the nonce IS a real absorb
            }
            _ => offset += op.absorbed_bytes(),
        }
    }
    let c = complete(offset);
    (
        offset,
        [
            offset.saturating_sub(1) / 64,
            c - c.count_ones() as usize,
            fin_blocks,
            fin_parents,
            xof,
        ],
    )
}

/// Prove + verify an element-only union proof, recording BOTH sides.
/// Panics if the verify rejects, so a truncated shape can never escape.
fn record_element_only(
    nu: usize,
    kappas: &[usize],
    counts: &[usize],
    seed: u64,
) -> (
    flock_core::transcript_record::TranscriptShape,
    flock_core::transcript_record::TranscriptShape,
) {
    let mut rng = Rng::new(seed);
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));

    let tys: Vec<Arc<ElementTableType>> = kappas.iter().map(|&k| gate_block(k, w0, w1)).collect();
    let registry = Registry::new(
        tys.iter().map(|t| TableType::element(t.clone())).collect(),
        nu,
    );
    let slot_tys: Vec<Arc<ElementTableType>> = registry
        .element_types()
        .iter()
        .map(|t| match &t.class {
            flock_core::schedule::TableClass::LargeField(e) => e.clone(),
            _ => unreachable!("element-only registry"),
        })
        .collect();
    let union = UnionInstance::new(&registry, counts.to_vec());
    let pcs_params = union_pcs_params(&union);

    let witnesses: Vec<Vec<F128>> = slot_tys
        .iter()
        .zip(counts)
        .map(|(t, &n)| gate_witness(t, nu, n, w0, w1, &mut rng))
        .collect();
    let element_slots: Vec<UnionElementSlotInput<'_>> = witnesses
        .iter()
        .map(|w| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(w)))
        .collect();

    let mut ch_p = RecordingChallenger::new(FsChallenger::new(DOMAIN));
    let (proof, commitment, _claims_p) = prover::prove_fast_ligerito_jagged_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        element_slots,
        &mut ch_p,
    );

    let mut ch_v = RecordingChallenger::new(FsChallenger::new(DOMAIN));
    verifier::verify_ligerito_jagged_union_mixed_class(
        &union,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| {
        panic!("verify rejected (nu={nu}, kappas={kappas:?}, counts={counts:?}): {e:?} — the recorded shape would be TRUNCATED")
    });

    (ch_p.shape(), ch_v.shape())
}

/// The FS shape is a function of the CONFIG only — not of counts, not of
/// witness values. This is the property the whole fixed-topology circuit rests
/// on, so it is checked across the utilization ladder rather than assumed.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_only_transcript_shape_is_data_independent() {
    let nu = 12;
    let kappas = [3usize];

    // Same config; full, non-power-of-two, single-row and empty utilization,
    // each on a different witness seed.
    let cases: [(usize, u64); 4] = [
        (1 << 12, 0xA11CE_0001),
        (2731, 0xA11CE_0002),
        (1, 0xA11CE_0003),
        (0, 0xA11CE_0004),
    ];

    let mut reference: Option<flock_core::transcript_record::TranscriptShape> = None;
    for (count, seed) in cases {
        let (shape_p, shape_v) = record_element_only(nu, &kappas, &[count], seed);

        // Prover and verifier share one transcript, so their shapes must be
        // equal — a differential over the entire FS order, for free.
        assert_eq!(
            shape_p.first_difference(&shape_v),
            None,
            "prover and verifier transcript shapes diverge at count={count} \
             (op {:?}); FS order is broken",
            shape_p.first_difference(&shape_v)
        );

        match &reference {
            None => reference = Some(shape_v),
            Some(r) => {
                if let Some(i) = r.first_difference(&shape_v) {
                    panic!(
                        "FS shape depends on DATA, not just config: count={count} diverges from \
                         the reference at op {i}\n  reference: {:?}\n  this run:  {:?}\n\
                         A fixed-topology circuit cannot be built until this is resolved.",
                        r.ops().get(i),
                        shape_v.ops().get(i),
                    );
                }
            }
        }
    }
}

/// The recorded shape, pinned. Any protocol change that moves the Fiat–Shamir
/// order or the message sizes lands here first.
///
/// Regenerate deliberately with `TRANSCRIPT_SHAPE_PRINT=1 ... --nocapture`,
/// and record why in the re-pin history below.
///
/// Re-pin history: initial pin (2026-07-31), element-only nu=12 kappa=3.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_only_transcript_shape_is_pinned() {
    const EXPECTED: &str = "2efd5caf0afbc530fc5df1ca2d859c128facddf15d7b99c4f852c0ac40ea1c79";

    let (_, shape) = record_element_only(12, &[3], &[1 << 12], 0xB0DD_1E01);

    // The inventory the FS chain table is sized from. Printed either way: it
    // is the number that matters and it should be visible when it moves.
    println!(
        "element-only nu=12 kappa=3: {} ops | {} absorbed bytes | {} squeezed bytes | \
         {} finalizations | {} squeezes addressed by role",
        shape.len(),
        shape.absorbed_bytes(),
        shape.squeezed_bytes(),
        shape.finalizations(),
        shape.squeeze_roles().len(),
    );

    // The FS chain's actual row inventory, derived from the schedule rather
    // than estimated. `finalize_parents` is the term a flat "one compression
    // per squeeze" model misses: a finalize collapses the chunk stack, so it
    // gets more expensive as the transcript grows.
    let inv = shape.blake3_inventory(DOMAIN.len());
    println!(
        "  BLAKE3 rows: absorb {} | chunk parents {} | finalize blocks {} | \
         finalize parents {} | xof {} = {} total",
        inv.absorb_blocks,
        inv.chunk_parents,
        inv.finalize_blocks,
        inv.finalize_parents,
        inv.xof_blocks,
        inv.total(),
    );
    println!(
        "    (a flat one-per-squeeze model would say {}, missing {} stack merges)",
        inv.total() - inv.finalize_parents,
        inv.finalize_parents,
    );

    let (nb, parts) = inventory_without_reabsorb(&shape, DOMAIN.len());
    let alt: usize = parts.iter().sum();
    println!(
        "  WITHOUT re-absorbing squeezes: {} -> {nb} absorbed bytes; rows \
         absorb {} | chunk parents {} | finalize blocks {} | finalize parents {} | xof {} \
         = {alt} ({:+.0}%)",
        shape.absorbed_bytes(),
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        100.0 * (alt as f64 / inv.total() as f64 - 1.0),
    );

    if std::env::var_os("TRANSCRIPT_SHAPE_PRINT").is_some() {
        println!("const EXPECTED: &str = \"{}\";", shape.digest_hex());
        return;
    }
    assert_eq!(
        shape.digest_hex(),
        EXPECTED,
        "the Fiat-Shamir transcript SHAPE moved. If that was intended, \
         regenerate with TRANSCRIPT_SHAPE_PRINT=1 and record the reason; \
         if not, stop and find out what changed the FS order."
    );
}
