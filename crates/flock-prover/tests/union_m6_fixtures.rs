//! M6 byte-identity oracle: support-proportional prover passes must produce
//! proofs BYTE-IDENTICAL to the pre-M6 prover at every count vector —
//! support-skipping only changes how sums are computed (the dropped terms are
//! honest zeros), never their values.
//!
//! The fixtures are SHA-256 digests of the full serialized proof bundle
//! (bincode of the proof + commitment root + claim values), captured at the
//! pre-M6 base commit (677d385, branch `multitable`) on deterministic
//! seeded witnesses. Everything downstream of the seeds is deterministic —
//! witness drivers are pure, the challenger is Fiat-Shamir, and all parallel
//! reductions are XOR/add in GF(2^128) (associative + commutative, so the
//! rayon split cannot change a value) — so the digests are stable across
//! runs and thread counts.
//!
//! Covers the mixed union path at full, partial, and zero-count utilization
//! (nu = 10: the M6 measurement geometry) AND the single-type direct jagged
//! anchors (BLAKE3, SHA-256), whose single-run fast paths M6 must not
//! perturb. The M1/M2 harness differentials in `tests/union_roundtrip.rs`
//! remain the live oracle for union-vs-direct plumbing; this file pins the
//! prover's absolute output bytes across M6's fold-skipping changes.
//!
//! Run with `cargo test --release -p flock-prover --test union_m6_fixtures
//! -- --ignored`. To regenerate digests after an INTENTIONAL transcript
//! change (a protocol change, not an M6-style optimization), run with
//! `M6_FIXTURES_PRINT=1 ... --nocapture` and update the constants.
//!
//! Re-pin history: integer-lane union commit (the shipped config); BLAKE3
//! I/O-region word alignment (M_BASE 513 → 512, out_hi/params words
//! aligned, const pin moved to the end — the circuit/wiring layer's
//! prerequisite). The SHA-256 anchor digest survived the latter unchanged,
//! pinning that the re-layout touched nothing outside BLAKE3.

use ::sha2 as sha2_hash;
use flock_core::proof::{R1csClaim, R1csProofJaggedLigerito, R1csProofMergedLigerito};
use flock_prover::challenger::FsChallenger;
use flock_prover::mixed::MixedRegistryId;
use flock_prover::pcs::{Commitment, PcsParams};
use flock_prover::proof_io::MixedProofBundleLigerito;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::{blake3, sha2};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use sha2_hash::Digest as _;

const DOMAIN: &[u8] = b"flock-m6-fixture-v0";

/// SplitMix64 PRNG, deterministic.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<blake3::Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

fn random_sha2_inputs(rng: &mut Rng, n: usize) -> Vec<sha2::Compression> {
    (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.next_u32()),
                std::array::from_fn(|_| rng.next_u32()),
            )
        })
        .collect()
}

/// SHA-256 over the full proof bundle: bincode(proof) ‖ commitment root ‖
/// the two claim values (the claim points are transcript-determined; the
/// values pin the lincheck/zerocheck outputs explicitly).
fn bundle_digest(
    proof: &R1csProofJaggedLigerito,
    commitment: &Commitment,
    claim: &R1csClaim,
) -> String {
    let mut h = sha2_hash::Sha256::new();
    h.update(bincode::serialize(proof).expect("proof serializes"));
    h.update(commitment.root);
    for v in [claim.ab.value, claim.c.value] {
        h.update(v.lo.to_le_bytes());
        h.update(v.hi.to_le_bytes());
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn check(label: &str, expected: &str, got: String) {
    if std::env::var_os("M6_FIXTURES_PRINT").is_some() {
        println!("(\"{label}\", \"{got}\"),");
        return;
    }
    assert_eq!(
        got, expected,
        "M6 byte-identity broken for fixture `{label}`: the prover's output \
         bytes diverged from the pre-M6 base commit"
    );
}

/// Mixed BLAKE3+SHA-256 union proofs at nu = 10 across the utilization
/// ladder: full (1024, 1024), partial non-powers-of-two (50, 37), the M6
/// low-utilization gate point (8, 8), and a zero count for one type (0, 64).
/// Counts are in slot order (SHA-256 first — capacity area descending).
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_mixed_union_proof_bytes_pinned() {
    const FIXTURES: [(&str, [usize; 2], &str); 4] = [
        (
            "mixed-nu10-1024-1024",
            [1024, 1024],
            "59b47c7b7010f8cb11a23cded29c8d7fabadb882a0b8eeaa681fd27b87d0d574",
        ),
        (
            "mixed-nu10-50-37",
            [50, 37],
            "fa2dbfb4e458028a3b72f1468aababc613ddaa5713821f3755408b1fb34b6e91",
        ),
        (
            "mixed-nu10-8-8",
            [8, 8],
            "d2ef6a3708f0efa26f1be9c530cf2a5bce49773d72a63f3a50261c224314443e",
        ),
        (
            "mixed-nu10-0-64",
            [0, 64],
            "41196de9b7e2b0651cdc69761357c68914213e8913254286316de4740bff41e6",
        ),
    ];

    let nu = 10usize;
    let sha2_r1cs = sha2::build_block_r1cs(nu);
    let blake3_r1cs = blake3::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&blake3_r1cs),
            TableType::from_block_r1cs(&sha2_r1cs),
        ],
        nu,
    );
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();

    for (label, counts, expected) in FIXTURES {
        let [n_sha2, n_blake3] = counts;
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: flock_core::pcs::ligerito::LigeritoProfile::Fast,
            // The shipped union configuration: the integer-lane commit skips
            // the whole zero lanes the power-of-two rounding of the dense
            // stack leaves behind (`UnionInstance::commit_lanes`). Pinning it
            // here keeps the fixtures a regression test for what actually
            // ships. The single-type anchors below stay on the power-of-two
            // path (identity compaction fills every lane), so they pin that
            // it is untouched.
            num_lanes: union.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        // Per-fixture seed so each count vector has its own witness stream.
        let mut rng = Rng::new(0x4D36_0000 ^ ((n_sha2 as u64) << 16) ^ n_blake3 as u64);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);

        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&sha2_inputs, nu),
                s2_circuit,
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(&blake3_inputs, nu),
                b3_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch);
        check(label, expected, bundle_digest(&proof, &commitment, &claim));
    }
}

/// Single-type anchors through the direct jagged path (full utilization,
/// single-run PaddingSpec): the fast single-run kernels M6 must not perturb.
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_single_type_anchor_proof_bytes_pinned() {
    // BLAKE3, 256 blocks (m = 22).
    {
        const EXPECTED: &str = "70ddf94f00ff6979bed209a7d8f83f9365d2ed7c1a71b24d2b444ce1afed5bc8";
        let n_blocks = 256usize;
        let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_B3B3);
        let inputs = random_blake3_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let mut ch = FsChallenger::new(DOMAIN);
        let (z, a, b, stripe) = blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
        let (proof, commitment, claim) = prover::prove_fast_ligerito_jagged_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            stripe,
            circuit,
            None,
            &mut ch,
        );
        check(
            "anchor-blake3-m22",
            EXPECTED,
            bundle_digest(&proof, &commitment, &claim),
        );
    }

    // SHA-256, 128 blocks (m = 22).
    {
        const EXPECTED: &str = "59ee6e30868277816735b9fe048deefc1cde752c84f7823a52d2031a6261c175";
        let n_blocks = 128usize;
        let setup = sha2::Sha256HybridSetup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_5252);
        let inputs = random_sha2_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let mut ch = FsChallenger::new(DOMAIN);
        let (z, a, b, stripe) = sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log());
        let (proof, commitment, claim) = prover::prove_fast_ligerito_jagged_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            stripe,
            circuit,
            None,
            &mut ch,
        );
        check(
            "anchor-sha2-m22",
            EXPECTED,
            bundle_digest(&proof, &commitment, &claim),
        );
    }
}

// ---------------------------------------------------------------------------
// MERGED-transport pins. The merged transport is what ships (wire v6), and
// until these fixtures it had NO absolute byte anchors — every pin above is
// jagged. Minted ahead of the jagged transport's removal so the shipped
// transcript is pinned before the differential oracle disappears.
// ---------------------------------------------------------------------------

/// SHA-256 over the merged proof bundle: bincode(proof) ‖ commitment root ‖
/// the two claim values — the merged mirror of [`bundle_digest`].
fn merged_bundle_digest(
    proof: &R1csProofMergedLigerito,
    commitment: &Commitment,
    claim: &R1csClaim,
) -> String {
    let mut h = sha2_hash::Sha256::new();
    h.update(bincode::serialize(proof).expect("proof serializes"));
    h.update(commitment.root);
    for v in [claim.ab.value, claim.c.value] {
        h.update(v.lo.to_le_bytes());
        h.update(v.hi.to_le_bytes());
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHIPPED mixed transcript, pinned at the WIRE encoding: the digest is
/// over `MixedProofBundleLigerito::to_bytes()` — magic, version, flavor,
/// registry id, counts vector, commitment, and the merged proof — plus the
/// claim values. The registry here (BLAKE3+SHA-256 at ν = 10) IS the
/// `Blake3Sha2Nu10` tier, so this pins exactly what `proof_io` puts on disk
/// for a v6 mixed proof. Same count ladder and witness streams as the
/// jagged fixture above, so the two pin the same statements.
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_merged_union_proof_bytes_pinned() {
    const FIXTURES: [(&str, [usize; 2], &str); 4] = [
        (
            "merged-nu10-1024-1024",
            [1024, 1024],
            "8433c864ee8652865ebd0c8515c35b92f9181593ca1a90b8fbdfd9eb2594a934",
        ),
        (
            "merged-nu10-50-37",
            [50, 37],
            "01d9649bdb1089ac7b37a37fa2551ae443d4222c083a9e8b0c4292584d4bc71b",
        ),
        (
            "merged-nu10-8-8",
            [8, 8],
            "6d29992db7b22abcad78a59eedd5963adaa5bdfe876e79fb480d44b11972771c",
        ),
        (
            "merged-nu10-0-64",
            [0, 64],
            "34bbecb1f003eb6da0e3d19b35c325af77a3bed55002b8a310d8456f3940d89b",
        ),
    ];

    let nu = 10usize;
    let sha2_r1cs = sha2::build_block_r1cs(nu);
    let blake3_r1cs = blake3::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&blake3_r1cs),
            TableType::from_block_r1cs(&sha2_r1cs),
        ],
        nu,
    );
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();

    for (label, counts, expected) in FIXTURES {
        let [n_sha2, n_blake3] = counts;
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: flock_core::pcs::ligerito::LigeritoProfile::Fast,
            // The shipped union configuration (integer-lane commit), as in
            // the jagged fixture above.
            num_lanes: union.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        // Same per-fixture seeds as the jagged fixture: same witnesses,
        // same statements — only the transport differs.
        let mut rng = Rng::new(0x4D36_0000 ^ ((n_sha2 as u64) << 16) ^ n_blake3 as u64);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);

        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&sha2_inputs, nu),
                s2_circuit,
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(&blake3_inputs, nu),
                b3_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_jagged_union_merged(&union, &pcs_params, slots, &mut ch);
        let bundle = MixedProofBundleLigerito {
            registry_id: MixedRegistryId::Blake3Sha2Nu10,
            counts: counts.iter().map(|&n| n as u64).collect(),
            commitment: commitment.clone(),
            proof: proof.clone(),
        };
        let mut h = sha2_hash::Sha256::new();
        h.update(bundle.to_bytes());
        for v in [claim.ab.value, claim.c.value] {
            h.update(v.lo.to_le_bytes());
            h.update(v.hi.to_le_bytes());
        }
        let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        check(label, expected, got);
    }
}

/// Single-slot MERGED anchors at full utilization: identity compaction
/// (q IS the padded buffer — no compaction copy), the power-of-two-lane
/// commit path (`num_lanes: None`, which the integer-lane mixed config
/// never exercises), and the full-utilization `generate_witness_batch_major`
/// drivers. These replace the single-table direct-jagged anchors when that
/// path is removed.
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_single_slot_merged_anchor_proof_bytes_pinned() {
    // BLAKE3, 256 blocks (m = 22).
    {
        const EXPECTED: &str = "4c2ec3a5625b4e5c9f6719b86b8d13747501dc036385a5392c3d3e06237f7a01";
        let n_blocks = 256usize;
        let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_B3B3);
        let inputs = random_blake3_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(
            vec![TableType::from_block_r1cs(&setup.r1cs)],
            setup.r1cs.n_log(),
        );
        let union = UnionInstance::new(&registry, vec![n_blocks]);
        assert!(union.compaction_is_identity());
        let slot = UnionSlotProverInput::new(
            blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
            circuit,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) = prover::prove_fast_ligerito_jagged_union_merged(
            &union,
            &setup.pcs_params,
            vec![slot],
            &mut ch,
        );
        check(
            "merged-anchor-blake3-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }

    // SHA-256, 128 blocks (m = 22).
    {
        const EXPECTED: &str = "2aa81599e41efd5b0d8f77ec10b309119d73c30f6df2cb1f0f17184060dbc904";
        let n_blocks = 128usize;
        let setup = sha2::Sha256HybridSetup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_5252);
        let inputs = random_sha2_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(
            vec![TableType::from_block_r1cs(&setup.r1cs)],
            setup.r1cs.n_log(),
        );
        let union = UnionInstance::new(&registry, vec![n_blocks]);
        assert!(union.compaction_is_identity());
        let slot = UnionSlotProverInput::new(
            sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
            circuit,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) = prover::prove_fast_ligerito_jagged_union_merged(
            &union,
            &setup.pcs_params,
            vec![slot],
            &mut ch,
        );
        check(
            "merged-anchor-sha2-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }
}
