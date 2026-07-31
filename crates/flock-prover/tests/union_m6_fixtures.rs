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
//! Re-pin history: integer-lane union commit (the shipped config); the
//! lincheck's `matrix_evals` report (the deferred/accumulated matrix work —
//! this moves the single-table anchors too, since even an empty vector
//! serializes); BLAKE3
//! I/O-region word alignment (M_BASE 513 → 512, out_hi/params words
//! aligned, const pin moved to the end — the circuit/wiring layer's
//! prerequisite). The SHA-256 anchor digest survived the latter unchanged,
//! pinning that the re-layout touched nothing outside BLAKE3.

use ::sha2 as sha2_hash;
use flock_core::proof::{R1csClaim, R1csProofJaggedLigerito};
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::{Commitment, PcsParams};
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
            "3375527896c6f364dbf986a4dda6389f08bfc4a244d20b17c50ab62e67476b42",
        ),
        (
            "mixed-nu10-50-37",
            [50, 37],
            "d08231e2c17732c775d41493bf0fb1215f6dee67dff5d2f29af5dc517b56b315",
        ),
        (
            "mixed-nu10-8-8",
            [8, 8],
            "57340829db5e8d4cf8eeb0e7d425a9566079aa45e31b6f5dfeec8799cfa3ca87",
        ),
        (
            "mixed-nu10-0-64",
            [0, 64],
            "ac9f2de9efc357498a780616d3a9d89766c7549fb61ecae123a22866bf323dca",
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
        const EXPECTED: &str = "3fc71fcc859e41939347b350f4bee3c6c4f5634a14a88b4a9837837b94f2fc4e";
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
        const EXPECTED: &str = "62c5dd92f540dcd900134cfb2c369f373ae9ef4b889be9e3139ea775ba6644b6";
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
