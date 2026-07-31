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
//! Ligerito query sampling **with replacement** (2026-07-31): the sampler
//! stopped rejecting repeats and moved to a single batched squeeze per level,
//! so every Fiat–Shamir stream downstream of the first query phase shifts.
//! All six digests moved, which is the correct signature — the change is
//! transcript-only, and the M1/M2 differentials plus every roundtrip and
//! tamper test stayed green across it.
//! **16-byte-aligned Fiat–Shamir framing** (2026-07-31): each op's header is
//! now a fixed 16-byte `[op][kind][0;6][len u64]` and byte payloads are padded
//! to a multiple of 16, so every observed `F128` starts on a 128-bit boundary
//! of the absorbed stream. That is what lets a recursion circuit place values
//! into BLAKE3's `m` words by PURE COPY — under the old 1–2 byte tags, seven
//! values in eight straddled two `m` words and would have needed a byte-shift
//! packing gate and a boolean glue table. Costs ~15% more FS compressions.
//! Domain separation is unchanged (op/kind bytes and the length still bind).
//! Every digest moved, as a framing change must.
//! **Squeezed output is no longer re-absorbed** (2026-07-31): the challenge is
//! `XOF(state)`, so feeding it back added no entropy and no binding the state
//! did not already carry — later observations bind to it because they are
//! absorbed into the very state that produced it. Halves the absorbed stream
//! and takes the FS chain 698 -> 515 rows. See circuit-wiring-design.tex
//! §"Re-absorbing squeezed output" for the argument.

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
            "c21d5ccebfb055edfa233b2b1e0c05b7935a43d4516c4cdc580943ce8ee2f019",
        ),
        (
            "mixed-nu10-50-37",
            [50, 37],
            "2ef70d45be5a060e2921c2f47fae38af5920846ba4eb16f7b0f873b542d059ae",
        ),
        (
            "mixed-nu10-8-8",
            [8, 8],
            "a07a815604ca5a88418b76731a1b74fb32b1fdadb666a879ae0049d9bb3be8ff",
        ),
        (
            "mixed-nu10-0-64",
            [0, 64],
            "0c92ac957c148e584fd704b5cb9148a11653b2b6686a0c6f6a716de964de76d1",
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
        const EXPECTED: &str = "39cec089515ddb592475a5d324d281b8cc3024498d067b0a38918fea12961c4d";
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
        const EXPECTED: &str = "bdec741a3a8f1d7514fba00626d2433d7b90ee4eea775137ba4e446e6a6cc62e";
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
