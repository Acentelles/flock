//! Reproducible full binary-relation gate for exact prover optimizations.
//! This has synthetic salt/message inputs and no Falcon signature lane.
//! Optional second argument writes the serialized proof for byte comparison.
//! Build with reference-gather for the pre-optimization evaluation algorithm.

use flock_core::challenger::{Challenger, FsChallenger};
use flock_prover::r1cs_hashes::hash_to_point_link::{prove_hash_to_point, verify_hash_to_point};
use flock_prover::r1cs_hashes::hash_to_point_slots::{MASK_REPS, SlotSetup};
use flock_prover::r1cs_hashes::hash_to_point_sponge::{SpongePublic, SpongeRecord, SpongeSetup};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let n: usize = args
        .first()
        .map_or(32, |n| n.parse().expect("record count"));
    let sponge = SpongeSetup::new(n);
    let slots = SlotSetup::new(n);
    let inputs: Vec<_> = (0..n)
        .map(|i| SpongeRecord {
            salt: std::array::from_fn(|j| (i * 41 + j * 7 + 1) as u8),
            hpk: std::array::from_fn(|j| (i * 13 + j * 3 + 5) as u8),
            message: (0..33).map(|j| (i + j) as u8).collect(),
        })
        .collect();
    let public: Vec<_> = inputs
        .iter()
        .map(|r| SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect();
    let masks: [[bool; 128]; MASK_REPS] =
        std::array::from_fn(|rep| std::array::from_fn(|bit| (n + rep + bit) % 5 == 0));
    let mut prover = FsChallenger::new(b"aerie-packed-gather-full-gate-v0");
    let start = std::time::Instant::now();
    let proof = prove_hash_to_point(&sponge, &slots, &inputs, &masks, &mut prover);
    let prove_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut verifier = FsChallenger::new(b"aerie-packed-gather-full-gate-v0");
    let start = std::time::Instant::now();
    verify_hash_to_point(&sponge, &slots, &public, &proof, &mut verifier).expect("valid proof");
    let verify_ms = start.elapsed().as_secs_f64() * 1e3;
    let bytes = bincode::serialize(&proof).expect("serialize");
    if let Some(path) = args.get(1) {
        std::fs::write(path, &bytes).expect("write proof");
    }
    let tail = prover.sample_f128();
    assert_eq!(tail, verifier.sample_f128(), "prover/verifier transcript");
    println!(
        "# Full binary relation; synthetic inputs; no Falcon signature lane; reference={}, faces={}",
        cfg!(feature = "reference-gather"),
        cfg!(feature = "face-batching")
    );
    println!("records\tprove_ms\tverify_ms\tproof_bytes\tproof_blake3\ttranscript_tail");
    println!(
        "{n}\t{prove_ms:.3}\t{verify_ms:.3}\t{}\t{}\t{:016x}{:016x}",
        bytes.len(),
        blake3::hash(&bytes),
        tail.hi,
        tail.lo
    );
}
