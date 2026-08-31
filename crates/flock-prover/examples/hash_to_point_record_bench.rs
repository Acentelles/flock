//! Bench harness for the aerie private-salt record lane.
//!
//! Usage: `cargo run --release --example hash_to_point_record_bench [N...]`
//! (record counts; default 32 64 128). Emits one TSV row per count.
//!
//! COVERAGE: this proof binds the candidate-slot relation (accept,
//! decomposition, centering, counter, write gate), the stable-compaction
//! scatter, the dense `Z_H` table, and the aerie fingerprint opening. The
//! Keccak sponge lane (that the slot words are the XOF stream of the
//! framed salt input) is NOT included, so this is NOT a Falcon
//! aggregate-signature or full HashToPoint proving result. Timings are
//! only comparable within one quiet host session.

use flock_core::challenger::FsChallenger;
use flock_prover::r1cs_hashes::hash_to_point_record::{prove_record, verify_record};
use flock_prover::r1cs_hashes::hash_to_point_slots::{SLOTS, SlotSetup};

fn main() {
    let mut counts: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("record count"))
        .collect();
    if counts.is_empty() {
        counts = vec![32, 64, 128];
    }
    println!(
        "# aerie private-salt record lane: slot relation + compaction scatter + Z_H + fingerprint."
    );
    println!(
        "# COVERAGE: Keccak sponge lane NOT included; not a Falcon aggregate-signature result."
    );
    println!("records\tslots\tsetup_ms\tprove_ms\tverify_ms\tproof_bytes");
    for &n in &counts {
        let t = std::time::Instant::now();
        let setup = SlotSetup::new(n);
        let setup_ms = t.elapsed().as_secs_f64() * 1e3;

        let blocks: Vec<[u16; SLOTS]> = (0..n)
            .map(|block| {
                let mut words = [0_u16; SLOTS];
                for (slot, word) in words.iter_mut().enumerate() {
                    *word = ((block * SLOTS + slot) as u16)
                        .wrapping_mul(9_973)
                        .wrapping_add(211);
                }
                words
            })
            .collect();

        let t = std::time::Instant::now();
        let mut prover_challenger = FsChallenger::new(b"aerie-record-bench");
        let (proof, _artifacts) = prove_record(&setup, &blocks, &mut prover_challenger);
        let prove_ms = t.elapsed().as_secs_f64() * 1e3;

        let proof_bytes = bincode::serialize(&proof).expect("serialize").len();

        let t = std::time::Instant::now();
        let mut verifier_challenger = FsChallenger::new(b"aerie-record-bench");
        verify_record(&setup, &proof, &mut verifier_challenger).expect("verifies");
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;

        println!(
            "{n}\t{}\t{setup_ms:.1}\t{prove_ms:.1}\t{verify_ms:.1}\t{proof_bytes}",
            n * SLOTS
        );
    }
}
