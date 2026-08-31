//! Bench harness for the COMPLETE private-salt HashToPoint relation.
//!
//! Usage: `cargo run --release --example hash_to_point_full_bench [N...]`
//! (record counts, powers of two; default 32 64). One TSV row per count.
//!
//! COVERAGE: this proof establishes the full spec Section 3.2 relation
//! with the salt as PRIVATE witness: the SHAKE256 sponge over
//! `salt || hpk || 0x00 || 0x00 || message` (default two-absorb,
//! nine-squeeze bucket), the candidate-slot relation (accept,
//! decomposition, centering, counter, gate), the stable-compaction
//! scatter, the dense `Z_H` table, the aerie fingerprint opening, and
//! the word linkage binding the slot words to the squeeze stream.
//! What remains OUTSIDE it: the aerie-side dual-commitment consistency
//! against the Akita lane and the composed Section 7 transcript. It is
//! NOT a Falcon aggregate-signature result (no signature relation here).
//! Timings are only comparable within one quiet host session.

use flock_core::challenger::FsChallenger;
use flock_prover::r1cs_hashes::hash_to_point_link::{prove_hash_to_point, verify_hash_to_point};
use flock_prover::r1cs_hashes::hash_to_point_slots::SlotSetup;
use flock_prover::r1cs_hashes::hash_to_point_sponge::{SpongePublic, SpongeRecord, SpongeSetup};

fn main() {
    let mut counts: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("record count"))
        .collect();
    if counts.is_empty() {
        counts = vec![32, 64];
    }
    println!("# aerie private-salt HashToPoint, COMPLETE relation (salt private):");
    println!(
        "# sponge lane + slot relation + compaction scatter + Z_H + fingerprint + word linkage."
    );
    println!(
        "# NOT included: the aerie-side dual-commitment consistency and the Section 7 transcript."
    );
    println!("# NOT a Falcon aggregate-signature result.");
    println!("records\tprove_ms\tverify_ms\tproof_bytes");
    for &n in &counts {
        let sponge_setup = SpongeSetup::new(n);
        let slot_setup = SlotSetup::new(n);
        let inputs: Vec<SpongeRecord> = (0..n)
            .map(|i| SpongeRecord {
                salt: std::array::from_fn(|j| (i * 41 + j * 7 + 1) as u8),
                hpk: std::array::from_fn(|j| (i * 13 + j * 3 + 5) as u8),
                message: (0..33).map(|j| (i + j) as u8).collect(),
            })
            .collect();
        let publics: Vec<SpongePublic> = inputs
            .iter()
            .map(|r| SpongePublic {
                hpk: r.hpk,
                message: r.message.clone(),
            })
            .collect();

        let t = std::time::Instant::now();
        let mut prover = FsChallenger::new(b"aerie-hash-to-point-bench");
        let proof = prove_hash_to_point(&sponge_setup, &slot_setup, &inputs, &mut prover);
        let prove_ms = t.elapsed().as_secs_f64() * 1e3;

        let proof_bytes = bincode::serialize(&proof).expect("serialize").len();

        let t = std::time::Instant::now();
        let mut verifier = FsChallenger::new(b"aerie-hash-to-point-bench");
        verify_hash_to_point(&sponge_setup, &slot_setup, &publics, &proof, &mut verifier)
            .expect("verifies");
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;

        println!("{n}\t{prove_ms:.1}\t{verify_ms:.1}\t{proof_bytes}");
    }
}
