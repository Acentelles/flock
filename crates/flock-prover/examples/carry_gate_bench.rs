//! Verified borrow-chain cost gate. No PCS or Falcon signature lane is timed.
//! Usage: carry_gate_bench [log_words ...], default 12 16.
use flock_core::challenger::{Challenger, FsChallenger};
use flock_prover::r1cs_hashes::carry_gate::{prove_chain_with, verify_chain};

fn main() {
    let mut logs: Vec<u32> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("log words"))
        .collect();
    if logs.is_empty() {
        logs = vec![12, 16];
    }
    println!("# Verified carry reduction only; input PCS claims remain to be opened");
    println!("log_words\treference_ms\tpacked_ms\tspeedup\tpacked_us_per_word");
    for log in logs {
        let n = 1_usize << log;
        let mut seed = 3_u64;
        let words: Vec<_> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 33) as u16
            })
            .collect();
        let mut reference = None;
        let mut reference_tail = None;
        let mut times = [0.0; 2];
        for (k, packed) in [false, true].into_iter().enumerate() {
            let mut transcript = FsChallenger::new(b"carry-bench-v1");
            let start = std::time::Instant::now();
            let (proof, _) = prove_chain_with(&words, packed, &mut transcript);
            times[k] = start.elapsed().as_secs_f64() * 1e3;
            let mut verifier = FsChallenger::new(b"carry-bench-v1");
            verify_chain(&proof, log as usize, &mut verifier).expect("valid reduction");
            let tail = transcript.sample_f128();
            assert_eq!(tail, verifier.sample_f128());
            if let Some(expected) = &reference {
                assert_eq!(&proof, expected);
                assert_eq!(Some(tail), reference_tail);
            } else {
                reference = Some(proof);
                reference_tail = Some(tail);
            }
        }
        println!(
            "{log}\t{:.3}\t{:.3}\t{:.2}\t{:.4}",
            times[0],
            times[1],
            times[0] / times[1],
            times[1] * 1000.0 / n as f64
        );
    }
}
