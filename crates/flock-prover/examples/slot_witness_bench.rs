//! Exact A/B kernel benchmark, not an aggregate-signature benchmark.
use flock_core::field::F128;
use flock_prover::r1cs_hashes::hash_to_point_slots::{
    build_block_witness_with, SlotSetup, K_LOG, MASK_REPS, SLOTS,
};
use rayon::prelude::*;
use std::time::Instant;

fn reference(setup: &SlotSetup, blocks: &[[u16; SLOTS]]) -> Vec<F128> {
    let mut packed = vec![F128::ZERO; 1 << (setup.r1cs.m - 7)];
    packed
        .par_chunks_mut(1 << (K_LOG - 7))
        .enumerate()
        .for_each(|(index, out)| {
            let zero = [0; SLOTS];
            let (block, _) = build_block_witness_with(
                &setup.r1cs.a_0,
                &setup.r1cs.b_0,
                blocks.get(index).unwrap_or(&zero),
            );
            for (word, bits) in out.iter_mut().zip(block.chunks_exact(128)) {
                for (bit, &set) in bits.iter().enumerate() {
                    if set {
                        if bit < 64 {
                            word.lo |= 1 << bit;
                        } else {
                            word.hi |= 1 << (bit - 64);
                        }
                    }
                }
            }
        });
    packed
}

fn main() {
    let records: usize = std::env::args()
        .nth(1)
        .unwrap_or("1024".into())
        .parse()
        .unwrap();
    let setup = SlotSetup::new(records);
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let blocks: Vec<_> = (0..records)
        .map(|_| {
            std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u16
            })
        })
        .collect();
    println!("# synthetic words; exact witness kernel comparison only");
    println!("records\trepetition\treference_ms\tbitsliced_ms\tspeedup");
    for repetition in 0..3 {
        let old = || {
            let t = Instant::now();
            let z = reference(&setup, &blocks);
            (z, t.elapsed())
        };
        let new = || {
            let t = Instant::now();
            let z = setup.generate_packed_witness_with_masks(&blocks, &[[false; 128]; MASK_REPS]);
            (z, t.elapsed())
        };
        let ((expected, a), (actual, b)) = if repetition % 2 == 0 {
            (old(), new())
        } else {
            let b = new();
            (old(), b)
        };
        assert_eq!(actual, expected);
        println!(
            "{records}\t{repetition}\t{:.3}\t{:.3}\t{:.3}",
            a.as_secs_f64() * 1000.,
            b.as_secs_f64() * 1000.,
            a.as_secs_f64() / b.as_secs_f64()
        );
    }
}
