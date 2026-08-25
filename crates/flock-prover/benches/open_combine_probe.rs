//! Micro-probe for the pcs open combine sweep (b_combined fold).
//!
//! Times the composed-table fold in isolation at the production shape
//! (L = 2^23, b = 2^15, 2 claims) so sweep variants can be bisected without
//! full proves. ST by default; respects RAYON_NUM_THREADS.

use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::pcs::combine_probe;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let l: usize = 1 << 23;
    let b: usize = 1 << 15;
    let mut rng = Rng(0xBEEF);
    let claims: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
        .map(|_| {
            (
                (0..b).map(|_| rng.f128()).collect(),
                (0..l / b).map(|_| rng.f128()).collect(),
                (0..combine_probe::FOLD_TABLE_LEN).map(|_| rng.f128()).collect(),
            )
        })
        .collect();
    let mut out = vec![F128::ZERO; l];

    for (name, variant) in combine_probe::VARIANTS {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let sink = variant(&claims, &mut out);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            best = best.min(ms);
            std::hint::black_box(sink);
        }
        println!("{name:<40} {best:8.2} ms");
    }
}
