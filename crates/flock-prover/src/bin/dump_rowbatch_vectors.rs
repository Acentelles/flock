//! Dump a row-batch fold oracle (interleaved codeword + lane challenges +
//! collapsed output) from the *real* `flock` field, so the CUDA port
//! (`cuda-ghash/test_rowbatch_fold.cu`) can be checked bit-for-bit against it.
//!
//! Step 2 of the GPU `pcs::open` (Ligerito) port (`cuda-ghash/GPU_OPEN_PLAN.md`):
//! the row-batch fold in `src/pcs/basefold.rs::row_batch_fold_all` (= the fused
//! single-pass form of `row_batch_fold_one`, basefold.rs:268). It collapses each
//! codeword position's `num_ntts = 2^log_batch_size` contiguous lanes
//! (SoA layout `codeword[pos*num_ntts + lane]`) to one F128 via `log_batch_size`
//! rounds of `buf[j] = u + r·(u + v)` (u = buf[2j], v = buf[2j+1]) — note: NO
//! twiddle, unlike the FRI fold. `row_batch_fold_*` are private to `basefold`,
//! so we replicate the spec here; the only nontrivial op (`F128` multiply) is
//! real flock code.
//!
//! Output: little-endian binary to argv[1] (default rowbatch_vectors.bin):
//!   magic          u32 = 0x52424631 ("RBF1")
//!   k_code         u32   (n_positions = 2^k_code)
//!   log_batch_size u32
//!   num_ntts       u32   (= 2^log_batch_size, lanes per position)
//!   cw_len         u32   (= n_positions * num_ntts)
//!   cw_len   * { lo, hi } : u64 each   — interleaved input codeword
//!   log_batch_size * { lo, hi } : u64  — lane challenges r_0..r_{k-1} (round order)
//!   out_len        u32   (= n_positions)
//!   out_len  * { lo, hi } : u64 each   — collapsed (one value per position)
//!
//! Run:
//!   cargo run --release --bin dump_rowbatch_vectors -- cuda-ghash/rowbatch_vectors.bin 12 5
//!   cargo run --release --bin dump_rowbatch_vectors -- out.bin 18 5

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;

/// SplitMix64 — same constants as the other `dump_*_vectors` bins.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

/// One position's lanes folded to a single F128 — verbatim port of the private
/// `basefold::row_batch_fold_one`: `buf[j] = u + r·(u + v)` per round.
fn row_batch_fold_one(lanes: &[F128], challenges: &[F128]) -> F128 {
    let mut buf = lanes.to_vec();
    for &r in challenges {
        let half = buf.len() / 2;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let u = buf[2 * j];
            let v = buf[2 * j + 1];
            next.push(u + r * (u + v));
        }
        buf = next;
    }
    debug_assert_eq!(buf.len(), 1);
    buf[0]
}

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "rowbatch_vectors.bin".to_string());
    let k_code: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let log_batch_size: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let num_ntts = 1usize << log_batch_size;
    let n_positions = 1usize << k_code;
    let cw_len = n_positions * num_ntts;

    let mut rng = Rng::new(0xC0FFEE);
    let codeword: Vec<F128> = (0..cw_len).map(|_| rng.next_f128()).collect();
    let challenges: Vec<F128> = (0..log_batch_size).map(|_| rng.next_f128()).collect();

    // Collapse each position's lanes (SoA: contiguous run of num_ntts).
    let out: Vec<F128> = (0..n_positions)
        .map(|p| row_batch_fold_one(&codeword[p * num_ntts..(p + 1) * num_ntts], &challenges))
        .collect();

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x5242_4631u32.to_le_bytes())?; // "RBF1"
    w.write_all(&(k_code as u32).to_le_bytes())?;
    w.write_all(&(log_batch_size as u32).to_le_bytes())?;
    w.write_all(&(num_ntts as u32).to_le_bytes())?;
    w.write_all(&(cw_len as u32).to_le_bytes())?;
    for &x in &codeword {
        write_f128(&mut w, x)?;
    }
    for &x in &challenges {
        write_f128(&mut w, x)?;
    }
    w.write_all(&(n_positions as u32).to_le_bytes())?;
    for &x in &out {
        write_f128(&mut w, x)?;
    }
    w.flush()?;
    eprintln!(
        "wrote row-batch oracle to {path}: k_code={k_code} log_batch_size={log_batch_size} \
         num_ntts={num_ntts} n_positions={n_positions} cw_len={cw_len}"
    );
    Ok(())
}
