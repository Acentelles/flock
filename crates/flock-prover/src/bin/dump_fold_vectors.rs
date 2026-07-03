//! Dump a FRI-fold oracle (initial codeword + per-round challenge + folded
//! codeword) from the *real* `flock` field + twiddle schedule, so the CUDA
//! port (`cuda-ghash/test_open_fold.cu`) can be checked bit-for-bit against it.
//!
//! This is the step-1 brick of the GPU `pcs::open` (Ligerito) port
//! (`cuda-ghash/GPU_OPEN_PLAN.md`): the FRI fold loop in
//! `src/pcs/basefold.rs::fri_fold_codeword`, which runs `log_dim` rounds, each
//! halving the (single-lane) codeword via `fold_pair` at
//! `layer = k_code - round - 1` (see `basefold.rs:604`). `fold_pair` is private
//! to `basefold`, so we replicate its 3-line spec here — but every nontrivial
//! part (the `F128` multiply and `AdditiveNttF128::twiddle`) is the real flock
//! code, exactly as `dump_commit_vectors` / `host_check_ntt` validate the
//! twiddle schedule.
//!
//! Output: little-endian binary to argv[1] (default fold_vectors.bin):
//!   magic         u32 = 0x464F4C44 ("FOLD")
//!   k_code        u32   (per-lane NTT size in log2; codeword len = 2^k_code)
//!   log_inv_rate  u32
//!   log_dim       u32   (= k_code - log_inv_rate = number of FRI rounds)
//!   init_len      u32   (= 2^k_code)
//!   init_len   * { lo, hi } : u64 each   — initial codeword
//!   for round j in 0..log_dim:
//!     challenge          { lo, hi } : u64
//!     out_len    u32     (= 2^(k_code - j - 1))
//!     out_len  * { lo, hi } : u64 each   — folded codeword after round j
//!
//! Run:
//!   cargo run --release --bin dump_fold_vectors -- cuda-ghash/fold_vectors.bin 12 1
//!   cargo run --release --bin dump_fold_vectors -- out.bin 22 1

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::ntt::AdditiveNttF128;

/// SplitMix64 — same constants as the other `dump_*_vectors` bins, so the
/// codeword and challenges are reproducible and stable across runs.
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

/// Verbatim port of the private `basefold::fold_pair` (DP24):
///   v = v_in + u_in;  u = u_in + v · twiddle;  result = u + r · (u + v)
fn fold_pair(twiddle: F128, u_in: F128, v_in: F128, r: F128) -> F128 {
    let v = v_in + u_in;
    let u = u_in + v * twiddle;
    u + r * (u + v)
}

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "fold_vectors.bin".to_string());
    let k_code: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let log_inv_rate: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(log_inv_rate < k_code, "need log_inv_rate < k_code");
    let log_dim = k_code - log_inv_rate; // number of FRI rounds

    // Same standard-basis NTT the real commit/open build (commit.rs:270,
    // ligerito.rs:2532...) — its twiddle schedule is what `fri_fold_codeword`
    // reads. Already validated bit-for-bit vs the CUDA `build_twiddle_table`.
    let ntt = AdditiveNttF128::standard(k_code);

    let mut rng = Rng::new(0xC0FFEE);
    let mut codeword: Vec<F128> = (0..(1usize << k_code)).map(|_| rng.next_f128()).collect();

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x464F_4C44u32.to_le_bytes())?; // "FOLD"
    w.write_all(&(k_code as u32).to_le_bytes())?;
    w.write_all(&(log_inv_rate as u32).to_le_bytes())?;
    w.write_all(&(log_dim as u32).to_le_bytes())?;
    w.write_all(&(codeword.len() as u32).to_le_bytes())?;
    for &x in &codeword {
        write_f128(&mut w, x)?;
    }

    // FRI rounds, mirroring basefold.rs:604-605 exactly.
    for j in 0..log_dim {
        let r = rng.next_f128();
        let layer = k_code - j - 1;
        let new_len = codeword.len() / 2;
        let mut out = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let u = codeword[2 * i];
            let v = codeword[2 * i + 1];
            let twiddle = ntt.twiddle(layer, i);
            out.push(fold_pair(twiddle, u, v, r));
        }
        write_f128(&mut w, r)?;
        w.write_all(&(new_len as u32).to_le_bytes())?;
        for &x in &out {
            write_f128(&mut w, x)?;
        }
        codeword = out;
    }
    w.flush()?;
    eprintln!(
        "wrote fold oracle to {path}: k_code={k_code} log_inv_rate={log_inv_rate} \
         log_dim={log_dim} rounds, init_len={}",
        1usize << k_code
    );
    Ok(())
}
