//! Dump a zerocheck **round-1 (univariate-skip URM)** oracle from the real
//! `flock` prover so the CUDA port (`cuda-ghash/test_zerocheck_round1.cu`) can
//! be checked bit-for-bit.
//!
//! Round-1 produces `round1_ab[64]`, `round1_c[64]` (F128) — the values
//! `prove_packed` observes into the challenger. We compute the golden via the
//! REAL optimized path (`round1_shift_reduce_extract_c_packed_padded` × `C_s`)
//! and assert it equals the canonical `round1_naive` (the algorithm the GPU
//! mirrors). The GPU computes round-1 canonically:
//!
//!   eq_full = build_eq(r[6..m])                         # 2^(m-6) F128
//!   for x_rest: A_Λ/B_Λ/C_Λ = extend(skip bits)         # S→Λ, F8-linear map M
//!     p_ab[i] += eq_full[x_rest] * φ8(A_Λ[i]*B_Λ[i]); p_c[i] += eq * φ8(C_Λ[i])
//!
//! The extension `extend = fwd_Λ ∘ inv_S` is F8-linear; for boolean input
//! `A_Λ = ⊕_{s: bit s set} M[:,s]`. We dump `M` (built from `AdditiveNttGf8`)
//! and the F8 mul table so the GPU needs no FFT / F8-poly knowledge.
//!
//! Output: little-endian binary to argv[1] (default zerocheck_round1_vectors.bin):
//!   magic   u32 = 0x5A435231 ("ZCR1")
//!   m, k_skip, k_log, useful_bits : u32 each
//!   r[m]                              {lo,hi} u64 each
//!   M  : 64*64 bytes (column-major: M[s*64 + i] = extend(e_s)[i], F8)
//!   f8mul : 64 bytes (f8mul[x*8 + y] = (F8(x)*F8(y)).0)
//!   a_packed, b_packed, c_packed   (2^m / 8 bytes each, LSB-first)
//!   round1_ab[64] {lo,hi};  round1_c[64] {lo,hi}
//!
//! Run:
//!   cargo run --release --bin dump_zerocheck_round1_vectors -- cuda-ghash/zerocheck_round1_vectors.bin 15

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::{F8, F128};
use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use flock_prover::zerocheck::PaddingSpec;
use flock_prover::zerocheck::univariate_skip::{pack_bits, round1_naive};
use flock_prover::zerocheck::univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

const K_SKIP: usize = 6;
const N_INNER: usize = 7;

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
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
    fn bit(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }
}

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "zerocheck_round1_vectors.bin".to_string());
    let m: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    // Optional padding (default: dense). k_log = block size in bits, useful_bits ≤ 2^k_log.
    let k_log: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(m);
    let useful_bits: usize = env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize << k_log);

    assert!(
        m >= K_SKIP + N_INNER,
        "m must be ≥ {} (k_skip+N_INNER)",
        K_SKIP + N_INNER
    );
    assert!(k_log <= m && useful_bits <= (1usize << k_log));
    let n_total = 1usize << m;

    let mut rng = Rng::new(0x2ECEC0 ^ ((m as u64) << 8));

    // Witness bits, with honest-zero padding when k_log < m.
    let mut a = vec![false; n_total];
    let mut b = vec![false; n_total];
    let mut c = vec![false; n_total];
    let block = 1usize << k_log;
    for i in 0..n_total {
        let within = i & (block - 1);
        if within < useful_bits {
            a[i] = rng.bit();
            b[i] = rng.bit();
            c[i] = rng.bit();
        }
    }
    let a_packed = pack_bits(&a);
    let b_packed = pack_bits(&b);
    let c_packed = pack_bits(&c);

    // r layout: skip (random), small (fixed), medium (fixed), outer (random).
    let mut r = vec![F128::ZERO; m];
    for v in r.iter_mut().take(K_SKIP) {
        *v = rng.f128();
    }
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[K_SKIP + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[K_SKIP + 3 + i] = *val;
    }
    for v in r.iter_mut().take(m).skip(K_SKIP + N_INNER) {
        *v = rng.f128();
    }

    // Golden via the REAL optimized round-1 (+ C_s), and the canonical naive.
    let padding = PaddingSpec {
        k_log,
        useful_bits_per_block: useful_bits,
    };
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let (ab_opt, c_opt) = round1_shift_reduce_extract_c_packed_padded(
        &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table, &padding,
    );
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = c_opt.iter().map(|x| c_s * *x).collect();

    // Sanity: canonical naive must equal the optimized (post-C_s) message.
    let (ab_naive, c_naive) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
    assert_eq!(ab_naive, round1_ab, "round1_naive AB != optimized*C_s");
    assert_eq!(c_naive, round1_c, "round1_naive C != optimized*C_s");

    // Extension matrix M (column-major) and F8 mul table.
    let mut mcol = vec![0u8; 64 * 64];
    for s in 0..64 {
        let mut col = vec![F8::ZERO; 64];
        col[s] = F8(1);
        ntt_s.inverse(&mut col);
        ntt_l.forward(&mut col);
        for i in 0..64 {
            mcol[s * 64 + i] = col[i].0;
        }
    }
    // F8 = GF(2^8): full 256×256 multiply table.
    let mut f8mul = vec![0u8; 256 * 256];
    for x in 0..256usize {
        for y in 0..256usize {
            f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
        }
    }

    let mut w = BufWriter::new(File::create(&path)?);
    write_u32(&mut w, 0x5A43_5231)?; // "ZCR1"
    write_u32(&mut w, m as u32)?;
    write_u32(&mut w, K_SKIP as u32)?;
    write_u32(&mut w, k_log as u32)?;
    write_u32(&mut w, useful_bits as u32)?;
    for &x in &r {
        write_f128(&mut w, x)?;
    }
    w.write_all(&mcol)?;
    w.write_all(&f8mul)?;
    w.write_all(&a_packed)?;
    w.write_all(&b_packed)?;
    w.write_all(&c_packed)?;
    for &x in &round1_ab {
        write_f128(&mut w, x)?;
    }
    for &x in &round1_c {
        write_f128(&mut w, x)?;
    }
    w.flush()?;

    eprintln!(
        "wrote zerocheck round-1 oracle to {path}: m={m} k_skip={K_SKIP} k_log={k_log} \
         useful_bits={useful_bits} rows={} (naive==optimized ✓)",
        1usize << (m - K_SKIP)
    );
    Ok(())
}
