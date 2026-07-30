//! Head-to-head benchmark of the two grand-product permutation checks on the
//! *same* statement and instance:
//!
//!   * `perm`     — committed product tree ([`flock_prover::permutation`]):
//!                  HyperPlonk-style, `2N` PCS oracle, 5-point Ligerito opening,
//!                  one batched degree-2 zerocheck.
//!   * `prod-GKR` — product-circuit GKR ([`flock_prover::product_gkr`]): two
//!                  product trees (LHS/RHS) reduced layer-by-layer, **no
//!                  committed oracle, no PCS, no field inversions** — the proof
//!                  is just two `O(μ²)` sumcheck transcripts.
//!
//! Both prove the multiset equality `{(f, s_id)} = {(g, s_σ)}` for a random
//! permutation `σ` with `f(x) = g(σ⁻¹(x))`, so the comparison isolates the
//! *construction* difference rather than the instance.
//!
//! Trimmed from `perm_vs_logup.rs` on `flock-dev`'s `recursive-verifier` branch
//! (which also carries three LogUp variants not ported here).
//!
//! Run:   `cargo bench --bench perm_vs_gkr`
//! On M-series, requires `.cargo/config.toml` so the `aes` feature is on.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::F128;
use flock_prover::{permutation, product_gkr};

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
        F128::new(self.next_u64(), self.next_u64())
    }
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// Honest instance: random `g`, permutation `σ`, and `f(x) = g(σ⁻¹(x))`.
fn honest_instance(mu: usize, seed: u64) -> (Vec<F128>, Vec<F128>, Vec<usize>) {
    let n = 1usize << mu;
    let mut rng = Rng::new(seed);
    let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let sigma = rng.permutation(n);
    let mut sinv = vec![0usize; n];
    for (x, &sx) in sigma.iter().enumerate() {
        sinv[sx] = x;
    }
    let f: Vec<F128> = (0..n).map(|x| g[sinv[x]]).collect();
    (f, g, sigma)
}

/// Absorb the statement `(f, g, σ)` — the PIOP caller contract for both schemes.
fn bind<C: Challenger>(ch: &mut C, f: &[F128], g: &[F128], sigma: &[usize]) {
    ch.observe_f128_slice(f);
    ch.observe_f128_slice(g);
    for &s in sigma {
        ch.observe_f128(F128::new(s as u64, 0));
    }
}

const SEED: u64 = 0xC0FFEE;
const DOMAIN: &[u8] = b"perm-vs-gkr-bench-v0";

struct Row {
    prove_ms: f64,
    verify_ms: f64,
    proof_kib: f64,
    oracle: &'static str,
}

/// Time one scheme via its `prove`/`verify`/serialize closures. Witness
/// generation and a warm-up run (priming NTT/convert tables and the scratch
/// pool) are hoisted out of the timed section; reports best-of-`n_runs`.
///
/// `mk_ch` builds a challenger with `(f, g, σ)` already absorbed. That
/// absorption is the caller's PIOP contract, not part of either scheme, and it
/// hashes `3·2^μ` field elements — timing it would swamp both schemes' verify
/// (~21 ms of the ~22 ms first measured at μ=20). So the clock starts on an
/// already-bound challenger, matching `perm_vs_logup.rs`.
fn time_scheme<M, P, V, S, Proof>(
    n_runs: usize,
    oracle: &'static str,
    mk_ch: M,
    prove: P,
    verify: V,
    size: S,
) -> Row
where
    M: Fn() -> FsChallenger,
    P: Fn(&mut FsChallenger) -> Proof,
    V: Fn(&Proof, &mut FsChallenger),
    S: Fn(&Proof) -> usize,
{
    let warm = prove(&mut mk_ch());
    verify(&warm, &mut mk_ch());

    let mut best_prove = f64::INFINITY;
    let mut proof = warm;
    for _ in 0..n_runs {
        let mut ch = mk_ch();
        let t0 = Instant::now();
        let p = prove(&mut ch);
        best_prove = best_prove.min(t0.elapsed().as_secs_f64() * 1e3);
        proof = p;
    }

    let mut best_verify = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = mk_ch();
        let t0 = Instant::now();
        verify(black_box(&proof), &mut ch);
        best_verify = best_verify.min(t0.elapsed().as_secs_f64() * 1e3);
    }

    Row {
        prove_ms: best_prove,
        verify_ms: best_verify,
        proof_kib: size(&proof) as f64 / 1024.0,
        oracle,
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    println!("\n  mu  scheme       prove ms  verify ms   proof KiB  oracle");

    for &mu in &[8usize, 10, 12, 14, 16, 18, 20] {
        let (f, g, sigma) = honest_instance(mu, SEED ^ mu as u64);
        let n_runs = if mu >= 16 { 3 } else { 5 };

        let mk_ch = || {
            let mut ch = FsChallenger::new(DOMAIN);
            bind(&mut ch, &f, &g, &sigma);
            ch
        };

        let perm = time_scheme(
            n_runs,
            "2N committed",
            mk_ch,
            |ch| permutation::prove(black_box(&f), black_box(&g), black_box(&sigma), ch).0,
            |p, ch| {
                permutation::verify(mu, p, ch).expect("perm verify");
            },
            |p| bincode::serialize(p).expect("serialize").len(),
        );

        let gkr = time_scheme(
            n_runs,
            "none",
            mk_ch,
            |ch| product_gkr::prove(black_box(&f), black_box(&g), black_box(&sigma), ch).0,
            |p, ch| {
                product_gkr::verify(mu, p, ch).expect("prod-GKR verify");
            },
            |p| bincode::serialize(p).expect("serialize").len(),
        );

        // Both circuits in lockstep under one λ-combined sumcheck per layer:
        // half the rounds, and a SINGLE reduction point ρ for the witness evals
        // (vs `prove`'s separate ρ_lhs / ρ_rhs).
        let gkr_b = time_scheme(
            n_runs,
            "none",
            mk_ch,
            |ch| product_gkr::prove_batched(black_box(&f), black_box(&g), black_box(&sigma), ch).0,
            |p, ch| {
                product_gkr::verify_batched(mu, p, ch).expect("prod-GKR-batched verify");
            },
            |p| bincode::serialize(p).expect("serialize").len(),
        );

        println!(
            "\n{mu:>4}  perm       {:>10.3} {:>10.3} {:>11.2}  {}",
            perm.prove_ms, perm.verify_ms, perm.proof_kib, perm.oracle
        );
        println!(
            "      prod-GKR   {:>10.3} {:>10.3} {:>11.2}  {}",
            gkr.prove_ms, gkr.verify_ms, gkr.proof_kib, gkr.oracle
        );
        println!(
            "      GKR-batch  {:>10.3} {:>10.3} {:>11.2}  {}",
            gkr_b.prove_ms, gkr_b.verify_ms, gkr_b.proof_kib, gkr_b.oracle
        );
        println!(
            "      vs perm    {:>9.2}× {:>9.2}× {:>10.2}×",
            perm.prove_ms / gkr_b.prove_ms,
            perm.verify_ms / gkr_b.verify_ms,
            perm.proof_kib / gkr_b.proof_kib,
        );
    }
}
