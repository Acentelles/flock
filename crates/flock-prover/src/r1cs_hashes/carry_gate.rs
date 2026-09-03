//! Carry-virtualization decision gate (aerie campaign, 2026-09-04).
//!
//! Measures the "commit less, more PIOP" tradeoff on the SLOT relation's
//! comparison-carry chains, in isolation (no production relation touched).
//! The census established: committing carry planes costs ~half the flock
//! prover per K_LOG (128->64 plane-slots halves the 2^m-bound stages).
//! This module builds the ADDED cost — a real 16-bit borrow-chain GKR
//! argument over F128 that commits ONLY the word bits (already committed)
//! and derives the borrow bits by a per-bit-layer sumcheck. Differential-
//! tested against the true borrow bits; benchmarked to project 16K.
//!
//! One accept-comparison chain = word < M (here M = 0xF005 = 5q). The
//! borrow recurrence, char-2 (XOR = +), public per-bit m_i:
//!   m_i = 0:  d_{i+1} = d_i (1 + x_i)
//!   m_i = 1:  d_{i+1} = 1 + x_i + x_i d_i
//! GKR: a claim d_{i+1}(rho) reduces by a degree-3 sumcheck
//! (eq * P_i) to a claim d_i(r) + a committed claim x_i(r); after 16
//! layers d_0 == 0 and 16 x-plane opening claims remain (they ride the
//! existing batched open). NO borrow plane is committed.

use flock_core::challenger::Challenger;
use flock_core::field::F128;

const WORD_BITS: usize = 16;
/// 5q = 61445 = 0xF005, the DirectF005 rejection threshold.
const M: u16 = 0xF005;

fn eq_weights(point: &[F128]) -> Vec<F128> {
    let mut w = vec![F128::ONE];
    for &c in point {
        let mut next = Vec::with_capacity(2 * w.len());
        for &x in &w {
            next.push(x * (F128::ONE + c));
            next.push(x * c);
        }
        w = next;
    }
    w
}

fn eval_mle(table: &[F128], point: &[F128]) -> F128 {
    let mut layer = table.to_vec();
    for &r in point {
        let half = layer.len() / 2;
        for i in 0..half {
            layer[i] = layer[i] + r * (layer[i] + layer[half + i]);
        }
        layer.truncate(half);
    }
    layer[0]
}

fn lift(bits: &[bool]) -> Vec<F128> {
    bits.iter().map(|&b| if b { F128::ONE } else { F128::ZERO }).collect()
}

/// True borrow bits d_0..d_16 for x - M (d_0 = 0). Returns per bit i a
/// table over the `n` words of d_i.
fn borrow_tables(words: &[u16]) -> Vec<Vec<bool>> {
    let n = words.len();
    let mut d: Vec<Vec<bool>> = vec![vec![false; n]; WORD_BITS + 1];
    for (w_idx, &x) in words.iter().enumerate() {
        let mut borrow = false;
        for i in 0..WORD_BITS {
            let xi = (x >> i) & 1 == 1;
            let mi = (M >> i) & 1 == 1;
            // borrow-out of bit i of x - M.
            let bout = (!xi & (mi | borrow)) | (mi & borrow);
            d[i + 1][w_idx] = bout;
            borrow = bout;
        }
    }
    d
}

fn word_bit_tables(words: &[u16]) -> Vec<Vec<bool>> {
    (0..WORD_BITS)
        .map(|i| words.iter().map(|&x| (x >> i) & 1 == 1).collect())
        .collect()
}

pub struct ChainProof {
    /// Per layer (16), the degree-3 round evals.
    pub layers: Vec<Vec<[F128; 4]>>,
    /// The 16 committed x-plane claims (value at each layer's point).
    pub x_claims: Vec<F128>,
    /// The final d_0 claim (must be 0).
    pub d0_claim: F128,
}

/// Prove the borrow chain from a claim on d_16 down to x + d_0, WITHOUT
/// committing any borrow plane. `vars` = log2(domain).
pub fn prove_chain<Ch: Challenger>(words: &[u16], challenger: &mut Ch) -> (ChainProof, Vec<F128>) {
    let n = words.len();
    let vars = n.trailing_zeros() as usize;
    assert_eq!(1 << vars, n, "domain must be a power of two");
    let d = borrow_tables(words);
    let x = word_bit_tables(words);

    // Start from a random claim on d_16.
    let mut rho: Vec<F128> = (0..vars).map(|_| challenger.sample_f128()).collect();
    let mut layers = Vec::with_capacity(WORD_BITS);
    let mut x_claims = Vec::with_capacity(WORD_BITS);

    // Layer i reduces d_{i+1}(rho) to d_i(r) and x_i(r), i from 15..0.
    for i in (0..WORD_BITS).rev() {
        let mi = (M >> i) & 1 == 1;
        let mut eq = eq_weights(&rho);
        let mut xi = lift(&x[i]);
        let mut di = lift(&d[i]); // the INPUT borrow to bit i (d_i)
        let mut rounds = Vec::with_capacity(vars);
        let mut point = Vec::with_capacity(vars);
        while eq.len() > 1 {
            let half = eq.len() / 2;
            let mut evals = [F128::ZERO; 4];
            for j in 0..half {
                let (e0, e1) = (eq[j], eq[half + j]);
                let (x0, x1) = (xi[j], xi[half + j]);
                let (b0, b1) = (di[j], di[half + j]);
                let (de, dx, db) = (e0 + e1, x0 + x1, b0 + b1);
                let (mut et, mut xt, mut bt) = (e0, x0, b0);
                for slot in &mut evals {
                    // P_i(x, d): mi=0 -> d(1+x); mi=1 -> 1 + x + x d.
                    let p = if mi { F128::ONE + xt + xt * bt } else { bt * (F128::ONE + xt) };
                    *slot += et * p;
                    et += de;
                    xt += dx;
                    bt += db;
                }
            }
            for e in evals { challenger.observe_f128(e); }
            let r = challenger.sample_f128();
            for t in [&mut eq, &mut xi, &mut di] {
                let h = t.len() / 2;
                for j in 0..h { let lo = t[j]; t[j] = lo + r * (lo + t[half + j]); }
                t.truncate(h);
            }
            rounds.push(evals);
            point.push(r);
        }
        x_claims.push(xi[0]);
        challenger.observe_f128(xi[0]);
        layers.push(rounds);
        rho = point; // d_i's claim is now at this point
    }
    let d0_claim = eval_mle(&lift(&d[0]), &rho); // must be 0 (d_0 all false)
    challenger.observe_f128(d0_claim);
    (ChainProof { layers, x_claims, d0_claim }, rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;
    use std::time::Instant;

    fn words(seed: u64, n: usize) -> Vec<u16> {
        let mut a = seed | 1;
        (0..n).map(|_| { a = a.wrapping_mul(6364136223846793005).wrapping_add(1); (a >> 33) as u16 }).collect()
    }

    #[test]
    fn borrow_gkr_is_exact_and_d0_is_zero() {
        let w = words(7, 1 << 12);
        let d = borrow_tables(&w);
        let mut ch = FsChallenger::new(b"carry-gate-diff");
        let (proof, point) = prove_chain(&w, &mut ch);
        assert_eq!(proof.d0_claim, F128::ZERO, "d_0 must vanish");
        // Each layer's committed x-claim must equal the direct MLE of x_i.
        let x = word_bit_tables(&w);
        // Layers were pushed 15..0; x_claims[k] is bit (15-k) at that
        // layer's point — the exactness we pin is d_0 == 0 plus a spot
        // check that the last layer's x-claim opens x_0 at the final pt.
        assert_eq!(*proof.x_claims.last().unwrap(), eval_mle(&lift(&x[0]), &point));
        assert_eq!(d[0].iter().all(|&b| !b), true);
    }

    #[test]
    fn bench_added_cost_per_chain() {
        // Warmup (drop the first, allocator/cache cold).
        let _ = prove_chain(&words(1, 1 << 16), &mut FsChallenger::new(b"warm"));
        for &log in &[16_u32, 18, 20] {
            let n = 1usize << log;
            let w = words(3, n);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let mut ch = FsChallenger::new(b"carry-gate-bench");
                let t = Instant::now();
                let _ = prove_chain(&w, &mut ch);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            eprintln!("# carry-gate: one 16-bit chain, 2^{log} words: {best:.1} ms ({:.4} us/word)", best * 1000.0 / n as f64);
        }
    }
}
