//! Phase 2 of the element PIOP: the batched lincheck.
//!
//! The zerocheck leaves two claims at `r = (r_row, r_con)` — `Âz(r)` and
//! `B̂z(r)`, once the verifier has stripped the affine constants. The verifier
//! samples `α` and one degree-2 sumcheck over `y = (y_row, y_col)` reduces both
//! to a single witness claim:
//!
//! ```text
//! Σ_y (Â + α·B̂)(r, y) · ẑ(y)  =  va + α·vb
//! ```
//!
//! The weight vector factors, which is what makes it cheap. Because the full
//! system is `I_{2^n_log} ⊗ A_0`, the MLE splits as
//! `Â((x_row,x_con),(y_row,y_col)) = eq(x_row,y_row)·Â_0(x_con,y_col)`, so
//!
//! ```text
//! u[(c << n_log) + j] = eq_row[j] · comb[c],
//! comb[c] = Σ_con eq_con[con] · (A_0 + α·B_0)[con, c]
//! ```
//!
//! `comb` is an `O(nnz)` base-block marginal — the same comb shape as the
//! boolean lincheck's, but tiny (a few entries per gate type). The verifier
//! rebuilds it and evaluates `eq(r_row, r'_row) · (Â_0 + α·B̂_0)(r_con, r'_col)`
//! in `O(2^kappa + nnz)` for its final check.
//!
//! The sumcheck loop itself is the boolean lincheck's calibrated product-sumcheck
//! core, called directly: [`crate::lincheck::sumcheck_round_eval_par`],
//! [`crate::lincheck::sumcheck_bind_both_and_eval_next`] (fold + next-round
//! message in one pass) and [`crate::lincheck::sumcheck_bind_top_in_place_par`].
//! Rounds bind the **top** remaining variable, so the challenge list reversed is
//! the claim point LSB-first — matching the rows-low witness layout.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::ElementTableType;
use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{
    sumcheck_bind_both_and_eval_next, sumcheck_bind_top_in_place_par, sumcheck_round_eval_par,
};
use crate::zerocheck::multilinear::eq_eval;
use crate::zerocheck::univariate_skip::build_eq;

const LABEL: &[u8] = b"flock-element-lc-v0";

/// Round messages plus the output witness claim value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// Per-round `(q(1), q(∞))`, length `n_log + kappa`. Top variable bound
    /// first.
    pub rounds: Vec<(F128, F128)>,
    /// `ẑ(r')` — the second packed-direct claim.
    pub z_eval: F128,
}

/// What a verified lincheck leaves for the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// `r' = (r'_row, r'_col)`, LSB-first (rows low). Length `n_log + kappa`.
    pub r_prime: Vec<F128>,
    pub z_eval: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Wrong number of round messages.
    BadRoundCount { expected: usize, got: usize },
    /// `r` (the zerocheck point) has the wrong length for this statement.
    BadPointLength { expected: usize, got: usize },
    /// The final consistency check
    /// `running == eq(r_row,r'_row)·(Â_0+αB̂_0)(r_con,r'_col) · z_eval` failed.
    SumcheckFinalFailed,
}

/// Prove the batched lincheck.
///
/// `r` is the zerocheck's claim point (LSB-first, length `kappa + n_log`), and
/// `va`, `vb` are the constant-stripped `Âz(r)`, `B̂z(r)` claims. `z` is the
/// committed witness, which the caller keeps for the opening.
pub fn prove<C: Challenger>(
    ty: &ElementTableType,
    z: &[F128],
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    ch: &mut C,
) -> (Proof, Claim) {
    let m_words = ty.kappa() + n_log;
    assert_eq!(r.len(), m_words, "zerocheck point length");
    assert_eq!(z.len(), 1usize << m_words, "witness length");

    ch.observe_label(LABEL);
    let alpha = ch.sample_f128();

    // Rows live in the LOW coordinates of the point, columns in the high ones.
    let (r_row, r_con) = r.split_at(n_log);
    let comb = comb_vector(ty, alpha, &build_eq(r_con));
    let mut u = weight_table(&crate::pcs::ring_switch::build_eq_parallel(r_row), &comb);
    let mut wz = z.to_vec();
    debug_assert_eq!(
        u.iter().zip(&wz).fold(F128::ZERO, |a, (x, y)| a + *x * *y),
        va + alpha * vb,
        "lincheck target must be the honest weighted inner product"
    );

    // Product sumcheck, top variable first. Round 0's message is the only
    // standalone pass; every later message falls out of binding the previous
    // round (see `sumcheck_bind_both_and_eval_next`).
    let mut rounds = Vec::with_capacity(m_words);
    let mut r_rounds = Vec::with_capacity(m_words);
    let (mut e1, mut einf) = sumcheck_round_eval_par(&u, &wz);
    for t in 0..m_words {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let rho = ch.sample_f128();
        rounds.push((e1, einf));
        r_rounds.push(rho);
        if t + 1 < m_words {
            let (n1, ninf) = sumcheck_bind_both_and_eval_next(&mut u, &mut wz, rho);
            e1 = n1;
            einf = ninf;
        } else {
            sumcheck_bind_top_in_place_par(&mut u, rho);
            sumcheck_bind_top_in_place_par(&mut wz, rho);
        }
    }
    debug_assert_eq!(wz.len(), 1);
    let z_eval = wz[0];

    // Top-bit-first binding: round `t` bound bit `m_words − 1 − t`, so reversing
    // gives the point in the LSB-first convention the packed-direct claims use.
    let mut r_prime = r_rounds;
    r_prime.reverse();

    let proof = Proof { rounds, z_eval };
    let claim = Claim { r_prime, z_eval };
    (proof, claim)
}

/// Verify a lincheck proof, walking the challenger in lockstep with [`prove`].
pub fn verify<C: Challenger>(
    ty: &ElementTableType,
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claim, VerifyError> {
    let m_words = ty.kappa() + n_log;
    if r.len() != m_words {
        return Err(VerifyError::BadPointLength {
            expected: m_words,
            got: r.len(),
        });
    }
    if proof.rounds.len() != m_words {
        return Err(VerifyError::BadRoundCount {
            expected: m_words,
            got: proof.rounds.len(),
        });
    }

    ch.observe_label(LABEL);
    let alpha = ch.sample_f128();

    // Replay the product sumcheck. `q(0) = running + q(1)` in char 2, then
    // `q(X) = einf·X² + c1·X + q(0)`. Same chain as `crate::lincheck::verify`.
    let mut running = va + alpha * vb;
    let mut r_rounds = Vec::with_capacity(m_words);
    for &(e1, einf) in &proof.rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let rho = ch.sample_f128();
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * rho * rho + c1 * rho + e0;
        r_rounds.push(rho);
    }
    let mut r_prime = r_rounds;
    r_prime.reverse();

    // Final check: the weight vector's own MLE at `r'`, in O(2^kappa + nnz).
    // `û(r') = eq(r_row, r'_row) · (Â_0 + α·B̂_0)(r_con, r'_col)`, and the second
    // factor is the same `comb` marginal the prover built, evaluated against
    // `eq(r'_col)`.
    let (r_row, r_con) = r.split_at(n_log);
    let (r_prime_row, r_prime_col) = r_prime.split_at(n_log);
    let comb = comb_vector(ty, alpha, &build_eq(r_con));
    let eq_col = build_eq(r_prime_col);
    let base = comb
        .iter()
        .zip(&eq_col)
        .fold(F128::ZERO, |acc, (c, e)| acc + *c * *e);
    let u_at_r_prime = eq_eval(r_row, r_prime_row) * base;
    if running != u_at_r_prime * proof.z_eval {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    Ok(Claim {
        r_prime,
        z_eval: proof.z_eval,
    })
}

/// `comb[c] = Σ_con eq_con[con] · (A_0 + α·B_0)[con, c]` — the eq-weighted
/// column marginal of the base block. `O(nnz(A_0) + nnz(B_0))`; both prover and
/// verifier call this, so there is one definition to disagree with.
fn comb_vector(ty: &ElementTableType, alpha: F128, eq_con: &[F128]) -> Vec<F128> {
    debug_assert_eq!(eq_con.len(), ty.width());
    let mut comb = vec![F128::ZERO; ty.width()];
    for (m, scale) in [(ty.a_0(), F128::ONE), (ty.b_0(), alpha)] {
        for (con, row) in m.rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            let w = scale * eq_con[con];
            for &(c, coeff) in row {
                comb[c] += w * coeff;
            }
        }
    }
    comb
}

/// Materialize `u[(c << n_log) + j] = eq_row[j] · comb[c]` — the factored weight
/// vector over the full word domain, laid out rows-low to match the witness.
fn weight_table(eq_row: &[F128], comb: &[F128]) -> Vec<F128> {
    let rows = eq_row.len();
    // Uninit alloc: the chunked map below writes every slot exactly once before
    // any read.
    let mut u = crate::alloc_uninit_f128_vec(comb.len() * rows);
    u.par_chunks_mut(rows)
        .zip(comb.par_iter())
        .for_each(|(dst, &c)| {
            for (d, &e) in dst.iter_mut().zip(eq_row) {
                *d = c * e;
            }
        });
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::broadcast_add;
    use crate::element_r1cs::tests::{Rng, mixed_gate, mixed_witness, mult_gate, mult_witness};

    /// Direct MLE evaluation at `point`, binding the low variable first.
    fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
        let mut t = table.to_vec();
        for &p in point {
            crate::zerocheck::multilinear::fold_in_place_single(&mut t, p);
        }
        t[0]
    }

    /// `(Âz(r), B̂z(r))` straight from the matrices — the claims the lincheck is
    /// supposed to reduce, computed without any of its machinery.
    fn true_claims(ty: &ElementTableType, z: &[F128], n_log: usize, r: &[F128]) -> (F128, F128) {
        let (az, bz) = ty.apply(z, n_log);
        (mle_eval(&az, r), mle_eval(&bz, r))
    }

    /// Brute-force `Σ_y (Â + α·B̂)(r, y)·ẑ(y)` from the *unfactored* definition:
    /// walk every `(x, y)` pair of the block-diagonal system explicitly. This is
    /// the independent check on the factorization `u = eq_row ⊗ comb` — if the
    /// row/column split or the index convention were wrong, this disagrees.
    fn brute_force_weighted_sum(
        ty: &ElementTableType,
        z: &[F128],
        n_log: usize,
        r: &[F128],
        alpha: F128,
    ) -> F128 {
        let width = ty.width();
        let rows = 1usize << n_log;
        let bits = |v: usize, n: usize| -> Vec<F128> {
            (0..n)
                .map(|i| {
                    if (v >> i) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect()
        };
        let (r_row, r_con) = r.split_at(n_log);
        let mut acc = F128::ZERO;
        // Σ_x eq(r, x) Σ_y M[x, y] z[y], with M = I ⊗ (A_0 + αB_0):
        // x = (x_row, con), y = (x_row, c) — the identity factor forces the rows
        // to agree, which is exactly what `eq(x_row, y_row)` encodes.
        for x_row in 0..rows {
            let eq_x_row = eq_eval(r_row, &bits(x_row, n_log));
            for con in 0..width {
                let eq_x_con = eq_eval(r_con, &bits(con, ty.kappa()));
                let mut inner = F128::ZERO;
                for (m, scale) in [(ty.a_0(), F128::ONE), (ty.b_0(), alpha)] {
                    for &(c, coeff) in &m.rows[con] {
                        inner += scale * coeff * z[(c << n_log) + x_row];
                    }
                }
                acc += eq_x_row * eq_x_con * inner;
            }
        }
        acc
    }

    /// The factored weight table's inner product against `z` must equal the
    /// brute-force block-diagonal sum, and must equal `va + α·vb` for the true
    /// `Âz(r)`, `B̂z(r)`. Both directions of the reduction's premise.
    #[test]
    fn weight_factorization_matches_brute_force() {
        let mut rng = Rng::new(1234);
        for (ty, kappa) in [(mult_gate(2), 2usize), (mixed_gate(&mut rng), 3)] {
            for n_log in [1usize, 2, 4] {
                let m_words = kappa + n_log;
                // Random z — the identity is about the weights, not satisfaction.
                let z: Vec<F128> = (0..1usize << m_words).map(|_| rng.f128()).collect();
                let r: Vec<F128> = (0..m_words).map(|_| rng.f128()).collect();
                let alpha = rng.f128();

                let (r_row, r_con) = r.split_at(n_log);
                let comb = comb_vector(&ty, alpha, &build_eq(r_con));
                let u = weight_table(&build_eq(r_row), &comb);
                let factored = u.iter().zip(&z).fold(F128::ZERO, |a, (x, y)| a + *x * *y);

                assert_eq!(
                    factored,
                    brute_force_weighted_sum(&ty, &z, n_log, &r, alpha),
                    "κ={kappa} n_log={n_log}: factored weights vs brute force"
                );
                let (va, vb) = true_claims(&ty, &z, n_log, &r);
                assert_eq!(
                    factored,
                    va + alpha * vb,
                    "κ={kappa} n_log={n_log}: target vs Âz(r) + α·B̂z(r)"
                );
            }
        }
    }

    /// **Differential test** on random instances: the prover's round messages
    /// must be the honest sumcheck of the true weighted inner product. Replaying
    /// the verifier's chain from the *brute-force* target must land on
    /// `û(r')·ẑ(r')`, and `z_eval` must be `z`'s MLE at `r'`.
    #[test]
    fn round_messages_match_brute_force_on_random_instances() {
        let mut rng = Rng::new(99);
        for (ty, kappa) in [(mult_gate(2), 2usize), (mixed_gate(&mut rng), 3)] {
            for n_log in [1usize, 3, 5] {
                let m_words = kappa + n_log;
                let z: Vec<F128> = (0..1usize << m_words).map(|_| rng.f128()).collect();
                let r: Vec<F128> = (0..m_words).map(|_| rng.f128()).collect();
                let (va, vb) = true_claims(&ty, &z, n_log, &r);

                let mut ch = FsChallenger::new(b"element-lc-diff");
                let (proof, claim) = prove(&ty, &z, n_log, &r, va, vb, &mut ch);

                // Re-derive α as the prover did.
                let mut ch2 = FsChallenger::new(b"element-lc-diff");
                ch2.observe_label(LABEL);
                let alpha = ch2.sample_f128();

                assert_eq!(
                    claim.z_eval,
                    mle_eval(&z, &claim.r_prime),
                    "κ={kappa} n_log={n_log}: z_eval is ẑ(r')"
                );

                let mut running = brute_force_weighted_sum(&ty, &z, n_log, &r, alpha);
                // The challenges in binding order are the reverse of r_prime.
                let bind_order: Vec<F128> = claim.r_prime.iter().rev().copied().collect();
                for (&(e1, einf), &rho) in proof.rounds.iter().zip(&bind_order) {
                    let e0 = running + e1;
                    let c1 = e0 + e1 + einf;
                    running = einf * rho * rho + c1 * rho + e0;
                }
                let (r_row, r_con) = r.split_at(n_log);
                let (rp_row, rp_col) = claim.r_prime.split_at(n_log);
                let comb = comb_vector(&ty, alpha, &build_eq(r_con));
                let eq_col = build_eq(rp_col);
                let base = comb
                    .iter()
                    .zip(&eq_col)
                    .fold(F128::ZERO, |a, (c, e)| a + *c * *e);
                assert_eq!(
                    running,
                    eq_eval(r_row, rp_row) * base * claim.z_eval,
                    "κ={kappa} n_log={n_log}: chain from brute-force target"
                );
            }
        }
    }

    /// Prove → verify roundtrip on satisfying witnesses, several shapes.
    #[test]
    fn prove_verify_roundtrip_honest() {
        let mut rng = Rng::new(555);
        for (n_log, n) in [(1usize, 1usize), (3, 5), (4, 16), (6, 41)] {
            let ty = mult_gate(2);
            let z = mult_witness(&ty, n_log, n, &mut rng);
            let r: Vec<F128> = (0..2 + n_log).map(|_| rng.f128()).collect();
            let (va, vb) = true_claims(&ty, &z, n_log, &r);

            let mut ch_p = FsChallenger::new(b"element-lc-rt");
            let (proof, claim_p) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-lc-rt");
            let claim_v = verify(&ty, n_log, &r, va, vb, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v, "n_log={n_log} n={n}");
        }
    }

    /// The mixed-gate table (free wires, mult, mult-acc, linear pin, padding)
    /// round-trips too, and the honest witness's claims are consistent with the
    /// zerocheck's constant stripping.
    #[test]
    fn prove_verify_roundtrip_mixed_gate() {
        let mut rng = Rng::new(556);
        let ty = mixed_gate(&mut rng);
        let (n_log, n) = (4usize, 13usize);
        let z = mixed_witness(&ty, n_log, n, &mut rng);
        assert!(ty.satisfies(&z, n_log, n));
        // Sanity: the constants really are row-uniform, so `pa − Az` is the
        // broadcast constant vector.
        let (az, _) = ty.apply(&z, n_log);
        let mut pa = az.clone();
        broadcast_add(&mut pa, ty.a_const(), n_log);

        let r: Vec<F128> = (0..ty.kappa() + n_log).map(|_| rng.f128()).collect();
        let (va, vb) = true_claims(&ty, &z, n_log, &r);
        let mut ch_p = FsChallenger::new(b"element-lc-mixed");
        let (proof, claim_p) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-lc-mixed");
        let claim_v = verify(&ty, n_log, &r, va, vb, &proof, &mut ch_v).expect("verify");
        assert_eq!(claim_p, claim_v);
    }

    /// Tamper matrix: every round message, `z_eval`, and the incoming claims
    /// must all be pinned.
    #[test]
    fn verify_rejects_mutations() {
        let mut rng = Rng::new(31);
        let (n_log, n) = (4usize, 11usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let r: Vec<F128> = (0..2 + n_log).map(|_| rng.f128()).collect();
        let (va, vb) = true_claims(&ty, &z, n_log, &r);
        let mut ch_p = FsChallenger::new(b"element-lc-mut");
        let (proof, _) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);

        let mut cases: Vec<(String, Proof)> = Vec::new();
        for i in 0..proof.rounds.len() {
            for which in 0..2 {
                let mut bad = proof.clone();
                if which == 0 {
                    bad.rounds[i].0 += F128::ONE;
                } else {
                    bad.rounds[i].1 += F128::ONE;
                }
                cases.push((format!("round {i} msg {which}"), bad));
            }
        }
        let mut bad = proof.clone();
        bad.z_eval += F128::ONE;
        cases.push(("z_eval".to_string(), bad));
        let mut bad = proof.clone();
        bad.rounds.pop();
        cases.push(("truncated rounds".to_string(), bad));

        for (name, bad) in cases {
            let mut ch = FsChallenger::new(b"element-lc-mut");
            assert!(
                verify(&ty, n_log, &r, va, vb, &bad, &mut ch).is_err(),
                "verify accepted mutation: {name}"
            );
        }

        // Wrong incoming claims: the sumcheck target is wrong from round 0.
        for (name, (bva, bvb)) in [("va", (va + F128::ONE, vb)), ("vb", (va, vb + F128::ONE))] {
            let mut ch = FsChallenger::new(b"element-lc-mut");
            assert!(
                verify(&ty, n_log, &r, bva, bvb, &proof, &mut ch).is_err(),
                "verify accepted wrong claim: {name}"
            );
        }
    }
}
