//! Phase 1 of the element PIOP: the large-field zerocheck.
//!
//! With `x = [x_row (n_log low bits) | x_con (kappa high bits)]`, the verifier
//! sends `τ` and the prover proves
//!
//! ```text
//! Σ_x eq(τ, x) · ( (Az+a_const)(x) · (Bz+b_const)(x) + z(x) ) = 0
//! ```
//!
//! (char 2, so the relation's `− z` is `+ z`). This is a plain eq-weighted
//! degree-3 sumcheck over `n_log + kappa` rounds — **no univariate skip, no
//! packing, no φ8**. Rounds 2+ of the boolean zerocheck are essentially this
//! protocol, and the round-message/verifier conventions here are deliberately
//! the same so the two verifiers stay structurally parallel:
//!
//! - low bit bound first, so the challenge list *is* the claim point LSB-first;
//! - **Convention A** — the prover sends the bare inner `(G(1), G(∞))` and the
//!   verifier absorbs the current variable's eq factor via the consistency
//!   identity `G_{r-1}(ρ) = (1+τ_r)·G_r(0) + τ_r·G_r(1)`, one inversion per
//!   round (`crate::zerocheck::verify`, zerocheck.rs:835);
//! - the running claim is the inner value, never eq-weighted, so the initial
//!   target is `0` and the final one is `ea·eb + ec`.
//!
//! Output claims at the final point `r = (r_row, r_con)`:
//!
//! - `ea = MLE of (Az+a_const)` and `eb = MLE of (Bz+b_const)` at `r`, which
//!   Phase 2 reduces once the verifier has subtracted the closed-form constants
//!   ([`super::strip_constants`]);
//! - `ec = ẑ(r)`. Because `C = I` this is *directly* a witness evaluation, so it
//!   leaves as a packed-direct claim with no lincheck term.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::univariate_skip::SplitEqGhash;

const LABEL: &[u8] = b"flock-element-zc-v0";

/// The prover's round messages plus the three final evaluations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// Per-round `(G(1), G(∞))` — Convention A, bare (no eq prefactor).
    /// Length `n_log + kappa`.
    pub rounds: Vec<(F128, F128)>,
    /// `(Âz + â_const)(r)`.
    pub ea: F128,
    /// `(B̂z + b̂_const)(r)`.
    pub eb: F128,
    /// `ẑ(r)` — the C-claim.
    pub ec: F128,
}

/// What a verified zerocheck leaves for Phase 2 and the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// `r = (r_row, r_con)`, LSB-first (rows low). Length `n_log + kappa`.
    pub r: Vec<F128>,
    pub ea: F128,
    pub eb: F128,
    pub ec: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Wrong number of round messages.
    BadRoundCount { expected: usize, got: usize },
    /// The final consistency check `running == ea·eb + ec` failed. Any
    /// inconsistency in a round message or in the three final evaluations
    /// propagates here.
    SumcheckFinalFailed,
}

/// Prove the zerocheck for one element table.
///
/// `pa`, `pb` are `(Az + a_const)` and `(Bz + b_const)` over the whole padded
/// domain; both are consumed (folded in place). `z` is the committed witness,
/// cloned into a working table so the caller keeps it for Phase 2 and the
/// opening.
///
/// All three tables are laid out `[(y or c) << n_log | j]`, i.e. rows low, which
/// makes the sumcheck's low-bit-first binding walk the row variables first and
/// the column variables last.
pub fn prove<C: Challenger>(
    pa: Vec<F128>,
    pb: Vec<F128>,
    z: &[F128],
    n_log: usize,
    kappa: usize,
    ch: &mut C,
) -> (Proof, Claim) {
    let m_words = n_log + kappa;
    let n_words = 1usize << m_words;
    assert_eq!(pa.len(), n_words, "pa length");
    assert_eq!(pb.len(), n_words, "pb length");
    assert_eq!(z.len(), n_words, "z length");
    assert!(m_words >= 1, "need at least one variable");

    ch.observe_label(LABEL);
    let tau = ch.sample_f128_vec(m_words);

    let (mut wa, mut wb) = (pa, pb);
    let mut wz = z.to_vec();
    let mut rounds = Vec::with_capacity(m_words);
    let mut r = Vec::with_capacity(m_words);
    for i in 0..m_words {
        let eq = SplitEqGhash::new(&tau[i + 1..]);
        let (g1, g_inf) = round_message(&wa, &wb, &wz, &eq);
        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho = ch.sample_f128();
        rounds.push((g1, g_inf));
        r.push(rho);
        fold_low(&mut wa, rho);
        fold_low(&mut wb, rho);
        fold_low(&mut wz, rho);
    }
    debug_assert_eq!(wa.len(), 1);

    let (ea, eb, ec) = (wa[0], wb[0], wz[0]);
    // Bind all three final claims BEFORE the next challenge is drawn (which is
    // Phase 2's α). The α-batched reduction of `ea`/`eb` is only sound if α
    // comes after them — a prover that knew α could pick a product-preserving
    // (ea, eb) pair satisfying the one batched equation. `ec` rides along at the
    // same position; the opening binds it again as a claim value.
    ch.observe_f128(ea);
    ch.observe_f128(eb);
    ch.observe_f128(ec);

    // Recycle the folded tables (each still owns its full round-1 capacity).
    for v in [wa, wb, wz] {
        crate::scratch::give_f128(v);
    }

    let proof = Proof { rounds, ea, eb, ec };
    let claim = Claim { r, ea, eb, ec };
    (proof, claim)
}

/// Verify a zerocheck proof over `n_log + kappa` variables, walking the
/// challenger in lockstep with [`prove`].
pub fn verify<C: Challenger>(
    n_log: usize,
    kappa: usize,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claim, VerifyError> {
    let m_words = n_log + kappa;
    if proof.rounds.len() != m_words {
        return Err(VerifyError::BadRoundCount {
            expected: m_words,
            got: proof.rounds.len(),
        });
    }

    ch.observe_label(LABEL);
    let tau = ch.sample_f128_vec(m_words);

    // Convention A chain, identical in shape to `crate::zerocheck::verify`: the
    // running claim is the bare inner value `G(ρ)`; the just-bound variable's eq
    // factor is absorbed by reconstructing `G(0)` from the consistency identity.
    // A zerocheck starts at target 0.
    let mut running = F128::ZERO;
    let mut r = Vec::with_capacity(m_words);
    for (i, &(g1, g_inf)) in proof.rounds.iter().enumerate() {
        let t = tau[i];
        let one_plus_t = F128::ONE + t;
        let g0 = (running + t * g1) * one_plus_t.inv();

        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho = ch.sample_f128();
        r.push(rho);

        let one_plus_rho = F128::ONE + rho;
        // G(ρ) = G(0)·(1+ρ) + G(1)·ρ + G(∞)·ρ·(1+ρ).
        running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }

    // The eq factors never accumulated into the running claim, so what is left
    // is the bare summand at `r`: `(Az+a_const)·(Bz+b_const) + z`.
    if running != proof.ea * proof.eb + proof.ec {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    // Same transcript position as the prover — before Phase 2's α.
    ch.observe_f128(proof.ea);
    ch.observe_f128(proof.eb);
    ch.observe_f128(proof.ec);

    Ok(Claim {
        r,
        ea: proof.ea,
        eb: proof.eb,
        ec: proof.ec,
    })
}

/// One eq-weighted round message `(G(1), G(∞))` for the summand
/// `wa·wb + wz`, with the current variable's eq factor left to the verifier.
///
/// `eq` carries the eq weights of the *not-yet-bound* variables split as
/// `eq = eq_lo ⊗ eq_hi` ([`SplitEqGhash`]), so only `2^n_lo + 2^n_hi` eq entries
/// are built instead of the full product. Low-bit binding: index `2x'` is
/// `(0, x')` and `2x'+1` is `(1, x')`.
///
/// `wz` is linear in the bound variable, so it contributes to `G(1)` only — the
/// `∞` (leading) coefficient of a degree-2 polynomial sees the quadratic term
/// alone.
fn round_message(wa: &[F128], wb: &[F128], wz: &[F128], eq: &SplitEqGhash) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len(); // 2^n_lo x_lo values per x_hi
    let n_blocks = hi.len(); // 2^n_hi
    debug_assert_eq!(block * n_blocks, wa.len() / 2);

    // One outer block (fixed x_hi): inner sum weighted by eq_lo, scaled once by
    // eq_hi[x_hi].
    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);
            let el = lo[x_lo];
            s1 += el * (wa[i1] * wb[i1] + wz[i1]);
            // Char 2: (a1 − a0)(b1 − b0) = (a0 + a1)(b0 + b1).
            s_inf += el * ((wa[i0] + wa[i1]) * (wb[i0] + wb[i1]));
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    let pairs = block * n_blocks;
    match crate::sumcheck_round_min_len(pairs, n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a1, ainf), (b1, binf)| (a1 + b1, ainf + binf),
            ),
        None => {
            let (mut g1, mut g_inf) = (F128::ZERO, F128::ZERO);
            for x_hi in 0..n_blocks {
                let (o, i) = block_fn(x_hi);
                g1 += o;
                g_inf += i;
            }
            (g1, g_inf)
        }
    }
}

/// Bind the low variable of one table at `rho`, halving it:
/// `u[x] ← u[2x] + rho·(u[2x+1] + u[2x])`.
///
/// Wide folds read the old table and write a **pooled** buffer, then swap and
/// recycle it — no per-round allocation, and no in-place aliasing (slot `x` is
/// also read as `2x'` for `x' = x/2`). Narrow folds fall through to the shared
/// serial kernel. Gating is the crate's [`crate::fold_min_len`], the same rule
/// the other sub-gate folds use.
fn fold_low(u: &mut Vec<F128>, rho: F128) {
    let half = u.len() / 2;
    match crate::fold_min_len(half) {
        Some(min_len) => {
            // `take_f128(half)` returns a length-`half` buffer; the map writes
            // every slot, satisfying the write-before-read contract.
            let mut out = crate::scratch::take_f128(half);
            {
                let src: &[F128] = u;
                out.par_iter_mut()
                    .with_min_len(min_len)
                    .enumerate()
                    .for_each(|(x, o)| {
                        let a0 = src[2 * x];
                        *o = a0 + rho * (src[2 * x + 1] + a0);
                    });
            }
            let old = std::mem::replace(u, out);
            crate::scratch::give_f128(old);
        }
        None => crate::zerocheck::multilinear::fold_in_place_single(u, rho),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::tests::{Rng, mixed_gate, mixed_witness, mult_gate, mult_witness};
    use crate::element_r1cs::{ElementTableType, broadcast_add};
    use crate::zerocheck::multilinear::eq_eval;

    /// Direct MLE evaluation of `table` at `point`, binding the low variable
    /// first — the same order [`fold_low`] uses.
    fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
        let mut t = table.to_vec();
        for &p in point {
            crate::zerocheck::multilinear::fold_in_place_single(&mut t, p);
        }
        t[0]
    }

    /// `(pa, pb)` for a witness — the same preparation [`super::super::prove`]
    /// does.
    fn prepare(ty: &ElementTableType, z: &[F128], n_log: usize) -> (Vec<F128>, Vec<F128>) {
        let (mut pa, mut pb) = ty.apply(z, n_log);
        broadcast_add(&mut pa, ty.a_const(), n_log);
        broadcast_add(&mut pb, ty.b_const(), n_log);
        (pa, pb)
    }

    /// Brute-force the zerocheck sum `Σ_x eq(τ,x)·(pa·pb + z)(x)` over the
    /// hypercube, evaluating `eq` from its definition. Low-bit-first index
    /// convention: bit `i` of `x` is coordinate `i` of the point.
    fn brute_force_sum(pa: &[F128], pb: &[F128], z: &[F128], tau: &[F128]) -> F128 {
        let mut acc = F128::ZERO;
        for x in 0..pa.len() {
            let bits: Vec<F128> = (0..tau.len())
                .map(|i| {
                    if (x >> i) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            acc += eq_eval(tau, &bits) * (pa[x] * pb[x] + z[x]);
        }
        acc
    }

    /// The statement the zerocheck proves is TRUE for a satisfying witness: the
    /// eq-weighted sum is zero at any τ, for every shape and count including the
    /// n=0 edge. This is the differential anchor — if the relation encoding or
    /// the padding convention were wrong, this fails before any sumcheck runs.
    #[test]
    fn satisfying_witness_has_zero_eq_weighted_sum() {
        let mut rng = Rng::new(4242);
        for kappa in [2usize, 3] {
            let ty = if kappa == 2 {
                mult_gate(2)
            } else {
                mixed_gate(&mut rng)
            };
            for n_log in [2usize, 4] {
                for n in [0usize, 1, 3, 1 << n_log] {
                    if n > 1 << n_log {
                        continue;
                    }
                    let z = if kappa == 2 {
                        mult_witness(&ty, n_log, n, &mut rng)
                    } else {
                        mixed_witness(&ty, n_log, n, &mut rng)
                    };
                    assert!(ty.satisfies(&z, n_log, n), "κ={kappa} n_log={n_log} n={n}");
                    let (pa, pb) = prepare(&ty, &z, n_log);
                    let tau: Vec<F128> = (0..n_log + kappa).map(|_| rng.f128()).collect();
                    assert_eq!(
                        brute_force_sum(&pa, &pb, &z, &tau),
                        F128::ZERO,
                        "κ={kappa} n_log={n_log} n={n}"
                    );
                }
            }
        }
    }

    /// **Differential test.** For a *random* (not necessarily satisfying)
    /// instance, the prover's messages must be the honest sumcheck of the real
    /// polynomial. We check that against brute force in two independent ways:
    ///
    /// 1. the final evaluations equal the tables' MLEs at the claim point `r`,
    ///    computed by direct folding;
    /// 2. re-running the verifier's chain from the true initial target
    ///    (`brute_force_sum`) lands exactly on `ea·eb + ec`.
    ///
    /// (2) is the strong statement: every round message is pinned, because a
    /// wrong `(G(1), G(∞))` anywhere breaks the chain.
    #[test]
    fn round_messages_match_brute_force_on_random_instances() {
        let mut rng = Rng::new(31337);
        for m_words in [1usize, 2, 5, 8] {
            let n = 1usize << m_words;
            let pa: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let pb: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let z: Vec<F128> = (0..n).map(|_| rng.f128()).collect();

            // n_log/kappa only split the point; the sumcheck sees m_words.
            let n_log = m_words / 2;
            let kappa = m_words - n_log;
            let mut ch = FsChallenger::new(b"element-zc-diff");
            let (proof, claim) = prove(pa.clone(), pb.clone(), &z, n_log, kappa, &mut ch);

            // Re-derive τ the way the prover did.
            let mut ch2 = FsChallenger::new(b"element-zc-diff");
            ch2.observe_label(LABEL);
            let tau = ch2.sample_f128_vec(m_words);

            // (1) finals are the MLEs at r.
            assert_eq!(claim.ea, mle_eval(&pa, &claim.r), "ea m={m_words}");
            assert_eq!(claim.eb, mle_eval(&pb, &claim.r), "eb m={m_words}");
            assert_eq!(claim.ec, mle_eval(&z, &claim.r), "ec m={m_words}");

            // (2) the chain from the true target closes on ea·eb + ec.
            let mut running = brute_force_sum(&pa, &pb, &z, &tau);
            for (i, &(g1, g_inf)) in proof.rounds.iter().enumerate() {
                let t = tau[i];
                let g0 = (running + t * g1) * (F128::ONE + t).inv();
                let rho = claim.r[i];
                let one_plus_rho = F128::ONE + rho;
                running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
            }
            assert_eq!(
                running,
                claim.ea * claim.eb + claim.ec,
                "chain from brute-force target, m={m_words}"
            );
        }
    }

    /// Prove → verify roundtrip on satisfying witnesses at several shapes and
    /// counts (including non-power-of-two `n`, full utilization, and `n = 0`).
    #[test]
    fn prove_verify_roundtrip_honest() {
        let mut rng = Rng::new(909);
        for (n_log, n) in [(2usize, 0usize), (2, 3), (3, 5), (4, 16), (6, 37)] {
            let ty = mult_gate(2);
            let z = mult_witness(&ty, n_log, n, &mut rng);
            let (pa, pb) = prepare(&ty, &z, n_log);

            let mut ch_p = FsChallenger::new(b"element-zc-rt");
            let (proof, claim_p) = prove(pa, pb, &z, n_log, 2, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-zc-rt");
            let claim_v = verify(n_log, 2, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected at n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v, "n_log={n_log} n={n}");
        }
    }

    /// A witness violating ONE constraint in ONE row must be rejected.
    #[test]
    fn unsatisfying_witness_rejected() {
        let mut rng = Rng::new(6161);
        let (n_log, n) = (4usize, 11usize);
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Break the product in row 5.
        z[2 * (1 << n_log) + 5] += F128::ONE;
        assert!(!ty.satisfies(&z, n_log, n));
        let (pa, pb) = prepare(&ty, &z, n_log);

        let mut ch_p = FsChallenger::new(b"element-zc-bad");
        let (proof, _) = prove(pa, pb, &z, n_log, 2, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-zc-bad");
        assert_eq!(
            verify(n_log, 2, &proof, &mut ch_v),
            Err(VerifyError::SumcheckFinalFailed)
        );
    }

    /// A non-zero entry in a dummy row is a violation too — the padding rows are
    /// inside the sum, not skipped.
    #[test]
    fn dirty_dummy_row_rejected() {
        let mut rng = Rng::new(717);
        let (n_log, n) = (4usize, 9usize);
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Row 12 is dummy; set its product column without its operands.
        z[2 * (1 << n_log) + 12] = F128::ONE;
        let (pa, pb) = prepare(&ty, &z, n_log);

        let mut ch_p = FsChallenger::new(b"element-zc-dirty");
        let (proof, _) = prove(pa, pb, &z, n_log, 2, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-zc-dirty");
        assert!(verify(n_log, 2, &proof, &mut ch_v).is_err());
    }

    /// Every proof component must be transcript-bound: flipping a bit anywhere
    /// makes the verifier reject.
    #[test]
    fn verify_rejects_mutations() {
        let mut rng = Rng::new(5);
        let (n_log, kappa) = (4usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 13, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-mut");
        let (proof, _) = prove(pa, pb, &z, n_log, kappa, &mut ch_p);

        let n_rounds = proof.rounds.len();
        let mut cases: Vec<(String, Proof)> = Vec::new();
        for i in 0..n_rounds {
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
        for (name, field) in [("ea", 0usize), ("eb", 1), ("ec", 2)] {
            let mut bad = proof.clone();
            match field {
                0 => bad.ea += F128::ONE,
                1 => bad.eb += F128::ONE,
                _ => bad.ec += F128::ONE,
            }
            cases.push((name.to_string(), bad));
        }
        for (name, bad) in cases {
            let mut ch = FsChallenger::new(b"element-zc-mut");
            assert!(
                verify(n_log, kappa, &bad, &mut ch).is_err(),
                "verify accepted mutation: {name}"
            );
        }
    }

    /// AUDIT (Fiat–Shamir binding of the final claims). A *product-preserving*
    /// tamper `(ea, eb) → (ea·t, eb·t⁻¹)` leaves the zerocheck's own final check
    /// `running == ea·eb + ec` satisfied, so `verify` still returns `Ok` — the
    /// zerocheck alone is blind to it, exactly as in
    /// `crate::zerocheck::tests::audit_final_ab_claims_bound_to_transcript`.
    ///
    /// The defense is that all three finals are observed last, so the next
    /// challenge — the slot Phase 2 draws α from — must diverge from the honest
    /// run. Without that observe the α-batched reduction of `ea`/`eb` would be
    /// unsound: a prover that already knew α could pick the pair.
    #[test]
    fn audit_final_claims_bound_to_transcript() {
        let mut rng = Rng::new(0xF1A7_5A11);
        let (n_log, kappa) = (4usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 13, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-bind");
        let (proof, _) = prove(pa, pb, &z, n_log, kappa, &mut ch_p);

        let mut ch_honest = FsChallenger::new(b"element-zc-bind");
        assert!(verify(n_log, kappa, &proof, &mut ch_honest).is_ok());
        let alpha_honest = ch_honest.sample_f128();

        let t = F128::new(3, 0);
        let mut bad = proof.clone();
        bad.ea = proof.ea * t;
        bad.eb = proof.eb * t.inv();
        let mut ch_bad = FsChallenger::new(b"element-zc-bind");
        assert!(
            verify(n_log, kappa, &bad, &mut ch_bad).is_ok(),
            "product-preserving swap is invisible to the zerocheck's own check"
        );
        assert_ne!(
            alpha_honest,
            ch_bad.sample_f128(),
            "final claims are not bound to the transcript — α would be reusable"
        );
    }

    /// Shape rejection: a truncated round list.
    #[test]
    fn verify_rejects_bad_round_count() {
        let mut rng = Rng::new(8);
        let (n_log, kappa) = (3usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 4, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-shape");
        let (mut proof, _) = prove(pa, pb, &z, n_log, kappa, &mut ch_p);
        proof.rounds.pop();
        let mut ch = FsChallenger::new(b"element-zc-shape");
        assert!(matches!(
            verify(n_log, kappa, &proof, &mut ch),
            Err(VerifyError::BadRoundCount { .. })
        ));
    }

    /// `fold_low` must agree with the shared serial kernel at every width,
    /// including across the parallel gate — the pooled-buffer path is the one
    /// place this module writes its own kernel.
    #[test]
    fn fold_low_matches_serial_kernel() {
        let mut rng = Rng::new(2024);
        for log_n in 1..=18usize {
            let n = 1usize << log_n;
            let v: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let mut a = v.clone();
            fold_low(&mut a, rho);
            let mut b = v;
            crate::zerocheck::multilinear::fold_in_place_single(&mut b, rho);
            assert_eq!(a, b, "log_n={log_n}");
        }
    }
}
