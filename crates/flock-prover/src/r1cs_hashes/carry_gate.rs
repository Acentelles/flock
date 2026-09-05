//! Verified cost gate for virtualizing the 16-bit rejection borrow chain.
//!
//! This is a reduction to sixteen word-bit MLE claims, not a standalone proof:
//! the caller must authenticate those claims and bind the output claim. No
//! carry planes are committed. It is not enabled in the production slot lane.
//!
//! The packed kernel computes the first three rounds' coefficient tables as
//! weighted bitwise expressions, sharing equality tensors across packed MLE
//! gathers, and densifies the two inputs only after those challenges. A scalar reference emits the
//! same round messages and challenges. Interpolation nodes are the distinct
//! field elements with encodings 0, 1, 2, 3, never repeated addition of ONE.

use super::packed_mle::{evaluate_packed, evaluate_packed_tables};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use rayon::prelude::*;

const WORD_BITS: usize = 16;
const M: u16 = 0xF005;

fn eq_weights(point: &[F128]) -> Vec<F128> {
    let mut weights = vec![F128::ONE];
    for &r in point {
        let mut next = vec![F128::ZERO; weights.len() * 2];
        next.par_chunks_mut(2).zip(&weights).for_each(|(pair, &w)| {
            pair[1] = w * r;
            pair[0] = w + pair[1];
        });
        weights = next;
    }
    weights
}

fn eq_at(a: &[F128], b: &[F128]) -> F128 {
    a.iter()
        .zip(b)
        .fold(F128::ONE, |acc, (&a, &b)| acc * (F128::ONE + a + b))
}

fn node(t: usize) -> F128 {
    F128 {
        lo: t as u64,
        hi: 0,
    }
}
fn polynomial(x: F128, d: F128, mi: bool) -> F128 {
    if mi {
        F128::ONE + x + x * d
    } else {
        d * (F128::ONE + x)
    }
}
fn pack(bits: &[bool]) -> Vec<F128> {
    bits.par_chunks(128)
        .map(|chunk| {
            let mut w = F128::ZERO;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    if i < 64 {
                        w.lo |= 1 << i;
                    } else {
                        w.hi |= 1 << (i - 64);
                    }
                }
            }
            w
        })
        .collect()
}
fn bit_eval(bits: &[bool], point: &[F128]) -> F128 {
    evaluate_packed(&pack(bits), &[point.to_vec()])[0]
}
fn lift(bits: &[bool]) -> Vec<F128> {
    bits.par_iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect()
}
fn fold(table: &mut Vec<F128>, r: F128) {
    let half = table.len() / 2;
    let (lo, hi) = table.split_at_mut(half);
    lo.par_iter_mut()
        .zip(hi)
        .for_each(|(a, b)| *a = *a + r * (*a + *b));
    table.truncate(half);
}
fn fold_bits_prefix(bits: &[bool], point: &[F128]) -> Vec<F128> {
    let weights = eq_weights(point);
    let mut lut = vec![F128::ZERO; 1 << weights.len()];
    for mask in 1..lut.len() {
        lut[mask] = lut[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
    }
    let length = bits.len() >> point.len();
    (0..length)
        .into_par_iter()
        .map(|y| {
            let mut code = 0;
            for a in 0..weights.len() {
                code |= usize::from(bits[a * length + y]) << a;
            }
            lut[code]
        })
        .collect()
}

fn borrow_tables(words: &[u16]) -> Vec<Vec<bool>> {
    let mut d = vec![vec![false; words.len()]; WORD_BITS + 1];
    for (j, &word) in words.iter().enumerate() {
        let mut borrow = false;
        for i in 0..WORD_BITS {
            let x = word >> i & 1 != 0;
            let m = M >> i & 1 != 0;
            borrow = (!x & (m | borrow)) | (m & borrow);
            d[i + 1][j] = borrow;
        }
    }
    d
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerProof {
    pub rounds: Vec<[F128; 4]>,
    pub x_value: F128,
    pub d_value: F128,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainProof {
    pub output_value: F128,
    /// Layers are ordered from bit 15 down to bit 0.
    pub layers: Vec<LayerProof>,
}
#[derive(Debug)]
pub struct WordBitClaim {
    pub bit: usize,
    pub point: Vec<F128>,
    pub value: F128,
}
#[derive(Debug)]
pub struct ChainClaims {
    pub output_point: Vec<F128>,
    pub output_value: F128,
    pub inputs: Vec<WordBitClaim>,
}

fn packed_round(x: &[bool], d: &[bool], mi: bool, rho: &[F128], prefix: &[F128]) -> [F128; 4] {
    let round = prefix.len();
    let weights = eq_weights(prefix);
    let length = x.len() >> (round + 1);
    let mut x0 = Vec::new();
    let mut dx = Vec::new();
    let mut d0 = Vec::new();
    let mut dd = Vec::new();
    for a in 0..weights.len() {
        let lo = (2 * a) * length;
        let hi = lo + length;
        let xl = pack(&x[lo..hi]);
        let xh = pack(&x[hi..hi + length]);
        let dl = pack(&d[lo..hi]);
        let dh = pack(&d[hi..hi + length]);
        dx.push(xl.iter().zip(xh).map(|(&a, b)| a + b).collect::<Vec<_>>());
        dd.push(dl.iter().zip(dh).map(|(&a, b)| a + b).collect::<Vec<_>>());
        x0.push(xl);
        d0.push(dl);
    }
    let and = |a: F128, b: F128| F128 {
        lo: a.lo & b.lo,
        hi: a.hi & b.hi,
    };
    let mut tables = Vec::new();
    let mut coefficients = Vec::new();
    for (a, &wa) in weights.iter().enumerate() {
        tables.push(if mi { x0[a].clone() } else { d0[a].clone() });
        coefficients.push((0, wa));
        tables.push(if mi { dx[a].clone() } else { dd[a].clone() });
        coefficients.push((1, wa));
        for (b, &wb) in weights.iter().enumerate() {
            let weight = wa * wb;
            tables.push((0..x0[a].len()).map(|j| and(x0[a][j], d0[b][j])).collect());
            coefficients.push((0, weight));
            tables.push(
                (0..x0[a].len())
                    .map(|j| and(x0[a][j], dd[b][j]) + and(dx[a][j], d0[b][j]))
                    .collect(),
            );
            coefficients.push((1, weight));
            tables.push((0..x0[a].len()).map(|j| and(dx[a][j], dd[b][j])).collect());
            coefficients.push((2, weight));
        }
    }
    let values = evaluate_packed_tables(&tables, &rho[round + 1..]);
    let mut q = [
        if mi { F128::ONE } else { F128::ZERO },
        F128::ZERO,
        F128::ZERO,
    ];
    for ((coefficient, weight), value) in coefficients.into_iter().zip(values) {
        q[coefficient] += weight * value;
    }
    let eq_prefix = eq_at(&rho[..round], prefix);
    std::array::from_fn(|t| {
        let t = node(t);
        eq_prefix * (F128::ONE + rho[round] + t) * (q[0] + t * q[1] + t * t * q[2])
    })
}

fn prove_layer<Ch: Challenger>(
    x: &[bool],
    d: &[bool],
    mi: bool,
    rho: &[F128],
    packed: bool,
    ch: &mut Ch,
) -> (LayerProof, Vec<F128>) {
    let mut point = Vec::with_capacity(rho.len());
    let mut rounds = Vec::with_capacity(rho.len());
    let (mut eq, mut x, mut d) = if packed && !rho.is_empty() {
        for _ in 0..rho.len().min(3) {
            let evals = packed_round(x, d, mi, rho, &point);
            ch.observe_f128_slice(&evals);
            point.push(ch.sample_f128());
            rounds.push(evals);
        }
        let prefix = eq_at(&rho[..point.len()], &point);
        let mut eq = eq_weights(&rho[point.len()..]);
        eq.par_iter_mut().for_each(|e| *e *= prefix);
        (eq, fold_bits_prefix(x, &point), fold_bits_prefix(d, &point))
    } else {
        (eq_weights(rho), lift(x), lift(d))
    };
    while eq.len() > 1 {
        let half = eq.len() / 2;
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || [F128::ZERO; 4],
                |mut acc, j| {
                    let de = eq[j] + eq[j + half];
                    let dx = x[j] + x[j + half];
                    let dd = d[j] + d[j + half];
                    for (t, acc) in acc.iter_mut().enumerate() {
                        let t = node(t);
                        *acc += (eq[j] + t * de) * polynomial(x[j] + t * dx, d[j] + t * dd, mi);
                    }
                    acc
                },
            )
            .reduce(
                || [F128::ZERO; 4],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        *a += b;
                    }
                    a
                },
            );
        ch.observe_f128_slice(&evals);
        let r = ch.sample_f128();
        fold(&mut eq, r);
        fold(&mut x, r);
        fold(&mut d, r);
        point.push(r);
        rounds.push(evals);
    }
    ch.observe_f128(x[0]);
    ch.observe_f128(d[0]);
    (
        LayerProof {
            rounds,
            x_value: x[0],
            d_value: d[0],
        },
        point,
    )
}

fn start<Ch: Challenger>(vars: usize, ch: &mut Ch) -> Vec<F128> {
    ch.observe_label(b"flock-borrow-chain-cost-gate-v1");
    ch.observe_bytes(&(vars as u64).to_le_bytes());
    ch.observe_bytes(&M.to_le_bytes());
    ch.sample_f128_vec(vars)
}

pub fn prove_chain<Ch: Challenger>(words: &[u16], ch: &mut Ch) -> (ChainProof, Vec<F128>) {
    prove_chain_with(words, true, ch)
}
/// Differential and performance oracle. Both kernels use the SAME transcript.
pub fn prove_chain_with<Ch: Challenger>(
    words: &[u16],
    packed: bool,
    ch: &mut Ch,
) -> (ChainProof, Vec<F128>) {
    assert!(words.len().is_power_of_two());
    let vars = words.len().trailing_zeros() as usize;
    let d = borrow_tables(words);
    let mut rho = start(vars, ch);
    let output_value = bit_eval(&d[16], &rho);
    ch.observe_f128(output_value);
    let mut layers = Vec::with_capacity(WORD_BITS);
    for i in (0..WORD_BITS).rev() {
        let x: Vec<_> = words.iter().map(|&x| x >> i & 1 != 0).collect();
        let (layer, point) = prove_layer(&x, &d[i], M >> i & 1 != 0, &rho, packed, ch);
        layers.push(layer);
        rho = point;
    }
    (
        ChainProof {
            output_value,
            layers,
        },
        rho,
    )
}

/// Check every layer transition and return claims the caller MUST authenticate
/// against the word-bit commitment, plus the output claim it must bind.
pub fn verify_chain<Ch: Challenger>(
    proof: &ChainProof,
    vars: usize,
    ch: &mut Ch,
) -> Result<ChainClaims, &'static str> {
    if vars >= usize::BITS as usize || proof.layers.len() != WORD_BITS {
        return Err("invalid borrow-chain shape");
    }
    let mut rho = start(vars, ch);
    let output_point = rho.clone();
    let mut claim = proof.output_value;
    ch.observe_f128(claim);
    let inv_denominator = node(6).inv(); // product of the three nonzero differences of {0,1,2,3}
    let mut inputs = Vec::with_capacity(WORD_BITS);
    for (k, layer) in proof.layers.iter().enumerate() {
        if layer.rounds.len() != vars {
            return Err("invalid borrow layer arity");
        }
        let mut point = Vec::with_capacity(vars);
        for evals in &layer.rounds {
            if evals[0] + evals[1] != claim {
                return Err("borrow round sum mismatch");
            }
            ch.observe_f128_slice(evals);
            let r = ch.sample_f128();
            claim = evals.iter().enumerate().fold(F128::ZERO, |sum, (i, &e)| {
                sum + e
                    * (0..4)
                        .filter(|&j| j != i)
                        .fold(F128::ONE, |a, j| a * (r + node(j)))
            }) * inv_denominator;
            point.push(r);
        }
        let bit = WORD_BITS - 1 - k;
        if claim
            != eq_at(&rho, &point) * polynomial(layer.x_value, layer.d_value, M >> bit & 1 != 0)
        {
            return Err("borrow layer terminal mismatch");
        }
        ch.observe_f128(layer.x_value);
        ch.observe_f128(layer.d_value);
        inputs.push(WordBitClaim {
            bit,
            point: point.clone(),
            value: layer.x_value,
        });
        claim = layer.d_value;
        rho = point;
    }
    if claim != F128::ZERO {
        return Err("nonzero initial borrow");
    }
    Ok(ChainClaims {
        output_point,
        output_value: proof.output_value,
        inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;
    fn words(n: usize) -> Vec<u16> {
        let mut s = 7_u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as u16
            })
            .collect()
    }
    fn dense_eval(bits: &[bool], point: &[F128]) -> F128 {
        let mut values = lift(bits);
        for &r in point {
            fold(&mut values, r);
        }
        values[0]
    }
    #[test]
    fn packed_rounds_match_reference_and_all_claims_verify() {
        for n in [1, 2, 128, 256, 4096] {
            let words = words(n);
            let mut packed = FsChallenger::new(b"carry-exact");
            let (proof, point) = prove_chain(&words, &mut packed);
            let mut reference = FsChallenger::new(b"carry-exact");
            let (expected, expected_point) = prove_chain_with(&words, false, &mut reference);
            assert_eq!(proof, expected);
            assert_eq!(point, expected_point);
            assert_eq!(packed.sample_f128(), reference.sample_f128());
            let mut verifier = FsChallenger::new(b"carry-exact");
            let checked = verify_chain(&proof, n.trailing_zeros() as usize, &mut verifier).unwrap();
            let outputs: Vec<_> = words.iter().map(|&x| x < M).collect();
            assert_eq!(
                checked.output_value,
                dense_eval(&outputs, &checked.output_point)
            );
            for claim in checked.inputs {
                let bits: Vec<_> = words.iter().map(|&x| x >> claim.bit & 1 != 0).collect();
                assert_eq!(claim.value, dense_eval(&bits, &claim.point));
            }
        }
    }
    #[test]
    fn rejects_forged_rounds_terminals_and_output() {
        let words = words(128);
        let (proof, _) = prove_chain(&words, &mut FsChallenger::new(b"carry-reject"));
        for kind in 0..4 {
            let mut bad = proof.clone();
            match kind {
                0 => bad.layers[3].rounds[2][2] += F128::ONE,
                1 => bad.layers[4].x_value += F128::ONE,
                2 => bad.layers[15].d_value += F128::ONE,
                _ => bad.output_value += F128::ONE,
            }
            assert!(verify_chain(&bad, 7, &mut FsChallenger::new(b"carry-reject")).is_err());
        }
    }
}
