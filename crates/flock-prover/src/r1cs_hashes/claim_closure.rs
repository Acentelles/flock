//! Pre-ring-switch claim closure (aerie spec 7.3): collapse every
//! multilinear bit-MLE claim on one commitment into ONE opening claim,
//! so the batched PCS opening carries `[ab, c, closed]` instead of a
//! ring-switch instance per claim.
//!
//! With claims `(p_i, v_i)` on the bit witness `z` over the
//! `m`-variable cube and a `mu` sampled AFTER all values are absorbed:
//!
//! ```text
//! V = sum_i mu^i v_i  =  sum_x W(x) z(x),
//! W(x) = sum_i mu^i eq(p_i, x),
//! ```
//!
//! proven by an m-round degree-2 sumcheck whose terminal `W(r) z(r)`
//! the verifier checks with its OWN `W(r) = sum_i mu^i eq(p_i, r)`
//! (O(claims x m)); `z(r)` is the single surviving claim. Soundness:
//! `(claims - 1)/|F|` from the mu batching plus `2m/|F|` from the
//! sumcheck; the terminal is bound by the unchanged PCS opening of the
//! closed claim.
//!
//! The dense `W` build shares one eq tensor per claim FAMILY (claims
//! differing only in Boolean offsets), and round 1 gates on the packed
//! witness bits. Dense `W` costs `2^m x 16` bytes — the supported range
//! here (m <= 27); the coefficient-form rounds for m = 30 are the
//! documented follow-up in the profile note.

use flock_core::challenger::Challenger;
use flock_core::field::F128;

/// The closure sumcheck wire: degree-2 round evaluations at 0, 1, 2,
/// plus the claimed terminal witness evaluation `z(r)`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClosureProof {
    pub rounds: Vec<[F128; 3]>,
    pub z_value: F128,
}

fn split_boolean(point: &[F128]) -> (usize, Vec<(usize, F128)>) {
    let vars = point.len();
    let mut fixed = 0_usize;
    let mut free = Vec::new();
    for (i, &coord) in point.iter().enumerate() {
        let bit = vars - 1 - i;
        if coord == F128::ZERO {
        } else if coord == F128::ONE {
            fixed |= 1 << bit;
        } else {
            free.push((bit, coord));
        }
    }
    (fixed, free)
}

/// The family decomposition of `W = sum_i mu^i eq(p_i, .)`: one eq
/// tensor and relative-address table per distinct free-coordinate
/// signature, plus each member's Boolean offset and `mu`-power
/// coefficient. `W(x) = sum over (member, entry) with `x = fixed |
/// rel_addr[entry]` of `coef * tensor[entry]`.
struct WeightFamilies {
    families: Vec<Family>,
}

struct Family {
    tensor: Vec<F128>,
    rel_addr: Vec<usize>,
    /// `(fixed offset, mu-power coefficient)` per member.
    members: Vec<(usize, F128)>,
}

fn weight_families(points: &[Vec<F128>], mu: F128, vars: usize) -> WeightFamilies {
    let mut split = Vec::with_capacity(points.len());
    for point in points {
        assert_eq!(point.len(), vars, "claims share the bit domain");
        split.push(split_boolean(point));
    }
    let mut groups: Vec<(Vec<(usize, F128)>, Vec<usize>)> = Vec::new();
    for (index, (_, free)) in split.iter().enumerate() {
        if let Some((_, members)) = groups.iter_mut().find(|(sig, _)| sig == free) {
            members.push(index);
        } else {
            groups.push((free.clone(), vec![index]));
        }
    }
    let mut powers = Vec::with_capacity(points.len());
    let mut power = F128::ONE;
    for _ in points {
        powers.push(power);
        power *= mu;
    }
    let families = groups
        .into_iter()
        .map(|(free, member_ids)| {
            let free_vars = free.len();
            let mut tensor = vec![F128::ONE];
            for &(_, coord) in &free {
                let mut next = Vec::with_capacity(2 * tensor.len());
                for &value in &tensor {
                    next.push(value * (F128::ONE + coord));
                    next.push(value * coord);
                }
                tensor = next;
            }
            let rel_addr: Vec<usize> = (0..1_usize << free_vars)
                .map(|index| {
                    let mut address = 0_usize;
                    for (j, &(bit, _)) in free.iter().enumerate() {
                        if (index >> (free_vars - 1 - j)) & 1 == 1 {
                            address |= 1 << bit;
                        }
                    }
                    address
                })
                .collect();
            let members = member_ids
                .iter()
                .map(|&i| (split[i].0, powers[i]))
                .collect();
            Family {
                tensor,
                rel_addr,
                members,
            }
        })
        .collect();
    WeightFamilies { families }
}

impl WeightFamilies {
    /// Iterate every `(address, weight)` support entry of `W`.
    fn for_each_entry(&self, mut sink: impl FnMut(usize, F128)) {
        for family in &self.families {
            for &(fixed, coef) in &family.members {
                for (offset, &weight) in family.rel_addr.iter().zip(&family.tensor) {
                    sink(fixed | offset, coef * weight);
                }
            }
        }
    }
}

/// Prove the closure. `points` are full `vars`-coordinate bit-MLE
/// points (MSB-first), `values` their claimed evaluations, `z_packed`
/// the packed witness words. Returns the proof and the closed claim's
/// point `r` (MSB-first). Transcript: absorb all values, sample `mu`,
/// rounds, absorb `z_value`.
pub fn prove_closure<Ch: Challenger>(
    z_packed: &[F128],
    vars: usize,
    points: &[Vec<F128>],
    values: &[F128],
    challenger: &mut Ch,
) -> (ClosureProof, Vec<F128>) {
    use rayon::prelude::*;
    assert_eq!(points.len(), values.len());
    assert_eq!(z_packed.len() << 7, 1 << vars, "packed words fill the domain");
    challenger.observe_label(b"aerie-claim-closure-v0");
    challenger.observe_f128_slice(values);
    let mu = challenger.sample_f128();
    let families = weight_families(points, mu, vars);

    let mut rounds = Vec::with_capacity(vars);
    let mut point = Vec::with_capacity(vars);
    // Round 1, entry-wise: the FULL 2^vars weight table never exists.
    // e0 and e1 accumulate directly over W's support entries, gated on
    // the packed witness bits; e_T streams ONE scattered half-size
    // table V(y) = W(T, y) against z(T, y) built from bit pairs (T is
    // the field element with representation 2, the scatter convention:
    // f(T) = f0 + T (f0 + f1)). After the challenge the same buffer is
    // re-scattered into the folded W. Peak memory is one half-size
    // table; every value is the same field element as the dense path's
    // (entry-wise sums are exact regroupings).
    let half = 1_usize << (vars - 1);
    let t_node = F128 { lo: 2, hi: 0 };
    let bit_at = |index: usize| crate::chain::read_packed_bit(z_packed, index);
    let mut e0 = F128::ZERO;
    let mut e1 = F128::ZERO;
    let mut v_table = vec![F128::ZERO; half];
    families.for_each_entry(|address, weight| {
        let top = address >> (vars - 1) & 1 == 1;
        if bit_at(address) {
            if top {
                e1 += weight;
            } else {
                e0 += weight;
            }
        }
        // V(y) = W(T, y): the top coordinate contributes (1 + T) for a
        // 0-half entry and T for a 1-half entry.
        let factor = if top { t_node } else { F128::ONE + t_node };
        v_table[address & (half - 1)] += factor * weight;
    });
    let e2: F128 = (0..half)
        .into_par_iter()
        .fold(
            || F128::ZERO,
            |acc, y| {
                let (b0, b1) = (bit_at(y), bit_at(y + half));
                let z_t = match (b0, b1) {
                    (false, false) => return acc,
                    (true, true) => F128::ONE,
                    (false, true) => t_node,
                    (true, false) => F128::ONE + t_node,
                };
                acc + v_table[y] * z_t
            },
        )
        .reduce(|| F128::ZERO, |a, b| a + b);
    let evals = [e0, e1, e2];
    challenger.observe_f128_slice(&evals);
    let r0 = challenger.sample_f128();
    // Densify z at half size; re-scatter the SAME buffer into the
    // folded W (top factor lin(., r0)).
    let mut z: Vec<F128> = (0..half)
        .into_par_iter()
        .map(|y| {
            let (b0, b1) = (bit_at(y), bit_at(y + half));
            match (b0, b1) {
                (false, false) => F128::ZERO,
                (true, true) => F128::ONE,
                (false, true) => r0,
                (true, false) => F128::ONE + r0,
            }
        })
        .collect();
    let mut w = v_table;
    w.par_iter_mut().for_each(|slot| *slot = F128::ZERO);
    families.for_each_entry(|address, weight| {
        let top = address >> (vars - 1) & 1 == 1;
        let factor = if top { r0 } else { F128::ONE + r0 };
        w[address & (half - 1)] += factor * weight;
    });
    rounds.push(evals);
    point.push(r0);

    while z.len() > 1 {
        let half = z.len() / 2;
        let evals: [F128; 3] = {
            let (z_low, z_high) = z.split_at(half);
            let (w_low, w_high) = w.split_at(half);
            (0..half)
                .into_par_iter()
                .fold(
                    || [F128::ZERO; 3],
                    |mut acc, i| {
                        let (z0, z1) = (z_low[i], z_high[i]);
                        let (w0, w1) = (w_low[i], w_high[i]);
                        acc[0] += w0 * z0;
                        acc[1] += w1 * z1;
                        // Node T (field element 2): f(T) = f0 + T (f0 + f1).
                        let t_node = F128 { lo: 2, hi: 0 };
                        acc[2] += (w0 + t_node * (w0 + w1)) * (z0 + t_node * (z0 + z1));
                        acc
                    },
                )
                .reduce(
                    || [F128::ZERO; 3],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    },
                )
        };
        challenger.observe_f128_slice(&evals);
        let r = challenger.sample_f128();
        let fold = |table: &mut Vec<F128>| {
            let half = table.len() / 2;
            let (low, high) = table.split_at_mut(half);
            low.par_iter_mut()
                .zip(&high[..half])
                .for_each(|(l, &h)| *l += r * (*l + h));
            table.truncate(half);
        };
        fold(&mut z);
        fold(&mut w);
        rounds.push(evals);
        point.push(r);
    }
    let z_value = z[0];
    challenger.observe_f128(z_value);
    (ClosureProof { rounds, z_value }, point)
}

/// Verify the closure; returns the closed claim `(r, z_value)`.
pub fn verify_closure<Ch: Challenger>(
    proof: &ClosureProof,
    vars: usize,
    points: &[Vec<F128>],
    values: &[F128],
    challenger: &mut Ch,
) -> Result<(Vec<F128>, F128), &'static str> {
    if proof.rounds.len() != vars {
        return Err("closure round count does not match the domain");
    }
    challenger.observe_label(b"aerie-claim-closure-v0");
    challenger.observe_f128_slice(values);
    let mu = challenger.sample_f128();
    let mut expected = F128::ZERO;
    let mut power = F128::ONE;
    for &value in values {
        expected += power * value;
        power *= mu;
    }
    let mut point = Vec::with_capacity(vars);
    for evals in &proof.rounds {
        if evals[0] + evals[1] != expected {
            return Err("a closure round does not sum to its claim");
        }
        challenger.observe_f128_slice(evals);
        let r = challenger.sample_f128();
        // Quadratic through (0, e0), (1, e1), (2, e2) at r, char 2:
        // e(r) = e0 (r+1)(r+x2... use Lagrange over GF(2^128) nodes
        // {0, 1, x}, with the third node the field element 2 = x.
        expected = quadratic_at(evals, r);
        point.push(r);
    }
    challenger.observe_f128(proof.z_value);
    // Terminal: W(r) z(r) with the verifier's own W(r).
    let mut w_at = F128::ZERO;
    let mut power = F128::ONE;
    for p in points {
        let mut eq = F128::ONE;
        for (&c, &r) in p.iter().zip(&point) {
            eq *= c * r + (F128::ONE + c) * (F128::ONE + r);
        }
        w_at += power * eq;
        power *= mu;
    }
    if expected != w_at * proof.z_value {
        return Err("the closure terminal does not match the claims");
    }
    Ok((point, proof.z_value))
}

/// Interpolate the quadratic through nodes `0, 1, 2` at `r` over
/// `GF(2^128)` (the node `2` is the polynomial-basis element `x`).
fn quadratic_at(evals: &[F128; 3], r: F128) -> F128 {
    let two = F128 { lo: 2, hi: 0 };
    // Lagrange: L0 = (r-1)(r-2)/((0-1)(0-2)), etc.; char-2 signs vanish.
    let d0 = (F128::ONE * two).inv(); // (0+1)(0+2) = 2
    let d1 = (F128::ONE * (F128::ONE + two)).inv(); // (1)(1+2) = 3
    let d2 = (two * (two + F128::ONE)).inv(); // (2)(2+1) = 6
    evals[0] * (r + F128::ONE) * (r + two) * d0
        + evals[1] * r * (r + two) * d1
        + evals[2] * r * (r + F128::ONE) * d2
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    fn mle(bits: &[bool], point: &[F128]) -> F128 {
        let mut layer: Vec<F128> = bits
            .iter()
            .map(|&b| if b { F128::ONE } else { F128::ZERO })
            .collect();
        for &r in point {
            let half = layer.len() / 2;
            for i in 0..half {
                let low = layer[i];
                layer[i] = low + r * (low + layer[half + i]);
            }
            layer.truncate(half);
        }
        layer[0]
    }

    #[test]
    fn the_closure_collapses_claims_and_rejects_a_lie() {
        let vars = 12;
        let mut acc = 0x9e37_79b9_u64;
        let bits: Vec<bool> = (0..1 << vars)
            .map(|_| {
                acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1);
                acc >> 63 == 1
            })
            .collect();
        let z_packed = flock_core::pcs::pack_witness(&bits, vars);
        // A family-shaped claim set: shared free coords with Boolean
        // offsets, plus one fully dense point.
        let coord = |seed: u64| F128 {
            lo: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 5,
            hi: seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 3,
        };
        let shared: Vec<F128> = (0..7).map(|i| coord(i + 11)).collect();
        let mut points: Vec<Vec<F128>> = Vec::new();
        for offset in 0..12_u64 {
            let mut p = Vec::with_capacity(vars);
            for j in 0..5 {
                p.push(if (offset >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            p.extend_from_slice(&shared);
            points.push(p);
        }
        points.push((0..vars as u64).map(|i| coord(100 + i)).collect());
        let values: Vec<F128> = points.iter().map(|p| mle(&bits, p)).collect();

        let mut prover = FsChallenger::new(b"closure-test");
        let (proof, r) = prove_closure(&z_packed, vars, &points, &values, &mut prover);
        assert_eq!(proof.z_value, mle(&bits, &r), "terminal is the witness MLE");

        let mut verifier = FsChallenger::new(b"closure-test");
        let (r_v, z_v) =
            verify_closure(&proof, vars, &points, &values, &mut verifier).expect("honest");
        assert_eq!((r_v, z_v), (r, proof.z_value));

        // One lied value must fail (the mu batching catches it).
        let mut lied = values.clone();
        lied[3] += F128::ONE;
        let mut cheat = FsChallenger::new(b"closure-test");
        assert!(verify_closure(&proof, vars, &points, &lied, &mut cheat).is_err());
    }
}
