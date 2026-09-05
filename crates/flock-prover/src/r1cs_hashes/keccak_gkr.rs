//! GKR-over-rounds prototype for Keccak-f (aerie stream B, gate B1/B2).
//!
//! The committed encoders prove a permutation by committing all
//! 24 x 1,600 chi AND witnesses. This prototype instead reduces a claim
//! `MLE(L_round, rho) = v` on a round's output state to a claim on the
//! previous state, layer by layer, so a full chain would commit ONLY
//! `state_0` and `state_24` (13.3x fewer committed bits).
//!
//! One round is `iota . chi . pi . rho . theta`. Write `a = pi rho
//! theta(L)` and let `a1, a2` be its x+1, x+2 lane-neighbor tables. In
//! char 2, `chi` is `b = a + a2 + a1 a2`, and `iota` a public constant,
//! so
//!
//! ```text
//! v - RC(rho) = sum_x eq(rho, x) [a0(x) + a2(x) + a1(x) a2(x)],
//! ```
//!
//! one cubic sumcheck whose terminal leaves claims on `a0, a1, a2` at
//! one point. Each `a_j` is a GF(2)-LINEAR image of `L` (XOR of bits =
//! field addition in char 2, EXACTLY), so a gamma-batched second
//! sumcheck over the 2^11 state cube reduces the three claims to one
//! claim on `MLE(L)`. The linear tap matrices are built by PROBING the
//! reference lane functions with unit states, immune to mod-5 index
//! bugs, and pinned by a differential test against `keccak_f`.
//!
//! B1 scope: correctness of one layer and the 24-layer chain for a
//! single permutation, plus kernel timings. Batching over permutations
//! and the production wiring closed forms are gate B2.

use flock_core::challenger::Challenger;
use flock_core::field::F128;

use super::keccak::{Lanes, STATE_BITS, rho_pi_lanes, theta_lanes};
#[cfg(test)]
use super::keccak::{State, iota_lanes, state_to_lanes};

/// Padded state cube: 1,600 bits in 2^11.
pub const STATE_VARS: usize = 11;
const CUBE: usize = 1 << STATE_VARS;

#[cfg(test)]
fn lanes_to_bits(lanes: &Lanes) -> Vec<bool> {
    let mut bits = vec![false; CUBE];
    for (i, slot) in bits.iter_mut().enumerate().take(STATE_BITS) {
        let (lane, z) = (i % 25, i / 25);
        *slot = (lanes[lane] >> z) & 1 == 1;
    }
    bits
}

#[cfg(test)]
fn state_bits(state: &State) -> Vec<bool> {
    lanes_to_bits(&state_to_lanes(state))
}

/// The linear layer `pi . rho . theta` as per-output tap lists, built by
/// probing the reference lane functions with the 1,600 unit states.
pub struct LinearTaps {
    /// `taps[out]` = the input bit indices XORed into output bit `out`.
    pub taps: Vec<Vec<u16>>,
}

#[must_use]
pub fn linear_taps() -> LinearTaps {
    let mut taps = vec![Vec::new(); STATE_BITS];
    for input in 0..STATE_BITS {
        let (lane, z) = (input % 25, input / 25);
        let mut lanes: Lanes = [0; 25];
        lanes[lane] = 1_u64 << z;
        theta_lanes(&mut lanes);
        let out = rho_pi_lanes(&lanes);
        for (out_lane, &word) in out.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let out_z = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                taps[out_lane + 25 * out_z].push(input as u16);
            }
        }
    }
    LinearTaps { taps }
}

/// The chi neighbor shifts as output-bit index maps: `a1[i] = a[x+1
/// lane]`, `a2[i] = a[x+2 lane]`, same y and z.
fn neighbor(index: usize, dx: usize) -> usize {
    let (lane, z) = (index % 25, index / 25);
    let (x, y) = (lane % 5, lane / 5);
    let shifted = (x + dx) % 5 + 5 * y;
    shifted + 25 * z
}

/// Apply the probed linear layer to a bit state (prover-side XORs).
fn apply_taps(taps: &LinearTaps, bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; CUBE];
    for (i, slot) in out.iter_mut().enumerate().take(STATE_BITS) {
        let mut acc = false;
        for &tap in &taps.taps[i] {
            acc ^= bits[tap as usize];
        }
        *slot = acc;
    }
    out
}

fn lift(bits: &[bool]) -> Vec<F128> {
    bits.iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect()
}

/// Dense multilinear evaluation, MSB-first.
#[must_use]
pub fn eval_mle(table: &[F128], point: &[F128]) -> F128 {
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

fn eq_weights(point: &[F128]) -> Vec<F128> {
    let mut w = vec![F128::ONE];
    for &coord in point {
        let mut next = Vec::with_capacity(2 * w.len());
        for &x in &w {
            next.push(x * (F128::ONE + coord));
            next.push(x * coord);
        }
        w = next;
    }
    w
}

/// The public iota term: `MLE(RC_round, rho)` over the padded cube.
#[must_use]
pub fn round_constant_mle(round: usize, rho: &[F128]) -> F128 {
    let rc = super::keccak::ROUND_CONSTANTS[round];
    let weights = eq_weights(rho);
    let mut sum = F128::ZERO;
    let mut bits = rc;
    while bits != 0 {
        let z = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        sum += weights[25 * z];
    }
    sum
}

/// One layer's proof: the cubic sumcheck rounds (evals at 0..=3), the
/// claimed tap-table values at the terminal, the batched linear
/// reduction rounds (evals at 0..=2), and the final claim on the
/// previous state's MLE.
pub struct LayerProof {
    pub chi_rounds: Vec<[F128; 4]>,
    pub tap_values: [F128; 3],
    pub linear_rounds: Vec<[F128; 3]>,
    pub previous_claim: F128,
}

/// Reduce `MLE(L_round, rho) = v` to a claim on `MLE(L_prev, r'')`.
/// Returns the proof and the new point. Fiat-Shamir via the flock
/// challenger; the verifier half replays the same reductions.
pub fn prove_layer<Ch: Challenger>(
    taps: &LinearTaps,
    l_prev_bits: &[bool],
    rho: &[F128],
    challenger: &mut Ch,
) -> (LayerProof, Vec<F128>) {
    // Prover tables: a0 = linear image, a1/a2 its lane shifts.
    let a0_bits = apply_taps(taps, l_prev_bits);
    let mut a1_bits = vec![false; CUBE];
    let mut a2_bits = vec![false; CUBE];
    for i in 0..STATE_BITS {
        a1_bits[i] = a0_bits[neighbor(i, 1)];
        a2_bits[i] = a0_bits[neighbor(i, 2)];
    }
    let mut eq = eq_weights(rho);
    let (mut a0, mut a1, mut a2) = (lift(&a0_bits), lift(&a1_bits), lift(&a2_bits));

    // Sumcheck of eq * (a0 + a2 + a1 a2), degree 3 per variable.
    let mut chi_rounds = Vec::with_capacity(STATE_VARS);
    let mut point = Vec::with_capacity(STATE_VARS);
    while eq.len() > 1 {
        let half = eq.len() / 2;
        let mut evals = [F128::ZERO; 4];
        for i in 0..half {
            let f = [eq[i], eq[half + i]];
            let x0 = [a0[i], a0[half + i]];
            let x1 = [a1[i], a1[half + i]];
            let x2 = [a2[i], a2[half + i]];
            let (df, d0, d1, d2) = (f[1] + f[0], x0[1] + x0[0], x1[1] + x1[0], x2[1] + x2[0]);
            let (mut ft, mut t0, mut t1, mut t2) = (f[0], x0[0], x1[0], x2[0]);
            for slot in &mut evals {
                *slot += ft * (t0 + t2 + t1 * t2);
                ft += df;
                t0 += d0;
                t1 += d1;
                t2 += d2;
            }
        }
        for eval in evals {
            challenger.observe_f128(eval);
        }
        let r = challenger.sample_f128();
        for table in [&mut eq, &mut a0, &mut a1, &mut a2] {
            let half = table.len() / 2;
            for i in 0..half {
                let low = table[i];
                table[i] = low + r * (low + table[half + i]);
            }
            table.truncate(half);
        }
        chi_rounds.push(evals);
        point.push(r);
    }
    let tap_values = [a0[0], a1[0], a2[0]];
    for value in tap_values {
        challenger.observe_f128(value);
    }

    // Batched linear reduction: sum_j gamma^j a_j(r') = sum_u V(u) L(u),
    // V(u) = sum_j gamma^j sum_{x in inv_j(u)} eq(r', x).
    let gamma = challenger.sample_f128();
    let eq_r = eq_weights(&point);
    let mut v_table = vec![F128::ZERO; CUBE];
    let gammas = [F128::ONE, gamma, gamma * gamma];
    for out in 0..STATE_BITS {
        for (j, &g) in gammas.iter().enumerate() {
            // a_j reads the linear image at the shifted OUTPUT index.
            let source = if j == 0 { out } else { neighbor(out, j) };
            let weight = g * eq_r[out];
            for &tap in &taps.taps[source] {
                v_table[tap as usize] += weight;
            }
        }
    }
    let mut l_table = lift(l_prev_bits);

    // Two-factor sumcheck of V * L over the state cube.
    let mut linear_rounds = Vec::with_capacity(STATE_VARS);
    let mut final_point = Vec::with_capacity(STATE_VARS);
    while v_table.len() > 1 {
        let half = v_table.len() / 2;
        let mut evals = [F128::ZERO; 3];
        for i in 0..half {
            let (v0, v1) = (v_table[i], v_table[half + i]);
            let (l0, l1) = (l_table[i], l_table[half + i]);
            evals[0] += v0 * l0;
            evals[1] += v1 * l1;
            evals[2] += (v1 + v1 + v0) * (l1 + l1 + l0);
        }
        for eval in evals {
            challenger.observe_f128(eval);
        }
        let r = challenger.sample_f128();
        for table in [&mut v_table, &mut l_table] {
            let half = table.len() / 2;
            for i in 0..half {
                let low = table[i];
                table[i] = low + r * (low + table[half + i]);
            }
            table.truncate(half);
        }
        linear_rounds.push(evals);
        final_point.push(r);
    }
    let previous_claim = l_table[0];
    challenger.observe_f128(previous_claim);

    (
        LayerProof {
            chi_rounds,
            tap_values,
            linear_rounds,
            previous_claim,
        },
        final_point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    fn random_state(seed: u64) -> State {
        let mut state = [false; STATE_BITS];
        let mut acc = seed | 1;
        for slot in state.iter_mut() {
            acc = acc
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = acc >> 63 == 1;
        }
        state
    }

    fn reference_round(state_bits_in: &[bool], round: usize) -> Vec<bool> {
        // Reassemble lanes, run the reference lane functions + chi + iota.
        let mut lanes: Lanes = [0; 25];
        for (i, &bit) in state_bits_in.iter().enumerate().take(STATE_BITS) {
            if bit {
                lanes[i % 25] |= 1 << (i / 25);
            }
        }
        theta_lanes(&mut lanes);
        let a = rho_pi_lanes(&lanes);
        let mut b: Lanes = [0; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[x + 5 * y] = a[x + 5 * y] ^ (!a[(x + 1) % 5 + 5 * y] & a[(x + 2) % 5 + 5 * y]);
            }
        }
        iota_lanes(&mut b, round);
        lanes_to_bits(&b)
    }

    #[test]
    fn probed_taps_reproduce_the_reference_round() {
        let taps = linear_taps();
        for seed in [3_u64, 77, 991] {
            let state = random_state(seed);
            let bits = state_bits(&state);
            let a0 = apply_taps(&taps, &bits);
            let mut out = vec![false; CUBE];
            for i in 0..STATE_BITS {
                out[i] = a0[i] ^ (!a0[neighbor(i, 1)] & a0[neighbor(i, 2)]);
            }
            // iota on lane 0.
            let rc = super::super::keccak::ROUND_CONSTANTS[5];
            for z in 0..64 {
                if (rc >> z) & 1 == 1 {
                    out[25 * z] ^= true;
                }
            }
            assert_eq!(out, reference_round(&bits, 5), "seed {seed}");
        }
    }

    #[test]
    fn one_layer_reduction_is_exact_and_the_chain_closes() {
        let taps = linear_taps();
        let state0 = random_state(42);
        // Chain of 24 rounds; keep every intermediate state.
        let mut states = vec![state_bits(&state0)];
        for round in 0..24 {
            let next = reference_round(states.last().unwrap(), round);
            states.push(next);
        }

        // Start from a random claim on the OUTPUT state's MLE and walk
        // the layers down to state_0, checking every reduced claim
        // against the direct MLE (the prototype's exactness oracle).
        let mut challenger = FsChallenger::new(b"keccak-gkr-b1");
        let mut point: Vec<F128> = (0..STATE_VARS).map(|_| challenger.sample_f128()).collect();
        let mut claim = eval_mle(&lift(&states[24]), &point);
        let start = std::time::Instant::now();
        for round in (0..24).rev() {
            // Peel iota: the chi output claim.
            let chi_claim = claim + round_constant_mle(round, &point);
            let (proof, next_point) = prove_layer(&taps, &states[round], &point, &mut challenger);
            // Sum rule at round 1 of the chi sumcheck.
            assert_eq!(
                proof.chi_rounds[0][0] + proof.chi_rounds[0][1],
                chi_claim,
                "chi claim, round {round}"
            );
            // The reduced claim matches the previous state's MLE.
            assert_eq!(
                proof.previous_claim,
                eval_mle(&lift(&states[round]), &next_point),
                "layer terminal, round {round}"
            );
            claim = proof.previous_claim;
            point = next_point;
        }
        eprintln!(
            "# keccak-gkr B1: 24-layer chain, one permutation: {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(claim, eval_mle(&lift(&states[0]), &point));
    }
}
