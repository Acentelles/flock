//! The stable-compaction scatter argument, prototype (AERIE-ADAPTER.md
//! Section 3.3).
//!
//! Binds the dense `Z_H` output table (512 packed targets per record) to
//! the gated slot outputs of `hash_to_point_slots` with ONE degree-12
//! product sumcheck and ONE extra MLE opening, instead of R1CS
//! multiplexers. The two identities that make it work, both specific to
//! characteristic 2:
//!
//! 1. `beta^count = prod_b (1 + count_b (beta^(2^b) + 1))`, LINEAR in each
//!    counter bit, so the slot side
//!    `S = sum_s gate_s * val_s(gamma) * beta^(count_s)` is a product of
//!    12 multilinear factors over the slot domain (gate, ten counter
//!    factors, the gamma-combined value).
//! 2. The dense power sum `sum_{j,c} Z[j,c] beta^j gamma^c` equals
//!    `scale * MLE_Z(x)` at the derived point
//!    `x_b = beta^(2^b) / (1 + beta^(2^b))`,
//!    `scale = prod_b (1 + beta^(2^b))` (and likewise for `gamma` over the
//!    coordinate variables), i.e. one MLE opening at a public point.
//!
//! Because XOR of 0/1 values IS the field sum in characteristic 2, the
//! MLE of any XOR-of-wires table is the F128-linear combination of the
//! wire MLEs: the symbolic counter bits and residue bits never need
//! materializing, and every factor's sumcheck terminal is a
//! public-weighted linear functional of the committed witness. This
//! prototype discharges those terminals by direct evaluation of the
//! factor tables (the packed-direct/lincheck opening layer is the
//! remaining integration); the sumcheck messages and checks themselves
//! are the real protocol.
//!
//! Soundness of the binding: if the dense table differs from the gated
//! scatter, the two sides differ as polynomials in `beta` of degree at
//! most 611 (and degree 14 in `gamma`), so a post-commitment challenge
//! pair accepts with probability about `625 / 2^128`. Stability and
//! ordering need no argument: the R1CS forces the counter to increment
//! exactly on accepted slots from a structural zero, so gated slots hit
//! indices `0..512` in order.

use flock_core::challenger::Challenger;
use flock_core::field::F128;

use super::hash_to_point_slots::{
    centering_position, gate_position, COUNTER_BITS, SLOTS, SLOT_BASE, SLOT_STRIDE,
};

/// log2 of the padded slot domain (612 slots in 1,024).
pub const SLOT_VARS: usize = 10;
/// Dense table shape: 512 output indices, 16 padded coordinate planes
/// (15 live: 14 residue bits plus the centering flag).
pub const INDEX_VARS: usize = 9;
pub const COORD_VARS: usize = 4;
pub const LIVE_COORDS: usize = 15;

/// Factors per sumcheck term: the gate, ten counter factors, the value.
pub const FACTORS: usize = 2 + COUNTER_BITS;
/// Round-polynomial degree and its evaluation points `0..=FACTORS`.
pub const ROUND_DEGREE: usize = FACTORS;

fn small(v: u64) -> F128 {
    F128 { lo: v, hi: 0 }
}

/// `base^(2^b)` for `b in 0..bits` by repeated squaring.
fn frobenius_powers(base: F128, bits: usize) -> Vec<F128> {
    let mut powers = Vec::with_capacity(bits);
    let mut current = base;
    for _ in 0..bits {
        powers.push(current);
        current = current * current;
    }
    powers
}

/// The derived MLE point and public scale for a power sum: coordinate `b`
/// is `p_b / (1 + p_b)` with `p_b = base^(2^b)`, MOST SIGNIFICANT variable
/// first to match the fold order; the scale is `prod_b (1 + p_b)`.
/// Undefined (returns None) when some `p_b = 1`.
pub fn power_sum_point(base: F128, bits: usize) -> Option<(Vec<F128>, F128)> {
    let mut point = Vec::with_capacity(bits);
    let mut scale = F128::ONE;
    for p in frobenius_powers(base, bits) {
        let denominator = F128::ONE + p;
        if denominator == F128::ZERO {
            return None;
        }
        scale = scale * denominator;
        point.push(p * denominator.inv());
    }
    // Bit b of the index is weight beta^(2^b); variable order in the MLE
    // is most significant first, so reverse.
    point.reverse();
    Some((point, scale))
}

/// Multilinear evaluation, most significant variable first (the
/// convention shared with `eval` folds throughout this module).
pub fn eval_mle(table: &[F128], point: &[F128]) -> F128 {
    assert_eq!(table.len(), 1 << point.len());
    let mut layer = table.to_vec();
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

/// The twelve factor tables over the padded slot domain, built from one
/// record block's witness bits (prototype: clear access; production reads
/// them through the packed-direct opening layer).
///
/// Padding slots get gate 0 and neutral 1 elsewhere.
pub fn factor_tables(block: &[bool], words: &[u16; SLOTS], gamma: F128, beta: F128) -> Vec<Vec<F128>> {
    let domain = 1 << SLOT_VARS;
    let beta_powers = frobenius_powers(beta, COUNTER_BITS);
    let gamma_powers: Vec<F128> = {
        let mut powers = Vec::with_capacity(LIVE_COORDS);
        let mut current = F128::ONE;
        for _ in 0..LIVE_COORDS {
            powers.push(current);
            current = current * gamma;
        }
        powers
    };

    let mut gate = vec![F128::ZERO; domain];
    let mut counter_factors = vec![vec![F128::ONE; domain]; COUNTER_BITS];
    let mut value = vec![F128::ONE; domain];
    let mut count_before = 0_u32;
    for (slot, &word) in words.iter().enumerate() {
        gate[slot] = if block[gate_position(slot)] {
            F128::ONE
        } else {
            F128::ZERO
        };
        for (bit, factor) in counter_factors.iter_mut().enumerate() {
            if (count_before >> bit) & 1 == 1 {
                factor[slot] = beta_powers[bit];
            }
        }
        // val = sum_c gamma^c * bit_c(a) + gamma^14 * u, from the committed
        // quotient wires and the centering wire.
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        let mut quotient = 0_u16;
        for i in 0..3 {
            quotient |= u16::from(block[base + 16 + i]) << i;
        }
        let residue = word.wrapping_sub(12_289_u16.wrapping_mul(quotient));
        let mut val = F128::ZERO;
        for (c, &power) in gamma_powers.iter().enumerate().take(14) {
            if (residue >> c) & 1 == 1 {
                val = val + power;
            }
        }
        if block[centering_position(slot)] {
            val = val + gamma_powers[14];
        }
        value[slot] = val;

        if u32::from(word) < 61_445 {
            count_before += 1;
        }
    }

    let mut factors = Vec::with_capacity(FACTORS);
    factors.push(gate);
    factors.extend(counter_factors);
    factors.push(value);
    factors
}

/// The dense table `Z[j, c]` a correct prover commits: the gated slot
/// outputs in counter order, bit c of the packed `(a, u)` value.
pub fn dense_table(block: &[bool], words: &[u16; SLOTS]) -> Vec<F128> {
    let mut dense = vec![F128::ZERO; 1 << (INDEX_VARS + COORD_VARS)];
    let mut index = 0_usize;
    for (slot, &word) in words.iter().enumerate() {
        if !block[gate_position(slot)] {
            continue;
        }
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        let mut quotient = 0_u16;
        for i in 0..3 {
            quotient |= u16::from(block[base + 16 + i]) << i;
        }
        let residue = word.wrapping_sub(12_289_u16.wrapping_mul(quotient));
        for c in 0..14 {
            if (residue >> c) & 1 == 1 {
                dense[(index << COORD_VARS) | c] = F128::ONE;
            }
        }
        if block[centering_position(slot)] {
            dense[(index << COORD_VARS) | 14] = F128::ONE;
        }
        index += 1;
    }
    assert_eq!(index, 512, "a full record gates exactly 512 outputs");
    dense
}

/// The dense side of the identity: `sum_{j,c} Z[j,c] beta^j gamma^c`,
/// evaluated as one MLE opening at the derived point, times the scales.
pub fn dense_power_sum(dense: &[F128], beta: F128, gamma: F128) -> Option<F128> {
    let (index_point, index_scale) = power_sum_point(beta, INDEX_VARS)?;
    let (coord_point, coord_scale) = power_sum_point(gamma, COORD_VARS)?;
    let point = [index_point, coord_point].concat();
    Some(index_scale * coord_scale * eval_mle(dense, &point))
}

/// One round message: the round polynomial's evaluations at `0..=12`.
pub type RoundEvals = Vec<F128>;

#[derive(Clone, Debug)]
pub struct ScatterProof {
    pub claim: F128,
    pub rounds: Vec<RoundEvals>,
}

/// Lagrange interpolation at `r` through `(t, evals[t])`, `t = 0..=12`.
fn interpolate(evals: &[F128], r: F128) -> F128 {
    let points: Vec<F128> = (0..evals.len() as u64).map(small).collect();
    let mut result = F128::ZERO;
    for (t, &eval) in evals.iter().enumerate() {
        let mut numerator = F128::ONE;
        let mut denominator = F128::ONE;
        for (u, &p_u) in points.iter().enumerate() {
            if u != t {
                numerator = numerator * (r + p_u);
                denominator = denominator * (points[t] + p_u);
            }
        }
        result = result + eval * numerator * denominator.inv();
    }
    result
}

/// Prove `claim = sum_s prod_f factors[f][s]` over the slot domain.
pub fn prove<Ch: Challenger>(
    mut factors: Vec<Vec<F128>>,
    challenger: &mut Ch,
) -> (ScatterProof, Vec<F128>) {
    assert_eq!(factors.len(), FACTORS);
    let claim = (0..factors[0].len()).fold(F128::ZERO, |sum, s| {
        sum + factors.iter().map(|f| f[s]).fold(F128::ONE, |p, v| p * v)
    });
    challenger.observe_label(b"aerie-scatter-v0");
    challenger.observe_f128(claim);

    let mut rounds = Vec::with_capacity(SLOT_VARS);
    let mut point = Vec::with_capacity(SLOT_VARS);
    while factors[0].len() > 1 {
        let half = factors[0].len() / 2;
        let mut evals = vec![F128::ZERO; ROUND_DEGREE + 1];
        for (t, eval) in evals.iter_mut().enumerate() {
            let p_t = small(t as u64);
            for i in 0..half {
                let mut term = F128::ONE;
                for factor in &factors {
                    let low = factor[i];
                    term = term * (low + p_t * (low + factor[half + i]));
                }
                *eval = *eval + term;
            }
        }
        challenger.observe_f128_slice(&evals);
        let challenge = challenger.sample_f128();
        for factor in &mut factors {
            for i in 0..half {
                let low = factor[i];
                factor[i] = low + challenge * (low + factor[half + i]);
            }
            factor.truncate(half);
        }
        rounds.push(evals);
        point.push(challenge);
    }
    (ScatterProof { claim, rounds }, point)
}

/// Verify the round messages; the caller checks the returned terminal
/// against its own factor evaluations at the returned point.
pub fn verify<Ch: Challenger>(
    proof: &ScatterProof,
    challenger: &mut Ch,
) -> Result<(Vec<F128>, F128), &'static str> {
    if proof.rounds.len() != SLOT_VARS {
        return Err("scatter sumcheck has the wrong round count");
    }
    challenger.observe_label(b"aerie-scatter-v0");
    challenger.observe_f128(proof.claim);
    let mut running = proof.claim;
    let mut point = Vec::with_capacity(SLOT_VARS);
    for evals in &proof.rounds {
        if evals.len() != ROUND_DEGREE + 1 {
            return Err("scatter round has the wrong degree");
        }
        if evals[0] + evals[1] != running {
            return Err("scatter round does not sum to the running claim");
        }
        challenger.observe_f128_slice(evals);
        let challenge = challenger.sample_f128();
        running = interpolate(evals, challenge);
        point.push(challenge);
    }
    Ok((point, running))
}

#[cfg(test)]
mod tests {
    use flock_core::challenger::FsChallenger;

    use super::super::hash_to_point_slots::build_block_witness;
    use super::*;

    fn record() -> ([u16; SLOTS], Vec<bool>) {
        let mut words = [0_u16; SLOTS];
        for (index, word) in words.iter_mut().enumerate() {
            *word = (index as u16).wrapping_mul(31_337).wrapping_add(77);
        }
        let (block, _counter) = build_block_witness(&words);
        (words, block)
    }

    #[test]
    fn the_power_sum_identity_holds() {
        // sum_j c_j beta^j == scale * MLE_c(derived point), checked directly
        // on a small table.
        let beta = F128 {
            lo: 0x1234_5678_9abc,
            hi: 0x42,
        };
        let table: Vec<F128> = (0..512_u64).map(|j| small(j.wrapping_mul(97) ^ 5)).collect();
        let mut direct = F128::ZERO;
        let mut power = F128::ONE;
        for &c in &table {
            direct = direct + c * power;
            power = power * beta;
        }
        let (point, scale) = power_sum_point(beta, 9).expect("beta powers != 1");
        assert_eq!(direct, scale * eval_mle(&table, &point));
    }

    #[test]
    fn slot_and_dense_sides_agree_on_an_honest_record() {
        let (words, block) = record();
        let beta = F128 {
            lo: 0xdead_beef_1234,
            hi: 0x9,
        };
        let gamma = F128 {
            lo: 0x5555_aaaa,
            hi: 0x77,
        };
        let factors = factor_tables(&block, &words, gamma, beta);
        let slot_side = (0..1 << SLOT_VARS).fold(F128::ZERO, |sum, s| {
            sum + factors.iter().map(|f| f[s]).fold(F128::ONE, |p, v| p * v)
        });
        let dense = dense_table(&block, &words);
        let dense_side = dense_power_sum(&dense, beta, gamma).expect("derived point defined");
        assert_eq!(slot_side, dense_side);
    }

    #[test]
    fn a_tampered_dense_table_breaks_the_identity() {
        let (words, block) = record();
        let beta = small(0x1111_2222);
        let gamma = small(0x3333);
        let factors = factor_tables(&block, &words, gamma, beta);
        let slot_side = (0..1 << SLOT_VARS).fold(F128::ZERO, |sum, s| {
            sum + factors.iter().map(|f| f[s]).fold(F128::ONE, |p, v| p * v)
        });
        let mut dense = dense_table(&block, &words);
        // Swap two adjacent outputs: order matters.
        for c in 0..16 {
            dense.swap((100 << COORD_VARS) | c, (101 << COORD_VARS) | c);
        }
        let dense_side = dense_power_sum(&dense, beta, gamma).expect("defined");
        assert_ne!(slot_side, dense_side);
    }

    #[test]
    fn scatter_sumcheck_roundtrips_and_matches_the_dense_side() {
        let (words, block) = record();
        let mut prover_challenger = FsChallenger::new(b"aerie-scatter-test");
        // Challenges sampled post-commitment in the real transcript; here
        // they are fixed public test values.
        let beta = F128 {
            lo: 0xabcd_ef01_2345,
            hi: 0x1,
        };
        let gamma = F128 {
            lo: 0x9876_5432,
            hi: 0x2,
        };
        let factors = factor_tables(&block, &words, gamma, beta);
        let (proof, prover_point) = prove(factors.clone(), &mut prover_challenger);

        // The claim equals the dense side: the scatter binds the tables.
        let dense = dense_table(&block, &words);
        assert_eq!(
            proof.claim,
            dense_power_sum(&dense, beta, gamma).expect("defined")
        );

        let mut verifier_challenger = FsChallenger::new(b"aerie-scatter-test");
        let (point, terminal) =
            verify(&proof, &mut verifier_challenger).expect("honest scatter verifies");
        assert_eq!(point, prover_point);
        // Terminal discharge: the product of the factor MLEs at the point
        // (prototype: direct evaluation; production: packed-direct opening).
        let expected = factors
            .iter()
            .map(|factor| eval_mle(factor, &point))
            .fold(F128::ONE, |p, v| p * v);
        assert_eq!(terminal, expected);

        // A tampered round rejects or mismatches the terminal.
        let mut wrong = proof.clone();
        wrong.rounds[4][2] = wrong.rounds[4][2] + F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-scatter-test");
        match verify(&wrong, &mut fresh) {
            Err(_) => {}
            Ok((tampered_point, tampered_terminal)) => {
                let recomputed = factors
                    .iter()
                    .map(|factor| eval_mle(factor, &tampered_point))
                    .fold(F128::ONE, |p, v| p * v);
                assert_ne!(tampered_terminal, recomputed);
            }
        }
    }
}
