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
    COUNTER_BITS, PLANE, SLOTS, borrow_position, centering_position, gate_position,
    increment_position, plane_of, quotient_position, word_position,
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
        scale *= denominator;
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
/// them through the opening layer via [`prove_discharge`]).
///
/// The padding conventions are CANONICAL, matching the plane formulas the
/// discharge reconstructs: the gate and value are zero at padded slots
/// (their planes are forced-zero there), and the counter factors freeze at
/// the final count (the prefix parity of the zero-extended h planes).
pub fn factor_tables(
    block: &[bool],
    words: &[u16; SLOTS],
    gamma: F128,
    beta: F128,
) -> Vec<Vec<F128>> {
    let domain = 1 << SLOT_VARS;
    let beta_powers = frobenius_powers(beta, COUNTER_BITS);
    let gamma_powers: Vec<F128> = {
        let mut powers = Vec::with_capacity(LIVE_COORDS);
        let mut current = F128::ONE;
        for _ in 0..LIVE_COORDS {
            powers.push(current);
            current *= gamma;
        }
        powers
    };

    let mut gate = vec![F128::ZERO; domain];
    let mut counter_factors = vec![vec![F128::ONE; domain]; COUNTER_BITS];
    let mut value = vec![F128::ZERO; domain];
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
        let mut quotient = 0_u16;
        for i in 0..3 {
            quotient |= u16::from(block[quotient_position(slot, i)]) << i;
        }
        let residue = word.wrapping_sub(12_289_u16.wrapping_mul(quotient));
        let mut val = F128::ZERO;
        for (c, &power) in gamma_powers.iter().enumerate().take(14) {
            if (residue >> c) & 1 == 1 {
                val += power;
            }
        }
        if block[centering_position(slot)] {
            val += gamma_powers[14];
        }
        value[slot] = val;

        if u32::from(word) < 61_445 {
            count_before += 1;
        }
    }
    // Padding: the counter freezes (zero-extended h planes), so the
    // counter factors keep the final count's bits.
    for slot in SLOTS..domain {
        for (bit, factor) in counter_factors.iter_mut().enumerate() {
            if (count_before >> bit) & 1 == 1 {
                factor[slot] = beta_powers[bit];
            }
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
        let mut quotient = 0_u16;
        for i in 0..3 {
            quotient |= u16::from(block[quotient_position(slot, i)]) << i;
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
                numerator *= r + p_u;
                denominator *= points[t] + p_u;
            }
        }
        result += eval * numerator * denominator.inv();
    }
    result
}

/// Prove `claim = sum_s prod_f factors[f][s]` over a power-of-two domain,
/// for any number of multilinear factors (round degree = factor count).
pub fn prove<Ch: Challenger>(
    mut factors: Vec<Vec<F128>>,
    challenger: &mut Ch,
) -> (ScatterProof, Vec<F128>) {
    use rayon::prelude::*;
    let degree = factors.len();
    assert!(degree < 32, "stack buffer sized for degree < 32");
    let claim = (0..factors[0].len())
        .into_par_iter()
        .map(|s| factors.iter().map(|f| f[s]).fold(F128::ONE, |p, v| p * v))
        .reduce(|| F128::ZERO, |a, b| a + b);
    challenger.observe_label(b"aerie-scatter-v0");
    challenger.observe_f128(claim);

    let points: Vec<F128> = (0..=degree as u64).map(small).collect();
    let vars = factors[0].len().trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(vars);
    let mut point = Vec::with_capacity(vars);
    let mut first_round = factors[0].len() > 1;
    while factors[0].len() > 1 {
        let half = factors[0].len() / 2;
        if first_round {
            first_round = false;
            // Round-1 bit kernel: before any fold the factor tables are
            // largely 0/1-valued (slot bits, gates), so a pair
            // (low, diff = low + high) classifies as (0,0) -> the term
            // is zero at EVERY evaluation point (whole element skipped
            // -- honest gated rows are mostly zero), or one of the
            // classes 1, p_t, 1 + p_t, whose product over the bit
            // factors depends only on the class COUNTS. Precomputed
            // power tables close each element in ~3 multiplications per
            // point plus its dense factors. Pure regrouping of the same
            // field products: round messages are byte-identical.
            let max_pow = degree + 1;
            let pow_of = |base_at: &dyn Fn(usize) -> F128| -> Vec<Vec<F128>> {
                (0..=degree)
                    .map(|t| {
                        let base = base_at(t);
                        let mut row = Vec::with_capacity(max_pow + 1);
                        row.push(F128::ONE);
                        for k in 0..max_pow {
                            let prev = row[k];
                            row.push(prev * base);
                        }
                        row
                    })
                    .collect()
            };
            // Class (1,0): value 1 at every t (no table needed).
            let pow_p = pow_of(&|t| points[t]); // class (0,1): p_t
            let pow_1p = pow_of(&|t| F128::ONE + points[t]); // class (1,1)
            let evals = (0..half)
                .into_par_iter()
                .fold(
                    || vec![F128::ZERO; degree + 1],
                    |mut acc, i| {
                        let mut n_p = 0usize;
                        let mut n_1p = 0usize;
                        let mut dense: [(F128, F128); 32] = [(F128::ZERO, F128::ZERO); 32];
                        let mut n_dense = 0usize;
                        for factor in &factors {
                            let low = factor[i];
                            let diff = low + factor[half + i];
                            let low_is_bit = low == F128::ZERO || low == F128::ONE;
                            let diff_is_bit = diff == F128::ZERO || diff == F128::ONE;
                            if low_is_bit && diff_is_bit {
                                match (low == F128::ONE, diff == F128::ONE) {
                                    (false, false) => return acc, // zero term
                                    (true, false) => {}
                                    (false, true) => n_p += 1,
                                    (true, true) => n_1p += 1,
                                }
                            } else {
                                dense[n_dense] = (low, diff);
                                n_dense += 1;
                            }
                        }
                        for (t, slot) in acc.iter_mut().enumerate() {
                            let mut term = pow_p[t][n_p] * pow_1p[t][n_1p];
                            for &(low, diff) in &dense[..n_dense] {
                                term *= low + points[t] * diff;
                            }
                            *slot += term;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![F128::ZERO; degree + 1],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(&b) {
                            *x += *y;
                        }
                        a
                    },
                );
            challenger.observe_f128_slice(&evals);
            let challenge = challenger.sample_f128();
            factors.par_iter_mut().for_each(|factor| {
                for i in 0..half {
                    let low = factor[i];
                    factor[i] = low + challenge * (low + factor[half + i]);
                }
                factor.truncate(half);
            });
            rounds.push(evals);
            point.push(challenge);
            continue;
        }
        // One pass over the domain: load each factor pair once and extend
        // it to every evaluation point, instead of degree + 1 passes.
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || vec![F128::ZERO; degree + 1],
                |mut acc, i| {
                    let mut terms = [F128::ONE; 32];
                    let terms = &mut terms[..degree + 1];
                    for factor in &factors {
                        let low = factor[i];
                        let diff = low + factor[half + i];
                        for (t, term) in terms.iter_mut().enumerate() {
                            *term *= low + points[t] * diff;
                        }
                    }
                    for (t, term) in terms.iter().enumerate() {
                        acc[t] += *term;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F128::ZERO; degree + 1],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += *y;
                    }
                    a
                },
            );
        challenger.observe_f128_slice(&evals);
        let challenge = challenger.sample_f128();
        factors.par_iter_mut().for_each(|factor| {
            for i in 0..half {
                let low = factor[i];
                factor[i] = low + challenge * (low + factor[half + i]);
            }
            factor.truncate(half);
        });
        rounds.push(evals);
        point.push(challenge);
    }
    (ScatterProof { claim, rounds }, point)
}

/// Verify the round messages for a `degree`-factor, `vars`-variable
/// product sumcheck; the caller checks the returned terminal against its
/// own factor evaluations at the returned point.
pub fn verify<Ch: Challenger>(
    proof: &ScatterProof,
    vars: usize,
    degree: usize,
    challenger: &mut Ch,
) -> Result<(Vec<F128>, F128), &'static str> {
    if proof.rounds.len() != vars {
        return Err("scatter sumcheck has the wrong round count");
    }
    challenger.observe_label(b"aerie-scatter-v0");
    challenger.observe_f128(proof.claim);
    let mut running = proof.claim;
    let mut point = Vec::with_capacity(vars);
    for evals in &proof.rounds {
        if evals.len() != degree + 1 {
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

// ---------------------------------------------------------------------------
// Terminal discharge: every factor MLE at the scatter point reduces to
// plane openings (sub-cubes of the committed witness) plus one 2-factor
// sumcheck for the counter prefix parities, whose weight is the
// transparent greater-than multilinear.
// ---------------------------------------------------------------------------

/// The greater-than multilinear `GT(x, y) = sum_{s > s'} eq(x, s) eq(y, s')`
/// extended multilinearly, evaluated in `O(n)`:
/// `sum_i (prod_{j<i} (x_j y_j + (1+x_j)(1+y_j))) * x_i * (1 + y_i)`,
/// most significant variable first, characteristic 2.
pub fn gt_mle(x: &[F128], y: &[F128]) -> F128 {
    assert_eq!(x.len(), y.len());
    let mut prefix = F128::ONE;
    let mut sum = F128::ZERO;
    for (&xi, &yi) in x.iter().zip(y) {
        sum += prefix * xi * (F128::ONE + yi);
        prefix *= xi * yi + (F128::ONE + xi) * (F128::ONE + yi);
    }
    sum
}

/// Equality weight table for a point, most significant variable first.
fn eq_table(point: &[F128]) -> Vec<F128> {
    let mut weights = vec![F128::ONE];
    for &r in point {
        let mut next = Vec::with_capacity(2 * weights.len());
        for &w in &weights {
            next.push(w * (F128::ONE + r));
            next.push(w * r);
        }
        weights = next;
    }
    weights
}

/// Prover-side weight table `w_r(s') = sum_{s > s'} eq(r, s)`: the reverse
/// suffix sums of the equality table.
fn weight_table(r: &[F128]) -> Vec<F128> {
    let eq = eq_table(r);
    let mut weights = vec![F128::ZERO; eq.len()];
    let mut suffix = F128::ZERO;
    for s in (0..eq.len() - 1).rev() {
        suffix += eq[s + 1];
        weights[s] = suffix;
    }
    weights
}

/// The discharge messages: the claimed `cnt_b` prefix-parity MLEs at the
/// scatter point, and the batched 2-factor sumcheck backing them.
#[derive(Clone, Debug)]
pub struct DischargeProof {
    pub counter_claims: Vec<F128>,
    pub counter_sumcheck: ScatterProof,
}

/// The plane whose prefix parity is counter bit `bit`: the increment INTO
/// bit b at a slot is the accept flag for b = 0 and the carry OUT of bit
/// b - 1 (the stored product plane H_{b-1}) for b >= 1.
fn counter_source_position(slot: usize, bit: usize) -> usize {
    if bit == 0 {
        super::hash_to_point_slots::accept_position(slot)
    } else {
        increment_position(slot, bit - 1)
    }
}

fn counter_source_table(block: &[bool], bit: usize) -> Vec<F128> {
    (0..PLANE)
        .map(|slot| {
            if block[counter_source_position(slot, bit)] {
                F128::ONE
            } else {
                F128::ZERO
            }
        })
        .collect()
}

/// Prove the counter prefix-parity claims at the scatter point `r`.
///
/// The prover sends the ten claimed `cnt_b` MLE values; the transcript
/// samples a combiner `rho`, and one 2-factor sumcheck proves their
/// rho-combination as `<w_r, sum_b rho^b h_b>`. Everything else the
/// verifier needs is plane openings at `r` itself.
pub fn prove_discharge<Ch: Challenger>(
    block: &[bool],
    r: &[F128],
    challenger: &mut Ch,
) -> DischargeProof {
    let weights = weight_table(r);
    let h_tables: Vec<Vec<F128>> = (0..COUNTER_BITS)
        .map(|bit| counter_source_table(block, bit))
        .collect();
    let counter_claims: Vec<F128> = h_tables
        .iter()
        .map(|h| {
            weights
                .iter()
                .zip(h)
                .fold(F128::ZERO, |sum, (&w, &v)| sum + w * v)
        })
        .collect();
    challenger.observe_label(b"aerie-scatter-discharge-v0");
    challenger.observe_f128_slice(&counter_claims);
    let rho = challenger.sample_f128();

    let mut combined = vec![F128::ZERO; PLANE];
    let mut power = F128::ONE;
    for h in &h_tables {
        for (slot, &v) in h.iter().enumerate() {
            combined[slot] += power * v;
        }
        power *= rho;
    }
    let (counter_sumcheck, _point) = prove(vec![weights, combined], challenger);
    DischargeProof {
        counter_claims,
        counter_sumcheck,
    }
}

/// Verify the discharge and reconstruct the scatter terminal from plane
/// openings, checking it against `terminal`.
///
/// `open(plane, point)` is the authenticated sub-cube opening oracle: the
/// witness MLE at `(plane bits, point)`. The prototype backs it with
/// direct evaluation; production backs it with the batched PCS opening.
#[allow(clippy::too_many_arguments)]
pub fn verify_discharge<Ch: Challenger>(
    proof: &DischargeProof,
    r: &[F128],
    beta: F128,
    gamma: F128,
    terminal: F128,
    open: &dyn Fn(usize, &[F128]) -> F128,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    if proof.counter_claims.len() != COUNTER_BITS {
        return Err("discharge has the wrong claim count");
    }
    challenger.observe_label(b"aerie-scatter-discharge-v0");
    challenger.observe_f128_slice(&proof.counter_claims);
    let rho = challenger.sample_f128();

    // The batched sumcheck's claim must combine the individual claims.
    let mut combined_claim = F128::ZERO;
    let mut power = F128::ONE;
    for &claim in &proof.counter_claims {
        combined_claim += power * claim;
        power *= rho;
    }
    if proof.counter_sumcheck.claim != combined_claim {
        return Err("discharge claims do not combine to the sumcheck claim");
    }
    let (point, sub_terminal) = verify(&proof.counter_sumcheck, SLOT_VARS, 2, challenger)?;
    // Terminal: transparent GT weight times the rho-combined h opening.
    let mut h_combined = F128::ZERO;
    let mut power = F128::ONE;
    for bit in 0..COUNTER_BITS {
        h_combined += power * open(plane_of(counter_source_position(0, bit)), &point);
        power *= rho;
    }
    if sub_terminal != gt_mle(r, &point) * h_combined {
        return Err("discharge sumcheck terminal does not match the openings");
    }

    // Reconstruct the scatter terminal from openings at r and the claims.
    let gate = open(plane_of(gate_position(0)), r);
    let beta_powers = frobenius_powers(beta, COUNTER_BITS);
    let mut product = gate;
    for (bit, &claim) in proof.counter_claims.iter().enumerate() {
        product *= F128::ONE + (beta_powers[bit] + F128::ONE) * claim;
    }
    // val-hat = sum_{c<14} gamma^c (x_c + M_c + borrow_c) + gamma^14 u,
    // with M_0..2 = q_0..2, M_12 = q_0, M_13 = q_0 + q_1 (t_0, t_1), and
    // borrow_c = sum_{i<c} g_i, all plane openings at r.
    let x_open: Vec<F128> = (0..14)
        .map(|c| open(plane_of(word_position(0, c)), r))
        .collect();
    let q_open: Vec<F128> = (0..3)
        .map(|c| open(plane_of(quotient_position(0, c)), r))
        .collect();
    let g_open: Vec<F128> = (0..14)
        .map(|i| open(plane_of(borrow_position(0, i)), r))
        .collect();
    let u_open = open(plane_of(centering_position(0)), r);
    let mut gamma_power = F128::ONE;
    let mut val = F128::ZERO;
    let mut borrow = F128::ZERO;
    for c in 0..14 {
        let m_c = match c {
            0..=2 => q_open[c],
            12 => q_open[0],
            13 => q_open[0] + q_open[1],
            _ => F128::ZERO,
        };
        val += gamma_power * (x_open[c] + m_c + borrow);
        borrow += g_open[c];
        gamma_power *= gamma;
    }
    val += gamma_power * u_open;
    product *= val;

    if product != terminal {
        return Err("the reconstructed scatter terminal does not match");
    }
    Ok(())
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
        let table: Vec<F128> = (0..512_u64)
            .map(|j| small(j.wrapping_mul(97) ^ 5))
            .collect();
        let mut direct = F128::ZERO;
        let mut power = F128::ONE;
        for &c in &table {
            direct += c * power;
            power *= beta;
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

    /// Clear-model sub-cube opening oracle: the witness MLE of one block
    /// at `(plane bits, point)` equals the plane table's MLE at `point`.
    fn plane_open(block: &[bool]) -> impl Fn(usize, &[F128]) -> F128 + '_ {
        move |plane, point| {
            let table: Vec<F128> = (0..PLANE)
                .map(|slot| {
                    if block[plane * PLANE + slot] {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            eval_mle(&table, point)
        }
    }

    #[test]
    fn plane_openings_are_sub_cube_openings_of_the_block() {
        // The discharge's oracle is justified by the sub-cube identity:
        // the 17-variable block MLE at (plane bits, r) equals the plane
        // table MLE at r.
        let (_words, block) = record();
        let full: Vec<F128> = block
            .iter()
            .map(|&b| if b { F128::ONE } else { F128::ZERO })
            .collect();
        let r: Vec<F128> = (0..SLOT_VARS as u64).map(|i| small(i * 71 + 3)).collect();
        let open = plane_open(&block);
        for plane in [
            super::super::hash_to_point_slots::plane_of(gate_position(0)),
            super::super::hash_to_point_slots::plane_of(centering_position(0)),
            super::super::hash_to_point_slots::plane_of(increment_position(0, 4)),
        ] {
            let mut point: Vec<F128> = (0..7)
                .map(|j| small(u64::from((plane >> (6 - j)) & 1 == 1)))
                .collect();
            point.extend_from_slice(&r);
            assert_eq!(eval_mle(&full, &point), open(plane, &r), "plane {plane}");
        }
    }

    #[test]
    fn full_scatter_with_terminal_discharge_verifies() {
        use flock_core::challenger::FsChallenger;
        let (words, block) = record();
        let beta = F128 {
            lo: 0x7777_1234,
            hi: 0x3,
        };
        let gamma = F128 {
            lo: 0x2468_ace0,
            hi: 0x5,
        };
        let factors = factor_tables(&block, &words, gamma, beta);

        let mut prover_challenger = FsChallenger::new(b"aerie-scatter-full");
        let (proof, prover_point) = prove(factors, &mut prover_challenger);
        let discharge = prove_discharge(&block, &prover_point, &mut prover_challenger);

        // The scatter claim still equals the dense side.
        let dense = dense_table(&block, &words);
        assert_eq!(
            proof.claim,
            dense_power_sum(&dense, beta, gamma).expect("defined")
        );

        let mut verifier_challenger = FsChallenger::new(b"aerie-scatter-full");
        let (point, terminal) =
            verify(&proof, SLOT_VARS, FACTORS, &mut verifier_challenger).expect("scatter verifies");
        let open = plane_open(&block);
        verify_discharge(
            &discharge,
            &point,
            beta,
            gamma,
            terminal,
            &open,
            &mut verifier_challenger,
        )
        .expect("discharge verifies");

        // A tampered counter claim rejects.
        let mut wrong = discharge.clone();
        wrong.counter_claims[3] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-scatter-full");
        let (point, terminal) =
            verify(&proof, SLOT_VARS, FACTORS, &mut fresh).expect("scatter verifies");
        assert!(
            verify_discharge(&wrong, &point, beta, gamma, terminal, &open, &mut fresh).is_err()
        );
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
        let (point, terminal) = verify(&proof, SLOT_VARS, FACTORS, &mut verifier_challenger)
            .expect("honest scatter verifies");
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
        wrong.rounds[4][2] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-scatter-test");
        match verify(&wrong, SLOT_VARS, FACTORS, &mut fresh) {
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
