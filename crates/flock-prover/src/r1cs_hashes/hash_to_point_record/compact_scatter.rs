//! Exact two-round compact prefix for the thirteen-factor record scatter.
//! The counter factors remain evaluations of the committed counter bits.
//! Tables indexed by two or four input bits eliminate their early dense
//! expansion; the remaining sumcheck and its opening claims are unchanged.

use super::{frobenius, scatter, slot_residue_bits, slots, COUNTER_BITS, SLOTS, SLOT_VARS};
use flock_core::{challenger::Challenger, field::F128};
use rayon::prelude::*;

const DEGREE: usize = 3 + COUNTER_BITS;
const SLOT_COUNT: usize = 1 << SLOT_VARS;

#[derive(Clone, Copy, Default)]
struct Row {
    count: u16,
    value: u16,
    gate: bool,
}

pub(super) struct Factors {
    rows: Vec<Row>,
    record_vars: usize,
    beta: Vec<F128>,
    gamma: Vec<F128>,
    delta: Vec<F128>,
    delta_powers: Vec<F128>,
}

fn affine(low: F128, high: F128, r: F128) -> F128 {
    low + r * (low + high)
}
fn bit(value: bool) -> F128 {
    if value {
        F128::ONE
    } else {
        F128::ZERO
    }
}

impl Factors {
    pub(super) fn new(
        bit_at: impl Fn(usize) -> bool + Sync,
        record_vars: usize,
        beta: F128,
        gamma: F128,
        delta: F128,
    ) -> Self {
        assert!(record_vars >= 2);
        let records = 1 << record_vars;
        let mut rows = vec![Row::default(); records * SLOT_COUNT];
        rows.par_chunks_mut(SLOT_COUNT)
            .enumerate()
            .for_each(|(record, row)| {
                let base = record << slots::K_LOG;
                for (slot, row) in row[..SLOTS].iter_mut().enumerate() {
                    for b in 0..COUNTER_BITS {
                        row.count |=
                            u16::from(bit_at(base + slots::counter_position(slot, b))) << b;
                    }
                    for (b, set) in slot_residue_bits(&bit_at, base, slot)
                        .into_iter()
                        .enumerate()
                    {
                        row.value |= u16::from(set) << b;
                    }
                    row.value |= u16::from(bit_at(base + slots::centering_position(slot))) << 14;
                    row.gate = bit_at(base + slots::gate_position(slot));
                }
            });
        let mut gamma_lut = vec![F128::ZERO; 1 << 15];
        let mut power = F128::ONE;
        for b in 0..15 {
            for i in 0..1 << b {
                gamma_lut[i | (1 << b)] = gamma_lut[i] + power;
            }
            power *= gamma;
        }
        let mut delta_values = Vec::with_capacity(records);
        let mut power = F128::ONE;
        for _ in 0..records {
            delta_values.push(power);
            power *= delta;
        }
        Self {
            rows,
            record_vars,
            beta: frobenius(beta, COUNTER_BITS),
            gamma: gamma_lut,
            delta: delta_values,
            delta_powers: frobenius(delta, record_vars),
        }
    }

    /// Counter-bit MLE from its two/four Boolean inputs. The first bound
    /// coordinate is the most significant record coordinate.
    fn counter_value(&self, b: usize, code: usize, prior: &[F128], r: F128) -> F128 {
        let v = |i: usize| bit((code >> i) & 1 != 0);
        let value = if prior.is_empty() {
            affine(v(0), v(1), r)
        } else {
            affine(
                affine(v(0), v(1), prior[0]),
                affine(v(2), v(3), prior[0]),
                r,
            )
        };
        F128::ONE + (F128::ONE + self.beta[b]) * value
    }

    fn counter_tables(&self, prior: &[F128], r: F128) -> Vec<Vec<F128>> {
        let width = if prior.is_empty() { 5 } else { 2 };
        let inputs = if prior.is_empty() { 2 } else { 4 };
        (0..COUNTER_BITS / width)
            .map(|group| {
                (0..1 << (width * inputs))
                    .map(|code| {
                        (0..width).fold(F128::ONE, |prod, b| {
                            let bits = (0..inputs)
                                .fold(0, |bits, i| bits | (((code >> (i * width + b)) & 1) << i));
                            prod * self.counter_value(group * width + b, bits, prior, r)
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn counter_codes(&self, i: usize, round: usize) -> [usize; 5] {
        let n = self.rows.len();
        let counts = [
            self.rows[i].count,
            self.rows[i + n / 2].count,
            if round == 1 {
                self.rows[i + n / 4].count
            } else {
                0
            },
            if round == 1 {
                self.rows[i + n / 4 + n / 2].count
            } else {
                0
            },
        ];
        let width = if round == 0 { 5 } else { 2 };
        let inputs = if round == 0 { 2 } else { 4 };
        let mut codes = [0; 5];
        for (group, code) in codes[..COUNTER_BITS / width].iter_mut().enumerate() {
            for (j, &count) in counts[..inputs].iter().enumerate() {
                *code |=
                    ((usize::from(count) >> (group * width)) & ((1 << width) - 1)) << (j * width);
            }
        }
        codes
    }

    pub(super) fn prove<Ch: Challenger>(
        self,
        challenger: &mut Ch,
    ) -> (scatter::ScatterProof, Vec<F128>) {
        let n = self.rows.len();
        let mut rounds = Vec::with_capacity(self.record_vars + SLOT_VARS);
        let mut point = Vec::with_capacity(self.record_vars + SLOT_VARS);
        let mut gates = Vec::new();
        let mut values = Vec::new();
        let mut delta_prefix = F128::ONE;
        let mut claim = F128::ZERO;
        for round in 0..2 {
            let half = n >> (round + 1);
            let delta_step = self.delta_powers[self.record_vars - 1 - round];
            let nodes: Vec<_> = (0..=DEGREE)
                .map(|t| F128 {
                    lo: t as u64,
                    hi: 0,
                })
                .collect();
            let tables: Vec<_> = nodes
                .iter()
                .map(|&t| self.counter_tables(&point, t))
                .collect();
            let delta_at: Vec<_> = nodes
                .iter()
                .map(|&t| delta_prefix * affine(F128::ONE, delta_step, t))
                .collect();
            let evals = (0..half / SLOT_COUNT)
                .into_par_iter()
                .fold(
                    || vec![F128::ZERO; DEGREE + 1],
                    |mut total, record| {
                        let mut sums = [F128::ZERO; DEGREE + 1];
                        for slot in 0..SLOTS {
                            let i = record * SLOT_COUNT + slot;
                            let (g0, g1, v0, v1) = if round == 0 {
                                let (a, b) = (self.rows[i], self.rows[i + half]);
                                (
                                    bit(a.gate),
                                    bit(b.gate),
                                    self.gamma[usize::from(a.value)],
                                    self.gamma[usize::from(b.value)],
                                )
                            } else {
                                (gates[i], gates[i + half], values[i], values[i + half])
                            };
                            if g0 == F128::ZERO && g1 == F128::ZERO {
                                continue;
                            }
                            let codes = self.counter_codes(i, round);
                            for t in 0..=DEGREE {
                                let mut term = affine(g0, g1, nodes[t]) * affine(v0, v1, nodes[t]);
                                for (table, &code) in tables[t].iter().zip(&codes) {
                                    term *= table[code];
                                }
                                sums[t] += term;
                            }
                        }
                        for (t, sum) in sums.into_iter().enumerate() {
                            total[t] += sum * self.delta[record] * delta_at[t];
                        }
                        total
                    },
                )
                .reduce(
                    || vec![F128::ZERO; DEGREE + 1],
                    |mut a, b| {
                        for (a, b) in a.iter_mut().zip(b) {
                            *a += b;
                        }
                        a
                    },
                );
            if round == 0 {
                claim = evals[0] + evals[1];
                challenger.observe_label(b"aerie-scatter-v0");
                challenger.observe_f128(claim);
            }
            challenger.observe_f128_slice(&evals);
            let r = challenger.sample_f128();
            if round == 0 {
                gates = (0..half)
                    .into_par_iter()
                    .map(|i| affine(bit(self.rows[i].gate), bit(self.rows[i + half].gate), r))
                    .collect();
                values = (0..half)
                    .into_par_iter()
                    .map(|i| {
                        affine(
                            self.gamma[usize::from(self.rows[i].value)],
                            self.gamma[usize::from(self.rows[i + half].value)],
                            r,
                        )
                    })
                    .collect();
            } else {
                for factor in [&mut gates, &mut values] {
                    let (lo, hi) = factor.split_at_mut(half);
                    lo.par_iter_mut()
                        .zip(hi)
                        .for_each(|(lo, hi)| *lo = affine(*lo, *hi, r));
                    factor.truncate(half);
                }
            }
            delta_prefix *= affine(F128::ONE, delta_step, r);
            rounds.push(evals);
            point.push(r);
        }
        let quarter = n / 4;
        let mut factors = Vec::with_capacity(DEGREE);
        factors.push(
            (0..quarter)
                .into_par_iter()
                .map(|i| delta_prefix * self.delta[i >> SLOT_VARS])
                .collect(),
        );
        factors.push(gates);
        for b in 0..COUNTER_BITS {
            let lut: Vec<_> = (0..16)
                .map(|code| self.counter_value(b, code, &point[..1], point[1]))
                .collect();
            factors.push(
                (0..quarter)
                    .into_par_iter()
                    .map(|i| {
                        let mut code = 0;
                        for (j, offset) in [0, n / 2, n / 4, 3 * n / 4].into_iter().enumerate() {
                            code |= (usize::from(self.rows[i + offset].count >> b) & 1) << j;
                        }
                        lut[code]
                    })
                    .collect(),
            );
        }
        factors.push(values);
        // Multiplication is commutative. Put the gate first so the dense
        // kernel can skip padding pairs before extending any other factor.
        factors.swap(0, 1);
        drop(self);
        scatter::prove_remaining(factors, challenger, false, &mut rounds, &mut point);
        (scatter::ScatterProof { claim, rounds }, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    #[test]
    fn compact_scatter_matches_dense_on_arbitrary_committed_bits() {
        for record_vars in [2, 3, 5] {
            for special in [false, true] {
                let bit_at = |address: usize| {
                    let mut x = (address as u64).wrapping_add(0x9e3779b97f4a7c15);
                    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                    ((x ^ (x >> 31)) & 1) != 0
                };
                let mut source = FsChallenger::new(b"compact-scatter-challenges");
                let beta = if special {
                    F128::ONE
                } else {
                    source.sample_f128()
                };
                let gamma = source.sample_f128();
                let delta = if special {
                    F128::ZERO
                } else {
                    source.sample_f128()
                };
                let dense =
                    super::super::record_factor_tables(bit_at, record_vars, beta, gamma, delta);
                let mut reference = FsChallenger::new(b"compact-scatter-test");
                let (expected, expected_point) = scatter::prove(dense, &mut reference);
                let mut optimized = FsChallenger::new(b"compact-scatter-test");
                let (actual, actual_point) =
                    Factors::new(bit_at, record_vars, beta, gamma, delta).prove(&mut optimized);
                assert_eq!(actual.claim, expected.claim);
                assert_eq!(actual.rounds, expected.rounds);
                assert_eq!(actual_point, expected_point);
                assert_eq!(optimized.sample_f128(), reference.sample_f128());
            }
        }
    }
}
