use super::scatter::polynomial_product as product;
use super::*;

impl Factors {
    fn counter_luts(&self, point: &[F128]) -> Vec<Vec<F128>> {
        assert!((2..=4).contains(&point.len()));
        let corners = 1 << point.len();
        let mut weights = vec![F128::ZERO; corners];
        weights[0] = F128::ONE;
        for (bit, &r) in point.iter().enumerate() {
            for j in 0..1 << bit {
                let high = weights[j] * r;
                weights[j | (1 << bit)] = high;
                weights[j] += high;
            }
        }
        self.beta
            .par_iter()
            .map(|&beta| {
                let mut table = vec![F128::ZERO; 1 << corners];
                table[0] = F128::ONE;
                for (bit, &weight) in weights.iter().enumerate() {
                    let high = (F128::ONE + beta) * weight;
                    let (low, rest) = table.split_at_mut(1 << bit);
                    for (lo, hi) in low.iter().zip(rest) {
                        *hi = *lo + high;
                    }
                }
                table
            })
            .collect()
    }

    pub(super) fn continue_compact<Ch: Challenger>(
        mut self,
        mut gates: Vec<F128>,
        mut values: Vec<F128>,
        mut delta_prefix: F128,
        challenger: &mut Ch,
        rounds: &mut Vec<Vec<F128>>,
        point: &mut Vec<F128>,
    ) {
        let n = self.rows.len();
        let quarter = n / 4;
        let mut codes: Vec<Vec<u16>> = (0..COUNTER_BITS)
            .into_par_iter()
            .map(|b| {
                (0..quarter)
                    .map(|i| {
                        let mut code = 0;
                        for (j, offset) in [0, n / 2, n / 4, 3 * n / 4].into_iter().enumerate() {
                            code |= ((self.rows[i + offset].count >> b) & 1) << j;
                        }
                        code
                    })
                    .collect()
            })
            .collect();
        self.rows = Vec::new();
        let nodes: Vec<_> = (0..=DEGREE)
            .map(|lo| F128 {
                lo: lo as u64,
                hi: 0,
            })
            .collect();
        let mut dense = Vec::new();
        for round in 2..self.record_vars.min(5) {
            let half = gates.len() / 2;
            let tables = self.counter_luts(point);
            let delta_step = self.delta_powers[self.record_vars - 1 - round];
            let delta_at: Vec<_> = nodes
                .iter()
                .map(|&t| delta_prefix * affine(F128::ONE, delta_step, t))
                .collect();
            let evals = (0..half / SLOT_COUNT)
                .into_par_iter()
                .fold(
                    || vec![F128::ZERO; DEGREE + 1],
                    |mut total, record| {
                        let mut coefficients = [F128::ZERO; 14];
                        for slot in 0..SLOTS {
                            let i = record * SLOT_COUNT + slot;
                            if gates[i] == F128::ZERO && gates[i + half] == F128::ZERO {
                                continue;
                            }
                            // Delta is transparent and constant over each record's
                            // slots. Sum the degree-12 polynomial before applying it.
                            let mut factors = [[F128::ONE, F128::ZERO]; 13];
                            factors[0] = [gates[i], gates[i] + gates[i + half]];
                            for b in 0..COUNTER_BITS {
                                let a = tables[b][usize::from(codes[b][i])];
                                let z = tables[b][usize::from(codes[b][i + half])];
                                factors[b + 1] = [a, a + z];
                            }
                            factors[11] = [values[i], values[i] + values[i + half]];
                            let product = product::product13(&factors);
                            for (a, b) in coefficients.iter_mut().zip(product) {
                                *a += b;
                            }
                        }
                        debug_assert_eq!(coefficients[13], F128::ZERO);
                        for (t, &node) in nodes.iter().enumerate() {
                            let value = coefficients[..13]
                                .iter()
                                .rev()
                                .fold(F128::ZERO, |sum, &coefficient| sum * node + coefficient);
                            total[t] += value * self.delta[record] * delta_at[t];
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
            challenger.observe_f128_slice(&evals);
            let r = challenger.sample_f128();
            for values in [&mut gates, &mut values] {
                let (low, high) = values.split_at_mut(half);
                low.par_iter_mut()
                    .zip(high)
                    .for_each(|(low, high)| *low = affine(*low, *high, r));
                values.truncate(half);
            }
            if round < 4 {
                codes.par_iter_mut().for_each(|plane| {
                    let (low, high) = plane.split_at_mut(half);
                    for (low, high) in low.iter_mut().zip(high) {
                        *low |= *high << (1 << round);
                    }
                    plane.truncate(half);
                });
            } else {
                dense = codes
                    .par_iter()
                    .zip(&tables)
                    .map(|(plane, table)| {
                        (0..half)
                            .map(|i| {
                                affine(
                                    table[usize::from(plane[i])],
                                    table[usize::from(plane[i + half])],
                                    r,
                                )
                            })
                            .collect()
                    })
                    .collect();
                codes = Vec::new();
            }
            delta_prefix *= affine(F128::ONE, delta_step, r);
            rounds.push(evals);
            point.push(r);
        }
        if dense.is_empty() {
            let tables = self.counter_luts(point);
            dense = codes
                .par_iter()
                .zip(&tables)
                .map(|(plane, table)| plane.iter().map(|&code| table[usize::from(code)]).collect())
                .collect();
        }
        drop(codes);
        let length = gates.len();
        let mut factors = Vec::with_capacity(DEGREE);
        factors.push(gates);
        factors.push(
            (0..length)
                .into_par_iter()
                .map(|i| delta_prefix * self.delta[i >> SLOT_VARS])
                .collect(),
        );
        factors.extend(dense);
        factors.push(values);
        drop(self);
        scatter::prove_remaining(factors, challenger, false, rounds, point);
    }
}
