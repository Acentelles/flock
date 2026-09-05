use super::*;
// Equal counter bits contribute a constant. Differing bits have only two
// orientations, giving three states per bit after constants are removed.
struct CounterPolynomials {
    coefficients: Vec<[F128; 11]>,
    degrees: Vec<usize>,
    ternary: [usize; 1024],
    common: [F128; 1024],
}

impl CounterPolynomials {
    fn new(beta: &[F128]) -> Self {
        assert_eq!(beta.len(), 10);
        let mut coefficients = vec![[F128::ZERO; 11]; 59049];
        let mut degrees = vec![0; 59049];
        coefficients[0][0] = F128::ONE;
        let mut ternary = [0; 1024];
        let mut common = [F128::ONE; 1024];
        let mut width = 1;
        for b in 0..10 {
            for i in 0..1 << b {
                ternary[i + (1 << b)] = ternary[i] + width;
                common[i + (1 << b)] = common[i] * beta[b];
            }
            let slope = F128::ONE + beta[b];
            for i in 0..width {
                let p = coefficients[i];
                let degree = degrees[i];
                let mut up = [F128::ZERO; 11];
                let mut down = [F128::ZERO; 11];
                for j in 0..=degree {
                    let shifted = p[j] * slope;
                    up[j] += p[j];
                    up[j + 1] += shifted;
                    down[j] += p[j] * beta[b];
                    down[j + 1] += shifted;
                }
                coefficients[i + width] = up;
                coefficients[i + 2 * width] = down;
                degrees[i + width] = degree + 1;
                degrees[i + 2 * width] = degree + 1;
            }
            width *= 3;
        }
        Self {
            coefficients,
            degrees,
            ternary,
            common,
        }
    }

    fn code(&self, a: u16, b: u16) -> usize {
        self.ternary[usize::from(a ^ b)] + self.ternary[usize::from(a & !b)]
    }
}

impl Factors {
    pub(super) fn first_round_polynomial(&self) -> Vec<F128> {
        let lookup = CounterPolynomials::new(&self.beta);
        let half = self.rows.len() / 2;
        let delta_step = self.delta_powers[self.record_vars - 1];
        (0..half / SLOT_COUNT)
            .into_par_iter()
            .fold(
                || vec![F128::ZERO; DEGREE + 1],
                |mut total, record| {
                    let mut coefficients = [F128::ZERO; 13];
                    for slot in 0..SLOTS {
                        let i = record * SLOT_COUNT + slot;
                        let a = self.rows[i];
                        let b = self.rows[i + half];
                        if !a.gate && !b.gate {
                            continue;
                        }
                        let code = lookup.code(a.count, b.count);
                        let common = lookup.common[usize::from(a.count & b.count)];
                        let va = self.gamma[usize::from(a.value)];
                        let vb = self.gamma[usize::from(b.value)];
                        let low = va * common;
                        let slope = (va + vb) * common;
                        for j in 0..=lookup.degrees[code] {
                            let p = lookup.coefficients[code][j];
                            let x = p * low;
                            let y = p * slope;
                            if a.gate {
                                coefficients[j] += x;
                                coefficients[j + 1] += y;
                            }
                            if a.gate != b.gate {
                                coefficients[j + 1] += x;
                                coefficients[j + 2] += y;
                            }
                        }
                    }
                    for (t, out) in total.iter_mut().enumerate() {
                        let t = F128 {
                            lo: t as u64,
                            hi: 0,
                        };
                        let value = coefficients
                            .iter()
                            .rev()
                            .fold(F128::ZERO, |s, &c| s * t + c);
                        *out += value * self.delta[record] * affine(F128::ONE, delta_step, t);
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
            )
    }
}

#[cfg(test)]
mod polynomial_tests {
    use super::*;
    #[test]
    fn all_counter_polynomials_match_their_affine_factors() {
        for beta in [
            F128::ZERO,
            F128::ONE,
            F128 {
                lo: 0xa392_b841_f430_32bf,
                hi: 0xb149_44ac_3218_91ff,
            },
        ] {
            let powers = frobenius(beta, 10);
            let lookup = CounterPolynomials::new(&powers);
            let t = F128 {
                lo: 0x890d_4781_943b_6621,
                hi: 0x4427_d21b_9876_5531,
            };
            for code in 0..59049 {
                let mut trits = code;
                let mut expected = F128::ONE;
                for &b in &powers {
                    match trits % 3 {
                        0 => {}
                        1 => expected *= affine(F128::ONE, b, t),
                        2 => expected *= affine(b, F128::ONE, t),
                        _ => unreachable!(),
                    }
                    trits /= 3;
                }
                let got = lookup.coefficients[code]
                    .iter()
                    .rev()
                    .fold(F128::ZERO, |s, &c| s * t + c);
                assert_eq!(got, expected, "code {code}");
            }
            for a in [0, 1, 31, 32, 511, 512, 1023] {
                for b in 0..1024 {
                    let p = &lookup.coefficients[lookup.code(a, b)];
                    let got = p.iter().rev().fold(F128::ZERO, |s, &c| s * t + c)
                        * lookup.common[usize::from(a & b)];
                    let expected = powers.iter().enumerate().fold(F128::ONE, |p, (j, &q)| {
                        p * affine(
                            if a >> j & 1 == 0 { F128::ONE } else { q },
                            if b >> j & 1 == 0 { F128::ONE } else { q },
                            t,
                        )
                    });
                    assert_eq!(got, expected);
                }
            }
        }
    }
}
