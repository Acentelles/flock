//! Exact bit-MLE evaluation without a full-size extension-field tensor.
//!
//! Contract each physical 128-bit word with its seven low-coordinate weights,
//! then dot those values with the equality tensor over free high coordinates.
//! Fixed coordinates, scattered free coordinates, and sub-word domains are
//! supported. Neither the field representation nor any transcript changes.

use flock_core::field::{F256Unreduced, F128};
use rayon::prelude::*;

type Free = Vec<(usize, F128)>;

enum LowMap {
    Sparse(Vec<(usize, F128)>),
    Bytes(Vec<(usize, Box<[F128; 256]>)>),
}

fn eq_tensor(point: &[F128]) -> Vec<F128> {
    let mut tensor = vec![F128::ONE];
    for &r in point {
        let mut next = Vec::with_capacity(tensor.len() * 2);
        for value in tensor {
            let high = value * r;
            next.push(value + high);
            next.push(high);
        }
        tensor = next;
    }
    tensor
}

impl LowMap {
    fn new(point: &[F128; 7]) -> Self {
        let weights = eq_tensor(point);
        let terms: Vec<_> = weights
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, weight)| *weight != F128::ZERO)
            .collect();
        if terms.len() <= 8 {
            return Self::Sparse(terms);
        }
        let mut tables = Vec::new();
        for (byte, weights) in weights.chunks_exact(8).enumerate() {
            if weights.iter().all(|&w| w == F128::ZERO) {
                continue;
            }
            let mut table = Box::new([F128::ZERO; 256]);
            for mask in 1_usize..256 {
                let bit = mask.trailing_zeros() as usize;
                table[mask] = table[mask & (mask - 1)] + weights[bit];
            }
            tables.push((byte, table));
        }
        Self::Bytes(tables)
    }

    #[inline]
    fn apply(&self, word: F128) -> F128 {
        let mut value = F128::ZERO;
        match self {
            Self::Sparse(terms) => {
                for &(bit, weight) in terms {
                    let limb = if bit < 64 { word.lo } else { word.hi };
                    if (limb >> (bit & 63)) & 1 != 0 {
                        value += weight;
                    }
                }
            }
            Self::Bytes(tables) => {
                for (byte, table) in tables {
                    let limb = if *byte < 8 { word.lo } else { word.hi };
                    value += table[((limb >> (8 * (byte & 7))) & 255) as usize];
                }
            }
        }
        value
    }
}

/// Build the high tensor and its word offsets in linear time, including
/// arbitrary gaps between free address bits. No per-element bit-deposit loop.
fn high_tensor(free: &[(usize, F128)]) -> (Vec<usize>, Vec<F128>) {
    let mut addresses = vec![0];
    let mut weights = vec![F128::ONE];
    for &(bit, r) in free {
        if weights.len() >= 1 << 20 {
            let mut next_addresses = Box::<[usize]>::new_uninit_slice(2 * addresses.len());
            let mut next_weights = Box::<[F128]>::new_uninit_slice(2 * weights.len());
            next_addresses
                .par_chunks_mut(2)
                .zip(next_weights.par_chunks_mut(2))
                .zip(addresses.par_iter().zip(weights.par_iter()))
                .for_each(|((out_addresses, out_weights), (&address, &weight))| {
                    let high = weight * r;
                    out_addresses[0].write(address);
                    out_addresses[1].write(address | (1 << bit));
                    out_weights[0].write(weight + high);
                    out_weights[1].write(high);
                });
            // SAFETY: both outputs have exactly twice the input length. Every
            // disjoint two-cell chunk is written above, and the parallel
            // traversal has completed before either allocation is assumed
            // initialized.
            addresses = unsafe { next_addresses.assume_init() }.into_vec();
            weights = unsafe { next_weights.assume_init() }.into_vec();
        } else {
            let mut next_addresses = Vec::with_capacity(2 * addresses.len());
            let mut next_weights = Vec::with_capacity(2 * weights.len());
            for (&address, &weight) in addresses.iter().zip(&weights) {
                let high = weight * r;
                next_addresses.extend([address, address | (1 << bit)]);
                next_weights.extend([weight + high, high]);
            }
            addresses = next_addresses;
            weights = next_weights;
        }
    }
    (addresses, weights)
}

fn evaluate(
    vars: usize,
    points: &[Vec<F128>],
    word_at: impl Fn(usize) -> F128 + Sync,
) -> Vec<F128> {
    let mut groups: Vec<(Free, Vec<(usize, usize, [F128; 7])>)> = Vec::new();
    for (index, point) in points.iter().enumerate() {
        assert_eq!(point.len(), vars, "mixed bit-MLE domains");
        let mut fixed = 0;
        let mut free = Vec::new();
        // Missing low coordinates of a sub-word domain are fixed to zero.
        let mut low = [F128::ZERO; 7];
        for (i, &r) in point.iter().enumerate() {
            let bit = vars - 1 - i;
            if bit < 7 {
                low[6 - bit] = r;
            } else if r == F128::ONE {
                fixed |= 1 << (bit - 7);
            } else if r != F128::ZERO {
                free.push((bit - 7, r));
            }
        }
        if let Some((_, members)) = groups.iter_mut().find(|(key, _)| *key == free) {
            members.push((index, fixed, low));
        } else {
            groups.push((free, vec![(index, fixed, low)]));
        }
    }
    let mut values = vec![F128::ZERO; points.len()];
    for (free, members) in groups {
        let (addresses, weights) = high_tensor(&free);
        let mut maps: Vec<([F128; 7], LowMap)> = Vec::new();
        let mapped: Vec<_> = members
            .iter()
            .map(|&(index, fixed, point)| {
                let map = if let Some(i) = maps.iter().position(|(p, _)| *p == point) {
                    i
                } else {
                    maps.push((point, LowMap::new(&point)));
                    maps.len() - 1
                };
                (index, fixed, map)
            })
            .collect();
        let evaluated: Vec<_> = mapped
            .par_iter()
            .map(|&(index, fixed, map)| {
                let low = &maps[map].1;
                let sum = addresses
                    .par_chunks(16_384)
                    .zip(weights.par_chunks(16_384))
                    .map(|(addresses, weights)| {
                        let mut sum = F256Unreduced::ZERO;
                        for (&address, &weight) in addresses.iter().zip(weights) {
                            let value = low.apply(word_at(fixed | address));
                            sum ^= value.mul_unreduced(weight);
                        }
                        sum
                    })
                    .reduce(|| F256Unreduced::ZERO, |a, b| a ^ b);
                (index, sum.reduce())
            })
            .collect();
        for (index, value) in evaluated {
            values[index] = value;
        }
    }
    values
}

/// Evaluate MSB-first points against physical consecutive 128-bit words.
pub fn evaluate_packed(words: &[F128], points: &[Vec<F128>]) -> Vec<F128> {
    let Some(point) = points.first() else {
        return Vec::new();
    };
    let vars = point.len();
    assert!(vars < usize::BITS as usize);
    assert!(
        words.len() >= (1_usize << vars).div_ceil(128),
        "short packed witness"
    );
    evaluate(vars, points, |address| words[address])
}

/// Evaluate several packed bit tables at one point, sharing both the high
/// equality tensor and low-byte lookup maps. No concatenated witness is built.
pub fn evaluate_packed_tables(tables: &[Vec<F128>], point: &[F128]) -> Vec<F128> {
    if tables.is_empty() {
        return Vec::new();
    }
    let vars = point.len();
    if vars < 7 {
        return tables
            .iter()
            .map(|table| evaluate_packed(table, &[point.to_vec()])[0])
            .collect();
    }
    assert!(vars < usize::BITS as usize);
    let stride = 1_usize << (vars - 7);
    assert!(tables.iter().all(|table| table.len() >= stride));
    let selector = tables.len().next_power_of_two().trailing_zeros() as usize;
    assert!(vars + selector < usize::BITS as usize);
    let points: Vec<_> = (0..tables.len())
        .map(|i| {
            let mut p: Vec<_> = (0..selector)
                .rev()
                .map(|bit| {
                    if i & (1 << bit) != 0 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            p.extend_from_slice(point);
            p
        })
        .collect();
    evaluate(vars + selector, &points, |address| {
        tables[address / stride][address % stride]
    })
}

/// Bool-table adapter. Callers that already hold the packed witness should
/// use evaluate_packed instead of packing the same words again here.
pub fn evaluate_bits(bits: &[bool], points: &[Vec<F128>]) -> Vec<F128> {
    let Some(point) = points.first() else {
        return Vec::new();
    };
    let vars = point.len();
    assert!(vars < usize::BITS as usize);
    assert_eq!(bits.len(), 1_usize << vars);
    evaluate(vars, points, |address| {
        let start = address * 128;
        let mut word = F128::ZERO;
        for (bit, &value) in bits[start..bits.len().min(start + 128)].iter().enumerate() {
            if value {
                if bit < 64 {
                    word.lo |= 1 << bit;
                } else {
                    word.hi |= 1 << (bit - 64);
                }
            }
        }
        word
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::hash_to_point_record::bit_mle;

    fn pseudo(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn mixed_shapes_match_dense_fold_and_bool_adapter() {
        let mut seed = 0xabcdef432111;
        for vars in [0, 1, 5, 7, 8, 10, 15] {
            let bits: Vec<_> = (0..1 << vars).map(|_| pseudo(&mut seed) & 1 == 1).collect();
            let mut packed = vec![F128::ZERO; bits.len().div_ceil(128)];
            for (i, &bit) in bits.iter().enumerate() {
                if bit {
                    if i % 128 < 64 {
                        packed[i / 128].lo |= 1 << (i % 64);
                    } else {
                        packed[i / 128].hi |= 1 << (i % 64);
                    }
                }
            }
            let base: Vec<_> = (0..vars)
                .map(|_| F128::new(pseudo(&mut seed), pseudo(&mut seed)))
                .collect();
            let points: Vec<_> = (0..24)
                .map(|pattern| {
                    base.iter()
                        .enumerate()
                        .map(|(i, &r)| match (pattern + 3 * i) % 7 {
                            0 | 1 => F128::ZERO,
                            2 | 3 => F128::ONE,
                            _ => r,
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let expected: Vec<_> = points.iter().map(|p| bit_mle(&bits, p)).collect();
            assert_eq!(
                evaluate_packed(&packed, &points),
                expected,
                "packed vars={vars}"
            );
            assert_eq!(evaluate_bits(&bits, &points), expected, "bool vars={vars}");
        }
    }

    #[test]
    fn all_low_boolean_patterns_select_exactly_the_requested_bit() {
        let word = F128::new(0x0123456789abcdef, 0xfedcba9876543210);
        for i in 0..128 {
            let point = std::array::from_fn(|j| {
                if (i >> (6 - j)) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            });
            let limb = if i < 64 { word.lo } else { word.hi };
            let expected = if (limb >> (i % 64)) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            };
            assert_eq!(LowMap::new(&point).apply(word), expected);
        }
        assert!(evaluate_packed(&[], &[]).is_empty());
        assert!(evaluate_bits(&[], &[]).is_empty());
    }
    fn high_tensor_reference(free: &[(usize, F128)]) -> (Vec<usize>, Vec<F128>) {
        let mut addresses = vec![0];
        let mut weights = vec![F128::ONE];
        for &(bit, r) in free {
            let mut next_addresses = Vec::with_capacity(2 * addresses.len());
            let mut next_weights = Vec::with_capacity(2 * weights.len());
            for (&address, &weight) in addresses.iter().zip(&weights) {
                let high = weight * r;
                next_addresses.extend([address, address | (1 << bit)]);
                next_weights.extend([weight + high, high]);
            }
            addresses = next_addresses;
            weights = next_weights;
        }
        (addresses, weights)
    }

    #[test]
    fn parallel_high_tensor_matches_every_address_and_weight() {
        for count in [0usize, 1, 4, 12, 13, 17, 20, 21] {
            for boolean in [false, true] {
                let free: Vec<_> = (0..count)
                    .map(|i| {
                        let value = if boolean && i % 5 == 0 {
                            F128::ONE
                        } else if boolean && i % 5 == 1 {
                            F128::ZERO
                        } else {
                            F128::new(
                                (i as u64 + 11).wrapping_mul(0x9e3779b97f4a7c15),
                                (i as u64 + 7).wrapping_mul(0xc2b2ae3d27d4eb4f),
                            )
                        };
                        (2 * (count - i) - 1, value)
                    })
                    .collect();
                assert_eq!(
                    high_tensor(&free),
                    high_tensor_reference(&free),
                    "count={count}, boolean={boolean}"
                );
            }
        }
    }
}
