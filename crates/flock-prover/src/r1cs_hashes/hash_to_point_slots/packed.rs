//! Exact bit-sliced execution of the existing slot R1CS. The matrix shape
//! is checked before specializing; neither constraints nor wire positions
//! change. Each u128 contains 128 independent Boolean candidates.

use super::*;
use flock_core::field::F128;

const PACKED_PLANE: usize = PLANE / 128;
const WIRE_PLANES: usize = K / PLANE;
const WORD_LOW_BITS: u64 = 0x0001_0001_0001_0001;
const GATHER_FOUR: u64 = 0x0001_0002_0004_0008;

struct Linear {
    constant: bool,
    planes: Vec<usize>,
}

impl Linear {
    fn from_row(row: &[usize]) -> Option<Self> {
        if row.iter().any(|&c| c >= K || c % PLANE != 0) {
            return None;
        }
        Some(Self {
            constant: row.iter().filter(|&&c| c == Z_CONST_POS).count() % 2 == 1,
            planes: row
                .iter()
                .filter(|&&c| c != Z_CONST_POS)
                .map(|&c| c / PLANE)
                .collect(),
        })
    }

    fn evaluate(&self, values: &[u128; WIRE_PLANES], live: u128) -> u128 {
        self.planes
            .iter()
            .fold(if self.constant { live } else { 0 }, |v, &p| v ^ values[p])
    }
}

struct Product {
    plane: usize,
    a: Linear,
    b: Linear,
}

pub(super) struct Program {
    products: Vec<Product>,
}

fn same_shifted_row(row: &[usize], first: &[usize], slot: usize) -> bool {
    row.len() == first.len()
        && row.iter().zip(first).all(|(&actual, &base)| {
            actual
                == if base == Z_CONST_POS {
                    base
                } else {
                    base + slot
                }
        })
}

impl Program {
    pub(super) fn new(a: &SparseBinaryMatrix, b: &SparseBinaryMatrix) -> Option<Self> {
        if a.rows.len() != K || b.rows.len() != K || SLOTS % 4 != 0 || SLOTS >= (1 << COUNTER_BITS)
        {
            return None;
        }
        let planes = (D..D + 16)
            .chain([ACC])
            .chain(E..E + 3)
            .chain(G..G + 16)
            .chain(U_CHAIN..U_CHAIN + 14)
            .chain([U])
            .chain(R_CHAIN..R_CHAIN + 14)
            .chain([PIN_RANGE, PIN_A14, PIN_A15, PIN_B16, PIN_T4]);
        let mut products = Vec::new();
        for plane in planes {
            let first = pos(plane, 0);
            let left = Linear::from_row(&a.rows[first])?;
            let right = Linear::from_row(&b.rows[first])?;
            for slot in 1..SLOTS {
                let row = pos(plane, slot);
                if !same_shifted_row(&a.rows[row], &a.rows[first], slot)
                    || !same_shifted_row(&b.rows[row], &b.rows[first], slot)
                {
                    return None;
                }
            }
            if (SLOTS..PLANE).any(|slot| {
                !a.rows[pos(plane, slot)].is_empty() || !b.rows[pos(plane, slot)].is_empty()
            }) {
                return None;
            }
            products.push(Product {
                plane,
                a: left,
                b: right,
            });
        }
        // Check the exact increment and gate rows before using integer
        // counters. Padded slots have empty rows and must stay all zero.
        for slot in 0..PLANE {
            let counter_planes = [GATE]
                .into_iter()
                .chain(H..H + COUNTER_BITS)
                .chain(CNT..CNT + COUNTER_BITS);
            if slot >= SLOTS {
                if counter_planes
                    .into_iter()
                    .any(|p| !a.rows[pos(p, slot)].is_empty() || !b.rows[pos(p, slot)].is_empty())
                {
                    return None;
                }
                continue;
            }
            let previous = |bit| {
                if slot == 0 {
                    Vec::new()
                } else {
                    wire(pos(CNT + bit, slot - 1))
                }
            };
            if a.rows[pos(GATE, slot)] != wire(pos(ACC, slot))
                || b.rows[pos(GATE, slot)] != lin_xor(&previous(COUNTER_BITS - 1), &constant_one())
            {
                return None;
            }
            for bit in 0..COUNTER_BITS {
                let input = if bit == 0 {
                    wire(pos(ACC, slot))
                } else {
                    wire(pos(H + bit - 1, slot))
                };
                let prev = previous(bit);
                if a.rows[pos(H + bit, slot)] != prev
                    || b.rows[pos(H + bit, slot)] != input
                    || a.rows[pos(CNT + bit, slot)] != lin_xor(&prev, &input)
                    || b.rows[pos(CNT + bit, slot)] != constant_one()
                {
                    return None;
                }
            }
        }
        Some(Self { products })
    }

    pub(super) fn write(&self, words: &[u16; SLOTS], out: &mut [F128]) {
        assert_eq!(out.len(), K / 128);
        out.fill(F128::ZERO);
        out[0].lo = 1;
        let mut count = 0_u16;
        for (block, chunk) in words.chunks(128).enumerate() {
            let mut values = [0_u128; WIRE_PLANES];
            let live = u128::MAX >> (128 - chunk.len());
            // Four 16-bit words transpose into sixteen four-bit columns
            // using a mask and one wrapping integer multiply per column.
            for (packet, words4) in chunk.chunks_exact(4).enumerate() {
                let mut xs = 0_u64;
                let mut qs = 0_u64;
                for (lane, &word) in words4.iter().enumerate() {
                    xs |= u64::from(word) << (16 * lane);
                    qs |= u64::from((u32::from(word) / FALCON_Q) as u16) << (16 * lane);
                }
                for bit in 0..16 {
                    values[X + bit] |= u128::from(gather_four(xs >> bit)) << (4 * packet);
                }
                for bit in 0..3 {
                    values[Q + bit] |= u128::from(gather_four(qs >> bit)) << (4 * packet);
                }
            }
            for product in &self.products {
                values[product.plane] =
                    product.a.evaluate(&values, live) & product.b.evaluate(&values, live);
            }
            for (packet, words4) in chunk.chunks_exact(4).enumerate() {
                let mut counters = 0_u64;
                let mut carries = 0_u64;
                for (lane, &word) in words4.iter().enumerate() {
                    let offset = 4 * packet + lane;
                    let accept = (values[ACC] >> offset) & 1 != 0;
                    let center = (values[U] >> offset) & 1 != 0;
                    let residue = u32::from(word) % FALCON_Q;
                    assert_eq!(accept, u32::from(word) < ACCEPT_BOUND);
                    assert_eq!(center, residue > (FALCON_Q - 1) / 2);
                    let before = count;
                    count += u16::from(accept);
                    counters |= u64::from(count) << (16 * lane);
                    // For increment by a Boolean, carry i equals bit i+1
                    // of before XOR after. No overflow: count <= 612.
                    carries |= u64::from((before ^ count) >> 1) << (16 * lane);
                    if accept && before < 512 {
                        values[GATE] |= 1_u128 << offset;
                        let target = residue as u16 | (u16::from(center) << 14);
                        let address = Z_BASE * PLANE + usize::from(before) * 16;
                        let output = &mut out[address / 128];
                        if address % 128 < 64 {
                            output.lo |= u64::from(target) << (address % 64);
                        } else {
                            output.hi |= u64::from(target) << (address % 64);
                        }
                    }
                }
                for bit in 0..COUNTER_BITS {
                    values[CNT + bit] |= u128::from(gather_four(counters >> bit)) << (4 * packet);
                    values[H + bit] |= u128::from(gather_four(carries >> bit)) << (4 * packet);
                }
            }
            for plane in 1..Z_BASE {
                out[plane * PACKED_PLANE + block] = F128 {
                    lo: values[plane] as u64,
                    hi: (values[plane] >> 64) as u64,
                };
            }
        }
    }
}

fn gather_four(word: u64) -> u64 {
    ((word & WORD_LOW_BITS).wrapping_mul(GATHER_FOUR) >> 48) & 15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_every_input_bit() {
        for input_bit in 0..64 {
            let word = 1_u64 << input_bit;
            for bit in 0..16 {
                assert_eq!(
                    gather_four(word >> bit),
                    if input_bit % 16 == bit {
                        1 << (input_bit / 16)
                    } else {
                        0
                    }
                );
            }
        }
        assert_eq!(gather_four(u64::MAX), 15);
    }

    #[test]
    fn packed_executor_matches_matrix_witness_for_every_word_and_counter_boundaries() {
        let (a, b) = build_matrices();
        let program = Program::new(&a, &b).expect("canonical slot matrices");
        let compare = |words: &[u16; SLOTS]| {
            let (reference, _) = build_block_witness_with(&a, &b, words);
            let mut packed = vec![F128::ZERO; K / 128];
            program.write(words, &mut packed);
            assert_eq!(packed, flock_core::pcs::pack_witness(&reference, K_LOG));
        };
        for first in (0..=u16::MAX as usize).step_by(SLOTS) {
            compare(&std::array::from_fn(|offset| (first + offset) as u16));
        }
        for accepted in [0, 1, 255, 256, 511, 512, 513, SLOTS] {
            compare(&std::array::from_fn(|slot| {
                if slot < accepted {
                    0
                } else {
                    u16::MAX
                }
            }));
            compare(&std::array::from_fn(|slot| {
                if slot >= SLOTS - accepted {
                    6144
                } else {
                    61445
                }
            }));
        }
    }

    #[test]
    fn specialization_rejects_changed_wiring_and_nonzero_padding_rows() {
        let (mut a, b) = build_matrices();
        a.rows[pos(D, 7)].push(pos(X, 8));
        assert!(Program::new(&a, &b).is_none());
        let (mut a, b) = build_matrices();
        a.rows[pos(H, 7)].clear();
        assert!(Program::new(&a, &b).is_none());
        let (mut a, b) = build_matrices();
        a.rows[pos(U, SLOTS)].push(Z_CONST_POS);
        assert!(Program::new(&a, &b).is_none());
    }
}
