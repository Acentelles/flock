//! Candidate-slot R1CS for the aerie private-salt HashToPoint relation.
//!
//! One block proves one RECORD's worth of rejection-sampling candidates:
//! `SLOTS = 612` (nine SHAKE256 rate blocks, the default squeeze bucket;
//! longer buckets are separate block families). Per 16-bit candidate word
//! `x`, the block proves the accept comparison
//! `x < 61,445`, the decomposition `x = 12,289 q + a` with `0 <= a < 12,289`
//! and `q <= 5`, the centering flag `u = (a > 6,144)`, and the running
//! acceptance counter. Because the whole record lives in one block, the
//! counter chains across all 612 slots inside the block, and its reset to
//! zero is structural: there are no counter-input wires and the initial
//! symbolic counter is the empty support. No cross-block chaining
//! machinery is needed for the counter.
//! See `AERIE-ADAPTER.md` Sections 3.2 and 3.3; the
//! scatter of accepted `(a, u)` values onto the dense `Z_H` region is a
//! claim-level argument (the counter's monotone increments make it a stable
//! permutation) and is NOT part of this block.
//!
//! ## Encoding
//!
//! Circuit R1CS `(A z) & (B z) = z` with `C_0 = I`. Only AND outputs and
//! free inputs are materialized wires; all linear structure is carried
//! symbolically into row taps (the SHA-2 encoder's discipline). Full-adder
//! carries cost one row each via `maj(p, s, c) = (p ^ c)(s ^ c) ^ c`.
//!
//! Values the relation FORCES (rather than defines) use self-referential
//! rows: `(expr ^ w)(1) = w` holds exactly when `expr = 0`, with `w` left
//! free (witness 0). The zerocheck itself enforces these, so no claim-level
//! pin machinery is needed and `C_0 = I` is preserved (the generic verifier
//! converts the c-claim to a z-claim under that assumption). The forced
//! zeros are, per slot: the subtraction bits 14 and 15 of `x - M` (so `a`
//! fits 14 bits), the final borrow (so `x >= M`), `t_4` of `3 q` (so
//! `q <= 5`), and the range flag (so `a < 12,289`). The constant-one wire
//! still uses the upstream `const_pin`.
//!
//! ## Layout (K_LOG = 16, 65,536 wires per block = one record)
//!
//! ```text
//! 0              constant-one wire (const_pin)
//! 16 .. 61,216   612 slots, stride 100 (see slot offsets in the code)
//! rest           zero padding (empty rows)
//! ```

use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

use super::common::build_block_r1cs_with_matrices;

pub const K_LOG: usize = 16;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

/// Squeeze blocks in the default bucket (spec Section 4).
pub const SQUEEZE_BLOCKS: usize = 9;
/// Candidates per record block: nine SHAKE256 rate blocks.
pub const SLOTS: usize = SQUEEZE_BLOCKS * 68;

pub const COUNTER_BITS: usize = 10;
pub const Z_CONST_POS: usize = 0;
pub const SLOT_BASE: usize = 16;
pub const SLOT_STRIDE: usize = 100;
pub const USEFUL_BITS: usize = SLOT_BASE + SLOTS * SLOT_STRIDE;

// Per-slot wire offsets.
const X: usize = 0; // 16 candidate word bits, free inputs
const Q: usize = 16; // 3 quotient bits, free inputs
const D: usize = 19; // 16 accept-chain carry products
const ACC: usize = 35; // materialized accept flag
const E: usize = 36; // 3 carry products of the 3q mini-adder
const G: usize = 39; // 16 borrow products of x - M
const U_CHAIN: usize = 55; // 14 centering-chain carry products
const U: usize = 69; // materialized centering flag
const R_CHAIN: usize = 70; // 14 range-chain carry products
const PIN_RANGE: usize = 84; // forced zero: carry-out of a + 4,095
const PIN_A14: usize = 85; // forced zero: subtraction bit 14
const PIN_A15: usize = 86; // forced zero: subtraction bit 15
const PIN_B16: usize = 87; // forced zero: final borrow of x - M
const H: usize = 88; // 10 counter-increment carry products
const PIN_T4: usize = 98; // forced zero: t_4 of 3q, so q <= 5
const GATE: usize = 99; // write gate: acc AND (count-before < 512)
const SLOT_WIRES: usize = 100;

/// Falcon rejection bound: accept exactly when `x < 5 * 12,289`.
const ACCEPT_BOUND: u32 = 61_445;
/// `2^16 - 61,445`: carry-out of `x + 4,091` is the reject flag.
const ACCEPT_K: u32 = 4_091;
/// `2^14 - 12,289`: carry-out of `a + 4,095` violates `a < 12,289`.
const RANGE_K: u32 = 4_095;
/// `2^14 - 6,145`: carry-out of `a + 10,239` is `u = (a > 6,144)`.
const CENTER_K: u32 = 10_239;
const FALCON_Q: u32 = 12_289;

/// Symbolic GF(2) linear form: a sorted set of wire columns whose XOR is the
/// value. The constant 1 is a tap on [`Z_CONST_POS`].
type Lin = Vec<usize>;

fn lin_xor(a: &Lin, b: &Lin) -> Lin {
    // Symmetric difference of two sorted column sets.
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
            }
            (Some(&x), Some(&y)) if x < y => {
                out.push(x);
                i += 1;
            }
            (Some(_), Some(&y)) => {
                out.push(y);
                j += 1;
            }
            (Some(&x), None) => {
                out.push(x);
                i += 1;
            }
            (None, Some(&y)) => {
                out.push(y);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    out
}

fn wire(col: usize) -> Lin {
    vec![col]
}

fn constant_one() -> Lin {
    vec![Z_CONST_POS]
}

struct Builder {
    a: Vec<Vec<usize>>,
    b: Vec<Vec<usize>>,
}

impl Builder {
    fn new() -> Self {
        Self {
            a: vec![Vec::new(); K],
            b: vec![Vec::new(); K],
        }
    }

    /// Row `w`: `(a_lin)(b_lin) = z_w`.
    fn product(&mut self, w: usize, a_lin: &Lin, b_lin: &Lin) {
        assert!(self.a[w].is_empty() && self.b[w].is_empty(), "row reused");
        self.a[w] = a_lin.clone();
        self.b[w] = b_lin.clone();
    }

    /// Free input: the vacuous self-loop `(z_w)(1) = z_w`.
    fn input(&mut self, w: usize) {
        self.product(w, &wire(w), &constant_one());
    }

    /// Defined wire: `(lin)(1) = z_w`.
    fn define(&mut self, w: usize, lin: &Lin) {
        self.product(w, lin, &constant_one());
    }

    /// Forced zero: `(lin ^ z_w)(1) = z_w` is satisfiable exactly when
    /// `lin = 0`; the zerocheck enforces it and `z_w` stays free (0).
    fn force_zero(&mut self, w: usize, lin: &Lin) {
        self.product(w, &lin_xor(lin, &wire(w)), &constant_one());
    }
}

/// One maj-form carry step: appends the product row for
/// `carry' = maj(p, s, c) = (p ^ c)(s ^ c) ^ c` at wire `w` and returns the
/// new symbolic carry `{w} ^ c`.
fn carry_step(builder: &mut Builder, w: usize, p: &Lin, s: &Lin, c: &Lin) -> Lin {
    builder.product(w, &lin_xor(p, c), &lin_xor(s, c));
    lin_xor(&wire(w), c)
}

/// Carry chain for `value + CONSTANT` over `bits` bits with per-bit value
/// forms `value_bit(i)`. Returns the symbolic carry-out.
fn constant_add_chain(
    builder: &mut Builder,
    products_base: usize,
    constant: u32,
    bits: usize,
    value_bit: impl Fn(usize) -> Lin,
) -> Lin {
    let mut carry: Lin = Vec::new();
    for i in 0..bits {
        let k_bit = if (constant >> i) & 1 == 1 {
            constant_one()
        } else {
            Vec::new()
        };
        carry = carry_step(builder, products_base + i, &value_bit(i), &k_bit, &carry);
    }
    carry
}

/// Build the shared per-block matrices plus the per-block symbolic counter
/// outputs (used by future chaining; also returned for tests).
fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut builder = Builder::new();
    builder.input(Z_CONST_POS);

    // Running counter, symbolic across all slots of the record; the reset
    // to zero is structural (empty initial supports, no input wires).
    let mut counter: Vec<Lin> = vec![Vec::new(); COUNTER_BITS];

    for slot in 0..SLOTS {
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        let x = |i: usize| wire(base + X + i);
        for i in 0..16 {
            builder.input(base + X + i);
        }
        for i in 0..3 {
            builder.input(base + Q + i);
        }
        let q = |i: usize| wire(base + Q + i);

        // Accept: reject flag is the carry-out of x + 4,091.
        let reject = constant_add_chain(&mut builder, base + D, ACCEPT_K, 16, x);
        builder.define(base + ACC, &lin_xor(&reject, &constant_one()));
        let acc = wire(base + ACC);

        // t = 3 q = q + (q << 1); bits: t_0 = q_0, t_1 = q_1 ^ q_0,
        // t_2 = q_2 ^ q_1 ^ e_2, t_3 = q_2 ^ e_3, t_4 = e_4.
        let e2 = carry_step(&mut builder, base + E, &q(1), &q(0), &Vec::new());
        let e2 = lin_xor(&e2, &Vec::new());
        let e3 = carry_step(&mut builder, base + E + 1, &q(2), &q(1), &e2);
        let e4 = carry_step(&mut builder, base + E + 2, &Vec::new(), &q(2), &e3);
        let t = [
            q(0),
            lin_xor(&q(1), &q(0)),
            lin_xor(&lin_xor(&q(2), &q(1)), &e2),
            lin_xor(&q(2), &e3),
            e4.clone(),
        ];
        // M = 12,289 q = (3 q << 12) ^ q: disjoint bit ranges, pure wiring.
        let m_bit = |i: usize| -> Lin {
            if i < 3 {
                q(i)
            } else if (12..16).contains(&i) {
                t[i - 12].clone()
            } else {
                Vec::new()
            }
        };

        // Subtraction a = x - M via borrows: b' = maj(~x_i, M_i, b).
        let mut borrow: Lin = Vec::new();
        let mut a_bits: Vec<Lin> = Vec::new();
        for i in 0..16 {
            a_bits.push(lin_xor(&lin_xor(&x(i), &m_bit(i)), &borrow));
            let not_x = lin_xor(&x(i), &constant_one());
            borrow = carry_step(&mut builder, base + G + i, &not_x, &m_bit(i), &borrow);
        }

        // Centering flag: carry-out of a + 10,239 over 14 bits.
        let center = constant_add_chain(&mut builder, base + U_CHAIN, CENTER_K, 14, |i| {
            a_bits[i].clone()
        });
        builder.define(base + U, &center);

        // Range flag: carry-out of a + 4,095 over 14 bits; forced zero.
        let range = constant_add_chain(&mut builder, base + R_CHAIN, RANGE_K, 14, |i| {
            a_bits[i].clone()
        });
        builder.force_zero(base + PIN_RANGE, &range);
        builder.force_zero(base + PIN_A14, &a_bits[14]);
        builder.force_zero(base + PIN_A15, &a_bits[15]);
        builder.force_zero(base + PIN_B16, &borrow);
        // t_4 is the SYMBOLIC carry e4 (product xor e3), not the raw product
        // wire; forcing it zero is the q <= 5 range bound.
        builder.force_zero(base + PIN_T4, &e4);

        // Write gate: acc AND (count-before-this-slot < 512). The counter
        // never reaches 1,024 in a 612-slot record, so the condition is
        // one bit: gate = acc * (1 ^ cnt_9).
        builder.product(
            base + GATE,
            &acc,
            &lin_xor(&counter[COUNTER_BITS - 1], &constant_one()),
        );

        // Counter increment by the accept flag: h_0 = acc,
        // h_{i+1} = cnt_i & h_i, cnt_i' = cnt_i ^ h_i.
        let mut increment = acc;
        for (bit, cnt_bit) in counter.iter_mut().enumerate() {
            builder.product(base + H + bit, cnt_bit, &increment);
            let carried = wire(base + H + bit);
            *cnt_bit = lin_xor(cnt_bit, &increment);
            increment = carried;
        }
        // A record has at most 680 slots, so the counter cannot wrap; the
        // final increment carry is left dangling by design.
        let _ = increment;
        let _ = base + SLOT_WIRES;
    }

    let to_matrix = |rows: Vec<Vec<usize>>| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_matrix(builder.a), to_matrix(builder.b))
}

/// Block accessors for tests and the future scatter argument.
pub fn accept_position(slot: usize) -> usize {
    SLOT_BASE + slot * SLOT_STRIDE + ACC
}
pub fn centering_position(slot: usize) -> usize {
    SLOT_BASE + slot * SLOT_STRIDE + U
}
/// The scatter's write gate: accepted and before the 512th acceptance.
pub fn gate_position(slot: usize) -> usize {
    SLOT_BASE + slot * SLOT_STRIDE + GATE
}

/// Reusable slot-prover state: the shared block R1CS plus PCS parameters.
///
/// One block is one record. The default Ligerito profile needs `m >= 22`,
/// so the smallest legal setup is 64 records (39,168 candidate slots).
pub struct SlotSetup {
    pub n_blocks: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl SlotSetup {
    pub fn new(n_blocks: usize) -> Self {
        let n_blocks_log = n_blocks.next_power_of_two().trailing_zeros().max(3) as usize;
        let r1cs = build_block_r1cs(n_blocks_log);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: Default::default(),
            merkle_hash: Default::default(),
        };
        Self {
            n_blocks,
            r1cs,
            pcs_params,
        }
    }

    /// Witness for `n_blocks` record blocks, zero-word blocks as padding
    /// (valid computations with the constant wire set, as `const_pin`
    /// requires). The counter reset per record is structural.
    pub fn generate_witness(&self, blocks: &[[u16; SLOTS]]) -> Vec<bool> {
        assert_eq!(blocks.len(), self.n_blocks);
        let padded = 1usize << (self.r1cs.m - K_LOG);
        let zero_block = [0_u16; SLOTS];
        let mut z = Vec::with_capacity(1 << self.r1cs.m);
        for index in 0..padded {
            let words = blocks.get(index).unwrap_or(&zero_block);
            let (block, _counter) = build_block_witness_with(&self.r1cs.a_0, &self.r1cs.b_0, words);
            z.extend_from_slice(&block);
        }
        z
    }

    /// Generic matrix-driven prover over the packed witness.
    pub fn prove_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        blocks: &[[u16; SLOTS]],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z = self.generate_witness(blocks);
        let z_packed = flock_core::pcs::pack_witness(&z, self.r1cs.m);
        crate::prover::prove_ligerito(&self.r1cs, z_packed, &self.pcs_params, challenger)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        Some(Z_CONST_POS),
    )
}

/// Build one block's witness by evaluating the rows in wire order, so the
/// witness is consistent with the matrices by construction. Free inputs are
/// the candidate words, the quotient bits, and the counter input.
///
/// Returns the block bits and the counter value after the block.
pub fn build_block_witness(words: &[u16; SLOTS]) -> (Vec<bool>, u16) {
    let (a_0, b_0) = build_matrices();
    build_block_witness_with(&a_0, &b_0, words)
}

/// [`build_block_witness`] against prebuilt matrices (one build per setup).
pub fn build_block_witness_with(
    a_0: &SparseBinaryMatrix,
    b_0: &SparseBinaryMatrix,
    words: &[u16; SLOTS],
) -> (Vec<bool>, u16) {
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;
    for (slot, &word) in words.iter().enumerate() {
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        for i in 0..16 {
            z[base + X + i] = (word >> i) & 1 == 1;
        }
        // The honest quotient; for rejected words it is still the true
        // quotient (capped by the pin only through q <= 5, x >= M, a < q).
        let quotient = (u32::from(word) / FALCON_Q).min(5) as u16;
        for i in 0..3 {
            z[base + Q + i] = (quotient >> i) & 1 == 1;
        }
    }
    let eval = |lin: &[usize], z: &[bool]| lin.iter().fold(false, |acc, &col| acc ^ z[col]);
    for w in 0..K {
        let is_input = w == Z_CONST_POS
            || (w >= SLOT_BASE && {
                let offset = (w - SLOT_BASE) % SLOT_STRIDE;
                (w - SLOT_BASE) / SLOT_STRIDE < SLOTS && offset < Q + 3
            });
        if is_input {
            continue;
        }
        z[w] = eval(&a_0.rows[w], &z) & eval(&b_0.rows[w], &z);
    }

    // Differential against the integer reference semantics.
    let mut counter = 0_u16;
    for (slot, &word) in words.iter().enumerate() {
        let accept = u32::from(word) < ACCEPT_BOUND;
        let residue = u32::from(word) % FALCON_Q;
        let centered = residue > (FALCON_Q - 1) / 2;
        assert_eq!(z[accept_position(slot)], accept, "accept flag, slot {slot}");
        assert_eq!(
            z[centering_position(slot)],
            centered,
            "centering flag, slot {slot}"
        );
        counter += u16::from(accept);
    }
    (z, counter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_battery() -> [u16; SLOTS] {
        // 612 words: the boundary edges first, pseudo-random fill after.
        let mut words = [0_u16; SLOTS];
        let edges = [
            0_u16, 1, 6_144, 6_145, 12_288, 12_289, 24_577, 36_866, 49_155, 61_444, 61_445, 61_446,
            65_535,
        ];
        words[..edges.len()].copy_from_slice(&edges);
        for (index, word) in words.iter_mut().enumerate().skip(edges.len()) {
            *word = (index as u16).wrapping_mul(9_973).wrapping_add(12_345);
        }
        words
    }

    fn single_block_r1cs() -> BlockR1cs {
        // n_blocks_log = 3 is the lincheck minimum; satisfies() needs the
        // full domain, so tile the same block eight times.
        build_block_r1cs(3)
    }

    fn tiled(z_block: &[bool]) -> Vec<bool> {
        let mut z = Vec::with_capacity(8 * K);
        for _ in 0..8 {
            z.extend_from_slice(z_block);
        }
        z
    }

    #[test]
    fn honest_block_satisfies() {
        let words = word_battery();
        let (block, counter) = build_block_witness(&words);
        let expected = words
            .iter()
            .filter(|&&word| u32::from(word) < ACCEPT_BOUND)
            .count() as u16;
        assert_eq!(counter, expected);
        assert!(single_block_r1cs().satisfies(&tiled(&block)));
    }

    #[test]
    fn counter_counts_all_record_slots() {
        let words = word_battery();
        let (block, counter) = build_block_witness(&words);
        let accepted = words
            .iter()
            .filter(|&&word| u32::from(word) < ACCEPT_BOUND)
            .count() as u16;
        assert_eq!(counter, accepted);
        // Honest random words accept at rate 61,445/65,536, so a full
        // record's 612 slots always clear 512 acceptances here.
        assert!(counter >= 512);
        let _ = block;
    }

    #[test]
    fn tampered_wires_fail_satisfies() {
        let words = word_battery();
        let (block, _counter) = build_block_witness(&words);
        let r1cs = single_block_r1cs();
        for position in [
            accept_position(0),
            centering_position(3),
            SLOT_BASE + G + 5,
            SLOT_BASE + 2 * SLOT_STRIDE + D + 9,
            SLOT_BASE + H + 1,
        ] {
            let mut tampered = block.clone();
            tampered[position] = !tampered[position];
            assert!(
                !r1cs.satisfies(&tiled(&tampered)),
                "tampered wire {position} must break a row"
            );
        }
    }

    #[test]
    fn a_wrong_quotient_breaks_a_forced_zero_row() {
        // Every ordinary wire is definitional, so a wrong free input still
        // satisfies those rows; the self-referential forced-zero rows are
        // what reject it, inside the zerocheck itself.
        let words = word_battery();
        let (mut block, _counter) = build_block_witness(&words);

        // Re-derive slot 5 (an accepted word) with quotient + 1.
        let slot = 5;
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        let word = words[slot];
        let wrong_quotient = (u32::from(word) / FALCON_Q + 1) as u16;
        for i in 0..3 {
            block[base + Q + i] = (wrong_quotient >> i) & 1 == 1;
        }
        let r1cs = single_block_r1cs();
        let eval = |lin: &[usize], z: &[bool]| lin.iter().fold(false, |acc, &col| acc ^ z[col]);
        for w in SLOT_BASE..USEFUL_BITS {
            let offset = (w - SLOT_BASE) % SLOT_STRIDE;
            if offset < Q + 3 {
                continue;
            }
            block[w] = eval(&r1cs.a_0.rows[w], &block) & eval(&r1cs.b_0.rows[w], &block);
        }
        assert!(
            !r1cs.satisfies(&tiled(&block)),
            "the wrong quotient must break a forced-zero row"
        );
    }

    #[test]
    fn real_shake_stream_matches_the_reference_compaction() {
        // The full-record differential on a genuinely Falcon-framed XOF
        // stream: SHAKE256(salt || hpk || 0x00 || 0x00 || message), 612
        // big-endian candidates, against an independent integer reference.
        use sha3::Shake256;
        use sha3::digest::{ExtendableOutput, Update, XofReader};

        let salt = [7_u8; 40];
        let hpk = [42_u8; 64];
        let mut hasher = Shake256::default();
        hasher.update(&salt);
        hasher.update(&hpk);
        hasher.update(&[0x00, 0x00]);
        hasher.update(b"hash-to-point record differential");
        let mut reader = hasher.finalize_xof();
        let mut words = [0_u16; SLOTS];
        for word in words.iter_mut() {
            let mut bytes = [0_u8; 2];
            reader.read(&mut bytes);
            *word = u16::from_be_bytes(bytes);
        }

        let (block, counter) = build_block_witness(&words);
        assert!(
            counter >= 512,
            "612 candidates yield 512 acceptances w.h.p."
        );
        assert!(single_block_r1cs().satisfies(&tiled(&block)));

        // Independent reference: the first 512 accepted (a, u) in order.
        let mut reference = Vec::new();
        for &word in &words {
            if u32::from(word) < ACCEPT_BOUND && reference.len() < 512 {
                let residue = u32::from(word) % FALCON_Q;
                reference.push((residue as u16, residue > (FALCON_Q - 1) / 2));
            }
        }
        assert_eq!(reference.len(), 512);

        // The gated slots reproduce exactly that sequence, in order, with
        // `a` recovered from the committed quotient wires.
        let mut gated = Vec::new();
        for (slot, &word) in words.iter().enumerate() {
            if block[gate_position(slot)] {
                let base = SLOT_BASE + slot * SLOT_STRIDE;
                let mut quotient = 0_u16;
                for i in 0..3 {
                    quotient |= u16::from(block[base + Q + i]) << i;
                }
                gated.push((word - 12_289 * quotient, block[centering_position(slot)]));
            }
        }
        assert_eq!(gated, reference);
    }

    #[test]
    fn prove_verify_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        // 64 records is the smallest default-Ligerito-legal size (m = 22).
        let n_blocks = 64;
        let setup = SlotSetup::new(n_blocks);
        let blocks: Vec<[u16; SLOTS]> = (0..n_blocks)
            .map(|block| {
                let mut words = [0_u16; SLOTS];
                for (slot, word) in words.iter_mut().enumerate() {
                    *word = ((block * SLOTS + slot) as u16)
                        .wrapping_mul(9_973)
                        .wrapping_add(211);
                }
                words
            })
            .collect();
        let mut prover_challenger = FsChallenger::new(b"aerie-hash-to-point-slots");
        let (proof, commitment, claim) = setup.prove_ligerito(&blocks, &mut prover_challenger);
        let mut verifier_challenger = FsChallenger::new(b"aerie-hash-to-point-slots");
        let verified = setup
            .verify(&commitment, &proof, &mut verifier_challenger)
            .expect("honest slot proof verifies");
        assert_eq!(verified, claim);

        // A domain-separated transcript must reject the same proof.
        let mut wrong_domain = FsChallenger::new(b"aerie-hash-to-point-other");
        assert!(
            setup
                .verify(&commitment, &proof, &mut wrong_domain)
                .is_err()
        );
    }
}
