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
//! ## Layout (K_LOG = 17, PLANE-MAJOR, one record per block)
//!
//! Each wire class occupies its own 1,024-slot aligned plane:
//! `wire = plane * 1,024 + slot`. This is what makes the scatter's
//! terminal discharge succinct: every per-slot class table is a sub-cube
//! of the witness, so its MLE at `r` is an ordinary opening at
//! `(plane bits, r)`, and the counter prefix parities reduce through one
//! less-than-weighted sumcheck (the LT multilinear is verifier-evaluable
//! in `O(n)`). 101 planes, `USEFUL_BITS = 103,424` of 131,072 (79%).
//!
//! ```text
//! plane 0        constant-one wire at slot 0 (const_pin); rest forced zero
//! planes 1..17   x bits (free inputs)
//! planes 17..20  q bits (free inputs)
//! planes 20..36  accept-chain products;   36 accept flag
//! planes 37..40  3q-adder products;       40..56 borrow products
//! planes 56..70  centering products;      70 centering flag
//! planes 71..85  range products;          85..90 forced-zero rows
//! planes 90..100 counter products;        100 write gate
//! planes 101..111 materialized counter-after bits
//! planes 112..120 the dense Z_H region (free inputs, scatter-bound)
//! rest           zero padding (empty rows)
//! ```

use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

use super::common::build_block_r1cs_with_matrices;

pub const K_LOG: usize = 17;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

/// Squeeze blocks in the default bucket (spec Section 4).
pub const SQUEEZE_BLOCKS: usize = 9;
/// Candidates per record block: nine SHAKE256 rate blocks.
pub const SLOTS: usize = SQUEEZE_BLOCKS * 68;

/// The padded slot domain: each wire class is one aligned plane of this
/// many slots, so class tables are sub-cubes of the witness.
pub const PLANE: usize = 1 << SLOT_VARS;
/// log2 of the plane size.
pub const SLOT_VARS: usize = 10;

pub const COUNTER_BITS: usize = 10;
pub const Z_CONST_POS: usize = 0;

// Plane indices (wire = plane * PLANE + slot).
const X: usize = 1; // 16 planes: candidate word bits, free inputs
const Q: usize = 17; // 3 planes: quotient bits, free inputs
const D: usize = 20; // 16 planes: accept-chain carry products
const ACC: usize = 36; // materialized accept flag
const E: usize = 37; // 3 planes: carry products of the 3q mini-adder
const G: usize = 40; // 16 planes: borrow products of x - M
const U_CHAIN: usize = 56; // 14 planes: centering-chain carry products
const U: usize = 70; // materialized centering flag
const R_CHAIN: usize = 71; // 14 planes: range-chain carry products
const PIN_RANGE: usize = 85; // forced zero: carry-out of a + 4,095
const PIN_A14: usize = 86; // forced zero: subtraction bit 14
const PIN_A15: usize = 87; // forced zero: subtraction bit 15
const PIN_B16: usize = 88; // forced zero: final borrow of x - M
const PIN_T4: usize = 89; // forced zero: t_4 of 3q, so q <= 5
const H: usize = 90; // 10 planes: counter-increment carry products
const GATE: usize = 100; // write gate: acc AND (count-before < 512)
/// Materialized counter-AFTER bits (ten planes): slot `s` holds the
/// count after `s`'s increment. Keeping these as wires makes every
/// counter tap O(1) (the symbolic prefix supports were O(s) per slot)
/// and turns the scatter's counter factors into plain sub-cube openings:
/// on gated slots `beta^(count-before) = beta^(-1) beta^(count-after)`.
const CNT: usize = 101;
/// The dense `Z_H` region: 512 output indices x 16 coordinate planes
/// (15 live) = 2^13 bits per record, as eight planes at an eight-aligned
/// base so the region is a sub-cube (top-four plane bits = 14). Free
/// inputs; the scatter argument binds them to the gated slot outputs.
pub const Z_BASE: usize = 112;
const Z_PLANES: usize = 8;
const PLANES: usize = Z_BASE + Z_PLANES;
pub const USEFUL_BITS: usize = PLANES * PLANE;

/// The wire of `plane` at `slot`.
const fn pos(plane: usize, slot: usize) -> usize {
    plane * PLANE + slot
}

/// Free inputs: the constant wire, the x bits and q bits of live slots,
/// and the `Z_H` region. Shared by witness generation and the tests'
/// re-derivation.
fn is_input(w: usize) -> bool {
    let plane = w / PLANE;
    let slot = w % PLANE;
    w == Z_CONST_POS
        || (slot < SLOTS && (X..Q + 3).contains(&plane))
        || (Z_BASE..Z_BASE + Z_PLANES).contains(&plane)
}

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
/// forms `value_bit(i)`; product `i` lands in plane `plane_base + i` at
/// `slot`. Returns the symbolic carry-out.
fn constant_add_chain(
    builder: &mut Builder,
    plane_base: usize,
    slot: usize,
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
        carry = carry_step(
            builder,
            pos(plane_base + i, slot),
            &value_bit(i),
            &k_bit,
            &carry,
        );
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
        let x = |i: usize| wire(pos(X + i, slot));
        for i in 0..16 {
            builder.input(pos(X + i, slot));
        }
        for i in 0..3 {
            builder.input(pos(Q + i, slot));
        }
        let q = |i: usize| wire(pos(Q + i, slot));

        // Accept: reject flag is the carry-out of x + 4,091.
        let reject = constant_add_chain(&mut builder, D, slot, ACCEPT_K, 16, x);
        builder.define(pos(ACC, slot), &lin_xor(&reject, &constant_one()));
        let acc = wire(pos(ACC, slot));

        // t = 3 q = q + (q << 1); bits: t_0 = q_0, t_1 = q_1 ^ q_0,
        // t_2 = q_2 ^ q_1 ^ e_2, t_3 = q_2 ^ e_3, t_4 = e_4.
        let e2 = carry_step(&mut builder, pos(E, slot), &q(1), &q(0), &Vec::new());
        let e2 = lin_xor(&e2, &Vec::new());
        let e3 = carry_step(&mut builder, pos(E + 1, slot), &q(2), &q(1), &e2);
        let e4 = carry_step(&mut builder, pos(E + 2, slot), &Vec::new(), &q(2), &e3);
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
            borrow = carry_step(&mut builder, pos(G + i, slot), &not_x, &m_bit(i), &borrow);
        }

        // Centering flag: carry-out of a + 10,239 over 14 bits.
        let center = constant_add_chain(&mut builder, U_CHAIN, slot, CENTER_K, 14, |i| {
            a_bits[i].clone()
        });
        builder.define(pos(U, slot), &center);

        // Range flag: carry-out of a + 4,095 over 14 bits; forced zero.
        let range = constant_add_chain(&mut builder, R_CHAIN, slot, RANGE_K, 14, |i| {
            a_bits[i].clone()
        });
        builder.force_zero(pos(PIN_RANGE, slot), &range);
        builder.force_zero(pos(PIN_A14, slot), &a_bits[14]);
        builder.force_zero(pos(PIN_A15, slot), &a_bits[15]);
        builder.force_zero(pos(PIN_B16, slot), &borrow);
        // t_4 is the SYMBOLIC carry e4 (product xor e3), not the raw product
        // wire; forcing it zero is the q <= 5 range bound.
        builder.force_zero(pos(PIN_T4, slot), &e4);

        // Write gate: acc AND (count-before-this-slot < 512). The counter
        // never reaches 1,024 in a 612-slot record, so the condition is
        // one bit: gate = acc * (1 ^ cnt_9).
        builder.product(
            pos(GATE, slot),
            &acc,
            &lin_xor(&counter[COUNTER_BITS - 1], &constant_one()),
        );

        // Counter increment by the accept flag: h_0 = acc,
        // h_{i+1} = cnt_i & h_i, cnt_i' = cnt_i ^ h_i; the new counter
        // bits are MATERIALIZED so every later tap is O(1).
        let mut increment = acc;
        for (bit, cnt_bit) in counter.iter_mut().enumerate() {
            builder.product(pos(H + bit, slot), cnt_bit, &increment);
            let carried = wire(pos(H + bit, slot));
            builder.define(pos(CNT + bit, slot), &lin_xor(cnt_bit, &increment));
            *cnt_bit = wire(pos(CNT + bit, slot));
            increment = carried;
        }
        // A record has at most 680 slots, so the counter cannot wrap; the
        // final increment carry is left dangling by design.
        let _ = increment;
    }

    // The Z_H region: free inputs, bound by the scatter argument.
    for plane in Z_BASE..Z_BASE + Z_PLANES {
        for slot in 0..PLANE {
            builder.input(pos(plane, slot));
        }
    }

    let to_matrix = |rows: Vec<Vec<usize>>| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_matrix(builder.a), to_matrix(builder.b))
}

/// Block accessors for tests and the scatter argument.
pub fn accept_position(slot: usize) -> usize {
    pos(ACC, slot)
}
pub fn centering_position(slot: usize) -> usize {
    pos(U, slot)
}
/// The scatter's write gate: accepted and before the 512th acceptance.
pub fn gate_position(slot: usize) -> usize {
    pos(GATE, slot)
}
/// Committed quotient bit `bit` of `slot`.
pub fn quotient_position(slot: usize, bit: usize) -> usize {
    pos(Q + bit, slot)
}
/// Committed candidate word bit `bit` of `slot`.
pub fn word_position(slot: usize, bit: usize) -> usize {
    pos(X + bit, slot)
}
/// Committed borrow product `bit` of `slot` (`x - M` chain).
pub fn borrow_position(slot: usize, bit: usize) -> usize {
    pos(G + bit, slot)
}
/// Committed counter-increment product `bit` of `slot`.
pub fn increment_position(slot: usize, bit: usize) -> usize {
    pos(H + bit, slot)
}
/// Materialized counter-AFTER bit `bit` of `slot`.
pub fn counter_position(slot: usize, bit: usize) -> usize {
    pos(CNT + bit, slot)
}
/// The plane index of a wire, for sub-cube opening points.
pub fn plane_of(position: usize) -> usize {
    position / PLANE
}
/// The `Z_H` region wire for output index `j`, coordinate plane `c`
/// (`c < 15` live: 14 residue bits, then the centering flag).
pub fn zh_position(j: usize, c: usize) -> usize {
    let index = (j << 4) | c;
    pos(Z_BASE + (index >> SLOT_VARS), index & (PLANE - 1))
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
        for i in 0..16 {
            z[pos(X + i, slot)] = (word >> i) & 1 == 1;
        }
        // The honest quotient; for rejected words it is still the true
        // quotient (capped by the pin only through q <= 5, x >= M, a < q).
        let quotient = (u32::from(word) / FALCON_Q).min(5) as u16;
        for i in 0..3 {
            z[pos(Q + i, slot)] = (quotient >> i) & 1 == 1;
        }
    }
    let eval = |lin: &[usize], z: &[bool]| lin.iter().fold(false, |acc, &col| acc ^ z[col]);
    // Dependency-aware order: the chains below plane H are slot-local and
    // ascending; the counter family alternates between planes across
    // slots (H(s) taps CNT(s-1), CNT(s) taps H(s)), so it is evaluated
    // slot-major; everything above is inputs or forced-zero padding.
    for w in 0..H * PLANE {
        if is_input(w) {
            continue;
        }
        z[w] = eval(&a_0.rows[w], &z) & eval(&b_0.rows[w], &z);
    }
    for slot in 0..PLANE {
        let gate = pos(GATE, slot);
        z[gate] = eval(&a_0.rows[gate], &z) & eval(&b_0.rows[gate], &z);
        for bit in 0..COUNTER_BITS {
            let h = pos(H + bit, slot);
            z[h] = eval(&a_0.rows[h], &z) & eval(&b_0.rows[h], &z);
            let cnt = pos(CNT + bit, slot);
            z[cnt] = eval(&a_0.rows[cnt], &z) & eval(&b_0.rows[cnt], &z);
        }
    }
    for w in (CNT + COUNTER_BITS) * PLANE..K {
        if is_input(w) {
            continue;
        }
        z[w] = eval(&a_0.rows[w], &z) & eval(&b_0.rows[w], &z);
    }

    // Fill the Z_H region from the gated slot outputs, in counter order.
    let mut index = 0_usize;
    for (slot, &word) in words.iter().enumerate() {
        if !z[pos(GATE, slot)] {
            continue;
        }
        let quotient = (u32::from(word) / FALCON_Q) as u16;
        let residue = word - 12_289 * quotient;
        for c in 0..14 {
            z[zh_position(index, c)] = (residue >> c) & 1 == 1;
        }
        z[zh_position(index, 14)] = z[pos(U, slot)];
        index += 1;
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
            pos(G + 5, 0),
            pos(D + 9, 2),
            pos(H + 1, 0),
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
        let word = words[slot];
        let wrong_quotient = (u32::from(word) / FALCON_Q + 1) as u16;
        for i in 0..3 {
            block[quotient_position(slot, i)] = (wrong_quotient >> i) & 1 == 1;
        }
        let r1cs = single_block_r1cs();
        let eval = |lin: &[usize], z: &[bool]| lin.iter().fold(false, |acc, &col| acc ^ z[col]);
        for w in 0..USEFUL_BITS {
            if is_input(w) {
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
                let mut quotient = 0_u16;
                for i in 0..3 {
                    quotient |= u16::from(block[quotient_position(slot, i)]) << i;
                }
                gated.push((word - 12_289 * quotient, block[centering_position(slot)]));
            }
        }
        assert_eq!(gated, reference);
    }

    #[test]
    fn prove_verify_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        // 32 records is the smallest default-Ligerito-legal size (m = 22).
        let n_blocks = 32;
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
