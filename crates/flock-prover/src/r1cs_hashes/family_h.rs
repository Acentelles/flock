//! **Family-H tables**: the boolean side of the ring-switch verifier's
//! F₂-linear maps, for the recursion circuit (route A, 2026-08-04 — see the
//! handoff's "ROUTE A's REFINED DESIGN" and the wiring doc §boundary).
//!
//! The recursion parent replays a child verifier whose ring-switch region
//! computes two scalars natively:
//!
//! - `rs_half = Σ_k γ_k·⟨transpose(s_hat_v_k), eq(r″_k)⟩` — the TensorAlgebra
//!   BIT transpose ([`flock_core::pcs::ring_switch::tensor_algebra_transpose`])
//!   dotted with the fold-challenge eq table, and
//! - `V_rs = Σ_k Σ_j c_{k,j}·B_{k,j}^{2^j}` — the linearized-coefficient
//!   recombination.
//!
//! Element-class gates do field arithmetic and cannot see bits, so the
//! transpose lives here as ONE boolean table row: 128 input words in, 128
//! output words out, relation = the pure bit permutation
//! `bit i of u[b] == bit b of v[i]` (F₂-linear — one two-entry `A` row per
//! output bit against the constant column, ~49k nonzeros total, invisible
//! next to BLAKE3's 21M). The dot against the eq wires then happens
//! element-side over the word-aligned outputs (cross-class copy is the
//! identity in the standard basis).
//!
//! The committed cost is trivial — height-`n_t` stacking commits
//! `n_t × used_cols` for every class, so the two rows (one per RS region)
//! commit ~512 words. The real cost is CELL SLOTS: 256 schema words. That is
//! this table's share of the mu 24→26 step route A accepted.
//!
//! (`V_rs`'s coefficient half is NOT a boolean table — `c_{k,j} =
//! γ_k·M̂inv_j(r″_k)` is a constant-lane MLE fold at the wired challenge
//! point, an element-class gate; see
//! `ring_switch::linearized_coefficients_are_moore_row_mles`.)

use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_core::schedule::IoWord;

/// The 128×128 bit-matrix transpose as one boolean table row.
///
/// Column layout (`k_log = 16` — 32,769 useful columns, the constant column
/// lands just past the 2^15 boundary):
///
/// ```text
///   0      .. 16384   v: input word i's bits at [128·i, 128·i+128)
///   16384  .. 32768   u: output word b's bits at [16384 + 128·b, ..+128)
///   32768             the constant-one column
/// ```
///
/// Relation, per output bit: `u[b] bit i == v[i] bit b` (`A = [that v bit]`,
/// `B = [const]`, `C = identity`). Input bits are free rows — their bit-ness
/// is inherited (the wired sources are committed boolean words), and the
/// relation binds every one of them through the outputs.
pub struct TransposeTable;

impl TransposeTable {
    pub const K_LOG: usize = 16;
    pub const V: usize = 0;
    pub const U: usize = 128 * 128;
    pub const CONST: usize = 2 * 128 * 128;
    pub const USEFUL_BITS: usize = Self::CONST + 1;

    pub fn k() -> usize {
        1usize << Self::K_LOG
    }

    /// 128 inputs (`v`, word columns 0..128) then 128 outputs (`u`, word
    /// columns 128..256) — every word wired, which is the point: binding the
    /// inputs to the absorbed `s_hat_v` stream words IS the content.
    pub fn io_schema() -> Vec<IoWord> {
        (0..128)
            .map(IoWord::input)
            .chain((0..128).map(|b| IoWord::output(128 + b)))
            .collect()
    }

    pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = Self::k();
        let gc = Self::CONST;
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];
        // Input bits ride free.
        for r in 0..128 * 128 {
            a[Self::V + r] = vec![Self::V + r];
            b[Self::V + r] = vec![gc];
        }
        // Output bit (b_idx, i) copies input bit (i, b_idx).
        for b_idx in 0..128 {
            for i in 0..128 {
                let r = Self::U + b_idx * 128 + i;
                a[r] = vec![Self::V + i * 128 + b_idx];
                b[r] = vec![gc];
            }
        }
        a[gc] = vec![gc];
        b[gc] = vec![gc];
        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        };
        (m(a), m(b))
    }

    pub fn build_block_r1cs(n_log: usize) -> BlockR1cs {
        let (a_0, b_0) = Self::build_matrices();
        BlockR1cs {
            m: n_log + Self::K_LOG,
            k_log: Self::K_LOG,
            k_skip: flock_core::zerocheck::K_SKIP,
            useful_bits: Self::USEFUL_BITS,
            a_0,
            b_0,
            c_0: super::common::identity(Self::k()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(Self::CONST),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }

    /// The `(z, a, b)` row-witness for one transpose: inputs `v` (the 128
    /// absorbed `s_hat_v` words), outputs their bit transpose.
    pub fn build_witness(v: &[F128]) -> [Vec<bool>; 3] {
        assert_eq!(v.len(), 128, "one transpose row carries 128 words");
        let u = flock_core::pcs::ring_switch::tensor_algebra_transpose(v);
        let k = Self::k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        let word_bit = |w: F128, t: usize| {
            if t < 64 {
                (w.lo >> t) & 1 == 1
            } else {
                (w.hi >> (t - 64)) & 1 == 1
            }
        };
        for i in 0..128 {
            for t in 0..128 {
                let set = word_bit(v[i], t);
                z[Self::V + i * 128 + t] = set;
                a[Self::V + i * 128 + t] = set;
                b[Self::V + i * 128 + t] = true;
            }
        }
        for b_idx in 0..128 {
            for i in 0..128 {
                let r = Self::U + b_idx * 128 + i;
                let set = word_bit(u[b_idx], i);
                z[r] = set;
                a[r] = word_bit(v[i], b_idx);
                b[r] = true;
            }
        }
        z[Self::CONST] = true;
        a[Self::CONST] = true;
        b[Self::CONST] = true;
        [z, a, b]
    }
}

impl TransposeTable {
    /// The union witness for a batch of transpose rows (each row = its 128
    /// input words), BatchMajor — the [`SwapTable`]-family driver contract.
    ///
    /// PREFER [`Self::generate_witness_into`]: this materializes the slot's
    /// FULL capacity block (`512 · 2^ν` words per buffer — half a GiB at
    /// ν 16) for a handful of live rows. Kept for shape tests.
    ///
    /// [`SwapTable`]: super::merkle_glue::SwapTable
    pub fn generate_witness_batch_major(
        rows: &[Vec<F128>],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(|v| Self::build_witness(v)).collect();
        super::merkle_glue::scatter_zab(&per, Self::k(), Self::USEFUL_BITS, nu)
    }

    /// The in-place, LIVE-PROPORTIONAL driver: write only the live rows'
    /// words into the slot's aligned union block
    /// ([`UnionSlotProverInput::in_place`]'s contract) and return the
    /// lincheck stripe. This slot's capacity block is `2^{16+ν−7}` words —
    /// 32 M at ν 16 for TWO live rows — so the full-block paths above are
    /// exactly the capacity tax this driver elides.
    ///
    /// When the destination's `elide_padding_writes` is false (the pooled
    /// zeroed mode) the dead words must be written as honest zeros; that
    /// path zero-fills first and is capacity-priced — the padding-dominant
    /// shapes this slot ships in never take it.
    ///
    /// The stripe is allocated lazily (`vec![0u8; ..]` gets untouched zero
    /// pages) and only the live groups are written; the union lincheck reads
    /// declared counts only, so the tail stays untouched.
    ///
    /// [`UnionSlotProverInput::in_place`]: ../prover/struct.UnionSlotProverInput.html
    pub fn generate_witness_into(
        rows: &[Vec<F128>],
        dst: flock_core::union::SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        let k = Self::k();
        let words_per_block = k / 128;
        assert_eq!(dst.z.len() % words_per_block, 0, "aligned slot block");
        let n_total = dst.z.len() / words_per_block;
        assert!(rows.len() <= n_total, "live rows fit the capacity");
        if !dst.elide_padding_writes {
            use rayon::prelude::*;
            for buf in [&mut *dst.z, &mut *dst.a, &mut *dst.b] {
                buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(F128::ZERO));
            }
        }
        let nu = n_total.trailing_zeros() as usize;
        let mut stripe = vec![0u8; (n_total / 8) * k];
        for (i, v) in rows.iter().enumerate() {
            let [pz, pa, pb] = Self::build_witness(v);
            for w in 0..words_per_block {
                let addr = (w << nu) + i;
                dst.z[addr] = pack_bits(&pz, w * 128);
                dst.a[addr] = pack_bits(&pa, w * 128);
                dst.b[addr] = pack_bits(&pb, w * 128);
            }
            let chunk = &mut stripe[(i / 8) * k..(i / 8 + 1) * k];
            for c in 0..Self::USEFUL_BITS {
                if pz[c] {
                    chunk[c] |= 1u8 << (i % 8);
                }
            }
        }
        stripe
    }
}

/// The 128 bools at `[base, base+128)` as one `F128` (the
/// `merkle_glue::pack_word` contract, local so the scatter helper stays
/// private there).
fn pack_bits(bits: &[bool], base: usize) -> F128 {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for t in 0..64 {
        if bits[base + t] {
            lo |= 1u64 << t;
        }
        if bits[base + 64 + t] {
            hi |= 1u64 << t;
        }
    }
    F128 { lo, hi }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row-witness satisfies the relation and its output words ARE the
    /// native transpose — the differential anchor for the circuit emission.
    #[test]
    fn transpose_row_satisfies_and_matches_native() {
        let mut seed = 0x7472_414Eu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        let v: Vec<F128> = (0..128)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let [z, a, b] = TransposeTable::build_witness(&v);
        let (a0, b0) = TransposeTable::build_matrices();
        // (A z) ∘ (B z) = z, bitwise over F2.
        for r in 0..TransposeTable::k() {
            let av = a0.rows[r].iter().fold(false, |acc, &c| acc ^ z[c]);
            let bv = b0.rows[r].iter().fold(false, |acc, &c| acc ^ z[c]);
            assert_eq!(av && bv, z[r], "constraint row {r}");
            assert_eq!(av, a[r], "a witness row {r}");
            assert_eq!(bv, b[r], "b witness row {r}");
        }
        // Output words == the native transpose.
        let u = flock_core::pcs::ring_switch::tensor_algebra_transpose(&v);
        for b_idx in 0..128 {
            let base = TransposeTable::U + b_idx * 128;
            let mut lo = 0u64;
            let mut hi = 0u64;
            for i in 0..64 {
                if z[base + i] {
                    lo |= 1 << i;
                }
                if z[base + 64 + i] {
                    hi |= 1 << i;
                }
            }
            assert_eq!(F128 { lo, hi }, u[b_idx], "output word {b_idx}");
        }
        // And the recombination the dot consumes: Σ_b eq[b]·u[b] equals the
        // native inner product (sanity that word packing agrees end to end).
        let eq: Vec<F128> = (0..128)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let dot = flock_core::pcs::ring_switch::inner_product(&u, &eq);
        let manual = (0..128).fold(F128::ZERO, |acc, b_idx| acc + eq[b_idx] * u[b_idx]);
        assert_eq!(dot, manual);
    }
}
