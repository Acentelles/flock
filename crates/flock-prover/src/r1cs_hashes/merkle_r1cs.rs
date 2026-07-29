//! Merkle-path verification as a **single monolithic R1CS block** — one
//! table row per path.
//!
//! Distinct from [`super::merkle_path_common`], which proves Merkle paths
//! with a bespoke shift sumcheck layered on top of a batch of independent
//! per-level compressions. Here the whole path lives in ONE witness block,
//! so the level-to-level dataflow is expressed as ordinary R1CS rows. That
//! is what makes a Merkle path a legal **table type** for the multi-table
//! union (`flock_core::schedule::TableType`): the union model forbids
//! constraints between rows (design doc §3, "no constraints connecting
//! different rows, neither within a table nor across tables"), so one row
//! must be one self-contained path.
//!
//! ## Statement
//!
//! For a leaf digest `L`, an index `i ∈ [0, 2^depth)`, sibling digests
//! `S_0 … S_{depth−1}` and a root `R`, each level `l` computes
//!
//! ```text
//!   b_l    = bit l of i                       (the indicator bit)
//!   left   = b_l·prev ⊕ (1 ⊕ b_l)·S_l
//!   right  = left ⊕ prev ⊕ S_l
//!   prev'  = H(left ‖ right)
//! ```
//!
//! with `prev = L` at level 0 and `R = prev` after the last level. Over
//! GF(2) the pair `(left, right)` is exactly the conditional swap: `b_l = 1`
//! puts the running digest on the left, `b_l = 0` on the right.
//!
//! ## The swap gadget under `C = I`
//!
//! The R1CS is the circuit shape `(A·z) ⊙ (B·z) = z`: every witness column
//! is the output of exactly one row, so a row's right-hand side is a single
//! wire and linear relations need the constant-one wire on the `B` side.
//! `left` is quadratic in the witness, so it needs one AND per bit:
//!
//! ```text
//!   t_j      = b_l · (prev_j ⊕ S_{l,j})     A = [b_l],        B = [prev_j, S_{l,j}]
//!   left_j   = S_{l,j} ⊕ t_j                A = [S_{l,j}, t_j], B = [const]
//!   right_j  = prev_j  ⊕ t_j                A = [prev_j, t_j],  B = [const]
//! ```
//!
//! (`right = left ⊕ prev ⊕ S = t ⊕ prev`, so both halves cost one linear
//! row.) Crucially `left_j`/`right_j` are **not** new columns: they ARE the
//! hash block's 512-bit message region, whose rows the composite *replaces*
//! — the base encoder makes them free inputs, we make them gadget outputs.
//! So the gadget costs only the `2^8` AND columns `t_j` per level.
//!
//! ## Composite layout (`depth = D`, base block `useful_bits = U`)
//!
//! ```text
//!   z[0]                                  = 1            (the table's const_pin)
//!   z[1        .. 257)                    = leaf digest        (free input)
//!   z[257      .. 257+D)                  = index bits b_0..b_D (free input)
//!   per level l, base LB(l) = 257+D + l·(512+U):
//!     z[LB(l)       .. LB(l)+256)         = sibling S_l        (free input)
//!     z[LB(l)+256   .. LB(l)+512)         = t_l (the ANDs)
//!     z[LB(l)+512   .. LB(l)+512+U)       = the hash block, verbatim
//!   z[useful_bits .. 2^k_log)             = padding (forced 0 by empty rows)
//! ```
//!
//! Each level embeds the base encoder's block by a **pure column offset**;
//! the composite then overrides exactly three row groups per level: the
//! block's own constant wire (re-derived from the global one), the 512-bit
//! message region (the swap gadget above), and every other free input —
//! the input chaining value, counter, block length and flags — which is
//! pinned to the Merkle node constants. Everything else, all `2^k_log`-wide
//! rows of the hash relation, is the base matrix with its column indices
//! shifted.
//!
//! ## What this does NOT enforce
//!
//! Public-input binding, exactly as in the per-hash encoders: the leaf, the
//! index bits and the root are free witness columns at fixed offsets
//! ([`MerkleTreeLayout::leaf_bit`], [`MerkleTreeLayout::index_bit`],
//! [`MerkleTreeLayout::root_bit`]). Binding them to public values is the job
//! of the claim-level glue (or the planned lookup/bus layer — design doc
//! §8, §11).

use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

use super::blake3;
use super::common::{empty_matrix, identity};

/// Bits in one digest / chaining value. Both supported encoders lay their
/// input and output chaining values out as aligned `2^8`-bit slots.
pub const SLOT_BITS: usize = 256;
/// 32-bit words per digest.
pub const SLOT_WORDS: usize = SLOT_BITS / 32;

// ---------------------------------------------------------------------------
// Merkle node constants
// ---------------------------------------------------------------------------

/// Counter input to every node compression. Merkle parent nodes are not
/// chunks, so the chunk counter is 0.
pub const NODE_COUNTER: u64 = 0;
/// Block length: a parent node compresses exactly two 32-byte digests.
pub const NODE_BLOCK_LEN: u32 = 64;
/// BLAKE3 `PARENT` domain flag. Note this is applied at EVERY level,
/// including the top one — real BLAKE3 tree hashing also sets `ROOT` on the
/// final parent. Keeping all levels uniform is what lets the composite be
/// `depth` copies of one base block; set [`HashSpec::flags`] if you need the
/// bit-exact BLAKE3 tree.
pub const BLAKE3_FLAG_PARENT: u32 = 4;

// ---------------------------------------------------------------------------
// Hash backend description
// ---------------------------------------------------------------------------

/// Geometry and witness hooks of the per-level hash's R1CS block.
///
/// The composite needs to know only where the base encoder keeps its input
/// chaining value, its output chaining value, its message region and its
/// constant wire — plus how to build one block's witness. Both shipped
/// encoders use the same "I/O-aligned" shape (input CV in aligned slot 0,
/// output CV in aligned slot 1, message right after), so a second backend is
/// one constructor.
#[derive(Clone)]
pub struct HashSpec {
    pub name: &'static str,
    /// log2 of the base block width.
    pub k_log: usize,
    /// Useful columns of the base block.
    pub useful_bits: usize,
    /// The base block's own constant-one column.
    pub z_const_pos: usize,
    /// Base offset of the 256-bit input chaining value.
    pub in_cv_base: usize,
    /// Base offset of the 256-bit output chaining value (the node digest).
    pub out_cv_base: usize,
    /// Base offset of the 512-bit message region: `left ‖ right`.
    pub msg_base: usize,
    /// Domain flags passed to every node compression.
    pub flags: u32,
    /// The base encoder's sparse `(A_0, B_0)`.
    pub build_matrices: fn() -> (SparseBinaryMatrix, SparseBinaryMatrix),
    /// One node compression's boolean witness block (length `2^k_log`),
    /// given the two child digests.
    pub node_witness: fn(&[u32; SLOT_WORDS], &[u32; SLOT_WORDS]) -> Vec<bool>,
    /// One node compression's ROW-witness `(z, a, b)` into three
    /// `2^k_log / 64`-word buffers, zero on entry.
    ///
    /// The composite copies these verbatim — the base encoder's row kinds
    /// already match every row the composite overrides, *provided* pin-to-zero
    /// uses `A = [], B = [const]` rather than an empty `B` (see
    /// `MerkleTreeLayout::extras`, override 3). That is why `a`/`b` never need
    /// a matrix application anywhere on this path.
    pub node_witness_ab:
        fn(&[u32; SLOT_WORDS], &[u32; SLOT_WORDS], &mut [u64], &mut [u64], &mut [u64]),
    /// Base-block columns the composite pins to constants, as
    /// `(column, value)`: every free input of the base encoder EXCEPT the
    /// message region, which the swap gadget drives.
    pub fixed_bits: fn() -> Vec<(usize, bool)>,
}

/// BLAKE3 backend (the default). One level = one
/// `compress(IV, left‖right, 0, 64, PARENT)`.
pub fn blake3_spec() -> HashSpec {
    HashSpec {
        name: "blake3",
        k_log: blake3::K_LOG,
        useful_bits: blake3::USEFUL_BITS,
        z_const_pos: blake3::Z_CONST_POS,
        in_cv_base: blake3::CV_BASE,
        out_cv_base: blake3::OUT_LO_BASE,
        msg_base: blake3::M_BASE,
        flags: BLAKE3_FLAG_PARENT,
        build_matrices: blake3::build_matrices,
        node_witness: blake3_node_witness,
        node_witness_ab: blake3_node_witness_ab,
        fixed_bits: blake3_fixed_bits,
    }
}

/// `left ‖ right` as BLAKE3's 16-word message block.
fn node_msg(left: &[u32; SLOT_WORDS], right: &[u32; SLOT_WORDS]) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[..SLOT_WORDS].copy_from_slice(left);
    m[SLOT_WORDS..].copy_from_slice(right);
    m
}

fn blake3_node_witness(left: &[u32; SLOT_WORDS], right: &[u32; SLOT_WORDS]) -> Vec<bool> {
    blake3::build_block_witness(
        &blake3::BLAKE3_IV,
        &node_msg(left, right),
        NODE_COUNTER,
        NODE_BLOCK_LEN,
        BLAKE3_FLAG_PARENT,
    )
}

fn blake3_node_witness_ab(
    left: &[u32; SLOT_WORDS],
    right: &[u32; SLOT_WORDS],
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    blake3::build_block_witness_ab_packed_into(
        &blake3::BLAKE3_IV,
        &node_msg(left, right),
        NODE_COUNTER,
        NODE_BLOCK_LEN,
        BLAKE3_FLAG_PARENT,
        z,
        a,
        b,
    );
}

/// BLAKE3's free inputs other than the message: `cv = IV`, `counter = 0`,
/// `block_len = 64`, `flags = PARENT`.
fn blake3_fixed_bits() -> Vec<(usize, bool)> {
    let w = blake3::WORD_BITS;
    let mut out = Vec::with_capacity(SLOT_BITS + 4 * w);
    for (word, iv) in blake3::BLAKE3_IV.iter().enumerate() {
        for b in 0..w {
            out.push((blake3::CV_BASE + word * w + b, (iv >> b) & 1 == 1));
        }
    }
    for (base, val) in [
        (blake3::T_LO_BASE, NODE_COUNTER as u32),
        (blake3::T_HI_BASE, (NODE_COUNTER >> 32) as u32),
        (blake3::BLEN_BASE, NODE_BLOCK_LEN),
        (blake3::FLAGS_BASE, BLAKE3_FLAG_PARENT),
    ] {
        for b in 0..w {
            out.push((base + b, (val >> b) & 1 == 1));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Composite layout
// ---------------------------------------------------------------------------

/// Column layout of the composite Merkle-path block. All offsets are bit
/// indices into one table row (one path).
#[derive(Clone)]
pub struct MerkleTreeLayout {
    pub spec: HashSpec,
    /// Number of levels = tree depth.
    pub depth: usize,
    /// log2 of the composite block width (smallest power of two ≥
    /// [`Self::useful_bits`]).
    pub k_log: usize,
    /// Useful columns of the composite block.
    pub useful_bits: usize,
    /// First column of a level's region; level `l` starts at
    /// `levels_base + l * level_stride`.
    pub levels_base: usize,
    /// Columns per level: `2·SLOT_BITS + spec.useful_bits`.
    pub level_stride: usize,
}

impl MerkleTreeLayout {
    /// The global constant-one column, and the table's `const_pin`.
    pub const CONST_POS: usize = 0;
    /// First column of the leaf digest.
    pub const LEAF_BASE: usize = 1;
    /// First column of the index bits.
    pub const INDEX_BASE: usize = Self::LEAF_BASE + SLOT_BITS;

    /// Lay out a `depth`-level Merkle path over `spec`.
    pub fn new(depth: usize, spec: HashSpec) -> Self {
        assert!(depth >= 1, "depth must be ≥ 1");
        let levels_base = Self::INDEX_BASE + depth;
        let level_stride = 2 * SLOT_BITS + spec.useful_bits;
        let useful_bits = levels_base + depth * level_stride;
        let k_log = useful_bits.next_power_of_two().trailing_zeros() as usize;
        assert!(
            k_log >= 7,
            "the union's BatchMajor chunking requires k_log ≥ 7"
        );
        Self {
            spec,
            depth,
            k_log,
            useful_bits,
            levels_base,
            level_stride,
        }
    }

    /// Composite width `2^k_log`.
    pub fn k(&self) -> usize {
        1usize << self.k_log
    }

    /// Bit `j` of the leaf digest.
    pub fn leaf_bit(&self, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        Self::LEAF_BASE + j
    }

    /// Indicator bit of level `l` (bit `l` of the index).
    pub fn index_bit(&self, level: usize) -> usize {
        debug_assert!(level < self.depth);
        Self::INDEX_BASE + level
    }

    fn level_base(&self, level: usize) -> usize {
        debug_assert!(level < self.depth);
        self.levels_base + level * self.level_stride
    }

    /// Bit `j` of level `l`'s sibling digest.
    pub fn sibling_bit(&self, level: usize, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        self.level_base(level) + j
    }

    /// Bit `j` of level `l`'s AND column `t = b_l · (prev ⊕ sibling)`.
    fn t_bit(&self, level: usize, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        self.level_base(level) + SLOT_BITS + j
    }

    /// Base-block column `c` of level `l`'s embedded hash block.
    pub fn hash_bit(&self, level: usize, c: usize) -> usize {
        debug_assert!(c < self.spec.useful_bits);
        self.level_base(level) + 2 * SLOT_BITS + c
    }

    /// Bit `j` of the digest entering level `l`: the leaf at level 0, else
    /// the previous level's output chaining value.
    pub fn prev_bit(&self, level: usize, j: usize) -> usize {
        if level == 0 {
            self.leaf_bit(j)
        } else {
            self.hash_bit(level - 1, self.spec.out_cv_base + j)
        }
    }

    /// Bit `j` of the root — the last level's output chaining value.
    pub fn root_bit(&self, j: usize) -> usize {
        self.hash_bit(self.depth - 1, self.spec.out_cv_base + j)
    }

    fn left_bit(&self, level: usize, j: usize) -> usize {
        self.hash_bit(level, self.spec.msg_base + j)
    }

    fn right_bit(&self, level: usize, j: usize) -> usize {
        self.hash_bit(level, self.spec.msg_base + SLOT_BITS + j)
    }

    // -----------------------------------------------------------------------
    // Matrices
    // -----------------------------------------------------------------------

    /// Which base-block ROWS the composite replaces, as a `spec.useful_bits`
    /// mask. Everything else is embedded verbatim at a column offset.
    ///
    /// Shared by [`Self::build_matrices`] and [`Self::build_walker`] so the
    /// materialized and walked forms cannot drift apart.
    fn base_overridden(&self) -> Vec<bool> {
        let mut ovr = vec![false; self.spec.useful_bits];
        ovr[self.spec.z_const_pos] = true;
        for j in 0..2 * SLOT_BITS {
            ovr[self.spec.msg_base + j] = true;
        }
        for &(c, _) in &(self.spec.fixed_bits)() {
            ovr[c] = true;
        }
        ovr
    }

    /// Every composite row that is NOT a shifted copy of a base-block row:
    /// the globals (constant, leaf, index), the per-level gadget columns
    /// (sibling, `t`), and the three override groups inside each hash block.
    ///
    /// These are the ONLY rows that reference columns outside their own
    /// level — which is exactly why the base embedding can be walked with a
    /// pure offset.
    fn extras(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let k = self.k();
        let gc = Self::CONST_POS;
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];

        // A free witness column: `z_c · 1 = z_c`, satisfied by any bit.
        let free = |a: &mut Vec<Vec<usize>>, b: &mut Vec<Vec<usize>>, r: usize| {
            a[r] = vec![r];
            b[r] = vec![gc];
        };

        // The global constant: `z_0 · z_0 = z_0`. Boolean idempotence leaves
        // 0 admissible here; `const_pin` is what forces it to 1 (see
        // `docs/const-wire-pin.md`).
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        for j in 0..SLOT_BITS {
            free(&mut a, &mut b, self.leaf_bit(j));
        }
        for l in 0..self.depth {
            free(&mut a, &mut b, self.index_bit(l));
        }

        let fixed = (self.spec.fixed_bits)();
        for l in 0..self.depth {
            for j in 0..SLOT_BITS {
                free(&mut a, &mut b, self.sibling_bit(l, j));
            }

            // t_j = b_l · (prev_j ⊕ sibling_j) — the only AND per bit.
            for j in 0..SLOT_BITS {
                let r = self.t_bit(l, j);
                a[r] = vec![self.index_bit(l)];
                b[r] = vec![self.prev_bit(l, j), self.sibling_bit(l, j)];
            }

            // Override 1: the block's own constant wire, re-derived from the
            // global one so a single column carries the table's const_pin.
            let r = self.hash_bit(l, self.spec.z_const_pos);
            a[r] = vec![gc];
            b[r] = vec![gc];

            // Override 2: the message region IS the swap gadget's output.
            for j in 0..SLOT_BITS {
                let rl = self.left_bit(l, j);
                a[rl] = vec![self.sibling_bit(l, j), self.t_bit(l, j)];
                b[rl] = vec![gc];
                let rr = self.right_bit(l, j);
                a[rr] = vec![self.prev_bit(l, j), self.t_bit(l, j)];
                b[rr] = vec![gc];
            }

            // Override 3: pin the remaining free inputs to the node
            // constants. Value 1 → `1·1 = 1`; value 0 → `0·1 = 0` (empty A,
            // const on B — NOT an empty B, so that the row-witness `b` bit
            // stays 1 and matches what the base encoder emits for the
            // free-input row this replaces).
            for &(c, v) in &fixed {
                let r = self.hash_bit(l, c);
                a[r] = if v { vec![gc] } else { Vec::new() };
                b[r] = vec![gc];
            }
        }
        (a, b)
    }

    /// Build the composite `(A_0, B_0)` in full.
    ///
    /// **Memory warning.** This materializes `depth` copies of the base
    /// encoder's matrices. BLAKE3's block carries ~21M nonzeros (its
    /// encoding trades density for a small `k` — see the `blake3` module
    /// docs), so depth 26 is ~547M nonzeros ≈ 4.4 GB. Use this at small
    /// depth as the reference oracle; at real depth use
    /// [`Self::build_walker`], which stores one base copy.
    pub fn build_matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = self.k();
        let (mut a, mut b) = self.extras();
        let (base_a, base_b) = (self.spec.build_matrices)();
        let ovr = self.base_overridden();
        debug_assert_eq!(base_a.num_rows, 1usize << self.spec.k_log);

        for l in 0..self.depth {
            // The hash block, embedded by a pure column offset. Overridden
            // rows are already written by `extras`.
            for c in 0..self.spec.useful_bits {
                if ovr[c] {
                    continue;
                }
                let r = self.hash_bit(l, c);
                a[r] = base_a.rows[c].iter().map(|&x| self.hash_bit(l, x)).collect();
                b[r] = base_b.rows[c].iter().map(|&x| self.hash_bit(l, x)).collect();
            }
        }

        // Rows [useful_bits, k) stay empty: `0 · 0 = z_r` forces the padding
        // columns to zero, which is what the union's run-list gating and the
        // dense-stack compaction rely on.
        debug_assert!(
            a.iter()
                .chain(b.iter())
                .flatten()
                .all(|&c| c < self.useful_bits),
            "a matrix row references a padding column"
        );
        (
            SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: a,
            },
            SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: b,
            },
        )
    }

    /// The composite [`BlockR1cs`] over `2^n_paths_log` rows (paths), with
    /// materialized matrices.
    ///
    /// Carries the same memory warning as [`Self::build_matrices`]. For the
    /// walker path use [`Self::build_block_r1cs_stub`].
    pub fn build_block_r1cs(&self, n_paths_log: usize) -> BlockR1cs {
        let (a_0, b_0) = self.build_matrices();
        self.block_r1cs_with(n_paths_log, a_0, b_0)
    }

    /// [`BlockR1cs`] with **empty `(A_0, B_0)` stubs** — the walker path.
    /// The constraints live in [`Self::build_walker`]'s
    /// [`LincheckCircuit`], following the same convention as the Keccak
    /// encoders (`r1cs_hashes::common`).
    ///
    /// Consequence, inherited by `Registry::digest`: the statement digest
    /// binds `k_log`, `useful_bits` and `const_pin` — hence the depth and,
    /// in practice, the hash backend — but **not** the constraint system.
    /// A verifier's guarantee rests on it constructing the matching walker
    /// out of band.
    pub fn build_block_r1cs_stub(&self, n_paths_log: usize) -> BlockR1cs {
        let k = self.k();
        self.block_r1cs_with(n_paths_log, empty_matrix(k), empty_matrix(k))
    }

    fn block_r1cs_with(
        &self,
        n_paths_log: usize,
        a_0: SparseBinaryMatrix,
        b_0: SparseBinaryMatrix,
    ) -> BlockR1cs {
        BlockR1cs {
            m: n_paths_log + self.k_log,
            k_log: self.k_log,
            k_skip: flock_core::zerocheck::K_SKIP,
            useful_bits: self.useful_bits,
            a_0,
            b_0,
            c_0: identity(self.k()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(Self::CONST_POS),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }

    /// Build the walker [`LincheckCircuit`]: ONE copy of the base block's
    /// CSC transpose, walked once per level at a column offset, plus the
    /// composite's small extras set. See [`MerkleWalkerCircuit`].
    pub fn build_walker(&self) -> MerkleWalkerCircuit {
        let (base_a, base_b) = (self.spec.build_matrices)();
        let ovr = self.base_overridden();

        // Empty the overridden rows before transposing: their base entries
        // do not apply to the composite, and their replacements are in
        // `extras`.
        let strip = |m: &SparseBinaryMatrix| -> SparseBinaryMatrix {
            let mut rows = m.rows.clone();
            for (c, o) in ovr.iter().enumerate() {
                if *o {
                    rows[c].clear();
                }
            }
            SparseBinaryMatrix {
                num_rows: m.num_rows,
                num_cols: m.num_cols,
                rows,
            }
        };
        let base = Csc::pair(&strip(&base_a), &strip(&base_b));
        drop((base_a, base_b));

        let k = self.k();
        let (xa, xb) = self.extras();
        let extras = Csc::pair(
            &SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: xa,
            },
            &SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: xb,
            },
        );

        MerkleWalkerCircuit {
            n_cols: k,
            depth: self.depth,
            levels_base: self.levels_base,
            level_stride: self.level_stride,
            base_useful: self.spec.useful_bits,
            base,
            extras,
            const_pin: Some(Self::CONST_POS),
        }
    }

    // -----------------------------------------------------------------------
    // Witness
    // -----------------------------------------------------------------------

    /// Boolean witness for ONE path (length `2^k_log`).
    ///
    /// Panics if `input.siblings.len() != depth` or the index does not fit
    /// in `depth` bits.
    pub fn build_witness(&self, input: &PathInput) -> Vec<bool> {
        assert_eq!(
            input.siblings.len(),
            self.depth,
            "need one sibling digest per level"
        );
        assert!(
            self.depth >= 64 || input.index < (1u64 << self.depth),
            "index {} does not fit in {} bits",
            input.index,
            self.depth
        );
        let mut z = vec![false; self.k()];
        z[Self::CONST_POS] = true;
        write_digest(&mut z, self.leaf_bit(0), &input.leaf);
        for l in 0..self.depth {
            z[self.index_bit(l)] = (input.index >> l) & 1 == 1;
        }

        let mut prev = input.leaf;
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            write_digest(&mut z, self.sibling_bit(l, 0), &sib);
            for j in 0..SLOT_BITS {
                z[self.t_bit(l, j)] = bit && (digest_bit(&prev, j) ^ digest_bit(&sib, j));
            }
            // b_l = 1 puts the running digest on the left.
            let (left, right) = if bit { (prev, sib) } else { (sib, prev) };
            let block = (self.spec.node_witness)(&left, &right);
            let dst = self.hash_bit(l, 0);
            z[dst..dst + self.spec.useful_bits]
                .copy_from_slice(&block[..self.spec.useful_bits]);
            prev = read_digest(&block, self.spec.out_cv_base);
        }
        z
    }

    /// Read the root digest back out of a witness built by
    /// [`Self::build_witness`].
    pub fn read_root(&self, z: &[bool]) -> [u32; SLOT_WORDS] {
        read_digest(z, self.root_bit(0))
    }

    /// Full ROW-witness `(z, a, b)` for ONE path, each of length `2^k_log`.
    ///
    /// `a = A_0·z` and `b = B_0·z` are emitted directly, never by matrix
    /// application: the base encoder's row-witness is copied in at the level
    /// offset (its row kinds already match the composite's overrides) and the
    /// gadget rows are written from the values at hand.
    pub fn build_witness_zab(&self, input: &PathInput) -> [Vec<bool>; 3] {
        assert_eq!(
            input.siblings.len(),
            self.depth,
            "need one sibling digest per level"
        );
        let k = self.k();
        let mut z = vec![false; k];
        let mut a = vec![false; k];
        let mut b = vec![false; k];

        // A free column: `z_c · 1 = z_c`.
        let free = |z: &mut [bool], a: &mut [bool], b: &mut [bool], c: usize, v: bool| {
            z[c] = v;
            a[c] = v;
            b[c] = true;
        };

        // The global constant: 1·1 = 1.
        z[Self::CONST_POS] = true;
        a[Self::CONST_POS] = true;
        b[Self::CONST_POS] = true;

        for j in 0..SLOT_BITS {
            free(&mut z, &mut a, &mut b, self.leaf_bit(j), digest_bit(&input.leaf, j));
        }
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            free(&mut z, &mut a, &mut b, self.index_bit(l), bit);
        }

        let base_words = (1usize << self.spec.k_log) / 64;
        let mut wz = vec![0u64; base_words];
        let mut wa = vec![0u64; base_words];
        let mut wb = vec![0u64; base_words];

        let mut prev = input.leaf;
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            for j in 0..SLOT_BITS {
                free(
                    &mut z,
                    &mut a,
                    &mut b,
                    self.sibling_bit(l, j),
                    digest_bit(&sib, j),
                );
            }
            // t_j = b_l · (prev_j ⊕ sibling_j): the AND row's operands are the
            // index bit and the XOR of the two candidate children.
            for j in 0..SLOT_BITS {
                let xor = digest_bit(&prev, j) ^ digest_bit(&sib, j);
                let c = self.t_bit(l, j);
                z[c] = bit && xor;
                a[c] = bit;
                b[c] = xor;
            }

            // b_l = 1 puts the running digest on the left.
            let (left, right) = if bit { (prev, sib) } else { (sib, prev) };
            wz.fill(0);
            wa.fill(0);
            wb.fill(0);
            (self.spec.node_witness_ab)(&left, &right, &mut wz, &mut wa, &mut wb);
            let dst = self.hash_bit(l, 0);
            for c in 0..self.spec.useful_bits {
                z[dst + c] = word_bit(&wz, c);
                a[dst + c] = word_bit(&wa, c);
                b[dst + c] = word_bit(&wb, c);
            }
            prev = read_digest_words(&wz, self.spec.out_cv_base);
        }
        [z, a, b]
    }

    /// One union slot's packed witness: `2^nu` rows (paths) in **BatchMajor**
    /// address order, plus the lincheck stripe.
    ///
    /// `paths.len()` may be less than `2^nu`; the trailing dummy rows are
    /// identically zero in all four outputs — **including the const-pin
    /// bit** — as the union's count-derived lincheck target requires
    /// (`flock_core::lincheck::union`). Do NOT substitute a "path over zero
    /// inputs": that would set the pin and be rejected.
    pub fn generate_witness_batch_major_partial(
        &self,
        paths: &[PathInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        let n_total = 1usize << nu;
        assert!(
            paths.len() <= n_total,
            "{} paths > 2^{nu} = {n_total} rows",
            paths.len()
        );
        assert!(
            n_total.is_multiple_of(8),
            "the lincheck stripe needs 2^nu ≥ 8 (nu ≥ 3)"
        );
        let k = self.k();
        let words_per_block = k / 128;
        let total = n_total * words_per_block;

        // Per-path row-witnesses first (embarrassingly parallel), then the
        // BatchMajor scatter + stripe transpose.
        let per_path: Vec<[Vec<bool>; 3]> = paths
            .par_iter()
            .map(|p| self.build_witness_zab(p))
            .collect();

        let mut z = vec![F128::ZERO; total];
        let mut a = vec![F128::ZERO; total];
        let mut b = vec![F128::ZERO; total];
        for (i, [pz, pa, pb]) in per_path.iter().enumerate() {
            for w in 0..words_per_block {
                // BatchMajor: word `w` of row `i` lives at `(w << nu) + i`.
                let addr = (w << nu) + i;
                z[addr] = pack_word(pz, w * 128);
                a[addr] = pack_word(pa, w * 128);
                b[addr] = pack_word(pb, w * 128);
            }
        }

        // Stripe: `stripe[g*k + c]` bit `r` = z of row `8g + r` at column c.
        // Rows past `paths.len()` contribute 0, so dummy rows stay zero.
        let mut stripe = vec![0u8; (n_total / 8) * k];
        stripe
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(g, chunk)| {
                for r in 0..8 {
                    let row = 8 * g + r;
                    if row >= per_path.len() {
                        continue;
                    }
                    let pz = &per_path[row][0];
                    for c in 0..self.useful_bits {
                        if pz[c] {
                            chunk[c] |= 1u8 << r;
                        }
                    }
                }
            });

        (z, a, b, stripe)
    }
}

/// Bit `i` of a u64-word buffer (LSB-first within each word).
#[inline]
fn word_bit(w: &[u64], i: usize) -> bool {
    (w[i / 64] >> (i % 64)) & 1 == 1
}

/// The 128 bools at `[base, base+128)` as one `F128` (bit `t` → `lo`/`hi`).
#[inline]
fn pack_word(bits: &[bool], base: usize) -> F128 {
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

fn read_digest_words(w: &[u64], base: usize) -> [u32; SLOT_WORDS] {
    let mut d = [0u32; SLOT_WORDS];
    for j in 0..SLOT_BITS {
        if word_bit(w, base + j) {
            d[j / 32] |= 1u32 << (j % 32);
        }
    }
    d
}

// ---------------------------------------------------------------------------
// The walker LincheckCircuit
// ---------------------------------------------------------------------------

/// A CSC-transposed matrix pair: column `c`'s nonzero ROWS are
/// `rows[col_ptr[c] as usize .. col_ptr[c+1] as usize]`. Same shape as
/// `flock_core::lincheck::CscCircuit`'s internals, kept local because the
/// walker needs to gather from two of them at different row offsets.
struct Csc {
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,
}

impl Csc {
    fn pair(a: &SparseBinaryMatrix, b: &SparseBinaryMatrix) -> Self {
        let (a_col_ptr, a_rows) = csc_from_rows(a);
        let (b_col_ptr, b_rows) = csc_from_rows(b);
        Self {
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
        }
    }

    /// `(Σ_{r ∈ colA(c)} eq[row_off + r], Σ_{r ∈ colB(c)} eq[row_off + r])`.
    #[inline]
    fn gather(&self, c: usize, eq: &[F128], row_off: usize) -> (F128, F128) {
        let mut sa = F128::ZERO;
        for &r in &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize] {
            sa += eq[row_off + r as usize];
        }
        let mut sb = F128::ZERO;
        for &r in &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize] {
            sb += eq[row_off + r as usize];
        }
        (sa, sb)
    }

    fn nnz(&self) -> (usize, usize) {
        (self.a_rows.len(), self.b_rows.len())
    }
}

fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_rows <= u32::MAX as usize);
    assert!(m.num_cols <= u32::MAX as usize);
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

/// [`LincheckCircuit`] for the composite Merkle block that never
/// materializes the `depth` per-level matrix copies.
///
/// The composite's rows split cleanly in two:
///
/// * **Base rows** — level `l`'s hash relation is the base block's matrix
///   with every row AND column index shifted by the same per-level offset.
///   A column marginal over these is therefore the base matrix's own column
///   marginal evaluated against an `eq` slice starting at that offset, so
///   one CSC transpose of the base block serves all `depth` levels.
/// * **Extras** — the globals, the swap-gadget columns, and the overridden
///   rows inside each block. These are the only rows referencing columns
///   outside their own level, and there are ~3.6K nonzeros per level, so
///   they are transposed once over the full composite.
///
/// At depth 26 over BLAKE3 this is ~88 MB resident instead of ~4.4 GB (plus
/// a second 4.4 GB for the transpose). It performs the *same* ~547M gather
/// operations — the saving is storage, not arithmetic. That is an acceptable
/// trade because the matrices are off the prove hot path entirely: the
/// batch-major witness drivers emit `a` and `b` during hashing, and the
/// zerocheck runs on the packed witness, so `fold_alpha_batched` is called
/// once per proof.
pub struct MerkleWalkerCircuit {
    n_cols: usize,
    depth: usize,
    levels_base: usize,
    level_stride: usize,
    base_useful: usize,
    base: Csc,
    extras: Csc,
    const_pin: Option<usize>,
}

impl std::fmt::Debug for MerkleWalkerCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ba, bb) = self.base.nnz();
        let (xa, xb) = self.extras.nnz();
        f.debug_struct("MerkleWalkerCircuit")
            .field("n_cols", &self.n_cols)
            .field("depth", &self.depth)
            .field("base_nnz", &(ba + bb))
            .field("extras_nnz", &(xa + xb))
            .field("effective_nnz", &(self.depth * (ba + bb) + xa + xb))
            .finish()
    }
}

impl MerkleWalkerCircuit {
    /// Resident bytes of the two CSC transposes — the whole point of the
    /// walker. Compare `depth · base_nnz · 8` for the materialized form.
    pub fn resident_bytes(&self) -> usize {
        let sz = |c: &Csc| {
            4 * (c.a_col_ptr.len() + c.a_rows.len() + c.b_col_ptr.len() + c.b_rows.len())
        };
        sz(&self.base) + sz(&self.extras)
    }

    /// Nonzeros the walker traverses per `fold_alpha_batched` — equal to the
    /// materialized matrices' nonzero count.
    pub fn effective_nnz(&self) -> usize {
        let (ba, bb) = self.base.nnz();
        let (xa, xb) = self.extras.nnz();
        self.depth * (ba + bb) + xa + xb
    }

    /// Decode a composite column into `(level, base column)` if it lands
    /// inside a level's embedded hash block. Globals, gadget columns and
    /// padding return `None` — those draw only on `extras`.
    #[inline]
    fn decode(&self, c: usize) -> Option<(usize, usize)> {
        if c < self.levels_base {
            return None;
        }
        let off = c - self.levels_base;
        let level = off / self.level_stride;
        if level >= self.depth {
            return None; // padding past the last level
        }
        let within = off % self.level_stride;
        // [0, 256) sibling, [256, 512) t, [512, 512+useful) the hash block.
        let c_base = within.checked_sub(2 * SLOT_BITS)?;
        debug_assert!(c_base < self.base_useful);
        Some((level, c_base))
    }

    /// First composite column of level `l`'s hash block — also the row
    /// offset the base gather uses, since the embedding shifts rows and
    /// columns alike.
    #[inline]
    fn hash_base(&self, level: usize) -> usize {
        self.levels_base + level * self.level_stride + 2 * SLOT_BITS
    }
}

impl flock_core::lincheck::LincheckCircuit for MerkleWalkerCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), self.n_cols);
        let one_col = |c: usize| -> F128 {
            let (mut sa, mut sb) = self.extras.gather(c, eq_inner, 0);
            if let Some((level, c_base)) = self.decode(c) {
                let (ba, bb) = self.base.gather(c_base, eq_inner, self.hash_base(level));
                sa += ba;
                sb += bb;
            }
            alpha * sa + sb
        };
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(c, slot)| *slot = one_col(c));
        out
    }
}

/// One Merkle path: the leaf digest, its index, and one sibling per level
/// (level 0 = closest to the leaf).
#[derive(Clone, Debug)]
pub struct PathInput {
    pub leaf: [u32; SLOT_WORDS],
    pub index: u64,
    pub siblings: Vec<[u32; SLOT_WORDS]>,
}

// ---------------------------------------------------------------------------
// Digest bit helpers — word `j/32`, bit `j%32`, matching the encoders'
// `write_word` (LSB-first within a 32-bit word).
// ---------------------------------------------------------------------------

#[inline]
fn digest_bit(d: &[u32; SLOT_WORDS], j: usize) -> bool {
    (d[j / 32] >> (j % 32)) & 1 == 1
}

fn write_digest(z: &mut [bool], base: usize, d: &[u32; SLOT_WORDS]) {
    for j in 0..SLOT_BITS {
        z[base + j] = digest_bit(d, j);
    }
}

fn read_digest(z: &[bool], base: usize) -> [u32; SLOT_WORDS] {
    let mut d = [0u32; SLOT_WORDS];
    for j in 0..SLOT_BITS {
        if z[base + j] {
            d[j / 32] |= 1u32 << (j % 32);
        }
    }
    d
}

/// Reference Merkle root: fold the path natively with the same node
/// compression the R1CS encodes. Mirrors [`MerkleTreeLayout::build_witness`]
/// without touching the witness, so a test can cross-check the two.
pub fn reference_root(
    spec: &HashSpec,
    input: &PathInput,
) -> [u32; SLOT_WORDS] {
    let mut prev = input.leaf;
    for (l, sib) in input.siblings.iter().enumerate() {
        let bit = (input.index >> l) & 1 == 1;
        let (left, right) = if bit { (prev, *sib) } else { (*sib, prev) };
        let block = (spec.node_witness)(&left, &right);
        prev = read_digest(&block, spec.out_cv_base);
    }
    prev
}
