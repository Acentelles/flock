//! Large-field (element-level) R1CS: a table class whose witness entries are
//! **F128 elements** — one field element per variable, one committed word per
//! variable — instead of GF(2) bits.
//!
//! The point of the class is arithmetic density. In the bit-level tables a
//! single F128 multiplication costs ~2187 constraints; here it costs one. The
//! price is that every variable occupies a full 128-bit word, so element tables
//! pay for the wires they use rather than for the bits they touch (see
//! `docs/local/recursion-verifier-map.md` §4.2).
//!
//! ## The relation
//!
//! A table type is a **base block** over F128 = GF(2^128) (GHASH modulus,
//! char 2 — so subtraction *is* addition). With `k` witness columns per row
//! padded to `2^kappa`, sparse matrices `A_0, B_0 ∈ F128^{2^kappa × 2^kappa}`
//! and affine constant vectors `a_const, b_const ∈ F128^{2^kappa}` (part of the
//! statement, not the witness), every row `j` and column `y` must satisfy
//!
//! ```text
//! (A_0[y]·z_j + a_const[y]) · (B_0[y]·z_j + b_const[y]) = z_j[y]
//! ```
//!
//! **C is the identity**: the constraint domain *is* the column domain, one
//! constraint per column. That is what lets the zerocheck's C-claim go straight
//! out as a witness evaluation claim with no lincheck term.
//!
//! The type constructor enforces `a_const[y] · b_const[y] = 0` for every `y`
//! (disjoint supports). That is what makes an all-zero row satisfying —
//! `(0 + a_const)(0 + b_const) = 0 = z_y` — so dummy/padding rows are
//! definitionally satisfying, zero-contributing, and consistent with the jagged
//! "dropped words are zero" convention.
//!
//! ## Witness layout
//!
//! BatchMajor at **word** level with rows in the LOW bits: the committed word
//! index of (column `c`, row `j`) is `(c << n_log) + j`. There is no in-word
//! packing structure — the element index *is* the packed-word index — so the
//! committed polynomial has `m_words = kappa + n_log` variables. The rows-low
//! convention is load-bearing for the future wiring layer.
//!
//! Because the full system is `I_{2^n_log} ⊗ A_0` (block diagonal per row), the
//! MLEs factor as `M̂((x_row,x_con),(y_row,y_col)) = eq(x_row,y_row)·M̂_0(x_con,y_col)`,
//! and the constant vectors — uniform across rows — collapse by partition of
//! unity: `â_const(r_row, r_con) = â_const_base(r_con)`, with no row and no
//! count dependence.
//!
//! ## Protocol
//!
//! Spartan-style, all in the large field:
//!
//! 1. [`zerocheck`] — a plain eq-weighted degree-3 sumcheck over
//!    `n_log + kappa` variables proving
//!    `Σ_x eq(τ,x)·((Az+a_const)(x)·(Bz+b_const)(x) + z(x)) = 0`. No univariate
//!    skip, no packing, no φ8. Outputs `ea`, `eb` (which Phase 2 reduces) and
//!    `ec = ẑ(r)` (a witness claim already).
//! 2. [`lincheck`] — one degree-2 sumcheck batching `ea`/`eb` into a single
//!    witness claim `ẑ(r')`.
//! 3. [`prove`] / [`verify`] — commit, bind the statement, run both phases, and
//!    open `ec` and `ẑ(r')` as **packed-direct** claims. No ring-switching
//!    anywhere: the witness words already are field elements, so there is no
//!    bit-MLE ↔ packed-MLE bridge to cross.
//!
//! Fiat–Shamir order: commit → bind statement → τ → zerocheck rounds → α →
//! lincheck rounds → γ-batched opening.

pub mod lincheck;
pub mod zerocheck;

use std::sync::OnceLock;

use crate::challenger::Challenger;
use crate::field::F128;
use crate::merkle::Hash;
use crate::pcs::ligerito::{ProverConfig, VerifierConfig};
use crate::pcs::{
    self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
    commit,
};
use crate::zerocheck::PaddingSpec;
use serde::{Deserialize, Serialize};

/// Statement-binding domain label. Absorbed before any challenge is squeezed.
const DOMAIN: &[u8] = b"flock-element-r1cs-v0";

// ---------------------------------------------------------------------------
// Sparse F128 matrix
// ---------------------------------------------------------------------------

/// Sparse matrix over F128. `rows[i]` lists the `(col, coeff)` entries of row
/// `i`; coefficients are non-zero and column indices within a row are distinct
/// (both enforced by [`SparseF128Matrix::from_rows`], so the canonical form is
/// unique up to the order of a row's entries).
///
/// Row `i` is read as the linear form `M[i]·v = Σ_(c,w) w·v[c]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseF128Matrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub rows: Vec<Vec<(usize, F128)>>,
}

impl SparseF128Matrix {
    /// The all-zero `num_rows × num_cols` matrix.
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            rows: vec![Vec::new(); num_rows],
        }
    }

    /// Validating constructor. `which` names the matrix in any error.
    pub fn from_rows(
        which: &'static str,
        num_cols: usize,
        rows: Vec<Vec<(usize, F128)>>,
    ) -> Result<Self, TypeError> {
        let m = Self {
            num_rows: rows.len(),
            num_cols,
            rows,
        };
        m.validate(which)?;
        Ok(m)
    }

    fn validate(&self, which: &'static str) -> Result<(), TypeError> {
        for (row, entries) in self.rows.iter().enumerate() {
            for (i, &(col, coeff)) in entries.iter().enumerate() {
                if col >= self.num_cols {
                    return Err(TypeError::ColumnOutOfRange { which, row, col });
                }
                if coeff.is_zero() {
                    return Err(TypeError::ZeroCoefficient { which, row, col });
                }
                if entries[..i].iter().any(|&(c, _)| c == col) {
                    return Err(TypeError::DuplicateColumn { which, row, col });
                }
            }
        }
        Ok(())
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// `M[row] · v` for a length-`num_cols` slice.
    pub fn row_dot(&self, row: usize, v: &[F128]) -> F128 {
        debug_assert_eq!(v.len(), self.num_cols);
        let mut acc = F128::ZERO;
        for &(col, coeff) in &self.rows[row] {
            acc += coeff * v[col];
        }
        acc
    }

    /// Absorb the matrix into a statement digest, length-prefixed per row so no
    /// two distinct matrices share an encoding. Mirrors
    /// `crate::r1cs::absorb_matrix` with F128 coefficients appended.
    fn absorb(&self, h: &mut blake3::Hasher) {
        h.update(&(self.num_rows as u64).to_le_bytes());
        h.update(&(self.num_cols as u64).to_le_bytes());
        for row in &self.rows {
            h.update(&(row.len() as u64).to_le_bytes());
            for &(col, coeff) in row {
                h.update(&(col as u64).to_le_bytes());
                h.update(&coeff.lo.to_le_bytes());
                h.update(&coeff.hi.to_le_bytes());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Table type
// ---------------------------------------------------------------------------

/// Why an [`ElementTableType`] could not be constructed. Every variant is a
/// *statement* defect — caught once at construction, never at proving time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeError {
    /// A matrix is not `2^kappa × 2^kappa`.
    MatrixShape {
        which: &'static str,
        num_rows: usize,
        num_cols: usize,
        expected: usize,
    },
    /// A constant vector is not length `2^kappa`.
    ConstLen {
        which: &'static str,
        got: usize,
        expected: usize,
    },
    /// `k` real columns exceeds the padded width `2^kappa`.
    TooManyColumns { k: usize, kappa: usize },
    /// A matrix row references a column outside `[0, 2^kappa)`.
    ColumnOutOfRange {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// A matrix row lists the same column twice.
    DuplicateColumn {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// An explicitly-stored zero coefficient — the canonical sparse form keeps
    /// non-zeros only, so this would make the type digest ambiguous.
    ZeroCoefficient {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// The validity rule `a_const[y] · b_const[y] = 0` is violated at `y`.
    /// Without disjoint supports an all-zero row does NOT satisfy the relation,
    /// so dummy/padding rows would be unsatisfiable and the jagged
    /// "dropped words are zero" convention would break.
    ConstantsOverlap { y: usize },
    /// A padding column `y ≥ k` does not carry the all-zero constraint row
    /// (`A_0[y] = B_0[y] = 0`, `a_const[y] = b_const[y] = 0`) that pins
    /// `z_y = 0`. Self-enforcing zero padding is the declared convention for
    /// the columns past the `k` real ones.
    PaddingRowNotZero { y: usize },
}

/// A large-field table type: the base block of the relation in the module docs,
/// plus the padded width and the count `k` of real columns.
///
/// Fields are private so the construction-time invariants (shape, disjoint
/// constant supports, zero padding rows) hold for the whole lifetime of the
/// value — the prover and verifier both rely on them.
#[derive(Debug)]
pub struct ElementTableType {
    kappa: usize,
    k: usize,
    a_0: SparseF128Matrix,
    b_0: SparseF128Matrix,
    a_const: Vec<F128>,
    b_const: Vec<F128>,
    digest_cache: OnceLock<[u8; 32]>,
}

impl ElementTableType {
    /// Validating constructor. `k` is the number of real columns; columns
    /// `[k, 2^kappa)` must carry all-zero rows (self-enforcing zero padding).
    pub fn new(
        kappa: usize,
        k: usize,
        a_0: SparseF128Matrix,
        b_0: SparseF128Matrix,
        a_const: Vec<F128>,
        b_const: Vec<F128>,
    ) -> Result<Self, TypeError> {
        let width = 1usize << kappa;
        for (which, m) in [("a_0", &a_0), ("b_0", &b_0)] {
            if m.num_rows != width || m.num_cols != width {
                return Err(TypeError::MatrixShape {
                    which,
                    num_rows: m.num_rows,
                    num_cols: m.num_cols,
                    expected: width,
                });
            }
            m.validate(which)?;
        }
        for (which, v) in [("a_const", &a_const), ("b_const", &b_const)] {
            if v.len() != width {
                return Err(TypeError::ConstLen {
                    which,
                    got: v.len(),
                    expected: width,
                });
            }
        }
        if k > width {
            return Err(TypeError::TooManyColumns { k, kappa });
        }
        // The validity rule. Checked over the FULL padded width, not just the
        // real columns: the zerocheck sums over every column of every row.
        for y in 0..width {
            if !(a_const[y] * b_const[y]).is_zero() {
                return Err(TypeError::ConstantsOverlap { y });
            }
        }
        for y in k..width {
            if !a_0.rows[y].is_empty()
                || !b_0.rows[y].is_empty()
                || !a_const[y].is_zero()
                || !b_const[y].is_zero()
            {
                return Err(TypeError::PaddingRowNotZero { y });
            }
        }
        Ok(Self {
            kappa,
            k,
            a_0,
            b_0,
            a_const,
            b_const,
            digest_cache: OnceLock::new(),
        })
    }

    /// log2 of the padded column count.
    pub fn kappa(&self) -> usize {
        self.kappa
    }
    /// Padded column count `2^kappa` — the width of one row's witness.
    pub fn width(&self) -> usize {
        1usize << self.kappa
    }
    /// Number of real (non-padding) columns.
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn a_0(&self) -> &SparseF128Matrix {
        &self.a_0
    }
    pub fn b_0(&self) -> &SparseF128Matrix {
        &self.b_0
    }
    /// The affine constant vector the spec calls `a0`.
    pub fn a_const(&self) -> &[F128] {
        &self.a_const
    }
    /// The affine constant vector the spec calls `b0`.
    pub fn b_const(&self) -> &[F128] {
        &self.b_const
    }

    /// Statement digest over the whole base block.
    ///
    /// Absorbs, in order: the domain tag `b"flock-element-type-v0"` (distinct
    /// from the bit-level `b"flock-registry-v1"` / `b"flock-r1cs-stmt-v1"`, so
    /// an element digest can never collide with a boolean one), a format
    /// version byte, `kappa` and `k` as u32 LE, the two matrices via
    /// [`SparseF128Matrix::absorb`], then the two constant vectors
    /// length-prefixed. Lazily cached.
    pub fn digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(b"flock-element-type-v0");
            h.update(&[0u8]);
            h.update(&(self.kappa as u32).to_le_bytes());
            h.update(&(self.k as u32).to_le_bytes());
            self.a_0.absorb(&mut h);
            self.b_0.absorb(&mut h);
            for v in [&self.a_const, &self.b_const] {
                h.update(&(v.len() as u64).to_le_bytes());
                for e in v {
                    h.update(&e.lo.to_le_bytes());
                    h.update(&e.hi.to_le_bytes());
                }
            }
            *h.finalize().as_bytes()
        })
    }

    /// `Az` and `Bz` for a BatchMajor witness (`z[(c << n_log) + j]`), by sparse
    /// gather: one pass per stored matrix entry per row, i.e. `O(nnz · 2^n_log)`
    /// with no matrix application on any hot path. For a mult gate the outputs
    /// are literally the operand values.
    ///
    /// Output layout matches the input: `az[(y << n_log) + j] = A_0[y]·z_j`.
    pub fn apply(&self, z: &[F128], n_log: usize) -> (Vec<F128>, Vec<F128>) {
        let rows = 1usize << n_log;
        assert_eq!(z.len(), self.width() << n_log, "witness length");
        let gather = |m: &SparseF128Matrix| {
            use rayon::prelude::*;
            let mut out = crate::alloc_zeroed_vec::<F128>(z.len());
            out.par_chunks_mut(rows).enumerate().for_each(|(y, dst)| {
                for &(col, coeff) in &m.rows[y] {
                    let src = &z[col << n_log..(col << n_log) + rows];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += coeff * *s;
                    }
                }
            });
            out
        };
        (gather(&self.a_0), gather(&self.b_0))
    }

    /// Brute-force check that every row `j < n` satisfies the relation and that
    /// rows `[n, 2^n_log)` are honestly zero. Reference for the tests, and the
    /// contract [`prove`] assumes of its caller.
    pub fn satisfies(&self, z: &[F128], n_log: usize, n: usize) -> bool {
        let width = self.width();
        let rows = 1usize << n_log;
        if z.len() != width << n_log || n > rows {
            return false;
        }
        let row_of = |j: usize| -> Vec<F128> { (0..width).map(|c| z[(c << n_log) + j]).collect() };
        for j in 0..rows {
            let zj = row_of(j);
            if j >= n && zj.iter().any(|e| !e.is_zero()) {
                return false;
            }
            for y in 0..width {
                let lhs = (self.a_0.row_dot(y, &zj) + self.a_const[y])
                    * (self.b_0.row_dot(y, &zj) + self.b_const[y]);
                if lhs != zj[y] {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Row-encoding builder
// ---------------------------------------------------------------------------

/// Incremental builder for the standard row encodings. Starts from the all-zero
/// block of width `2^kappa` (every column self-pinned to zero) and lets each
/// gate claim its output column.
///
/// `k` — the real-column count — is `1 + max(column touched)`, so the untouched
/// tail keeps its all-zero rows and is therefore *padding* in the
/// [`ElementTableType::new`] sense.
///
/// This is deliberately a handful of test gates, not a gate library.
#[derive(Clone, Debug)]
pub struct ElementTableBuilder {
    kappa: usize,
    a_rows: Vec<Vec<(usize, F128)>>,
    b_rows: Vec<Vec<(usize, F128)>>,
    a_const: Vec<F128>,
    b_const: Vec<F128>,
    k: usize,
}

impl ElementTableBuilder {
    pub fn new(kappa: usize) -> Self {
        let width = 1usize << kappa;
        Self {
            kappa,
            a_rows: vec![Vec::new(); width],
            b_rows: vec![Vec::new(); width],
            a_const: vec![F128::ZERO; width],
            b_const: vec![F128::ZERO; width],
            k: 0,
        }
    }

    fn touch(&mut self, y: usize) {
        assert!(y < 1usize << self.kappa, "column {y} exceeds 2^kappa");
        self.k = self.k.max(y + 1);
        self.a_rows[y].clear();
        self.b_rows[y].clear();
        self.a_const[y] = F128::ZERO;
        self.b_const[y] = F128::ZERO;
    }

    /// Multiplication `z_out = z_a · z_b`: `A_0[out] = e_a`, `B_0[out] = e_b`,
    /// both constants zero.
    pub fn mult(&mut self, out: usize, a: usize, b: usize) -> &mut Self {
        self.touch(out);
        self.a_rows[out] = vec![(a, F128::ONE)];
        self.b_rows[out] = vec![(b, F128::ONE)];
        self
    }

    /// Free wire — an input constrained only by future wiring. The tautology row
    /// `(z_y)(1) = z_y`: `A_0[y] = e_y`, `B_0[y] = 0`, `b_const[y] = 1`.
    pub fn free_wire(&mut self, y: usize) -> &mut Self {
        self.touch(y);
        self.a_rows[y] = vec![(y, F128::ONE)];
        self.b_const[y] = F128::ONE;
        self
    }

    /// Linear constraint pinning a linear combination to a wire:
    /// `(Σ w·z_c)(1) = z_y`. `terms` must have distinct, non-zero-weighted
    /// columns.
    pub fn linear(&mut self, y: usize, terms: &[(usize, F128)]) -> &mut Self {
        self.touch(y);
        self.a_rows[y] = terms.to_vec();
        self.b_const[y] = F128::ONE;
        self
    }

    /// Multiply-accumulate `z_out = z_a·z_b + z_addend`, spelled as the two rows
    /// the relation shape allows: a [`Self::mult`] into `tmp`, then a
    /// [`Self::linear`] summing `tmp` and `addend` into `out`. (One row cannot
    /// do it: the right-hand side of a row is exactly one column.)
    pub fn mult_acc(
        &mut self,
        out: usize,
        a: usize,
        b: usize,
        addend: usize,
        tmp: usize,
    ) -> &mut Self {
        self.mult(tmp, a, b);
        self.linear(out, &[(tmp, F128::ONE), (addend, F128::ONE)])
    }

    /// Finish, validating every invariant of [`ElementTableType::new`].
    pub fn build(self) -> Result<ElementTableType, TypeError> {
        let width = 1usize << self.kappa;
        ElementTableType::new(
            self.kappa,
            self.k,
            SparseF128Matrix::from_rows("a_0", width, self.a_rows)?,
            SparseF128Matrix::from_rows("b_0", width, self.b_rows)?,
            self.a_const,
            self.b_const,
        )
    }
}

// ---------------------------------------------------------------------------
// Statement
// ---------------------------------------------------------------------------

/// The public statement of one element table: the type, the row capacity
/// `2^n_log`, and the declared count `n ≤ 2^n_log` of real rows.
#[derive(Clone, Copy, Debug)]
pub struct ElementStatement<'a> {
    pub ty: &'a ElementTableType,
    pub n_log: usize,
    pub n: usize,
}

impl ElementStatement<'_> {
    /// Committed word-variable count `m_words = kappa + n_log`.
    pub fn m_words(&self) -> usize {
        self.ty.kappa() + self.n_log
    }

    /// Total committed words `2^m_words`.
    pub fn n_words(&self) -> usize {
        1usize << self.m_words()
    }

    /// Absorb label, type digest, capacity, count and commitment root — the
    /// whole statement — BEFORE any challenge is squeezed. Prover and verifier
    /// call this at the same transcript position.
    fn bind<C: Challenger>(&self, root: &Hash, ch: &mut C) {
        ch.observe_label(DOMAIN);
        ch.observe_bytes(&self.ty.digest());
        ch.observe_bytes(&(self.n_log as u64).to_le_bytes());
        ch.observe_bytes(&(self.n as u64).to_le_bytes());
        ch.observe_bytes(root);
    }
}

// ---------------------------------------------------------------------------
// Commitment parameters
// ---------------------------------------------------------------------------

/// RS inverse rate (log2) and interleaving batch size (log2) for the element
/// witness commitment. Both backends' L0 commit and Ligerito's `default_config`
/// must agree on these, so they live in one place. Same choice as
/// [`crate::permutation`].
const PCS_LOG_INV_RATE: usize = 1;
const PCS_LOG_BATCH_SIZE: usize = 1;

/// PCS parameters for a witness of `m_words` F128 words.
///
/// `PcsParams::m` is the **bit**-level variable count, so it is
/// `m_words + LOG_PACKING`; for an element table that offset is pure
/// bookkeeping (there is no real in-word packing — the element index *is* the
/// packed-word index), and it makes `log_msg_len() == m_words` so the
/// packed-direct opening points have length `m_words`. Deterministic in
/// `m_words`, so the verifier rebuilds these from the statement and the proof
/// carries only the root.
fn pcs_params(m_words: usize) -> PcsParams {
    PcsParams {
        m: m_words + LOG_PACKING,
        log_inv_rate: PCS_LOG_INV_RATE,
        log_batch_size: PCS_LOG_BATCH_SIZE,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    }
}

/// Smallest `m_words` Ligerito's recursion can open at the parameters above
/// (the L0 block must hold `udr_queries(1) = 243` distinct queries). Below it
/// [`prove`] cannot run; the PIOP phases themselves have no such floor.
pub const MIN_M_WORDS: usize = 8;

fn ligerito_prover_config(m_words: usize) -> ProverConfig {
    pcs::ligerito::default_config(m_words, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito config for the element witness; requires m_words >= MIN_M_WORDS")
}

fn ligerito_verifier_config(m_words: usize) -> VerifierConfig {
    pcs::ligerito::default_verifier_config(m_words, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito verifier config for the element witness; requires m_words >= MIN_M_WORDS")
}

// ---------------------------------------------------------------------------
// End-to-end proof
// ---------------------------------------------------------------------------

/// A standalone single-table element proof: the witness commitment root, the two
/// PIOP phases, and one batched opening of both output claims.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementProof {
    /// Merkle root of the committed witness words.
    pub root: Hash,
    pub zerocheck: zerocheck::Proof,
    pub lincheck: lincheck::Proof,
    /// Packed-direct opening of `ec = ẑ(r)` and `ẑ(r')` — the mixed open with
    /// zero ring-switched claims.
    pub open: pcs::BatchOpeningProofLigerito,
}

/// Why an element proof was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `m_words` is below Ligerito's feasibility floor ([`MIN_M_WORDS`]).
    TooSmall {
        m_words: usize,
        min: usize,
    },
    /// The declared count exceeds the row capacity.
    CountExceedsCapacity {
        n: usize,
        n_log: usize,
    },
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    /// The packed-direct opening rejected.
    Open(pcs::VerifyError),
}

/// The claims a verified element proof leaves behind: the two witness
/// evaluation points and values, both already discharged by the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementClaim {
    /// Zerocheck point `r = (r_row, r_con)`, LSB-first (rows low).
    pub r: Vec<F128>,
    /// `ẑ(r)` — the C-claim, direct because C is the identity.
    pub ec: F128,
    /// Lincheck point `r'`, LSB-first.
    pub r_prime: Vec<F128>,
    /// `ẑ(r')`.
    pub z_eval: F128,
}

/// Prove that `z` satisfies `stmt`.
///
/// `z` is the BatchMajor witness (`z[(c << n_log) + j]`, length `2^m_words`)
/// with rows `[n, 2^n_log)` all zero. Committed at full height — dummy rows are
/// zero, so this is honest; count-proportional/jagged heights are the union
/// integration's job.
pub fn prove<C: Challenger>(
    stmt: &ElementStatement<'_>,
    z: &[F128],
    ch: &mut C,
) -> (ElementProof, ElementClaim) {
    let m_words = stmt.m_words();
    assert!(
        m_words >= MIN_M_WORDS,
        "element prove needs m_words >= {MIN_M_WORDS} (got {m_words})"
    );
    assert!(stmt.n <= 1usize << stmt.n_log, "count exceeds capacity");
    assert_eq!(z.len(), stmt.n_words(), "witness length");

    // ---- 1. Commit the witness words, then bind the whole statement. ----
    let params = pcs_params(m_words);
    let (commitment, pdata) = commit(z, &params);
    stmt.bind(&commitment.root, ch);

    // ---- 2. Phase 1: element zerocheck. ----
    //
    // `Az`/`Bz` by sparse gather, then the affine constants folded in — the
    // zerocheck works directly on `(Az + a_const)` and `(Bz + b_const)`.
    let (mut pa, mut pb) = stmt.ty.apply(z, stmt.n_log);
    broadcast_add(&mut pa, stmt.ty.a_const(), stmt.n_log);
    broadcast_add(&mut pb, stmt.ty.b_const(), stmt.n_log);
    let (zc_proof, zc_claim) = zerocheck::prove(pa, pb, z, stmt.n_log, stmt.ty.kappa(), ch);

    // ---- 3. Phase 2: batched lincheck. ----
    //
    // The verifier's own correction: `ea`/`eb` are claims on `Az + a_const`, and
    // the constants' MLEs collapse to the base-block evaluation at `r_con` with
    // no row dependence, so subtracting them (char 2: adding) leaves the pure
    // `Âz(r)` / `B̂z(r)` claims the lincheck reduces.
    let (va, vb) = strip_constants(stmt.ty, &zc_claim);
    let (lc_proof, lc_claim) = lincheck::prove(stmt.ty, z, stmt.n_log, &zc_claim.r, va, vb, ch);

    // ---- 4. Open both witness claims, packed-direct, no ring-switch. ----
    let claims = packed_direct_claims(&zc_claim.r, zc_claim.ec, &lc_claim.r_prime, lc_claim.z_eval);
    let open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z.to_vec(),
        &pdata,
        &commitment,
        &[],
        &[],
        &claims,
        &PaddingSpec::dense(params.m),
        &ligerito_prover_config(m_words),
        ch,
    );

    let proof = ElementProof {
        root: commitment.root,
        zerocheck: zc_proof,
        lincheck: lc_proof,
        open,
    };
    let claim = ElementClaim {
        r: zc_claim.r,
        ec: zc_claim.ec,
        r_prime: lc_claim.r_prime,
        z_eval: lc_claim.z_eval,
    };
    (proof, claim)
}

/// Verify an element proof against `stmt`. Walks the challenger in lockstep with
/// [`prove`].
pub fn verify<C: Challenger>(
    stmt: &ElementStatement<'_>,
    proof: &ElementProof,
    ch: &mut C,
) -> Result<ElementClaim, VerifyError> {
    let m_words = stmt.m_words();
    if m_words < MIN_M_WORDS {
        return Err(VerifyError::TooSmall {
            m_words,
            min: MIN_M_WORDS,
        });
    }
    if stmt.n > 1usize << stmt.n_log {
        return Err(VerifyError::CountExceedsCapacity {
            n: stmt.n,
            n_log: stmt.n_log,
        });
    }

    // Rebuild the commitment from the proof's root + statement-derived params,
    // and bind at the prover's transcript position.
    let commitment = Commitment {
        root: proof.root,
        params: pcs_params(m_words),
    };
    stmt.bind(&commitment.root, ch);

    let zc_claim = zerocheck::verify(stmt.n_log, stmt.ty.kappa(), &proof.zerocheck, ch)
        .map_err(VerifyError::Zerocheck)?;
    let (va, vb) = strip_constants(stmt.ty, &zc_claim);
    let lc_claim = lincheck::verify(
        stmt.ty,
        stmt.n_log,
        &zc_claim.r,
        va,
        vb,
        &proof.lincheck,
        ch,
    )
    .map_err(VerifyError::Lincheck)?;

    let points = [zc_claim.r.as_slice(), lc_claim.r_prime.as_slice()];
    let values = [zc_claim.ec, lc_claim.z_eval];
    let refs: Vec<PackedDirectClaimRef<'_>> = points
        .iter()
        .zip(values)
        .map(|(point, value)| PackedDirectClaimRef { point, value })
        .collect();
    pcs::verify_opening_batch_ligerito_mixed(
        &commitment,
        &[],
        &[],
        &[],
        &refs,
        &proof.open,
        &ligerito_verifier_config(m_words),
        ch,
    )
    .map_err(VerifyError::Open)?;

    Ok(ElementClaim {
        r: zc_claim.r,
        ec: zc_claim.ec,
        r_prime: lc_claim.r_prime,
        z_eval: lc_claim.z_eval,
    })
}

/// The two packed-direct claims, in the fixed order `[ec at r, ẑ(r')]`. Both
/// points are fully random (the challenger never hands out an exactly-zero
/// coordinate in practice), so the dense `eq_ind` is the right representation.
fn packed_direct_claims(
    r: &[F128],
    ec: F128,
    r_prime: &[F128],
    z_eval: F128,
) -> Vec<PackedDirectClaim> {
    use crate::pcs::ring_switch::build_eq_parallel;
    [(r, ec), (r_prime, z_eval)]
        .into_iter()
        .map(|(point, value)| PackedDirectClaim {
            point: point.to_vec(),
            value,
            eq_ind: DirectEqInd::Dense(build_eq_parallel(point)),
        })
        .collect()
}

/// `v[(y << n_log) + j] += c[y]` — broadcast the row-uniform constant vector
/// across every row.
fn broadcast_add(v: &mut [F128], c: &[F128], n_log: usize) {
    use rayon::prelude::*;
    let rows = 1usize << n_log;
    debug_assert_eq!(v.len(), c.len() << n_log);
    v.par_chunks_mut(rows)
        .zip(c.par_iter())
        .for_each(|(dst, &cy)| {
            if !cy.is_zero() {
                for d in dst {
                    *d += cy;
                }
            }
        });
}

/// Turn the zerocheck's `(ea, eb)` — claims on `(Az + a_const)`, `(Bz + b_const)`
/// at `r` — into the pure `Âz(r)`, `B̂z(r)` claims the lincheck reduces, by
/// subtracting the constants' closed-form MLEs.
///
/// `x ↦ a_const[x_con]` is uniform in `x_row`, so its MLE is
/// `â_const_base(r_con)` — no row dependence, no count dependence (partition of
/// unity over the row block). `O(2^kappa)` for the verifier.
fn strip_constants(ty: &ElementTableType, zc: &zerocheck::Claim) -> (F128, F128) {
    let r_con = &zc.r[zc.r.len() - ty.kappa()..];
    let eq_con = crate::zerocheck::univariate_skip::build_eq(r_con);
    let dot = |c: &[F128]| -> F128 {
        eq_con
            .iter()
            .zip(c)
            .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
    };
    (zc.ea + dot(ty.a_const()), zc.eb + dot(ty.b_const()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// SplitMix64 PRNG, matching the repo's test RNG convention.
    pub(crate) struct Rng(u64);
    impl Rng {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed)
        }
        pub(crate) fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub(crate) fn f128(&mut self) -> F128 {
            F128::new(self.next_u64(), self.next_u64())
        }
        /// Non-zero field element (so it is a legal sparse coefficient).
        pub(crate) fn nonzero(&mut self) -> F128 {
            loop {
                let v = self.f128();
                if !v.is_zero() {
                    return v;
                }
            }
        }
    }

    /// The canonical test gate: `kappa = 2`, columns `0,1` free wires (the
    /// operands), column `2` their product, column `3` padding.
    pub(crate) fn mult_gate(kappa: usize) -> ElementTableType {
        let mut b = ElementTableBuilder::new(kappa);
        b.free_wire(0).free_wire(1).mult(2, 0, 1);
        b.build().expect("mult gate is valid")
    }

    /// A mixed table exercising every row encoding: free wires, a mult, a
    /// linear pin, a mult-acc (which itself is mult + linear), and padding.
    pub(crate) fn mixed_gate(rng: &mut Rng) -> ElementTableType {
        // kappa = 3 → width 8. Columns: 0,1,2 free; 3 = z0·z1; 4 = w3·z0 + z2
        // via tmp column 5; 6 = a·z0 + b·z1 (linear); 7 padding.
        let mut b = ElementTableBuilder::new(3);
        let (wa, wb) = (rng.nonzero(), rng.nonzero());
        b.free_wire(0)
            .free_wire(1)
            .free_wire(2)
            .mult(3, 0, 1)
            .mult_acc(4, 3, 0, 2, 5)
            .linear(6, &[(0, wa), (1, wb)]);
        b.build().expect("mixed gate is valid")
    }

    /// Fill a satisfying witness for [`mult_gate`]: `n` real rows with random
    /// operands, the rest zero.
    pub(crate) fn mult_witness(
        ty: &ElementTableType,
        n_log: usize,
        n: usize,
        rng: &mut Rng,
    ) -> Vec<F128> {
        let rows = 1usize << n_log;
        let mut z = vec![F128::ZERO; ty.width() << n_log];
        for j in 0..n {
            let (a, b) = (rng.f128(), rng.f128());
            z[j] = a;
            z[rows + j] = b;
            z[2 * rows + j] = a * b;
        }
        z
    }

    /// Fill a satisfying witness for [`mixed_gate`].
    pub(crate) fn mixed_witness(
        ty: &ElementTableType,
        n_log: usize,
        n: usize,
        rng: &mut Rng,
    ) -> Vec<F128> {
        let at = |c: usize, j: usize| (c << n_log) + j;
        let mut z = vec![F128::ZERO; ty.width() << n_log];
        let (wa, wb) = (ty.a_0().rows[6][0].1, ty.a_0().rows[6][1].1);
        for j in 0..n {
            let (z0, z1, z2) = (rng.f128(), rng.f128(), rng.f128());
            z[at(0, j)] = z0;
            z[at(1, j)] = z1;
            z[at(2, j)] = z2;
            z[at(3, j)] = z0 * z1;
            z[at(5, j)] = z[at(3, j)] * z0;
            z[at(4, j)] = z[at(5, j)] + z2;
            z[at(6, j)] = wa * z0 + wb * z1;
        }
        z
    }

    // ---- type construction -------------------------------------------------

    #[test]
    fn builder_rows_are_satisfiable() {
        let mult = mult_gate(2);
        assert!(
            mult.satisfies(&mult_witness(&mult, 4, 9, &mut Rng::new(2)), 4, 9),
            "mult gate"
        );
        let mixed = mixed_gate(&mut Rng::new(1));
        assert!(
            mixed.satisfies(&mixed_witness(&mixed, 4, 9, &mut Rng::new(3)), 4, 9),
            "mixed gate (free wires, mult, mult-acc, linear, padding)"
        );
    }

    #[test]
    fn free_wire_is_a_tautology() {
        // A free wire constrains nothing: any value satisfies it.
        let ty = mult_gate(2);
        let mut rng = Rng::new(11);
        let mut z = mult_witness(&ty, 4, 16, &mut rng);
        // Perturbing an operand AND its product stays satisfying.
        let (a, b) = (rng.f128(), rng.f128());
        z[0] = a;
        z[16] = b;
        z[32] = a * b;
        assert!(ty.satisfies(&z, 4, 16));
    }

    #[test]
    fn padding_columns_are_pinned_to_zero() {
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, 4, 16, &mut Rng::new(12));
        // Column 3 is padding: its row is all-zero, forcing z_3 = 0.
        z[3 * 16] = F128::ONE;
        assert!(!ty.satisfies(&z, 4, 16), "padding column must be pinned");
    }

    #[test]
    fn dummy_rows_satisfy_by_the_validity_rule() {
        // Every row past the count is all-zero, and the disjoint-support rule
        // makes that satisfying for EVERY row encoding — free wires included
        // (`(0)(1) = 0`).
        let ty = mult_gate(2);
        for n in [0usize, 1, 7, 16] {
            let z = mult_witness(&ty, 4, n, &mut Rng::new(100 + n as u64));
            assert!(ty.satisfies(&z, 4, n), "n={n}");
        }
    }

    #[test]
    fn overlapping_constants_are_rejected_at_construction() {
        let width = 4usize;
        let a_const = {
            let mut v = vec![F128::ZERO; width];
            v[0] = F128::ONE;
            v
        };
        let b_const = {
            let mut v = vec![F128::ZERO; width];
            v[0] = F128::new(3, 0);
            v
        };
        let err = ElementTableType::new(
            2,
            1,
            SparseF128Matrix::zeros(width, width),
            SparseF128Matrix::zeros(width, width),
            a_const,
            b_const,
        )
        .expect_err("a0 ⊙ b0 ≠ 0 must be rejected");
        assert_eq!(err, TypeError::ConstantsOverlap { y: 0 });
    }

    #[test]
    fn disjoint_constants_are_accepted() {
        // The free-wire encoding has a_const = 0, b_const = 1 — disjoint.
        let width = 4usize;
        let mut b_const = vec![F128::ZERO; width];
        b_const[0] = F128::ONE;
        assert!(
            ElementTableType::new(
                2,
                1,
                SparseF128Matrix::zeros(width, width),
                SparseF128Matrix::zeros(width, width),
                vec![F128::ZERO; width],
                b_const,
            )
            .is_ok()
        );
    }

    #[test]
    fn shape_and_sparsity_errors_are_rejected() {
        let width = 4usize;
        let zeros = || SparseF128Matrix::zeros(width, width);
        let cz = || vec![F128::ZERO; width];

        // Wrong matrix shape.
        assert!(matches!(
            ElementTableType::new(2, 1, SparseF128Matrix::zeros(3, 4), zeros(), cz(), cz()),
            Err(TypeError::MatrixShape { .. })
        ));
        // Wrong constant length.
        assert!(matches!(
            ElementTableType::new(2, 1, zeros(), zeros(), vec![F128::ZERO; 3], cz()),
            Err(TypeError::ConstLen { .. })
        ));
        // k > width.
        assert!(matches!(
            ElementTableType::new(2, 5, zeros(), zeros(), cz(), cz()),
            Err(TypeError::TooManyColumns { .. })
        ));
        // Column out of range.
        let mut m = zeros();
        m.rows[0].push((width, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::ColumnOutOfRange { .. })
        ));
        // Explicit zero coefficient.
        let mut m = zeros();
        m.rows[0].push((0, F128::ZERO));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::ZeroCoefficient { .. })
        ));
        // Duplicate column.
        let mut m = zeros();
        m.rows[0].push((1, F128::ONE));
        m.rows[0].push((1, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::DuplicateColumn { .. })
        ));
        // Padding row must be all-zero.
        let mut m = zeros();
        m.rows[3].push((0, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::PaddingRowNotZero { y: 3 })
        ));
    }

    #[test]
    fn digest_is_deterministic_and_sensitive() {
        let a = mult_gate(2);
        let b = mult_gate(2);
        assert_eq!(a.digest(), b.digest(), "same type, same digest");
        assert_eq!(a.digest(), a.digest(), "cached digest is stable");

        // Any change to the block moves the digest.
        let mut wider = ElementTableBuilder::new(3);
        wider.free_wire(0).free_wire(1).mult(2, 0, 1);
        assert_ne!(a.digest(), wider.build().unwrap().digest(), "kappa");

        let mut swapped = ElementTableBuilder::new(2);
        swapped.free_wire(0).free_wire(1).mult(2, 1, 0);
        // Operand order is a real difference in A_0 vs B_0.
        assert_ne!(a.digest(), swapped.build().unwrap().digest(), "operands");

        let mut scaled = ElementTableBuilder::new(2);
        scaled
            .free_wire(0)
            .free_wire(1)
            .linear(2, &[(0, F128::new(7, 0))]);
        assert_ne!(a.digest(), scaled.build().unwrap().digest(), "coefficients");
    }

    // ---- Az / Bz -----------------------------------------------------------

    #[test]
    fn apply_matches_per_row_matrix_product() {
        let mut rng = Rng::new(77);
        let ty = mixed_gate(&mut rng);
        let n_log = 5usize;
        let rows = 1usize << n_log;
        let z: Vec<F128> = (0..ty.width() << n_log).map(|_| rng.f128()).collect();
        let (az, bz) = ty.apply(&z, n_log);
        for j in 0..rows {
            let zj: Vec<F128> = (0..ty.width()).map(|c| z[(c << n_log) + j]).collect();
            for y in 0..ty.width() {
                assert_eq!(
                    az[(y << n_log) + j],
                    ty.a_0().row_dot(y, &zj),
                    "az y={y} j={j}"
                );
                assert_eq!(
                    bz[(y << n_log) + j],
                    ty.b_0().row_dot(y, &zj),
                    "bz y={y} j={j}"
                );
            }
        }
    }

    #[test]
    fn broadcast_add_is_row_uniform() {
        let n_log = 3usize;
        let c = vec![F128::new(5, 0), F128::ZERO, F128::new(9, 1), F128::ONE];
        let mut v = vec![F128::ZERO; c.len() << n_log];
        broadcast_add(&mut v, &c, n_log);
        for (y, &cy) in c.iter().enumerate() {
            for j in 0..1usize << n_log {
                assert_eq!(v[(y << n_log) + j], cy);
            }
        }
    }
}
