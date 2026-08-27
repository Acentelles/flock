//! Monolithic per-block R1CS encoders for cryptographic hashes (BLAKE3,
//! SHA-2). Each submodule packages: per-instance witness
//! layout, sparse `(A_0, B_0)` matrix construction (`C_0 = I`), `prove_fast`
//! helpers (the c-aliased fast path), and a `*Setup` convenience type
//! wrapping R1CS + PCS params.
//!
//! Submodules share low-level bit-packing / matrix-row utilities via
//! [`common`].

pub mod blake3;
/// Shared low-level bit-packing / R1CS-row utilities (carry-save adders,
/// fused adders, lin-id slot helpers) used by the per-hash encoders.
pub mod common;
/// The Fiat–Shamir chain: BLAKE3 over a transcript with a finalize forked at
/// every squeeze — the FS chain's witness generator, over [`blake3`]'s rows.
pub mod fs_chain;
/// Generic Merkle-path glue ([`MerkleLayout`]-parameterized prove/verify),
/// with a per-row bit selector.
///
/// [`MerkleLayout`]: merkle_path_common::MerkleLayout
pub mod merkle_glue;
pub mod merkle_path_common;
/// Merkle-path verification as ONE monolithic R1CS block per path — the
/// multi-table-legal form (one table row = one whole path), as opposed to
/// [`merkle_path_common`]'s shift-sumcheck composition over a batch of
/// independent per-level compressions.
pub mod merkle_r1cs;
pub mod sha2;
