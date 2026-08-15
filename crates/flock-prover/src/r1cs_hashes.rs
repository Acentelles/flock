//! Monolithic per-block R1CS encoders for cryptographic hashes (BLAKE3,
//! SHA-2, Keccak-f[1600]). Each submodule packages: per-instance witness
//! layout, sparse `(A_0, B_0)` matrix construction (`C_0 = I`), `prove_fast`
//! helpers (the c-aliased fast path), and a `*Setup` convenience type
//! wrapping R1CS + PCS params.
//!
//! Submodules share low-level bit-packing / matrix-row utilities via
//! [`common`].

pub mod blake3;
/// Generic hash-chain glue ([`ChainLayout`]-parameterized prove/verify) shared
/// by the per-hash `*_chain` modules.
///
/// [`ChainLayout`]: chain_common::ChainLayout
pub mod common;
/// The Fiat–Shamir chain: BLAKE3 over a transcript with a finalize forked at
/// every squeeze — the FS chain's witness generator, over [`blake3`]'s rows.
pub mod fs_chain;
pub mod keccak;
/// 3-wide Keccak-f[1600] R1CS (3 independent permutations per K_LOG=17 block)
/// for tighter PCS utilization (~97% vs the single encoder's ~65%).
pub mod keccak3;
/// Generic Merkle-path glue ([`MerkleLayout`]-parameterized prove/verify),
/// analogous to [`chain_common`] but with a per-row bit selector.
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
