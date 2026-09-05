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
pub mod chain_common;
pub mod claim_closure;
pub mod common;
/// Exact packed bit-MLE evaluations, shared by the private-salt lanes.
pub mod packed_mle;
pub mod face_closure;
/// The cross-lane word linkage plus the complete HashToPoint driver.
pub mod hash_to_point_link;
/// Record assembly for the aerie private-salt lane: real commitments and
/// openings for the slot proof, scatter, Z_H table, and fingerprint.
pub mod hash_to_point_record;
/// The aerie private-salt stable-compaction scatter argument (prototype);
/// see `AERIE-ADAPTER.md` Section 3.3.
pub mod hash_to_point_scatter;
/// Candidate-slot R1CS for the aerie private-salt HashToPoint relation; see
/// `AERIE-ADAPTER.md`.
pub mod hash_to_point_slots;
/// The aerie private-salt sponge lane: Keccak chains with the absorption
/// edge, salt-free record-start pinning, all as sub-cube openings.
pub mod hash_to_point_sponge;
pub mod keccak;
/// 3-wide Keccak-f[1600] R1CS (3 independent permutations per K_LOG=17 block)
/// for tighter PCS utilization (~97% vs the single encoder's ~65%).
pub mod keccak3;
pub mod keccak_gkr;
pub mod carry_gate;
/// Generic Merkle-path glue ([`MerkleLayout`]-parameterized prove/verify),
/// analogous to [`chain_common`] but with a per-row bit selector.
///
/// [`MerkleLayout`]: merkle_path_common::MerkleLayout
pub mod merkle_path_common;
pub mod sha2;
