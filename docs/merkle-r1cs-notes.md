# Merkle-path R1CS as a multi-table type — working notes

Branch `merkle-path-r1cs`, off `multitable`. Status: **working end to end at
depth 26**, both as a standalone single-type registry and as a shipped
`merkle26+blake3` mixed tier.

## What exists

| file | what |
|---|---|
| `crates/flock-prover/src/r1cs_hashes/merkle_r1cs.rs` | the composite R1CS, `HashSpec` backend abstraction, witness + row-witness generators, the walker `LincheckCircuit` |
| `crates/flock-prover/tests/merkle_r1cs.rs` | R1CS correctness, `a`/`b` emission, BatchMajor layout, walker equivalence |
| `crates/flock-prover/tests/merkle_union.rs` | union + mixed-tier roundtrips at depth 26 |
| `crates/flock-prover/src/mixed.rs` | `RegistryFamily`, `MerkleMixedSetup`, `MerkleMixedCounts`, tier codes 3 & 4 |

Run: `cargo +nightly test -p flock-prover --release --test merkle_r1cs --test merkle_union`
(add `-- --ignored` for the real-depth proofs).

## The design in one page

The union model forbids constraints between rows (design doc §3), so **one
table row = one whole Merkle path**. Level `l` computes

```
b_l   = bit l of the index
left  = b_l·prev ⊕ (1 ⊕ b_l)·S_l          (conditional swap over GF(2))
right = left ⊕ prev ⊕ S_l
prev' = H(left ‖ right)
```

Under the circuit shape `(A·z) ⊙ (B·z) = z`, the swap costs one AND per bit:

```
t_j     = b_l · (prev_j ⊕ S_j)     A = [b_l],          B = [prev_j, S_j]
left_j  = S_j  ⊕ t_j               A = [S_j, t_j],     B = [const]
right_j = prev_j ⊕ t_j             A = [prev_j, t_j],  B = [const]
```

`left`/`right` are **not new columns** — they ARE the hash block's 512-bit
message region, whose rows the composite overrides. So the only new columns
per level are the 256 `t_j`.

Each level embeds the base encoder's block by a **pure column offset**. Only
three row groups per level are overridden: the block's own constant wire
(re-derived from the global one), the message region (the gadget), and the
remaining free inputs (`cv = IV`, `counter = 0`, `block_len = 64`,
`flags = PARENT`) pinned to constants.

**Load-bearing subtlety.** Pin-to-zero uses `A = [], B = [const]`, *not* an
empty `B`. That makes the base encoder's own row-witness `(z, a, b)` correct
for every overridden row, so `a`/`b` are emitted by copying the base block in
at the level offset — no matrix application anywhere.
`emitted_ab_matches_matrix_application` pins this.

## Why BLAKE3 needs a walker

Measured nonzeros per base block: **BLAKE3 21.0M** (929/row, max 2514) vs
**SHA-256 1.30M** (28.6/row). BLAKE3's encoder deliberately trades density for
a small `k` (`blake3.rs:14-31`, "Option D"): no sum-bit columns, and lanes 0-3
/ 8-11 cascade, inlining prior carry chains.

That trade is right for a batch of *independent* compressions (`A = I_n ⊗ A_0`,
one matrix shared by all `n`). It inverts under *concatenation*: 26 copies at
distinct column offsets → 547M nonzeros ≈ **4.4 GB**, plus another 4.4 GB for
the CSC transpose.

`MerkleWalkerCircuit` stores ONE base CSC and walks it per level:
**88.8 MB for 546,771,284 effective nonzeros, a 49× reduction.** Same gather
count — storage only. Licensed by `walker_matches_materialized`, which compares
against `CscCircuit` over the fully materialized matrices at depths 1-3.

`build_matrices()` (materialized) is retained as the small-depth oracle;
`build_block_r1cs_stub()` + `build_walker()` is the real path.

## Geometry

depth 26, BLAKE3: `k_log = 19`, `useful_bits = 414,229` (79% utilization),
3,237 chunk-columns/path.

| nu | paths | dense words | dense_m | M (single-type) |
|---|---|---|---|---|
| 3 | 8 | 25,896 | 22 | 22 |
| 4 | 16 | 51,792 | 23 | 23 |
| 7 | 128 | 414,336 | 26 | 26 |

Mixed `merkle26+blake3`: Merkle (κ=19) sorts before BLAKE3 (κ=14); areas
`33·2^(nu+14)` round to `2^(nu+20)`, so `M = nu + 20`, ~52% utilization.

Measured, `merkle26+blake3@nu3`, dense_m 22:

```
8 merkle + 8 blake3   prove 246 ms   verify 501 ms
8 merkle + 0 blake3   prove 204 ms   verify 496 ms
0 merkle + 8 blake3   prove  79 ms   verify 387 ms
```

---

# NEXT: a ~25× faster lincheck fold (in flight, not implemented)

**Benedikt's observation, confirmed.** The verifier's cost is dominated by
`fold_alpha_batched`, which is `O(nnz) = 547M` field ops — and both sides pay
it. But the composite is `A ≈ I_depth ⊗ A_blake3`, so the column marginal
should factor.

It does. `build_quirky_eq_table` (`lincheck.rs:947`) produces

```
eq_inner[i_skip + i_rest·2^k_skip] = L_{i_skip}(z_skip) · eq(x_inner_rest, i_rest)
```

with `eq_rest = build_eq_table(...)` a per-bit product — so `eq_inner` is
**rank-1**. If level `l` occupies the *aligned* subcube
`[l·2^κ_base, (l+1)·2^κ_base)`, the level index is a set of high address bits
and

```
eq_inner[l·2^κ + r_b] = eq_hi[l] · eq_base[r_b]
⇒ ξ_A(l·2^κ + c_b)   = eq_hi[l] · ξ_base(c_b)
```

One base fold (21M) + one multiply per composite column (0.5M) + extras (94K)
≈ **21.6M vs 547M, a 25× cut on prover AND verifier.**

## What blocks it today

`level_stride = 15,921` is **not** a power of two, so the level index is not a
set of address bits and `eq` does not factor across level boundaries. Alignment
is exactly what's missing. (Verified: without it, each level's `eq` slice is an
arbitrary slice of the tensor, not rank-1 in `r_b`.)

## The re-layout

Set `level_stride = 2^κ_base = 16,384` — i.e. level `l`'s subcube IS the base
block. The gadget columns move into the base block's own padding region
`[15409, 16384)`, which has 975 free columns:

```
level l at l·2^14:
  [0, 15409)        base block, verbatim
  [15409, 15665)    sibling S_l
  [15665, 15921)    t_l
level 0 additionally (fits: 15409 + 769 + depth ≤ 16384 for depth ≤ 206):
  [15921, 15922)          global const  (const_pin)
  [15922, 16178)          leaf
  [16178, 16178+depth)    index bits
```

`k_log = κ_base + ceil(log2(depth))`. Unchanged at depths 1, 2, 3, 26.

Cost: `useful_bits` 414,229 → 425,521 (+2.7%), chunk-cols 3237 → 3325 (+2.7%).
**`dense_m` at nu=3 is unchanged (22), and `k_log` is unchanged (19)** — so at
the tier we ship, the re-layout is free.

## Implementation note: extracting `eq_hi`

`fold_alpha_batched` receives the *table*, not the point, so `eq_hi` must be
recovered from it. Since the `2^(k_log−κ) × 2^κ` view is rank-1, all level
slices are scalar multiples of each other:

1. find the first `(l_ref, r0)` with `eq_inner[l_ref·2^κ + r0] ≠ 0`;
2. `ρ_l = eq_inner[l·2^κ + r0] · inv(eq_inner[l_ref·2^κ + r0])`;
3. `comb` on level `l` = `ρ_l · fold(eq_inner[l_ref·2^κ .. ])`.

Needs one F128 inversion (see `permutation.rs:138 batch_inverse` for the
existing helper). All-zero `eq_inner` ⇒ all-zero comb, handle trivially.
Add a `debug_assert` sampling `eq_inner[l·2^κ + r] == ρ_l · eq_inner[l_ref·2^κ + r]`.

The alternative — extending `LincheckCircuit` with a method that receives the
structured point — is cleaner but touches a core trait and all implementors.
The ratio trick needs no core change.

`walker_matches_materialized` is the safety net: it compares the walker against
materialized matrices column-for-column on random `eq_inner`, so a broken
factorization fails there immediately. **Do the re-layout and the factorization
as separate commits, running that test between them.**

---

## Other open items

- **`proof_io` / CLI don't know the new tiers.** `MixedProofBundleLigerito`
  already carries `registry_id` + `counts: Vec<u64>`, so serialization works,
  but `cmd_prove_mix` / `cmd_verify_mix` hardcode the SHA-256+BLAKE3 shape
  (including a `counts.len() != 2` check that happens to pass).
  `--mix merkle=N,blake3=M` needs parsing.
- **Two `#![feature(isolate_most_least_significant_one)]` lines** in
  `flock-core/src/lib.rs` and `flock-prover/src/lib.rs`, committed separately
  so they are trivially revertible. The repo does not compile on any installed
  toolchain without them — this is pre-existing on `main`, not from this work
  (`ntt/inv_table.rs:87`, `inv_table_deg4.rs:110`,
  `zerocheck/multilinear.rs:385`, `chain.rs:399`). Drop the lines on a
  toolchain where the feature is stable, or rewrite the four sites as
  `x & x.wrapping_neg()` to be stable-clean.
- **The digest gap.** With stub matrices, `Registry::digest` binds the Merkle
  slot's `k_log`, `useful_bits` and `const_pin` — hence its depth, and in
  practice its backend — but **not** its constraint system. The verifier's
  guarantee rests on rebuilding the same walker from the tier id. Same
  convention as the Keccak encoders; deliberate, but worth revisiting if these
  proofs ever leave the lab.
- **`flags = PARENT` on every level, including the top.** Real BLAKE3 tree
  hashing also sets `ROOT` on the final parent. Uniformity is what makes the
  composite `depth` identical blocks; `HashSpec::flags` is the knob.
- **No public-input binding.** Leaf, index bits and root are free columns at
  `leaf_bit` / `index_bit` / `root_bit`. Binding them needs claim-level glue or
  the planned bus layer (design doc §8, §11).
- **SHA-256 backend.** `HashSpec` is ready for it (identical I/O-aligned region
  shape: in-CV `[0,256)`, out-CV `[256,512)`, message `[512,1024)`); needs
  `node_witness`, `node_witness_ab`, `fixed_bits`. Would be 16× cheaper on the
  fold (34M vs 547M nonzeros) at 2× the committed area — a real alternative to
  the factorization above, or complementary to it.
