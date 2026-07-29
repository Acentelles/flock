# Merkle-path R1CS as a multi-table type — working notes

Branch `merkle-path-r1cs`, off `multitable`. Status: **working end to end at
depth 26**, both as a standalone single-type registry and as a shipped
`merkle26+blake3` mixed tier.

## What exists

| file | what |
|---|---|
| `crates/flock-prover/src/r1cs_hashes/merkle_r1cs.rs` | the composite R1CS, `HashSpec` backend abstraction, witness + row-witness generators, the walker `LincheckCircuit` + its `eq` factorization |
| `crates/flock-prover/tests/merkle_r1cs.rs` | R1CS correctness, `a`/`b` emission, BatchMajor layout, walker equivalence |
| `crates/flock-prover/tests/merkle_union.rs` | union + mixed-tier roundtrips at depth 26 |
| `crates/flock-prover/src/mixed.rs` | `RegistryFamily`, `MerkleMixedSetup`, `MerkleMixedCounts`, tier codes 3 & 4 |

Run: `cargo test -p flock-prover --release --test merkle_r1cs --test merkle_union`
(add `-- --include-ignored` for the real-depth proofs). No `+nightly` — see the
resolved toolchain item under open items.

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
**88.8 MB for 546,771,284 effective nonzeros, a 49× reduction.** Licensed by
`walker_matches_materialized`, which compares against `CscCircuit` over the
fully materialized matrices at depths 1-3.

Storage was the walker's first purpose; holding the base block *as a unit* is
also what makes the `eq` factorization expressible, so it is now the arithmetic
win too — see the fold section below. (The line that used to sit here, "same
gather count — storage only", is no longer true.)

`build_matrices()` (materialized) is retained as the small-depth oracle;
`build_block_r1cs_stub()` + `build_walker()` is the real path.

## Geometry

Levels occupy **aligned `2^14` subcubes** (see the factorization section —
the alignment is load-bearing). depth 26, BLAKE3: `k_log = 19`,
`useful_bits = 425,521`, 3,325 chunk-columns/path.

| nu | paths | dense words | dense_m | M (single-type) |
|---|---|---|---|---|
| 3 | 8 | 26,600 | 22 | 22 |
| 4 | 16 | 53,200 | 23 | 23 |
| 7 | 128 | 425,600 | 26 | 26 |

Mixed `merkle26+blake3`: Merkle (κ=19) sorts before BLAKE3 (κ=14); areas
`33·2^(nu+14)` round to `2^(nu+20)`, so `M = nu + 20`, ~52% utilization.

Measured, `merkle26+blake3@nu3`, dense_m 22, **before the fold factorization**
(and on a different machine, so compare ratios not absolutes — the current
numbers, both paths measured in one process, are in the fold section below):

```
8 merkle + 8 blake3   prove 246 ms   verify 501 ms
8 merkle + 0 blake3   prove 204 ms   verify 496 ms
0 merkle + 8 blake3   prove  79 ms   verify 387 ms
```

---

# The ~25× lincheck fold: **DONE** (alignment + factorization)

**Benedikt's observation, confirmed.** `fold_alpha_batched` is `O(nnz) = 547M`
field ops and both sides pay
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

## Step 1 — the aligned re-layout: **DONE**

The old `level_stride = 15,921` was not a power of two, so the level index was
not a set of address bits and `eq` did not factor. Now `level_stride = 2^κ`:
level `l`'s subcube IS the base block, and the gadget columns live in the base
block's own padding region `[15409, 16384)` (975 free columns):

```
level l at l·2^14:
  [0, 15409)        base block, verbatim
  [15409, 15665)    sibling S_l
  [15665, 15921)    t_l
level 0 additionally (asserted to fit; holds for depth ≤ 206):
  [15921, 15922)          global const  (const_pin)
  [15922, 16178)          leaf
  [16178, 16178+depth)    index bits
```

`k_log = κ + log2(next_pow2(depth))` — unchanged at depths 1, 2, 3, 26.
`CONST_POS` is no longer 0; it is `layout.const_pos()`.

Cost as predicted: `useful_bits` 414,229 → 425,521 (+2.7%), chunk-cols
3237 → 3325. **`dense_m` at nu=3 unchanged (22), `k_log` unchanged (19)** — free
at the tier we ship. All tests including `walker_matches_materialized` and the
real-depth roundtrips pass unchanged.

## Step 2 — the factorization: **DONE**

`MerkleWalkerCircuit::fold_alpha_batched` now dispatches:

| | what | cost at depth 26 |
|---|---|---|
| `factor_eq` | recover `(eq_base, ρ)`, **verify** it | 0.43M muls |
| `fold_factored` | one base fold, then one mul per composite column | ~21.5M |
| `fold_per_level` | the old `depth` gathers — general fallback | 547M |

Measured (`walker_factored_matches_per_level_at_depth26`):

```
546,771,284 effective nonzeros
 parallel:  factored   3.2 ms   per-level   47.6 ms   (15.1×)
 1 thread:  factored  12.9 ms   per-level  258.9 ms   (20.1×)
```

**The 1-thread column is the one that matters**: the verifier runs its whole
PIOP replay inside a dedicated single-thread pool (`flock_core::verifier`,
"intentionally single-threaded — matching binius64/plonky3/hashcaster"), so the
parallel column *understates* the win. The per-level walk parallelizes better
than the base fold (547M independent gathers vs 2^14 skewed columns), which is
why the ratio improves when you take the threads away.

12.9 ms is essentially the arithmetic floor for this backend: at the per-level
path's 0.47 ns/gather, 21.5M gathers alone is ~10.2 ms. The remaining base fold
is BLAKE3's own block density — the only way further down is the SHA-256
backend (1.3M/block, see open items), not more cleverness here.

End to end, `merkle26+blake3@nu3`, dense_m 22, A/B in one process via
`FLOCK_MERKLE_FOLD_PER_LEVEL`:

| | verify per-level | verify factored |
|---|---|---|
| 5 merkle + 6 blake3 | 346 ms | **119 ms** |
| 8 merkle + 8 blake3 | 345 ms | **115 ms** |
| 8 merkle + 0 blake3 | 347 ms | **113 ms** |
| 0 merkle + 8 blake3 | 269 ms | **39 ms** |

**3.0× on total verify** (345 → 115 ms), and the 246 ms saved matches the
1-thread fold delta (259 → 13 ms) almost exactly, which is the cross-check that
the win is where we think it is. Prove also drops (215 → 176 ms), less
dramatically — it has other dominant costs.

Note the last row: verify folds **every registry slot regardless of its
count**, so a proof with zero Merkle paths still paid the full 547M walk. That
is why 0+8 improves 7×.

Correction to an earlier claim in these notes: the verifier is *not* dominated
by `fold_alpha_batched` to the extent assumed. The residual is
`jagged::verify_frobenius_assist` — see the phase breakdown in the
Merkle-vs-BLAKE3 section, which measures it rather than guessing (an earlier
revision of this line guessed "PCS opening + zerocheck replay" and both were
wrong: each is ~250 µs).

## How `eq_hi` is recovered

`fold_alpha_batched` receives the *table*, not the point, so ρ is recovered
from it by the ratio trick: pick the first `(l_ref, r0)` with a nonzero entry,
set `eq_base := eq_inner[l_ref·2^κ ..]` (absorbing `ρ_{l_ref}`, so it is 1) and
`ρ_l := eq_inner[l·2^κ + r0] · inv(eq_base[r0])`. One F128 inversion.

Departure from the original plan: the factorization is **verified in full**,
not `debug_assert`-sampled. Every `l < depth`, `r < 2^κ` entry is checked
against `ρ_l · eq_base[r]` — 0.43M multiplies, 2% of the folded cost and 0.08%
of what it replaces. That buys two things worth far more than 2%:

* `fold_alpha_batched` keeps its old contract (correct for *any* `eq_inner`)
  rather than silently requiring a structured table, so no caller can be
  broken by a future non-product table;
* `walker_matches_materialized` stays meaningful. Random `eq` legitimately does
  *not* factor, so it now exercises the fallback, and a separate
  `walker_factored_matches_materialized` feeds real `build_quirky_eq_table`
  tables and asserts (i) they DO factor — otherwise the fast path would be
  silently dead — and (ii) the result still matches the materialized matrices
  column for column.

At depth 26 there is no materialized oracle, so
`walker_factored_matches_per_level_at_depth26` pins factored against
per-level, and the small-depth tests pin per-level against real matrices.

The alternative — extending `LincheckCircuit` with a method that receives the
structured point — is still cleaner in principle but touches a core trait and
all implementors. The ratio trick needed no core change.

Why the table is rank-1, recorded because the fast path silently depends on it:
`build_quirky_eq_table` is `L_{i_skip}(z_skip) · eq(x_rest, i_rest)` with the
skip dim in the **low** `K_SKIP = 6` bits and `eq` a per-bit product above
them, so it factors at every bit boundary ≥ 6; levels tile aligned `2^κ`
subcubes with `κ = 14 > 6`. The union's per-slot `w_t` prefix weight scales the
*comb*, not the table, so it does not interfere.

---

# What the structure costs: Merkle table vs plain BLAKE3 table

`benches/merkle_vs_plain_blake3.rs`. A depth-26 path is 26 compressions, so `n`
paths is measured against **26n loose BLAKE3 compressions** — same hashing work,
different R1CS encoding. Both sides go through the same union entry as a
single-type registry, so only the table type differs. M4, 4 P-cores, median of 7.

Caveat on what this compares: the Merkle side also proves the level-to-level
dataflow (each digest feeds the next, plus a conditional swap per level); the
plain side proves 26n *unrelated* compressions and nothing about how they
connect. The plain side is a **lower bound** on any real Merkle circuit, not an
equivalent statement.

| | hashes | rows | dense_m | proof | wit/mt | prove/mt | prove/1t | verify/1t | verify/4t |
|---|---|---|---|---|---|---|---|---|---|
| merkle ×8 | 208 | 8 | 22 | 226 KiB | 10 ms | 126 ms | 256 ms | 95.0 ms | 32.1 ms |
| blake3 | 208 | 208 | 22 | 220 KiB | 0 ms | 14 ms | 21 ms | 12.0 ms | 4.7 ms |
| merkle ×16 | 416 | 16 | 23 | 237 KiB | 14 ms | 140 ms | 286 ms | 100.2 ms | 35.7 ms |
| blake3 | 416 | 416 | 23 | 231 KiB | 0 ms | 19 ms | 26 ms | 12.6 ms | 4.4 ms |
| merkle ×32 | 832 | 32 | 24 | 249 KiB | 20 ms | 154 ms | 331 ms | 105.0 ms | 35.9 ms |
| blake3 | 832 | 832 | 24 | 242 KiB | 1 ms | 17 ms | 31 ms | 12.7 ms | 6.1 ms |
| merkle ×64 | 1664 | 64 | 25 | 269 KiB | 41 ms | 183 ms | 401 ms | 109.5 ms | 36.5 ms |
| blake3 | 1664 | 1664 | 25 | 261 KiB | 1 ms | 22 ms | 44 ms | 13.4 ms | 4.7 ms |

**The structure costs roughly 8×** — prove 8.5–11× multi-threaded, 9–12×
single-threaded, verify 6–8×. Stable across a 8× range of sizes.

Three things worth keeping:

* **The circuit is the same size — this is not a bigger circuit.** Measured:

  | | k_log | useful bits | bits/hash | base nnz | system nnz | nnz/hash |
  |---|---|---|---|---|---|---|
  | merkle ×8 | 19 | 3,404,168 | 16,366 | 546,771,284 | 4,374,170,272 | 21,029,664 |
  | blake3 ×208 | 14 | 3,205,072 | 15,409 | 21,028,097 | 4,373,844,176 | 21,028,097 |

  The two constraint systems differ by **0.0075%** in nonzeros (1,567 per
  compression — the swap gadget) and 6.2% in witness bits. `dense_m` and proof
  size match for the same reason. So the 8× is not circuit size and not data
  volume; it is the *shape*, same area split as (few rows × `2^19` block)
  instead of (many rows × `2^14`). See the phase breakdown below for which
  term that actually hits.
* **Both verifies are nearly flat in size** (Merkle 95 → 110 ms while the work
  grows 8×), which is the same statement: verify is fixed cost. So Merkle
  verify-per-compression falls 457 → 66 µs from 8 to 64 paths and keeps
  improving; the 8× ratio is what survives after both sides amortize.
* **Witness generation is not the story.** Broken out on purpose: Merkle
  witness gen is 10–41 ms (BLAKE3's is 0–1 ms — 26n independent compressions
  parallelize perfectly, a path's 26 levels are a sequential chain over only
  8–64 paths). That is 8–22% of Merkle prove, so ~90% of the gap is genuinely
  proving, not hashing.

## Where the 8× actually is: `verify_frobenius_assist`

Measured, not inferred (`VERIFY_TRACE=1 MVB_REPS=1 MVB_PATHS=8`, 1 thread,
steady-state rep). **Two earlier guesses in these notes were wrong; this
replaces them.**

| phase | merkle (k_log 19) | blake3 (k_log 14) | ratio |
|---|---|---|---|
| `bind_statement` | 0.3 µs | 0.3 µs | — |
| `zerocheck::verify` (m_total 22 both) | 268 µs | 291 µs | **1.0×** |
| `lincheck::verify_union` | 14.4 ms | 9.9 ms | 1.5× |
|  · of which `fold_alpha_batched` | 12.4 ms | 9.6 ms | 1.3× |
|  · of which sumcheck replay | 530 µs (13 rd) | 91 µs (8 rd) | — |
| `ring_switch::verify_succinct` ×2 | 234 µs | 245 µs | **1.0×** |
| `JaggedParams::from_heights` | 6.5 µs | 0.5 µs | — |
| coeffs (fold byte tables) | 53 µs | 53 µs | **1.0×** |
| **`jagged::verify_frobenius_assist`** | **81.5 ms** | **3.3 ms** | **24.5×** |
| `verify_opening_batch_ligerito_mixed` | 252 µs | 224 µs | **1.0×** |
| total | 96.8 ms | 14.1 ms | 6.9× |

**`verify_frobenius_assist` is 78 ms of the ~83 ms gap — 94% of it.** Everything
else is at parity or nearly so.

Note what this rules out:

* **The zerocheck is identical** (268 vs 291 µs) — `m_total = nu + k_log = 22`
  on both sides, so it never could have been the difference.
* **The actual PCS opening is identical** (252 vs 224 µs). The Ligerito
  recursive verify is ~0.24 ms total (`LIG_VERIFY_TRACE`). "PCS is expensive"
  was wrong; the *jagged transport around it* is.
* **The lincheck is only 1.5× apart**, and the fold inside it 1.3× — the eq
  factorization brought a 26× structural disadvantage to near parity, which is
  the real vindication of that work. It also kills the "13 rounds vs 8 rounds
  and a 2^19 comb vector" story: those cost 530 µs and 91 µs.

**Mechanism.** The Frobenius assist scales with the number of committed jagged
**columns**, and that count is set by the BLOCK WIDTH, not the row count:

```
columns = 2^col_log,  col_log = m_total − LOG_PACKING − nu = k_log − 7
  merkle  k_log 19 → 2^12 = 4096 columns   (used: ⌈425521/128⌉ = 3325)
  blake3  k_log 14 → 2^7  =  128 columns   (used: ⌈ 15409/128⌉ =  121)
```

32× the columns, 24.5× the time. And since `col_log` cancels `nu`, **adding
paths does not help** — which is exactly why both verifies are flat in size
(Merkle 95 → 110 ms while work grows 8×) and why per-compression verify
improves purely by amortization.

**Consequence for what to do next.** Merkle verify is a function of `k_log`
alone, so the lever is narrowing the composite block, not more paths and not
the fold. Note this makes the SHA-256 backend *counterproductive* for verify:
κ = 15 gives a composite `k_log = 15 + 5 = 20`, i.e. **more** columns than
BLAKE3's 19 — it would cut the fold (already at parity) and grow the term that
actually dominates. The candidates worth costing are making the assist itself
cheaper, or reducing `depth`'s contribution to `k_log`.

Verify parallelizes ~2.9× on 4 cores (95 → 32 ms) but production ships the
1-thread pool deliberately; `FLOCK_VERIFY_THREADS` exists only for this bench.

---

## Other open items

- **`proof_io` / CLI don't know the new tiers.** `MixedProofBundleLigerito`
  already carries `registry_id` + `counts: Vec<u64>`, so serialization works,
  but `cmd_prove_mix` / `cmd_verify_mix` hardcode the SHA-256+BLAKE3 shape
  (including a `counts.len() != 2` check that happens to pass).
  `--mix merkle=N,blake3=M` needs parsing.
- ~~**Two `#![feature(isolate_most_least_significant_one)]` lines**~~
  **RESOLVED, the second way.** Taking the notes' own second option: the four
  sites are now `flock_core::bits::lowest_one` (`x & x.wrapping_neg()`), and
  both `#![feature]` lines are gone. The method itself is no longer used
  anywhere.

  Dropping the gates alone was not enough. `isolate_most_least_significant_one`
  stabilized in **1.97.0**, so gate-removal makes the tree need stable ≥ 1.97 —
  but `edition = "2024"` otherwise builds on anything from 1.85, so that would
  have silently raised the floor twelve releases and broken exactly the older
  toolchain that motivated the gates in the first place. The rewrite depends on
  no version at all.

  Note also that the gates had stopped being merely redundant and become
  *breaking*: `#![feature]` is a hard error (E0554) on stable, so the gated tree
  did not build on stable 1.97.1 **at all**, only on nightly. That is the
  build failure to expect if you check out any commit before `ba3dea6`.

  Verified: `cargo build --workspace --all-targets` (release and debug) on
  stable 1.97.1 **and** nightly 1.99. `bits::lowest_one` is pinned against a
  `trailing_zeros` reference — deliberately not against
  `usize::isolate_lowest_one`, since the point is that the tree builds without
  it, tests included. `main` never carried the gates, so this restores parity
  rather than diverging. **Drop `+nightly` from the run commands.**

  Not declared: a `rust-version` MSRV. The true floor is now whatever edition
  2024 needs (1.85), but only 1.97.1 and 1.99 were actually tested here, and an
  untested MSRV claim is worse than none.
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
  `node_witness`, `node_witness_ab`, `fixed_bits`. Now **complementary** to the
  factorization rather than an alternative to it: the factored fold's cost is
  one base block, so SHA-256 would take that from 21M to 1.3M nonzeros — the
  only remaining lever on the fold, at 2× the committed area. Worth it only if
  the fold is still the bottleneck after the PCS-opening work, which at 115 ms
  total verify it no longer is.
