# Merkle-path R1CS as a multi-table type — working notes

Branch `merkle-path-r1cs`, off `multitable`. Status: **working end to end at
depth 26**, both as a standalone single-type registry and as a shipped
`merkle26+blake3` mixed tier.

## Where this stands (read this first)

Done since the "alignment DONE, factorization NEXT" state:

1. **The `eq` tensor factorization** — the fold is 20× faster single-threaded
   (259 → 13 ms at depth 26), which cut end-to-end verify 3× (365 → 119 ms on
   `merkle26+blake3@nu3`).
2. **The repo builds on stable** — the `isolate_lowest_one` feature gates are
   gone, replaced by `bits::lowest_one`, with no version floor. Drop `+nightly`.
   CI's fmt/clippy gates now pass (they did not before).
3. **A matched-hash-count benchmark vs a plain BLAKE3 table**
   (`benches/merkle_vs_plain_blake3.rs`), swept to 26,624 compressions.
4. **Witness gen on the batch-major driver** — 32× (735 → 23 ms at 26k),
   taking Merkle prove 4.4× (1451 → 328 ms) and the prove gap from 8.2× to
   ~2.2×.

5. **The Frobenius assist, both sides.** Verify: `eq(pairs, σ)` hoisted out of
   the per-statement loop (it was recomputed 128·K times on identical inputs) —
   `W(σ)` 116.62 → 1.45 ms, **Merkle verify 140.6 → 25.0 ms (5.6×)**. Prove: the
   suffix tail is statement-independent above layer `params.n`, so it is built
   once — footprint 1247 → 522 MiB, prove/1t ~780 → 631 ms.

**Current state at 26,624 compressions**: prove ratio ~1.85×, verify ~2.50×
(was ~8× on both). At **depth 8** it is 1.24× / 1.47× — see the depth-8 section
for why, and for the free 23% available by using a power-of-two depth.

**What is left** is no longer one thing. The assist is down to 15% of verify at
depth 8 and the **lincheck fold is now 74%** of it. The assist's residual cost is
still `k_log`-driven (columns `2^(k_log-7)`), so narrowing the block helps and
more paths never do — which still makes the SHA-256 backend
**counterproductive** (κ = 15 ⇒ composite `k_log` 20 ⇒ more columns), reversing
what the open items below say. The bigger prover lever is the
**multipoint-twisted prototype** already in `pcs/jagged.rs`: it deletes the
128·K-statement structure rather than sharing within it (so it subsumes both
optimizations above), is tested against brute force, and is *not* wired into
`verify_batch_merged` — integration is the unfinished part.

Three claims in earlier revisions of this file were measured to be wrong and are
corrected in place: the verifier is not dominated by `fold_alpha_batched`; the
PCS *opening* is at parity (it is the jagged transport around it); and the
`zerocheck` asymmetry was page-fault noise, not real.

## What exists

| file | what |
|---|---|
| `crates/flock-prover/src/r1cs_hashes/merkle_r1cs.rs` | the composite R1CS, `HashSpec` backend abstraction, the batch-major witness driver (+ the `Vec<bool>` oracle), the walker `LincheckCircuit` + its `eq` factorization |
| `crates/flock-prover/tests/merkle_r1cs.rs` | R1CS correctness, `a`/`b` emission, BatchMajor layout, walker equivalence, packed-vs-bool witness equality |
| `crates/flock-prover/tests/merkle_union.rs` | union + mixed-tier roundtrips at depth 26 |
| `crates/flock-prover/src/mixed.rs` | `RegistryFamily`, `MerkleMixedSetup`, `MerkleMixedCounts`, tier codes 3 & 4 |
| `crates/flock-prover/benches/merkle_vs_plain_blake3.rs` | the matched-hash-count comparison; `MVB_PATHS` / `MVB_REPS` |

Env knobs added for A/B and attribution, all off by default:
`FLOCK_MERKLE_FOLD_PER_LEVEL` (skip the `eq` factorization),
`FLOCK_VERIFY_THREADS` (the verify pool is 1 thread in production),
`VERIFY_TRACE` (per-phase verify breakdown, incl. the merged union entry),
`PCS_TRACE` (per-phase prove breakdown — pre-existing).

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

## Scaling to 27k compressions: the ratio does NOT amortize away

`MVB_PATHS=8,128,256,512,1024`, 4 P-cores, verify 1 thread. Peak RSS 2.6 GB at
1024 paths.

| compressions | paths | dense_m | merkle prove/mt | blake3 | ratio | merkle verify | blake3 | ratio |
|---|---|---|---|---|---|---|---|---|
| 208 | 8 | 22 | 153 ms | 15 ms | 9.0× | 99 ms | 13.0 ms | 7.9× |
| 3,328 | 128 | 26 | 284 ms | 40 ms | 7.1× | 118.7 ms | 13.6 ms | 8.7× |
| 6,656 | 256 | 27 | 429 ms | 58 ms | 7.4× | 141.4 ms | 14.4 ms | 9.8× |
| 13,312 | 512 | 28 | 636 ms | 87 ms | 7.3× | 135.6 ms | 14.6 ms | 9.3× |
| 26,624 | 1024 | 29 | 1451 ms | 176 ms | 8.2× | 147.2 ms | 18.5 ms | 8.0× |

**The ~8× is scale-invariant over a 128× range.** Both sides amortize at the
same rate (merkle prove 734 → 54.5 µs/compression, blake3 73 → 6.6 µs), so the
ratio survives. **This refutes a prediction made in this session**: since the
Frobenius assist looked near-fixed (its column count is `k_log`-derived and
independent of row count), the gap "should" have collapsed at scale. It didn't,
because a second Merkle-specific cost grows with area.

Verify is the good news: nearly flat in absolute terms on both sides (Merkle
99 → 147 ms while the work grows 128×), so verify-per-compression falls
478 → 5.5 µs (Merkle) and 62 → 0.7 µs (BLAKE3).

### What grows: witness generation

| compressions | merkle wit/mt | blake3 wit/mt | share of merkle prove |
|---|---|---|---|
| 208 | 13 ms | 0 ms | 8% |
| 3,328 | 88 ms | 3 ms | 31% |
| 6,656 | 179 ms | 4 ms | 42% |
| 13,312 | 348 ms | 5 ms | 55% |
| 26,624 | 735 ms | 19 ms | 51% |

So the answer is **scale-dependent**: at 208 compressions the assist is ~94% of
the prove gap and witness gen ~8%; by 13k–27k, witness gen is over half of
Merkle prove and ~40× BLAKE3's per compression.

This was an implementation gap, not a protocol one. **FIXED** — see the next
section. The old path (`generate_witness_batch_major_partial_bool`, retained as
the test oracle) called the *packed* `node_witness_ab` hook into u64 buffers and
then threw the packing away:

1. unpacked bit-by-bit into three `vec![false; 2^19]` (1.5 MB of `Vec<bool>`
   per path, and `.collect()` held all of them at once — 1.6 GB at 1024 paths);
2. a **sequential** scatter loop repacked those bools into `F128` via
   `pack_word` (128 bit-tests per word);
3. the stripe pass walked the bools bit-by-bit a *third* time.

## The packed Merkle driver: witness gen 32×, prove 4.4× at 27k

The Merkle slot now runs on the **same** driver as the per-hash encoders —
`common::drive_witness_batch_major_partial_into` with a `BM_V = 8` lane-parallel
group builder — and reuses BLAKE3's own `build_group_batch_major` per level.

**Why that composes at all: the `2^κ` alignment, again.** Level `l`'s subcube is
exactly the base block at u64-row offset `l · 2^κ/64`, so the base encoder's
packed output drops in with **no bit shifting**. And per the pin-to-zero
subtlety above, the base encoder's `(z, a, b)` is already correct for every
overridden row — including the message region, where the gadget's
`a = S_j ⊕ t_j = left_j` coincides with the base free column's `a = left_j`. So
the driver writes the hash block verbatim and this code adds only the swap
gadget (sibling, `t`), level 0's globals, and a 4-word read-back of the output
CV to chain to the next level.

| compressions | wit before | wit after | speedup | prove/mt before | after | speedup | m/b3 ratio before → after |
|---|---|---|---|---|---|---|---|
| 208 | 13 ms | 0 ms | — | 153 ms | 121 ms | 1.3× | 9.0× → 9.3× |
| 6,656 | 179 ms | 5 ms | **36×** | 429 ms | 189 ms | **2.3×** | 7.4× → **3.2×** |
| 26,624 | 735 ms | 23 ms | **32×** | 1451 ms | 328 ms | **4.4×** | 8.2× → **2.2×** |

Merkle witness gen vs BLAKE3's at 27k is now **23 ms vs 13 ms — 1.8×**, down
from 39×. The residual 1.8× is the 6% extra bits plus the fact that a path's 26
levels are a sequential chain (only lanes parallelize, not levels). `prove/1t`
at 27k: 2933 → 765 ms. Verify is unchanged (147 → 143 ms), which is the sanity
check: witness gen is prover-only.

**Correction to a claim made earlier in this session.** I wrote that the 1.6 GB
of `Vec<bool>` was "most of the 2.6 GB peak RSS". It was not: peak RSS at 1024
paths measured **2.59 GB before and 2.56 GB after**, i.e. unchanged. Peak is set
by the PCS commit/open phase, and the witness peak sat below it. There is no
memory win here, only a time win.

Licensed by `packed_driver_matches_bool_reference`: the packed driver must be
**bit-identical** to the `Vec<bool>` oracle in all of `(z, a, b, stripe)`, at
depths 1/2/3/26, at full and partial counts (including 1-of-8 declared, so
dummy rows and the const pin are covered), with per-path varying indices so both
swap directions occur at every level.

### Full breakdown at 26,624 compressions, after the packed driver

Both columns are from **one** traced run each, and each reconciles with that
run's own total — the only way these are meaningful (see the caveat below).

**Prove** (4 threads; run totals: Merkle 424 ms, BLAKE3 176 ms):

| phase | merkle | blake3 | ratio | Δ |
|---|---|---|---|---|
| witness gen | 31 ms | 15 ms | 2.1× | +16 |
| `compact q` | 4.79 ms | 2.43 ms | 2.0× | +2 |
| `commit` | 34.39 ms | 38.19 ms | **0.9×** | −4 |
| `zerocheck + s_hat_v_c` | 63.12 ms | 60.76 ms | **1.04×** | +2 |
| `lincheck` | 15.45 ms | 10.50 ms | 1.5× | +5 |
| open · ring_switch | 0.22 ms | 0.20 ms | 1.1× | 0 |
| open · W build + round-0 | 6.20 ms | 6.09 ms | **1.0×** | 0 |
| open · merged sumcheck (22 rd) | 5.31 ms | 5.78 ms | **0.9×** | −1 |
| **open · coeffs + frobenius assist** | **226.55 ms** | **8.36 ms** | **27×** | **+218** |
| open · inner Ligerito | 35.47 ms | 30.87 ms | 1.1× | +5 |
| total | 424 ms | 176 ms | 2.4× | +248 |

**The assist is +218 of the +248 ms prove gap — 88%.** Everything else is at
parity or within 2×.

**Verify** (1 thread, production config; run totals: Merkle 158.9 ms, BLAKE3
15.8 ms):

| phase | merkle | blake3 | ratio | Δ |
|---|---|---|---|---|
| `bind_statement` | 0.3 µs | 0.4 µs | — | 0 |
| `zerocheck::verify` (m_total 29 both) | 375 µs | 295 µs | **1.3×** | 0 |
| `lincheck::verify_union` | 15.65 ms | 9.16 ms | 1.7× | +6 |
|  · of which `fold_alpha_batched` | 13.57 ms | 8.93 ms | 1.5× | +5 |
| pcs · `ring_switch::verify_succinct` | 246 µs | 251 µs | **1.0×** | 0 |
| pcs · `JaggedParams::from_heights` | 2.6 µs | 0.2 µs | — | 0 |
| pcs · coeffs | 676 µs | 53 µs | 12.7× | +1 |
| **pcs · `verify_frobenius_assist`** | **141.15 ms** | **5.37 ms** | **26×** | **+136** |
| pcs · Ligerito open | 545 µs | 534 µs | **1.0×** | 0 |
| total | 158.9 ms | 15.8 ms | 10.1× | +143 |

**The assist is +136 of the +143 ms verify gap — 95%.** `k_cols` is 12 vs 7 as
always (4096 vs 128 columns).

So after the witness fix there is **exactly one term left** on either side, and
it is the same term. Post-fix ratios at 26,624: prove/mt 2.2–3.0×, prove/1t
**1.9×**, verify 10.1× (verify was untouched — witness gen is prover-only).

### Re-run after merging `multitable` (8 commits, incl. a zerocheck rewrite)

`multitable` was merged in (`e5de8fb`): the `product_gkr` grand-product port,
one shared `eq(ρ)` across the permutation's five v-openings, two zerocheck
changes that stop scanning/scattering over the **padded** domain, and scratch
prewarm sized by buffer class. Those zerocheck changes were the ones plausibly
relevant here, since the Merkle table carries more padding per row than BLAKE3's
(interior holes per level, 81% vs 94% utilization).

A/B at 26,624 compressions, 7-rep medians, back to back on the same machine:

| | pre-merge | post-merge |
|---|---|---|
| merkle prove/mt | 348 ms | **321 ms** (−8%) |
| merkle prove/1t | 766 ms | 733 ms (−4%) |
| merkle verify | 140.3 ms | 141.5 ms (flat) |
| blake3 prove/mt | 156 ms | 156 ms (flat) |
| blake3 verify | 14.4 ms | 14.1 ms (flat) |

So: a modest Merkle prove gain, nothing else. Treat the 8% as suggestive rather
than established — earlier in this session two 3-rep runs of the *same* build
differed by 17% (328 vs 385 ms), so 8% is near the edge of what this harness
distinguishes. BLAKE3 being flat while Merkle moves is at least consistent with
the padded-domain story.

**The structure is unchanged.** Post-merge phase breakdown at 26,624, each
column from one traced run and reconciled against that run's own total:

Prove (totals 335 vs 159 ms): witness 22/15, `compact q` 4.26/2.09, `commit`
32.51/32.09 (**1.0×**), `zerocheck` 54.11/50.23 (**1.08×**), `lincheck`
13.19/10.12, ring_switch 0.21/0.19, W build 6.88/5.59, merged sumcheck
5.68/6.34 (**0.9×**), **frobenius assist 161.55/7.08 (23×)**, inner Ligerito
30.00/35.63 (**0.8×** — Merkle faster). Assist = **88%** of the +176 ms gap.

Verify (totals 139.9 vs 14.6 ms): `zerocheck::verify` 272/262 µs (**1.0×**),
`lincheck::verify_union` 14.26/8.96 ms (fold 12.43/8.72), ring_switch 223/243 µs
(**0.9×**), coeffs 307/256 µs, **`verify_frobenius_assist` 124.04/4.98 ms
(25×)**, Ligerito open 496/493 µs (**1.0×**). Assist = **95%** of the +125 ms
gap.

Ratios post-merge: prove/mt 2.1–2.5×, prove/1t **1.8×**, verify 9.9×. Identical
88%/95% attribution to pre-merge — the merge changed the constant, not the
shape. `verify_frobenius_assist` remains the only thing worth attacking.

### The prover's analogue: a shared suffix tail — 58% less memory, prove/1t 14%

The verifier's hoist does not apply to the prover (its suffix-row streaming is a
different algorithm), but a *different* sharing does, and it comes from two facts
in the code rather than from the protocol:

* `assist_suffix_rows` destructures `&(_, t_c, t_next)` — it reads only the
  height PAIR, never the per-column weights.
* `point_bit(z, layer)` returns `ZERO` past `z.len()`. So
  `eq4s[layer] = eq([point_bit(z_row, layer), rho[layer]])` depends **only on
  rho** for every `layer >= z_row.len() = params.n` — and squaring `ZERO` is
  `ZERO`, so this holds at every Frobenius power.

The backward recurrence reads only `eq4s[layer..]` plus the pairs, so by
induction **every suffix block from layer `params.n` up is identical across all
128·K statements**. At 1024 paths (`n = 10`, `m = 22`) that is blocks 10..23 —
14 of 24. `assist_shared_suffix_tail` builds them once; each statement
materializes only layers `0..n`, seeded from the tail.

| | before | after | |
|---|---|---|---|
| suffix footprint | 1247 MiB | **522 MiB** | **58% less** |
| statements + suffix build | 92.32 ms | 60.54 ms | 1.5× |
| `v` + round loop | 49.83 ms | 50.70 ms | unchanged, as predicted |
| `free()` the suffix rows | 44.00 ms | 26.23 ms | 1.7× |
| **assist total** | **186.22 ms** | **137.64 ms** | **1.35×** |
| Merkle prove/1t | 733–838 ms | **631 ms** | ~14% |
| peak RSS | 2.78 GB | **2.24 GB** | |

The 58% memory saving matched the prediction exactly (`1 − n/(m+2)` = 58.3%).
The build saved less than that (35%, not 58%) because the per-spec pass also does
`assist_columns` — a `2^k`-entry `eq` table per statement — which does **not**
share, since each statement's `z_col` is genuinely Frobenius-powered. The round
loop is unchanged by design: it reads the same blocks, just some of them from the
shared array. Verify is unaffected (it passes `None` and never builds suffixes).

Note the sharing fraction is `1 − n/(m+2)`, so it **grows as rows shrink** —
Merkle at `nu = 10` saves 58% where BLAKE3 at `nu = 15` saves 37%. The Merkle
table benefits more, which is the right direction for once.

Licensed by `split_suffix_matches_monolithic`: the shared tail plus the
per-statement low blocks must reassemble into exactly what `assist_suffix_rows`
builds in one piece, checked block-by-block over 6 shapes including
`n_row == m` (only the seed shared) and `n_row > m+1` (nothing shared). Writing
that test *before* rewiring the prover paid for itself — it caught an index
underflow immediately (`assist_low_blocks` clamped to `m+2`, leaving the tail
with no seed block for the per-statement pass to start from), which would
otherwise have surfaced as a corrupt proof deep in a roundtrip.

### Depth 8: near parity, and 19% of every depth-26 row is wasted

`MVB_DEPTH=8`. A depth-8 path is 8 compressions, so 4096 paths = 32,768 of them
— matched against 32,768 loose BLAKE3 blocks as usual.

| | depth 8 (4096 paths, 32,768) | depth 26 (1024 paths, 26,624) |
|---|---|---|
| `k_log` / jagged columns | 17 / 1024 (1022 merged) | 19 / 4096 (3326 merged) |
| row utilization | **99.6%** | 81.2% |
| `dense_m` | 29 | 29 |
| merkle prove/mt | 214 ms | 287 ms |
| blake3 prove/mt | 172 ms | 155 ms |
| **prove ratio** | **1.24×** | 1.85× |
| merkle verify | **14.4 ms** | 25.7 ms |
| blake3 verify | 9.8 ms | 10.3 ms |
| **verify ratio** | **1.47×** | 2.50× |

The column model predicted this and it held: 4× fewer columns bought **4.5×
cheaper assist** on both sides — verify 9.35 → 2.07 ms, prove 137.64 → 29.66 ms,
suffix footprint 522 → 192 MiB. Circuit size is identical to BLAKE3 again
(21,029,709 vs 21,028,097 nnz/hash, +0.008%).

**The lincheck fold now dominates at depth 8.** Of the 14.4 ms verify:
`lincheck::verify_union` 10.23 ms (**74%**, `fold_alpha_batched` 9.66 ms), assist
2.07 ms (15%), Ligerito open 0.50 ms. Since BLAKE3's fold is ~8.7 ms the fold is
near parity, so most of the residual 1.47× is the assist's last 1.5 ms. That is a
complete reversal of where this investigation started.

**19% of every depth-26 row is wasted, and it is avoidable.**
`k_log = 14 + log2(next_pow2(depth))`, so depth 26 allocates **32** level slots
and uses 26; depth 8 uses 8 of 8. Hence 99.6% vs 81.2%, and why depth 8 fits
32,768 compressions in the same `dense_m = 29` where depth 26 fits 26,624.

The actionable form: **depth 32 costs exactly what depth 26 costs** — same
`k_log = 19`, same column count, same assist, same commitment — while proving 32
compressions per row instead of 26. That is 23% more hashing for free. If the
depth is a design choice, powers of two are strictly better in this layout; if 26
is fixed by the application, the 19% is the price of the aligned-subcube layout
(which is what buys the `eq` factorization, so it is not a regression — just a
cost worth knowing).

Verify is flat in path count at depth 8 too (13.1 → 14.6 → 14.4 ms across
32 → 1024 → 4096 paths), consistent with cost set by block width, not batch size.

`packed_driver_matches_bool_reference` covers depth 8 at full and partial counts;
every bench configuration round-trips prove + verify + claim equality before any
timing is reported, so all six depth-8 configs are verified proofs.

### Recycling the suffix buffers: the `free()` third disappears

The suffix drop was ~24-31% of the assist prove. It is **per-page** work, not
per-call — 0.77 µs per 16 KiB page at both depth 8 (192 MiB, 9.3 ms) and depth 26
(522 MiB, 26.2 ms) — so consolidating the 256 allocations into one would not have
helped. Only not returning the pages to the OS helps, which is precisely what
`crate::scratch` exists for, and its module doc already names this exact cost.
That pool is bounded at 24 buffers though, sized for the prove cycle's few
giants, so `128·K` of these would evict everything. Hence a dedicated
`sfx_pool` (same contract; `clear_suffix_pool()` releases it).

Depth 26, per-proof, from one traced run:

| | pre-pool (steady) | cold proof #1 | warm proof #3 |
|---|---|---|---|
| statements + suffix build | 60.54 ms | 118.21 ms | **20.73 ms** |
| `v` + round loop | 50.70 ms | 59.72 ms | **29.93 ms** |
| free / recycle | 26.23 ms | 0.13 ms | **0.05 ms** |
| **assist total** | **137.64 ms** | 178.15 ms | **51.00 ms** |

**Read the two right-hand columns separately — they are different claims.**

* The `free()` → `recycle` saving (26.23 → 0.05 ms, 525×) is unconditional: even
  a single cold proof gets it, because the buffers go to the pool instead of
  `munmap`.
* The build and round-loop savings (60.54 → 20.73 and 50.70 → 29.93) are
  **warm-pool page-residency effects, not algorithmic**. Cold proof #1 is
  *slower* than the pre-pool steady state (118.21 vs 60.54), because pre-pool the
  allocator could recycle pages the previous proof had just freed, whereas the
  first pooled proof faults 522 MiB fresh. Proof #2 is 38.30 and #3 is 20.73, so
  it takes a couple of proofs to converge.

So: a long-running prover gets the full 2.7× on the assist; a one-shot CLI
invocation gets ~26 ms. The bench warms up before timing, so its medians are the
steady-state figure.

Headline, 7-rep medians:

| | merkle | blake3 | ratio |
|---|---|---|---|
| depth 26, 26,624 · prove/mt | **208 ms** (was 287) | 146 ms | **1.42×** (was 1.85) |
| depth 26 · prove/1t | **569 ms** (was 631) | 401 ms | 1.42× |
| depth 8, 32,768 · prove/mt | **181 ms** (was 214) | 165 ms | **1.10×** (was 1.24) |
| depth 8 · prove/1t | **508 ms** (was 553) | 457 ms | 1.11× |

The trade is explicit: the prover now keeps the suffix footprint resident
between proofs. Peak RSS moved 2.24 → 2.51 GB at depth 26.

### The zerocheck asymmetry: RESOLVED, it was not real

An earlier revision of these notes flagged `zerocheck + s_hat_v_c` at 224 ms vs
54 ms at 1024 paths as an unexplained asymmetry, despite `m_total = 29` on both
sides. It measures **63.12 vs 60.76 ms — parity** — after the packed driver.

Nothing in the zerocheck changed. What changed is that the old witness path
allocated and freed 1.6 GB of `Vec<bool>` immediately beforehand, so the
zerocheck then ran on cold, faulting pages. It was allocator/page-fault noise
attributed to the wrong phase. Worth remembering as a measurement hazard: a
phase can be slow because of what the *previous* phase did to the page tables.

## Inside the Frobenius assist

`VERIFY_TRACE` / `PCS_TRACE` now split the assist itself. At 26,624
compressions (`k_cols` 12 vs 7 ⇒ 3326 vs 122 merged columns, 256 statements,
m = 22):

**Verify** — the assist is **one loop**:

| phase | merkle | blake3 | ratio | share of merkle assist |
|---|---|---|---|---|
| round replay (46 rounds) | 3.3 µs | 3.2 µs | 1.0× | 0.0% |
| `frobenius_statements` | 5.75 ms | 277 µs | 21× | 4.6% |
| · of which spec enumeration | 12 µs | 13 µs | 1.0× | 0.0% |
| **`assist_w_at` — the `W(σ)` walk** | **116.62 ms** | **4.23 ms** | **27.6×** | **93%** |
| boundary DP (4-state, m+1 layers) | ~0.2 ms | ~0.2 ms | 1.0× | 0.2% |
| assist total | 125.05 ms | 4.74 ms | 26× | |

The 99.8%/93% split of `W(σ)` vs the DP is measured, not inferred (per-statement
nanos accumulator; the verify pool is 1 thread so the sum is wall-clock). And
27.6× is the column ratio 3326/122 = 27.3× — so `assist_w_at` *is* the
`O(2^k)` term, exactly as its own doc comment claims ("`2(m+1)`
multiplications per distinct column — the verifier's only `2^k`-scale work").

Its body is `cols × (m+1) × 2` multiplies: for each column, a fresh
`Π_layer eq(t_{y-1}[layer], σ_c) · eq(t_y[layer], σ_d)`. That is
256 × 3326 × 46 ≈ **39.2M multiplies**, and 116.62 ms / 39.2M = **2.97 ns per
F128 multiply** — i.e. the loop is already at raw multiply cost. It will not
yield to micro-optimization; it needs *fewer products*. Two openings worth
costing: Merkle's columns are **one run of equal heights** (all 3325 used
columns have height `n_t`, so `t_y = y·n_t` is an arithmetic progression —
consecutive `eq` products are related), and the 256 statements all walk the
*same* column set with only `σ` differing.

### The hoist: `W(σ)` computed once, not 128·K times — verify 5.6×

Writing out what `W(σ)` computes exposed pure redundancy. In
`verify_frobenius_assist`, across the 256 statements:

* `σ` is the sumcheck's final point — **sampled once, shared**;
* the column height pairs `(t_{y-1}, t_y)` come from `params.col_prefix_sums`
  alone, and `assist_columns` merges purely on pair equality — so the pair
  sequence, its order, and the merge boundaries are **identical** in every
  statement;
* only the per-column *weight* differs (each statement's Frobenius-powered
  `eq(z_c, ·)` scaled by `c_{i,j}`).

So the whole `Π_layer eq(t[layer], σ)` tensor product — all `2(m+1)` multiplies
per column — was being recomputed 256 times on identical inputs.
[`assist_eq_pairs_at`] now computes it **once** into a shared table, and each
statement becomes a dot product `Σ_y w_y · E_y`.

| | before | after | |
|---|---|---|---|
| `eq(pairs, σ)` | — | 473 µs | computed once |
| per-statement dot + DP | — | 976 µs | (dot 756 µs summed) |
| **`W(σ)` total** | **116.62 ms** | **1.45 ms** | **80×** |
| `frobenius_statements` | 5.75 ms | 6.24 ms | unchanged |
| **assist total** | **125.05 ms** | **9.35 ms** | **13.4×** |
| **Merkle verify** | **140.6 ms** | **25.0 ms** | **5.6×** |
| BLAKE3 verify | 14.2 ms | 10.2 ms | 1.4× |
| verify ratio | 9.9× | **2.45×** | |

80×, better than the 38× the multiply count predicted (39.2M → 1.0M) — the old
inner loop selected `σ` or `1+σ` per bit, so it was branch-bound as well as
multiply-bound, while the dot product is a clean stream. Prove is untouched and
unchanged (~160 ms BLAKE3 / ~320–390 ms Merkle across runs; a 209 ms BLAKE3
reading in one run was thermal noise, confirmed by re-running).

Licensed by `hoisted_eq_pairs_match_assist_w_at`, which asserts **bit-identical**
results against `assist_w_at` (same products, different association order — and
GF(2^128) has no rounding to hide behind) *and* pins the shared-pair-sequence
invariant directly rather than leaving it to a `debug_assert`, since if that ever
breaks the hoist is silently wrong for every statement but the first.

**Verify at 26,624 is now 25.3 ms, and the lincheck fold is the biggest phase
again**: `lincheck::verify_union` 14.64 ms (58%, of which `fold_alpha_batched`
~12.4 ms vs BLAKE3's 8.7 ms — only 1.4×) against the assist's 9.35 ms. Inside
the assist, `frobenius_statements` is now dominant at 6.24 ms (67%) — that one
does not hoist, since each statement's `z_col` is genuinely different (Frobenius
powers), so its `2^k`-entry `eq` table really is per-statement. ~1.7 ms of the
9.35 ms is unattributed; by analogy with the prover it is likely the statement
drop (~34 MB), not verified.

**Prove** — three roughly equal thirds, all column-scaled. **Not** improved by
the hoist above, which is verifier-only; the prover's suffix-row streaming is a
different algorithm:

(Superseded by the shared-suffix-tail section below — the numbers here are the
pre-optimization baseline.)

| phase | merkle | blake3 | ratio | share |
|---|---|---|---|---|
| `linearized_coefficients` (×2) | 0.03 ms | 0.03 ms | 1.0× | 0.0% |
| statements + suffix rows | 92.32 ms | 2.45 ms | 38× | 50% |
| `v` + round loop | 49.83 ms | 2.37 ms | 21× | 27% |
| **`free()` the suffix rows** | **44.00 ms** | 1.73 ms | 25× | **24%** |
| assist total | 186.22 ms | 6.62 ms | 28× | |

**A quarter of the Merkle assist prove time is deallocation.** The suffix rows
are `(m+2)·n_cols` × `[F128; 4]` per statement — **1247 MiB across the 256
statements** (4.9 MiB each) versus 45 MiB for BLAKE3. That is the assist
prover's whole footprint, it is transient, and it explains both the `free()` cost
and why this phase is memory-bound rather than arithmetic-bound. Note this also
resolves the residual these notes previously logged as unattributed (~39 ms with
"no sub-timer"): it was the drop, happening after the last timer printed.

`linearized_coefficients` was the other suspect for that residual. It is 0.03 ms
— not a factor at all.

### Remaining caveat

(The former caveat here — an unattributed residual inside
`coeffs + frobenius assist` — is resolved above: it was the suffix-row
deallocation, which happened after the last timer printed.)

Also: prove varies run to run at these sizes, and `PCS_TRACE=1` itself costs
time, so phase breakdowns are only valid *within* a single run — never compared
against a table from another. Both tables above are internally reconciled
against their own run's totals for exactly this reason.

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
