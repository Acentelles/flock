# RS-path port log

Record of porting engineering optimizations from the Yukon challenge repo
(`Layr-Labs/flock-challenge` @ `c576e68`) into this tree's RS zerocheck path.
Goal: close the gap while keeping added code minimal, and keep every change
attributable to a measured per-phase delta.

Every number here is `benchmarks/breakdown_phases.sh` at 2^18 BLAKE3 on an
**Apple M1 Max (8 P-cores, 32 GB)**, back-to-back A/B in the same session.
Reproduce a row with:

```sh
LABEL=<tag> OUT=phases.tsv ./benchmarks/breakdown_phases.sh
```

## Baseline (`9793190`) and the target

| phase | ours ST | ours 8T | challenge ST | challenge 8T |
|---|---:|---:|---:|---:|
| witness | 351.4 | 81.1 | *not reported* | *not reported* |
| commit | 1277.4 | 196.2 | 1796.7 | 255.9 |
| zc round 1 | 1441.4 | 196.7 | 314.1 | 43.7 |
| zc round 2 | 433.1 | 58.8 | *(combined)* | *(combined)* |
| zc rounds 3+ | 452.4 | 66.4 | 538.2 (r2+r3+) | 78.0 |
| lincheck | 183.4 | 30.6 | 31.9 | 13.4 |
| open | 636.9 | 100.2 | 565.2 | 89.5 |
| **total** | **4781.0** | **723.9** | — | — |

Two caveats on the challenge-repo column, both established with the session
that produced it:

1. Those numbers come from `prove_fast_timed`, which in that repo is a
   structurally separate path from `prove_fast` and measured **~15% slower**
   (405.7-420.7 ms vs 473-484 ms for the same work). Ours agrees with its own
   headline `prove_fast` to within 1.4%, so the two columns are not on equal
   footing. Where the 15% sits is unknown; if it is concentrated in `commit`
   (the dominant phase there, and the one touching its pinned allocation) then
   its commit figure is overstated by a lot.
2. That repo moves work **across phase boundaries** —
   `round1_c_fold4_from_lincheck_stripe` and `stage_c_prelude_for_tail_fill`
   shift C-side work between lincheck and the zerocheck. Its very low lincheck
   number is therefore partly re-attribution, not necessarily a real win.

Treat the column as directional, not as a scoreboard.

## Attempts

| # | change | phase | ST | 8T | verdict |
|---|---|---|---|---|---|
| 1 | batch reduced msg muls via `ghash_mul_vec2_neon` | rounds 3+ | 455.2 → 500.8 (**+10.0%**) | 69.2 → 78.3 (+13.2%) | **reverted** |
| 2 | `WideNeon` register-resident accumulator | round 2 | 433.1 → 375.2 (**−13.4%**) | 58.8 → 55.0 (−6.4%) | **kept** |
| 3 | same, rounds 3+ | rounds 3+ | 452.5 → 455.7 (+0.7%) | 68.6 → 69.2 (+0.9%) | dropped |
| 4 | defer round-1 partial-sum reduction to once per x_hi | round 1 | 1441.4 → 1447.1 (+0.4%) | 196.7 → 203.2 (+3.3%) | **reverted** |
| 5 | port the multilinear lookahead to the RS tail | rounds 3+ | −7.5% *(pre-measured)* | — | not attempted |
| 6 | fetch inv-NTT rows with one `LD1 x4` | round 1 | 1455.8 → 1634.6 (**+12.3%**) | 192.1 → 218.4 (+13.7%) | **reverted** |
| 7 | fold XOR accumulation into `EOR3` pairs | round 1 | 1444.3 → 1469.9 (+1.8%) | *(8T arm discarded, drift)* | **reverted** |
| 8 | hoist the challenge-independent AB transform out of the zerocheck, `rayon::join`ed with the commit (+ `stnp` non-temporal stores) | round 1 | 1474.6 → 601.1 (**−59%**) but **total unchanged** | 219.8 → 81.4 (−63%), total unchanged | **reverted** |
| 9 | nibble-split convert tables (64 KiB → 8 KiB hot table, gathers 48 → 96/lane) | round-1 drain | headline 53508 → 51582 comp/s (**−3.6%**, base 8/8) | — | **reverted** |
| 10 | lincheck-stripe dedup for round 1's C input | round-1 drain | *impossible* — byte groupings run on disjoint axes (stride 64 vs 2^14) | — | **closed on algebra** |
| 11 | geometric eq-build in lincheck | lincheck | whole eq build is 0.13 ms; geometric variant *slower* at 5/6 sizes | — | **closed for free** |
| 12 | **skip structurally-zero b K-rows** | round 1 | **1475.4 → 1433.0 ms (−42.4, −2.9%), head 8/8** | — | **KEPT** |
| 13 | constant-fold all-ones b K-rows | round 1 | 1420.0 → 1413.9 (−6.1, −0.43%), 7/8 | — | reverted (~90 lines for 6 ms) |
| 14 | **two lanes per iteration in the drain** | round-1 drain | **1483.1 → 1420.0 ms (−63.1, −4.3%), head 8/8** | — | **KEPT** |
| 15 | **unreduced pmull accumulate + x^K weight split (x⁴ table image · x² byte-mul · u16 shift)** | round-1 prep | **1401.4 → 1273.4 ms (−128.0, −9.1%), head 8/8, every pair −8.7..−10.1%** | — | **KEPT** |
| 16 | **stripe-fold C side: round-1 C banks from one multilinear fold of the lincheck stripe; drain runs AB-only, C transpose deleted** | round 1 | **1277.5 → 1204.7 ms (−72.7, −5.7%), head 8/8** | — | **KEPT** |
| 17 | **q-resident round 2: fold outputs stay in q registers, in-register karatsuba `mul_q` (5 PMULLs), `WideNeon` fed directly** | round 2 | **358.1 → 311.6 ms (−13.0%), 8/8, every pair −12.6..−13.3%** | — | **KEPT** |
| 18 | **fused q-resident rounds-3+ tail: fold+message in one pass, second read pass over multi-MB chunks deleted** | rounds 3+ | **449.5 → 306.6 ms (−31.8%), 8/8, every pair −31.1..−32.6% — largest single win of the effort** | — | **KEPT** |
| 19 | **stripe fold through lincheck's tiled dispatcher** (was calling the portable fallback; its 256 KiB accumulator thrashes L2 at k_log=14) | round-1 C fold | **1191.0 → 1023.7 ms (−167.3, −14.0%), 8/8, every pair −13.6..−14.4%** | — | **KEPT** |
| 21 | b === all-ones round-2 pair degeneration | round 2 | 312.0 → 312.4 ms, sign 4-4 — **null** (third partial-skip-vs-ILP confirmation) | — | reverted |
| 22 | non-temporal (STNP) round-2 output stores | round 2 | 313.1 → 350.3 ms (**+12%**), base 8/8 — STNP *inverts* on M1 (M4-specific idiom, their ablation +1.2%) | — | **reverted** |
| 23 | **four lanes per iteration in the AB-only drain** | round-1 drain | 960.5 → 954.4 ms (−6.0, −0.6%), 7/8 | — | **KEPT** (10 lines) |
| 25 | word-extract in the round-2 fold | round 2 | 313.1 → 305.3 ms (−7.9, −2.5%), 6/8 | — | **KEPT** |
| 26 | two round-2 pairs per iteration | round 2 | 306.5 → 317.9 (+3.7%), base 8/8; **re-verified under challenge**: 307.1 vs 325.4, unroll worse 8/8 disjoint, regression larger under load (register spills amplify with memory contention) | — | reverted, double-sourced |
| 24 | static-b partial loads — bounded by probe, not implemented | round-1 prep | deleting ALL b gathers: 954 → 750.6 ms, so ceiling = 33.7% × 204 ≈ **69 ms**; realistic ≈ 15–20 at measured partial-skip capture rates, vs 200–800 lines | — | **closed on the bound** |
| 20 | **word-extract addressing in the prep** (16 byte-loads per K-row → 2 word loads + shifts) | round-1 prep | **1027.6 → 974.8 ms (−52.7, −5.1%), 6/6** | — | **KEPT** |

Net kept: **round 2 −13.4% ST**, total 4781.0 → 4716.2 ST (−1.4%), for 179
added lines.

## What the negative results tell us

- **Round 1 is not multiply-bound** (attempt 4: reductions cut by a factor of
  `big_lo_size`, products from 6 PMULLs to 3, no movement). An earlier version
  of this log inferred from that "round 1 is gather-bound on the convert
  table". **That inference was wrong.** Measuring the split directly
  (temporary `FLOCK_R1_SPLIT` scaffold, since removed) gives, at 2^18 ST:

  | round-1 component | ST ms | of round 1 | of whole prove |
  |---|---:|---:|---:|
  | `shift_reduce_inner_ab` | ~810 | 56% | 17% |
  | `bit_transpose_64bytes` | ~136 | 9% | 3% |
  | `accumulate_convert` | ~521 | 36% | 11% |

  So the convert-table accumulate is only a third of round 1; the prep pass
  dominates. Reproduce by re-adding two `Instant` counters around the b_med
  prep loop and the `accumulate_convert_with_s_hat_v` call in
  `process_one_x_hi_with_s_hat_v` (per-x_outer_lo granularity; per-b_med timers
  add ~470 ms of their own overhead and only the ratio survives).
- **`shift_reduce_inner_ab` is limited by neither load-issue nor XOR-issue
  count.** Attempts 6 and 7 cut load instructions 4x (8 per byte-pair to 2) and
  XOR ops ~1.75x (56 to 32) respectively; the first cost 12.3% and the second
  1.8%. On Apple cores the four-register structured `LD1` is microcoded and
  loses to four independent 128-bit loads, and `EOR3` bought nothing. What is
  left as the plausible limiter is load *latency* / L1 port pressure against
  the 16 KB inv-NTT table — whose gathers are data-dependent and cannot be
  batched — plus the `gf8_mul_vec16` work. Neither yields to a local rewrite,
  which is consistent with the challenge repo needing specialized and generated
  kernels here rather than a tidier loop.
- **Interface shape can outweigh instruction counts.** Attempt 1 reduced PMULLs
  (4/mul vs binius's 6) and still lost 10%, because `ghash_mul_vec2_neon` takes
  and returns `[F128; 2]` and forces operands through memory. It earns its
  place in the NTT and `f128_slice` call sites only because one operand there
  is a loop-invariant broadcast.
- **Lookahead loses on this hardware.** `cargo bench --bench ag_lookahead_ab`
  (paired, same-process, proofs asserted bit-identical) gives classic 285.9 ms
  vs lookahead 307.2 ms ST at m=30 — lookahead faster 0/4 runs. The code's own
  comments target an M4; do not port it to the RS tail on an M1 without
  re-measuring.
- **Our `commit` is already ~1.4x faster than theirs** (1277 vs 1797 ms ST),
  plausibly from #29/#30 which `c576e68` predates. Cross-pollination runs both
  ways; commit is not a target for us.

## Where the branch ended up

Three changes kept, everything else reverted with its measurement in the commit
message. Against `9793190` (the harness-only commit, before any optimization),
2^18 BLAKE3 ST, paired n=8 with alternating arm order:

| level | base | head | delta |
|---|---:|---:|---|
| round 1 (`round1 URM`) | 1477.30 ms | 1023.72 ms | **-30.7%**, each step 8/8 |
| round 2 (`round2 fused fold`) | ~433 ms | 311.56 ms | **-28%** (WideNeon + q-resident) |
| rounds 3+ (`rounds 3+ tail`) | ~455 ms | 306.56 ms | **-33%** (fused one-pass q-resident) |
| end-to-end headline | ~52,400 comp/s | ~58,100 comp/s | **~+11%**, 8/8 (clean-band read; per-pair delta stable +9.6..+12.4% even through throttled pairs) |

(Supersedes the interim +5.1% figure from the four-win state. The final run was
taken with an active browser session; pairs 4-8 form tight bands on both arms
and cross-check against the predicted sum of the individual wins, ~+7.9%.)

Caveat on the end-to-end figure: the base arm spread that run was wide
(46894-51655, ~10%) while head was tight (51231-53209), so the point estimate is
soft even though the sign test is not. The round-1 number is the better
measured of the two.

The seven kept changes:

1. **Round-2 NEON register accumulator** (~179 lines) -- `WideNeon`, a 256-bit
   product held as two uint64x2_t instead of the GPR-resident F256Unreduced.
   -13% on `zc_round2`.
2. **Structurally-zero b K-row skip** (~10 lines) -- see above. Part of the
   -5.0%.
3. **Two lanes per drain iteration** (~50 lines) -- see above.
4. **Unreduced pmull accumulate + x^K weight split** (~80 lines) -- the
   challenge repo's top-attributed AB-prep mechanism, ported as an idea.
   -128 ms on round 1 by itself (1401.4 -> 1273.4, 8/8, predicted 100-130 from
   their attribution).
5. **Stripe-fold C side** (~150 lines) -- round-1 C banks from one multilinear
   fold of the lincheck stripe; drain AB-only, C transpose deleted. -72.7 ms.
6. **q-resident round 2** (~120 lines) -- fold outputs stay in q registers,
   in-register karatsuba mul_q, WideNeon fed directly. -46.6 ms (-13.0%).
7. **Fused q-resident rounds-3+ tail** (~90 lines) -- fold and message in one
   pass; the second read over multi-MB chunks deleted. -142.9 ms (-31.8%),
   the largest single win. Two earlier failures on this exact loop (pair-mul
   kernel +10%, WideNeon-alone 0%) had located the cost in the pass structure
   and struct crossings, not the arithmetic.

## What bounds each round-1 kernel (measured, not inferred)

Five local rewrites of round 1 have now failed. Taken together they say
something fairly precise about the two kernels, which is more useful than any
of the individual results:

- **The drain is bound by gather COUNT.** Doubling gathers (48 → 96 per lane)
  while shrinking the hot table 8x (64 KiB → 8 KiB) cost 3.6%. M1 has 128 KiB
  of L1D per performance core, so the 256-row convert table already fit and
  there was no footprint problem to fix. Calibrating from that regression, the
  drain's 48 gathers/lane are worth roughly 170 ms of its ~521 ms.
- **The prep kernel is bound by neither load-issue nor XOR-issue count.**
  Cutting load instructions 4x (LD1 x4) cost 12.3%; cutting XOR ops 1.75x
  (EOR3 pairing) gave +1.8%. Deferring its reduction entirely moved nothing.
- **And it is not bound by anything the witness's structure could unlock.**
  Byte statistics of the packed BLAKE3 witness at 2^12 blocks:

  | buffer | zeros | dominant byte | all-0xff rows | uniform 8-byte rows |
  |---|---:|---|---:|---:|
  | a | 6.6% | — | 0.0% | 5.9% |
  | b | 6.1% | **0xff at 28.9%** | 9.4% | 15.3% |
  | z | 12.6% | — | 0.0% | 5.9% |

  `b` is strikingly non-random, which is presumably why the challenge repo has
  a `static_b` / `mixed_const_b` / `single_k0_static_b` kernel family. But the
  0xff bytes are scattered inside mixed rows rather than clustered: only 9.4%
  of b's aligned 8-byte rows are uniformly 0xff, and the 5.9% all-zero rows
  (identical in all three buffers) are padding that `b_med_counts` already
  skips. Row-level constant specialization is therefore worth ~4% of prep
  loads here -- order 10 ms -- not the 262 ms of the AB-prep gap.

The uncomfortable implication: their AB prep is 548 ms against our ~810 ms
while doing strictly MORE memory work (it streams 512 MiB out through
non-temporal stores; ours writes a 1 KiB L1 scratch). So their kernel is
genuinely ~1.5x better code at the same computation, and none of the
structural explanations we can test account for it. That points at
`fused_apply_one_k_fast` / `fast_shift_reduce_with_policy` / the 839-line
generated `aarch64_bstatic_gen.rs` -- i.e. the specialized-and-generated
kernel zoo, which is exactly the bloat this effort set out to avoid.

## Anatomy of the remaining zerocheck gap (from the challenge-repo session)

The session holding the c576e68 checkout read its own kernels and explained the
mechanisms behind each sub-phase advantage. Recorded here because the *shape*
of each answer matters for what upstream should do next.

**AB prep (548 vs ~770 ms): one real arithmetic-kernel win.** Their
`fused_apply_one_k_fast` replaces the incumbent's per-lane REDUCED GF(2^8)
multiply (`gf8_mul_vec16`) with an UNREDUCED carry-less multiply (raw
PMULL/PMULL2) and defers all reduction into an incremental Horner fold
(`acc = acc*x XOR lo XOR hi`, one fused BCAX per step, precomputed carry
constant x^16 mod p = 0x5e). Same gathers, same passes; cheaper arithmetic.
~50-60% of the prep gap, per their attribution. The rest is traffic: zero-copy
views into the precompute buffer instead of a per-row scratch memcpy
(DIRECT_AB_ROWS) and skipping the zero-fill of dead tail rows
(AB_COMPACT_STORE). This is the one honest kernel-quality gap, and it is
portable as a technique.

**Round-1 drain (314 vs ~590 ms): structural — they do not run this
computation.** Their active path never reads c_packed, never bit-transposes,
never touches the convert table. Because C = I (C aliases z), the round-1
C-claim derives from the LINCHECK STRIPE via an eq-fold
(`partial_fold_packed_z_best`) plus ring-switch's own fold8
(`s_hat_v_fold8_from_z_vec`), a small quad fold, and one collapse to the
two-half s_hat_v_c layout. This refines the "stripe dedup does not exist"
section below: the index algebra there is correct — the stripe cannot feed
*this tree's drain shape* — but they compute the same OUTPUT by a different
algorithm, so the dedup exists at the algorithm level, not the buffer level.
Their per-lane gather cost model simply does not apply. (Unresolved caveat:
part of their 314 ms may be a Metal GPU prefix they could not isolate.)

Decomposition of OUR 590 ms drain, all measured: ~136 ms C bit-transpose,
~75 ms per-lane eq multiplies (timing probe: removing the muls took round 1
from ~1403 to ~1330 ms), ~380 ms gathers+XORs.

**Rounds 2+ (538 vs ~800 ms): NOT lookahead — lookahead is a shared regression.**
At our request they kill-switched their sumcheck lookahead on their own machine:
591 -> 514 ms with it OFF (13-15% loss when on, all 6 paired samples),
matching this tree's M1 measurement (285.9 vs 307.2 in the AG tail). It is
default-ON in their tree and loses on both chips. Cascade2/cascade3 (composed
double-folds via a 32 KiB rho byte table) was the initial candidate for their
remaining rounds-2+ edge, but a follow-up kill-switch run on their machine says
otherwise: with cascade OFF and lookahead ON, rounds 2+ measured ~736 ms avg
(noisy, 612-964, low confidence), while their fastest configuration remains
lookahead fully OFF (~514 ms) -- where cascade structurally never fires, since
it requires lookahead. So their rounds-2+ advantage over this tree's ~800 ms
lives in the BASE per-round kernels: the register-resident wide-arithmetic
family in their multilinear kernels (2779 lines against this tree's 56; the
same family our round-2 WideNeon win was one piece of), not the fusion
machinery. Cascade's own contribution could not be cleanly isolated.

Direct consequence for THIS tree: the ligerito OPEN runs the same lookahead
family by default (landed in #30, presumably tuned on the M4 Max reference
machine) with the kill-switch `LIG_LOOKAHEAD_DISABLE=1`. Measured separately —
see below if a result was recorded.

**Connective tissue:** a size-classed scratch allocator (take_f128/give_f128)
recycling large buffers across prove calls; the same recycle-don't-allocate
pattern as their largest historical non-GPU win.

Net revision to the earlier "no big idea" conclusion: the gap is NOT dozens of
micro-tunings. It is one structural algorithm change (drain), one arithmetic
kernel (prep), one fusion family (rounds 2+), plus a shared lookahead
regression that is an upstream opportunity rather than a deficit.

## The lincheck-stripe dedup does not exist (checked, closed)

The most promising remaining idea was that `z` gets transposed twice -- once
into the lincheck stripe, once again by round 1's `bit_transpose_64bytes` --
and that round 1 could read its C input out of the stripe instead, deleting
~136 ms of transpose plus the ~113 ms of C gathers it feeds. The challenge
repo's active path is even named `round1_c_fold8_from_lincheck_stripe`.

It does not work: the two byte-groupings run along different axes. Traced by
setting one logical witness bit at a time (m=18, k_log=14):

| byte | logical bits feeding it | stride |
|---|---|---:|
| round-1 C `[b_med=0][lane=0]` | 0, 64, 128, ..., 448 | 64 |
| stripe `[byte_idx=0][i_inner=0]` | 0, 16384, ..., 114688 | 16384 |

With `logical = i_inner + i_outer·K` and `K = 2^14`, round 1's byte runs along
logical bits 6-8 (three of the within-block inner dims) while the stripe's runs
along `i_outer`'s low three bits, logical bits 14-16. Disjoint. Converting one
grouping into the other is precisely the transpose we wanted to skip.

Two corollaries:

- The challenge repo does not avoid this transpose either. Its C drain still
  calls `bit_transpose_64bytes` into a local scratch (confirmed by the session
  that has that checkout), so `..._from_lincheck_stripe` names the buffer it
  reads, not an avoided pass.
- The z transpose is already better handled here than "in parallel" would be:
  `generate_witness_with_ab_packed_and_lincheck` fuses it into witness
  generation, bit-transposing z u64s into the stripe while they are still hot
  in L1, and replaces the standalone `pack_z_lincheck_from_packed` on the fast
  path. Splitting it out to overlap it would add a 512 MiB DRAM round trip to
  buy concurrency on an already-saturated pool -- the same trap as the AB
  hoist. The remaining `pack_z_lincheck_from_packed` call sites are the generic
  `prove_ligerito` path only.

## The lincheck gap is re-attribution, and its fold has no reachable headroom

Their lincheck is 32 ms against our 183 ms -- the largest ratio in the table
(5.7x) and the row we explored last. It is not a real gap.

**Physics.** `partial_fold_packed_z_neon_*` is byte-table driven: per input byte
it loads the byte, loads a 16-byte `build_sum_table` entry, and XORs into an
accumulator pinned in a Q register. At the 2^18 BLAKE3 shape z_packed is 512
MiB, so 2^30 loads. M1 sustains ~3 loads/cycle, giving a floor of ~112 ms
(~84 ms allowing for the `useful_bits` padding skip). Their 32 ms works out to
**0.19 cycles per input byte**, roughly 3.5x below that floor, and even a
16-bit-table variant (two bytes per lookup) only floors at ~0.34. So their
lincheck cannot be performing this fold -- the C-side fold is presumably
computed in round 1 from the stripe and reused, consistent with the active path
being named `round1_c_fold8_from_lincheck_stripe`.

Correcting for this, the real comparable gap is ~970 ms, not ~1120 ms.

**And the genuine headroom is not reachable.** Our fold does sit 1.6-2.2x above
its own floor, but:

- It is insensitive to blocking. `FOLD_AB=1` A/Bs the size-aware dispatch
  against forced `iblock` interleaved per m: 0.988x at m=26, 0.972x at m=28,
  0.994x at m=29. Both strategies land within 3%.
- The only load-reducing transform is arithmetically dominated. `build_sum_table`
  builds 256 entries with 255 XORs by doubling; a 16-bit table needs 65535 XORs
  to enable only `k = 2^14 = 16384` lookups per stripe, so the build costs 4x
  more than it saves. (And the two stripe bytes for one `i_inner` are `k` bytes
  apart, so forming a u16 index needs two loads regardless -- loads would go
  4 -> 3, not 4 -> 2.) It could only pay at much larger `k_log`.

**Stale comment worth fixing.** `benches/lincheck.rs` documents oblock beating
iblock by "≈1.4-1.7x by m=28-29 at this k_log". That does not reproduce here --
see the ratios above. Whoever tuned `OBLOCK_MIN_N_LOG = 16` did it on different
hardware or the win has since regressed; do not trust that comment on M1.

**Also closed for free:** the geometric eq-build (`build_eq_table_optimized` in
their tree, prototyped here in `benches/eq_build_probe.rs`). Running that probe
shows the entire `SplitEqGhash::new` at the round-2 shape costs 0.13 ms, and the
geometric variant is *slower* than the standard build at 5 of 6 sizes. Worth
approximately zero.

## The measured round-1 decomposition (post-stripe-C), and what it closed

A second FLOCK_R1_SPLIT probe (since removed) replaced the estimated
decomposition with a measured one and immediately found a defect:

| component | estimated | measured | note |
|---|---:|---:|---|
| AB prep | ~640 | ~715 | 90% gathers: a gathers-only probe put the whole multiply tail at ~67 ms, killing the h4-Horner port idea (ceiling ~15 ms for ~100 lines) |
| stripe fold | ~180 | **340 -> 166** | was calling the PORTABLE fold; `partial_fold_packed_z_best` (lincheck's tiled NEON dispatcher) halves it -- the portable kernel's length-k accumulator is 256 KiB at k_log=14, twice M1's L1D |
| AB drain | ~340 | ~167 | near its gather floor all along; the estimate that made it look like a target was wrong |

With the dispatcher fix, drain+fold is ~333 ms against their ~314 --
effectively at parity (and theirs may include an unquantified Metal prefix).
The entire remaining round-1 gap (~150 ms after word-extract, if it lands)
is AB-prep gather machinery, where the remaining lever is the static-b
partial-load import previously ruled out as bloat.

Zerocheck like-for-like standing: ~1643 vs ~1376 = **1.19x** (was 1.74x
fairly accounted, "2.7x" as first misread). Rounds 3+ (1.03x) and drain+fold
(1.06x) are closed; round 2 (1.45x, ~97 ms, their compact-fold mechanism) is
the largest remaining relative gap.

## Machine-specific tunings: three inversions/nulls on M1

Mechanisms measured good on the challenge tree's M4 that fail on M1:
oblock fold gating (comment claims 1.4-1.7x, measures 0.97-0.99x here),
zerocheck-tail lookahead (default-ON there, loses 7-15% on BOTH machines),
and STNP output stores (+1.2% there, **+12% regression** here -- the
non-temporal hint costs store throughput on M1 instead of saving RFO
traffic). Port memory-system hints only with a local paired measurement.

## Final ST+MT comparison vs the session baseline (lightweight, 2026-08-25)

Single instrumented invocation per cell; untouched phases (witness, commit,
lincheck, open) matched across arms to <1% at both thread counts, validating
the run. All 13 kept optimizations, vs `9793190`:

| phase | ST base | ST head | delta | 8T base | 8T head | delta |
|---|---:|---:|---:|---:|---:|---:|
| zerocheck | 2357 | 1638 | **-30.5%** | 322 | 221 | **-31.4%** |
| -- round 1 | 1465 | 961 | -34% | 191 | 129 | -33% |
| -- round 2 | 437 | 304 | -30% | 59 | 41 | -31% |
| -- rounds 3+ | 454 | 310 | -32% | 65 | 48 | -26% |
| **headline** | 52.5k | **62.5k c/s** | **+18.9%** | 349k | **406k c/s** | **+16.1%** |

The MT columns are the first multithreaded measurement since any optimization
landed: every win transferred to 8T at essentially its ST magnitude, and the
end-to-end gain in the threaded (production/scored) configuration is +16%.

Known anomaly, comparison-safe: the open phase measured ~850 ms ST on BOTH
arms today vs ~636 in earlier sessions -- a day-scale bimodality also seen
once before (the 857 ms lookahead-test reading). Same on both arms, so no
delta is affected; flag for the commit/open campaign.

## Cross-tree comparisons: GPU-status-uncertain (major caveat, 2026-08-25)

The challenge-tree session re-ran its own round-2 measurement (identical
config, same machine, days apart) and got **342 ms where it had reported
215** -- a 127 ms swing it flagged rather than explained away. Its raw dump
shows round-1 samples of 92-95 ms jumping to 309 ms MID-RUN with no code
change. 92 ms is implausibly fast for its CPU drain+fold; it is exactly what
its Metal GPU round-1 prefix produces, and that prefix "fires if the shape
matches and Metal's available" with no warmup latch. The Metal-assist hypothesis was TESTED AND REFUTED as a complete explanation:
with both GPU arms force-disabled (FLOCK_NO_GPU_ZEROCHECK=1 and the separately
gated FLOCK_NO_GPU_ZC_R2=1), their first samples remained chaotic (a 1182 ms
round-2 with no GPU involved). The broader driver: this shared machine ran
builds and benches from three concurrent Claude sessions that day, plus
ambient load. Widen the caveat from GPU-status to: fine-grained (sub-100 ms)
cross-tree bucket deltas from this date are unresolvable, period.

Consequences for this log:
- Every "theirs" column in the cross-tree tables is soft until their GPU
  test reports. Their bracket accounting itself was verified clean (buffer
  takes, tables, and padding all inside their round-2 timer; their tail
  honestly carries the compact-format reconstruction).
- What the stable window of their full-GPU-off run DOES support (samples 3-7,
  tight): their pure-CPU round 2 is 274-285 ms against our same-conditions
  305, i.e. a residual of ~27 ms -- the size of their compact format's
  modeled store saving (~25 ms), the one mechanism deliberately not ported.
  The original "90 ms gap" therefore decomposes as ~25 ms format + ~65 ms
  measurement conditions. Their tail (300-307) matches ours (307) exactly.
  Coarse conclusion that survives all of this: the trees are within ~10% on
  the zerocheck CPU-vs-CPU, and bucket deltas below ~50 ms cannot be
  adjudicated on this machine this week.
- Every KEPT win in this log is unaffected: all were internally paired A/B
  on this tree alone and never depended on their numbers.

## The SHA-256 cross-circuit control

Question: is the challenge repo's remaining advantage BLAKE3 specialization
(its static-b census, degen flags) or generic kernel quality? Control: SHA-256
at 2^16 (m=31), ST, identical command on both trees, where neither side's
structure guards fire at their tuned density.

| tree | best prove_fast | throughput |
|---|---:|---:|
| this branch | 2.05 s | 32,027 h/s |
| challenge (c576e68 frontier) | 1.72 s | 38,170 h/s |

Their advantage on SHA-256: **1.19x** -- statistically the same as the ~1.16x
comparable whole-proof gap on BLAKE3. Conclusion: their edge is uniform,
circuit-agnostic kernel quality (commit path, round-2 compact/NT stores, prep
tail, allocator recycling), and the BLAKE3-specific structure machinery is
performance noise at end-to-end scale on both sides -- consistent with their
own per-switch ablations (1-3% each) and with our ports of that family
(zero-skip -42 ms; b===1 degen null).

Also measured by the control: this branch's campaign improved SHA-256 by
**+18% for free** (27.1k -> 32.0k h/s vs the session-baseline matrix) --
no SHA-specific work was ever done, confirming the kept wins are
circuit-agnostic. The b===1 degeneration port (row 21, reverted) was the
last BLAKE3-structural candidate; with this control there is no reason to
pursue that family further.

## What finally worked, and why

Two round-1 wins landed after eleven failures, and they share a property none
of the failures had: they change **how much work exists** or **how much of it
can be in flight**, rather than how the same work is encoded.

**1. Skip structurally-zero b K-rows (−42.4 ms, 8/8).** A census of the packed
BLAKE3 witness -- 256 word positions per block, 256 blocks, 3 independent
witnesses -- found the circuit pins 38 of 256 8-byte b K-rows regardless of the
inputs, taking only three distinct values:

| value | positions | |
|---|---:|---|
| `0xffffffffffffffff` | 22 (8.6%) | const-one wires |
| `0x0000000000000000` | 15 (5.9%) | structural zeros |
| `0x0001ffffffffffff` | 1 | |

33.7% of all b bytes are fixed, cross-checking the byte histogram (28.9% 0xff,
6.1% zero) from the other direction. The zero case is the strongest: the
inv-NTT transform is F_2-linear, so row(0) = 0, so db = 0, so
y = gf8_mul(da, 0) = 0 and the K-row contributes nothing at all --
`fused_apply_one_k` returns immediately, skipping all 64 table loads and the
four F_8 multiplies. One u64 compare, no census data shipped, no position
tracking, and a disagreeing witness falls through to the generic path.

This is strictly better than the challenge repo's `static_b` fast path, which
still loads a precomputed partial for these rows.

**3. Unreduced pmull accumulate in the prep kernel (−128.0 ms, 8/8) — the
largest single win of the effort, and the challenge repo's own top-attributed
mechanism, ported as an idea (~80 lines).** `gf8_mul_vec16` spent 6 PMULLs per
K-row per block — 2 for the raw product, 4 for a reduction that was redundant,
since the accumulator gets one final reduce anyway. Now the raw product
accumulates unreduced, with the x^K row weight decomposed as x^4 (a pre-scaled
second table image to gather from — F_2-linearity makes scaled entries scale
the XOR-sum) times x^2 (a 6-op byte-mul) times x^(K&1) (a u16 shift). Terms
reach degree 15; both reducers were verified exact over the full 16-bit domain
first (exhaustive tests now permanent in gf2_8). Predicted 100-130 ms from the
challenge session's attribution; measured 128.

**2. Two lanes per drain iteration (−63.1 ms, 8/8).** The drain carries three
XOR chains per lane, each of depth `n_b_med` = 16. The gathers feeding them are
independent but the accumulations are serial, so one lane exposes only three
chains. Interleaving a second doubles that to six with no change in work.

**The all-ones case is the instructive failure.** Predicted ~27 ms from the
zero case's calibration; delivered 6.1. Halving a row's loads halves its
memory-level parallelism at the same time, and the remaining dependency chain
goes latency-bound -- the same mechanism that sank the LD1 x4 attempt. Whole-row
elimination avoids it because no chain survives. That result is what motivated
the lane unroll, which then outperformed the win that inspired it.

**Two censuses that closed leads without any code:**

- `a`-side pinned zeros are *exactly* the same 15 positions as `b`'s (union 15,
  a-only 0) -- the padding rows where both operands vanish. An a-side check
  would add nothing.
- The zero words cluster into the block tail (parity 1, b_med 14-15), giving
  only one fully-zero `(parity, b_med)` group of 32, worth ~1% of drain
  gathers. Whole-`b_med` elimination is not there.

**Combined effect, measured directly.** The two wins together, against the
pre-zero-skip commit in a single session, paired n=8 with alternating arm
order:

  round1 URM  base median 1477.30 ms -> head median 1403.07 ms
              -74.2 ms (-5.0%), head 8/8, ranges disjoint
              (base min 1469.55 > head max 1433.81)

That is less than the 42.4 + 63.1 = 105 ms the individual measurements suggest,
and the combined figure is the one to trust: it is the only one where both arms
ran under the same conditions. The individual runs were taken in different
sessions, and cross-run drift on this machine is large enough to swamp the
difference -- identical code measured 1420 ms in one run and 1483 ms in another.

**Methodology that made these findable.** Earlier attempts were measured on the
end-to-end headline, where a 40 ms effect is ~1% and sits under the noise. These
were measured on `round1 URM` directly via `FLOCK_ZC_TIMING`, taking the min
across the ~5 zerocheck calls in a run, 8 alternating-order pairs per verdict --
about 3x faster per sample and aimed at the phase actually being changed. Note
cross-run drift remains large: the same code measured 1420 ms in one run and
1483 ms in another, so only within-run paired deltas are trustworthy.

## Not attempted, and why

- Anything GPU-gated (`partial_fold_packed_z_best_gpu_split`,
  `ranked_lincheck_fold_gpu_shape`) — out of scope by request.
- The `Round1AbInner` staged pipeline, `c_fold4` mask tables, static-B
  specialization and its 839-line generated kernel. This is where the round-1
  win actually lives, but it is ~6000 production lines across
  `univariate_skip_optimized.rs`, its NEON kernels, and `zerocheck.rs`.
- `build_eq_table_optimized` in lincheck. The geometric-medium trick is
  prototyped in `benches/eq_build_probe.rs` but never landed; lincheck is only
  3.9% of ST here and its cross-repo gap is partly re-attribution (see above),
  so it was not the best next move.

## The cross-repo round-1 comparison was never like-for-like

This is the most important correction in this log. The challenge repo's
round-1 figure **excludes its AB precompute**. `commit_with_round1_ab_precompute`
in its `prover.rs` runs

    rayon::join(commit_arm, precompute_ab_arm)

so `precompute_round1_ab_inner_packed_padded` — the same
`shift_reduce_inner_ab` work that is 56% of our round 1 — lands in its *commit*
bucket, and its `t.commit_s` wraps the whole join. Comparing their 314 ms
round 1 against our 1444 ms was comparing a drain against a prep-plus-drain.
Combined:

| phase | ours ST | theirs ST |
|---|---:|---:|
| commit | 1277.4 | 1796.7 |
| zc round 1 | 1444.3 | 314.1 |
| **commit + round 1** | **2721.7** | **2110.7** |

The honest gap is ~611 ms, not ~1130 ms. An earlier claim in this log that
"our commit is already ~1.4x faster than theirs" was wrong for the same
reason — they do strictly more work in that phase.

We implemented the same architecture to check whether it is a speedup or an
accounting choice, and it is the latter. The transform really is
challenge-independent (the challenge reaches round 1 only via `eq_lo_scaled`
and the convert table, both owned by the drain), a
`[x_outer][b_med][64]` buffer lets the drain consume it by borrowing with no
copy, and the result is bit-identical. But:

- **ST is a wash.** `rayon::join` is sequential on one thread, so the locality
  won by no longer interleaving AB with the C transpose and the 64 KB
  convert-table drain is spent again on a 512 MiB write plus 512 MiB read the
  interleaved version never did. Non-temporal `stnp` stores, which skip the
  read-for-ownership on that write-once surface, did not change it.
- **8T is a wash.** Both join arms compete for the same saturated pool, so
  there is no idle capacity for the overlap to fill.

Measured three ways (ST paired n=8 order-alternating with NT stores: base
53288 vs 53037, base 5/8; 8T paired n=6: 325007 vs 320655, base 4/6; ST
paired n=10 without NT stores: indistinguishable once warm). Reverted.

**Methodology note worth keeping: check the power source before measuring.**
Two grand-total runs were invalidated in one evening by power state. On low
battery, macOS caps frequencies (one base arm carried a sample ~20% low); while
fast-charging a nearly-empty battery, it is even worse -- the charger's power
budget is shared with the SoC and a paired run swung -13% to +57% per pair
with a 51% base-arm spread. Run `pmset -g batt` first: measure only on AC with
the battery above ~60%, where the charge rate has tapered. The alternating
paired design protects the sign test through slow drift (arms sit within ~80 s
of each other), and min-of-5 within a run rejects transient dips -- which is
how the round-1 results cross-validated and survived -- but end-to-end
magnitudes from a throttled window are unusable.

**Methodology note worth keeping.** An earlier version of the paired script
always ran the base arm first. Throughput declines monotonically across a run
as the machine heats (342707 → 315100 over six 8T pairs), so a fixed order
biases the comparison by roughly the size of the effect being measured. Always
alternate which arm runs first, and discard a warm-up of each arm — an
apparent +3.3% win for the hoist evaporated once both were done.

## The idea behind their `accumulate_convert` win

Worth recording even though it did not transplant, because the algebra is the
interesting part. Their C-side drain does **no table gathers at all**.

`convert[b][v] = γ^b · φ_8(v)`, and **γ = X** — the comments confirm it, the
rows are built by `mul_by_x` doubling. So `Σ_b γ^b · (bit_b)` is *literally* a
16-bit mask in the field's coefficient representation: `F128 { lo: mask }`.
Since `φ_8` is F2-linear, the per-lane C contribution decomposes over the 8 bit
positions of the byte into 8 such masks — the "eight-bank C drain" — each
accumulated with pure bit operations. The only field work left is one multiply
by `eq`, and `F128 { lo: m } * eq` is itself F2-linear in the mask's 16 bits,
so even that becomes `T_lo[m & 0xff] + T_hi[m >> 8]` from tables built **once
per prove** and shared read-only (8 MiB at their shape; building per call would
cost ~4 GiB of L1 stores).

Why it does not transplant on its own: profitability depends on data layout,
not just the algebra. Building the masks needs one lane's bytes gathered
*across* `b_med`, but `chunk_c_bytes` is `[b_med][lane]`, so extracting them
costs exactly the 16 strided loads the trick was meant to remove. Their
pipeline gets the transposed layout for free from the `Round1AbInner`
precompute pass. The AB side is handled separately by the tensor split
`eq.lo[(w << s) | u] · D^-1 == eq_top_scaled[w] · eq_bot[u]`, pre-scaling the
convert tables by `eq_top` so the inner loop is pure XOR with no per-lane
multiply, keeping `2^s` bank accumulators and applying `eq_bot` once at the
end.

Any future round-1 attempt should start from the layout, not the algebra.

## Note on the geometric trick

The three layered optimizations in `univariate_skip_optimized.rs` — geometric
small-eq + shift_reduce, geometric medium-eq + 64 KB convert-table lookups, and
D^-1 absorbed into eq_lo — are **already in this tree**; the doc headers are
byte-identical to the challenge repo's. They are inherited upstream code, not a
Yukon addition, and nothing there needs porting.

## Commit phase, ST: decomposition and the refuted butterfly rewrite (2026-08-25)

Commit-phase breakdown at m=32, single-threaded (`FLOCK_COMMIT_M=32
FLOCK_NTT_SPLIT=1 RAYON_NUM_THREADS=1`, bench `pcs_commit`):
alloc/pad ~90 ms (prefault-hidden in the real prover), **NTT 858 ms** (top 9
fused-2 layers 329 ms, deep 11 blocked layers 529 ms), **merkle 417 ms** —
merkle is at the SHA-256 silicon floor (~2.6 GB/s) and closed.

**Refuted experiment — "q-resident 3-PMULL karatsuba butterflies"** (reverted,
this commit). The premise was a misdiagnosis: `butterfly_row_pair` /
`butterfly_fused_2layer` have no aarch64 dispatch arm, so I read the portable
fallback as "scalar, 6 PMULLs through the struct/GPR interface". Wrong — the
portable butterfly is generic over `F128` ops, and `F128::mul` on aarch64
inlines `ghash_mul_binius` (gf2_128.rs:104-115, where the comment records that
M-series picked binius over karatsuba). The baseline was already running the
M1-tuned mul, register-resident after inlining.

Measured, same session, same machine state, m=32 ST:

| arm | top 9 | deep 11 | NTT total |
|---|---|---|---|
| baseline (binius via generic path) | 328.6 ms | 529.5 ms | 858 ms |
| karatsuba + q-resident kernels, GPR half-sums | 540 ms | 873 ms | 1410 ms |
| same, half-sums moved to NEON (veor+vext) | 525 ms | 835 ms | 1360 ms |

+58% regression, reproduced across two runs pre-fix and confirmed post-fix;
the GPR-vs-NEON sum was worth only ~4 points of the 58. Fewer PMULLs lost to
binius's shape: karatsuba's mid-term chain plus a per-butterfly vzip/reduce/
vunzip repack costs more than the three PMULLs it saves. This is the sixth
confirmation that re-encoding fixed work never pays on this machine (0/6), and
it extends the rule to PMULL count itself: **binius's 6-PMULL mul beats
3-PMULL karatsuba in situ on M1, not just in the latency microbench**.

Kept from the episode: the `FLOCK_COMMIT_M` bench knob and the temporary
`FLOCK_NTT_SPLIT` probe (strip the probe when the commit campaign closes).
Remaining commit-ST headroom candidates, unmeasured: aarch64 fused-4 for the
top layers (`fused4_ok` is currently x86-only; top layers are full-buffer
sweeps, so deeper fusion removes memory passes — the win category with the
best track record), and nothing else obvious; cross-tree, our commit was
already at parity or ahead.

## Cross-tree commit is measured parity, not inferred (2026-08-25)

The earlier claims ("our commit is 1.4x faster", later corrected, then
"parity or ahead") all came from in-prover bucket timers, which are
confounded: their `commit_s` wraps a `rayon::join` that includes their
round-1 AB precompute. Today: direct primitive-level A/B, m=31 packed
breakdown, ST, alternating arms, 3 pairs, same minute, both trees' own
`pcs_commit` bench (byte-identical bench code; theirs got the same
FLOCK_COMMIT_M knob temporarily and was restored after; both merkle
defaults are SHA-256; FLOCK_NO_GPU_COMMIT=1 on their arm):

| arm | NTT (3 runs) | merkle | total |
|---|---|---|---|
| ours | 418 / 421 / 419 ms | 212.8 / 212.6 / 212.4 | 692 / 692 / 690 |
| theirs (c576e68) | 422 / 437 / 420 ms | 212.2 / 213.8 / 212.2 | 690 / 706 / 684 |

Identical to within ~1% on every bucket. Their NTT source files DO differ
from ours (5 files), so this is a measured null, not shared code: whatever
they changed there is performance-neutral at this shape, and there is no
commit-phase port pool. Caveats: run on battery power with an active Zoom
call (a real >15% gap could not hide in data this tight, but treat the
third decimal as weather); and their tree has a Metal `gpu_commit.rs` path
we did not exercise — CPU-vs-CPU is parity, GPU-on is untested and out of
scope for this campaign.

## Open campaign, ST (2026-08-25): the combine port, and a full two-tree reconciliation

Protocol parity first: both trees produce byte-identical proofs at n=65536
BLAKE3 (395,919 bytes) with identical verify times, so open comparisons are
clean. ST decomposition (PCS_TRACE + LIG_PROVE_TRACE / their
FLOCK_OPEN_TIMING, same day, same machine):

| sub-phase | ours (pre) | theirs | ours (post-port) |
|---|---:|---:|---:|
| combine (b_combined fold+prime) | 94.4 | 64.4 | **78.5–79.4** |
| initial sumcheck | ~36 | 44.6 | ~36 |
| recursive commits (NTT+merkle) | 19.0 | 14.9 | 19.0 |
| induce_sumcheck_poly | 7.5 | 5.6 | 7.5 |
| ring_switch / folds / glue / OOD | ~3.7 | ~3.7 | ~3.7 |
| **open TOTAL** | **~160** | **132.7** | **~145** |

**The port (commit 6baccea): composed-table fold.** `fold_one_slot(·, T)` is
F₂-linear, so `lo ↦ fold_one_slot(lo·e_hi, T)` collapses into one composed
byte table per claim per block (x-ladder monomial walk + subset-sum
doubling), deleting the per-slot field multiply — 2·L muls — from the sweep.
Needs the coarse deferred split (eq_lo 2^15, not the balanced 2^11) so the
~4.3k-op build amortizes. Bit-identical (equivalence test + verify). Their
tree had both pieces; they never A/B'd it as a unit — it predates their
session. Prediction was −30 ms; measured −15 ms, and the sub-timer probe
explains the rest (below). MT: combine 10.9 ms at 8T (near-linear transfer).

**Corrected accounting (in-fold sub-timers + open_combine_probe micro).**
Sweeps alone: ours compose 1.2 + sweep0 22.7 + sweep1 24.5 = 48.4 ms —
EXACTLY their sweep cost (64.4 bucket − ~16 prime). The remaining bucket
difference is the tail pass: our fused prime+round-1-lookahead costs 24.5 ms
vs their plain prime ~16 ms, and the lookahead buys ~12 ms back in initial
sumcheck (36 vs 44.6). Tail+initial: ours 60.5, theirs 60.6 — **the
lookahead placement is a wash**, another instance of "moving fixed work
between buckets is not a speedup." It stays only because it is inherited
code (zero new lines to keep). Earlier probes that "showed lookahead free"
were wrong: LIG_LOOKAHEAD_DISABLE only gates the ligerito consumer, never
the combine's producer pass.

**Nulls, measured.** (1) EOR3 depth-3 fold tree: flat (79.5 vs 79.1) — LLVM
already fuses XOR pairs into EOR3 under target-cpu=native, and the sweep is
load-port bound (32 loads/slot). Not kept. (2) Fusing both claims into one
sweep (two live 64 KiB composed tables, single store, no RMW read-back):
55.6 vs 48.4 ms in the micro — the doubled gather footprint thrashes L1.
Validates the claim-sequential design note in the challenge tree.

**Residual vs theirs, and why it is closed for now**: ~6 ms structural
(recursive commits 19 vs 15, induce 7.5 vs 5.6) comes from their
sparse/windowed transpose-NTT + truncated-final-NTT machinery — thousands
of lines whose four kill-switches all measured null at this shape in their
own tree (peer-session test; several gate on their ranked 2^18 shape and
cannot fire here). Fails the bloat bar decisively at ~6 ms. The remaining
~7 ms is unattributed noise; Fiat-Shamir grinding lives inside "initial
sumcheck" and swings 1.8–8.4 ms per sample (their measurement, same bucket
convention in both trees).

Instrumentation kept (strip at campaign close): `open_combine_probe` bench +
`pcs::combine_probe` module, and the `b=` field in the combine trace line.
Conditions caveat: battery power + active Zoom all afternoon; every kept
number is an internal same-run comparison or reproduced across ≥3 samples.

**Addendum (peer measurement, same day): their truncated-final-NTT is a real
null even at its designed shape.** At the ranked 2^18 config (the exact shape
`is_ranked_induce_truncated_final_ntt_shape` pins: log_msg_cols=19,
n_queries=218), 7 paired ST samples with the production switch
`FLOCK_NO_LIG_INDUCE_TRUNCATED_NTT`: lig-prove 132.39 ms ON vs 132.79 OFF,
induce 12.23 vs 11.80 — flat. So the truncation contributes nothing anywhere;
whatever induce/commit edge their tree holds (~6 ms at our shape) rides on
the always-on sparse transpose-NTT, not this. The not-ported decision stands
with their own numbers behind it. Their run also showed the familiar
environmental spike (two trailing samples at ~2× on BOTH arms identically) —
same all-week pattern, comparison-safe, logged for the record.

## Witness gen, ST: streamed full-write builder (2026-08-25, kept)

Focused bench (`genwitness_phase`, n=65536, m=30): ours 95.9 ms best /
123 avg; their default 61.4/62.4; their scalar path (FLOCK_NO_WITGEN_SIMD=1)
73.8/76.1. So their edge decomposed as: streamed full-write + unrolled Gs
(−22 ms) then SIMD quad lockstep (−12 ms more).

Ported the first, natively: three `PackedWordWriter`s publish complete u64s
sequentially (rows are contiguous through USEFUL_BITS), killing BOTH the
driver's per-group memset and the OR path's read-modify-write on every
store; the 56-G sequence unrolls with literal state/message indices. The
one out-of-order region (out_lo, 256-bit aligned) is reserved and
overwritten at the end. Bit-identical through the whole driver (new test:
real + padding slots, both values of the prefix carry bit).

Paired alternating A/B, best-of-12 per invocation: **ST 87.1–90.0 →
64.8–68.9 ms (−24%, 3/3 disjoint ranges); 8T 18.0–18.3 → 14.4–15.8 ms
(−20%, 2/2)**. Avg variance also fell (113 → 88 ms ST). In-prove witness
bucket: 89.5 → 67.3 ms. Our streamed scalar now BEATS their scalar (73.8);
their remaining SIMD-quad edge is ~3–7 ms ST for ~400+ lines of NEON
lockstep + NT-drain + scratch-provenance machinery — fails the bloat bar.

## Re-baseline: full ST cross-tree table after the open + witness ports

Same day, same machine, GPU off on their arm (their Aug 22 binary),
n=65536: ours witness 67.3 / commit 300.8 / zerocheck 396.9 / lincheck
56.4 / open 146.7 / **total 967.4 ms (67.7k comp/s)**; theirs 30.4 /
424.8 / 217.3 / 42.2 / 132.7 / **total 847.1 ms (77.4k comp/s)**. Headline
gap **1.46× (start of day) → 1.14×**. Two buckets remain confounded, both
in their favor's appearance only: their commit carries their round-1 AB
prep (known since the zerocheck campaign) — commit+zerocheck combined is
697.7 vs 642.1 (1.09×, matching the ~10% kernel-quality verdict) — and
their in-prove witness bucket (30.4) is HALF their own focused bench
(61.4), so something (seed-pipe speculation or the rate2-codeword fusion)
moves witness work out of that bucket; under investigation. Honest
remaining real gaps: lincheck 1.34×, their witness accounting, ~9% kernel
quality in zerocheck+commit.

## Lincheck: two-stripe word-load fold reaches the load floor (2026-08-25, kept)

The prior "no reachable headroom" verdict on the lincheck fold examined
blocking strategies and table geometry; the challenge tree's newer asm
kernel wins differently: the inner loop is load-port bound, and their
kernel grabs each stripe's 8 index bytes as ONE paired load (UBFX
extracts) while folding two stripes per iteration with EOR3. Ported the
idea as intrinsics (u64 load + shift extraction, two stripes per
iteration, XOR pairs LLVM fuses to EOR3): 16 loads/stripe -> ~9,
bit-identical XOR multiset. Paired A/B: **partial_fold_z ST 40.7-40.8 ->
27.8-28.0 ms (-32%, 3/3 disjoint, at the ~28 ms computed floor); 8T
6.0-6.1 -> 4.3-4.4 ms (-28%)**. Lincheck bucket 56.4 -> 42.6 ST — parity
with theirs (42.2). One measurement-hygiene note for the record: two
paired runs were invalidated before the real one — a stale-binary
overwrite refusal (aliased interactive cp) and a stash left behind by a
failed && chain built both arms from the same source; both caught by the
disjoint-range check and a binary cmp before trusting any numbers.

## End-of-day cumulative (n=65536, m=30): 1.46x -> ~1.10x

ST: witness 71.6 / commit 300.7 / zerocheck 379.7 / lincheck 42.6 /
open 144.6 — **936 ms, 70.0k comp/s**. 8T: 152.9 ms, **428.7k comp/s**.
Vs their same-day CPU-only 847 ms ST: 1.10x, with their commit+zerocheck
bucket confounds unwound this is within the ~9% uniform-kernel-quality
band established by the SHA-256 control. Today's three kept ports:
composed-table open fold (-15 ms), streamed witness builder (-22 ms),
two-stripe lincheck fold (-13 ms) — ~50 ms ST total, all bit-identical,
all paired-decisive, all transferring to 8T.

## Round 1, final pass: the "bigish gap" was mostly an estimation error (2026-08-25)

Skip-arm probes (FLOCK_R1_SKIP_PREP / _DRAIN, since stripped; the stripe-fold
timer under FLOCK_ZC_TIMING was kept) split our round 1 at m=30 ST:
**AB prep 160 + AB drain 39 + stripe fold 28 ≈ 230 ms** — the stripe fold
already carries today's two-stripe lincheck kernel (was ~41).

The peer session then measured their prep arm directly (their
FLOCK_PHASE_TIMING probe inside the commit rayon::join, 7 ST samples,
GPU off): **~140 ms**, not the ~105–125 my commit-bucket subtraction
estimated. Corrected comparison:

| piece | ours | theirs |
|---|---:|---:|
| AB prep | 160 | ~140 (measured) |
| drain + fold | 67 | 72.5 |
| **round 1 total** | **~230** | **~213** |

So round 1 is ~7% apart, we are AHEAD on drain+fold, and the prep delta is
12.5%, not 40%. Their prep mechanism (their read): `fused_apply_one_k_fast`
— identical gather structure, but unreduced PMULL/Horner accumulation with
one fused BCAX reduction per step instead of a full reduced GF(2^8)
multiply per K-row. Arithmetic-only; our multiply tail is ~17 ms of the
160 (gathers ~90%), so the port ceiling is ~8–15 ms for ~100 lines of
kernel restructure — below the bloat bar. Their other two prep levers
(DIRECT_AB_ROWS zero-copy views, AB_COMPACT_STORE) address the
materialize-then-read-back architecture ours doesn't have: our prep is
fused into the drain and never writes the 128 MB buffer at all.

**Round-1 verdict, this time with both sides measured: closed.** The
remaining zerocheck delta decomposes as r1 arithmetic ~8–15 (priced, not
taken), r2 compact format ~7–10 (priced, not taken), tail parity.

## Official end-state grid: clean-conditions, no-timer, both trees (2026-08-25 night)

Machine quieted to just the two Claude sessions, AC power, 100% charged.
15 runs: 3 per config, interleaved between trees, bare `blake3_proof`
n=65536 (no instrumentation env), best-of-3 proves per run. Ours at HEAD;
theirs the Aug 22 binary at c576e68.

| config | ours (best, spread) | theirs (best, spread) | gap |
|---|---|---|---|
| ST CPU | 930.4 ms / 70.4k c/s (0.3%) | 843.7 ms / 77.7k c/s (0.9%) | 1.10x |
| 8T CPU | 156.1 ms / 419.8k c/s (0.4%) | 131.3 ms / 499.3k c/s (2.0%) | 1.19x |
| 8T GPU-on | — (no GPU path) | 131.0 ms / 500.3k c/s (2.3%) | 1.19x |

Findings: (1) no-timer headlines match the instrumented runs within noise —
instrumentation overhead confirmed ~zero, all bucket analyses stand;
(2) run spread at 0.3–0.9 % ST confirms every prior "day-mode"/spike
anomaly was ambient load, not code; (3) their GPU is worth nothing at 8T
(131.0 vs 131.3) — their CPU path caught up to their own Metal offload;
(4) the MT gap is 1.19x under clean conditions (1.23x on battery), and it
is scheduling (helper threads / epool P+E / allocator recycling), not
kernels — ST stands at 1.10x with every bucket at parity or priced.

## Their GPU, resolved: an ST-only, ranked-shape-only effect (2026-08-25 night)

The clean grid showed GPU-on worth ~nothing at n=65536, contradicting the
campaign-era "+10.7% ST" table. Both were right — different cells. Full
GPU value map (their Aug 22 binary, clean machine, AC, paired same-minute):

| shape / threads | GPU-on | GPU-off | GPU worth |
|---|---:|---:|---:|
| m=30 ST | 832 ms | 847 ms | +1.7% |
| m=30 8T | 131.0 ms | 131.3 ms | 0 |
| m=32 ST | 2.63 s | 2.87 s | **+9.2%** |
| m=32 8T | 413.5 ms | 406.1 ms | −1.8% |

Mechanism: their heavy offloads are shape-pinned to the ranked m=32
geometry (dormant at m=30 — Metal initializes but the cpu= telemetry shows
all threads busy doing the work), and the GPU only adds value when the CPU
is starved (ST). At 8T the CPU saturates the same memory system, the GPU
graph "finishes with 0.00 ms host wait" (their comment), and sync overhead
turns it slightly negative. In the threaded production configuration the
GPU is worth nothing on either shape; the +10.7% campaign figure was the
m=32 ST cell (reproduced tonight at +9.2%), not a general advantage.

## CORRECTION + the ranked-config picture: the GPU verdict was a config artifact (2026-08-25 late)

**Retraction.** The "their GPU is worth nothing threaded" section above was
measured with the SHA-256 merkle default — which silently fails their
ranked GPU gates (`merkle_hash == Blake3` is a hard condition on the big
offload paths). The user's suspicion that "a flag needed to be turned on"
was correct: with FLOCK_MERKLE_HASH=blake3 at m=32, their GPU is worth
**+35%** at 8T on this M1 Max. The +9.2% ST figure earlier is also
understated for the same reason.

Ranked-config grid (m=32, this machine, clean, same half-hour):

| m=32 8T | ours | theirs |
|---|---:|---:|
| SHA merkle, CPU | 433.6k c/s | 645.6k c/s |
| Blake3 merkle, CPU | 410.3k (no fast blake3-merkle kernel here) | 663.4k |
| Blake3 merkle, GPU 8T | — | 897.9k |
| Blake3 merkle, GPU 10T | — | **942.7k** |

Consequences:
1. The MT gap is SHAPE-DEPENDENT: 1.19x at m=30 but **1.49x at m=32
   CPU-vs-CPU** — their scheduling stack (seed-pipe, epool, allocator
   recycling, AB-prep overlap) is gated on the ranked m=32 geometry and
   never fired in the m=30 comparisons. Their throughput scales +29%
   from m=30 to m=32; ours +3%.
2. Full scored-config gap on this machine: 433.6k vs 942.7k = **2.17x**
   (their reported 600k/900k reproduced here as 663k CPU / 898-943k GPU).
3. The ST kernel campaign remains validly closed (1.10x, same-hash,
   same-shape); what it never measured is the ranked-config stack:
   MT scheduling at m=32, GPU offload behind the Blake3 gate, 10-thread
   epool, and a fast BLAKE3 merkle kernel. Those are the remaining
   campaign, in descending order of measured value.

**10-thread addendum (2026-08-25 late).** All-core (8P+2E) CPU-only runs:
ours m=30 414.3k (−1.3% vs 8T) / m=32 443.1k (+2.2%); theirs m=30 502.1k
(+0.6%) / m=32 595.3k (−7.8% vs their 8T). E-cores are ~worthless for
CPU-only proving on both trees — their 10T mode only pays with the GPU
overlap (943k). Best-CPU-vs-best-CPU at the ranked shape: 443.1k vs
645.6k = **1.46×**, all scheduling, not thread count. Ours reproduced to
5 digits across runs (414,339 vs 414,337 c/s).

## BLAKE3 merkle: the neon8 idea in 290 intrinsics lines (2026-08-25, kept)

Their blake3 merkle edge is a 2.6k-line generated-asm 8-wide kernel; the
mechanism is just ILP (the crate's 4-wide NEON state is latency-bound on
the G chain). Re-derived as intrinsics: two transposed 4-wide states
interleaved G-for-G, dispatched from blake3_hash_many for groups of 8,
crate path for tails, bit-identical by equivalence test.

merkle_tree ST: blake3 1.63→2.30 GB/s at 512 B leaves (+41%, now 1.08×
faster than SHA-256), 1.71→2.42 GB/s at the ranked 1 KB leaves (+42%,
parity with SHA silicon; their asm ≈2.58, i.e. within 6% for 9× fewer
lines). E2E ranked config m=32 8T blake3-merkle: 410.3k → 431.5k c/s —
the −5% blake3 penalty is erased and the ranked hash choice is now free
for this tree. LLVM handled the 32-register pressure without measurable
spill cost; the asm fallback (their .S) was not needed.

## MT campaign, night 1 (2026-08-26): one keeper, seven nulls, a map of what's left

Target (user directive): CPU-only MT within 10% of the challenge tree,
no GPU. Start: 1.19× at m=30 8T.

**Kept — AB hoist v2 (commit d963445):** prep under the commit via
rayon::join, with the two defects that nulled v1 fixed: the ab_pre buffer
comes from the scratch pool uninitialized (fresh vec![0u8] zero+fault cost
was eating the entire gain) and the join window runs on the all-core (P+E)
pool while the rest of the prove stays on P-cores. The E-cores are the
active ingredient: prep is gather/PMULL compute they can add without
stealing the DRAM bandwidth the NTT saturates. Paired 3/3 at m=30
(149.6–151.3 vs 152.5–156.3), 2/2 at m=32 (best clean pair −57 ms).
Best production number: 149.6 ms / 438.1k c/s.

**Nulls/inversions, all paired, all on this M1 Max:** P-pool-only join
(wash, third confirmation); NT stores in witness writers (~0 — M1 ignores
the stnp hint, third confirmation of the model); lincheck-stripe transpose
on E during commit (NEGATIVE: bandwidth task in a bandwidth-bound window,
commit +7 ms); FLOCK_ALLCORE combine (0); NTT fused-4 on aarch64 (+19–26%,
16 live F128s spill — the old code comment was right); two-block scalar
witgen interleave (+40% ST, GPR blowout); quad-lite SIMD witgen (state
math 4-wide, scalar packing — null even after removing 4.5k lane
extractions: the packing is the cost, not the G math).

**Decisive ablations on their tree (same day):** their witgen SIMD is
worth 2× IN-PROVE (24.9→12.1 at 8T) but their SIMD-without-elision
(16.7) ≈ our streamed scalar (16.2) — i.e. the entire remaining witness
gap is their scratch-provenance CONSTANT-REGION ELISION (−4.6 ms their
tree; ceiling probe on ours: −4.8 ms paired, degraded-machine caveat).
The focused genwitness bench measures only their scalar path (the SIMD
gate lives in their prove method), which earlier mislead this log.

Standing at checkpoint: ~1.15× at m=30 (149.6–152.4 vs 130.5–130.9
same-minute). Queued with measured ceilings: witgen constant-region
elision via pool provenance tags (−3..5 ms, ~120 lines), zerocheck
round-2 compact format (−2 ms, ~150 lines, previously priced). Those two
land ≈1.10–1.12×; anything past that is their 2.6k-line lane-wise
vectorized packing. Measurements paused: machine degraded after ~6 h of
continuous benching (witness bench 14.3→37.9 ms both arms) — resume
after cooldown per the discipline.

## MT campaign, morning session (2026-08-26): elision kept; the last 4–6 ms priced

Post-cooldown conditions verified (witness bench back to 14.2 ms from the
degraded 37.9). **Kept — witness constant-region elision (4dc8742):**
scratch-pool provenance tags, derived independently at give (from BlockR1cs)
and take (from the encoder constants), gate skipping b's MAX prefix /
reserved words and all three zero tails on a hit; any other custody clears
the tag. Paired kill-switch A/B, production config: 4/5, mean −2.5 ms,
best 146.98 ms / 445.9k c/s. Byte-identity test drives a tagged give/take
cycle vs a fresh run. (The −4.8 ms ceiling probe from last night was
degraded-machine-inflated; −2.5 is the clean value, in line with their
−4.6 on the larger constant share their layout elides.)

**Final standing, same-minute paired:** m=30 production 149.3–150.6 vs
their 130.2–130.9 = **1.146×** (campaign start 1.19×; best single run
146.98). m=32: ours 572.6 (457.8k, +5.6% on the shape since yesterday) vs
theirs 394.9 = 1.45× — the ranked-shape residual is their m=32-gated
machinery.

**The remaining ~4–6 ms at m=30, priced:** (1) compact round-2
anchor+delta (−1.8 ms @MT measured from their r2 8.9 vs ours 10.7) — ~800
lines in their tree, resurfaces through our three tuned r2/r3 kernels,
worst lines-per-ms of the campaign; their newer symbolic lookahead+cascade
(rounds 3+4 and 5+6 collapsed into earlier passes) sits on top of it,
m=32-gated there, several hundred more lines. (2) Their generated
lane-wise vectorized packing network (−2–3 ms; our quad-lite probe
confirmed the packing, not the hash math, is the cost — vectorizing it is
their 2.6k-line codegen). Both exceed the standing bloat bar; parked for
an explicit call rather than taken unilaterally.

## The m=32 shape, decomposed and probed (2026-08-26)

First m=32 bucket decomposition, both trees (8T-class, same minute):
ours witness 71.2 / commit 260.3 / zc 131.2 / lincheck 27.9 / open 93.6
(sum 584, best total 572.6); theirs 19.2 / 271.6 / 142.4 / 15.9 / 84.9
(sum 534, best total **394.9** — 139 ms of phase OVERLAP that exists only
at m=32, where their ranked stack's gates open). Notably our commit AND
zerocheck buckets are BETTER than theirs at m=32 — the kernel campaign
transferred; the 1.45× lives in witness (−52: their m==32-gated deferred
stripe + witgen hetero drains), lincheck (−12: their round-1 stripe-fold
reuse), open (−9), and the wholesale pipeline overlap.

**Probed and rejected:** deferring our lincheck stripe into the commit's
all-core join window as a third arm, gated to n_blocks_log ≥ 17 —
NEGATIVE 3/3 at m=32 (628.9–641.6 on vs 616.4–633.8 off). Their own
m==32 gate on the same idea works only inside their epool/GPU-window
architecture; re-streaming 512 MB from DRAM into our already-saturated
join window loses to the L1-fused eager transpose both at m=30 (measured
earlier) and m=32. Reverted.

m=32 conclusion: closing it means porting the pipeline architecture
(phase-overlap scheduling), not any single mechanism — same class of
decision as the r2 complex and the packing network. Parked with the rest.

**Correction to the m=32 entry above (blake arms).** The 1.45× figure
compared SHA-merkle arms — which silently disables the blake-gated half of
their ranked stack (the deferred stripe requires HashKind::Blake3, and the
witness attribution above is accordingly wrong: that gate was closed in the
SHA runs). With blake arms (their true ranked config, CPU verified via
util telemetry): theirs 920–950k c/s vs our best (SHA) 457.8k —
**~2.1× at the ranked shape**. What the blake gates open, bucket-level:
their lincheck 15.9→7.6 ms, open 84.9→21.9 ms, plus the deferred-stripe
witness path. Their dev-bench blake-CPU (950k) also exceeds their
worker-scored GPU-off number (630.8k), so ranked scoring overhead is
large; cross-methodology caution applies. Conclusion unchanged in kind
but bigger in degree: the m=32 gap is the integrated blake+m32-gated
pipeline architecture, a deliberate port-project, not a mechanism list.

**RETRACTION of the blake-arms correction above.** The 920–1036k "CPU-only"
figures were GPU-contaminated: the challenge tree's GPU merkle paths
(recursive merkle, L1 overlap) are BLAKE3-only — GPU shaders hash blake,
not SHA — and sat outside the kill-switch list used here; and the GPU-
utilization "verification" sampled only the bench's 3-minute setup window,
killing the process before any timed prove ran (worthless both arms). The
tree's owner reproduced with an airtight fresh-build kill: **their true
CPU-only ceiling at m=32+blake(+blake FS, the worker's hardcoded config)
is 638.9k c/s — consistent with their worker-scored GPU-off 630.8k**,
which is the cross-check that settles it. Their GPU at the ranked config
is worth +57% dev-warm / +43% scored.

Corrected m=32 standing: ours 457.8k (SHA; blake costs us ~4%) vs theirs
639k CPU-only = **1.40×** — in line with the original SHA-arms 1.45×, so
the earlier pipeline-architecture analysis stands as written; the "2.1×"
interlude is void. Also noted from the owner: the ranked worker hardcodes
Blake3 for BOTH merkle and Fiat–Shamir (no env); dev-bench defaults are
SHA — worth +3.7% on their tree; our FS hash config at the ranked point
is an open item. Lesson for the log: a kill-switch list is only as
airtight as the tree's owner says it is, and GPU telemetry must bracket
the timed region, not the process.

## Repo default switched to Blake3 (merkle + Fiat–Shamir), 2026-08-26

Matches the ranked worker's hardcoded config (surfaced by the peer session:
BENCHMARK_HASH = Blake3 for both, no env). HashKind::default(),
FsChallenger::new, and all embedded ligerito TOMLs flipped; SHA-256 remains
selectable per component and the cross-hash tests now exercise it as the
non-default arm. Same-binary A/B: blake-vs-sha −1% at m=30, +2.3% at m=32
— neutral-to-positive thanks to the neon8 merkle kernel. All future
default-config numbers are now at the scored hash point; historical log
entries above used SHA defaults unless marked otherwise.

## m=32 blake, CPU-vs-CPU, finally clean (2026-08-26) — and the real root cause

**The contamination mechanism was a shell bug, not a switch-coverage hole:**
`GPUOFF="A=1 B=1 ..."` as a STRING does not word-split in zsh, so
`env $GPUOFF cmd` set one garbage variable and none of the kill switches —
verified by a 1 s GPU trace showing 94–98% utilization through an "all-off"
run. The array form (used by the original official grid) and inline env
lists were always valid; every blake "GPU-off" cell used the string form.
The earlier "blake-gated GPU-merkle paths escaped the kill list" hypothesis
is withdrawn — the switches were simply never delivered. (Same zsh footgun
as the `for arm in $order` loop earlier in this campaign; now twice bitten.)

**Clean paired comparison, m=32, Blake3 merkle+FS (the ranked hash point),
CPU-only both arms, inline envs:**

| pair | ours (default config, prod) | theirs (8T, all kills) | ratio |
|---|---:|---:|---:|
| 1 | 592.3 ms / 442.6k c/s | 392.1 ms / 668.6k c/s | 1.51× |
| 2 | 629.0 ms / 416.8k c/s | 413.4 ms / 634.1k c/s | 1.52× |

Their arm agrees with the owner's airtight fresh-build (638.9k) and their
worker-scored GPU-off (630.8k) — three independent methodologies within
5%. **Verified standing at the ranked shape and hash: 1.51× CPU-only.**
(Slightly above the SHA-arms 1.40–1.45× because the blake hash point
benefits their tree ~4% and ours ~2%.) The composition of that gap is the
previously-logged one: their m=32-gated pipeline (phase overlap, deferred
stripe, round-collapsing r2 complex) plus their blake-tuned kernels.

## The "pipeline architecture" was an accounting mirage (2026-08-26)

Stage 1 of the pipeline port (Merkle leaf hashing fused into the NTT deep
pass's sub-group tasks, ~200 lines, bit-identical, kill-switched) measured
3/5 pairs, mean −0.6% at m=32 — a null (the deep pass is PMULL-compute-
bound, so hash compute doesn't ride stalls; only the codeword re-read
saving survives). REVERTED.

The null prompted re-examining the "139 ms of phase overlap" that motivated
the pipeline theory — and it dissolves: their multi-run breakdown
(BLAKE3_BREAKDOWN_RUNS) shows per-run buckets summing to ~471 ms against
~400 ms headline runs, which is exactly their prove_fast_TIMED wrapper
being ~15% slower than the untimed path (documented in week one and
forgotten). Their buckets are self-consistent within the timed path; the
"overlap" was timed-buckets-vs-untimed-best. WITHDRAWN.

**The real m=32 (blake, CPU-only) gap, timed-vs-timed, finally solid:**

| phase | ours | theirs | delta | mechanism |
|---|---:|---:|---:|---|
| witness | 71.2 | 29.5 | −41.7 | their witgen SIMD packing (2× in-prove) + scaling |
| commit(+prep) | 260.3 | 231.0 | −29.3 | window packing/alloc details, NTT+merkle parity |
| zerocheck | 131.2 | 113.6 | −17.6 | their r2 lookahead+cascade (m==32-gated) |
| lincheck | 27.9 | 16.2 | −11.7 | their round-1 stripe-fold reuse |
| open | 93.6 | 79.8 | −13.8 | ranked open machinery |

No scheduling magic — five kernel/structure ports, all previously priced,
with m=32 values now attached. The big mover is the witgen SIMD packing
network: worth only ~2–3 ms at m=30 (why it was declined) but ~40 ms at
m=32. Revised menu, m=32 value per effort: witgen packing (−40, 2.6k-line
class), r2 complex (−18, ~800 lines), lincheck stripe-reuse (−12,
unscoped), open ranked pieces (−14, partially GPU-adjacent), commit misc
(−29, undecomposed). Ceiling if all land: ~600 → ~490 vs their ~400
untimed — the last ~90 is their untimed-path leanness itself.

## The SIMD packing port that became an allocation fix (2026-08-26)

The witgen SIMD packing network — the full lane-wise design: u32-granular
writers whose pending word lives in a vector register, every push one
vsli with compile-time constants, an L1 stage per stream, vld4-deinterleave
contiguous dump — was implemented via a build.rs generator (committed
artifact = the ~200-line generator, not the 2.6k-line unrolled output) and
was bit-identical on the first full test run. It then measured a NULL
in-prove at both shapes, because the real in-prove witness cost was never
the builder: **the lincheck stripe buffer was a fresh zeroed 128–512 MB
allocation every prove**, faulted during the transpose. Pooling that one
buffer (c8d36b6): witness m=30 18.5→8.0 ms, m=32 74→33 ms, both 2/2
decisive — more than the SIMD port's entire predicted value. The quad
kernel was reverted unmerged per the bloat rule; its full design and the
generator live in the commit history via c8d36b6's message.

Their in-prove witness advantage is now INVERTED at m=30 (ours 7.9 vs
their 12.1) and mostly closed at m=32 (ours ~33 vs their timed 29.5).
Third lesson of this genre in the campaign: measure the allocation story
before porting a kernel (elision, ab_pre pooling, and now this).

## Certification grid: MT target met at m=30 (2026-08-26)

Final verification grid, both trees end-to-end untimed-best, blake3
merkle+FS both arms, CPU-only both arms (their five GPU kill switches as a
zsh array, verified by comp/s sanity), 8T, interleaved same-minute pairs
with alternating order. Ours = HEAD (feb41428 build: pool fix c8d36b6,
blake defaults fc78188); theirs = their fresh Aug-26 build (5d405d46).

| pair | ours m=30 | theirs m=30 | ratio |
|---|---:|---:|---:|
| 1 | 141.85 | 133.89 | 1.059× |
| 2 | 144.69 | 137.35 | 1.053× |
| 3 | 143.28 | 132.08 | 1.085× |

Best-vs-best **141.85 vs 132.08 = 1.074×**, every pair ≤1.09×: the
"MT within 10% of yukon" target is CERTIFIED at the prod shape
(462.0k vs 496.2k comp/s). Our best buckets: witness 7.6 / commit 64.7 /
zc 33.7 / lincheck 8.1 / open 27.9.

m=32 pairs: 542.22/397.56 = 1.364× and 541.32/405.82 = 1.334×
(483.5k vs 659.4k comp/s best) — matches the priced-menu projection;
the residual is the five parked ports (r2 complex, lincheck stripe-reuse,
open ranked, commit misc, witgen tail).

Conditions: AC power, ambient GUI load (WindowServer ~55%, Texifier ~30%)
— absolute numbers mildly inflated vs quiet-window bests (their m=32 read
384-386 in a quieter window an hour earlier; ratios are interleaved and
robust). Two grid attempts discarded first: one ran a stale pre-pool
binary left newest by the A/B stash build (witness bucket 18/68 ms gave
it away), one was launched under bash where `$ARRAY` expands to its first
element — kill switches silently dropped, their GPU came alive (1.02M
comp/s tell). Both hazards now in the protocol notes.

## Commit-bucket decomposition: the gap was absorption economics (2026-08-26)

FLOCK_COMMIT_TIMING splits our m=32 commit bucket (270ms): replicate-fill
13ms solo, NTT-from-layer-2 124ms, merkle (neon8) 54ms = **191ms of commit
work** — plus the hoisted AB prep (85ms solo) absorbed at nearly full
cost. The join window is thread-THROUGHPUT-bound: both arms scale threads
near-perfectly, so wall = (191+85 work)/(pool) = 276 predicted, 270-278
measured. Sequencing the fill before the join just moved the contention
onto the NTT (124→230 beside prep; fill 90→13): thread-work is conserved,
order can't matter. Paired A/B 3/3 old-schedule (best 558.4 vs 584.8,
noisy window); reverted with numbers in the commit message.

Peer anatomy (clean window, verified): their commit is ONE fused pipelined
pass — replicate+NTT+merkle-LEAVES prints as a single 223-298ms number
(min 223, cluster 223-242) + merkle-parents 0.1ms. Their AB-prep arm
(116.9ms) rides the join FREE because the fused pass is bandwidth-bound
and leaves idle thread-time. Their zc bucket (120.4) contains no prep
(r1 41.3 + r2 49.4 + r3+ 29.7 sums exactly). So: commit work ours 191 vs
theirs 223 — WE are ahead on work; bucket ours 270 vs theirs ~231 —
they win on absorption. The earlier "-29 commit misc" menu item is
re-attributed: it is join-contention, not kernel deficit.

Consequence: the m=32 commit lever is DELETING THREAD-WORK from the join
window, not scheduling. Two candidates: (1) fuse the replicate-fill into
the first computed NTT layer-block (read z directly per replica; deletes
the 2GB fill write + its re-read, ~25-30ms thread-work); (2) retry
NTT→merkle-leaf fusion — measured null SOLO earlier (deep pass
compute-bound) but under a thread-bound join, thread-work cuts pay even
when solo wall doesn't. Together ≈ bucket parity with their 223-231.

Our zc r1/r2/r3+ split at m=32: attempted, contaminated (user interactive
on the machine; mins r1 38.7 / r2 70.1 / r3+ 64.7 exceed the known-clean
134ms zc total — upper bounds only). Redo in a quiet window; their
r2 49.4 vs our clean r2 (TBD) prices the r2-complex port properly.

## Fill→NTT fusion: NULL, reverted (2026-08-26)

Implemented `forward_transform_interleaved_from_message`: the first fused-2
top pass copies its four input rows straight from z into the codeword rows
and butterflies them in place L1-hot, deleting the standalone 2GB
replicate pass (bit-identical: garbage-start equivalence test over 5
shapes, NTT oracle, prove/verify roundtrips ×3 circuits; kill switch
FLOCK_NO_FILL_FUSE=1). Paired same-binary A/B, 8 pairs at m=32 (ambient
noisy, user interactive): commit-bucket sign test 6-2 AGAINST fusion,
totals 4-4, min-vs-min commit 308.3 fuse vs 297.9 nofuse. Reverted;
diff preserved at scratchpad/fillfuse.patch (566 lines) and in this entry.

MODEL REFINEMENT (the valuable part): the fill's 90ms under the join was
QUEUEING, not work — a memcpy pass contributes few thread-ms, so deleting
its DRAM traffic doesn't shorten a thread-bound critical path, and the
per-row copies added overhead inside the butterfly tasks. The commit
bucket is compute-limited: NTT ~124 + merkle ~54 + prep ~85 ≈ 263 of
thread-work ≈ the measured 270-278 wall. Corollary: the planned
NTT→merkle-leaf fusion retry is ALSO downgraded — it deletes a 2GB READ
(bandwidth, not thread-work) and keeps all the hashing compute; the
earlier solo null likely stands under the join too.

Surviving m=32 commit levers, by the compute model: make PREP cheaper
(unreduced-PMULL Horner arithmetic, priced ~100 lines at m=30 and
declined at ~8-15ms; prep is 85ms at m=32 so the same idea re-prices to
an est. −20-30 bucket) — everything else in the window is already at its
measured floor (fused-4 top NEON: tried, register spill, +19-26%).

## RETRACTION: the "prep Horner" menu item was already banked (2026-08-26)

Before implementing the recommended unreduced-PMULL Horner port, archaeology
killed it: the headline win behind that name is ALREADY MERGED as e1398be
(Aug 24, "accumulate round-1 prep products unreduced (pmull + weight
split)") — the −128 ms / 8-of-8 ST result, §pmull of the writeup, one of
the seven kept changes. What the menu item actually referred to was the
RESIDUAL after that merge, priced at round-1 closure (74802fc): ~8-15 ms
ceiling at m=30 ST for ~100 lines through the hottest kernel — declined
then, and the decline stands.

Decisive at today's target shape: at m=32 8T under the join, OUR prep arm
measures 85-101 ms vs THEIR prep arm's 116.9 (their own clean sample).
Our prep is already faster than theirs in the current architecture; there
is nothing left in their tree's prep worth porting. My "−20-30 ms"
estimate from earlier today was an error — I re-priced the menu label
without checking that the mechanism behind it was already in.

Corrected m=32 menu (nothing cheap left in commit): zc r2 anchor+delta
complex (−15..30, ~800 lines), open ranked pieces (−10..20), lincheck
stripe-fold reuse (−5, unscoped). The commit bucket's remaining −40 vs
theirs is absorption economics (their bandwidth-bound fused pass hides
prep free); by the compute-limited model it has no sub-800-line lever.

### Amendment (same day, prompted by Benedikt)

"Our prep is already faster than theirs" overstated an ARM-WALL comparison
into a kernel claim. Scope is symmetric (both arms = the full
challenge-independent AB transform; the challenge-dependent drain is
outside both, pinned by Fiat-Shamir), but the contexts aren't: their 116.9
runs beside a bandwidth-bound pass (near-solo), our 85-101 beside
compute-saturating passes (contended). Scaling the clean ST closure
numbers (160 vs 140 at m=30, post-e1398be) to m=32 8T: theirs ~80 solo vs
ours ~91 — their prep kernel is likely still ~12% cheaper in isolation.
The port stays dead for the corrected reason: the reachable ~10-11 ms has
no named mechanism left (unreduced accumulation, zero-copy rows, and
dead-row-fill skip are all banked here; their BCAX fold vs our
shift+x2-byte absorb is a few vector ops in a gather-dominated kernel) —
it's the campaign's unattributed uniform-kernel-quality band.

## 8-P-core pool pinning retested at m=32: still correct (2026-08-26)

Benedikt asked whether the deliberate 8-thread (P-core) global pool is now
a slowdown at the ranked shape. Paired A/B, default-8 vs
RAYON_NUM_THREADS=10, 3 pairs m=32 (noisy window): sign 2-1 for t8,
cleanest pair dead even (574.2 vs 576.1). Per-phase on the clean samples:
zc +14 ms and lincheck +6 ms on 10T (E-core stragglers at barriers),
witness/open unchanged. The early-campaign "8 beats 10 on NTT-shaped
phases" verdict holds with today's kernels. E-cores remain harvested
selectively (join window, NTT deep pass, open combine — the phases where
they add compute to bandwidth-bound windows) and nowhere else. Pool
pinning is NOT part of the m=32 residual.

### Epool priced by the peer's own kill-switch A/B: below bar (2026-08-26)

The peer added FLOCK_NO_EPOOL to their tree (it didn't exist; I'd assumed
it from naming) and ran 3 alternating pairs at ranked CPU config: epool
is worth ~6.6% / ~26.5ms total THERE, 3/3 clean — but decomposed:
commit −10.7 (their AB-prep-hetero, the analog of the all-core join we
ALREADY have), zc −6.7, lincheck −0.6, open +3.1 (reversed, ~noise).
Portable NEW value for us = zc+lincheck ≈ −7ms for a queue primitive +
call-site conversions (~200 lines): below the bloat bar and priced-and-
parked (zc-only variant listed on the menu at −6..7). The "uniform
kernel-quality band" is NOT hidden E-core scheduling; the m=32 menu
stands: r2 complex (−15..30, ~800 lines), open ranked (−10..20),
lincheck stripe-reuse (−5), epool-zc (−6..7, ~200 lines).

### epool: NULL-TO-NEGATIVE on our kernels, reverted (2026-08-26)

Paired kill-switch A/B, 3 pairs m=32: tail 3/3 WORSE with helpers
(49.7-51.4 off vs 53.6-56.2 on), zc bucket 3/3 worse (131-149 off vs
142-171 on), r2 2/3 worse (42.2-47.4 off vs 45.3-46.7 on). Reverted (the
implementation survives in history, commit "zerocheck: bounded-tail
E-core chunk queue"). Mechanism: our r2/tail are bandwidth-heavy
streaming passes; background-QoS E-cores add DRAM pressure the P-cores
need — 4th confirmation of the bandwidth-on-bandwidth rule. Their epool
pays on THEIR kernels plausibly because anchor+delta/compact formats
made those passes compute-relative. RETEST epool after the anchor+delta
port lands.

Also learned from the off arm: our incumbent r2 is 42-47ms at m=32 —
already FASTER than their timed 49.4 (≈ parity with their scored ~43).
The r2-side of the anchor+delta port is ≈0; the port's value is
concentrated in the TAIL (ours ~50 vs their timed 29.7): the r3
table-combine, lookahead, and cascades. Task #3 rescoped accordingly.

## Cascade tail: built, verified, staged opt-in — coupled to anchor+delta (2026-08-26)

Three A/B rounds at m=32 told the full story:
1. Scalar lookahead passes (as the AG tail ships them): tail 57-59 vs
   classic 52-54 — the historical "lookahead = 13-15% regression" verdict
   reproduced, and diagnosed as KERNEL QUALITY (generic scalar F128 muls
   vs the classic path's q-resident NEON kernel), not scheduling.
2. NEON lookahead kernel + fold1 entry: tail 50.4-51.7 vs 50.5-51.4 —
   WASH, explained by pass accounting: the fold1 entry (the largest pass)
   saves no traffic and pays the full product bill.
3. Integrated r2 lookahead (r2 emits the 8 sums; all tail passes 4→1):
   tail 35.0-37.4 vs 49.6-53.1, 3/3 DISJOINT (−15) — but r2 67-80 vs
   42.6-47.8 (+24). The surcharge is the mul-count floor (8 mul_q + 8
   wide per group vs classic's 4+4 = 36 extra PMULL/group × 2^24); a
   two-sweep de-spill restructure changed nothing. Net zc ≈ +9. 

CONCLUSION: cascade and anchor+delta are COUPLED — their tree affords the
surcharge only because their compact r2 pays it from a lower base
(deferred odd-element folds + cheaper unreduced product accumulation).
Staged opt-in (FLOCK_ZC_LOOKAHEAD=1, transcript byte-identical by test);
anchor+delta port is the remaining piece, anatomy question out to the
peer: where do the odd folded values for THEIR Q products come from —
paid delta-gathers in r2, or a product formulation in anchor/delta space?
