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
| round 1 (`round1 URM`) | 1477.30 ms | 1403.07 ms | **-5.0%**, 8/8, disjoint ranges |
| end-to-end headline | 50143 comp/s | 52566 comp/s | **+4.8%**, 8/8 |

Caveat on the end-to-end figure: the base arm spread that run was wide
(46894-51655, ~10%) while head was tight (51231-53209), so the point estimate is
soft even though the sign test is not. The round-1 number is the better
measured of the two.

The three kept changes:

1. **Round-2 NEON register accumulator** (~179 lines) -- `WideNeon`, a 256-bit
   product held as two uint64x2_t instead of the GPR-resident F256Unreduced.
   -13% on `zc_round2`.
2. **Structurally-zero b K-row skip** (~10 lines) -- see above. Part of the
   -5.0%.
3. **Two lanes per drain iteration** (~50 lines) -- see above. Part of the
   -5.0%.

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
default-ON in their tree and loses on both chips. Their actual rounds-2+
mechanism is cascade2/cascade3: composing rho into a 32 KiB byte table so round
pairs (5+6, 7+8) collapse into one composed double-fold each, deleting a full
DRAM pass and a Fiat-Shamir round boundary per fusion — exact reassociation,
unmeasured individually.

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
