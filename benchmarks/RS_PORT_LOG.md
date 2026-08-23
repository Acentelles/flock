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

Net kept: **round 2 −13.4% ST**, total 4781.0 → 4716.2 ST (−1.4%), for 179
added lines.

## What the negative results tell us

- **Round 1 is gather-bound, not PMULL-bound.** Attempt 4 did exactly what it
  claimed — reductions cut by a factor of `big_lo_size`, products from 6 PMULLs
  to 3 — and round 1 did not move. `accumulate_convert_with_s_hat_v` issues 3
  gathers per lane per b_med into the 64 KB convert table, up to 3072 loads per
  chunk, and that traffic hides the arithmetic completely. **Round 1's
  remaining ~1130 ms is not reachable by cheaper arithmetic.** Whoever picks
  this up should attack the gather pattern; that is what the challenge repo's
  AB-precompute-with-NT-stores and `c_fold4` mask tables do, and it is the
  bloat-heavy part.
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

## Note on the geometric trick

The three layered optimizations in `univariate_skip_optimized.rs` — geometric
small-eq + shift_reduce, geometric medium-eq + 64 KB convert-table lookups, and
D^-1 absorbed into eq_lo — are **already in this tree**; the doc headers are
byte-identical to the challenge repo's. They are inherited upstream code, not a
Yukon addition, and nothing there needs porting.
