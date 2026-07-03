# GPU acceleration — PCS commit (CUDA, throughput-first)

Handoff notes for moving the next step onto the Blackwell box. Decisions are
locked; P1 is done on the M4; P2+ are authored/run on the GPU.

## Decisions

- **Platform: NVIDIA / CUDA.** Builds on the existing `cuda-ghash/` PoC, whose
  `F128` (GHASH-form GF(2¹²⁸)) is already validated bit-for-bit against the Rust
  `F128` via `clmad`. (Apple/Metal was rejected for the first slice: no native
  carryless multiply, none of the `clmad` work transfers.)
- **Goal: throughput / large instances.** Where the GPU structurally wins and
  host↔device transfer amortizes. *Not* single-proof latency — the M4 NEON path
  is already near the compute floor, and PCIe would eat a one-shot win.
- **First slice: PCS commit** (`src/pcs/commit.rs::finalize_commit`). Most
  GPU-classic phase, narrow clean host/device boundary, field building block
  already validated.

## The pipeline we're porting (`src/pcs/commit.rs:263`, `finalize_commit`)

1. `replicate_message_fill` — cheap copies (= the first `log_inv_rate` NTT
   layers on `[z, 0, …, 0]`, which are pure copies).
2. `AdditiveNttF128::forward_transform_interleaved_from_layer(&mut codeword,
   num_ntts, log_inv_rate)` — **interleaved additive (binary-field) forward
   NTT**. `num_ntts = 2^log_batch_size` independent sub-NTTs share twiddles;
   SoA layout `codeword[pos * num_ntts + lane]`. Butterfly is
   `x[i] ^= twiddle ⊗ x[j]` with the `clmad` multiply. **Additive FFT, NOT
   Cooley–Tukey.**
3. `merkle::merkle_tree(codeword_bytes, n_leaves)` — SHA-256, one leaf per
   codeword position (`num_ntts · 16` bytes), thousands of independent leaf
   compressions then a pairwise reduction. Output: `root` + `merkle_tree`.

## Staged plan (each step gated on a byte-exact oracle)

- **P1 — CPU oracle dumper. ✅ DONE.** `src/bin/dump_commit_vectors.rs` runs the
  real `commit()` and emits a self-describing file: input `z_packed`, all
  derived params, golden post-NTT codeword, golden Merkle root. Same workflow as
  `dump_ghash_vectors`, at the whole-pipeline level.

  ```sh
  # small m → bit-exact correctness target:
  cargo run --release --bin dump_commit_vectors -- cuda-ghash/commit_vectors.bin 16 1 5
  # args: <out> <m> <log_inv_rate> <log_batch_size>   (defaults: 20 1 5)
  ```

  File format is documented in the binary's header comment ("CMT1" magic).

- **P2 — GPU additive-NTT kernel. ✅ CORRECT (GPU-validated); optimization pending.**
  Port `AdditiveNttF128`'s twiddle schedule onto `f128.cuh`/`clmad`. **The one
  correctness risk is matching the twiddle schedule exactly** — reference:
  `forward_transform_interleaved_from_layer` and `twiddle(layer, block)` in
  `src/ntt/additive_ntt_f128.rs`. Correctness-first (one layer per launch,
  global mem, SoA).

  Files added:
  - `ntt_host.hpp` — pure-C++ GF(2¹²⁸) math + twiddle table
    (`build_twiddle_table`/`twiddle_from_table`), shared host/device.
  - `ntt_f128.cuh` — the `ntt_layer_kernel` device butterfly (one thread per
    `(block,row,lane)`), reusing the host twiddle table + `ghash_mul_binius`.
  - `host_check_ntt.cpp` — pure-host (g++) checker: replicate-fill + scalar
    interleaved NTT vs the CMT1 golden codeword. **The twiddle schedule was
    confirmed bit-for-bit on CPU** across rate 1/2·1/4·1/8, batch 2–5, m≤22
    (`make` has no target; build with `g++ -O2 -std=c++17`). This de-risks the
    schedule before the GPU box; the Blackwell run only has to match launch
    mechanics + clmad.
  - `test_commit_ntt.cu` (+ `make test_commit_ntt` / `make commit_ntt`) — the
    CUDA equivalent: runs `ntt_layer_kernel` per layer, diffs the full codeword.

  Key fact that pinned the schedule (`src/pcs/commit.rs:270`): the NTT is built
  with `dim == k_code` and the per-lane buffer is `2^k_code`, so **L == log_d**;
  `twiddle(l,b) = evals[L−l−1][1..]` spanned by the bits of `b`, with the row's
  0-th (normalized 1) absorbed into the butterfly.

  **Done:** the CUDA kernel (`make test_commit_ntt`) matches the flare oracle
  bit-for-bit on an RTX 5090 (sm_120) across rate 1/2·1/4, batch 3·5, m=16–22.
  **Remaining:** optimize — shared-mem tiling, multi-layer fusion per launch,
  coalesced lane access — then measure throughput vs the CPU baseline.

- **P3 — GPU SHA-256 Merkle kernel. ✅ CORRECT (GPU-validated) + benched.**
  Files: `sha256.cuh` (FIPS-180-4 scalar SHA-256, byte-identical to the `sha2`
  crate — sm_120 has no SHA hardware, so the win is pure parallelism, one thread
  per hash), `merkle.cuh` (`merkle_leaf_kernel` one thread/leaf,
  `merkle_level_kernel` one thread/parent, `launch_merkle` = leaf kernel + one
  launch per level), `test_commit_merkle.cu` (feeds the GOLDEN codeword, checks
  the root — isolates P3 from P2), `bench_merkle.cu`. Flat layout + leaf/node
  hashing mirror `src/merkle.rs` exactly (no domain separation). **Root matches
  the flare oracle bit-for-bit** across batch 1–5 (32–512 B leaves, both tail
  shapes), m=14–24. Used one-thread-per-leaf, not warp-per-leaf — simpler and
  already 9× the CPU; revisit warp/leaf only if P4 shows Merkle dominating.

- **P4 — Fuse + measure with transfer.** Keep the codeword **resident on device**
  between NTT and Merkle (only H2D the small message, D2H the 32-byte root).
  Time H2D / NTT / Merkle / D2H separately. Throughput story: a **batch of
  commits across CUDA streams**, overlapping next-instance H2D with current
  compute — a single m=29 commit won't saturate Blackwell; many concurrent ones
  will. Compare against the CPU baseline (`FLOCK_COMMIT_TIMING=1` splits NTT vs
  Merkle on the M4).

- **P5 — Integration decision.** Only if P4 clears the gate, wire a
  feature-gated `commit_cuda` into Rust (C-ABI `.so` + `build.rs`, or `cudarc`),
  with a CPU-vs-GPU byte-identical cross-check test mirroring
  `commit_matches_full_ntt_oracle` in `src/pcs/commit.rs`.

## P2 throughput (RTX 5090 vs Threadripper 7970X, same box)

Rate 1/2, batch 5. GPU = `bench_commit_ntt`. CPU = real interleaved `commit()`
NTT, `FLOCK_COMMIT_TIMING=1`, 64 threads. `single` = one layer per launch;
`fused` = greedy 4/2/1-layer fusion (registers).

| m  | buffer | single   | **fused** | CPU NTT (64T) | fused vs single | fused vs CPU |
|----|--------|----------|-----------|---------------|-----------------|--------------|
| 26 | 16 MB  | 0.115 ms | 0.127 ms  | 5.87 ms       | 0.9×            | 46×          |
| 28 | 64 MB  | 0.397 ms | 0.443 ms  | 12.96 ms      | 0.9×            | 29×          |
| 29 | 128 MB | 2.98 ms  | **1.005 ms** | 19.03 ms   | **3.0×**        | **19×**      |
| 30 | 256 MB | 6.36 ms  | **1.95 ms**  | —          | **3.3×**        | —            |
| 31 | 512 MB | 13.5 ms  | **4.46 ms**  | —          | **3.0×**        | —            |

**The single-layer kernel hits an L2 cliff at m=29**: throughput peaks at m=28
(buffer fits the ~96 MB L2) then collapses to ~24 GMul/s / ~1.5 TB/s (≈ HBM
peak) once it spills, because each of `log_dim` layers does a full-buffer
read+write. **Layer fusion removes the cliff** — fusing 4 layers per launch
cuts m=29 from 17 passes to 5 (4+4+4+4+1), holding ~70–77 GMul/s across all
large m. Fused-4 uses 138 registers with **zero spills** (16 F128 stay in
registers). Below the cliff (m≤28) fusion is marginally slower — the
single-layer kernel was already cache-resident there, and the fused kernel's
register pressure lowers occupancy — but that's noise next to the large-m win.

(`bench_commit_ntt`'s GB/s column counts *logical* per-layer traffic; with
fusion the real HBM traffic is ~K× lower, so GMul/s is the honest metric.)

**Shared-mem deep-layer tiling was tried and REGRESSED (~25% slower, m=29
1.00→1.26 ms).** `ntt_deep_smem_kernel` + `launch_ntt(deep_smem=true)` load a
2^dt-position tile to shared mem and run the deepest dt layers on-chip in one
pass. It loses because the premise is wrong: under pure fusion the deep layers
have *small strides and are already L2-resident*, so they're cheap — replacing
those L2-cached fused passes with a 64 KB-shared-mem kernel just tanks occupancy
(1–2 blocks/SM) and adds `__syncthreads` barriers. The L2 already does the
tiling. Kept in-tree behind the default-off flag as a documented negative
result. **Don't re-try deep tiling.**

State: the fused kernel is HBM-bound on the top passes (~1.27 TB/s ≈ 71% of the
5090's HBM peak) and L2-bound on the deep passes. The only remaining lever is a
transpose-based 4-step decomposition (fewer full-buffer passes) — but the
transpose itself costs HBM passes, so it's unlikely to beat fusion. Treating the
NTT as optimized; the next real win for end-to-end commit is P3 (Merkle).

## P3 throughput (Merkle, RTX 5090 vs Threadripper 7970X)

Rate 1/2, batch 5 (512-byte leaves). GPU = `bench_merkle`. CPU = real
`commit()` Merkle, `FLOCK_COMMIT_TIMING=1`, 64 threads.

| m  | leaves | GPU Merkle | CPU Merkle (64T) | speedup |
|----|--------|------------|------------------|---------|
| 26 | 32 K   | 0.202 ms   | 3.94 ms          | 19×     |
| 28 | 131 K  | 0.434 ms   | 5.65 ms          | 13×     |
| 29 | 262 K  | 0.649 ms   | 5.85 ms          | 9×      |
| 31 | 1.05 M | 1.934 ms   | —                | —       |

SHA-256 is compute-bound (no SHA hardware on sm_120; ~4.4 Gcompr/s at m=29).
Leaf hashing dominates (9 compressions/leaf at 512 B vs 2/node).

**Plonky3-derived micro-opts evaluated, both REGRESS on GPU** (mined from
~/plonky3 dft + merkle-tree crates). They're CPU cache/SIMD techniques that
don't transfer to the GPU's throughput model:
- *NTT shared-mem deep tiling* (their L1-aware chunking) — see the P2 note;
  deep layers are already L2-resident under fusion. ~25% slower.
- *Merkle K-way leaf interleave* (their vertical packing / the CPU 4-way SHA) —
  `sha256_kway` + `launch_merkle(kway=2/4)`. Monotonically slower (m=29:
  0.65→0.78→0.90 ms). One-thread-per-leaf already hides SHA's chain via TLP and
  is throughput-bound; per-thread ILP just inflates registers (212/255) and
  kills occupancy. Default stays kway=1; code kept as a documented negative.
- What plonky3 confirmed we already do right: *multi-layer fusion* (their 3
  layers/kernel; we fuse 4). The remaining untried NTT lever is the six-step
  transpose decomposition (deferred — modest payoff, high risk for the additive
  field, transpose adds a pass).

## Full commit @ m=29 (NTT + Merkle, on the same box)

| stage  | GPU       | CPU (64T)  |
|--------|-----------|------------|
| NTT    | 1.004 ms  | 19.03 ms   |
| Merkle | 0.649 ms  | 5.85 ms    |
| **sum**| **1.65 ms** | **24.9 ms** (**15×**) |

This is compute only (codeword already on device). P4 measures it end-to-end
with H2D message in / D2H root out, and batches commits across streams.

## Gotchas / notes

- **GPU multiply ranking is the REVERSE of the CPU.** Measured on RTX 5090
  (sm_120, `make bench_f128`): **karatsuba 139 GMul/s > schoolbook 103 > binius
  68** (software shift-XOR ~17). Karatsuba also wins single-thread latency (179
  vs 245 vs 295 ns/op). This inverts the M4 CPU, where `binius` is fastest
  (`INSIGHTS.md` §1: +26% via PMULL-pair-friendly reduction) — its 2-stage
  serial recursive reduction stalls on Blackwell, while karatsuba's 3
  independent carryless products (6 CLMAD) feed the pipeline. **The NTT butterfly
  uses `ghash_mul_karatsuba`; do NOT copy the CPU's binius choice.**
- **Not apples-to-apples.** Blackwell-GPU vs M4-CPU is a different machine; frame
  results as commit *throughput* on their own merits plus the phase speedup vs
  the CPU.
- **Open also reads the codeword.** The opening phase rebuilds the T₁/T₂ FRI
  Merkle trees over the same codeword (`src/pcs.rs` module docs). A full
  throughput win eventually wants the codeword to stay on-device through open
  too — but commit-first is the right beachhead.
- **Don't dump giant golden codewords for perf runs.** At m=29 the codeword is
  128 MB → the oracle file is ~256 MB. Validate bit-exactly at small m; for
  large-m throughput runs carry only the seed + root and regenerate the message
  on-device.
- **Baseline numbers** (M4 Max, from `INSIGHTS.md`): commit is dominated by the
  NTT butterflies + SHA-256 Merkle hashing; full proof gen ~26.5 ms at m=29.
