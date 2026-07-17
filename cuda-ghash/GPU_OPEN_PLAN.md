# GPU acceleration — pcs::open (Ligerito recursive), CUDA

Successor target after `GPU_COMMIT_PLAN.md` (commit done) and the roadmap in
`GPU_PROVER_PLAN.md`. Re-measured phase breakdown on this box (m=29, `SHA2_LOG2S=14
cargo bench --bench sha2_proof`, ~71 ms total):

| phase | ms | share | status |
|-------|----|-------|--------|
| gen_witness + lincheck setup | 4.37 | 6.2% | — |
| pcs::commit (NTT + Merkle) | 14.71 | 20.9% | ✅ ported |
| **zerocheck::prove_packed** | 22.35 | **31.7%** | future |
| lincheck::prove | 10.17 | 14.4% | future |
| **pcs::open (ligerito)** | 19.48 | **27.6%** | ← THIS PLAN |

Open is the chosen next target over zerocheck: it **reuses the already-validated
NTT + Merkle kernels** and the **resident codeword buffer** (the #1 resident
buffer per `GPU_PROVER_PLAN.md`), so the only genuinely new compute is the
degree-2 sumcheck fold — and that fold primitive is what zerocheck will reuse
afterward. The architectural payoff: commit→open share the codeword on-device,
paying PCIe for neither.

## Backend note — Ligerito only

The repo's PCS backend is **Ligerito**: `pcs::open_batch_mixed_ligerito` →
`ligerito::recursive_prover_with_basis` — recursive ladder, induced-basis
sumcheck. (A legacy BaseFold backend existed when this port began; it was
removed from the Rust tree in #9, and the GPU-side FRI-fold / row-batch-fold
kernels that mirrored it — the original steps 1–2 — were dropped with it.)

The core on-path primitive is the **sumcheck fold+message** (step 3):
Ligerito's `fold_and_msg_lsb` is `nf[j] = f[2j]·(1+r) + f[2j+1]·r`, and the
kernel is **validated against the real `SumcheckProver`**
(`dump_ligerito_sumcheck_vectors.rs` → `test_sumcheck_ab`, bit-exact to L=20).

Ligerito-specific compute still to port: `induce_sumcheck_poly` (induced basis
from opened rows), `introduce_new`/`glue` (α-batched basis), per-level NTT+Merkle
commit (✅ have it from the commit port), query openings, recursive orchestration.

## Validation discipline (mirror `test_commit_ntt`)

- Rust oracle dumper (sibling to `src/bin/dump_ghash_vectors.rs`): run the real
  CPU rounds on fixed inputs; emit inputs, per-round challenges, per-round
  messages (`u_0,u_2`), and the folded buffer after each round.
- Per-kernel `test_*.cu`: load the dump, run device kernels round-by-round,
  assert **bit-for-bit** on both round messages and the folded buffer every round.

## Sequencing

1. ~~FRI `fold_pair` kernel~~ — removed with the BaseFold backend (#9).
2. ~~Row-batch fold kernel~~ — removed with the BaseFold backend (#9).
3. ✅ **Sumcheck fold+message `{u_0,u_2}`** — `sumcheck_ab.cuh` /
   `test_sumcheck_ab.cu`. Reduce-per-term message + adjacent-pair (LSB) fold,
   bit-exact to L=22. **Re-validated against the real Ligerito `SumcheckProver`**
   via `dump_ligerito_sumcheck_vectors.rs` (`make sumcheck_ab_ligerito`,
   bit-exact to L=20). This is Ligerito's `fold_and_msg_lsb` — folds
   `(f, combined_basis)`. Also the primitive zerocheck reuses.
   `bench_sumcheck_ab.cu`: throughput at m=33–35.

### Ligerito-specific roadmap (the recursive ladder)

4. ✅ **`induce_sumcheck_poly`** (`ligerito.rs:1757`) — `induce_sumcheck.cuh` /
   `test_induce.cu` / `dump_induce_vectors.rs`. Host setup (eq tables, `next_s`
   s_k chain, per-query low/high tensors, inverses via `f128_inv_host`) + device
   O(n·n_queries) accumulation (reduce-per-term). **Bit-exact vs the real
   `induce_sumcheck_poly`** (made `pub`) — basis_poly + enforced_sum, to log_n=20,
   FRI query counts 148/243, odd log_n. (`make induce`.)
5. ✅ **`introduce_new` + `glue`** (`ligerito.rs:2402`/`:2438`) —
   `introduce_glue.cuh` / `test_introduce_glue.cu` / `dump_introduce_glue_vectors.rs`.
   3-sum (F128) message+eval kernel (`u_0,u_2,h_new=Σ f·b_new`) + `glue` AXPY
   (`combined_basis += β·b_new`). **Bit-exact vs the real
   `SumcheckProver::introduce_new_with_eval`** + glue, to L=22. (`make introduce_glue`.)
6. 🚧 **Recursive orchestration** — in progress.
   - ✅ **Device-resident `SumcheckProver`** — `test_sumcheck_prover.cu` /
     `dump_sumcheck_prover_vectors.rs`. Composes step-3 (fold+message) + step-5
     (glue) into the real state machine: `(f, combined_basis)` stay in VRAM the
     whole run; only `{u_0,u_2}` messages cross to host. Driven by a scripted
     op sequence (fold | introduce+glue) from the real `SumcheckProver`;
     **full transcript + final f bit-exact** to L=22. (`make sumcheck_prover`.)
   - ✅ **Host `FsChallenger` (Fiat-Shamir)** — `challenger.hpp` /
     `test_challenger.cpp` / `dump_challenger_vectors.rs`. Host C++ SHA-256
     duplex sponge: observe (f128/bytes/label/slice), sample (scalar/vec),
     `grind_pow` (PoW leading-zero search), `sample_distinct_queries`.
     **Byte-exact vs the real `FsChallenger`** (samples + grinds + queries).
     (`make challenger`.) Derives every challenge/β/query/nonce.
   - Note: `ligero_commit` (per-level commit) = `replicate_message_fill` +
     interleaved NTT + Merkle — the **exact commit-port pipeline** (already
     validated by `test_commit_ntt` + `test_commit_merkle`), no new kernel.
   - ✅ **Merkle multi-proof query opening** — `merkle_open.hpp` /
     `test_merkle_open.cpp` / `dump_merkle_open_vectors.rs`. Host port of
     `merkle::merkle_multi_proof` (deduplicated batched opening). **Byte-exact vs
     the real function** at FRI query counts 50/100/243 (e.g. 243 queries →
     1735 siblings vs 3888 independent paths). (`make merkle_open`.)
   - ✅ **L0 orchestrator (initial sumcheck + commit f¹)** — `test_ligerito_l0.cu`
     / `dump_ligerito_l0_vectors.rs`. Challenger-driven composition: host
     `FsChallenger` **derives** the `initial_k` fold challenges (+ fold-grinding)
     from the observed transcript; the device-resident sumcheck consumes them;
     then **commit f¹ on device** (replicate-fill + interleaved NTT + Merkle →
     root) is observed and a post-commit probe challenge binds to it. **Byte-exact**
     (folds + grind nonces + messages + device Merkle root + post-commit probe)
     vs the real `FsChallenger`+`SumcheckProver`+`ligero_commit`, across grinding
     on/off and rate 1/2 & 1/4. (`make ligerito_l0`.)
   - ✅ **FULL L0 phase** (`test_ligerito_l0.cu` / `dump_ligerito_l0_vectors.rs`):
     real **L0 commit** (device NTT+Merkle) → observe → `initial_k` folds → commit
     f¹ → OOD intro/glue (on-device MLE-eval) → query grind + `sample_distinct_queries`
     + α → **open rows + multi-proof** (gather + host multi-proof over the resident
     L0 tree) → **`induce` basis₀** (real opened rows, `eval_sk_at_vks` host-ported)
     → `introduce`/`glue`. **Byte-exact** vs the real prover across grinding on/off,
     rate 1/2 & 1/4, OOD 0–3, queries 30–100, sizes to log_n=18. (`make ligerito_l0`.)
     This is the complete template for the recursive levels.
   - ✅ **E2E for general `r`** (`test_ligerito_l0.cu` / `dump_ligerito_l0_vectors.rs`):
     full L0 phase **+ the recursive ladder** — a loop over `r` levels, each
     folding k, querying the *previous* commit, and (if not last) committing
     f^{i+2} → OOD → query/open → **induce** → introduce/glue; the last level emits
     the final `yr` + opening. The **entire prove transcript** — every Merkle root,
     fold/OOD/β challenge, message, grind nonce, query position, multi-proof, and
     induced basis — matches the real `recursive_prover_with_basis` **byte-for-bit**.
     Validated at **r = 1, 2, 3**, grinding at every site, rate 1/2 & 1/4, OOD 0–3,
     sizes to log_n=20. The whole Ligerito prover runs on device + host challenger.
   - ✅ **Perf pass** (`bench_ligerito.cu`): full prove benchmarked + optimized
     **160 ms → 11.3 ms** at log_n=22 (≈ m=29 witness) on the RTX 5090 — **~1.7×
     faster than the 19.48 ms CPU open**. Fixes (all kept byte-exact, re-validated):
     (1) **eq tables on device** (`build_eq_device`) — OOD 47→0.2ms; (2) **induce
     qtensor build on device** (`induce_setup_device`/`build_qtensors_kernel`) —
     induce 47→5.2ms; (3) **codewords stay resident on device, gather only the
     queried rows** (`gather_rows_k`) instead of a 128MB D2H — commit 60→3.8ms;
     (4) twiddle tables cached by k_code (precomputed static data). Breakdown @22:
     commit 3.8, induce 5.2, fold 1.1, ood 0.2, open 0.1.
   - ✅ **Device multi-proof** (`merkle_open_device.cuh`): compute emitted node
     indices on host (positions-only), gather just those ~q·d sibling hashes from
     the resident device tree — no full-tree D2H. **log_n=24: 49.9→22.85 ms**
     (commit 33→8 ms), log_n=22: 11.3→10.3 ms. Validated host+device byte-exact
     (`test_merkle_open`) + full e2e.
   - ✅ **Device w-chain** (`compute_w_kernel`): the per-query s_k chain moved off
     host software-clmul onto hardware clmad (enforced_sum stays host — it's
     w-independent). **induce 5.2→3.1 ms; total @22 now 8.1 ms (~2.4× the CPU
     open)**, @24 19.6 ms. Validated byte-exact via e2e (basis compared directly).
     Remaining levers: inv_sks/enforced_sum on device (~2 ms host left in induce),
     msg↔fold fusion (fold 1.1 ms), pinned buffers / fewer launches.
   - **Remaining**: serialize the `LigeritoProof` struct and compare proof bytes —
     layout-only (bincode field order); the transcript that determines the proof is
     already proven equal.
7. Fuse with commit: codeword/f resident commit→open; measure the **fused**
   number (the only trusted one).

## Verification

- End-state: fused commit→open device time beats the 14.7+19.5 ms CPU sum, with
  the codeword never leaving VRAM.

## Transpose-NTT induce (GPU fast path) — the big induce win

Upstream's `induce_sumcheck_poly_via_ntt` (transposed forward additive-NTT over
the scattered query weights) **regresses ~2× on a 32-thread CPU** (poor
parallelism) but is a structured, bandwidth-bound NTT — ideal on the GPU.
`ntt_transpose.cuh`: scatter sparse query-weights → transposed Fᵀ-NTT (butterfly
`a'=a+b; b'=t·(a+b)+b`, reverse layers) → truncate to the message coords. The
sparse-window prefix isn't needed (the GPU affords the full dense transpose).

- `test_transpose_induce.cu` — **byte-for-bit vs `induce_sumcheck_poly_via_ntt`**
  across all m29_fast per-level rates (log_inv_rate 1..5).
- **induce basis build: dense GPU ~5.6 ms → transpose-NTT 0.02–0.07 ms (~80×).**
- Full prove (m29_fast, `bench_ligerito fast29`): **11.2 → 5.74 ms**
  (induce 5.59 → 0.33 ms) — now **~4× the CPU open**.

## Perf pass 2 — fold-fusion + buffer pooling (m=35 focus)

Re-measured at **m=35** (log_n=28, m35_fast, RTX 5090) — the large-m regime where
the phase mix differs from m=29 (folds dominate, not induce).

- **Witness-buffer elision**: fill the sumcheck state (df,dcb) directly, commit L0
  from it — no separate d_f/d_b1. Needed to fit m=35 in 32 GB (peak ~24 GB).
- **Fused fold+message** (`sumcheck_fold_msg_partial`, ligerito's `fold_and_msg_lsb`):
  compute round k+1's {u_0,u_2} during round k's fold pass — f/cb read once/round,
  not twice. **Byte-for-bit re-validated** via test_ligerito_l0 (every fold msg +
  folded head). m=35 fold 29.1 → 24.5 ms (the per-round message passes; the
  irreducible first-fold + prime pass remain).
- **Buffer pooling**: (a) the borrowed **L0 codeword/tree is freed AFTER the timer**
  (a real open doesn't free its input — the caller does); freeing 8 GB inside the
  open wrongly inflated it. (b) induce `d_c` scratch pooled grow-only. Together:
  the ~12.5 ms unattributed gap (large-buffer cudaFree) → ~1.3 ms, and removing the
  per-iteration 8 GB free de-fragmented the allocator (commit 11.7→8.7, induce
  5.7→4.8). Recursive-commit mallocs measured negligible (0.09 ms) — no pooling needed.

**m=35 open: 60.6 → 41.1 ms** (commit 8.7, fold 24.5, induce 4.8, rest ~3).
**m=29 open: 4.18 → 3.41 ms.** Fold (60%) + commit (21%) now dominate; both are at
their memory/compute floor (fold = first-fold+prime full-witness passes; commit =
NTT+Merkle over the recursive codewords).
