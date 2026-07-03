# GPU acceleration — full Ligerito prover (CUDA, resident-on-device)

Successor to `GPU_COMMIT_PLAN.md`. The PCS commit (NTT + Merkle) is ported,
validated bit-for-bit, and ~15× the 64-thread CPU. But commit is only **19%** of
`prove_fast`, so the next wins are the other phases — and the *real* payoff is an
end-to-end GPU prove that keeps the big buffers resident and pays PCIe only for
the witness in / proof out.

Approach (decided): **end-to-end on the whiteboard, piecemeal at the keyboard,
fused on the bench.** Design the resident data-flow up front; implement and
validate each phase against a CPU oracle in isolation (as we did for NTT/Merkle);
compose adjacent phases resident as soon as two exist; only trust the fused
end-to-end number, never per-phase-in-isolation (those include transfer the
fused pipeline won't pay).

## Measured phase breakdown (m=29, Threadripper 7970X, 32 threads)

From `SHA2_LOG2S=14 cargo bench --bench sha2_proof` (`[prove_fast breakdown]`,
uses `prove_fast_ligerito_timed`). Total ~72–86 ms:

| phase | ms | share | status |
|-------|----|-------|--------|
| gen_witness + lincheck setup | 4.6 | 6% | — |
| pcs::commit (NTT + Merkle) | 14.7 | 19% | ✅ ported (`cuda-ghash/`) |
| **zerocheck::prove_packed** | 23.6 | **31%** | ← target 1 (biggest, most parallel) |
| lincheck::prove | 10.7 | 14% | target 3 |
| **pcs::open (Ligerito recursive)** | 22.8 | **30%** | ← target 2 (reuses NTT+Merkle) |

Re-measure on the actual GPU box — shares shift by machine (M4 NEON vs x86 vs the
GPU's CPU host). zerocheck + open = **61%** combined.

## Prover data-flow (from `src/prover.rs::prove_fast_ligerito_timed`, ~770–880)

Phases are chained two ways: **big buffers** consumed downstream, and the
**challenger** (Fiat-Shamir transcript) threaded through every phase with each
phase's output challenges feeding the next.

```
witness z_packed ─┬─► commit ──► (commitment, codeword[128MB@m29], merkle_tree)
                  │                    │
a_packed,b_packed─┼──► zerocheck ◄─────┤ (c_packed = z_packed)
                  │      │  └─► zc_claim ─► x_ab
z_packed_lincheck─┼──► lincheck ◄── x_ab
                  │      └─► lc_claim ─► claims {ab, c} + s_hat_v
                  └─► open (Ligerito) ◄── codeword, commitment, claims
                         └─► pcs_open
challenger (host transcript) ──── binds/absorbs/squeezes through ALL phases ────►
```

Residency targets, by buffer:
- **codeword** (`prover_data.codeword`, 128 MB @ m=29): produced by commit,
  reused by open (open rebuilds FRI Merkle trees + NTT folds over it). **#1
  resident buffer** — commit→open must share it on device, never round-trip.
- **z_packed** (witness): used by commit, zerocheck (as `c_packed`), and open.
  Resident the whole proof.
- **a_packed / b_packed**: zerocheck-only; freed to scratch right after
  (`scratch::give_f128`). Allocate from the device pool, free after zerocheck.
- **transcript / claims** (zc_claim, x_ab, lc_claim, s_hat_v): small, sequential,
  host-side. Cross PCIe between phases as D2H proof-messages + H2D challenges.

## Orchestration model (mined from sp1-gpu `crates/cuda`, `shard_prover`)

Patterns to copy (file refs are in sp1-gpu):
- **One allocator scope per proof** (`task.rs:351`): a stream + its async mempool
  back every device buffer for the whole prove. Buffers are passed *by reference*
  across phases; phase N+1 consumes phase N's device buffer with zero copy
  (`shard_prover/prover.rs:615–763` — traces never touch host across 4 phases).
- **Stream-bound async malloc + pool** (`task.rs:188–343`):
  `cudaMallocAsync`/`cudaFreeAsync` on the proof's stream; mempool release
  threshold = MAX so freed blocks (e.g. a/b after zerocheck) are *reused*, not
  returned to the OS. No per-phase realloc.
- **Pinned host buffers** for the only PCIe crossings — witness in, proof out
  (`pinned.rs`); a reusable pinned buffer pool (`prover.rs:68,158`).
- **Host-side Fiat-Shamir**, device-side only where heavy (grinding)
  (`challenger/duplex.rs:78–119`). The transcript is inherently sequential, so
  derive challenges on host between phases; copy the small eval points/claims to
  device for GPU-native polynomial evaluation (`prover.rs:714–752`).
- **One-time upload of static data** (`prover.rs:78`): twiddle tables, round
  constants, R1CS structure — upload once, reuse across proves/shards.

What crosses PCIe, and when:
- once: static tables (twiddles, SHA/round constants, R1CS metadata).
- per proof in: `z_packed` (+ a/b or generate on device), pinned H2D.
- per phase boundary: small transcript messages D2H, challenges H2D (KB-scale).
- per proof out: the proof bytes, D2H.
- **never**: codeword, Merkle tree, intermediate MLEs.

## Staged build order (piecemeal + compose, each oracle-gated)

- **S0 — Device scaffolding.** A `cudarc`- or C-ABI-based proof scope: stream +
  async mempool, pinned witness buffer, a device buffer type passed across
  phases, and the host↔device transcript bridge (serialize challenger state /
  squeeze challenges around each GPU phase). Smallest thing that can run the
  existing commit kernels under the resident model and reproduce the commit
  oracle. Establishes the interfaces every later phase plugs into.

- **S1 — GPU zerocheck (target 1, 31%).** The GF(2) sumcheck
  (`zerocheck::prove_packed_padded_capture_s_hat_v_c`). Mine the existing
  `cuda-ghash/bench_sumcheck.cu` / `bench_full_sumcheck.cu` against
  `src/zerocheck.rs`. CPU oracle: dump round polynomials + challenges + final
  claim from the real prover (like `dump_commit_vectors`), assert the device
  sumcheck transcript matches bit-for-bit. The univariate-skip round-1 and the
  eq-weighting are the correctness-risk spots (cf. `INSIGHTS.md` §2).

- **S2 — Compose commit + zerocheck resident.** Wire z_packed/codeword to live on
  device across both; transcript ping-pongs. First honest fused number; surfaces
  layout/transcript mismatches early with only two phases in play.

- **S3 — GPU pcs::open (target 2, 30%).** Ligerito recursive open reuses the NTT
  + Merkle kernels over the resident codeword (FRI fold = deepest-layers NTT in
  reverse + new Merkle trees per epoch). High leverage: most kernels exist.
  Oracle: the existing `commit_matches_full_ntt_oracle`-style cross-check plus a
  full proof that *verifies*.

- **S4 — GPU lincheck (target 3, 14%) + witness.** Fill in the remainder; then
  the whole prove is resident.

- **S5 — Integration + end-to-end gate.** Feature-gated `prove_cuda` with a
  CPU-vs-GPU byte-identical proof cross-check (same challenger seed ⇒ identical
  proof bytes), and the end-to-end throughput vs CPU with transfer included.

## Gotchas / open questions

- **Challenger consistency is the integration crux.** Every GPU phase must absorb
  its proof messages and squeeze challenges in the *exact* order the CPU prover
  does, or the Fiat-Shamir transcript diverges and the proof won't verify. The
  byte-identical cross-check (same seed ⇒ same proof) is the only real guard —
  build it early (S0/S2), not at the end.
- **Don't trust per-phase-isolated throughput.** Standalone numbers include H2D/D2H
  that the fused pipeline elides; commit is 19%, so isolated offload is
  Amdahl/PCIe-bound. Only the composed-resident number counts.
- **Binary-field tax persists.** GF(2¹²⁸) elements are 16 B / 4 shuffle-words —
  cross-thread exchange and any sumcheck fold over the extension field costs
  more than sp1's 32-bit KoalaBear. Profile sumcheck folds for this.
- **Layout discipline.** Keep the codeword SoA layout (`codeword[pos*num_ntts +
  lane]`) consistent commit→open so no transpose is needed between them; design
  zerocheck's a/b/c access to match the witness packing.
- **Reuse, don't re-port.** open's NTT fold and Merkle are the *same* kernels as
  commit — factor `ntt_f128.cuh` / `merkle.cuh` so both phases call them.
