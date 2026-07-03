# cuda-ghash — GF(2¹²⁸) GHASH on the GPU with `clmad`

A CUDA port of flare-avx's `src/field/gf2_128.rs`, using NVIDIA's native
carryless multiply-add instruction **`clmad`** (PTX ISA 9.3; SASS
`CLMAD.LO`/`CLMAD.HI` on Blackwell `sm_120`). It validates bit-for-bit against
the real `flare` implementation and benchmarks several multiply strategies.

## Field

GF(2¹²⁸) in GHASH form, irreducible `p = x¹²⁸ + x⁷ + x² + x + 1`, layout
`lo = x⁰..x⁶³`, `hi = x⁶⁴..x¹²⁷` — identical to the Rust `F128`.

`clmad` maps onto it naturally: a 64×64→128 carryless product is one
`clmad.hi` + one `clmad.lo`, and GHASH's pervasive cross-term/reduction XORs
fold into `clmad`'s free `^ c` operand. See `f128.cuh`.

## Files

| File | Purpose |
|------|---------|
| `f128.cuh` | `F128`, `clmad` wrappers, `ghash_reduce`, and four multiplies: `binius`, `schoolbook` (both clmad-fused), `software` (shift-XOR baseline), plus deferred `mul_unreduced`. |
| `test_f128.cu` | Loads `vectors.bin` and checks every device variant **bit-for-bit vs flare** + mutual agreement. |
| `bench_f128.cu` | Latency / throughput / GHASH-GCM benchmarks (mirrors `benches/field.rs`). |
| `../crates/flock-prover/src/bin/dump_ghash_vectors.rs` | Emits `vectors.bin` from the real `flock::field::F128`. |

## Requirements

- A `clmad`-capable `ptxas` (CUDA 13.3 build `V13.3.33`+ on this machine works).
- **Always compile AOT** (`-gencode arch=compute_120,code=sm_120`, as in the
  Makefile): `ptxas` assembles `clmad` → SASS at build time, so the GPU driver's
  PTX-JIT version is irrelevant. Do **not** rely on runtime PTX JIT.
- An NVIDIA Blackwell GPU (`sm_120`); `clmad` itself needs `sm_80`+.

## Usage

```bash
make test     # build + regenerate flare vectors + run correctness check
make bench    # build + run benchmarks
make sass     # confirm the hot loops emit native CLMAD instructions
make clean
```

`make test` runs `cargo run --bin dump_ghash_vectors` from the repo root, so it
needs the Rust toolchain. Run `cargo bench --bench field` for this host's CPU
(AVX PCLMULQDQ) numbers to compare against the GPU.
