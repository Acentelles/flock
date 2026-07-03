// Row-batch fold kernel — step 2 of the GPU pcs::open (Ligerito) port
// (GPU_OPEN_PLAN.md). Direct port of the fused single-pass row-batch fold in
//   src/pcs/basefold.rs::row_batch_fold_all  (= row_batch_fold_one, :268)
// which collapses each codeword position's `num_ntts = 2^log_batch_size`
// contiguous lanes (SoA: codeword[pos*num_ntts + lane]) down to a single F128,
// in `log_batch_size` rounds of `buf[j] = u + r·(u + v)` with u = buf[2j],
// v = buf[2j+1]. NOTE: no twiddle here (unlike the FRI fold) — the row-batch
// challenges are pure sumcheck folds. Reads the codeword once: the per-position
// lane stack is folded in registers/local memory, only buf[0] is written.
//
// One thread per OUTPUT position. The multiply is `ghash_mul_karatsuba` (the
// fastest on this GPU, bench_f128); adds are XOR (`f128_add`).
#pragma once
#include "f128.cuh"
#include "ntt_host.hpp"   // F128,u64 (from f128.cuh); shares the build conventions

// Max lanes a single thread folds = 2^MAX_LOG_BATCH. 64 covers log_batch_size
// up to 6 (typical is 5 → 32 lanes). Per-thread local F128 buf[MAX_LANES].
#ifndef RBF_MAX_LANES
#define RBF_MAX_LANES 64
#endif

// Collapse each position's `num_ntts` lanes to out[k] via `n_chal` row-batch
// folds. `chal` is a device array of `n_chal` challenges (round order).
__global__ void row_batch_fold_kernel(const F128* __restrict__ in,
                                      F128* __restrict__ out,
                                      const F128* __restrict__ chal,
                                      int n_chal, int num_ntts,
                                      long long n_positions) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_positions) return;

    F128 buf[RBF_MAX_LANES];
    long long base = k * (long long)num_ntts;
    for (int l = 0; l < num_ntts; l++) buf[l] = in[base + l];

    int len = num_ntts;
    for (int c = 0; c < n_chal; c++) {
        F128 r = chal[c];
        int half = len >> 1;
        for (int j = 0; j < half; j++) {
            F128 u = buf[2 * j];
            F128 v = buf[2 * j + 1];
            buf[j] = f128_add(u, ghash_mul_karatsuba(r, f128_add(u, v)));
        }
        len = half;
    }
    out[k] = buf[0];
}

inline void launch_row_batch_fold(const F128* d_in, F128* d_out, const F128* d_chal,
                                  int n_chal, int num_ntts, long long n_positions,
                                  int tpb = 256) {
    long long blocks = (n_positions + tpb - 1) / tpb;
    row_batch_fold_kernel<<<(unsigned)blocks, tpb>>>(d_in, d_out, d_chal, n_chal,
                                                     num_ntts, n_positions);
}
