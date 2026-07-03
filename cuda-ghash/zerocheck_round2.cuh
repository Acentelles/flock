// Zerocheck round-2 fold-at-z on GPU — port of the fold half of
// src/zerocheck/multilinear.rs::uni_skip_fold_and_round_pair_optimized_packed.
// Folds the packed witness a/b at the URM challenge z (over the skip domain)
// into a_mlv/b_mlv (F128, length 2^(m-6)):
//   a_mlv[row] = Σ_{j=0..8} foldtable[j*256 + a_packed[row*8 + j]]   (UniSkipFoldTable)
// The first multilinear message is then the eq-weighted deg-2 message over
// (a_mlv, b_mlv) — reuse zerocheck_tail.cuh::launch_zt_msg.
#pragma once
#include "f128.cuh"

#ifndef ZR2_TPB
#define ZR2_TPB 256
#endif

// One thread per output row: 8 byte-lookups into the 8×256 F128 fold table.
__global__ void zc_round2_fold(const uint8_t* __restrict__ a_packed,
                               const uint8_t* __restrict__ b_packed,
                               const F128* __restrict__ foldtable, long long n_out,
                               F128* __restrict__ a_mlv, F128* __restrict__ b_mlv) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out) return;
    const uint8_t* ar = a_packed + row * 8;
    const uint8_t* br = b_packed + row * 8;
    F128 av{0, 0}, bv{0, 0};
#pragma unroll
    for (int j = 0; j < 8; j++) {
        av = f128_add(av, foldtable[j * 256 + ar[j]]);
        bv = f128_add(bv, foldtable[j * 256 + br[j]]);
    }
    a_mlv[row] = av;
    b_mlv[row] = bv;
}

inline void launch_zc_round2_fold(const uint8_t* d_a, const uint8_t* d_b, const F128* d_foldtable,
                                  long long n_out, F128* d_a_mlv, F128* d_b_mlv) {
    long long blocks = (n_out + ZR2_TPB - 1) / ZR2_TPB;
    zc_round2_fold<<<(unsigned)blocks, ZR2_TPB>>>(d_a, d_b, d_foldtable, n_out, d_a_mlv, d_b_mlv);
}
