// Zerocheck multilinear sumcheck tail on GPU — port of
// src/zerocheck/multilinear.rs::round_pair_naive (message) + fold_in_place_pair
// (fold). The message is the eq-weighted degree-2 form (adjacent pairing):
//   g_one = Σ_x eq[x]·a[2x+1]·b[2x+1]
//   g_inf = Σ_x eq[x]·(a[2x]+a[2x+1])·(b[2x]+b[2x+1])
//   message = (r[0]·g_one, g_inf)            (r[0]=ONE in zerocheck → msg_1=g_one)
// eq = build_eq(r[1..]). The fold a[x]=a[2x]+ρ·(a[2x+1]+a[2x]) is the same
// adjacent-pair LSB fold as sumcheck_ab.cuh::sumcheck_fold (reused).
#pragma once
#include "f128.cuh"
#include "sumcheck_ab.cuh"   // sumcheck_fold / launch_sumcheck_fold (adjacent LSB fold)

#ifndef ZT_TPB
#define ZT_TPB 256
#endif
#ifndef ZT_MAX_BLOCKS
#define ZT_MAX_BLOCKS 2048
#endif

// Block-partial eq-weighted message reduction (adjacent pairing). Grid-stride.
__global__ void zt_msg_partial(const F128* __restrict__ A, const F128* __restrict__ B,
                               const F128* __restrict__ eq, long long half,
                               F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < half; x += stride) {
        F128 a0 = A[2 * x], a1 = A[2 * x + 1];
        F128 b0 = B[2 * x], b1 = B[2 * x + 1];
        F128 e = eq[x];
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(a1, b1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(a0, a1), f128_add(b0, b1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}

// Parallel partials reduction (one 256-thread block). The single-thread loop
// this replaces cost ~200 us at 2048 blocks — same order as the big message
// kernels themselves. XOR-sum order is irrelevant, so this is bit-identical.
__global__ void zt_msg_combine(const F128* p1, const F128* pinf, int blocks,
                               F128* m1, F128* minf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    F128 a1{0, 0}, ai{0, 0};
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) { a1 = f128_add(a1, p1[b]); ai = f128_add(ai, pinf[b]); }
    int x = threadIdx.x;
    s1[x] = a1; sinf[x] = ai;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { *m1 = s1[0]; *minf = sinf[0]; }
}

// FUSED fold-by-r + next eq-weighted message in ONE pass. Folds A,B (length len) ->
// Ao,Bo (length len/2) by r, and simultaneously computes the next round's message
// (g_one,g_inf) over the folded data weighted by eq (length out_pairs=len/4). Saves a
// whole fold kernel + a data pass per tail round vs separate launch_sumcheck_fold +
// launch_zt_msg. Each thread owns one output message-pair x: reads A[4x..4x+3], folds
// to Ao[2x]=af0, Ao[2x+1]=af1, accumulates eq[x]·(af1·bf1) and eq[x]·(af0+af1)(bf0+bf1).
__global__ void zt_fold_msg_partial(const F128* __restrict__ A, const F128* __restrict__ B,
                                    F128* __restrict__ Ao, F128* __restrict__ Bo,
                                    const F128* __restrict__ eq, long long out_pairs, F128 r,
                                    F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < out_pairs; x += stride) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));   // folded nA[2x]
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));   // folded nA[2x+1]
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = eq[x];
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}

// Incremental eq fold: eq(r[j+1..m])[y] = eq(r[j..m])[2y]·(1+r[j])^{-1}. Halves a length-2n
// eq table to length-n (gather even entries, scale) — replaces a full per-round rebuild in
// the tail (eq tables are nested). inv_scale = (1+r[j])^{-1}, host-precomputed.
__global__ void eq_halve_scale_k(const F128* __restrict__ in, F128* __restrict__ out,
                                 long long n, F128 inv_scale) {
    long long y = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= n) return;
    out[y] = ghash_mul_karatsuba(in[2 * y], inv_scale);
}
inline void launch_eq_halve_scale(const F128* d_in, F128* d_out, long long n, F128 inv_scale, int tpb = 256) {
    eq_halve_scale_k<<<(unsigned)((n + tpb - 1) / tpb), tpb>>>(d_in, d_out, n, inv_scale);
}

inline int zt_blocks(long long half) {
    long long b = (half + ZT_TPB - 1) / ZT_TPB;
    if (b < 1) b = 1;
    if (b > ZT_MAX_BLOCKS) b = ZT_MAX_BLOCKS;
    return (int)b;
}

// One round's eq-weighted message over (dA,dB) with eq table dEq (length half).
// Leaves (g_one, g_inf) in d_m1/d_minf. r[0]=ONE in zerocheck, so msg_1 = g_one.
inline void launch_zt_msg(const F128* dA, const F128* dB, const F128* dEq, long long half,
                          F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(half);
    zt_msg_partial<<<blocks, ZT_TPB>>>(dA, dB, dEq, half, d_p1, d_pinf);
    zt_msg_combine<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
}

// Device-rho variant: reads the fold challenge r from *r_ptr (device) instead of a host
// scalar — for the RESIDENT tail where rho is produced on-device by the challenger kernel.
__global__ void zt_fold_msg_partial_dev(const F128* __restrict__ A, const F128* __restrict__ B,
                                        F128* __restrict__ Ao, F128* __restrict__ Bo,
                                        const F128* __restrict__ eq, long long out_pairs,
                                        const F128* __restrict__ r_ptr, F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    F128 r = *r_ptr;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < out_pairs; x += stride) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = eq[x];
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}
inline void launch_zt_fold_msg_dev(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                                   const F128* dEq, long long out_pairs, const F128* d_r,
                                   F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(out_pairs);
    zt_fold_msg_partial_dev<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEq, out_pairs, d_r, d_p1, d_pinf);
    zt_msg_combine<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
}

// Fused fold(A,B by r, len -> len/2 into Ao,Bo) + next message over the folded data
// weighted by dEq (length out_pairs = len/4). Leaves (g_one,g_inf) in d_m1/d_minf.
inline void launch_zt_fold_msg(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                               const F128* dEq, long long out_pairs, F128 r,
                               F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(out_pairs);
    zt_fold_msg_partial<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEq, out_pairs, r, d_p1, d_pinf);
    zt_msg_combine<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
}

// ---- SPLIT-EQ tail (port of the CPU SplitEqGhash structure) ----
//
// The per-round eq table eq(r[7+k..m]) is never materialized. Build once:
//   eqlo = build_eq(r[7 .. 7+lobits])       (2^lobits entries, lobits = (m-7)-7)
//   eqhi = build_eq(r[7+lobits .. m])       (<= 2^7 = 128 entries)
// Then for the round that has dropped the k lowest of these vars (k=0 is the
// round-2 message, tail round i has k=i+1), with z = y << k:
//   eq(r[7+k..m])[y] = S_k · eqlo[z & (2^lobits-1)] · eqhi[z >> lobits]
// because build_eq puts a (1+r_j) factor at every zero bit, so the shifted-in
// low zero bits contribute exactly Π_{j=7}^{6+k}(1+r_j) — cancelled by the host
// scalar S_k = Π_{j=7}^{6+k}(1+r_j)^{-1}, applied once to the two message sums
// in the combine kernel (GF(2^128) is exact, so this is bit-identical to the
// materialized eq). Kills the per-round eq_halve_scale pass and the full-size
// eq stream in the message: eqlo/eqhi are small enough to live in L2.
__device__ __forceinline__ F128 zt_split_eq(const F128* __restrict__ eqlo,
                                            const F128* __restrict__ eqhi,
                                            long long z, int lobits) {
    return ghash_mul_karatsuba(eqlo[z & ((1LL << lobits) - 1)], eqhi[z >> lobits]);
}

__global__ void zt_msg_combine_scaled(const F128* p1, const F128* pinf, int blocks,
                                      F128 scale, F128* m1, F128* minf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    F128 a1{0, 0}, ai{0, 0};
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) { a1 = f128_add(a1, p1[b]); ai = f128_add(ai, pinf[b]); }
    int x = threadIdx.x;
    s1[x] = a1; sinf[x] = ai;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { *m1 = ghash_mul_karatsuba(s1[0], scale); *minf = ghash_mul_karatsuba(sinf[0], scale); }
}

__global__ void zt_msg_partial_split(const F128* __restrict__ A, const F128* __restrict__ B,
                                     const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                     int shift, int lobits, long long half,
                                     F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < half; x += stride) {
        F128 a0 = A[2 * x], a1 = A[2 * x + 1];
        F128 b0 = B[2 * x], b1 = B[2 * x + 1];
        F128 e = zt_split_eq(eqlo, eqhi, x << shift, lobits);
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(a1, b1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(a0, a1), f128_add(b0, b1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}

__global__ void zt_fold_msg_partial_split(const F128* __restrict__ A, const F128* __restrict__ B,
                                          F128* __restrict__ Ao, F128* __restrict__ Bo,
                                          const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                          int shift, int lobits, long long out_pairs, F128 r,
                                          F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < out_pairs; x += stride) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = zt_split_eq(eqlo, eqhi, x << shift, lobits);
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}

// Device-rho split variant (for the resident on-device-challenger tail).
__global__ void zt_fold_msg_partial_split_dev(const F128* __restrict__ A, const F128* __restrict__ B,
                                              F128* __restrict__ Ao, F128* __restrict__ Bo,
                                              const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                              int shift, int lobits, long long out_pairs,
                                              const F128* __restrict__ r_ptr,
                                              F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    F128 r = *r_ptr;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 g1{0, 0}, ginf{0, 0};
    for (long long x = t; x < out_pairs; x += stride) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = zt_split_eq(eqlo, eqhi, x << shift, lobits);
        g1 = f128_add(g1, ghash_mul_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        ginf = f128_add(ginf, ghash_mul_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = g1; sinf[x] = ginf;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p1[blockIdx.x] = s1[0]; pinf[blockIdx.x] = sinf[0]; }
}

// ---- hi-factored variants (the CPU SplitEqGhash inner-loop structure) ----
//
// When each block's contiguous point range maps to a SINGLE eqhi entry, the
// per-point mul is eqlo only; the block's partial sums are multiplied by that
// one eqhi value at the end of the shared-memory reduction (2 muls per block
// instead of 1 per point). Distributivity over the XOR-sum is exact in
// GF(2^128), so this is bit-identical to the per-point form. These kernels use
// contiguous per-block chunks (not grid-stride): block b owns points
// [b·chunk, (b+1)·chunk); zt_hib_plan checks the single-eqhi condition
// (power-of-two chunk with chunk·2^shift ≤ 2^lobits ⇒ every chunk lies inside
// one eqhi segment) — callers fall back to the per-point kernels otherwise
// (only the tiny latency-bound rounds).
inline bool zt_hib_plan(long long n, int shift, int lobits, int blocks, long long& chunk) {
    if (n <= 0 || (n & (n - 1))) return false;
    chunk = (n + blocks - 1) / blocks;
    if (chunk & (chunk - 1)) return false;
    return (chunk << shift) <= (1LL << lobits);
}

__global__ void zt_msg_partial_split_hib(const F128* __restrict__ A, const F128* __restrict__ B,
                                         const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                         int shift, int lobits, long long half, long long chunk,
                                         F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long start = (long long)blockIdx.x * chunk;
    long long end = start + chunk < half ? start + chunk : half;
    F256 g1{0, 0, 0, 0}, ginf{0, 0, 0, 0};   // deferred reduction (ghash_reduce is F2-linear)
    for (long long x = start + threadIdx.x; x < end; x += blockDim.x) {
        F128 a0 = A[2 * x], a1 = A[2 * x + 1];
        F128 b0 = B[2 * x], b1 = B[2 * x + 1];
        F128 e = eqlo[(x << shift) & ((1LL << lobits) - 1)];
        f256_xor(g1, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(a1, b1)));
        f256_xor(ginf, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(a0, a1), f128_add(b0, b1))));
    }
    int x = threadIdx.x;
    s1[x] = f256_reduce(g1); sinf[x] = f256_reduce(ginf);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) {
        if (start < half) {
            F128 eh = eqhi[(start << shift) >> lobits];
            p1[blockIdx.x] = ghash_mul_karatsuba(s1[0], eh);
            pinf[blockIdx.x] = ghash_mul_karatsuba(sinf[0], eh);
        } else { p1[blockIdx.x] = F128{0, 0}; pinf[blockIdx.x] = F128{0, 0}; }
    }
}

__global__ void zt_fold_msg_partial_split_hib(const F128* __restrict__ A, const F128* __restrict__ B,
                                              F128* __restrict__ Ao, F128* __restrict__ Bo,
                                              const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                              int shift, int lobits, long long out_pairs, long long chunk,
                                              F128 r, F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    long long start = (long long)blockIdx.x * chunk;
    long long end = start + chunk < out_pairs ? start + chunk : out_pairs;
    F256 g1{0, 0, 0, 0}, ginf{0, 0, 0, 0};   // deferred reduction (ghash_reduce is F2-linear)
    for (long long x = start + threadIdx.x; x < end; x += blockDim.x) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = eqlo[(x << shift) & ((1LL << lobits) - 1)];
        f256_xor(g1, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        f256_xor(ginf, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = f256_reduce(g1); sinf[x] = f256_reduce(ginf);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) {
        if (start < out_pairs) {
            F128 eh = eqhi[(start << shift) >> lobits];
            p1[blockIdx.x] = ghash_mul_karatsuba(s1[0], eh);
            pinf[blockIdx.x] = ghash_mul_karatsuba(sinf[0], eh);
        } else { p1[blockIdx.x] = F128{0, 0}; pinf[blockIdx.x] = F128{0, 0}; }
    }
}

__global__ void zt_fold_msg_partial_split_hib_dev(const F128* __restrict__ A, const F128* __restrict__ B,
                                                  F128* __restrict__ Ao, F128* __restrict__ Bo,
                                                  const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                                  int shift, int lobits, long long out_pairs, long long chunk,
                                                  const F128* __restrict__ r_ptr, F128* p1, F128* pinf) {
    __shared__ F128 s1[ZT_TPB];
    __shared__ F128 sinf[ZT_TPB];
    F128 r = *r_ptr;
    long long start = (long long)blockIdx.x * chunk;
    long long end = start + chunk < out_pairs ? start + chunk : out_pairs;
    F256 g1{0, 0, 0, 0}, ginf{0, 0, 0, 0};   // deferred reduction (ghash_reduce is F2-linear)
    for (long long x = start + threadIdx.x; x < end; x += blockDim.x) {
        long long i = 4 * x;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * x] = af0; Ao[2 * x + 1] = af1; Bo[2 * x] = bf0; Bo[2 * x + 1] = bf1;
        F128 e = eqlo[(x << shift) & ((1LL << lobits) - 1)];
        f256_xor(g1, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(af1, bf1)));
        f256_xor(ginf, mul_unreduced_karatsuba(e, ghash_mul_karatsuba(
                                  f128_add(af0, af1), f128_add(bf0, bf1))));
    }
    int x = threadIdx.x;
    s1[x] = f256_reduce(g1); sinf[x] = f256_reduce(ginf);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s1[x] = f128_add(s1[x], s1[x + s]); sinf[x] = f128_add(sinf[x], sinf[x + s]); }
        __syncthreads();
    }
    if (x == 0) {
        if (start < out_pairs) {
            F128 eh = eqhi[(start << shift) >> lobits];
            p1[blockIdx.x] = ghash_mul_karatsuba(s1[0], eh);
            pinf[blockIdx.x] = ghash_mul_karatsuba(sinf[0], eh);
        } else { p1[blockIdx.x] = F128{0, 0}; pinf[blockIdx.x] = F128{0, 0}; }
    }
}

inline void launch_zt_msg_split(const F128* dA, const F128* dB,
                                const F128* dEqLo, const F128* dEqHi, int shift, int lobits,
                                long long half, F128 scale,
                                F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(half);
    long long chunk;
    if (zt_hib_plan(half, shift, lobits, blocks, chunk))
        zt_msg_partial_split_hib<<<blocks, ZT_TPB>>>(dA, dB, dEqLo, dEqHi, shift, lobits, half, chunk,
                                                     d_p1, d_pinf);
    else
        zt_msg_partial_split<<<blocks, ZT_TPB>>>(dA, dB, dEqLo, dEqHi, shift, lobits, half, d_p1, d_pinf);
    zt_msg_combine_scaled<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, scale, d_m1, d_minf);
}

inline void launch_zt_fold_msg_split(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                                     const F128* dEqLo, const F128* dEqHi, int shift, int lobits,
                                     long long out_pairs, F128 r, F128 scale,
                                     F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(out_pairs);
    long long chunk;
    if (zt_hib_plan(out_pairs, shift, lobits, blocks, chunk))
        zt_fold_msg_partial_split_hib<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEqLo, dEqHi, shift, lobits,
                                                          out_pairs, chunk, r, d_p1, d_pinf);
    else
        zt_fold_msg_partial_split<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEqLo, dEqHi, shift, lobits,
                                                      out_pairs, r, d_p1, d_pinf);
    zt_msg_combine_scaled<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, scale, d_m1, d_minf);
}

inline void launch_zt_fold_msg_split_dev(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                                         const F128* dEqLo, const F128* dEqHi, int shift, int lobits,
                                         long long out_pairs, const F128* d_r, F128 scale,
                                         F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(out_pairs);
    long long chunk;
    if (zt_hib_plan(out_pairs, shift, lobits, blocks, chunk))
        zt_fold_msg_partial_split_hib_dev<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEqLo, dEqHi, shift, lobits,
                                                              out_pairs, chunk, d_r, d_p1, d_pinf);
    else
        zt_fold_msg_partial_split_dev<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEqLo, dEqHi, shift, lobits,
                                                          out_pairs, d_r, d_p1, d_pinf);
    zt_msg_combine_scaled<<<1, ZT_TPB>>>(d_p1, d_pinf, blocks, scale, d_m1, d_minf);
}
