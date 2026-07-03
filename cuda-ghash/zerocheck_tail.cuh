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

__global__ void zt_msg_combine(const F128* p1, const F128* pinf, int blocks,
                               F128* m1, F128* minf) {
    F128 a1{0, 0}, ai{0, 0};
    for (int b = 0; b < blocks; b++) { a1 = f128_add(a1, p1[b]); ai = f128_add(ai, pinf[b]); }
    *m1 = a1; *minf = ai;
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
    zt_msg_combine<<<1, 1>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
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
    zt_msg_combine<<<1, 1>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
}

// Fused fold(A,B by r, len -> len/2 into Ao,Bo) + next message over the folded data
// weighted by dEq (length out_pairs = len/4). Leaves (g_one,g_inf) in d_m1/d_minf.
inline void launch_zt_fold_msg(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                               const F128* dEq, long long out_pairs, F128 r,
                               F128* d_p1, F128* d_pinf, F128* d_m1, F128* d_minf) {
    int blocks = zt_blocks(out_pairs);
    zt_fold_msg_partial<<<blocks, ZT_TPB>>>(dA, dB, dAo, dBo, dEq, out_pairs, r, d_p1, d_pinf);
    zt_msg_combine<<<1, 1>>>(d_p1, d_pinf, blocks, d_m1, d_minf);
}
