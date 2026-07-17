// a·b multilinear sumcheck kernels — step 3 of the GPU pcs::open (Ligerito)
// port (GPU_OPEN_PLAN.md). The degree-2 sumcheck of S = Σ_x a(x)·b(x) that
// the Ligerito prover runs over `(f, combined_basis)`
// (`src/pcs/ligerito.rs`'s `SumcheckProver` / `fold_and_msg_lsb`).
//
// Per round, over the CURRENT a,b with ADJACENT pairing (a[2j], a[2j+1]) —
// matching the CPU prover (NOT the strided (i, i+half) layout in bench_full_sumcheck):
//   message:  u_0 = Σ_j a[2j]·b[2j]                     (= u(0))
//             u_2 = Σ_j (a[2j]+a[2j+1])·(b[2j]+b[2j+1]) (= u(∞), leading coeff)
//   fold:     a'[j] = a[2j] + r·(a[2j]+a[2j+1])  (and b)
// The middle coeff is recovered by the verifier from the running claim, so only
// {u_0, u_2} are produced (the CPU `SumcheckMessage`).
//
// The message is a global reduction: reduce-per-term (F128 accumulate). Deferred
// reduction (F256, reduce once) was measured a wash-to-slight-loss on this GPU
// and doubles the reduction-tree's shared memory, so plain F128 is used. Two-pass
// per round (message reduce, then fold) for correctness-first clarity; fusing the next
// round's message into the fold (as `fold_and_msg_lsb` does) is the later optimization.
#pragma once
#include "f128.cuh"

#ifndef SMC_TPB
#define SMC_TPB 256
#endif

// Block-partial message reduction (adjacent pairing). Grid-stride so the
// launched block count can be capped; each block writes one (p0, p2) F128.
__global__ void sumcheck_msg_partial(const F128* __restrict__ A,
                                     const F128* __restrict__ B,
                                     long long half, F128* p0, F128* p2) {
    // Reduce-per-term (F128) rather than deferred (F256): halves shared memory
    // (better occupancy) and is a measured wash-to-win on this GPU, since
    // ghash_reduce pipelines behind the CLMAD multiply. Bit-identical.
    __shared__ F128 s0[SMC_TPB];
    __shared__ F128 s2[SMC_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 e0{0, 0}, e2{0, 0};
    for (long long j = t; j < half; j += stride) {
        F128 a0 = A[2 * j], a1 = A[2 * j + 1];
        F128 b0 = B[2 * j], b1 = B[2 * j + 1];
        e0 = f128_add(e0, ghash_mul_karatsuba(a0, b0));
        e2 = f128_add(e2, ghash_mul_karatsuba(f128_add(a0, a1), f128_add(b0, b1)));
    }
    int x = threadIdx.x;
    s0[x] = e0; s2[x] = e2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s0[x] = f128_add(s0[x], s0[x + s]); s2[x] = f128_add(s2[x], s2[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p0[blockIdx.x] = s0[0]; p2[blockIdx.x] = s2[0]; }
}

// Combine block partials → u_0, u_2. One 256-thread block: the single-thread
// loop this replaces cost ~200 us at 2048 blocks — same order as the partial
// kernel itself. XOR order is irrelevant → bit-identical.
__global__ void sumcheck_msg_combine(const F128* p0, const F128* p2, int blocks,
                                     F128* u0, F128* u2) {
    __shared__ F128 s0[SMC_TPB];
    __shared__ F128 s2[SMC_TPB];
    F128 a0{0, 0}, a2{0, 0};
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) { a0 = f128_add(a0, p0[b]); a2 = f128_add(a2, p2[b]); }
    int x = threadIdx.x;
    s0[x] = a0; s2[x] = a2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s0[x] = f128_add(s0[x], s0[x + s]); s2[x] = f128_add(s2[x], s2[x + s]); }
        __syncthreads();
    }
    if (x == 0) { *u0 = s0[0]; *u2 = s2[0]; }
}

// Fold a,b by r (adjacent pairing), ping-pong: out[j] from in[2j],in[2j+1].
__global__ void sumcheck_fold(const F128* __restrict__ A, const F128* __restrict__ B,
                              F128* __restrict__ Ao, F128* __restrict__ Bo,
                              long long half, F128 r) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    F128 a0 = A[2 * j], a1 = A[2 * j + 1];
    F128 b0 = B[2 * j], b1 = B[2 * j + 1];
    Ao[j] = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));
    Bo[j] = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
}

// FUSED fold + next-round message (ligerito's fold_and_msg_lsb). One pass over
// (A,B): fold by r into (Ao,Bo), AND accumulate the message of the FOLDED arrays
// (= the next round's {u_0,u_2}) — so A,B are read once per round instead of
// twice (separate message pass eliminated). Each thread handles one output PAIR
// (Ao[2j],Ao[2j+1]) from inputs A[4j..4j+4]; out_pairs = half/2 (half=folded len).
// Requires half>=2 (even); the lone half==1 tail uses sumcheck_fold + zero msg.
__global__ void sumcheck_fold_msg_partial(const F128* __restrict__ A, const F128* __restrict__ B,
                                          F128* __restrict__ Ao, F128* __restrict__ Bo,
                                          long long out_pairs, F128 r, F128* p0, F128* p2) {
    __shared__ F128 s0[SMC_TPB];
    __shared__ F128 s2[SMC_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 e0{0, 0}, e2{0, 0};
    for (long long j = t; j < out_pairs; j += stride) {
        long long i = 4 * j;
        F128 a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
        F128 b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
        F128 af0 = f128_add(a0, ghash_mul_karatsuba(r, f128_add(a0, a1)));  // fold pair 2j
        F128 af1 = f128_add(a2, ghash_mul_karatsuba(r, f128_add(a2, a3)));  // fold pair 2j+1
        F128 bf0 = f128_add(b0, ghash_mul_karatsuba(r, f128_add(b0, b1)));
        F128 bf1 = f128_add(b2, ghash_mul_karatsuba(r, f128_add(b2, b3)));
        Ao[2 * j] = af0; Ao[2 * j + 1] = af1; Bo[2 * j] = bf0; Bo[2 * j + 1] = bf1;
        e0 = f128_add(e0, ghash_mul_karatsuba(af0, bf0));                                  // u_0 over folded
        e2 = f128_add(e2, ghash_mul_karatsuba(f128_add(af0, af1), f128_add(bf0, bf1)));    // u_2 over folded
    }
    int x = threadIdx.x;
    s0[x] = e0; s2[x] = e2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s0[x] = f128_add(s0[x], s0[x + s]); s2[x] = f128_add(s2[x], s2[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p0[blockIdx.x] = s0[0]; p2[blockIdx.x] = s2[0]; }
}

// Host driver for one round's message: returns (u_0, u_2) on device. `d_p0`,
// `d_p2` are scratch of length >= SMC_MAX_BLOCKS.
#ifndef SMC_MAX_BLOCKS
#define SMC_MAX_BLOCKS 2048
#endif

inline int sumcheck_blocks(long long half) {
    long long b = (half + SMC_TPB - 1) / SMC_TPB;
    if (b < 1) b = 1;
    if (b > SMC_MAX_BLOCKS) b = SMC_MAX_BLOCKS;
    return (int)b;
}

inline void launch_sumcheck_msg(const F128* dA, const F128* dB, long long half,
                                F128* d_p0, F128* d_p2, F128* d_u0, F128* d_u2) {
    int blocks = sumcheck_blocks(half);
    sumcheck_msg_partial<<<blocks, SMC_TPB>>>(dA, dB, half, d_p0, d_p2);
    sumcheck_msg_combine<<<1, SMC_TPB>>>(d_p0, d_p2, blocks, d_u0, d_u2);
}

inline void launch_sumcheck_fold(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                                 long long half, F128 r) {
    long long blocks = (half + SMC_TPB - 1) / SMC_TPB;
    sumcheck_fold<<<(unsigned)blocks, SMC_TPB>>>(dA, dB, dAo, dBo, half, r);
}

// Fused fold-by-r + next-round message in one pass. Folds (dA,dB)→(dAo,dBo) of
// length `half`, and leaves the FOLDED arrays' message in (d_u0,d_u2). half>=2.
inline void launch_sumcheck_fold_msg(const F128* dA, const F128* dB, F128* dAo, F128* dBo,
                                     long long half, F128 r,
                                     F128* d_p0, F128* d_p2, F128* d_u0, F128* d_u2) {
    if (half < 2) {  // folded length <2 → message is empty (0,0); just fold the tail.
        long long b = (half + SMC_TPB - 1) / SMC_TPB; if (b < 1) b = 1;
        sumcheck_fold<<<(unsigned)b, SMC_TPB>>>(dA, dB, dAo, dBo, half, r);
        cudaMemset(d_u0, 0, sizeof(F128)); cudaMemset(d_u2, 0, sizeof(F128));
        return;
    }
    long long out_pairs = half >> 1;
    int blocks = sumcheck_blocks(out_pairs);
    sumcheck_fold_msg_partial<<<blocks, SMC_TPB>>>(dA, dB, dAo, dBo, out_pairs, r, d_p0, d_p2);
    sumcheck_msg_combine<<<1, SMC_TPB>>>(d_p0, d_p2, blocks, d_u0, d_u2);
}
