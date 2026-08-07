// CPU-structured (shift-reduce + convert-table) GPU port of zerocheck round-1.
// Validated end-to-end in test_zc_round1_cpustyle (host Ref B). The point: do the
// outer F128 ghash only once per (outer-chunk, λ) instead of once per row — folding
// the 8 "small" dims in F8 (shift-then-reduce) and the 16 "medium" dims via a 64 KB
// convert table T[j][v]=γ^j·φ8(v). vs warp5's per-row eqB form this trades the 16
// eqB-shuffles/row for table lookups and cuts the ghash count ~8x.
//
// Layout: rows 2^(m-6) = 8 small(k) × 16 medium(j) × N_out(2^(m-13)). Column index
// col = o*128 + j*8 + k is contiguous, so one outer chunk = 128 contiguous columns.
// One WARP processes G outer chunks; lane owns output coords {lane, lane+32}.
// Accumulates raw = Σ_o eq_out[o]·chunk; the host/finalize scales by C_s·C_med.
#pragma once
#include "zerocheck_round1.cuh"   // F128 helpers, ZC_BASIS, g_zc_t0, g_zc_f8mul, ZC_WG

#ifndef ZC_CS_WG
#define ZC_CS_WG 2        // outer chunks accumulated serially per warp
#endif

// GF(2^8) product strategy for the k-loop. Discrete-log tables in shared are the
// measured DEFAULT on sm_120 (5090): round-1 1.59 -> 1.12 ms at m=29 — the kernel
// is shared-load-throughput bound (L1/LDS pipe 71%, 111M LDS instructions), so the
// 768 B log/antilog tables beat both the 64 KB L2 f8mul gather (1.59) and deferred
// clmad (-DZC_CS_CLMAD, 1.16). Define ZC_CS_F8MUL to get the L2-table path back.
// Re-confirmed on cpustyle3 at m=33: log 15.78, clmad 16.62, clmul 16.76, f8mul
// 24.70 ms. Staging the columns through s_scol also still pays (-DZC_CS_NOSCOL,
// which reads them straight from global, costs +9%).
#if !defined(ZC_CS_LOG) && !defined(ZC_CS_CLMUL) && !defined(ZC_CS_CLMAD) && !defined(ZC_CS_F8MUL)
#define ZC_CS_LOG
#endif

// Experiment (-DZC_T0_CONST): T0 in __constant__ memory. Measured 2.2x SLOWER than
// shared at m=29 — constant memory broadcasts but SERIALIZES the divergent (data-
// dependent) reads our extend gather makes. -DZC_T0_LDG (read-only cache) is 1.7x
// slower (L2-backed). Shared is the right home for a divergent 16 KB LUT gather.
#ifdef ZC_T0_CONST
__constant__ u64 c_zc_t0[256 * 8];
inline void zc_cpustyle_upload_const_t0() {
    cudaMemcpyToSymbol(c_zc_t0, g_zc_t0, 256 * 8 * sizeof(u64), 0, cudaMemcpyDeviceToDevice);
}
#endif

// Discrete-log GF(2^8) tables (-DZC_CS_LOG): A·B = antilog[log[A]+log[B]] (0-masked).
// 768 bytes total → live in shared (vs the 64 KB f8mul table in L2). Generator g=0x03;
// antilog sized 512 so log[A]+log[B] (≤508) needs no mod. Built on-device, once.
__device__ uint8_t d_zc_log[256];
__device__ uint8_t d_zc_antilog[512];
__global__ void zc_build_log_tables() {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t a = 1;
    for (int i = 0; i < 255; i++) {
        d_zc_antilog[i] = a; d_zc_antilog[i + 255] = a; d_zc_log[a] = (uint8_t)i;
        a = (uint8_t)((a << 1) ^ ((a >> 7) * 0x1b)) ^ a;   // a *= 0x03
    }
    d_zc_antilog[510] = 1; d_zc_antilog[511] = d_zc_antilog[1];
    d_zc_log[0] = 0;       // sentinel; masked out when either operand is 0
}

// GF(2^8) reduce of a <=15-bit poly, AES poly x^8+x^4+x^3+x+1 (matches Rust gf8_reduce).
__device__ __forceinline__ uint8_t zc_gf8_reduce(uint16_t p) {
    uint16_t h = p >> 8;
    uint16_t t = (p & 0xff) ^ h ^ (uint16_t)(h << 1) ^ (uint16_t)(h << 3) ^ (uint16_t)(h << 4);
    uint16_t h2 = t >> 8;
    return (uint8_t)((t & 0xff) ^ h2 ^ (uint16_t)(h2 << 1) ^ (uint16_t)(h2 << 3) ^ (uint16_t)(h2 << 4));
}

// Reduce a wider poly (deg < ~24, e.g. a deferred Σ_k clmul(A_k,B_k)·x^k) mod the AES
// poly via repeated 8-bit XOR-folds (x^8 ≡ x^4+x^3+x+1). 4 folds converge from deg<22.
__device__ __forceinline__ uint8_t zc_gf8_reduce_wide(u64 p) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
        u64 h = p >> 8;
        p = (p & 0xff) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    }
    return (uint8_t)p;
}

template <int W>
__global__ void zc_round1_cpustyle(const uint8_t* __restrict__ a_packed,
                                   const uint8_t* __restrict__ b_packed,
                                   const uint8_t* __restrict__ c_packed,
                                   const F128* __restrict__ eq_out,
                                   const u64* __restrict__ t0,
                                   const uint8_t* __restrict__ f8mul,
                                   const F128* __restrict__ convert,   // [16*256] γ^j·φ8(v)
                                   long long n_out, long long G,
                                   F128* __restrict__ raw_ab, F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];                 // 16 KB unified extend table
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;

    int wsrc = lane >> 3, wcol = lane & 7;        // lane<24 builds extend-word wcol of witness wsrc
    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};

    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        long long col_base = o * 128;
        for (int j = 0; j < 16; j++) {
            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                long long col = col_base + j * 8 + k;
                u64 a64 = *(const u64*)(a_packed + col * 8);
                u64 b64 = *(const u64*)(b_packed + col * 8);
                u64 c64 = *(const u64*)(c_packed + col * 8);
                u64 src = (wsrc == 0) ? a64 : (wsrc == 1) ? b64 : c64;
                u64 word = 0;
#ifdef ZC_CS_SKIP_EXT
                word = src;
#else
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
#endif
                u64 aw0 = __shfl_sync(0xffffffffu, word, wlo),     aw1 = __shfl_sync(0xffffffffu, word, whi);
                u64 bw0 = __shfl_sync(0xffffffffu, word, 8 + wlo), bw1 = __shfl_sync(0xffffffffu, word, 8 + whi);
                u64 cw0 = __shfl_sync(0xffffffffu, word, 16 + wlo),cw1 = __shfl_sync(0xffffffffu, word, 16 + whi);
                uint8_t aL = (uint8_t)(aw0 >> sh), aH = (uint8_t)(aw1 >> sh);
                uint8_t bL = (uint8_t)(bw0 >> sh), bH = (uint8_t)(bw1 >> sh);
                uint8_t cL = (uint8_t)(cw0 >> sh), cH = (uint8_t)(cw1 >> sh);
                aacc0 ^= (uint16_t)f8mul[(int)aL * 256 + bL] << k;
                aacc1 ^= (uint16_t)f8mul[(int)aH * 256 + bH] << k;
                cacc0 ^= (uint16_t)cL << k;
                cacc1 ^= (uint16_t)cH << k;
            }
#ifdef ZC_CS_SKIP_CONV
            chunk_ab0.lo ^= aacc0; chunk_ab1.lo ^= aacc1; chunk_c0.lo ^= cacc0; chunk_c1.lo ^= cacc1;
#else
            chunk_ab0 = f128_add(chunk_ab0, convert[j * 256 + zc_gf8_reduce(aacc0)]);
            chunk_ab1 = f128_add(chunk_ab1, convert[j * 256 + zc_gf8_reduce(aacc1)]);
            chunk_c0  = f128_add(chunk_c0,  convert[j * 256 + zc_gf8_reduce(cacc0)]);
            chunk_c1  = f128_add(chunk_c1,  convert[j * 256 + zc_gf8_reduce(cacc1)]);
#endif
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

// Variant: 32-lane cooperative extend. The cpustyle extend builds 24 words/column
// (a,b,c × 8) on lanes 0..23 — 8 lanes idle (~75% util). Here all 32 lanes build the
// 192 words of one j-group (8 columns) into a shared word-buffer (6 builds/lane), then
// Phase B reads the needed bytes from shared instead of __shfl. Cuts T0 lookups 1024 ->
// 768 per lane per outer chunk.
template <int W>
__global__ void zc_round1_cpustyle2(const uint8_t* __restrict__ a_packed,
                                    const uint8_t* __restrict__ b_packed,
                                    const uint8_t* __restrict__ c_packed,
                                    const F128* __restrict__ eq_out,
                                    const u64* __restrict__ t0,
                                    const uint8_t* __restrict__ f8mul,
                                    const F128* __restrict__ convert,
                                    long long n_out, long long G,
                                    F128* __restrict__ raw_ab, F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];                 // 16 KB
    __shared__ u64 s_scol[W * 24];                // 8 cols × 3 witnesses per warp
    __shared__ u64 s_wbuf[W * 192];               // 8 cols × 24 words per warp (1.5 KB/warp)
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64* scol = s_scol + wid * 24;
    u64* wbuf = s_wbuf + wid * 192;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;

    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};

    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        for (int j = 0; j < 16; j++) {
            long long base = (o * 128 + j * 8) * 8;        // byte offset of this j's 8 columns
            // preload 24 witness words (a:0..7, b:8..15, c:16..23) coalesced
            if (lane < 24) {
                const uint8_t* Wt = (lane < 8) ? a_packed : (lane < 16) ? b_packed : c_packed;
                int cl = lane & 7;
                scol[lane] = *(const u64*)(Wt + base + cl * 8);
            }
            __syncwarp();
            // Phase A: all 32 lanes build the 192 extend-words (6 builds/lane).
#pragma unroll
            for (int r = 0; r < 6; r++) {
                int bld = lane + 32 * r;               // 0..191
                int cl = bld / 24, win = bld - cl * 24, wt = win >> 3, wcol = win & 7;
                u64 src = scol[wt * 8 + cl];
                u64 word = 0;
#ifdef ZC_CS_SKIP_EXT
                word = src ^ s_t0[wcol];
#elif defined(ZC_T0_CONST)
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= c_zc_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
#elif defined(ZC_T0_LDG)
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= __ldg(&t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)]);
#else
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
#endif
                wbuf[cl * 24 + win] = word;
            }
            __syncwarp();
            // Phase B: shift-reduce over the 8 small dims, reading bytes from wbuf.
            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wbuf + k * 24;
                uint8_t aL = (uint8_t)(wc[wlo]      >> sh), aH = (uint8_t)(wc[whi]      >> sh);
                uint8_t bL = (uint8_t)(wc[8 + wlo]  >> sh), bH = (uint8_t)(wc[8 + whi]  >> sh);
                uint8_t cL = (uint8_t)(wc[16 + wlo] >> sh), cH = (uint8_t)(wc[16 + whi] >> sh);
                aacc0 ^= (uint16_t)f8mul[(int)aL * 256 + bL] << k;
                aacc1 ^= (uint16_t)f8mul[(int)aH * 256 + bH] << k;
                cacc0 ^= (uint16_t)cL << k;
                cacc1 ^= (uint16_t)cH << k;
            }
#ifdef ZC_CS_SKIP_CONV
            chunk_ab0.lo ^= aacc0; chunk_ab1.lo ^= aacc1; chunk_c0.lo ^= cacc0; chunk_c1.lo ^= cacc1;
#else
            chunk_ab0 = f128_add(chunk_ab0, convert[j * 256 + zc_gf8_reduce(aacc0)]);
            chunk_ab1 = f128_add(chunk_ab1, convert[j * 256 + zc_gf8_reduce(aacc1)]);
            chunk_c0  = f128_add(chunk_c0,  convert[j * 256 + zc_gf8_reduce(cacc0)]);
            chunk_c1  = f128_add(chunk_c1,  convert[j * 256 + zc_gf8_reduce(cacc1)]);
#endif
            __syncwarp();                              // wbuf reuse next j
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

// GHASH multiply-by-γ (γ=0x02): left shift by 1 bit, reduce by 0x87.
__device__ __forceinline__ F128 zc_mul_by_x(F128 z) {
    u64 mask = (u64)0 - (z.hi >> 63);
    return F128{ (z.lo << 1) ^ (0x87 & mask), (z.hi << 1) | (z.lo >> 63) };
}

// Variant 3: drop the 64 KB convert table. chunk = Σ_j γ^j·φ8(sn_j) is a reverse
// Horner fold in γ (γ-multiply = zc_mul_by_x, a shift — not a ghash), so we only need
// the 4 KB φ8 table in shared (keeps occupancy high). j-loop runs 15→0 so the fold
// fuses with no extra storage. Same extend (32-lane) as cpustyle2.
template <int W>
__global__ void zc_round1_cpustyle3(const uint8_t* __restrict__ a_packed,
                                    const uint8_t* __restrict__ b_packed,
                                    const uint8_t* __restrict__ c_packed,
                                    const F128* __restrict__ eq_out,
                                    const u64* __restrict__ t0,
                                    const uint8_t* __restrict__ f8mul,
                                    const F128* __restrict__ phi,
                                    long long n_out, long long G,
                                    F128* __restrict__ raw_ab, F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];                 // 16 KB
    __shared__ F128 s_phi[256];                   // 4 KB  (replaces the 64 KB convert table)
    __shared__ u64 s_scol[W * 24];
    __shared__ u64 s_wbuf[W * 192];
#ifdef ZC_CS_LOG
    __shared__ uint8_t s_log[256];
    __shared__ uint8_t s_antilog[512];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_log[i] = d_zc_log[i];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_antilog[i] = d_zc_antilog[i];
#endif
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_phi[i] = phi[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64* scol = s_scol + wid * 24;
    u64* wbuf = s_wbuf + wid * 192;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;

    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};

    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        for (int j = 15; j >= 0; j--) {            // reverse for Horner
            long long col0 = o * 128 + j * 8;
#ifndef ZC_CS_NOSCOL
            long long base = col0 * 8;
            if (lane < 24) {
                const uint8_t* Wt = (lane < 8) ? a_packed : (lane < 16) ? b_packed : c_packed;
                int cl = lane & 7;
                scol[lane] = *(const u64*)(Wt + base + cl * 8);
            }
            __syncwarp();
#endif
#pragma unroll
            for (int r = 0; r < 6; r++) {
                int bld = lane + 32 * r;
                int cl = bld / 24, win = bld - cl * 24, wt = win >> 3, wcol = win & 7;
#ifdef ZC_CS_NOSCOL
                const uint8_t* Wt = (wt == 0) ? a_packed : (wt == 1) ? b_packed : c_packed;
                u64 src = *(const u64*)(Wt + (col0 + cl) * 8);
#else
                u64 src = scol[wt * 8 + cl];
#endif
                u64 word = 0;
#ifdef ZC_CS_SKIP_EXT
                word = src ^ s_t0[wcol];
#else
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
#endif
                wbuf[cl * 24 + win] = word;
            }
            __syncwarp();
#if defined(ZC_CS_CLMAD)
            // Deferred clmad: Â·B̂·x^k = clmul(Â, B̂<<k) (carryless mult is linear in
            // shifts), fused-accumulated unreduced via the native clmad, reduced ONCE
            // per j. No 64 KB f8mul table. C is linear (just shifts).
            u64 aacc0 = 0, aacc1 = 0; uint16_t cacc0 = 0, cacc1 = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wbuf + k * 24;
                u64 aL = (uint8_t)(wc[wlo] >> sh), aH = (uint8_t)(wc[whi] >> sh);
                u64 bL = (uint8_t)(wc[8 + wlo] >> sh), bH = (uint8_t)(wc[8 + whi] >> sh);
                aacc0 = clmad_lo(aL, bL << k, aacc0);
                aacc1 = clmad_lo(aH, bH << k, aacc1);
                cacc0 ^= (uint16_t)(uint8_t)(wc[16 + wlo] >> sh) << k;
                cacc1 ^= (uint16_t)(uint8_t)(wc[16 + whi] >> sh) << k;
            }
            chunk_ab0 = f128_add(zc_mul_by_x(chunk_ab0), s_phi[zc_gf8_reduce_wide(aacc0)]);
            chunk_ab1 = f128_add(zc_mul_by_x(chunk_ab1), s_phi[zc_gf8_reduce_wide(aacc1)]);
            chunk_c0  = f128_add(zc_mul_by_x(chunk_c0),  s_phi[zc_gf8_reduce(cacc0)]);
            chunk_c1  = f128_add(zc_mul_by_x(chunk_c1),  s_phi[zc_gf8_reduce(cacc1)]);
#else
            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wbuf + k * 24;
                uint8_t aL = (uint8_t)(wc[wlo]      >> sh), aH = (uint8_t)(wc[whi]      >> sh);
                uint8_t bL = (uint8_t)(wc[8 + wlo]  >> sh), bH = (uint8_t)(wc[8 + whi]  >> sh);
                uint8_t cL = (uint8_t)(wc[16 + wlo] >> sh), cH = (uint8_t)(wc[16 + whi] >> sh);
#if defined(ZC_CS_LOG)
                // A·B = antilog[log A + log B], 0-masked; 768 B tables in shared.
                uint8_t p0 = s_antilog[(int)s_log[aL] + s_log[bL]] & (uint8_t)(-(int)(aL && bL));
                uint8_t p1 = s_antilog[(int)s_log[aH] + s_log[bH]] & (uint8_t)(-(int)(aH && bH));
                aacc0 ^= (uint16_t)p0 << k;
                aacc1 ^= (uint16_t)p1 << k;
#elif defined(ZC_CS_CLMUL)
                aacc0 ^= (uint16_t)zc_gf8_reduce((uint16_t)clmul_lo((u64)aL, (u64)bL)) << k;
                aacc1 ^= (uint16_t)zc_gf8_reduce((uint16_t)clmul_lo((u64)aH, (u64)bH)) << k;
#else
                aacc0 ^= (uint16_t)f8mul[(int)aL * 256 + bL] << k;
                aacc1 ^= (uint16_t)f8mul[(int)aH * 256 + bH] << k;
#endif
                cacc0 ^= (uint16_t)cL << k;
                cacc1 ^= (uint16_t)cH << k;
            }
            chunk_ab0 = f128_add(zc_mul_by_x(chunk_ab0), s_phi[zc_gf8_reduce(aacc0)]);
            chunk_ab1 = f128_add(zc_mul_by_x(chunk_ab1), s_phi[zc_gf8_reduce(aacc1)]);
            chunk_c0  = f128_add(zc_mul_by_x(chunk_c0),  s_phi[zc_gf8_reduce(cacc0)]);
            chunk_c1  = f128_add(zc_mul_by_x(chunk_c1),  s_phi[zc_gf8_reduce(cacc1)]);
#endif
            __syncwarp();
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

__global__ void zc_cpustyle_finalize(F128* ab, F128* c, F128 scale);   // fwd decl

// cpustyle3db: software-pipelined / double-buffered variant of cpustyle3. wbuf & scol
// are double-buffered so each reverse-j iteration BUILDs j-1 into the other buffer while
// CONSUMEing j from the current one — the build's shared T0 loads overlap the consume's
// ALU, and the per-j barrier drops from 2 syncwarp to 1. Costs 2x wbuf/scol smem, so W
// must be lower to stay resident (occupancy vs pipelining trade).
template <int W>
__global__ void zc_round1_cpustyle3db(const uint8_t* __restrict__ a_packed,
                                      const uint8_t* __restrict__ b_packed,
                                      const uint8_t* __restrict__ c_packed,
                                      const F128* __restrict__ eq_out,
                                      const u64* __restrict__ t0,
                                      const uint8_t* __restrict__ f8mul,
                                      const F128* __restrict__ phi,
                                      long long n_out, long long G,
                                      F128* __restrict__ raw_ab, F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];
    __shared__ F128 s_phi[256];
    __shared__ u64 s_scol[W * 2 * 24];
    __shared__ u64 s_wbuf[W * 2 * 192];
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_phi[i] = phi[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;
    u64* scol2 = s_scol + wid * 2 * 24;       // scol2[buf*24 + i]
    u64* wbuf2 = s_wbuf + wid * 2 * 192;       // wbuf2[buf*192 + i]

    // preload + build extend-words for column-group `j` into buffer `buf`.
    auto preload = [&](int buf, long long col0) {
        if (lane < 24) {
            const uint8_t* Wt = (lane < 8) ? a_packed : (lane < 16) ? b_packed : c_packed;
            scol2[buf * 24 + lane] = *(const u64*)(Wt + (col0 + (lane & 7)) * 8);
        }
    };
    auto build = [&](int buf) {
        u64* sc = scol2 + buf * 24; u64* wb = wbuf2 + buf * 192;
#pragma unroll
        for (int r = 0; r < 6; r++) {
            int bld = lane + 32 * r;
            int cl = bld / 24, win = bld - cl * 24, wt = win >> 3, wcol = win & 7;
            u64 src = sc[wt * 8 + cl];
            u64 word = 0;
#pragma unroll
            for (int b = 0; b < 8; b++)
                word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
            wb[cl * 24 + win] = word;
        }
    };

    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        long long cb = o * 128;
        preload(0, cb + 15 * 8); __syncwarp(); build(0);     // prologue: j=15 -> buf 0
        for (int j = 15; j >= 0; j--) {
            int cur = (15 - j) & 1, nxt = cur ^ 1, jn = j - 1;
            if (jn >= 0) preload(nxt, cb + jn * 8);
            __syncwarp();                       // wbuf2[cur] built (prev), scol2[nxt] loaded
            if (jn >= 0) build(nxt);            // build j-1 (writes nxt) — overlaps consume(cur)
            const u64* wb = wbuf2 + cur * 192;  // consume j (reads cur)
            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wb + k * 24;
                uint8_t aL = (uint8_t)(wc[wlo] >> sh), aH = (uint8_t)(wc[whi] >> sh);
                uint8_t bL = (uint8_t)(wc[8 + wlo] >> sh), bH = (uint8_t)(wc[8 + whi] >> sh);
                aacc0 ^= (uint16_t)f8mul[(int)aL * 256 + bL] << k;
                aacc1 ^= (uint16_t)f8mul[(int)aH * 256 + bH] << k;
                cacc0 ^= (uint16_t)(uint8_t)(wc[16 + wlo] >> sh) << k;
                cacc1 ^= (uint16_t)(uint8_t)(wc[16 + whi] >> sh) << k;
            }
            chunk_ab0 = f128_add(zc_mul_by_x(chunk_ab0), s_phi[zc_gf8_reduce(aacc0)]);
            chunk_ab1 = f128_add(zc_mul_by_x(chunk_ab1), s_phi[zc_gf8_reduce(aacc1)]);
            chunk_c0  = f128_add(zc_mul_by_x(chunk_c0),  s_phi[zc_gf8_reduce(cacc0)]);
            chunk_c1  = f128_add(zc_mul_by_x(chunk_c1),  s_phi[zc_gf8_reduce(cacc1)]);
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

inline void launch_zc_round1_cpustyle3db(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                         const F128* d_eq_out, long long n_out,
                                         const F128* d_convert, F128 scale,
                                         F128* d_round1_ab, F128* d_round1_c) {
#ifndef ZC_CS3DB_W
#define ZC_CS3DB_W 8
#endif
    constexpr int W = ZC_CS3DB_W;
    long long G = ZC_CS_WG;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_cpustyle3db<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_f8mul,
                                                 g_zc_phi, n_out, G, d_round1_ab, d_round1_c);
    zc_cpustyle_finalize<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}

// Multiply all 64 outputs by the global scale C_s·C_med (= eq_small[0]·eq_med[0]).
__global__ void zc_cpustyle_finalize(F128* ab, F128* c, F128 scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 64) { ab[i] = ghash_mul_karatsuba(ab[i], scale); c[i] = ghash_mul_karatsuba(c[i], scale); }
}

// d_eq_out: eq(r[13..m]), length n_out=2^(m-13). d_convert: 16*256 F128. scale: C_s·C_med.
inline void launch_zc_round1_cpustyle(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                      const F128* d_eq_out, long long n_out,
                                      const F128* d_convert, F128 scale,
                                      F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;
    long long G = ZC_CS_WG;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_cpustyle<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_f8mul,
                                              d_convert, n_out, G, d_round1_ab, d_round1_c);
    zc_cpustyle_finalize<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}

inline void launch_zc_round1_cpustyle2(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                       const F128* d_eq_out, long long n_out,
                                       const F128* d_convert, F128 scale,
                                       F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;
    long long G = ZC_CS_WG;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
#ifdef ZC_T0_CONST
    static bool s_up = false; if (!s_up) { zc_cpustyle_upload_const_t0(); s_up = true; }
#endif
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_cpustyle2<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_f8mul,
                                               d_convert, n_out, G, d_round1_ab, d_round1_c);
    zc_cpustyle_finalize<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}

// Drop-in replacement for launch_zc_round1 (same signature: eq_full, rows). Uses the fast
// cpustyle3 kernel. Trick: cpustyle3 computes Σ_o eq_out[o]·chunk then ×(C_s·C_med); but
// eq_full[o*128] = C_s·C_med·eq_out[o] already, so we feed the stride-128 subsample of
// eq_full and scale=1 — no r, no convert table needed. Tables come from
// zc_round1_upload_tables (g_zc_t0, g_zc_phi, g_zc_f8mul), already called by the pipeline.
inline void launch_zc_round1_cpustyle3(const uint8_t*, const uint8_t*, const uint8_t*,
                                       const F128*, long long, const F128*, F128, F128*, F128*); // fwd
__global__ void zc_eq_stride128(const F128* __restrict__ eq_full, F128* __restrict__ eq_out,
                                long long n_out) {
    long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (o < n_out) eq_out[o] = eq_full[o * 128];
}
inline void launch_zc_round1_fast(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                  const F128* d_eq_full, long long rows,
                                  F128* d_round1_ab, F128* d_round1_c) {
    long long n_out = rows >> 7;                 // rows / 128 = 2^(m-13)
    static F128* s_eqs = nullptr; static long long s_cap = 0;
    if (n_out > s_cap) { if (s_eqs) cudaFree(s_eqs); cudaMalloc(&s_eqs, n_out * sizeof(F128)); s_cap = n_out; }
    int t = 256;
    zc_eq_stride128<<<(int)((n_out + t - 1) / t), t>>>(d_eq_full, s_eqs, n_out);
    launch_zc_round1_cpustyle3(d_a, d_b, d_c, s_eqs, n_out, /*convert*/ nullptr,
                               F128{1, 0}, d_round1_ab, d_round1_c);
}

// cpustyle3: φ8-in-shared + Horner; d_convert is ignored (kept for a drop-in signature).
inline void launch_zc_round1_cpustyle3(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                       const F128* d_eq_out, long long n_out,
                                       const F128* d_convert, F128 scale,
                                       F128* d_round1_ab, F128* d_round1_c) {
// Warps per block. 14, not the 16 that fits the 48 KB static-shared ceiling: at 72
// registers/thread a second block per SM needs 2·W·32·72 <= 65536, i.e. W <= 14.22.
// W=16 clears the shared limit but misses the register file, so it runs ONE block
// per SM (16 warps) where W=14 runs two (28 warps) — measured -8% round-1 at m=29,
// 32 and 33 alike, and W=15 falls straight back to W=16's time. Re-derive this if
// the kernel's register count moves; the cliff is sharp on both sides.
#ifndef ZC_CS3_W
#define ZC_CS3_W 14
#endif
    constexpr int W = ZC_CS3_W;
#ifdef ZC_CS_LOG
    static bool s_log_up = false; if (!s_log_up) { zc_build_log_tables<<<1, 1>>>(); s_log_up = true; }
#endif
    long long G = ZC_CS_WG;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_cpustyle3<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_f8mul,
                                               g_zc_phi, n_out, G, d_round1_ab, d_round1_c);
    zc_cpustyle_finalize<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}
