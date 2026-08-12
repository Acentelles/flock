// Zerocheck round-1 (univariate-skip URM message) on GPU — validated bit-exact
// against the real optimized prove_packed round-1 (shift-reduce + C_s).
// All forms compute the SAME linear algebra (M, φ8 BASIS, f8mul, eq_full),
// reorganized for the GPU. `launch_zc_round1` -> the warp3 winner.
//
// Bake-off at m=29 on RTX 5090 (CPU optimized prover ~5 ms):
//   groups (zc_round1_groups)       22.8 ms  — 64 outputs split 16/thread, 254 regs,
//                                              ~8% occupancy: register-wall bound.
//   warp  (per-byte extend)          9.6 ms  — warp-coop, lane owns coords {L,L+32},
//                                              4 accum regs, eqB computed once/warp.
//   warp2 (coop-word extend)         5.9 ms  — 24 lanes build one extend-u64 each.
//   warp3 (nibble-LUT extend)        3.9 ms  — extend = 16 nibble lookups (16 KB shared).
//   warp4 (per-coord φ8+ghash)       8.6 ms  — drops eqB shuffles but clmul-bound.
//   warp5 (unified byte LUT)         3.7 ms  — WINNER. Cauchy single-table collapse
//                                              (docs/urm_optimizations.tex): 8 byte
//                                              lookups from one 16 KB table.
//
// Why the groups form loses on GPU: it keeps all 128 F128 output accumulators in
// registers (254 regs -> low occupancy) and the canonical CPU FLOP-minimization
// (2^(m-13) heavy threads) starves the GPU. The warp forms hold only 4 F128
// accumulators/lane (coords {L, L+32}), compute eqB = eq_x·φ8(2^k) once per warp
// (8 ghash, broadcast via __shfl), and do extend cooperatively. See git history
// for the original CPU-strategy variant.
//
// k_skip=6, ell=64, rows=2^(m-6). Output: round1_ab[64], round1_c[64] (F128).
//   for x_rest in 0..rows:
//     A_Λ,B_Λ,C_Λ = extend(64 boolean skip bits)         # S→Λ via M (boolean XOR)
//     eq_x = eq_full[x_rest]                              # eq(r[6..m], x_rest)
//     for i in 0..64: p_ab[i] += eq_x·φ8(A·B); p_c[i] += eq_x·φ8(C)
// φ8 is F2-linear: eq_x·φ8(v) = ⊕_{k: bit k of v} (eq_x·φ8(2^k)) = ⊕ eqB[k], so the
// per-element ghash becomes 8 ghash/row (eqB) + XORs (F8 = GF(2^8), 256-wide).
#pragma once
#include "f128.cuh"
#include "ntt_host.hpp"   // host f128 math for build helpers (unused on device)
#include <cstdint>
#include <vector>

#ifndef ZC_TPB
#define ZC_TPB 128
#endif

__device__ __constant__ u64 ZC_M[64 * 8];     // 64 columns × 8 u64 (64 F8 bytes)
__device__ __constant__ F128 ZC_BASIS[8];     // φ8(2^k), k=0..8
static uint8_t* g_zc_f8mul = nullptr;          // [a*256 + b] = f8mul(a,b)
// Transposed M for the warp kernel: ZC_MT[s*64 + i] = byte i of column s
// (= the contribution of input skip-bit s to output coordinate i). So
// extend(row) byte i = XOR_{s: bit s of row} ZC_MT[s*64 + i].
static uint8_t* g_zc_mt = nullptr;
static u64* g_zc_nib = nullptr;        // nibble LUT, 16*16*8 u64 (16 KB)
static F128* g_zc_phi = nullptr;       // φ8: F8->F128, 256 entries (4 KB)
static u64* g_zc_t0 = nullptr;         // unified byte LUT (Cauchy collapse), 256*8 u64 (16 KB)
static int g_zc_t0_ok = 0;             // 1 iff M has the Cauchy structure (single-table valid)

inline cudaError_t zc_round1_upload_tables(const uint8_t* mcol, const uint8_t* f8mul,
                                           const F128* phi8_256) {
    u64 mpacked[64 * 8];
    uint8_t mt[64 * 64];
    for (int s = 0; s < 64; s++)
        for (int w = 0; w < 8; w++) {
            u64 v = 0;
            for (int t = 0; t < 8; t++) v |= (u64)mcol[s * 64 + w * 8 + t] << (8 * t);
            mpacked[s * 8 + w] = v;
        }
    for (int s = 0; s < 64; s++)
        for (int i = 0; i < 64; i++)
            mt[s * 64 + i] = (uint8_t)(mpacked[s * 8 + (i >> 3)] >> (8 * (i & 7)));
    cudaError_t err = cudaMemcpyToSymbol(ZC_M, mpacked, sizeof(mpacked));
    if (err != cudaSuccess) return err;
    F128 basis[8];
    for (int k = 0; k < 8; k++) basis[k] = phi8_256[1 << k];
    err = cudaMemcpyToSymbol(ZC_BASIS, basis, sizeof(basis));
    if (err != cudaSuccess) return err;
    if (!g_zc_f8mul) {
        err = cudaMalloc(&g_zc_f8mul, (size_t)256 * 256);
        if (err != cudaSuccess) return err;
    }
    err = cudaMemcpy(g_zc_f8mul, f8mul, (size_t)256 * 256, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    if (!g_zc_mt) {
        err = cudaMalloc(&g_zc_mt, sizeof(mt));
        if (err != cudaSuccess) return err;
    }
    err = cudaMemcpy(g_zc_mt, mt, sizeof(mt), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    // Nibble (4-Russians) table: NIB[np][v][w] = XOR of M columns (4*np .. 4*np+3)
    // selected by nibble bits v, word w. extend word w = XOR_np NIB[np][nibble_np][w].
    u64 nib[16 * 16 * 8];
    for (int np = 0; np < 16; np++)
        for (int v = 0; v < 16; v++)
            for (int w = 0; w < 8; w++) {
                u64 acc = 0;
                for (int bit = 0; bit < 4; bit++)
                    if ((v >> bit) & 1) acc ^= mpacked[(np * 4 + bit) * 8 + w];
                nib[(np * 16 + v) * 8 + w] = acc;
            }
    if (!g_zc_nib) {
        err = cudaMalloc(&g_zc_nib, sizeof(nib));
        if (err != cudaSuccess) return err;
    }
    err = cudaMemcpy(g_zc_nib, nib, sizeof(nib), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    if (!g_zc_phi) {
        err = cudaMalloc(&g_zc_phi, 256 * sizeof(F128));
        if (err != cudaSuccess) return err;
    }
    err = cudaMemcpy(g_zc_phi, phi8_256, 256 * sizeof(F128), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // Unified single-table collapse (Cauchy structure, docs/urm_optimizations.tex
    // §Single-table collapse): the additive-NTT M satisfies M[i', 8b+t] = M[i'^8b, t],
    // so the 8 per-byte-position tables collapse to one 256-entry base table built
    // from columns 0..7, with the output index XOR-shifted by 8b at lookup time.
    // For lane-owned word w: extend word w = XOR_{b=0..7} T0[a_b * 8 + (w ^ b)].
    auto Mbyte = [&](int ip, int j) -> uint8_t {        // byte i' of column j
        return (uint8_t)(mpacked[j * 8 + (ip >> 3)] >> (8 * (ip & 7)));
    };
    g_zc_t0_ok = 1;
    for (int ip = 0; ip < 64 && g_zc_t0_ok; ip++)
        for (int b = 0; b < 8; b++)
            for (int t = 0; t < 8; t++)
                if (Mbyte(ip, 8 * b + t) != Mbyte(ip ^ (8 * b), t)) { g_zc_t0_ok = 0; break; }
    u64 t0[256 * 8];                                    // T0[v*8 + w] = word w of (XOR_{t:bit t of v} column_t)
    for (int v = 0; v < 256; v++)
        for (int w = 0; w < 8; w++) {
            u64 acc = 0;
            for (int t = 0; t < 8; t++)
                if ((v >> t) & 1) acc ^= mpacked[t * 8 + w];
            t0[v * 8 + w] = acc;
        }
    if (!g_zc_t0) {
        err = cudaMalloc(&g_zc_t0, sizeof(t0));
        if (err != cudaSuccess) return err;
    }
    return cudaMemcpy(g_zc_t0, t0, sizeof(t0), cudaMemcpyHostToDevice);
}

__device__ __forceinline__ void zc_extend(const uint8_t* row8, u64 out[8]) {
#pragma unroll
    for (int w = 0; w < 8; w++) out[w] = 0;
#pragma unroll
    for (int byte_idx = 0; byte_idx < 8; byte_idx++) {
        uint8_t bits = row8[byte_idx];
        while (bits) {
            int s = byte_idx * 8 + (__ffs(bits) - 1);
#pragma unroll
            for (int w = 0; w < 8; w++) out[w] ^= ZC_M[s * 8 + w];
            bits &= bits - 1;
        }
    }
}
__device__ __forceinline__ uint8_t zc_byte(const u64 v[8], int i) {
    return (uint8_t)(v[i >> 3] >> (8 * (i & 7)));
}

// Pass 1: each thread owns ZC_NI=16 of the 64 outputs (register-resident), with
// ZC_NBLK=4 threads per row-group covering all 64.
#ifndef ZC_NI
#define ZC_NI 16
#endif
#ifndef ZC_NBLK
#define ZC_NBLK 4
#endif
__global__ void zc_round1_groups(const uint8_t* __restrict__ a_packed,
                                 const uint8_t* __restrict__ b_packed,
                                 const uint8_t* __restrict__ c_packed,
                                 const F128* __restrict__ eq_full,
                                 const uint8_t* __restrict__ f8mul,
                                 long long rows, long long spg, long long G,
                                 F128* __restrict__ partials) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long g = tid / ZC_NBLK;
    if (g >= G) return;
    int i0 = (int)(tid % ZC_NBLK) * ZC_NI;

    F128 pab[ZC_NI], pc[ZC_NI];
#pragma unroll
    for (int j = 0; j < ZC_NI; j++) { pab[j] = F128{0, 0}; pc[j] = F128{0, 0}; }

    long long s0 = g * spg, s1 = s0 + spg;
    if (s1 > rows) s1 = rows;
    for (long long row = s0; row < s1; row++) {
        u64 A[8], B[8], C[8];
        zc_extend(a_packed + row * 8, A);
        zc_extend(b_packed + row * 8, B);
        zc_extend(c_packed + row * 8, C);
        F128 eq_x = eq_full[row];
        F128 eqB[8];
#pragma unroll
        for (int k = 0; k < 8; k++) eqB[k] = ghash_mul_karatsuba(eq_x, ZC_BASIS[k]);
#pragma unroll
        for (int j = 0; j < ZC_NI; j++) {
            int i = i0 + j;
            uint8_t ai = zc_byte(A, i), bi = zc_byte(B, i), ci = zc_byte(C, i);
            uint8_t ab = f8mul[(int)ai * 256 + bi];
            F128 sab{0, 0}, sc{0, 0};
#pragma unroll
            for (int k = 0; k < 8; k++) {
                u64 ma = (u64)0 - (u64)((ab >> k) & 1);
                sab.lo ^= eqB[k].lo & ma; sab.hi ^= eqB[k].hi & ma;
                u64 mc = (u64)0 - (u64)((ci >> k) & 1);
                sc.lo ^= eqB[k].lo & mc; sc.hi ^= eqB[k].hi & mc;
            }
            pab[j] = f128_add(pab[j], sab);
            pc[j] = f128_add(pc[j], sc);
        }
    }
    F128* out = partials + g * 128 + i0;
#pragma unroll
    for (int j = 0; j < ZC_NI; j++) { out[j] = pab[j]; out[64 + j] = pc[j]; }
}

#define ZC_RED_TPB 256
__global__ void zc_round1_reduce(const F128* __restrict__ partials, long long G,
                                 F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ F128 sh[ZC_RED_TPB];
    int col = blockIdx.x, tid = threadIdx.x;
    F128 acc{0, 0};
    for (long long g = tid; g < G; g += blockDim.x) acc = f128_add(acc, partials[g * 128 + col]);
    sh[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] = f128_add(sh[tid], sh[tid + s]); __syncthreads(); }
    if (tid == 0) { if (col < 64) round1_ab[col] = sh[0]; else round1_c[col - 64] = sh[0]; }
}

// ============================================================================
// Warp-cooperative form. Lane L (0..31) owns output coords {L, L+32}, holding
// only 4 F128 accumulators in registers (vs 64 in the groups kernel -> 254 regs
// -> 8% occupancy). Per row: eqB[8] = eq_x·φ8(2^k) is computed ONCE per warp
// (lanes 0..7, one ghash each) and broadcast via __shfl (8 ghash/warp/row, not
// 32); extend is done per-output-byte from the transposed M in shared memory.
// This trades a little redundant extend arithmetic for ~6x occupancy.
#ifndef ZC_WG
#define ZC_WG 64          // rows accumulated serially per warp before atomics
#endif
template <int W>
__global__ void zc_round1_warp(const uint8_t* __restrict__ a_packed,
                               const uint8_t* __restrict__ b_packed,
                               const uint8_t* __restrict__ c_packed,
                               const F128* __restrict__ eq_full,
                               const uint8_t* __restrict__ mt,
                               const uint8_t* __restrict__ f8mul,
                               long long rows, long long G,
                               F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ uint8_t s_mt[64 * 64];
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) s_mt[i] = mt[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > rows) o1 = rows;
    if (o0 >= o1) return;

    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        u64 a64 = *(const u64*)(a_packed + o * 8);
        u64 b64 = *(const u64*)(b_packed + o * 8);
        u64 c64 = *(const u64*)(c_packed + o * 8);
        // eqB[k] = eq_x · φ8(2^k): lanes 0..7 each do one ghash, then broadcast.
        F128 eq_x = eq_full[o];
        F128 my = (lane < 8) ? ghash_mul_karatsuba(eq_x, ZC_BASIS[lane]) : F128{0, 0};
        F128 eqB[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            eqB[k].lo = __shfl_sync(0xffffffffu, my.lo, k);
            eqB[k].hi = __shfl_sync(0xffffffffu, my.hi, k);
        }
        // per-output-byte extend for this lane's two coords (L and L+32).
        uint8_t aL = 0, aH = 0, bL = 0, bH = 0, cL = 0, cH = 0;
#pragma unroll
        for (int s = 0; s < 64; s++) {
            uint8_t ma = (uint8_t)(0u - (unsigned)((a64 >> s) & 1));
            uint8_t mb = (uint8_t)(0u - (unsigned)((b64 >> s) & 1));
            uint8_t mc = (uint8_t)(0u - (unsigned)((c64 >> s) & 1));
            const uint8_t* row = s_mt + s * 64;
            uint8_t tL = row[lane], tH = row[lane + 32];
            aL ^= tL & ma; aH ^= tH & ma;
            bL ^= tL & mb; bH ^= tH & mb;
            cL ^= tL & mc; cH ^= tH & mc;
        }
        uint8_t ab0 = f8mul[(int)aL * 256 + bL], ab1 = f8mul[(int)aH * 256 + bH];
        // accumulate φ8(ab)·eq and φ8(c)·eq via the eqB basis masks.
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u64 m;
            m = (u64)0 - (u64)((ab0 >> k) & 1); pab0.lo ^= eqB[k].lo & m; pab0.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((ab1 >> k) & 1); pab1.lo ^= eqB[k].lo & m; pab1.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cL  >> k) & 1); pc0.lo  ^= eqB[k].lo & m; pc0.hi  ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cH  >> k) & 1); pc1.lo  ^= eqB[k].lo & m; pc1.hi  ^= eqB[k].hi & m;
        }
    }
    atomicXor((unsigned long long*)&round1_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&round1_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&round1_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&round1_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&round1_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&round1_c[lane + 32].hi, pc1.hi);
}

// Variant: cooperative-word extend. The per-byte extend above costs ~64 byte-MACs
// per output coord per lane (it dominates). Instead, lanes 0..23 each compute ONE
// full u64 extend-word (a:0..7, b:8..15, c:16..23) by XOR-ing M columns (in shared
// as u64), then the 6 bytes each lane needs (coords L, L+32 of a,b,c) are gathered
// via __shfl. ~3x less extend work, all u64 ops.
template <int W>
__global__ void zc_round1_warp2(const uint8_t* __restrict__ a_packed,
                                const uint8_t* __restrict__ b_packed,
                                const uint8_t* __restrict__ c_packed,
                                const F128* __restrict__ eq_full,
                                const uint8_t* __restrict__ f8mul,
                                long long rows, long long G,
                                F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ u64 s_zcm[64 * 8];                 // M columns as u64 (4 KB)
    for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) s_zcm[i] = ZC_M[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > rows) o1 = rows;
    if (o0 >= o1) return;

    // lane 0..23 compute extend-word `wword` of witness `wsrc` (0=a,1=b,2=c).
    int wsrc = lane >> 3, wcol = lane & 7;       // valid for lane < 24
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        u64 a64 = *(const u64*)(a_packed + o * 8);
        u64 b64 = *(const u64*)(b_packed + o * 8);
        u64 c64 = *(const u64*)(c_packed + o * 8);
        F128 eq_x = eq_full[o];
        F128 my = (lane < 8) ? ghash_mul_karatsuba(eq_x, ZC_BASIS[lane]) : F128{0, 0};
        F128 eqB[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            eqB[k].lo = __shfl_sync(0xffffffffu, my.lo, k);
            eqB[k].hi = __shfl_sync(0xffffffffu, my.hi, k);
        }
        // each of lanes 0..23 builds its extend-word
        u64 src = (wsrc == 0) ? a64 : (wsrc == 1) ? b64 : c64;
        u64 word = 0;
#pragma unroll
        for (int s = 0; s < 64; s++) {
            u64 m = (u64)0 - ((src >> s) & 1);
            word ^= s_zcm[s * 8 + wcol] & m;
        }
        // gather the 6 bytes this lane needs. coord L -> A byte L = byte(L&7) of
        // a-word(L>>3) on lane (L>>3); coord L+32 -> a-word((L>>3)+4) on lane +4.
        int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
        u64 aw0 = __shfl_sync(0xffffffffu, word, wlo);
        u64 aw1 = __shfl_sync(0xffffffffu, word, whi);
        u64 bw0 = __shfl_sync(0xffffffffu, word, 8 + wlo);
        u64 bw1 = __shfl_sync(0xffffffffu, word, 8 + whi);
        u64 cw0 = __shfl_sync(0xffffffffu, word, 16 + wlo);
        u64 cw1 = __shfl_sync(0xffffffffu, word, 16 + whi);
        uint8_t aL = (uint8_t)(aw0 >> sh), aH = (uint8_t)(aw1 >> sh);
        uint8_t bL = (uint8_t)(bw0 >> sh), bH = (uint8_t)(bw1 >> sh);
        uint8_t cL = (uint8_t)(cw0 >> sh), cH = (uint8_t)(cw1 >> sh);
        uint8_t ab0 = f8mul[(int)aL * 256 + bL], ab1 = f8mul[(int)aH * 256 + bH];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u64 m;
            m = (u64)0 - (u64)((ab0 >> k) & 1); pab0.lo ^= eqB[k].lo & m; pab0.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((ab1 >> k) & 1); pab1.lo ^= eqB[k].lo & m; pab1.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cL  >> k) & 1); pc0.lo  ^= eqB[k].lo & m; pc0.hi  ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cH  >> k) & 1); pc1.lo  ^= eqB[k].lo & m; pc1.hi  ^= eqB[k].hi & m;
        }
    }
    atomicXor((unsigned long long*)&round1_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&round1_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&round1_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&round1_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&round1_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&round1_c[lane + 32].hi, pc1.hi);
}

// Variant 3: cooperative-word extend via a nibble (4-Russians) LUT in shared.
// extend-word = XOR over 16 nibble lookups instead of 64 column XORs.
template <int W>
__global__ void zc_round1_warp3(const uint8_t* __restrict__ a_packed,
                                const uint8_t* __restrict__ b_packed,
                                const uint8_t* __restrict__ c_packed,
                                const F128* __restrict__ eq_full,
                                const u64* __restrict__ nib,
                                const uint8_t* __restrict__ f8mul,
                                long long rows, long long G,
                                F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ u64 s_nib[16 * 16 * 8];            // 16 KB
    for (int i = threadIdx.x; i < 16 * 16 * 8; i += blockDim.x) s_nib[i] = nib[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > rows) o1 = rows;
    if (o0 >= o1) return;

    int wsrc = lane >> 3, wcol = lane & 7;        // lane < 24 builds a word
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        u64 a64 = *(const u64*)(a_packed + o * 8);
        u64 b64 = *(const u64*)(b_packed + o * 8);
        u64 c64 = *(const u64*)(c_packed + o * 8);
        F128 eq_x = eq_full[o];
        F128 my = (lane < 8) ? ghash_mul_karatsuba(eq_x, ZC_BASIS[lane]) : F128{0, 0};
        F128 eqB[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            eqB[k].lo = __shfl_sync(0xffffffffu, my.lo, k);
            eqB[k].hi = __shfl_sync(0xffffffffu, my.hi, k);
        }
        u64 src = (wsrc == 0) ? a64 : (wsrc == 1) ? b64 : c64;
        u64 word = 0;
#pragma unroll
        for (int np = 0; np < 16; np++)
            word ^= s_nib[(np * 16 + (int)((src >> (4 * np)) & 0xf)) * 8 + wcol];
        int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
        u64 aw0 = __shfl_sync(0xffffffffu, word, wlo),     aw1 = __shfl_sync(0xffffffffu, word, whi);
        u64 bw0 = __shfl_sync(0xffffffffu, word, 8 + wlo), bw1 = __shfl_sync(0xffffffffu, word, 8 + whi);
        u64 cw0 = __shfl_sync(0xffffffffu, word, 16 + wlo),cw1 = __shfl_sync(0xffffffffu, word, 16 + whi);
        uint8_t aL = (uint8_t)(aw0 >> sh), aH = (uint8_t)(aw1 >> sh);
        uint8_t bL = (uint8_t)(bw0 >> sh), bH = (uint8_t)(bw1 >> sh);
        uint8_t cL = (uint8_t)(cw0 >> sh), cH = (uint8_t)(cw1 >> sh);
        uint8_t ab0 = f8mul[(int)aL * 256 + bL], ab1 = f8mul[(int)aH * 256 + bH];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u64 m;
            m = (u64)0 - (u64)((ab0 >> k) & 1); pab0.lo ^= eqB[k].lo & m; pab0.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((ab1 >> k) & 1); pab1.lo ^= eqB[k].lo & m; pab1.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cL  >> k) & 1); pc0.lo  ^= eqB[k].lo & m; pc0.hi  ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cH  >> k) & 1); pc1.lo  ^= eqB[k].lo & m; pc1.hi  ^= eqB[k].hi & m;
        }
    }
    atomicXor((unsigned long long*)&round1_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&round1_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&round1_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&round1_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&round1_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&round1_c[lane + 32].hi, pc1.hi);
}

// Variant 4: nibble-LUT extend + per-coord φ8 table lookup + single ghash.
// Drops the eqB basis (16 shuffles + 16 accumulator regs) and the 8-way accumulate
// loop; instead Σ_{k:bit k of byte} eqB[k] = ghash(eq_x, φ8(byte)) directly, with
// φ8(byte) from a 256-entry shared table. eq_x is broadcast-loaded (no shuffle).
template <int W>
__global__ void zc_round1_warp4(const uint8_t* __restrict__ a_packed,
                                const uint8_t* __restrict__ b_packed,
                                const uint8_t* __restrict__ c_packed,
                                const F128* __restrict__ eq_full,
                                const u64* __restrict__ nib,
                                const F128* __restrict__ phi,
                                const uint8_t* __restrict__ f8mul,
                                long long rows, long long G,
                                F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ u64 s_nib[16 * 16 * 8];            // 16 KB
    __shared__ F128 s_phi[256];                   // 4 KB
    for (int i = threadIdx.x; i < 16 * 16 * 8; i += blockDim.x) s_nib[i] = nib[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_phi[i] = phi[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > rows) o1 = rows;
    if (o0 >= o1) return;

    int wsrc = lane >> 3, wcol = lane & 7;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        u64 a64 = *(const u64*)(a_packed + o * 8);
        u64 b64 = *(const u64*)(b_packed + o * 8);
        u64 c64 = *(const u64*)(c_packed + o * 8);
        F128 eq_x = eq_full[o];
        u64 src = (wsrc == 0) ? a64 : (wsrc == 1) ? b64 : c64;
        u64 word = 0;
#pragma unroll
        for (int np = 0; np < 16; np++)
            word ^= s_nib[(np * 16 + (int)((src >> (4 * np)) & 0xf)) * 8 + wcol];
        int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
        u64 aw0 = __shfl_sync(0xffffffffu, word, wlo),     aw1 = __shfl_sync(0xffffffffu, word, whi);
        u64 bw0 = __shfl_sync(0xffffffffu, word, 8 + wlo), bw1 = __shfl_sync(0xffffffffu, word, 8 + whi);
        u64 cw0 = __shfl_sync(0xffffffffu, word, 16 + wlo),cw1 = __shfl_sync(0xffffffffu, word, 16 + whi);
        uint8_t aL = (uint8_t)(aw0 >> sh), aH = (uint8_t)(aw1 >> sh);
        uint8_t bL = (uint8_t)(bw0 >> sh), bH = (uint8_t)(bw1 >> sh);
        uint8_t cL = (uint8_t)(cw0 >> sh), cH = (uint8_t)(cw1 >> sh);
        uint8_t ab0 = f8mul[(int)aL * 256 + bL], ab1 = f8mul[(int)aH * 256 + bH];
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_x, s_phi[ab0]));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_x, s_phi[ab1]));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_x, s_phi[cL]));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_x, s_phi[cH]));
    }
    atomicXor((unsigned long long*)&round1_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&round1_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&round1_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&round1_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&round1_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&round1_c[lane + 32].hi, pc1.hi);
}

inline void launch_zc_round1_warp4(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                   const F128* d_eq_full, long long rows,
                                   F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;
    long long G = ZC_WG;
    long long warps = (rows + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_warp4<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_full, g_zc_nib, g_zc_phi,
                                           g_zc_f8mul, rows, G, d_round1_ab, d_round1_c);
}

// Variant 5: unified byte LUT via the Cauchy single-table collapse (the paper's
// intended encode). extend word w = XOR_{b=0..7} T0[a_b*8 + (w^b)] — 8 byte
// lookups from a 16 KB shared table, vs warp3's 16 nibble lookups (same size).
template <int W>
__global__ void zc_round1_warp5(const uint8_t* __restrict__ a_packed,
                                const uint8_t* __restrict__ b_packed,
                                const uint8_t* __restrict__ c_packed,
                                const F128* __restrict__ eq_full,
                                const u64* __restrict__ t0,
                                const uint8_t* __restrict__ f8mul,
                                long long rows, long long G,
                                F128* __restrict__ round1_ab, F128* __restrict__ round1_c) {
    __shared__ u64 s_t0[256 * 8];                 // 16 KB
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > rows) o1 = rows;
    if (o0 >= o1) return;

    int wsrc = lane >> 3, wcol = lane & 7;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};
    for (long long o = o0; o < o1; o++) {
        u64 a64 = *(const u64*)(a_packed + o * 8);
        u64 b64 = *(const u64*)(b_packed + o * 8);
        u64 c64 = *(const u64*)(c_packed + o * 8);
        F128 eq_x = eq_full[o];
        F128 my = (lane < 8) ? ghash_mul_karatsuba(eq_x, ZC_BASIS[lane]) : F128{0, 0};
        F128 eqB[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            eqB[k].lo = __shfl_sync(0xffffffffu, my.lo, k);
            eqB[k].hi = __shfl_sync(0xffffffffu, my.hi, k);
        }
        u64 src = (wsrc == 0) ? a64 : (wsrc == 1) ? b64 : c64;
        u64 word = 0;
#pragma unroll
        for (int b = 0; b < 8; b++)
            word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
        int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
        u64 aw0 = __shfl_sync(0xffffffffu, word, wlo),     aw1 = __shfl_sync(0xffffffffu, word, whi);
        u64 bw0 = __shfl_sync(0xffffffffu, word, 8 + wlo), bw1 = __shfl_sync(0xffffffffu, word, 8 + whi);
        u64 cw0 = __shfl_sync(0xffffffffu, word, 16 + wlo),cw1 = __shfl_sync(0xffffffffu, word, 16 + whi);
        uint8_t aL = (uint8_t)(aw0 >> sh), aH = (uint8_t)(aw1 >> sh);
        uint8_t bL = (uint8_t)(bw0 >> sh), bH = (uint8_t)(bw1 >> sh);
        uint8_t cL = (uint8_t)(cw0 >> sh), cH = (uint8_t)(cw1 >> sh);
        uint8_t ab0 = f8mul[(int)aL * 256 + bL], ab1 = f8mul[(int)aH * 256 + bH];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u64 m;
            m = (u64)0 - (u64)((ab0 >> k) & 1); pab0.lo ^= eqB[k].lo & m; pab0.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((ab1 >> k) & 1); pab1.lo ^= eqB[k].lo & m; pab1.hi ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cL  >> k) & 1); pc0.lo  ^= eqB[k].lo & m; pc0.hi  ^= eqB[k].hi & m;
            m = (u64)0 - (u64)((cH  >> k) & 1); pc1.lo  ^= eqB[k].lo & m; pc1.hi  ^= eqB[k].hi & m;
        }
    }
    atomicXor((unsigned long long*)&round1_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&round1_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&round1_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&round1_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&round1_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&round1_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&round1_c[lane + 32].hi, pc1.hi);
}

inline void launch_zc_round1_warp5(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                   const F128* d_eq_full, long long rows,
                                   F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;
    long long G = ZC_WG;
    long long warps = (rows + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_warp5<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_full, g_zc_t0, g_zc_f8mul,
                                           rows, G, d_round1_ab, d_round1_c);
}

inline void launch_zc_round1_warp3(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                   const F128* d_eq_full, long long rows,
                                   F128* d_round1_ab, F128* d_round1_c) {
#ifndef ZC_W
#define ZC_W 4
#endif
    constexpr int W = ZC_W;
    long long G = ZC_WG;
    long long warps = (rows + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_warp3<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_full, g_zc_nib, g_zc_f8mul,
                                           rows, G, d_round1_ab, d_round1_c);
}

inline void launch_zc_round1_warp2(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                   const F128* d_eq_full, long long rows,
                                   F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;
    long long G = ZC_WG;
    long long warps = (rows + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_warp2<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_full, g_zc_f8mul,
                                           rows, G, d_round1_ab, d_round1_c);
}

inline void launch_zc_round1_warp(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                  const F128* d_eq_full, long long rows,
                                  F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 4;                 // warps per block
    long long G = ZC_WG;
    long long warps = (rows + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc_round1_warp<W><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_full, g_zc_mt, g_zc_f8mul,
                                          rows, G, d_round1_ab, d_round1_c);
}

// Reference (groups) launcher — kept for cross-checking; superseded by the warp
// path below. rows = 2^(m-6); d_eq_full = eq(r[6..m]) length rows.
inline void launch_zc_round1_groups(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                                    const F128* d_eq_full, long long rows,
                                    F128* d_round1_ab, F128* d_round1_c) {
    long long G = rows < (1LL << 17) ? rows : (1LL << 17);
    if (G < 1) G = 1;
    long long spg = (rows + G - 1) / G;
    G = (rows + spg - 1) / spg;
    static F128* s_part = nullptr; static long long s_cap = 0;
    long long need = G * 128;
    if (need > s_cap) { if (s_part) cudaFree(s_part); cudaMalloc(&s_part, need * sizeof(F128)); s_cap = need; }
    long long nthreads = G * ZC_NBLK;
    int blocks = (int)((nthreads + ZC_TPB - 1) / ZC_TPB);
    zc_round1_groups<<<blocks, ZC_TPB>>>(d_a, d_b, d_c, d_eq_full, g_zc_f8mul, rows, spg, G, s_part);
    zc_round1_reduce<<<128, ZC_RED_TPB>>>(s_part, G, d_round1_ab, d_round1_c);
}

// Production round-1 launcher: the warp-cooperative unified-byte-LUT path (warp5),
// ~6.2x faster than the groups kernel at m=29 on RTX 5090 (22.8 -> 3.7 ms) and
// faster than the optimized CPU prover (~5 ms). Falls back to the nibble-LUT path
// (warp3) if M lacks the Cauchy single-table structure. rows = 2^(m-6).
inline void launch_zc_round1(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c,
                             const F128* d_eq_full, long long rows,
                             F128* d_round1_ab, F128* d_round1_c) {
    if (g_zc_t0_ok)
        launch_zc_round1_warp5(d_a, d_b, d_c, d_eq_full, rows, d_round1_ab, d_round1_c);
    else
        launch_zc_round1_warp3(d_a, d_b, d_c, d_eq_full, rows, d_round1_ab, d_round1_c);
}
