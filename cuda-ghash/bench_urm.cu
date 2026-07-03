// Binary AG-code URM round-1 AB message — encode bake-off on GPU.
// Three encode strategies, same oracle (host scalar_ref), same fold:
//   1. vanilla        — one thread/block, per-coordinate popcount-parity encode
//   2. bitslice-matvec— one threadblock/block, 128 rows as bit-planes; each output
//                       plane = XOR-reduction of input planes per M (register-light
//                       dense GF(2) matvec). Bitsliced throughout -> free fold.
//   3. lut+transpose  — four-Russians table (8x256x160b) on row-major messages,
//                       by-point product per row, then 128x160 bit-transpose -> fold.
// Bitslice gets pre-bitsliced input (transform not timed, as in the bench).
// LUT does its transpose inside the timed region.
//
// Build: make bench_urm    Run: ./bench_urm [n_blocks]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "f128.cuh"
#include "ntt_host.hpp"
#include "urm_mmask.h"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

__device__ __constant__ u64 M_MASK[160] = URM_MMASK_INIT;
static const u64 M_MASK_H[160] = URM_MMASK_INIT;

struct Rng { u64 s; u64 n() {
    s += 0x9E3779B97F4A7C15ull; u64 z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
} };

static inline int parH(u64 m, u64 x) { return __builtin_popcountll(m & x) & 1; }

static void scalar_ref(const std::vector<std::vector<u64>>& a, const std::vector<std::vector<u64>>& b,
                       const std::vector<F128>& eq, int n, F128* res) {
    for (int j = 0; j < 160; j++) res[j] = F128{0, 0};
    for (int o = 0; o < n; o++) {
        F128 w[160]; for (int j = 0; j < 160; j++) w[j] = F128{0, 0};
        for (int r = 0; r < 128; r++) {
            u64 am = a[o][r], bm = b[o][r];
            int af[160], bf[160];
            for (int k = 0; k < 160; k++) { af[k] = parH(M_MASK_H[k], am); bf[k] = parH(M_MASK_H[k], bm); }
            int pr[160];
            for (int p = 0; p < 64; p++) pr[p] = (af[p] & ((bm>>p)&1)) ^ (((am>>p)&1) & bf[p]);
            for (int p = 0; p < 64; p++) pr[64+p] = (af[64+p] & ((bm>>p)&1)) ^ (af[p] & bf[p]) ^ (((am>>p)&1) & bf[64+p]);
            for (int p = 0; p < 32; p++) pr[128+p] = (af[128+p] & ((bm>>p)&1)) ^ (af[64+p] & bf[p]) ^ (af[p] & bf[64+p]) ^ (((am>>p)&1) & bf[128+p]);
            for (int j = 0; j < 160; j++) if (pr[j]) { if (r<64) w[j].lo|=1ull<<r; else w[j].hi|=1ull<<(r-64); }
        }
        for (int j = 0; j < 160; j++) res[j] = f128_add_hd(res[j], f128_mul_hd(eq[o], w[j]));
    }
}

// Four-Russians table as u32[8*256*5] (5 u32 = 160 bits per (pos,byte) entry).
static std::vector<uint32_t> build_lut_u32() {
    std::vector<uint32_t> t((size_t)8*256*5, 0);
    for (int pos = 0; pos < 8; pos++) for (int byte = 0; byte < 256; byte++) {
        uint32_t* e = &t[((size_t)pos*256+byte)*5];
        for (int k = 0; k < 160; k++)
            if ((__builtin_popcountll((M_MASK_H[k] >> (pos*8)) & (u64)byte) & 1)) e[k/32] |= 1u << (k%32);
    }
    return t;
}

// ---- device helpers ----
__device__ __forceinline__ uint4 x4(uint4 a, uint4 b){ return make_uint4(a.x^b.x,a.y^b.y,a.z^b.z,a.w^b.w); }
__device__ __forceinline__ uint4 a4(uint4 a, uint4 b){ return make_uint4(a.x&b.x,a.y&b.y,a.z&b.z,a.w&b.w); }
__device__ __forceinline__ F128 plane_f128(uint4 p){ return F128{ (u64)p.x | ((u64)p.y<<32), (u64)p.z | ((u64)p.w<<32) }; }
__device__ __forceinline__ void fold_atomic(F128* res, int k, F128 eq, F128 word){
    F128 pr = ghash_mul_karatsuba(eq, word);
    atomicXor((unsigned long long*)&res[k].lo, pr.lo);
    atomicXor((unsigned long long*)&res[k].hi, pr.hi);
}

// 1. Vanilla: one thread per block.
__global__ void urm_vanilla(const u64* amsg, const u64* bmsg, const F128* eq, F128* res, int n) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    F128 w[160]; for (int j = 0; j < 160; j++) w[j] = F128{0,0};
    for (int r = 0; r < 128; r++) {
        u64 am = amsg[(size_t)o*128+r], bm = bmsg[(size_t)o*128+r];
        unsigned char af[160], bf[160];
        for (int k = 0; k < 160; k++) { af[k]=__popcll(M_MASK[k]&am)&1; bf[k]=__popcll(M_MASK[k]&bm)&1; }
        for (int p = 0; p < 64; p++)  if ((af[p]&((bm>>p)&1)) ^ (((am>>p)&1)&bf[p])) { if(r<64) w[p].lo|=1ull<<r; else w[p].hi|=1ull<<(r-64); }
        for (int p = 0; p < 64; p++)  if ((af[64+p]&((bm>>p)&1)) ^ (af[p]&bf[p]) ^ (((am>>p)&1)&bf[64+p])) { if(r<64) w[64+p].lo|=1ull<<r; else w[64+p].hi|=1ull<<(r-64); }
        for (int p = 0; p < 32; p++)  if ((af[128+p]&((bm>>p)&1)) ^ (af[64+p]&bf[p]) ^ (af[p]&bf[64+p]) ^ (((am>>p)&1)&bf[128+p])) { if(r<64) w[128+p].lo|=1ull<<r; else w[128+p].hi|=1ull<<(r-64); }
    }
    F128 e = eq[o];
    for (int j = 0; j < 160; j++) fold_atomic(res, j, e, w[j]);
}

// 2. Bitslice-matvec: one threadblock per data block; a_pl/b_pl pre-bitsliced
//    (a_pl[o*64+k], b_pl[o*64+k] are 128-bit input planes as uint4).
__global__ void urm_bitslice(const uint4* a_pl, const uint4* b_pl, const F128* eq, F128* res, int n) {
    int o = blockIdx.x; if (o >= n) return;
    __shared__ uint4 sa[64], sb[64], saf[160], sbf[160];
    for (int i = threadIdx.x; i < 64; i += blockDim.x) { sa[i]=a_pl[(size_t)o*64+i]; sb[i]=b_pl[(size_t)o*64+i]; }
    __syncthreads();
    int k = threadIdx.x;
    if (k < 160) {
        u64 mask = M_MASK[k];
        uint4 af = make_uint4(0,0,0,0), bf = af;
        for (int j = 0; j < 64; j++) if ((mask>>j)&1) { af = x4(af, sa[j]); bf = x4(bf, sb[j]); }
        saf[k] = af; sbf[k] = bf;
    }
    __syncthreads();
    if (k < 160) {
        uint4 prod;
        if (k < 64) { int p=k; prod = x4(a4(saf[k],sb[p]), a4(sa[p],sbf[k])); }
        else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[p],sbf[p])), a4(sa[p],sbf[k])); }
        else { int p=k-128; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[64+p],sbf[p])), x4(a4(saf[p],sbf[64+p]), a4(sa[p],sbf[k]))); }
        fold_atomic(res, k, eq[o], plane_f128(prod));
    }
}

// Diagnostic: bitslice but atomics spread across SLOTS copies of res to test
// whether global atomic contention on 160 addresses is the bottleneck.
__global__ void urm_bitslice_spread(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                    F128* res, int n, int slots) {
    int o = blockIdx.x; if (o >= n) return;
    __shared__ uint4 sa[64], sb[64], saf[160], sbf[160];
    for (int i = threadIdx.x; i < 64; i += blockDim.x) { sa[i]=a_pl[(size_t)o*64+i]; sb[i]=b_pl[(size_t)o*64+i]; }
    __syncthreads();
    int k = threadIdx.x;
    if (k < 160) {
        u64 mask = M_MASK[k];
        uint4 af = make_uint4(0,0,0,0), bf = af;
        for (int j = 0; j < 64; j++) if ((mask>>j)&1) { af = x4(af, sa[j]); bf = x4(bf, sb[j]); }
        saf[k] = af; sbf[k] = bf;
    }
    __syncthreads();
    if (k < 160) {
        uint4 prod;
        if (k < 64) { int p=k; prod = x4(a4(saf[k],sb[p]), a4(sa[p],sbf[k])); }
        else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[p],sbf[p])), a4(sa[p],sbf[k])); }
        else { int p=k-128; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[64+p],sbf[p])), x4(a4(saf[p],sbf[64+p]), a4(sa[p],sbf[k]))); }
        int slot = o & (slots-1);
        fold_atomic(res + (size_t)slot*160, k, eq[o], plane_f128(prod));
    }
}

// 2b. Bitslice, accumulated: each CUDA block processes G data-blocks serially,
//     keeping coord k's running fold in a register; 160 atomics per CUDA block
//     (G x fewer) instead of per data-block. Kills the atomic contention.
__global__ void urm_bitslice_acc(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                 F128* res, int n, int G) {
    __shared__ uint4 sa[64], sb[64], saf[160], sbf[160];
    int k = threadIdx.x;
    F128 acc = F128{0,0};
    int base = blockIdx.x * G;
    for (int g = 0; g < G; g++) {
        int o = base + g; if (o >= n) break;
        for (int i = threadIdx.x; i < 64; i += blockDim.x) { sa[i]=a_pl[(size_t)o*64+i]; sb[i]=b_pl[(size_t)o*64+i]; }
        __syncthreads();
        if (k < 160) {
            u64 mask = M_MASK[k];
            uint4 af = make_uint4(0,0,0,0), bf = af;
            for (int j = 0; j < 64; j++) if ((mask>>j)&1) { af = x4(af, sa[j]); bf = x4(bf, sb[j]); }
            saf[k] = af; sbf[k] = bf;
        }
        __syncthreads();
        if (k < 160) {
            uint4 prod;
            if (k < 64) { int p=k; prod = x4(a4(saf[k],sb[p]), a4(sa[p],sbf[k])); }
            else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[p],sbf[p])), a4(sa[p],sbf[k])); }
            else { int p=k-128; prod = x4(x4(a4(saf[k],sb[p]), a4(saf[64+p],sbf[p])), x4(a4(saf[p],sbf[64+p]), a4(sa[p],sbf[k]))); }
            F128 pr = ghash_mul_karatsuba(eq[o], plane_f128(prod));
            acc.lo ^= pr.lo; acc.hi ^= pr.hi;
        }
        __syncthreads();
    }
    if (k < 160) {
        atomicXor((unsigned long long*)&res[k].lo, acc.lo);
        atomicXor((unsigned long long*)&res[k].hi, acc.hi);
    }
}

// 3. LUT + transpose: one threadblock per data block.
__global__ void urm_lut(const u64* amsg, const u64* bmsg, const uint32_t* table, const F128* eq, F128* res, int n) {
    int o = blockIdx.x; if (o >= n) return;
    __shared__ uint32_t stab[8*256*5];   // 40 KB
    __shared__ uint32_t sprod[128*5];     // 2.5 KB
    for (int i = threadIdx.x; i < 8*256*5; i += blockDim.x) stab[i] = table[i];
    __syncthreads();
    for (int r = threadIdx.x; r < 128; r += blockDim.x) {
        u64 am = amsg[(size_t)o*128+r], bm = bmsg[(size_t)o*128+r];
        uint32_t af[5]={0,0,0,0,0}, bf[5]={0,0,0,0,0};
        for (int pos = 0; pos < 8; pos++) {
            uint32_t* ea = &stab[((size_t)pos*256 + ((am>>(pos*8))&0xff))*5];
            uint32_t* eb = &stab[((size_t)pos*256 + ((bm>>(pos*8))&0xff))*5];
            for (int i=0;i<5;i++){ af[i]^=ea[i]; bf[i]^=eb[i]; }
        }
        uint32_t ax0=(uint32_t)am, ax1=(uint32_t)(am>>32), bx0=(uint32_t)bm, bx1=(uint32_t)(bm>>32);
        uint32_t pr[5];
        pr[0]=(af[0]&bx0)^(ax0&bf[0]);
        pr[1]=(af[1]&bx1)^(ax1&bf[1]);
        pr[2]=(af[2]&bx0)^(af[0]&bf[0])^(ax0&bf[2]);
        pr[3]=(af[3]&bx1)^(af[1]&bf[1])^(ax1&bf[3]);
        pr[4]=(af[4]&bx0)^(af[2]&bf[0])^(af[0]&bf[2])^(ax0&bf[4]);
        for (int i=0;i<5;i++) sprod[r*5+i]=pr[i];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < 160; k += blockDim.x) {
        uint4 plane = make_uint4(0,0,0,0);
        int wi = k/32, bit = k%32;
        for (int r = 0; r < 128; r++) {
            if ((sprod[r*5+wi]>>bit)&1) {
                if (r<32) plane.x|=1u<<r; else if (r<64) plane.y|=1u<<(r-32);
                else if (r<96) plane.z|=1u<<(r-64); else plane.w|=1u<<(r-96);
            }
        }
        fold_atomic(res, k, eq[o], plane_f128(plane));
    }
}

// 4. Warp-cooperative: 1 warp = 1 data-block, W warps/CUDA-block (independent).
//    32 lanes, lane L owns coords {L, L+32, L+64, L+96, L+128}. Planes exchanged
//    via shared (per-warp region) synced with __syncwarp() (no block barrier).
//    Each lane reads sa[j] once and applies to its 5 coords (5x fewer LDS).
template <int W>
__global__ void urm_bitslice_warp(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                  F128* res, int n, int G) {
    __shared__ uint4 sa[W*64], sb[W*64], saf[W*160], sbf[W*160];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int sin = wid*64, sco = wid*160;
    F128 acc[5] = { {0,0},{0,0},{0,0},{0,0},{0,0} };
    u64 m0=M_MASK[lane], m1=M_MASK[lane+32], m2=M_MASK[lane+64], m3=M_MASK[lane+96], m4=M_MASK[lane+128];
    int base = (blockIdx.x * W + wid) * G;
    for (int g = 0; g < G; g++) {
        int o = base + g; if (o >= n) break;
        for (int i = lane; i < 64; i += 32) { sa[sin+i]=a_pl[(size_t)o*64+i]; sb[sin+i]=b_pl[(size_t)o*64+i]; }
        __syncwarp();
        uint4 af[5], bf[5];
        #pragma unroll
        for (int c=0;c<5;c++){ af[c]=make_uint4(0,0,0,0); bf[c]=make_uint4(0,0,0,0); }
        for (int j = 0; j < 64; j++) {
            uint4 sj = sa[sin+j], tj = sb[sin+j];
            if ((m0>>j)&1){ af[0]=x4(af[0],sj); bf[0]=x4(bf[0],tj); }
            if ((m1>>j)&1){ af[1]=x4(af[1],sj); bf[1]=x4(bf[1],tj); }
            if ((m2>>j)&1){ af[2]=x4(af[2],sj); bf[2]=x4(bf[2],tj); }
            if ((m3>>j)&1){ af[3]=x4(af[3],sj); bf[3]=x4(bf[3],tj); }
            if ((m4>>j)&1){ af[4]=x4(af[4],sj); bf[4]=x4(bf[4],tj); }
        }
        #pragma unroll
        for (int c=0;c<5;c++){ saf[sco+lane+32*c]=af[c]; sbf[sco+lane+32*c]=bf[c]; }
        __syncwarp();
        #pragma unroll
        for (int c=0;c<5;c++){
            int k = lane + 32*c; uint4 prod;
            if (k < 64) { int p=k; prod = x4(a4(saf[sco+k],sb[sin+p]), a4(sa[sin+p],sbf[sco+k])); }
            else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[sco+k],sb[sin+p]), a4(saf[sco+p],sbf[sco+p])), a4(sa[sin+p],sbf[sco+k])); }
            else { int p=k-128; prod = x4(x4(a4(saf[sco+k],sb[sin+p]), a4(saf[sco+64+p],sbf[sco+p])), x4(a4(saf[sco+p],sbf[sco+64+p]), a4(sa[sin+p],sbf[sco+k]))); }
            F128 pr = ghash_mul_karatsuba(eq[o], plane_f128(prod));
            acc[c].lo ^= pr.lo; acc[c].hi ^= pr.hi;
        }
        __syncwarp();
    }
    #pragma unroll
    for (int c=0;c<5;c++){ int k=lane+32*c; atomicXor((unsigned long long*)&res[k].lo, acc[c].lo); atomicXor((unsigned long long*)&res[k].hi, acc[c].hi); }
}

// 4c. Warp-cooperative, NO cross-coord exchange. Lane L owns coords {L,L+32,
//     L+64,L+96,L+128} = all coords ≡ L (mod 32). The Hasse product for coord k
//     only couples coords ≡ k (mod 32) = L, so every af/bf it needs is in this
//     lane's own registers -- no saf/sbf shared, no second __syncwarp, no
//     cross-coord reads. Only the raw value planes sa[L],sa[L+32] (and b) come
//     from shared. Frees ~20KB shared -> higher occupancy.
template <int W>
__global__ void urm_bitslice_warp_nx(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                     F128* res, int n, int G) {
    __shared__ uint4 sa[W*64], sb[W*64];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int sin = wid*64;
    F128 acc[5] = { {0,0},{0,0},{0,0},{0,0},{0,0} };
    u64 m0=M_MASK[lane], m1=M_MASK[lane+32], m2=M_MASK[lane+64], m3=M_MASK[lane+96], m4=M_MASK[lane+128];
    int base = (blockIdx.x * W + wid) * G;
    for (int g = 0; g < G; g++) {
        int o = base + g; if (o >= n) break;
        for (int i = lane; i < 64; i += 32) { sa[sin+i]=a_pl[(size_t)o*64+i]; sb[sin+i]=b_pl[(size_t)o*64+i]; }
        __syncwarp();
        uint4 af0=make_uint4(0,0,0,0),af1=af0,af2=af0,af3=af0,af4=af0, bf0=af0,bf1=af0,bf2=af0,bf3=af0,bf4=af0;
        for (int j = 0; j < 64; j++) {
            uint4 sj = sa[sin+j], tj = sb[sin+j];
            if ((m0>>j)&1){ af0=x4(af0,sj); bf0=x4(bf0,tj); }
            if ((m1>>j)&1){ af1=x4(af1,sj); bf1=x4(bf1,tj); }
            if ((m2>>j)&1){ af2=x4(af2,sj); bf2=x4(bf2,tj); }
            if ((m3>>j)&1){ af3=x4(af3,sj); bf3=x4(bf3,tj); }
            if ((m4>>j)&1){ af4=x4(af4,sj); bf4=x4(bf4,tj); }
        }
        // value planes for p=L and p=L+32 (raw input planes, from shared)
        uint4 saL=sa[sin+lane], sbL=sb[sin+lane], saH=sa[sin+lane+32], sbH=sb[sin+lane+32];
        // products, all operands in registers (+ raw value planes)
        uint4 p0 = x4(a4(af0,sbL), a4(saL,bf0));                                  // D1, k=L
        uint4 p1 = x4(a4(af1,sbH), a4(saH,bf1));                                  // D1, k=L+32
        uint4 p2 = x4(x4(a4(af2,sbL), a4(af0,bf0)), a4(saL,bf2));                 // D2, k=L+64 (p=L)
        uint4 p3 = x4(x4(a4(af3,sbH), a4(af1,bf1)), a4(saH,bf3));                 // D2, k=L+96 (p=L+32)
        uint4 p4 = x4(x4(a4(af4,sbL), a4(af2,bf0)), x4(a4(af0,bf2), a4(saL,bf4))); // D3, k=L+128 (p=L)
        F128 e = eq[o];
        F128 r0=ghash_mul_karatsuba(e,plane_f128(p0)); acc[0].lo^=r0.lo; acc[0].hi^=r0.hi;
        F128 r1=ghash_mul_karatsuba(e,plane_f128(p1)); acc[1].lo^=r1.lo; acc[1].hi^=r1.hi;
        F128 r2=ghash_mul_karatsuba(e,plane_f128(p2)); acc[2].lo^=r2.lo; acc[2].hi^=r2.hi;
        F128 r3=ghash_mul_karatsuba(e,plane_f128(p3)); acc[3].lo^=r3.lo; acc[3].hi^=r3.hi;
        F128 r4=ghash_mul_karatsuba(e,plane_f128(p4)); acc[4].lo^=r4.lo; acc[4].hi^=r4.hi;
        __syncwarp();
    }
    #pragma unroll
    for (int c=0;c<5;c++){ int k=lane+32*c; atomicXor((unsigned long long*)&res[k].lo, acc[c].lo); atomicXor((unsigned long long*)&res[k].hi, acc[c].hi); }
}

// lop3.b32 with truth table 0x78 = A ^ (B & C). Used to fuse the masked
// accumulate af ^= (sa[j] & mask_bit) into one unconditional instruction.
__device__ __forceinline__ unsigned lop3_78(unsigned a, unsigned b, unsigned c) {
    unsigned d; asm("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ uint4 xandc4(uint4 a, uint4 b, unsigned mb) {
    return make_uint4(lop3_78(a.x,b.x,mb), lop3_78(a.y,b.y,mb), lop3_78(a.z,b.z,mb), lop3_78(a.w,b.w,mb));
}

// 4d. Warp-cooperative, encode via lop3 0x78 (af ^= sa[j] & mask_bit, branchless).
template <int W>
__global__ void urm_bitslice_warp_l3(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                     F128* res, int n, int G) {
    __shared__ uint4 sa[W*64], sb[W*64], saf[W*160], sbf[W*160];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int sin = wid*64, sco = wid*160;
    F128 acc[5] = { {0,0},{0,0},{0,0},{0,0},{0,0} };
    u64 m0=M_MASK[lane], m1=M_MASK[lane+32], m2=M_MASK[lane+64], m3=M_MASK[lane+96], m4=M_MASK[lane+128];
    int base = (blockIdx.x * W + wid) * G;
    for (int g = 0; g < G; g++) {
        int o = base + g; if (o >= n) break;
        for (int i = lane; i < 64; i += 32) { sa[sin+i]=a_pl[(size_t)o*64+i]; sb[sin+i]=b_pl[(size_t)o*64+i]; }
        __syncwarp();
        uint4 af[5], bf[5];
        #pragma unroll
        for (int c=0;c<5;c++){ af[c]=make_uint4(0,0,0,0); bf[c]=make_uint4(0,0,0,0); }
        for (int j = 0; j < 64; j++) {
            uint4 sj = sa[sin+j], tj = sb[sin+j];
            unsigned b0=-(unsigned)((m0>>j)&1), b1=-(unsigned)((m1>>j)&1), b2=-(unsigned)((m2>>j)&1), b3=-(unsigned)((m3>>j)&1), b4=-(unsigned)((m4>>j)&1);
            af[0]=xandc4(af[0],sj,b0); bf[0]=xandc4(bf[0],tj,b0);
            af[1]=xandc4(af[1],sj,b1); bf[1]=xandc4(bf[1],tj,b1);
            af[2]=xandc4(af[2],sj,b2); bf[2]=xandc4(bf[2],tj,b2);
            af[3]=xandc4(af[3],sj,b3); bf[3]=xandc4(bf[3],tj,b3);
            af[4]=xandc4(af[4],sj,b4); bf[4]=xandc4(bf[4],tj,b4);
        }
        #pragma unroll
        for (int c=0;c<5;c++){ saf[sco+lane+32*c]=af[c]; sbf[sco+lane+32*c]=bf[c]; }
        __syncwarp();
        #pragma unroll
        for (int c=0;c<5;c++){
            int k = lane + 32*c; uint4 prod;
            if (k < 64) { int p=k; prod = x4(a4(saf[sco+k],sb[sin+p]), a4(sa[sin+p],sbf[sco+k])); }
            else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[sco+k],sb[sin+p]), a4(saf[sco+p],sbf[sco+p])), a4(sa[sin+p],sbf[sco+k])); }
            else { int p=k-128; prod = x4(x4(a4(saf[sco+k],sb[sin+p]), a4(saf[sco+64+p],sbf[sco+p])), x4(a4(saf[sco+p],sbf[sco+64+p]), a4(sa[sin+p],sbf[sco+k]))); }
            F128 pr = ghash_mul_karatsuba(eq[o], plane_f128(prod));
            acc[c].lo ^= pr.lo; acc[c].hi ^= pr.hi;
        }
        __syncwarp();
    }
    #pragma unroll
    for (int c=0;c<5;c++){ int k=lane+32*c; atomicXor((unsigned long long*)&res[k].lo, acc[c].lo); atomicXor((unsigned long long*)&res[k].hi, acc[c].hi); }
}

// cp.async helpers (16-byte global->shared, sm_80+).
__device__ __forceinline__ void cpasync16(uint4* dst_smem, const uint4* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cpasync_commit() { asm volatile("cp.async.commit_group;\n"); }

// 4b. Warp-cooperative + cp.async double-buffering: overlap the next data-block's
//     global->shared load with the current block's compute (hides memory latency).
template <int W>
__global__ void urm_bitslice_warp_ca(const uint4* a_pl, const uint4* b_pl, const F128* eq,
                                     F128* res, int n, int G) {
    __shared__ uint4 sa[2][W*64], sb[2][W*64], saf[W*160], sbf[W*160];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int sin = wid*64, sco = wid*160;
    F128 acc[5] = { {0,0},{0,0},{0,0},{0,0},{0,0} };
    u64 m0=M_MASK[lane], m1=M_MASK[lane+32], m2=M_MASK[lane+64], m3=M_MASK[lane+96], m4=M_MASK[lane+128];
    int base = (blockIdx.x * W + wid) * G;

    auto load = [&](int buf, int o) {
        for (int i = lane; i < 64; i += 32) { cpasync16(&sa[buf][sin+i], &a_pl[(size_t)o*64+i]); cpasync16(&sb[buf][sin+i], &b_pl[(size_t)o*64+i]); }
        cpasync_commit();
    };
    if (base < n) load(0, base);
    for (int g = 0; g < G; g++) {
        int o = base + g; if (o >= n) break;
        bool has_next = (g+1 < G) && (base+g+1 < n);
        if (has_next) load((g+1)&1, base+g+1);
        if (has_next) asm volatile("cp.async.wait_group 1;\n"); else asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
        int cb = g & 1;
        uint4 af[5], bf[5];
        #pragma unroll
        for (int c=0;c<5;c++){ af[c]=make_uint4(0,0,0,0); bf[c]=make_uint4(0,0,0,0); }
        for (int j = 0; j < 64; j++) {
            uint4 sj = sa[cb][sin+j], tj = sb[cb][sin+j];
            if ((m0>>j)&1){ af[0]=x4(af[0],sj); bf[0]=x4(bf[0],tj); }
            if ((m1>>j)&1){ af[1]=x4(af[1],sj); bf[1]=x4(bf[1],tj); }
            if ((m2>>j)&1){ af[2]=x4(af[2],sj); bf[2]=x4(bf[2],tj); }
            if ((m3>>j)&1){ af[3]=x4(af[3],sj); bf[3]=x4(bf[3],tj); }
            if ((m4>>j)&1){ af[4]=x4(af[4],sj); bf[4]=x4(bf[4],tj); }
        }
        #pragma unroll
        for (int c=0;c<5;c++){ saf[sco+lane+32*c]=af[c]; sbf[sco+lane+32*c]=bf[c]; }
        __syncwarp();
        #pragma unroll
        for (int c=0;c<5;c++){
            int k = lane + 32*c; uint4 prod;
            if (k < 64) { int p=k; prod = x4(a4(saf[sco+k],sb[cb][sin+p]), a4(sa[cb][sin+p],sbf[sco+k])); }
            else if (k < 128) { int p=k-64; prod = x4(x4(a4(saf[sco+k],sb[cb][sin+p]), a4(saf[sco+p],sbf[sco+p])), a4(sa[cb][sin+p],sbf[sco+k])); }
            else { int p=k-128; prod = x4(x4(a4(saf[sco+k],sb[cb][sin+p]), a4(saf[sco+64+p],sbf[sco+p])), x4(a4(saf[sco+p],sbf[sco+64+p]), a4(sa[cb][sin+p],sbf[sco+k]))); }
            F128 pr = ghash_mul_karatsuba(eq[o], plane_f128(prod));
            acc[c].lo ^= pr.lo; acc[c].hi ^= pr.hi;
        }
        __syncwarp();
    }
    #pragma unroll
    for (int c=0;c<5;c++){ int k=lane+32*c; atomicXor((unsigned long long*)&res[k].lo, acc[c].lo); atomicXor((unsigned long long*)&res[k].hi, acc[c].hi); }
}

__global__ void fill_planes(uint4* d, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)(i*2654435761u + 1);
    d[i] = make_uint4(x, x*9u+1u, x*5u+3u, x*7u+11u);
}

int main(int argc, char** argv) {
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s | %d SMs\n", p.name, p.multiProcessorCount);

    // Scaling mode: timing-only (device-generated data, no host oracle), warp W=4 G=4,
    // across m = 30..33.  ./bench_urm scale
    if (argc > 1 && strcmp(argv[1], "scale") == 0) {
        size_t nmax = 1u<<20;                       // m=33
        uint4 *dapl,*dbpl; F128 *deq,*dres;
        CK(cudaMalloc(&dapl, nmax*64*sizeof(uint4))); CK(cudaMalloc(&dbpl, nmax*64*sizeof(uint4)));
        CK(cudaMalloc(&deq, nmax*sizeof(F128)));     CK(cudaMalloc(&dres, 160*sizeof(F128)));
        { size_t tot=nmax*64; int t=256; fill_planes<<<(unsigned)((tot+t-1)/t),t>>>(dapl,tot); fill_planes<<<(unsigned)((tot+t-1)/t),t>>>(dbpl,tot);
          fill_planes<<<(unsigned)((nmax*2+t-1)/t),t>>>((uint4*)deq, nmax/2); CK(cudaDeviceSynchronize()); }
        printf("\n== URM warp W=4 G=4 scaling (timing only) ==\n");
        int G=4;
        for (int m = 30; m <= 33; m++) {
            int nn = 1<<(m-13);                     // blocks = 2^(m-13)
            double gib = (double)nn*64*sizeof(uint4)*2/(1024.0*1024*1024);
            auto launch=[&]{ int gb=(nn+(4*G)-1)/(4*G); urm_bitslice_warp<4><<<gb,128>>>(dapl,dbpl,deq,dres,nn,G); };
            CK(cudaMemset(dres,0,160*sizeof(F128))); launch(); CK(cudaDeviceSynchronize());
            cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            int it=30; CK(cudaEventRecord(a)); for(int i=0;i<it;i++) launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=it;
            printf("  m=%d  n=%-8d rows=%-10lld in=%5.2f GiB | %8.3f ms  %6.4f ns/row\n",
                   m, nn, (long long)nn*128, gib, ms, ms*1e6/((double)nn*128.0));
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        }
        return 0;
    }

    int n = argc > 1 ? atoi(argv[1]) : 8192;
    Rng rng{0xC0DE};
    std::vector<std::vector<u64>> a(n, std::vector<u64>(128)), b(n, std::vector<u64>(128));
    for (int o=0;o<n;o++) for (int r=0;r<128;r++){ a[o][r]=rng.n(); b[o][r]=rng.n(); }
    std::vector<F128> eq(n); for (int o=0;o<n;o++) eq[o]=F128{rng.n(),rng.n()};
    F128 want[160]; scalar_ref(a,b,eq,n,want);

    // row-major flat
    std::vector<u64> amf((size_t)n*128), bmf((size_t)n*128);
    for (int o=0;o<n;o++) for (int r=0;r<128;r++){ amf[(size_t)o*128+r]=a[o][r]; bmf[(size_t)o*128+r]=b[o][r]; }
    // pre-bitsliced planes: a_pl[o*64+k], uint4 {rows0-31,32-63,64-95,96-127}
    std::vector<uint4> apl((size_t)n*64), bpl((size_t)n*64);
    for (int o=0;o<n;o++) for (int k=0;k<64;k++){
        uint32_t w[4]={0,0,0,0}, v[4]={0,0,0,0};
        for (int r=0;r<128;r++){ if((a[o][r]>>k)&1) w[r/32]|=1u<<(r%32); if((b[o][r]>>k)&1) v[r/32]|=1u<<(r%32); }
        apl[(size_t)o*64+k]=make_uint4(w[0],w[1],w[2],w[3]);
        bpl[(size_t)o*64+k]=make_uint4(v[0],v[1],v[2],v[3]);
    }
    auto lut = build_lut_u32();

    u64 *da,*db; uint4 *dapl,*dbpl; uint32_t* dlut; F128 *deq,*dres;
    CK(cudaMalloc(&da,amf.size()*8)); CK(cudaMalloc(&db,bmf.size()*8));
    CK(cudaMalloc(&dapl,apl.size()*16)); CK(cudaMalloc(&dbpl,bpl.size()*16));
    CK(cudaMalloc(&dlut,lut.size()*4)); CK(cudaMalloc(&deq,n*sizeof(F128))); CK(cudaMalloc(&dres,160*sizeof(F128)));
    CK(cudaMemcpy(da,amf.data(),amf.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,bmf.data(),bmf.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dapl,apl.data(),apl.size()*16,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbpl,bpl.data(),bpl.size()*16,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlut,lut.data(),lut.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(deq,eq.data(),n*sizeof(F128),cudaMemcpyHostToDevice));

    auto check = [&](const char* name){
        F128 got[160]; CK(cudaMemcpy(got,dres,160*sizeof(F128),cudaMemcpyDeviceToHost));
        int bad=0; for(int j=0;j<160;j++) if(got[j].lo!=want[j].lo||got[j].hi!=want[j].hi) bad++;
        printf("  %-16s %s\n", name, bad? "FAIL" : "OK (oracle)");
        return bad==0;
    };
    auto timeit = [&](const char* name, auto launch){
        // correctness
        CK(cudaMemset(dres,0,160*sizeof(F128))); launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        bool ok = check(name);
        // timing
        cudaEvent_t a2,b2; CK(cudaEventCreate(&a2)); CK(cudaEventCreate(&b2));
        int iters=50; launch(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a2)); for(int i=0;i<iters;i++) launch(); CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        float ms=0; CK(cudaEventElapsedTime(&ms,a2,b2)); ms/=iters;
        double nspr = ms*1e6/((double)n*128.0);
        printf("  %-16s %8.3f ms  %6.3f ns/row %s\n", name, ms, nspr, ok?"":"(WRONG)");
        CK(cudaEventDestroy(a2)); CK(cudaEventDestroy(b2));
    };

    printf("\n== URM encode bake-off, n=%d blocks (%d messages) ==\n", n, n*128);
    timeit("vanilla", [&]{ int t=128; urm_vanilla<<<(n+t-1)/t,t>>>(da,db,deq,dres,n); });
    timeit("bitslice-matvec", [&]{ urm_bitslice<<<n,192>>>(dapl,dbpl,deq,dres,n); });
    timeit("lut+transpose", [&]{ urm_lut<<<n,128>>>(da,db,dlut,deq,dres,n); });
    timeit("bitslice-acc G8", [&]{ int gb=(n+7)/8; urm_bitslice_acc<<<gb,192>>>(dapl,dbpl,deq,dres,n,8); });
    timeit("warp W4 G4", [&]{ int gb=(n+15)/16; urm_bitslice_warp<4><<<gb,128>>>(dapl,dbpl,deq,dres,n,4); });          // WINNER
    timeit("warp-noexch(reg)", [&]{ int gb=(n+15)/16; urm_bitslice_warp_nx<4><<<gb,128>>>(dapl,dbpl,deq,dres,n,4); }); // 5x slower: live-range pressure
    timeit("warp+cp.async", [&]{ int gb=(n+15)/16; urm_bitslice_warp_ca<4><<<gb,128>>>(dapl,dbpl,deq,dres,n,4); });    // neutral: not mem-bound
    timeit("warp+lop3.78", [&]{ int gb=(n+15)/16; urm_bitslice_warp_l3<4><<<gb,128>>>(dapl,dbpl,deq,dres,n,4); });     // branchless masked encode

    // Diagnostic: spread atomics over SLOTS res copies (skip oracle; tests contention).
    int slots = 4096; F128* dbig; CK(cudaMalloc(&dbig, (size_t)slots*160*sizeof(F128)));
    {
        cudaEvent_t a2,b2; CK(cudaEventCreate(&a2)); CK(cudaEventCreate(&b2));
        auto launch=[&]{ urm_bitslice_spread<<<n,192>>>(dapl,dbpl,deq,dbig,n,slots); };
        CK(cudaMemset(dbig,0,(size_t)slots*160*sizeof(F128))); launch(); CK(cudaDeviceSynchronize());
        int it=50; CK(cudaEventRecord(a2)); for(int i=0;i<it;i++) launch(); CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
        float ms=0; CK(cudaEventElapsedTime(&ms,a2,b2)); ms/=it;
        printf("  %-16s %8.3f ms  %6.3f ns/row  (diagnostic, %d slots)\n", "bitslice-spread", ms, ms*1e6/((double)n*128.0), slots);
    }
    return 0;
}
