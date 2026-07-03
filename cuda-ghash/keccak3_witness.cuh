// GPU witness generator for the 3-wide Keccak-f[1600] R1CS (src/r1cs_hashes/keccak3.rs,
// K_LOG=17, K=131072: three independent permutations per block). One thread per block:
// runs 24 real Keccak-f rounds for each of 3 sub-permutations, captures the χ AND-marginals
// t[i,r], and writes the K-bit z/a/b slices — the same total write volume as BLAKE3 but with
// the heavier Keccak-f compute. Timing-faithful (real permutation + real t + full writes);
// inputs are generated inline (this is a timing/dataflow bench, like fill_compressions).
#pragma once
#include "f128.cuh"

#define KC_K_LOG 17
#define KC_K (1 << KC_K_LOG)            // 131072 z-slots/block
#define KC_U64_PER_BLOCK (KC_K / 64)    // 2048 u64 per z/a/b slice
#define KC_N_SUB 3
#define KC_SLOT_U64 32                  // 2048-bit slot = 32 u64
#define KC_ZCONST_U64 192               // Z_CONST=12288 bit -> u64 192
#define KC_T_BASE_U64 193               // T_PACKED_BIT_BASE=12352 bit -> u64 193
#define KC_LANES 25                     // 1600-bit state = 25 u64 lanes

__device__ __constant__ u64 KC_RC[24] = {
    0x0000000000000001ull,0x0000000000008082ull,0x800000000000808Aull,0x8000000080008000ull,
    0x000000000000808Bull,0x0000000080000001ull,0x8000000080008081ull,0x8000000000008009ull,
    0x000000000000008Aull,0x0000000000000088ull,0x0000000080008009ull,0x000000008000000Aull,
    0x000000008000808Bull,0x800000000000008Bull,0x8000000000008089ull,0x8000000000008003ull,
    0x8000000000008002ull,0x8000000000000080ull,0x000000000000800Aull,0x800000008000000Aull,
    0x8000000080008081ull,0x8000000000008080ull,0x0000000080000001ull,0x8000000080008008ull};
// RHO_OFFSETS[a][b], a=row, b=col (keccak.rs).
__device__ __constant__ int KC_RHO[5][5] = {
    {0,36,3,41,18},{1,44,10,45,2},{62,6,43,15,61},{28,55,25,21,56},{27,20,39,8,14}};

__device__ __forceinline__ u64 kc_rotl(u64 v, int r) { return r ? ((v << r) | (v >> (64 - r))) : v; }
#define KC_LI(x,y) ((x) + 5*(y))

int constexpr KC_TPB = 64;

// theta in place on 25 lanes.
__device__ __forceinline__ void kc_theta(u64* s) {
    u64 c[5], d[5];
#pragma unroll
    for (int x = 0; x < 5; x++)
        c[x] = s[KC_LI(x,0)]^s[KC_LI(x,1)]^s[KC_LI(x,2)]^s[KC_LI(x,3)]^s[KC_LI(x,4)];
#pragma unroll
    for (int x = 0; x < 5; x++) d[x] = c[(x+4)%5] ^ kc_rotl(c[(x+1)%5], 1);
#pragma unroll
    for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) s[KC_LI(x,y)] ^= d[x];
}
// rho∘pi: out[x,y] = rotl(in[(x+3y)%5, x], rho).
__device__ __forceinline__ void kc_rho_pi(const u64* in, u64* out) {
#pragma unroll
    for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) {
        int a = (x + 3*y) % 5, b = x;
        out[KC_LI(x,y)] = kc_rotl(in[KC_LI(a,b)], KC_RHO[a][b] % 64);
    }
}

// Build one block's z/a/b slices (KC_U64_PER_BLOCK u64 each) from 3 Keccak-f's.
__global__ void keccak3_witness_blocks(int n_blocks, long long n_total,
                                       u64* __restrict__ z, u64* __restrict__ a, u64* __restrict__ b) {
    long long blk = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_total) return;
    u64* zb = z + blk * KC_U64_PER_BLOCK;
    u64* ab = a + blk * KC_U64_PER_BLOCK;
    u64* bb = b + blk * KC_U64_PER_BLOCK;
    if (blk >= n_blocks) {            // padding block: zero slices
        for (int j = 0; j < KC_U64_PER_BLOCK; j++) { zb[j]=0; ab[j]=0; bb[j]=0; }
        return;
    }
    for (int j = 0; j < KC_U64_PER_BLOCK; j++) { zb[j]=0; ab[j]=0; bb[j]=0; }
    const u64 ALL = ~0ull;

    for (int sub = 0; sub < KC_N_SUB; sub++) {
        u64 s[KC_LANES];
        // pseudo-random input state (timing bench).
        u64 seed = (u64)blk * 0x9E3779B97F4A7C15ull + (u64)sub * 0xD1B54A32D192ED03ull + 1;
        for (int i = 0; i < KC_LANES; i++) {
            u64 v = seed + (u64)i * 0xBF58476D1CE4E5B9ull;
            v = (v ^ (v >> 30)) * 0x94D049BB133111EBull; v ^= v >> 31; s[i] = v;
        }
        // state_0 → slot 2*sub (z & a = value, b = all-ones for the linear self-loops).
        u64* z0 = zb + (2*sub) * KC_SLOT_U64; u64* a0 = ab + (2*sub) * KC_SLOT_U64; u64* b0 = bb + (2*sub) * KC_SLOT_U64;
        for (int i = 0; i < KC_LANES; i++) { z0[i] = s[i]; a0[i] = s[i]; b0[i] = ALL; }

        for (int r = 0; r < 24; r++) {
            kc_theta(s);
            u64 phi[KC_LANES]; kc_rho_pi(s, phi);
            // χ AND-marginals: t[x,y] = ¬phi[(x+1)%5,y] & phi[(x+2)%5,y]; state' = phi ^ t.
            u64 t[KC_LANES], na[KC_LANES], nb[KC_LANES];
#pragma unroll
            for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) {
                u64 p1 = phi[KC_LI((x+1)%5, y)], p2 = phi[KC_LI((x+2)%5, y)];
                na[KC_LI(x,y)] = ~p1; nb[KC_LI(x,y)] = p2;
                t[KC_LI(x,y)] = (~p1) & p2;
                s[KC_LI(x,y)] = phi[KC_LI(x,y)] ^ t[KC_LI(x,y)];
            }
            s[KC_LI(0,0)] ^= KC_RC[r];
            // write t (z) and the AND operands (a=¬phi_{x+1}, b=phi_{x+2}).
            long long toff = KC_T_BASE_U64 + (long long)(sub*24 + r) * KC_LANES;
            for (int i = 0; i < KC_LANES; i++) { zb[toff+i] = t[i]; ab[toff+i] = na[i]; bb[toff+i] = nb[i]; }
        }
        // state_24 → slot 2*sub+1 (pin rows: z & a = value, b = all-ones).
        u64* z1 = zb + (2*sub+1) * KC_SLOT_U64; u64* a1 = ab + (2*sub+1) * KC_SLOT_U64; u64* b1 = bb + (2*sub+1) * KC_SLOT_U64;
        for (int i = 0; i < KC_LANES; i++) { z1[i] = s[i]; a1[i] = s[i]; b1[i] = ALL; }
    }
    zb[KC_ZCONST_U64] |= 1ull; ab[KC_ZCONST_U64] |= 1ull; bb[KC_ZCONST_U64] |= 1ull;  // Z_CONST const row
}

inline void launch_keccak3_witness_blocks(int n_blocks, long long n_total, u64* z, u64* a, u64* b) {
    long long blocks = (n_total + KC_TPB - 1) / KC_TPB;
    keccak3_witness_blocks<<<(unsigned)blocks, KC_TPB>>>(n_blocks, n_total, z, a, b);
}

// Lincheck stripe transpose for K_LOG=17 (port of blake3_lincheck_transpose with the
// keccak block size). One thread per (group-of-8-blocks, word i): the 64-byte output row
// is 8 independent 8x8 bit-transposes. Produces d_zlin for lincheck at k_log=17.
__device__ __forceinline__ u64 kc_transpose8(u64 x) {
    u64 t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAull; x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCull; x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ull; x ^= t ^ (t << 28);
    return x;
}
__global__ void keccak3_lincheck_transpose(const u64* __restrict__ z, long long n_total,
                                           uint8_t* __restrict__ z_lincheck) {
    long long total = (n_total / 8) * (long long)KC_U64_PER_BLOCK;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    long long g = tid / KC_U64_PER_BLOCK;
    int i = (int)(tid - g * KC_U64_PER_BLOCK);
    u64 lanes[8];
#pragma unroll
    for (int lane = 0; lane < 8; lane++)
        lanes[lane] = z[(8 * g + lane) * (long long)KC_U64_PER_BLOCK + i];
    u64* dst = (u64*)(z_lincheck + g * (long long)KC_K + (long long)i * 64);
#pragma unroll
    for (int b_chunk = 0; b_chunk < 8; b_chunk++) {
        u64 src = 0;
#pragma unroll
        for (int r = 0; r < 8; r++) src |= ((lanes[r] >> (8 * b_chunk)) & 0xFFull) << (8 * r);
        dst[b_chunk] = kc_transpose8(src);
    }
}
inline void launch_keccak3_lincheck_transpose(const u64* z, long long n_total, uint8_t* z_lincheck) {
    long long total = (n_total / 8) * (long long)KC_U64_PER_BLOCK;
    keccak3_lincheck_transpose<<<(unsigned)((total + 255) / 256), 256>>>(z, n_total, z_lincheck);
}
