// BLAKE3 R1CS witness generation on GPU — byte-exact port of
// src/r1cs_hashes/blake3.rs::build_block_witness_ab_packed_into driven by
// common.rs::drive_witness_packed_and_lincheck (the S4 "GPU witness" target).
//
// Per BLAKE3 block (Compression = cv[8], m[16], counter, block_len, flags):
//   - run the 7-round / 8-G trace, materializing every carry/word row,
//   - OR the bits into the block's K/64 = 256-u64 slices of z / a / b
//     (z = witness, a/b = the A·z / B·z products: a gets `left`+word,
//      b gets `right`+0xFFFFFFFF, z gets `carry`+word),
//   - then a separate kernel bit-transposes each 8-block group into the
//     lincheck stripe (K bytes per group).
//
// Pure integer ops (u32/u64) — no field math, independent of f128.cuh.
#pragma once
#include <cstdint>

typedef unsigned long long b3u64;

// ---- constants (verbatim from blake3.rs) ----------------------------------
#define B3_K_LOG 14
#define B3_K (1 << B3_K_LOG)            // 16384
#define B3_U64_PER_BLOCK (B3_K / 64)    // 256
#define B3_N_ROUNDS 7
#define B3_N_G_PER_ROUND 8
#define B3_N_G (B3_N_ROUNDS * B3_N_G_PER_ROUND) // 56
#define B3_WORD_BITS 32
#define B3_CARRY_BITS 31                // WORD_BITS - 1
#define B3_ADDS_PER_G 6
#define B3_G_STRIDE 250                 // 6*31 + 2*32

// layout bases
#define B3_CV_BASE 0
#define B3_OUT_LO_BASE 256
#define B3_Z_CONST_POS 512
#define B3_M_BASE 513
#define B3_T_LO_BASE 1025
#define B3_T_HI_BASE 1057
#define B3_BLEN_BASE 1089
#define B3_FLAGS_BASE 1121
#define B3_GS_BASE 1153
#define B3_OUT_HI_BASE 15153

// record-relative tags
#define B3_REC_C0 0
#define B3_REC_C1 31
#define B3_REC_C2 62
#define B3_REC_C3 93
#define B3_REC_C4 124
#define B3_REC_C5 155
#define B3_REC_LIN0 186                 // ADDS_PER_G * CARRY_BITS
#define B3_REC_LIN1 218                 // REC_LIN0 + WORD_BITS

__device__ __constant__ uint32_t B3_IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

__device__ __constant__ int B3_G_LANES[8][4] = {
    {0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15},
    {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};

__device__ __constant__ int B3_G_MSG_IDX[8][2] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}};

__device__ __constant__ int B3_MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

// ---- bit-packing primitives (verbatim from common.rs) ---------------------
__device__ __forceinline__ void b3_or_bit_at(b3u64* buf, int bit) {
    buf[bit >> 6] |= 1ull << (bit & 63);
}
__device__ __forceinline__ void b3_or_u32_at_bit(b3u64* buf, int bit, uint32_t val) {
    int idx = bit >> 6, s = bit & 63;
    buf[idx] |= (b3u64)val << s;
    if (s > 32) buf[idx + 1] |= (b3u64)val >> (64 - s);
}
// z gets val, a gets val, b gets all-ones (the linear "= word" R1CS rows).
__device__ __forceinline__ void b3_write_lin_word(b3u64* z, b3u64* a, b3u64* b,
                                                  int bit, uint32_t val) {
    b3_or_u32_at_bit(z, bit, val);
    b3_or_u32_at_bit(a, bit, val);
    b3_or_u32_at_bit(b, bit, 0xFFFFFFFFu);
}
__device__ __forceinline__ uint32_t b3_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
// add_carry_parts: sum, left=(x^cin)&0x7FFFFFFF, right=(y^cin)&…, carry=left&right.
__device__ __forceinline__ uint32_t b3_add_carry(uint32_t x, uint32_t y,
                                                 uint32_t* left, uint32_t* right,
                                                 uint32_t* carry) {
    uint32_t sum = x + y;
    uint32_t cin = sum ^ x ^ y;
    uint32_t l = (x ^ cin) & 0x7FFFFFFFu;
    uint32_t r = (y ^ cin) & 0x7FFFFFFFu;
    *left = l; *right = r; *carry = l & r;
    return sum;
}

// BitRecord<4>: push a u32 at record-bit POS, flush ORs the 4-word record into
// `buf` at bit `base` with spill into buf[bi+4].
__device__ __forceinline__ void b3_rec_push(b3u64 rec[4], int pos, uint32_t val) {
    int idx = pos >> 6, s = pos & 63;
    rec[idx] |= (b3u64)val << s;
    if (s > 32) rec[idx + 1] |= (b3u64)val >> (64 - s);
}
__device__ __forceinline__ void b3_rec_flush(const b3u64 rec[4], b3u64* buf, int base) {
    int bi = base >> 6, s = base & 63;
    b3u64 spill = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        buf[bi + j] |= (rec[j] << s) | spill;
        spill = (rec[j] >> 1) >> (63 - s);  // = rec[j] >> (64 - s), no UB at s=0
    }
    buf[bi + 4] |= spill;
}

// ---- per-block witness builder -------------------------------------------
// One thread per block. z/a/b slices must be pre-zeroed (cudaMemset). Padding
// blocks (blk >= n_blocks) leave their slices zero.
__global__ void blake3_witness_blocks(const uint32_t* __restrict__ cv_all,
                                      const uint32_t* __restrict__ m_all,
                                      const b3u64* __restrict__ ctr_all,
                                      const uint32_t* __restrict__ blen_all,
                                      const uint32_t* __restrict__ flags_all,
                                      int n_blocks, long long n_total,
                                      b3u64* __restrict__ z, b3u64* __restrict__ a,
                                      b3u64* __restrict__ b) {
    long long blk = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_total) return;
    if (blk >= n_blocks) return;  // padding: slices already zero

    uint32_t cv[8], m[16];
#pragma unroll
    for (int w = 0; w < 8; w++) cv[w] = cv_all[blk * 8 + w];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = m_all[blk * 16 + i];
    b3u64 counter = ctr_all[blk];
    uint32_t block_len = blen_all[blk];
    uint32_t flags = flags_all[blk];
    uint32_t counter_lo = (uint32_t)counter;
    uint32_t counter_hi = (uint32_t)(counter >> 32);

    // Three-pass build: one 2KB local buffer at a time (which = 0:z, 1:a, 2:b),
    // re-running the cheap BLAKE3 trace each pass. A single 6KB z+a+b footprint
    // is memory-latency-bound (it thrashes L1 across resident warps); a 2KB hot
    // buffer fits far better, and the recomputed trace is pure ALU work that runs
    // ~200x under compute peak (i.e. hidden behind the memory latency).
    //   z ← carry/word, a ← left/word, b ← right/0xFFFFFFFF.  Linear "= word"
    //   rows put `word` in z & a and all-ones in b → LINVAL.
#define LINVAL(v) ((which == 2) ? 0xFFFFFFFFu : (uint32_t)(v))
    b3u64* gout[3] = {z, a, b};
    for (int which = 0; which < 3; which++) {
        b3u64 buf[B3_U64_PER_BLOCK];
#pragma unroll 8
        for (int j = 0; j < B3_U64_PER_BLOCK; j++) buf[j] = 0;

        b3_or_bit_at(buf, B3_Z_CONST_POS);  // z[0]=1 in all three
#pragma unroll
        for (int w = 0; w < 8; w++) b3_or_u32_at_bit(buf, B3_CV_BASE + 32 * w, LINVAL(cv[w]));
#pragma unroll
        for (int i = 0; i < 16; i++) b3_or_u32_at_bit(buf, B3_M_BASE + 32 * i, LINVAL(m[i]));
        b3_or_u32_at_bit(buf, B3_T_LO_BASE, LINVAL(counter_lo));
        b3_or_u32_at_bit(buf, B3_T_HI_BASE, LINVAL(counter_hi));
        b3_or_u32_at_bit(buf, B3_BLEN_BASE, LINVAL(block_len));
        b3_or_u32_at_bit(buf, B3_FLAGS_BASE, LINVAL(flags));

        uint32_t state[16];
#pragma unroll
        for (int w = 0; w < 8; w++) state[w] = cv[w];
#pragma unroll
        for (int w = 0; w < 8; w++) state[8 + w] = B3_IV[w];
        state[12] = counter_lo; state[13] = counter_hi;
        state[14] = block_len;  state[15] = flags;

        int perm[16];
#pragma unroll
        for (int i = 0; i < 16; i++) perm[i] = i;

        for (int r = 0; r < B3_N_ROUNDS; r++) {
            for (int gi = 0; gi < B3_N_G_PER_ROUND; gi++) {
                int g = r * B3_N_G_PER_ROUND + gi;
                int la = B3_G_LANES[gi][0], lb = B3_G_LANES[gi][1];
                int lc = B3_G_LANES[gi][2], ld = B3_G_LANES[gi][3];
                uint32_t mx = m[perm[B3_G_MSG_IDX[gi][0]]];
                uint32_t my = m[perm[B3_G_MSG_IDX[gi][1]]];

                uint32_t a_val = state[la], b_val = state[lb];
                uint32_t c_val = state[lc], d_val = state[ld];

                b3u64 rec[4] = {0, 0, 0, 0};
                uint32_t L, R, C;
#define SEL ((which == 0) ? C : (which == 1) ? L : R)

                uint32_t tmp_0 = b3_add_carry(a_val, b_val, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C0, SEL);
                uint32_t a_1 = b3_add_carry(tmp_0, mx, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C1, SEL);
                uint32_t d_1 = b3_rotr(d_val ^ a_1, 16);
                uint32_t c_1 = b3_add_carry(c_val, d_1, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C2, SEL);
                uint32_t b_1 = b3_rotr(b_val ^ c_1, 12);
                uint32_t tmp_1 = b3_add_carry(a_1, b_1, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C3, SEL);
                uint32_t a_2 = b3_add_carry(tmp_1, my, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C4, SEL);
                uint32_t d_2 = b3_rotr(d_1 ^ a_2, 8);
                uint32_t c_2 = b3_add_carry(c_1, d_2, &L, &R, &C);
                b3_rec_push(rec, B3_REC_C5, SEL);
                uint32_t b_new = b3_rotr(b_1 ^ c_2, 7);
                uint32_t d_new = d_2;
                b3_rec_push(rec, B3_REC_LIN0, LINVAL(b_new));
                b3_rec_push(rec, B3_REC_LIN1, LINVAL(d_new));
#undef SEL
                b3_rec_flush(rec, buf, B3_GS_BASE + B3_G_STRIDE * g);

                state[la] = a_2; state[lb] = b_new; state[lc] = c_2; state[ld] = d_new;
            }
            int next[16];
#pragma unroll
            for (int i = 0; i < 16; i++) next[i] = perm[B3_MSG_PERM[i]];
#pragma unroll
            for (int i = 0; i < 16; i++) perm[i] = next[i];
        }

        // finalization XOR rows
#pragma unroll
        for (int w = 0; w < 8; w++) {
            uint32_t lo = state[w] ^ state[w + 8];
            uint32_t hi = state[w + 8] ^ cv[w];
            b3_or_u32_at_bit(buf, B3_OUT_LO_BASE + 32 * w, LINVAL(lo));
            b3_or_u32_at_bit(buf, B3_OUT_HI_BASE + 32 * w, LINVAL(hi));
        }

        // Write out as 16-byte stores (STG.128); block-major is uncoalesced
        // (2KB-strided per thread), so halving the store count is the lever.
        ulonglong2* gd = (ulonglong2*)(gout[which] + blk * B3_U64_PER_BLOCK);
        const ulonglong2* ld = (const ulonglong2*)buf;
#pragma unroll 8
        for (int j = 0; j < B3_U64_PER_BLOCK / 2; j++) gd[j] = ld[j];
    }
#undef LINVAL
}

// 8x8 bit-matrix transpose of a u64 (byte r = row r, bit t = col t): output
// byte t bit r = input byte r bit t (Hacker's Delight). Matches the scalar
// bit_transpose_64bytes per byte-chunk.
__device__ __forceinline__ b3u64 b3_transpose8(b3u64 x) {
    b3u64 t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAull; x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCull; x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ull; x ^= t ^ (t << 28);
    return x;
}

// ---- lincheck stripe transpose (port of transpose_8_u64s_to_64_bytes ->
// bit_transpose_64bytes). One thread per (group, word i): the 64-byte output
// row is 8 independent 8x8 bit-transposes (one per byte-chunk). ----
__global__ void blake3_lincheck_transpose(const b3u64* __restrict__ z, long long n_total,
                                          uint8_t* __restrict__ z_lincheck) {
    long long total = (n_total / 8) * (long long)B3_U64_PER_BLOCK;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    long long g = tid / B3_U64_PER_BLOCK;
    int i = (int)(tid - g * B3_U64_PER_BLOCK);

    b3u64 lanes[8];
#pragma unroll
    for (int lane = 0; lane < 8; lane++)
        lanes[lane] = z[(8 * g + lane) * (long long)B3_U64_PER_BLOCK + i];

    b3u64* dst = (b3u64*)(z_lincheck + g * (long long)B3_K + (long long)i * 64);
#pragma unroll
    for (int b_chunk = 0; b_chunk < 8; b_chunk++) {
        // src byte r = byte b_chunk of lanes[r].
        b3u64 src = 0;
#pragma unroll
        for (int r = 0; r < 8; r++)
            src |= ((lanes[r] >> (8 * b_chunk)) & 0xFFull) << (8 * r);
        dst[b_chunk] = b3_transpose8(src);  // LE: byte t → out[b_chunk*8 + t]
    }
}

// ---- host launchers -------------------------------------------------------
#ifndef B3_TPB
#define B3_TPB 128
#endif

inline void launch_blake3_witness_blocks(const uint32_t* cv, const uint32_t* m,
                                         const b3u64* ctr, const uint32_t* blen,
                                         const uint32_t* flags, int n_blocks,
                                         long long n_total, b3u64* z, b3u64* a, b3u64* b) {
    long long blocks = (n_total + B3_TPB - 1) / B3_TPB;
    blake3_witness_blocks<<<(unsigned)blocks, B3_TPB>>>(cv, m, ctr, blen, flags,
                                                        n_blocks, n_total, z, a, b);
}

inline void launch_blake3_lincheck_transpose(const b3u64* z, long long n_total,
                                             uint8_t* z_lincheck) {
    long long total = (n_total / 8) * (long long)B3_U64_PER_BLOCK;
    long long blocks = (total + 255) / 256;
    blake3_lincheck_transpose<<<(unsigned)blocks, 256>>>(z, n_total, z_lincheck);
}
