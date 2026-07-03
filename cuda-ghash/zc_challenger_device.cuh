// On-device Fiat-Shamir challenger for the resident zerocheck tail. Bit-identical
// device port of challenger.hpp's Sha256 + FsChallenger observe/sample, so the tail's
// 23 rounds can run as a kernel sequence on one stream with NO host round-trips: each
// round a single-thread kernel observes the message and samples rho entirely on device.
// Only the challenger state is shipped H2D once before the loop and D2H once after.
#pragma once
#include "f128.cuh"
#include "sha256.cuh"     // sha256_compress (bit-identical to the host process)

// Device SHA-256 incremental state (mirrors challenger.hpp::Sha256).
struct ZcSha {
    uint32_t h[8];
    unsigned long long total_len;
    uint8_t buf[64];
    unsigned buf_len;
};
__device__ __forceinline__ void zcsha_reset(ZcSha& s) {
    s.h[0]=0x6a09e667u; s.h[1]=0xbb67ae85u; s.h[2]=0x3c6ef372u; s.h[3]=0xa54ff53au;
    s.h[4]=0x510e527fu; s.h[5]=0x9b05688cu; s.h[6]=0x1f83d9abu; s.h[7]=0x5be0cd19u;
    s.total_len = 0; s.buf_len = 0;
}
__device__ __forceinline__ void zcsha_update(ZcSha& s, const uint8_t* data, unsigned len) {
    s.total_len += len;
    while (len > 0) {
        unsigned take = 64 - s.buf_len; if (take > len) take = len;
        for (unsigned i = 0; i < take; i++) s.buf[s.buf_len + i] = data[i];
        s.buf_len += take; data += take; len -= take;
        if (s.buf_len == 64) { sha256_compress(s.h, s.buf); s.buf_len = 0; }
    }
}
// Finalize MUTATES (call on a copy to keep absorbing). Writes 32-byte big-endian digest.
__device__ __forceinline__ void zcsha_finalize(ZcSha& s, uint8_t out[32]) {
    unsigned long long bitlen = s.total_len * 8ull;
    s.buf[s.buf_len++] = 0x80;
    if (s.buf_len > 56) { while (s.buf_len < 64) s.buf[s.buf_len++] = 0; sha256_compress(s.h, s.buf); s.buf_len = 0; }
    while (s.buf_len < 56) s.buf[s.buf_len++] = 0;
    for (int i = 0; i < 8; i++) s.buf[56 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    sha256_compress(s.h, s.buf);
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(s.h[i] >> 24); out[4*i+1] = (uint8_t)(s.h[i] >> 16);
        out[4*i+2] = (uint8_t)(s.h[i] >> 8);  out[4*i+3] = (uint8_t)(s.h[i]);
    }
}
// FsChallenger op/kind tags (must match challenger.hpp).
#define ZC_OP_OBSERVE 0x03
#define ZC_OP_SQUEEZE 0x04
#define ZC_KIND_SCALAR 0x01
__device__ __forceinline__ void zc_le64(uint8_t* b, unsigned long long v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
}
__device__ __forceinline__ unsigned long long zc_rd_le64(const uint8_t* b) {
    unsigned long long v = 0; for (int i = 0; i < 8; i++) v |= (unsigned long long)b[i] << (8 * i); return v;
}
__device__ __forceinline__ void zc_observe_f128(ZcSha& s, F128 v) {
    uint8_t op[2] = {ZC_OP_OBSERVE, ZC_KIND_SCALAR}; zcsha_update(s, op, 2);
    uint8_t b[16]; zc_le64(b, v.lo); zc_le64(b + 8, v.hi); zcsha_update(s, b, 16);
}
// sample_f128: squeeze 16 bytes as SHA256(state || ctr=0) without mutating, then re-absorb.
__device__ __forceinline__ F128 zc_sample_f128(ZcSha& s) {
    uint8_t op[2] = {ZC_OP_SQUEEZE, ZC_KIND_SCALAR}; zcsha_update(s, op, 2);
    ZcSha h = s;                       // clone live state
    uint8_t cb[8]; zc_le64(cb, 0ull); zcsha_update(h, cb, 8);
    uint8_t block[32]; zcsha_finalize(h, block);
    uint8_t buf[16]; for (int i = 0; i < 16; i++) buf[i] = block[i];
    zcsha_update(s, buf, 16);          // re-absorb
    return F128{zc_rd_le64(buf), zc_rd_le64(buf + 8)};
}

// One tail round on device: observe (m1, mi), sample rho. Updates the persistent state
// in *st, writes rho to *rho_out (read by the next fold) and *rho_store (kept for host).
__global__ void zt_chal_round(ZcSha* st, const F128* m1, const F128* mi,
                              F128* rho_out, F128* rho_store, F128* m1log, F128* milog) {
    if (threadIdx.x || blockIdx.x) return;
    ZcSha s = *st;
    F128 a = *m1, b = *mi;
    zc_observe_f128(s, a);
    zc_observe_f128(s, b);
    F128 rho = zc_sample_f128(s);
    *st = s; *rho_out = rho; if (rho_store) *rho_store = rho;
    if (m1log) *m1log = a; if (milog) *milog = b;   // optional per-round log for validation
}
