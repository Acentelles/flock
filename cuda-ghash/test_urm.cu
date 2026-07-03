// Vanilla GPU port of the binary AG-code URM round-1 AB message, oracle-gated
// against a host port of benches/urm_bitslice.rs::scalar_ref.
//
// Per 128-row block o (am[o][r], bm[o][r] are the 64-bit skip messages; eq[o] an
// F128 weight):
//   af[k] = parity(M_MASK[k] & am),  bf[k] = parity(M_MASK[k] & bm)   k<160  (encode)
//   pr[j] = D1/D2/D3 Hasse product (AND/XOR)                                  (product)
//   word[j] |= pr[j] << r  over the 128 rows  -> F128                         (free reinterpret)
//   res[j] += eq[o] * word[j]                                                  (fold, karatsuba)
//
// "Vanilla" = one thread per block, encode via per-coordinate popcount-parity
// (neither bitsliced nor LUT). This is the correctness baseline; the two encode
// variants to compare next are (a) vec-matrix bitslice, (b) LUT + transpose.
//
// Build: make test_urm     Run: ./test_urm
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "f128.cuh"
#include "ntt_host.hpp"      // host F128 mul/add (f128_mul_hd / f128_add_hd)
#include "urm_mmask.h"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

__device__ __constant__ u64 M_MASK[160] = URM_MMASK_INIT;
static const u64 M_MASK_H[160] = URM_MMASK_INIT;

// splitmix64, identical to benches/urm_bitslice.rs::Rng::n
struct Rng { u64 s; u64 n() {
    s += 0x9E3779B97F4A7C15ull; u64 z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
} };

static inline int parH(u64 mask, u64 msg) { return __builtin_popcountll(mask & msg) & 1; }

// Host ground truth — faithful port of scalar_ref (by-point layout).
static void scalar_ref(const std::vector<std::vector<u64>>& a, const std::vector<std::vector<u64>>& b,
                       const std::vector<F128>& eq, int n, F128* res) {
    for (int j = 0; j < 160; j++) res[j] = F128{0, 0};
    for (int o = 0; o < n; o++) {
        F128 word[160]; for (int j = 0; j < 160; j++) word[j] = F128{0, 0};
        for (int r = 0; r < 128; r++) {
            u64 am = a[o][r], bm = b[o][r];
            int af[160], bf[160];
            for (int k = 0; k < 160; k++) { af[k] = parH(M_MASK_H[k], am); bf[k] = parH(M_MASK_H[k], bm); }
            int pr[160];
            for (int p = 0; p < 64; p++) pr[p] = (af[p] & ((bm>>p)&1)) ^ (((am>>p)&1) & bf[p]);
            for (int p = 0; p < 64; p++) pr[64+p] = (af[64+p] & ((bm>>p)&1)) ^ (af[p] & bf[p]) ^ (((am>>p)&1) & bf[64+p]);
            for (int p = 0; p < 32; p++) pr[128+p] = (af[128+p] & ((bm>>p)&1)) ^ (af[64+p] & bf[p]) ^ (af[p] & bf[64+p]) ^ (((am>>p)&1) & bf[128+p]);
            for (int j = 0; j < 160; j++) if (pr[j]) { if (r < 64) word[j].lo |= 1ull<<r; else word[j].hi |= 1ull<<(r-64); }
        }
        for (int j = 0; j < 160; j++) res[j] = f128_add_hd(res[j], f128_mul_hd(eq[o], word[j]));
    }
}

// Vanilla device kernel: one thread per block.
__global__ void urm_vanilla(const u64* amsg, const u64* bmsg, const F128* eq, F128* res, int n) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    F128 word[160];
    for (int j = 0; j < 160; j++) word[j] = F128{0, 0};
    for (int r = 0; r < 128; r++) {
        u64 am = amsg[(size_t)o*128 + r], bm = bmsg[(size_t)o*128 + r];
        unsigned char af[160], bf[160];
        for (int k = 0; k < 160; k++) { af[k] = __popcll(M_MASK[k] & am) & 1; bf[k] = __popcll(M_MASK[k] & bm) & 1; }
        for (int p = 0; p < 64; p++) {
            int ax = (am>>p)&1, bx = (bm>>p)&1;
            if ((af[p] & bx) ^ (ax & bf[p])) { if (r<64) word[p].lo|=1ull<<r; else word[p].hi|=1ull<<(r-64); }
        }
        for (int p = 0; p < 64; p++) {
            int ax = (am>>p)&1, bx = (bm>>p)&1;
            if ((af[64+p] & bx) ^ (af[p] & bf[p]) ^ (ax & bf[64+p])) { if (r<64) word[64+p].lo|=1ull<<r; else word[64+p].hi|=1ull<<(r-64); }
        }
        for (int p = 0; p < 32; p++) {
            int ax = (am>>p)&1, bx = (bm>>p)&1;
            if ((af[128+p] & bx) ^ (af[64+p] & bf[p]) ^ (af[p] & bf[64+p]) ^ (ax & bf[128+p])) { if (r<64) word[128+p].lo|=1ull<<r; else word[128+p].hi|=1ull<<(r-64); }
        }
    }
    F128 e = eq[o];
    for (int j = 0; j < 160; j++) {
        F128 pr = ghash_mul_karatsuba(e, word[j]);
        atomicXor((unsigned long long*)&res[j].lo, pr.lo);
        atomicXor((unsigned long long*)&res[j].hi, pr.hi);
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;     // blocks (128 rows each)
    Rng rng{0xC0DE};
    std::vector<std::vector<u64>> a(n, std::vector<u64>(128)), b(n, std::vector<u64>(128));
    for (int o = 0; o < n; o++) for (int r = 0; r < 128; r++) { a[o][r] = rng.n(); b[o][r] = rng.n(); }
    std::vector<F128> eq(n);
    for (int o = 0; o < n; o++) eq[o] = F128{rng.n(), rng.n()};

    F128 want[160]; scalar_ref(a, b, eq, n, want);

    // Flatten witness for device.
    std::vector<u64> amf((size_t)n*128), bmf((size_t)n*128);
    for (int o = 0; o < n; o++) for (int r = 0; r < 128; r++) { amf[(size_t)o*128+r]=a[o][r]; bmf[(size_t)o*128+r]=b[o][r]; }

    u64 *da, *db; F128 *deq, *dres;
    CK(cudaMalloc(&da, amf.size()*8)); CK(cudaMalloc(&db, bmf.size()*8));
    CK(cudaMalloc(&deq, n*sizeof(F128))); CK(cudaMalloc(&dres, 160*sizeof(F128)));
    CK(cudaMemcpy(da, amf.data(), amf.size()*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, bmf.data(), bmf.size()*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(deq, eq.data(), n*sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemset(dres, 0, 160*sizeof(F128)));

    int tpb = 128, blocks = (n + tpb - 1)/tpb;
    urm_vanilla<<<blocks, tpb>>>(da, db, deq, dres, n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    F128 got[160]; CK(cudaMemcpy(got, dres, 160*sizeof(F128), cudaMemcpyDeviceToHost));

    int bad = 0, first = -1;
    for (int j = 0; j < 160; j++) if (got[j].lo != want[j].lo || got[j].hi != want[j].hi) { if (first<0) first=j; bad++; }
    printf("URM vanilla, n=%d blocks (%d messages)\n", n, n*128);
    printf("want[0] = %016llx:%016llx\n", want[0].hi, want[0].lo);
    if (bad) {
        printf("FAIL: %d/160 coords mismatch; first @%d: got %016llx:%016llx exp %016llx:%016llx\n",
               bad, first, got[first].hi, got[first].lo, want[first].hi, want[first].lo);
        return 1;
    }
    printf("OK: all 160 coords match the scalar oracle bit-for-bit\n");
    cudaFree(da); cudaFree(db); cudaFree(deq); cudaFree(dres);
    return 0;
}
