// Throughput for the zerocheck round-1 (univariate-skip URM, canonical form).
// No oracle — correctness in test_zerocheck_round1. Compare to CPU round-1 (~5 ms @ m=29).
//
// Build:  make bench_zerocheck_round1
// Run:    ./bench_zerocheck_round1            (default sweep)   |   ./bench_zerocheck_round1 29 20
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "zerocheck_round1.cuh"
#include "phi8_table.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void fill_bytes(uint8_t* z, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    u64 x = (u64)i * 0x9E3779B97F4A7C15ull + 1; z[i] = (uint8_t)(x ^ (x >> 13) ^ (x >> 29));
}
__global__ void fill_f128(F128* a, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    u64 x = (u64)i * 0x9E3779B97F4A7C15ull + 1; a[i] = F128{x, x * 0xBF58476D1CE4E5B9ull};
}
static float ev_ms(cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }

static void run_one(int m, int iters) {
    long long rows = 1LL << (m - 6);
    size_t packed = (size_t)1 << (m - 3);
    std::vector<uint8_t> mcol(64 * 64), f8mul((size_t)256 * 256);
    for (size_t i = 0; i < mcol.size(); i++) mcol[i] = (uint8_t)(i * 7 + 1);
    for (size_t i = 0; i < f8mul.size(); i++) f8mul[i] = (uint8_t)(i ^ (i >> 8));
    zc_round1_upload_tables(mcol.data(), f8mul.data(), PHI_8_TABLE);
    // The synthetic M above lacks the Cauchy structure, so the dispatcher would
    // fall back to warp3; the REAL protocol M is Cauchy (see test_zerocheck_round1),
    // so force the production (warp5) path here for representative timing. Kernel
    // timing is data-independent, so the synthetic witness values are fine.
    g_zc_t0_ok = 1;

    uint8_t *d_a, *d_b, *d_c; F128 *d_eq, *d_ab, *d_c_out;
    CK(cudaMalloc(&d_a, packed)); CK(cudaMalloc(&d_b, packed)); CK(cudaMalloc(&d_c, packed));
    CK(cudaMalloc(&d_eq, rows * sizeof(F128)));
    CK(cudaMalloc(&d_ab, 64 * sizeof(F128))); CK(cudaMalloc(&d_c_out, 64 * sizeof(F128)));
    fill_bytes<<<(unsigned)((packed + 255) / 256), 256>>>(d_a, packed);
    fill_bytes<<<(unsigned)((packed + 255) / 256), 256>>>(d_b, packed);
    fill_bytes<<<(unsigned)((packed + 255) / 256), 256>>>(d_c, packed);
    fill_f128<<<(unsigned)((rows + 255) / 256), 256>>>(d_eq, rows);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t = 0;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(e0);
        launch_zc_round1(d_a, d_b, d_c, d_eq, rows, d_ab, d_c_out);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        t += ev_ms(e0, e1);
    }
    t /= iters;
    printf("m=%2d rows=%10lld | round1 %8.3f ms  (witness 3x%.2f GiB)\n",
           m, rows, t, packed / (1024.0 * 1024.0 * 1024.0));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_eq); cudaFree(d_ab); cudaFree(d_c_out);
}

int main(int argc, char** argv) {
    if (argc >= 2) { run_one(atoi(argv[1]), argc > 2 ? atoi(argv[2]) : 20); return 0; }
    printf("zerocheck round-1 (URM, canonical) throughput (RTX 5090, sm_120)\n");
    for (int m = 24; m <= 29; m++) run_one(m, 15);
    return 0;
}
