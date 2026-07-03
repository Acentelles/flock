// Throughput baseline for the FRI fold kernel (GPU pcs::open / Ligerito,
// step 1; GPU_OPEN_PLAN.md). No oracle needed — correctness is validated
// separately by test_open_fold; here we only time. The codeword is generated
// on-device, so this scales to realistic m.
//
// What's timed: the full FRI fold *cascade* the prover runs in basefold —
// `log_dim` rounds, each halving the (single-lane, post-row-batch) codeword at
// `layer = k_code - round - 1` (src/pcs/basefold.rs:604). The buffer here is the
// single-lane codeword `2^k_code` (the row-batch fold already collapsed the
// `num_ntts` lanes), NOT the full interleaved commit codeword — so for a given
// m it is `num_ntts`× smaller than bench_commit_ntt's buffer.
//
// Params follow src/pcs/commit.rs (LOG_PACKING = 7), same mapping as
// bench_commit_ntt so m lines up across the two benches:
//   log_msg_len = m - 7;  log_dim = log_msg_len - log_batch_size
//   k_code      = log_dim + log_inv_rate          (single-lane codeword = 2^k_code)
//   n_rounds    = log_dim  FRI folds
//
// Per fold_pair: 2 GF(2^128) muls (v·twiddle, r·(u+v)); per round it reads the
// whole current buffer + writes its half.
//
// Build:  make bench_open_fold
// Run:    ./bench_open_fold 33 1 5 [iters]   (default sweep if no args)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "fri_fold.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

// Deterministic on-device fill (any bit pattern is a valid F128).
__global__ void fill_kernel(F128* d, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x = (u64)i * 0x9E3779B97F4A7C15ull + 1;
    d[i] = F128{x, x * 0xBF58476D1CE4E5B9ull};
}

static void run_one(int m, int log_inv_rate, int log_batch_size, int iters) {
    const int LOG_PACKING = 7;
    int log_msg_len = m - LOG_PACKING;
    int log_dim     = log_msg_len - log_batch_size;
    int k_code      = log_dim + log_inv_rate;       // single-lane codeword = 2^k_code
    long long init_len = 1LL << k_code;
    int n_rounds    = log_dim;                       // FRI folds (k_code -> log_inv_rate)

    double gib = init_len * 16.0 / (1024.0 * 1024.0 * 1024.0);

    F128 *d_a = nullptr, *d_b = nullptr, *d_tw = nullptr;
    CK(cudaMalloc(&d_a, init_len * sizeof(F128)));
    CK(cudaMalloc(&d_b, init_len * sizeof(F128)));
    TwiddleTable tt = build_twiddle_table(k_code);
    CK(cudaMalloc(&d_tw, tt.data.size() * sizeof(F128)));
    CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));

    // Per-round fold challenges (values irrelevant for timing).
    std::vector<F128> chal(n_rounds);
    u64 s = 0xC0FFEEull;
    for (int j = 0; j < n_rounds; j++) {
        s += 0x9E3779B97F4A7C15ull;
        u64 z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        chal[j] = F128{z ^ (z >> 31), z * 0x2545F4914F6CDD1Dull};
    }

    int tpb = 256;
    long long fill_blocks = (init_len + tpb - 1) / tpb;
    fill_kernel<<<(unsigned)fill_blocks, tpb>>>(d_a, init_len);
    CK(cudaGetLastError());

    auto run_cascade = [&]() {
        F128* cur = d_a; F128* nxt = d_b;
        long long len = init_len;
        for (int j = 0; j < n_rounds; j++) {
            int layer = k_code - j - 1;
            long long new_len = len / 2;
            launch_fri_fold(cur, nxt, d_tw, tt, layer, chal[j], new_len, tpb);
            len = new_len;
            F128* t = cur; cur = nxt; nxt = t;
        }
    };

    run_cascade();                  // warm-up
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int it = 0; it < iters; it++) run_cascade();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms_total = 0; CK(cudaEventElapsedTime(&ms_total, a, b));
    double ms = ms_total / iters;

    // Work tallies summed across the halving cascade.
    double muls = 0.0, bytes = 0.0;
    long long len = init_len;
    for (int j = 0; j < n_rounds; j++) {
        long long nl = len / 2;
        muls  += 2.0 * (double)nl;                  // 2 muls per fold_pair
        bytes += (double)(len + nl) * 16.0;         // read full + write half
        len = nl;
    }
    double gmuls = muls / (ms * 1e-3) / 1e9;
    double gbps  = bytes / (ms * 1e-3) / 1e9;

    printf("  m=%-2d rate=1/%-2d batch=%d | k_code=%2d rounds=%2d buf=%6.3f GiB | "
           "%8.3f ms  %7.2f GMul/s  %7.1f GB/s\n",
           m, 1 << log_inv_rate, log_batch_size, k_code, n_rounds, gib, ms, gmuls, gbps);

    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_tw);
}

int main(int argc, char** argv) {
    int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s | %d SMs | sm_%d%d\n\n", p.name, p.multiProcessorCount, p.major, p.minor);

    if (argc >= 4) {
        int m = atoi(argv[1]), r = atoi(argv[2]), b = atoi(argv[3]);
        int iters = argc >= 5 ? atoi(argv[4]) : 20;
        printf("== FRI fold cascade (basefold, karatsuba+clmad), %d iters ==\n", iters);
        run_one(m, r, b, iters);
        return 0;
    }

    printf("== FRI fold cascade (basefold, karatsuba+clmad), rate 1/2, batch 5 ==\n");
    for (int m = 20; m <= 31; m += (m < 26 ? 2 : 1)) {
        int iters = m >= 28 ? 10 : 20;
        run_one(m, 1, 5, iters);
    }
    return 0;
}
