// Bit-for-bit validation of the CUDA FRI fold kernel (step 1 of the GPU
// pcs::open / Ligerito port, GPU_OPEN_PLAN.md) against the flock CPU oracle
// dumped by `src/bin/dump_fold_vectors.rs` (FOLD format).
//
// Pipeline (mirrors src/pcs/basefold.rs::fri_fold_codeword, called log_dim
// times at layer = k_code - round - 1, basefold.rs:604):
//   1. upload the initial single-lane codeword + twiddle table (k_code).
//   2. for each round: device fri_fold at this round's layer + challenge,
//      halving the codeword.
//   3. after EVERY round, compare the device codeword to the golden folded
//      codeword bit-for-bit (not just the final round).
//
// Build:  make test_open_fold
// Run:    (from repo root)
//           cargo run --release --bin dump_fold_vectors -- cuda-ghash/fold_vectors.bin 12 1
//         (from cuda-ghash/)
//           ./test_open_fold fold_vectors.bin
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "fri_fold.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

static uint32_t rd_u32(FILE* f) {
    uint32_t v = 0;
    if (fread(&v, 4, 1, f) != 1) { printf("short read (u32)\n"); exit(1); }
    return v;
}
static F128 rd_f128(FILE* f) {
    u64 v[2];
    if (fread(v, 8, 2, f) != 2) { printf("short read (f128)\n"); exit(1); }
    return F128{v[0], v[1]};
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "fold_vectors.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s (run dump_fold_vectors first)\n", path); return 1; }

    uint32_t magic = rd_u32(f);
    if (magic != 0x464F4C44u) { printf("bad file (magic=%08x, want FOLD)\n", magic); return 1; }
    int k_code       = (int)rd_u32(f);
    int log_inv_rate = (int)rd_u32(f);
    int log_dim      = (int)rd_u32(f);
    uint32_t init_len = rd_u32(f);
    if (init_len != (1u << k_code)) { printf("init_len %u != 2^k_code\n", init_len); return 1; }

    std::vector<F128> codeword(init_len);
    for (uint32_t i = 0; i < init_len; i++) codeword[i] = rd_f128(f);

    printf("FOLD: k_code=%d rate=1/%d log_dim=%d rounds, init_len=%u\n",
           k_code, 1 << log_inv_rate, log_dim, init_len);

    // Twiddle table (host build), L = k_code — same path the NTT validates.
    TwiddleTable tt = build_twiddle_table(k_code);

    // Ping-pong device buffers (init_len is the max size).
    F128 *d_a = nullptr, *d_b = nullptr, *d_tw = nullptr;
    CK(cudaMalloc(&d_a, (size_t)init_len * sizeof(F128)));
    CK(cudaMalloc(&d_b, (size_t)init_len * sizeof(F128)));
    CK(cudaMalloc(&d_tw, tt.data.size() * sizeof(F128)));
    CK(cudaMemcpy(d_a, codeword.data(), (size_t)init_len * sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));

    long long cur_len = init_len;
    F128* d_cur = d_a;
    F128* d_nxt = d_b;
    std::vector<F128> got, golden;

    for (int j = 0; j < log_dim; j++) {
        F128 r = rd_f128(f);
        uint32_t out_len = rd_u32(f);
        golden.resize(out_len);
        for (uint32_t i = 0; i < out_len; i++) golden[i] = rd_f128(f);

        long long new_len = cur_len / 2;
        if ((long long)out_len != new_len) {
            printf("round %d: oracle out_len %u != cur_len/2 %lld\n", j, out_len, new_len);
            return 1;
        }
        int layer = k_code - j - 1;
        launch_fri_fold(d_cur, d_nxt, d_tw, tt, layer, r, new_len);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        got.resize(new_len);
        CK(cudaMemcpy(got.data(), d_nxt, (size_t)new_len * sizeof(F128), cudaMemcpyDeviceToHost));

        size_t bad = 0, first = 0;
        for (long long i = 0; i < new_len; i++) {
            if (got[i].lo != golden[i].lo || got[i].hi != golden[i].hi) {
                if (!bad) first = i;
                bad++;
            }
        }
        if (bad) {
            F128 g = got[first], e = golden[first];
            printf("FOLD FAIL round %d (layer %d): %zu/%lld mismatch; first @%zu: "
                   "got %016llx:%016llx exp %016llx:%016llx\n",
                   j, layer, bad, new_len, first,
                   (unsigned long long)g.hi, (unsigned long long)g.lo,
                   (unsigned long long)e.hi, (unsigned long long)e.lo);
            return 1;
        }
        printf("  round %2d  layer %2d  len %8lld -> %8lld  OK\n", j, layer, cur_len, new_len);

        cur_len = new_len;
        F128* t = d_cur; d_cur = d_nxt; d_nxt = t;
    }
    fclose(f);

    printf("FOLD OK: all %d FRI rounds match flock bit-for-bit (final len %lld = 2^%d)\n",
           log_dim, cur_len, log_inv_rate);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_tw);
    return 0;
}
