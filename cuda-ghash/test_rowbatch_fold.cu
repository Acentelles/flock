// Bit-for-bit validation of the CUDA row-batch fold kernel (step 2 of the GPU
// pcs::open / Ligerito port, GPU_OPEN_PLAN.md) against the flock CPU oracle
// dumped by `src/bin/dump_rowbatch_vectors.rs` (RBF1 format).
//
// Pipeline (mirrors src/pcs/basefold.rs::row_batch_fold_all):
//   1. upload the interleaved codeword + lane challenges.
//   2. device row_batch_fold: collapse each position's num_ntts lanes to one.
//   3. compare the device output to the golden collapsed codeword bit-for-bit.
//
// Build:  make test_rowbatch_fold
// Run:    (from repo root)
//           cargo run --release --bin dump_rowbatch_vectors -- cuda-ghash/rowbatch_vectors.bin 12 5
//         (from cuda-ghash/)
//           ./test_rowbatch_fold rowbatch_vectors.bin
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "row_batch_fold.cuh"

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
    const char* path = argc > 1 ? argv[1] : "rowbatch_vectors.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s (run dump_rowbatch_vectors first)\n", path); return 1; }

    uint32_t magic = rd_u32(f);
    if (magic != 0x52424631u) { printf("bad file (magic=%08x, want RBF1)\n", magic); return 1; }
    int k_code         = (int)rd_u32(f);
    int log_batch_size = (int)rd_u32(f);
    int num_ntts       = (int)rd_u32(f);
    uint32_t cw_len    = rd_u32(f);
    if (num_ntts > RBF_MAX_LANES) {
        printf("num_ntts %d exceeds RBF_MAX_LANES %d (raise the define)\n", num_ntts, RBF_MAX_LANES);
        return 1;
    }

    std::vector<F128> codeword(cw_len);
    for (uint32_t i = 0; i < cw_len; i++) codeword[i] = rd_f128(f);
    std::vector<F128> chal(log_batch_size);
    for (int i = 0; i < log_batch_size; i++) chal[i] = rd_f128(f);
    uint32_t out_len = rd_u32(f);
    std::vector<F128> golden(out_len);
    for (uint32_t i = 0; i < out_len; i++) golden[i] = rd_f128(f);
    fclose(f);

    long long n_positions = (long long)out_len;
    if ((long long)cw_len != n_positions * num_ntts) {
        printf("inconsistent file: cw_len=%u != n_positions*num_ntts=%lld\n",
               cw_len, n_positions * num_ntts);
        return 1;
    }
    printf("RBF1: k_code=%d log_batch_size=%d num_ntts=%d n_positions=%lld cw_len=%u\n",
           k_code, log_batch_size, num_ntts, n_positions, cw_len);

    F128 *d_in = nullptr, *d_out = nullptr, *d_chal = nullptr;
    CK(cudaMalloc(&d_in, (size_t)cw_len * sizeof(F128)));
    CK(cudaMalloc(&d_out, (size_t)n_positions * sizeof(F128)));
    CK(cudaMalloc(&d_chal, (size_t)log_batch_size * sizeof(F128)));
    CK(cudaMemcpy(d_in, codeword.data(), (size_t)cw_len * sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_chal, chal.data(), (size_t)log_batch_size * sizeof(F128), cudaMemcpyHostToDevice));

    launch_row_batch_fold(d_in, d_out, d_chal, log_batch_size, num_ntts, n_positions);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<F128> got(n_positions);
    CK(cudaMemcpy(got.data(), d_out, (size_t)n_positions * sizeof(F128), cudaMemcpyDeviceToHost));

    size_t bad = 0, first = 0;
    for (long long i = 0; i < n_positions; i++) {
        if (got[i].lo != golden[i].lo || got[i].hi != golden[i].hi) {
            if (!bad) first = i;
            bad++;
        }
    }
    if (bad) {
        F128 g = got[first], e = golden[first];
        printf("ROW-BATCH FAIL: %zu/%lld positions mismatch; first @%zu: "
               "got %016llx:%016llx exp %016llx:%016llx\n",
               bad, n_positions, first,
               (unsigned long long)g.hi, (unsigned long long)g.lo,
               (unsigned long long)e.hi, (unsigned long long)e.lo);
        return 1;
    }
    printf("ROW-BATCH OK: all %lld positions match flock bit-for-bit "
           "(%d lanes -> 1 each)\n", n_positions, num_ntts);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_chal);
    return 0;
}
