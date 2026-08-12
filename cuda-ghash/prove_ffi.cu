// C-ABI GPU prover for the full R1CS Ligerito proof, linked into Rust by
// crates/flock-cuda-ffi. Reproduces crates/flock-prover/src/prover.rs::
// prove_ligerito byte-for-byte on the transcript:
//   commit -> bind_statement -> zerocheck (+s_hat_v_c) -> lincheck (+z_vec)
//   -> ring-switch batch -> ligerito recursion, capturing every proof field.
//
// Orchestration uses the optimized kernels exercised by bench_ligerito.cu.
// Protocol constants the Rust side owns (statement digest, zerocheck tables,
// ligerito config) are passed in so both sides share one source of truth.
//
// Output: flat little-endian byte stream (see FfiWriter) that the Rust test
// parses back into the typed proof structs; layout must match
// crates/flock-cuda-ffi/tests/gpu_roundtrip.rs::parse_proof.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "ntt_f128.cuh"
#include "merkle.cuh"
#include "merkle_open.hpp"
#include "merkle_open_device.cuh"
#include "induce_sumcheck.cuh"
#include "ntt_transpose.cuh"
#include "introduce_glue.cuh"
#include "sumcheck_ab.cuh"
#include "lincheck.cuh"
#include "blake3_witness.cuh"
#include "zerocheck_round1.cuh"
#include "zerocheck_round1_cpustyle.cuh"
#include "zerocheck_round2.cuh"
#include "zerocheck_tail.cuh"
#include "phi8_table.cuh"
#include "challenger.hpp"
#include "zc_challenger_device.cuh"
#include "pow_grind.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("FFI CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 100; } } while(0)

namespace {

using std::vector;

template <class T>
cudaError_t ffi_malloc(T** ptr, size_t bytes) {
    return cudaMallocAsync((void**)ptr, bytes, 0);
}

cudaError_t ffi_free(void* ptr) { return cudaFreeAsync(ptr, 0); }

cudaError_t configure_memory_pool() {
    cudaMemPool_t pool;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, 0);
    if (err != cudaSuccess) return err;
    uint64_t keep = UINT64_MAX;
    return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
}

F128 ADD(F128 a, F128 b) { return f128_add_hd(a, b); }
F128 MUL(F128 a, F128 b) { return f128_mul_hd(a, b); }
const F128 ONE{1, 0};
ChF128 toch(F128 x) { return ChF128{x.lo, x.hi}; }
F128 frch(ChF128 x) { return F128{x.lo, x.hi}; }

F128 interp3(F128 h0, F128 h1, F128 hinf, F128 rho) {
    F128 c1 = ADD(ADD(h0, h1), hinf);
    return ADD(h0, MUL(rho, ADD(c1, MUL(rho, hinf))));
}

ZcSha pack_challenger(const Sha256& s) {
    ZcSha z;
    for (int i = 0; i < 8; i++) z.h[i] = s.h[i];
    z.total_len = s.total_len;
    for (int i = 0; i < 64; i++) z.buf[i] = s.buf[i];
    z.buf_len = (unsigned)s.buf_len;
    return z;
}

void unpack_challenger(Sha256& s, const ZcSha& z) {
    for (int i = 0; i < 8; i++) s.h[i] = z.h[i];
    s.total_len = z.total_len;
    for (int i = 0; i < 64; i++) s.buf[i] = z.buf[i];
    s.buf_len = z.buf_len;
}

vector<F128> build_eq_host(const F128* r, int n) {
    vector<F128> t; t.reserve((size_t)1 << n); t.push_back(ONE);
    for (int j = 0; j < n; j++) {
        F128 rj = r[j], omr = ADD(ONE, rj);
        size_t len = (size_t)1 << j; t.resize(2 * len);
        for (size_t x = 0; x < len; x++) { F128 v = t[x]; t[x + len] = MUL(v, rj); t[x] = MUL(v, omr); }
    }
    return t;
}

// Lagrange weights at z over 2^k nodes of the PHI domain (off=0: S, off=64: Λ).
vector<F128> lagrange_phi(int k, F128 z, int off) {
    int ell = 1 << k; vector<F128> w(ell);
    for (int i = 0; i < ell; i++) {
        F128 si = PHI_8_TABLE[off + i], num = ONE, den = ONE;
        for (int j = 0; j < ell; j++) {
            if (j == i) continue;
            F128 sj = PHI_8_TABLE[off + j];
            num = MUL(num, ADD(z, sj)); den = MUL(den, ADD(si, sj));
        }
        w[i] = MUL(num, f128_inv_host(den));
    }
    return w;
}

// ---- flat output writer (all integers u64 LE, F128 = lo,hi LE) ----
struct FfiWriter {
    vector<uint8_t> buf;
    void u64(uint64_t v) { for (int i = 0; i < 8; i++) buf.push_back((uint8_t)(v >> (8 * i))); }
    void f128(F128 v) { u64(v.lo); u64(v.hi); }
    void f128s(const vector<F128>& v) { u64(v.size()); for (auto& x : v) f128(x); }
    void hash(const uint8_t* h) { buf.insert(buf.end(), h, h + 32); }
    void hashes(const vector<MHash>& v) { u64(v.size()); for (auto& h : v) hash(h.b); }
    void rows(const vector<F128>& flat, size_t n_rows, size_t row_len) {
        u64(n_rows); u64(row_len);
        for (size_t i = 0; i < n_rows * row_len; i++) f128(flat[i]);
    }
};

// tensor-algebra transpose: out[i].bit[j] = in[j].bit[i] (128x128 bit matrix).
vector<F128> ta_transpose_host(const vector<F128>& in) {
    vector<F128> out(128, F128{0, 0});
    for (int j = 0; j < 128; j++) {
        for (int half = 0; half < 2; half++) {
            u64 bits = half ? in[j].hi : in[j].lo;
            while (bits) {
                int i = 64 * half + __builtin_ctzll(bits);
                if (j < 64) out[i].lo |= (u64)1 << j; else out[i].hi |= (u64)1 << (j - 64);
                bits &= bits - 1;
            }
        }
    }
    return out;
}

// Bench-style deterministic pseudo-random compression inputs (copy of
// bench_ligerito.cu::fill_compressions — every slot is a REAL compression, so
// the witness satisfies the R1CS and the const-pin column is 1 everywhere).
__global__ void ffi_fill_compressions(uint32_t* cv, uint32_t* m, b3u64* ctr, uint32_t* blen,
                                      uint32_t* flags, int n_blocks) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;
    b3u64 s = (b3u64)blk * 0x9E3779B97F4A7C15ull + 1;
#define NXT (s = s * 6364136223846793005ull + 1, (uint32_t)(s >> 33))
    for (int w = 0; w < 8; w++) cv[blk * 8 + w] = NXT;
    for (int i = 0; i < 16; i++) m[blk * 16 + i] = NXT;
    ctr[blk] = ((b3u64)NXT << 32) | NXT; blen[blk] = NXT; flags[blk] = NXT;
#undef NXT
}

// replicate-fill for the unfused commit fallback.
__global__ void replicate_fill_ffi(const F128* __restrict__ msg, F128* __restrict__ cw,
                                   long long cw_len, long long msg_len) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cw_len) return;
    cw[i] = msg[i % msg_len];
}

__global__ void gather_codeword_rows(const F128* __restrict__ codeword,
                                     const unsigned long long* __restrict__ positions,
                                     int n_rows, int row_len, F128* __restrict__ rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)n_rows * row_len;
    if (i >= n) return;
    int row = (int)(i / row_len);
    int col = (int)(i % row_len);
    rows[i] = codeword[positions[row] * row_len + col];
}

__global__ void ring_switch_fold_rows_grouped(const F128* __restrict__ witness,
                                              const F128* __restrict__ suffix, long long len,
                                              int chunks, F128* __restrict__ partials) {
    int bit_base = blockIdx.y * 8;
    F128 acc[8]{};
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < len; i += (long long)chunks * blockDim.x) {
        F128 row = witness[i];
        u64 word = bit_base < 64 ? row.lo : row.hi;
        word >>= bit_base & 63;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if ((word >> j) & 1) {
                acc[j].lo ^= suffix[i].lo;
                acc[j].hi ^= suffix[i].hi;
            }
        }
    }
    __shared__ F128 partial[256];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        partial[threadIdx.x] = acc[j];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x].lo ^= partial[threadIdx.x + stride].lo;
                partial[threadIdx.x].hi ^= partial[threadIdx.x + stride].hi;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            partials[((long long)(bit_base + j) * chunks) + blockIdx.x] = partial[0];
        __syncthreads();
    }
}

__global__ void ring_switch_fold_rows_reduce(const F128* __restrict__ partials,
                                             int chunks, F128* __restrict__ shat) {
    int bit = blockIdx.x;
    F128 acc{0, 0};
    for (int i = threadIdx.x; i < chunks; i += blockDim.x) {
        acc.lo ^= partials[(long long)bit * chunks + i].lo;
        acc.hi ^= partials[(long long)bit * chunks + i].hi;
    }
    __shared__ F128 shared[256];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x].lo ^= shared[threadIdx.x + stride].lo;
            shared[threadIdx.x].hi ^= shared[threadIdx.x + stride].hi;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shat[bit] = shared[0];
}

__global__ void ring_switch_combine_basis(const F128* __restrict__ suffix_ab,
                                          const F128* __restrict__ suffix_c,
                                          const F128* __restrict__ weights_ab,
                                          const F128* __restrict__ weights_c,
                                          long long len, F128* __restrict__ combined) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    F128 acc{0, 0};
    const F128 suffixes[2] = {suffix_ab[i], suffix_c[i]};
    const F128* weights[2] = {weights_ab, weights_c};
    for (int claim = 0; claim < 2; claim++) {
        for (int half = 0; half < 2; half++) {
            u64 bits = half ? suffixes[claim].hi : suffixes[claim].lo;
            while (bits) {
                int bit = __ffsll((long long)bits) - 1;
                F128 w = weights[claim][64 * half + bit];
                acc.lo ^= w.lo;
                acc.hi ^= w.hi;
                bits &= bits - 1;
            }
        }
    }
    combined[i] = acc;
}

int copy_codeword_rows(const F128* d_codeword, const vector<size_t>& positions,
                       int row_len, vector<F128>& rows) {
    vector<unsigned long long> host_positions(positions.begin(), positions.end());
    rows.resize(positions.size() * (size_t)row_len);
    if (rows.empty()) return 0;

    unsigned long long* d_positions;
    F128* d_rows;
    CK(ffi_malloc(&d_positions, host_positions.size() * sizeof(unsigned long long)));
    CK(ffi_malloc(&d_rows, rows.size() * sizeof(F128)));
    CK(cudaMemcpy(d_positions, host_positions.data(),
                  host_positions.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    int threads = 256;
    gather_codeword_rows<<<(unsigned)((rows.size() + threads - 1) / threads), threads>>>(
        d_codeword, d_positions, (int)positions.size(), row_len, d_rows);
    CK(cudaGetLastError());
    CK(cudaMemcpy(rows.data(), d_rows, rows.size() * sizeof(F128), cudaMemcpyDeviceToHost));
    CK(ffi_free(d_positions));
    CK(ffi_free(d_rows));
    return 0;
}

} // namespace

// ---- C ABI ----
extern "C" {

typedef struct {
    // instance
    int m;                        // = 14 + n_blocks_log
    const uint8_t* statement_digest;   // 32 bytes (BLAKE3, computed Rust-side)
    const uint8_t* domain;             // challenger domain
    uint32_t domain_len;
    // lincheck CSC matrices (K = 2^14 columns)
    const uint32_t* a_col_ptr; const uint32_t* a_rows; uint32_t a_nnz;
    const uint32_t* b_col_ptr; const uint32_t* b_rows; uint32_t b_nnz;
    int const_pin_col;            // -1 = none; BLAKE3: 512
    int useful_bits;              // 15409
    int k_log;                    // 14
    // zerocheck round-1 tables (protocol constants, Rust-side source of truth)
    const uint8_t* zc_mcol;       // 64*64
    const uint8_t* zc_f8mul;      // 256*256
    // ligerito config (embedded TOML on the Rust side)
    int initial_k;                // 6
    int num_levels;               // e.g. 5
    const int* log_inv_rates;     // [num_levels]
    const int* recursive_ks;      // [recursive_steps]
    const int* queries;           // [num_levels]
    const int* grinding_bits;     // [num_levels]
    const int* fold_grinding_bits; // [num_levels]
    const int* ood_samples;       // [num_levels]
    int recursive_steps;
    // Optional: dump the generated packed witness (len F128, raw LE) here so a
    // host-side Rust prover can replay the identical instance. NULL = no dump.
    const char* dump_z_path;
} FlockCudaProveParams;

int flock_cuda_device_count() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return -(int)e;
    return n;
}

void flock_cuda_free(uint8_t* p) { free(p); }

// Returns 0 on success; out/out_len receive a malloc'd flat proof stream.
int flock_cuda_prove_blake3(const FlockCudaProveParams* P, uint8_t** out, size_t* out_len) {
    static std::mutex prover_mutex;
    std::lock_guard<std::mutex> prover_lock(prover_mutex);
    const int m = P->m, k_log = P->k_log, k_skip = 6;
    const int log_n = m - 7;                    // packed-witness log length
    const long long len = 1LL << log_n;         // packed F128 elements
    const int n_blocks_log = m - 14;
    const long long n_total = 1LL << n_blocks_log;
    const int n_blocks = (int)n_total;
    if (n_blocks_log < 3) { printf("FFI: m too small\n"); return 101; }
    CK(configure_memory_pool());
    // ================= witness (real BLAKE3 compressions, deterministic) ====
    F128 *df, *d_a, *d_b; uint8_t* d_zlin;
    CK(ffi_malloc(&df, len * sizeof(F128)));
    CK(ffi_malloc(&d_a, len * sizeof(F128)));
    CK(ffi_malloc(&d_b, len * sizeof(F128)));
    CK(ffi_malloc(&d_zlin, (size_t)len * 16));
    {
        uint32_t *d_cv, *d_m, *d_blen, *d_flags; b3u64* d_ctr;
        CK(ffi_malloc(&d_cv, (size_t)n_blocks * 8 * 4)); CK(ffi_malloc(&d_m, (size_t)n_blocks * 16 * 4));
        CK(ffi_malloc(&d_blen, (size_t)n_blocks * 4)); CK(ffi_malloc(&d_flags, (size_t)n_blocks * 4));
        CK(ffi_malloc(&d_ctr, (size_t)n_blocks * 8));
        ffi_fill_compressions<<<(unsigned)((n_blocks + 127) / 128), 128>>>(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks);
        CK(cudaGetLastError());
        launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks, n_total,
                                     (b3u64*)df, (b3u64*)d_a, (b3u64*)d_b);
        launch_blake3_lincheck_transpose((const b3u64*)df, n_total, d_zlin);
        CK(cudaDeviceSynchronize());
        ffi_free(d_cv); ffi_free(d_m); ffi_free(d_blen); ffi_free(d_flags); ffi_free(d_ctr);
    }
    if (P->dump_z_path && P->dump_z_path[0]) {
        vector<F128> z_host(len);
        CK(cudaMemcpy(z_host.data(), df, len * sizeof(F128), cudaMemcpyDeviceToHost));
        FILE* zf = fopen(P->dump_z_path, "wb");
        if (!zf) { printf("FFI: cannot open dump path %s\n", P->dump_z_path); return 103; }
        fwrite(z_host.data(), sizeof(F128), z_host.size(), zf);
        fclose(zf);
    }

    // ================= L0 commit ============================================
    F128* d_cw0; uint8_t* d_tree0; long long l0_bl; int l0_ni; uint8_t l0root[32];
    {
        int k_code = (log_n - P->initial_k) + P->log_inv_rates[0];
        int num_ntts = 1 << P->initial_k;
        l0_bl = 1LL << k_code; l0_ni = num_ntts;
        long long cw_len = l0_bl * num_ntts;
        TwiddleTable tt = build_twiddle_table(k_code);
        F128* d_tw;
        CK(ffi_malloc(&d_cw0, cw_len * sizeof(F128)));
        CK(ffi_malloc(&d_tw, tt.data.size() * sizeof(F128)));
        CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));
        CK(ffi_malloc(&d_tree0, (size_t)(2 * l0_bl - 1) * 32));
        if (ntt_can_fuse_src(k_code - P->log_inv_rates[0])) {
            launch_ntt(d_cw0, d_tw, tt, P->log_inv_rates[0], k_code, num_ntts, 256, false, df, len - 1);
        } else {
            printf("FFI: unfused L0 rate-extend not wired\n"); return 102;
        }
        CK(cudaGetLastError());
        launch_merkle((const uint8_t*)d_cw0, d_tree0, l0_bl, num_ntts * 16);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(l0root, d_tree0 + (size_t)(2 * l0_bl - 2) * 32, 32, cudaMemcpyDeviceToHost));
        ffi_free(d_tw);
    }

    // ================= challenger + statement binding =======================
    FsChallenger ch(P->domain, P->domain_len);
    ch.observe_label((const uint8_t*)"flock-r1cs-v0", 13);
    ch.observe_bytes(P->statement_digest, 32);
    ch.observe_bytes(l0root, 32);

    FfiWriter W;
    W.hash(l0root);                       // commitment root

    // ================= zerocheck (test_zerocheck_full flow) =================
    vector<F128> zc_r1ab(64), zc_r1c(64), zc_m1s, zc_mis;
    F128 zc_z, zc_fa, zc_fb, zc_fc;
    vector<F128> zc_r(m), mlv_rhos;
    {
        const long long n_out = 1LL << (m - 6);
        static std::once_flag zerocheck_setup_once;
        static cudaError_t zerocheck_setup_error = cudaSuccess;
        static vector<uint8_t> setup_zc_mcol, setup_zc_f8mul;
        std::call_once(zerocheck_setup_once, [&] {
            setup_zc_mcol.assign(P->zc_mcol, P->zc_mcol + 64 * 64);
            setup_zc_f8mul.assign(P->zc_f8mul, P->zc_f8mul + 256 * 256);
            zerocheck_setup_error = zc_round1_upload_tables(P->zc_mcol, P->zc_f8mul, PHI_8_TABLE);
        });
        CK(zerocheck_setup_error);
        if (memcmp(P->zc_mcol, setup_zc_mcol.data(), setup_zc_mcol.size()) != 0 ||
            memcmp(P->zc_f8mul, setup_zc_f8mul.data(), setup_zc_f8mul.size()) != 0) {
            printf("FFI: zerocheck tables changed after setup\n");
            return 105;
        }
        F128 *d_eq, *d_r1ab, *d_r1c, *d_ft, *d_am, *d_bm, *d_amn, *d_bmn, *d_p1, *d_pinf, *d_m1d, *d_mid;
        CK(ffi_malloc(&d_eq, (1LL << (m - 13)) * sizeof(F128)));
        CK(ffi_malloc(&d_r1ab, 64 * sizeof(F128))); CK(ffi_malloc(&d_r1c, 64 * sizeof(F128)));
        CK(ffi_malloc(&d_ft, 8 * 256 * sizeof(F128)));
        CK(ffi_malloc(&d_am, n_out * sizeof(F128))); CK(ffi_malloc(&d_bm, n_out * sizeof(F128)));
        CK(ffi_malloc(&d_amn, n_out * sizeof(F128))); CK(ffi_malloc(&d_bmn, n_out * sizeof(F128)));
        CK(ffi_malloc(&d_p1, ZT_MAX_BLOCKS * sizeof(F128))); CK(ffi_malloc(&d_pinf, ZT_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_m1d, sizeof(F128))); CK(ffi_malloc(&d_mid, sizeof(F128)));
        const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
        F128 *d_eqlo, *d_eqhi;
        CK(ffi_malloc(&d_eqlo, (1LL << zt_lobits) * sizeof(F128)));
        CK(ffi_malloc(&d_eqhi, (1LL << (zt_dfull - zt_lobits)) * sizeof(F128)));

        ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);
        std::vector<ChF128> rs(6); ch.sample_f128_vec(rs.data(), 6);
        std::vector<ChF128> ro(m - 13); ch.sample_f128_vec(ro.data(), m - 13);
        for (int i = 0; i < 6; i++) zc_r[i] = frch(rs[i]);
        int sm[3] = {0xF7, 0x53, 0xB5};
        for (int i = 0; i < 3; i++) zc_r[6 + i] = PHI_8_TABLE[sm[i]];
        F128 gm[4] = {F128{2, 0}, F128{4, 0}, F128{16, 0}, F128{256, 0}};
        for (int i = 0; i < 4; i++) zc_r[9 + i] = MUL(gm[i], f128_inv_host(ADD(ONE, gm[i])));
        for (int i = 0; i < m - 13; i++) zc_r[13 + i] = frch(ro[i]);

        // Round one needs only eq(r[13..]). The fixed rounds contribute one scale.
        build_eq_device(d_eq, &zc_r[13], m - 13);
        F128 round1_scale = ONE;
        for (int i = 6; i < 13; i++) round1_scale = MUL(round1_scale, ADD(ONE, zc_r[i]));
        launch_zc_round1_cpustyle3((const uint8_t*)d_a, (const uint8_t*)d_b,
                                   (const uint8_t*)df, d_eq, 1LL << (m - 13), nullptr,
                                   round1_scale, d_r1ab, d_r1c);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(zc_r1ab.data(), d_r1ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(zc_r1c.data(), d_r1c, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        { std::vector<ChF128> s(64);
          for (int i = 0; i < 64; i++) s[i] = toch(zc_r1ab[i]); ch.observe_f128_slice(s.data(), 64);
          for (int i = 0; i < 64; i++) s[i] = toch(zc_r1c[i]);  ch.observe_f128_slice(s.data(), 64); }
        zc_z = frch(ch.sample_f128());

        // c-interp at z over Λ (final_c_eval)
        { vector<F128> wl = lagrange_phi(6, zc_z, 64);
          zc_fc = F128{0, 0};
          for (int i = 0; i < 64; i++) zc_fc = ADD(zc_fc, MUL(wl[i], zc_r1c[i])); }

        // Round two folds at z and returns the first two multilinear messages.
        { vector<F128> ws = lagrange_phi(6, zc_z, 0);
          vector<F128> ft(8 * 256, F128{0, 0});
          for (int j = 0; j < 8; j++) for (int v = 0; v < 256; v++) { F128 acc{0, 0};
              for (int bb = 0; bb < 8; bb++) if ((v >> bb) & 1) acc = ADD(acc, ws[8 * j + bb]);
              ft[j * 256 + v] = acc; }
          CK(cudaMemcpy(d_ft, ft.data(), 8 * 256 * sizeof(F128), cudaMemcpyHostToDevice)); }
        build_eq_device(d_eqlo, &zc_r[7], zt_lobits);
        build_eq_device(d_eqhi, &zc_r[7 + zt_lobits], zt_dfull - zt_lobits);
        F128 *cA = d_am, *cB = d_bm, *nA = d_amn, *nB = d_bmn;
        long long L = n_out;
        int n_tail = (m - 6) - 1;
        vector<F128> S(n_tail);
        { vector<F128> values(n_tail), prefixes(n_tail); F128 acc = ONE;
          for (int i = 0; i < n_tail; i++) values[i] = ADD(ONE, zc_r[7 + i]);
          for (int i = 0; i < n_tail; i++) { prefixes[i] = acc; acc = MUL(acc, values[i]); }
          F128 inv = f128_inv_host(acc);
          for (int i = n_tail - 1; i >= 0; i--) { S[i] = MUL(prefixes[i], inv); inv = MUL(inv, values[i]); }
          for (int i = 1; i < n_tail; i++) S[i] = MUL(S[i - 1], S[i]); }

        F128 *d_part8, *d_out8, *d_scales, *d_rhos, *d_m1_log, *d_mi_log;
        ZcSha* d_state;
        CK(ffi_malloc(&d_part8, 8 * ZT_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_out8, 8 * sizeof(F128)));
        CK(ffi_malloc(&d_scales, n_tail * sizeof(F128)));
        CK(ffi_malloc(&d_rhos, n_tail * sizeof(F128)));
        CK(ffi_malloc(&d_m1_log, n_tail * sizeof(F128)));
        CK(ffi_malloc(&d_mi_log, n_tail * sizeof(F128)));
        CK(ffi_malloc(&d_state, sizeof(ZcSha)));

        launch_zc_round2_fold_msg2((const uint8_t*)d_a, (const uint8_t*)d_b, d_ft,
                                   d_eqlo, d_eqhi, zt_lobits, n_out, d_am, d_bm,
                                   S[0], d_part8, d_out8);
        CK(cudaGetLastError());

        auto observe_message = [&](F128 m1, F128 mi) {
            zc_m1s.push_back(m1); zc_mis.push_back(mi);
            ch.observe_f128(toch(m1)); ch.observe_f128(toch(mi));
            mlv_rhos.push_back(frch(ch.sample_f128()));
        };
        F128 h[8];
        CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
        observe_message(h[0], h[1]);
        F128 rho0 = mlv_rhos.back();
        observe_message(interp3(h[2], h[3], h[4], rho0),
                        interp3(h[5], h[6], h[7], rho0));

        int k = 2;
        while (k + 1 <= n_tail && L / 16 > ZT_FINISH_OP) {
            launch_zt_fold_msg2(cA, cB, nA, nB, d_eqlo, d_eqhi, k, zt_lobits, L / 16,
                                mlv_rhos[mlv_rhos.size() - 2], mlv_rhos.back(),
                                S[k - 1], S[k], d_part8, d_out8);
            CK(cudaGetLastError());
            { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L /= 4;
            CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
            observe_message(h[0], h[1]);
            F128 rho = mlv_rhos.back();
            observe_message(interp3(h[2], h[3], h[4], rho),
                            interp3(h[5], h[6], h[7], rho));
            k += 2;
        }

        // One pending challenge aligns the remaining single-round or finisher path.
        { long long half = L / 2;
          launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos[mlv_rhos.size() - 2]);
          CK(cudaGetLastError());
          F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t;
          L = half; }
        int i = k - 1;
        for (; i < n_tail && L / 4 > ZT_FINISH_OP; i++) {
            launch_zt_fold_msg_split(cA, cB, nA, nB, d_eqlo, d_eqhi, i + 1, zt_lobits,
                                     L / 4, mlv_rhos.back(), S[i],
                                     d_p1, d_pinf, d_m1d, d_mid);
            CK(cudaGetLastError());
            { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L /= 2;
            F128 m1, mi;
            CK(cudaMemcpy(&m1, d_m1d, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&mi, d_mid, sizeof(F128), cudaMemcpyDeviceToHost));
            observe_message(m1, mi);
        }
        if (i < n_tail) {
            int rounds = n_tail - i;
            ZcSha state = pack_challenger(ch.hasher);
            CK(cudaMemcpy(d_state, &state, sizeof(ZcSha), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_scales, S.data() + i, rounds * sizeof(F128), cudaMemcpyHostToDevice));
            zt_tail_finisher<<<1, ZT_FIN_TPB>>>(cA, cB, nA, nB, d_eqlo, d_eqhi,
                                                zt_lobits, i + 1, d_scales,
                                                mlv_rhos.back(), rounds, L, d_state,
                                                d_rhos, d_m1_log, d_mi_log);
            CK(cudaGetLastError());
            vector<F128> rhos(rounds), m1s(rounds), mis(rounds);
            CK(cudaMemcpy(&state, d_state, sizeof(ZcSha), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rhos.data(), d_rhos, rounds * sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(m1s.data(), d_m1_log, rounds * sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(mis.data(), d_mi_log, rounds * sizeof(F128), cudaMemcpyDeviceToHost));
            unpack_challenger(ch.hasher, state);
            for (int j = 0; j < rounds; j++) {
                zc_m1s.push_back(m1s[j]); zc_mis.push_back(mis[j]); mlv_rhos.push_back(rhos[j]);
            }
            if (rounds & 1) { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L >>= rounds;
        }

        // final binding + evals
        { long long half = L / 2; launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos.back());
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
        CK(cudaMemcpy(&zc_fa, cA, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&zc_fb, cB, sizeof(F128), cudaMemcpyDeviceToHost));
        ch.observe_f128(toch(zc_fa)); ch.observe_f128(toch(zc_fb));

        ffi_free(d_eq); ffi_free(d_r1ab); ffi_free(d_r1c); ffi_free(d_ft);
        ffi_free(d_am); ffi_free(d_bm); ffi_free(d_amn); ffi_free(d_bmn);
        ffi_free(d_p1); ffi_free(d_pinf); ffi_free(d_m1d); ffi_free(d_mid);
        ffi_free(d_eqlo); ffi_free(d_eqhi);
        ffi_free(d_part8); ffi_free(d_out8); ffi_free(d_scales); ffi_free(d_rhos);
        ffi_free(d_m1_log); ffi_free(d_mi_log); ffi_free(d_state);
    }
    ffi_free(d_a); ffi_free(d_b);
    // zerocheck proof section
    W.f128s(zc_r1ab); W.f128s(zc_r1c);
    W.u64(zc_m1s.size());
    for (size_t i = 0; i < zc_m1s.size(); i++) { W.f128(zc_m1s[i]); W.f128(zc_mis[i]); }
    W.f128(zc_fa); W.f128(zc_fb); W.f128(zc_fc);

    // x_ab (RowMajor): z_skip = zc_z, inner = mlv[..k_log-6], outer = mlv[k_log-6..]
    const int irl = k_log - k_skip;
    vector<F128> xab_inner(mlv_rhos.begin(), mlv_rhos.begin() + irl);
    vector<F128> xab_outer(mlv_rhos.begin() + irl, mlv_rhos.end());

    // ================= lincheck (test_lincheck flow + const-pin beta) =======
    vector<F128> lc_e1s, lc_einfs, lc_zpart(64), lc_rrounds, z_vec((size_t)1 << k_log);
    F128 lc_rskip, lc_w;
    {
        const int K = 1 << k_log;
        const int n_log = m - k_log;
        const long long n_outer = 1LL << n_log, n_stripes = n_outer / 8;
        ch.observe_label((const uint8_t*)"flock-lincheck-v0", 17);
        F128 alpha = frch(ch.sample_f128());
        F128 beta{0, 0}; bool has_pin = P->const_pin_col >= 0;
        if (has_pin) beta = frch(ch.sample_f128());

        static std::once_flag matrix_setup_once;
        static cudaError_t matrix_setup_error = cudaSuccess;
        static uint32_t *d_acp = nullptr, *d_ar = nullptr, *d_bcp = nullptr, *d_br = nullptr;
        static uint32_t setup_a_nnz = 0, setup_b_nnz = 0;
        static int setup_k_log = 0, setup_useful_bits = 0, setup_const_pin = -1;
        static vector<uint32_t> setup_a_col_ptr, setup_a_rows, setup_b_col_ptr, setup_b_rows;
        std::call_once(matrix_setup_once, [&] {
            setup_a_nnz = P->a_nnz; setup_b_nnz = P->b_nnz;
            setup_k_log = P->k_log; setup_useful_bits = P->useful_bits;
            setup_const_pin = P->const_pin_col;
            setup_a_col_ptr.assign(P->a_col_ptr, P->a_col_ptr + K + 1);
            if (P->a_nnz) setup_a_rows.assign(P->a_rows, P->a_rows + P->a_nnz);
            setup_b_col_ptr.assign(P->b_col_ptr, P->b_col_ptr + K + 1);
            if (P->b_nnz) setup_b_rows.assign(P->b_rows, P->b_rows + P->b_nnz);
            matrix_setup_error = cudaMalloc(&d_acp, (K + 1) * sizeof(uint32_t));
            if (matrix_setup_error == cudaSuccess)
                matrix_setup_error = cudaMalloc(&d_ar, (P->a_nnz ? P->a_nnz : 1) * sizeof(uint32_t));
            if (matrix_setup_error == cudaSuccess)
                matrix_setup_error = cudaMalloc(&d_bcp, (K + 1) * sizeof(uint32_t));
            if (matrix_setup_error == cudaSuccess)
                matrix_setup_error = cudaMalloc(&d_br, (P->b_nnz ? P->b_nnz : 1) * sizeof(uint32_t));
            if (matrix_setup_error == cudaSuccess)
                matrix_setup_error = cudaMemcpy(d_acp, P->a_col_ptr, (K + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (matrix_setup_error == cudaSuccess && P->a_nnz)
                matrix_setup_error = cudaMemcpy(d_ar, P->a_rows, P->a_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (matrix_setup_error == cudaSuccess)
                matrix_setup_error = cudaMemcpy(d_bcp, P->b_col_ptr, (K + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (matrix_setup_error == cudaSuccess && P->b_nnz)
                matrix_setup_error = cudaMemcpy(d_br, P->b_rows, P->b_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice);
        });
        CK(matrix_setup_error);
        if (P->a_nnz != setup_a_nnz || P->b_nnz != setup_b_nnz ||
            P->k_log != setup_k_log || P->useful_bits != setup_useful_bits ||
            P->const_pin_col != setup_const_pin ||
            memcmp(P->a_col_ptr, setup_a_col_ptr.data(), (K + 1) * sizeof(uint32_t)) != 0 ||
            (P->a_nnz && memcmp(P->a_rows, setup_a_rows.data(), P->a_nnz * sizeof(uint32_t)) != 0) ||
            memcmp(P->b_col_ptr, setup_b_col_ptr.data(), (K + 1) * sizeof(uint32_t)) != 0 ||
            (P->b_nnz && memcmp(P->b_rows, setup_b_rows.data(), P->b_nnz * sizeof(uint32_t)) != 0)) {
            printf("FFI: BLAKE3 lincheck matrix changed after setup\n");
            return 104;
        }

        F128 *d_eq_inner, *d_comb, *d_zvec, *d_eq_outer, *d_nC, *d_nZ, *d_p1, *d_pinf, *d_e1, *d_einf;
        CK(ffi_malloc(&d_eq_inner, K * sizeof(F128))); CK(ffi_malloc(&d_comb, K * sizeof(F128)));
        CK(ffi_malloc(&d_zvec, K * sizeof(F128))); CK(ffi_malloc(&d_nC, K * sizeof(F128)));
        CK(ffi_malloc(&d_nZ, K * sizeof(F128)));
        CK(ffi_malloc(&d_eq_outer, n_outer * sizeof(F128)));
        CK(ffi_malloc(&d_p1, LC_MAX_BLOCKS * sizeof(F128))); CK(ffi_malloc(&d_pinf, LC_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&d_e1, sizeof(F128))); CK(ffi_malloc(&d_einf, sizeof(F128)));

        build_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip);
        CK(cudaGetLastError());
        launch_lincheck_csc_fold(d_eq_inner, d_acp, d_ar, d_bcp, d_br, alpha, K, d_comb);
        CK(cudaGetLastError());
        if (has_pin) {   // comb_vec[pin] += beta
            F128 v; CK(cudaMemcpy(&v, d_comb + P->const_pin_col, sizeof(F128), cudaMemcpyDeviceToHost));
            v = ADD(v, beta);
            CK(cudaMemcpy(d_comb + P->const_pin_col, &v, sizeof(F128), cudaMemcpyHostToDevice));
        }

        build_eq_device(d_eq_outer, xab_outer.data(), n_log);
        CK(cudaGetLastError());
        launch_lincheck_partial_fold(d_zlin, d_eq_outer, n_stripes, K, P->useful_bits, d_zvec);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(z_vec.data(), d_zvec, K * sizeof(F128), cudaMemcpyDeviceToHost));

        F128 *cC = d_comb, *cZ = d_zvec, *nC = d_nC, *nZ = d_nZ;
        long long L = K;
        for (int rnd = 0; rnd < irl; rnd++) {
            long long half = L / 2;
            launch_lincheck_msg(cC, cZ, half, d_p1, d_pinf, d_e1, d_einf);
            CK(cudaGetLastError());
            F128 e1, einf;
            CK(cudaMemcpy(&e1, d_e1, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&einf, d_einf, sizeof(F128), cudaMemcpyDeviceToHost));
            lc_e1s.push_back(e1); lc_einfs.push_back(einf);
            ch.observe_f128(toch(e1)); ch.observe_f128(toch(einf));
            F128 r = frch(ch.sample_f128()); lc_rrounds.push_back(r);
            launch_lincheck_fold2(cC, cZ, nC, nZ, half, r);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            F128* t; t = cC; cC = nC; nC = t; t = cZ; cZ = nZ; nZ = t;
            L = half;
        }
        CK(cudaMemcpy(lc_zpart.data(), cZ, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
        { std::vector<ChF128> s(64); for (int i = 0; i < 64; i++) s[i] = toch(lc_zpart[i]);
          ch.observe_f128_slice(s.data(), 64); }
        lc_rskip = frch(ch.sample_f128());
        { vector<F128> lam = lagrange_weights_host(k_skip, lc_rskip);
          lc_w = F128{0, 0};
          for (int i = 0; i < 64; i++) lc_w = ADD(lc_w, MUL(lam[i], lc_zpart[i])); }

        ffi_free(d_eq_inner); ffi_free(d_comb); ffi_free(d_zvec); ffi_free(d_nC); ffi_free(d_nZ);
        ffi_free(d_eq_outer);
        ffi_free(d_p1); ffi_free(d_pinf); ffi_free(d_e1); ffi_free(d_einf);
    }
    ffi_free(d_zlin);
    // lincheck proof section
    W.u64(lc_e1s.size());
    for (size_t i = 0; i < lc_e1s.size(); i++) { W.f128(lc_e1s[i]); W.f128(lc_einfs[i]); }
    W.f128s(lc_zpart);

    // r_inner_rest = reverse(rounds)
    vector<F128> r_inner_rest(lc_rrounds.rbegin(), lc_rrounds.rend());

    // ================= ring-switch batch ====================================
    // x_full per claim: ab = r_inner_rest ++ xab_outer; c = zc_r[6..].
    vector<F128> xfull_ab; xfull_ab.reserve(m - 6);
    xfull_ab.insert(xfull_ab.end(), r_inner_rest.begin(), r_inner_rest.end());
    xfull_ab.insert(xfull_ab.end(), xab_outer.begin(), xab_outer.end());
    vector<F128> xfull_c(zc_r.begin() + 6, zc_r.end());

    // s_hat_v_ab from z_vec: s[b] = Σ_k eq(tail)[k]·z_vec[k*128+b], tail = r_inner_rest[1..]
    vector<F128> shat_ab(128, F128{0, 0});
    { vector<F128> eq_tail = build_eq_host(r_inner_rest.data() + 1, irl - 1);
      for (size_t k = 0; k < eq_tail.size(); k++)
          for (int b = 0; b < 128; b++)
              shat_ab[b] = ADD(shat_ab[b], MUL(eq_tail[k], z_vec[k * 128 + b])); }
    F128 *d_suffix_ab, *d_suffix_c, *d_shat_c, *d_shat_partials;
    const int ring_switch_chunks = 64;
    CK(ffi_malloc(&d_suffix_ab, len * sizeof(F128)));
    CK(ffi_malloc(&d_suffix_c, len * sizeof(F128)));
    CK(ffi_malloc(&d_shat_c, 128 * sizeof(F128)));
    CK(ffi_malloc(&d_shat_partials, 128 * ring_switch_chunks * sizeof(F128)));
    build_eq_device(d_suffix_ab, xfull_ab.data() + 1, (int)xfull_ab.size() - 1);
    build_eq_device(d_suffix_c, xfull_c.data() + 1, (int)xfull_c.size() - 1);
    ring_switch_fold_rows_grouped<<<dim3(ring_switch_chunks, 16), 256>>>(
        df, d_suffix_c, len, ring_switch_chunks, d_shat_partials);
    ring_switch_fold_rows_reduce<<<128, 256>>>(d_shat_partials, ring_switch_chunks, d_shat_c);
    CK(cudaGetLastError());
    vector<F128> shat_c(128);
    CK(cudaMemcpy(shat_c.data(), d_shat_c, 128 * sizeof(F128), cudaMemcpyDeviceToHost));
    CK(ffi_free(d_shat_c));
    CK(ffi_free(d_shat_partials));

    ch.observe_label((const uint8_t*)"flock-pcs-open-batch-v0", 23);
    struct RsWork { vector<F128> shat, eq_rd; F128 claim; };
    RsWork rsw[2];
    const vector<F128>* shats[2] = { &shat_ab, &shat_c };
    for (int i = 0; i < 2; i++) {
        ch.observe_label((const uint8_t*)"flock-ring-switch-v0", 20);
        { std::vector<ChF128> s(128); for (int j = 0; j < 128; j++) s[j] = toch((*shats[i])[j]);
          ch.observe_f128_slice(s.data(), 128); }
        std::vector<ChF128> rd(7); ch.sample_f128_vec(rd.data(), 7);
        vector<F128> rdf(7); for (int j = 0; j < 7; j++) rdf[j] = frch(rd[j]);
        rsw[i].shat = *shats[i];
        rsw[i].eq_rd = build_eq_host(rdf.data(), 7);
        vector<F128> shat_u = ta_transpose_host(*shats[i]);
        F128 c{0, 0};
        for (int j = 0; j < 128; j++) c = ADD(c, MUL(shat_u[j], rsw[i].eq_rd[j]));
        rsw[i].claim = c;
    }
    F128 gam[2]; gam[0] = frch(ch.sample_f128()); gam[1] = frch(ch.sample_f128());
    F128 target = ADD(MUL(gam[0], rsw[0].claim), MUL(gam[1], rsw[1].claim));

    // Build the combined basis on the device. Keep it resident for sumcheck.
    vector<F128> weights_ab(128), weights_c(128);
    for (int j = 0; j < 128; j++) {
        weights_ab[j] = MUL(gam[0], rsw[0].eq_rd[j]);
        weights_c[j] = MUL(gam[1], rsw[1].eq_rd[j]);
    }
    F128 *d_weights_ab, *d_weights_c, *dcb;
    CK(ffi_malloc(&d_weights_ab, 128 * sizeof(F128)));
    CK(ffi_malloc(&d_weights_c, 128 * sizeof(F128)));
    CK(ffi_malloc(&dcb, len * sizeof(F128)));
    CK(cudaMemcpy(d_weights_ab, weights_ab.data(), 128 * sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_weights_c, weights_c.data(), 128 * sizeof(F128), cudaMemcpyHostToDevice));
    ring_switch_combine_basis<<<(unsigned)((len + 255) / 256), 256>>>(
        d_suffix_ab, d_suffix_c, d_weights_ab, d_weights_c, len, dcb);
    CK(cudaGetLastError());
    CK(ffi_free(d_suffix_ab)); CK(ffi_free(d_suffix_c));
    CK(ffi_free(d_weights_ab)); CK(ffi_free(d_weights_c));

    F128 *p0, *p2, *du0, *du2;
    CK(ffi_malloc(&p0, SMC_MAX_BLOCKS * sizeof(F128)));
    CK(ffi_malloc(&p2, SMC_MAX_BLOCKS * sizeof(F128)));
    CK(ffi_malloc(&du0, sizeof(F128))); CK(ffi_malloc(&du2, sizeof(F128)));
    launch_sumcheck_msg(df, dcb, len / 2, p0, p2, du0, du2);
    CK(cudaGetLastError());
    F128 r0u0, r0u2;
    CK(cudaMemcpy(&r0u0, du0, sizeof(F128), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&r0u2, du2, sizeof(F128), cudaMemcpyDeviceToHost));
    // ring-switch proof section
    W.f128s(shat_ab); W.f128s(shat_c);

    // ================= ligerito recursion (test_ligerito_l0 flow) ===========
    vector<F128> sc_transcript;          // (u0,u2) pairs in transcript order
    vector<F128> ood_values;
    vector<uint64_t> grind_nonces, fold_grind_nonces;
    vector<MHash> rec_roots;
    struct LevelOpen { vector<F128> rows_flat; size_t n_rows, row_len; vector<MHash> proof; };
    vector<LevelOpen> level_opens;       // [0] = initial (L0), then per recursive level
    vector<F128> yr_out;

    {
        ch.observe_label((const uint8_t*)"flock-ligerito-basis-v0", 23);
        ch.observe_f128(toch(target));
        ch.observe_bytes(l0root, 32);
        ch.observe_f128(toch(r0u0)); ch.observe_f128(toch(r0u2));
        sc_transcript.push_back(r0u0); sc_transcript.push_back(r0u2);

        // resident sumcheck state (f, b)
        F128 *dfp, *df2, *dcb2;
        CK(ffi_malloc(&dfp, len * sizeof(F128)));
        CK(ffi_malloc(&df2, len * sizeof(F128))); CK(ffi_malloc(&dcb2, len * sizeof(F128)));
        CK(cudaMemcpy(dfp, df, len * sizeof(F128), cudaMemcpyDeviceToDevice));
        F128 *cf = dfp, *ccb = dcb, *nf = df2, *ncb = dcb2;
        long long slen = len;
        F128 u0, u2;

        vector<F128> r_lane;
        for (int j = 0; j < P->initial_k; j++) {
            int bits = P->fold_grinding_bits[0] - j; if (bits < 0) bits = 0;
            if (bits > 0) {
                uint64_t nonce;
                CK(grind_pow_device(ch, (uint32_t)bits, nonce));
                fold_grind_nonces.push_back(nonce);
            }
            F128 r = frch(ch.sample_f128());
            long long half = slen / 2;
            launch_sumcheck_fold_msg(cf, ccb, nf, ncb, half, r, p0, p2, du0, du2);
            CK(cudaGetLastError());
            { F128* t; t = cf; cf = nf; nf = t; t = ccb; ccb = ncb; ncb = t; }
            slen = half;
            CK(cudaMemcpy(&u0, du0, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&u2, du2, sizeof(F128), cudaMemcpyDeviceToHost));
            ch.observe_f128(toch(u0)); ch.observe_f128(toch(u2));
            sc_transcript.push_back(u0); sc_transcript.push_back(u2);
            r_lane.push_back(r);
        }

        // scratch for OOD / intro
        F128 *d_bnew, *ep0, *ep2, *epodd, *eu0, *eu2, *ehnew;
        long long n1_len = slen;
        CK(ffi_malloc(&d_bnew, n1_len * sizeof(F128)));
        CK(ffi_malloc(&ep0, IGL_MAX_BLOCKS * sizeof(F128))); CK(ffi_malloc(&ep2, IGL_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&epodd, IGL_MAX_BLOCKS * sizeof(F128)));
        CK(ffi_malloc(&eu0, sizeof(F128))); CK(ffi_malloc(&eu2, sizeof(F128))); CK(ffi_malloc(&ehnew, sizeof(F128)));

        F128* d_prev_cw = d_cw0;
        uint8_t* d_prev_tree = d_tree0;
        long long prev_bl = l0_bl; int prev_ni = l0_ni;

        // per-level loop: level 0 handles commit-L1 + OOD + L0 queries; then r-1
        // more middle levels; the last level ships yr.
        int r_steps = P->recursive_steps;
        for (int lvl = 0; ; lvl++) {
            if (lvl > 0) {
                int k_rec = P->recursive_ks[lvl - 1];
                vector<F128> level_rs;
                for (int j = 0; j < k_rec; j++) {
                    int bits = P->fold_grinding_bits[lvl] - j; if (bits < 0) bits = 0;
                    if (bits > 0) {
                        uint64_t nonce;
                        CK(grind_pow_device(ch, (uint32_t)bits, nonce));
                        fold_grind_nonces.push_back(nonce);
                    }
                    F128 r = frch(ch.sample_f128());
                    long long half = slen / 2;
                    launch_sumcheck_fold_msg(cf, ccb, nf, ncb, half, r, p0, p2, du0, du2);
                    CK(cudaGetLastError());
                    { F128* t; t = cf; cf = nf; nf = t; t = ccb; ccb = ncb; ncb = t; }
                    slen = half;
                    CK(cudaMemcpy(&u0, du0, sizeof(F128), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&u2, du2, sizeof(F128), cudaMemcpyDeviceToHost));
                    ch.observe_f128(toch(u0)); ch.observe_f128(toch(u2));
                    sc_transcript.push_back(u0); sc_transcript.push_back(u2);
                    level_rs.push_back(r);
                }
                r_lane = std::move(level_rs);
            }

            if (lvl == r_steps) {
                // final level: yr in clear, grind, queries, open prev
                yr_out.resize(slen);
                CK(cudaMemcpy(yr_out.data(), cf, (size_t)slen * sizeof(F128), cudaMemcpyDeviceToHost));
                for (long long i = 0; i < slen; i++) ch.observe_f128(toch(yr_out[i]));
                uint64_t nonce;
                CK(grind_pow_device(ch, (uint32_t)P->grinding_bits[lvl], nonce));
                grind_nonces.push_back(nonce);
                std::vector<size_t> q = ch.sample_distinct_queries((size_t)prev_bl, P->queries[lvl]);
                LevelOpen lo; lo.n_rows = q.size(); lo.row_len = prev_ni;
                if (copy_codeword_rows(d_prev_cw, q, prev_ni, lo.rows_flat) != 0) return 100;
                lo.proof = merkle_multi_proof_device(d_prev_tree, (size_t)prev_bl, q);
                level_opens.push_back(std::move(lo));
                break;
            }

            // middle level: commit next, OOD, grind, queries, open prev, induce, intro
            int n_next = 0; { long long s = slen; while (s > 1) { s >>= 1; n_next++; } }
            long long nn_len = 1LL << n_next;
            int k_comm = P->recursive_ks[lvl];   // lanes of the next level's commit
            int rate_next = P->log_inv_rates[lvl + 1];
            F128* d_cwn; uint8_t* d_treen; long long bln; int lanesn; uint8_t rn[32];
            {
                int k_code = (n_next - k_comm) + rate_next;
                int num_ntts = 1 << k_comm;
                bln = 1LL << k_code; lanesn = num_ntts;
                long long cw_len = bln * num_ntts;
                TwiddleTable tt = build_twiddle_table(k_code);
                F128* d_tw;
                CK(ffi_malloc(&d_cwn, cw_len * sizeof(F128)));
                CK(ffi_malloc(&d_tw, tt.data.size() * sizeof(F128)));
                CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));
                CK(ffi_malloc(&d_treen, (size_t)(2 * bln - 1) * 32));
                // Always the replicate path here: the recursive codewords are
                // tiny, and the fused src read diverges from the Rust commit at
                // (k_code - rate) == 8 (first hit at m=33's L4; L0 at every
                // tested m is fine, so L0 keeps the fusion for the big buffer).
                {
                    int tpb = 256;
                    replicate_fill_ffi<<<(unsigned)((cw_len + tpb - 1) / tpb), tpb>>>(cf, d_cwn, cw_len, nn_len);
                    launch_ntt(d_cwn, d_tw, tt, rate_next, k_code, num_ntts);
                }
                CK(cudaGetLastError());
                launch_merkle((const uint8_t*)d_cwn, d_treen, bln, num_ntts * 16);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(rn, d_treen + (size_t)(2 * bln - 2) * 32, 32, cudaMemcpyDeviceToHost));
                ffi_free(d_tw);
            }
            ch.observe_bytes(rn, 32);
            { MHash h; memcpy(h.b, rn, 32); rec_roots.push_back(h); }

            for (int o = 0; o < P->ood_samples[lvl + 1]; o++) {
                std::vector<ChF128> zc_(n_next); ch.sample_f128_vec(zc_.data(), n_next);
                vector<F128> zf(n_next);
                for (int i = 0; i < n_next; i++) zf[i] = frch(zc_[i]);
                build_eq_device(d_bnew, zf.data(), n_next);
                launch_msg_eval(cf, d_bnew, nn_len / 2, ep0, ep2, epodd, eu0, eu2, ehnew);
                CK(cudaGetLastError());
                F128 iu0, iu2, y;
                CK(cudaMemcpy(&iu0, eu0, sizeof(F128), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&iu2, eu2, sizeof(F128), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&y, ehnew, sizeof(F128), cudaMemcpyDeviceToHost));
                ch.observe_f128(toch(y));
                ood_values.push_back(y);
                ch.observe_f128(toch(iu0)); ch.observe_f128(toch(iu2));
                sc_transcript.push_back(iu0); sc_transcript.push_back(iu2);
                ChF128 bc = ch.sample_f128();
                launch_glue(ccb, d_bnew, frch(bc), nn_len);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            }

            uint64_t nonce;
            CK(grind_pow_device(ch, (uint32_t)P->grinding_bits[lvl], nonce));
            grind_nonces.push_back(nonce);
            std::vector<size_t> q = ch.sample_distinct_queries((size_t)prev_bl, P->queries[lvl]);
            int al = 0; { int mq = P->queries[lvl] - 1; while (mq) { al++; mq >>= 1; } }
            if (P->queries[lvl] <= 1) al = 0;
            std::vector<ChF128> alpha(al); ch.sample_f128_vec(alpha.data(), al);
            vector<F128> alpha_f(al);
            for (int i = 0; i < al; i++) alpha_f[i] = frch(alpha[i]);

            LevelOpen lo; lo.n_rows = q.size(); lo.row_len = prev_ni;
            if (copy_codeword_rows(d_prev_cw, q, prev_ni, lo.rows_flat) != 0) return 100;
            lo.proof = merkle_multi_proof_device(d_prev_tree, (size_t)prev_bl, q);

            std::vector<unsigned long long> qull(q.size());
            for (size_t i = 0; i < q.size(); i++) qull[i] = q[i];
            vector<F128> sks = eval_sk_at_vks_hd(n_next);
            InduceSetupDev S = induce_setup_device(n_next, sks, r_lane, alpha_f, qull, lo.rows_flat, prev_ni);
            level_opens.push_back(std::move(lo));
            F128* d_basis;
            CK(ffi_malloc(&d_basis, (size_t)nn_len * sizeof(F128)));
            launch_induce_accumulate(S.d_sh, S.d_low, S.n_queries, S.low_n, S.high_n, d_basis, nn_len);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

            // introduce + glue
            { long long quads = nn_len / 2;
              int pblocks = sumcheck_blocks(quads);
              sumcheck_msg_partial<<<pblocks, SMC_TPB>>>(cf, d_basis, quads, p0, p2);
              sumcheck_msg_combine<<<1, SMC_TPB>>>(p0, p2, pblocks, du0, du2);
              CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
              CK(cudaMemcpy(&u0, du0, sizeof(F128), cudaMemcpyDeviceToHost));
              CK(cudaMemcpy(&u2, du2, sizeof(F128), cudaMemcpyDeviceToHost)); }
            ch.observe_f128(toch(u0)); ch.observe_f128(toch(u2));
            sc_transcript.push_back(u0); sc_transcript.push_back(u2);
            ChF128 bi = ch.sample_f128();
            launch_glue(ccb, d_basis, frch(bi), nn_len);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            ffi_free(S.d_low); ffi_free(S.d_sh); ffi_free(d_basis);

            ffi_free(d_prev_cw); ffi_free(d_prev_tree);
            d_prev_cw = d_cwn; d_prev_tree = d_treen;
            prev_bl = bln; prev_ni = lanesn;
        }

        ffi_free(d_prev_cw); ffi_free(d_prev_tree);

        ffi_free(dfp); ffi_free(dcb); ffi_free(df2); ffi_free(dcb2);
        ffi_free(p0); ffi_free(p2); ffi_free(du0); ffi_free(du2);
        ffi_free(d_bnew); ffi_free(ep0); ffi_free(ep2); ffi_free(epodd);
        ffi_free(eu0); ffi_free(eu2); ffi_free(ehnew);
    }
    ffi_free(df);

    // ================= ligerito proof section ===============================
    W.u64(rec_roots.size()); for (auto& h : rec_roots) W.hash(h.b);
    W.u64(level_opens.size());
    for (auto& lo : level_opens) { W.rows(lo.rows_flat, lo.n_rows, lo.row_len); W.hashes(lo.proof); }
    W.f128s(yr_out);
    W.u64(sc_transcript.size() / 2);
    for (size_t i = 0; i < sc_transcript.size(); i += 2) { W.f128(sc_transcript[i]); W.f128(sc_transcript[i + 1]); }
    W.f128s(ood_values);
    W.u64(grind_nonces.size()); for (auto n : grind_nonces) W.u64(n);
    W.u64(fold_grind_nonces.size()); for (auto n : fold_grind_nonces) W.u64(n);

    *out_len = W.buf.size();
    *out = (uint8_t*)malloc(W.buf.size());
    memcpy(*out, W.buf.data(), W.buf.size());
    return 0;
}

} // extern "C"
