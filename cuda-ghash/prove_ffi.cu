// C-ABI GPU prover for the full R1CS Ligerito proof, linked into Rust by
// crates/flock-cuda-ffi. Reproduces crates/flock-prover/src/prover.rs::
// prove_ligerito byte-for-byte on the transcript:
//   commit -> bind_statement -> zerocheck (+s_hat_v_c) -> lincheck (+z_vec)
//   -> ring-switch batch -> ligerito recursion, capturing every proof field.
//
// Orchestration is lifted from the byte-validated vector tests
// (test_zerocheck_full.cu, test_lincheck.cu, test_ligerito_l0.cu); the bench
// (bench_ligerito.cu) stays a timing tool and is not touched. Protocol
// constants the Rust side owns (statement digest, zerocheck tables, ligerito
// config) are passed IN so both sides share one source of truth.
//
// Output: flat little-endian byte stream (see FfiWriter) that the Rust test
// parses back into the typed proof structs; layout must match
// crates/flock-cuda-ffi/tests/gpu_roundtrip.rs::parse_proof.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ntt_f128.cuh"
#include "merkle.cuh"
#include "merkle_open.hpp"
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

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("FFI CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 100; } } while(0)

namespace {

using std::vector;

F128 ADD(F128 a, F128 b) { return f128_add_hd(a, b); }
F128 MUL(F128 a, F128 b) { return f128_mul_hd(a, b); }
const F128 ONE{1, 0};
ChF128 toch(F128 x) { return ChF128{x.lo, x.hi}; }
F128 frch(ChF128 x) { return F128{x.lo, x.hi}; }

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

// s_hat_v[b] = Σ_t bit_b(z[t]) · suffix[t]  (ring_switch::fold_1b_rows).
vector<F128> fold_1b_rows_host(const vector<F128>& z, const vector<F128>& suffix) {
    vector<F128> s(128, F128{0, 0});
    for (size_t t = 0; t < z.size(); t++) {
        F128 w = suffix[t];
        for (int half = 0; half < 2; half++) {
            u64 bits = half ? z[t].hi : z[t].lo;
            while (bits) {
                int b = __builtin_ctzll(bits);
                s[64 * half + b] = ADD(s[64 * half + b], w);
                bits &= bits - 1;
            }
        }
    }
    return s;
}

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

// phi(e) = Σ_i bit_i(e)·w[i] — the F2-linear ring-switch fold of one slot.
F128 fold_one_slot_host(F128 e, const vector<F128>& w) {
    F128 acc{0, 0};
    for (int half = 0; half < 2; half++) {
        u64 bits = half ? e.hi : e.lo;
        while (bits) {
            acc = ADD(acc, w[64 * half + __builtin_ctzll(bits)]);
            bits &= bits - 1;
        }
    }
    return acc;
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
    const int m = P->m, k_log = P->k_log, k_skip = 6;
    const int log_n = m - 7;                    // packed-witness log length
    const long long len = 1LL << log_n;         // packed F128 elements
    const int n_blocks_log = m - 14;
    const long long n_total = 1LL << n_blocks_log;
    const int n_blocks = (int)n_total;
    if (n_blocks_log < 3) { printf("FFI: m too small\n"); return 101; }

    // ================= witness (real BLAKE3 compressions, deterministic) ====
    F128 *df, *d_a, *d_b; uint8_t* d_zlin;
    CK(cudaMalloc(&df, len * sizeof(F128)));
    CK(cudaMalloc(&d_a, len * sizeof(F128)));
    CK(cudaMalloc(&d_b, len * sizeof(F128)));
    CK(cudaMalloc(&d_zlin, (size_t)len * 16));
    {
        uint32_t *d_cv, *d_m, *d_blen, *d_flags; b3u64* d_ctr;
        CK(cudaMalloc(&d_cv, (size_t)n_blocks * 8 * 4)); CK(cudaMalloc(&d_m, (size_t)n_blocks * 16 * 4));
        CK(cudaMalloc(&d_blen, (size_t)n_blocks * 4)); CK(cudaMalloc(&d_flags, (size_t)n_blocks * 4));
        CK(cudaMalloc(&d_ctr, (size_t)n_blocks * 8));
        ffi_fill_compressions<<<(unsigned)((n_blocks + 127) / 128), 128>>>(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks);
        CK(cudaGetLastError());
        launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks, n_total,
                                     (b3u64*)df, (b3u64*)d_a, (b3u64*)d_b);
        launch_blake3_lincheck_transpose((const b3u64*)df, n_total, d_zlin);
        CK(cudaDeviceSynchronize());
        cudaFree(d_cv); cudaFree(d_m); cudaFree(d_blen); cudaFree(d_flags); cudaFree(d_ctr);
    }
    vector<F128> z_host(len);   // host witness copy for ring-switch folds
    CK(cudaMemcpy(z_host.data(), df, len * sizeof(F128), cudaMemcpyDeviceToHost));

    // ================= L0 commit ============================================
    F128* d_cw0; uint8_t* d_tree0; long long l0_bl; int l0_ni; uint8_t l0root[32];
    {
        int k_code = (log_n - P->initial_k) + P->log_inv_rates[0];
        int num_ntts = 1 << P->initial_k;
        l0_bl = 1LL << k_code; l0_ni = num_ntts;
        long long cw_len = l0_bl * num_ntts;
        TwiddleTable tt = build_twiddle_table(k_code);
        F128* d_tw;
        CK(cudaMalloc(&d_cw0, cw_len * sizeof(F128)));
        CK(cudaMalloc(&d_tw, tt.data.size() * sizeof(F128)));
        CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_tree0, (size_t)(2 * l0_bl - 1) * 32));
        if (ntt_can_fuse_src(k_code - P->log_inv_rates[0])) {
            launch_ntt(d_cw0, d_tw, tt, P->log_inv_rates[0], k_code, num_ntts, 256, false, df, len - 1);
        } else {
            printf("FFI: unfused L0 rate-extend not wired\n"); return 102;
        }
        CK(cudaGetLastError());
        launch_merkle((const uint8_t*)d_cw0, d_tree0, l0_bl, num_ntts * 16);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(l0root, d_tree0 + (size_t)(2 * l0_bl - 2) * 32, 32, cudaMemcpyDeviceToHost));
        cudaFree(d_tw);
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
        zc_round1_upload_tables(P->zc_mcol, P->zc_f8mul, PHI_8_TABLE);
        F128 *d_eq, *d_r1ab, *d_r1c, *d_ft, *d_am, *d_bm, *d_amn, *d_bmn, *d_p1, *d_pinf, *d_m1d, *d_mid;
        CK(cudaMalloc(&d_eq, n_out * sizeof(F128)));
        CK(cudaMalloc(&d_r1ab, 64 * sizeof(F128))); CK(cudaMalloc(&d_r1c, 64 * sizeof(F128)));
        CK(cudaMalloc(&d_ft, 8 * 256 * sizeof(F128)));
        CK(cudaMalloc(&d_am, n_out * sizeof(F128))); CK(cudaMalloc(&d_bm, n_out * sizeof(F128)));
        CK(cudaMalloc(&d_amn, n_out * sizeof(F128))); CK(cudaMalloc(&d_bmn, n_out * sizeof(F128)));
        CK(cudaMalloc(&d_p1, ZT_MAX_BLOCKS * sizeof(F128))); CK(cudaMalloc(&d_pinf, ZT_MAX_BLOCKS * sizeof(F128)));
        CK(cudaMalloc(&d_m1d, sizeof(F128))); CK(cudaMalloc(&d_mid, sizeof(F128)));
        const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
        F128 *d_eqlo, *d_eqhi;
        CK(cudaMalloc(&d_eqlo, (1LL << zt_lobits) * sizeof(F128)));
        CK(cudaMalloc(&d_eqhi, (1LL << (zt_dfull - zt_lobits)) * sizeof(F128)));

        ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);
        std::vector<ChF128> rs(6); ch.sample_f128_vec(rs.data(), 6);
        std::vector<ChF128> ro(m - 13); ch.sample_f128_vec(ro.data(), m - 13);
        for (int i = 0; i < 6; i++) zc_r[i] = frch(rs[i]);
        int sm[3] = {0xF7, 0x53, 0xB5};
        for (int i = 0; i < 3; i++) zc_r[6 + i] = PHI_8_TABLE[sm[i]];
        F128 gm[4] = {F128{2, 0}, F128{4, 0}, F128{16, 0}, F128{256, 0}};
        for (int i = 0; i < 4; i++) zc_r[9 + i] = MUL(gm[i], f128_inv_host(ADD(ONE, gm[i])));
        for (int i = 0; i < m - 13; i++) zc_r[13 + i] = frch(ro[i]);

        // round 1 over the full eq(r[6..]) table
        vector<F128> eqf6 = build_eq_host(&zc_r[6], m - 6);
        CK(cudaMemcpy(d_eq, eqf6.data(), n_out * sizeof(F128), cudaMemcpyHostToDevice));
        launch_zc_round1_fast((const uint8_t*)d_a, (const uint8_t*)d_b, (const uint8_t*)df,
                              d_eq, n_out, d_r1ab, d_r1c);
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

        // round 2: fold-at-z + first message
        { vector<F128> ws = lagrange_phi(6, zc_z, 0);
          vector<F128> ft(8 * 256, F128{0, 0});
          for (int j = 0; j < 8; j++) for (int v = 0; v < 256; v++) { F128 acc{0, 0};
              for (int bb = 0; bb < 8; bb++) if ((v >> bb) & 1) acc = ADD(acc, ws[8 * j + bb]);
              ft[j * 256 + v] = acc; }
          CK(cudaMemcpy(d_ft, ft.data(), 8 * 256 * sizeof(F128), cudaMemcpyHostToDevice)); }
        launch_zc_round2_fold((const uint8_t*)d_a, (const uint8_t*)d_b, d_ft, n_out, d_am, d_bm);
        CK(cudaGetLastError());

        { vector<F128> eqlo = build_eq_host(&zc_r[7], zt_lobits);
          vector<F128> eqhi = build_eq_host(&zc_r[7 + zt_lobits], zt_dfull - zt_lobits);
          CK(cudaMemcpy(d_eqlo, eqlo.data(), eqlo.size() * sizeof(F128), cudaMemcpyHostToDevice));
          CK(cudaMemcpy(d_eqhi, eqhi.data(), eqhi.size() * sizeof(F128), cudaMemcpyHostToDevice)); }
        F128 *cA = d_am, *cB = d_bm, *nA = d_amn, *nB = d_bmn;
        long long L = n_out;
        F128 m1, mi;
        { launch_zt_msg_split(cA, cB, d_eqlo, d_eqhi, 0, zt_lobits, L / 2, ONE, d_p1, d_pinf, d_m1d, d_mid);
          CK(cudaMemcpy(&m1, d_m1d, sizeof(F128), cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(&mi, d_mid, sizeof(F128), cudaMemcpyDeviceToHost)); }
        zc_m1s.push_back(m1); zc_mis.push_back(mi);
        ch.observe_f128(toch(m1)); ch.observe_f128(toch(mi));
        F128 rho = frch(ch.sample_f128()); mlv_rhos.push_back(rho);

        // tail rounds (host-challenger form; correctness path)
        int n_tail = (m - 6) - 1;
        vector<F128> S(n_tail);
        { F128 acc = ONE;
          for (int i = 0; i < n_tail; i++) { acc = MUL(acc, f128_inv_host(ADD(ONE, zc_r[7 + i]))); S[i] = acc; } }
        for (int i = 0; i < n_tail; i++) {
            launch_zt_fold_msg_split(cA, cB, nA, nB, d_eqlo, d_eqhi, i + 1, zt_lobits, L / 4,
                                     mlv_rhos.back(), S[i], d_p1, d_pinf, d_m1d, d_mid);
            CK(cudaGetLastError());
            { F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
            L /= 2;
            CK(cudaMemcpy(&m1, d_m1d, sizeof(F128), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&mi, d_mid, sizeof(F128), cudaMemcpyDeviceToHost));
            zc_m1s.push_back(m1); zc_mis.push_back(mi);
            ch.observe_f128(toch(m1)); ch.observe_f128(toch(mi));
            mlv_rhos.push_back(frch(ch.sample_f128()));
        }
        // final binding + evals
        { long long half = L / 2; launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos.back());
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          F128* t; t = cA; cA = nA; nA = t; t = cB; cB = nB; nB = t; }
        CK(cudaMemcpy(&zc_fa, cA, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&zc_fb, cB, sizeof(F128), cudaMemcpyDeviceToHost));
        ch.observe_f128(toch(zc_fa)); ch.observe_f128(toch(zc_fb));

        cudaFree(d_eq); cudaFree(d_r1ab); cudaFree(d_r1c); cudaFree(d_ft);
        cudaFree(d_am); cudaFree(d_bm); cudaFree(d_amn); cudaFree(d_bmn);
        cudaFree(d_p1); cudaFree(d_pinf); cudaFree(d_m1d); cudaFree(d_mid);
        cudaFree(d_eqlo); cudaFree(d_eqhi);
    }
    cudaFree(d_a); cudaFree(d_b);
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

        F128 *d_eq_inner, *d_comb, *d_zvec, *d_eq_outer, *d_nC, *d_nZ, *d_p1, *d_pinf, *d_e1, *d_einf;
        uint32_t *d_acp, *d_ar, *d_bcp, *d_br;
        CK(cudaMalloc(&d_eq_inner, K * sizeof(F128))); CK(cudaMalloc(&d_comb, K * sizeof(F128)));
        CK(cudaMalloc(&d_zvec, K * sizeof(F128))); CK(cudaMalloc(&d_nC, K * sizeof(F128)));
        CK(cudaMalloc(&d_nZ, K * sizeof(F128)));
        CK(cudaMalloc(&d_eq_outer, n_outer * sizeof(F128)));
        CK(cudaMalloc(&d_acp, (K + 1) * sizeof(uint32_t)));
        CK(cudaMalloc(&d_ar, (P->a_nnz ? P->a_nnz : 1) * sizeof(uint32_t)));
        CK(cudaMalloc(&d_bcp, (K + 1) * sizeof(uint32_t)));
        CK(cudaMalloc(&d_br, (P->b_nnz ? P->b_nnz : 1) * sizeof(uint32_t)));
        CK(cudaMalloc(&d_p1, LC_MAX_BLOCKS * sizeof(F128))); CK(cudaMalloc(&d_pinf, LC_MAX_BLOCKS * sizeof(F128)));
        CK(cudaMalloc(&d_e1, sizeof(F128))); CK(cudaMalloc(&d_einf, sizeof(F128)));

        vector<F128> eq_inner = build_quirky_eq_table_host(zc_z, xab_inner, k_skip);
        CK(cudaMemcpy(d_eq_inner, eq_inner.data(), K * sizeof(F128), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_acp, P->a_col_ptr, (K + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (P->a_nnz) CK(cudaMemcpy(d_ar, P->a_rows, P->a_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bcp, P->b_col_ptr, (K + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (P->b_nnz) CK(cudaMemcpy(d_br, P->b_rows, P->b_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));
        launch_lincheck_csc_fold(d_eq_inner, d_acp, d_ar, d_bcp, d_br, alpha, K, d_comb);
        CK(cudaGetLastError());
        if (has_pin) {   // comb_vec[pin] += beta
            F128 v; CK(cudaMemcpy(&v, d_comb + P->const_pin_col, sizeof(F128), cudaMemcpyDeviceToHost));
            v = ADD(v, beta);
            CK(cudaMemcpy(d_comb + P->const_pin_col, &v, sizeof(F128), cudaMemcpyHostToDevice));
        }

        { vector<F128> eq_outer = build_eq_host(xab_outer.data(), n_log);
          CK(cudaMemcpy(d_eq_outer, eq_outer.data(), n_outer * sizeof(F128), cudaMemcpyHostToDevice)); }
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

        cudaFree(d_eq_inner); cudaFree(d_comb); cudaFree(d_zvec); cudaFree(d_nC); cudaFree(d_nZ);
        cudaFree(d_eq_outer); cudaFree(d_acp); cudaFree(d_ar); cudaFree(d_bcp); cudaFree(d_br);
        cudaFree(d_p1); cudaFree(d_pinf); cudaFree(d_e1); cudaFree(d_einf);
    }
    cudaFree(d_zlin);
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
    // s_hat_v_c from the packed witness against eq(xfull_c[1..])
    vector<F128> suffix_c = build_eq_host(xfull_c.data() + 1, (int)xfull_c.size() - 1);
    vector<F128> shat_c = fold_1b_rows_host(z_host, suffix_c);

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

    // b_combined + round-0 prime
    vector<F128> b_comb(len, F128{0, 0});
    {
        const vector<F128>* sufs[2] = { nullptr, &suffix_c };
        vector<F128> suffix_ab = build_eq_host(xfull_ab.data() + 1, (int)xfull_ab.size() - 1);
        sufs[0] = &suffix_ab;
        for (int i = 0; i < 2; i++) {
            vector<F128> wscaled(128);
            for (int j = 0; j < 128; j++) wscaled[j] = MUL(gam[i], rsw[i].eq_rd[j]);
            for (long long t = 0; t < len; t++)
                b_comb[t] = ADD(b_comb[t], fold_one_slot_host((*sufs[i])[t], wscaled));
        }
    }
    F128 r0u0{0, 0}, r0u2{0, 0};
    for (long long t = 0; t < len / 2; t++) {
        r0u0 = ADD(r0u0, MUL(z_host[2 * t], b_comb[2 * t]));
        r0u2 = ADD(r0u2, MUL(ADD(z_host[2 * t], z_host[2 * t + 1]), ADD(b_comb[2 * t], b_comb[2 * t + 1])));
    }
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
        F128 *dfp, *dcb, *df2, *dcb2, *p0, *p2, *du0, *du2;
        CK(cudaMalloc(&dfp, len * sizeof(F128))); CK(cudaMalloc(&dcb, len * sizeof(F128)));
        CK(cudaMalloc(&df2, len * sizeof(F128))); CK(cudaMalloc(&dcb2, len * sizeof(F128)));
        CK(cudaMalloc(&p0, SMC_MAX_BLOCKS * sizeof(F128))); CK(cudaMalloc(&p2, SMC_MAX_BLOCKS * sizeof(F128)));
        CK(cudaMalloc(&du0, sizeof(F128))); CK(cudaMalloc(&du2, sizeof(F128)));
        CK(cudaMemcpy(dfp, df, len * sizeof(F128), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dcb, b_comb.data(), len * sizeof(F128), cudaMemcpyHostToDevice));
        F128 *cf = dfp, *ccb = dcb, *nf = df2, *ncb = dcb2;
        long long slen = len;
        F128 u0, u2;

        // host copies of the current tree/codeword for opening
        vector<F128> h_cw0((size_t)l0_bl * l0_ni);
        CK(cudaMemcpy(h_cw0.data(), d_cw0, h_cw0.size() * sizeof(F128), cudaMemcpyDeviceToHost));
        vector<MHash> h_tree0(2 * l0_bl - 1);
        CK(cudaMemcpy(h_tree0.data(), d_tree0, h_tree0.size() * 32, cudaMemcpyDeviceToHost));
        cudaFree(d_cw0); cudaFree(d_tree0);

        vector<F128> r_lane;
        for (int j = 0; j < P->initial_k; j++) {
            int bits = P->fold_grinding_bits[0] - j; if (bits < 0) bits = 0;
            if (bits > 0) fold_grind_nonces.push_back(ch.grind_pow((uint32_t)bits));
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
        CK(cudaMalloc(&d_bnew, n1_len * sizeof(F128)));
        CK(cudaMalloc(&ep0, IGL_MAX_BLOCKS * sizeof(F128))); CK(cudaMalloc(&ep2, IGL_MAX_BLOCKS * sizeof(F128)));
        CK(cudaMalloc(&epodd, IGL_MAX_BLOCKS * sizeof(F128)));
        CK(cudaMalloc(&eu0, sizeof(F128))); CK(cudaMalloc(&eu2, sizeof(F128))); CK(cudaMalloc(&ehnew, sizeof(F128)));

        vector<F128> prev_cw = std::move(h_cw0);
        vector<MHash> prev_tree = std::move(h_tree0);
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
                    if (bits > 0) fold_grind_nonces.push_back(ch.grind_pow((uint32_t)bits));
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
                grind_nonces.push_back(ch.grind_pow((uint32_t)P->grinding_bits[lvl]));
                std::vector<size_t> q = ch.sample_distinct_queries((size_t)prev_bl, P->queries[lvl]);
                LevelOpen lo; lo.n_rows = q.size(); lo.row_len = prev_ni;
                lo.rows_flat.resize(lo.n_rows * lo.row_len);
                for (size_t i = 0; i < q.size(); i++)
                    memcpy(&lo.rows_flat[i * prev_ni], &prev_cw[q[i] * prev_ni], prev_ni * sizeof(F128));
                lo.proof = merkle_multi_proof_host(prev_tree.data(), (size_t)prev_bl, q);
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
                CK(cudaMalloc(&d_cwn, cw_len * sizeof(F128)));
                CK(cudaMalloc(&d_tw, tt.data.size() * sizeof(F128)));
                CK(cudaMemcpy(d_tw, tt.data.data(), tt.data.size() * sizeof(F128), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&d_treen, (size_t)(2 * bln - 1) * 32));
                if (ntt_can_fuse_src(k_code - rate_next)) {
                    launch_ntt(d_cwn, d_tw, tt, rate_next, k_code, num_ntts, 256, false, cf, nn_len - 1);
                } else {
                    int tpb = 256;
                    replicate_fill_ffi<<<(unsigned)((cw_len + tpb - 1) / tpb), tpb>>>(cf, d_cwn, cw_len, nn_len);
                    launch_ntt(d_cwn, d_tw, tt, rate_next, k_code, num_ntts);
                }
                CK(cudaGetLastError());
                launch_merkle((const uint8_t*)d_cwn, d_treen, bln, num_ntts * 16);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(rn, d_treen + (size_t)(2 * bln - 2) * 32, 32, cudaMemcpyDeviceToHost));
                cudaFree(d_tw);
            }
            ch.observe_bytes(rn, 32);
            { MHash h; memcpy(h.b, rn, 32); rec_roots.push_back(h); }
            vector<F128> next_cw((size_t)bln * lanesn);
            CK(cudaMemcpy(next_cw.data(), d_cwn, next_cw.size() * sizeof(F128), cudaMemcpyDeviceToHost));
            vector<MHash> next_tree(2 * bln - 1);
            CK(cudaMemcpy(next_tree.data(), d_treen, next_tree.size() * 32, cudaMemcpyDeviceToHost));
            cudaFree(d_cwn); cudaFree(d_treen);

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

            grind_nonces.push_back(ch.grind_pow((uint32_t)P->grinding_bits[lvl]));
            std::vector<size_t> q = ch.sample_distinct_queries((size_t)prev_bl, P->queries[lvl]);
            int al = 0; { int mq = P->queries[lvl] - 1; while (mq) { al++; mq >>= 1; } }
            if (P->queries[lvl] <= 1) al = 0;
            std::vector<ChF128> alpha(al); ch.sample_f128_vec(alpha.data(), al);
            vector<F128> alpha_f(al);
            for (int i = 0; i < al; i++) alpha_f[i] = frch(alpha[i]);

            LevelOpen lo; lo.n_rows = q.size(); lo.row_len = prev_ni;
            lo.rows_flat.resize(lo.n_rows * lo.row_len);
            for (size_t i = 0; i < q.size(); i++)
                memcpy(&lo.rows_flat[i * prev_ni], &prev_cw[q[i] * prev_ni], prev_ni * sizeof(F128));
            lo.proof = merkle_multi_proof_host(prev_tree.data(), (size_t)prev_bl, q);

            std::vector<unsigned long long> qull(q.size());
            for (size_t i = 0; i < q.size(); i++) qull[i] = q[i];
            vector<F128> sks = eval_sk_at_vks_hd(n_next);
            InduceSetupDev S = induce_setup_device(n_next, sks, r_lane, alpha_f, qull, lo.rows_flat, prev_ni);
            level_opens.push_back(std::move(lo));
            F128* d_basis;
            CK(cudaMalloc(&d_basis, (size_t)nn_len * sizeof(F128)));
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
            cudaFree(S.d_low); cudaFree(S.d_sh); cudaFree(d_basis);

            prev_cw = std::move(next_cw); prev_tree = std::move(next_tree);
            prev_bl = bln; prev_ni = lanesn;
        }

        cudaFree(dfp); cudaFree(dcb); cudaFree(df2); cudaFree(dcb2);
        cudaFree(p0); cudaFree(p2); cudaFree(du0); cudaFree(du2);
        cudaFree(d_bnew); cudaFree(ep0); cudaFree(ep2); cudaFree(epodd);
        cudaFree(eu0); cudaFree(eu2); cudaFree(ehnew);
    }
    cudaFree(df);

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
