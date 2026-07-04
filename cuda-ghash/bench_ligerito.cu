// End-to-end Ligerito open prover benchmark (GPU pcs::open, step 7) — runs the
// full prove (no oracle, no validation): host FsChallenger derives every
// challenge, device kernels do the compute, host does multi-proof + induce
// setup. Times each phase via wall clock (device synced at boundaries).
//
// Mirrors test_ligerito_l0.cu's prove exactly, minus the byte-for-byte checks.
//
// Build:  make bench_ligerito
// Run:    ./bench_ligerito log_n initial_k log_inv_rate_0 num_queries_0 \
//                          log_inv_rate_1 ood r k_rec rate_rec ood_rec nq_rec [iters]
//   default: 22 5 1 148 1 1 2 3 1 1 148
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <chrono>
#include <string>
#include <fstream>
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
#include "zerocheck_round1_cpustyle.cuh"   // fast cpustyle3 round-1 (launch_zc_round1_fast)
#include "zerocheck_round2.cuh"
#include "zerocheck_tail.cuh"
#include "phi8_table.cuh"
#include "challenger.hpp"
#include "zc_challenger_device.cuh"   // resident on-device challenger for the tail
static ZcSha zc_pack(const Sha256& s){ ZcSha z; for(int i=0;i<8;i++)z.h[i]=s.h[i]; z.total_len=s.total_len;
    for(int i=0;i<64;i++)z.buf[i]=s.buf[i]; z.buf_len=(unsigned)s.buf_len; return z; }
static void zc_unpack(Sha256& s, const ZcSha& z){ for(int i=0;i<8;i++)s.h[i]=z.h[i]; s.total_len=z.total_len;
    for(int i=0;i<64;i++)s.buf[i]=z.buf[i]; s.buf_len=z.buf_len; }

#define CK(x) do { cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)
static const uint8_t PROVER_LABEL[] = "flock-ligerito-basis-v0";
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { CK(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

__global__ void fill2(F128* A, F128* B, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    u64 x = (u64)i * 0x9E3779B97F4A7C15ull + 1; A[i] = F128{x, x*0xBF58476D1CE4E5B9ull};
    B[i] = F128{x ^ 0x55, x*0x2545F4914F6CDD1Dull};
}
__global__ void replicate_fill(const F128* __restrict__ m, F128* __restrict__ cw, long long cw_len, long long ml) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= cw_len) return; cw[i] = m[i % ml];
}
// Gather the `nq` queried rows (ni F128 each) from the device codeword.
__global__ void gather_rows_k(const F128* __restrict__ cw, const unsigned long long* __restrict__ q,
                              int nq, int ni, F128* __restrict__ out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)nq * ni) return;
    int i = (int)(idx / ni), l = (int)(idx % ni);
    out[idx] = cw[q[i] * ni + l];
}
// Gather rows from a device codeword into a host vector (small D2H).
static std::vector<F128> gather_host(const F128* d_cw, const std::vector<size_t>& q, int ni) {
    int nq = (int)q.size();
    std::vector<unsigned long long> qh(nq); for (int i = 0; i < nq; i++) qh[i] = q[i];
    unsigned long long* d_q; CK(cudaMalloc(&d_q, nq*sizeof(unsigned long long)));
    CK(cudaMemcpy(d_q, qh.data(), nq*sizeof(unsigned long long), cudaMemcpyHostToDevice));
    F128* d_rows; CK(cudaMalloc(&d_rows, (size_t)nq*ni*sizeof(F128)));
    int tpb=256; gather_rows_k<<<(unsigned)(((long long)nq*ni+tpb-1)/tpb),tpb>>>(d_cw, d_q, nq, ni, d_rows);
    std::vector<F128> out((size_t)nq*ni);
    CK(cudaMemcpy(out.data(), d_rows, out.size()*sizeof(F128), cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_rows);
    return out;
}
static ChF128 to_ch(F128 x){ return ChF128{x.lo,x.hi}; }

// Twiddle tables are data-independent static data (the CPU ligero_commit takes
// a precomputed AdditiveNttF128) — build + upload once per k_code, reuse.
static std::map<int, TwiddleTable> g_tt;
static std::map<int, F128*> g_dtw;
static const TwiddleTable& cached_tt(int k_code, F128*& d_tw) {
    auto it = g_tt.find(k_code);
    if (it == g_tt.end()) {
        g_tt[k_code] = build_twiddle_table(k_code);
        F128* dtw; CK(cudaMalloc(&dtw, g_tt[k_code].data.size()*sizeof(F128)));
        CK(cudaMemcpy(dtw, g_tt[k_code].data.data(), g_tt[k_code].data.size()*sizeof(F128), cudaMemcpyHostToDevice));
        g_dtw[k_code] = dtw;
    }
    d_tw = g_dtw[k_code];
    return g_tt[k_code];
}

// device ligero_commit; returns root (host), leaves codeword+tree on device.
static void commit_dev(const F128* d_src, int msg_log, int log_msg_cols, int log_ni, int log_inv_rate,
                       F128*& d_cw, uint8_t*& d_tree, long long& block_len, int& num_ntts, uint8_t root[32]) {
    int k_code = log_msg_cols + log_inv_rate; num_ntts = 1 << log_ni; block_len = 1LL << k_code;
    long long cw_len = block_len * num_ntts, msg_len = 1LL << msg_log;
    F128* d_tw; const TwiddleTable& tt = cached_tt(k_code, d_tw);
    CK(cudaMalloc(&d_cw, cw_len*sizeof(F128)));
    CK(cudaMalloc(&d_tree, (size_t)(2*block_len-1)*32));
    int tpb=256; replicate_fill<<<(unsigned)((cw_len+tpb-1)/tpb),tpb>>>(d_src,d_cw,cw_len,msg_len);
    launch_ntt(d_cw,d_tw,tt,log_inv_rate,k_code,num_ntts);
    launch_merkle((const uint8_t*)d_cw,d_tree,block_len,num_ntts*16,256,4);   // kway=4 ILP
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(root, d_tree+(size_t)(2*block_len-2)*32, 32, cudaMemcpyDeviceToHost));
}
static void msg(const F128*A,const F128*B,long long len,F128*p0,F128*p2,F128*du0,F128*du2,F128&u0,F128&u2){
    launch_sumcheck_msg(A,B,len/2,p0,p2,du0,du2);
    CK(cudaMemcpy(&u0,du0,sizeof(F128),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&u2,du2,sizeof(F128),cudaMemcpyDeviceToHost));
}

struct Phase { double commit=0, fold=0, ood=0, open=0, induce=0, intro=0, lincheck=0, witness=0, zerocheck=0, l0commit=0; };

// Full zerocheck prove_packed orchestration, resident on the witness products
// a=A·z, b=B·z (c=z=df), threading the shared challenger `ch`. Times into
// ph.zerocheck and returns the x_ab quirky point (z_skip + mlv challenges) that
// lincheck consumes — closing the zerocheck→lincheck hand-off on-GPU. Tables M/
// f8mul are data-independent; arbitrary patterns here (this is a timing/dataflow
// bench — correctness is in test_zerocheck_full). m = log_n + 7, k_skip = 6.
static void zerocheck_phase(F128* da, F128* db, F128* dc, int m, FsChallenger& ch, Phase& ph,
                            F128& z_skip_out, std::vector<F128>& x_inner_rest,
                            std::vector<F128>& x_outer, int lc_k_log) {
    const int k_skip = 6;
    long long rows = 1LL << (m - 6);          // round-1 rows / a_mlv length

    static bool tables_done = false;
    if (!tables_done) {
        std::vector<uint8_t> mcol(64 * 64), f8mul((size_t)256 * 256);
        for (size_t i = 0; i < mcol.size(); i++) mcol[i] = (uint8_t)(i * 7 + 1);
        for (size_t i = 0; i < f8mul.size(); i++) f8mul[i] = (uint8_t)(i ^ (i >> 8));
        zc_round1_upload_tables(mcol.data(), f8mul.data(), PHI_8_TABLE);
        tables_done = true;
    }

    F128 *d_eq, *d_r1ab, *d_r1c, *d_ft, *d_am, *d_bm, *d_amn, *d_bmn, *d_p1, *d_pinf, *d_m1, *d_minf;
    CK(cudaMalloc(&d_eq, rows * sizeof(F128)));
    CK(cudaMalloc(&d_r1ab, 64 * sizeof(F128))); CK(cudaMalloc(&d_r1c, 64 * sizeof(F128)));
    CK(cudaMalloc(&d_ft, 8 * 256 * sizeof(F128)));
    CK(cudaMalloc(&d_am, rows * sizeof(F128))); CK(cudaMalloc(&d_bm, rows * sizeof(F128)));
    CK(cudaMalloc(&d_amn, rows * sizeof(F128))); CK(cudaMalloc(&d_bmn, rows * sizeof(F128)));
    CK(cudaMalloc(&d_p1, ZT_MAX_BLOCKS * sizeof(F128))); CK(cudaMalloc(&d_pinf, ZT_MAX_BLOCKS * sizeof(F128)));
    CK(cudaMalloc(&d_m1, sizeof(F128))); CK(cudaMalloc(&d_minf, sizeof(F128)));
    ZcSha* d_state; F128 *d_rho, *d_rhos, *d_eqlo, *d_eqhi;
    CK(cudaMalloc(&d_state, sizeof(ZcSha))); CK(cudaMalloc(&d_rho, sizeof(F128)));
    CK(cudaMalloc(&d_rhos, (m - 6) * sizeof(F128)));
    F128* d_scales; CK(cudaMalloc(&d_scales, (m - 6) * sizeof(F128)));
    // split-eq tables (see zerocheck_tail.cuh): lo = (m-7)-7 vars, hi = 7 vars.
    const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
    CK(cudaMalloc(&d_eqlo, (1LL << zt_lobits) * sizeof(F128)));
    CK(cudaMalloc(&d_eqhi, (1LL << (zt_dfull - zt_lobits)) * sizeof(F128)));
    const F128 ONE{1, 0};

    auto t = Clock::now();
    ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);
    // r = [r_skip(6) | small(3) | medium(4) | r_outer(m-13)].
    std::vector<ChF128> rs(6); ch.sample_f128_vec(rs.data(), 6);
    std::vector<ChF128> ro(m - 13); ch.sample_f128_vec(ro.data(), m - 13);
    std::vector<F128> r(m);
    for (int i = 0; i < 6; i++) r[i] = F128{rs[i].lo, rs[i].hi};
    int sm[3] = {0xF7, 0x53, 0xB5};
    for (int i = 0; i < 3; i++) r[6 + i] = PHI_8_TABLE[sm[i]];
    F128 gm[4] = {F128{2, 0}, F128{4, 0}, F128{16, 0}, F128{256, 0}};
    for (int i = 0; i < 4; i++) r[9 + i] = f128_mul_hd(gm[i], f128_inv_host(f128_add_hd(ONE, gm[i])));
    for (int i = 0; i < m - 13; i++) r[13 + i] = F128{ro[i].lo, ro[i].hi};

    double t_r1=0,t_r1eq=0,t_r2=0,t_msg1=0,t_tail=0,t_fin=0; auto _s=Clock::now();
    static int zc_call = 0; zc_call++; const bool zc_det = (zc_call == 2);
    cudaEvent_t ev0, ev1, evk; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1)); CK(cudaEventCreate(&evk));
    float det_r1k=0, det_r2k=0, det_eqb=0, det_msg1k=0;
    double det_r2host=0, tr_wall[40]={0}; float tr_gpu[40]={0}; int tr_n=0; long long tr_op[40]={0};
    // round-1 URM. cpustyle3 only needs eq_out = eq(r[13..m]) (the stride-128 subsample),
    // NOT the full eq(r[6..m]) — build the small one directly (128x less) + the C_s.C_med
    // scale = prod_{i=6..12}(1+r[i]), and call cpustyle3 directly (skip launch_zc_round1_fast's
    // wasteful full-eq build + stride kernel).
    { std::vector<F128> ro13(r.begin() + 13, r.end()); build_eq_device(d_eq, ro13.data(), m - 13); }
    F128 r1scale = ONE; for (int i = 6; i < 13; i++) r1scale = f128_mul_hd(r1scale, f128_add_hd(ONE, r[i]));
    t_r1eq=ms_since(_s); _s=Clock::now();
    cudaEventRecord(ev0);
    launch_zc_round1_cpustyle3((const uint8_t*)da, (const uint8_t*)db, (const uint8_t*)dc,
                               d_eq, 1LL << (m - 13), nullptr, r1scale, d_r1ab, d_r1c);  // cpustyle3 direct
    cudaEventRecord(ev1);
    std::vector<F128> r1ab(64), r1c(64);
    CK(cudaMemcpy(r1ab.data(), d_r1ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r1c.data(), d_r1c, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
    { std::vector<ChF128> s(64); for (int i = 0; i < 64; i++) s[i] = ChF128{r1ab[i].lo, r1ab[i].hi};
      ch.observe_f128_slice(s.data(), 64);
      for (int i = 0; i < 64; i++) s[i] = ChF128{r1c[i].lo, r1c[i].hi}; ch.observe_f128_slice(s.data(), 64); }
    ChF128 zc = ch.sample_f128(); F128 z{zc.lo, zc.hi};
    t_r1=ms_since(_s); _s=Clock::now();
    cudaEventElapsedTime(&det_r1k, ev0, ev1);

    // round-2 fold-at-z + first message.
    auto _h2 = Clock::now();
    std::vector<F128> ws = lagrange_weights_host(6, z);
    std::vector<F128> ft(8 * 256, F128{0, 0});
    for (int j = 0; j < 8; j++) for (int v = 0; v < 256; v++) { F128 acc{0, 0};
        for (int bb = 0; bb < 8; bb++) if ((v >> bb) & 1) acc = f128_add_hd(acc, ws[8 * j + bb]); ft[j * 256 + v] = acc; }
    det_r2host = std::chrono::duration<double, std::milli>(Clock::now() - _h2).count();
    CK(cudaMemcpy(d_ft, ft.data(), 8 * 256 * sizeof(F128), cudaMemcpyHostToDevice));
    cudaEventRecord(ev0);
    launch_zc_round2_fold((const uint8_t*)da, (const uint8_t*)db, d_ft, rows, d_am, d_bm);
    cudaEventRecord(ev1);
    t_r2=ms_since(_s); _s=Clock::now();
    cudaEventElapsedTime(&det_r2k, ev0, ev1);

    F128 *cA = d_am, *cB = d_bm, *nA = d_amn, *nB = d_bmn;
    long long len = rows;
    std::vector<F128> mlv_rhos;
    int n_mlv = m - 6;
    // SPLIT-EQ tail (host challenger): eqlo/eqhi built ONCE, each round's eq is an
    // index shift into them plus a scalar S_k = prod_{j=7}^{6+k}(1+r[j])^{-1} applied
    // to the two message sums in the combine kernel (see zerocheck_tail.cuh). No
    // per-round halve+scale pass, no full-size eq table ever built or streamed.
    // (A fully-resident on-device-challenger variant was tried — byte-exact but ~0.1 ms slower:
    // single-thread GPU SHA > CPU SHA; the tail was never round-trip-bound. See zc_challenger_device.cuh.)
    cudaEventRecord(ev0);
    build_eq_device(d_eqlo, &r[7], zt_lobits);
    build_eq_device(d_eqhi, &r[7 + zt_lobits], zt_dfull - zt_lobits);
    cudaEventRecord(ev1);
    {   // round-2 message: shift 0, scale ONE (eq = eq(r[7..m]) exactly)
        long long half = len / 2;
        launch_zt_msg_split(cA, cB, d_eqlo, d_eqhi, 0, zt_lobits, half, ONE, d_p1, d_pinf, d_m1, d_minf);
        cudaEventRecord(evk);
        F128 m1, mi; CK(cudaMemcpy(&m1, d_m1, sizeof(F128), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&mi, d_minf, sizeof(F128), cudaMemcpyDeviceToHost));
        ch.observe_f128(ChF128{m1.lo, m1.hi}); ch.observe_f128(ChF128{mi.lo, mi.hi});
        ChF128 rr = ch.sample_f128(); mlv_rhos.push_back(F128{rr.lo, rr.hi});
    }
    t_msg1=ms_since(_s); _s=Clock::now();
    cudaEventElapsedTime(&det_eqb, ev0, ev1); cudaEventElapsedTime(&det_msg1k, ev1, evk);
    int n_tail = n_mlv - 1;
    std::vector<F128> sc(n_tail);                                        // S_i = prod_{j=7}^{7+i}(1+r[j])^{-1}
    { std::vector<F128> v(n_tail), pre(n_tail); F128 acc = ONE;
      for (int i = 0; i < n_tail; i++) v[i] = f128_add_hd(ONE, r[7 + i]);
      for (int i = 0; i < n_tail; i++) { pre[i] = acc; acc = f128_mul_hd(acc, v[i]); }
      F128 inv = f128_inv_host(acc);
      for (int i = n_tail - 1; i >= 0; i--) { sc[i] = f128_mul_hd(pre[i], inv); inv = f128_mul_hd(inv, v[i]); }
      for (int i = 1; i < n_tail; i++) sc[i] = f128_mul_hd(sc[i - 1], sc[i]); }   // prefix products
    double fin_wall = 0; float fin_gpu = 0; int fin_rem = 0; long long fin_op0 = 0;
    { long long L = len;
      int i = 0;
      // host-challenger rounds while bandwidth-bound...
      for (; i < n_tail && L / 4 > ZT_FINISH_OP; i++) {
          long long op = L / 4;
          auto _rw = Clock::now();
          cudaEventRecord(ev0);
          launch_zt_fold_msg_split(cA, cB, nA, nB, d_eqlo, d_eqhi, i + 1, zt_lobits, op,
                                   mlv_rhos.back(), sc[i], d_p1, d_pinf, d_m1, d_minf);
          cudaEventRecord(evk);
          { F128* t2; t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; } len = len / 2;
          F128 m1, mi; CK(cudaMemcpy(&m1, d_m1, sizeof(F128), cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(&mi, d_minf, sizeof(F128), cudaMemcpyDeviceToHost));
          ch.observe_f128(ChF128{m1.lo, m1.hi}); ch.observe_f128(ChF128{mi.lo, mi.hi});
          ChF128 rr = ch.sample_f128();
          tr_wall[i] = std::chrono::duration<double, std::milli>(Clock::now() - _rw).count();
          cudaEventElapsedTime(&tr_gpu[i], ev0, evk); tr_op[i] = op; tr_n = i + 1;
          mlv_rhos.push_back(F128{rr.lo, rr.hi});
          L /= 2; }
      // ...then ONE fused finisher launch for the latency-floor rounds: fold +
      // message + on-device challenger every remaining round, no host round-trips.
      if (i < n_tail) {
          int rem = n_tail - i;
          auto _rw = Clock::now();
          ZcSha zs = zc_pack(ch.hasher);
          CK(cudaMemcpy(d_state, &zs, sizeof(ZcSha), cudaMemcpyHostToDevice));
          CK(cudaMemcpy(d_scales, sc.data() + i, rem * sizeof(F128), cudaMemcpyHostToDevice));
          cudaEventRecord(ev0);
          zt_tail_finisher<<<1, ZT_FIN_TPB>>>(cA, cB, nA, nB, d_eqlo, d_eqhi, zt_lobits, i + 1,
                                              d_scales, mlv_rhos.back(), rem, L,
                                              d_state, d_rhos, nullptr, nullptr);
          cudaEventRecord(evk);
          CK(cudaMemcpy(&zs, d_state, sizeof(ZcSha), cudaMemcpyDeviceToHost));
          zc_unpack(ch.hasher, zs);
          std::vector<F128> rh(rem);
          CK(cudaMemcpy(rh.data(), d_rhos, rem * sizeof(F128), cudaMemcpyDeviceToHost));
          for (int t = 0; t < rem; t++) mlv_rhos.push_back(rh[t]);
          if (rem & 1) { F128* t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; }
          fin_op0 = L / 4; len >>= rem; L >>= rem;
          fin_wall = std::chrono::duration<double, std::milli>(Clock::now() - _rw).count();
          cudaEventElapsedTime(&fin_gpu, ev0, evk); fin_rem = rem;
      } }
    t_tail=ms_since(_s); _s=Clock::now();
    (void)d_state; (void)d_rho; (void)d_rhos;
    { long long half = len / 2; launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos.back()); len = half; }  // final binding
    t_fin=ms_since(_s);
    CK(cudaDeviceSynchronize());
    ph.zerocheck += ms_since(t);
    static bool _pr=false; if(!_pr){_pr=true; printf("  [zc] r1-eqbuild %.2f  round1-kernel %.2f  round2(lag+fold) %.2f  msg1 %.2f  tail(%d) %.2f  final %.2f ms\n", t_r1eq,t_r1,t_r2,t_msg1,n_tail,t_tail,t_fin);}
    if (zc_det) {
        printf("  [zc-detail] (iter 1, post-warmup)\n");
        printf("  [zc-detail] round1: kernel %.3f | d2h+fiat-shamir %.3f ms  (reads a/b/c bit-packed: 3 x %lld MB)\n",
               det_r1k, t_r1 - det_r1k, (1LL << m) / 8 / (1 << 20));
        printf("  [zc-detail] round2: host lagrange/table %.3f | fold kernel %.3f | other %.3f ms\n",
               det_r2host, det_r2k, t_r2 - det_r2host - det_r2k);
        printf("  [zc-detail] msg1:   eqlo/eqhi build(%d+%d vars, once) %.3f | msg kernel %.3f | d2h+fs %.3f ms\n",
               zt_lobits, zt_dfull - zt_lobits, det_eqb, det_msg1k, t_msg1 - det_eqb - det_msg1k);
        double wsum = 0, gsum = 0;
        printf("  [zc-detail] tail per-round:\n");
        for (int i = 0; i < tr_n; i++) {
            wsum += tr_wall[i]; gsum += tr_gpu[i];
            printf("    r%-2d op=2^%-2d wall %6.3f  gpu %6.3f  host/launch %6.3f ms\n",
                   i, (int)(63 - __builtin_clzll(tr_op[i])), tr_wall[i], tr_gpu[i], tr_wall[i] - tr_gpu[i]);
        }
        if (fin_rem) {
            wsum += fin_wall; gsum += fin_gpu;
            printf("    finisher: %d rounds (op 2^%d..2^0) in ONE launch: wall %6.3f  gpu %6.3f  state-ship %6.3f ms\n",
                   fin_rem, (int)(63 - __builtin_clzll(fin_op0)), fin_wall, fin_gpu, fin_wall - fin_gpu);
        }
        printf("  [zc-detail] tail totals: wall %.3f = gpu-kernels %.3f + host+latency %.3f ms\n", wsum, gsum, wsum - gsum);
    }
    CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1)); CK(cudaEventDestroy(evk));

    // x_ab = QuirkyPoint{ z_skip=z, x_inner_rest=mlv_rhos[..inner_rest_len], x_outer=mlv_rhos[inner_rest_len..] }.
    int inner_rest_len = lc_k_log - k_skip;
    z_skip_out = z;
    x_inner_rest.assign(mlv_rhos.begin(), mlv_rhos.begin() + inner_rest_len);
    x_outer.assign(mlv_rhos.begin() + inner_rest_len, mlv_rhos.end());

    cudaFree(d_eq); cudaFree(d_r1ab); cudaFree(d_r1c); cudaFree(d_ft);
    cudaFree(d_am); cudaFree(d_bm); cudaFree(d_amn); cudaFree(d_bmn);
    cudaFree(d_p1); cudaFree(d_pinf); cudaFree(d_m1); cudaFree(d_minf);
    cudaFree(d_eqlo); cudaFree(d_eqhi);
    cudaFree(d_state); cudaFree(d_rho); cudaFree(d_rhos); cudaFree(d_scales);
}

// Fill SoA BLAKE3 Compression inputs with deterministic pseudo-random values.
__global__ void fill_compressions(uint32_t* cv, uint32_t* m, b3u64* ctr, uint32_t* blen,
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

// Resident BLAKE3 witness generation (S4 GPU target). Produces the real witness
// z (= df, the commit input) plus a = A·z, b = B·z (resident outputs for the
// future zerocheck) and the lincheck stripe-packed z (d_zlin), all on-device —
// the only H2D in the full prover would be the Compression inputs themselves.
// n_blocks_log = log_n - 7 (K_LOG=14), fully populated so no padding/memset.
static void witness_phase(F128* df, F128* da, F128* db, uint8_t* d_zlin, int log_n, Phase& ph) {
    int n_blocks_log = log_n - 7;
    long long n_total = 1LL << n_blocks_log;
    int n_blocks = (int)n_total;
    uint32_t *d_cv, *d_m, *d_blen, *d_flags; b3u64* d_ctr;
    CK(cudaMalloc(&d_cv, (size_t)n_blocks * 8 * 4)); CK(cudaMalloc(&d_m, (size_t)n_blocks * 16 * 4));
    CK(cudaMalloc(&d_blen, (size_t)n_blocks * 4)); CK(cudaMalloc(&d_flags, (size_t)n_blocks * 4));
    CK(cudaMalloc(&d_ctr, (size_t)n_blocks * 8));
    fill_compressions<<<(unsigned)((n_blocks + 127) / 128), 128>>>(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks);
    CK(cudaDeviceSynchronize());
    auto t = Clock::now();
    launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks, n_total,
                                 (b3u64*)df, (b3u64*)da, (b3u64*)db);
    launch_blake3_lincheck_transpose((b3u64*)df, n_total, d_zlin);
    ph.witness += ms_since(t);
    cudaFree(d_cv); cudaFree(d_m); cudaFree(d_blen); cudaFree(d_flags); cudaFree(d_ctr);
}

// Lincheck phase (src/lincheck.rs) run resident on the committed witness. In the
// full prover lincheck sits between zerocheck and PCS-open, reducing zerocheck's
// (â, b̂) claims to one z-claim. GPU zerocheck isn't ported yet, so here we drive
// the three lincheck kernels (CSC fold → comb_vec, partial fold → z_vec, top-bit
// product-sumcheck) over the RESIDENT L0 witness to measure their cost in place.
//
// Now fed the REAL resident stripe-packed witness `d_zlin` (from the GPU
// witness-gen transpose) AND the REAL x_ab (z_skip + mlv challenges) from the
// resident GPU zerocheck — closing the zerocheck→lincheck hand-off on-GPU.
// The CSC base matrices are the REAL BLAKE3 R1CS (A_0, B_0) loaded from
// blake3_lincheck_matrices.bin (dump_blake3_lincheck_matrices); α is arbitrary
// (timing bench, fold_alpha_batched is α-linear so timing is α-independent).

// Real BLAKE3 R1CS lincheck CSC matrices (GF(2), implicit ones), loaded once and
// cached. File: dump_blake3_lincheck_matrices.bin (magic "BL3M").
struct B3LincheckMatrices {
    int n_cols = 0, useful_bits = 0;
    std::vector<uint32_t> a_col_ptr, a_rows, b_col_ptr, b_rows;
};
static const B3LincheckMatrices& load_b3_lincheck_matrices() {
    static B3LincheckMatrices M;
    static bool done = false;
    if (done) return M;
    const char* path = "blake3_lincheck_matrices.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("FATAL: cannot open %s (run: make blake3_lincheck_matrices)\n", path); exit(1); }
    auto ru32 = [&](uint32_t& v){ f.read((char*)&v, 4); };
    auto rvec = [&](std::vector<uint32_t>& v, size_t n){ v.resize(n); f.read((char*)v.data(), n*4); };
    uint32_t magic; ru32(magic);
    if (magic != 0x424C334Du) { printf("FATAL: %s bad magic 0x%08X\n", path, magic); exit(1); }
    uint32_t ncols, ub, annz, bnnz;
    ru32(ncols); ru32(ub); M.n_cols = (int)ncols; M.useful_bits = (int)ub;
    ru32(annz); rvec(M.a_col_ptr, ncols + 1); rvec(M.a_rows, annz);
    ru32(bnnz); rvec(M.b_col_ptr, ncols + 1); rvec(M.b_rows, bnnz);
    if (!f) { printf("FATAL: %s truncated\n", path); exit(1); }
    done = true;
    return M;
}

static void lincheck_phase(const uint8_t* d_zlin, int m, int k_log, int k_skip,
                           F128 z_skip, const std::vector<F128>& x_inner_rest,
                           const std::vector<F128>& x_outer, Phase& ph) {
    int n_log = m - k_log;
    if (n_log < 3) return;                       // too small for byte stripes
    int k = 1 << k_log;
    int inner_rest_len = k_log - k_skip;
    long long n_outer = 1LL << n_log, n_stripes = n_outer / 8;

    F128 alpha{0x9abc, 0xdef0};
    std::vector<F128> eq_inner = build_quirky_eq_table_host(z_skip, x_inner_rest, k_skip);
    std::vector<F128> eq_outer = build_eq_table_host(x_outer);

    // --- REAL BLAKE3 R1CS base matrices (A_0, B_0), GF(2) CSC, loaded once.
    const B3LincheckMatrices& M = load_b3_lincheck_matrices();
    if (M.n_cols != k) { printf("FATAL: matrix n_cols %d != k %d (k_log mismatch)\n", M.n_cols, k); exit(1); }
    const std::vector<uint32_t>& a_col_ptr = M.a_col_ptr;
    const std::vector<uint32_t>& a_rows    = M.a_rows;
    const std::vector<uint32_t>& b_col_ptr = M.b_col_ptr;
    const std::vector<uint32_t>& b_rows    = M.b_rows;
    int useful_bits = M.useful_bits;            // 15409 for BLAKE3 (rest is padding)

    F128 *d_eq_inner, *d_comb, *d_zvec, *d_eq_outer, *d_nC, *d_nZ, *d_p1, *d_pinf, *d_e1, *d_einf;
    uint32_t *d_acp, *d_ar, *d_bcp, *d_br;
    CK(cudaMalloc(&d_eq_inner, k*sizeof(F128))); CK(cudaMalloc(&d_comb, k*sizeof(F128)));
    CK(cudaMalloc(&d_zvec, k*sizeof(F128))); CK(cudaMalloc(&d_nC, k*sizeof(F128))); CK(cudaMalloc(&d_nZ, k*sizeof(F128)));
    CK(cudaMalloc(&d_eq_outer, n_outer*sizeof(F128)));
    CK(cudaMalloc(&d_acp,(k+1)*sizeof(uint32_t))); CK(cudaMalloc(&d_ar,a_rows.size()*sizeof(uint32_t)));
    CK(cudaMalloc(&d_bcp,(k+1)*sizeof(uint32_t))); CK(cudaMalloc(&d_br,b_rows.size()*sizeof(uint32_t)));
    CK(cudaMalloc(&d_p1,LC_MAX_BLOCKS*sizeof(F128))); CK(cudaMalloc(&d_pinf,LC_MAX_BLOCKS*sizeof(F128)));
    CK(cudaMalloc(&d_e1,sizeof(F128))); CK(cudaMalloc(&d_einf,sizeof(F128)));
    CK(cudaMemcpy(d_eq_inner, eq_inner.data(), k*sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_eq_outer, eq_outer.data(), n_outer*sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_acp,a_col_ptr.data(),(k+1)*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ar,a_rows.data(),a_rows.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bcp,b_col_ptr.data(),(k+1)*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_br,b_rows.data(),b_rows.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));

    std::vector<F128> chal(inner_rest_len);
    for (int r = 0; r < inner_rest_len; r++) chal[r] = F128{(u64)(r*2654435761ull+1), (u64)(r*40503+7)};

    auto t = Clock::now();
    launch_lincheck_csc_fold(d_eq_inner, d_acp, d_ar, d_bcp, d_br, alpha, k, d_comb);
    launch_lincheck_partial_fold(d_zlin, d_eq_outer, n_stripes, k, useful_bits, d_zvec);
    F128 *cC=d_comb,*cZ=d_zvec,*nC=d_nC,*nZ=d_nZ; long long len=k;
    for (int r = 0; r < inner_rest_len; r++) {
        long long half = len/2;
        launch_lincheck_msg(cC, cZ, half, d_p1, d_pinf, d_e1, d_einf);
        launch_lincheck_fold2(cC, cZ, nC, nZ, half, chal[r]);
        F128* z; z=cC;cC=nC;nC=z; z=cZ;cZ=nZ;nZ=z; len=half;
    }
    ph.lincheck += ms_since(t);

    cudaFree(d_eq_inner); cudaFree(d_comb); cudaFree(d_zvec); cudaFree(d_nC); cudaFree(d_nZ);
    cudaFree(d_eq_outer);
    cudaFree(d_acp); cudaFree(d_ar); cudaFree(d_bcp); cudaFree(d_br);
    cudaFree(d_p1); cudaFree(d_pinf); cudaFree(d_e1); cudaFree(d_einf);
}

static double prove(int log_n,int initial_k,int log_inv_rate_0,int num_queries_0,int log_inv_rate_1,
                    int ood1,int r,int k_rec,int ood_rec,
                    const std::vector<int>& rec_rates,const std::vector<int>& rec_queries, Phase& ph) {
    long long len = 1LL << log_n; int n1 = log_n - initial_k; long long n1_len = 1LL << n1;
    int log_ni1 = k_rec;
    // Sumcheck state allocated up front; the witness is filled directly into
    // (df, dcb) — no separate d_f/d_b1 (saves 2 full-size buffers, matters at m≥34).
    F128 *df,*dcb,*df2,*dcb2,*du0,*du2,*p0,*p2;
    CK(cudaMalloc(&df,len*sizeof(F128)));CK(cudaMalloc(&dcb,len*sizeof(F128)));CK(cudaMalloc(&df2,len*sizeof(F128)));CK(cudaMalloc(&dcb2,len*sizeof(F128)));
    CK(cudaMalloc(&p0,SMC_MAX_BLOCKS*sizeof(F128)));CK(cudaMalloc(&p2,SMC_MAX_BLOCKS*sizeof(F128)));CK(cudaMalloc(&du0,sizeof(F128)));CK(cudaMalloc(&du2,sizeof(F128)));
    { int tpb=256; fill2<<<(unsigned)((len+tpb-1)/tpb),tpb>>>(df,dcb,len); CK(cudaDeviceSynchronize()); }

    // ---- GPU witness generation (S4): produce the REAL witness z into `df`
    // (overwriting the random fill — `dcb` keeps its random basis), plus a/b and
    // the lincheck stripe `d_zlin`, all resident. `df` then feeds commit + the
    // open with no H2D. Requires n_blocks_log = log_n-7 >= 3.
    F128 *d_a=nullptr,*d_b=nullptr; uint8_t* d_zlin=nullptr;
    bool do_witness = (log_n - 7) >= 3;
    if (do_witness) {
        CK(cudaMalloc(&d_a,len*sizeof(F128))); CK(cudaMalloc(&d_b,len*sizeof(F128)));
        CK(cudaMalloc(&d_zlin,(size_t)len*16));   // 2^m/8 bytes = len*16
        witness_phase(df, d_a, d_b, d_zlin, log_n, ph);
        // a/b stay resident — consumed by zerocheck below, then freed before the open.
    }

    // L0 commit is the UPSTREAM commit phase (pcs::commit), NOT the open — the
    // open receives l0_codeword + l0_tree as borrowed inputs. Committed from the
    // witness (df, before any fold), before timing starts, excluded from the open.
    F128 *d_prev_cw; uint8_t *d_tree0; long long l0bl; int l0lanes; uint8_t l0root[32];
    auto t_l0c = Clock::now();
    commit_dev(df, log_n, log_n-initial_k, initial_k, log_inv_rate_0, d_prev_cw,d_tree0,l0bl,l0lanes,l0root);
    CK(cudaDeviceSynchronize()); ph.l0commit += ms_since(t_l0c);
    uint8_t* d_prev_tree = d_tree0;
    long long prev_bl=l0bl; int prev_ni=l0lanes;
    F128* d_l0_cw=d_prev_cw; uint8_t* d_l0_tree=d_prev_tree;  // borrowed input — freed after timing

    // ---- Shared Fiat-Shamir challenger, threaded through the whole chain:
    //   observe commitment → zerocheck → lincheck → open. This is the residency
    //   assembly: the resident witness products a/b feed zerocheck, whose x_ab
    //   feeds lincheck, all on-GPU with one transcript; the open continues on it.
    FsChallenger ch(PROVER_LABEL+0, 0); // domain unimportant for timing
    F128 target{0x1234,0x5678};
    ch.observe_label(PROVER_LABEL,sizeof(PROVER_LABEL)-1); ch.observe_f128(to_ch(target)); ch.observe_bytes(l0root,32);

    if (do_witness) {
        // Zerocheck resident on a=A·z, b=B·z, c=z(=df) → x_ab quirky point.
        F128 z_skip; std::vector<F128> x_inner_rest, x_outer;
        zerocheck_phase(d_a, d_b, df, log_n + 7, ch, ph, z_skip, x_inner_rest, x_outer, B3_K_LOG);
        cudaFree(d_a); d_a = nullptr; cudaFree(d_b); d_b = nullptr;   // consumed by zerocheck
        // Lincheck on the resident stripe witness with the REAL x_ab.
        lincheck_phase(d_zlin, log_n + 7, B3_K_LOG, 6, z_skip, x_inner_rest, x_outer, ph);
        cudaFree(d_zlin); d_zlin = nullptr;   // free before the open's codeword allocs
    }

    auto t_all = Clock::now();   // time the OPEN (commit + zerocheck + lincheck already done)

    F128 *cf=df,*ccb=dcb,*nf=df2,*ncb=dcb2; long long slen=len;
    F128 u0,u2;
    auto t=Clock::now(); msg(cf,ccb,slen,p0,p2,du0,du2,u0,u2); ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2));
    std::vector<F128> r_lane;
    for(int k=0;k<initial_k;k++){ ChF128 rc=ch.sample_f128(); F128 rr{rc.lo,rc.hi};
        long long half=slen/2; launch_sumcheck_fold_msg(cf,ccb,nf,ncb,half,rr,p0,p2,du0,du2); // fused fold + next msg (1 pass)
        {F128*z;z=cf;cf=nf;nf=z;z=ccb;ccb=ncb;ncb=z;} slen=half;
        CK(cudaMemcpy(&u0,du0,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&u2,du2,sizeof(F128),cudaMemcpyDeviceToHost));
        ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2)); r_lane.push_back(rr); }
    ph.fold += ms_since(t);

    // commit f1
    t=Clock::now();
    F128 *d_cw1; uint8_t *d_tree1; long long bl1; int lanes1; uint8_t root1[32];
    commit_dev(cf,n1,n1-log_ni1,log_ni1,log_inv_rate_1,d_cw1,d_tree1,bl1,lanes1,root1); ch.observe_bytes(root1,32);
    ph.commit += ms_since(t);  // keep d_cw1 + d_tree1 on device

    // OOD scratch
    F128 *d_bnew,*ep0,*ep2,*epodd,*eu0,*eu2,*ehnew;
    CK(cudaMalloc(&d_bnew,n1_len*sizeof(F128)));CK(cudaMalloc(&ep0,IGL_MAX_BLOCKS*sizeof(F128)));CK(cudaMalloc(&ep2,IGL_MAX_BLOCKS*sizeof(F128)));CK(cudaMalloc(&epodd,IGL_MAX_BLOCKS*sizeof(F128)));
    CK(cudaMalloc(&eu0,sizeof(F128)));CK(cudaMalloc(&eu2,sizeof(F128)));CK(cudaMalloc(&ehnew,sizeof(F128)));

    auto ood_loop=[&](int cnt,int nn){ long long nl=1LL<<nn; for(int o=0;o<cnt;o++){
        std::vector<ChF128> z(nn); ch.sample_f128_vec(z.data(),nn); std::vector<F128> zf(nn);
        for(int j=0;j<nn;j++) zf[j]=F128{z[j].lo,z[j].hi};
        build_eq_device(d_bnew, zf.data(), nn);   // device eq table (hardware clmad)
        launch_msg_eval(cf,d_bnew,nl/2,ep0,ep2,epodd,eu0,eu2,ehnew);
        F128 y,iu0,iu2; CK(cudaMemcpy(&iu0,eu0,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&iu2,eu2,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&y,ehnew,sizeof(F128),cudaMemcpyDeviceToHost));
        ch.observe_f128(to_ch(y));ch.observe_f128(to_ch(iu0));ch.observe_f128(to_ch(iu2));
        ChF128 bc=ch.sample_f128(); launch_glue(ccb,d_bnew,F128{bc.lo,bc.hi},nl); } };

    auto query_open_induce=[&](int nn,int nq,const F128* d_pcw,const uint8_t* d_ptree,long long pbl,int pni,std::vector<F128>&lvl_rs){
        long long nl=1LL<<nn;
        // grind(0) + sample queries + alpha
        ch.grind_pow(0);
        std::vector<size_t> q=ch.sample_distinct_queries((size_t)pbl,nq);
        int al=0;{int m=nq-1;while(m){al++;m>>=1;}} if(nq<=1)al=0;
        std::vector<ChF128> alpha(al); ch.sample_f128_vec(alpha.data(),al); std::vector<F128> af(al);
        for(int i=0;i<al;i++) af[i]=F128{alpha[i].lo,alpha[i].hi};
        (void)pni; (void)d_pcw; (void)lvl_rs;
        auto to=Clock::now();
        std::vector<MHash> mp=merkle_multi_proof_device(d_ptree,(size_t)pbl,q); ph.open += ms_since(to);
        // ---- transpose-NTT induce: scatter alpha_pows over the queried codeword
        // domain (pbl), Fᵀ-NTT, truncate to 2^nn = basis. (enforced_sum is not
        // transcript-affecting, so the prove bench omits it.) ----
        auto ti=Clock::now();
        int log_block=0; { long long b=pbl; while(b>1){ b>>=1; log_block++; } }
        long long ap_len = 1LL<<al;
        // Pooled grow-only induce scratch (d_c is pbl-sized = 128MB at m=35 L0):
        // reused across levels, no per-level malloc/free.
        static F128* d_ap=nullptr; static F128* d_c=nullptr; static unsigned long long* d_q=nullptr;
        static long long ap_cap=0, c_cap=0; static int q_cap=0;
        if(ap_len>ap_cap){ if(d_ap)cudaFree(d_ap); CK(cudaMalloc(&d_ap,ap_len*sizeof(F128))); ap_cap=ap_len; }
        if(pbl>c_cap){ if(d_c)cudaFree(d_c); CK(cudaMalloc(&d_c,pbl*sizeof(F128))); c_cap=pbl; }
        if(nq>q_cap){ if(d_q)cudaFree(d_q); CK(cudaMalloc(&d_q,nq*sizeof(unsigned long long))); q_cap=nq; }
        build_eq_device(d_ap, af.data(), al);
        std::vector<unsigned long long> qh(nq); for(int i=0;i<nq;i++) qh[i]=q[i];
        CK(cudaMemcpy(d_q,qh.data(),nq*sizeof(unsigned long long),cudaMemcpyHostToDevice));
        F128* d_tw; const TwiddleTable& tt=cached_tt(log_block,d_tw);
        int tpb2=256;
        zero_f128<<<(unsigned)((pbl+tpb2-1)/tpb2),tpb2>>>(d_c,pbl);
        scatter_weights<<<(unsigned)((nq+tpb2-1)/tpb2),tpb2>>>(d_c,d_q,d_ap,nq);
        launch_transpose_ntt(d_c,d_tw,tt,log_block);
        F128* dbasis=d_c;   // first nl elements are the truncated basis
        ph.induce += ms_since(ti);
        auto tg=Clock::now();
        msg(cf,dbasis,nl,p0,p2,du0,du2,u0,u2); ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2));
        ChF128 bi=ch.sample_f128(); launch_glue(ccb,dbasis,F128{bi.lo,bi.hi},nl); ph.intro += ms_since(tg);
        // pooled scratch — not freed per level
    };

    // L0 OOD + query/open/induce/introduce (query wtns_0)
    t=Clock::now(); ood_loop(ood1,n1); ph.ood += ms_since(t);
    query_open_induce(n1,num_queries_0,d_prev_cw,d_prev_tree,prev_bl,prev_ni,r_lane);
    // prev = wtns_1. wtns_0 (L0) is the BORROWED INPUT — a real open doesn't free
    // it (the caller owns it); freeing 8GB here would wrongly inflate the open. So
    // just adopt wtns_1; d_l0_cw/tree are released after the timer.
    d_prev_cw=d_cw1; d_prev_tree=d_tree1; prev_bl=bl1; prev_ni=lanes1;

    // recursive levels
    for(int lvl=0;lvl<r;lvl++){
        std::vector<F128> lvl_rs;
        t=Clock::now();
        for(int k=0;k<k_rec;k++){ ChF128 rc=ch.sample_f128(); F128 rr{rc.lo,rc.hi};
            long long half=slen/2; launch_sumcheck_fold_msg(cf,ccb,nf,ncb,half,rr,p0,p2,du0,du2); // fused fold + next msg (1 pass)
            {F128*z;z=cf;cf=nf;nf=z;z=ccb;ccb=ncb;ncb=z;} slen=half;
            CK(cudaMemcpy(&u0,du0,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&u2,du2,sizeof(F128),cudaMemcpyDeviceToHost));
            ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2)); lvl_rs.push_back(rr);}
        ph.fold += ms_since(t);
        if(lvl==r-1){ std::vector<F128> yr(slen); CK(cudaMemcpy(yr.data(),cf,(size_t)slen*sizeof(F128),cudaMemcpyDeviceToHost));
            for(long long i=0;i<slen;i++)ch.observe_f128(to_ch(yr[i]));
            ch.grind_pow(0); auto to=Clock::now(); std::vector<size_t> q=ch.sample_distinct_queries((size_t)prev_bl,rec_queries[lvl]);
            merkle_multi_proof_device(d_prev_tree,(size_t)prev_bl,q); ph.open += ms_since(to);
        } else {
            int nn=0;{long long s=slen;while(s>1){s>>=1;nn++;}}
            t=Clock::now(); F128*dcwn;uint8_t*dtn;long long bln;int ln;uint8_t rn[32];
            commit_dev(cf,nn,nn-k_rec,k_rec,rec_rates[lvl],dcwn,dtn,bln,ln,rn); ch.observe_bytes(rn,32);
            ph.commit += ms_since(t);   // keep dcwn + dtn on device
            t=Clock::now(); ood_loop(ood_rec,nn); ph.ood += ms_since(t);
            query_open_induce(nn,rec_queries[lvl],d_prev_cw,d_prev_tree,prev_bl,prev_ni,lvl_rs);
            cudaFree(d_prev_cw); cudaFree(d_prev_tree); d_prev_cw=dcwn; d_prev_tree=dtn; prev_bl=bln; prev_ni=ln;
        }
    }
    cudaFree(d_prev_cw); cudaFree(d_prev_tree);
    CK(cudaDeviceSynchronize());
    double total = std::chrono::duration<double,std::milli>(Clock::now()-t_all).count();
    cudaFree(d_l0_cw); cudaFree(d_l0_tree);   // borrowed input — released outside the timed open
    if (d_a) cudaFree(d_a); if (d_b) cudaFree(d_b); if (d_zlin) cudaFree(d_zlin);
    cudaFree(df);cudaFree(dcb);cudaFree(df2);cudaFree(dcb2);
    cudaFree(p0);cudaFree(p2);cudaFree(du0);cudaFree(du2);
    cudaFree(d_bnew);cudaFree(ep0);cudaFree(ep2);cudaFree(epodd);cudaFree(eu0);cudaFree(eu2);cudaFree(ehnew);
    return total;
}

int main(int argc,char**argv){
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("Device: %s | %d SMs\n",p.name,p.multiProcessorCount);
    int log_n, ik, r0, nq0, r1, ood1, r, k, oodr, iters;
    std::vector<int> rec_rates, rec_queries;

    if (argc > 1 && std::string(argv[1]) == "fast29") {
        // configs/ligerito/m29_fast.toml — grinding excluded (separate concern).
        log_n=22; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=4; k=3; oodr=1;
        rec_rates  = {3,4,5,5};        // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43};   // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):7;
        printf("Ligerito open [m29_fast config, grinding OFF]: log_n=22 initial_k=6 r=4 k_rec=3 "
               "rates=1,2,3,4,5  queries=218,106,71,53,43  ood=0,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast32") {
        // configs/ligerito/m32_fast.toml — grinding excluded.
        log_n=25; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=5; k=3; oodr=1;
        rec_rates  = {3,4,5,6,6};               // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36};         // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):5;
        printf("Ligerito open [m32_fast config, grinding OFF]: log_n=25 initial_k=6 r=5 k_rec=3 "
               "rates=1..6  queries=218,106,71,53,43,36  ood=0,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast33") {
        // configs/ligerito/m33_fast.toml — grinding excluded.
        log_n=26; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=5; k=3; oodr=1;
        rec_rates  = {3,4,5,6,6};               // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36};         // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):5;
        printf("Ligerito open [m33_fast config, grinding OFF]: log_n=26 initial_k=6 r=5 k_rec=3 "
               "rates=1..6  queries=218,106,71,53,43,36  ood=0,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast34") {
        // configs/ligerito/m34_fast.toml — grinding excluded.
        log_n=27; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=6; k=3; oodr=1;
        rec_rates  = {3,4,5,6,7,7};              // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36,32};       // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):3;
        printf("Ligerito open [m34_fast config, grinding OFF]: log_n=27 initial_k=6 r=6 k_rec=3 "
               "rates=1..7  queries=218,106,71,53,43,36,32  ood=0,1,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast35") {
        // configs/ligerito/m35_fast.toml — grinding excluded.
        log_n=28; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=6; k=3; oodr=1;
        rec_rates  = {3,4,5,6,7,7};               // log_inv_rates[lvl+2]
        rec_queries= {106,71,53,43,36,32};        // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):3;
        printf("Ligerito open [m35_fast config, grinding OFF]: log_n=28 initial_k=6 r=6 k_rec=3 "
               "rates=1..7  queries=218,106,71,53,43,36,32  ood=0,1,1,1,1,1,1\n");
    } else {
        auto A=[&](int i,int d){ return argc>i?atoi(argv[i]):d; };
        log_n=A(1,22);ik=A(2,5);r0=A(3,1);nq0=A(4,148);r1=A(5,1);ood1=A(6,1);r=A(7,2);k=A(8,3);
        int rr=A(9,1);oodr=A(10,1);int nqr=A(11,148);iters=A(12,5);
        rec_rates.assign(r, rr); rec_queries.assign(r, nqr);
        printf("Ligerito open: log_n=%d initial_k=%d r=%d k_rec=%d rate0=1/%d rate_rec=1/%d nq=%d/%d ood=%d/%d\n",
               log_n,ik,r,k,1<<r0,1<<rr,nq0,nqr,ood1,oodr);
    }

    Phase warm; prove(log_n,ik,r0,nq0,r1,ood1,r,k,oodr,rec_rates,rec_queries,warm); // warm-up
    Phase ph; double best=1e30;
    for(int it=0;it<iters;it++){ Phase p2; double t=prove(log_n,ik,r0,nq0,r1,ood1,r,k,oodr,rec_rates,rec_queries,p2); if(t<best){best=t;ph=p2;} }
    printf("  open total %.2f ms | commit %.2f  fold %.2f  ood %.2f  open(multiproof+gather) %.2f  induce %.2f  introduce/glue %.2f\n"
           "  resident chain: witness-gen %.2f  l0-commit %.2f  zerocheck %.2f  lincheck %.2f ms\n",
           best,ph.commit,ph.fold,ph.ood,ph.open,ph.induce,ph.intro,ph.witness,ph.l0commit,ph.zerocheck,ph.lincheck);
    double e2e = ph.witness + ph.l0commit + ph.zerocheck + ph.lincheck + best;
    printf("  >>> e2e prove total %.2f ms  (witness %.2f + l0-commit %.2f + zerocheck %.2f + lincheck %.2f + open %.2f)\n",
           e2e, ph.witness, ph.l0commit, ph.zerocheck, ph.lincheck, best);
    return 0;
}
