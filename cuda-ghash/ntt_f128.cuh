// Interleaved additive (LCH) NTT over GF(2^128) on CUDA — P2 of GPU_COMMIT_PLAN.
//
// Direct port of the scalar reference in
//   src/ntt/additive_ntt_f128.rs
// specifically `forward_transform_interleaved_scalar_from_layer` (the butterfly)
// and the twiddle schedule. Correctness-first: one layer per kernel launch,
// global memory, SoA layout `data[pos * num_ntts + lane]`. No shared-mem tiling
// / layer fusion yet — those are the P2 "then optimize" step, gated on this
// matching the oracle.
//
// THE one correctness risk per the plan is the twiddle schedule. Two facts pin
// it down (see src/pcs/commit.rs:270):
//   * the NTT is built with dim == k_code, and the per-lane buffer is 2^k_code,
//     so the basis length L == log_d. twiddle uses evals[L - layer - 1][1..].
//   * the 0-th element of each evals row is the normalized 1 and is "absorbed"
//     into the butterfly, hence the [1..] slice in the twiddle span.
// The twiddle table is built on the host (ntt_host.hpp) and validated on the
// CPU against the flare oracle (host_check_ntt.cpp) before this kernel runs.
//
// Field arithmetic: the device butterfly uses `ghash_mul_karatsuba` from
// f128.cuh — 3 carryless products (6 CLMAD) + reduction, the fastest multiply
// on this GPU in the bench_f128 experiments.
#pragma once
#include "f128.cuh"
#include "ntt_host.hpp"   // F128/u64 from f128.cuh above; host twiddle build

// ---------------------------------------------------------------------------
// One forward NTT layer over the interleaved SoA buffer. One thread per
// butterfly *lane* (block, row, lane). `tw_basis` points at layer `l`'s span
// basis (TwiddleTable::data + off[l]); it has `layer` entries.
//
// Matches the scalar reference exactly:
//   block_size = 1 << (log_d - layer);  half = block_size / 2
//   off_top = block*block_size*num_ntts + row*num_ntts + lane
//   off_bot = off_top + half*num_ntts
//   new_u = top + v*tw;  bot = v + new_u
// ---------------------------------------------------------------------------
__global__ void ntt_layer_kernel(F128* data, const F128* tw_basis,
                                 int layer, int log_d, int num_ntts) {
    long long half    = 1LL << (log_d - layer - 1);
    long long pairs   = half * (long long)num_ntts;      // butterfly lanes per block
    long long nblocks = 1LL << layer;
    long long total   = nblocks * pairs;

    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;

    long long block = tid / pairs;
    long long rem   = tid - block * pairs;
    long long row   = rem / num_ntts;
    long long lane  = rem - row * num_ntts;

    // twiddle(layer, block) = XOR of span-basis elements at set bits of block.
    F128 tw{0ull, 0ull};
    for (int j = 0; j < layer; j++) {
        if ((block >> j) & 1ull) tw = f128_add(tw, tw_basis[j]);
    }

    long long block_size  = half << 1;
    long long block_start = block * block_size * (long long)num_ntts;
    long long off_top = block_start + row * (long long)num_ntts + lane;
    long long off_bot = off_top + half * (long long)num_ntts;

    F128 v = data[off_bot];
    F128 u = f128_add(data[off_top], ghash_mul_karatsuba(v, tw));
    data[off_top] = u;
    data[off_bot] = f128_add(v, u);
}

// ---------------------------------------------------------------------------
// Multi-layer fusion. At realistic m the single-layer kernel is HBM-bound: one
// full-buffer read+write per layer. Fusing K consecutive layers loads each
// element once into registers, applies K butterfly layers, writes once —
// cutting full-buffer passes from log_dim to ceil(log_dim / K). Mirrors the
// CPU fused-2 / fused-4 kernels in src/ntt/additive_ntt_f128.rs.
// ---------------------------------------------------------------------------

// twiddle(layer, block) on device: XOR of layer's span basis at set bits of block.
__device__ __forceinline__ F128 dev_twiddle(const F128* basis, int layer, long long block) {
    F128 tw{0ull, 0ull};
    for (int j = 0; j < layer; j++)
        if ((block >> j) & 1ull) tw = f128_add(tw, basis[j]);
    return tw;
}

// One forward butterfly in a register array: nu = x[u] + x[v]*tw; x[v] += nu; x[u] = nu.
__device__ __forceinline__ void bf(F128* x, int u, int v, F128 tw) {
    F128 nu = f128_add(x[u], ghash_mul_karatsuba(x[v], tw));
    x[v] = f128_add(x[v], nu);
    x[u] = nu;
}

// Fuse 2 layers (L, L+1). One thread per (block, r, lane); 4 elements held in
// registers. Needs block_size = 2^(log_d-L) >= 4. Matches butterfly_fused_2layer.
__global__ void ntt_fused2_kernel(F128* data, const F128* bL, const F128* bL1,
                                  int L, int log_d, int num_ntts) {
    long long quarter    = 1LL << (log_d - L - 2);
    long long block_size = 1LL << (log_d - L);
    long long nblocks    = 1LL << L;
    long long total      = nblocks * quarter * (long long)num_ntts;

    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    long long lane  = tid % num_ntts;
    long long tmp   = tid / num_ntts;
    long long r     = tmp % quarter;
    long long block = tmp / quarter;

    long long stride = quarter * (long long)num_ntts;
    long long base   = block * block_size * (long long)num_ntts + r * (long long)num_ntts + lane;
    F128 x[4];
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = data[base + (long long)i * stride];

    F128 t0  = dev_twiddle(bL,  L,     block);
    F128 ta  = dev_twiddle(bL1, L + 1, 2 * block);
    F128 tb  = dev_twiddle(bL1, L + 1, 2 * block + 1);
    bf(x, 0, 2, t0); bf(x, 1, 3, t0);     // layer L:   (a,c) (b,d)
    bf(x, 0, 1, ta); bf(x, 2, 3, tb);     // layer L+1: (a,b) (c,d)

#pragma unroll
    for (int i = 0; i < 4; i++) data[base + (long long)i * stride] = x[i];
}

// Fuse 4 layers (L..L+3). One thread per (block, r, lane); 16 elements in
// registers. Needs block_size >= 16. Matches fused4_butterfly_scalar.
__global__ void ntt_fused4_kernel(F128* data, const F128* bL, const F128* bL1,
                                  const F128* bL2, const F128* bL3,
                                  int L, int log_d, int num_ntts) {
    long long sixteenth  = 1LL << (log_d - L - 4);
    long long block_size = 1LL << (log_d - L);
    long long nblocks    = 1LL << L;
    long long total      = nblocks * sixteenth * (long long)num_ntts;

    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    long long lane  = tid % num_ntts;
    long long tmp   = tid / num_ntts;
    long long r     = tmp % sixteenth;
    long long block = tmp / sixteenth;

    long long stride = sixteenth * (long long)num_ntts;
    long long base   = block * block_size * (long long)num_ntts + r * (long long)num_ntts + lane;
    F128 x[16];
#pragma unroll
    for (int i = 0; i < 16; i++) x[i] = data[base + (long long)i * stride];

    F128 t0 = dev_twiddle(bL, L, block);
#pragma unroll
    for (int i = 0; i < 8; i++) bf(x, i, i + 8, t0);                          // L  stride 8
#pragma unroll
    for (int s = 0; s < 2; s++) {
        F128 t = dev_twiddle(bL1, L + 1, 2 * block + s);
        for (int i = 0; i < 4; i++) bf(x, 8 * s + i, 8 * s + i + 4, t);       // L+1 stride 4
    }
#pragma unroll
    for (int s = 0; s < 4; s++) {
        F128 t = dev_twiddle(bL2, L + 2, 4 * block + s);
        for (int i = 0; i < 2; i++) bf(x, 4 * s + i, 4 * s + i + 2, t);       // L+2 stride 2
    }
#pragma unroll
    for (int s = 0; s < 8; s++) {
        F128 t = dev_twiddle(bL3, L + 3, 8 * block + s);
        bf(x, 2 * s, 2 * s + 1, t);                                          // L+3 stride 1
    }

#pragma unroll
    for (int i = 0; i < 16; i++) data[base + (long long)i * stride] = x[i];
}

// span basis pointer for layer l from the flattened table base: off[l] =
// l*(l-1)/2 (cumulative sum of 0..l-1), so no offset array is needed on device.
__device__ __forceinline__ const F128* tw_basis_for(const F128* d_tw, int l) {
    return d_tw + ((long long)l * (l - 1)) / 2;
}

// Deep-layer shared-memory tile. One threadblock owns one contiguous tile of
// `T = 2^dt` positions (× num_ntts lanes); after the top `top_layers` are done
// every remaining layer [top_layers, log_d) lives inside such a tile. Load the
// tile to shared mem once, run all `dt` layers on-chip with a barrier between
// them, write once — collapsing dt full-buffer passes into one. Mirrors the
// Rust cache-blocked Stage-2 (forward_transform_batched deep stage).
__global__ void ntt_deep_smem_kernel(F128* data, const F128* d_tw,
                                     int top_layers, int log_d, int num_ntts, int dt) {
    extern __shared__ F128 sm[];
    long long T      = 1LL << dt;
    long long tcount = T * num_ntts;                 // elements in this tile
    long long g      = blockIdx.x;                   // which tile
    long long base   = g * tcount;

    for (long long i = threadIdx.x; i < tcount; i += blockDim.x) sm[i] = data[base + i];
    __syncthreads();

    long long nbf = (T >> 1) * num_ntts;             // butterflies per layer in this tile
    for (int l = top_layers; l < log_d; l++) {
        int rel              = l - top_layers;        // 0..dt-1
        long long half       = 1LL << (dt - rel - 1); // sub-block half-size (positions)
        long long num_sub    = 1LL << rel;            // sub-blocks in this tile
        const F128* basis    = tw_basis_for(d_tw, l);
        for (long long t = threadIdx.x; t < nbf; t += blockDim.x) {
            long long lane = t % num_ntts;
            long long tmp  = t / num_ntts;            // 0..T/2-1
            long long sub  = tmp / half;
            long long row  = tmp % half;
            F128 tw = dev_twiddle(basis, l, g * num_sub + sub);
            long long sb_start = sub * (half << 1) * num_ntts;
            long long off_top  = sb_start + row * num_ntts + lane;
            long long off_bot  = off_top + half * num_ntts;
            F128 v = sm[off_bot];
            F128 u = f128_add(sm[off_top], ghash_mul_karatsuba(v, tw));
            sm[off_top] = u;
            sm[off_bot] = f128_add(v, u);
        }
        __syncthreads();
    }

    for (long long i = threadIdx.x; i < tcount; i += blockDim.x) data[base + i] = sm[i];
}

// Top-stage shared-memory fusion. Unlike the register-resident fusedK kernels
// (capped at K=4 by F128 register pressure — K=5 spills at 254 reg/thread), this
// holds the 2^K-position butterfly tile in SHARED memory, cooperatively across a
// block, so K can grow without per-thread register cost. The tile spans all
// `num_ntts` lanes (contiguous in the SoA layout) so global loads stay coalesced.
// Each block owns one (block, r) tile = 2^K positions x num_ntts lanes; it runs
// all K layers on-chip with a barrier between them, collapsing K full-buffer
// passes into one. smem index = pos*num_ntts + lane. Mirrors the fusedK butterfly
// schedule: layer L+j has within-tile stride 2^(K-1-j) and 2^j sub-blocks, each
// carrying twiddle dev_twiddle(basis_{L+j}, L+j, (block<<j)+sub).
// `lb` = lanes handled per block (a tile of the num_ntts batch dimension). At
// large num_ntts the full-lane tile (lb = num_ntts) blows the shared-mem budget
// and halves occupancy (e.g. 2^6·64·16 = 64 KB/block → ~3 blocks/SM on a 5090);
// tiling the lane dim keeps per-block smem ≈ 32 KB independent of num_ntts so
// occupancy (and thus throughput) stays high. Lanes are an independent batch
// axis — they don't affect the twiddle — so this is a pure partitioning, no math
// change. blockIdx.x enumerates (pos_tile, lane_tile): n_lane_tiles = num_ntts/lb
// inner-most. smem index = pos_in_tile*lb + lin (lin = lane within this block).
template <int K>
__global__ void ntt_smem_topK_kernel(F128* data, const F128* d_tw,
                                     int L, int log_d, int num_ntts, int lb,
                                     const F128* src, long long smask) {
    extern __shared__ F128 sm[];
    const int TILE       = 1 << K;
    const int NTW        = TILE - 1;          // distinct twiddles across the K layers
    long long seg        = 1LL << (log_d - L - K);
    long long block_size = 1LL << (log_d - L);
    int n_lane_tiles     = num_ntts / lb;
    long long pos_tile   = (long long)blockIdx.x / n_lane_tiles;
    long long lane_tile  = (long long)blockIdx.x % n_lane_tiles;
    long long r          = pos_tile % seg;
    long long block      = pos_tile / seg;
    long long lane_base  = lane_tile * (long long)lb;
    long long gbase      = block * block_size * (long long)num_ntts + r * (long long)num_ntts + lane_base;
    long long stride     = seg * (long long)num_ntts;
    long long tcount     = (long long)TILE * lb;
    F128* twid           = sm + tcount;       // NTW twiddles parked after the data tile

    // Coalesced load: smem[i*lb+lin] <- src[(gbase + i*stride + lin) & smask].
    // src/smask default to (data, -1) — identity. The rate-extend fusion passes
    // src = the pre-replication MESSAGE with smask = msg_elems-1: the codeword
    // before the NTT is cw[e] = msg[e mod msg_elems] (replicate_fill), so the
    // first pass can read the message directly (half the bytes) and the fill
    // pass disappears. Stores always go to data; src is a different buffer
    // there, and in the identity case all loads precede all stores per tile.
    for (long long e = threadIdx.x; e < tcount; e += blockDim.x) {
        long long i = e / lb, lin = e - i * lb;
        sm[e] = src[(gbase + i * stride + lin) & smask];
    }
    // Precompute the NTW distinct twiddles ONCE (was: dev_twiddle re-expanded per
    // butterfly — an O(layer) XOR loop recomputed across all strj*lb butterflies
    // that share a sub, up to ~1000x). Twiddle t encodes (j, sub) with
    // 2^j-1 <= t < 2^(j+1)-1 and sub = (t+1) - 2^j; value matches the per-layer
    // dev_twiddle(basis_{L+j}, L+j, (block<<j)+sub) bit-for-bit.
    for (int t = threadIdx.x; t < NTW; t += blockDim.x) {
        int j   = 31 - __clz(t + 1);          // floor(log2(t+1))
        int sub = (t + 1) - (1 << j);
        twid[t] = dev_twiddle(tw_basis_for(d_tw, L + j), L + j, (block << j) + sub);
    }
    __syncthreads();

    long long bpl = (long long)(TILE >> 1) * lb;         // butterflies per layer
#pragma unroll
    for (int j = 0; j < K; j++) {
        int strj  = 1 << (K - 1 - j);
        int twoff = (1 << j) - 1;                         // twid base for this layer
        for (long long q = threadIdx.x; q < bpl; q += blockDim.x) {
            long long lin  = q % lb;
            long long bi   = q / lb;                   // 0..TILE/2-1
            long long sub  = bi / strj;
            long long p    = bi - sub * strj;
            long long ubase= sub * (strj << 1) + p;
            F128 tw = twid[twoff + sub];               // shared-mem read, no re-expand
            long long ui = ubase * lb + lin;
            long long vi = (ubase + strj) * lb + lin;
            F128 a = sm[ui], b = sm[vi];
            F128 nu = f128_add(a, ghash_mul_karatsuba(b, tw));
            sm[vi] = f128_add(b, nu);
            sm[ui] = nu;
        }
        __syncthreads();
    }

    for (long long e = threadIdx.x; e < tcount; e += blockDim.x) {
        long long i = e / lb, lin = e - i * lb;
        data[gbase + i * stride + lin] = sm[e];
    }
}

// Launch one K-layer smem-fused chunk at `layer`, lane-tiling so per-block smem
// ≈ TOPK_SMEM_CAP regardless of num_ntts (keeps occupancy high; lb=num_ntts when
// it fits). Valid whenever layer+K <= log_d (always true for chunks of a balanced
// split of [from,to) with to <= log_d).
template <int K>
inline void launch_topK_chunk(F128* d_data, const F128* d_tw, int layer, int log_d,
                              int num_ntts, int tpb,
                              const F128* src = nullptr, long long smask = -1) {
    const size_t TOPK_SMEM_CAP = 32 * 1024;
    int lb = num_ntts;
    while ((size_t)(1LL << K) * (size_t)lb * sizeof(F128) > TOPK_SMEM_CAP && lb > 1)
        lb >>= 1;
    size_t smem = ((size_t)(1LL << K) * (size_t)lb + ((1u << K) - 1)) * sizeof(F128);
    cudaFuncSetAttribute(ntt_smem_topK_kernel<K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    long long tiles = (1LL << (log_d - K)) * (long long)(num_ntts / lb);
    ntt_smem_topK_kernel<K><<<(unsigned)tiles, tpb, smem>>>(d_data, d_tw, layer, log_d, num_ntts, lb,
                                                            src ? src : d_data, smask);
}

// Fused launches for layers [from, to). Each full-buffer pass costs one DRAM
// read+write, so we minimize PASS COUNT: split the layers into ceil(total/KMAX)
// balanced chunks (KMAX=7 — the deepest tile that still keeps smem ~32 KB at
// lb=16). Balanced (not greedy) avoids a trailing lone layer: 19 layers go
// 7+6+6 (3 passes) instead of 6+6+6+1 (4 passes). Chunks of 4/2/1 fall back to
// the register-fused / single-layer kernels (cheaper than a smem tile at small K).
inline void launch_top_fused(F128* d_data, const F128* d_tw, const TwiddleTable& tt,
                             int from, int to, int log_d, int num_ntts, int tpb,
                             const F128* src0 = nullptr, long long smask0 = -1) {
    int total = to - from;
    if (total <= 0) return;
    const int KMAX = 7;
    int npass = (total + KMAX - 1) / KMAX;
    int base = total / npass, extra = total % npass;
    int layer = from;
    for (int p = 0; p < npass; p++) {
        int c = base + (p < extra ? 1 : 0);     // this chunk's layer count
        // Rate-extend fusion: pass 0 may read from src0 (the pre-replication
        // message) via smask0 — topK chunks only (see ntt_can_fuse_src).
        const F128* src = (p == 0) ? src0 : nullptr;
        long long smask = (p == 0) ? smask0 : -1;
        long long total_bf, blocks;
        switch (c) {
            case 7: launch_topK_chunk<7>(d_data, d_tw, layer, log_d, num_ntts, tpb, src, smask); break;
            case 6: launch_topK_chunk<6>(d_data, d_tw, layer, log_d, num_ntts, tpb, src, smask); break;
            case 5: launch_topK_chunk<5>(d_data, d_tw, layer, log_d, num_ntts, tpb, src, smask); break;
            case 3: launch_topK_chunk<3>(d_data, d_tw, layer, log_d, num_ntts, tpb, src, smask); break;
            case 4:
                total_bf = (1LL << layer) * (1LL << (log_d - layer - 4)) * (long long)num_ntts;
                blocks = (total_bf + tpb - 1) / tpb;
                ntt_fused4_kernel<<<(unsigned)blocks, tpb>>>(
                    d_data, d_tw + tt.off[layer], d_tw + tt.off[layer + 1],
                    d_tw + tt.off[layer + 2], d_tw + tt.off[layer + 3], layer, log_d, num_ntts);
                break;
            case 2:
                total_bf = (1LL << layer) * (1LL << (log_d - layer - 2)) * (long long)num_ntts;
                blocks = (total_bf + tpb - 1) / tpb;
                ntt_fused2_kernel<<<(unsigned)blocks, tpb>>>(
                    d_data, d_tw + tt.off[layer], d_tw + tt.off[layer + 1], layer, log_d, num_ntts);
                break;
            default:  // c == 1
                total_bf = (1LL << (log_d - 1)) * (long long)num_ntts;
                blocks = (total_bf + tpb - 1) / tpb;
                ntt_layer_kernel<<<(unsigned)blocks, tpb>>>(
                    d_data, d_tw + tt.off[layer], layer, log_d, num_ntts);
                break;
        }
        layer += c;
    }
}

// Host launcher: forward interleaved NTT, layers [log_inv_rate, k_code).
// Pure fused 4/2/1 is the DEFAULT and the fastest path. `deep_smem=true` adds a
// shared-memory deep stage (top fused stage + one on-chip pass for the deepest
// `dt` layers) — it is a MEASURED REGRESSION (~25% slower at m=29) and kept only
// for reference: the deep layers have small strides and are already L2-resident
// under pure fusion, so explicit tiling just costs occupancy + barriers. Caller
// syncs. `d_tw`/`tt` come from build_twiddle_table (uploaded to device).
// True when the first fused pass is a smem-topK chunk (layer counts 3,5,6,7 of
// a balanced split — everything except totals 1/2/4, which use the register
// kernels without the src/smask load hook). Callers that want the rate-extend
// fusion (skip replicate_fill, first pass reads the message) must check this
// and fall back to replicate_fill + plain launch_ntt when false.
inline bool ntt_can_fuse_src(int total_layers) {
    return total_layers > 0 && total_layers != 1 && total_layers != 2 && total_layers != 4;
}

inline void launch_ntt(F128* d_data, const F128* d_tw, const TwiddleTable& tt,
                       int log_inv_rate, int k_code, int num_ntts,
                       int tpb = 256, bool deep_smem = false,
                       const F128* src0 = nullptr, long long smask0 = -1) {
    int log_d = k_code;
    int total_layers = k_code - log_inv_rate;
    if (total_layers <= 0) return;

    // Pick the deepest tile that fits the shared-mem budget (~100 KB), capped so
    // the top fused stage still has something to do and dt stays sane.
    const long long SMEM_BUDGET = 100 * 1024;
    int dt = 0;
    if (deep_smem) {
        while (dt + 1 <= total_layers && dt + 1 <= 12 &&
               ((1LL << (dt + 1)) * num_ntts * (long long)sizeof(F128)) <= SMEM_BUDGET) {
            dt++;
        }
    }
    if (dt < 1) {  // no useful tile — pure fused path
        launch_top_fused(d_data, d_tw, tt, log_inv_rate, k_code, log_d, num_ntts, tpb, src0, smask0);
        return;
    }

    int top_layers = k_code - dt;                       // deep stage = [top_layers, k_code)
    launch_top_fused(d_data, d_tw, tt, log_inv_rate, top_layers, log_d, num_ntts, tpb, src0, smask0);

    size_t smem = (size_t)(1LL << dt) * num_ntts * sizeof(F128);
    cudaFuncSetAttribute(ntt_deep_smem_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    long long tiles = 1LL << (log_d - dt);
    ntt_deep_smem_kernel<<<(unsigned)tiles, tpb, smem>>>(d_data, d_tw, top_layers, log_d, num_ntts, dt);
}
