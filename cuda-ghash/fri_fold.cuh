// FRI fold kernel — step 1 of the GPU pcs::open (Ligerito) port
// (GPU_OPEN_PLAN.md). Direct port of the per-round FRI fold in
//   src/pcs/basefold.rs::fri_fold_codeword  +  fold_pair  (DP24)
// which the prover runs `log_dim` times, halving the single-lane codeword each
// round at `layer = k_code - round - 1` (basefold.rs:604).
//
// One thread per OUTPUT position. fold_pair (basefold.rs:192):
//   v = v_in + u_in;  u = u_in + v · twiddle;  result = u + r · (u + v)
// Adds are XOR (`f128_add`); the two multiplies use `ghash_mul_karatsuba` from
// f128.cuh — the fastest multiply on this GPU (bench_f128). The twiddle is the
// SAME standard-basis schedule as the forward NTT: XOR of layer `layer`'s span
// basis at the set bits of the output index (mirrors ntt_f128.cuh's inline and
// AdditiveNttF128::twiddle, whose doc notes it is "for the forward NTT and FRI
// fold"). Twiddle table comes from ntt_host.hpp's build_twiddle_table(k_code).
#pragma once
#include "f128.cuh"
#include "ntt_host.hpp"   // TwiddleTable / build_twiddle_table; F128,u64 from f128.cuh

// One FRI fold round: out[i] = fold_pair(twiddle(layer,i), in[2i], in[2i+1], r),
// for i in [0, new_len). `tw_basis` points at layer `layer`'s span basis
// (d_tw + tt.off[layer]); it has `layer` entries.
__global__ void fri_fold_round(const F128* __restrict__ in,
                               F128* __restrict__ out,
                               const F128* __restrict__ tw_basis,
                               int layer, F128 r, long long new_len) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= new_len) return;

    F128 u_in = in[2 * i];
    F128 v_in = in[2 * i + 1];

    // twiddle(layer, i) = XOR of span-basis elements at set bits of i.
    F128 tw{0ull, 0ull};
    for (int j = 0; j < layer; j++) {
        if ((i >> j) & 1ull) tw = f128_add(tw, tw_basis[j]);
    }

    F128 v = f128_add(v_in, u_in);
    F128 u = f128_add(u_in, ghash_mul_karatsuba(v, tw));
    out[i] = f128_add(u, ghash_mul_karatsuba(r, f128_add(u, v)));
}

// Host launcher for one round. `d_tw` is the full flattened twiddle table on
// device; `layer`'s basis starts at d_tw + tt.off[layer].
inline void launch_fri_fold(const F128* d_in, F128* d_out, const F128* d_tw,
                            const TwiddleTable& tt, int layer, F128 r,
                            long long new_len, int tpb = 256) {
    const F128* tw_basis = d_tw + tt.off[layer];
    long long blocks = (new_len + tpb - 1) / tpb;
    fri_fold_round<<<(unsigned)blocks, tpb>>>(d_in, d_out, tw_basis, layer, r, new_len);
}
