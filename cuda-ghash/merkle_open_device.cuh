// Device-resident Merkle multi-proof — perf path for the Ligerito query opening.
// Instead of copying the whole tree to host (64MB at log_n=24), compute the
// emitted node-index list on host (positions-only, cheap) and gather just those
// ~q·d sibling hashes from the device tree (~tens of KB). Byte-identical to
// merkle_multi_proof_host.
#pragma once
#include <vector>
#include "merkle_open.hpp"   // MHash, merkle_multi_proof_indices

__global__ void gather_tree_nodes(const uint8_t* __restrict__ tree,
                                  const unsigned long long* __restrict__ idxs,
                                  int n, uint8_t* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* src = tree + idxs[i] * 32;
    uint8_t* dst = out + (size_t)i * 32;
    #pragma unroll
    for (int j = 0; j < 32; j++) dst[j] = src[j];
}

// Multi-proof over a device-resident tree `d_tree`. No full-tree D2H.
inline std::vector<MHash> merkle_multi_proof_device(const uint8_t* d_tree, size_t num_leaves,
                                                    const std::vector<size_t>& positions) {
    std::vector<size_t> idxs = merkle_multi_proof_indices(num_leaves, positions);
    int n = (int)idxs.size();
    std::vector<MHash> out(n);
    if (n == 0) return out;
    // Pooled (grow-only) device scratch + pinned host staging — the per-call
    // cudaMalloc/cudaFree was the whole cost of this tiny gather. Pinned H/D
    // staging makes the two small copies fast.
    static unsigned long long* d_idx = nullptr;
    static uint8_t* d_out = nullptr;
    static unsigned long long* h_idx = nullptr;   // pinned
    static uint8_t* h_out = nullptr;               // pinned
    static int cap = 0;
    if (n > cap) {
        if (d_idx) { cudaFree(d_idx); cudaFree(d_out); cudaFreeHost(h_idx); cudaFreeHost(h_out); }
        cap = n + (n >> 1);  // headroom to avoid frequent regrow
        (void)cudaMalloc(&d_idx, (size_t)cap * sizeof(unsigned long long));
        (void)cudaMalloc(&d_out, (size_t)cap * 32);
        (void)cudaHostAlloc(&h_idx, (size_t)cap * sizeof(unsigned long long), cudaHostAllocDefault);
        (void)cudaHostAlloc(&h_out, (size_t)cap * 32, cudaHostAllocDefault);
    }
    for (int i = 0; i < n; i++) h_idx[i] = (unsigned long long)idxs[i];
    (void)cudaMemcpy(d_idx, h_idx, (size_t)n * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    int tpb = 128;
    gather_tree_nodes<<<(n + tpb - 1) / tpb, tpb>>>(d_tree, d_idx, n, d_out);
    (void)cudaMemcpy(h_out, d_out, (size_t)n * 32, cudaMemcpyDeviceToHost);
    memcpy(out.data(), h_out, (size_t)n * 32);
    return out;
}
