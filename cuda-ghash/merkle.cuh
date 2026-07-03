// GPU Merkle tree (SHA-256) for the PCS commit — P3 of GPU_COMMIT_PLAN.
//
// Mirrors src/merkle.rs::merkle_tree exactly:
//   * flat layout: tree[0..n] = leaf hashes, then each level above, root last;
//     total 2n-1 nodes.
//   * leaf i = SHA256(codeword bytes [i*leaf_size, (i+1)*leaf_size)) where
//     leaf_size = num_ntts*16 (one codeword position's lanes, no domain sep).
//   * parent = SHA256(left || right), 64-byte preimage.
//
// Leaf hashing dominates (each leaf is leaf_size/64 + 1 compressions vs 2 per
// node), and all leaves / all nodes within a level are independent — one thread
// per leaf, one thread per node, one kernel launch per level (levels are
// sequential, ordered by the stream).
#pragma once
#include <cstdint>
#include "sha256.cuh"

typedef uint8_t Hash[32];

// One thread per leaf: hash leaf_size contiguous bytes into tree[leaf].
__global__ void merkle_leaf_kernel(const uint8_t* data, uint8_t* tree,
                                   long long num_leaves, int leaf_size) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= num_leaves) return;
    sha256(data + i * (long long)leaf_size, (uint32_t)leaf_size, tree + i * 32);
}

// K leaves per thread, interleaved (ILP-hidden SHA dependency chain).
template <int K>
__global__ void merkle_leaf_kernel_kway(const uint8_t* data, uint8_t* tree,
                                        long long num_leaves, int leaf_size) {
    long long grp = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long first = grp * K;
    if (first >= num_leaves) return;
    if (first + K <= num_leaves) {
        sha256_kway<K>(data + first * (long long)leaf_size, leaf_size, (uint32_t)leaf_size,
                       tree + first * 32, 32);
    } else {
        for (long long i = first; i < num_leaves; i++)
            sha256(data + i * (long long)leaf_size, (uint32_t)leaf_size, tree + i * 32);
    }
}

// One thread per parent: hash the 64-byte child pair into the parent node.
// read_start/write_start are node indices into the flat tree.
__global__ void merkle_level_kernel(uint8_t* tree, long long read_start,
                                    long long write_start, long long num_parents) {
    long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= num_parents) return;
    const uint8_t* children = tree + (read_start + 2 * j) * 32;   // 2 contiguous 32-byte hashes
    sha256(children, 64, tree + (write_start + j) * 32);
}

// Build the full Merkle tree in d_tree (must hold 2*num_leaves-1 Hash nodes)
// over d_data (num_leaves * leaf_size bytes). Caller syncs; root is the last
// node, d_tree + (2*num_leaves-2)*32.
// `kway` = leaves hashed per thread (1, 2, or 4); interleaves the SHA chains
// for ILP. kway=1 is the simple one-thread-per-leaf path.
inline void launch_merkle(const uint8_t* d_data, uint8_t* d_tree,
                          long long num_leaves, int leaf_size, int tpb = 256, int kway = 1) {
    if (kway == 2) {
        long long groups = (num_leaves + 1) / 2;
        long long b = (groups + tpb - 1) / tpb;
        merkle_leaf_kernel_kway<2><<<(unsigned)b, tpb>>>(d_data, d_tree, num_leaves, leaf_size);
    } else if (kway == 4) {
        long long groups = (num_leaves + 3) / 4;
        long long b = (groups + tpb - 1) / tpb;
        merkle_leaf_kernel_kway<4><<<(unsigned)b, tpb>>>(d_data, d_tree, num_leaves, leaf_size);
    } else {
        long long blocks = (num_leaves + tpb - 1) / tpb;
        merkle_leaf_kernel<<<(unsigned)blocks, tpb>>>(d_data, d_tree, num_leaves, leaf_size);
    }

    long long read_start = 0, read_len = num_leaves;
    while (read_len > 1) {
        long long next = read_len >> 1;
        long long write_start = read_start + read_len;
        long long b = (next + tpb - 1) / tpb;
        merkle_level_kernel<<<(unsigned)b, tpb>>>(d_tree, read_start, write_start, next);
        read_start += read_len;
        read_len = next;
    }
}
