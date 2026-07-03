// Host-side Merkle query opening — port of src/merkle.rs::merkle_multi_proof
// (+ the opened-rows gather), step 6 of the GPU pcs::open / Ligerito port
// (GPU_OPEN_PLAN.md). This is host proof-assembly, NOT a GPU compute kernel: it
// reads sibling hashes from the resident Merkle tree (built by the commit-port
// kernel) for the challenger-sampled query positions.
//
// merkle_multi_proof: deduplicated batched opening. Bottom-up, per level, for
// each active position emit its sibling hash UNLESS the sibling is also active
// (then both fold into the same parent, no hash needed). Output order is the
// canonical sorted-by-position traversal. Byte-identical to the Rust.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
#include <algorithm>

struct MHash { uint8_t b[32]; };
inline bool mhash_eq(const MHash& a, const MHash& c) { return memcmp(a.b, c.b, 32) == 0; }

// Flat tree-node INDICES the multi-proof emits, in canonical order — depends
// only on `positions` (no hash values), so it can be computed without the tree
// and used to gather just those nodes from a device-resident tree.
inline std::vector<size_t> merkle_multi_proof_indices(size_t num_leaves,
                                                      const std::vector<size_t>& positions) {
    std::vector<size_t> idxs;
    if (positions.empty() || num_leaves == 1) return idxs;
    std::vector<size_t> active(positions.begin(), positions.end());
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    size_t level_start = 0, level_len = num_leaves;
    while (level_len > 1) {
        std::vector<size_t> next;
        next.reserve(active.size());
        size_t i = 0;
        while (i < active.size()) {
            size_t p = active[i];
            bool sib_active = (i + 1 < active.size()) && (active[i + 1] == (p ^ 1));
            if (sib_active) { i += 2; }
            else { idxs.push_back(level_start + (p ^ 1)); i += 1; }
            next.push_back(p >> 1);
        }
        active.swap(next);
        level_start += level_len;
        level_len >>= 1;
    }
    return idxs;
}

// Multi-proof sibling hashes for `positions` against the tree's root (host tree).
inline std::vector<MHash> merkle_multi_proof_host(const MHash* tree, size_t num_leaves,
                                                  const std::vector<size_t>& positions) {
    std::vector<size_t> idxs = merkle_multi_proof_indices(num_leaves, positions);
    std::vector<MHash> proof(idxs.size());
    for (size_t i = 0; i < idxs.size(); i++) proof[i] = tree[idxs[i]];
    return proof;
}

// Sorted-unique positions (the opened_rows / multi-proof alignment).
inline std::vector<size_t> sorted_unique(const std::vector<size_t>& positions) {
    std::vector<size_t> v(positions.begin(), positions.end());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}
