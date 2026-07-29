//! Correctness of the composite Merkle-path R1CS (`r1cs_hashes::merkle_r1cs`).
//!
//! These tests materialize the composite matrices, which costs `depth` copies
//! of BLAKE3's ~21M-nonzero block (~170 MB per level), so they run at small
//! depth. They are the reference oracle for the depth-26 walker path.

use flock_core::field::F128;
use flock_core::lincheck::{CscCircuit, LincheckCircuit};
use flock_prover::r1cs_hashes::merkle_r1cs::{
    blake3_spec, reference_root, MerkleTreeLayout, PathInput, SLOT_WORDS,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn digest(&mut self) -> [u32; SLOT_WORDS] {
        std::array::from_fn(|_| self.next_u32())
    }
}

fn path(rng: &mut Rng, depth: usize, index: u64) -> PathInput {
    PathInput {
        leaf: rng.digest(),
        index,
        siblings: (0..depth).map(|_| rng.digest()).collect(),
    }
}

/// Geometry: the layout is `depth` hash blocks plus the per-level gadget
/// columns, and it lands on the `k_log` the density analysis predicted.
#[test]
fn layout_geometry() {
    let spec = blake3_spec();
    // 1 const + 256 leaf + depth index bits + depth · (512 + 15,409).
    for (depth, want_useful, want_k_log) in [
        (1usize, 257 + 1 + 15_921, 14usize),
        (2, 257 + 2 + 2 * 15_921, 15),
        (26, 257 + 26 + 26 * 15_921, 19),
    ] {
        let layout = MerkleTreeLayout::new(depth, spec.clone());
        assert_eq!(layout.useful_bits, want_useful, "depth {depth} useful_bits");
        assert_eq!(layout.k_log, want_k_log, "depth {depth} k_log");
        assert!(layout.useful_bits <= layout.k());
    }
    // The depth-26 experiment target: 79% column utilization, 3,237 packed
    // chunk-columns per path (of the 4,096 the padded block would hold).
    let l26 = MerkleTreeLayout::new(26, spec);
    assert_eq!(l26.useful_bits, 414_229);
    assert_eq!(l26.k_log, 19);
    assert_eq!(l26.useful_bits.div_ceil(128), 3_237);
}

/// The composite R1CS accepts an honest path, at every index pattern that
/// exercises both swap directions, and the witness's root column matches an
/// independent native fold.
#[test]
fn honest_paths_satisfy() {
    let depth = 2;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0x4D33_2B1E);

    for index in 0..(1u64 << depth) {
        let input = path(&mut rng, depth, index);
        let z = layout.build_witness(&input);
        assert!(
            r1cs.satisfies(&z),
            "honest witness rejected at index {index}"
        );
        assert_eq!(
            layout.read_root(&z),
            reference_root(&layout.spec, &input),
            "root column disagrees with the native fold at index {index}"
        );
    }
}

/// Depth 3 — the chaining is exercised twice, so a mis-wired `prev` region
/// that survives depth 2 shows up here.
#[test]
#[ignore] // ~510 MB of matrices; run with `-- --ignored`.
fn honest_paths_satisfy_depth3() {
    let depth = 3;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0x0D3E_9743);
    for index in 0..(1u64 << depth) {
        let input = path(&mut rng, depth, index);
        let z = layout.build_witness(&input);
        assert!(r1cs.satisfies(&z), "depth-3 witness rejected at {index}");
        assert_eq!(layout.read_root(&z), reference_root(&layout.spec, &input));
    }
}

/// The depth-26 experiment target. Witness generation needs no matrices, so
/// this runs at the real depth: it pins the layout, checks the witness fills
/// exactly the useful prefix (padding stays zero — the union's compaction
/// invariant), and cross-checks the root column against a native fold.
#[test]
fn depth26_witness() {
    let depth = 26;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    assert_eq!(layout.k_log, 19);
    let mut rng = Rng::new(0x1A_26_00_5F);

    for index in [0u64, 1, 0x2AA_AAAA, 0x155_5555, (1 << depth) - 1] {
        let input = path(&mut rng, depth, index);
        let z = layout.build_witness(&input);
        assert_eq!(z.len(), 1 << 19);
        assert_eq!(
            layout.read_root(&z),
            reference_root(&layout.spec, &input),
            "depth-26 root disagrees with the native fold at index {index:#x}"
        );
        assert!(
            z[layout.useful_bits..].iter().all(|&b| !b),
            "padding is not zero at index {index:#x}"
        );
        // The running digest really does thread through every level: level
        // l's `prev` region must equal level l−1's output region.
        for l in 1..depth {
            for j in 0..256 {
                assert_eq!(
                    z[layout.prev_bit(l, j)],
                    z[layout.hash_bit(l - 1, layout.spec.out_cv_base + j)],
                    "level {l} prev bit {j} is not aliased to level {} out",
                    l - 1
                );
            }
        }
    }
}

/// Every column the statement rests on is genuinely constrained: flipping a
/// leaf bit, an index bit, a sibling bit, a swap-gadget AND, or a root bit
/// must break the system.
#[test]
fn tampering_is_rejected() {
    let depth = 2;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0x7A_4D_93_11);
    let input = path(&mut rng, depth, 0b10);
    let z = layout.build_witness(&input);
    assert!(r1cs.satisfies(&z));

    let cases: Vec<(&str, usize)> = vec![
        ("leaf bit 0", layout.leaf_bit(0)),
        ("leaf bit 137", layout.leaf_bit(137)),
        ("index bit 0", layout.index_bit(0)),
        ("index bit 1", layout.index_bit(1)),
        ("sibling L0 bit 5", layout.sibling_bit(0, 5)),
        ("sibling L1 bit 200", layout.sibling_bit(1, 200)),
        ("root bit 0", layout.root_bit(0)),
        ("root bit 255", layout.root_bit(255)),
        ("const wire", MerkleTreeLayout::CONST_POS),
    ];
    for (what, col) in cases {
        let mut bad = z.clone();
        bad[col] = !bad[col];
        assert!(
            !r1cs.satisfies(&bad),
            "flipping {what} (column {col}) was NOT rejected"
        );
    }
}

/// A path that is internally consistent but hashes a *swapped* pair at a
/// level whose index bit says otherwise must be rejected — this is the swap
/// gadget's whole job.
#[test]
fn wrong_swap_direction_is_rejected() {
    let depth = 2;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0x5A_4B_00_71);
    let input = path(&mut rng, depth, 0b00);

    // Honest witness for index 0b00, then relabel the index bits to 0b11
    // without redoing the hashing: the gadget rows now demand the other
    // swap, so the message region no longer matches.
    let mut z = layout.build_witness(&input);
    assert!(r1cs.satisfies(&z));
    z[layout.index_bit(0)] = true;
    z[layout.index_bit(1)] = true;
    assert!(
        !r1cs.satisfies(&z),
        "relabelling the index bits without reswapping was accepted"
    );
}

// ---------------------------------------------------------------------------
// The row-witness (a, b) emission
// ---------------------------------------------------------------------------

/// THE `a`/`b` test: the emitted row-witness must equal honest matrix
/// application, `a = A_0·z` and `b = B_0·z`. This is what licenses emitting
/// them directly at depth 26, where applying the matrices is not an option.
#[test]
fn emitted_ab_matches_matrix_application() {
    for depth in [1usize, 2] {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        let r1cs = layout.build_block_r1cs(0);
        let mut rng = Rng::new(0x_AB_0F_11_23);
        for index in 0..(1u64 << depth) {
            let input = path(&mut rng, depth, index);
            let [z, a, b] = layout.build_witness_zab(&input);
            // The z half must agree with the plain witness builder.
            assert_eq!(z, layout.build_witness(&input), "depth {depth} idx {index} z");
            let want_a = r1cs.apply_a(&z);
            let want_b = r1cs.apply_b(&z);
            for (name, got, want) in [("a", &a, &want_a), ("b", &b, &want_b)] {
                if got != want {
                    let c = (0..got.len()).find(|&c| got[c] != want[c]).unwrap();
                    panic!(
                        "depth {depth} idx {index}: {name}[{c}] = {} but A/B·z gives {}",
                        got[c], want[c]
                    );
                }
            }
            // And the system is satisfied: a ⊙ b = z.
            assert!(r1cs.satisfies(&z));
        }
    }
}

/// The BatchMajor scatter and the stripe must place bits where the union
/// expects, and dummy rows must be identically zero in all four outputs —
/// const-pin bit included, which is the union's count-check contract.
#[test]
fn batch_major_slot_layout_and_padding() {
    let depth = 1;
    let nu = 3;
    let layout = MerkleTreeLayout::new(depth, blake3_spec());
    let k = layout.k();
    let n_total = 1usize << nu;
    let n_paths = 5; // deliberately partial: rows 5..8 are dummies
    let mut rng = Rng::new(0x_B4_7C_44_01);
    let paths: Vec<PathInput> = (0..n_paths).map(|i| path(&mut rng, depth, i as u64 & 1)).collect();

    let (z, a, b, stripe) = layout.generate_witness_batch_major_partial(&paths, nu);
    let words_per_block = k / 128;
    assert_eq!(z.len(), n_total * words_per_block);
    assert_eq!(stripe.len(), (n_total / 8) * k);

    // Declared rows: the packed words must match the per-path row-witness at
    // the BatchMajor address (w << nu) + i, and the stripe must carry z.
    for (i, p) in paths.iter().enumerate() {
        let [pz, pa, pb] = layout.build_witness_zab(p);
        for w in 0..words_per_block {
            let addr = (w << nu) + i;
            for (buf, bits, name) in [(&z, &pz, "z"), (&a, &pa, "a"), (&b, &pb, "b")] {
                for t in 0..128 {
                    let got = if t < 64 {
                        (buf[addr].lo >> t) & 1 == 1
                    } else {
                        (buf[addr].hi >> (t - 64)) & 1 == 1
                    };
                    assert_eq!(got, bits[w * 128 + t], "{name} row {i} bit {}", w * 128 + t);
                }
            }
        }
        for c in 0..layout.useful_bits {
            let got = (stripe[(i / 8) * k + c] >> (i % 8)) & 1 == 1;
            assert_eq!(got, pz[c], "stripe row {i} col {c}");
        }
    }

    // Dummy rows 5..8: zero in z, a, b AND the stripe, pin included.
    for i in n_paths..n_total {
        for w in 0..words_per_block {
            let addr = (w << nu) + i;
            assert_eq!(z[addr], F128::ZERO, "dummy row {i} z word {w}");
            assert_eq!(a[addr], F128::ZERO, "dummy row {i} a word {w}");
            assert_eq!(b[addr], F128::ZERO, "dummy row {i} b word {w}");
        }
        let pin = MerkleTreeLayout::CONST_POS;
        assert_eq!(
            (stripe[(i / 8) * k + pin] >> (i % 8)) & 1,
            0,
            "dummy row {i} carries the const pin — the union lincheck rejects this"
        );
    }
}

// ---------------------------------------------------------------------------
// The walker circuit
// ---------------------------------------------------------------------------

impl Rng {
    fn f128(&mut self) -> F128 {
        F128 {
            lo: ((self.next_u32() as u64) << 32) | self.next_u32() as u64,
            hi: ((self.next_u32() as u64) << 32) | self.next_u32() as u64,
        }
    }
}

/// THE walker test: `fold_alpha_batched` must agree with the CSC circuit over
/// the fully materialized composite matrices, column for column, on random
/// `eq_inner`. This is what licenses using the walker at depth 26, where the
/// materialized form does not fit in memory.
#[test]
fn walker_matches_materialized() {
    for depth in [1usize, 2, 3] {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let reference = CscCircuit::from_matrices(&a_0, &b_0)
            .with_const_pin(Some(MerkleTreeLayout::CONST_POS));
        let walker = layout.build_walker();

        assert_eq!(walker.n_cols(), reference.n_cols(), "depth {depth} n_cols");
        assert_eq!(
            walker.const_pin_col(),
            reference.const_pin_col(),
            "depth {depth} const_pin — the union asserts these match the TableType"
        );
        // The walker traverses exactly the materialized nonzero count.
        let nnz: usize = a_0.rows.iter().chain(b_0.rows.iter()).map(|r| r.len()).sum();
        assert_eq!(walker.effective_nnz(), nnz, "depth {depth} nnz");

        let mut rng = Rng::new(0x_A17C_0057);
        for trial in 0..3 {
            let eq: Vec<F128> = (0..walker.n_cols()).map(|_| rng.f128()).collect();
            let alpha = rng.f128();
            let got = walker.fold_alpha_batched(alpha, &eq);
            let want = reference.fold_alpha_batched(alpha, &eq);
            assert_eq!(got.len(), want.len());
            if got != want {
                let bad = (0..got.len()).find(|&c| got[c] != want[c]).unwrap();
                panic!(
                    "depth {depth} trial {trial}: comb mismatch at column {bad} \
                     (of {}): walker {:?} vs materialized {:?}",
                    got.len(),
                    got[bad],
                    want[bad]
                );
            }
        }
    }
}

/// The walker is the memory story: at the real depth it must stay small,
/// while representing the full ~547M-nonzero system.
#[test]
fn walker_is_compact_at_depth26() {
    let layout = MerkleTreeLayout::new(26, blake3_spec());
    let walker = layout.build_walker();
    let resident = walker.resident_bytes();
    let materialized = walker.effective_nnz() * std::mem::size_of::<usize>();
    println!(
        "depth 26: walker resident {:.1} MB, effective nnz {} ({:.2} GB materialized), \
         ratio {:.0}x",
        resident as f64 / 1e6,
        walker.effective_nnz(),
        materialized as f64 / 1e9,
        materialized as f64 / resident as f64
    );
    assert!(
        resident < 200_000_000,
        "walker resident {resident} bytes should stay well under 200 MB"
    );
    assert!(
        walker.effective_nnz() > 500_000_000,
        "should still represent >500M nonzeros"
    );
}

/// Padding columns are forced to zero (empty rows), so a nonzero padding bit
/// must be rejected — the union's dense-stack compaction depends on it.
#[test]
fn padding_must_be_zero() {
    let layout = MerkleTreeLayout::new(1, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0xF4_DD_01_29);
    let input = path(&mut rng, 1, 0);
    let mut z = layout.build_witness(&input);
    assert!(r1cs.satisfies(&z));
    assert!(layout.useful_bits < layout.k(), "this depth has padding");
    z[layout.useful_bits] = true;
    assert!(!r1cs.satisfies(&z), "nonzero padding column was accepted");
}
