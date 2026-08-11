//! Correctness of the composite Merkle-path R1CS (`r1cs_hashes::merkle_r1cs`).
//!
//! These tests materialize the composite matrices, which costs `depth` copies
//! of BLAKE3's ~21M-nonzero block (~170 MB per level), so they run at small
//! depth. They are the reference oracle for the depth-26 walker path.

use flock_core::field::F128;
use flock_core::lincheck::{CscCircuit, LincheckCircuit, build_quirky_eq_table};
use flock_core::zerocheck::K_SKIP;
use flock_prover::r1cs_hashes::merkle_r1cs::{
    MerkleTreeLayout, PathInput, SLOT_WORDS, blake3_spec, reference_root,
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

/// Geometry: each level occupies an aligned `2^14` subcube (the alignment the
/// walker's `eq` factorization needs), so `k_log = 14 + log2(next_pow2(depth))`
/// and the useful region ends after the last level's `t` block.
#[test]
fn layout_geometry() {
    let spec = blake3_spec();
    const U: usize = 11_825; // blake3 useful bits
    const STRIDE: usize = 1 << 14;
    for (depth, want_k_log) in [(1usize, 14usize), (2, 15), (3, 16), (26, 19)] {
        let layout = MerkleTreeLayout::new(depth, spec.clone());
        // Level 0 additionally carries const + leaf + index bits; every later
        // level ends after its 512 gadget columns.
        let globals_end = U + 512 + 1 + 256 + depth;
        let last_level_end = (depth - 1) * STRIDE + U + 512;
        assert_eq!(
            layout.useful_bits,
            globals_end.max(last_level_end),
            "depth {depth} useful_bits"
        );
        assert_eq!(layout.k_log, want_k_log, "depth {depth} k_log");
        assert!(layout.useful_bits <= layout.k());
        // Every level's subcube is exactly the base block, and levels are
        // aligned — this is what makes the level index a set of address bits.
        assert_eq!(layout.hash_bit(0, 0), 0);
        for l in 0..depth {
            assert_eq!(layout.hash_bit(l, 0), l * STRIDE, "level {l} not aligned");
        }
    }

    // The depth-26 experiment target. Padding each level to 2^14 costs 31%
    // more columns than the tightly-packed layout (321,045 → 421,937 — the
    // fraction grew when the lin-id drop narrowed the base block) and buys
    // the walker's 26× eq factorization; k_log is unchanged at 19.
    let l26 = MerkleTreeLayout::new(26, spec);
    assert_eq!(l26.useful_bits, 421_937);
    assert_eq!(l26.k_log, 19);
    assert_eq!(l26.useful_bits.div_ceil(128), 3_297);
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
        ("const wire", layout.const_pos()),
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
            assert_eq!(
                z,
                layout.build_witness(&input),
                "depth {depth} idx {index} z"
            );
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
    let paths: Vec<PathInput> = (0..n_paths)
        .map(|i| path(&mut rng, depth, i as u64 & 1))
        .collect();

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
        let pin = layout.const_pos();
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

/// Assert two combs are equal, naming the first offending column.
fn assert_comb_eq(got: &[F128], want: &[F128], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: comb length");
    if got != want {
        let bad = (0..got.len()).find(|&c| got[c] != want[c]).unwrap();
        panic!(
            "{ctx}: comb mismatch at column {bad} (of {}): {:?} vs {:?}",
            got.len(),
            got[bad],
            want[bad]
        );
    }
}

/// THE walker test: `fold_alpha_batched` must agree with the CSC circuit over
/// the fully materialized composite matrices, column for column, on random
/// `eq_inner`. This is what licenses using the walker at depth 26, where the
/// materialized form does not fit in memory.
///
/// A random table does not factor over the level subcubes (for `depth > 1`),
/// so this also pins the general `fold_per_level` fallback — both the
/// dispatching entry point and the fallback directly.
#[test]
fn walker_matches_materialized() {
    for depth in [1usize, 2, 3] {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let reference =
            CscCircuit::from_matrices(&a_0, &b_0).with_const_pin(Some(layout.const_pos()));
        let walker = layout.build_walker();

        assert_eq!(walker.n_cols(), reference.n_cols(), "depth {depth} n_cols");
        assert_eq!(
            walker.const_pin_col(),
            reference.const_pin_col(),
            "depth {depth} const_pin — the union asserts these match the TableType"
        );
        // The walker traverses exactly the materialized nonzero count.
        let nnz: usize = a_0
            .rows
            .iter()
            .chain(b_0.rows.iter())
            .map(|r| r.len())
            .sum();
        assert_eq!(walker.effective_nnz(), nnz, "depth {depth} nnz");

        let mut rng = Rng::new(0x_A17C_0057);
        for trial in 0..3 {
            let eq: Vec<F128> = (0..walker.n_cols()).map(|_| rng.f128()).collect();
            let alpha = rng.f128();
            // A random table is rank-1 across levels only when there is a
            // single level, where the claim is vacuous.
            assert_eq!(
                walker.eq_factors_over_levels(&eq),
                depth == 1,
                "depth {depth} trial {trial}: random eq factorability"
            );
            let want = reference.fold_alpha_batched(alpha, &eq);
            assert_comb_eq(
                &walker.fold_alpha_batched(alpha, &eq),
                &want,
                &format!("depth {depth} trial {trial} dispatched"),
            );
            assert_comb_eq(
                &walker.fold_per_level(alpha, &eq),
                &want,
                &format!("depth {depth} trial {trial} per-level"),
            );
        }
    }
}

/// THE packed-driver test: the batch-major driver must produce **bit-identical**
/// `(z, a, b, stripe)` to the original per-path `Vec<bool>` builder.
///
/// This is what licenses replacing the witness generator: the packed path
/// reuses BLAKE3's lane-parallel group builder per level and writes only the
/// swap gadget and globals itself, so a single misplaced bit offset would be
/// invisible to the R1CS-satisfaction tests at some depths but caught here at
/// every depth. Partial counts are covered too (dummy rows must stay
/// identically zero, const pin included — the union's lincheck target
/// depends on it).
#[test]
fn packed_driver_matches_bool_reference() {
    for (depth, nu, n_paths) in [
        (1usize, 3usize, 8usize),
        (2, 3, 8),
        (3, 3, 5),  // partial: 3 dummy rows
        (8, 3, 8),  // power-of-two depth: no wasted level slots
        (8, 4, 13), // partial, power-of-two depth
        (26, 3, 8),
        (26, 4, 11), // partial at a wider capacity
        (26, 3, 1),  // single declared path, 7 dummies
    ] {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        let mut rng = Rng::new(0x_9AC7_ED00 + (depth as u64) * 16 + nu as u64);
        // Vary the index across paths so both swap directions occur at every
        // level; a fixed index would leave one branch of the gadget untested.
        let paths: Vec<PathInput> = (0..n_paths)
            .map(|i| path(&mut rng, depth, (i as u64).wrapping_mul(0x9E37_79B9)))
            .collect();

        let (gz, ga, gb, gs) = layout.generate_witness_batch_major_partial(&paths, nu);
        let (wz, wa, wb, ws) = layout.generate_witness_batch_major_partial_bool(&paths, nu);

        let ctx = format!("depth {depth}, nu {nu}, {n_paths} paths");
        for (name, got, want) in [("z", &gz, &wz), ("a", &ga, &wa), ("b", &gb, &wb)] {
            assert_eq!(got.len(), want.len(), "{ctx}: {name} length");
            if got != want {
                let i = (0..got.len()).find(|&i| got[i] != want[i]).unwrap();
                panic!(
                    "{ctx}: {name} differs at word {i} of {}: packed {:?} vs bool {:?}",
                    got.len(),
                    got[i],
                    want[i]
                );
            }
        }
        assert_eq!(gs.len(), ws.len(), "{ctx}: stripe length");
        if gs != ws {
            let i = (0..gs.len()).find(|&i| gs[i] != ws[i]).unwrap();
            panic!(
                "{ctx}: stripe differs at byte {i} of {}: packed {:#04x} vs bool {:#04x}",
                gs.len(),
                gs[i],
                ws[i]
            );
        }
    }
}

/// A lincheck-shaped `eq_inner`: `build_quirky_eq_table` at a random point,
/// exactly what `lincheck::prove` / `verify_union` hand the circuit.
fn quirky_eq(rng: &mut Rng, k_log: usize) -> Vec<F128> {
    let x_inner_rest: Vec<F128> = (0..k_log - K_SKIP).map(|_| rng.f128()).collect();
    build_quirky_eq_table(rng.f128(), &x_inner_rest, K_SKIP)
}

/// The factorization, against the materialized oracle. On a *real* lincheck
/// table the walker takes the factored path — one base fold plus a multiply
/// per column — and it must still agree with the full composite matrices
/// column for column.
#[test]
fn walker_factored_matches_materialized() {
    for depth in [1usize, 2, 3] {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let reference =
            CscCircuit::from_matrices(&a_0, &b_0).with_const_pin(Some(layout.const_pos()));
        let walker = layout.build_walker();

        let mut rng = Rng::new(0x_FAC7_0000 + depth as u64);
        for trial in 0..3 {
            let eq = quirky_eq(&mut rng, layout.k_log);
            assert_eq!(eq.len(), walker.n_cols(), "depth {depth} eq length");
            let alpha = rng.f128();
            assert!(
                walker.eq_factors_over_levels(&eq),
                "depth {depth} trial {trial}: a quirky eq table MUST factor over \
                 the aligned level subcubes — if this fails the fast fold is \
                 silently never used"
            );
            let want = reference.fold_alpha_batched(alpha, &eq);
            assert_comb_eq(
                &walker.fold_alpha_batched(alpha, &eq),
                &want,
                &format!("depth {depth} trial {trial} factored vs materialized"),
            );
            assert_comb_eq(
                &walker.fold_per_level(alpha, &eq),
                &want,
                &format!("depth {depth} trial {trial} per-level vs materialized"),
            );
        }
    }
}

/// The factorization at the depth we ship, where no materialized reference
/// exists: the factored fold must equal the general per-level walk, which
/// `walker_factored_matches_materialized` ties to the real matrices at small
/// depth. Also reports the speedup, since that is the whole point.
#[test]
fn walker_factored_matches_per_level_at_depth26() {
    use std::time::Instant;

    let layout = MerkleTreeLayout::new(26, blake3_spec());
    let walker = layout.build_walker();
    let mut rng = Rng::new(0x_FAC7_001A);
    let eq = quirky_eq(&mut rng, layout.k_log);
    let alpha = rng.f128();
    assert!(
        walker.eq_factors_over_levels(&eq),
        "depth 26 quirky eq must factor"
    );

    // Time both on the full pool AND on one thread. The verifier runs its
    // PIOP replay inside a dedicated single-thread pool
    // (`flock_core::verifier`), so the 1-thread column is the one that
    // predicts verify time; the parallel column understates the win because
    // the per-level walk parallelizes better than the base fold.
    let solo = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let bench = |label: &str, pool: Option<&rayon::ThreadPool>| {
        let run = |f: &(dyn Fn() -> Vec<F128> + Sync)| match pool {
            Some(p) => p.install(f),
            None => f(),
        };
        let t = Instant::now();
        let fast = run(&|| walker.fold_alpha_batched(alpha, &eq));
        let t_fast = t.elapsed();
        let t = Instant::now();
        let slow = run(&|| walker.fold_per_level(alpha, &eq));
        let t_slow = t.elapsed();
        assert_comb_eq(
            &fast,
            &slow,
            &format!("depth 26 factored vs per-level ({label})"),
        );
        println!(
            "depth 26 fold, {label:>9}: factored {:>8.1} ms  per-level {:>8.1} ms  ({:.1}×)",
            t_fast.as_secs_f64() * 1e3,
            t_slow.as_secs_f64() * 1e3,
            t_slow.as_secs_f64() / t_fast.as_secs_f64(),
        );
    };
    println!("{} effective nonzeros", walker.effective_nnz());
    bench("parallel", None);
    bench("1 thread", Some(&solo));
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

// ---------------------------------------------------------------------------
// The chunk-leaf layout (BLAKE3 L0 openings: 1 KiB leaf + PARENT path)
// ---------------------------------------------------------------------------

use flock_core::merkle::{self as core_merkle, HashKind};
use flock_prover::r1cs_hashes::merkle_r1cs::ChunkPathInput;

fn chunk_input(rng: &mut Rng, depth: usize, leaf_bytes: usize, index: u128) -> ChunkPathInput {
    ChunkPathInput {
        leaf_data: (0..leaf_bytes).map(|_| rng.next_u32() as u8).collect(),
        index,
        siblings: (0..depth).map(|_| rng.digest()).collect(),
    }
}

fn digest_to_hash(d: &[u32; SLOT_WORDS]) -> [u8; 32] {
    let mut h = [0u8; 32];
    for (w, word) in d.iter().enumerate() {
        h[4 * w..4 * w + 4].copy_from_slice(&word.to_le_bytes());
    }
    h
}

fn hash_to_digest(h: &[u8; 32]) -> [u32; SLOT_WORDS] {
    std::array::from_fn(|w| u32::from_le_bytes(h[4 * w..4 * w + 4].try_into().unwrap()))
}

/// Geometry of the chunk-leaf layout: chunk blocks tile first, node levels
/// after; the globals shrink to const + index bits in chunk block 0's
/// padding; the L0 shape (1 KiB leaf, depth 13) lands on k_log 19 — the
/// same width as the depth-26 digest table.
#[test]
fn chunk_layout_geometry() {
    let spec = blake3_spec();
    const U: usize = 11_825;
    const STRIDE: usize = 1 << 14;
    for (depth, leaf_bytes, want_k_log) in
        [(1usize, 64usize, 15usize), (2, 128, 16), (13, 1024, 19)]
    {
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, spec.clone());
        let blocks = leaf_bytes / 64 + depth;
        assert_eq!(layout.leaf_blocks, leaf_bytes / 64);
        assert_eq!(layout.total_blocks(), blocks);
        assert_eq!(layout.k_log, want_k_log, "depth {depth} leaf {leaf_bytes}");
        assert_eq!(
            layout.useful_bits,
            (blocks - 1) * STRIDE + U + 512,
            "depth {depth} leaf {leaf_bytes} useful_bits"
        );
        assert_eq!(
            layout.const_pos(),
            U,
            "const sits at chunk block 0's padding start"
        );
        // The index is a word-aligned 128-bit WORD, not a tight run after the
        // constant — that is what makes it wireable, and it is why the
        // Fiat-Shamir query binding needs no mask gadget: the query index IS
        // the low `depth` bits of the challenge word.
        assert_eq!(layout.index_word_base() % 128, 0, "index word is aligned");
        assert!(
            layout.index_word_base() > U,
            "index word clears the constant-one column"
        );
        assert_eq!(layout.index_bit(0), layout.index_word_base());
        for l in 0..depth {
            assert_eq!(layout.index_bit(l), layout.index_word_base() + l);
        }
        assert!(
            layout.index_word_base() + 128 <= STRIDE,
            "the whole index word fits chunk block 0's padding"
        );
        // Node levels sit after the chunk segment.
        for l in 0..depth {
            assert_eq!(
                layout.hash_bit(l, 0),
                (leaf_bytes / 64 + l) * STRIDE,
                "level {l} not aligned past the chunk segment"
            );
        }
        // The leaf data enters through the chunk blocks' message regions.
        for i in 0..layout.leaf_blocks {
            assert_eq!(
                layout.leaf_data_bit(i, 0),
                i * STRIDE + layout.spec.msg_base,
                "chunk block {i} msg"
            );
        }
    }
}

/// An honest chunk-leaf opening satisfies the materialized composite, at
/// both swap directions, and the root column agrees with the native fold —
/// covering the chunk chain's in_cv copy rows (2 chunk blocks) and both
/// flag patterns.
#[test]
fn honest_chunk_openings_satisfy() {
    let (depth, leaf_bytes) = (1usize, 128usize); // 2 chunk blocks + 1 node level
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);
    let mut rng = Rng::new(0x_C4_09_77_31);

    for index in 0..(1u128 << depth) {
        let input = chunk_input(&mut rng, depth, leaf_bytes, index);
        let z = layout.build_witness_chunk(&input);
        assert!(
            r1cs.satisfies(&z),
            "honest chunk witness rejected at index {index}"
        );
        assert_eq!(
            layout.read_root(&z),
            layout.reference_root_chunk(&input),
            "root column disagrees with the native fold at index {index}"
        );

        // zab: emitted a/b must equal the matrix application, exactly.
        let [z2, a, b] = layout.build_witness_zab_chunk(&input);
        assert_eq!(z2, z, "zab z half");
        assert_eq!(a, r1cs.apply_a(&z), "zab a half");
        assert_eq!(b, r1cs.apply_b(&z), "zab b half");
    }

    // Tampering: leaf data, index bit, root, and a chunk-chain bit must all
    // be load-bearing.
    let input = chunk_input(&mut rng, depth, leaf_bytes, 1);
    let z0 = layout.build_witness_chunk(&input);
    assert!(r1cs.satisfies(&z0));
    for (name, col) in [
        ("leaf data bit", layout.leaf_data_bit(1, 7)),
        ("index bit", layout.index_bit(0)),
        (
            "root bit",
            layout.hash_bit(depth - 1, layout.spec.out_cv_base + 3),
        ),
        (
            "chunk chain cv bit",
            layout.hash_bit(0, layout.spec.in_cv_base + 11),
        ),
    ] {
        let mut z = z0.clone();
        z[col] ^= true;
        assert!(!r1cs.satisfies(&z), "{name} flip was accepted");
    }
}

/// **The production pin**: the chunk-leaf table must reproduce
/// `flock_core::merkle`'s BLAKE3 tree bit-for-bit — leaves as non-root
/// chunk chaining values of the raw 1 KiB leaf bytes, internal nodes as
/// non-root PARENT compressions — at the real L0 shape (depth 13, 1 KiB
/// leaves) and a small shape. Index convention: the table's bit `l` puts
/// the running digest LEFT, `flock_core` puts it left on an even node
/// index, so the table index is the complement of the tree position.
#[test]
fn chunk_root_matches_flock_core_blake3_tree() {
    let mut rng = Rng::new(0x_1F_5A_33_D7);
    for (depth, leaf_bytes) in [(3usize, 256usize), (13, 1024)] {
        let n_leaves = 1usize << depth;
        let data: Vec<u8> = (0..n_leaves * leaf_bytes)
            .map(|_| rng.next_u32() as u8)
            .collect();
        let tree = core_merkle::merkle_tree(&data, n_leaves, HashKind::Blake3);
        let root = tree[tree.len() - 1];

        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let positions: Vec<usize> = if depth <= 3 {
            (0..n_leaves).collect()
        } else {
            vec![0, 1, 4097, 5000, n_leaves - 1]
        };
        for pos in positions {
            // Siblings bottom-up out of the flat tree (leaves first, then
            // each level, root last).
            let mut siblings = Vec::with_capacity(depth);
            let mut seg = 0usize; // start of current level
            let mut width = n_leaves;
            let mut idx = pos;
            for _ in 0..depth {
                siblings.push(hash_to_digest(&tree[seg + (idx ^ 1)]));
                seg += width;
                width /= 2;
                idx >>= 1;
            }
            let input = ChunkPathInput {
                leaf_data: data[pos * leaf_bytes..(pos + 1) * leaf_bytes].to_vec(),
                // The table index IS the tree position now (no
                // complement) — that is what the polarity flip bought.
                index: pos as u128,
                siblings,
            };
            assert_eq!(
                digest_to_hash(&layout.reference_root_chunk(&input)),
                root,
                "depth {depth} pos {pos}: native fold vs flock_core tree"
            );
            let z = layout.build_witness_chunk(&input);
            assert_eq!(
                digest_to_hash(&layout.read_root(&z)),
                root,
                "depth {depth} pos {pos}: witness root vs flock_core tree"
            );
        }
        // And the leaf hash itself matches hash_leaf.
        let one = core_merkle::hash_leaf(&data[..leaf_bytes], HashKind::Blake3);
        let input = ChunkPathInput {
            leaf_data: data[..leaf_bytes].to_vec(),
            index: !0u128 & ((1u128 << depth) - 1),
            siblings: (0..depth).map(|_| rng.digest()).collect(),
        };
        let z = layout.build_witness_chunk(&input);
        let leaf_cv: [u32; SLOT_WORDS] = std::array::from_fn(|w| {
            let base = layout.hash_bit(0, 0) - (1 << 14) + layout.spec.out_cv_base; // last chunk block
            let mut word = 0u32;
            for b in 0..32 {
                if z[base + 32 * w + b] {
                    word |= 1 << b;
                }
            }
            word
        });
        assert_eq!(digest_to_hash(&leaf_cv), one, "chunk CV vs hash_leaf");
    }
}

/// The chunk packed driver must be bit-identical to the per-path bool
/// builder — including partial counts with zero dummy rows — at the real
/// L0 shape and small shapes.
#[test]
fn chunk_packed_driver_matches_bool_reference() {
    for (depth, leaf_bytes, nu, n_paths) in [
        (1usize, 64usize, 3usize, 5usize),
        (2, 128, 3, 8),
        (13, 1024, 3, 5),
    ] {
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let mut rng = Rng::new(0x_7E_00_D1_44 ^ (depth as u64) << 8 ^ leaf_bytes as u64);
        // Full index WORDS, high bits and all — the two drivers must agree on
        // every one of the 128 columns, not just the `depth` the relation reads.
        let paths: Vec<ChunkPathInput> = (0..n_paths)
            .map(|i| {
                let hi = (0..4).fold(0u128, |acc, _| (acc << 32) | rng.next_u32() as u128);
                chunk_input(&mut rng, depth, leaf_bytes, (hi << depth) | i as u128)
            })
            .collect();
        let packed = layout.generate_witness_batch_major_partial_chunk(&paths, nu);
        let reference = layout.generate_witness_batch_major_partial_bool_chunk(&paths, nu);
        assert_eq!(packed.0, reference.0, "z (depth {depth} leaf {leaf_bytes})");
        assert_eq!(packed.1, reference.1, "a (depth {depth} leaf {leaf_bytes})");
        assert_eq!(packed.2, reference.2, "b (depth {depth} leaf {leaf_bytes})");
        assert_eq!(
            packed.3, reference.3,
            "stripe (depth {depth} leaf {leaf_bytes})"
        );
    }
}

/// **What makes the index wireable**: the index column is a full 128-bit
/// word, and every bit at or above `depth` is genuinely free — the relation
/// reads only the low `depth`, so an opening whose index word carries
/// arbitrary high bits still satisfies, and still folds to the same root.
///
/// That is the whole mechanism behind binding a Merkle opening to a
/// Fiat–Shamir query without a masking gadget. `sample_queries` derives a
/// position as `challenge.lo & (block_len − 1)`; a circuit wires the challenge
/// WORD into this column, and the `& (block_len − 1)` is not computed anywhere
/// — it is expressed by which columns the relation reads. If the high bits
/// were pinned to zero (as they were before they became a word) the copy
/// constraint against a real challenge would be unsatisfiable.
///
/// The complement is checked too: the low `depth` bits are NOT free.
#[test]
fn index_word_high_bits_are_free() {
    let (depth, leaf_bytes) = (2usize, 128usize);
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let r1cs = layout.build_block_r1cs(0);

    for pos in 0..(1u128 << depth) {
        let mut rng = Rng::new(0x_B1_7E_5A_09 ^ pos as u64);
        let bare = chunk_input(&mut rng, depth, leaf_bytes, pos);

        // The same opening, with a full challenge word above the position.
        let hi = (0..4).fold(0u128, |acc, _| (acc << 32) | rng.next_u32() as u128);
        let dressed = ChunkPathInput {
            index: (hi << depth) | pos,
            ..bare.clone()
        };
        assert_ne!(dressed.index >> depth, 0, "test needs nonzero high bits");

        let (zb, zd) = (
            layout.build_witness_chunk(&bare),
            layout.build_witness_chunk(&dressed),
        );
        assert!(
            r1cs.satisfies(&zd),
            "a nonzero high half was rejected at position {pos}"
        );
        assert_eq!(
            layout.read_root(&zd),
            layout.reference_root_chunk(&dressed),
            "high bits perturbed the root at position {pos}"
        );
        assert_eq!(
            layout.read_root(&zd),
            layout.read_root(&zb),
            "high bits are read by the fold at position {pos}"
        );

        // The two witnesses differ in exactly the index word's high columns —
        // nothing else in the trace moved.
        let differ: Vec<usize> = (0..zb.len()).filter(|&c| zb[c] != zd[c]).collect();
        let want: Vec<usize> = (depth..128)
            .filter(|&j| (hi >> (j - depth)) & 1 == 1)
            .map(|j| layout.index_word_base() + j)
            .collect();
        assert_eq!(differ, want, "high bits leaked outside the index word");

        // ...and the low bits really are load-bearing.
        for l in 0..depth {
            let mut bad = zd.clone();
            bad[layout.index_bit(l)] ^= true;
            assert!(
                !r1cs.satisfies(&bad),
                "flipping index bit {l} was accepted at position {pos}"
            );
        }
    }
}

/// `root_chunk` — the allocation-free fold a circuit gate uses — must agree
/// with `reference_root_chunk`, which reaches the same digest by
/// materializing a full witness block per compression. The fast path skips
/// the witness entirely, so nothing else would catch it drifting.
///
/// Both swap directions at every position, plus full random index words, so
/// the high bits are confirmed not to reach the fold.
#[test]
fn fast_root_matches_the_witness_fold() {
    for (depth, leaf_bytes) in [(1usize, 64usize), (2, 128), (3, 256), (13, 1024)] {
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let mut rng = Rng::new(0x_FA_57_00_07u64.wrapping_mul(depth as u64 + 1));
        for pos in 0..(1u128 << depth).min(8) {
            let hi = (0..4).fold(0u128, |acc, _| (acc << 32) | rng.next_u32() as u128);
            let input = chunk_input(&mut rng, depth, leaf_bytes, (hi << depth) | pos);
            assert_eq!(
                layout.root_chunk(&input),
                layout.reference_root_chunk(&input),
                "depth {depth} leaf {leaf_bytes} pos {pos}: fast fold disagrees"
            );
        }
    }
}

/// The walker must agree with the materialized composite on the chunk-leaf
/// layout too — the chunk blocks embed the same stripped base, with their
/// pin/copy/free-message replacements in the extras.
#[test]
fn chunk_walker_matches_materialized() {
    for (depth, leaf_bytes) in [(1usize, 64usize), (1, 128)] {
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let reference =
            CscCircuit::from_matrices(&a_0, &b_0).with_const_pin(Some(layout.const_pos()));
        let walker = layout.build_walker();
        assert_eq!(walker.n_cols(), reference.n_cols());
        assert_eq!(walker.const_pin_col(), reference.const_pin_col());
        let nnz: usize = a_0
            .rows
            .iter()
            .chain(b_0.rows.iter())
            .map(|r| r.len())
            .sum();
        assert_eq!(walker.effective_nnz(), nnz, "leaf {leaf_bytes} nnz");

        let mut rng = Rng::new(0x_3C_71_88_2F);
        for trial in 0..2 {
            let eq: Vec<F128> = (0..walker.n_cols()).map(|_| rng.f128()).collect();
            let alpha = rng.f128();
            let want = reference.fold_alpha_batched(alpha, &eq);
            assert_comb_eq(
                &walker.fold_alpha_batched(alpha, &eq),
                &want,
                &format!("chunk leaf {leaf_bytes} trial {trial} dispatched"),
            );
            assert_comb_eq(
                &walker.fold_per_level(alpha, &eq),
                &want,
                &format!("chunk leaf {leaf_bytes} trial {trial} per-level"),
            );
        }
    }

    // The quirky eq table must still factor over the subcubes at the real
    // L0 shape — the walker's fast path at prove time.
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(13, 1024, blake3_spec());
    let walker = layout.build_walker();
    let mut rng = Rng::new(0x_66_11_09_AB);
    let m_inner = layout.k_log;
    let z_skip = rng.f128();
    let xs: Vec<F128> = (0..m_inner - K_SKIP).map(|_| rng.f128()).collect();
    let eq = build_quirky_eq_table(z_skip, &xs, K_SKIP);
    assert_eq!(eq.len(), walker.n_cols());
    assert!(
        walker.eq_factors_over_levels(&eq),
        "lincheck's table must factor over the 29 subcubes"
    );
}

// ---------------------------------------------------------------------------
// The chunk-only layout (leaf half of an L0 opening: no node levels)
// ---------------------------------------------------------------------------

fn chunk_only_input(rng: &mut Rng, leaf_words: usize) -> ChunkPathInput {
    ChunkPathInput {
        leaf_data: (0..16 * leaf_words).map(|_| rng.next_u32() as u8).collect(),
        index: 0,
        siblings: Vec::new(),
    }
}

/// The width shapes a chunk-only row can take (see
/// `docs/local/chunk-leaf-l0-table.md`): sub-block leaves, whole blocks,
/// mixed-union lane counts (arbitrary integers — partial final blocks), and
/// the full 16-block chunk.
const CHUNK_ONLY_WIDTHS: [usize; 8] = [1, 3, 4, 8, 25, 47, 61, 64];

/// Geometry of the chunk-only layout: chunk blocks tile the whole row, the
/// only global is the constant, the output is the LAST block's chaining
/// value, and the schema is `words` inputs + 2 outputs with NO index word.
#[test]
fn chunk_only_layout_geometry() {
    use flock_core::schedule::IoWord;
    let spec = blake3_spec();
    const U: usize = 11_825;
    const STRIDE: usize = 1 << 14;
    for words in CHUNK_ONLY_WIDTHS {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, spec.clone());
        let blocks = words.div_ceil(4);
        assert_eq!(layout.depth, 0, "{words} words");
        assert_eq!(layout.leaf_blocks, blocks, "{words} words");
        assert_eq!(layout.leaf_words, words, "{words} words");
        assert_eq!(layout.total_blocks(), blocks, "{words} words");
        assert_eq!(
            layout.k_log,
            14 + blocks.next_power_of_two().trailing_zeros() as usize,
            "{words} words k_log"
        );
        assert_eq!(
            layout.useful_bits,
            ((blocks - 1) * STRIDE + U).max(U + 1),
            "{words} words useful_bits"
        );
        assert_eq!(layout.const_pos(), U, "const at chunk block 0's padding");

        // The schema: the used leaf words in block order, then the chunk CV.
        let schema = layout.io_schema();
        assert_eq!(schema.len(), words + 2, "{words} words schema len");
        for w in 0..words {
            assert_eq!(
                schema[w],
                IoWord::input((w / 4) * (STRIDE / 128) + 4 + (w % 4)),
                "{words} words: leaf word {w}"
            );
            assert_eq!(layout.io_leaf(w / 4, w % 4), w, "{words} words io_leaf");
        }
        let out0 = (blocks - 1) * (STRIDE / 128) + 2;
        assert_eq!(schema[words], IoWord::output(out0), "{words} words out lo");
        assert_eq!(
            schema[words + 1],
            IoWord::output(out0 + 1),
            "{words} words out hi"
        );
        assert_eq!(layout.io_root(0), words, "{words} words io_root(0)");
        assert_eq!(layout.io_root(1), words + 1, "{words} words io_root(1)");
    }
}

/// **The production pin**: a chunk-only row's output IS
/// `flock_core::merkle::hash_leaf`'s BLAKE3 leaf — the non-root chunk
/// chaining value of the raw bytes — at every width shape, partial final
/// blocks included, through all three folds (fast root, reference root, and
/// the witness's root column).
#[test]
fn chunk_only_cv_matches_flock_core_hash_leaf() {
    let mut rng = Rng::new(0x_C4_11_F0_07);
    for words in CHUNK_ONLY_WIDTHS {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, blake3_spec());
        let input = chunk_only_input(&mut rng, words);
        let want = core_merkle::hash_leaf(&input.leaf_data, HashKind::Blake3);
        assert_eq!(
            digest_to_hash(&layout.root_chunk(&input)),
            want,
            "{words} words: fast root vs hash_leaf"
        );
        assert_eq!(
            digest_to_hash(&layout.reference_root_chunk(&input)),
            want,
            "{words} words: reference root vs hash_leaf"
        );
        let z = layout.build_witness_chunk(&input);
        assert_eq!(
            digest_to_hash(&layout.read_root(&z)),
            want,
            "{words} words: witness root vs hash_leaf"
        );
    }
}

/// An honest chunk-only witness satisfies the materialized composite; the
/// emitted `(a, b)` equal the matrix application exactly; and the leaf data,
/// the output chaining value, the zero-pinned message tail of a partial
/// final block, and the absent index word's columns are all load-bearing.
#[test]
fn honest_chunk_only_rows_satisfy() {
    // 4 = one full block; 5 = two blocks with a partial final block (the
    // cv-copy row AND the zero-pinned tail in one shape); 61 = the recorded
    // mixed-union lane count (16 blocks, 16-byte final block).
    for words in [4usize, 5, 61] {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, blake3_spec());
        let r1cs = layout.build_block_r1cs(0);
        let mut rng = Rng::new(0x_5E_AF_00_1D ^ words as u64);
        let input = chunk_only_input(&mut rng, words);

        let z = layout.build_witness_chunk(&input);
        assert!(r1cs.satisfies(&z), "{words} words: honest witness rejected");

        let [z2, a, b] = layout.build_witness_zab_chunk(&input);
        assert_eq!(z2, z, "{words} words zab z half");
        assert_eq!(a, r1cs.apply_a(&z), "{words} words zab a half");
        assert_eq!(b, r1cs.apply_b(&z), "{words} words zab b half");

        let mut cols = vec![
            ("leaf data bit", layout.leaf_data_bit(0, 7)),
            ("chunk cv bit", layout.root_bit(3)),
            // No index word exists on this layout: its would-be columns are
            // padding, forced zero by empty rows.
            ("absent index word bit", layout.index_word_base()),
        ];
        if words % 4 != 0 {
            cols.push((
                "zero-pinned message tail bit",
                layout.leaf_data_bit(words / 4, 128 * (words % 4) + 5),
            ));
        }
        for (name, col) in cols {
            let mut zt = z.clone();
            zt[col] ^= true;
            assert!(!r1cs.satisfies(&zt), "{words} words: {name} flip accepted");
        }
    }
}

/// The chunk-only walker must agree with the CSC circuit over the fully
/// materialized matrices, column for column — same license as
/// [`walker_matches_materialized`], at 1-, 2- and 4-block widths including
/// a partial final block.
#[test]
fn chunk_only_walker_matches_materialized() {
    for words in [4usize, 5, 13] {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let reference =
            CscCircuit::from_matrices(&a_0, &b_0).with_const_pin(Some(layout.const_pos()));
        let walker = layout.build_walker();

        assert_eq!(walker.n_cols(), reference.n_cols(), "{words} words n_cols");
        assert_eq!(
            walker.const_pin_col(),
            reference.const_pin_col(),
            "{words} words const_pin"
        );
        let nnz: usize = a_0
            .rows
            .iter()
            .chain(b_0.rows.iter())
            .map(|r| r.len())
            .sum();
        assert_eq!(walker.effective_nnz(), nnz, "{words} words nnz");

        let mut rng = Rng::new(0x_C0_11_A9_5E ^ words as u64);
        for trial in 0..2 {
            let eq: Vec<F128> = (0..walker.n_cols()).map(|_| rng.f128()).collect();
            let alpha = rng.f128();
            let want = reference.fold_alpha_batched(alpha, &eq);
            assert_comb_eq(
                &walker.fold_alpha_batched(alpha, &eq),
                &want,
                &format!("{words} words trial {trial} dispatched"),
            );
            assert_comb_eq(
                &walker.fold_per_level(alpha, &eq),
                &want,
                &format!("{words} words trial {trial} per-level"),
            );
        }
    }

    // Lincheck's quirky eq table must factor over the block subcubes at a
    // multi-block width — the walker's fast path at prove time.
    let layout = MerkleTreeLayout::with_blake3_chunk_only(61, blake3_spec());
    let walker = layout.build_walker();
    let mut rng = Rng::new(0x_66_11_09_AC);
    let z_skip = rng.f128();
    let xs: Vec<F128> = (0..layout.k_log - K_SKIP).map(|_| rng.f128()).collect();
    let eq = build_quirky_eq_table(z_skip, &xs, K_SKIP);
    assert_eq!(eq.len(), walker.n_cols());
    assert!(
        walker.eq_factors_over_levels(&eq),
        "lincheck's table must factor over the 16 subcubes"
    );
}

/// The chunk-only packed driver must be bit-identical to the per-path bool
/// builder — partial row counts and partial final blocks included.
#[test]
fn chunk_only_packed_driver_matches_bool_reference() {
    for (words, nu, n_paths) in [(1usize, 3usize, 5usize), (5, 3, 8), (61, 3, 5)] {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, blake3_spec());
        let mut rng = Rng::new(0x_7E_00_D1_45 ^ (words as u64) << 8);
        let paths: Vec<ChunkPathInput> =
            (0..n_paths).map(|_| chunk_only_input(&mut rng, words)).collect();
        let packed = layout.generate_witness_batch_major_partial_chunk(&paths, nu);
        let reference = layout.generate_witness_batch_major_partial_bool_chunk(&paths, nu);
        assert_eq!(packed.0, reference.0, "z ({words} words)");
        assert_eq!(packed.1, reference.1, "a ({words} words)");
        assert_eq!(packed.2, reference.2, "b ({words} words)");
        assert_eq!(packed.3, reference.3, "stripe ({words} words)");
    }
}

/// The walked [`ChunkFoldMatrix`] must agree with the materialized
/// composite in BOTH marginal directions, on factoring (eq-tensor) and
/// non-factoring (random) weights alike, and discharge the same bilinear
/// claims — the license for handing stub-matrix chunk types to the
/// matrix-claim fold and the root discharge.
#[test]
fn chunk_only_fold_matrices_match_materialized() {
    use flock_core::matrix_fold::{FoldMatrix, MatrixClaim, Weight};
    for words in [4usize, 5, 13] {
        let layout = MerkleTreeLayout::with_blake3_chunk_only(words, blake3_spec());
        let (a_0, b_0) = layout.build_matrices();
        let (fa, fb) = layout.fold_matrices();
        let k = 1usize << layout.k_log;
        assert_eq!(fa.n_rows(), k, "{words} words n_rows");
        assert_eq!(fa.n_cols(), k, "{words} words n_cols");

        let mut rng = Rng::new(0x_F0_1D_00_07 ^ words as u64);
        // A rank-1 eq tensor (factors across subcubes) and a random vector
        // (does not, at >1 block) — both must agree with the sparse form.
        let eq_pt: Vec<F128> = (0..layout.k_log).map(|_| rng.f128()).collect();
        let eq_w = Weight::eq(eq_pt).materialize();
        let rand_w: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
        for (m, f, name) in [(&a_0, &fa, "A"), (&b_0, &fb, "B")] {
            for (w, wname) in [(&eq_w, "eq"), (&rand_w, "random")] {
                assert_eq!(
                    f.row_marginal(w, k),
                    m.row_marginal(w, k),
                    "{words} words {name} row_marginal ({wname})"
                );
                assert_eq!(
                    f.col_marginal(w, k),
                    m.col_marginal(w, k),
                    "{words} words {name} col_marginal ({wname})"
                );
            }
            // An honest claim about the sparse form discharges against the
            // walked form, and a perturbed one does not.
            let row = Weight::eq((0..layout.k_log).map(|_| rng.f128()).collect::<Vec<_>>());
            let col = Weight::eq((0..layout.k_log).map(|_| rng.f128()).collect::<Vec<_>>());
            let claim = MatrixClaim::honest(row.clone(), col.clone(), m);
            assert!(
                claim.check_direct(f),
                "{words} words {name}: honest claim discharges walked"
            );
            let bad = MatrixClaim {
                value: claim.value + F128::ONE,
                ..claim
            };
            assert!(
                !bad.check_direct(f),
                "{words} words {name}: perturbed claim rejected"
            );
        }
    }
}
