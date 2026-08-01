//! **MVP-3: PCS query openings as circuit gates, and the arithmetic on them.**
//!
//! A recursive verifier's dominant cost is checking PCS query openings — 218
//! depth-13 Merkle paths over 1 KiB leaves, per the L0 measurement. This file
//! makes one such opening a [`GateType`], so a circuit can instantiate as many
//! as it needs and the builder produces the wiring and the witness together.
//!
//! Three things are being validated, and they are the three that were open:
//!
//! - **The sibling path travels as a [`GateType::Hint`].** It is not a schema
//!   word and never can be: a sibling sits at
//!   [`sibling_bit`](MerkleTreeLayout::sibling_bit), inside its level's
//!   base-block padding at an unaligned offset, and a wire carries a whole
//!   128-bit word. It needs no wire — no other gate reads it, and the relation
//!   binds it through the root the schema does export.
//!
//! - **The index word is wireable, and is wired to a hash output.** This is
//!   the Fiat–Shamir query binding in miniature: a BLAKE3 gate produces a
//!   challenge word, that word is connected to the opening's index, and the
//!   masking `sample_queries` does natively (`& (block_len − 1)`) costs no
//!   gadget at all — it is which columns the Merkle relation reads. The word
//!   alignment this needs is why the index moved.
//!
//! - **A boolean slot with a hint proves and verifies** against the real
//!   union prover, over the walker lincheck circuit rather than materialized
//!   matrices.
//!
//! MVP-3b adds the other half — `LeafEvalGate`, an ELEMENT gate consuming the
//! same leaf words to compute `Σ_i α_i · ⟨row_i, eq(v, ·)⟩`, the `enforced_sum`
//! the Ligerito verifier checks. The 64 leaf words are one wire class each with
//! cells in both slots, so the copy constraint is what makes "the leaf that is
//! in the tree" and "the leaf the arithmetic ran on" the same leaf — across the
//! boolean/element class boundary.
//!
//! Small shapes where the geometry allows; `l0_shape_circuit_cost` runs the
//! real one (218 openings, depth 13, 1 KiB leaves).

use flock_core::circuit::builder::{CircuitBuilder, GateType, ShapeBuilder, SlotWitness, Wire};
use flock_core::field::F128;
use flock_core::merkle::{self as core_merkle, HashKind};
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::verifier;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::merkle_r1cs::{
    ChunkPathInput, MerkleTreeLayout, SLOT_WORDS, blake3_spec,
};
use flock_prover::schedule::TableType;
use flock_prover::union::UnionInstance;

const DOMAIN: &[u8] = b"flock-circuit-merkle-v0";

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;

const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

fn pack4(w: [u32; 4]) -> F128 {
    F128::new(
        w[0] as u64 | ((w[1] as u64) << 32),
        w[2] as u64 | ((w[3] as u64) << 32),
    )
}

fn unpack4(v: F128) -> [u32; 4] {
    [
        v.lo as u32,
        (v.lo >> 32) as u32,
        v.hi as u32,
        (v.hi >> 32) as u32,
    ]
}

fn pack8(w: &[u32; 8]) -> [F128; 2] {
    [
        pack4([w[0], w[1], w[2], w[3]]),
        pack4([w[4], w[5], w[6], w[7]]),
    ]
}

fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}

fn unpack_params(v: F128) -> (u64, u32, u32) {
    (v.lo, v.hi as u32, (v.hi >> 32) as u32)
}

/// A 128-bit word of leaf data: bytes `[o, o+16)` little-endian, which is
/// exactly how the message region reads them (`leaf_msg_words` is LE `u32`s,
/// and committed bit `t` of a word is bit `t` of `lo`).
fn leaf_word(data: &[u8], o: usize) -> F128 {
    F128::new(
        u64::from_le_bytes(data[o..o + 8].try_into().unwrap()),
        u64::from_le_bytes(data[o + 8..o + 16].try_into().unwrap()),
    )
}

fn digest_words(d: &[u32; SLOT_WORDS]) -> [F128; 2] {
    [
        pack4([d[0], d[1], d[2], d[3]]),
        pack4([d[4], d[5], d[6], d[7]]),
    ]
}

fn hash_to_digest(h: &[u8; 32]) -> [u32; SLOT_WORDS] {
    std::array::from_fn(|w| u32::from_le_bytes(h[4 * w..4 * w + 4].try_into().unwrap()))
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

/// One BLAKE3 compression, the challenge source. (Same gate as
/// `circuit_builder.rs`; duplicated rather than shared because these are
/// separate test binaries.)
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = blake3::Compression;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&blake3::build_block_r1cs(self.nu))
            .with_io_schema(blake3::io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let cv: [u32; 8] = {
            let (a, b) = (unpack4(inputs[0]), unpack4(inputs[1]));
            [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        };
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = unpack_params(inputs[6]);
        let out = blake3::blake3_compress(&cv, &m, counter, block_len, flags);
        let lo = pack8(&out[0..8].try_into().unwrap());
        let hi = pack8(&out[8..16].try_into().unwrap());
        (
            vec![lo[0], lo[1], hi[0], hi[1]],
            (cv, m, counter, block_len, flags),
        )
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// One chunk-leaf Merkle opening: leaf data and an index word in, the root
/// out, the sibling path as a hint.
struct MerklePathGate {
    layout: MerkleTreeLayout,
    nu: usize,
}

impl MerklePathGate {
    /// `block_len` is the PCS block length the opening's index will be
    /// sampled against — `sample_queries` masks a challenge with
    /// `block_len − 1`, and the relation reads the index word's low `depth`
    /// bits, so the two agree only when `depth = log2(block_len)`. Asserting
    /// it here means a circuit cannot silently wire a challenge that the
    /// relation truncates differently than the sampler did.
    fn new(depth: usize, leaf_bytes: usize, nu: usize, block_len: usize) -> Self {
        assert!(
            block_len.is_power_of_two() && block_len.trailing_zeros() as usize == depth,
            "tree depth {depth} does not match block_len {block_len}: the index \
             word's low {depth} bits are not the query sample_queries picked"
        );
        Self {
            layout: MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec()),
            nu,
        }
    }
}

impl GateType for MerklePathGate {
    type Row = ChunkPathInput;
    /// The sibling path, level 0 closest to the leaf. Unwireable by
    /// construction — see the module docs.
    type Hint = Vec<[u32; SLOT_WORDS]>;

    fn table(&self) -> TableType {
        // Stub matrices: the constraints live in the walker, as on the
        // production path. The digest binds `k_log`, `useful_bits` and the
        // const pin — hence the depth — and the verifier builds the matching
        // walker out of band.
        TableType::from_block_r1cs(&self.layout.build_block_r1cs_stub(self.nu))
            .with_io_schema(self.layout.io_schema())
    }

    fn eval(&self, inputs: &[F128], hint: &Self::Hint) -> (Vec<F128>, Self::Row) {
        assert_eq!(hint.len(), self.layout.depth, "one sibling per level");
        // Schema In-order: 4 words per chunk block, then the index word.
        let mut leaf_data = Vec::with_capacity(64 * self.layout.leaf_blocks);
        for w in &inputs[..4 * self.layout.leaf_blocks] {
            leaf_data.extend_from_slice(&w.lo.to_le_bytes());
            leaf_data.extend_from_slice(&w.hi.to_le_bytes());
        }
        let index_word = inputs[4 * self.layout.leaf_blocks];
        let row = ChunkPathInput {
            leaf_data,
            // The WHOLE word. The low `depth` bits are the position; the rest
            // ride along committed, pinned by the copy constraint, and read by
            // no row of the relation.
            index: (index_word.lo as u128) | ((index_word.hi as u128) << 64),
            siblings: hint.clone(),
        };
        let root = self.layout.root_chunk(&row);
        (digest_words(&root).to_vec(), row)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

// ---------------------------------------------------------------------------

/// A tree, and one opening's siblings out of it.
struct Tree {
    data: Vec<u8>,
    flat: Vec<[u8; 32]>,
    root: [u8; 32],
    depth: usize,
    leaf_bytes: usize,
}

impl Tree {
    fn new(depth: usize, leaf_bytes: usize, rng: &mut Rng) -> Self {
        let n_leaves = 1usize << depth;
        let data: Vec<u8> = (0..n_leaves * leaf_bytes)
            .map(|_| rng.next_u32() as u8)
            .collect();
        let flat = core_merkle::merkle_tree(&data, n_leaves, HashKind::Blake3);
        let root = flat[flat.len() - 1];
        Self {
            data,
            flat,
            root,
            depth,
            leaf_bytes,
        }
    }

    fn leaf(&self, pos: usize) -> &[u8] {
        &self.data[pos * self.leaf_bytes..(pos + 1) * self.leaf_bytes]
    }

    fn siblings(&self, pos: usize) -> Vec<[u32; SLOT_WORDS]> {
        let mut out = Vec::with_capacity(self.depth);
        let (mut seg, mut width, mut idx) = (0usize, 1usize << self.depth, pos);
        for _ in 0..self.depth {
            out.push(hash_to_digest(&self.flat[seg + (idx ^ 1)]));
            seg += width;
            width /= 2;
            idx >>= 1;
        }
        out
    }
}

/// The table's index bit `l` means "the running digest goes LEFT at level
/// `l`", while `flock_core::merkle` puts it left when the node index is even.
/// So the table index is the complement of the tree position.
fn table_index(pos: usize, depth: usize) -> u128 {
    !(pos as u128) & ((1u128 << depth) - 1)
}

/// **The gate, standalone**: the builder's rows reproduce the openings, the
/// computed roots are the real tree root, and the sibling path reached `eval`
/// as a hint without ever being a wire.
#[test]
fn merkle_openings_through_the_builder() {
    let (depth, leaf_bytes, nu) = (2usize, 128usize, 6usize);
    let mut rng = Rng(0x_3E_11_5E_ED);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));

    // Every opening's inputs first, then every root — `public_value` and
    // `publish` share one public vector, so keeping the publishes last is what
    // makes the roots the trailing `2n` entries.
    let positions: Vec<usize> = (0..1usize << depth).collect();
    let roots: Vec<Vec<Wire>> = positions
        .iter()
        .map(|&pos| {
            let leaf = tree.leaf(pos);
            let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
                .map(|w| b.public_value(leaf_word(leaf, 16 * w)))
                .collect();
            inputs.push(b.public_value(F128::new(table_index(pos, depth) as u64, 0)));
            b.gate_with_hint(g, &inputs, tree.siblings(pos))
        })
        .collect();
    for root in &roots {
        b.publish(root[0]);
        b.publish(root[1]);
    }

    let built = b.finish().expect("builder produces a valid circuit");
    assert_eq!(built.shape.counts, vec![positions.len()]);

    // Every opening's row is the opening we asked for, and every published
    // root is the tree's.
    let rows = built.rows::<MerklePathGate>(g);
    assert_eq!(rows.len(), positions.len());
    let want = digest_words(&hash_to_digest(&tree.root));
    for (i, &pos) in positions.iter().enumerate() {
        assert_eq!(rows[i].leaf_data, tree.leaf(pos), "row {i} leaf");
        assert_eq!(rows[i].index, table_index(pos, depth), "row {i} index");
        assert_eq!(rows[i].siblings, tree.siblings(pos), "row {i} siblings");
    }
    let published = &built.witness.public[built.witness.public.len() - 2 * positions.len()..];
    for i in 0..positions.len() {
        assert_eq!(
            [published[2 * i], published[2 * i + 1]],
            want,
            "opening {i} did not fold to the tree root"
        );
    }
}

/// **The Fiat–Shamir query binding.** A BLAKE3 gate emits a challenge word;
/// that word is wired straight into a Merkle opening's index. Nothing masks
/// it — the relation reads the low `depth` columns and the other 115 ride
/// along, pinned by the copy constraint and read by nothing.
///
/// Then the whole thing proves and verifies: two slot types, one boolean
/// each, joined by a wire that crosses from a hash output to a Merkle input.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn merkle_index_wired_to_a_challenge() {
    let (depth, leaf_bytes) = (2usize, 128usize);
    // The Merkle composite is 4 base blocks ⇒ k_log = 16, so nu = 6 puts the
    // union at the M = 22 Ligerito floor.
    let nu = 6usize;
    let mut rng = Rng(0x_C0FF_EE01);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let merkle = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));

    // One compression over a public message: its `out_lo0` is the challenge.
    let iv = pack8(&IV);
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let mut hash_in = vec![b.public_value(iv[0]), b.public_value(iv[1])];
    for j in 0..4 {
        hash_in.push(b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())));
    }
    hash_in.push(b.public_value(pack_params(0, 64, CHUNK_START | CHUNK_END)));
    let out = b.gate(hash, &hash_in);

    // What the challenge word selects, natively — the table reads its low
    // `depth` bits as the index, whose complement is the tree position.
    let compressed = blake3::blake3_compress(&IV, &m, 0, 64, CHUNK_START | CHUNK_END);
    let challenge = pack8(&compressed[0..8].try_into().unwrap())[0];
    let index = (challenge.lo as u128) | ((challenge.hi as u128) << 64);
    let pos = (!index & ((1u128 << depth) - 1)) as usize;
    assert_ne!(index >> depth, 0, "the challenge must have high bits set");

    // The opening: leaf data public, index NOT public — it comes off the wire.
    let leaf = tree.leaf(pos);
    let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
        .map(|w| b.public_value(leaf_word(leaf, 16 * w)))
        .collect();
    inputs.push(out[0]); // ← the binding: challenge word IS the index word
    let root = b.gate_with_hint(merkle, &inputs, tree.siblings(pos));
    b.publish(root[0]);
    b.publish(root[1]);

    let built = b.finish().expect("builder produces a valid circuit");

    // The row really did receive the full challenge word, not a masked copy —
    // and it arrived by WIRE: the public segment is the 7 compression inputs,
    // the 8 leaf words and the 2 published root halves, with no index among
    // them. Nothing but the copy constraint put it there.
    assert_eq!(built.witness.public.len(), 7 + leaf_bytes / 16 + 2);
    let rows = built.rows::<MerklePathGate>(merkle);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].index, index,
        "the whole challenge word reached the row"
    );
    let published = &built.witness.public[built.witness.public.len() - 2..];
    assert_eq!(
        [published[0], published[1]],
        digest_words(&hash_to_digest(&tree.root)),
        "the wired query did not fold to the tree root"
    );

    // ---- prove / verify ----
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };

    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();

    // Slot order is the registry's, not the declaration's.
    let (hash_slot, merkle_slot) = (built.registry_slot(hash), built.registry_slot(merkle));
    let mut slots: Vec<(usize, UnionSlotProverInput)> = vec![
        (
            hash_slot,
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu),
                blake_lc,
            ),
        ),
        (
            merkle_slot,
            UnionSlotProverInput::new(
                layout.generate_witness_batch_major_partial_chunk(rows, nu),
                &walker,
            ),
        ),
    ];
    slots.sort_by_key(|(i, _)| *i);
    let mut lcs: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> =
        vec![(hash_slot, blake_lc), (merkle_slot, &walker)];
    lcs.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs.into_iter().map(|(_, c)| c).collect();

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        slots.into_iter().map(|(_, s)| s).collect(),
        Vec::new(),
        &mut ch,
    );

    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("a challenge-derived Merkle opening verifies");

    // A wrong claimed root breaks the wiring — the opening is doing work.
    let mut bad = built.witness.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_jagged_union_circuit(
            &union,
            &built.shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered root must be rejected"
    );
}

/// **The real recursion shape, timed**: 218 depth-13 openings over 1 KiB
/// leaves — the L0 workload a Ligerito verifier at dense m = 25 checks — as a
/// circuit rather than a bare table.
///
/// The point of the measurement is the delta against `merkle_l0_opening`,
/// which proves the identical 218 rows with no wiring at all. Everything the
/// circuit adds is the copy-constraint layer: `product_gkr` over the cell
/// space, whose size is `2^(nu + c)` with `c = ceil(log2(67 schema words))`,
/// so mu = 8 + 7 = 15 here. That is tiny next to the k_log-19 slot, and the
/// numbers should say so.
#[test]
#[ignore] // The real shape: ~8 MiB tree, minutes of proving. `-- --ignored`.
fn l0_shape_circuit_cost() {
    use std::time::Instant;

    // Pin rayon to the physical P-cores, as `merkle_l0_opening` does. Without
    // it the default pool spreads across efficiency cores and `prove` swings
    // 53-91 ms run to run, which swamps everything being measured.
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    let (depth, leaf_bytes, n_paths) = (13usize, 1024usize, 218usize);
    let nu = 8usize; // 218 rows ⇒ capacity 256
    let mut rng = Rng(0x_10_5A_4E_11);
    let t = Instant::now();
    let tree = Tree::new(depth, leaf_bytes, &mut rng);
    let tree_ms = t.elapsed().as_secs_f64() * 1e3;

    // How expensive is just materializing the TableType at k_log 19? It
    // carries a 2^19 identity `c_0`, and `finish` handles one per slot.
    std::hint::black_box(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth).table());
    let t = Instant::now(); // second call: the first pays cold-allocator faults
    std::hint::black_box(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth).table());
    let table_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- SETUP: no values, no field arithmetic, paid once ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let g = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
    let roots: Vec<Vec<Wire>> = (0..n_paths)
        .map(|_| {
            // Leaf data is free witness here — bound by the root, not pinned.
            // In the full verifier it is wired to the arithmetic gate instead.
            let mut ins: Vec<Wire> = (0..leaf_bytes / 16).map(|_| sb.input()).collect();
            ins.push(sb.public_input());
            sb.gate_hinted(g, &ins)
        })
        .collect();
    for root in &roots {
        sb.publish(root[0]);
        sb.publish(root[1]);
    }
    let shape = sb.finish().expect("builder produces a valid circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- ONLINE: this proof's values ----
    let positions: Vec<usize> = (0..n_paths).map(|i| (i * 37 + 11) % (1 << depth)).collect();
    let gather = || {
        let mut vals = Vec::with_capacity(shape.num_inputs());
        let mut hints = Vec::with_capacity(n_paths);
        for &pos in &positions {
            let leaf = tree.leaf(pos);
            vals.extend((0..leaf_bytes / 16).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(F128::new(table_index(pos, depth) as u64, 0));
            hints.push(tree.siblings(pos));
        }
        (vals, hints)
    };
    let (vals, hints) = gather();
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();

    std::hint::black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();
    let rows = built.rows::<MerklePathGate>(g);

    let witgen = || layout.generate_witness_batch_major_partial_chunk(rows, nu);
    let prove = |witness| {
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_jagged_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(witness, &walker)],
            Vec::new(),
            &mut ch,
        )
    };

    // WARM-UP. A first prove in a fresh process pays lazy initialization —
    // twiddle tables, `OnceLock` caches, thread-pool spin-up, cold allocator —
    // and came out 20-40% high. `merkle_l0_opening` reports a median after
    // warm-up, so timing a cold single shot against it compares two different
    // statistics. Discard one round, then measure.
    std::hint::black_box(prove(witgen()));

    let t = Instant::now();
    let witness = witgen();
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;

    // How much of `build` is the gate re-executing BLAKE3 natively?
    let t = Instant::now();
    for row in rows {
        std::hint::black_box(layout.root_chunk(row));
    }
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let (proof, commitment, _) = prove(witness);
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![&walker];
    // No thread-pool wrapper: `flock_core::verifier` pins its own 1-thread
    // `verifier_pool` around the verify cores, and the bench calls verify on
    // the default pool exactly like this. Wrapping would change the regime,
    // not match it.
    let mut ch = FsChallenger::new(DOMAIN);
    let t = Instant::now();
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the L0-shape circuit verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    let proof_kib = bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0;

    // Split by what recurs. `circuit_structure_does_not_depend_on_the_witness`
    // licenses the setup line: the statement is identical every proof, so
    // `ShapeBuilder::finish` is paid once and `CircuitShape::run` per proof.
    let per_proof = online_ms + wit_ms + prove_ms;
    println!(
        "\nL0 shape as a CIRCUIT: {n_paths} openings, depth {depth}, {leaf_bytes} B leaves\n\
           k_log {}  nu {nu}  dense_m {}  public {}  wires {}  proof {proof_kib:.1} KiB  \
         {threads} threads\n\
         \n\
           PER PROOF     {per_proof:6.0} ms = online {online_ms:.0} (of which eval \
         {eval_ms:.0}) + witgen {wit_ms:.0} + prove {prove_ms:.0}\n\
           verifier side {verify_ms:6.1} ms\n\
           SETUP         {setup_ms:6.0} ms = shape + finish, of which TableType \
         {table_ms:.0}   (tree gen {tree_ms:.0} ms is the test's own fixture)\n\
         \n\
        compare `cargo bench --bench merkle_l0_opening`: 48 ms prove for the \
         same rows unwired.\n",
        layout.k_log,
        union.dense_m(),
        built.public.len(),
        shape.circuit.wires().len(),
    );
}

/// **The circuit structure is witness-independent** — which is what makes
/// `finish`'s cost amortizable across proofs.
///
/// Two circuits over different trees, opening different positions with
/// different siblings, must produce the same `Circuit::digest`: the statement
/// binds the registry, the cell space and sigma, none of which depend on a
/// value. Only the witness and the public values differ.
///
/// The consequence for the timing in `l0_shape_circuit_cost`: `finish`'s work
/// could be done once per recursion shape and reused, while the gate phase —
/// `eval` computing each root — is genuinely per-proof, because that IS the
/// witness.
#[test]
fn circuit_structure_does_not_depend_on_the_witness() {
    let (depth, leaf_bytes, nu) = (2usize, 128usize, 6usize);

    let build = |seed: u64, shift: usize| {
        let mut rng = Rng(seed);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        let mut b = CircuitBuilder::new(nu);
        let g = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
        let roots: Vec<Vec<Wire>> = (0..1usize << depth)
            .map(|i| {
                let pos = (i + shift) % (1 << depth);
                let leaf = tree.leaf(pos);
                let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
                    .map(|w| b.value(leaf_word(leaf, 16 * w)))
                    .collect();
                inputs.push(b.public_value(F128::new(table_index(pos, depth) as u64, 0)));
                b.gate_with_hint(g, &inputs, tree.siblings(pos))
            })
            .collect();
        for root in &roots {
            b.publish(root[0]);
            b.publish(root[1]);
        }
        (b.finish().expect("valid circuit"), g)
    };

    let (a, ga) = build(0x_A1_11_00_01, 0);
    let (c, gc) = build(0x_B2_22_00_02, 3);

    assert_eq!(
        a.shape.circuit.digest(),
        c.shape.circuit.digest(),
        "the statement moved when only the witness did"
    );
    assert_eq!(a.shape.counts, c.shape.counts);
    assert_eq!(a.witness.public.len(), c.witness.public.len());
    // ...and the witnesses really are different, so the check is not vacuous.
    let (ra, rc) = (a.rows::<MerklePathGate>(ga), c.rows::<MerklePathGate>(gc));
    assert_ne!(ra[0].leaf_data, rc[0].leaf_data, "same leaf data");
    assert_ne!(a.witness.public, c.witness.public, "same public values");
}

// ---------------------------------------------------------------------------
// MVP-3b: the leaf arithmetic
// ---------------------------------------------------------------------------

/// What the verifier actually computes from the opened leaves, at one level:
///
/// ```text
/// enforced_sum = Σ_i  α_i · ⟨row_i, eq(v_challenges, ·)⟩
/// ```
///
/// (`ligerito::induce_sumcheck_enforced_sum`.) The inner product against an
/// `eq` table IS the multilinear evaluation of the leaf's 64 lanes at the
/// 6-dimensional point `v`, so this gate evaluates it by folding, one variable
/// per level, and folds the α-weighted result into a running accumulator so
/// the whole level's sum falls out of the last row.
///
/// Layout (`kappa = 8`, 256 columns, 200 real):
///
/// ```text
///   0  .. 64   leaf lanes         (In — the SAME wires the Merkle gate reads)
///  64  .. 70   v challenges       (In)
///  70          alpha_i            (In)
///  71          prev accumulator   (In)
///  72  ..198   fold tree          (2 columns per node: d, then the folded value)
/// 198          alpha_i · y
/// 199          accumulator out    (Out)
/// ```
///
/// **The fold, and why 2 columns per node.** `build_eq_table` is LSB-first, so
/// variable `j` is bit `j` of the lane index and folding pairs `(2i, 2i+1)`:
///
/// ```text
///   new[i] = (1+v)·f[2i] + v·f[2i+1] = f[2i] + v·(f[2i] + f[2i+1])
/// ```
///
/// A row's left-hand side is a product of two linear forms, so `v·(f+f)` is
/// one `mult_lin` — the addition rides the multiplication's `A_0` row for
/// free — but the trailing `+ f[2i]` is outside the product and costs a
/// `linear` row of its own. Hence 2 rows per node, 126 for the 63 nodes.
///
/// That is not the floor. Materializing only `d[i] = v·(f[2i]+f[2i+1])` and
/// leaving `new[i]` as the *linear form* `f[2i] + d[i]` would let the next
/// level's `mult_lin` absorb it, giving 1 row per node — at the price of `A_0`
/// rows that grow to 127 terms at the last level. Left for later on purpose:
/// this is the MVP.
struct LeafEvalGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

const LE_LANES: usize = 64;
const LE_VARS: usize = 6;
const LE_V: usize = LE_LANES;
const LE_ALPHA: usize = LE_V + LE_VARS;
const LE_PREV: usize = LE_ALPHA + 1;
const LE_FOLD: usize = LE_PREV + 1;
const LE_N_IN: usize = LE_FOLD;
const LE_T: usize = LE_FOLD + 2 * (LE_LANES - 1);
const LE_ACC: usize = LE_T + 1;
const LE_K: usize = LE_ACC + 1;
const LE_KAPPA: usize = 8;

/// First column of fold level `l` (`1..=LE_VARS`); level `l` has `64 >> l`
/// nodes and each node owns two columns.
fn le_base(l: usize) -> usize {
    (1..l).fold(LE_FOLD, |acc, k| acc + 2 * (LE_LANES >> k))
}

/// The column holding entry `j` of the array entering fold level `l`.
fn le_prev(l: usize, j: usize) -> usize {
    if l == 1 {
        j
    } else {
        le_base(l - 1) + 2 * j + 1
    }
}

/// The fully folded value: the last level's single node.
fn le_y() -> usize {
    le_base(LE_VARS) + 1
}

impl LeafEvalGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let mut b = ElementTableBuilder::new(LE_KAPPA);
        for c in 0..LE_N_IN {
            b.free_wire(c);
        }
        for l in 1..=LE_VARS {
            for i in 0..(LE_LANES >> l) {
                let (p0, p1) = (le_prev(l, 2 * i), le_prev(l, 2 * i + 1));
                let d = le_base(l) + 2 * i;
                b.mult_lin(d, &[(p0, one), (p1, one)], &[(LE_V + l - 1, one)]);
                b.linear(d + 1, &[(p0, one), (d, one)]);
            }
        }
        b.mult(LE_T, LE_ALPHA, le_y());
        b.linear(LE_ACC, &[(LE_PREV, one), (LE_T, one)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("leaf-eval block is valid")),
        }
    }
}

impl GateType for LeafEvalGate {
    /// The row's committed columns, verbatim.
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..LE_N_IN).map(IoWord::input).collect();
        schema.push(IoWord::output(LE_ACC));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; LE_K];
        z[..LE_N_IN].copy_from_slice(&inputs[..LE_N_IN]);
        for l in 1..=LE_VARS {
            for i in 0..(LE_LANES >> l) {
                let (p0, p1) = (z[le_prev(l, 2 * i)], z[le_prev(l, 2 * i + 1)]);
                let d = le_base(l) + 2 * i;
                z[d] = (p0 + p1) * z[LE_V + l - 1];
                z[d + 1] = p0 + z[d];
            }
        }
        z[LE_T] = z[LE_ALPHA] * z[le_y()];
        z[LE_ACC] = z[LE_PREV] + z[LE_T];
        (vec![z[LE_ACC]], z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                z[(c << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// **MVP-3b: the opened leaf reaches the arithmetic.**
///
/// The two halves of checking a PCS query, in one circuit and one proof:
///
/// - a boolean [`MerklePathGate`] binds each leaf to the committed root, and
/// - an element [`LeafEvalGate`] consumes the SAME leaf words and computes
///   `α_i · ⟨row_i, eq(v, ·)⟩`, accumulating across openings.
///
/// The 64 leaf words are one wire class each with cells in BOTH slots, so the
/// copy constraint is what makes "the leaf that is in the tree" and "the leaf
/// the arithmetic ran on" the same leaf. That is the join the whole wiring
/// layer exists for, and it crosses the class boundary: a `k_log`-19 boolean
/// slot and a `kappa`-8 element slot in one union.
///
/// The published accumulator is checked against `enforced_sum` computed
/// natively the way `ligerito::induce_sumcheck_enforced_sum` computes it.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn leaf_arithmetic_joins_the_merkle_openings() {
    use flock_core::lincheck::build_eq_table;
    use flock_prover::prover::UnionElementSlotInput;

    // 1 KiB leaves: the L0 shape, and what makes a leaf exactly LE_LANES words.
    let (depth, leaf_bytes, n_open) = (2usize, 1024usize, 4usize);
    let nu = 3usize; // Merkle k_log is 19 here, so M = 22 at the Ligerito floor
    let mut rng = Rng(0x_3B_1EA_F00);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let v: Vec<F128> = (0..LE_VARS)
        .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
        .collect();
    let alpha: Vec<F128> = (0..n_open)
        .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
        .collect();
    let positions: Vec<usize> = (0..n_open).map(|i| i % (1 << depth)).collect();

    // ---- setup ----
    let mut sb = ShapeBuilder::new(nu);
    let merkle = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
    let leafeval = sb.slot(LeafEvalGate::new());

    let v_w: Vec<Wire> = (0..LE_VARS).map(|_| sb.public_input()).collect();
    let mut acc = sb.public_input(); // the accumulator's seed, published as zero
    let mut roots = Vec::new();
    for _ in 0..n_open {
        let leaf_w: Vec<Wire> = (0..LE_LANES).map(|_| sb.input()).collect();
        let idx_w = sb.public_input();

        // The join: `leaf_w` feeds the Merkle gate AND the arithmetic gate.
        let mut m_in = leaf_w.clone();
        m_in.push(idx_w);
        roots.push(sb.gate_hinted(merkle, &m_in));

        let mut a_in = leaf_w;
        a_in.extend_from_slice(&v_w);
        a_in.push(sb.public_input()); // alpha_i
        a_in.push(acc);
        acc = sb.gate(leafeval, &a_in)[0];
    }
    for r in &roots {
        sb.publish(r[0]);
        sb.publish(r[1]);
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid circuit");

    // ---- online ----
    let mut vals: Vec<F128> = v.clone();
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (i, &pos) in positions.iter().enumerate() {
        let leaf = tree.leaf(pos);
        vals.extend((0..LE_LANES).map(|w| leaf_word(leaf, 16 * w)));
        vals.push(F128::new(table_index(pos, depth) as u64, 0));
        vals.push(alpha[i]);
        hints.push(tree.siblings(pos));
    }
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    let built = shape.run(&vals, &hint_refs);

    // ---- the join is structural, not incidental ----
    // Every leaf word must be ONE wire class holding a cell in the Merkle
    // slot's leaf region and a cell in the leaf-eval slot's. Without this the
    // test would pass just as well on two unrelated circuits that happen to
    // agree numerically, which is exactly what the wiring layer is for.
    let iota_base = |reg: usize| -> usize {
        (0..reg)
            .map(|i| shape.registry.types()[i].io_schema.len())
            .sum()
    };
    let (m_leaf, l_leaf) = (
        iota_base(shape.registry_slot(merkle)),
        iota_base(shape.registry_slot(leafeval)),
    );
    let joined = flock_core::circuit::wire_cells(&shape.circuit)
        .iter()
        .filter(|cls| {
            let has = |lo: usize| cls.iter().any(|c| (lo..lo + LE_LANES).contains(&c.slot));
            has(m_leaf) && has(l_leaf)
        })
        .count();
    assert_eq!(
        joined,
        LE_LANES * n_open,
        "leaf words are not shared between the Merkle and arithmetic slots"
    );

    // ---- the accumulator IS the verifier's enforced_sum ----
    let eq = build_eq_table(&v);
    let want = positions
        .iter()
        .enumerate()
        .fold(F128::ZERO, |s, (i, &pos)| {
            let leaf = tree.leaf(pos);
            let dot = (0..LE_LANES)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            s + alpha[i] * dot
        });
    assert_eq!(
        *built.public.last().unwrap(),
        want,
        "the circuit's accumulator disagrees with enforced_sum"
    );
    // ...and every opening still folds to the tree root.
    let root_want = digest_words(&hash_to_digest(&tree.root));
    let base = built.public.len() - 1 - 2 * n_open;
    for i in 0..n_open {
        assert_eq!(
            [built.public[base + 2 * i], built.public[base + 2 * i + 1]],
            root_want,
            "opening {i} root"
        );
    }

    // ---- prove / verify, both classes ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();
    let el = match &built.witnesses[shape.registry_slot(leafeval)] {
        SlotWitness::Element(z) => z.clone(),
        other => panic!("leaf-eval slot produced {other:?}"),
    };
    // The fold is pinned by the relation, not merely consistent with it:
    // an honest witness satisfies, and perturbing any one intermediate — a
    // `mult_lin` product, its `linear` partner, or the alpha product — does
    // not. Otherwise the accumulator could be reached without doing the
    // arithmetic.
    {
        let ty = &LeafEvalGate::new().ty;
        assert!(ty.satisfies(&el, nu, n_open), "honest leaf-eval witness");
        for (what, col) in [
            ("fold product", le_base(1)),
            ("fold sum", le_base(4) + 1),
            ("alpha product", LE_T),
        ] {
            let mut bad = el.clone();
            bad[col << nu] += F128::ONE;
            assert!(
                !ty.satisfies(&bad, nu, n_open),
                "{what} (column {col}) is not constrained"
            );
        }
    }

    let m_witness =
        layout.generate_witness_batch_major_partial_chunk(built.rows::<MerklePathGate>(merkle), nu);

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(m_witness, &walker)],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut ch,
    );

    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![&walker];
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the joined Merkle + leaf-arithmetic circuit verifies");

    // Tampering with the claimed sum breaks it — the accumulator is wired to
    // the same leaf words the Merkle openings bind.
    let mut bad = built.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_jagged_union_circuit(
            &union,
            &shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered enforced_sum must be rejected"
    );
}
