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
const PARENT: u32 = 1 << 2;

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

fn unpack8(a: F128, b: F128) -> [u32; SLOT_WORDS] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
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

/// The table's index and the tree position are THE SAME NUMBER (`0776f64`
/// flipped the swap gadget's polarity to make it so). Kept as a named function
/// because it is the identity that has to hold for a Fiat-Shamir challenge to
/// be wireable straight into the index: `sample_queries` masks the challenge
/// to a position, so the circuit must open that position and not its
/// complement.
fn table_index(pos: usize, _depth: usize) -> u128 {
    pos as u128
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

    // What the challenge word selects, natively. This is `sample_queries`'
    // own arithmetic — `challenge.lo & (block_len - 1)` — and the circuit must
    // open exactly this position. It does, with no masking gadget and no
    // complement: the relation reads the index word's low `depth` columns, and
    // the gadget's polarity makes the index the tree position.
    let compressed = blake3::blake3_compress(&IV, &m, 0, 64, CHUNK_START | CHUNK_END);
    let challenge = pack8(&compressed[0..8].try_into().unwrap())[0];
    let index = (challenge.lo as u128) | ((challenge.hi as u128) << 64);
    let pos = (index & ((1u128 << depth) - 1)) as usize;
    assert_eq!(
        pos,
        (challenge.lo as usize) & ((1 << depth) - 1),
        "sample_queries' mask"
    );
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
    lay: LeafLayout,
}

/// The column layout of a [`LeafEvalGate`] over `lanes` leaf words.
///
/// Parameterised because the levels differ: L0's leaves are 1 KiB (64 lanes
/// at `log_batch_size = 6`) and every recursive level's are 128 B (8 lanes).
/// Same shape, different width — and two levels with the same lane count
/// share one table type, hence one slot.
#[derive(Clone, Copy)]
struct LeafLayout {
    lanes: usize,
    vars: usize,
    v: usize,
    alpha: usize,
    prev: usize,
    fold: usize,
    n_in: usize,
    t: usize,
    acc: usize,
    k: usize,
    kappa: usize,
}

impl LeafLayout {
    fn new(lanes: usize) -> Self {
        assert!(lanes.is_power_of_two() && lanes >= 2);
        let vars = lanes.trailing_zeros() as usize;
        let (v, alpha) = (lanes, lanes + vars);
        let (prev, fold) = (alpha + 1, alpha + 2);
        let t = fold + 2 * (lanes - 1);
        let k = t + 2;
        Self {
            lanes,
            vars,
            v,
            alpha,
            prev,
            fold,
            n_in: fold,
            t,
            acc: t + 1,
            k,
            kappa: k.next_power_of_two().trailing_zeros().max(2) as usize,
        }
    }

    /// First column of fold level `l` (`1..=vars`); level `l` has
    /// `lanes >> l` nodes and each node owns two columns.
    fn base(&self, l: usize) -> usize {
        (1..l).fold(self.fold, |acc, k| acc + 2 * (self.lanes >> k))
    }

    /// The column holding entry `j` of the array entering fold level `l`.
    fn prev_col(&self, l: usize, j: usize) -> usize {
        if l == 1 {
            j
        } else {
            self.base(l - 1) + 2 * j + 1
        }
    }

    /// The fully folded value: the last level's single node.
    fn y(&self) -> usize {
        self.base(self.vars) + 1
    }
}

/// L0's lane count, and the width the single-level tests use.
const LE_LANES: usize = 64;
const LE_VARS: usize = 6;

impl LeafEvalGate {
    fn new(lanes: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let lay = LeafLayout::new(lanes);
        let mut b = ElementTableBuilder::new(lay.kappa);
        for c in 0..lay.n_in {
            b.free_wire(c);
        }
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (lay.prev_col(l, 2 * i), lay.prev_col(l, 2 * i + 1));
                let d = lay.base(l) + 2 * i;
                b.mult_lin(d, &[(p0, one), (p1, one)], &[(lay.v + l - 1, one)]);
                b.linear(d + 1, &[(p0, one), (d, one)]);
            }
        }
        b.mult(lay.t, lay.alpha, lay.y());
        b.linear(lay.acc, &[(lay.prev, one), (lay.t, one)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("leaf-eval block is valid")),
            lay,
        }
    }
}

impl GateType for LeafEvalGate {
    /// The row's committed columns, verbatim.
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.lay.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.lay.acc));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let lay = self.lay;
        let mut z = vec![F128::ZERO; lay.k];
        z[..lay.n_in].copy_from_slice(&inputs[..lay.n_in]);
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (z[lay.prev_col(l, 2 * i)], z[lay.prev_col(l, 2 * i + 1)]);
                let d = lay.base(l) + 2 * i;
                z[d] = (p0 + p1) * z[lay.v + l - 1];
                z[d + 1] = p0 + z[d];
            }
        }
        z[lay.t] = z[lay.alpha] * z[lay.y()];
        z[lay.acc] = z[lay.prev] + z[lay.t];
        (vec![z[lay.acc]], z)
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
    let leafeval = sb.slot(LeafEvalGate::new(LE_LANES));

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
        let ty = &LeafEvalGate::new(LE_LANES).ty;
        assert!(ty.satisfies(&el, nu, n_open), "honest leaf-eval witness");
        for (what, col) in [
            ("fold product", LeafLayout::new(LE_LANES).base(1)),
            ("fold sum", LeafLayout::new(LE_LANES).base(4) + 1),
            ("alpha product", LeafLayout::new(LE_LANES).t),
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

// ---------------------------------------------------------------------------
// MVP-4: the vertical slice
// ---------------------------------------------------------------------------

/// **The query phase of a PCS verification, entirely in-circuit.**
///
/// Every earlier MVP proved one link. This joins them, and the join is that
/// *nothing about which leaves get opened is an input*:
///
/// ```text
///   transcript bytes ──(FS chain, BLAKE3)──▶ challenge words
///                                                │  copy constraint
///                                                ▼
///                                          index word of a Merkle opening
///                                                │  copy constraint (leaf words)
///                                                ▼
///                                          leaf-eval ──▶ enforced_sum
/// ```
///
/// The commitment is Ligerito-shaped: `block_len` codeword rows of 64 `F128`
/// lanes each — a 1 KiB leaf under `log_batch_size = 6` — hashed into a BLAKE3
/// Merkle tree by `flock_core::merkle`, which is exactly what `ligero_commit`
/// builds and what `verify_level_opens` checks against
/// (`chunk_root_matches_flock_core_blake3_tree` pins the table to it).
///
/// The query rule is the protocol's: `sample_queries` is
/// `challenger.sample_f128_vec(count)` masked with `block_len - 1`, and the
/// circuit reproduces it with no gadget at all — the challenge word is wired
/// into the index and the relation reads its low `depth` columns.
///
/// Six queries on purpose: a squeeze spans 64-byte XOF blocks and challenge
/// `k` is output `k % 4` of block `k / 4`, so six crosses a block boundary.
#[test]
#[ignore] // Heavy — run with `-- --ignored`.
fn mvp4_query_phase_end_to_end() {
    mvp4_slice(4, 6, 3);
}

/// The same slice at the **real L0 shape** — 218 queries over a 2^13 x 1 KiB
/// commitment, what a Ligerito verifier at dense m = 25 actually checks — with
/// the phases timed. Compare `l0_shape_circuit_cost`, which proves the Merkle
/// openings alone at this shape: the delta is what deriving the queries and
/// running the arithmetic on them costs.
#[test]
#[ignore] // The real shape. `-- --ignored`.
fn mvp4_l0_shape_cost() {
    mvp4_slice(13, 218, 8);
}

fn mvp4_slice(depth: usize, n_queries: usize, nu: usize) {
    use flock_core::challenger::Challenger as _;
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};

    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp4-query-phase-v0";
    // Pin the P-cores and measure warm, as `l0_shape_circuit_cost` does — the
    // default pool and a cold first prove each move the numbers ~2x.
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let block_len = 1usize << depth;
    let leaf_bytes = 16 * LE_LANES; // 1 KiB: 64 F128 lanes

    // ---- a Ligerito-shaped L0 commitment ----
    let mut rng = Rng(0x_4E_C0_DE_01);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    // ---- the transcript, and the queries it determines ----
    let mut ch = FsChallenger::with_hash(SLICE, HashKind::Blake3);
    ch.observe_bytes(&tree.root);
    let want_positions: Vec<usize> = ch
        .sample_f128_vec(n_queries)
        .iter()
        .map(|v| (v.lo as usize) & (block_len - 1))
        .collect();

    // Record the same transcript: the shape drives the circuit.
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    rec.observe_bytes(&tree.root);
    let derived = rec.sample_f128_vec(n_queries);
    assert_eq!(
        derived
            .iter()
            .map(|v| (v.lo as usize) & (block_len - 1))
            .collect::<Vec<_>>(),
        want_positions,
        "recording the transcript changed it"
    );
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();
    assert_eq!(challenges.len(), n_queries);

    // ---- replay it through the FS chain ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    assert_eq!(trace.squeezes.len(), 1, "one squeeze: the query draw");

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let t0 = Instant::now();
    let hash = sb.slot(Blake3Gate { nu });
    let slot_hash_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let merkle = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, block_len));
    let slot_merkle_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let leafeval = sb.slot(LeafEvalGate::new(LE_LANES));
    let slot_leaf_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t_wiring = Instant::now();

    // The FS chain, verbatim from MVP-1: every row's cv and message come from
    // an earlier row's output or from a transcript word.
    let iv = [sb.public_input(), sb.public_input()];
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    // Input values in declaration order. Every FS wire is a `public_input`,
    // declared and published in the same order, so a value's index here IS its
    // index in the public segment — which the tamper check below relies on.
    let mut fs_values: Vec<F128> = Vec::new();
    let mut first_msg_pub: Option<usize> = None;
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    fs_values.extend_from_slice(&iv_w);

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = sb.public_input();
        fs_values.push(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(hash, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        fs_values.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                fs_values.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                first_msg_pub.get_or_insert(fs_values.len() - 1);
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(hash, &g_in));
    }

    // **The binding.** Challenge `k` is output `k % 4` of the squeeze's block
    // `k / 4` — so this is the wire, and there is no other route from the
    // transcript to a query.
    let sq = &trace.squeezes[0];
    let challenge_w: Vec<Wire> = (0..n_queries).map(|k| outs[sq[k / 4]][k % 4]).collect();

    // The openings, and the arithmetic on them.
    let v_w: Vec<Wire> = (0..LE_VARS).map(|_| sb.public_input()).collect();
    let mut acc = sb.public_input();
    let mut roots = Vec::new();
    for (k, &cw) in challenge_w.iter().enumerate() {
        let leaf_w: Vec<Wire> = (0..LE_LANES).map(|_| sb.input()).collect();
        let mut m_in = leaf_w.clone();
        m_in.push(cw); // ← the challenge word IS the index word
        roots.push(sb.gate_hinted(merkle, &m_in));

        let mut a_in = leaf_w;
        a_in.extend_from_slice(&v_w);
        a_in.push(sb.public_input()); // alpha_k
        a_in.push(acc);
        acc = sb.gate(leafeval, &a_in)[0];
        let _ = k;
    }
    for r in &roots {
        sb.publish(r[0]);
        sb.publish(r[1]);
    }
    sb.publish(acc);
    let wiring_ms = t_wiring.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let shape = sb.finish().expect("valid circuit");
    let finish_ms = t0.elapsed().as_secs_f64() * 1e3;
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let v: Vec<F128> = (0..LE_VARS)
        .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
        .collect();
    let alpha: Vec<F128> = (0..n_queries)
        .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
        .collect();

    let mut vals = fs_values;
    vals.extend_from_slice(&v);
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (k, &pos) in want_positions.iter().enumerate() {
        let leaf = tree.leaf(pos);
        vals.extend((0..LE_LANES).map(|w| leaf_word(leaf, 16 * w)));
        vals.push(alpha[k]);
        hints.push(tree.siblings(pos));
    }
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    std::hint::black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    // **The chain is structural.** Each query's wire class must hold a cell in
    // the BLAKE3 slot's output region and a cell at the Merkle slot's index
    // word, and each leaf word a cell in the Merkle slot and one in the
    // leaf-eval slot. Values agreeing is not enough: without these classes the
    // circuit would be three unrelated computations that happen to line up.
    {
        let iota = |reg: usize| -> usize {
            (0..reg)
                .map(|i| shape.registry.types()[i].io_schema.len())
                .sum::<usize>()
        };
        let (h, m, l) = (
            iota(shape.registry_slot(hash)),
            iota(shape.registry_slot(merkle)),
            iota(shape.registry_slot(leafeval)),
        );
        let classes = flock_core::circuit::wire_cells(&shape.circuit);
        let spans = |lo_a: usize, n_a: usize, lo_b: usize, n_b: usize| {
            classes
                .iter()
                .filter(|cls| {
                    let has =
                        |lo: usize, n: usize| cls.iter().any(|c| (lo..lo + n).contains(&c.slot));
                    has(lo_a, n_a) && has(lo_b, n_b)
                })
                .count()
        };
        // blake3 outputs are schema words 7..11; the Merkle index is word 64.
        assert_eq!(
            spans(h + blake3::IO_OUT_LO0, 4, m + 4 * (leaf_bytes / 64), 1),
            n_queries,
            "challenge words are not wired to the Merkle index words"
        );
        assert_eq!(
            spans(m, LE_LANES, l, LE_LANES),
            LE_LANES * n_queries,
            "leaf words are not shared with the arithmetic slot"
        );
    }

    // The circuit opened the positions the TRANSCRIPT chose. If the wiring
    // from challenge to index were wrong, `run`'s own equality check on the
    // Merkle rows' index word would already have fired; assert it anyway,
    // because this is the sentence the whole slice exists to make true.
    let rows = built.rows::<MerklePathGate>(merkle);
    for (k, &pos) in want_positions.iter().enumerate() {
        assert_eq!(
            (rows[k].index & (block_len as u128 - 1)) as usize,
            pos,
            "opening {k} did not open the query the transcript derived"
        );
        assert_eq!(
            rows[k].index,
            (challenges[k].lo as u128) | ((challenges[k].hi as u128) << 64),
            "opening {k}'s index is not the whole challenge word"
        );
        assert_eq!(rows[k].leaf_data, tree.leaf(pos), "opening {k} leaf");
    }
    // ...every opening folds to the committed root...
    let root_want = digest_words(&hash_to_digest(&tree.root));
    let base = built.public.len() - 1 - 2 * n_queries;
    for k in 0..n_queries {
        assert_eq!(
            [built.public[base + 2 * k], built.public[base + 2 * k + 1]],
            root_want,
            "opening {k} root"
        );
    }
    // ...and the accumulator is the verifier's enforced_sum over them.
    let eq = build_eq_table(&v);
    let want_sum = want_positions
        .iter()
        .enumerate()
        .fold(F128::ZERO, |s, (k, &pos)| {
            let leaf = tree.leaf(pos);
            let dot = (0..LE_LANES)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            s + alpha[k] * dot
        });
    assert_eq!(
        *built.public.last().unwrap(),
        want_sum,
        "enforced_sum disagrees with the native computation"
    );

    // ---- prove / verify: three slots, both classes ----
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
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();
    let el = match &built.witnesses[shape.registry_slot(leafeval)] {
        SlotWitness::Element(z) => z.clone(),
        other => panic!("leaf-eval slot produced {other:?}"),
    };

    // Boolean slots go in REGISTRY order.
    let t = Instant::now();
    let m_wit = layout.generate_witness_batch_major_partial_chunk(rows, nu);
    let h_wit = blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu);
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut bool_slots = vec![
        (
            shape.registry_slot(merkle),
            UnionSlotProverInput::new(m_wit, &walker),
        ),
        (
            shape.registry_slot(hash),
            UnionSlotProverInput::new(h_wit, b3_lc),
        ),
    ];
    bool_slots.sort_by_key(|(i, _)| *i);
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape.registry_slot(merkle), &walker),
        (shape.registry_slot(hash), b3_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        bool_slots.into_iter().map(|(_, s)| s).collect(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut c,
    );

    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut c,
    )
    .expect("the query phase verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    // Tampering with a TRANSCRIPT word must break it. This is the sharp one:
    // that word is hashed into the challenge, the challenge is the index, and
    // the index selects the leaf — so a transcript the prover did not commit to
    // cannot be made to justify the openings it already produced.
    let msg_pub = first_msg_pub.expect("the transcript has message words");
    let mut bad = built.public.clone();
    bad[msg_pub] += F128::ONE;
    let mut c = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_jagged_union_circuit(
            &union,
            &shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .is_err(),
        "a tampered transcript word must be rejected"
    );

    // ---- trace size ----
    // Two different sizes, and they behave differently:
    //  * COMMITTED is jagged — `dense_words` sums `used_cols(ty) * n_t` over
    //    real row counts, then rounds to a power of two. This is what the PCS
    //    commits and opens.
    //  * ADDRESS SPACE is capacity-based — `m_bool` is
    //    `next_pow2(sum 2^(nu + k_log))`, so every slot pays a full `2^nu`
    //    rows whether it uses them or not. This is what the zerocheck and
    //    lincheck run over, and it is where the padding bites.
    {
        let mut hdr = format!(
            "\nTRACE SIZE\n  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12} {:>7}\n",
            "slot", "k_log", "useful bits", "words", "rows", "dense words", "used"
        );
        let mut addr_cells = 0usize;
        for (i, ty) in shape.registry.types().iter().enumerate() {
            let words = ty.useful_bits.div_ceil(128);
            let n_t = shape.counts[i];
            let cells = 1usize << (nu + ty.k_log);
            addr_cells += cells;
            let name = if i == shape.registry_slot(hash) {
                "blake3"
            } else if i == shape.registry_slot(merkle) {
                "merkle"
            } else {
                "leaf-eval"
            };
            hdr += &format!(
                "  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12} {:>6.1}%\n",
                name,
                ty.k_log,
                ty.useful_bits,
                words,
                format!("{}/{}", n_t, 1usize << nu),
                words * n_t,
                100.0 * (words * n_t) as f64 / (cells / 128) as f64,
            );
        }
        hdr += &format!(
            "  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12}\n",
            "TOTAL",
            "",
            "",
            "",
            "",
            union.dense_words()
        );
        // What the LINCHECK actually sweeps — O(nnz), and unrelated to trace
        // size. The Merkle walker keeps ONE copy of the base CSC and walks it
        // per block; BLAKE3's own slot sweeps the full materialized block.
        let (ba, bb) = {
            let (a, b) = blake3::build_matrices();
            (
                a.rows.iter().map(|r| r.len()).sum::<usize>(),
                b.rows.iter().map(|r| r.len()).sum::<usize>(),
            )
        };
        hdr += &format!(
            "  lincheck nnz: merkle walker {} (effective) | blake3 block {}\n",
            walker.effective_nnz(),
            ba + bb,
        );
        hdr += &format!(
            "  committed {} words (2^{}) | address space 2^{} cells = {} words \
             ({:.1}% used)\n  M_bool {} | M_elem {} | M_total {}\n",
            union.committed_words(),
            union.dense_m() - 7,
            (addr_cells as f64).log2().ceil() as usize,
            addr_cells / 128,
            100.0 * union.dense_words() as f64 / (addr_cells / 128) as f64,
            union.m_bool(),
            union.m_elem(),
            union.m_total(),
        );
        println!("{hdr}");
    }

    println!(
        "\nMVP-4 query phase: {n_queries} queries over a 2^{depth} x 1 KiB commitment\n\
           slots: blake3 {} rows, merkle {} rows, leaf-eval {} rows | public {} | \
         dense_m {} | proof {:.1} KiB | {threads} threads\n\
         \n\
           PER PROOF     {:6.0} ms = online {online_ms:.0} + witgen {wit_ms:.0} + \
         prove {prove_ms:.0}\n\
           verifier side {verify_ms:6.1} ms\n\
           SETUP         {setup_ms:6.0} ms = slot(blake3) {slot_hash_ms:.0} + \
         slot(merkle) {slot_merkle_ms:.0} + slot(leaf) {slot_leaf_ms:.0} + \
         wiring {wiring_ms:.0} + finish {finish_ms:.0}\n",
        shape.counts[shape.registry_slot(hash)],
        shape.counts[shape.registry_slot(merkle)],
        shape.counts[shape.registry_slot(leafeval)],
        built.public.len(),
        union.dense_m(),
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        online_ms + wit_ms + prove_ms,
    );
}

// ---------------------------------------------------------------------------
// MVP-5: every level's query phase
// ---------------------------------------------------------------------------

/// One Ligerito level's commitment shape.
#[derive(Clone, Copy)]
struct Level {
    /// `log2(block_len)` = `log_msg_cols + log_inv_rate`; the tree's depth.
    depth: usize,
    /// Interleaved lanes per codeword row; the leaf is `16 * lanes` bytes.
    lanes: usize,
    queries: usize,
}

/// **The whole query phase**: all four levels of the m=26 Fast ladder, in one
/// circuit and one proof.
///
/// The ladder (`docs/local/recursion-verifier-map.md` §2.5c, from
/// `configs/ligerito/m26_fast.toml`) is
/// `(log_inv_rate, log_msg_cols, lanes, queries)` per level; `block_len` is
/// `msg_cols << log_inv_rate`, so the tree depth is their sum:
///
/// ```text
///   L0  rate 1  cols 13  lanes 64  218 queries  ⇒  depth 14, 1 KiB leaves
///   L1  rate 2  cols 10  lanes  8  106 queries  ⇒  depth 12, 128 B leaves
///   L2  rate 3  cols  7  lanes  8   71 queries  ⇒  depth 10, 128 B leaves
///   L3  rate 4  cols  4  lanes  8   53 queries  ⇒  depth  8, 128 B leaves
/// ```
///
/// **What this measures.** Each level's tree shape yields a DIFFERENT
/// `MerkleTreeLayout` — different `useful_bits`, different matrices — so each
/// is its own table type, and each one's walker stores its own copy of
/// BLAKE3's base CSC. The lincheck sweeps once per slot, so the prediction is
/// ~5 x 21M nonzeros against MVP-4's ~2 x. This is the measurement that
/// decides whether collapsing the composites into one plain BLAKE3 table is
/// worth the conditional-swap and bit-decomposition glue it would need.
///
/// L1..L3 share one 8-lane leaf-eval slot: same lane count, same table type.
#[test]
#[ignore] // The full shape. `-- --ignored`.
fn mvp5_all_levels_query_phase() {
    use flock_core::challenger::Challenger as _;
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};
    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp5-all-levels-v0";
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let levels = [
        Level {
            depth: 14,
            lanes: 64,
            queries: 218,
        },
        Level {
            depth: 12,
            lanes: 8,
            queries: 106,
        },
        Level {
            depth: 10,
            lanes: 8,
            queries: 71,
        },
        Level {
            depth: 8,
            lanes: 8,
            queries: 53,
        },
    ];
    let nu = 8usize; // 218 is the largest row count

    // ---- one commitment per level ----
    let mut rng = Rng(0x_5EED_0005);
    let trees: Vec<Tree> = levels
        .iter()
        .map(|l| Tree::new(l.depth, 16 * l.lanes, &mut rng))
        .collect();

    // ---- the transcript: absorb each root, then draw that level's queries ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    let mut want: Vec<Vec<usize>> = Vec::new();
    for (l, tree) in levels.iter().zip(&trees) {
        rec.observe_bytes(&tree.root);
        want.push(
            rec.sample_f128_vec(l.queries)
                .iter()
                .map(|v| (v.lo as usize) & ((1usize << l.depth) - 1))
                .collect(),
        );
    }
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();
    assert_eq!(
        challenges.len(),
        levels.iter().map(|l| l.queries).sum::<usize>()
    );

    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    assert_eq!(trace.squeezes.len(), levels.len(), "one squeeze per level");

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let hash = sb.slot(Blake3Gate { nu });
    let merkle: Vec<_> = levels
        .iter()
        .map(|l| sb.slot(MerklePathGate::new(l.depth, 16 * l.lanes, nu, 1 << l.depth)))
        .collect();
    // One leaf-eval slot per distinct lane count; L1..L3 share.
    let mut leaf_slot: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
    let leafeval: Vec<_> = levels
        .iter()
        .map(|l| match leaf_slot.iter().find(|(n, _)| *n == l.lanes) {
            Some((_, s)) => *s,
            None => {
                let s = sb.slot(LeafEvalGate::new(l.lanes));
                leaf_slot.push((l.lanes, s));
                s
            }
        })
        .collect();

    let iv = [sb.public_input(), sb.public_input()];
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    let mut fs_values: Vec<F128> = Vec::new();
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    fs_values.extend_from_slice(&iv_w);

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = sb.public_input();
        fs_values.push(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(hash, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        fs_values.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                fs_values.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(hash, &g_in));
    }

    // Per level: challenge words → indices, openings, arithmetic.
    let mut v_w: Vec<Vec<Wire>> = Vec::new();
    let mut acc = sb.public_input();
    let mut all_roots: Vec<Vec<Vec<Wire>>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        let sq = &trace.squeezes[li];
        let vars = l.lanes.trailing_zeros() as usize;
        let vs: Vec<Wire> = (0..vars).map(|_| sb.public_input()).collect();
        let mut roots = Vec::with_capacity(l.queries);
        for k in 0..l.queries {
            let cw = outs[sq[k / 4]][k % 4];
            let leaf_w: Vec<Wire> = (0..l.lanes).map(|_| sb.input()).collect();
            let mut m_in = leaf_w.clone();
            m_in.push(cw);
            roots.push(sb.gate_hinted(merkle[li], &m_in));

            let mut a_in = leaf_w;
            a_in.extend_from_slice(&vs);
            a_in.push(sb.public_input()); // alpha
            a_in.push(acc);
            acc = sb.gate(leafeval[li], &a_in)[0];
        }
        v_w.push(vs);
        all_roots.push(roots);
    }
    for roots in &all_roots {
        for r in roots {
            sb.publish(r[0]);
            sb.publish(r[1]);
        }
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let vees: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.lanes.trailing_zeros() as usize)
                .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
                .collect()
        })
        .collect();
    let alphas: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.queries)
                .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
                .collect()
        })
        .collect();

    let mut vals = fs_values;
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        vals.extend_from_slice(&vees[li]);
        for k in 0..l.queries {
            let pos = want[li][k];
            let leaf = trees[li].leaf(pos);
            vals.extend((0..l.lanes).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(alphas[li][k]);
            hints.push(trees[li].siblings(pos));
        }
    }
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    std::hint::black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    // Every level opened the queries its own squeeze determined, and every
    // opening folds to that level's root.
    for (li, l) in levels.iter().enumerate() {
        let rows = built.rows::<MerklePathGate>(merkle[li]);
        assert_eq!(rows.len(), l.queries);
        for k in 0..l.queries {
            assert_eq!(
                (rows[k].index & ((1u128 << l.depth) - 1)) as usize,
                want[li][k],
                "L{li} opening {k} is not the query the transcript derived"
            );
        }
    }
    let mut at = built.public.len() - 1 - 2 * levels.iter().map(|l| l.queries).sum::<usize>();
    for (li, l) in levels.iter().enumerate() {
        let rw = digest_words(&hash_to_digest(&trees[li].root));
        for k in 0..l.queries {
            assert_eq!(
                [built.public[at], built.public[at + 1]],
                rw,
                "L{li} root {k}"
            );
            at += 2;
        }
    }
    // ...and the accumulator is enforced_sum over ALL levels.
    let mut want_sum = F128::ZERO;
    for (li, l) in levels.iter().enumerate() {
        let eq = build_eq_table(&vees[li]);
        for k in 0..l.queries {
            let leaf = trees[li].leaf(want[li][k]);
            let dot = (0..l.lanes)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            want_sum += alphas[li][k] * dot;
        }
    }
    assert_eq!(*built.public.last().unwrap(), want_sum, "enforced_sum");

    // ---- prove / verify ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layouts: Vec<MerkleTreeLayout> = levels
        .iter()
        .map(|l| MerkleTreeLayout::with_blake3_chunk_leaf(l.depth, 16 * l.lanes, blake3_spec()))
        .collect();
    let walkers: Vec<_> = layouts.iter().map(|l| l.build_walker()).collect();
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();

    let t = Instant::now();
    let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![(
        shape.registry_slot(hash),
        UnionSlotProverInput::new(
            blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu),
            b3_lc,
        ),
    )];
    for (li, _) in levels.iter().enumerate() {
        bool_slots.push((
            shape.registry_slot(merkle[li]),
            UnionSlotProverInput::new(
                layouts[li].generate_witness_batch_major_partial_chunk(
                    built.rows::<MerklePathGate>(merkle[li]),
                    nu,
                ),
                &walkers[li],
            ),
        ));
    }
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;

    bool_slots.sort_by_key(|(i, _)| *i);
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> =
        vec![(shape.registry_slot(hash), b3_lc)];
    for (li, _) in levels.iter().enumerate() {
        lcs_ord.push((shape.registry_slot(merkle[li]), &walkers[li]));
    }
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    // Element slots go in registry order too.
    let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
        .iter()
        .zip(els)
        .map(|((_, s), z)| (shape.registry_slot(*s), z))
        .collect();
    el_ord.sort_by_key(|(i, _)| *i);
    let el_inputs: Vec<UnionElementSlotInput> = el_ord
        .into_iter()
        .map(|(_, z)| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z)))
        .collect();

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        bool_slots.into_iter().map(|(_, s)| s).collect(),
        el_inputs,
        &mut c,
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut c,
    )
    .expect("the full query phase verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- what it cost ----
    let mut nnz_total = 0usize;
    let mut report = String::from("\nMVP-5 FULL QUERY PHASE (m=26 Fast ladder)\n");
    report += &format!(
        "  {:<12} {:>6} {:>8} {:>7} {:>14}\n",
        "slot", "k_log", "rows", "words", "lincheck nnz"
    );
    for (i, ty) in shape.registry.types().iter().enumerate() {
        let words = ty.useful_bits.div_ceil(128);
        let nnz = if i == shape.registry_slot(hash) {
            let (a, b) = blake3::build_matrices();
            let n = a.rows.iter().map(|r| r.len()).sum::<usize>()
                + b.rows.iter().map(|r| r.len()).sum::<usize>();
            nnz_total += n;
            n
        } else if let Some(li) = (0..levels.len()).find(|&li| shape.registry_slot(merkle[li]) == i)
        {
            let n = walkers[li].effective_nnz() / layouts[li].total_blocks();
            nnz_total += n;
            n
        } else {
            0
        };
        report += &format!(
            "  {:<12} {:>6} {:>8} {:>7} {:>14}\n",
            format!("slot{i}"),
            ty.k_log,
            shape.counts[i],
            words * shape.counts[i],
            nnz
        );
    }
    report += &format!(
        "  stored lincheck nnz total {nnz_total} | dense {} words | dense_m {} | \
         M_bool {} | M_total {}\n\n  \
         PER PROOF     {:6.0} ms = online {online_ms:.0} + witgen {wit_ms:.0} + prove {prove_ms:.0}\n  \
         verifier side {verify_ms:6.1} ms | proof {:.1} KiB | {threads} threads\n  \
         SETUP         {setup_ms:6.0} ms\n",
        union.dense_words(),
        union.dense_m(),
        union.m_bool(),
        union.m_total(),
        online_ms + wit_ms + prove_ms,
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
    println!("{report}");
}

// ---------------------------------------------------------------------------
// The collapsed opening: wiring over ONE BLAKE3 table
// ---------------------------------------------------------------------------

use flock_prover::r1cs_hashes::merkle_glue::{BitSpreadTable, SwapInput, SwapTable};

/// One Merkle level's conditional swap. The sibling is a [`GateType::Hint`] —
/// it is not word-aligned-wireable in the composite and nothing else reads it
/// here either, so it stays free witness.
struct SwapGate {
    nu: usize,
}

impl GateType for SwapGate {
    type Row = SwapInput;
    type Hint = [u32; SLOT_WORDS];

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&SwapTable::build_block_r1cs(self.nu))
            .with_io_schema(SwapTable::io_schema())
    }

    fn eval(&self, inputs: &[F128], hint: &Self::Hint) -> (Vec<F128>, Self::Row) {
        let row = SwapInput {
            bit_word: (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64),
            prev: unpack8(inputs[1], inputs[2]),
            sib: *hint,
        };
        let (left, right) = SwapTable::outputs(&row);
        let (lw, rw) = (digest_words(&left), digest_words(&right));
        (vec![lw[0], lw[1], rw[0], rw[1]], row)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// Relocate each of the index word's low `depth` bits into its own word, so a
/// per-level swap row can read it at the one column its uniform relation is
/// allowed to look at.
struct BitSpreadGate {
    ty: BitSpreadTable,
    nu: usize,
}

impl GateType for BitSpreadGate {
    type Row = u128;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.ty.build_block_r1cs(self.nu))
            .with_io_schema(self.ty.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, u128) {
        let idx = (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64);
        let outs = (0..self.ty.depth)
            .map(|l| F128::new(((idx >> l) & 1) as u64, 0))
            .collect();
        (outs, idx)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The three slots a collapsed opening writes into.
#[derive(Clone, Copy)]
struct CollapsedSlots {
    b3: flock_core::circuit::builder::SlotId,
    swap: flock_core::circuit::builder::SlotId,
    spread: flock_core::circuit::builder::SlotId,
}

/// Emit one Merkle opening as rows of the shipped BLAKE3 table plus glue,
/// wired together. Returns the two words of the root.
///
/// This is what replaces a composite row. The dataflow, per level:
///
/// ```text
///   index word ─▶ BitSpread ─bit_l─▶ Swap ─left‖right─▶ BLAKE3 ─out_lo─▶ next Swap.prev
/// ```
///
/// and before it, the chunk chain: `leaf_blocks` BLAKE3 rows whose out_lo
/// threads row to row, seeded by the IV, `CHUNK_START` on the first and
/// `CHUNK_END` on the last. Every arrow is a copy constraint on whole words.
#[allow(clippy::too_many_arguments)]
fn emit_opening(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    leaf_w: &[Wire],
    index_w: Wire,
    depth: usize,
    pubs: &mut Vec<F128>,
) -> [Wire; 2] {
    let blocks = leaf_w.len() / 4;
    assert_eq!(leaf_w.len(), 4 * blocks, "leaf is whole 64-byte blocks");

    // The index word's bits, one per level.
    let bits = sb.gate(s.spread, &[index_w]);

    // Chunk chain: the leaf hashed as a BLAKE3 chunk.
    let mut cv = iv;
    for i in 0..blocks {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == blocks {
            flags |= CHUNK_END;
        }
        pubs.push(pack_params(0, 64, flags));
        let params = sb.public_input();
        let out = sb.gate(
            s.b3,
            &[
                cv[0],
                cv[1],
                leaf_w[4 * i],
                leaf_w[4 * i + 1],
                leaf_w[4 * i + 2],
                leaf_w[4 * i + 3],
                params,
            ],
        );
        cv = [out[0], out[1]];
    }

    // Node levels: swap, then a PARENT compression over the swapped pair.
    // The sibling is the swap's hint, supplied at `run` time in this call
    // order — setup has no values.
    for l in 0..depth {
        let sw = sb.gate_hinted(s.swap, &[bits[l], cv[0], cv[1]]);
        pubs.push(pack_params(0, 64, PARENT));
        let params = sb.public_input();
        let out = sb.gate(s.b3, &[iv[0], iv[1], sw[0], sw[1], sw[2], sw[3], params]);
        cv = [out[0], out[1]];
    }
    cv
}

/// **The collapse, one opening at a time.** Emit an opening as BLAKE3 rows
/// plus glue and check the root it computes is the one the composite computes
/// — for the real L0 shape and a small one, at every index polarity that
/// exercises both swap directions.
///
/// If this holds, a composite row and `leaf_blocks + depth` wired rows are
/// interchangeable, and the whole collapse is a matter of scale.
#[test]
#[ignore] // Builds real BLAKE3 matrices. `-- --ignored`.
fn collapsed_opening_matches_the_composite() {
    for (depth, leaf_bytes, n_open) in [(3usize, 128usize, 8usize), (14, 1024, 3)] {
        let blocks = leaf_bytes / 64;
        let mut rng = Rng(0x_C0_11_09_5E ^ depth as u64);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        // Rows: one bit-spread + `blocks + depth` BLAKE3 + `depth` swaps each.
        let nu = (((blocks + depth) * n_open)
            .next_power_of_two()
            .trailing_zeros() as usize)
            .max(3);

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(depth),
                nu,
            }),
        };

        let mut pubs: Vec<F128> = Vec::new();
        let iv_w = pack8(&IV);
        pubs.push(iv_w[0]);
        pubs.push(iv_w[1]);
        let iv = [sb.public_input(), sb.public_input()];

        let positions: Vec<usize> = (0..n_open).map(|i| (i * 5 + 1) % (1 << depth)).collect();
        let mut leaf_vals: Vec<F128> = Vec::new();
        let mut idx_vals: Vec<F128> = Vec::new();
        let mut roots = Vec::new();
        for &pos in &positions {
            let leaf_w: Vec<Wire> = (0..4 * blocks).map(|_| sb.input()).collect();
            let index_w = sb.input();
            roots.push(emit_opening(
                &mut sb, slots, iv, &leaf_w, index_w, depth, &mut pubs,
            ));
            let leaf = tree.leaf(pos);
            leaf_vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            idx_vals.push(F128::new(pos as u64, 0));
        }
        for r in &roots {
            sb.publish(r[0]);
            sb.publish(r[1]);
        }
        let shape = sb.finish().expect("valid collapsed circuit");

        // Inputs in declaration order: iv, then per opening (params are
        // public_inputs emitted INSIDE emit_opening, already in `pubs`), and
        // the leaf/index free wires. Rebuild that order exactly.
        let mut vals: Vec<F128> = vec![iv_w[0], iv_w[1]];
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        for (i, &pos) in positions.iter().enumerate() {
            let leaf = tree.leaf(pos);
            vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(idx_vals[i]);
            // Then `emit_opening`'s own public params, in its call order.
            for b in 0..blocks {
                let mut f = 0u32;
                if b == 0 {
                    f |= CHUNK_START;
                }
                if b + 1 == blocks {
                    f |= CHUNK_END;
                }
                vals.push(pack_params(0, 64, f));
            }
            for _ in 0..depth {
                vals.push(pack_params(0, 64, PARENT));
            }
            hints.extend(tree.siblings(pos));
        }
        let hint_refs: Vec<&dyn std::any::Any> =
            hints.iter().map(|h| h as &dyn std::any::Any).collect();
        let built = shape.run(&vals, &hint_refs);

        // Every opening's root is the tree's — i.e. the collapsed rows fold
        // exactly as `root_chunk` does.
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let want = digest_words(&hash_to_digest(&tree.root));
        let base = built.public.len() - 2 * n_open;
        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(
                [built.public[base + 2 * i], built.public[base + 2 * i + 1]],
                want,
                "depth {depth} opening {i} (pos {pos}) root"
            );
            // ...and it agrees with the composite on the same input.
            let composite = layout.root_chunk(&ChunkPathInput {
                leaf_data: tree.leaf(pos).to_vec(),
                index: pos as u128,
                siblings: tree.siblings(pos),
            });
            assert_eq!(
                digest_words(&composite),
                want,
                "depth {depth} opening {i}: composite disagrees with the tree"
            );
        }
        println!(
            "  depth {depth} leaf {leaf_bytes}: {n_open} openings = {} blake3 + {} swap + \
             {} spread rows, nu {nu}, mu {}",
            shape.counts[shape.registry_slot(slots.b3)],
            shape.counts[shape.registry_slot(slots.swap)],
            shape.counts[shape.registry_slot(slots.spread)],
            shape.circuit.cells().mu(),
        );
    }
}

/// **MVP-6: the full query phase, collapsed.** The same four levels as
/// [`mvp5_all_levels_query_phase`], with every Merkle composite replaced by
/// wiring over ONE BLAKE3 table.
///
/// The prediction from `wiring_scaling.rs` and the glue tables' measured nnz:
/// lincheck ~105.1M → ~21.0M, wiring ~1.7 → ~18 ms, prove ~174 → ~119 ms.
/// Note the FS chain's compressions and the openings' share the SAME slot —
/// that is the point.
///
/// ONE `BitSpreadTable`, sized for the deepest level; shallower levels leave
/// its extra outputs unwired, and an unwired schema word is sigma-fixed and
/// costs nothing. Four depth-specific spread tables would have been four more
/// types, which is the thing being removed.
#[test]
#[ignore] // The full shape. `-- --ignored`.
fn mvp6_all_levels_collapsed() {
    use flock_core::challenger::Challenger as _;
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};
    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp6-collapsed-v0";
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let levels = [
        Level {
            depth: 14,
            lanes: 64,
            queries: 218,
        },
        Level {
            depth: 12,
            lanes: 8,
            queries: 106,
        },
        Level {
            depth: 10,
            lanes: 8,
            queries: 71,
        },
        Level {
            depth: 8,
            lanes: 8,
            queries: 53,
        },
    ];
    let max_depth = levels.iter().map(|l| l.depth).max().unwrap();

    let mut rng = Rng(0x_5EED_0006);
    let trees: Vec<Tree> = levels
        .iter()
        .map(|l| Tree::new(l.depth, 16 * l.lanes, &mut rng))
        .collect();

    // ---- transcript ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    let mut want: Vec<Vec<usize>> = Vec::new();
    for (l, tree) in levels.iter().zip(&trees) {
        rec.observe_bytes(&tree.root);
        want.push(
            rec.sample_f128_vec(l.queries)
                .iter()
                .map(|v| (v.lo as usize) & ((1usize << l.depth) - 1))
                .collect(),
        );
    }
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());

    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();

    // Rows in the ONE blake3 slot: the FS chain plus every opening's
    // compressions.
    let b3_rows: usize = trace.rows.len()
        + levels
            .iter()
            .map(|l| (16 * l.lanes / 64 + l.depth) * l.queries)
            .sum::<usize>();
    let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let slots = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(max_depth),
            nu,
        }),
    };
    let mut leaf_slot: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
    let leafeval: Vec<_> = levels
        .iter()
        .map(|l| match leaf_slot.iter().find(|(n, _)| *n == l.lanes) {
            Some((_, s)) => *s,
            None => {
                let s = sb.slot(LeafEvalGate::new(l.lanes));
                leaf_slot.push((l.lanes, s));
                s
            }
        })
        .collect();

    // Values are pushed in DECLARATION order throughout; `emit_opening` pushes
    // its own params into the same vector as it declares them.
    let mut vals: Vec<F128> = Vec::new();
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv = [sb.public_input(), sb.public_input()];

    // The FS chain, into the same blake3 slot.
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        vals.push(pack_params(counter, blen, flags));
        let params = sb.public_input();
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(slots.b3, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        vals.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                vals.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(slots.b3, &g_in));
    }

    // The openings.
    let vees: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.lanes.trailing_zeros() as usize)
                .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
                .collect()
        })
        .collect();
    let alphas: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.queries)
                .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
                .collect()
        })
        .collect();

    vals.push(F128::ZERO);
    let mut acc = sb.public_input();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut all_roots: Vec<Vec<[Wire; 2]>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        let sq = &trace.squeezes[li];
        let vars = l.lanes.trailing_zeros() as usize;
        vals.extend_from_slice(&vees[li]);
        let vs: Vec<Wire> = (0..vars).map(|_| sb.public_input()).collect();
        let blocks = 16 * l.lanes / 64;
        let mut roots = Vec::with_capacity(l.queries);
        for k in 0..l.queries {
            let pos = want[li][k];
            let leaf = trees[li].leaf(pos);
            vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            let leaf_w: Vec<Wire> = (0..4 * blocks).map(|_| sb.input()).collect();

            // The challenge word IS the index word — no masking gadget.
            let cw = outs[sq[k / 4]][k % 4];
            roots.push(emit_opening(
                &mut sb, slots, iv, &leaf_w, cw, l.depth, &mut vals,
            ));
            hints.extend(trees[li].siblings(pos));

            // The same leaf words feed the arithmetic.
            let mut a_in = leaf_w;
            a_in.extend_from_slice(&vs);
            vals.push(alphas[li][k]);
            a_in.push(sb.public_input());
            a_in.push(acc);
            acc = sb.gate(leafeval[li], &a_in)[0];
        }
        all_roots.push(roots);
    }
    for roots in &all_roots {
        for r in roots {
            sb.publish(r[0]);
            sb.publish(r[1]);
        }
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid collapsed circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    std::hint::black_box(shape.run(&vals, &hint_refs));
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    // Every opening folds to its level's root, and the accumulator is
    // enforced_sum — the same two claims MVP-5 makes, now over wired rows.
    let mut at = built.public.len() - 1 - 2 * levels.iter().map(|l| l.queries).sum::<usize>();
    for (li, l) in levels.iter().enumerate() {
        let rw = digest_words(&hash_to_digest(&trees[li].root));
        for k in 0..l.queries {
            assert_eq!(
                [built.public[at], built.public[at + 1]],
                rw,
                "L{li} root {k}"
            );
            at += 2;
        }
    }
    let mut want_sum = F128::ZERO;
    for (li, l) in levels.iter().enumerate() {
        let eq = build_eq_table(&vees[li]);
        for k in 0..l.queries {
            let leaf = trees[li].leaf(want[li][k]);
            let dot = (0..l.lanes)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            want_sum += alphas[li][k] * dot;
        }
    }
    assert_eq!(*built.public.last().unwrap(), want_sum, "enforced_sum");

    // ---- prove / verify ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();
    let swap_r1cs = SwapTable::build_block_r1cs(nu);
    let swap_lc = swap_r1cs.csc_lincheck_circuit();
    let spread_ty = BitSpreadTable::new(max_depth);
    let spread_r1cs = spread_ty.build_block_r1cs(nu);
    let spread_lc = spread_r1cs.csc_lincheck_circuit();

    let t = Instant::now();
    let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
        (
            shape.registry_slot(slots.b3),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(
                    built.rows::<Blake3Gate>(slots.b3),
                    nu,
                ),
                b3_lc,
            ),
        ),
        (
            shape.registry_slot(slots.swap),
            UnionSlotProverInput::new(
                SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu),
                swap_lc,
            ),
        ),
        (
            shape.registry_slot(slots.spread),
            UnionSlotProverInput::new(
                spread_ty
                    .generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu),
                spread_lc,
            ),
        ),
    ];
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;

    bool_slots.sort_by_key(|(i, _)| *i);
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape.registry_slot(slots.b3), b3_lc),
        (shape.registry_slot(slots.swap), swap_lc),
        (shape.registry_slot(slots.spread), spread_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
        .iter()
        .zip(els)
        .map(|((_, s), z)| (shape.registry_slot(*s), z))
        .collect();
    el_ord.sort_by_key(|(i, _)| *i);
    let el_inputs: Vec<UnionElementSlotInput> = el_ord
        .into_iter()
        .map(|(_, z)| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z)))
        .collect();

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        bool_slots.into_iter().map(|(_, s)| s).collect(),
        el_inputs,
        &mut c,
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut c,
    )
    .expect("the collapsed query phase verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    let nnz = |r: &flock_core::r1cs::BlockR1cs| {
        r.a_0.rows.iter().map(|x| x.len()).sum::<usize>()
            + r.b_0.rows.iter().map(|x| x.len()).sum::<usize>()
    };
    println!(
        "\nMVP-6 FULL QUERY PHASE, COLLAPSED (m=26 Fast ladder)\n  \
         blake3 {} rows | swap {} | spread {} | leaf-eval {}+{}\n  \
         lincheck nnz {} (MVP-5: 105145720) | dense {} words | dense_m {} | \
         M_bool {} | mu {}\n\n  \
         PER PROOF     {:6.0} ms = online {online_ms:.0} + witgen {wit_ms:.0} + \
         prove {prove_ms:.0}\n  \
         verifier side {verify_ms:6.1} ms | proof {:.1} KiB | {threads} threads\n  \
         SETUP         {setup_ms:6.0} ms\n",
        shape.counts[shape.registry_slot(slots.b3)],
        shape.counts[shape.registry_slot(slots.swap)],
        shape.counts[shape.registry_slot(slots.spread)],
        shape.counts[shape.registry_slot(leaf_slot[0].1)],
        shape.counts[shape.registry_slot(leaf_slot[1].1)],
        nnz(&b3) + nnz(&swap_r1cs) + nnz(&spread_r1cs),
        union.dense_words(),
        union.dense_m(),
        union.m_bool(),
        shape.circuit.cells().mu(),
        online_ms + wit_ms + prove_ms,
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}
