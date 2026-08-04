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
    ///
    /// Real-protocol paths are CAPPED since Merkle capping landed: they are
    /// `d − c` deep and the index's high `c` bits select a node of the
    /// absorbed cap layer rather than folding to a root. The COLLAPSED path
    /// models this (`emit_opening` + the boundary select in `mvp6`): the
    /// select is done by the checker on published words, so no mux gadget
    /// exists in-circuit. This COMPOSITE gate stays full-depth against its
    /// synthetic trees, where the assert above IS the index-binding
    /// argument; it is the uncapped differential oracle, not the protocol.
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
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        slots.into_iter().map(|(_, s)| s).collect(),
        Vec::new(),
        &mut ch,
    );

    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
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
        verifier::verify_ligerito_union_circuit(
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
        prover::prove_fast_ligerito_union_circuit(
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
    verifier::verify_ligerito_union_circuit(
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
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
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
    verifier::verify_ligerito_union_circuit(
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
        verifier::verify_ligerito_union_circuit(
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
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
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
    verifier::verify_ligerito_union_circuit(
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
        verifier::verify_ligerito_union_circuit(
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

/// Median wall-clock of `reps` timed runs after one discarded warm-up, plus
/// the spread.
///
/// Single-shot timings on these tests vary by 15-20% — more than several of
/// the effects being compared — so a lone number is not evidence. Every
/// figure quoted from `mvp5`/`mvp6` should come from here.
#[derive(Clone, Copy)]
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}

impl std::fmt::Display for Timing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.0} [{:.0}-{:.0}]", self.median, self.min, self.max)
    }
}

/// Timed repetitions per phase. Five is enough to see the spread without
/// making an `#[ignore]`d test tedious.
const REPS: usize = 5;

fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, Timing) {
    let mut out = f(); // warm-up, discarded
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        out = f();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        out,
        Timing {
            median: ms[ms.len() / 2],
            min: ms[0],
            max: ms[ms.len() - 1],
        },
    )
}

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
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

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

    // Witnesses once; each prove rep rebuilds its slot inputs from CLONES,
    // outside the timer, because `UnionSlotProverInput::new` consumes them.
    let (hash_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu)
    });
    let merkle_wit: Vec<_> = (0..levels.len())
        .map(|li| {
            layouts[li].generate_witness_batch_major_partial_chunk(
                built.rows::<MerklePathGate>(merkle[li]),
                nu,
            )
        })
        .collect();
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> =
        vec![(shape.registry_slot(hash), b3_lc)];
    for (li, _) in levels.iter().enumerate() {
        lcs_ord.push((shape.registry_slot(merkle[li]), &walkers[li]));
    }
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((proof, commitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![(
            shape.registry_slot(hash),
            UnionSlotProverInput::new(hash_wit.clone(), b3_lc),
        )];
        for (li, _) in levels.iter().enumerate() {
            bool_slots.push((
                shape.registry_slot(merkle[li]),
                UnionSlotProverInput::new(merkle_wit[li].clone(), &walkers[li]),
            ));
        }
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape.counts[i], nu))
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (proof, commitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .expect("the full query phase verifies")
    });


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
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         MERGED OPEN   frobenius {:.1} KiB, {} rounds | {} gather claims\n  \
         SETUP         {setup_ms:6.0} ms\n",
        union.dense_words(),
        union.dense_m(),
        union.m_bool(),
        union.m_total(),
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        bincode::serialize(&proof.pcs_open.frobenius).map(|b| b.len()).unwrap_or(0) as f64
            / 1024.0,
        proof.pcs_open.merged_rounds.len(),
        shape.circuit.cells().num_gate_slots(),
    );
    println!("{report}");
}

// ---------------------------------------------------------------------------
// The collapsed opening: wiring over ONE BLAKE3 table
// ---------------------------------------------------------------------------
// MVP-7: the query phase of a REAL inner proof
// ---------------------------------------------------------------------------

/// A shared-constant public: one public input PER DISTINCT VALUE, wired to
/// every use through copy constraints — the `zw`/`ow` pattern generalized.
/// The per-row structural words (params, zero pads) collapse from one
/// public per ROW to one per VALUE; being few and public they are also the
/// auditable surface the checker contract pins (the fixed-shape statement).
fn cw(sb: &mut ShapeBuilder, vals: &mut Vec<F128>, consts: &mut Vec<(F128, Wire)>, v: F128) -> Wire {
    match consts.iter().find(|&&(x, _)| x == v) {
        Some(&(_, w)) => w,
        None => {
            vals.push(v);
            let w = sb.public_input();
            consts.push((v, w));
            w
        }
    }
}

/// Which byte payloads of a tape stay PUBLIC under the witness/public
/// split: every `observe_bytes` payload — the STATEMENT surfaces (registry
/// digest, counts, caps, a child's circuit digest + public words) and
/// nothing else. PoW nonces share the payload counter but are witness (their
/// wires publish separately where the grinding checker reads them).
fn bytes_payload_mask(ops: &[flock_core::transcript_record::TranscriptOp]) -> Vec<bool> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut v = Vec::new();
    for op in ops {
        match op {
            Op::ObserveBytes(_) => v.push(true),
            Op::Pow { .. } => v.push(false),
            _ => {}
        }
    }
    v
}

/// Replay a recorded transcript's FS chain into the blake3 slot; squeeze
/// rows chain off prior outputs. Returns the per-row output wires
/// (`trace.squeezes[fin]` indexes into them) and the per-stream-word wires.
///
/// **The witness/public split** (the recursion-composition fix): the child
/// PROOF BODY is existentially quantified — its stream words enter as
/// WITNESS inputs, bound in-circuit by the chain compressions and the
/// region gates that consume them, never read natively. What stays public:
/// the byte payloads `pub_payloads` selects (the STATEMENT: digests,
/// counts, caps — the caps' wires also feed the in-circuit cap trees the
/// openings connect to), domain constants, and the shared structural
/// constants through `consts`.
fn emit_fs_chain(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    trace: &flock_prover::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_core::transcript_record::Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    pub_payloads: &[bool],
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    use flock_core::transcript_record::StreamWord;
    use flock_prover::r1cs_hashes::fs_chain::CvSource;
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = cw(sb, vals, consts, pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(b3, &g_in));
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
                        cw(sb, vals, consts, F128::ZERO)
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
                                let public = match &stream.words[wi] {
                                    StreamWord::Bytes { payload, .. } => {
                                        pub_payloads.get(*payload).copied().unwrap_or(true)
                                    }
                                    StreamWord::Const(_) => true,
                                    StreamWord::Value(_) => false,
                                };
                                let w = if public { sb.public_input() } else { sb.input() };
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
        outs.push(sb.gate(b3, &g_in));
    }
    (outs, word_wire)
}

/// The sumcheck-spine gate: one fold-and-eval step of the verifier's running
/// quadratic, `RoundQuad` in circuit form (char-2, so `u1 = t + u0` is the
/// linear coefficient trick):
///
///   c' = c + beta*u0     b' = b + beta*(y + u2)     a' = a + beta*u2
///   tr' = tr + beta*y    t' = c' + r*b' + r^2*a'
///
/// Three degenerate uses cover every verifier step with ONE table type:
/// BUILD `from_msg` (zero quad in, beta = 1, y = the running target),
/// EVAL a held quad (beta = 0; only t' consumed), and INTRO-FOLD an OOD or
/// enforced-sum claim (consume c', b', a', tr'; t' unwired).
struct SpineGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

const SP_IN: usize = 9; // c b a tr u0 u2 y beta r
const SP_K: usize = 21;

impl SpineGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let (c, b, a, tr, u0, u2, y, beta, r) = (0, 1, 2, 3, 4, 5, 6, 7, 8);
        let (pc, pb, pa, pt, co, bo, ao, tro, r2, m1, m2, to) =
            (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let mut bld = ElementTableBuilder::new(5);
        for w in 0..SP_IN {
            bld.free_wire(w);
        }
        bld.mult(pc, beta, u0)
            .mult_lin(pb, &[(y, one), (u2, one)], &[(beta, one)])
            .mult(pa, beta, u2)
            .mult(pt, beta, y)
            .linear(co, &[(c, one), (pc, one)])
            .linear(bo, &[(b, one), (pb, one)])
            .linear(ao, &[(a, one), (pa, one)])
            .linear(tro, &[(tr, one), (pt, one)])
            .mult(r2, r, r)
            .mult(m1, r, bo)
            .mult(m2, r2, ao)
            .linear(to, &[(co, one), (m1, one), (m2, one)]);
        Self {
            ty: std::sync::Arc::new(bld.build().expect("spine gate is valid")),
        }
    }
}

impl GateType for SpineGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..SP_IN).map(IoWord::input).collect();
        for o in [13, 14, 15, 16, 20] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; SP_K];
        z[..SP_IN].copy_from_slice(&inputs[..SP_IN]);
        let (c, b, a, tr, u0, u2, y, beta, r) =
            (z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8]);
        z[9] = beta * u0;
        z[10] = (y + u2) * beta;
        z[11] = beta * u2;
        z[12] = beta * y;
        z[13] = c + z[9];
        z[14] = b + z[10];
        z[15] = a + z[11];
        z[16] = tr + z[12];
        z[17] = r * r;
        z[18] = r * z[14];
        z[19] = z[17] * z[15];
        z[20] = z[13] + z[18] + z[19];
        (vec![z[13], z[14], z[15], z[16], z[20]], z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The residual-basis gate (step 2b): one query's contribution to a level's
/// `induce_sumcheck_evaluate_at_residual`, at every residual position `y`.
///
/// From `q_field` the novel-basis chain runs `s_{k+1} = s_k (s_k + c_k)`
/// (`c_k = s_k(v_k)`, a constant; the `1/s_k(v_k)` normalizations fold into
/// downstream weights). The level's post-intro fold challenges `ris` build
/// `prefix = prod_k (1 + ris_k (1 + W_k))`, the suffix `W`s form subset
/// products over the `2^yr` residual positions (`1 + p_j(1+w) = w` iff the
/// bit is set), and `aw * prefix * subset(y)` accumulates into `2^yr` running
/// sums. One gate row per (level, query); the accumulators chain across
/// queries like `LeafEvalGate`'s.
///
/// `q_field` is a public input, bound at the boundary: the checker masks the
/// (already published) challenge word natively — same pattern as the cap
/// select.
/// Smallest kappa whose column budget holds `c_need` (floored at MacGate's
/// 3). Tight envelopes matter: the element region's size is the sum of
/// 2^kappa envelopes rounded to a power of two, and the union's column
/// domain (claim-point lengths, the eq-dot loops, run counts) follows it.
fn gate_kappa(c_need: usize) -> usize {
    assert!(c_need <= 256, "gate spills kappa=8 ({c_need} cols)");
    (c_need.next_power_of_two().trailing_zeros() as usize).max(3)
}

struct ResidualGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    sks_vks: Vec<F128>,
    acc_out: Vec<usize>,
    lmc: usize,
    pl: usize,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl ResidualGate {
    /// Column layout, in declaration order:
    ///   q(0), ris(1..=pl), aw, one, acc_in[yr]          — inputs (n_in)
    ///   s_1..s_{lmc-1}                                   — the chain
    ///   pk_0..pk_{pl-1}, pr_1..pr_{pl-1}                 — prefix terms/products
    ///   sp for each y with >=2 bits                      — subset products
    ///   t = aw*prefix (pl>0 only), c_y (y>0), acc_out[yr] — the contributions
    ///
    /// The normalized suffix factors W_j = s_{pl+j}(q)/s_{pl+j}(v) have no
    /// cells of their own: each use is fused into a product as a mult_lin
    /// side `(s_col, inv)` — the envelope program.
    fn new(log_msg_cols: usize, prefix_len: usize, yr_log_n: usize, sks_vks: &[F128]) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one_w = F128::ONE;
        let (lmc, pl, yl) = (log_msg_cols, prefix_len, yr_log_n);
        assert_eq!(pl + yl, lmc);
        let yr = 1usize << yl;
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let n_in = 1 + pl + 1 + 1 + yr;
        let (q, aw, one, acc0) = (0usize, 1 + pl, 2 + pl, 3 + pl);
        // Total columns, counted up front so kappa ADAPTS: inputs, the
        // chain (lmc−1), prefix (2·pl), multi-bit subset products
        // (yr−1−yl), t only when pl>0, contributions (yr−1) and
        // accumulators (yr). Sums to 4·pl + 4·yr + [pl>0].
        let c_need = 4 * pl + 4 * yr + usize::from(pl > 0);
        let kappa = gate_kappa(c_need);
        let mut c = n_in; // next free column
        let mut b = ElementTableBuilder::new(kappa);
        for wcol in 0..n_in {
            b.free_wire(wcol);
        }
        // s columns: s_col[k] holds s_k(q); s_0 IS the q input column.
        let mut s_col = vec![q];
        for k in 1..lmc {
            b.mult_lin(
                c,
                &[(s_col[k - 1], one_w)],
                &[(s_col[k - 1], one_w), (one, sks_vks[k - 1])],
            );
            s_col.push(c);
            c += 1;
        }
        // prefix: pk = ris_k * (1 + W_k), pr = running product of (1 + pk).
        let mut pr = one; // empty product
        for k in 0..pl {
            let ivk = inv(sks_vks[k]);
            b.mult_lin(c, &[(1 + k, one_w)], &[(one, one_w), (s_col[k], ivk)]);
            let pk = c;
            c += 1;
            b.mult_lin(c, &[(pr, one_w)], &[(one, one_w), (pk, one_w)]);
            pr = c;
            c += 1;
        }
        // Suffix factor for bit j, as a fused mult_lin side (no cell).
        let wf = |j: usize| (s_col[pl + j], inv(sks_vks[pl + j]));
        // Subset products, multi-bit y only; y=0 is the empty product and
        // single-bit y is a fused factor at the point of use.
        let mut sp: Vec<Option<usize>> = vec![None; yr];
        for y in 1..yr {
            if !y.is_power_of_two() {
                let low = y & y.wrapping_neg();
                let jl = low.trailing_zeros() as usize;
                let rest = y ^ low;
                if rest.is_power_of_two() {
                    b.mult_lin(c, &[wf(rest.trailing_zeros() as usize)], &[wf(jl)]);
                } else {
                    b.mult_lin(c, &[(sp[rest].unwrap(), one_w)], &[wf(jl)]);
                }
                sp[y] = Some(c);
                c += 1;
            }
        }
        // t = aw * prefix (aw itself when the prefix is empty);
        // contributions and accumulators.
        let t = if pl > 0 {
            b.mult(c, aw, pr);
            c += 1;
            c - 1
        } else {
            aw
        };
        let mut acc_out = Vec::with_capacity(yr);
        for y in 0..yr {
            let cy = if y == 0 {
                t
            } else if y.is_power_of_two() {
                b.mult_lin(c, &[(t, one_w)], &[wf(y.trailing_zeros() as usize)]);
                c += 1;
                c - 1
            } else {
                b.mult(c, t, sp[y].unwrap());
                c += 1;
                c - 1
            };
            b.linear(c, &[(acc0 + y, one_w), (cy, one_w)]);
            acc_out.push(c);
            c += 1;
        }
        assert_eq!(c, c_need, "the residual column count is the counted one");
        Self {
            ty: std::sync::Arc::new(b.build().expect("residual gate is valid")),
            sks_vks: sks_vks.to_vec(),
            acc_out,
            lmc,
            pl,
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for ResidualGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &o in &self.acc_out {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        // A structural mirror of `new()`: same column cursor, same order.
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let (lmc, pl) = (self.lmc, self.pl);
        let yl = lmc - pl;
        let acc0 = 3 + pl;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut s_col = vec![0usize];
        for k in 1..lmc {
            z[c] = z[s_col[k - 1]] * (z[s_col[k - 1]] + self.sks_vks[k - 1]);
            s_col.push(c);
            c += 1;
        }
        let mut pr_v = F128::ONE;
        for k in 0..pl {
            z[c] = z[1 + k] * (F128::ONE + z[s_col[k]] * inv(self.sks_vks[k]));
            let pk = z[c];
            c += 1;
            z[c] = pr_v * (F128::ONE + pk);
            pr_v = z[c];
            c += 1;
        }
        let w_v: Vec<F128> = (0..yl)
            .map(|j| z[s_col[pl + j]] * inv(self.sks_vks[pl + j]))
            .collect();
        let mut sp: Vec<Option<usize>> = vec![None; self.yr];
        for y in 1..self.yr {
            if !y.is_power_of_two() {
                let low = y & y.wrapping_neg();
                let jl = low.trailing_zeros() as usize;
                let rest = y ^ low;
                z[c] = if rest.is_power_of_two() {
                    w_v[rest.trailing_zeros() as usize] * w_v[jl]
                } else {
                    z[sp[rest].unwrap()] * w_v[jl]
                };
                sp[y] = Some(c);
                c += 1;
            }
        }
        let t_v = if pl > 0 {
            z[c] = z[1 + pl] * pr_v;
            c += 1;
            z[c - 1]
        } else {
            z[1 + pl]
        };
        let mut outs = Vec::with_capacity(self.yr);
        for y in 0..self.yr {
            let cy = if y == 0 {
                t_v
            } else {
                z[c] = if y.is_power_of_two() {
                    t_v * w_v[y.trailing_zeros() as usize]
                } else {
                    t_v * z[sp[y].unwrap()]
                };
                c += 1;
                z[c - 1]
            };
            z[c] = z[acc0 + y] + cy;
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// An element slot input that writes only the LIVE row prefix of each
/// column. The destination block arrives freshly zeroed (element unions
/// never pool dirty buffers) and the source's dead words are zero by the
/// element closure contract, so skipping them is value-identical — the
/// witgen counterpart of the wiring layer's live gather fold. Copies
/// `count` rows of each of the `dst.len() >> nu` columns.
fn live_element_input(
    z: Vec<F128>,
    count: usize,
    nu: usize,
) -> flock_prover::prover::UnionElementSlotInput<'static> {
    flock_prover::prover::UnionElementSlotInput::new(move |dst: &mut [F128]| {
        debug_assert_eq!(dst.len(), z.len());
        let rows = 1usize << nu;
        if count >= rows {
            dst.copy_from_slice(&z);
            return;
        }
        let width = dst.len() >> nu;
        for col in 0..width {
            let base = col << nu;
            dst[base..base + count].copy_from_slice(&z[base..base + count]);
        }
    })
}

/// `s_{k+1}(x) = s_k(x)^2 + s_k(v_k) s_k(x)` — the subspace-polynomial chain,
/// and its `s_k(v_k)` constants (a replica of ligerito's pub(crate)
/// `eval_sk_at_vks`, pinned by the residual boundary check below).
fn sk_at_vks(log_n: usize) -> Vec<F128> {
    let next = |s: F128, c: F128| s * s + c * s;
    let mut sks = vec![F128::ZERO; log_n + 1];
    sks[0] = F128::ONE;
    if log_n == 0 {
        return sks;
    }
    let mut layer: Vec<F128> = (1..=log_n).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut cur = log_n;
    for i in 0..log_n {
        for j in 0..cur {
            let v = next(layer[j], sks[i]);
            if j == 0 {
                sks[i + 1] = v;
            } else {
                layer[j - 1] = v;
            }
        }
        cur -= 1;
    }
    sks
}

/// 2b stage 2, split into kappa<=6 gates (kappa=7 breaks the union's
/// column-split): PrefixGate computes `seed * prod_j (1 + a_j + b_j)` — the
/// char-2 eq prefix of a packed-direct claim (seed = gamma, a = point,
/// b = fold challenges) or an OOD claim (seed = beta, a = z). SuffixGate
/// tensors the point's tail over the 2^yr binary positions and accumulates;
/// PartialCombineGate folds `beta * resid` into the running combined vector;
/// FinalDotGate dots against the absorbed yr words.
struct PrefixGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    pl: usize,
    n_in: usize,
    k: usize,
}

impl PrefixGate {
    fn new(pl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let n_in = 2 + 2 * pl; // seed, a[pl], b[pl], one
        let one = n_in - 1;
        // FUSED: each factor is ONE mult_lin cell, pr' = pr·(1 + a + b) —
        // the B side is a linear combination (the envelope program).
        let c_need = n_in + pl;
        let kappa = gate_kappa(c_need);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(kappa);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut pr = 0;
        for j in 0..pl {
            bl.mult_lin(c, &[(pr, o)], &[(one, o), (1 + j, o), (1 + pl + j, o)]);
            pr = c;
            c += 1;
        }
        assert_eq!(c, c_need, "the prefix column count is the counted one");
        Self {
            ty: std::sync::Arc::new(bl.build().expect("prefix gate")),
            pl,
            n_in,
            k: c,
        }
    }
}

impl GateType for PrefixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.k - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut pr = z[0];
        for j in 0..self.pl {
            z[c] = pr * (F128::ONE + z[1 + j] + z[1 + self.pl + j]);
            pr = z[c];
            c += 1;
        }
        (vec![pr], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct SuffixGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    acc_out: Vec<usize>,
    yl: usize,
    n_in: usize,
    k: usize,
}

impl SuffixGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 2 + yl + yr; // p, ptS[yl], one, acc[yr]
        let (pt0, one, acc0) = (1, 1 + yl, 2 + yl);
        let c_need = 2 * yl + 5 * yr;
        let kappa = gate_kappa(c_need);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(kappa);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut e = vec![one];
        for j in 0..yl {
            bl.linear(c, &[(one, o), (pt0 + j, o)]);
            let neg = c;
            c += 1;
            let mut nx = Vec::new();
            for &pv in &e {
                bl.mult(c, pv, neg);
                nx.push(c);
                c += 1;
            }
            for &pv in &e {
                bl.mult(c, pv, pt0 + j);
                nx.push(c);
                c += 1;
            }
            e = nx;
        }
        let mut acc_out = Vec::new();
        for (y, &ey) in e.iter().enumerate() {
            bl.mult(c, 0, ey);
            c += 1;
            bl.linear(c, &[(acc0 + y, o), (c - 1, o)]);
            acc_out.push(c);
            c += 1;
        }
        assert_eq!(c, c_need, "the suffix column count is the counted one");
        Self {
            ty: std::sync::Arc::new(bl.build().expect("suffix gate")),
            acc_out,
            yl,
            n_in,
            k: c,
        }
    }
}

impl GateType for SuffixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &oc in &self.acc_out {
            schema.push(IoWord::output(oc));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yl = self.yl;
        let (pt0, acc0) = (1, 2 + yl);
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut e = vec![F128::ONE];
        for j in 0..yl {
            z[c] = F128::ONE + z[pt0 + j];
            let neg = z[c];
            c += 1;
            let mut nx = Vec::new();
            for &pv in &e {
                z[c] = pv * neg;
                nx.push(z[c]);
                c += 1;
            }
            for &pv in &e {
                z[c] = pv * z[pt0 + j];
                nx.push(z[c]);
                c += 1;
            }
            e = nx;
        }
        let mut outs = Vec::new();
        for (y, &ey) in e.iter().enumerate() {
            z[c] = z[0] * ey;
            c += 1;
            z[c] = z[acc0 + y] + z[c - 1];
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct PartialCombineGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    acc_out: Vec<usize>,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl PartialCombineGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 1 + 2 * yr; // beta, acc[yr], resid[yr]
        let c_need = 1 + 4 * yr;
        let kappa = gate_kappa(c_need);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(kappa);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut acc_out = Vec::new();
        for y in 0..yr {
            bl.mult(c, 0, 1 + yr + y);
            c += 1;
            bl.linear(c, &[(1 + y, o), (c - 1, o)]);
            acc_out.push(c);
            c += 1;
        }
        Self {
            ty: std::sync::Arc::new(bl.build().expect("partial combine")),
            acc_out,
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for PartialCombineGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &oc in &self.acc_out {
            schema.push(IoWord::output(oc));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yr = self.yr;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut outs = Vec::new();
        for y in 0..yr {
            z[c] = z[0] * z[1 + yr + y];
            c += 1;
            z[c] = z[1 + y] + z[c - 1];
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct FinalDotGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl FinalDotGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 2 * yr; // yr words, combined
        let c_need = 3 * yr + 1;
        let kappa = gate_kappa(c_need);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(kappa);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut terms = Vec::new();
        for y in 0..yr {
            bl.mult(c, y, yr + y);
            terms.push((c, o));
            c += 1;
        }
        bl.linear(c, &terms);
        c += 1;
        Self {
            ty: std::sync::Arc::new(bl.build().expect("final dot")),
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for FinalDotGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.k - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yr = self.yr;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut inner = F128::ZERO;
        for y in 0..yr {
            z[c] = z[y] * z[yr + y];
            inner += z[c];
            c += 1;
        }
        z[c] = inner;
        (vec![inner], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One merged W-round of the verifier (`jagged::fold_round_claim`):
/// `t' = (t + g1) + (t + gi) r + gi r^2` — messages `(G(1), G(inf))` wire
/// from the absorbed stream, `r` from the chain squeeze; the chain of these
/// binds rho and carries the outer gamma-combination down to `running`.
struct MergedRoundGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl MergedRoundGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: t(0), g1(1), gi(2), r(3)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..4 {
            b.free_wire(w);
        }
        b.mult_lin(4, &[(0, o), (2, o)], &[(3, o)]); // (t+gi) r
        b.mult(5, 3, 3); // r^2
        b.mult(6, 5, 2); // gi r^2
        b.linear(7, &[(0, o), (1, o), (4, o), (6, o)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("merged round gate")),
        }
    }
}

impl GateType for MergedRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..4).map(IoWord::input).collect();
        schema.push(IoWord::output(7));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 8];
        z[..4].copy_from_slice(&inputs[..4]);
        z[4] = (z[0] + z[2]) * z[3];
        z[5] = z[3] * z[3];
        z[6] = z[5] * z[2];
        z[7] = z[0] + z[1] + z[4] + z[6];
        (vec![z[7]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// Multiply-accumulate: `out = acc + x·y` — the workhorse of the multipoint
/// intake (gamma-power chains, the `T0`/`V` sums, and zero-delta joins:
/// `mac(a, b, one) = a + b` is the char-2 equality delta).
struct MacGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl MacGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: acc(0), x(1), y(2)
        let mut b = ElementTableBuilder::new(3);
        for w in 0..3 {
            b.free_wire(w);
        }
        b.mult(3, 1, 2);
        b.linear(4, &[(0, o), (3, o)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("mac gate")),
        }
    }
}

impl GateType for MacGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..3).map(IoWord::input).collect();
        schema.push(IoWord::output(4));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 5];
        z[..3].copy_from_slice(&inputs[..3]);
        z[3] = z[1] * z[2];
        z[4] = z[0] + z[3];
        (vec![z[4]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One layer of the multipoint anchor's 4-state boundary DP (MVP-8 step 3;
/// oracle-tested standalone in `circuit_assist.rs` — this is the same gate,
/// both sourcing the transition table from `flock_core::pcs::jagged`).
struct AssistLayerGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

const AL_IN: usize = 9; // g0..g3, za, rb, rc, rd, one
const AL_OUT0: usize = 49;

impl AssistLayerGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let sparse = flock_core::pcs::jagged::assist_sparse_transitions();
        let mut b = ElementTableBuilder::new(6);
        for w in 0..AL_IN {
            b.free_wire(w);
        }
        b.mult(9, 4, 5)
            .linear(10, &[(8, one), (4, one), (5, one), (9, one)])
            .linear(11, &[(4, one), (9, one)])
            .linear(12, &[(5, one), (9, one)]);
        let eq4 = [10usize, 11, 12, 9];
        b.mult(13, 6, 7)
            .linear(14, &[(8, one), (6, one), (7, one), (13, one)])
            .linear(15, &[(6, one), (13, one)])
            .linear(16, &[(7, one), (13, one)]);
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                b.mult(p(i, o), eq4[i], o);
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                b.mult_lin(
                    33 + 4 * cd + s,
                    &[(p(i0, o0), one), (p(i1, o1), one)],
                    &[(e[cd], one)],
                );
            }
        }
        for s in 0..4 {
            b.linear(
                AL_OUT0 + s,
                &[(33 + s, one), (37 + s, one), (41 + s, one), (45 + s, one)],
            );
        }
        Self {
            ty: std::sync::Arc::new(b.build().expect("assist layer gate")),
        }
    }
}

impl GateType for AssistLayerGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..AL_IN).map(IoWord::input).collect();
        for s in 0..4 {
            schema.push(IoWord::output(AL_OUT0 + s));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let sparse = flock_core::pcs::jagged::assist_sparse_transitions();
        let mut z = vec![F128::ZERO; 53];
        z[..AL_IN].copy_from_slice(&inputs[..AL_IN]);
        let one = F128::ONE;
        z[9] = z[4] * z[5];
        z[10] = one + z[4] + z[5] + z[9];
        z[11] = z[4] + z[9];
        z[12] = z[5] + z[9];
        let eq4 = [10usize, 11, 12, 9];
        z[13] = z[6] * z[7];
        z[14] = one + z[6] + z[7] + z[13];
        z[15] = z[6] + z[13];
        z[16] = z[7] + z[13];
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                z[p(i, o)] = z[eq4[i]] * z[o];
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                z[33 + 4 * cd + s] = z[e[cd]] * (z[p(i0, o0)] + z[p(i1, o1)]);
            }
        }
        for s in 0..4 {
            z[AL_OUT0 + s] = z[33 + s] + z[37 + s] + z[41 + s] + z[45 + s];
        }
        (z[AL_OUT0..AL_OUT0 + 4].to_vec(), z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One element-zerocheck round (degree-3, convention A): `g0` rides as
/// ADVICE (a public input) and the gate enforces its defining identity as a
/// published-zero delta — the family-I pattern, no in-circuit inversion:
///
///   delta = g0 (1+t) + running + t g1          (must be 0)
///   running' = g0 (1+rho) + g1 rho + g_inf rho (1+rho)
struct ZcRoundGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl ZcRoundGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: running(0) g1(1) gi(2) t(3) rho(4) g0(5) one(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult_lin(7, &[(5, o)], &[(6, o), (3, o)]); // g0(1+t)
        b.mult(8, 3, 1); // t g1
        b.linear(9, &[(7, o), (0, o), (8, o)]); // delta
        b.mult_lin(10, &[(5, o)], &[(6, o), (4, o)]); // g0(1+rho)
        b.mult(11, 1, 4); // g1 rho
        b.mult_lin(12, &[(4, o)], &[(6, o), (4, o)]); // rho(1+rho)
        b.mult(13, 2, 12); // gi rho(1+rho)
        b.linear(14, &[(10, o), (11, o), (13, o)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("zc round gate")),
        }
    }
}

impl GateType for ZcRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        schema.push(IoWord::output(9));
        schema.push(IoWord::output(14));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 15];
        z[..7].copy_from_slice(&inputs[..7]);
        z[7] = z[5] * (z[6] + z[3]);
        z[8] = z[3] * z[1];
        z[9] = z[7] + z[0] + z[8];
        z[10] = z[5] * (z[6] + z[4]);
        z[11] = z[1] * z[4];
        z[12] = z[4] * (z[6] + z[4]);
        z[13] = z[2] * z[12];
        z[14] = z[10] + z[11] + z[13];
        (vec![z[9], z[14]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One Λ-node of the univariate skip's interpolation (the family-H pass,
/// first item): the barycentric NUMERATOR recurrence
///
///   A' = A·(z+λ),   B' = B·(z+λ) + v·A
///
/// accumulates `B_final = Σ_i v_i · Π_{j≠i}(z+λ_j)` in one forward pass —
/// no inversions, no advice, and (unlike the native closed form's
/// `Z(z)/(z+λ_i)` branch) EXACT even when z lands on a node. Both
/// interpolations (`round1_c` alone and the combined `ab+c`) share the
/// node products, so one row carries both accumulators.
struct SkipNodeGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl SkipNodeGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: A(0) bc(1) bab(2) z(3) lam(4) vc(5) vab(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult_lin(7, &[(0, o)], &[(3, o), (4, o)]); // A' = A (z+lam)
        b.mult_lin(8, &[(1, o)], &[(3, o), (4, o)]); // bc (z+lam)
        b.mult(9, 5, 0); // vc A
        b.linear(10, &[(8, o), (9, o)]); // bc'
        b.mult_lin(11, &[(2, o)], &[(3, o), (4, o)]); // bab (z+lam)
        b.mult_lin(12, &[(6, o), (5, o)], &[(0, o)]); // (vab+vc) A
        b.linear(13, &[(11, o), (12, o)]); // bab'
        Self {
            ty: std::sync::Arc::new(b.build().expect("skip node gate")),
        }
    }
}

impl GateType for SkipNodeGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        schema.push(IoWord::output(7));
        schema.push(IoWord::output(10));
        schema.push(IoWord::output(13));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 14];
        z[..7].copy_from_slice(&inputs[..7]);
        let zl = z[3] + z[4];
        z[7] = z[0] * zl;
        z[8] = z[1] * zl;
        z[9] = z[5] * z[0];
        z[10] = z[8] + z[9];
        z[11] = z[2] * zl;
        z[12] = (z[6] + z[5]) * z[0];
        z[13] = z[11] + z[12];
        (vec![z[7], z[10], z[13]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The skip round's close-out (one row): scales the two numerator sums by
/// the subspace-denominator inverses and the linearized `Z_S(z)`.
///
/// The φ8 node sets are F₂-subspaces, so `Z_S(X) = Σ_j c_j·X^(2^j)` is
/// LINEARIZED — its evaluation is a constant-coefficient combination of the
/// squaring-chain wires `z^(2^j)`, free inside the row — and the Lagrange
/// denominator is the same constant for every node (it is `Z'_S = c_0`, the
/// formal derivative). Outputs: `rc = interpolate_on_lambda(round1_c)(z)` —
/// which BINDS `final_c_eval` (never absorbed; the native verifier
/// recomputes-and-compares, so the published wire is its binding) — and
/// `seed = combined_at_z + rc`, the multilinear chain's entry, replacing
/// the zc-seed advice.
struct SkipCloseGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl SkipCloseGate {
    /// Linearized coefficients of `Z_{V_m}` over `V_m = φ_8({0..2^m−1})` via
    /// the subspace-polynomial recursion `Z_{k+1} = Z_k² + Z_k(φ(2^k))·Z_k`
    /// (squaring shifts the coefficient basis: `(Σ c_j X^(2^j))² =
    /// Σ c_j²·X^(2^(j+1))`).
    fn linearized_coeffs(dim: usize) -> Vec<F128> {
        use flock_core::field::PHI_8_TABLE;
        let mut c = vec![F128::ONE];
        for k in 0..dim {
            let bk = PHI_8_TABLE[1 << k];
            let mut t = F128::ZERO;
            let mut p = bk;
            for &cj in &c {
                t += cj * p;
                p = p * p;
            }
            let mut nc = vec![F128::ZERO; c.len() + 1];
            for (j, &cj) in c.iter().enumerate() {
                nc[j + 1] += cj * cj;
                nc[j] += t * cj;
            }
            c = nc;
        }
        c
    }

    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        use flock_core::field::PHI_8_TABLE;
        let o = F128::ONE;
        let c = Self::linearized_coeffs(6);
        // Z_{V6} vanishes on V6 and matches the product form off it; the
        // denominator is its formal derivative c_0. One-time construction
        // checks — the constants below are load-bearing for soundness.
        let zv6 = |x: F128| {
            let mut acc = F128::ZERO;
            let mut p = x;
            for &cj in &c {
                acc += cj * p;
                p = p * p;
            }
            acc
        };
        assert_eq!(zv6(PHI_8_TABLE[1]), F128::ZERO, "Z_V6 vanishes on V6");
        assert_eq!(zv6(PHI_8_TABLE[63]), F128::ZERO, "Z_V6 vanishes on V6");
        let den6 = (1..64).fold(F128::ONE, |acc, i| acc * PHI_8_TABLE[i]);
        assert_eq!(c[0], den6, "den6 is Z_V6's formal derivative");
        let den7 = zv6(PHI_8_TABLE[64]) * den6;
        // in: bc(0) bab(1) zp_j = z^(2^j) at (2+j) for j in 0..7
        let mut b = ElementTableBuilder::new(4);
        for w in 0..9 {
            b.free_wire(w);
        }
        let zs: Vec<(usize, F128)> = c
            .iter()
            .enumerate()
            .filter(|&(_, &cj)| cj != F128::ZERO)
            .map(|(j, &cj)| (2 + j, cj))
            .collect();
        b.mult_lin(9, &[(1, den7.inv())], &zs); // combined_at_z
        b.linear(10, &[(0, den6.inv())]); // rc
        b.linear(11, &[(9, o), (10, o)]); // seed
        Self {
            ty: std::sync::Arc::new(b.build().expect("skip close gate")),
        }
    }
}

impl GateType for SkipCloseGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..9).map(IoWord::input).collect();
        schema.push(IoWord::output(10));
        schema.push(IoWord::output(11));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        use flock_core::field::PHI_8_TABLE;
        let c = Self::linearized_coeffs(6);
        let den6 = c[0];
        let mut t6 = F128::ZERO;
        let mut p = PHI_8_TABLE[64];
        for &cj in &c {
            t6 += cj * p;
            p = p * p;
        }
        let mut z = vec![F128::ZERO; 12];
        z[..9].copy_from_slice(&inputs[..9]);
        let zs = c
            .iter()
            .enumerate()
            .fold(F128::ZERO, |acc, (j, &cj)| acc + cj * z[2 + j]);
        z[9] = (t6 * den6).inv() * z[1] * zs;
        z[10] = den6.inv() * z[0];
        z[11] = z[9] + z[10];
        (vec![z[10], z[11]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The zerocheck's closing identity `running = ea eb + ec` (published-zero
/// delta) and the constant-strip + lincheck entry: with the inner shape's
/// single kappa=2 slot at full prefix, `va = ea + <eq(r_con), a_const>` and
/// likewise `vb` — the four eq-tensor weights over the last two zerocheck
/// challenges, with the table's affine constants baked as weights.
struct ZcJoinGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    ac: Vec<F128>,
    bc: Vec<F128>,
}

impl ZcJoinGate {
    fn new(a_const: &[F128], b_const: &[F128]) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        assert_eq!(a_const.len(), 4, "kappa=2 slot");
        // in: running(0) ea(1) eb(2) ec(3) r0(4) r1(5) one(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult(7, 1, 2); // ea eb
        b.linear(8, &[(0, o), (7, o), (3, o)]); // delta
        b.mult_lin(9, &[(6, o), (4, o)], &[(6, o), (5, o)]); // (1+r0)(1+r1)
        b.mult_lin(10, &[(4, o)], &[(6, o), (5, o)]); // r0(1+r1)
        b.mult_lin(11, &[(6, o), (4, o)], &[(5, o)]); // (1+r0)r1
        b.mult(12, 4, 5); // r0 r1
        // The builder rejects zero coefficients — free-wire columns have
        // zero affine constants, so filter.
        let mut ta = vec![(1usize, o)];
        let mut tb = vec![(2usize, o)];
        for (c, (&wa, &wb)) in a_const.iter().zip(b_const).enumerate() {
            if wa != F128::ZERO {
                ta.push((9 + c, wa));
            }
            if wb != F128::ZERO {
                tb.push((9 + c, wb));
            }
        }
        b.linear(13, &ta); // va
        b.linear(14, &tb); // vb
        Self {
            ty: std::sync::Arc::new(b.build().expect("zc join gate")),
            ac: a_const.to_vec(),
            bc: b_const.to_vec(),
        }
    }
}

impl GateType for ZcJoinGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        for o in [8, 13, 14] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 15];
        z[..7].copy_from_slice(&inputs[..7]);
        z[7] = z[1] * z[2];
        z[8] = z[0] + z[7] + z[3];
        z[9] = (z[6] + z[4]) * (z[6] + z[5]);
        z[10] = z[4] * (z[6] + z[5]);
        z[11] = (z[6] + z[4]) * z[5];
        z[12] = z[4] * z[5];
        z[13] = z[1]
            + z[9] * self.ac[0]
            + z[10] * self.ac[1]
            + z[11] * self.ac[2]
            + z[12] * self.ac[3];
        z[14] = z[2]
            + z[9] * self.bc[0]
            + z[10] * self.bc[1]
            + z[11] * self.bc[2]
            + z[12] * self.bc[3];
        (vec![z[8], z[13], z[14]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One packed-direct claim on the tape: its absorbed point/value and gamma.
struct PdRec {
    pt_v: usize,
    pt_len: usize,
    val_v: usize,
    fin: usize,
    ch: usize,
}

/// One merged W-round: the (G(1), G(inf)) value index and the rho squeeze.
#[derive(Clone)]
struct RoundRec {
    g_v: usize,
    fin: usize,
    ch: usize,
}

/// The inner ligerito intake's single claim: q_eval's value index + gamma'.
struct InnerPd {
    q_v: usize,
    fin: usize,
    ch: usize,
}

/// The element PIOP on the tape: tau, the zerocheck rounds, (ea, eb, ec),
/// alpha, and the lincheck rounds.
struct PiopRec {
    tau_fin: usize,
    tau_ch: usize,
    tau_len: usize,
    zc_rounds: Vec<RoundRec>,
    eab_v: usize,
    alpha_fin: usize,
    alpha_ch: usize,
    lc_rounds: Vec<RoundRec>,
}

/// One OOD claim on the tape: where its `z` squeezed, its `y`/intro-msg
/// values, and its beta.
struct OodRec {
    z_fin: usize,
    z_ch: usize,
    z_len: usize,
    y_v: usize,
    intro_v: usize,
    beta_fin: usize,
    beta_ch: usize,
}

/// The multipoint region of the merged open, located on the tape (MVP-8):
/// the group values' absorb, the batching gamma, the two-product sumcheck
/// rounds, and the anchor assist's `v` + rounds. For a pure-element inner
/// (R = 0) the RS values are absent and the sumcheck is the single
/// untwisted product — see docs/multipoint-twisted-assist.tex.
struct MpRec {
    /// Value indices of the P group values `B_k` (stream-wireable).
    val_vs: Vec<usize>,
    /// The batching gamma squeeze: `(fin, ch)`.
    gamma_fin: usize,
    gamma_ch: usize,
    /// The m two-product sumcheck rounds.
    rounds: Vec<RoundRec>,
    /// The anchor's claimed twisted evaluation `v` (value index).
    anchor_v: usize,
    /// The anchor's `2(m + 1)` rounds.
    anchor_rounds: Vec<RoundRec>,
}

/// One open-phase level, located on a recorded op tape. `*_fin` are finalize
/// ordinals (indices into `FsChainTrace::squeezes`); `*_ch` index into
/// `RecordingChallenger::challenges()`.
struct OpenLevel {
    fold_fins: Vec<usize>,
    fold_chs: Vec<usize>,
    /// Value index of each fold round's message `u_0` (`u_2` is `+1`).
    fold_msg_vs: Vec<usize>,
    /// OOD claims folded before this level's queries.
    ood: Vec<OodRec>,
    /// The level's intro message value idx (unused for the final level).
    intro_v: usize,
    /// The intro/final beta: `(fin, ch)`.
    beta_fin: usize,
    beta_ch: usize,
    q_fin: usize,
    q_ch: usize,
    q_count: usize,
    a_fin: usize,
    a_ch: usize,
    a_count: usize,
}

/// Walk the recorded transcript ops and locate every open-phase squeeze the
/// circuit needs. The walk MIRRORS the succinct verifier's structure (folds,
/// cap absorbs, OOD groups, PoW, queries, alpha, beta per level), asserting
/// each op kind — a config change that moves the shape fails here, loudly,
/// not as a wrong wire.
#[allow(clippy::type_complexity)]
fn parse_open_levels(
    ops: &[flock_core::transcript_record::TranscriptOp],
    cap0_bytes: usize,
    r: usize,
) -> (
    usize,
    Option<PiopRec>,
    Vec<PdRec>,
    Vec<RoundRec>,
    MpRec,
    InnerPd,
    usize,
    Vec<OpenLevel>,
) {
    use flock_core::transcript_record::TranscriptOp as Op;
    struct Cur<'a> {
        ops: &'a [Op],
        i: usize,
        fin: usize,
        ch: usize,
        v: usize,
    }
    impl Cur<'_> {
        fn bump(&mut self) {
            let op = &self.ops[self.i];
            if op.finalizes() {
                self.fin += 1;
            }
            match op {
                Op::SqueezeScalar => self.ch += 1,
                Op::SqueezeSlice(n) => self.ch += n,
                Op::ObserveScalar => self.v += 1,
                Op::ObserveSlice(n) => self.v += n,
                _ => {}
            }
            self.i += 1;
        }
        fn expect_obs_scalar(&mut self) {
            assert!(
                matches!(self.ops[self.i], Op::ObserveScalar),
                "op {}: expected ObserveScalar, got {:?}",
                self.i,
                self.ops[self.i]
            );
            self.bump();
        }
    }

    // Open start: the LAST ObserveBytes of the L0 cap's size —
    // `bind_statement` absorbs the same cap once, earlier.
    let starts: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, Op::ObserveBytes(n) if *n == cap0_bytes))
        .map(|(i, _)| i)
        .collect();
    assert!(starts.len() >= 2, "expected bind + open cap absorbs");
    let start = *starts.last().unwrap();
    // The merged intake absorbs, per packed-direct claim, [point slice,
    // value, gamma squeeze] right after the merged-open label — so the claim
    // POINTS are stream words (wireable), and each gamma is a scalar squeeze.
    let mut cur = Cur { ops, i: 0, fin: 0, ch: 0, v: 0 };
    let mut gammas: Vec<PdRec> = Vec::new();
    let mut rounds: Vec<RoundRec> = Vec::new();
    let mut mp: Option<MpRec> = None;
    let mut inner_pd: Option<InnerPd> = None;
    let mut piop: Option<PiopRec> = None;
    let mut in_pd = false;
    while cur.i < start {
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-zc-v0") {
            cur.bump();
            let (tau_fin, tau_ch, tau_len) = match ops[cur.i] {
                Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
                ref o => panic!("tau, got {o:?}"),
            };
            cur.bump();
            let mut zc_rounds = Vec::with_capacity(tau_len);
            for _ in 0..tau_len {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "zc rho");
                zc_rounds.push(RoundRec { g_v, fin: cur.fin, ch: cur.ch });
                cur.bump();
            }
            let eab_v = cur.v;
            cur.expect_obs_scalar(); // ea
            cur.expect_obs_scalar(); // eb
            cur.expect_obs_scalar(); // ec
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-lc-v0"),
                "lc label"
            );
            cur.bump();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "alpha");
            let (alpha_fin, alpha_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut lc_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "lc rho");
                lc_rounds.push(RoundRec { g_v, fin: cur.fin, ch: cur.ch });
                cur.bump();
            }
            piop = Some(PiopRec {
                tau_fin,
                tau_ch,
                tau_len,
                zc_rounds,
                eab_v,
                alpha_fin,
                alpha_ch,
                lc_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            in_pd = true;
            cur.bump();
            continue;
        }
        if in_pd {
            // Ring-switched claims front the intake on boolean-bearing
            // tapes: [label, s_hat_v slice, r_dprime slice] each, then the
            // bare gamma squeezes — walk over them (mvp9 pins them
            // separately).
            if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
                cur.bump(); // label
                cur.bump(); // s_hat_v slice
                cur.bump(); // r_dprime slice
                continue;
            }
            if matches!(ops[cur.i], Op::SqueezeScalar) {
                // an rs gamma — bare squeeze, no absorb
                cur.bump();
                continue;
            }
            if let Op::ObserveSlice(n) = ops[cur.i] {
                let pt_v = cur.v;
                cur.bump();
                let val_v = cur.v;
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "pd gamma");
                gammas.push(PdRec {
                    pt_v,
                    pt_len: n,
                    val_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
                continue;
            }
            in_pd = false;
            // The merged W-rounds follow the intake immediately: one
            // [ObserveScalar x2, SqueezeScalar] triplet per dense variable,
            // running until the multipoint label — count-free, so boolean
            // tapes (no packed-direct claims) parse identically.
            while matches!(ops[cur.i], Op::ObserveScalar)
                && matches!(ops[cur.i + 1], Op::ObserveScalar)
                && matches!(ops[cur.i + 2], Op::SqueezeScalar)
            {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1") {
            // The multipoint region: P group-value absorbs, the batching
            // gamma, m two-product rounds, then the anchor's label + v +
            // 2(m + 1) rounds. Each loop terminates on the next label /
            // squeeze, so a shape change fails loudly here.
            cur.bump();
            let mut val_vs = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                val_vs.push(cur.v);
                cur.bump();
            }
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint gamma");
            let (gamma_fin, gamma_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut mp_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint round");
                mp_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
                "op {}: expected the anchor label, got {:?}",
                cur.i,
                ops[cur.i]
            );
            cur.bump();
            let anchor_v = cur.v;
            cur.expect_obs_scalar();
            let mut anchor_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "anchor round");
                anchor_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            mp = Some(MpRec {
                val_vs,
                gamma_fin,
                gamma_ch,
                rounds: mp_rounds,
                anchor_v,
                anchor_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-pcs-packed-direct-v0") {
            cur.bump();
            let q_v = cur.v;
            cur.expect_obs_scalar(); // q_eval
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "inner gamma");
            inner_pd = Some(InnerPd {
                q_v,
                fin: cur.fin,
                ch: cur.ch,
            });
            cur.bump();
            continue;
        }
        cur.bump();
    }
    let inner_pd = inner_pd.expect("the inner ligerito intake");
    let mp = mp.expect("the multipoint region");
    cur.bump(); // the open-phase initial cap absorb
    let start_v = cur.v;
    cur.expect_obs_scalar(); // sumcheck start msg u_0
    cur.expect_obs_scalar(); // ... u_2

    let mut levels = Vec::new();
    let mut yr_v = 0usize;
    for li in 0..=r {
        // Fold batch: [Pow?] SqueezeScalar + ObserveScalar x2 per round. Only
        // consume a Pow that fronts a fold — the query-grinding Pow follows
        // this loop and must survive it.
        let mut fold_fins = Vec::new();
        let mut fold_chs = Vec::new();
        let mut fold_msg_vs = Vec::new();
        loop {
            match cur.ops[cur.i] {
                Op::Pow { .. }
                    if matches!(cur.ops.get(cur.i + 1), Some(Op::SqueezeScalar)) =>
                {
                    cur.bump()
                }
                Op::SqueezeScalar => {
                    fold_fins.push(cur.fin);
                    fold_chs.push(cur.ch);
                    cur.bump();
                    fold_msg_vs.push(cur.v);
                    cur.expect_obs_scalar();
                    cur.expect_obs_scalar();
                }
                _ => break,
            }
        }
        let mut ood = Vec::new();
        if li < r {
            // The NEXT commitment's cap, then its OOD groups.
            assert!(
                matches!(cur.ops[cur.i], Op::ObserveBytes(_)),
                "op {}: expected the next cap absorb, got {:?}",
                cur.i,
                cur.ops[cur.i]
            );
            cur.bump();
            while !matches!(cur.ops[cur.i], Op::Pow { .. }) {
                let z_len = match cur.ops[cur.i] {
                    Op::SqueezeSlice(n) => n,
                    ref o => panic!("OOD z, got {o:?}"),
                };
                let (z_fin, z_ch) = (cur.fin, cur.ch);
                cur.bump();
                let y_v = cur.v;
                cur.expect_obs_scalar(); // y
                let intro_v = cur.v;
                cur.expect_obs_scalar(); // intro u_0
                cur.expect_obs_scalar(); // intro u_2
                assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "OOD beta");
                ood.push(OodRec {
                    z_fin,
                    z_ch,
                    z_len,
                    y_v,
                    intro_v,
                    beta_fin: cur.fin,
                    beta_ch: cur.ch,
                });
                cur.bump();
            }
        } else {
            // Final level: the yr observes.
            yr_v = cur.v;
            while matches!(cur.ops[cur.i], Op::ObserveScalar) {
                cur.bump();
            }
        }
        assert!(
            matches!(cur.ops[cur.i], Op::Pow { .. }),
            "op {}: expected query-grinding Pow, got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        cur.bump();
        let (q_fin, q_ch, q_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected queries squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let (a_fin, a_ch, a_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected alpha squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let intro_v = cur.v;
        if li < r {
            cur.expect_obs_scalar(); // intro u_0
            cur.expect_obs_scalar(); // intro u_2
        }
        assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "beta");
        let (beta_fin, beta_ch) = (cur.fin, cur.ch);
        cur.bump();
        levels.push(OpenLevel {
            fold_fins,
            fold_chs,
            fold_msg_vs,
            ood,
            intro_v,
            beta_fin,
            beta_ch,
            q_fin,
            q_ch,
            q_count,
            a_fin,
            a_ch,
            a_count,
        });
    }
    (
        start_v,
        piop,
        gammas,
        rounds,
        mp,
        inner_pd,
        yr_v,
        levels,
    )
}

// ---------------------------------------------------------------------------
/// **MVP-7: the query phase of a REAL inner proof.** Everything MVP-6 models,
/// against an actual proof: a pure-element union proof (no ring-switch, no
/// Frobenius assist) is proven and verified natively, its verifier transcript
/// recorded, and the circuit then replays the REAL FS chain, opens the REAL
/// leaf rows out of the proof against the REAL cap layers, and computes each
/// level's enforced sum with FS-DERIVED weights:
///
/// - the query index words, the lane-fold challenges `v`, and the basis
///   challenges `alpha` are all chain squeeze outputs, WIRED — not public
///   inputs. `v` feeds `LeafEvalGate` directly; `alpha`'s eq-tensor expansion
///   happens at the boundary (the words are published, the checker expands
///   `eq(alpha, k)` natively and supplies the weights as public inputs).
/// - the circuit's witness IS the proof: opened rows are the leaf words,
///   `merkle_proof` supplies the per-query sibling hints, the caps are the
///   absorbed commitment. Nothing is plumbed out of the prover.
/// - per query the boundary select publishes (challenge word, terminal
///   digest); per level the alpha words and the enforced-sum accumulator.
///   The checker masks, indexes the cap, and compares the sums against
///   natively recomputed `enforced_sum` values.
///
/// **Step 2a, the sumcheck spine, is IN**: one `SpineGate` element type
/// replays the verifier's running quadratic across all levels — build from
/// each message, eval at each chain challenge, intro-fold every OOD claim and
/// enforced sum (the LeafEval accumulators, consumed in-circuit rather than
/// boundary-checked) — and publishes the final `t_r`, checked against a
/// native replay. The start target is a fixed public input until the merged
/// intake lands (2c — now DONE, see below). **2b stage 1**: per-level `ResidualGate`s evaluate the
/// induced-basis residuals (`next_s` chain, prefix over later fold
/// challenges, suffix subset products) with `q_field` boundary-bound like
/// the alpha expansion; the `2^yr` accumulators publish and check against a
/// native replica. **2c (the merged intake)**: the outer target is the
/// gamma-combination of the ABSORBED claim values (SpineGate tr-rows), the
/// W-rounds fold it through `MergedRoundGate` binding rho, and the ligerito
/// spine starts from gamma' * q_eval — every scalar in the statement is
/// transcript-bound, and `inner == t_r` closes BETWEEN CIRCUIT OUTPUTS.
/// **The PIOP spine extension**: the element zerocheck replays with `g0` as
/// advice (published-zero identity deltas — family I, no in-circuit
/// inversion), closes `running = ea eb + ec`, strips the slot's affine
/// constants (ZcJoinGate), enters the lincheck at `va + alpha vb`, and
/// reuses MergedRoundGate for the lincheck rounds; the published target IS
/// the deferred ElementAssertion's target, and the ec join forces the merged
/// intake's first absorbed value to be the zerocheck's output. **MVP-8
/// (2026-08-03) closed the rest**: the multipoint region is fully
/// in-circuit (MacGate T0/V chains, mrslot rounds, the AssistLayerGate
/// anchor DP with claim-POINT joins load-bearing, three tail zero-deltas —
/// the native `v` is gone), the PoW bit predicate binds through published
/// digest/nonce wires, and the ElementAssertion exits as bound publics.
/// Nothing in the pure-element verifier is native. The inner proof commits with
/// the lane grid at full utilization — count 2^13 x 4 cols = 2^15 words =
/// exactly 2^22 dense bits, t = 64 — so L0 is the real 64-lane / 1 KiB-leaf
/// shape with zero padding.
#[test]
#[ignore] // Proves a real m=22 inner proof first. `-- --ignored`.
fn mvp7_real_query_phase() {
    use flock_core::element_r1cs::ElementTableBuilder;
    use flock_core::transcript_record::RecordingChallenger;
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::FsChain;
    use flock_prover::schedule::Registry;
    use std::sync::Arc;
    use std::time::Instant;

    const DOMAIN7: &[u8] = b"flock-mvp7-real-v0";
    const INNER_NU: usize = 13;
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    // ---- the inner proof: pure element, m=22 dense, BLAKE3 everywhere ----
    let mut rng = Rng(0x_5EED_0007);
    let rf = |rng: &mut Rng| {
        F128::new(
            ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
        )
    };
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));
    let inner_ty = {
        let mut b = ElementTableBuilder::new(2);
        b.free_wire(0)
            .free_wire(1)
            .mult(2, 0, 1)
            .linear(3, &[(0, w0), (1, w1)]);
        Arc::new(b.build().expect("gate block is valid"))
    };
    let registry = Registry::new(vec![TableType::element(inner_ty.clone())], INNER_NU);
    let n_rows = 1usize << INNER_NU; // full utilization: dense = 4 * 2^20 = 2^22
    let witness: Vec<F128> = {
        let at = |c: usize, j: usize| (c << INNER_NU) + j;
        let mut z = vec![F128::ZERO; inner_ty.width() << INNER_NU];
        for j in 0..n_rows {
            let (a, b) = (rf(&mut rng), rf(&mut rng));
            z[at(0, j)] = a;
            z[at(1, j)] = b;
            z[at(2, j)] = a * b;
            z[at(3, j)] = w0 * a + w1 * b;
        }
        z
    };
    let inner_union = UnionInstance::new(&registry, vec![n_rows]);
    assert_eq!(inner_union.dense_m(), 22, "the inner commit is exactly m=22");
    let inner_params = PcsParams {
        m: inner_union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: inner_union.commit_lanes(6), // = 64 at full utilization
        merkle_hash: HashKind::Blake3,
    };
    let t = Instant::now();
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_mixed_class(
        &inner_union,
        &inner_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(|dst: &mut [F128]| {
            dst.copy_from_slice(&witness)
        })],
        &mut FsChallenger::with_hash(DOMAIN7, HashKind::Blake3),
    );
    let inner_prove_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- record the REAL verifier transcript ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(DOMAIN7, HashKind::Blake3));
    let claims_v = verifier::verify_ligerito_union_mixed_class(
        &inner_union,
        &[],
        &commitment,
        &proof,
        &inner_params,
        &mut rec,
    )
    .expect("the inner proof verifies natively");
    let el_claims = claims_v.element.as_ref().expect("element claims");
    // The packed-direct intake order (verifier.rs): [(c_point, c_value),
    // (lc_point, lc_value)].
    let pd_pts: Vec<Vec<F128>> = vec![el_claims.c_point.clone(), el_claims.lc_point.clone()];
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(DOMAIN7);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let chals: Vec<F128> = rec.challenges().to_vec();

    // ---- level geometry, from the proof alone ----
    let lig = &proof.pcs_open.inner.ligerito;
    assert_eq!(commitment.cap, lig.initial_cap, "commitment IS the L0 cap");
    let r = lig.recursive_caps.len();
    assert_eq!(lig.recursive_proofs.len(), r - 1, "levels with opens = r + 1");
    let lvl_src = level_sources(lig);
    let (start_v, piop, gammas, w_rounds, mp, inner_pd, yr_v, levels) =
        parse_open_levels(t_shape.ops(), 32 * lig.initial_cap.len(), r);
    let piop = piop.expect("the element PIOP");
    assert_eq!(levels.len(), r + 1);

    // The multipoint region, validated field-for-field against the proof:
    // the located stream words ARE the proof's group values / round
    // messages / anchor transcript, in order — so the wires the MVP-8
    // assembly reads are the scalars the native verifier consumed. R = 0
    // for this pure-element inner: no RS values exist.
    {
        let fro = &proof.pcs_open.frobenius;
        let vals_rec = rec.values();
        assert!(fro.values.is_empty(), "pure-element inner has R = 0");
        assert_eq!(mp.val_vs.len(), fro.group_values.len(), "group value count");
        for (vi, want) in mp.val_vs.iter().zip(&fro.group_values) {
            assert_eq!(vals_rec[*vi], *want, "group value stream word");
        }
        assert_eq!(mp.rounds.len(), fro.rounds.len(), "two-product round count");
        for (rr, want) in mp.rounds.iter().zip(&fro.rounds) {
            assert_eq!((vals_rec[rr.g_v], vals_rec[rr.g_v + 1]), *want, "mp round msg");
        }
        assert_eq!(vals_rec[mp.anchor_v], fro.anchor.v, "anchor v stream word");
        assert_eq!(
            mp.anchor_rounds.len(),
            fro.anchor.rounds.len(),
            "anchor round count"
        );
        for (rr, want) in mp.anchor_rounds.iter().zip(&fro.anchor.rounds) {
            assert_eq!(
                (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]),
                *want,
                "anchor round msg"
            );
        }
        // The native accept relations, replayed from the located pieces:
        // T0 = sum gamma^k B_k folds through the rounds to T_m == anchor.v,
        // and the anchor.v folds through its rounds to the expect the DP
        // gates will compute. This is the exact statement the in-circuit
        // assembly must publish as zero-deltas.
        let gamma = chals[mp.gamma_ch];
        let mut t = F128::ZERO;
        let mut pw = F128::ONE;
        for &vi in &mp.val_vs {
            t += pw * vals_rec[vi];
            pw *= gamma;
        }
        for rr in &mp.rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let r = chals[rr.ch];
            let g0 = t + g1;
            t = g0 + (g1 + g0 + gi) * r + gi * r * r;
        }
        assert_eq!(t, fro.anchor.v, "T_m must equal the anchor's claimed v");
    }

    // ---- PoW grinding ops, located on the tape ----
    // Each Pow op finalizes the chain (the state digest — output wires the
    // replay already computes) and absorbs one aligned 8-byte nonce word
    // (a Bytes payload). Record (finalize ordinal, payload ordinal, bits).
    struct PowRec {
        fin: usize,
        pay: usize,
        bits: u32,
    }
    let pows: Vec<PowRec> = {
        use flock_core::transcript_record::TranscriptOp as Op;
        let mut out = Vec::new();
        let (mut fin, mut pay) = (0usize, 0usize);
        for op in t_shape.ops() {
            if let Op::Pow { bits } = op {
                out.push(PowRec {
                    fin,
                    pay,
                    bits: *bits,
                });
            }
            if op.finalizes() {
                fin += 1;
            }
            match op {
                Op::ObserveBytes(_) | Op::Pow { .. } => pay += 1,
                _ => {}
            }
        }
        out
    };
    assert!(!pows.is_empty(), "the Fast profile grinds");

    // Per level: q, c, path depth d-c, lanes — and the native cross-checks
    // that pin every piece of the plumbing before the circuit exists: each
    // opened row verifies against its cap under the recorded challenge, and
    // the recorded weights reproduce `induce_sumcheck_enforced_sum`.
    let (geo, native_sums) = level_geometry(&levels, &lvl_src, &chals, HashKind::Blake3);

    // ---- the FS chain over the real byte stream ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&flock_core::transcript_record::TranscriptOp> =
        t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    let b3_rows: usize = trace.rows.len()
        + geo
            .iter()
            .map(|g| (g.lanes / 4 + g.depth) * g.q + (1usize << g.c) - 1)
            .sum::<usize>();
    let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);

    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let slots = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(spread_w),
            nu,
        }),
    };
    let mut leaf_slot: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
    // ONE 8-lane leaf-eval type serves every level: a 64-lane leaf is 8
    // CHAINED rows, row h taking lanes [8h, 8h+8) with alpha input
    // `alpha_k·eq(v[3..], h)` — the same boundary-expanded-public pattern
    // alpha itself rides — because `y_64 = Σ_h eq(v[3..], h)·y_8(group h)`
    // (split the lane index at bit 3; build_eq_table is LSB-first). This
    // deletes the kappa-8 slot: 2^21 element-region words (27% of the
    // region) and a 73-word schema from the cell space. The high v wires
    // stay bound through the residual/prefix gates, which consume every
    // level's fold wires.
    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match leaf_slot.iter().find(|(n, _)| *n == lanes) {
                Some((_, s)) => *s,
                None => {
                    let s = sb.slot(LeafEvalGate::new(lanes));
                    leaf_slot.push((lanes, s));
                    s
                }
            }
        })
        .collect();

    let spine = sb.slot(SpineGate::new());
    leaf_slot.push((0, spine));

    let mut vals: Vec<F128> = Vec::new();
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv = [sb.public_input(), sb.public_input()];
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let mut pub_payloads = bytes_payload_mask(t_shape.ops());
    let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
    for &p in &cap_pays[1..] {
        pub_payloads[p] = false;
    }
    let (outs, word_wire) = emit_fs_chain(
        &mut sb,
        slots.b3,
        iv,
        &trace,
        &stream,
        &bytes,
        &mut vals,
        &mut consts,
        &pub_payloads,
    );
    // Observed-value index -> absorbed-stream word index, for wiring the
    // sumcheck messages (they are absorbed proof scalars, so their wires
    // already exist as the chain's block inputs).
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
            if vmap.len() <= vi {
                vmap.resize(vi + 1, None);
            }
            vmap[vi] = Some(wi);
        }
    }

    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    // The query phase, via the shared emitter. Its `to_publish` is published
    // AFTER every input is declared: `built.public` lists entries in
    // DECLARATION order, so publishing inside the loop would interleave with
    // the next level's public inputs and break the tail walk below.
    let cap_w = cap_wires(&stream, &word_wire, &cap_pays);
    let (to_publish, level_accs) = emit_query_phase(
        &mut sb,
        slots,
        iv,
        &leafeval,
        &levels,
        &geo,
        &lvl_src,
        &trace.squeezes,
        &outs,
        &chals,
        &cap_w,
        &mut vals,
        &mut consts,
        &mut hints,
    );

    // ---- the merged intake + W-rounds (2c): binding the start target ----
    // The outer target is the gamma-combination of the element claims'
    // absorbed values; each merged round folds it through the quadratic
    // (t+g1) + (t+gi)r + gi r^2, binding rho; the boundary check below
    // closes `running == q_eval * v` with the assist's v native. The
    // ligerito spine then starts from gamma' * q_eval — every scalar in the
    // statement is now transcript-bound.
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let chw = |outs: &Vec<Vec<Wire>>, trace_sq: &Vec<Vec<usize>>, fin: usize| -> Wire {
        outs[trace_sq[fin][0]][0]
    };
    let wv = |vi: usize| -> Wire { word_wire[vmap[vi].expect("stream word")].expect("wired") };
    // ---- the element PIOP (spine extension): zerocheck + lincheck ----
    // g0 rides as advice per zerocheck round; its defining identity and the
    // closing `running = ea eb + ec` publish as zero-deltas. The strip gate
    // folds the slot's affine constants at the last two zerocheck challenges,
    // and the lincheck's two rounds reuse MergedRoundGate. The `ec` join
    // (published-zero) forces the merged intake's first absorbed value to BE
    // the zerocheck's output.
    let zc_natives: Vec<F128> = {
        // native g0 chain, needed as the advice values
        let mut out = Vec::with_capacity(piop.tau_len);
        let mut running = F128::ZERO;
        for (i, rr) in piop.zc_rounds.iter().enumerate() {
            let (g1, gi) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let t = chals[piop.tau_ch + i];
            let rho = chals[rr.ch];
            let g0 = (running + t * g1) * (F128::ONE + t).inv();
            out.push(g0);
            running = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        out
    };
    let zslot = sb.slot(ZcRoundGate::new());
    leaf_slot.push((500, zslot));
    let mut zr = zw;
    let mut zc_deltas: Vec<Wire> = Vec::new();
    for (i, rr) in piop.zc_rounds.iter().enumerate() {
        let sqt = &trace.squeezes[piop.tau_fin];
        let t_w = outs[sqt[i / 4]][i % 4];
        let rho_w = chw(&outs, &trace.squeezes, rr.fin);
        vals.push(zc_natives[i]);
        let g0w = sb.public_input();
        let g = sb.gate(
            zslot,
            &[zr, wv(rr.g_v), wv(rr.g_v + 1), t_w, rho_w, g0w, ow],
        );
        zc_deltas.push(g[0]);
        zr = g[1];
    }
    let jslot = sb.slot(ZcJoinGate::new(inner_ty.a_const(), inner_ty.b_const()));
    leaf_slot.push((501, jslot));
    let nzc = piop.zc_rounds.len();
    let jr = sb.gate(
        jslot,
        &[
            zr,
            wv(piop.eab_v),
            wv(piop.eab_v + 1),
            wv(piop.eab_v + 2),
            chw(&outs, &trace.squeezes, piop.zc_rounds[nzc - 2].fin),
            chw(&outs, &trace.squeezes, piop.zc_rounds[nzc - 1].fin),
            ow,
        ],
    );
    let (zc_fin_delta, va_w, vb_w) = (jr[0], jr[1], jr[2]);
    let alpha_w = chw(&outs, &trace.squeezes, piop.alpha_fin);
    let lc0 = sb.gate(spine, &[zw, zw, zw, va_w, zw, zw, vb_w, alpha_w, zw])[3];
    let mut lr = lc0;
    // (MergedRoundGate is declared just below for the W-rounds; the lincheck
    // rounds share it, so hoist the slot.)
    let mrslot = sb.slot(MergedRoundGate::new());
    leaf_slot.push((400, mrslot));
    for rr in &piop.lc_rounds {
        lr = sb.gate(
            mrslot,
            &[lr, wv(rr.g_v), wv(rr.g_v + 1), chw(&outs, &trace.squeezes, rr.fin)],
        )[0];
    }
    let lc_target = lr;
    // ec join: the intake's first absorbed value == the zerocheck's ec.
    let ec_join = sb.gate(
        spine,
        &[zw, zw, zw, wv(piop.eab_v + 2), zw, zw, wv(gammas[0].val_v), ow, zw],
    )[3];

    // Outer target: SpineGate tr-rows accumulate gamma_k * value_k.
    let mut mt = zw;
    for pd in &gammas {
        let gw = chw(&outs, &trace.squeezes, pd.fin);
        let f = sb.gate(spine, &[zw, zw, zw, mt, zw, zw, wv(pd.val_v), gw, zw]);
        mt = f[3];
    }
    // The W-rounds, binding rho.
    for rr in &w_rounds {
        let rw = chw(&outs, &trace.squeezes, rr.fin);
        mt = sb.gate(mrslot, &[mt, wv(rr.g_v), wv(rr.g_v + 1), rw])[0];
    }
    let running_w = mt;
    // The ligerito start target: gamma' * q_eval.
    let gpw = chw(&outs, &trace.squeezes, inner_pd.fin);
    let tw0 = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, wv(inner_pd.q_v), gpw, zw]);
    let mut tw = tw0[3];
    let st = sb.gate(spine, &[zw, zw, zw, zw, wv(start_v), wv(start_v + 1), tw, ow, zw]);
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = chw(&outs, &trace.squeezes, lvl.fold_fins[j]);
            let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
            tw = ev[4];
            let bld = sb.gate(spine, &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tw, ow, zw]);
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r {
            for od in &lvl.ood {
                let bw = chw(&outs, &trace.squeezes, od.beta_fin);
                let f = sb.gate(
                    spine,
                    &[qc, qb, qa, tw, wv(od.intro_v), wv(od.intro_v + 1), wv(od.y_v), bw, zw],
                );
                (qc, qb, qa, tw) = (f[0], f[1], f[2], f[3]);
            }
            let bw = chw(&outs, &trace.squeezes, lvl.beta_fin);
            let f = sb.gate(
                spine,
                &[qc, qb, qa, tw, wv(lvl.intro_v), wv(lvl.intro_v + 1), level_accs[li], bw, zw],
            );
            (qc, qb, qa, tw) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = chw(&outs, &trace.squeezes, lvl.beta_fin);
            let f = sb.gate(spine, &[zw, zw, zw, tw, zw, zw, level_accs[li], bw, zw]);
            tw = f[3];
        }
    }
    let t_final = tw;

    // ---- the residual region (2b): the shared emitter ----
    // Per-level ResidualGate accumulators, then the prefix/suffix/partial-
    // combine/final-dot close-out ending in the residual-side inner. (The
    // close-out once needed MVP7_CLOSEOUT=1: the extra element types pushed
    // the outer union's boolean RS claims off the DeferredDense shape.
    // ring_switch now defers every claim, so it is unconditional.)
    assert_eq!(gammas.len(), pd_pts.len(), "one gamma per claim");
    for (k, pd) in gammas.iter().enumerate() {
        for j in 0..pd.pt_len {
            assert_eq!(rec.values()[pd.pt_v + j], pd_pts[k][j], "pt {k}:{j} on tape");
        }
    }
    let yr_len = proof.pcs_open.inner.ligerito.final_proof.yr.len();
    let yr_wires: Vec<Wire> = (0..yr_len).map(|y| wv(yr_v + y)).collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        &mut sb,
        &mut leaf_slot,
        &levels,
        &geo,
        &w_rounds,
        inner_pd.fin,
        &yr_wires,
        &trace.squeezes,
        &outs,
        &chals,
        &mut vals,
        zw,
        ow,
    );

    // ---- MVP-8 step 2: the multipoint intake in-circuit ----
    // T0 = Σ gamma^k·B_k (Mac chain over the absorbed group values), the m
    // two-product rounds through the SAME MergedRoundGate slot the W-rounds
    // use, and two zero-delta joins: T_m == anchor.v binds the round chain
    // to the anchor's claimed evaluation, and running_W == q_eval·V with
    // V = Σ B_k IN-CIRCUIT replaces the boundary's native v. The anchor's
    // own rounds + expect (the AssistLayerGate chains) are step 3.
    let macslot = sb.slot(MacGate::new());
    leaf_slot.push((600, macslot));
    let gamma_w = chw(&outs, &trace.squeezes, mp.gamma_fin);
    let mut t0 = zw;
    let mut vsum = zw;
    let mut pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp.val_vs.iter().enumerate() {
        t0 = sb.gate(macslot, &[t0, pws[k], wv(vi)])[0];
        vsum = sb.gate(macslot, &[vsum, wv(vi), ow])[0];
        if k + 1 < mp.val_vs.len() {
            let p = sb.gate(macslot, &[zw, pws[k], gamma_w])[0];
            pws.push(p);
        }
    }
    let mut tm = t0;
    let mut rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp.rounds {
        let r_w = chw(&outs, &trace.squeezes, rr.fin);
        rho2_w.push(r_w);
        tm = sb.gate(mrslot, &[tm, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let delta_tm = sb.gate(macslot, &[tm, wv(mp.anchor_v), ow])[0];
    let qv = sb.gate(macslot, &[zw, wv(inner_pd.q_v), vsum])[0];
    let delta_rq = sb.gate(macslot, &[running_w, qv, ow])[0];

    // ---- MVP-8 step 3: the anchor in-circuit ----
    // 3a: the anchor's claimed v folds through its 2(m+1) rounds (mrslot
    // reuse); the squeezes are the sigma wires the expect consumes.
    let mut acl = wv(mp.anchor_v);
    let mut sig_w: Vec<Wire> = Vec::new();
    for rr in &mp.anchor_rounds {
        let r_w = chw(&outs, &trace.squeezes, rr.fin);
        sig_w.push(r_w);
        acl = sb.gate(mrslot, &[acl, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let m_mp = mp.rounds.len();
    assert_eq!(sig_w.len(), 2 * (m_mp + 1), "sigma spans the anchor layers");

    // The residual region's prefix slot (width pf_w); a chunked product of
    // (1 + a + b) factors, seed-chained across rows, padded factors
    // (zw, zw) = 1.
    let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
        let mut seed = ow;
        for chunk in factors.chunks(pf_w) {
            let mut g_in = vec![seed];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            seed = sb.gate(pfslot, &g_in)[0];
        }
        seed
    };

    // 3b: e_at = eq(rho, rho'') — rho is the W-round challenge wires.
    let rho_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| chw(&outs, &trace.squeezes, rr.fin))
        .collect();
    let factors: Vec<(Wire, Wire)> = rho_w.iter().copied().zip(rho2_w.iter().copied()).collect();
    let e_at_w = prefix_product(&mut sb, &factors);

    // 3c: the expect. Groups by shared row part (structural for this fixed
    // shape — asserted one dual value per group), run structure from the
    // inner's jagged boundaries (pub, same source as the verifier).
    let n_log_i = INNER_NU;
    let k_cols_i = gammas[0].pt_len - n_log_i;
    let mut groups_ix: Vec<Vec<usize>> = Vec::new();
    for (i, pt) in pd_pts.iter().enumerate() {
        match groups_ix
            .iter_mut()
            .find(|g| pd_pts[g[0]][..n_log_i] == pt[..n_log_i])
        {
            Some(g) => g.push(i),
            None => groups_ix.push(vec![i]),
        }
    }
    assert_eq!(groups_ix.len(), mp.val_vs.len(), "one dual value per group");
    let params_i = flock_core::pcs::jagged::JaggedParams::from_heights(
        &inner_union.jagged_heights(),
        n_log_i,
        m_mp,
    );
    let bounds = flock_core::pcs::jagged::assist_boundaries(&params_i);
    // Shape assumption: singleton used-column runs, plus AT MOST one
    // zero-height tail run (absent at full utilization — this inner).
    let n_runs = bounds.len();
    let has_tail = bounds[n_runs - 1].0 == bounds[n_runs - 1].1;
    let n_single = if has_tail { n_runs - 1 } else { n_runs };
    for &(_, _, len) in &bounds[..n_single] {
        assert_eq!(len, 1, "used columns are singleton runs");
    }

    // Per-run boundary eq products at sigma (statement-independent).
    let eqc: Vec<Wire> = bounds
        .iter()
        .map(|&(t_c, t_next, _)| {
            let mut factors = Vec::with_capacity(2 * (m_mp + 1));
            for l in 0..=m_mp {
                factors.push((sig_w[2 * l], if (t_c >> l) & 1 == 1 { ow } else { zw }));
                factors.push((sig_w[2 * l + 1], if (t_next >> l) & 1 == 1 { ow } else { zw }));
            }
            prefix_product(&mut sb, &factors)
        })
        .collect();

    // Per (claim, singleton run) column-eq sums, the tail by char-2
    // complement (total eq mass is 1), then the gamma_pd combination and
    // the per-statement dot + DP + coefficient.
    let alslot = sb.slot(AssistLayerGate::new());
    leaf_slot.push((601, alslot));
    let mut expect = zw;
    for (g_ix, members) in groups_ix.iter().enumerate() {
        let mut run_w: Vec<Wire> = vec![zw; n_runs];
        for &i in members {
            let pd = &gammas[i];
            let gpd_w = chw(&outs, &trace.squeezes, pd.fin);
            let mut tail = ow;
            for r in 0..n_single {
                let y = r as u64;
                let factors: Vec<(Wire, Wire)> = (0..k_cols_i)
                    .map(|j| {
                        (
                            wv(pd.pt_v + n_log_i + j),
                            if (y >> j) & 1 == 1 { ow } else { zw },
                        )
                    })
                    .collect();
                let s = prefix_product(&mut sb, &factors);
                tail = sb.gate(macslot, &[tail, s, ow])[0];
                run_w[r] = sb.gate(macslot, &[run_w[r], gpd_w, s])[0];
            }
            if has_tail {
                run_w[n_runs - 1] = sb.gate(macslot, &[run_w[n_runs - 1], gpd_w, tail])[0];
            }
        }
        let mut w_st = zw;
        for (r, &rw) in run_w.iter().enumerate() {
            w_st = sb.gate(macslot, &[w_st, rw, eqc[r]])[0];
        }
        let mut g = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        let row0 = pd_pts[members[0]].len(); // silence: row wires below
        let _ = row0;
        for layer in (0..=m_mp).rev() {
            let za = if layer < n_log_i {
                wv(gammas[members[0]].pt_v + layer)
            } else {
                zw
            };
            let rb = if layer < m_mp { rho2_w[layer] } else { zw };
            let mut a_in = g.to_vec();
            a_in.extend_from_slice(&[za, rb, sig_w[2 * layer], sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            g = [o[0], o[1], o[2], o[3]];
        }
        let coeff = sb.gate(macslot, &[zw, pws[g_ix], e_at_w])[0];
        let wd = sb.gate(macslot, &[zw, w_st, g[0]])[0];
        expect = sb.gate(macslot, &[expect, coeff, wd])[0];
    }
    // 3d: the join — the anchor's folded claim equals the expect.
    let delta_anchor = sb.gate(macslot, &[acl, expect, ow])[0];

    // ---- the PoW bit predicate (boundary pattern) ----
    // Per Pow op, publish (state-digest words, nonce word): the digest is
    // the chain finalize's first two output words, the nonce its aligned
    // stream word. The checker recomputes H(digest ‖ nonce) natively and
    // applies the leading-zero predicate — the same trust structure as the
    // alpha expansion.
    let pow_pub: Vec<[Wire; 3]> = pows
        .iter()
        .map(|pr| {
            let sq = &trace.squeezes[pr.fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pr.pay))
                .expect("pow nonce stream word");
            let nw = word_wire[wi].expect("pow nonce wired");
            [outs[sq[0]][0], outs[sq[0]][1], nw]
        })
        .collect();

    // ---- ASSERTION EMISSION: the ElementAssertion exits as bound publics.
    // For this one-slot inner every field is already a wire: alpha (chain),
    // r_con = the zerocheck challenges, r_col = the lincheck challenges
    // (chain squeezes), evals = the ZcJoin-derived (va, vb) — the
    // strip_constants values the native assertion carries — z_eval = the
    // element c claim's absorbed value (ec-joined to the zerocheck), and
    // target = lc_target (published below with the PIOP tail). The
    // accumulator can reconstruct the assertion from the public segment
    // alone. Multi-slot inners absorb per-slot eval pairs — a shape
    // extension for the mixed phase.
    let mut assertion_pub: Vec<Wire> = vec![alpha_w];
    for rr in &piop.zc_rounds {
        assertion_pub.push(chw(&outs, &trace.squeezes, rr.fin));
    }
    for rr in &piop.lc_rounds {
        assertion_pub.push(chw(&outs, &trace.squeezes, rr.fin));
    }
    assertion_pub.extend_from_slice(&[va_w, vb_w, wv(gammas[0].val_v)]);

    for a_wires in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    sb.publish(t_final);
    sb.publish(running_w);
    for d in &zc_deltas {
        sb.publish(*d);
    }
    sb.publish(zc_fin_delta);
    sb.publish(ec_join);
    sb.publish(lc_target);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(*w);
        }
    }
    sb.publish(inner_w);
    sb.publish(delta_tm);
    sb.publish(delta_rq);
    sb.publish(delta_anchor);
    for p in &pow_pub {
        for w in p {
            sb.publish(*w);
        }
    }
    for w in &assertion_pub {
        sb.publish(*w);
    }
    let shape = sb.finish().expect("valid real-query circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

    // ---- the boundary checks ----
    // The tail publics: three multipoint zero-deltas (T_m == anchor.v,
    // running_W == q_eval·V, claim == expect), per-Pow (digest word0,
    // digest word1, nonce word) triples the checker validates natively,
    // then the emitted ElementAssertion fields.
    let n_assert = 1 + piop.zc_rounds.len() + piop.lc_rounds.len() + 3;
    let assert_base = built.public.len() - n_assert;
    {
        let vals_rec = rec.values();
        let mut at = assert_base;
        assert_eq!(built.public[at], chals[piop.alpha_ch], "assertion alpha");
        at += 1;
        for rr in &piop.zc_rounds {
            assert_eq!(built.public[at], chals[rr.ch], "assertion r_con");
            at += 1;
        }
        for rr in &piop.lc_rounds {
            assert_eq!(built.public[at], chals[rr.ch], "assertion r_col");
            at += 1;
        }
        // evals = (va, vb) are strip_constants derivations — validated
        // transitively by the lincheck target equality; z_eval is the
        // absorbed c-claim value.
        assert_eq!(built.public[at + 2], vals_rec[gammas[0].val_v], "assertion z_eval");
    }
    let pow_base = assert_base - 3 * pows.len();
    for (i, off) in [3, 2, 1].into_iter().enumerate() {
        assert_eq!(
            built.public[pow_base - off],
            F128::ZERO,
            "multipoint zero-delta {i}"
        );
    }
    for (i, pr) in pows.iter().enumerate() {
        let d0 = built.public[pow_base + 3 * i];
        let d1 = built.public[pow_base + 3 * i + 1];
        let nn = built.public[pow_base + 3 * i + 2];
        let mut digest = [0u8; 32];
        digest[..8].copy_from_slice(&d0.lo.to_le_bytes());
        digest[8..16].copy_from_slice(&d0.hi.to_le_bytes());
        digest[16..24].copy_from_slice(&d1.lo.to_le_bytes());
        digest[24..].copy_from_slice(&d1.hi.to_le_bytes());
        assert_eq!(nn.hi, 0, "pow {i}: nonce word is 8 bytes zero-padded");
        if pr.bits == 0 {
            assert_eq!(nn.lo, 0, "pow {i}: canonical zero nonce");
        } else {
            assert!(
                flock_core::challenger::pow_has_leading_zero_bits(
                    &digest,
                    nn.lo,
                    pr.bits,
                    HashKind::Blake3,
                ),
                "pow {i}: grinding predicate on the published wires"
            );
        }
    }
    let yr_pub = levels.len() * yr_len
        + 1
        + 1
        + piop.zc_rounds.len()
        + 3
        + 3
        + 3 * pows.len()
        + n_assert;
    let total_pub: usize = 1 + yr_pub
        + levels.iter().map(|l| l.a_count).sum::<usize>();
    let mut at = built.public.len() - total_pub;
    // The openings bind to the absorbed caps by COPY CONSTRAINT (the
    // in-circuit cap tree) — no per-query publics, no checker walk.
    for (li, lvl) in levels.iter().enumerate() {
        for j in 0..lvl.a_count {
            assert_eq!(built.public[at + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
        }
        at += lvl.a_count;
    }
    // The spine, replayed natively over the recorded transcript: same quad
    // math, same start target, native enforced sums. Equality transitively
    // validates every eval/build/fold gate AND the LeafEval accumulators.
    let vals_rec = rec.values();
    let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
    let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
    // The bound start target: gamma' * q_eval.
    let mut nt = chals[inner_pd.ch] * vals_rec[inner_pd.q_v];
    let mut nq = quad(vals_rec[start_v], vals_rec[start_v + 1], nt);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            nt = evalq(nq, chals[lvl.fold_chs[j]]);
            nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
        }
        if li < levels.len() - 1 {
            for od in &lvl.ood {
                let b = chals[od.beta_ch];
                let iq = quad(vals_rec[od.intro_v], vals_rec[od.intro_v + 1], vals_rec[od.y_v]);
                nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                nt += b * vals_rec[od.y_v];
            }
            let b = chals[lvl.beta_ch];
            let iq = quad(vals_rec[lvl.intro_v], vals_rec[lvl.intro_v + 1], native_sums[li]);
            nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
            nt += b * native_sums[li];
        } else {
            nt += chals[lvl.beta_ch] * native_sums[li];
        }
    }
    assert_eq!(built.public[at], nt, "the spine's final t_r");
    at += 1;
    // The merged rounds, natively: outer gamma-combination through
    // fold_round_claim. The native verify already enforced
    // `running == q_eval * v` (the assist stays native), so the published
    // running matching this replica closes the outer chain.
    {
        let mut mt = F128::ZERO;
        for pd in &gammas {
            mt += chals[pd.ch] * vals_rec[pd.val_v];
        }
        for rr in &w_rounds {
            let (g1, gi, rch) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1], chals[rr.ch]);
            mt = (mt + g1) + (mt + gi) * rch + gi * (rch * rch);
        }
        assert_eq!(built.public[at], mt, "the merged running claim");
        at += 1;
    }
    // The PIOP: every zero-delta must BE zero, and the lincheck target must
    // match a native replay (which is the deferred assertion's target).
    {
        for i in 0..piop.zc_rounds.len() {
            assert_eq!(built.public[at + i], F128::ZERO, "zc delta {i}");
        }
        at += piop.zc_rounds.len();
        assert_eq!(built.public[at], F128::ZERO, "zc final delta");
        at += 1;
        assert_eq!(built.public[at], F128::ZERO, "ec join");
        at += 1;
        // native lincheck replay
        let mut running = F128::ZERO;
        for (i, rr) in piop.zc_rounds.iter().enumerate() {
            let (g1, gi) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let (t, rho) = (chals[piop.tau_ch + i], chals[rr.ch]);
            let g0 = (running + t * g1) * (F128::ONE + t).inv();
            running = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        let (ea, eb, ec) = (
            rec.values()[piop.eab_v],
            rec.values()[piop.eab_v + 1],
            rec.values()[piop.eab_v + 2],
        );
        assert_eq!(running, ea * eb + ec, "native zc closes");
        let nzc = piop.zc_rounds.len();
        let (r0, r1) = (
            chals[piop.zc_rounds[nzc - 2].ch],
            chals[piop.zc_rounds[nzc - 1].ch],
        );
        let eqt = [
            (F128::ONE + r0) * (F128::ONE + r1),
            r0 * (F128::ONE + r1),
            (F128::ONE + r0) * r1,
            r0 * r1,
        ];
        let (mut va, mut vb) = (ea, eb);
        for c in 0..4 {
            va += eqt[c] * inner_ty.a_const()[c];
            vb += eqt[c] * inner_ty.b_const()[c];
        }
        let mut lrn = va + chals[piop.alpha_ch] * vb;
        for rr in &piop.lc_rounds {
            let (e1, ei) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let rho = chals[rr.ch];
            let e0 = lrn + e1;
            lrn = ei * rho * rho + (e0 + e1 + ei) * rho + e0;
        }
        assert_eq!(built.public[at], lrn, "the deferred assertion's target");
        at += 1;
    }
    // The residual region against its native replica (shared checker); the
    // published inner equals the TRUE t_r of the native verify (which
    // accepted).
    let inner_n = check_residual_publics(
        &built.public,
        at,
        &levels,
        &geo,
        &w_rounds,
        inner_pd.ch,
        &rec.values()[yr_v..yr_v + yr_len],
        &chals,
    );
    // THE CLOSURE: with the start target bound, the spine's t_r and the
    // residual side's inner are the same statement scalar — the native
    // verifier's final check, now enforced between two circuit outputs.
    assert_eq!(inner_n, nt, "inner == t_r: the statement closes");

    // ---- prove / verify the circuit itself ----
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
    let spread_ty = BitSpreadTable::new(spread_w);
    let spread_r1cs = spread_ty.build_block_r1cs(nu);
    let spread_lc = spread_r1cs.csc_lincheck_circuit();

    let (b3_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu)
    });
    let swap_wit = SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
    let spread_wit =
        spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape.registry_slot(slots.b3), b3_lc),
        (shape.registry_slot(slots.swap), swap_lc),
        (shape.registry_slot(slots.spread), spread_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((cproof, ccommitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape.registry_slot(slots.b3),
                UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
            ),
            (
                shape.registry_slot(slots.swap),
                UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
            ),
            (
                shape.registry_slot(slots.spread),
                UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
            ),
        ];
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape.counts[i], nu))
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (cproof, ccommitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (cproof, ccommitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &ccommitment,
            &cproof,
            &pcs_params,
            &mut c,
        )
        .expect("the real query phase verifies")
    });

    println!(
        "\nMVP-7 REAL QUERY PHASE (inner: pure-element m=22, BLAKE3)\n  \
         inner prove {inner_prove_ms:.0} ms | levels {:?} (q, depth, cap, path)\n  \
         blake3 {} rows ({} chain) | swap {} | spread {} | leaf-eval slots {:?}\n  \
         nu {nu} | dense_m {} | mu {}\n\n  \
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         SETUP         {setup_ms:6.0} ms\n",
        geo.iter()
            .map(|g| (g.q, g.depth, g.c, g.path))
            .collect::<Vec<_>>(),
        shape.counts[shape.registry_slot(slots.b3)],
        trace.rows.len(),
        shape.counts[shape.registry_slot(slots.swap)],
        shape.counts[shape.registry_slot(slots.spread)],
        leaf_slot.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        union.dense_m(),
        shape.circuit.cells().mu(),
        bincode::serialize(&cproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

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

/// One opened Ligerito level's geometry, as the proof itself reports it.
struct Lvl {
    q: usize,
    c: usize,
    path: usize,
    depth: usize,
    /// The FOLD width `2^folds` — the lane-weight domain.
    lanes: usize,
    /// The COMMITTED width: `num_lanes` active lanes, which for a mixed
    /// union is an arbitrary integer `<= lanes` (the top lanes are
    /// definitionally zero and never encoded). Equal to `lanes` whenever
    /// the lane count happens to be a power of two.
    row_words: usize,
}

/// The per-level `(cap, opened rows, flat sibling paths)` triples a Ligerito
/// proof reports, in level order: L0's initial cap, then each recursive cap,
/// with the FINAL level reusing the last recursive cap. Since Merkle
/// capping, this is the whole witness a query phase needs — the proof itself
/// carries it, so no prover data is ever plumbed through.
fn level_sources(
    lig: &flock_core::pcs::ligerito::LigeritoProof,
) -> Vec<(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)> {
    let r = lig.recursive_caps.len();
    (0..=r)
        .map(|li| {
            if li == 0 {
                (
                    lig.initial_cap.as_slice(),
                    &lig.initial_proof.opened_rows,
                    &lig.initial_proof.merkle_proof,
                )
            } else if li < r {
                (
                    lig.recursive_caps[li - 1].as_slice(),
                    &lig.recursive_proofs[li - 1].opened_rows,
                    &lig.recursive_proofs[li - 1].merkle_proof,
                )
            } else {
                (
                    lig.recursive_caps[r - 1].as_slice(),
                    &lig.final_proof.opened_rows,
                    &lig.final_proof.merkle_proof,
                )
            }
        })
        .collect()
}

/// Per level: `q`, cap bits `c`, path length `d − c`, depth, lanes — plus
/// the NATIVE cross-checks that pin every piece of the plumbing before any
/// circuit exists: each opened row verifies against its cap under the
/// recorded challenge, and the recorded weights reproduce
/// `induce_sumcheck_enforced_sum`. Returns `(geo, native_sums)`; the sums
/// are what the in-circuit leaf-eval accumulators must equal.
fn level_geometry(
    levels: &[OpenLevel],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
    chals: &[F128],
    hash: HashKind,
) -> (Vec<Lvl>, Vec<F128>) {
    use flock_core::lincheck::build_eq_table;
    let mut geo: Vec<Lvl> = Vec::new();
    let mut native_sums: Vec<F128> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let (cap, rows, paths) = lvl_src[li];
        let q = lvl.q_count;
        assert_eq!(rows.len(), q, "L{li}: one opened row per query");
        let c = cap.len().trailing_zeros() as usize;
        assert_eq!(cap.len(), 1 << c, "L{li}: cap is a power of two");
        let path = paths.len() / q;
        assert_eq!(paths.len(), q * path, "L{li}: flat paths divide evenly");
        let depth = path + c;
        // The lane-fold weights are `2^folds` wide; the committed row may be
        // NARROWER (its top lanes are definitionally zero), and the dot below
        // zips — which IS the zero-fill, exactly as the native verifier does.
        let lanes = 1usize << lvl.fold_fins.len();
        let row_words = rows[0].len();
        assert!(
            row_words >= 1 && row_words <= lanes,
            "L{li}: opened width {row_words} must fit the fold width {lanes}"
        );
        let fold_vals: Vec<F128> = lvl.fold_chs.iter().map(|&i| chals[i]).collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let eqv = build_eq_table(&fold_vals);
        let aw = build_eq_table(&alpha_vals);
        let mut sum = F128::ZERO;
        for (k, row) in rows.iter().enumerate() {
            let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << depth) - 1);
            let mut leaf_bytes = Vec::with_capacity(16 * lanes);
            for f in row {
                leaf_bytes.extend_from_slice(&f.lo.to_le_bytes());
                leaf_bytes.extend_from_slice(&f.hi.to_le_bytes());
            }
            let lh = core_merkle::hash_leaf(&leaf_bytes, hash);
            assert!(
                core_merkle::verify_merkle_proof_capped(
                    cap,
                    1 << depth,
                    &lh,
                    pos,
                    &paths[k * path..(k + 1) * path],
                    hash,
                ),
                "L{li} query {k}: capped path verifies natively"
            );
            let dot = row
                .iter()
                .zip(eqv.iter())
                .map(|(&x, &e)| x * e)
                .fold(F128::ZERO, |a, v| a + v);
            sum += aw[k] * dot;
        }
        native_sums.push(sum);
        geo.push(Lvl {
            q,
            c,
            path,
            depth,
            lanes,
            row_words,
        });
    }
    (geo, native_sums)
}

/// Stream-word indices per `observe_bytes` payload, in payload-word order.
fn payload_words(stream: &flock_core::transcript_record::Stream) -> Vec<Vec<usize>> {
    use flock_core::transcript_record::StreamWord;
    let mut pay_words: Vec<Vec<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let StreamWord::Bytes { payload, word } = *w {
            if pay_words.len() <= payload {
                pay_words.resize(payload + 1, Vec::new());
            }
            assert_eq!(pay_words[payload].len(), word, "payload words in order");
            pay_words[payload].push(wi);
        }
    }
    pay_words
}

/// Locate each level's absorbed cap payload in the stream: one payload
/// index per level, in level order.
///
/// Payloads are CONTENT-matched — the flattened cap bytes must equal a
/// whole `observe_bytes` payload — searching FORWARD (levels absorb their
/// caps in transcript order: the statement's L0 cap first, then each
/// recursion round's), so a size collision with another absorbed surface
/// (the sigma V cap, a child's publics payload) cannot mislocate: a
/// different tree's 32-byte digests never reproduce this cap's bytes.
///
/// Entry 0 is the L0 cap — the COMMITMENT, a statement surface that stays
/// public. Entries 1.. are the recursive caps — PROOF BODY: since the
/// in-circuit cap trees bind them (chain + root connects, nothing
/// checker-read), their payloads demote to witness in `pub_payloads`.
fn cap_payloads(
    stream: &flock_core::transcript_record::Stream,
    bytes: &[u8],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
) -> Vec<usize> {
    let pay_words = payload_words(stream);
    let mut out = Vec::with_capacity(lvl_src.len());
    let mut from = 0usize;
    for (li, (cap, _, _)) in lvl_src.iter().enumerate() {
        let flat: Vec<u8> = cap.iter().flatten().copied().collect();
        let words = flat.len() / 16;
        let p = (from..pay_words.len())
            .find(|&p| {
                pay_words[p].len() == words
                    && pay_words[p]
                        .iter()
                        .enumerate()
                        .all(|(j, &wi)| bytes[wi * 16..wi * 16 + 16] == flat[j * 16..j * 16 + 16])
            })
            .unwrap_or_else(|| panic!("L{li}: absorbed cap payload located"));
        from = p + 1;
        out.push(p);
    }
    out
}

/// The absorbed caps' node wires: per level, `2^c` word-wire pairs in
/// cap-layer order, read off the [`cap_payloads`]-located payloads.
fn cap_wires(
    stream: &flock_core::transcript_record::Stream,
    word_wire: &[Option<Wire>],
    cap_pays: &[usize],
) -> Vec<Vec<[Wire; 2]>> {
    let pay_words = payload_words(stream);
    cap_pays
        .iter()
        .map(|&p| {
            pay_words[p]
                .chunks(2)
                .map(|c| {
                    [
                        word_wire[c[0]].expect("cap word wired"),
                        word_wire[c[1]].expect("cap word wired"),
                    ]
                })
                .collect()
        })
        .collect()
}

/// ROUND 2 — the H(publics) region: re-derive the child's publics
/// commitment ([`flock_core::union::publics_digest`]) from WITNESS wires
/// and CONNECT it to the absorbed digest payload words.
///
/// Under the v2 statement binding the child's transcript absorbs 32 bytes,
/// not the segment, so the child's public words enter the PARENT as
/// witness; this region is what makes the digest binding structural: 1 KiB
/// chunk chains per leaf (the emit_opening chunk shape, pinned == the
/// native `hash_leaf`), LEFT-FOLDED through PARENT rows (== `hash_pair`) —
/// exactly the `publics_digest` chain, ending in an output-output connect
/// with no gate consumers (no cycles, no checker item). Returns the
/// public-word wires — the future consumers' handle (the wiring
/// recombination's publics-MLE evaluation is the recorded upgrade).
fn emit_publics_hash(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    child_public: &[F128],
    digest_w: [Wire; 2],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> Vec<Wire> {
    assert!(!child_public.is_empty(), "a circuit child has publics");
    let pw: Vec<Wire> = child_public
        .iter()
        .map(|v| {
            vals.push(*v);
            sb.input()
        })
        .collect();
    let pad_w = cw(sb, vals, consts, F128::ZERO);
    let mut cv: Option<[Wire; 2]> = None;
    for leaf in pw.chunks(64) {
        let blocks = leaf.len().div_ceil(4);
        let mut lcv = iv;
        for i in 0..blocks {
            let mut flags = 0u32;
            if i == 0 {
                flags |= CHUNK_START;
            }
            if i + 1 == blocks {
                flags |= CHUNK_END;
            }
            let words = (leaf.len() - 4 * i).min(4);
            let params = cw(sb, vals, consts, pack_params(0, 16 * words as u32, flags));
            let mw = |j: usize| if j < words { leaf[4 * i + j] } else { pad_w };
            let out = sb.gate(s.b3, &[lcv[0], lcv[1], mw(0), mw(1), mw(2), mw(3), params]);
            lcv = [out[0], out[1]];
        }
        cv = Some(match cv {
            None => lcv,
            Some(prev) => {
                let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
                let out = sb.gate(
                    s.b3,
                    &[iv[0], iv[1], prev[0], prev[1], lcv[0], lcv[1], params],
                );
                [out[0], out[1]]
            }
        });
    }
    let root = cv.expect("at least one leaf");
    sb.connect(root[0], digest_w[0]);
    sb.connect(root[1], digest_w[1]);
    pw
}

/// Emit the whole QUERY PHASE — every level's Merkle openings against the
/// absorbed caps, plus the leaf-eval accumulators — as circuit rows.
///
/// This is the class-agnostic half of a deferred verifier: it reads the
/// proof's own rows and paths, wires each query's challenge word straight
/// into the opening (no masking gadget — the relation reads the low `depth`
/// columns), and folds the opened rows against the fold challenges into one
/// accumulator per level.
///
/// **ROUND 1 — the cap is hashed, not selected.** Per level, `2^c − 1`
/// PARENT rows fold the ABSORBED cap wires (`cap_w`, from [`cap_wires`])
/// to one root in fixed positional order — no swaps, the cap layer IS the
/// tree's depth-`c` slice — and every opening runs FULL depth (path
/// siblings from the proof, cap-internal siblings recomputed natively from
/// the cap) and CONNECTS to that root. The absorbed cap is bound to the
/// openings by copy constraint; the per-query boundary select — 3 publics
/// per query and its checker tier — is gone. MVP-7, MVP-9 and MVP-10 all
/// need exactly this, so it lives here once rather than three times.
///
/// Appends the sibling `hints` and the public `vals` in declaration order;
/// returns the per-level alpha wires to publish AFTER every input is
/// declared — publishing inside the loop would interleave with the next
/// level's inputs and misindex the public segment (the recorded MVP-7
/// gotcha) — and the accumulators.
#[allow(clippy::too_many_arguments)]
fn emit_query_phase(
    sb: &mut ShapeBuilder,
    slots: CollapsedSlots,
    iv: [Wire; 2],
    leafeval: &[flock_core::circuit::builder::SlotId],
    levels: &[OpenLevel],
    geo: &[Lvl],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
    sq: &[Vec<usize>],
    outs: &[Vec<Wire>],
    chals: &[F128],
    cap_w: &[Vec<[Wire; 2]>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
) -> (Vec<Vec<Wire>>, Vec<Wire>) {
    use flock_core::lincheck::build_eq_table;
    let mut to_publish: Vec<Vec<Wire>> = Vec::new();
    let mut level_accs: Vec<Wire> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let g = &geo[li];
        let (cap, rows, paths) = lvl_src[li];
        let sqq = &sq[lvl.q_fin];
        let sqa = &sq[lvl.a_fin];
        // The cap-internal tree, natively: level 0 is the cap layer, each
        // next level pairs — the openings' cap-side sibling hints and the
        // in-circuit root's expected value both read from here.
        let mut tree_lvls: Vec<Vec<[u8; 32]>> = vec![cap.to_vec()];
        while tree_lvls.last().unwrap().len() > 1 {
            let next: Vec<[u8; 32]> = tree_lvls
                .last()
                .unwrap()
                .chunks(2)
                .map(|p| core_merkle::hash_pair(&p[0], &p[1], HashKind::Blake3))
                .collect();
            tree_lvls.push(next);
        }
        // The cap tree, in-circuit: 2^c − 1 PARENT rows over the absorbed
        // cap wires in fixed positional order — no swap gates.
        let mut nodes: Vec<[Wire; 2]> = cap_w[li].clone();
        while nodes.len() > 1 {
            let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
            nodes = nodes
                .chunks(2)
                .map(|p| {
                    let out = sb.gate(
                        slots.b3,
                        &[iv[0], iv[1], p[0][0], p[0][1], p[1][0], p[1][1], params],
                    );
                    [out[0], out[1]]
                })
                .collect();
        }
        let cap_root = nodes[0];
        // alpha words: chain outputs, PUBLISHED for the checker's expansion.
        let a_wires: Vec<Wire> = (0..lvl.a_count).map(|j| outs[sqa[j / 4]][j % 4]).collect();
        // v: this level's fold challenges, chain outputs, wired straight in.
        let v_wires: Vec<Wire> = lvl.fold_fins.iter().map(|&f| outs[sq[f][0]][0]).collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        // The hi-group weights of the leaf-eval split: eq over the native
        // values of the fold challenges past the 8-lane gate's three.
        let le_vars = g.lanes.min(8).trailing_zeros() as usize;
        let le_groups = g.lanes >> le_vars;
        let hw = {
            let v_hi: Vec<F128> = lvl.fold_chs[le_vars..].iter().map(|&i| chals[i]).collect();
            build_eq_table(&v_hi)
        };
        let mut acc = cw(sb, vals, consts, F128::ZERO);
        // Zero wire for the fold's known-zero top lanes (only declared when
        // the committed row is narrower than the fold).
        let pad_w = if g.row_words < g.lanes {
            Some(cw(sb, vals, consts, F128::ZERO))
        } else {
            None
        };
        for k in 0..g.q {
            vals.extend_from_slice(&rows[k]);
            let leaf_w: Vec<Wire> = (0..g.row_words).map(|_| sb.input()).collect();
            let cw = outs[sqq[k / 4]][k % 4];
            let cv = emit_opening(sb, slots, iv, &leaf_w, cw, g.depth, 0, Some(consts), vals);
            // Full-depth hints: the proof's path siblings, then the c
            // cap-internal siblings from the native cap tree.
            hints.extend(paths[k * g.path..(k + 1) * g.path].iter().map(hash_to_digest));
            let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << g.depth) - 1);
            let mut idx = pos >> g.path;
            for i in 0..g.c {
                hints.push(hash_to_digest(&tree_lvls[i][idx ^ 1]));
                idx >>= 1;
            }
            // Output-output connects: a multi-producer class with no gate
            // consumers — witgen asserts agreement, no dataflow cycle.
            sb.connect(cv[0], cap_root[0]);
            sb.connect(cv[1], cap_root[1]);
            // The fold reads the full `2^folds` domain: the committed words
            // then the definitionally-zero top lanes.
            let mut fold_w = leaf_w.clone();
            fold_w.resize(g.lanes, pad_w.unwrap_or(leaf_w[0]));
            let lanes = g.lanes.min(8);
            for h in 0..le_groups {
                let mut a_in: Vec<Wire> = fold_w[lanes * h..lanes * (h + 1)].to_vec();
                a_in.extend_from_slice(&v_wires[..le_vars]);
                vals.push(aw[k] * hw[h]);
                a_in.push(sb.input());
                a_in.push(acc);
                acc = sb.gate(leafeval[li], &a_in)[0];
            }
        }
        to_publish.push(a_wires);
        level_accs.push(acc);
    }
    (to_publish, level_accs)
}

/// Emit the RESIDUAL region — the third shared piece of the deferred
/// verifier, after the FS chain and the query phase. Per level, ResidualGate
/// rows accumulate `induce_sumcheck_evaluate_at_residual` (the `next_s`
/// chain from a boundary-bound q_field, a prefix over the LATER levels' fold
/// wires, suffix subset products over the `2^yr` residual positions); the
/// close-out then assembles `eval_b` from gamma' and the W-round wires
/// through ONE `pl_full`-wide prefix slot (shorter calls pad their (a, b)
/// blocks with zero pairs — each padded factor is 1 + 0 + 0 = 1, so the wide
/// gate is exact), folds in each OOD claim and each level's beta-weighted
/// accumulators, and dots the absorbed `yr` words into the residual-side
/// `inner`. MVP-7, MVP-9 and MVP-10 all need exactly this, so it lives here
/// once rather than three times.
///
/// Appends public `vals` in declaration order; the caller publishes the
/// returned accumulators and `inner` AFTER all inputs are declared (the
/// recorded MVP-7 gotcha). Also returns the prefix slot AND ITS WIDTH
/// `min(pl_full, 8)`, which the anchor-expect machinery reuses for its
/// chunked products — longer factor lists seed-chain across rows. (The
/// cap keeps the schema at 19 IO words instead of 2·pl_full + 3; every
/// gate cell-slot is also a wiring gather claim, so schema words are the
/// μ AND claim-count budget.)
///
/// **CHUNKING (the mu-25 fix).** Every gate instantiates at
/// `chunk_log = min(yr_log, 3)` — kappa 6 REGARDLESS of the proof's yr.
/// The real inner's yr = 32 otherwise pushed the close-out schemas to
/// kappa 7-8 (~600 IO words, cell space c = 10, and every O(2^mu) pass
/// paid 16x). A yr > 8 region runs as `2^(yr_log-3)` chunks of 8:
/// - the close-out claims' HIGH-bit eq factors ride the PREFIX SLOT
///   (seed = the claim's prefix product, factors = high coords vs the
///   chunk bits) — wire-bound, no new trust;
/// - the residual rows' high subset factor `sp_hi(h)` rides the CHECKER
///   tier (`awp = aw·sp_hi`, recomputed natively from the validated
///   position by `check_residual_publics` — the alpha-expansion trust
///   class; a wrong value fails the published accumulators);
/// - cross-chunk dots sum through a degenerate `SuffixGate(0)` adder.
/// Shapes with yr <= 8 take the single-chunk path BIT-IDENTICALLY.
#[allow(clippy::too_many_arguments)]
fn emit_residual_region(
    sb: &mut ShapeBuilder,
    leaf_slot: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    levels: &[OpenLevel],
    geo: &[Lvl],
    w_rounds: &[RoundRec],
    inner_pd_fin: usize,
    yr_wires: &[Wire],
    sq: &[Vec<usize>],
    outs: &[Vec<Wire>],
    chals: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
) -> (
    Vec<Vec<Wire>>,
    Wire,
    (flock_core::circuit::builder::SlotId, usize),
) {
    use flock_core::lincheck::build_eq_table;
    let yr_len = yr_wires.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let chunk_log = yr_log.min(3);
    let chunk = 1usize << chunk_log;
    let n_chunks = 1usize << (yr_log - chunk_log);
    let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
    let chw = |fin: usize| -> Wire { outs[sq[fin][0]][0] };
    let mut resid_pub: Vec<Vec<Wire>> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
        let lmc_full = pl + yr_log;
        let sks_full = sk_at_vks(lmc_full);
        let lmc = pl + chunk_log;
        let sks = sk_at_vks(lmc);
        debug_assert_eq!(&sks[..], &sks_full[..lmc + 1], "sk_at_vks is prefix-stable");
        // Cache-keyed so a SECOND same-shape region (the mvp11 merge node's
        // two children) reuses the slot instead of duplicating its columns.
        // Reuse is sound exactly when the constructor parameters match —
        // per-level keys, so same-shape callers only.
        let rslot = match leaf_slot.iter().find(|&&(k, _)| k == 100 + li) {
            Some(&(_, s)) => s,
            None => {
                let s = sb.slot(ResidualGate::new(lmc, pl, chunk_log, &sks));
                leaf_slot.push((100 + li, s));
                s
            }
        };
        let ris_w: Vec<Wire> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_fins.iter().map(|&f| chw(f)))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        let mut accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
        for k in 0..geo[li].q {
            let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
            // The high subset factors sp_hi(h), natively from the full
            // chain (the checker tier — see the doc comment).
            let sp_hi: Vec<F128> = {
                let mut sk = Vec::with_capacity(lmc_full);
                if lmc_full > 0 {
                    sk.push(F128::new(pos as u64, 0));
                    for j in 1..lmc_full {
                        sk.push(sk[j - 1] * sk[j - 1] + sks_full[j - 1] * sk[j - 1]);
                    }
                }
                let w_hi: Vec<F128> = (chunk_log..yr_log)
                    .map(|j| sk[pl + j] * inv(sks_full[pl + j]))
                    .collect();
                (0..n_chunks)
                    .map(|h| {
                        let mut p = F128::ONE;
                        for (j, &wj) in w_hi.iter().enumerate() {
                            if (h >> j) & 1 == 1 {
                                p *= wj;
                            }
                        }
                        p
                    })
                    .collect()
            };
            for (h, &sph) in sp_hi.iter().enumerate() {
                // WITNESS advice (the alpha-expansion tier): the checker
                // recomputes aw·sp_hi natively and validates the published
                // ACCS — these values were never read as publics.
                vals.push(F128::new(pos as u64, 0));
                let qf = sb.input();
                vals.push(aw[k] * sph);
                let awp = sb.input();
                let mut g_in = vec![qf];
                g_in.extend_from_slice(&ris_w);
                g_in.push(awp);
                g_in.push(ow);
                g_in.extend_from_slice(&accs[h * chunk..(h + 1) * chunk]);
                let out = sb.gate(rslot, &g_in);
                accs[h * chunk..(h + 1) * chunk].copy_from_slice(&out);
            }
        }
        resid_pub.push(accs);
    }
    // The close-out. The ligerito layer sees ONE packed-direct claim:
    // (rho, q_eval) with gamma'; rho's coords are the W-round squeezes —
    // chain wires. The OOD claims are the same shape, seed = beta, point =
    // the squeezed z.
    let pl_full: usize = levels.iter().map(|l| l.fold_fins.len()).sum();
    let ris_full: Vec<Wire> = levels
        .iter()
        .flat_map(|l| l.fold_fins.iter().map(|&f| chw(f)))
        .collect();
    let sxslot = match leaf_slot.iter().find(|&&(k, _)| k == 300) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(SuffixGate::new(chunk_log));
            leaf_slot.push((300, s));
            s
        }
    };
    let pf_w = pl_full.min(8);
    let pfslot = match leaf_slot.iter().find(|&&(k, _)| k == 310 + pf_w) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PrefixGate::new(pf_w));
            leaf_slot.push((310 + pf_w, s));
            s
        }
    };
    // Seed-chained prefix product: any factor list, `pf_w` per row.
    let prefix_chain = |sb: &mut ShapeBuilder, seed: Wire, factors: &[(Wire, Wire)]| -> Wire {
        let mut s = seed;
        for chunk_f in factors.chunks(pf_w) {
            let mut g_in = vec![s];
            for (a, _) in chunk_f {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk_f.len()));
            for (_, b) in chunk_f {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk_f.len()));
            g_in.push(ow);
            s = sb.gate(pfslot, &g_in)[0];
        }
        s
    };
    let mut evb_accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
    // Fold one claim (prefix product p at full-yl coord wires) into the
    // accumulators: per chunk, the high-coord eq factor is a prefix row
    // seeded by p (wire-bound), then a suffix row over the low coords.
    let apply_suffix =
        |sb: &mut ShapeBuilder, evb_accs: &mut [Wire], p: Wire, coords: &[Wire]| {
            assert_eq!(coords.len(), yr_log, "the claim tail spans yr");
            for h in 0..n_chunks {
                let ph = if n_chunks == 1 {
                    p
                } else {
                    let factors: Vec<(Wire, Wire)> = coords[chunk_log..]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, if (h >> j) & 1 == 1 { ow } else { zw }))
                        .collect();
                    prefix_chain(sb, p, &factors)
                };
                let mut s_in = vec![ph];
                s_in.extend_from_slice(&coords[..chunk_log]);
                s_in.push(ow);
                s_in.extend_from_slice(&evb_accs[h * chunk..(h + 1) * chunk]);
                let out = sb.gate(sxslot, &s_in);
                evb_accs[h * chunk..(h + 1) * chunk].copy_from_slice(&out);
            }
        };
    {
        assert_eq!(w_rounds.len(), pl_full + yr_log, "rho spans the dense domain");
        let factors: Vec<(Wire, Wire)> = w_rounds[..pl_full]
            .iter()
            .map(|rr| chw(rr.fin))
            .zip(ris_full.iter().copied())
            .collect();
        let pw = prefix_chain(sb, chw(inner_pd_fin), &factors);
        let coords: Vec<Wire> = w_rounds[pl_full..].iter().map(|rr| chw(rr.fin)).collect();
        apply_suffix(sb, &mut evb_accs, pw, &coords);
    }
    for (li, lvl) in levels.iter().enumerate() {
        for od in &lvl.ood {
            let folded = od.z_len - yr_log;
            let later: Vec<Wire> = levels[li + 1..]
                .iter()
                .flat_map(|l| l.fold_fins.iter().map(|&f| chw(f)))
                .collect();
            assert_eq!(later.len(), folded, "OOD prefix = later folds");
            let sqz = &sq[od.z_fin];
            let factors: Vec<(Wire, Wire)> = (0..folded)
                .map(|j| (outs[sqz[j / 4]][j % 4], later[j]))
                .collect();
            let pw = prefix_chain(sb, chw(od.beta_fin), &factors);
            let coords: Vec<Wire> = (0..yr_log)
                .map(|j| {
                    let jj = folded + j;
                    outs[sqz[jj / 4]][jj % 4]
                })
                .collect();
            apply_suffix(sb, &mut evb_accs, pw, &coords);
        }
    }
    // beta-weighted residuals fold in per level, then the yr dot. The
    // combine rows are pure accumulate — no cross-position structure — so
    // they sub-chunk to 4-wide (17 cols, kappa 5): rows are live-prefix
    // cheap, columns are the envelope.
    let pc_log = chunk_log.min(2);
    let pc = 1usize << pc_log;
    let pcslot = match leaf_slot.iter().find(|&&(k, _)| k == 301) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PartialCombineGate::new(pc_log));
            leaf_slot.push((301, s));
            s
        }
    };
    let mut comb = evb_accs;
    for (li, lvl) in levels.iter().enumerate() {
        for h in 0..(yr_len / pc) {
            let mut g_in = vec![chw(lvl.beta_fin)];
            g_in.extend_from_slice(&comb[h * pc..(h + 1) * pc]);
            g_in.extend_from_slice(&resid_pub[li][h * pc..(h + 1) * pc]);
            let out = sb.gate(pcslot, &g_in);
            comb[h * pc..(h + 1) * pc].copy_from_slice(&out);
        }
    }
    let fdslot = match leaf_slot.iter().find(|&&(k, _)| k == 302) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(FinalDotGate::new(chunk_log));
            leaf_slot.push((302, s));
            s
        }
    };
    let mut inner_w: Option<Wire> = None;
    for h in 0..n_chunks {
        let mut g_in: Vec<Wire> = yr_wires[h * chunk..(h + 1) * chunk].to_vec();
        g_in.extend_from_slice(&comb[h * chunk..(h + 1) * chunk]);
        let dot = sb.gate(fdslot, &g_in)[0];
        inner_w = Some(match inner_w {
            None => dot,
            Some(acc) => {
                // Cross-chunk adder as a SUFFIX row with a zero point:
                // e = [1, 0, ..], so out[0] = acc + dot·1 — no extra type.
                let mut s_in = vec![dot];
                s_in.extend(std::iter::repeat_n(zw, chunk_log));
                s_in.push(ow);
                s_in.push(acc);
                s_in.extend(std::iter::repeat_n(zw, chunk - 1));
                sb.gate(sxslot, &s_in)[0]
            }
        });
    }
    let inner_w = inner_w.expect("at least one chunk");
    (resid_pub, inner_w, (pfslot, pf_w))
}

/// Check the residual region's published wires against a NATIVE replica:
/// `induce_sumcheck_evaluate_at_residual` per level (sks replicated via
/// `sk_at_vks` — the mvp7 discipline), then the close-out's gamma-weighted
/// char-2 eq products and the yr dot. Walks `public` from `at` (the first
/// accumulator, `levels × 2^yr` entries, then the inner), asserting each.
/// Returns the native inner — the residual-side t_r — so the caller asserts
/// the `inner == t_r` closure in its own indexing.
#[allow(clippy::too_many_arguments)]
fn check_residual_publics(
    public: &[F128],
    at: usize,
    levels: &[OpenLevel],
    geo: &[Lvl],
    w_rounds: &[RoundRec],
    inner_pd_ch: usize,
    yr_vals: &[F128],
    chals: &[F128],
) -> F128 {
    use flock_core::lincheck::build_eq_table;
    let yr_len = yr_vals.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let mut at = at;
    let mut resid_native: Vec<Vec<F128>> = vec![vec![F128::ZERO; yr_len]; levels.len()];
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
        let lmc = pl + yr_log;
        let sks = sk_at_vks(lmc);
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let ris: Vec<F128> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        for y in 0..yr_len {
            let mut sum = F128::ZERO;
            for k in 0..geo[li].q {
                let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
                let mut sk = Vec::with_capacity(lmc);
                if lmc > 0 {
                    sk.push(F128::new(pos as u64, 0));
                    for j in 1..lmc {
                        sk.push(sk[j - 1] * sk[j - 1] + sks[j - 1] * sk[j - 1]);
                    }
                }
                let mut prod = F128::ONE;
                for j in 0..pl {
                    prod *= F128::ONE + ris[j] * (F128::ONE + sk[j] * inv(sks[j]));
                }
                for j in 0..yr_log {
                    if (y >> j) & 1 == 1 {
                        prod *= sk[pl + j] * inv(sks[pl + j]);
                    }
                }
                sum += aw[k] * prod;
            }
            assert_eq!(public[at], sum, "L{li} residual y={y}");
            resid_native[li][y] = sum;
            at += 1;
        }
    }
    // evb + combine, natively: gamma-weighted char-2 eq products, then the
    // yr dot.
    let ris_v: Vec<F128> = levels
        .iter()
        .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
        .collect();
    let pl_full = ris_v.len();
    let mut inner_n = F128::ZERO;
    for y in 0..yr_len {
        let mut evb = chals[inner_pd_ch];
        for j in 0..pl_full {
            evb *= F128::ONE + chals[w_rounds[j].ch] + ris_v[j];
        }
        for j in 0..yr_log {
            evb *= if (y >> j) & 1 == 1 {
                chals[w_rounds[pl_full + j].ch]
            } else {
                F128::ONE + chals[w_rounds[pl_full + j].ch]
            };
        }
        let mut comb = evb;
        for (li, lvl) in levels.iter().enumerate() {
            comb += chals[lvl.beta_ch] * resid_native[li][y];
            for od in &lvl.ood {
                let folded = od.z_len - yr_log;
                let later: Vec<F128> = levels[li + 1..]
                    .iter()
                    .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
                    .collect();
                let mut t = chals[od.beta_ch];
                for j in 0..folded {
                    t *= F128::ONE + chals[od.z_ch + j] + later[j];
                }
                for j in 0..yr_log {
                    t *= if (y >> j) & 1 == 1 {
                        chals[od.z_ch + folded + j]
                    } else {
                        F128::ONE + chals[od.z_ch + folded + j]
                    };
                }
                comb += t;
            }
        }
        inner_n += yr_vals[y] * comb;
    }
    assert_eq!(public[at], inner_n, "the close-out inner");
    inner_n
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
    cap_depth: usize,
    mut consts: Option<&mut Vec<(F128, Wire)>>,
    pubs: &mut Vec<F128>,
) -> [Wire; 2] {
    // A leaf need NOT be a whole number of 64-byte blocks: a mixed circuit
    // union commits `num_lanes` ACTIVE lanes (`dense_words.div_ceil(2^log_dim)`
    // — an arbitrary integer, since the top lanes are definitionally zero and
    // never encoded), so a row can be e.g. 61 words = 976 bytes. BLAKE3 hashes
    // that as 16 blocks whose last carries b = 16 bytes with the rest of the
    // message zero — and the compression's `b` is already a free input here,
    // so the partial block costs one zero-padding wire, not a wire-format
    // change. `blocks` counts up to a chunk's 16; larger leaves would need
    // real chunk merging, which nothing here produces.
    assert!(!leaf_w.is_empty(), "a leaf has data");
    let blocks = leaf_w.len().div_ceil(4);
    assert!(blocks <= 16, "a leaf is one BLAKE3 chunk (<= 1024 bytes)");
    let mut shared = |sb: &mut ShapeBuilder, pubs: &mut Vec<F128>, v: F128| -> Wire {
        match consts.as_deref_mut() {
            Some(c) => cw(sb, pubs, c, v),
            None => {
                pubs.push(v);
                sb.public_input()
            }
        }
    };
    let pad_w = if leaf_w.len() % 4 == 0 {
        None
    } else {
        Some(shared(sb, pubs, F128::ZERO))
    };

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
        // The final block carries only the bytes that remain.
        let words = (leaf_w.len() - 4 * i).min(4);
        let params = shared(sb, pubs, pack_params(0, 16 * words as u32, flags));
        let mw = |j: usize| -> Wire {
            if j < words {
                leaf_w[4 * i + j]
            } else {
                pad_w.expect("a short block needs the zero pad")
            }
        };
        let out = sb.gate(
            s.b3,
            &[cv[0], cv[1], mw(0), mw(1), mw(2), mw(3), params],
        );
        cv = [out[0], out[1]];
    }

    // Node levels: swap, then a PARENT compression over the swapped pair.
    // The sibling is the swap's hint, supplied at `run` time in this call
    // order — setup has no values. Under capping the fold stops `cap_depth`
    // levels below the root: the returned digest is the depth-`cap_depth`
    // ancestor, which the CHECKER compares against the absorbed cap layer
    // (the boundary select — the circuit never touches the cap words).
    // `cap_depth = 0` is the uncapped statement, terminal = root.
    for l in 0..(depth - cap_depth) {
        let sw = sb.gate_hinted(s.swap, &[bits[l], cv[0], cv[1]]);
        let params = shared(sb, pubs, pack_params(0, 64, PARENT));
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
                &mut sb, slots, iv, &leaf_w, index_w, depth, 0, None, &mut pubs,
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
///
/// **Capped openings (the real protocol since Merkle capping)**: each level
/// absorbs its depth-c cap layer (`c = cap_depth(q, d)`) instead of a root,
/// paths fold only `d − c` levels, and the terminal digest is bound by the
/// BOUNDARY SELECT — the circuit publishes `(challenge, digest)` per query
/// and the checker compares the digest against `cap[pos >> (d − c)]`
/// natively. The c-bit select costs the circuit nothing; the price is one
/// extra published word per query and ~18 KiB more FS absorb (the caps),
/// against ~3.3k fewer BLAKE3 rows and ~3.3k fewer swaps.
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
    // Capping: the commitment is the depth-c cap layer, absorbed flattened
    // where the root used to be. The synthetic trees are core merkle trees,
    // so `cap_layer` reads the caps straight out of them.
    let cap_depths: Vec<usize> = levels
        .iter()
        .map(|l| core_merkle::cap_depth(l.queries, l.depth))
        .collect();
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    let mut chals: Vec<Vec<F128>> = Vec::new();
    let mut want: Vec<Vec<usize>> = Vec::new();
    for (li, (l, tree)) in levels.iter().zip(&trees).enumerate() {
        let cap = core_merkle::cap_layer(&tree.flat, 1 << l.depth, cap_depths[li]);
        rec.observe_bytes(cap.as_flattened());
        let cs = rec.sample_f128_vec(l.queries);
        want.push(
            cs.iter()
                .map(|v| (v.lo as usize) & ((1usize << l.depth) - 1))
                .collect(),
        );
        chals.push(cs);
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
            .zip(&cap_depths)
            .map(|(l, &c)| (16 * l.lanes / 64 + l.depth - c) * l.queries)
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
    let spine = sb.slot(SpineGate::new());
    leaf_slot.push((0, spine));

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
    let mut all_opens: Vec<Vec<(Wire, [Wire; 2])>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        let sq = &trace.squeezes[li];
        let c = cap_depths[li];
        let vars = l.lanes.trailing_zeros() as usize;
        vals.extend_from_slice(&vees[li]);
        let vs: Vec<Wire> = (0..vars).map(|_| sb.public_input()).collect();
        let blocks = 16 * l.lanes / 64;
        let mut opens = Vec::with_capacity(l.queries);
        for k in 0..l.queries {
            let pos = want[li][k];
            let leaf = trees[li].leaf(pos);
            vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            let leaf_w: Vec<Wire> = (0..4 * blocks).map(|_| sb.input()).collect();

            // The challenge word IS the index word — no masking gadget.
            let cw = outs[sq[k / 4]][k % 4];
            let cv = emit_opening(&mut sb, slots, iv, &leaf_w, cw, l.depth, c, None, &mut vals);
            opens.push((cw, cv));
            hints.extend(trees[li].siblings(pos).into_iter().take(l.depth - c));

            // The same leaf words feed the arithmetic.
            let mut a_in = leaf_w;
            a_in.extend_from_slice(&vs);
            vals.push(alphas[li][k]);
            a_in.push(sb.public_input());
            a_in.push(acc);
            acc = sb.gate(leafeval[li], &a_in)[0];
        }
        all_opens.push(opens);
    }
    // The boundary select: each opening publishes its challenge word and its
    // terminal digest. The checker derives the cap index from the challenge
    // natively and compares the digest against the absorbed cap — the c-bit
    // select never enters the circuit. Publishing `cw` (the same wire the
    // spread gate consumes) is what binds the high bits: no index bit floats.
    for opens in &all_opens {
        for (cw, cv) in opens {
            sb.publish(*cw);
            sb.publish(cv[0]);
            sb.publish(cv[1]);
        }
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid collapsed circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

    // Every opening folds to its cap node, and the accumulator is
    // enforced_sum — the root equality of the uncapped statement became a
    // per-query cap-node equality, checked HERE, natively: read the
    // published challenge word, mask, take the high c bits, index the cap.
    let mut at = built.public.len() - 1 - 3 * levels.iter().map(|l| l.queries).sum::<usize>();
    for (li, l) in levels.iter().enumerate() {
        let c = cap_depths[li];
        let cap = core_merkle::cap_layer(&trees[li].flat, 1 << l.depth, c);
        for k in 0..l.queries {
            assert_eq!(built.public[at], chals[li][k], "L{li} challenge {k}");
            let pos = (chals[li][k].lo as usize) & ((1usize << l.depth) - 1);
            assert_eq!(pos, want[li][k], "L{li} position {k}");
            let node = digest_words(&hash_to_digest(&cap[pos >> (l.depth - c)]));
            assert_eq!(
                [built.public[at + 1], built.public[at + 2]],
                node,
                "L{li} cap node {k}"
            );
            at += 3;
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

    // Witnesses once; each prove rep rebuilds its inputs from CLONES outside
    // the timer (`UnionSlotProverInput::new` consumes them).
    let (b3_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu)
    });
    let swap_wit = SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
    let spread_wit =
        spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape.registry_slot(slots.b3), b3_lc),
        (shape.registry_slot(slots.swap), swap_lc),
        (shape.registry_slot(slots.spread), spread_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((proof, commitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape.registry_slot(slots.b3),
                UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
            ),
            (
                shape.registry_slot(slots.swap),
                UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
            ),
            (
                shape.registry_slot(slots.spread),
                UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
            ),
        ];
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape.counts[i], nu))
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (proof, commitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .expect("the collapsed query phase verifies")
    });


    let nnz = |r: &flock_core::r1cs::BlockR1cs| {
        r.a_0.rows.iter().map(|x| x.len()).sum::<usize>()
            + r.b_0.rows.iter().map(|x| x.len()).sum::<usize>()
    };
    println!(
        "\nMVP-6 FULL QUERY PHASE, COLLAPSED (m=26 Fast ladder)\n  \
         blake3 {} rows | swap {} | spread {} | leaf-eval {}+{}\n  \
         lincheck nnz {} (MVP-5: 105145720) | dense {} words | dense_m {} | \
         M_bool {} | mu {}\n\n  \
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         MERGED OPEN   frobenius {:.1} KiB, {} rounds | {} gather claims\n  \
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
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        bincode::serialize(&proof.pcs_open.frobenius).map(|b| b.len()).unwrap_or(0) as f64
            / 1024.0,
        proof.pcs_open.merged_rounds.len(),
        shape.circuit.cells().num_gate_slots(),
    );
}

/// **MVP-9: the boolean LEAF — the recursion tree's real leaf shape
/// (rs×2, pd = 0).** A real blake3 workload proof (blake3 for BOTH the
/// FS chain and the Merkle trees — each default diverges silently) is
/// natively verified under a RecordingChallenger, and the outer circuit
/// grows over the recorded tape in the mvp7 pattern:
///
/// - TAPE PINS, all field-for-field vs the proof: the R = 2 multipoint
///   region (2×128 RS dual values, γ^{128i+j} schedule, T0 → rounds →
///   T_m == anchor.v), the boolean PIOP (zerocheck tau/skip slices/
///   rounds/finals, lincheck rounds + z_partial; matrix_evals are NOT
///   absorbed — deferred proof-side by design), and the ring-switch
///   regions (s_hat_v slices, r_dprime/gamma ordinals) with the whole
///   R = 2 merged boundary replayed: succinct outputs, W-fold,
///   linearized coefficients, running == q_eval·V.
/// - IN-CIRCUIT: the full FS chain; the full query phase (collapsed
///   openings against the absorbed caps, FS-derived v, boundary-
///   expanded alpha, per-level enforced sums == native replicas); the
///   PoW bit predicate (published digest/nonce wires, checker-applied);
///   and the intake W-rounds (target as checker-validated advice — the
///   sc dots are family-H, the bit-matrix transpose — with rho BOUND
///   in-circuit and running published). The outer proves and verifies
///   over the circuit path.
///
/// Remaining for the leaf node: the ligerito spine, the boolean PIOP
/// gates (skip round checker-native first), the R = 2 T0/anchor deltas,
/// MatrixAssertion emission, then the family-H upgrade batch.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp9_boolean_leaf_tape() {
    build_leaf_outer();
}

/// The leaf outer's artifacts, returned by [`build_leaf_outer`] so the
/// recursion swap can consume the proof as ITS inner: the circuit shape
/// (owning registry + counts — `UnionInstance::new(&shape.registry,
/// shape.counts.clone())` reconstructs the instance), the public segment,
/// the BLAKE3/BLAKE3 circuit proof, and the boolean tables whose lincheck
/// circuits a verifier needs (in registry order via the `*_slot` indices).
struct LeafOuter {
    shape: flock_core::circuit::builder::CircuitShape,
    public: Vec<F128>,
    proof: flock_core::proof::R1csProofCircuitMerged,
    commitment: flock_core::pcs::Commitment,
    pcs: PcsParams,
    b3_r1cs: flock_core::r1cs::BlockR1cs,
    swap_r1cs: flock_core::r1cs::BlockR1cs,
    spread_r1cs: flock_core::r1cs::BlockR1cs,
    b3_slot: usize,
    swap_slot: usize,
    spread_slot: usize,
}

/// mvp9's WHOLE construction as the shared builder the swap consumes: the
/// real blake3 workload leaf, its recorded native verify, the outer circuit
/// carrying the leaf's complete deferred verification, and the outer's own
/// prove/verify over the circuit path — BLAKE3 for BOTH the FS chain and
/// the Merkle trees, so the proof is recursable (each default diverges
/// silently otherwise; the two recorded hash gotchas). Every tape pin and
/// native replica stays inside: the builder IS the mvp9 test.
fn build_leaf_outer() -> LeafOuter {
    build_leaf_outer_seeded(0x4D50_9B00)
}

/// [`build_leaf_outer`] with the workload seed exposed: the seed varies only
/// the leaf's compression inputs (cv/message/counter words), so two seeds
/// yield the SAME outer circuit — the 2→1 merge's foldability key — with
/// claims at unrelated FS points, which is what a merge node actually sees.
fn build_leaf_outer_seeded(seed: u64) -> LeafOuter {
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
    // Pin the perf pool BEFORE any rayon touch (the native leaf prove would
    // otherwise auto-initialize the global pool at all cores).
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng(seed);
    let inputs: Vec<blake3::Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect();
    let circuit = setup.r1cs.csc_lincheck_circuit();
    let registry = flock_prover::schedule::Registry::new(
        vec![TableType::from_block_r1cs(&setup.r1cs)],
        setup.r1cs.n_log(),
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    let slot = UnionSlotProverInput::new(
        blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        circuit,
    );
    // BLAKE3 for BOTH the Merkle trees and the FS chain — the circuit's
    // opening gates and chain rows model blake3; the setup's defaults
    // (SHA-256 Merkle) diverge silently otherwise.
    let mut leaf_pcs = setup.pcs_params.clone();
    leaf_pcs.merkle_hash = flock_core::merkle::HashKind::Blake3;
    let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
    let (proof, commitment, _claim) =
        prover::prove_fast_ligerito_union(&union, &leaf_pcs, vec![slot], &mut ch);

    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(DOMAIN, HashKind::Blake3));
    let native_claims = verifier::verify_ligerito_union(
        &union,
        &[circuit],
        &commitment,
        &proof,
        &leaf_pcs,
        &mut rec,
    )
    .expect("the leaf workload proof verifies");
    let t_shape = rec.shape();
    let chals: Vec<F128> = rec.challenges().to_vec();
    let vals_rec = rec.values();

    if std::env::var("MVP9_DUMP").is_ok() {
        for (i, op) in t_shape.ops().iter().enumerate().take(160) {
            eprintln!("op {i:>3}: {op:?}");
        }
    }

    // ---- the boolean PIOP region, located and pinned (phase 2a) ----
    // bind -> zerocheck (skip slices + rounds + finals) -> lincheck
    // (rounds + z_partial + matrix_evals). Every absorbed value is
    // identified against the proof field it carries, so the assembly's
    // wires have named indices — the MVP-8-step-1 pattern.
    {
        use flock_core::transcript_record::TranscriptOp as Op2;
        let ops = t_shape.ops();
        let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
        let bump = |op: &Op2, v: &mut usize, c: &mut usize| match op {
            Op2::SqueezeScalar => *c += 1,
            Op2::SqueezeSlice(n) => *c += n,
            Op2::ObserveScalar => *v += 1,
            Op2::ObserveSlice(n) => *v += n,
            _ => {}
        };
        while !matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-zerocheck-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeSlice(_)), "zc tau lo");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeSlice(_)), "zc tau hi");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        let r1ab_v = v;
        assert!(matches!(ops[i], Op2::ObserveSlice(64)), "round1_ab");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        let r1c_v = v;
        assert!(matches!(ops[i], Op2::ObserveSlice(64)), "round1_c");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeScalar), "z_skip");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert_eq!(&vals_rec[r1ab_v..r1ab_v + 64], &proof.zerocheck.round1_ab[..], "round1_ab words");
        assert_eq!(&vals_rec[r1c_v..r1c_v + 64], &proof.zerocheck.round1_c[..], "round1_c words");
        let mut zc_rounds = Vec::new();
        loop {
            // rounds are [obs, obs, squeeze]; the finals are obs NOT
            // followed by a squeeze.
            if matches!(ops[i], Op2::ObserveScalar)
                && matches!(ops[i + 1], Op2::ObserveScalar)
                && matches!(ops[i + 2], Op2::SqueezeScalar)
            {
                zc_rounds.push((v, c + 1));
                for _ in 0..3 {
                    bump(&ops[i], &mut v, &mut c);
                    i += 1;
                }
            } else {
                break;
            }
        }
        assert_eq!(zc_rounds.len(), proof.zerocheck.multilinear_rounds.len(), "zc rounds");
        for ((g_v, _), want) in zc_rounds.iter().zip(&proof.zerocheck.multilinear_rounds) {
            assert_eq!((vals_rec[*g_v], vals_rec[*g_v + 1]), *want, "zc round msg");
        }
        let mut zc_finals = Vec::new();
        while matches!(ops[i], Op2::ObserveScalar) {
            zc_finals.push(vals_rec[v]);
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let want_finals = [
            proof.zerocheck.final_a_eval,
            proof.zerocheck.final_b_eval,
            proof.zerocheck.final_c_eval,
        ];
        assert_eq!(&want_finals[..zc_finals.len()], &zc_finals[..], "zc finals");
        assert!(
            matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-lincheck-v0"),
            "lincheck label, got {:?}",
            ops[i]
        );
        i += 1;
        let mut lc_pre_squeezes = 0usize;
        while matches!(ops[i], Op2::SqueezeScalar) {
            lc_pre_squeezes += 1;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(lc_pre_squeezes >= 1, "lc alpha");
        let mut lc_rounds = Vec::new();
        while matches!(ops[i], Op2::ObserveScalar)
            && matches!(ops[i + 1], Op2::ObserveScalar)
            && matches!(ops[i + 2], Op2::SqueezeScalar)
        {
            lc_rounds.push((v, c + 1));
            for _ in 0..3 {
                bump(&ops[i], &mut v, &mut c);
                i += 1;
            }
        }
        assert_eq!(lc_rounds.len(), proof.lincheck.rounds.len(), "lc rounds");
        for ((g_v, _), want) in lc_rounds.iter().zip(&proof.lincheck.rounds) {
            assert_eq!((vals_rec[*g_v], vals_rec[*g_v + 1]), *want, "lc round msg");
        }
        // The lincheck tail: z_partial (a 64-slice) and the matrix_evals
        // pairs, in whatever op order — located by value identification.
        let mut zp_v = None;
        let mut tail_scalars = Vec::new();
        while !matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            match ops[i] {
                Op2::ObserveSlice(64) if zp_v.is_none() => zp_v = Some(v),
                Op2::ObserveScalar => tail_scalars.push(vals_rec[v]),
                Op2::ObserveSlice(n) => {
                    tail_scalars.extend_from_slice(&vals_rec[v..v + n]);
                }
                _ => {}
            }
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let zp_v = zp_v.expect("z_partial slice on the tape");
        assert_eq!(
            &vals_rec[zp_v..zp_v + 64],
            &proof.lincheck.z_partial[..],
            "z_partial words"
        );
        // matrix_evals are NOT on the tape — the deferral leaves them
        // proof-side, pinned only by the lincheck's final one-equation
        // check and the accumulator's root discharge. In the outer
        // circuit they enter as published advice bound by that equation
        // (the assertion-emission shape for the leaf).
        assert!(
            tail_scalars.is_empty(),
            "the lincheck tail carries only z_partial"
        );
        assert!(!proof.lincheck.matrix_evals.is_empty(), "the deferred matrix work");
    }

    // The leaf's opening shape: rs×2, pd = 0 — R = 2, P = 0.
    let fro = &proof.pcs_open.frobenius;
    assert_eq!(fro.values.len(), 2, "the boolean class contributes rs×2");
    assert!(fro.values.iter().all(|v| v.len() == 128), "128 values per RS claim");
    assert!(fro.group_values.is_empty(), "no packed-direct claims at the leaf");

    // Locate the multipoint region with a minimal cursor (value/challenge
    // ordinals), exactly as parse_open_levels does for the element inner.
    let ops = t_shape.ops();
    let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
    let bump = |op: &Op, v: &mut usize, c: &mut usize| {
        match op {
            Op::SqueezeScalar => *c += 1,
            Op::SqueezeSlice(n) => *c += n,
            Op::ObserveScalar => *v += 1,
            Op::ObserveSlice(n) => *v += n,
            _ => {}
        }
    };
    while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1") {
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    i += 1;
    let mut val_vs = Vec::new();
    while matches!(ops[i], Op::ObserveScalar) {
        val_vs.push(v);
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert_eq!(val_vs.len(), 256, "2×128 RS dual values absorbed");
    assert!(matches!(ops[i], Op::SqueezeScalar), "multipoint gamma");
    let gamma = chals[c];
    bump(&ops[i], &mut v, &mut c);
    i += 1;
    let mut rounds = Vec::new();
    while matches!(ops[i], Op::ObserveScalar) {
        let g_v = v;
        for _ in 0..2 {
            assert!(matches!(ops[i], Op::ObserveScalar));
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(matches!(ops[i], Op::SqueezeScalar), "mp round");
        rounds.push((g_v, c));
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert!(
        matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
        "anchor label"
    );
    i += 1;
    let anchor_v = v;
    assert!(matches!(ops[i], Op::ObserveScalar));
    bump(&ops[i], &mut v, &mut c);
    i += 1;
    let mut anchor_rounds = 0usize;
    while matches!(ops[i], Op::ObserveScalar) {
        for _ in 0..2 {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(matches!(ops[i], Op::SqueezeScalar), "anchor round");
        anchor_rounds += 1;
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert_eq!(rounds.len(), fro.rounds.len(), "two-product round count");
    assert_eq!(anchor_rounds, fro.anchor.rounds.len(), "anchor round count");
    assert_eq!(vals_rec[anchor_v], fro.anchor.v, "anchor v on the tape");

    // The located stream words ARE the proof's RS dual values, in order.
    for (k, &vi) in val_vs.iter().enumerate() {
        assert_eq!(vals_rec[vi], fro.values[k / 128][k % 128], "RS value {k}");
    }

    // The accept chain with the R = 2 schedule: T0 = Σ γ^{128 i + j}·A_ij
    // folds through the rounds to T_m == anchor.v.
    let mut t = F128::ZERO;
    let mut pw = F128::ONE;
    for &vi in &val_vs {
        t += pw * vals_rec[vi];
        pw *= gamma;
    }
    for &(g_v, ch_ix) in &rounds {
        let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
        let r = chals[ch_ix];
        let g0 = t + g1;
        t = g0 + (g1 + g0 + gi) * r + gi * r * r;
    }
    assert_eq!(t, fro.anchor.v, "T_m must equal the anchor's claimed v (R = 2)");

    // ---- phase 2b step 1: the ring-switch regions pinned, and the R = 2
    // merged boundary replayed from located pieces: the two s_hat_v slices
    // ARE the proof's, r_dprime/gamma ordinals named, the succinct outputs
    // (claim transpose dot) rebuilt, the W-rounds folded to `running`, the
    // linearized coefficients derived from the SAME pub helpers, and
    // running == q_eval·V closed with the R = 2 recombination. This is the
    // exact relation chain the leaf circuit will publish as zero-deltas.
    let (native_target, native_running) = {
        use flock_core::pcs::ring_switch as rs;
        use flock_core::zerocheck::univariate_skip::build_eq;
        let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
        let bump = |op: &Op, v: &mut usize, c: &mut usize| match op {
            Op::SqueezeScalar => *c += 1,
            Op::SqueezeSlice(n) => *c += n,
            Op::ObserveScalar => *v += 1,
            Op::ObserveSlice(n) => *v += n,
            _ => {}
        };
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        i += 1;
        let mut rs_recs: Vec<(usize, usize)> = Vec::new();
        while matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
            i += 1;
            assert!(matches!(ops[i], Op::ObserveSlice(128)), "s_hat_v slice");
            let sv = v;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeSlice(7)), "r_dprime");
            let rc = c;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            rs_recs.push((sv, rc));
        }
        assert_eq!(rs_recs.len(), 2, "rs×2 at the leaf");
        let mut gs = Vec::new();
        for _ in 0..2 {
            assert!(matches!(ops[i], Op::SqueezeScalar), "rs gamma");
            gs.push(chals[c]);
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let mut target = F128::ZERO;
        let mut coeffs: Vec<Vec<F128>> = Vec::new();
        for (k, &(sv, rc)) in rs_recs.iter().enumerate() {
            let shv = &vals_rec[sv..sv + 128];
            assert_eq!(
                shv,
                &proof.pcs_open.ring_switches[k].s_hat_v[..],
                "s_hat_v {k} on the stream"
            );
            let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
            let eq = build_eq(&rdp);
            target += gs[k] * rs::inner_product(&rs::tensor_algebra_transpose(shv), &eq);
            let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
            coeffs.push(rs::linearized_coefficients(&rs::build_fold_byte_table(&scaled)));
        }
        let mut running = target;
        while matches!(ops[i], Op::ObserveScalar)
            && matches!(ops[i + 1], Op::ObserveScalar)
            && matches!(ops[i + 2], Op::SqueezeScalar)
        {
            let (g1, gi) = (vals_rec[v], vals_rec[v + 1]);
            for _ in 0..2 {
                bump(&ops[i], &mut v, &mut c);
                i += 1;
            }
            let r = chals[c];
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            let g0 = running + g1;
            running = g0 + (g1 + g0 + gi) * r + gi * r * r;
        }
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-pcs-packed-direct-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let q_eval = vals_rec[v];
        let mut big_v = F128::ZERO;
        for (k, cs) in coeffs.iter().enumerate() {
            for (j, &cj) in cs.iter().enumerate() {
                if cj.is_zero() {
                    continue;
                }
                let mut x = fro.values[k][j];
                for _ in 0..j {
                    x = x * x;
                }
                big_v += cj * x;
            }
        }
        assert_eq!(running, q_eval * big_v, "the R = 2 merged boundary replays");
        (target, running)
    };

    // ---- phase 2a step 2 + the query-phase port: the leaf's chain AND
    // query phase in-circuit — Merkle openings against the absorbed caps,
    // FS-derived v, boundary-expanded alpha, per-level enforced sums
    // published and checked against native replicas. The mvp7 machinery,
    // instantiated on the boolean leaf tape.
    {
        use flock_core::lincheck::build_eq_table;
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let lig = &proof.pcs_open.inner.ligerito;
        assert_eq!(commitment.cap, lig.initial_cap, "commitment IS the L0 cap");
        let r = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v, piop_o, gammas_o, w_rounds, _mp2, inner_pd2, yr_v2, levels) =
            parse_open_levels(t_shape.ops(), 32 * lig.initial_cap.len(), r);
        assert!(piop_o.is_none(), "a boolean tape has no element PIOP");
        assert!(gammas_o.is_empty(), "no packed-direct claims at the leaf");
        assert_eq!(levels.len(), r + 1);

        let (geo, native_sums) =
            level_geometry(&levels, &lvl_src, &chals, HashKind::Blake3);

        let stream = t_shape.stream_words(DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(stream.finalize_after.len(), fin_ops.len(), "finalize alignment");
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();
        let b3_rows: usize = trace.rows.len()
            + geo
                .iter()
                .map(|g| (g.lanes / 4 + g.depth) * g.q + (1usize << g.c) - 1)
                .sum::<usize>();
        let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
        let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(spread_w),
                nu,
            }),
        };
        let mut leaf_slot: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
        let leafeval: Vec<_> = geo
            .iter()
            .map(|g| {
                let lanes = g.lanes.min(8);
                match leaf_slot.iter().find(|(n, _)| *n == lanes) {
                    Some((_, sl)) => *sl,
                    None => {
                        let sl = sb.slot(LeafEvalGate::new(lanes));
                        leaf_slot.push((lanes, sl));
                        sl
                    }
                }
            })
            .collect();
        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv = [sb.public_input(), sb.public_input()];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let mut pub_payloads = bytes_payload_mask(ops);
        let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
        for &p in &cap_pays[1..] {
            pub_payloads[p] = false;
        }
        let (outs, ww) = emit_fs_chain(
            &mut sb,
            slots.b3,
            iv,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
        );

        // The PoW grinding ops, located and bound (the mvp7 machinery).
        struct PowRec {
            fin: usize,
            pay: usize,
            bits: u32,
        }
        let pows: Vec<PowRec> = {
            use flock_core::transcript_record::TranscriptOp as Op3;
            let mut out = Vec::new();
            let (mut fin, mut pay) = (0usize, 0usize);
            for op in t_shape.ops() {
                if let Op3::Pow { bits } = op {
                    out.push(PowRec {
                        fin,
                        pay,
                        bits: *bits,
                    });
                }
                if op.finalizes() {
                    fin += 1;
                }
                match op {
                    Op3::ObserveBytes(_) | Op3::Pow { .. } => pay += 1,
                    _ => {}
                }
            }
            out
        };
        assert!(!pows.is_empty(), "the Fast profile grinds");
        let pow_pub: Vec<[Wire; 3]> = pows
            .iter()
            .map(|pr| {
                let sq = &trace.squeezes[pr.fin];
                let wi = stream
                    .words
                    .iter()
                    .position(|w| matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pr.pay))
                    .expect("pow nonce stream word");
                let nw = ww[wi].expect("pow nonce wired");
                [outs[sq[0]][0], outs[sq[0]][1], nw]
            })
            .collect();

        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let cap_w = cap_wires(&stream, &ww, &cap_pays);
        let (to_publish, level_accs) = emit_query_phase(
            &mut sb,
            slots,
            iv,
            &leafeval,
            &levels,
            &geo,
            &lvl_src,
            &trace.squeezes,
            &outs,
            &chals,
            &cap_w,
            &mut vals,
            &mut consts,
            &mut hints,
        );
        // ---- the intake W-rounds in-circuit: the RS target enters as
        // CHECKER-VALIDATED advice (its sc dots are family-H — the bit
        // transpose — deferred to the boundary batch), the W-rounds fold
        // it through mrslot binding rho, and `running` publishes; the
        // checker closes target == Σ γ·sc and running == q_eval·V
        // natively (the 2b replay above is the reference).
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
        let mrslot = sb.slot(MergedRoundGate::new());
        leaf_slot.push((400, mrslot));
        vals.push(native_target);
        let tw = sb.public_input();
        let mut runw = tw;
        for rr in &w_rounds {
            let r_w = outs[trace.squeezes[rr.fin][0]][0];
            runw = sb.gate(mrslot, &[runw, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
        }

        // ---- the ligerito spine (2a on the leaf): start = gamma'·q_eval,
        // eval/build per fold round, intro-folds for OODs and levels with
        // the LeafEval accumulators consumed IN-CIRCUIT; the final t_r
        // publishes and is checked against a native replay — equality
        // transitively validates every eval/build/fold gate AND the accs.
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();
        // The assert-zero anchor: a dedicated zero public NO gate consumes,
        // so the zero-delta outputs connected into its class add no
        // dataflow edges (connecting them to the ubiquitous `zw` creates
        // cycles — the acyclicity check draws producer→consumer edges).
        vals.push(F128::ZERO);
        let zassert = sb.public_input();
        let spine = sb.slot(SpineGate::new());
        leaf_slot.push((0, spine));
        let gpw = outs[trace.squeezes[inner_pd2.fin][0]][0];
        let tw0 = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, wv(inner_pd2.q_v), gpw, zw]);
        let mut tsp = tw0[3];
        let st = sb.gate(
            spine,
            &[zw, zw, zw, zw, wv(start_v), wv(start_v + 1), tsp, ow, zw],
        );
        let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
        for (li, lvl) in levels.iter().enumerate() {
            for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                let rw = outs[trace.squeezes[lvl.fold_fins[j]][0]][0];
                let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
                tsp = ev[4];
                let bld = sb.gate(
                    spine,
                    &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tsp, ow, zw],
                );
                (qc, qb, qa) = (bld[0], bld[1], bld[2]);
            }
            if li < r {
                for od in &lvl.ood {
                    let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                    let f = sb.gate(
                        spine,
                        &[
                            qc,
                            qb,
                            qa,
                            tsp,
                            wv(od.intro_v),
                            wv(od.intro_v + 1),
                            wv(od.y_v),
                            bw,
                            zw,
                        ],
                    );
                    (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
                }
                let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
                let f = sb.gate(
                    spine,
                    &[
                        qc,
                        qb,
                        qa,
                        tsp,
                        wv(lvl.intro_v),
                        wv(lvl.intro_v + 1),
                        level_accs[li],
                        bw,
                        zw,
                    ],
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            } else {
                let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
                let f = sb.gate(spine, &[zw, zw, zw, tsp, zw, zw, level_accs[li], bw, zw]);
                tsp = f[3];
            }
        }
        let t_final = tsp;

        // ---- the boolean zerocheck in-circuit ----
        // The SKIP ROUND is fully bound (the family-H pass, item 1): the
        // barycentric numerator recurrence over the Λ nodes (SkipNodeGate,
        // 64 rows — no inversions, no advice), z^(2^j) via spine tr-rows,
        // and the close gate baking the linearized Z_S coefficients and
        // subspace denominators. Its rc output binds final_c_eval and its
        // seed output IS the multilinear chain's entry — the zc-seed
        // advice is gone. The 16 rounds are ZcRoundGate rows: eq weights
        // t_i are the 7 baked ghash constants then the 9 r_outer squeeze
        // wires, rho from the chain, msgs from the stream, g0 as advice
        // with published-zero deltas — family I, no in-circuit inversion.
        let zc0 = {
            use flock_core::transcript_record::TranscriptOp as Op4;
            let ops2 = t_shape.ops();
            let (mut v2, mut c2, mut f2, mut i2) = (0usize, 0usize, 0usize, 0usize);
            let bump2 = |op: &Op4, v: &mut usize, c: &mut usize, f: &mut usize| {
                if op.finalizes() {
                    *f += 1;
                }
                match op {
                    Op4::SqueezeScalar => *c += 1,
                    Op4::SqueezeSlice(n) => *c += n,
                    Op4::ObserveScalar => *v += 1,
                    Op4::ObserveSlice(n) => *v += n,
                    _ => {}
                }
            };
            while !matches!(&ops2[i2], Op4::Label(l) if l.as_slice() == b"flock-zerocheck-v0") {
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                i2 += 1;
            }
            i2 += 1;
            // r_skip: SqueezeSlice(6)
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            // r_outer: SqueezeSlice(9)
            let (outer_ch, outer_fin) = (c2, f2);
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let r1ab_v = v2;
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let r1c_v = v2;
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let (z_ch, z_fin) = (c2, f2);
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let mut rounds2 = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar)
                && matches!(ops2[i2 + 1], Op4::ObserveScalar)
                && matches!(ops2[i2 + 2], Op4::SqueezeScalar)
            {
                let g_v = v2;
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                bump2(&ops2[i2 + 1], &mut v2, &mut c2, &mut f2);
                let (ch, fin) = (c2, f2);
                bump2(&ops2[i2 + 2], &mut v2, &mut c2, &mut f2);
                rounds2.push((g_v, ch, fin));
                i2 += 3;
            }
            // the zc finals (v_a, v_b — the lincheck's entry values)
            let mut finals_v = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar) {
                finals_v.push(v2);
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                i2 += 1;
            }
            assert!(
                matches!(&ops2[i2], Op4::Label(l) if l.as_slice() == b"flock-lincheck-v0"),
                "lincheck label"
            );
            i2 += 1;
            let (alpha_ch2, alpha_fin2) = (c2, f2);
            assert!(matches!(ops2[i2], Op4::SqueezeScalar), "lc alpha");
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let (beta_ch2, beta_fin2) = (c2, f2);
            assert!(matches!(ops2[i2], Op4::SqueezeScalar), "lc beta (const pin)");
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let mut lc_rounds2 = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar)
                && matches!(ops2[i2 + 1], Op4::ObserveScalar)
                && matches!(ops2[i2 + 2], Op4::SqueezeScalar)
            {
                let g_v = v2;
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                bump2(&ops2[i2 + 1], &mut v2, &mut c2, &mut f2);
                let (ch, fin) = (c2, f2);
                bump2(&ops2[i2 + 2], &mut v2, &mut c2, &mut f2);
                lc_rounds2.push((g_v, ch, fin));
                i2 += 3;
            }
            (
                (outer_ch, outer_fin, r1ab_v, r1c_v, z_ch, z_fin),
                rounds2,
                finals_v,
                (alpha_ch2, alpha_fin2, beta_ch2, beta_fin2),
                lc_rounds2,
            )
        };
        let (zc0, zc_rounds2, zc_finals_v, lc_chs, lc_rounds2) = zc0;
        let (outer_ch, outer_fin, r1ab_v, r1c_v, z_ch, z_fin) = zc0;
        let (alpha_ch2, alpha_fin2, beta_ch2, beta_fin2) = lc_chs;
        assert!(zc_finals_v.len() >= 2, "zc finals (v_a, v_b) on the tape");
        // Native seed + g0/running chain.
        use flock_core::zerocheck::multilinear::{
            interpolate_at_z_combined, interpolate_at_z_on_lambda,
        };
        use flock_core::zerocheck::univariate_skip_optimized::{
            medium_challenges_ghash, small_challenges_ghash,
        };
        let zval = chals[z_ch];
        let c_eval = interpolate_at_z_on_lambda(&vals_rec[r1c_v..r1c_v + 64], 6, zval);
        let combined: Vec<F128> = vals_rec[r1ab_v..r1ab_v + 64]
            .iter()
            .zip(&vals_rec[r1c_v..r1c_v + 64])
            .map(|(a, b)| *a + *b)
            .collect();
        let zc_seed = interpolate_at_z_combined(&combined, 6, zval) + c_eval;
        let mut t_vals: Vec<F128> = Vec::new();
        t_vals.extend_from_slice(&small_challenges_ghash());
        t_vals.extend_from_slice(&medium_challenges_ghash());
        for j in 0..zc_rounds2.len() - 7 {
            t_vals.push(chals[outer_ch + j]);
        }
        let mut g0_native = Vec::new();
        let mut zc_run = zc_seed;
        for (k2, &(g_v, ch, _)) in zc_rounds2.iter().enumerate() {
            let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
            let t = t_vals[k2];
            let g0 = (zc_run + t * g1) * (F128::ONE + t).inv();
            g0_native.push(g0);
            let rho = chals[ch];
            zc_run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        let zc_end_native = zc_run;
        // The circuit chain — the skip round first.
        let z_w = outs[trace.squeezes[z_fin][0]][0];
        let mut zpw = vec![z_w];
        for j in 1..7 {
            let p = zpw[j - 1];
            zpw.push(sb.gate(spine, &[zw, zw, zw, zw, zw, zw, p, p, zw])[3]);
        }
        let skslot = sb.slot(SkipNodeGate::new());
        leaf_slot.push((510, skslot));
        let (mut ska, mut skc, mut skab) = (ow, zw, zw);
        for i in 0..64 {
            let lam_w = cw(
                &mut sb,
                &mut vals,
                &mut consts,
                flock_core::field::PHI_8_TABLE[64 + i],
            );
            let g = sb.gate(
                skslot,
                &[ska, skc, skab, z_w, lam_w, wv(r1c_v + i), wv(r1ab_v + i)],
            );
            (ska, skc, skab) = (g[0], g[1], g[2]);
        }
        let scslot = sb.slot(SkipCloseGate::new());
        leaf_slot.push((511, scslot));
        let mut cin = vec![skc, skab];
        cin.extend_from_slice(&zpw);
        let cl = sb.gate(scslot, &cin);
        let (rc_w, seed_w) = (cl[0], cl[1]);
        let zslot = sb.slot(ZcRoundGate::new());
        leaf_slot.push((500, zslot));
        let mut zrw = seed_w;
        // The eq-weight wires, kept in round order: they are exactly the
        // zerocheck's r_rest — the c claim's point — which the anchor
        // expect consumes below.
        let mut zc_t_w: Vec<Wire> = Vec::new();
        for (k2, &(g_v, _, fin)) in zc_rounds2.iter().enumerate() {
            let t_w = if k2 < 7 {
                cw(&mut sb, &mut vals, &mut consts, t_vals[k2])
            } else {
                let j = k2 - 7;
                let sq = &trace.squeezes[outer_fin];
                outs[sq[j / 4]][j % 4]
            };
            zc_t_w.push(t_w);
            let rho_w = outs[trace.squeezes[fin][0]][0];
            vals.push(g0_native[k2]);
            let g0w = sb.input();
            let g = sb.gate(zslot, &[zrw, wv(g_v), wv(g_v + 1), t_w, rho_w, g0w, ow]);
            sb.connect(g[0], zassert);
            zrw = g[1];
        }

        // ---- the lincheck rounds in-circuit ----
        // The entry is FULLY BOUND: target = alpha·v_a + v_b + beta with
        // alpha/beta chain squeezes and v_a/v_b the zerocheck finals on the
        // stream (SpineGate tr-rows). The rounds are MergedRoundGate; the
        // end publishes and equals the native chain (the comb-side final
        // consistency and the deferred matrix equation stay checker-native).
        let alpha_w2 = outs[trace.squeezes[alpha_fin2][0]][0];
        let beta_w2 = outs[trace.squeezes[beta_fin2][0]][0];
        let s1 = sb.gate(
            spine,
            &[zw, zw, zw, zw, zw, zw, wv(zc_finals_v[0]), alpha_w2, zw],
        )[3];
        let s2 = sb.gate(spine, &[zw, zw, zw, s1, zw, zw, wv(zc_finals_v[1]), ow, zw])[3];
        let mut lcw = sb.gate(spine, &[zw, zw, zw, s2, zw, zw, beta_w2, ow, zw])[3];
        for &(g_v, _, fin) in &lc_rounds2 {
            let rho_w = outs[trace.squeezes[fin][0]][0];
            lcw = sb.gate(mrslot, &[lcw, wv(g_v), wv(g_v + 1), rho_w])[0];
        }
        // Native replica of the same chain.
        let lc_end_native = {
            let mut lrn = chals[alpha_ch2] * vals_rec[zc_finals_v[0]]
                + vals_rec[zc_finals_v[1]]
                + chals[beta_ch2];
            for &(g_v, ch, _) in &lc_rounds2 {
                let (e1, ei) = (vals_rec[g_v], vals_rec[g_v + 1]);
                let rho = chals[ch];
                let e0 = lrn + e1;
                lrn = ei * rho * rho + (e0 + e1 + ei) * rho + e0;
            }
            lrn
        };

        // ---- the residual region (mvp7's 2b, ported to the leaf) ----
        // Per-level ResidualGates compute the induced-basis residuals
        // (next_s chain from a boundary-bound q_field, prefix over LATER
        // levels' fold wires, suffix subset products), and the close-out
        // (prefix/suffix/partial-combine/final-dot) assembles eval_b and
        // dots the absorbed yr words — `inner == t_r` then closes between
        // circuit outputs, exactly as on mvp7.
        let yr_len = proof.pcs_open.inner.ligerito.final_proof.yr.len();
        let yr_wires: Vec<Wire> = (0..yr_len).map(|y| wv(yr_v2 + y)).collect();
        let (resid_pub, inner_w, (pfslot2, pf_w)) = emit_residual_region(
            &mut sb,
            &mut leaf_slot,
            &levels,
            &geo,
            &w_rounds,
            inner_pd2.fin,
            &yr_wires,
            &trace.squeezes,
            &outs,
            &chals,
            &mut vals,
            zw,
            ow,
        );
        // THE CLOSURE, in-circuit: inner == t_r as a copy constraint.
        sb.connect(inner_w, t_final);

        // ---- the R = 2 multipoint chains ----
        // T0 = Σ gamma^{128i+j}·A_ij over the 256 absorbed dual values
        // (gamma-power chain + tr-row MACs), the rounds via mrslot,
        // T_m + anchor.v published as a zero-delta, and the anchor's own
        // rounds folded to an endpoint checked against the native replay.
        // The anchor EXPECT (RS statements, ĝ closed form) stays
        // checker-native with the family-H batch.
        let (mp_gamma_ch, mp_gamma_fin, mp_rounds3, mp_anchor_v, mp_anchor_rounds3) = {
            use flock_core::transcript_record::TranscriptOp as Op5;
            let ops3 = t_shape.ops();
            let (mut v3, mut c3, mut f3, mut i3) = (0usize, 0usize, 0usize, 0usize);
            let bump3 = |op: &Op5, v: &mut usize, c: &mut usize, f: &mut usize| {
                if op.finalizes() {
                    *f += 1;
                }
                match op {
                    Op5::SqueezeScalar => *c += 1,
                    Op5::SqueezeSlice(n) => *c += n,
                    Op5::ObserveScalar => *v += 1,
                    Op5::ObserveSlice(n) => *v += n,
                    _ => {}
                }
            };
            while !matches!(&ops3[i3], Op5::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1")
            {
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                i3 += 1;
            }
            i3 += 1;
            while matches!(ops3[i3], Op5::ObserveScalar) {
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                i3 += 1;
            }
            let (gch, gfin) = (c3, f3);
            bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
            i3 += 1;
            let mut rds = Vec::new();
            while matches!(ops3[i3], Op5::ObserveScalar) && !matches!(ops3[i3], Op5::Label(_)) {
                if !matches!(ops3[i3 + 2], Op5::SqueezeScalar) {
                    break;
                }
                let g_v = v3;
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                bump3(&ops3[i3 + 1], &mut v3, &mut c3, &mut f3);
                let (ch, fin) = (c3, f3);
                bump3(&ops3[i3 + 2], &mut v3, &mut c3, &mut f3);
                rds.push((g_v, ch, fin));
                i3 += 3;
            }
            assert!(
                matches!(&ops3[i3], Op5::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0")
            );
            i3 += 1;
            let av = v3;
            bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
            i3 += 1;
            let mut ards = Vec::new();
            while matches!(ops3[i3], Op5::ObserveScalar) {
                let g_v = v3;
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                bump3(&ops3[i3 + 1], &mut v3, &mut c3, &mut f3);
                let (ch, fin) = (c3, f3);
                bump3(&ops3[i3 + 2], &mut v3, &mut c3, &mut f3);
                ards.push((g_v, ch, fin));
                i3 += 3;
            }
            (gch, gfin, rds, av, ards)
        };
        let mp_gamma_w = outs[trace.squeezes[mp_gamma_fin][0]][0];
        let mut mp_t0 = zw;
        let mut mp_pw = ow;
        let mut mp_pws: Vec<Wire> = vec![ow];
        for (k3, &vi) in val_vs.iter().enumerate() {
            mp_t0 = sb.gate(spine, &[zw, zw, zw, mp_t0, zw, zw, wv(vi), mp_pw, zw])[3];
            if k3 + 1 < val_vs.len() {
                mp_pw = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pw, mp_gamma_w, zw])[3];
                mp_pws.push(mp_pw);
            }
        }
        let mut mp_tm = mp_t0;
        let mut mp_rho2_w: Vec<Wire> = Vec::new();
        for &(g_v, _, fin) in &mp_rounds3 {
            let r_w = outs[trace.squeezes[fin][0]][0];
            mp_rho2_w.push(r_w);
            mp_tm = sb.gate(mrslot, &[mp_tm, wv(g_v), wv(g_v + 1), r_w])[0];
        }
        sb.connect(mp_tm, wv(mp_anchor_v));
        let mut anc = wv(mp_anchor_v);
        let mut mp_sig_w: Vec<Wire> = Vec::new();
        for &(g_v, _, fin) in &mp_anchor_rounds3 {
            let r_w = outs[trace.squeezes[fin][0]][0];
            mp_sig_w.push(r_w);
            anc = sb.gate(mrslot, &[anc, wv(g_v), wv(g_v + 1), r_w])[0];
        }
        let anc_end_native = {
            let mut t = vals_rec[mp_anchor_v];
            for &(g_v, ch, _) in &mp_anchor_rounds3 {
                let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
                let r = chals[ch];
                let g0 = t + g1;
                t = g0 + (g1 + g0 + gi) * r + gi * r * r;
            }
            t
        };

        // ---- the R=2 anchor EXPECT in-circuit (family-H pass, item 2) ----
        // The anchor's accept `claim == expect` becomes a published
        // zero-delta: expect = Σ_i γ^{128·i}·ĝ(ρ″)·(w_i·DP_i) over the two
        // RS statements (P = 0 at the leaf; each RS anchor claim has only
        // c[0] ≠ 0, so the singleton statements sit at the UNSQUARED claim
        // points). ĝ(ρ″) = Σ_j γ^j·eq(ρ^{2^-j}, ρ″), with the
        // inverse-Frobenius points as ADVICE bound by forward squaring
        // deltas y·y + prev = 0 (squaring is a bijection in char 2, so the
        // deltas pin the advice exactly — no checker item); every eq
        // product rides the existing prefix slot (char-2 eq factors are
        // affine, zw-padded pairs contribute 1); the DP is AssistLayerGate.
        let m_mp2 = mp_rounds3.len();
        assert_eq!(mp_sig_w.len(), 2 * (m_mp2 + 1), "sigma spans the anchor layers");
        assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
        let n_log_i = union.n_log();
        let params_i = flock_core::pcs::jagged::JaggedParams::from_heights(
            &union.jagged_heights(),
            n_log_i,
            m_mp2,
        );
        let k_cols_i = params_i.k;
        let bounds_i = flock_core::pcs::jagged::assist_boundaries(&params_i);
        let n_runs = bounds_i.len();
        let has_tail = bounds_i[n_runs - 1].0 == bounds_i[n_runs - 1].1;
        let n_single = if has_tail { n_runs - 1 } else { n_runs };
        for &(_, _, len) in &bounds_i[..n_single] {
            assert_eq!(len, 1, "used columns are singleton runs");
        }

        // The statements' points, as (native value, wire) pairs pinned
        // against the native claims: ab = [lc r0 | zc mlv[1..1+ν] | lc
        // r1..], c = r_rest = [7 ghash weights | r_outer] verbatim.
        let mlv_pw: Vec<(F128, Wire)> = zc_rounds2
            .iter()
            .map(|&(_, ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
            .collect();
        // The lincheck binds the TOP bit each round, so r_inner_rest is its
        // round challenges REVERSED (LSB-first address order).
        let lc_pw: Vec<(F128, Wire)> = lc_rounds2
            .iter()
            .rev()
            .map(|&(_, ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
            .collect();
        assert_eq!(lc_pw.len(), 1 + k_cols_i, "lc rounds = 1 + col bits");
        let mut xab_pw: Vec<(F128, Wire)> = vec![lc_pw[0]];
        xab_pw.extend_from_slice(&mlv_pw[1..1 + n_log_i]);
        xab_pw.extend_from_slice(&lc_pw[1..]);
        let mut xc_pw: Vec<(F128, Wire)> = Vec::new();
        for (k2, &tw2) in zc_t_w.iter().enumerate() {
            let nv = if k2 < 7 {
                t_vals[k2]
            } else {
                chals[outer_ch + (k2 - 7)]
            };
            xc_pw.push((nv, tw2));
        }
        let x_ab_n: Vec<F128> = {
            let p = &native_claims.ab.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        let x_c_n: Vec<F128> = {
            let p = &native_claims.c.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        assert_eq!(x_ab_n.len(), 1 + n_log_i + k_cols_i, "ab point split");
        assert_eq!(x_c_n.len(), 1 + n_log_i + k_cols_i, "c point split");
        for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&x_ab_n).enumerate() {
            assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
        }
        for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&x_c_n).enumerate() {
            assert_eq!(nv, xn, "c point coord {i2} is the located wire");
        }

        // Native replica of the whole expect — validates the formula
        // against the accepted proof before any gate exists.
        let frob_inv_native = |x: F128| {
            let mut y = x;
            for _ in 0..127 {
                y = y * y;
            }
            y
        };
        let gamma_n = chals[mp_gamma_ch];
        let mut gpow_n = vec![F128::ONE];
        for j in 1..129 {
            gpow_n.push(gpow_n[j - 1] * gamma_n);
        }
        let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
        let point_n: Vec<F128> = mp_rounds3.iter().map(|&(_, ch, _)| chals[ch]).collect();
        let sig_n: Vec<F128> = mp_anchor_rounds3
            .iter()
            .map(|&(_, ch, _)| chals[ch])
            .collect();
        let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
        let g_at_n = {
            let mut rinv = rho_mrg_n.clone();
            let mut acc = F128::ZERO;
            for (j, &gp) in gpow_n.iter().enumerate().take(128) {
                if j > 0 {
                    for x in rinv.iter_mut() {
                        *x = frob_inv_native(*x);
                    }
                }
                let mut prod = gp;
                for (t2, &x) in point_n.iter().enumerate() {
                    prod *= F128::ONE + rinv[t2] + x;
                }
                acc += prod;
            }
            acc
        };
        let eqc_n: Vec<F128> = bounds_i
            .iter()
            .map(|&(t_c, t_next, _)| {
                let mut p = F128::ONE;
                for l in 0..=m_mp2 {
                    p *= F128::ONE + sig_n[2 * l] + bit((t_c >> l) & 1 == 1);
                    p *= F128::ONE + sig_n[2 * l + 1] + bit((t_next >> l) & 1 == 1);
                }
                p
            })
            .collect();
        let expect_n = {
            let sparse = flock_core::pcs::jagged::assist_sparse_transitions();
            let mut acc = F128::ZERO;
            for (si, xs) in [&x_ab_n, &x_c_n].iter().enumerate() {
                let z_row = &xs[1..1 + n_log_i];
                let z_col = &xs[1 + n_log_i..];
                let mut run_n = vec![F128::ZERO; n_runs];
                let mut tail = F128::ONE;
                for (r, slot2) in run_n.iter_mut().take(n_single).enumerate() {
                    let mut s = F128::ONE;
                    for (jj, &zc) in z_col.iter().enumerate() {
                        s *= F128::ONE + zc + bit((r >> jj) & 1 == 1);
                    }
                    *slot2 = s;
                    tail += s;
                }
                if has_tail {
                    run_n[n_runs - 1] = tail;
                }
                let w_n = run_n
                    .iter()
                    .zip(&eqc_n)
                    .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                let mut g = [F128::ZERO; 4];
                g[flock_core::pcs::jagged::STATE_SUCCESS] = F128::ONE;
                for layer in (0..=m_mp2).rev() {
                    let za = if layer < n_log_i { z_row[layer] } else { F128::ZERO };
                    let rb = if layer < m_mp2 { point_n[layer] } else { F128::ZERO };
                    let eq4 = build_eq_table(&[za, rb]);
                    let (rc, rd) = (sig_n[2 * layer], sig_n[2 * layer + 1]);
                    let e = [
                        (F128::ONE + rc) * (F128::ONE + rd),
                        rc * (F128::ONE + rd),
                        (F128::ONE + rc) * rd,
                        rc * rd,
                    ];
                    let mut prev = [F128::ZERO; 4];
                    for (cd, &ecd) in e.iter().enumerate() {
                        for (s2, slot2) in prev.iter_mut().enumerate() {
                            let (i0, o0) = sparse[cd][s2][0];
                            let (i1, o1) = sparse[cd][s2][1];
                            *slot2 += ecd * (eq4[i0] * g[o0] + eq4[i1] * g[o1]);
                        }
                    }
                    g = prev;
                }
                let coeff = if si == 0 { g_at_n } else { gpow_n[128] * g_at_n };
                acc += coeff * (w_n * g[flock_core::pcs::jagged::STATE_INITIAL]);
            }
            acc
        };
        assert_eq!(
            expect_n, anc_end_native,
            "the R=2 anchor expect replays natively"
        );

        // The circuit side. Advice square-root chains for rho^(2^-j).
        let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
            let mut seed = ow;
            for chunk in factors.chunks(pf_w) {
                let mut g_in = vec![seed];
                for (a, _) in chunk {
                    g_in.push(*a);
                }
                g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
                for (_, b) in chunk {
                    g_in.push(*b);
                }
                g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
                g_in.push(ow);
                seed = sb.gate(pfslot2, &g_in)[0];
            }
            seed
        };
        let mut rinv_n: Vec<F128> = rho_mrg_n.clone();
        let mut rinv_w: Vec<Wire> = w_rounds
            .iter()
            .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
            .collect();

        let mut ghat = zw;
        for j in 0..128 {
            if j > 0 {
                let mut lvl_w = Vec::with_capacity(m_mp2);
                for t2 in 0..m_mp2 {
                    let y = frob_inv_native(rinv_n[t2]);
                    rinv_n[t2] = y;
                    vals.push(y);
                    let yw = sb.input();
                    let d = sb.gate(spine, &[zw, zw, zw, rinv_w[t2], zw, zw, yw, yw, zw])[3];
                    sb.connect(d, zassert);
                    lvl_w.push(yw);
                }
                rinv_w = lvl_w;
            }
            let factors: Vec<(Wire, Wire)> = rinv_w
                .iter()
                .copied()
                .zip(mp_rho2_w.iter().copied())
                .collect();
            let eqj = prefix_product(&mut sb, &factors);
            ghat = sb.gate(spine, &[zw, zw, zw, ghat, zw, zw, mp_pws[j], eqj, zw])[3];
        }
        // Per-run boundary eq products at sigma (statement-independent).
        let eqc_w: Vec<Wire> = bounds_i
            .iter()
            .map(|&(t_c, t_next, _)| {
                let mut factors = Vec::with_capacity(2 * (m_mp2 + 1));
                for l in 0..=m_mp2 {
                    factors.push((mp_sig_w[2 * l], if (t_c >> l) & 1 == 1 { ow } else { zw }));
                    factors
                        .push((mp_sig_w[2 * l + 1], if (t_next >> l) & 1 == 1 { ow } else { zw }));
                }
                prefix_product(&mut sb, &factors)
            })
            .collect();
        // Per statement: run weights, the w dot, the DP, the coefficient.
        let alslot = sb.slot(AssistLayerGate::new());
        leaf_slot.push((601, alslot));
        let mut expect_w = zw;
        for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
            let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
            let z_col_w: Vec<Wire> = xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect();
            let mut run_w: Vec<Wire> = vec![zw; n_runs];
            let mut tail_w = ow;
            for (r, slot2) in run_w.iter_mut().take(n_single).enumerate() {
                let factors: Vec<(Wire, Wire)> = z_col_w
                    .iter()
                    .enumerate()
                    .map(|(jj, &zc)| (zc, if (r >> jj) & 1 == 1 { ow } else { zw }))
                    .collect();
                let s = prefix_product(&mut sb, &factors);
                *slot2 = s;
                tail_w = sb.gate(spine, &[zw, zw, zw, tail_w, zw, zw, s, ow, zw])[3];
            }
            if has_tail {
                run_w[n_runs - 1] = tail_w;
            }
            let mut w_st = zw;
            for (r, &rw) in run_w.iter().enumerate() {
                w_st = sb.gate(spine, &[zw, zw, zw, w_st, zw, zw, rw, eqc_w[r], zw])[3];
            }
            let mut g = [zw, zw, ow, zw]; // STATE_SUCCESS seed
            for layer in (0..=m_mp2).rev() {
                let za = if layer < n_log_i { z_row_w[layer] } else { zw };
                let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
                let mut a_in = g.to_vec();
                a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
                let o = sb.gate(alslot, &a_in);
                g = [o[0], o[1], o[2], o[3]];
            }
            let coeff = if si == 0 {
                ghat
            } else {
                sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pws[128], ghat, zw])[3]
            };
            let wd = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, w_st, g[0], zw])[3];
            expect_w = sb.gate(spine, &[zw, zw, zw, expect_w, zw, zw, coeff, wd, zw])[3];
        }
        // The join: the anchor's folded claim equals the in-circuit expect.
        sb.connect(anc, expect_w);

        // ---- MatrixAssertion emission ----
        // The deferred matrix work exits as bound publics: alpha and the
        // lincheck round challenges (chain wires — the assertion's point),
        // plus the matrix_evals as advice publics (they are NOT absorbed —
        // deferral leaves them proof-side, pinned by the one-equation final
        // check and the root discharge, both checker-native for now).
        let mut assert_pub: Vec<Wire> = vec![alpha_w2];
        for &(_, _, fin) in &lc_rounds2 {
            assert_pub.push(outs[trace.squeezes[fin][0]][0]);
        }
        for &(a, b) in &proof.lincheck.matrix_evals {
            vals.push(a);
            assert_pub.push(sb.public_input());
            vals.push(b);
            assert_pub.push(sb.public_input());
        }

        for a_wires in &to_publish {
            for w in a_wires {
                sb.publish(*w);
            }
        }
        for w in &level_accs {
            sb.publish(*w);
        }
        for pp in &pow_pub {
            for w in pp {
                sb.publish(*w);
            }
        }
        for accs in &resid_pub {
            for w in accs {
                sb.publish(*w);
            }
        }
        sb.publish(inner_w);
        sb.publish(tw);
        sb.publish(runw);
        sb.publish(t_final);
        sb.publish(rc_w);
        sb.publish(seed_w);
        sb.publish(zrw);
        sb.publish(lcw);
        sb.publish(anc);
        for w in &assert_pub {
            sb.publish(*w);
        }
        let shape = sb.finish().expect("valid leaf query-phase circuit");
        let hint_refs: Vec<&dyn std::any::Any> =
            hints.iter().map(|h| h as &dyn std::any::Any).collect();
        let built = shape.run(&vals, &hint_refs);

        // ---- boundary checks: alphas and the enforced sums.
        // The anchor-expect tail (sqrt-chain deltas + the claim==expect
        // delta) is appended after everything else; `plen` is the public
        // length BEFORE it, so every older from-the-end offset holds.
        // The sqrt-chain, anchor-expect, zc-round and T_m == anchor.v
        // identities are COPY CONSTRAINTS — no publics, no checker items.
        let plen = built.public.len();
        let n_assert_pub = 1 + lc_rounds2.len() + 2 * proof.lincheck.matrix_evals.len();
        let total_pub: usize = levels.len()
            + levels.len() * yr_len
            + 1
            + 3
            + 5
            + n_assert_pub
            + 3 * pows.len()
            + levels.iter().map(|l| l.a_count).sum::<usize>();
        let mut at2 = plen - total_pub;
        // The openings bind to the absorbed caps by COPY CONSTRAINT (the
        // in-circuit cap tree) — no per-query publics, no checker walk.
        for (li, lvl) in levels.iter().enumerate() {
            for j in 0..lvl.a_count {
                assert_eq!(built.public[at2 + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
            }
            at2 += lvl.a_count;
        }
        for (li, want) in native_sums.iter().enumerate() {
            assert_eq!(
                built.public[at2 + li],
                *want,
                "L{li} enforced sum matches the native replica"
            );
        }
        // The residual region against the shared native replica (sks via
        // sk_at_vks — the mvp7 discipline).
        let resid_base = at2 + native_sums.len() + 3 * pows.len();
        {
            let inner_n = check_residual_publics(
                &built.public,
                resid_base,
                &levels,
                &geo,
                &w_rounds,
                inner_pd2.ch,
                &vals_rec[yr_v2..yr_v2 + yr_len],
                &chals,
            );
            // THE CLOSURE, between circuit outputs: the residual side's
            // inner and the spine's t_r are the same statement scalar.
            let zc_tail2 = n_assert_pub + 5;
            assert_eq!(
                built.public[plen - zc_tail2 - 1],
                inner_n,
                "inner == t_r: the leaf statement closes"
            );
        }
        let pow_base = at2 + native_sums.len();
        for (k, pr) in pows.iter().enumerate() {
            let d0 = built.public[pow_base + 3 * k];
            let d1 = built.public[pow_base + 3 * k + 1];
            let nn = built.public[pow_base + 3 * k + 2];
            let mut digest = [0u8; 32];
            digest[..8].copy_from_slice(&d0.lo.to_le_bytes());
            digest[8..16].copy_from_slice(&d0.hi.to_le_bytes());
            digest[16..24].copy_from_slice(&d1.lo.to_le_bytes());
            digest[24..].copy_from_slice(&d1.hi.to_le_bytes());
            assert_eq!(nn.hi, 0, "pow {k}: nonce word zero-padded");
            if pr.bits == 0 {
                assert_eq!(nn.lo, 0, "pow {k}: canonical zero nonce");
            } else {
                assert!(
                    flock_core::challenger::pow_has_leading_zero_bits(
                        &digest,
                        nn.lo,
                        pr.bits,
                        HashKind::Blake3,
                    ),
                    "pow {k}: grinding predicate on the published wires"
                );
            }
        }
        // Tail order: [.., zc_end, lc_end, mp_delta, anc, assertion fields].
        {
            let base = plen - n_assert_pub;
            assert_eq!(built.public[base], chals[alpha_ch2], "assertion alpha");
            for (k4, &(_, ch, _)) in lc_rounds2.iter().enumerate() {
                assert_eq!(built.public[base + 1 + k4], chals[ch], "assertion r {k4}");
            }
            let me_base = base + 1 + lc_rounds2.len();
            for (k4, &(a, b)) in proof.lincheck.matrix_evals.iter().enumerate() {
                assert_eq!(built.public[me_base + 2 * k4], a, "matrix eval a {k4}");
                assert_eq!(built.public[me_base + 2 * k4 + 1], b, "matrix eval b {k4}");
            }
        }
        let tail0 = plen - n_assert_pub;
        assert_eq!(
            built.public[tail0 - 1],
            anc_end_native,
            "the anchor rounds end at the native claim"
        );
        assert_eq!(
            built.public[tail0 - 2],
            lc_end_native,
            "the lincheck chain ends at the native running claim"
        );
        let zc_tail = n_assert_pub + 5;
        assert_eq!(
            built.public[plen - zc_tail],
            proof.zerocheck.final_c_eval,
            "the skip interpolation binds final_c_eval"
        );
        assert_eq!(
            built.public[plen - zc_tail + 1],
            zc_seed,
            "the in-circuit skip seed equals the native interpolation"
        );
        assert_eq!(
            built.public[tail0 - 3],
            zc_end_native,
            "the zc chain ends at the native running claim"
        );
        // The intake boundary: the advice target and the in-circuit
        // running, both checker-validated against the native replay.
        assert_eq!(
            built.public[plen - zc_tail - 3],
            native_target,
            "the RS target advice is the native gamma-combination"
        );
        assert_eq!(
            built.public[plen - zc_tail - 2],
            native_running,
            "the W-rounds fold the target to the native running claim"
        );
        // The spine's t_r, against the native quad replay.
        {
            let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
            let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
            let mut nt = chals[inner_pd2.ch] * vals_rec[inner_pd2.q_v];
            let mut nq = quad(vals_rec[start_v], vals_rec[start_v + 1], nt);
            for (li, lvl) in levels.iter().enumerate() {
                for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                    nt = evalq(nq, chals[lvl.fold_chs[j]]);
                    nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
                }
                if li < levels.len() - 1 {
                    for od in &lvl.ood {
                        let b = chals[od.beta_ch];
                        let iq =
                            quad(vals_rec[od.intro_v], vals_rec[od.intro_v + 1], vals_rec[od.y_v]);
                        nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                        nt += b * vals_rec[od.y_v];
                    }
                    let b = chals[lvl.beta_ch];
                    let iq =
                        quad(vals_rec[lvl.intro_v], vals_rec[lvl.intro_v + 1], native_sums[li]);
                    nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                    nt += b * native_sums[li];
                } else {
                    nt += chals[lvl.beta_ch] * native_sums[li];
                }
            }
            assert_eq!(
                built.public[plen - zc_tail - 1],
                nt,
                "the spine's final t_r matches the native replay"
            );
        }

        // ---- prove / verify the leaf query-phase circuit ----
        // BLAKE3 for both hashes: the outer proof is the SWAP's inner, so
        // it must be recursable — the same two gotchas the leaf hit.
        let union_o = UnionInstance::new(&shape.registry, shape.counts.clone());
        let pcs_o = PcsParams {
            m: union_o.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union_o.commit_lanes(6),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs = blake3::build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let swap_r1cs = SwapTable::build_block_r1cs(nu);
        let swap_lc = swap_r1cs.csc_lincheck_circuit();
        let spread_ty = BitSpreadTable::new(spread_w);
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let b3_wit =
            blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu);
        let swap_wit =
            SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
        let spread_wit =
            spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
        let els: Vec<Vec<F128>> = leaf_slot
            .iter()
            .map(|(_, sl)| match &built.witnesses[shape.registry_slot(*sl)] {
                SlotWitness::Element(z) => z.clone(),
                other => panic!("leaf-eval slot produced {other:?}"),
            })
            .collect();
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (shape.registry_slot(slots.b3), b3_lc),
            (shape.registry_slot(slots.swap), swap_lc),
            (shape.registry_slot(slots.spread), spread_lc),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.into_iter().map(|(_, cc)| cc).collect();
        let ((oproof, ocommit), prove_t) = timed(REPS, || {
            let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
                (
                    shape.registry_slot(slots.b3),
                    UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
                ),
                (
                    shape.registry_slot(slots.swap),
                    UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
                ),
                (
                    shape.registry_slot(slots.spread),
                    UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
                ),
            ];
            bool_slots.sort_by_key(|(i, _)| *i);
            let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
                .iter()
                .zip(els.clone())
                .map(|((_, sl), z)| (shape.registry_slot(*sl), z))
                .collect();
            el_ord.sort_by_key(|(i, _)| *i);
            let el_inputs: Vec<UnionElementSlotInput> = el_ord
                .into_iter()
                .map(|(i, z)| live_element_input(z, shape.counts[i], nu))
                .collect();
            let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
            let (p, c, _) = prover::prove_fast_ligerito_union_circuit(
                &union_o,
                &shape.circuit,
                &built.public,
                &pcs_o,
                bool_slots.into_iter().map(|(_, x)| x).collect(),
                el_inputs,
                &mut ch,
            );
            (p, c)
        });
        let (_, verify_t) = timed(REPS, || {
            let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
            verifier::verify_ligerito_union_circuit(
                &union_o,
                &shape.circuit,
                &built.public,
                &lcs,
                &ocommit,
                &oproof,
                &pcs_o,
                &mut ch,
            )
            .expect("the leaf query-phase circuit verifies")
        });
        println!(
            "\nMVP-9 BOOLEAN LEAF (inner: blake3 workload m=22, rs×2 pd=0)\n  \
             nu {} | dense_m {} | mu {} | b3 rows {}\n\n  \
             medians of {REPS} runs, spread in brackets\n  \
             PER PROOF     prove {prove_t} ms\n  \
             verifier side {verify_t} ms | proof {:.1} KiB | {} threads\n",
            nu,
            union_o.dense_m(),
            shape.circuit.cells().mu(),
            b3_rows,
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
            threads,
        );
        let (b3_slot, swap_slot, spread_slot) = (
            shape.registry_slot(slots.b3),
            shape.registry_slot(slots.swap),
            shape.registry_slot(slots.spread),
        );
        LeafOuter {
            public: built.public.clone(),
            shape,
            proof: oproof,
            commitment: ocommit,
            pcs: pcs_o,
            b3_r1cs,
            swap_r1cs,
            spread_r1cs,
            b3_slot,
            swap_slot,
            spread_slot,
        }
    }
}

/// **THE SWAP, step 1 — mvp9's outer becomes the inner.** The leaf-outer
/// circuit proof (the first real recursion node, BLAKE3/BLAKE3 from the
/// shared builder) is natively verified under a RecordingChallenger and its
/// tape walked by the SAME machinery mvp10's assembly consumes:
/// parse_open_levels, the region label map, level_geometry (native capped
/// paths + enforced-sum replicas per level), and the R=2 + P multipoint
/// schedule replayed to the anchor's claimed v — pinned before any
/// assembly, the step-1 pattern every phase ran. What it establishes about
/// the REAL inner's shape: the element PIOP parses at multi-slot scale, the
/// packed-direct claims are the element (c, lc) pair plus every wiring
/// gather, the R=2 + P>0 schedule holds, and the committed lane count is
/// once more an arbitrary integer.
// ---------------------------------------------------------------------------
// The REAL child: the leaf outer's deferred verifier as a reusable region
// (the swap test's assembly, extracted so the 2→1 merge node can
// instantiate a real child-tape region per child — the emit_child_region
// precedent at leaf-outer scale)
// ---------------------------------------------------------------------------

/// One recorded REAL-child verification (the leaf outer as inner), parsed:
/// the tape pinned op-for-op, every region located, and every native
/// replica the emitter and checker consume — the swap test's step-1 walk as
/// a reusable unit. `new` runs the RECORDING verify itself, so every
/// instantiation re-asserts the whole map on that child's tape.
struct RealTape<'p> {
    lo: &'p LeafOuter,
    // the recorded tape
    vals_rec: Vec<F128>,
    chals: Vec<F128>,
    /// Which byte payloads stay PUBLIC under the witness/public split.
    pub_payloads: Vec<bool>,
    /// Per level, the absorbed cap's payload index ([`cap_payloads`]).
    cap_pays: Vec<usize>,
    // chain materials
    trace: flock_prover::r1cs_hashes::fs_chain::FsChainTrace,
    stream: flock_core::transcript_record::Stream,
    bytes: Vec<u8>,
    b3_rows: usize,
    spread_w: usize,
    // located regions
    gkr: GkrRec,
    piop_i: PiopRec,
    start_v_i: usize,
    gammas_i: Vec<PdRec>,
    w_rounds: Vec<RoundRec>,
    w_resid: Vec<RoundRec>,
    mp_i: MpRec,
    inner_pd_i: InnerPd,
    yr_v_i: usize,
    yr_len: usize,
    levels: Vec<OpenLevel>,
    lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    geo: Vec<Lvl>,
    native_sums: Vec<F128>,
    /// The grinding ops: (fin ordinal, payload ordinal, bits).
    pows: Vec<(usize, usize, u32)>,
    n_p: usize,
    n_gather: usize,
    // the boolean PIOP's round ordinals ((ch, fin) pairs) + surfaces
    zc_rounds_b: Vec<(usize, usize)>,
    outer_b: (usize, usize),
    #[allow(dead_code)] // The r_outer slice length — wall-4 shape data.
    outer_len: usize,
    bl_alpha: (usize, usize),
    /// The const-pin beta squeezes: (ch, fin) per pinned boolean type.
    betas_b: Vec<(usize, usize)>,
    /// The zerocheck finals' value ordinal (v_a at, v_b at +1).
    zc_finals_v: usize,
    /// Per pinned type, eq_prefix_sum(x_outer, n_t) — the count-derived
    /// beta term (advice, checker-recomputable from published wires).
    eps_n: Vec<F128>,
    /// (g_v, ch, fin) per boolean lc round — messages feed the in-circuit
    /// lincheck replay.
    lc_rounds_b: Vec<(usize, usize, usize)>,
    zskip_ch: usize,
    zskip_fin: usize,
    zp_v: usize,
    /// The rs regions: (s_hat_v ordinal, r_dprime fin, r_dprime ch), plus
    /// the two rs gammas' first (fin, ch) — the family-H re-exposure set.
    rs_recs: Vec<(usize, usize, usize)>,
    rs_gam_fin: usize,
    rs_gam_ch: usize,
    // native references + replicas
    mat_assert: flock_core::lincheck::MatrixAssertion,
    el_assert: flock_core::element_r1cs::union::ElementAssertion,
    sigma_native: flock_core::circuit::SigmaAssertion,
    /// Which pd claim carries z_eval (order varies per tape).
    z_ix: usize,
    el_g0: Vec<F128>,
    el_run_n: F128,
    a_sum_n: F128,
    b_sum_n: F128,
    native_rs_half: F128,
    native_vrs: F128,
    native_target: F128,
    native_running: F128,
    t_final_n: F128,
    anc_end_n: F128,
    mid_n: F128,
    live_n: F128,
    mu_i: usize,
    // anchor-expect geometry — statement constants of the real inner
    n_log_i: usize,
    k_cols_i: usize,
    m_mp2: usize,
    bounds_i: Vec<(u64, u64, u32)>,
    run_of: Vec<usize>,
    x_ab_n: Vec<F128>,
    x_c_n: Vec<F128>,
    groups_ix: Vec<Vec<usize>>,
}

impl<'p> RealTape<'p> {
    fn new(lo: &'p LeafOuter, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let union_i = UnionInstance::new(&lo.shape.registry, lo.shape.counts.clone());
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (lo.b3_slot, lo.b3_r1cs.csc_lincheck_circuit()),
            (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
            (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.into_iter().map(|(_, cc)| cc).collect();
        let mut rec = RecordingChallenger::new(FsChallenger::with_hash(domain, HashKind::Blake3));
        let claims = verifier::verify_ligerito_union_circuit(
            &union_i,
            &lo.shape.circuit,
            &lo.public,
            &lcs,
            &lo.commitment,
            &lo.proof,
            &lo.pcs,
            &mut rec,
        )
        .expect("the leaf outer verifies as the inner");
        assert!(claims.boolean.is_some(), "boolean claims from the real inner");
        assert!(claims.element.is_some(), "element claims from the real inner");
        // The DEFERRED verify of the same proof: the independent reference
        // for the three assertion families — the method-note discipline,
        // verifier-exported over formulas-written-twice.
        let (mat_assert, el_assert, sigma_native) = {
            let mut ch = FsChallenger::with_hash(domain, HashKind::Blake3);
            let (_, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
                &union_i,
                &lo.shape.circuit,
                &lo.public,
                &lcs,
                &lo.commitment,
                &lo.proof,
                &lo.pcs,
                &mut ch,
            )
            .expect("the deferred verify accepts the leaf outer");
            (
                work.boolean.expect("a boolean PIOP ran"),
                work.element.expect("an element PIOP ran"),
                sigma,
            )
        };
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops = t_shape.ops();
        let mut pub_payloads = bytes_payload_mask(ops);
        let vc_at = |end: usize| -> (usize, usize) {
            let (mut v, mut c) = (0usize, 0usize);
            for op in &ops[..end] {
                match op {
                    Op::SqueezeScalar => c += 1,
                    Op::SqueezeSlice(n) => c += n,
                    Op::ObserveScalar => v += 1,
                    Op::ObserveSlice(n) => v += n,
                    _ => {}
                }
            }
            (v, c)
        };
        let fin_at = |end: usize| ops[..end].iter().filter(|o| o.finalizes()).count();

        // The region order, by label — identical to the minimal mixed inner's.
        let find = |label: &[u8]| -> Vec<usize> {
            ops.iter()
                .enumerate()
                .filter_map(|(i, op)| match op {
                    Op::Label(l) if l.as_slice() == label => Some(i),
                    _ => None,
                })
                .collect()
        };
        let zc_l = find(b"flock-zerocheck-v0");
        let lc_l = find(b"flock-lincheck-v0");
        let elzc_l = find(b"flock-element-union-zc-v0");
        let el_l = find(b"flock-element-union-lc-v0");
        let gkr_l = find(b"flock-product-gkr-batched-v0");
        let mo_l = find(b"flock-merged-open-v0");
        let rs_l = find(b"flock-ring-switch-v0");
        let mp_l = find(b"flock-multipoint-twisted-v1");
        let fa_l = find(b"flock-frobenius-assist-v0");
        assert_eq!(
            (zc_l.len(), lc_l.len(), elzc_l.len(), el_l.len(), gkr_l.len()),
            (1, 1, 1, 1, 1),
            "one region each"
        );
        assert_eq!((mo_l.len(), rs_l.len(), mp_l.len(), fa_l.len()), (1, 2, 1, 1));
        assert!(zc_l[0] < lc_l[0] && lc_l[0] < elzc_l[0] && elzc_l[0] < el_l[0]);
        assert!(el_l[0] < gkr_l[0] && gkr_l[0] < mo_l[0]);
        assert!(mo_l[0] < rs_l[0] && rs_l[1] < mp_l[0] && mp_l[0] < fa_l[0]);

        // parse_open_levels + level_geometry — the assembly's own walkers,
        // unchanged, on the real-inner tape.
        let lig = &lo.proof.pcs_open.inner.ligerito;
        let r = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v_i, piop_i, gammas_i, w_rounds, mp_i, inner_pd_i, yr_v_i, levels) =
            parse_open_levels(ops, 32 * lig.initial_cap.len(), r);
        assert_eq!(levels.len(), r + 1);
        let piop_i = piop_i.expect("the real inner HAS an element PIOP");
        assert!(!piop_i.zc_rounds.is_empty() && !piop_i.lc_rounds.is_empty());
        let n_gather = lo.proof.wiring.gather.len();
        assert_eq!(
            gammas_i.len(),
            2 + n_gather,
            "pd claims = the element (c, lc) pair + the outer's gathers"
        );
        assert_eq!(w_rounds.len(), lo.pcs.m - 7, "W spans the dense domain");
        let (geo, native_sums) = level_geometry(&levels, &lvl_src, &chals, HashKind::Blake3);
        assert!(geo[0].row_words <= geo[0].lanes, "committed width fits the fold");

        // The R=2 + P schedule replays to the anchor's claimed v.
        let n_p = lo.proof.pcs_open.frobenius.group_values.len();
        assert!(n_p > 0, "the mixed inner groups its pd claims");
        assert_eq!(
            mp_i.val_vs.len(),
            256 + n_p,
            "T0 spans the RS dual values then the P group values"
        );
        let gamma_mp = chals[mp_i.gamma_ch];
        let mut pw = F128::ONE;
        let mut t0 = F128::ZERO;
        for &vi in &mp_i.val_vs {
            t0 += pw * vals_rec[vi];
            pw *= gamma_mp;
        }
        let mut tm = t0;
        for rr in &mp_i.rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let rch = chals[rr.ch];
            let g0 = tm + g1;
            tm = g0 + (g1 + g0 + gi) * rch + gi * rch * rch;
        }
        assert_eq!(
            tm,
            vals_rec[mp_i.anchor_v],
            "T0 folds to the anchor's claimed v"
        );

        // ---- the wiring GKR walk (the mvp10 walker, real-inner layers) ----
        // Records every ordinal the transcription wires against and replays
        // the whole layer recursion natively in lockstep, input checks
        // included — the rhs consuming the DEFERRED s_sigma from the proof.
        let gkr_rec = {
            let gkr = &lo.proof.wiring.gkr;
            let mut i = gkr_l[0] + 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr alpha");
            let (_, c_alpha) = vc_at(i);
            let alpha_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr beta");
            let beta_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::ObserveScalar), "top lhs");
            let (tv, _) = vc_at(i);
            assert_eq!(vals_rec[tv], gkr.top_lhs, "top_lhs on the stream");
            assert_eq!(vals_rec[tv + 1], gkr.top_rhs, "top_rhs on the stream");
            assert_eq!(gkr.top_lhs, gkr.top_rhs, "the grand products agree");
            i += 2;
            let (mut claim_l, mut claim_r) = (gkr.top_lhs, gkr.top_rhs);
            let mut r_pt: Vec<F128> = Vec::new();
            let mut lrecs: Vec<GkrLayerRec> = Vec::new();
            for (k, layer) in gkr.layers.iter().enumerate() {
                assert_eq!(layer.rounds.len(), k, "layer {k} has k rounds");
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} lambda");
                let (_, lc2) = vc_at(i);
                let lambda = chals[lc2];
                let lam_fin = fin_at(i);
                i += 1;
                let mut c_run = claim_l + lambda * claim_r;
                let mut r_prime = Vec::with_capacity(k + 1);
                let mut rrecs: Vec<(usize, usize)> = Vec::new();
                let mut g0s: Vec<F128> = Vec::new();
                for (t2, &(g1, gi)) in layer.rounds.iter().enumerate() {
                    assert!(matches!(ops[i], Op::ObserveScalar), "round obs g1");
                    let (gv, _) = vc_at(i);
                    assert_eq!(vals_rec[gv], g1, "layer {k} round {t2} g1");
                    assert_eq!(vals_rec[gv + 1], gi, "layer {k} round {t2} g_inf");
                    assert!(matches!(ops[i + 2], Op::SqueezeScalar), "round rho");
                    let (_, rc2) = vc_at(i + 2);
                    let rho = chals[rc2];
                    rrecs.push((gv, fin_at(i + 2)));
                    i += 3;
                    let r_eq = r_pt[t2];
                    let g0 = (c_run + r_eq * g1) * (F128::ONE + r_eq).inv();
                    g0s.push(g0);
                    c_run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
                    r_prime.push(rho);
                }
                let (vv, _) = vc_at(i);
                for (j, want) in [layer.vl0, layer.vl1, layer.vr0, layer.vr1]
                    .into_iter()
                    .enumerate()
                {
                    assert!(matches!(ops[i], Op::ObserveScalar), "layer value obs");
                    assert_eq!(vals_rec[vv + j], want, "layer {k} value {j}");
                    i += 1;
                }
                assert_eq!(
                    c_run,
                    layer.vl0 * layer.vl1 + lambda * (layer.vr0 * layer.vr1),
                    "layer {k} closes"
                );
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} c_k");
                let (_, cc2) = vc_at(i);
                let c_k = chals[cc2];
                let ck_fin = fin_at(i);
                i += 1;
                claim_l = (F128::ONE + c_k) * layer.vl0 + c_k * layer.vl1;
                claim_r = (F128::ONE + c_k) * layer.vr0 + c_k * layer.vr1;
                r_prime.push(c_k);
                r_pt = r_prime;
                lrecs.push(GkrLayerRec {
                    lam_fin,
                    rounds: rrecs,
                    g0s,
                    v_v: vv,
                    ck_fin,
                });
            }
            let mu_i = lo.shape.circuit.cells().mu();
            assert_eq!(r_pt.len(), mu_i, "the GKR point spans the inner cell space");
            let alpha2 = chals[c_alpha];
            let beta2 = chals[c_alpha + 1];
            let basis = flock_core::product_gkr::s_id_basis(mu_i);
            // The LIVE-IDENTITY padding: leaves are w + α·(live⊙s_id) +
            // (β+1)·live + 1 (dead cells = 1), so the input checks carry
            // the masked closed forms.
            let mask_w = lo.shape.circuit.live_mask();
            let tail_w2 = (beta2 + F128::ONE) * mask_w.live_eval(&r_pt) + F128::ONE;
            assert_eq!(
                claim_l,
                gkr.f_eval + alpha2 * mask_w.masked_id_eval(&basis, &r_pt) + tail_w2,
                "lhs input check replays (masked)"
            );
            assert_eq!(
                claim_r,
                gkr.g_eval + alpha2 * gkr.s_sigma_eval + tail_w2,
                "rhs input check replays with the DEFERRED (masked) sigma value"
            );
            let (fv, _) = vc_at(i);
            assert!(matches!(ops[i], Op::ObserveScalar), "f_eval obs");
            assert_eq!(vals_rec[fv], gkr.f_eval, "f_eval on the stream");
            assert_eq!(vals_rec[fv + 1], gkr.g_eval, "g_eval on the stream");
            assert_eq!(vals_rec[fv + 2], gkr.s_sigma_eval, "s_sigma on the stream");
            GkrRec {
                alpha_fin,
                beta_fin,
                top_v: tv,
                layers: lrecs,
                fgs_v: fv,
                r_pt,
            }
        };
        // ROUND 2: the H(publics) region's rows — a chunk chain per 1 KiB
        // leaf of the child's public segment plus the left-fold parents.
        let h_rows = lo.public.len().div_ceil(4) + 2 * lo.public.len().div_ceil(64);

        // ---- the chain materials ----
        let stream = t_shape.stream_words(domain);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let trace = {
            let mut chain = FsChain::new();
            let mut at = 0usize;
            let fin_ops: Vec<_> = ops.iter().filter(|o| o.finalizes()).collect();
            assert_eq!(
                stream.finalize_after.len(),
                fin_ops.len(),
                "finalize alignment"
            );
            for (k, &upto) in stream.finalize_after.iter().enumerate() {
                chain.absorb(&bytes[at * 16..upto * 16]);
                at = upto;
                chain.finalize(fin_ops[k].squeezed_bytes());
            }
            chain.absorb(&bytes[at * 16..]);
            chain.finish()
        };
        let b3_rows = trace.rows.len()
            + h_rows
            + geo
                .iter()
                .map(|g| (g.row_words.div_ceil(4) + g.depth) * g.q + (1usize << g.c) - 1)
                .sum::<usize>();
        let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);
        // Recursive caps are PROOF BODY — the in-circuit cap trees bind them
        // (chain + root connects, nothing checker-read); only the L0 cap —
        // the commitment — stays a statement public.
        let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
        for &p in &cap_pays[1..] {
            pub_payloads[p] = false;
        }

        // The PoW grinding ops, located (the mvp7 machinery).
        let pows: Vec<(usize, usize, u32)> = {
            let mut out = Vec::new();
            let (mut fin, mut pay) = (0usize, 0usize);
            for op in ops {
                if let Op::Pow { bits } = op {
                    out.push((fin, pay, *bits));
                }
                if op.finalizes() {
                    fin += 1;
                }
                match op {
                    Op::ObserveBytes(_) | Op::Pow { .. } => pay += 1,
                    _ => {}
                }
            }
            out
        };
        assert!(!pows.is_empty(), "the Fast profile grinds");

        // ---- the rs×2 regions + the two-halves target, natively ----
        let (rs_recs2, rs_gam_ch2, rs_gam_fin2) = {
            let mut i2 = rs_l[0];
            // (s_hat_v ordinal, r_dprime fin, r_dprime ch) per region.
            let mut recs: Vec<(usize, usize, usize)> = Vec::new();
            for k in 0..2 {
                assert!(
                    matches!(&ops[i2], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0"),
                    "rs region {k}"
                );
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(128)), "s_hat_v slice");
                let (sv, _) = vc_at(i2);
                assert_eq!(
                    &vals_rec[sv..sv + 128],
                    &lo.proof.pcs_open.ring_switches[k].s_hat_v[..],
                    "s_hat_v {k} on the stream"
                );
                i2 += 1;
                assert!(matches!(ops[i2], Op::SqueezeSlice(7)), "r_dprime");
                recs.push((sv, fin_at(i2), vc_at(i2).1));
                i2 += 1;
            }
            let gch = vc_at(i2).1;
            let gfin = fin_at(i2);
            for _ in 0..2 {
                assert!(matches!(ops[i2], Op::SqueezeScalar), "rs gamma");
                i2 += 1;
            }
            (recs, gch, gfin)
        };
        // The two-halves target and V, split into their family-H (RS) and
        // in-circuit-computable (packed-direct / group) parts — the round-0
        // production posture: the pd/group halves become MAC chains in the
        // circuit, the RS halves stay advice checked over RE-EXPOSED words.
        let (native_rs_half, native_target, native_vrs, native_running) = {
            use flock_core::pcs::ring_switch as rsw;
            use flock_core::zerocheck::univariate_skip::build_eq;
            let gs: Vec<F128> = (0..2).map(|k| chals[rs_gam_ch2 + k]).collect();
            let mut rs_half = F128::ZERO;
            let mut coeffs: Vec<Vec<F128>> = Vec::new();
            for (k, &(sv, _, rc)) in rs_recs2.iter().enumerate() {
                let shv = &vals_rec[sv..sv + 128];
                let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
                let eq = build_eq(&rdp);
                rs_half += gs[k] * rsw::inner_product(&rsw::tensor_algebra_transpose(shv), &eq);
                let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
                coeffs
                    .push(rsw::linearized_coefficients(&rsw::build_fold_byte_table(&scaled)));
            }
            let mut target = rs_half;
            for pd in &gammas_i {
                target += chals[pd.ch] * vals_rec[pd.val_v];
            }
            let mut running = target;
            for rr in &w_rounds {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let rc = chals[rr.ch];
                let g0 = running + g1;
                running = g0 + (g1 + g0 + gi) * rc + gi * rc * rc;
            }
            let fro = &lo.proof.pcs_open.frobenius;
            let mut vrs = F128::ZERO;
            for (k, cs) in coeffs.iter().enumerate() {
                for (j, &cj) in cs.iter().enumerate() {
                    if cj.is_zero() {
                        continue;
                    }
                    let mut x = fro.values[k][j];
                    for _ in 0..j {
                        x = x * x;
                    }
                    vrs += cj * x;
                }
            }
            let mut big_v = vrs;
            for &v in &fro.group_values {
                big_v += v;
            }
            assert_eq!(
                running,
                vals_rec[inner_pd_i.q_v] * big_v,
                "the R=2 + P merged boundary replays at real-inner scale"
            );
            (rs_half, target, vrs, running)
        };

        // ---- the spine's native quad replay ----
        let t_final_n = {
            let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
            let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
            let mut nt = chals[inner_pd_i.ch] * vals_rec[inner_pd_i.q_v];
            let mut nq = quad(vals_rec[start_v_i], vals_rec[start_v_i + 1], nt);
            for (li, lvl) in levels.iter().enumerate() {
                for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                    nt = evalq(nq, chals[lvl.fold_chs[j]]);
                    nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
                }
                if li < r {
                    for od in &lvl.ood {
                        let b = chals[od.beta_ch];
                        let iq = quad(
                            vals_rec[od.intro_v],
                            vals_rec[od.intro_v + 1],
                            vals_rec[od.y_v],
                        );
                        nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                        nt += b * vals_rec[od.y_v];
                    }
                    let b = chals[lvl.beta_ch];
                    let iq = quad(
                        vals_rec[lvl.intro_v],
                        vals_rec[lvl.intro_v + 1],
                        native_sums[li],
                    );
                    nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                    nt += b * native_sums[li];
                } else {
                    nt += chals[lvl.beta_ch] * native_sums[li];
                }
            }
            nt
        };

        // ---- the residual pairing's rotation (lane-major, 56/64) ----
        let yr_len = lo.proof.pcs_open.inner.ligerito.final_proof.yr.len();
        let lane_major = geo[0].row_words < geo[0].lanes;
        assert!(lane_major, "the real inner commits integer lanes");
        let w_resid: Vec<RoundRec> = {
            let k_rot = w_rounds.len() - levels[0].fold_fins.len();
            let mut v = w_rounds[k_rot..].to_vec();
            v.extend_from_slice(&w_rounds[..k_rot]);
            v
        };

        // ---- the element PIOP's natives: the GENERAL strip + g0 chain ----
        assert_eq!(
            piop_i.zc_rounds.len(),
            piop_i.tau_len,
            "one element zc round per tau coordinate"
        );
        assert_eq!(
            el_assert.alpha,
            chals[piop_i.alpha_ch],
            "the located alpha is the assertion's"
        );
        let (a_sum_n, b_sum_n) = {
            let slots_el = flock_core::element_r1cs::union::region_slots(&union_i);
            let nu_i = union_i.n_log();
            let mut a_sum = F128::ZERO;
            let mut b_sum = F128::ZERO;
            for s in &slots_el {
                let kappa = s.ty.kappa();
                let eq_con = flock_core::zerocheck::univariate_skip::build_eq(
                    &el_assert.r_con[..kappa],
                );
                let prefix = s.layout.region_prefix(nu_i);
                let mut w = F128::ONE;
                for (j, &x) in el_assert.r_con[kappa..].iter().enumerate() {
                    w *= if (prefix >> j) & 1 == 1 { x } else { F128::ONE + x };
                }
                let dot = |c: &[F128]| -> F128 {
                    eq_con
                        .iter()
                        .zip(c)
                        .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
                };
                a_sum += w * dot(s.ty.a_const());
                b_sum += w * dot(s.ty.b_const());
            }
            (a_sum, b_sum)
        };
        let mut el_g0: Vec<F128> = Vec::new();
        let el_run_n = {
            let mut run = F128::ZERO;
            for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let t2 = chals[piop_i.tau_ch + k];
                let rho = chals[rr.ch];
                let g0 = (run + t2 * g1) * (F128::ONE + t2).inv();
                el_g0.push(g0);
                run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
            }
            run
        };
        // The element c claim's position among the pd claims varies with
        // the tape — identify it by the assertion's own value.
        let z_ix = gammas_i
            .iter()
            .position(|pd| vals_rec[pd.val_v] == el_assert.z_eval)
            .expect("z_eval is one of the absorbed pd values");

        // ---- the anchor's native endpoint ----
        let anc_end_n = {
            let mut t2 = vals_rec[mp_i.anchor_v];
            for rr in &mp_i.anchor_rounds {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let r3 = chals[rr.ch];
                let g0 = t2 + g1;
                t2 = g0 + (g1 + g0 + gi) * r3 + gi * r3 * r3;
            }
            t2
        };

        // ---- the GKR input-check advice (masked M̂ and livê) ----
        let mu_i = lo.shape.circuit.cells().mu();
        let (mid_n, live_n) = {
            let basis_i = flock_core::product_gkr::s_id_basis(mu_i);
            let mask_i = lo.shape.circuit.live_mask();
            (
                mask_i.masked_id_eval(&basis_i, &gkr_rec.r_pt),
                mask_i.live_eval(&gkr_rec.r_pt),
            )
        };

        // ---- the anchor-expect geometry + boolean locate + replica ----
        let m_mp2 = mp_i.rounds.len();
        assert_eq!(
            mp_i.anchor_rounds.len(),
            2 * (m_mp2 + 1),
            "sigma spans the anchor layers"
        );
        assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
        let n_log_i = union_i.n_log();
        let params_i = flock_core::pcs::jagged::JaggedParams::from_heights(
            &union_i.jagged_heights(),
            n_log_i,
            m_mp2,
        );
        let k_cols_i = params_i.k;
        let bounds_i = flock_core::pcs::jagged::assist_boundaries(&params_i);
        let n_runs = bounds_i.len();
        let run_y0: Vec<usize> = bounds_i
            .iter()
            .scan(0usize, |y, &(_, _, len)| {
                let s = *y;
                *y += len as usize;
                Some(s)
            })
            .collect();
        let comp_ix = (0..n_runs)
            .max_by_key(|&r3| bounds_i[r3].2)
            .expect("at least one run");
        let run_of: Vec<usize> = {
            let mut v = Vec::with_capacity(1usize << k_cols_i);
            for (r3, &(_, _, len)) in bounds_i.iter().enumerate() {
                v.extend(std::iter::repeat_n(r3, len as usize));
            }
            assert_eq!(v.len(), 1usize << k_cols_i, "runs partition the columns");
            v
        };
        // The boolean PIOP's round ordinals, located with fins — plus the
        // MatrixAssertion surfaces the 2→1 merge connects to (z_skip's
        // squeeze, z_partial's slice).
        let (
            zc_rounds_b,
            (zskip_ch, zskip_fin),
            (outer_ch_b, outer_fin_b, outer_len),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_rounds_b,
            zp_v,
        ) = {
            let mut i2 = zc_l[0] + 1;
            assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_skip slice");
            i2 += 1;
            let outer_len = match ops[i2] {
                Op::SqueezeSlice(n) => n,
                ref o => panic!("r_outer slice, got {o:?}"),
            };
            let outer = (vc_at(i2).1, fin_at(i2), outer_len);
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_ab");
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_c");
            i2 += 1;
            assert!(matches!(ops[i2], Op::SqueezeScalar), "z_skip");
            let zskip = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            let mut zc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar)
                && matches!(ops[i2 + 1], Op::ObserveScalar)
                && matches!(ops[i2 + 2], Op::SqueezeScalar)
            {
                zc_r.push((vc_at(i2 + 2).1, fin_at(i2 + 2)));
                i2 += 3;
            }
            // The zerocheck finals (v_a, v_b, ...) — the lincheck entry's
            // absorbed operands.
            let (zcf, _) = vc_at(i2);
            while matches!(ops[i2], Op::ObserveScalar) {
                i2 += 1;
            }
            assert_eq!(i2, lc_l[0], "the zerocheck runs straight into the lincheck");
            i2 += 1;
            assert!(matches!(ops[i2], Op::SqueezeScalar), "lc alpha");
            let lc_alpha = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            // The const-pin beta squeezes, one per pinned boolean type.
            let mut betas: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::SqueezeScalar) {
                betas.push((vc_at(i2).1, fin_at(i2)));
                i2 += 1;
            }
            // (g_v, ch, fin) per lc round — the message ordinals feed the
            // round-0 in-circuit lincheck replay.
            let mut lc_r: Vec<(usize, usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar)
                && matches!(ops[i2 + 1], Op::ObserveScalar)
                && matches!(ops[i2 + 2], Op::SqueezeScalar)
            {
                lc_r.push((vc_at(i2).0, vc_at(i2 + 2).1, fin_at(i2 + 2)));
                i2 += 3;
            }
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "z_partial slice");
            let (zp, _) = vc_at(i2);
            (zc_r, zskip, outer, lc_alpha, betas, zcf, lc_r, zp)
        };
        // The surface→ordinal mapping asserts (the batch-major packing the
        // minimal child pinned): x_inner_rest[0] = mlv round 0, x_outer =
        // rounds 1..1+ν, x_inner_rest[1..] = the rest; rr = the lc rounds
        // REVERSED; z_skip and z_partial are the located ops.
        {
            let inner_b = mat_assert.x_inner_rest.len();
            assert_eq!(
                zc_rounds_b.len(),
                inner_b + n_log_i,
                "zc mlv rounds = x_inner_rest + x_outer"
            );
            for (j, &x) in mat_assert.x_inner_rest.iter().enumerate() {
                let m = if j == 0 { 0 } else { n_log_i + j };
                assert_eq!(
                    chals[zc_rounds_b[m].0],
                    x,
                    "x_inner_rest {j} is located zc round {m}"
                );
            }
            assert_eq!(lc_rounds_b.len(), mat_assert.rr.len(), "lc round count");
            for (j, &x) in mat_assert.rr.iter().enumerate() {
                assert_eq!(
                    chals[lc_rounds_b[lc_rounds_b.len() - 1 - j].1],
                    x,
                    "rr {j} is the located lc round, reversed"
                );
            }
            assert_eq!(chals[zskip_ch], mat_assert.z_skip, "z_skip located");
            assert_eq!(
                &vals_rec[zp_v..zp_v + 64],
                &mat_assert.z_partial[..],
                "z_partial on the stream"
            );
            assert_eq!(
                mat_assert.alpha,
                chals[bl_alpha.0],
                "the located boolean lc alpha is the matrix assertion's"
            );
            // The element assertion's points: r_con = zc.r[ν..] (round
            // order), r_col = the lc bind order reversed.
            assert_eq!(
                piop_i.zc_rounds.len(),
                n_log_i + el_assert.r_con.len(),
                "element zc rounds = rows + r_con"
            );
            for (j, &x) in el_assert.r_con.iter().enumerate() {
                assert_eq!(
                    chals[piop_i.zc_rounds[n_log_i + j].ch],
                    x,
                    "el r_con {j} is a located element zc round"
                );
            }
            assert_eq!(
                piop_i.lc_rounds.len(),
                el_assert.r_col.len(),
                "element lc round count"
            );
            for (j, &x) in el_assert.r_col.iter().enumerate() {
                assert_eq!(
                    chals[piop_i.lc_rounds[piop_i.lc_rounds.len() - 1 - j].ch],
                    x,
                    "el r_col {j} is the located element lc round, reversed"
                );
            }
        }
        assert!(lc_rounds_b.len() <= 1 + k_cols_i, "lc rounds fit the col bits");
        // The boolean lincheck ENTRY, natively: target0 = α·v_a + v_b +
        // Σ β_t·eq_prefix_sum(x_outer, n_t), with x_outer the zc mlv rows
        // (batch-major: rounds 1..1+ν) — replayed through the located lc
        // rounds it must end at the deferred MatrixAssertion's own target
        // (the method-note discipline; this pre-assert is what licenses the
        // in-circuit replay's wire map).
        let (eps_n, entry_n) = {
            let x_outer_n: Vec<F128> = (0..n_log_i)
                .map(|j| chals[zc_rounds_b[1 + j].0])
                .collect();
            let pinned: Vec<usize> = mat_assert
                .betas
                .iter()
                .enumerate()
                .filter_map(|(t, b)| b.map(|_| t))
                .collect();
            assert_eq!(pinned.len(), betas_b.len(), "one squeeze per const pin");
            let mut eps = Vec::with_capacity(betas_b.len());
            let mut entry = mat_assert.alpha * vals_rec[zc_finals_v]
                + vals_rec[zc_finals_v + 1];
            for (k, &t) in pinned.iter().enumerate() {
                assert_eq!(
                    chals[betas_b[k].0],
                    mat_assert.betas[t].expect("pinned"),
                    "beta {k} is the located squeeze"
                );
                let e = flock_core::product_gkr::LiveMask::eq_prefix_sum(
                    &x_outer_n,
                    union_i.counts()[t],
                );
                entry += chals[betas_b[k].0] * e;
                eps.push(e);
            }
            let mut run = entry;
            for &(g_v, ch, _) in &lc_rounds_b {
                let (e1, einf) = (vals_rec[g_v], vals_rec[g_v + 1]);
                let r = chals[ch];
                let q0 = run + e1;
                run = einf * r * r + (q0 + e1 + einf) * r + q0;
            }
            assert_eq!(
                run, mat_assert.target,
                "the boolean lc entry replays to the assertion's target"
            );
            (eps, entry)
        };
        let _ = entry_n;
        let nat_b = claims.boolean.as_ref().expect("boolean claims");
        let x_ab_n: Vec<F128> = {
            let p = &nat_b.ab.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        let x_c_n: Vec<F128> = {
            let p = &nat_b.c.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        assert_eq!(x_ab_n.len(), 1 + n_log_i + k_cols_i, "ab point split");
        assert_eq!(x_c_n.len(), 1 + n_log_i + k_cols_i, "c point split");
        let pd_pts_n: Vec<Vec<F128>> = gammas_i
            .iter()
            .map(|pd| vals_rec[pd.pt_v..pd.pt_v + pd.pt_len].to_vec())
            .collect();
        for pd in &gammas_i {
            assert_eq!(pd.pt_len, n_log_i + k_cols_i, "pd point split");
        }
        let mut groups_ix: Vec<Vec<usize>> = Vec::new();
        for (i2, pt) in pd_pts_n.iter().enumerate() {
            match groups_ix
                .iter_mut()
                .find(|g2| pd_pts_n[g2[0]][..n_log_i] == pt[..n_log_i])
            {
                Some(g2) => g2.push(i2),
                None => groups_ix.push(vec![i2]),
            }
        }
        assert_eq!(groups_ix.len(), n_p, "P scalar groups by shared row");

        // Native replica of the WHOLE anchor expect — validated against
        // the accepted proof before any gate exists.
        {
            let gamma_n = chals[mp_i.gamma_ch];
            let mut gpow_n = vec![F128::ONE];
            for j in 1..257 + n_p {
                gpow_n.push(gpow_n[j - 1] * gamma_n);
            }
            let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
            let point_n: Vec<F128> = mp_i.rounds.iter().map(|rr| chals[rr.ch]).collect();
            let sig_n: Vec<F128> = mp_i.anchor_rounds.iter().map(|rr| chals[rr.ch]).collect();
            let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
            let g_at_n = {
                let mut rinv = rho_mrg_n.clone();
                let mut acc = F128::ZERO;
                for (j, &gp) in gpow_n.iter().enumerate().take(128) {
                    if j > 0 {
                        for x in rinv.iter_mut() {
                            *x = frob_inv_native(*x);
                        }
                    }
                    let mut prod = gp;
                    for (t3, &x) in point_n.iter().enumerate() {
                        prod *= F128::ONE + rinv[t3] + x;
                    }
                    acc += prod;
                }
                acc
            };
            let e_at_n = rho_mrg_n
                .iter()
                .zip(&point_n)
                .fold(F128::ONE, |a, (&r3, &x)| a * (F128::ONE + r3 + x));
            let eqc_n: Vec<F128> = bounds_i
                .iter()
                .map(|&(t_c, t_next, _)| {
                    let mut p = F128::ONE;
                    for l in 0..=m_mp2 {
                        p *= F128::ONE + sig_n[2 * l] + bit((t_c >> l) & 1 == 1);
                        p *= F128::ONE + sig_n[2 * l + 1] + bit((t_next >> l) & 1 == 1);
                    }
                    p
                })
                .collect();
            let sparse_t = flock_core::pcs::jagged::assist_sparse_transitions();
            let dp_native = |z_row: &[F128]| -> F128 {
                let mut gdp = [F128::ZERO; 4];
                gdp[flock_core::pcs::jagged::STATE_SUCCESS] = F128::ONE;
                for layer in (0..=m_mp2).rev() {
                    let za = if layer < n_log_i { z_row[layer] } else { F128::ZERO };
                    let rb = if layer < m_mp2 { point_n[layer] } else { F128::ZERO };
                    let eq4 = flock_core::lincheck::build_eq_table(&[za, rb]);
                    let (rc, rd) = (sig_n[2 * layer], sig_n[2 * layer + 1]);
                    let e = [
                        (F128::ONE + rc) * (F128::ONE + rd),
                        rc * (F128::ONE + rd),
                        (F128::ONE + rc) * rd,
                        rc * rd,
                    ];
                    let mut prev = [F128::ZERO; 4];
                    for (cd, &ecd) in e.iter().enumerate() {
                        for (s2, slot2) in prev.iter_mut().enumerate() {
                            let (i0, o0) = sparse_t[cd][s2][0];
                            let (i1, o1) = sparse_t[cd][s2][1];
                            *slot2 += ecd * (eq4[i0] * gdp[o0] + eq4[i1] * gdp[o1]);
                        }
                    }
                    gdp = prev;
                }
                gdp[flock_core::pcs::jagged::STATE_INITIAL]
            };
            let run_weights_n = |z_col: &[F128]| -> Vec<F128> {
                let mut w_at = vec![F128::ZERO; n_runs];
                let mut tot = F128::ONE;
                for (r3, &(_, _, len)) in bounds_i.iter().enumerate() {
                    if r3 == comp_ix {
                        continue;
                    }
                    let mut w = F128::ZERO;
                    for y in run_y0[r3]..run_y0[r3] + len as usize {
                        let mut s = F128::ONE;
                        for (jj, &zc2) in z_col.iter().enumerate() {
                            s *= F128::ONE + zc2 + bit((y >> jj) & 1 == 1);
                        }
                        w += s;
                    }
                    w_at[r3] = w;
                    tot += w;
                }
                w_at[comp_ix] = tot;
                w_at
            };
            let expect_n = {
                let mut acc = F128::ZERO;
                for (si, xs) in [&x_ab_n, &x_c_n].iter().enumerate() {
                    let z_row = &xs[1..1 + n_log_i];
                    let run_n = run_weights_n(&xs[1 + n_log_i..]);
                    let w_n = run_n
                        .iter()
                        .zip(&eqc_n)
                        .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                    let coeff = if si == 0 { g_at_n } else { gpow_n[128] * g_at_n };
                    acc += coeff * (w_n * dp_native(z_row));
                }
                for (g_ix, members) in groups_ix.iter().enumerate() {
                    let mut run_n = vec![F128::ZERO; n_runs];
                    for &i2 in members {
                        let pd = &gammas_i[i2];
                        let gpd = chals[pd.ch];
                        let w_at = run_weights_n(&pd_pts_n[i2][n_log_i..]);
                        for r3 in 0..n_runs {
                            run_n[r3] += gpd * w_at[r3];
                        }
                    }
                    let w_n = run_n
                        .iter()
                        .zip(&eqc_n)
                        .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                    let dp = dp_native(&pd_pts_n[members[0]][..n_log_i]);
                    acc += gpow_n[256 + g_ix] * e_at_n * (w_n * dp);
                }
                acc
            };
            assert_eq!(
                expect_n, anc_end_n,
                "the anchor expect replays natively at real-inner scale"
            );
        }

        RealTape {
            lo,
            vals_rec,
            chals,
            pub_payloads,
            cap_pays,
            trace,
            stream,
            bytes,
            b3_rows,
            spread_w,
            gkr: gkr_rec,
            piop_i,
            start_v_i,
            gammas_i,
            w_rounds,
            w_resid,
            mp_i,
            inner_pd_i,
            yr_v_i,
            yr_len,
            levels,
            lvl_src,
            geo,
            native_sums,
            pows,
            n_p,
            n_gather,
            zc_rounds_b,
            outer_b: (outer_ch_b, outer_fin_b),
            outer_len,
            bl_alpha,
            betas_b,
            zc_finals_v,
            eps_n,
            lc_rounds_b,
            zskip_ch,
            zskip_fin,
            zp_v,
            rs_recs: rs_recs2,
            rs_gam_fin: rs_gam_fin2,
            rs_gam_ch: rs_gam_ch2,
            mat_assert,
            el_assert,
            sigma_native,
            z_ix,
            el_g0,
            el_run_n,
            a_sum_n,
            b_sum_n,
            native_rs_half,
            native_vrs,
            native_target,
            native_running,
            t_final_n,
            anc_end_n,
            mid_n,
            live_n,
            mu_i,
            n_log_i,
            k_cols_i,
            m_mp2,
            bounds_i,
            run_of,
            x_ab_n,
            x_c_n,
            groups_ix,
        }
    }
}

/// What one emitted REAL child region hands back: where its public block
/// starts, the walk counts, and the assertion-emission wires the 2→1 merge
/// node CONNECTS the fold region's claim words to — all three families.
struct RealRegion {
    pub_base: usize,
    n_query_pub: usize,
    n_tail: usize,
    n_mat_pub: usize,
    /// The family-H re-exposure block length (the tail past z_skip).
    n_fam_pub: usize,
    n_ela_pub: usize,
    /// sigma: the deferred s_sigma stream word + the GKR squeeze point.
    #[allow(dead_code)]
    sig_w: Wire,
    #[allow(dead_code)]
    pt_w: Vec<Wire>,
    /// element: every zc/lc round rho (round order) and the per-slot eval
    /// advice pairs (bound publics — connectable, unlike the minimal child).
    #[allow(dead_code)]
    el_zc_rho_w: Vec<Wire>,
    #[allow(dead_code)]
    el_lc_rho_w: Vec<Wire>,
    #[allow(dead_code)]
    el_eval_w: Vec<(Wire, Wire)>,
    /// boolean: the zc mlv / lc round rhos (round order), the absorbed
    /// z_partial words, and the per-type matrix_evals advice pairs.
    #[allow(dead_code)]
    b_mlv_w: Vec<Wire>,
    #[allow(dead_code)]
    b_lc_w: Vec<Wire>,
    #[allow(dead_code)]
    b_zpartial_w: Vec<Wire>,
    #[allow(dead_code)]
    mat_eval_w: Vec<(Wire, Wire)>,
    /// The residual close-out's prefix slot (and width).
    #[allow(dead_code)]
    pf: (flock_core::circuit::builder::SlotId, usize),
}

/// Emit ONE real child's complete deferred-verifier region — the swap
/// test's whole assembly (chain, PoW, query phase, W-rounds, spine,
/// residual, wiring GKR + sigma, multi-slot element PIOP with the GENERAL
/// strip, multipoint intake, anchor expect with one-hot gathers and
/// eq-table dots, and all THREE assertion emissions) — into `sb` over the
/// shared [`ChildSlots`], publishing exactly what
/// [`check_real_child_region`] walks.
fn emit_real_child_region(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    rt: &RealTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
) -> RealRegion {
    let trace = &rt.trace;
    let stream = &rt.stream;
    let chals = &rt.chals[..];
    let levels = &rt.levels[..];
    let geo = &rt.geo[..];
    let w_rounds = &rt.w_rounds[..];
    let mp_i = &rt.mp_i;
    let inner_pd_i = &rt.inner_pd_i;
    let piop_i = &rt.piop_i;
    let gammas_i = &rt.gammas_i[..];
    let r = levels.len() - 1;
    let m_mp2 = rt.m_mp2;
    let n_log_i = rt.n_log_i;
    let k_cols_i = rt.k_cols_i;

    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match cs.le.iter().find(|(n, _)| *n == lanes) {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(LeafEvalGate::new(lanes));
                    cs.le.push((lanes, sl));
                    sl
                }
            }
        })
        .collect();
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [sb.public_input(), sb.public_input()];
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let (outs, ww) = emit_fs_chain(
        sb,
        cs.q.b3,
        iv2,
        trace,
        stream,
        &rt.bytes,
        vals,
        &mut consts,
        &rt.pub_payloads,
    );

    // The PoW grinding wires: (digest word0, word1, nonce word) per op.
    let pow_pub: Vec<[Wire; 3]> = rt
        .pows
        .iter()
        .map(|&(fin, pay, _)| {
            let sq = &trace.squeezes[fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pay))
                .expect("pow nonce stream word");
            let nw = ww[wi].expect("pow nonce wired");
            [outs[sq[0]][0], outs[sq[0]][1], nw]
        })
        .collect();

    // ---- ROUND 2: the H(publics) region (v2 statement binding) ----
    // Payload 4 of the circuit binding is the 32-byte publics digest; the
    // child's public words themselves are witness, bound here.
    {
        let pays = payload_words(stream);
        assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
        let dw = [
            ww[pays[4][0]].expect("digest word wired"),
            ww[pays[4][1]].expect("digest word wired"),
        ];
        emit_publics_hash(sb, cs.q, iv2, &rt.lo.public, dw, vals, &mut consts);
    }
    let cap_w = cap_wires(stream, &ww, &rt.cap_pays);
    let (to_publish, level_accs) = emit_query_phase(
        sb,
        cs.q,
        iv2,
        &leafeval,
        levels,
        geo,
        &rt.lvl_src,
        &trace.squeezes,
        &outs,
        chals,
        &cap_w,
        vals,
        &mut consts,
        hints,
    );

    // ---- intake W-rounds, spine, residual ----
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
            if vmap.len() <= vi {
                vmap.resize(vi + 1, None);
            }
            vmap[vi] = Some(wi);
        }
    }
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let mrslot = cs.mrs;
    let spine = cs.spine;
    // The assert-zero anchor: a dedicated zero public NO gate consumes,
    // so the zero-delta outputs connected into its class add no
    // dataflow edges (connecting them to the ubiquitous `zw` creates
    // cycles — the acyclicity check draws producer→consumer edges).
    vals.push(F128::ZERO);
    let zassert = sb.public_input();


    // The merged target's TWO HALVES, round-0 posture: the packed-direct
    // half is a MAC chain over absorbed value words × gamma squeeze wires
    // (fully in-circuit); only the RS half — the family-H transpose dots —
    // stays advice, production-checked over the RE-EXPOSED words below.
    let mut pdh_w = zw;
    for pd in gammas_i {
        let gw = outs[trace.squeezes[pd.fin][0]][0];
        pdh_w = sb.gate(cs.macs, &[pdh_w, gw, wv(pd.val_v)])[0];
    }
    vals.push(rt.native_rs_half);
    let rsh_w = sb.public_input();
    let tgt_w = sb.gate(cs.macs, &[rsh_w, ow, pdh_w])[0];
    let mut runw = tgt_w;
    for rr in w_rounds {
        let r_w = outs[trace.squeezes[rr.fin][0]][0];
        runw = sb.gate(mrslot, &[runw, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    // V, the other family-H item, same split: the group-value sum is a MAC
    // chain over absorbed words; V_rs stays advice; and the boundary
    // `running == q_eval·V` CLOSES IN-CIRCUIT as a copy constraint —
    // q_eval needs no exposure at all.
    let mut vgrp_w = zw;
    for &vi in &mp_i.val_vs[256..] {
        vgrp_w = sb.gate(cs.macs, &[vgrp_w, ow, wv(vi)])[0];
    }
    vals.push(rt.native_vrs);
    let vrs_w = sb.public_input();
    let v_w = sb.gate(cs.macs, &[vrs_w, ow, vgrp_w])[0];
    let rhs_v_w = sb.gate(cs.macs, &[zw, wv(inner_pd_i.q_v), v_w])[0];
    sb.connect(runw, rhs_v_w);

    // The ligerito SPINE: start gamma'·q_eval, eval/build per fold,
    // intro-folds consuming the query phase's accumulator wires.
    let gpw = outs[trace.squeezes[inner_pd_i.fin][0]][0];
    let tw0 = sb.gate(
        spine,
        &[zw, zw, zw, zw, zw, zw, wv(inner_pd_i.q_v), gpw, zw],
    );
    let mut tsp = tw0[3];
    let st = sb.gate(
        spine,
        &[zw, zw, zw, zw, wv(rt.start_v_i), wv(rt.start_v_i + 1), tsp, ow, zw],
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = outs[trace.squeezes[lvl.fold_fins[j]][0]][0];
            let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
            tsp = ev[4];
            let bld = sb.gate(spine, &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tsp, ow, zw]);
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r {
            for od in &lvl.ood {
                let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                let f = sb.gate(
                    spine,
                    &[
                        qc,
                        qb,
                        qa,
                        tsp,
                        wv(od.intro_v),
                        wv(od.intro_v + 1),
                        wv(od.y_v),
                        bw,
                        zw,
                    ],
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            }
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = sb.gate(
                spine,
                &[
                    qc,
                    qb,
                    qa,
                    tsp,
                    wv(lvl.intro_v),
                    wv(lvl.intro_v + 1),
                    level_accs[li],
                    bw,
                    zw,
                ],
            );
            (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = sb.gate(spine, &[zw, zw, zw, tsp, zw, zw, level_accs[li], bw, zw]);
            tsp = f[3];
        }
    }
    let t_final = tsp;

    // The RESIDUAL region via the shared emitter (lane-major rotation).
    let yr_wires: Vec<Wire> = (0..rt.yr_len).map(|y| wv(rt.yr_v_i + y)).collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        &rt.w_resid,
        inner_pd_i.fin,
        &yr_wires,
        &trace.squeezes,
        &outs,
        chals,
        vals,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: inner == t_r as a copy constraint.
    sb.connect(inner_w, t_final);

    // ---- the WIRING GKR in-circuit + the sigma emission ----
    let macs = cs.macs;
    let zcr = cs.zcr;
    let gr = &rt.gkr;
    let g_alpha_w = outs[trace.squeezes[gr.alpha_fin][0]][0];
    let g_beta_w = outs[trace.squeezes[gr.beta_fin][0]][0];
    // Every former published-zero delta in this region is a COPY
    // CONSTRAINT now — the proof itself fails on a broken identity.
    let (mut cl_w, mut cr_w) = (wv(gr.top_v), wv(gr.top_v + 1));
    sb.connect(cl_w, cr_w);
    let mut pt_w: Vec<Wire> = Vec::new();
    for lr in &gr.layers {
        let lam_w = outs[trace.squeezes[lr.lam_fin][0]][0];
        let mut run_w = sb.gate(macs, &[cl_w, lam_w, cr_w])[0];
        let mut pt_next: Vec<Wire> = Vec::with_capacity(lr.rounds.len() + 1);
        for (t2, &(gv, rfin)) in lr.rounds.iter().enumerate() {
            let rho_w = outs[trace.squeezes[rfin][0]][0];
            vals.push(lr.g0s[t2]);
            let g0w = sb.input();
            let o = sb.gate(zcr, &[run_w, wv(gv), wv(gv + 1), pt_w[t2], rho_w, g0w, ow]);
            sb.connect(o[0], zassert);
            run_w = o[1];
            pt_next.push(rho_w);
        }
        let (vl0, vl1) = (wv(lr.v_v), wv(lr.v_v + 1));
        let (vr0, vr1) = (wv(lr.v_v + 2), wv(lr.v_v + 3));
        let pl2 = sb.gate(macs, &[zw, vl0, vl1])[0];
        let pr2 = sb.gate(macs, &[zw, vr0, vr1])[0];
        let gate_w = sb.gate(macs, &[pl2, lam_w, pr2])[0];
        sb.connect(gate_w, run_w);
        let ck_w = outs[trace.squeezes[lr.ck_fin][0]][0];
        let sl2 = sb.gate(macs, &[vl0, vl1, ow])[0];
        let sr2 = sb.gate(macs, &[vr0, vr1, ow])[0];
        cl_w = sb.gate(macs, &[vl0, ck_w, sl2])[0];
        cr_w = sb.gate(macs, &[vr0, ck_w, sr2])[0];
        pt_next.push(ck_w);
        pt_w = pt_next;
    }
    assert_eq!(pt_w.len(), rt.mu_i, "the GKR point spans the inner cell space");
    // M̂(ρ) / livê(ρ) as checker-validated advice publics.
    vals.push(rt.mid_n);
    let mid_w = sb.public_input();
    vals.push(rt.live_n);
    let live_w = sb.public_input();
    let (f_w, g_w, sig_w) = (wv(gr.fgs_v), wv(gr.fgs_v + 1), wv(gr.fgs_v + 2));
    let l1 = sb.gate(macs, &[f_w, g_alpha_w, mid_w])[0];
    let l2 = sb.gate(macs, &[l1, g_beta_w, live_w])[0];
    let l3 = sb.gate(macs, &[l2, ow, live_w])[0];
    let l4 = sb.gate(macs, &[l3, ow, ow])[0];
    sb.connect(l4, cl_w);
    let r1 = sb.gate(macs, &[g_w, g_alpha_w, sig_w])[0];
    let r2 = sb.gate(macs, &[r1, g_beta_w, live_w])[0];
    let r3 = sb.gate(macs, &[r2, ow, live_w])[0];
    let r4 = sb.gate(macs, &[r3, ow, ow])[0];
    sb.connect(r4, cr_w);

    // ---- the MULTI-SLOT element PIOP (general strip) ----
    let mut el_zr = zw;
    let sqt = &trace.squeezes[piop_i.tau_fin];
    for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
        let t_w = outs[sqt[k / 4]][k % 4];
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        vals.push(rt.el_g0[k]);
        let g0w = sb.input();
        let o = sb.gate(zcr, &[el_zr, wv(rr.g_v), wv(rr.g_v + 1), t_w, rho_w, g0w, ow]);
        sb.connect(o[0], zassert);
        el_zr = o[1];
    }
    let el_alpha_w = outs[trace.squeezes[piop_i.alpha_fin][0]][0];
    let ea_w = wv(piop_i.eab_v);
    let eb_w = wv(piop_i.eab_v + 1);
    vals.push(rt.a_sum_n);
    let asum_w = sb.public_input();
    vals.push(rt.b_sum_n);
    let bsum_w = sb.public_input();
    let va_w = sb.gate(macs, &[ea_w, asum_w, ow])[0];
    let vb_w = sb.gate(macs, &[eb_w, bsum_w, ow])[0];
    let mut el_lcw = sb.gate(macs, &[va_w, el_alpha_w, vb_w])[0];
    for rr in &piop_i.lc_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        el_lcw = sb.gate(mrslot, &[el_lcw, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }

    // ---- the multipoint intake at R=2, P>0 ----
    let mp_gamma_w = outs[trace.squeezes[mp_i.gamma_fin][0]][0];
    let mut t0_w = zw;
    let mut pw_w = ow;
    let mut mp_pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp_i.val_vs.iter().enumerate() {
        t0_w = sb.gate(macs, &[t0_w, pw_w, wv(vi)])[0];
        if k + 1 < mp_i.val_vs.len() {
            pw_w = sb.gate(macs, &[zw, pw_w, mp_gamma_w])[0];
            mp_pws.push(pw_w);
        }
    }
    let mut tm_w = t0_w;
    let mut mp_rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp_i.rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_rho2_w.push(rho_w);
        tm_w = sb.gate(mrslot, &[tm_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    sb.connect(tm_w, wv(mp_i.anchor_v));
    let mut anc_w = wv(mp_i.anchor_v);
    let mut mp_sig_w: Vec<Wire> = Vec::new();
    for rr in &mp_i.anchor_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_sig_w.push(rho_w);
        anc_w = sb.gate(mrslot, &[anc_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    assert_eq!(mp_sig_w.len(), 2 * (m_mp2 + 1), "sigma spans the anchor layers");

    // ---- the anchor EXPECT at real-inner scale ----
    let extend_const = |pw: &mut Vec<(F128, Wire)>, xn: &[F128]| {
        for &cv2 in &xn[pw.len()..] {
            let w = if cv2 == F128::ZERO {
                zw
            } else {
                assert_eq!(cv2, F128::ONE, "constant point coord is a slot-prefix bit");
                ow
            };
            pw.push((cv2, w));
        }
    };
    use flock_core::zerocheck::univariate_skip_optimized::{
        medium_challenges_ghash, small_challenges_ghash,
    };
    let mut t_vals_b: Vec<F128> = Vec::new();
    t_vals_b.extend_from_slice(&small_challenges_ghash());
    t_vals_b.extend_from_slice(&medium_challenges_ghash());
    assert_eq!(t_vals_b.len(), 7, "the seven baked ghash weights");
    let mlv_pw: Vec<(F128, Wire)> = rt
        .zc_rounds_b
        .iter()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let lc_pw: Vec<(F128, Wire)> = rt
        .lc_rounds_b
        .iter()
        .rev()
        .map(|&(_, ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let mut xab_pw: Vec<(F128, Wire)> = vec![lc_pw[0]];
    xab_pw.extend_from_slice(&mlv_pw[1..1 + n_log_i]);
    xab_pw.extend_from_slice(&lc_pw[1..]);
    extend_const(&mut xab_pw, &rt.x_ab_n);
    let (outer_ch_b, outer_fin_b) = rt.outer_b;
    let mut xc_pw: Vec<(F128, Wire)> = (0..rt.zc_rounds_b.len())
        .map(|k2| {
            if k2 < 7 {
                (t_vals_b[k2], cw(sb, vals, &mut consts, t_vals_b[k2]))
            } else {
                let j = k2 - 7;
                let sq2 = &trace.squeezes[outer_fin_b];
                (chals[outer_ch_b + j], outs[sq2[j / 4]][j % 4])
            }
        })
        .collect();
    extend_const(&mut xc_pw, &rt.x_c_n);
    for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&rt.x_ab_n).enumerate() {
        assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
    }
    for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&rt.x_c_n).enumerate() {
        assert_eq!(nv, xn, "c point coord {i2} is the located wire");
    }

    let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
        let mut seed = ow;
        for chunk in factors.chunks(pf_w) {
            let mut g_in = vec![seed];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            seed = sb.gate(pfslot, &g_in)[0];
        }
        seed
    };
    let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
    let rho_mrg_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let mut rinv_n2: Vec<F128> = rho_mrg_n.clone();
    let mut rinv_w: Vec<Wire> = rho_mrg_w.clone();
    let mut ghat = zw;
    for j in 0..128 {
        if j > 0 {
            let mut lvl_w = Vec::with_capacity(m_mp2);
            for t3 in 0..m_mp2 {
                let y = frob_inv_native(rinv_n2[t3]);
                rinv_n2[t3] = y;
                vals.push(y);
                let yw = sb.input();
                let d = sb.gate(spine, &[zw, zw, zw, rinv_w[t3], zw, zw, yw, yw, zw])[3];
                sb.connect(d, zassert);
                lvl_w.push(yw);
            }
            rinv_w = lvl_w;
        }
        let factors: Vec<(Wire, Wire)> = rinv_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        let eqj = prefix_product(sb, &factors);
        ghat = sb.gate(spine, &[zw, zw, zw, ghat, zw, zw, mp_pws[j], eqj, zw])[3];
    }
    let e_at_w = {
        let factors: Vec<(Wire, Wire)> = rho_mrg_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        prefix_product(sb, &factors)
    };
    let eqc_w: Vec<Wire> = rt
        .bounds_i
        .iter()
        .map(|&(t_c, t_next, _)| {
            let mut factors = Vec::with_capacity(2 * (m_mp2 + 1));
            for l in 0..=m_mp2 {
                factors.push((mp_sig_w[2 * l], if (t_c >> l) & 1 == 1 { ow } else { zw }));
                factors.push((
                    mp_sig_w[2 * l + 1],
                    if (t_next >> l) & 1 == 1 { ow } else { zw },
                ));
            }
            prefix_product(sb, &factors)
        })
        .collect();
    // Column weights via EQ-TABLE DOUBLING (the committed-footprint fix).
    let col_eqc: Vec<Wire> = rt.run_of.iter().map(|&r3| eqc_w[r3]).collect();
    let lo_bits = k_cols_i / 2;
    let eq_dot = |sb: &mut ShapeBuilder, z_col: &[Wire]| -> Wire {
        let build = |sb: &mut ShapeBuilder, coords: &[Wire]| -> Vec<Wire> {
            let mut t2 = vec![ow];
            for &cw2 in coords {
                let mut lo_half = Vec::with_capacity(t2.len());
                let mut hi_half = Vec::with_capacity(t2.len());
                for &e in &t2 {
                    let m2 = sb.gate(macs, &[zw, e, cw2])[0];
                    lo_half.push(sb.gate(macs, &[e, e, cw2])[0]);
                    hi_half.push(m2);
                }
                lo_half.extend(hi_half);
                t2 = lo_half;
            }
            t2
        };
        let lo_t = build(sb, &z_col[..lo_bits]);
        let hi_t = build(sb, &z_col[lo_bits..]);
        let block = lo_t.len();
        let mut acc = zw;
        for (h2, &hw2) in hi_t.iter().enumerate() {
            let mut inner = zw;
            for (l2, &lw2) in lo_t.iter().enumerate() {
                inner = sb.gate(macs, &[inner, lw2, col_eqc[h2 * block + l2]])[0];
            }
            acc = sb.gate(macs, &[acc, hw2, inner])[0];
        }
        acc
    };
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        let z_col_w: Vec<Wire> = xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect();
        let w_st = eq_dot(sb, &z_col_w);
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i { z_row_w[layer] } else { zw };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = if si == 0 {
            ghat
        } else {
            sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pws[128], ghat, zw])[3]
        };
        let wd = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, w_st, gdp[0], zw])[3];
        expect_w = sb.gate(spine, &[zw, zw, zw, expect_w, zw, zw, coeff, wd, zw])[3];
    }
    for (g_ix, members) in rt.groups_ix.iter().enumerate() {
        // Bilinearity; one-hot members bind through ONE prefix row.
        let mut w_st = zw;
        for &i2 in members {
            let pd = &gammas_i[i2];
            let gpd_w = outs[trace.squeezes[pd.fin][0]][0];
            let z_col_n = &rt.vals_rec[pd.pt_v + n_log_i..pd.pt_v + n_log_i + k_cols_i];
            let hot: Option<usize> =
                z_col_n.iter().enumerate().try_fold(0usize, |acc, (j, &x)| {
                    if x == F128::ZERO {
                        Some(acc)
                    } else if x == F128::ONE {
                        Some(acc | (1 << j))
                    } else {
                        None
                    }
                });
            if let Some(h) = hot {
                let factors: Vec<(Wire, Wire)> = (0..k_cols_i)
                    .map(|j| {
                        (
                            wv(pd.pt_v + n_log_i + j),
                            if (h >> j) & 1 == 1 { ow } else { zw },
                        )
                    })
                    .collect();
                let s = prefix_product(sb, &factors);
                let e = sb.gate(macs, &[zw, s, eqc_w[rt.run_of[h]]])[0];
                w_st = sb.gate(macs, &[w_st, gpd_w, e])[0];
            } else {
                let z_col_w: Vec<Wire> = (0..k_cols_i)
                    .map(|j| wv(pd.pt_v + n_log_i + j))
                    .collect();
                let d = eq_dot(sb, &z_col_w);
                w_st = sb.gate(macs, &[w_st, gpd_w, d])[0];
            }
        }
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i {
                wv(gammas_i[members[0]].pt_v + layer)
            } else {
                zw
            };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = sb.gate(macs, &[zw, mp_pws[256 + g_ix], e_at_w])[0];
        let wd = sb.gate(macs, &[zw, w_st, gdp[0]])[0];
        expect_w = sb.gate(macs, &[expect_w, coeff, wd])[0];
    }
    sb.connect(anc_w, expect_w);

    // ---- the assertion EMISSIONS (all three families) ----
    let bl_alpha_w = outs[trace.squeezes[rt.bl_alpha.1][0]][0];
    let mut mat_pub: Vec<Wire> = vec![bl_alpha_w];
    for &(_, _, fin) in &rt.lc_rounds_b {
        mat_pub.push(outs[trace.squeezes[fin][0]][0]);
    }
    let bp_i = rt.lo.proof.boolean.as_ref().expect("boolean side present");
    let mut mat_eval_w: Vec<(Wire, Wire)> = Vec::new();
    for &(a, b) in &bp_i.lincheck.matrix_evals {
        vals.push(a);
        let aw = sb.public_input();
        vals.push(b);
        let bw = sb.public_input();
        mat_pub.push(aw);
        mat_pub.push(bw);
        mat_eval_w.push((aw, bw));
    }
    // ROUND 0: the MatrixAssertion equation's remaining data, published —
    // x_inner_rest (batch-major mlv map), x_outer (mlv rounds 1..1+ν),
    // the const-pin betas + their count-derived eps advice, the z_partial
    // words — and the ~20-row BOOLEAN LINCHECK REPLAY, so the published
    // chain end IS the equation's bound target: entry = α·v_a + v_b +
    // Σ β_t·eps_t from absorbed finals and squeeze wires, rounds through
    // the shared MergedRoundGate slot.
    let inner_b = rt.mat_assert.x_inner_rest.len();
    for j in 0..inner_b {
        let m = if j == 0 { 0 } else { n_log_i + j };
        mat_pub.push(mlv_pw[m].1);
    }
    for j in 0..n_log_i {
        mat_pub.push(mlv_pw[1 + j].1);
    }
    let zpartial_ws: Vec<Wire> = (0..64).map(|i| wv(rt.zp_v + i)).collect();
    let va_b = wv(rt.zc_finals_v);
    let vb_b = wv(rt.zc_finals_v + 1);
    let mut lcb_w = sb.gate(cs.macs, &[vb_b, bl_alpha_w, va_b])[0];
    for (k, &(_, bfin)) in rt.betas_b.iter().enumerate() {
        let bw = outs[trace.squeezes[bfin][0]][0];
        vals.push(rt.eps_n[k]);
        let ew = sb.public_input();
        lcb_w = sb.gate(cs.macs, &[lcb_w, bw, ew])[0];
        mat_pub.push(bw);
        mat_pub.push(ew);
    }
    for &(g_v, _, fin) in &rt.lc_rounds_b {
        let rw = outs[trace.squeezes[fin][0]][0];
        lcb_w = sb.gate(mrslot, &[lcb_w, wv(g_v), wv(g_v + 1), rw])[0];
    }
    mat_pub.extend_from_slice(&zpartial_ws);
    mat_pub.push(lcb_w);
    let mut ela_pub: Vec<Wire> = vec![el_alpha_w];
    for rr in &piop_i.zc_rounds {
        ela_pub.push(outs[trace.squeezes[rr.fin][0]][0]);
    }
    for rr in &piop_i.lc_rounds {
        ela_pub.push(outs[trace.squeezes[rr.fin][0]][0]);
    }
    ela_pub.extend_from_slice(&[va_w, vb_w, wv(gammas_i[rt.z_ix].val_v)]);
    let mut el_eval_w: Vec<(Wire, Wire)> = Vec::new();
    for &(a, b) in &rt.el_assert.evals {
        vals.push(a);
        let aw = sb.public_input();
        vals.push(b);
        let bw = sb.public_input();
        ela_pub.push(aw);
        ela_pub.push(bw);
        el_eval_w.push((aw, bw));
    }

    // ---- the publishes, in the swap's recorded order ----
    let pub_base = sb.public_len();
    for a_wires in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    for w in &level_accs {
        sb.publish(*w);
    }
    for p in &pow_pub {
        for w in p {
            sb.publish(*w);
        }
    }
    sb.publish(t_final);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(*w);
        }
    }
    sb.publish(inner_w);
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    sb.publish(el_zr);
    sb.publish(el_lcw);
    sb.publish(anc_w);
    for w in &mat_pub {
        sb.publish(*w);
    }
    for w in &ela_pub {
        sb.publish(*w);
    }
    // The z_skip squeeze wire, published: the boolean claims' lagrange row
    // lows derive from it — the 2→1 merge's checker rebuilds them from
    // THIS published value (the alpha-expansion trust class).
    sb.publish(outs[trace.squeezes[rt.zskip_fin][0]][0]);
    // ROUND 0's family-H RE-EXPOSURE: the words the rs_half / V_rs advice
    // checks reference — s_hat_v (2×128), the r_dprime squeeze wires
    // (2×7), the two rs gammas, and the 256+P multipoint value words. All
    // wires that already exist; published so the production checker can
    // recompute the transpose dots and the linearized-coefficient
    // combination from PUBLIC data alone. Removed again when the
    // family-H arithmetization lands.
    let mut n_fam_pub = 0usize;
    for &(sv, rfin, _) in &rt.rs_recs {
        for i in 0..128 {
            sb.publish(wv(sv + i));
        }
        let sq = &trace.squeezes[rfin];
        for j in 0..7 {
            sb.publish(outs[sq[j / 4]][j % 4]);
        }
        n_fam_pub += 135;
    }
    for k in 0..2 {
        sb.publish(outs[trace.squeezes[rt.rs_gam_fin + k][0]][0]);
        n_fam_pub += 1;
    }
    for &vi in &mp_i.val_vs {
        sb.publish(wv(vi));
        n_fam_pub += 1;
    }
    // The two family-H advice values themselves, at known tail positions.
    sb.publish(rsh_w);
    sb.publish(vrs_w);
    n_fam_pub += 2;

    let n_query_pub: usize = levels.iter().map(|l| l.a_count).sum();
    let n_tail = levels.len()
        + 3 * rt.pows.len()
        + 3
        + levels.len() * rt.yr_len
        + 1
        + 1
        + rt.mu_i
        + 2
        + 1
        + mat_pub.len()
        + ela_pub.len()
        + 1
        + n_fam_pub;
    RealRegion {
        pub_base,
        n_query_pub,
        n_tail,
        n_mat_pub: mat_pub.len(),
        n_fam_pub,
        n_ela_pub: ela_pub.len(),
        sig_w,
        pt_w,
        el_zc_rho_w: piop_i
            .zc_rounds
            .iter()
            .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
            .collect(),
        el_lc_rho_w: piop_i
            .lc_rounds
            .iter()
            .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
            .collect(),
        el_eval_w,
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: rt
            .lc_rounds_b
            .iter()
            .map(|&(_, _, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: zpartial_ws,
        mat_eval_w,
        pf: (pfslot, pf_w),
    }
}

/// Walk one emitted REAL child region's public block and hold every
/// published value against the tape's native replicas — the swap test's
/// checker, extracted and base-relative. Returns the entries consumed.
fn check_real_child_region(public: &[F128], rt: &RealTape<'_>, r: &RealRegion) -> usize {
    let chals = &rt.chals[..];
    let mut at2 = r.pub_base;
    // The openings bind to the absorbed caps by COPY CONSTRAINT (the
    // in-circuit cap tree) — no per-query publics, no checker walk.
    for (li, lvl) in rt.levels.iter().enumerate() {
        for j in 0..lvl.a_count {
            assert_eq!(public[at2 + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
        }
        at2 += lvl.a_count;
    }
    for (li, want) in rt.native_sums.iter().enumerate() {
        assert_eq!(
            public[at2 + li],
            *want,
            "L{li} enforced sum matches the native replica"
        );
    }
    let pow_base = at2 + rt.native_sums.len();
    for (k, &(_, _, bits)) in rt.pows.iter().enumerate() {
        let d0 = public[pow_base + 3 * k];
        let d1 = public[pow_base + 3 * k + 1];
        let nn = public[pow_base + 3 * k + 2];
        let mut digest = [0u8; 32];
        digest[..8].copy_from_slice(&d0.lo.to_le_bytes());
        digest[8..16].copy_from_slice(&d0.hi.to_le_bytes());
        digest[16..24].copy_from_slice(&d1.lo.to_le_bytes());
        digest[24..].copy_from_slice(&d1.hi.to_le_bytes());
        assert_eq!(nn.hi, 0, "pow {k}: nonce word zero-padded");
        if bits == 0 {
            assert_eq!(nn.lo, 0, "pow {k}: canonical zero nonce");
        } else {
            assert!(
                flock_core::challenger::pow_has_leading_zero_bits(
                    &digest,
                    nn.lo,
                    bits,
                    HashKind::Blake3,
                ),
                "pow {k}: grinding predicate on the published wires"
            );
        }
    }
    let sp_base = pow_base + 3 * rt.pows.len();
    assert_eq!(
        public[sp_base],
        rt.t_final_n,
        "the spine's t_r matches the native replay"
    );
    assert_eq!(
        public[sp_base + 1],
        rt.native_target,
        "the target advice is the native two-halves combination"
    );
    assert_eq!(
        public[sp_base + 2],
        rt.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    let inner_n = check_residual_publics(
        public,
        sp_base + 3,
        &rt.levels,
        &rt.geo,
        &rt.w_resid,
        rt.inner_pd_i.ch,
        &rt.vals_rec[rt.yr_v_i..rt.yr_v_i + rt.yr_len],
        chals,
    );
    assert_eq!(
        inner_n, rt.t_final_n,
        "inner == t_r: the real-inner statement closes"
    );
    // The GKR/element/multipoint/anchor identities are COPY CONSTRAINTS —
    // no publics, no checker items; the proof itself carries them.
    let sig_base = sp_base + 3 + rt.levels.len() * rt.yr_len + 1;
    assert_eq!(
        public[sig_base],
        rt.lo.proof.wiring.gkr.s_sigma_eval,
        "the emitted sigma value is the proof's deferred evaluation"
    );
    let sa = flock_core::circuit::SigmaAssertion {
        rho: public[sig_base + 1..sig_base + 1 + rt.mu_i].to_vec(),
        nu: rt.lo.shape.circuit.cells().nu(),
        value: public[sig_base],
    };
    assert_eq!(sa.rho, rt.sigma_native.rho, "the emitted sigma point");
    assert_eq!(sa.value, rt.sigma_native.value, "the emitted sigma value");
    assert_eq!(sa.nu, rt.sigma_native.nu, "the emitted sigma split");
    assert!(
        sa.check(&rt.lo.shape.circuit),
        "the emitted sigma assertion discharges against the real inner"
    );
    let el_base = sig_base + 1 + rt.mu_i;
    assert_eq!(
        public[el_base],
        rt.el_run_n,
        "the element zc chain ends at the native running claim"
    );
    assert_eq!(
        public[el_base + 1],
        rt.el_assert.target,
        "the element lc chain ends at the native assertion's target"
    );
    assert_eq!(
        public[el_base + 2],
        rt.anc_end_n,
        "the anchor rounds end at the native claim"
    );
    // The assertion emissions, held against the DEFERRED verify's own
    // assertions — a parent reads the accumulator inputs off the segment.
    let mat_base = el_base + 3;
    assert_eq!(
        public[mat_base],
        rt.mat_assert.alpha,
        "the emitted matrix alpha is the assertion's"
    );
    for (j, &(_, ch, _)) in rt.lc_rounds_b.iter().enumerate() {
        assert_eq!(
            public[mat_base + 1 + j],
            chals[ch],
            "matrix point coord {j} is the located round wire"
        );
    }
    let bp_i = rt.lo.proof.boolean.as_ref().expect("boolean side present");
    for (j, &(a, b)) in bp_i.lincheck.matrix_evals.iter().enumerate() {
        assert_eq!(
            (
                public[mat_base + 1 + rt.lc_rounds_b.len() + 2 * j],
                public[mat_base + 1 + rt.lc_rounds_b.len() + 2 * j + 1],
            ),
            (a, b),
            "matrix_evals pair {j} rides as bound advice"
        );
    }
    // ROUND 0's extension: every remaining datum of the MatrixAssertion
    // equation, published and held against the assertion itself.
    let mut mq = mat_base + 1 + rt.lc_rounds_b.len() + 2 * bp_i.lincheck.matrix_evals.len();
    for (j, &x) in rt.mat_assert.x_inner_rest.iter().enumerate() {
        assert_eq!(public[mq + j], x, "x_inner_rest {j} published");
    }
    mq += rt.mat_assert.x_inner_rest.len();
    for j in 0..rt.n_log_i {
        assert_eq!(
            public[mq + j],
            chals[rt.zc_rounds_b[1 + j].0],
            "x_outer {j} published"
        );
    }
    mq += rt.n_log_i;
    for (k, &(bch, _)) in rt.betas_b.iter().enumerate() {
        assert_eq!(public[mq], chals[bch], "beta {k} published");
        assert_eq!(public[mq + 1], rt.eps_n[k], "eps {k} advice");
        mq += 2;
    }
    for (j, &z) in rt.mat_assert.z_partial.iter().enumerate() {
        assert_eq!(public[mq + j], z, "z_partial {j} published");
    }
    mq += 64;
    assert_eq!(
        public[mq],
        rt.mat_assert.target,
        "the in-circuit boolean lc replay ends at the assertion's target"
    );
    assert_eq!(mq + 1, mat_base + r.n_mat_pub, "the mat block walk is complete");
    let ela_base = mat_base + r.n_mat_pub;
    assert_eq!(
        public[ela_base],
        rt.el_assert.alpha,
        "the emitted element alpha is the assertion's"
    );
    let n_er = rt.piop_i.zc_rounds.len() + rt.piop_i.lc_rounds.len();
    for (j, rr) in rt
        .piop_i
        .zc_rounds
        .iter()
        .chain(rt.piop_i.lc_rounds.iter())
        .enumerate()
    {
        assert_eq!(
            public[ela_base + 1 + j],
            chals[rr.ch],
            "element point coord {j} is the located round wire"
        );
    }
    assert_eq!(
        public[ela_base + 1 + n_er],
        rt.vals_rec[rt.piop_i.eab_v] + rt.a_sum_n,
        "the emitted va is the strip-derived value"
    );
    assert_eq!(
        public[ela_base + 1 + n_er + 1],
        rt.vals_rec[rt.piop_i.eab_v + 1] + rt.b_sum_n,
        "the emitted vb is the strip-derived value"
    );
    assert_eq!(
        public[ela_base + 1 + n_er + 2],
        rt.el_assert.z_eval,
        "the emitted z_eval is the assertion's"
    );
    for (j, &(a, b)) in rt.el_assert.evals.iter().enumerate() {
        assert_eq!(
            (
                public[ela_base + 1 + n_er + 3 + 2 * j],
                public[ela_base + 1 + n_er + 3 + 2 * j + 1],
            ),
            (a, b),
            "per-slot eval pair {j} rides as bound advice"
        );
    }
    assert_eq!(
        public[ela_base + r.n_ela_pub],
        chals[rt.zskip_ch],
        "the published z_skip is the located squeeze"
    );
    // The family-H re-exposure block: the words the rs_half / V_rs advice
    // reference, all published — validated here against the proof's own
    // fields and the located challenges.
    let mut fq = ela_base + r.n_ela_pub + 1;
    for (k, &(_, _, rc)) in rt.rs_recs.iter().enumerate() {
        for (i, &w) in rt.lo.proof.pcs_open.ring_switches[k].s_hat_v.iter().enumerate() {
            assert_eq!(public[fq + i], w, "s_hat_v[{k}][{i}] re-exposed");
        }
        fq += 128;
        for j in 0..7 {
            assert_eq!(public[fq + j], chals[rc + j], "r_dprime[{k}][{j}] re-exposed");
        }
        fq += 7;
    }
    for k in 0..2 {
        assert_eq!(public[fq + k], chals[rt.rs_gam_ch + k], "rs gamma {k} re-exposed");
    }
    fq += 2;
    let fro = &rt.lo.proof.pcs_open.frobenius;
    for (k, &vi) in rt.mp_i.val_vs.iter().enumerate() {
        let want = if k < 256 {
            fro.values[k / 128][k % 128]
        } else {
            fro.group_values[k - 256]
        };
        assert_eq!(public[fq + k], want, "mp value {k} re-exposed");
        let _ = vi;
    }
    fq += rt.mp_i.val_vs.len();
    assert_eq!(public[fq], rt.native_rs_half, "the rs_half advice");
    assert_eq!(public[fq + 1], rt.native_vrs, "the V_rs advice");
    assert_eq!(
        fq + 2,
        r.pub_base + r.n_query_pub + r.n_tail,
        "the family-H block is the very tail"
    );
    r.n_query_pub + r.n_tail
}

#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp10_leaf_outer_inner_tape() {
    use flock_prover::prover::UnionElementSlotInput;

    let lo = build_leaf_outer();
    let rt = RealTape::new(&lo, DOMAIN);

    // ---- the outer-of-outer: one REAL child region in one builder ----
    // The parse, the assembly and the checker are the extracted
    // [`RealTape`] / [`emit_real_child_region`] / [`check_real_child_region`]
    // — the SAME machinery the 2→1 merge node instantiates per child, so
    // this test keeping green is what makes the extraction faithful.
    let nu2 = (rt.b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, rt.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let region = emit_real_child_region(&mut sb, &mut cs, &rt, &mut vals, &mut hints);
    let shape2 = sb.finish().expect("the swap outer builds");
    // Cell-slot budget: every gate IO word is ALSO a wiring gather claim,
    // so schema words are the budget for both mu and claims. The anchor
    // expect's AssistLayerGate (+13 words) tipped the 256 boundary to
    // mu 24 — ACCEPTED per the sequencing decision (the per-class-nu
    // layout redesign subsumes word-level consolidation; do not spend
    // throwaway trims here).
    assert!(
        shape2.circuit.cells().slots().len() <= 512,
        "the swap outer's cell-slot budget regressed past mu 24 ({} slots)",
        shape2.circuit.cells().slots().len()
    );
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_real_child_region(&built2.public, &rt, &region);
    assert_eq!(
        region.pub_base + consumed,
        built2.public.len(),
        "the region's publics are the whole tail"
    );

    // The outer-of-outer proves and verifies over the circuit path.
    let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
    let pcs2 = PcsParams {
        m: union2.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union2.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let b3_r1cs2 = blake3::build_block_r1cs(nu2);
    let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
    let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
    let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
    let spread_ty2 = BitSpreadTable::new(rt.spread_w);
    let spread_r1cs2 = spread_ty2.build_block_r1cs(nu2);
    let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
    let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
        (
            shape2.registry_slot(cs.q.b3),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(
                    built2.rows::<Blake3Gate>(cs.q.b3),
                    nu2,
                ),
                b3_lc2,
            ),
        ),
        (
            shape2.registry_slot(cs.q.swap),
            UnionSlotProverInput::new(
                SwapTable::generate_witness_batch_major(built2.rows::<SwapGate>(cs.q.swap), nu2),
                swap_lc2,
            ),
        ),
        (
            shape2.registry_slot(cs.q.spread),
            UnionSlotProverInput::new(
                spread_ty2.generate_witness_batch_major(
                    built2.rows::<BitSpreadGate>(cs.q.spread),
                    nu2,
                ),
                spread_lc2,
            ),
        ),
    ];
    bslots.sort_by_key(|(i, _)| *i);
    let mut el_ord: Vec<(usize, Vec<F128>)> = cs
        .element_slot_ids()
        .into_iter()
        .map(|sl| {
            let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                SlotWitness::Element(z) => z.clone(),
                other => panic!("element slot produced {other:?}"),
            };
            (shape2.registry_slot(sl), z)
        })
        .collect();
    el_ord.sort_by_key(|(i, _)| *i);
    let el_inputs: Vec<UnionElementSlotInput> = el_ord
        .into_iter()
        .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
        .collect();
    let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape2.registry_slot(cs.q.b3), b3_lc2),
        (shape2.registry_slot(cs.q.swap), swap_lc2),
        (shape2.registry_slot(cs.q.spread), spread_lc2),
    ];
    lco.sort_by_key(|(i, _)| *i);
    let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lco.into_iter().map(|(_, c)| c).collect();
    let t0p = std::time::Instant::now();
    let mut ch2 = FsChallenger::new(DOMAIN);
    let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
        &union2,
        &shape2.circuit,
        &built2.public,
        &pcs2,
        bslots.into_iter().map(|(_, x)| x).collect(),
        el_inputs,
        &mut ch2,
    );
    let prove_ms = t0p.elapsed().as_secs_f64() * 1e3;
    let t0v = std::time::Instant::now();
    let mut ch2 = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union2,
        &shape2.circuit,
        &built2.public,
        &lcs2,
        &ocommit,
        &oproof,
        &pcs2,
        &mut ch2,
    )
    .expect("the swap outer verifies");
    let verify_ms = t0v.elapsed().as_secs_f64() * 1e3;
    // The DEFERRED verify — what a parent node actually runs: no native
    // sigma discharge (the O(2^mu) eval leaves as a foldable claim), no
    // matrix work. The plain-vs-deferred gap IS sigma v1's cost, and
    // route B is why recursion never pays it.
    let t0d = std::time::Instant::now();
    let mut ch2 = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit_deferred(
        &union2,
        &shape2.circuit,
        &built2.public,
        &lcs2,
        &ocommit,
        &oproof,
        &pcs2,
        &mut ch2,
    )
    .expect("the swap outer verifies deferred");
    let deferred_ms = t0d.elapsed().as_secs_f64() * 1e3;
    println!(
        "  outer verify: plain {verify_ms:.0} ms | DEFERRED {deferred_ms:.0} ms \
         (the gap = the native sigma discharge a parent never pays)"
    );

    println!(
        "\nMVP-10 SWAP steps 1-5 — the LEAF OUTER as the inner\n  \
         inner: dense_m {} | mu {} | levels (q, depth) {:?}\n  \
         pd claims {} (element pair + {} gathers) | P {} | L0 lanes {}/{}\n  \
         outer-of-outer: b3 rows {} | nu {} | dense_m {} | mu {}\n  \
         carries: chain, QUERY PHASE, PoW, W-rounds (rho bound), SPINE\n  \
         (t_r bound), RESIDUAL (rotated; inner == t_r closes), WIRING\n  \
         GKR (21 layers) + sigma (emitted, discharges), the MULTI-SLOT\n  \
         element PIOP (general strip), the MULTIPOINT intake, the ANCHOR\n  \
         EXPECT (one-hot gathers), and ALL THREE assertion emissions\n  \
         (matrix + element + sigma — the parent folds from publics)\n  \
         prove {:.0} ms | verify {:.0} ms | proof {:.1} KiB\n",
        lo.pcs.m,
        rt.mu_i,
        rt.geo.iter().map(|g| (g.q, g.depth)).collect::<Vec<_>>(),
        rt.gammas_i.len(),
        rt.n_gather,
        rt.n_p,
        rt.geo[0].row_words,
        rt.geo[0].lanes,
        rt.b3_rows,
        nu2,
        union2.dense_m(),
        shape2.circuit.cells().mu(),
        prove_ms,
        verify_ms,
        bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

// ---------------------------------------------------------------------------
// The minimal mixed inner and its reusable child-region machinery
// (mvp10's assembly, extracted so the mvp11 merge node can instantiate a
// child-tape region per child — the build_leaf_outer precedent)
// ---------------------------------------------------------------------------

/// A minimal MIXED circuit inner — a blake3 chain feeding MacGate rows across
/// the class boundary, ends published — proven over the circuit path and
/// verified DEFERRED. The shape mvp10 pins at `(256, 32)` and the mvp11 merge
/// children instantiate at `(128, 16)`. The seed varies only the witness
/// (message words), so same-parameter instances share the CIRCUIT — and its
/// digest, the key the accumulator folds sigma under — while their claims
/// land at unrelated FS points, which is what a merge node actually sees.
struct MixedInner {
    nu: usize,
    built: flock_core::circuit::builder::BuiltCircuit,
    proof: flock_core::proof::R1csProofCircuitMerged,
    commitment: flock_core::pcs::commit::Commitment,
    pcs: PcsParams,
    work: flock_core::verifier::DeferredMatrixWork,
    sigma: flock_core::circuit::SigmaAssertion,
}

fn build_mixed_inner(n_blocks: usize, mac_take: usize, seed: u64) -> MixedInner {
    use flock_prover::prover::UnionElementSlotInput;

    let nu = n_blocks.trailing_zeros() as usize;
    assert_eq!(1usize << nu, n_blocks, "block count is a power of two");
    let mut rng = Rng(seed);
    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let mac = b.slot(MacGate::new());
    let iv = pack8(&IV);
    let mut cv = [b.public_value(iv[0]), b.public_value(iv[1])];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let mut hash_in = vec![cv[0], cv[1]];
        for j in 0..4 {
            hash_in.push(b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())));
        }
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == n_blocks {
            flags |= CHUNK_END;
        }
        hash_in.push(b.public_value(pack_params(0, 64, flags)));
        let out = b.gate(hash, &hash_in);
        cv = [out[0], out[1]];
        outs.push(out);
    }
    // The cross-class wiring: element rows consuming hash outputs.
    let zero = b.public_value(F128::ZERO);
    let mut acc = zero;
    for out in outs.iter().take(mac_take) {
        acc = b.gate(mac, &[acc, out[2], out[3]])[0];
    }
    b.publish(acc);
    b.publish(cv[0]);
    b.publish(cv[1]);
    let built = b.finish().expect("the mixed inner builds");

    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        // BLAKE3 for BOTH Merkle and FS: the inner stays recursable — a
        // child-tape region replays this transcript in-circuit, and the
        // defaults diverge silently (both recorded gotchas).
        merkle_hash: HashKind::Blake3,
    };
    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let el_z = match MacGate::new().witness(built.rows::<MacGate>(mac), nu) {
        SlotWitness::Element(z) => z,
        other => panic!("mac witness is {other:?}"),
    };
    let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu),
            blake_lc,
        )],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el_z)
        })],
        &mut ch,
    );
    // The DEFERRED verify of the same inner: the independent reference for
    // everything a child-tape region re-derives — it exposes the boolean
    // MatrixAssertion, the ElementAssertion and the SigmaAssertion natively,
    // so assemblies are checked against the verifier's own data rather than
    // against a formula this file also wrote.
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
    let (_, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the deferred verify accepts an honest mixed inner");
    assert!(
        work.boolean.is_some(),
        "sigma never travels alone: the boolean matrix work rides with it"
    );
    assert!(
        work.element.is_some(),
        "the MacGate slot yields an element assertion"
    );
    MixedInner {
        nu,
        built,
        proof,
        commitment,
        pcs: pcs_params,
        work,
        sigma,
    }
}

/// One wiring-GKR layer, located on the tape (the assembly's wire map).
struct GkrLayerRec {
    lam_fin: usize,
    rounds: Vec<(usize, usize)>, // (g_v, squeeze fin)
    g0s: Vec<F128>,
    v_v: usize, // vl0; vl1/vr0/vr1 follow
    ck_fin: usize,
}
struct GkrRec {
    alpha_fin: usize,
    beta_fin: usize,
    top_v: usize,
    layers: Vec<GkrLayerRec>,
    fgs_v: usize, // f_eval, g_eval, s_sigma consecutive
    r_pt: Vec<F128>,
}

/// The ELEMENT PIOP region, located. Round tuples are `(g_v, fin, ch)`.
struct ElPiopRec {
    tau_fin: usize,
    tau_ch: usize,
    zc_rounds: Vec<(usize, usize, usize)>,
    eab_v: usize,
    alpha_fin: usize,
    alpha_ch: usize,
    lc_rounds: Vec<(usize, usize, usize)>,
}

fn frob_inv_native(x: F128) -> F128 {
    let mut y = x;
    for _ in 0..127 {
        y = y * y;
    }
    y
}

/// One recorded child verification, parsed: the tape pinned op-for-op, every
/// region located, and every native replica the emitter and checker consume.
/// This is mvp10's step-1 parsing as a reusable unit — `new` runs the
/// RECORDING verify itself, so instantiating a child region for the mvp11
/// merge node re-asserts the whole map on that child's tape.
struct ChildTape<'p> {
    inner: &'p MixedInner,
    // the recorded tape
    vals_rec: Vec<F128>,
    chals: Vec<F128>,
    /// Which byte payloads stay PUBLIC under the witness/public split.
    pub_payloads: Vec<bool>,
    /// Per level, the absorbed cap's payload index ([`cap_payloads`]).
    cap_pays: Vec<usize>,
    // chain materials
    trace: flock_prover::r1cs_hashes::fs_chain::FsChainTrace,
    stream: flock_core::transcript_record::Stream,
    bytes: Vec<u8>,
    b3_rows: usize,
    spread_w: usize,
    // located regions
    gkr: GkrRec,
    el: ElPiopRec,
    start_v: usize,
    gammas_o: Vec<PdRec>,
    w_rounds: Vec<RoundRec>,
    w_resid: Vec<RoundRec>,
    mp_o: MpRec,
    inner_pd2: InnerPd,
    yr_v2: usize,
    yr_len: usize,
    levels: Vec<OpenLevel>,
    lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    geo: Vec<Lvl>,
    native_sums: Vec<F128>,
    n_pd: usize,
    n_p: usize,
    // the boolean PIOP's round ordinals, located with fins ((ch, fin) pairs)
    zc_rounds_b: Vec<(usize, usize)>,
    outer_b: (usize, usize),
    lc_rounds_b: Vec<(usize, usize)>,
    // z_skip's ordinals (the boolean claims' lagrange row lows derive from
    // it — published, checker-validated) and z_partial's value ordinal (the
    // boolean claims' column lows — absorbed child words, connectable).
    zskip_ch: usize,
    zskip_fin: usize,
    zp_v: usize,
    // published chain ordinals
    ga_c: usize,
    ga_fin: usize,
    mg_c: usize,
    mg_fin: usize,
    // native references + replicas
    bool_assert: flock_core::lincheck::MatrixAssertion,
    el_assert: flock_core::element_r1cs::union::ElementAssertion,
    sigma_native: flock_core::circuit::SigmaAssertion,
    el_g0: Vec<F128>,
    el_run_n: F128,
    a_sum_n: F128,
    b_sum_n: F128,
    native_target: F128,
    native_running: F128,
    t_final_n: F128,
    anc_end_n: F128,
    mid_n: F128,
    live_n: F128,
    mu_i: usize,
    // anchor-expect geometry — statement constants of the inner shape
    n_log_i: usize,
    k_cols_i: usize,
    m_mp2: usize,
    bounds_i: Vec<(u64, u64, u32)>,
    run_y0: Vec<usize>,
    comp_ix: usize,
    x_ab_n: Vec<F128>,
    x_c_n: Vec<F128>,
    groups_ix: Vec<Vec<usize>>,
}

impl<'p> ChildTape<'p> {
    fn new(inner: &'p MixedInner, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let built = &inner.built;
        let proof = &inner.proof;
        let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
        let blake_r1cs = blake3::build_block_r1cs(inner.nu);
        let blake_lc = blake_r1cs.csc_lincheck_circuit();
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
        let mut rec = RecordingChallenger::new(FsChallenger::with_hash(domain, HashKind::Blake3));
        let native_claims = verifier::verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &inner.commitment,
            proof,
            &inner.pcs,
            &mut rec,
        )
        .expect("the mixed circuit inner verifies")
        .boolean
        .expect("the boolean class yields the RS (ab, c) claims");
        let bool_assert = inner.work.boolean.clone().expect("boolean matrix work");
        let el_assert = inner.work.element.clone().expect("an element PIOP ran");
        let sigma_native = inner.sigma.clone();
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops: Vec<Op> = t_shape.ops().to_vec();
        let mut pub_payloads = bytes_payload_mask(&ops);

        // ---- the label map: the region order the assembly builds against ----
        let find = |label: &[u8]| -> Vec<usize> {
            ops.iter()
                .enumerate()
                .filter_map(|(i, op)| match op {
                    Op::Label(l) if l.as_slice() == label => Some(i),
                    _ => None,
                })
                .collect()
        };
        let zc_l = find(b"flock-zerocheck-v0");
        let lc_l = find(b"flock-lincheck-v0");
        let elzc_l = find(b"flock-element-union-zc-v0");
        let el_l = find(b"flock-element-union-lc-v0");
        assert_eq!(elzc_l.len(), 1, "one element zerocheck");
        let gkr_l = find(b"flock-product-gkr-batched-v0");
        let mo_l = find(b"flock-merged-open-v0");
        let rs_l = find(b"flock-ring-switch-v0");
        let mp_l = find(b"flock-multipoint-twisted-v1");
        let fa_l = find(b"flock-frobenius-assist-v0");
        assert_eq!(zc_l.len(), 1, "one boolean zerocheck");
        assert_eq!(lc_l.len(), 1, "one boolean lincheck");
        assert_eq!(el_l.len(), 1, "one element lincheck region");
        assert!(elzc_l[0] < el_l[0], "element zc before element lc");
        assert_eq!(gkr_l.len(), 1, "one batched wiring GKR");
        assert_eq!(mo_l.len(), 1, "one merged open");
        assert_eq!(rs_l.len(), 2, "rs x 2 — one ab/c pair for the boolean class");
        assert_eq!(mp_l.len(), 1, "one multipoint region");
        assert_eq!(fa_l.len(), 1, "one anchor region");
        assert!(zc_l[0] < lc_l[0], "boolean zc before boolean lc");
        assert!(lc_l[0] < el_l[0], "boolean PIOP before element PIOP");
        assert!(el_l[0] < gkr_l[0], "element PIOP before the wiring GKR");
        assert!(gkr_l[0] < mo_l[0], "wiring GKR before the merged open");
        assert!(mo_l[0] < rs_l[0] && rs_l[1] < mp_l[0] && mp_l[0] < fa_l[0]);

        // (v, c) counters up to an op index — the walker every pin shares.
        let vc_at = |end: usize| -> (usize, usize) {
            let (mut v, mut c) = (0usize, 0usize);
            for op in &ops[..end] {
                match op {
                    Op::SqueezeScalar => c += 1,
                    Op::SqueezeSlice(n) => c += n,
                    Op::ObserveScalar => v += 1,
                    Op::ObserveSlice(n) => v += n,
                    _ => {}
                }
            }
            (v, c)
        };
        // fin ordinal of the op at `end` = finalizing ops strictly before it.
        let fin_at = |end: usize| ops[..end].iter().filter(|o| o.finalizes()).count();

        // ---- the boolean zerocheck slices, same shape as the leaf ----
        let bp = proof.boolean.as_ref().expect("boolean side present");
        assert!(proof.element.is_some(), "element side present");
        {
            let mut i = zc_l[0] + 1;
            assert!(matches!(ops[i], Op::SqueezeSlice(_)), "zc tau lo");
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeSlice(_)), "zc tau hi");
            i += 1;
            assert!(matches!(ops[i], Op::ObserveSlice(64)), "round1_ab");
            let (v0, _) = vc_at(i);
            assert_eq!(
                &vals_rec[v0..v0 + 64],
                &bp.zerocheck.round1_ab[..],
                "round1_ab on the stream"
            );
            i += 1;
            assert!(matches!(ops[i], Op::ObserveSlice(64)), "round1_c");
            let (v1, _) = vc_at(i);
            assert_eq!(
                &vals_rec[v1..v1 + 64],
                &bp.zerocheck.round1_c[..],
                "round1_c on the stream"
            );
        }

        // ---- the wiring GKR region, walked op by op ----
        // The transcription map: [alpha, beta squeezes | top pair observed |
        // per layer k: lambda squeeze, k x (2 obs + squeeze) rounds — the
        // ZcRoundGate shape VERBATIM — then (vl0, vl1, vr0, vr1) observed,
        // the layer check, and the c_k squeeze folding the claims | the
        // (f, g, s_sigma) triple observed last]. The walk also RECORDS the
        // ordinals the assembly wires against, and the per-round `g0` advice.
        let gkr_rec = {
            let gkr = &proof.wiring.gkr;
            let mut i = gkr_l[0] + 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr alpha");
            let (_, c_alpha) = vc_at(i);
            let alpha_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr beta");
            let beta_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::ObserveScalar), "top lhs");
            let (tv, _) = vc_at(i);
            assert_eq!(vals_rec[tv], gkr.top_lhs, "top_lhs on the stream");
            assert_eq!(vals_rec[tv + 1], gkr.top_rhs, "top_rhs on the stream");
            assert_eq!(gkr.top_lhs, gkr.top_rhs, "the grand products agree");
            i += 2;
            // The layer walk + native replay in lockstep.
            let (mut claim_l, mut claim_r) = (gkr.top_lhs, gkr.top_rhs);
            let mut r_pt: Vec<F128> = Vec::new();
            let mut lrecs: Vec<GkrLayerRec> = Vec::new();
            for (k, layer) in gkr.layers.iter().enumerate() {
                assert_eq!(layer.rounds.len(), k, "layer {k} has k rounds");
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} lambda");
                let (_, lc2) = vc_at(i);
                let lambda = chals[lc2];
                let lam_fin = fin_at(i);
                i += 1;
                let mut c_run = claim_l + lambda * claim_r;
                let mut r_prime = Vec::with_capacity(k + 1);
                let mut rrecs: Vec<(usize, usize)> = Vec::new();
                let mut g0s: Vec<F128> = Vec::new();
                for (t2, &(g1, gi)) in layer.rounds.iter().enumerate() {
                    assert!(matches!(ops[i], Op::ObserveScalar), "round obs g1");
                    let (gv, _) = vc_at(i);
                    assert_eq!(vals_rec[gv], g1, "layer {k} round {t2} g1");
                    assert_eq!(vals_rec[gv + 1], gi, "layer {k} round {t2} g_inf");
                    assert!(matches!(ops[i + 2], Op::SqueezeScalar), "round rho");
                    let (_, rc2) = vc_at(i + 2);
                    let rho = chals[rc2];
                    rrecs.push((gv, fin_at(i + 2)));
                    i += 3;
                    let r_eq = r_pt[t2];
                    let g0 = (c_run + r_eq * g1) * (F128::ONE + r_eq).inv();
                    g0s.push(g0);
                    c_run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
                    r_prime.push(rho);
                }
                let (vv, _) = vc_at(i);
                for (j, want) in [layer.vl0, layer.vl1, layer.vr0, layer.vr1]
                    .into_iter()
                    .enumerate()
                {
                    assert!(matches!(ops[i], Op::ObserveScalar), "layer value obs");
                    assert_eq!(vals_rec[vv + j], want, "layer {k} value {j}");
                    i += 1;
                }
                assert_eq!(
                    c_run,
                    layer.vl0 * layer.vl1 + lambda * (layer.vr0 * layer.vr1),
                    "layer {k} closes"
                );
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} c_k");
                let (_, cc2) = vc_at(i);
                let c_k = chals[cc2];
                let ck_fin = fin_at(i);
                i += 1;
                claim_l = (F128::ONE + c_k) * layer.vl0 + c_k * layer.vl1;
                claim_r = (F128::ONE + c_k) * layer.vr0 + c_k * layer.vr1;
                r_prime.push(c_k);
                r_pt = r_prime;
                lrecs.push(GkrLayerRec {
                    lam_fin,
                    rounds: rrecs,
                    g0s,
                    v_v: vv,
                    ck_fin,
                });
            }
            // The input checks: s_id(rho) closed-form NATIVE, s_sigma from
            // the PROOF — the deferred value the assertion carries.
            let mu2 = built.shape.circuit.cells().mu();
            assert_eq!(r_pt.len(), mu2, "the GKR point spans the cell space");
            let alpha2 = chals[c_alpha];
            let beta2 = chals[c_alpha + 1];
            let basis = flock_core::product_gkr::s_id_basis(mu2);
            // Masked input checks under the live-identity padding.
            let mask_w = built.shape.circuit.live_mask();
            let tail_w2 = (beta2 + F128::ONE) * mask_w.live_eval(&r_pt) + F128::ONE;
            assert_eq!(
                claim_l,
                gkr.f_eval + alpha2 * mask_w.masked_id_eval(&basis, &r_pt) + tail_w2,
                "lhs input check replays (masked)"
            );
            assert_eq!(
                claim_r,
                gkr.g_eval + alpha2 * gkr.s_sigma_eval + tail_w2,
                "rhs input check replays with the DEFERRED (masked) sigma value"
            );
            // The triple observed last — the assertion's value wire.
            let (fv, _) = vc_at(i);
            assert!(matches!(ops[i], Op::ObserveScalar), "f_eval obs");
            assert_eq!(vals_rec[fv], gkr.f_eval, "f_eval on the stream");
            assert_eq!(vals_rec[fv + 1], gkr.g_eval, "g_eval on the stream");
            assert_eq!(vals_rec[fv + 2], gkr.s_sigma_eval, "s_sigma on the stream");
            GkrRec {
                alpha_fin,
                beta_fin,
                top_v: tv,
                layers: lrecs,
                fgs_v: fv,
                r_pt,
            }
        };

        // ---- the ELEMENT PIOP region, located ----
        // Shape, per `parse_open_levels`' element branch: [tau slice |
        // tau_len rounds | ea, eb, ec | lc label | alpha | lc rounds].
        let el_rec = {
            let mut i = elzc_l[0] + 1;
            let (tau_fin, tau_ch, tau_len) = match ops[i] {
                Op::SqueezeSlice(n) => (fin_at(i), vc_at(i).1, n),
                ref o => panic!("element tau, got {o:?}"),
            };
            i += 1;
            let mut zc_rounds = Vec::with_capacity(tau_len);
            for _ in 0..tau_len {
                let (gv, _) = vc_at(i);
                assert!(matches!(ops[i], Op::ObserveScalar), "el zc msg");
                assert!(matches!(ops[i + 1], Op::ObserveScalar), "el zc msg");
                assert!(matches!(ops[i + 2], Op::SqueezeScalar), "el zc rho");
                zc_rounds.push((gv, fin_at(i + 2), vc_at(i + 2).1));
                i += 3;
            }
            let (eab_v, _) = vc_at(i);
            for _ in 0..3 {
                assert!(matches!(ops[i], Op::ObserveScalar), "el zc final");
                i += 1;
            }
            assert_eq!(i, el_l[0], "the lc label follows the finals");
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "el lc alpha");
            let (alpha_fin, alpha_ch) = (fin_at(i), vc_at(i).1);
            i += 1;
            let mut lc_rounds = Vec::new();
            while matches!(ops[i], Op::ObserveScalar)
                && matches!(ops[i + 1], Op::ObserveScalar)
                && matches!(ops[i + 2], Op::SqueezeScalar)
            {
                let (gv, _) = vc_at(i);
                lc_rounds.push((gv, fin_at(i + 2), vc_at(i + 2).1));
                i += 3;
            }
            assert!(!zc_rounds.is_empty() && !lc_rounds.is_empty(), "el rounds");
            ElPiopRec {
                tau_fin,
                tau_ch,
                zc_rounds,
                eab_v,
                alpha_fin,
                alpha_ch,
                lc_rounds,
            }
        };

        // ---- the merged open: rs x 2, then the packed-direct claims ----
        let (pd_recs, mp_val_v, rs_recs, rs_gam_ch) = {
            let mut i = mo_l[0] + 1;
            let mut rs_recs: Vec<(usize, usize)> = Vec::new(); // (s_hat_v index, r_dprime ch)
            for k in 0..2 {
                assert!(
                    matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0"),
                    "rs region {k}"
                );
                i += 1;
                assert!(matches!(ops[i], Op::ObserveSlice(128)), "s_hat_v slice");
                let (sv, _) = vc_at(i);
                assert_eq!(
                    &vals_rec[sv..sv + 128],
                    &proof.pcs_open.ring_switches[k].s_hat_v[..],
                    "s_hat_v {k} on the stream"
                );
                i += 1;
                assert!(matches!(ops[i], Op::SqueezeSlice(7)), "r_dprime");
                rs_recs.push((sv, vc_at(i).1));
                i += 1;
            }
            let rs_gam_ch = vc_at(i).1;
            for _ in 0..2 {
                assert!(matches!(ops[i], Op::SqueezeScalar), "rs gamma");
                i += 1;
            }
            // Packed-direct claims: [ObserveSlice(point), ObserveScalar(value),
            // SqueezeScalar(gamma)] each.
            let mut pd_recs: Vec<(usize, usize)> = Vec::new(); // (point_len, value index)
            while let Op::ObserveSlice(n) = ops[i] {
                let (_, _) = vc_at(i);
                i += 1;
                assert!(matches!(ops[i], Op::ObserveScalar), "pd value");
                let (pv, _) = vc_at(i);
                i += 1;
                assert!(matches!(ops[i], Op::SqueezeScalar), "pd gamma");
                i += 1;
                pd_recs.push((n, pv));
            }
            // W rounds until the multipoint label.
            let mut w_rounds = 0usize;
            while matches!(ops[i], Op::ObserveScalar) {
                assert!(matches!(ops[i + 1], Op::ObserveScalar), "w round pair");
                assert!(matches!(ops[i + 2], Op::SqueezeScalar), "w round squeeze");
                i += 3;
                w_rounds += 1;
            }
            assert_eq!(
                w_rounds,
                proof.pcs_open.merged_rounds.len(),
                "the W rounds fill the dense domain"
            );
            while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1")
            {
                i += 1;
            }
            i += 1;
            let (mv, _) = vc_at(i);
            (pd_recs, mv, rs_recs, rs_gam_ch)
        };
        // The pd claims are the element class's two (c, lc) plus one per
        // wiring GATHER; every gather value is absorbed, in proof order.
        assert_eq!(
            pd_recs.len(),
            2 + proof.wiring.gather.len(),
            "pd claims = element (c, lc) + the wiring gathers"
        );
        let pd_vals: Vec<F128> = pd_recs.iter().map(|&(_, pv)| vals_rec[pv]).collect();
        for (k, g) in proof.wiring.gather.iter().enumerate() {
            assert!(
                pd_vals.contains(g),
                "gather value {k} rides a packed-direct claim"
            );
        }

        // ---- the multipoint: the R=2 + P>0 schedule, pinned ----
        let fro = &proof.pcs_open.frobenius;
        let n_p = fro.group_values.len();
        assert!(n_p > 0, "a circuit inner carries scalar groups (P > 0)");
        {
            let mut i = mp_l[0] + 1;
            let mut n_vals = 0usize;
            while matches!(ops[i], Op::ObserveScalar) {
                n_vals += 1;
                i += 1;
            }
            assert_eq!(n_vals, 256 + n_p, "2x128 RS dual values + P group values");
            assert!(matches!(ops[i], Op::SqueezeScalar), "multipoint gamma");
            let (_, gc) = vc_at(i);
            let gamma = chals[gc];
            i += 1;
            // The located values ARE the proof's, in schedule order.
            for k in 0..n_vals {
                let want = if k < 256 {
                    fro.values[k / 128][k % 128]
                } else {
                    fro.group_values[k - 256]
                };
                assert_eq!(vals_rec[mp_val_v + k], want, "mp value {k}");
            }
            // T0 under the R=2 + P schedule folds through the rounds to the
            // anchor's claimed v — consecutive gamma powers across BOTH kinds.
            let mut t = F128::ZERO;
            let mut pw = F128::ONE;
            for k in 0..n_vals {
                t += pw * vals_rec[mp_val_v + k];
                pw *= gamma;
            }
            let mut rounds = 0usize;
            while matches!(ops[i], Op::ObserveScalar)
                && matches!(ops[i + 1], Op::ObserveScalar)
                && matches!(ops[i + 2], Op::SqueezeScalar)
            {
                let (gv, _) = vc_at(i);
                let (_, rc) = vc_at(i + 2);
                let (g1, gi) = (vals_rec[gv], vals_rec[gv + 1]);
                let r = chals[rc];
                let g0 = t + g1;
                t = g0 + (g1 + g0 + gi) * r + gi * r * r;
                i += 3;
                rounds += 1;
            }
            assert_eq!(rounds, fro.rounds.len(), "mp round count");
            assert!(
                matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
                "anchor label follows the rounds"
            );
            assert_eq!(t, fro.anchor.v, "T_m == anchor.v under the R=2+P schedule");
        }

        // ---- the published chain ordinals (GKR alpha, multipoint gamma) ----
        let ga_fin = fin_at(gkr_l[0] + 1);
        let (_, ga_c) = vc_at(gkr_l[0] + 1);
        let mut mp_i = mp_l[0] + 1;
        while matches!(ops[mp_i], Op::ObserveScalar) {
            mp_i += 1;
        }
        assert!(matches!(ops[mp_i], Op::SqueezeScalar), "mp gamma op");
        let mg_fin = fin_at(mp_i);
        let (_, mg_c) = vc_at(mp_i);
        // ROUND 2: the H(publics) region's rows — a chunk chain per 1 KiB
        // leaf of the child's public segment plus the left-fold parents.
        let n_pub_i = inner.built.witness.public.len();
        let h_rows = n_pub_i.div_ceil(4) + 2 * n_pub_i.div_ceil(64);

        // ---- the chain materials ----
        let stream = t_shape.stream_words(domain);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let trace = {
            let mut chain = FsChain::new();
            let mut at = 0usize;
            let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
            assert_eq!(
                stream.finalize_after.len(),
                fin_ops.len(),
                "finalize alignment"
            );
            for (k, &upto) in stream.finalize_after.iter().enumerate() {
                chain.absorb(&bytes[at * 16..upto * 16]);
                at = upto;
                chain.finalize(fin_ops[k].squeezed_bytes());
            }
            chain.absorb(&bytes[at * 16..]);
            chain.finish()
        };

        // ---- the open-phase walk + geometry ----
        let lig = &proof.pcs_open.inner.ligerito;
        assert_eq!(
            inner.commitment.cap, lig.initial_cap,
            "commitment IS the L0 cap"
        );
        let r_lvl = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v, piop_o, gammas_o, w_rounds, mp_o, inner_pd2, yr_v2, levels) =
            parse_open_levels(&ops, 32 * lig.initial_cap.len(), r_lvl);
        assert!(piop_o.is_some(), "a mixed tape carries the element PIOP");
        assert_eq!(
            gammas_o.len(),
            pd_recs.len(),
            "the parser and the region walk agree on the pd claims"
        );
        let (geo, native_sums) = level_geometry(&levels, &lvl_src, &chals, HashKind::Blake3);
        let b3_rows = trace.rows.len()
            + h_rows
            + geo
                .iter()
                .map(|g| (g.row_words.div_ceil(4) + g.depth) * g.q + (1usize << g.c) - 1)
                .sum::<usize>();
        let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);
        // Recursive caps are PROOF BODY — the in-circuit cap trees bind them
        // (chain + root connects, nothing checker-read); only the L0 cap —
        // the commitment — stays a statement public.
        let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
        for &p in &cap_pays[1..] {
            pub_payloads[p] = false;
        }

        // ---- the merged intake's natives (target, running, boundary) ----
        let (native_target, native_running) = {
            use flock_core::pcs::ring_switch as rs;
            use flock_core::zerocheck::univariate_skip::build_eq;
            let gs: Vec<F128> = (0..2).map(|k| chals[rs_gam_ch + k]).collect();
            let mut target = F128::ZERO;
            let mut coeffs: Vec<Vec<F128>> = Vec::new();
            for (k, &(sv, rc)) in rs_recs.iter().enumerate() {
                let shv = &vals_rec[sv..sv + 128];
                let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
                let eq = build_eq(&rdp);
                target += gs[k] * rs::inner_product(&rs::tensor_algebra_transpose(shv), &eq);
                let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
                coeffs.push(rs::linearized_coefficients(&rs::build_fold_byte_table(
                    &scaled,
                )));
            }
            // A MIXED tape's target carries the packed-direct claims too —
            // each absorbed value against its own gamma squeeze.
            for pd in &gammas_o {
                target += chals[pd.ch] * vals_rec[pd.val_v];
            }
            let mut running = target;
            for rr in &w_rounds {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let r = chals[rr.ch];
                let g0 = running + g1;
                running = g0 + (g1 + g0 + gi) * r + gi * r * r;
            }
            // The R = 2 recombination plus the P group values, against the
            // same q_eval the spine starts from.
            let mut big_v = F128::ZERO;
            for (k, cs) in coeffs.iter().enumerate() {
                for (j, &cj) in cs.iter().enumerate() {
                    if cj.is_zero() {
                        continue;
                    }
                    let mut x = fro.values[k][j];
                    for _ in 0..j {
                        x = x * x;
                    }
                    big_v += cj * x;
                }
            }
            for &v in &fro.group_values {
                big_v += v;
            }
            assert_eq!(
                running,
                vals_rec[inner_pd2.q_v] * big_v,
                "the R=2 + P merged boundary replays"
            );
            (target, running)
        };

        // ---- the spine's native quad replay ----
        let t_final_n = {
            let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
            let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
            let mut nt = chals[inner_pd2.ch] * vals_rec[inner_pd2.q_v];
            let mut nq = quad(vals_rec[start_v], vals_rec[start_v + 1], nt);
            for (li, lvl) in levels.iter().enumerate() {
                for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                    nt = evalq(nq, chals[lvl.fold_chs[j]]);
                    nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
                }
                if li < r_lvl {
                    for od in &lvl.ood {
                        let b = chals[od.beta_ch];
                        let iq = quad(
                            vals_rec[od.intro_v],
                            vals_rec[od.intro_v + 1],
                            vals_rec[od.y_v],
                        );
                        nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                        nt += b * vals_rec[od.y_v];
                    }
                    let b = chals[lvl.beta_ch];
                    let iq = quad(
                        vals_rec[lvl.intro_v],
                        vals_rec[lvl.intro_v + 1],
                        native_sums[li],
                    );
                    nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                    nt += b * native_sums[li];
                } else {
                    nt += chals[lvl.beta_ch] * native_sums[li];
                }
            }
            nt
        };

        // ---- the anchor's native endpoint ----
        let anc_end_n = {
            let mut t = vals_rec[mp_o.anchor_v];
            for rr in &mp_o.anchor_rounds {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let r = chals[rr.ch];
                let g0 = t + g1;
                t = g0 + (g1 + g0 + gi) * r + gi * r * r;
            }
            t
        };

        // ---- the element PIOP's native chain + strip sums ----
        let mut el_g0: Vec<F128> = Vec::new();
        let mut el_run_n = F128::ZERO;
        for (k, &(gv, _, ch)) in el_rec.zc_rounds.iter().enumerate() {
            let (g1, gi) = (vals_rec[gv], vals_rec[gv + 1]);
            let t = chals[el_rec.tau_ch + k];
            let rho = chals[ch];
            let g0 = (el_run_n + t * g1) * (F128::ONE + t).inv();
            el_g0.push(g0);
            el_run_n = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        assert_eq!(
            el_assert.alpha, chals[el_rec.alpha_ch],
            "the located alpha is the assertion's"
        );
        let (a_sum_n, b_sum_n) = {
            let mt = MacGate::new();
            let kappa = mt.ty.kappa();
            let eq_con =
                flock_core::zerocheck::univariate_skip::build_eq(&el_assert.r_con[..kappa]);
            // Single slot at the region start: the prefix bits are all
            // zero, so the region weight is the all-zero eq pattern.
            let w = el_assert.r_con[kappa..]
                .iter()
                .fold(F128::ONE, |acc, &x| acc * (F128::ONE + x));
            let dot = |c: &[F128]| -> F128 {
                eq_con
                    .iter()
                    .zip(c)
                    .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
            };
            (w * dot(mt.ty.a_const()), w * dot(mt.ty.b_const()))
        };

        // ---- the GKR input-check advice (masked M̂ and livê) ----
        let mu_i = built.shape.circuit.cells().mu();
        let (mid_n, live_n) = {
            let basis_i = flock_core::product_gkr::s_id_basis(mu_i);
            let mask_i = built.shape.circuit.live_mask();
            (
                mask_i.masked_id_eval(&basis_i, &gkr_rec.r_pt),
                mask_i.live_eval(&gkr_rec.r_pt),
            )
        };

        // ---- the anchor-expect geometry + its FULL native replica ----
        let m_mp2 = mp_o.rounds.len();
        assert_eq!(
            mp_o.anchor_rounds.len(),
            2 * (m_mp2 + 1),
            "sigma spans the anchor layers"
        );
        assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
        let n_log_i = union.n_log();
        let params_i = flock_core::pcs::jagged::JaggedParams::from_heights(
            &union.jagged_heights(),
            n_log_i,
            m_mp2,
        );
        let k_cols_i = params_i.k;
        let bounds_i = flock_core::pcs::jagged::assist_boundaries(&params_i);
        let n_runs = bounds_i.len();
        // A run longer than one column is ALWAYS a zero-height run, and the
        // mixed inner has INTERIOR zero runs — the per-run weight is the
        // general Σ eq over the run's columns; the LONGEST run takes the
        // char-2 complement (the eq masses sum to 1).
        let run_y0: Vec<usize> = bounds_i
            .iter()
            .scan(0usize, |y, &(_, _, len)| {
                let s = *y;
                *y += len as usize;
                Some(s)
            })
            .collect();
        let comp_ix = (0..n_runs)
            .max_by_key(|&r| bounds_i[r].2)
            .expect("at least one run");
        // The boolean PIOP's round ordinals, located with fins: the RS
        // statements sit at points made of its round challenges, and the
        // MatrixAssertion's surfaces (x_inner_rest, rr, z_skip, z_partial)
        // map onto the same walk — the merge node's connects consume them.
        let (zc_rounds_b, (zskip_ch, zskip_fin), (outer_ch_b, outer_fin_b), lc_rounds_b, zp_v) = {
            let mut i2 = zc_l[0] + 1;
            assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_skip slice");
            i2 += 1;
            assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_outer slice");
            let outer = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_ab");
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_c");
            i2 += 1;
            assert!(matches!(ops[i2], Op::SqueezeScalar), "z_skip");
            let zskip = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            let mut zc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar)
                && matches!(ops[i2 + 1], Op::ObserveScalar)
                && matches!(ops[i2 + 2], Op::SqueezeScalar)
            {
                zc_r.push((vc_at(i2 + 2).1, fin_at(i2 + 2)));
                i2 += 3;
            }
            while matches!(ops[i2], Op::ObserveScalar) {
                i2 += 1;
            }
            assert_eq!(i2, lc_l[0], "the zerocheck runs straight into the lincheck");
            i2 += 1;
            while matches!(ops[i2], Op::SqueezeScalar) {
                i2 += 1;
            }
            let mut lc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar)
                && matches!(ops[i2 + 1], Op::ObserveScalar)
                && matches!(ops[i2 + 2], Op::SqueezeScalar)
            {
                lc_r.push((vc_at(i2 + 2).1, fin_at(i2 + 2)));
                i2 += 3;
            }
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "z_partial slice");
            let (zp, _) = vc_at(i2);
            (zc_r, zskip, outer, lc_r, zp)
        };
        assert!(
            lc_rounds_b.len() <= 1 + k_cols_i,
            "lc rounds fit the col bits"
        );
        // The MatrixAssertion's surfaces map onto located ordinals — asserted
        // value-for-value so the merge node's connects consume VERIFIED wire
        // indices, not layout assumptions. The mlv rounds follow the
        // BATCH-MAJOR packing [k_skip | dim6 | rows | high col vars]:
        // round 0 binds x_inner_rest[0] (the dim-6 var), rounds 1..1+ν bind
        // x_outer (the rows — what mvp10's RS-point composition uses), and
        // the remaining rounds bind x_inner_rest[1..]. rr is the lc rounds
        // REVERSED; z_skip is the located squeeze; z_partial the located
        // slice. (Two wrong layout guesses died on these asserts before
        // this mapping — the method-note discipline earning its keep.)
        {
            let inner_b = bool_assert.x_inner_rest.len();
            assert_eq!(
                zc_rounds_b.len(),
                inner_b + n_log_i,
                "zc mlv rounds = x_inner_rest + x_outer"
            );
            for (j, &x) in bool_assert.x_inner_rest.iter().enumerate() {
                let m = if j == 0 { 0 } else { n_log_i + j };
                assert_eq!(
                    chals[zc_rounds_b[m].0],
                    x,
                    "x_inner_rest {j} is located zc round {m}"
                );
            }
            assert_eq!(lc_rounds_b.len(), bool_assert.rr.len(), "lc round count");
            for (j, &x) in bool_assert.rr.iter().enumerate() {
                assert_eq!(
                    chals[lc_rounds_b[lc_rounds_b.len() - 1 - j].0],
                    x,
                    "rr {j} is the located lc round, reversed"
                );
            }
            assert_eq!(chals[zskip_ch], bool_assert.z_skip, "z_skip located");
            assert_eq!(
                &vals_rec[zp_v..zp_v + 64],
                &bool_assert.z_partial[..],
                "z_partial on the stream"
            );
            // The element assertion's points: r_con = zc.r[ν..] (round
            // order), r_col = the lc bind order reversed.
            assert_eq!(
                el_rec.zc_rounds.len(),
                n_log_i + el_assert.r_con.len(),
                "element zc rounds = rows + r_con"
            );
            for (j, &x) in el_assert.r_con.iter().enumerate() {
                assert_eq!(
                    chals[el_rec.zc_rounds[n_log_i + j].2],
                    x,
                    "el r_con {j} is a located element zc round"
                );
            }
            assert_eq!(
                el_rec.lc_rounds.len(),
                el_assert.r_col.len(),
                "element lc round count"
            );
            for (j, &x) in el_assert.r_col.iter().enumerate() {
                assert_eq!(
                    chals[el_rec.lc_rounds[el_rec.lc_rounds.len() - 1 - j].2],
                    x,
                    "el r_col {j} is the located element lc round, reversed"
                );
            }
        }
        let x_ab_n: Vec<F128> = {
            let p = &native_claims.ab.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        let x_c_n: Vec<F128> = {
            let p = &native_claims.c.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        assert_eq!(x_ab_n.len(), 1 + n_log_i + k_cols_i, "ab point split");
        assert_eq!(x_c_n.len(), 1 + n_log_i + k_cols_i, "c point split");
        // The P scalar groups, by shared row part — the same structural
        // grouping the two-product build uses (first-occurrence order).
        let pd_pts_n: Vec<Vec<F128>> = gammas_o
            .iter()
            .map(|pd| vals_rec[pd.pt_v..pd.pt_v + pd.pt_len].to_vec())
            .collect();
        for pd in &gammas_o {
            assert_eq!(pd.pt_len, n_log_i + k_cols_i, "pd point split");
        }
        let mut groups_ix: Vec<Vec<usize>> = Vec::new();
        for (i2, pt) in pd_pts_n.iter().enumerate() {
            match groups_ix
                .iter_mut()
                .find(|g2| pd_pts_n[g2[0]][..n_log_i] == pt[..n_log_i])
            {
                Some(g2) => g2.push(i2),
                None => groups_ix.push(vec![i2]),
            }
        }
        assert_eq!(groups_ix.len(), n_p, "P scalar groups by shared row");

        // Native replica of the WHOLE anchor expect — validates the formula
        // against the accepted proof before any gate exists.
        {
            let gamma_n = chals[mp_o.gamma_ch];
            let mut gpow_n = vec![F128::ONE];
            for j in 1..257 + n_p {
                gpow_n.push(gpow_n[j - 1] * gamma_n);
            }
            let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
            let point_n: Vec<F128> = mp_o.rounds.iter().map(|rr| chals[rr.ch]).collect();
            let sig_n: Vec<F128> = mp_o.anchor_rounds.iter().map(|rr| chals[rr.ch]).collect();
            let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
            let g_at_n = {
                let mut rinv = rho_mrg_n.clone();
                let mut acc = F128::ZERO;
                for (j, &gp) in gpow_n.iter().enumerate().take(128) {
                    if j > 0 {
                        for x in rinv.iter_mut() {
                            *x = frob_inv_native(*x);
                        }
                    }
                    let mut prod = gp;
                    for (t2, &x) in point_n.iter().enumerate() {
                        prod *= F128::ONE + rinv[t2] + x;
                    }
                    acc += prod;
                }
                acc
            };
            let e_at_n = rho_mrg_n
                .iter()
                .zip(&point_n)
                .fold(F128::ONE, |a, (&r, &x)| a * (F128::ONE + r + x));
            let eqc_n: Vec<F128> = bounds_i
                .iter()
                .map(|&(t_c, t_next, _)| {
                    let mut p = F128::ONE;
                    for l in 0..=m_mp2 {
                        p *= F128::ONE + sig_n[2 * l] + bit((t_c >> l) & 1 == 1);
                        p *= F128::ONE + sig_n[2 * l + 1] + bit((t_next >> l) & 1 == 1);
                    }
                    p
                })
                .collect();
            let sparse_t = flock_core::pcs::jagged::assist_sparse_transitions();
            let dp_native = |z_row: &[F128]| -> F128 {
                let mut gdp = [F128::ZERO; 4];
                gdp[flock_core::pcs::jagged::STATE_SUCCESS] = F128::ONE;
                for layer in (0..=m_mp2).rev() {
                    let za = if layer < n_log_i {
                        z_row[layer]
                    } else {
                        F128::ZERO
                    };
                    let rb = if layer < m_mp2 { point_n[layer] } else { F128::ZERO };
                    let eq4 = flock_core::lincheck::build_eq_table(&[za, rb]);
                    let (rc, rd) = (sig_n[2 * layer], sig_n[2 * layer + 1]);
                    let e = [
                        (F128::ONE + rc) * (F128::ONE + rd),
                        rc * (F128::ONE + rd),
                        (F128::ONE + rc) * rd,
                        rc * rd,
                    ];
                    let mut prev = [F128::ZERO; 4];
                    for (cd, &ecd) in e.iter().enumerate() {
                        for (s2, slot2) in prev.iter_mut().enumerate() {
                            let (i0, o0) = sparse_t[cd][s2][0];
                            let (i1, o1) = sparse_t[cd][s2][1];
                            *slot2 += ecd * (eq4[i0] * gdp[o0] + eq4[i1] * gdp[o1]);
                        }
                    }
                    gdp = prev;
                }
                gdp[flock_core::pcs::jagged::STATE_INITIAL]
            };
            let run_weights_n = |z_col: &[F128]| -> Vec<F128> {
                let mut w_at = vec![F128::ZERO; n_runs];
                let mut tot = F128::ONE;
                for (r, &(_, _, len)) in bounds_i.iter().enumerate() {
                    if r == comp_ix {
                        continue;
                    }
                    let mut w = F128::ZERO;
                    for y in run_y0[r]..run_y0[r] + len as usize {
                        let mut s = F128::ONE;
                        for (jj, &zc2) in z_col.iter().enumerate() {
                            s *= F128::ONE + zc2 + bit((y >> jj) & 1 == 1);
                        }
                        w += s;
                    }
                    w_at[r] = w;
                    tot += w;
                }
                w_at[comp_ix] = tot;
                w_at
            };
            let expect_n = {
                let mut acc = F128::ZERO;
                for (si, xs) in [&x_ab_n, &x_c_n].iter().enumerate() {
                    let z_row = &xs[1..1 + n_log_i];
                    let run_n = run_weights_n(&xs[1 + n_log_i..]);
                    let w_n = run_n
                        .iter()
                        .zip(&eqc_n)
                        .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                    let coeff = if si == 0 { g_at_n } else { gpow_n[128] * g_at_n };
                    acc += coeff * (w_n * dp_native(z_row));
                }
                for (g_ix, members) in groups_ix.iter().enumerate() {
                    let mut run_n = vec![F128::ZERO; n_runs];
                    for &i2 in members {
                        let pd = &gammas_o[i2];
                        let gpd = chals[pd.ch];
                        let w_at = run_weights_n(&pd_pts_n[i2][n_log_i..]);
                        for r in 0..n_runs {
                            run_n[r] += gpd * w_at[r];
                        }
                    }
                    let w_n = run_n
                        .iter()
                        .zip(&eqc_n)
                        .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                    let dp = dp_native(&pd_pts_n[members[0]][..n_log_i]);
                    acc += gpow_n[256 + g_ix] * e_at_n * (w_n * dp);
                }
                acc
            };
            assert_eq!(
                expect_n, anc_end_n,
                "the R=2 + P anchor expect replays natively"
            );
        }

        // ---- the residual pairing's rotation (lane-major inners) ----
        let yr_len = proof.pcs_open.inner.ligerito.final_proof.yr.len();
        let lane_major = geo[0].row_words < geo[0].lanes;
        let w_resid: Vec<RoundRec> = if lane_major {
            let k_rot = w_rounds.len() - levels[0].fold_fins.len();
            let mut v = w_rounds[k_rot..].to_vec();
            v.extend_from_slice(&w_rounds[..k_rot]);
            v
        } else {
            w_rounds.to_vec()
        };

        ChildTape {
            inner,
            vals_rec,
            chals,
            pub_payloads,
            cap_pays,
            trace,
            stream,
            bytes,
            b3_rows,
            spread_w,
            gkr: gkr_rec,
            el: el_rec,
            start_v,
            gammas_o,
            w_rounds,
            w_resid,
            mp_o,
            inner_pd2,
            yr_v2,
            yr_len,
            levels,
            lvl_src,
            geo,
            native_sums,
            n_pd: pd_recs.len(),
            n_p,
            zc_rounds_b,
            outer_b: (outer_ch_b, outer_fin_b),
            lc_rounds_b,
            zskip_ch,
            zskip_fin,
            zp_v,
            ga_c,
            ga_fin,
            mg_c,
            mg_fin,
            bool_assert,
            el_assert,
            sigma_native,
            el_g0,
            el_run_n,
            a_sum_n,
            b_sum_n,
            native_target,
            native_running,
            t_final_n,
            anc_end_n,
            mid_n,
            live_n,
            mu_i,
            n_log_i,
            k_cols_i,
            m_mp2,
            bounds_i,
            run_y0,
            comp_ix,
            x_ab_n,
            x_c_n,
            groups_ix,
        }
    }
}

/// The gate slots a child-tape region emits into. Created ONCE by the outer
/// test and shared by every region in the builder — the mvp11 merge outer
/// instantiates two child regions (and the fold region) over the same slots,
/// so a second child adds rows, not columns. The `le`/`resid` caches fill on
/// demand during emission; cache hits require same-shape children (the keyed
/// constructor parameters must match, which the merge test asserts by
/// requiring one shared circuit).
struct ChildSlots {
    q: CollapsedSlots,
    macs: flock_core::circuit::builder::SlotId,
    zcr: flock_core::circuit::builder::SlotId,
    mrs: flock_core::circuit::builder::SlotId,
    spine: flock_core::circuit::builder::SlotId,
    alslot: flock_core::circuit::builder::SlotId,
    le: Vec<(usize, flock_core::circuit::builder::SlotId)>,
    resid: Vec<(usize, flock_core::circuit::builder::SlotId)>,
}

impl ChildSlots {
    fn new(sb: &mut ShapeBuilder, nu2: usize, spread_w: usize) -> Self {
        ChildSlots {
            q: CollapsedSlots {
                b3: sb.slot(Blake3Gate { nu: nu2 }),
                swap: sb.slot(SwapGate { nu: nu2 }),
                spread: sb.slot(BitSpreadGate {
                    ty: BitSpreadTable::new(spread_w),
                    nu: nu2,
                }),
            },
            macs: sb.slot(MacGate::new()),
            zcr: sb.slot(ZcRoundGate::new()),
            mrs: sb.slot(MergedRoundGate::new()),
            spine: sb.slot(SpineGate::new()),
            alslot: sb.slot(AssistLayerGate::new()),
            le: Vec::new(),
            resid: Vec::new(),
        }
    }

    /// Every element-class slot, for the outer prover's slot inputs.
    fn element_slot_ids(&self) -> Vec<flock_core::circuit::builder::SlotId> {
        let mut v = vec![self.macs, self.zcr, self.mrs, self.spine, self.alslot];
        v.extend(self.le.iter().map(|&(_, s)| s));
        v.extend(self.resid.iter().map(|&(_, s)| s));
        v
    }
}

/// What one emitted child region hands back: where its public block starts,
/// the walk counts the checker needs, and the assertion-emission wires the
/// mvp11 merge node CONNECTS the fold region's claim words to.
struct ChildRegion {
    pub_base: usize,
    n_query_pub: usize,
    n_tail: usize,
    /// The sigma assertion's wires: the deferred s_sigma stream word and the
    /// GKR's accumulated squeeze point.
    sig_w: Wire,
    pt_w: Vec<Wire>,
    /// The element assertion's point wires: every element zc round rho (in
    /// round order — r_con = zc.r[ν..]) and every element lc round rho (in
    /// round order — r_col is these reversed).
    el_zc_rho_w: Vec<Wire>,
    el_lc_rho_w: Vec<Wire>,
    /// The boolean MatrixAssertion's wires: the zc mlv round rhos (round
    /// order — [dim6 | x_outer | x_inner_rest]), the lc round rhos (round
    /// order — rr is these reversed), and the absorbed z_partial words.
    b_mlv_w: Vec<Wire>,
    b_lc_w: Vec<Wire>,
    b_zpartial_w: Vec<Wire>,
    /// The residual close-out's prefix slot (and width) — reusable by a
    /// caller emitting more prefix products into the same builder.
    pf: (flock_core::circuit::builder::SlotId, usize),
}

/// Emit ONE child's complete deferred-verifier region — chain, query phase,
/// wiring GKR, element PIOP, multipoint intake + anchor expect, W-rounds,
/// spine, residual, sigma emission — into `sb`, publishing exactly what
/// [`check_child_region`] walks. This is mvp10's assembly (steps 1-8),
/// extracted verbatim; the tape supplies every located ordinal and native
/// replica.
fn emit_child_region(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    ct: &ChildTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
) -> ChildRegion {
    let trace = &ct.trace;
    let stream = &ct.stream;
    let chals = &ct.chals[..];
    let levels = &ct.levels[..];
    let geo = &ct.geo[..];
    let w_rounds = &ct.w_rounds[..];
    let mp_o = &ct.mp_o;
    let inner_pd2 = &ct.inner_pd2;
    let el_rec = &ct.el;
    let r_lvl = levels.len() - 1;
    let n_p = ct.n_p;
    let m_mp2 = ct.m_mp2;
    let n_log_i = ct.n_log_i;
    let k_cols_i = ct.k_cols_i;
    let n_runs = ct.bounds_i.len();

    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match cs.le.iter().find(|(n, _)| *n == lanes) {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(LeafEvalGate::new(lanes));
                    cs.le.push((lanes, sl));
                    sl
                }
            }
        })
        .collect();
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [sb.public_input(), sb.public_input()];
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let (outs, ww) = emit_fs_chain(
        sb,
        cs.q.b3,
        iv2,
        trace,
        stream,
        &ct.bytes,
        vals,
        &mut consts,
        &ct.pub_payloads,
    );
    // ---- ROUND 2: the H(publics) region (v2 statement binding) ----
    {
        let pays = payload_words(stream);
        assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
        let dw = [
            ww[pays[4][0]].expect("digest word wired"),
            ww[pays[4][1]].expect("digest word wired"),
        ];
        emit_publics_hash(
            sb,
            cs.q,
            iv2,
            &ct.inner.built.witness.public,
            dw,
            vals,
            &mut consts,
        );
    }
    let cap_w = cap_wires(stream, &ww, &ct.cap_pays);
    let (to_publish, level_accs) = emit_query_phase(
        sb,
        cs.q,
        iv2,
        &leafeval,
        levels,
        geo,
        &ct.lvl_src,
        &trace.squeezes,
        &outs,
        chals,
        &cap_w,
        vals,
        &mut consts,
        hints,
    );
    let ga_w = outs[trace.squeezes[ct.ga_fin][0]][0];
    let mg_w = outs[trace.squeezes[ct.mg_fin][0]][0];

    // ---- the WIRING GKR in-circuit ----
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
            if vmap.len() <= vi {
                vmap.resize(vi + 1, None);
            }
            vmap[vi] = Some(wi);
        }
    }
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let macs = cs.macs;
    let zcr = cs.zcr;
    let mrs = cs.mrs;
    // The assert-zero anchor: a dedicated zero public NO gate consumes,
    // so the zero-delta outputs connected into its class add no
    // dataflow edges (connecting them to the ubiquitous `zw` creates
    // cycles — the acyclicity check draws producer→consumer edges).
    vals.push(F128::ZERO);
    let zassert = sb.public_input();

    let g = &ct.gkr;
    let alpha_w = outs[trace.squeezes[g.alpha_fin][0]][0];
    let beta_w = outs[trace.squeezes[g.beta_fin][0]][0];
    // The grand products agree: a COPY CONSTRAINT on the tops (every
    // former published-zero delta in this region is now a connect — the
    // proof itself fails on a broken identity; no public, no checker item).
    let (mut cl_w, mut cr_w) = (wv(g.top_v), wv(g.top_v + 1));
    sb.connect(cl_w, cr_w);
    let mut pt_w: Vec<Wire> = Vec::new();
    for lr in &g.layers {
        let lam_w = outs[trace.squeezes[lr.lam_fin][0]][0];
        let mut run_w = sb.gate(macs, &[cl_w, lam_w, cr_w])[0];
        let mut pt_next: Vec<Wire> = Vec::with_capacity(lr.rounds.len() + 1);
        for (t2, &(gv, rfin)) in lr.rounds.iter().enumerate() {
            let rho_w = outs[trace.squeezes[rfin][0]][0];
            vals.push(lr.g0s[t2]);
            let g0w = sb.input();
            let o = sb.gate(zcr, &[run_w, wv(gv), wv(gv + 1), pt_w[t2], rho_w, g0w, ow]);
            sb.connect(o[0], zassert);
            run_w = o[1];
            pt_next.push(rho_w);
        }
        // The layer close: run == vl0·vl1 + lambda·(vr0·vr1).
        let (vl0, vl1) = (wv(lr.v_v), wv(lr.v_v + 1));
        let (vr0, vr1) = (wv(lr.v_v + 2), wv(lr.v_v + 3));
        let pl = sb.gate(macs, &[zw, vl0, vl1])[0];
        let pr = sb.gate(macs, &[zw, vr0, vr1])[0];
        let gate_w = sb.gate(macs, &[pl, lam_w, pr])[0];
        sb.connect(gate_w, run_w);
        // The claim fold: claim' = v0 + c·(v0 + v1).
        let ck_w = outs[trace.squeezes[lr.ck_fin][0]][0];
        let sl = sb.gate(macs, &[vl0, vl1, ow])[0];
        let sr = sb.gate(macs, &[vr0, vr1, ow])[0];
        cl_w = sb.gate(macs, &[vl0, ck_w, sl])[0];
        cr_w = sb.gate(macs, &[vr0, ck_w, sr])[0];
        pt_next.push(ck_w);
        pt_w = pt_next;
    }
    assert_eq!(pt_w.len(), ct.mu_i, "the GKR point spans the inner cell space");
    // The input checks under the LIVE-IDENTITY padding: M̂(ρ) and livê(ρ)
    // as checker-validated advice publics.
    vals.push(ct.mid_n);
    let mid_w = sb.public_input();
    vals.push(ct.live_n);
    let live_w = sb.public_input();
    // The two input checks, as published-zero deltas.
    let (f_w, g_w, sig_w) = (wv(g.fgs_v), wv(g.fgs_v + 1), wv(g.fgs_v + 2));
    let l1 = sb.gate(macs, &[f_w, alpha_w, mid_w])[0];
    let l2 = sb.gate(macs, &[l1, beta_w, live_w])[0];
    let l3 = sb.gate(macs, &[l2, ow, live_w])[0];
    let l4 = sb.gate(macs, &[l3, ow, ow])[0];
    sb.connect(l4, cl_w);
    let r1 = sb.gate(macs, &[g_w, alpha_w, sig_w])[0];
    let r2 = sb.gate(macs, &[r1, beta_w, live_w])[0];
    let r3 = sb.gate(macs, &[r2, ow, live_w])[0];
    let r4 = sb.gate(macs, &[r3, ow, ow])[0];
    sb.connect(r4, cr_w);

    // ---- the MULTIPOINT intake at R = 2 AND P > 0 ----
    let mp_gamma_w = outs[trace.squeezes[mp_o.gamma_fin][0]][0];
    assert_eq!(
        mp_o.val_vs.len(),
        256 + n_p,
        "the R=2 + P schedule spans both claim kinds"
    );
    let mut t0_w = zw;
    let mut pw_w = ow;
    // The gamma-power wires are KEPT: the anchor expect consumes mp_pws[j]
    // (j < 128) for ĝ, mp_pws[128] for the second RS statement, and
    // mp_pws[256 + k] for the P group coefficients.
    let mut mp_pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp_o.val_vs.iter().enumerate() {
        t0_w = sb.gate(macs, &[t0_w, pw_w, wv(vi)])[0];
        if k + 1 < mp_o.val_vs.len() {
            pw_w = sb.gate(macs, &[zw, pw_w, mp_gamma_w])[0];
            mp_pws.push(pw_w);
        }
    }
    let mut tm_w = t0_w;
    let mut mp_rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp_o.rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_rho2_w.push(rho_w);
        tm_w = sb.gate(mrs, &[tm_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    sb.connect(tm_w, wv(mp_o.anchor_v));
    // The anchor's own rounds fold its claimed v to an endpoint, which
    // publishes and is held against the native replay; the squeezes are the
    // sigma wires the expect consumes below.
    let mut anc_w = wv(mp_o.anchor_v);
    let mut mp_sig_w: Vec<Wire> = Vec::new();
    for rr in &mp_o.anchor_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_sig_w.push(rho_w);
        anc_w = sb.gate(mrs, &[anc_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    assert_eq!(
        mp_sig_w.len(),
        2 * (m_mp2 + 1),
        "sigma spans the anchor layers"
    );

    // ---- the merged intake's W-ROUNDS ----
    // The RS target is FAMILY H — checker-validated advice; the W-rounds
    // fold it through the shared round gate, binding rho in-circuit.
    vals.push(ct.native_target);
    let tgt_w = sb.public_input();
    let mut runw = tgt_w;
    for rr in w_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        runw = sb.gate(mrs, &[runw, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }

    // ---- the LIGERITO SPINE ----
    let spine = cs.spine;
    let gpw = outs[trace.squeezes[inner_pd2.fin][0]][0];
    let tw0 = sb.gate(
        spine,
        &[zw, zw, zw, zw, zw, zw, wv(inner_pd2.q_v), gpw, zw],
    );
    let mut tsp = tw0[3];
    let st = sb.gate(
        spine,
        &[zw, zw, zw, zw, wv(ct.start_v), wv(ct.start_v + 1), tsp, ow, zw],
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = outs[trace.squeezes[lvl.fold_fins[j]][0]][0];
            let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
            tsp = ev[4];
            let bld = sb.gate(spine, &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tsp, ow, zw]);
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r_lvl {
            for od in &lvl.ood {
                let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                let f = sb.gate(
                    spine,
                    &[
                        qc,
                        qb,
                        qa,
                        tsp,
                        wv(od.intro_v),
                        wv(od.intro_v + 1),
                        wv(od.y_v),
                        bw,
                        zw,
                    ],
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            }
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = sb.gate(
                spine,
                &[
                    qc,
                    qb,
                    qa,
                    tsp,
                    wv(lvl.intro_v),
                    wv(lvl.intro_v + 1),
                    level_accs[li],
                    bw,
                    zw,
                ],
            );
            (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = sb.gate(spine, &[zw, zw, zw, tsp, zw, zw, level_accs[li], bw, zw]);
            tsp = f[3];
        }
    }
    let t_final = tsp;

    // ---- the RESIDUAL region (shared emitter) ----
    let yr_wires: Vec<Wire> = (0..ct.yr_len).map(|y| wv(ct.yr_v2 + y)).collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        &ct.w_resid,
        inner_pd2.fin,
        &yr_wires,
        &trace.squeezes,
        &outs,
        chals,
        vals,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: the residual side's inner and the spine's
    // t_r are the same statement scalar — a copy constraint, not a
    // checker item (both stay published as test cross-checks).
    sb.connect(inner_w, t_final);

    // ---- the ELEMENT PIOP rounds in-circuit ----
    // Zerocheck rounds are ZcRoundGate rows (tau slice wires as eq weights,
    // g0 advice + zero deltas); lincheck rounds are MergedRoundGate rows.
    // The entry is DERIVED: va = ea + a_sum, vb = eb + b_sum, entry =
    // va + alpha·vb — only the two constant-strip sums are advice.
    let mut el_zr = zw;
    for (k, &(gv, rfin, _)) in el_rec.zc_rounds.iter().enumerate() {
        let sqt = &trace.squeezes[el_rec.tau_fin];
        let t_w = outs[sqt[k / 4]][k % 4];
        let rho_w = outs[trace.squeezes[rfin][0]][0];
        vals.push(ct.el_g0[k]);
        let g0w = sb.input();
        let o = sb.gate(zcr, &[el_zr, wv(gv), wv(gv + 1), t_w, rho_w, g0w, ow]);
        sb.connect(o[0], zassert);
        el_zr = o[1];
    }
    let el_alpha_w = outs[trace.squeezes[el_rec.alpha_fin][0]][0];
    let ea_w = wv(el_rec.eab_v);
    let eb_w = wv(el_rec.eab_v + 1);
    vals.push(ct.a_sum_n);
    let asum_w = sb.public_input();
    vals.push(ct.b_sum_n);
    let bsum_w = sb.public_input();
    let va_w = sb.gate(macs, &[ea_w, asum_w, ow])[0];
    let vb_w = sb.gate(macs, &[eb_w, bsum_w, ow])[0];
    let mut el_lcw = sb.gate(macs, &[va_w, el_alpha_w, vb_w])[0];
    for &(gv, rfin, _) in &el_rec.lc_rounds {
        let rho_w = outs[trace.squeezes[rfin][0]][0];
        el_lcw = sb.gate(mrs, &[el_lcw, wv(gv), wv(gv + 1), rho_w])[0];
    }

    // ---- the anchor EXPECT in-circuit, at R = 2 AND P > 0 ----
    // expect = Σ_i γ^{128i}·ĝ(ρ″)·(w_i·DP_i) over the RS statements + Σ_k
    // γ^{256+k}·eq(ρ,ρ″)·(w_k·DP_k) over the P groups; claim == expect
    // publishes as a zero-delta. ĝ's inverse-Frobenius points ride as advice
    // bound by forward squaring deltas.
    use flock_core::zerocheck::univariate_skip_optimized::{
        medium_challenges_ghash, small_challenges_ghash,
    };
    let mut t_vals_b: Vec<F128> = Vec::new();
    t_vals_b.extend_from_slice(&small_challenges_ghash());
    t_vals_b.extend_from_slice(&medium_challenges_ghash());
    assert_eq!(t_vals_b.len(), 7, "the seven baked ghash weights");

    // The statements' points as (native value, wire) pairs, pinned against
    // the native claims: ab = [LAST lc round | zc mlv rounds 1..1+ν | lc
    // rounds REVERSED tail], c = the zerocheck's r_rest verbatim.
    let mlv_pw: Vec<(F128, Wire)> = ct
        .zc_rounds_b
        .iter()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let lc_pw: Vec<(F128, Wire)> = ct
        .lc_rounds_b
        .iter()
        .rev()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let extend_const = |pw: &mut Vec<(F128, Wire)>, xn: &[F128]| {
        for &cv2 in &xn[pw.len()..] {
            let w = if cv2 == F128::ZERO {
                zw
            } else {
                assert_eq!(cv2, F128::ONE, "constant point coord is a slot-prefix bit");
                ow
            };
            pw.push((cv2, w));
        }
    };
    let mut xab_pw: Vec<(F128, Wire)> = vec![lc_pw[0]];
    xab_pw.extend_from_slice(&mlv_pw[1..1 + n_log_i]);
    xab_pw.extend_from_slice(&lc_pw[1..]);
    extend_const(&mut xab_pw, &ct.x_ab_n);
    let (outer_ch_b, outer_fin_b) = ct.outer_b;
    let mut xc_pw: Vec<(F128, Wire)> = (0..ct.zc_rounds_b.len())
        .map(|k2| {
            if k2 < 7 {
                (t_vals_b[k2], cw(sb, vals, &mut consts, t_vals_b[k2]))
            } else {
                let j = k2 - 7;
                let sq2 = &trace.squeezes[outer_fin_b];
                (chals[outer_ch_b + j], outs[sq2[j / 4]][j % 4])
            }
        })
        .collect();
    extend_const(&mut xc_pw, &ct.x_c_n);
    for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&ct.x_ab_n).enumerate() {
        assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
    }
    for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&ct.x_c_n).enumerate() {
        assert_eq!(nv, xn, "c point coord {i2} is the located wire");
    }

    // The residual region's prefix slot (width pf_w) carries every chunked
    // (1 + a + b) product.
    let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
        let mut seed = ow;
        for chunk in factors.chunks(pf_w) {
            let mut g_in = vec![seed];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            seed = sb.gate(pfslot, &g_in)[0];
        }
        seed
    };
    // ĝ(ρ″): advice square-root chains for ρ^(2^-j), bound by forward
    // squaring deltas y·y + prev = 0.
    let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
    let rho_mrg_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let mut rinv_n2: Vec<F128> = rho_mrg_n.clone();
    let mut rinv_w: Vec<Wire> = rho_mrg_w.clone();
    let mut ghat = zw;
    for j in 0..128 {
        if j > 0 {
            let mut lvl_w = Vec::with_capacity(m_mp2);
            for t2 in 0..m_mp2 {
                let y = frob_inv_native(rinv_n2[t2]);
                rinv_n2[t2] = y;
                vals.push(y);
                let yw = sb.input();
                let d = sb.gate(spine, &[zw, zw, zw, rinv_w[t2], zw, zw, yw, yw, zw])[3];
                sb.connect(d, zassert);
                lvl_w.push(yw);
            }
            rinv_w = lvl_w;
        }
        let factors: Vec<(Wire, Wire)> = rinv_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        let eqj = prefix_product(sb, &factors);
        ghat = sb.gate(spine, &[zw, zw, zw, ghat, zw, zw, mp_pws[j], eqj, zw])[3];
    }
    // e_at = eq(ρ, ρ″) for the group coefficients.
    let e_at_w = {
        let factors: Vec<(Wire, Wire)> = rho_mrg_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        prefix_product(sb, &factors)
    };
    // Per-run boundary eq products at sigma (statement-independent).
    let eqc_w: Vec<Wire> = ct
        .bounds_i
        .iter()
        .map(|&(t_c, t_next, _)| {
            let mut factors = Vec::with_capacity(2 * (m_mp2 + 1));
            for l in 0..=m_mp2 {
                factors.push((mp_sig_w[2 * l], if (t_c >> l) & 1 == 1 { ow } else { zw }));
                factors.push((
                    mp_sig_w[2 * l + 1],
                    if (t_next >> l) & 1 == 1 { ow } else { zw },
                ));
            }
            prefix_product(sb, &factors)
        })
        .collect();
    // Per RS statement: run weights, the w dot, the DP, the coefficient.
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        let z_col_w: Vec<Wire> = xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect();
        let mut run_w: Vec<Wire> = vec![zw; n_runs];
        let mut tot_w = ow;
        for (r, &(_, _, len)) in ct.bounds_i.iter().enumerate() {
            if r == ct.comp_ix {
                continue;
            }
            let mut w: Option<Wire> = None;
            for y in ct.run_y0[r]..ct.run_y0[r] + len as usize {
                let factors: Vec<(Wire, Wire)> = z_col_w
                    .iter()
                    .enumerate()
                    .map(|(jj, &zc2)| (zc2, if (y >> jj) & 1 == 1 { ow } else { zw }))
                    .collect();
                let s = prefix_product(sb, &factors);
                w = Some(match w {
                    None => s,
                    Some(p) => sb.gate(spine, &[zw, zw, zw, p, zw, zw, s, ow, zw])[3],
                });
            }
            let w = w.expect("non-empty run");
            run_w[r] = w;
            tot_w = sb.gate(spine, &[zw, zw, zw, tot_w, zw, zw, w, ow, zw])[3];
        }
        run_w[ct.comp_ix] = tot_w;
        let mut w_st = zw;
        for (r, &rw) in run_w.iter().enumerate() {
            w_st = sb.gate(spine, &[zw, zw, zw, w_st, zw, zw, rw, eqc_w[r], zw])[3];
        }
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i { z_row_w[layer] } else { zw };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = if si == 0 {
            ghat
        } else {
            sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pws[128], ghat, zw])[3]
        };
        let wd = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, w_st, gdp[0], zw])[3];
        expect_w = sb.gate(spine, &[zw, zw, zw, expect_w, zw, zw, coeff, wd, zw])[3];
    }
    // Per group: γ_pd-combined run weights over the absorbed claim points,
    // coefficient γ^{256+k}·e_at.
    for (g_ix, members) in ct.groups_ix.iter().enumerate() {
        let mut run_w: Vec<Wire> = vec![zw; n_runs];
        for &i2 in members {
            let pd = &ct.gammas_o[i2];
            let gpd_w = outs[trace.squeezes[pd.fin][0]][0];
            let mut tot_w = ow;
            let mut w_at: Vec<Wire> = vec![zw; n_runs];
            for (r, &(_, _, len)) in ct.bounds_i.iter().enumerate() {
                if r == ct.comp_ix {
                    continue;
                }
                let mut w: Option<Wire> = None;
                for y in ct.run_y0[r]..ct.run_y0[r] + len as usize {
                    let factors: Vec<(Wire, Wire)> = (0..k_cols_i)
                        .map(|jj| {
                            (
                                wv(pd.pt_v + n_log_i + jj),
                                if (y >> jj) & 1 == 1 { ow } else { zw },
                            )
                        })
                        .collect();
                    let s = prefix_product(sb, &factors);
                    w = Some(match w {
                        None => s,
                        Some(p) => sb.gate(macs, &[p, s, ow])[0],
                    });
                }
                let w = w.expect("non-empty run");
                w_at[r] = w;
                tot_w = sb.gate(macs, &[tot_w, w, ow])[0];
            }
            w_at[ct.comp_ix] = tot_w;
            for r in 0..n_runs {
                run_w[r] = sb.gate(macs, &[run_w[r], gpd_w, w_at[r]])[0];
            }
        }
        let mut w_st = zw;
        for (r, &rw) in run_w.iter().enumerate() {
            w_st = sb.gate(macs, &[w_st, rw, eqc_w[r]])[0];
        }
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i {
                wv(ct.gammas_o[members[0]].pt_v + layer)
            } else {
                zw
            };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = sb.gate(macs, &[zw, mp_pws[256 + g_ix], e_at_w])[0];
        let wd = sb.gate(macs, &[zw, w_st, gdp[0]])[0];
        expect_w = sb.gate(macs, &[expect_w, coeff, wd])[0];
    }
    // The join: the anchor's folded claim equals the in-circuit expect.
    sb.connect(anc_w, expect_w);

    // Everything publishes HERE, after every public input is declared
    // (`built.public` lists entries in DECLARATION order — the recorded
    // MVP-7 gotcha). Tail order: [query phase (alphas), accs | ga, mg |
    // gkr deltas | el deltas, el zc end, el lc end | mp_delta, anc,
    // t_final, tgt, runw | resid | inner | s_sigma | rho... | sqrt deltas |
    // anchor delta].
    let pub_base = sb.public_len();
    for a_wires in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    for w in &level_accs {
        sb.publish(*w);
    }
    sb.publish(ga_w);
    sb.publish(mg_w);
    sb.publish(el_zr);
    sb.publish(el_lcw);
    sb.publish(anc_w);
    sb.publish(t_final);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(*w);
        }
    }
    sb.publish(inner_w);
    // ---- the SIGMA ASSERTION emission (route B, in-circuit) ----
    // The claim exits as bound publics: the value is the deferred s_sigma
    // stream word — the SAME wire the rhs input check just consumed — and
    // the point is the GKR's own accumulated squeeze wires.
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    // The z_skip squeeze wire, published: the boolean claims' lagrange row
    // lows derive from it, and the merge node's checker rebuilds them from
    // THIS published value (the alpha-expansion trust class — the
    // SkipNodeGate/φ8 in-circuit derivation is the recorded upgrade).
    sb.publish(outs[trace.squeezes[ct.zskip_fin][0]][0]);
    let n_tail = 2 + 2 + 4 + levels.len() * ct.yr_len + 1 + 1 + ct.mu_i + 1;
    let n_query_pub: usize =
        levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    ChildRegion {
        pub_base,
        n_query_pub,
        n_tail,
        sig_w,
        pt_w,
        el_zc_rho_w: el_rec
            .zc_rounds
            .iter()
            .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
            .collect(),
        el_lc_rho_w: el_rec
            .lc_rounds
            .iter()
            .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
            .collect(),
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: ct
            .lc_rounds_b
            .iter()
            .map(|&(_, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: (0..64).map(|i| wv(ct.zp_v + i)).collect(),
        pf: (pfslot, pf_w),
    }
}

/// Walk one emitted child region's public block and hold every published
/// value against the tape's native replicas — mvp10's checker, extracted.
/// Returns the number of public entries consumed (the region's publish
/// tail), so a multi-region caller can walk region after region.
fn check_child_region(public: &[F128], ct: &ChildTape<'_>, r: &ChildRegion) -> usize {
    let chals = &ct.chals[..];
    // The query-phase boundary: published alphas are the recorded
    // challenges and each accumulator equals the native enforced sum.
    {
        let mut at = r.pub_base;
        // The openings bind to the absorbed caps by COPY CONSTRAINT (the
        // in-circuit cap tree) — no per-query publics, no checker walk.
        for (li, lvl) in ct.levels.iter().enumerate() {
            for j in 0..lvl.a_count {
                assert_eq!(public[at + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
            }
            at += lvl.a_count;
        }
        for (li, want) in ct.native_sums.iter().enumerate() {
            assert_eq!(
                public[at + li],
                *want,
                "L{li} enforced sum matches the native replica"
            );
        }
        assert_eq!(
            at + ct.native_sums.len(),
            r.pub_base + r.n_query_pub,
            "the query publics walk consumed its whole block"
        );
    }
    let base2 = r.pub_base + r.n_query_pub;
    assert_eq!(
        public[base2],
        chals[ct.ga_c],
        "the GKR alpha derives in-circuit"
    );
    assert_eq!(
        public[base2 + 1],
        chals[ct.mg_c],
        "the multipoint gamma derives in-circuit"
    );
    // The GKR round/close/input identities, the element zc round deltas,
    // T_m == anchor.v and claim == expect are COPY CONSTRAINTS now — no
    // publics, no checker items; the proof itself carries them.
    let el_base = base2 + 2;
    assert_eq!(
        public[el_base],
        ct.el_run_n,
        "the element zc chain ends at the native running claim"
    );
    // THE INDEPENDENT CLOSE: the in-circuit lincheck chain ends exactly at
    // the native ElementAssertion's target.
    assert_eq!(
        public[el_base + 1],
        ct.el_assert.target,
        "the element lc chain ends at the native assertion's target"
    );
    let mp_base = el_base + 2;
    assert_eq!(
        public[mp_base],
        ct.anc_end_n,
        "the anchor rounds end at the native claim"
    );
    // THE LIGERITO CLOSE: the in-circuit spine reaches the native t_r.
    assert_eq!(
        public[mp_base + 1],
        ct.t_final_n,
        "the spine's final t_r matches the native replay"
    );
    // The merged intake: the advice target and the in-circuit running.
    assert_eq!(
        public[mp_base + 2],
        ct.native_target,
        "the RS target advice is the native gamma-combination"
    );
    assert_eq!(
        public[mp_base + 3],
        ct.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    // The residual region against the shared native replica — and THE
    // CLOSURE: the residual-side inner and the spine's t_r are the same
    // statement scalar, both held against published circuit outputs.
    let inner_n = check_residual_publics(
        public,
        mp_base + 4,
        &ct.levels,
        &ct.geo,
        &ct.w_resid,
        ct.inner_pd2.ch,
        &ct.vals_rec[ct.yr_v2..ct.yr_v2 + ct.yr_len],
        chals,
    );
    assert_eq!(inner_n, ct.t_final_n, "inner == t_r: the mixed statement closes");
    // The sigma assertion, as the accumulator would read it: the value and
    // the mu point coordinates, matched against the native claim.
    let sig_base = mp_base + 4 + ct.levels.len() * ct.yr_len + 1;
    assert_eq!(
        public[sig_base],
        ct.inner.proof.wiring.gkr.s_sigma_eval,
        "the emitted sigma value is the proof's deferred evaluation"
    );
    let sig_rho = &public[sig_base + 1..sig_base + 1 + ct.mu_i];
    {
        // The emitted pair IS a SigmaAssertion, rebuilt from the outer's
        // PUBLIC SEGMENT ALONE — equal to the deferred verify's own, and it
        // discharges against the inner circuit's sigma table.
        let sa = flock_core::circuit::SigmaAssertion {
            rho: sig_rho.to_vec(),
            nu: ct.inner.built.shape.circuit.cells().nu(),
            value: public[sig_base],
        };
        assert_eq!(sa.rho, ct.sigma_native.rho, "the emitted sigma point");
        assert_eq!(sa.value, ct.sigma_native.value, "the emitted sigma value");
        assert_eq!(sa.nu, ct.sigma_native.nu, "the emitted sigma split");
        assert!(
            sa.check(&ct.inner.built.shape.circuit),
            "the emitted sigma assertion discharges against the inner circuit"
        );
    }
    assert_eq!(
        public[sig_base + 1 + ct.mu_i],
        chals[ct.zskip_ch],
        "the published z_skip is the located squeeze"
    );
    r.n_query_pub + r.n_tail
}

/// **MVP-10 step 1 — the circuit-inner tape.** Phase 3's inner is a CIRCUIT
/// proof; this pins its transcript regions before any assembly — the same
/// step 1 every phase ran. The inner is a minimal MIXED circuit (one blake3
/// compression feeding a MacGate chain across the class boundary, end
/// published), proven over the circuit path and natively verified under a
/// RecordingChallenger. Pinned: the region order (boolean PIOP → element
/// PIOP → wiring GKR → merged open), the boolean zerocheck slices, the
/// wiring GKR's final (f, g, s_σ) triple on the value stream (the value the
/// σ route-B assertion carries), the packed-direct claims = element (2) +
/// the wiring GATHERS (count and every gather value absorbed), rs×2, and
/// the FIRST R=2 + P>0 multipoint schedule: T0 = Σ γ^{128i+j}·A_ij +
/// Σ γ^{256+k}·B_k folds through the rounds to T_m == anchor.v.
/// Scaffolding inner per the mvp8 precedent; the parse, the assembly and
/// the checker are the extracted [`ChildTape`] / [`emit_child_region`] /
/// [`check_child_region`] — the SAME machinery the mvp11 merge node
/// instantiates per child, so this test keeping green is what makes the
/// extraction faithful.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp10_circuit_inner_tape() {
    use flock_prover::prover::UnionElementSlotInput;

    let inner = build_mixed_inner(256, 32, 0x4D51_0001);
    let ct = ChildTape::new(&inner, DOMAIN);

    // Shape facts of THIS inner (the machinery is shared with the mvp11
    // children; these numbers are not): a mixed CIRCUIT union commits
    // `num_lanes` ACTIVE lanes — an arbitrary integer (61 here), NOT a
    // whole number of blocks, and narrower than the fold width.
    assert_eq!(ct.geo[0].row_words, 61, "the mixed inner's active lane count");
    assert_ne!(ct.geo[0].row_words % 4, 0, "not a whole number of blocks");
    assert!(
        ct.geo[0].row_words < ct.geo[0].lanes,
        "and narrower than the fold"
    );

    // ---- the outer: one child region in one builder ----
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let region = emit_child_region(&mut sb, &mut cs, &ct, &mut vals, &mut hints);
    let shape2 = sb.finish().expect("the mvp10 chain circuit builds");
    let hint_refs: Vec<&dyn std::any::Any> =
        hints.iter().map(|h| h as &dyn std::any::Any).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_child_region(&built2.public, &ct, &region);
    assert_eq!(
        region.pub_base + consumed,
        built2.public.len(),
        "the region's publics are the whole tail"
    );

    // The outer proves and verifies over the circuit path.
    let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
    let pcs2 = PcsParams {
        m: union2.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union2.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let b3_r1cs2 = blake3::build_block_r1cs(nu2);
    let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
    let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
    let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
    let spread_ty2 = BitSpreadTable::new(ct.spread_w);
    let spread_r1cs2 = spread_ty2.build_block_r1cs(nu2);
    let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
    let mut el_ord: Vec<(usize, Vec<F128>)> = cs
        .element_slot_ids()
        .into_iter()
        .map(|sl| {
            let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                SlotWitness::Element(z) => z.clone(),
                other => panic!("gkr slot produced {other:?}"),
            };
            (shape2.registry_slot(sl), z)
        })
        .collect();
    el_ord.sort_by_key(|(i, _)| *i);
    let el_inputs: Vec<UnionElementSlotInput> = el_ord
        .into_iter()
        .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
        .collect();
    let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
        (
            shape2.registry_slot(cs.q.b3),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(
                    built2.rows::<Blake3Gate>(cs.q.b3),
                    nu2,
                ),
                b3_lc2,
            ),
        ),
        (
            shape2.registry_slot(cs.q.swap),
            UnionSlotProverInput::new(
                SwapTable::generate_witness_batch_major(built2.rows::<SwapGate>(cs.q.swap), nu2),
                swap_lc2,
            ),
        ),
        (
            shape2.registry_slot(cs.q.spread),
            UnionSlotProverInput::new(
                spread_ty2.generate_witness_batch_major(
                    built2.rows::<BitSpreadGate>(cs.q.spread),
                    nu2,
                ),
                spread_lc2,
            ),
        ),
    ];
    bslots.sort_by_key(|(i, _)| *i);
    let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (shape2.registry_slot(cs.q.b3), b3_lc2),
        (shape2.registry_slot(cs.q.swap), swap_lc2),
        (shape2.registry_slot(cs.q.spread), spread_lc2),
    ];
    lco.sort_by_key(|(i, _)| *i);
    let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lco.into_iter().map(|(_, c)| c).collect();
    let mut ch2 = FsChallenger::new(DOMAIN);
    let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
        &union2,
        &shape2.circuit,
        &built2.public,
        &pcs2,
        bslots.into_iter().map(|(_, x)| x).collect(),
        el_inputs,
        &mut ch2,
    );
    let mut ch2 = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union2,
        &shape2.circuit,
        &built2.public,
        &lcs2,
        &ocommit,
        &oproof,
        &pcs2,
        &mut ch2,
    )
    .expect("the mvp10 chain circuit verifies");

    let union = UnionInstance::new(&inner.built.shape.registry, inner.built.shape.counts.clone());
    println!(
        "\nMVP-10 CIRCUIT-INNER TAPE (mixed: blake3 + mac, wired)\n  \
         inner: nu {} | dense_m {} | pd claims {} (2 element + {} gathers) | P {} | mu {}\n  \
         outer: chain b3 rows {} | nu {} | dense_m {} | mu {}\n  \
         outer carries: the chain, the QUERY PHASE (61-word leaves), the\n         \
         WIRING GKR ({} layers, identities as copy constraints), the\n         \
         element PIOP, the\n         \
         MULTIPOINT intake (R=2 and P={}), the SPINE (t_r bound), the\n         \
         RESIDUAL region (rotated lane-major pairing; inner == t_r closes),\n         \
         the ANCHOR EXPECT (RS + group statements, claim == expect closes),\n         \
         and the sigma assertion (value + {} point coords, discharges)\n  \
         proof {:.1} KiB\n",
        inner.nu,
        union.dense_m(),
        ct.n_pd,
        inner.proof.wiring.gather.len(),
        ct.n_p,
        ct.mu_i,
        ct.b3_rows,
        nu2,
        union2.dense_m(),
        shape2.circuit.cells().mu(),
        inner.proof.wiring.gkr.layers.len(),
        ct.n_p,
        ct.mu_i,
        bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// A Merkle leaf need not be a whole number of 64-byte blocks — and the
/// opening gate hashes the partial final block correctly.
///
/// This is the shape a MIXED CIRCUIT union produces: it commits `num_lanes`
/// ACTIVE lanes, `dense_words.div_ceil(2^log_dim)`, an arbitrary integer
/// (the top lanes are definitionally zero and never encoded — see
/// `ligerito`'s "high-bit-lane commit"). MVP-7 and MVP-9 never met it
/// because their inners' lane counts were powers of two by luck of
/// `dense_words`; MVP-10's realistic mixed inner opens 61-word rows.
///
/// BLAKE3 hashes 61 words = 976 bytes as one chunk of 16 blocks whose last
/// carries `b = 16`, and the compression's `b` is a free input to the gate,
/// so the only cost is one zero-padding wire. Pinned against
/// `merkle::hash_leaf` itself at every width, whole blocks and partial
/// alike.
#[test]
fn partial_block_leaves_hash_correctly() {
    for words in [1usize, 3, 4, 8, 61, 64] {
        let (depth, leaf_bytes) = (2usize, 16 * words);
        let nu = 6usize;
        let mut rng = Rng(0x_B10C_0000 ^ words as u64);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        let pos = 2usize;

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(depth),
                nu,
            }),
        };
        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&IV);
        vals.extend_from_slice(&iv_w);
        let iv = [sb.public_input(), sb.public_input()];
        let leaf = tree.leaf(pos);
        let leaf_w: Vec<Wire> = (0..words)
            .map(|w| {
                vals.push(leaf_word(leaf, 16 * w));
                sb.public_input()
            })
            .collect();
        vals.push(F128::new(pos as u64, 0));
        let idx_w = sb.public_input();
        let root = emit_opening(&mut sb, slots, iv, &leaf_w, idx_w, depth, 0, None, &mut vals);
        sb.publish(root[0]);
        sb.publish(root[1]);
        let shape = sb.finish().expect("the opening circuit builds");
        let hints: Vec<[u32; SLOT_WORDS]> = tree.siblings(pos);
        let hint_refs: Vec<&dyn std::any::Any> =
            hints.iter().map(|h| h as &dyn std::any::Any).collect();
        let built = shape.run(&vals, &hint_refs);

        // The in-circuit chunk chain reproduces `hash_leaf` on a leaf that
        // is NOT block-aligned, and the fold reaches the real root.
        let n = built.public.len();
        assert_eq!(
            [built.public[n - 2], built.public[n - 1]],
            digest_words(&hash_to_digest(&tree.root)),
            "width {words}: the opening folds to the tree root"
        );
    }
}

// ---------------------------------------------------------------------------
// MVP-11: the merge node — step 1, the sigma fold
// ---------------------------------------------------------------------------

/// One MVP-11 merge child: the mvp10-style minimal mixed inner (a blake3
/// chain feeding MacGate rows across the class boundary) proven over the
/// circuit path and verified DEFERRED. The seed varies only the witness
/// (message words), so two children share the CIRCUIT — and its digest, the
/// key the accumulator folds sigma under — while their claims land at
/// unrelated FS points, which is what a merge node actually sees.
fn mvp11_child(seed: u64) -> MixedInner {
    build_mixed_inner(128, 16, seed)
}

/// **MVP-11 step 1: the merge node's sigma fold, tape-pinned.**
///
/// The merge node arithmetises `verify_aggregate_classes` — `verify_fold`
/// replays that read NO matrix anywhere. This records the SMALLEST fold the
/// native merge-node test performs (the sigma group: 2 claims, one per
/// child) under a RecordingChallenger and pins its whole tape: the claims'
/// weights and values absorbed field-for-field, the two Convention-A
/// sumchecks' rounds and the bridge on the stream, and both endpoint
/// identities closing from LOCATED words alone — exactly the wires the
/// in-circuit replay consumes.
///
/// The fold runs under its own scaffolding domain; in the real merge node
/// ONE challenger spans bind + every per-type fold, and the claims' stream
/// words get CONNECTED to child-tape-derived wires. Both are later steps —
/// the mvp8 fixed-start precedent.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp11_sigma_fold_tape() {
    use flock_core::matrix_fold::{self, FoldMatrix, MatrixClaim, Weight};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const M11_DOMAIN: &[u8] = b"flock-mvp11-merge-fold-v0";

    // Two children, one circuit: different witnesses put the sigma claims
    // at unrelated points; the shared digest is the foldability key.
    let MixedInner {
        built: built0,
        sigma: sig0,
        ..
    } = mvp11_child(0x4D31_0001);
    let MixedInner {
        built: built1,
        sigma: sig1,
        ..
    } = mvp11_child(0x4D31_0002);
    assert_eq!(
        built0.shape.circuit.digest(),
        built1.shape.circuit.digest(),
        "the children share one circuit"
    );
    assert_ne!(sig0.rho, sig1.rho, "distinct witnesses, distinct FS points");
    let sigmas = [sig0, sig1];
    let (k_row, k_col) = (sigmas[0].nu, sigmas[0].rho.len() - sigmas[0].nu);

    // The fold, exactly as aggregate's sigma group runs it: claims in the
    // fixed order, per-claim column marginals (the k·nnz prover work,
    // native forever), prove under one challenger, verify under a
    // recording twin.
    let m_sig = flock_core::circuit::SigmaAssertion::matrix(&built0.shape.circuit);
    let claims: Vec<MatrixClaim> = sigmas.iter().map(|s| s.claim()).collect();
    let n_cols = FoldMatrix::n_cols(&m_sig);
    let combs: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| FoldMatrix::col_marginal(&m_sig, &c.row.materialize(), n_cols))
        .collect();
    let mut chp = FsChallenger::with_hash(M11_DOMAIN, HashKind::Blake3);
    let (fp, out_p) = matrix_fold::prove_fold(&m_sig, &combs, &claims, &mut chp);
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(M11_DOMAIN, HashKind::Blake3));
    let out_v =
        matrix_fold::verify_fold(&claims, &fp, &mut rec).expect("the honest sigma fold verifies");
    assert_eq!(out_p, out_v, "prover and verifier agree on the accumulator");
    assert!(
        out_v.check_direct(&m_sig),
        "the folded sigma claim discharges at the root"
    );

    // ---- the tape structure, pinned op-for-op ----
    // Everything the fold verifier touches is scalar squeezes and scalar /
    // slice observes — no PoW, no vec squeezes — so the challenge ordinal
    // IS the finalization ordinal, which is what wires the chain squeezes.
    let t_shape = rec.shape();
    let ops = t_shape.ops();
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![Op::Label(b"flock-matrix-fold-v0".to_vec())];
    for _ in 0..claims.len() {
        want.extend([
            Op::ObserveSlice(1),     // row.low — an eq weight's [1]
            Op::ObserveSlice(k_row), // row.point = rho[..nu]
            Op::ObserveSlice(1),     // col.low
            Op::ObserveSlice(k_col), // col.point = rho[nu..]
            Op::ObserveScalar,       // value
        ]);
    }
    want.extend([Op::SqueezeScalar, Op::SqueezeScalar]); // lambdas
    for _ in 0..k_col {
        want.extend([Op::ObserveScalar, Op::ObserveScalar, Op::SqueezeScalar]);
    }
    want.extend([Op::ObserveScalar, Op::ObserveScalar]); // bridge
    want.extend([Op::SqueezeScalar, Op::SqueezeScalar]); // mus
    for _ in 0..k_row {
        want.extend([Op::ObserveScalar, Op::ObserveScalar, Op::SqueezeScalar]);
    }
    want.push(Op::ObserveScalar); // the output value
    assert_eq!(ops, want.as_slice(), "the fold tape is the expected shape");

    // Value ordinals, by construction from the pinned shape — then held
    // against the proof and the claims field-for-field, so the formulas
    // below consume verified indices, not assumptions.
    let blk = k_row + k_col + 3;
    let v_cm = claims.len() * blk; // col-round messages
    let v_br = v_cm + 2 * k_col; // bridge
    let v_rm = v_br + 2; // row-round messages
    let v_out = v_rm + 2 * k_row; // output value
    assert_eq!(vals_rec.len(), v_out + 1, "nothing rides after the output");
    assert_eq!(
        chals.len(),
        4 + k_col + k_row,
        "lambdas, col rhos, mus, row rhos — all scalar squeezes"
    );
    for (k, c) in claims.iter().enumerate() {
        let base = k * blk;
        assert_eq!(vals_rec[base], F128::ONE, "claim {k}: row.low is eq's [1]");
        assert_eq!(
            &vals_rec[base + 1..base + 1 + k_row],
            &c.row.point[..],
            "claim {k}: row point on the stream"
        );
        assert_eq!(
            vals_rec[base + 1 + k_row],
            F128::ONE,
            "claim {k}: col.low is eq's [1]"
        );
        assert_eq!(
            &vals_rec[base + 2 + k_row..base + 2 + k_row + k_col],
            &c.col.point[..],
            "claim {k}: col point on the stream"
        );
        assert_eq!(
            vals_rec[base + blk - 1],
            c.value,
            "claim {k}: value on the stream"
        );
    }
    for (j, &(q1, qinf)) in fp.col_rounds.iter().enumerate() {
        assert_eq!(vals_rec[v_cm + 2 * j], q1, "col round {j} q(1)");
        assert_eq!(vals_rec[v_cm + 2 * j + 1], qinf, "col round {j} q(inf)");
    }
    assert_eq!(&vals_rec[v_br..v_br + 2], &fp.bridge[..], "the bridge");
    for (j, &(q1, qinf)) in fp.row_rounds.iter().enumerate() {
        assert_eq!(vals_rec[v_rm + 2 * j], q1, "row round {j} q(1)");
        assert_eq!(vals_rec[v_rm + 2 * j + 1], qinf, "row round {j} q(inf)");
    }
    assert_eq!(vals_rec[v_out], fp.value, "the output value on the stream");

    // ---- both endpoints, replayed from the LOCATED words alone ----
    // The in-circuit dataflow run natively first: stream words + squeezes
    // in, two zero-deltas out. Convention A: q(0) = running + q(1).
    let replay = |target: F128, base: usize, ch0: usize, n: usize| -> (F128, Vec<F128>) {
        let mut run = target;
        let mut rho = Vec::with_capacity(n);
        for j in 0..n {
            let (g1, gi) = (vals_rec[base + 2 * j], vals_rec[base + 2 * j + 1]);
            let r = chals[ch0 + j];
            let q0 = run + g1;
            run = gi * r * r + (q0 + g1 + gi) * r + q0;
            rho.push(r);
        }
        (run, rho)
    };
    // An eq weight's eval is the char-2 product Π (1 + p_j + r_j), SEEDED
    // by the absorbed low word — 1 here, but consumed from the stream so
    // the wire is bound, not assumed.
    let eq_prod = |low_v: usize, pt_base: usize, rho: &[F128]| -> F128 {
        let mut w = vals_rec[low_v];
        for (j, &r) in rho.iter().enumerate() {
            w *= F128::ONE + vals_rec[pt_base + j] + r;
        }
        w
    };
    let lam = [chals[0], chals[1]];
    let target_c = lam[0] * vals_rec[blk - 1] + lam[1] * vals_rec[2 * blk - 1];
    let (run_c, rho_col) = replay(target_c, v_cm, 2, k_col);
    let expect_c = (0..claims.len()).fold(F128::ZERO, |acc, k| {
        acc + lam[k]
            * eq_prod(k * blk + 1 + k_row, k * blk + 2 + k_row, &rho_col)
            * vals_rec[v_br + k]
    });
    assert_eq!(run_c, expect_c, "the col endpoint closes from located words");

    let mus = [chals[2 + k_col], chals[3 + k_col]];
    let target_r = mus[0] * vals_rec[v_br] + mus[1] * vals_rec[v_br + 1];
    let (run_r, rho_row) = replay(target_r, v_rm, 4 + k_col, k_row);
    let w_mu = (0..claims.len()).fold(F128::ZERO, |acc, k| {
        acc + mus[k] * eq_prod(k * blk, k * blk + 1, &rho_row)
    });
    assert_eq!(
        run_r,
        w_mu * vals_rec[v_out],
        "the row endpoint closes from located words"
    );

    // The accumulator IS (eq(rho_row), eq(rho_col), value) — the merge
    // node's public statement in miniature, every piece a located wire.
    assert_eq!(
        out_v,
        MatrixClaim {
            row: Weight::eq(rho_row),
            col: Weight::eq(rho_col),
            value: vals_rec[v_out],
        },
        "the accumulator is the located rho pair + the located value"
    );

    // ---- the in-circuit replay ----
    // The fold transcript replays through one b3 slot: λ/μ/ρ are chain
    // squeeze wires, the messages and claim coordinates are absorbed
    // stream words, the rounds are MergedRoundGate rows (the same
    // Convention-A quadratic), and the weight evals at ρ are PrefixGate
    // eq products seeded by the absorbed low words. Both endpoint
    // identities publish as zero-deltas, and the accumulator
    // (ρ_col, ρ_row, value) publishes as the merge node's statement —
    // rebuilt from the public segment alone and discharged at the root.
    let outer_stats = {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let stream = t_shape.stream_words(M11_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(
            stream.finalize_after.len(),
            fin_ops.len(),
            "finalize alignment"
        );
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();

        let b3_rows = trace.rows.len();
        // Floor the capacity at 2^7 rows: the registered Ligerito configs
        // start at m=22, and this outer is small enough that nu2=6 would
        // land at dense_m 21 — live-prefix rows make the slack ~free.
        let nu2 = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        let mut sb = ShapeBuilder::new(nu2);
        let b3s = sb.slot(Blake3Gate { nu: nu2 });
        let macs = sb.slot(MacGate::new());
        let mrs = sb.slot(MergedRoundGate::new());
        let pf_w = k_row.max(k_col).min(8);
        let pfslot = sb.slot(PrefixGate::new(pf_w));

        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [sb.public_input(), sb.public_input()];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(ops);
        let (outs, ww) = emit_fs_chain(
            &mut sb,
            b3s,
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
        );
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
        // Every finalizing op is a scalar squeeze (pinned above), so the
        // challenge ordinal addresses the chain squeeze directly.
        let chw = |fin: usize| -> Wire { outs[trace.squeezes[fin][0]][0] };
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();
        // The transcript TAIL past the last squeeze — the output value's
        // observe — is absorbed but never compressed (no later squeeze
        // flushes it; the real merge tape ends the same way), so its word
        // has no chain wire. It enters as its own input instead: the row
        // endpoint identity is what binds it, and it publishes as the
        // accumulator's value below.
        vals.push(vals_rec[v_out]);
        let val_w = sb.input();

        // seed · Π (1 + a_j + b_j) through the prefix slot, seed-chained
        // across chunks, padded factors (zw, zw) = 1.
        let prefix = |sb: &mut ShapeBuilder, seed: Wire, fs: &[(Wire, Wire)]| -> Wire {
            let mut s = seed;
            for chunk in fs.chunks(pf_w) {
                let mut g_in = vec![s];
                for (a, _) in chunk {
                    g_in.push(*a);
                }
                g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
                for (_, b) in chunk {
                    g_in.push(*b);
                }
                g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
                g_in.push(ow);
                s = sb.gate(pfslot, &g_in)[0];
            }
            s
        };

        // Col phase: target = Σ λ_k·value_k over the absorbed claim
        // values, then the rounds, then expect = Σ λ_k·colw_k·bridge_k.
        let lam_w = [chw(0), chw(1)];
        let mut run_w = zw;
        for k in 0..claims.len() {
            run_w = sb.gate(macs, &[run_w, lam_w[k], wv(k * blk + blk - 1)])[0];
        }
        let mut rho_col_w: Vec<Wire> = Vec::new();
        for j in 0..k_col {
            let r_w = chw(2 + j);
            rho_col_w.push(r_w);
            run_w = sb.gate(mrs, &[run_w, wv(v_cm + 2 * j), wv(v_cm + 2 * j + 1), r_w])[0];
        }
        let mut exp_w = zw;
        for k in 0..claims.len() {
            let fs: Vec<(Wire, Wire)> = (0..k_col)
                .map(|j| (wv(k * blk + 2 + k_row + j), rho_col_w[j]))
                .collect();
            let cw = prefix(&mut sb, wv(k * blk + 1 + k_row), &fs);
            let t = sb.gate(macs, &[zw, cw, wv(v_br + k)])[0];
            exp_w = sb.gate(macs, &[exp_w, lam_w[k], t])[0];
        }
        let delta_col = sb.gate(macs, &[run_w, exp_w, ow])[0];

        // Row phase: target = Σ μ_k·bridge_k, the rounds, and the closing
        // running == (Σ μ_k·roww_k) · value.
        let mu_w = [chw(2 + k_col), chw(3 + k_col)];
        let mut run2_w = zw;
        for k in 0..claims.len() {
            run2_w = sb.gate(macs, &[run2_w, mu_w[k], wv(v_br + k)])[0];
        }
        let mut rho_row_w: Vec<Wire> = Vec::new();
        for j in 0..k_row {
            let r_w = chw(4 + k_col + j);
            rho_row_w.push(r_w);
            run2_w = sb.gate(mrs, &[run2_w, wv(v_rm + 2 * j), wv(v_rm + 2 * j + 1), r_w])[0];
        }
        let mut wmu_w = zw;
        for k in 0..claims.len() {
            let fs: Vec<(Wire, Wire)> = (0..k_row)
                .map(|j| (wv(k * blk + 1 + j), rho_row_w[j]))
                .collect();
            let rw = prefix(&mut sb, wv(k * blk), &fs);
            wmu_w = sb.gate(macs, &[wmu_w, mu_w[k], rw])[0];
        }
        let rhs_w = sb.gate(macs, &[zw, wmu_w, val_w])[0];
        let delta_row = sb.gate(macs, &[run2_w, rhs_w, ow])[0];

        // Publishes AFTER every input is declared (the declaration-order
        // rule): the two zero-deltas, then the accumulator.
        sb.publish(delta_col);
        sb.publish(delta_row);
        for &w in &rho_col_w {
            sb.publish(w);
        }
        for &w in &rho_row_w {
            sb.publish(w);
        }
        sb.publish(val_w);

        let shape2 = sb.finish().expect("the mvp11 fold circuit builds");
        let built2 = shape2.run(&vals, &[]);

        // The checker: both endpoint deltas are zero, and the accumulator
        // rebuilt from the PUBLIC SEGMENT ALONE is the native fold output
        // — which discharges against the children's own sigma table.
        let tail = built2.public.len() - (2 + k_col + k_row + 1);
        assert_eq!(
            built2.public[tail],
            F128::ZERO,
            "the col endpoint zero-delta"
        );
        assert_eq!(
            built2.public[tail + 1],
            F128::ZERO,
            "the row endpoint zero-delta"
        );
        let rebuilt = MatrixClaim {
            row: Weight::eq(built2.public[tail + 2 + k_col..tail + 2 + k_col + k_row].to_vec()),
            col: Weight::eq(built2.public[tail + 2..tail + 2 + k_col].to_vec()),
            value: built2.public[tail + 2 + k_col + k_row],
        };
        assert_eq!(
            rebuilt, out_v,
            "the accumulator, rebuilt from the public segment alone"
        );
        assert!(rebuilt.check_direct(&m_sig), "and it discharges at the root");

        // The outer proves and verifies over the circuit path.
        let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let mut el_ord: Vec<(usize, Vec<F128>)> = [macs, mrs, pfslot]
            .into_iter()
            .map(|sl| {
                let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                    SlotWitness::Element(z) => z.clone(),
                    other => panic!("element slot produced {other:?}"),
                };
                (shape2.registry_slot(sl), z)
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
            .collect();
        let mut ch2 = FsChallenger::new(DOMAIN);
        let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &pcs2,
            vec![UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(built2.rows::<Blake3Gate>(b3s), nu2),
                b3_lc2,
            )],
            el_inputs,
            &mut ch2,
        );
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![b3_lc2];
        let mut ch2 = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &lcs2,
            &ocommit,
            &oproof,
            &pcs2,
            &mut ch2,
        )
        .expect("the mvp11 fold circuit verifies");
        (
            b3_rows,
            nu2,
            union2.dense_m(),
            shape2.circuit.cells().mu(),
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0),
        )
    };

    println!(
        "\nMVP-11 SIGMA FOLD TAPE + IN-CIRCUIT REPLAY (2 circuit children, one digest)\n  \
         child: nu {} | mu {} — fold: {} col rounds, {} row rounds, 2 bridge values\n  \
         tape: {} ops | {} stream values | {} squeezes — endpoints close from located words\n  \
         outer: chain b3 rows {} | nu {} | dense_m {} | mu {} — both deltas published zero,\n         \
         the accumulator rebuilt from the public segment discharges | proof {:.1} KiB\n",
        sigmas[0].nu,
        sigmas[0].rho.len(),
        k_col,
        k_row,
        ops.len(),
        vals_rec.len(),
        chals.len(),
        outer_stats.0,
        outer_stats.1,
        outer_stats.2,
        outer_stats.3,
        outer_stats.4 as f64 / 1024.0,
    );
}

/// One absorbed claim's stream ordinals on a fold tape: the four weight
/// slices and the value, in absorb order.
struct ClaimLoc {
    row_low_v: usize,
    row_low_n: usize,
    row_pt_v: usize,
    row_pt_n: usize,
    col_low_v: usize,
    col_low_n: usize,
    col_pt_v: usize,
    col_pt_n: usize,
    value_v: usize,
}

/// One fold group's ordinals: its claims, then the lambdas, col rounds,
/// bridge, mus, row rounds and output value.
struct FoldLoc {
    claims: Vec<ClaimLoc>,
    lam_ch0: usize,
    col_v: usize,
    col_ch0: usize,
    k_col: usize,
    bridge_v: usize,
    mu_ch0: usize,
    row_v: usize,
    row_ch0: usize,
    k_row: usize,
    out_v: usize,
}

/// (public index, fold, row side, h) of one boundary-expanded low-fold eq
/// public — checker-validated against the fold's PUBLISHED ρ coordinates.
type AlphaRec = (usize, usize, bool, usize);

/// One emitted fold group's wires: the accumulator claim (ρ_col, ρ_row,
/// value) to publish. The two endpoint identities are COPY CONSTRAINTS
/// (`connect`), not published zero-deltas — the proof itself fails on a
/// broken endpoint, and no public or checker item exists for it.
struct FoldPub {
    rho_col: Vec<Wire>,
    rho_row: Vec<Wire>,
    value: Wire,
}

/// The fold region's op tape for a claim-list set: per group, the
/// matrix-fold label, every claim's four weight slices + value, the
/// lambdas, col rounds, bridge, mus, row rounds, and the output value.
/// Width-driven, so mixed low widths and any claim count pin themselves.
fn fold_region_ops(
    fold_claims: &[Vec<flock_core::matrix_fold::MatrixClaim>],
) -> Vec<flock_core::transcript_record::TranscriptOp> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut want: Vec<Op> = Vec::new();
    for cs in fold_claims {
        want.push(Op::Label(b"flock-matrix-fold-v0".to_vec()));
        for c in cs {
            want.extend([
                Op::ObserveSlice(c.row.low.len()),
                Op::ObserveSlice(c.row.point.len()),
                Op::ObserveSlice(c.col.low.len()),
                Op::ObserveSlice(c.col.point.len()),
                Op::ObserveScalar,
            ]);
        }
        for _ in 0..cs.len() {
            want.push(Op::SqueezeScalar); // lambdas
        }
        for _ in 0..cs[0].col.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar, Op::SqueezeScalar]);
        }
        for _ in 0..cs.len() {
            want.push(Op::ObserveScalar); // bridge
        }
        for _ in 0..cs.len() {
            want.push(Op::SqueezeScalar); // mus
        }
        for _ in 0..cs[0].row.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar, Op::SqueezeScalar]);
        }
        want.push(Op::ObserveScalar); // the output value
    }
    want
}

/// Locate every fold's surfaces on the value/challenge streams (counters
/// start at 0 — the bind prefix carries only byte payloads) and pin them
/// field-for-field against the gathered claims and the `FoldProof`s.
fn locate_and_pin_folds(
    fold_claims: &[Vec<flock_core::matrix_fold::MatrixClaim>],
    fold_proofs: &[&flock_core::matrix_fold::FoldProof],
    vals_rec: &[F128],
    chals: &[F128],
) -> Vec<FoldLoc> {
    let (mut vcur, mut ccur) = (0usize, 0usize);
    let locs: Vec<FoldLoc> = fold_claims
        .iter()
        .map(|cs| {
            let claims = cs
                .iter()
                .map(|c| {
                    let l = ClaimLoc {
                        row_low_v: vcur,
                        row_low_n: c.row.low.len(),
                        row_pt_v: vcur + c.row.low.len(),
                        row_pt_n: c.row.point.len(),
                        col_low_v: vcur + c.row.low.len() + c.row.point.len(),
                        col_low_n: c.col.low.len(),
                        col_pt_v: vcur + c.row.low.len() + c.row.point.len() + c.col.low.len(),
                        col_pt_n: c.col.point.len(),
                        value_v: vcur
                            + c.row.low.len()
                            + c.row.point.len()
                            + c.col.low.len()
                            + c.col.point.len(),
                    };
                    vcur = l.value_v + 1;
                    l
                })
                .collect::<Vec<_>>();
            let (k_col, k_row) = (cs[0].col.n_vars(), cs[0].row.n_vars());
            let lam_ch0 = ccur;
            ccur += cs.len();
            let col_v = vcur;
            let col_ch0 = ccur;
            vcur += 2 * k_col;
            ccur += k_col;
            let bridge_v = vcur;
            vcur += cs.len();
            let mu_ch0 = ccur;
            ccur += cs.len();
            let row_v = vcur;
            let row_ch0 = ccur;
            vcur += 2 * k_row;
            ccur += k_row;
            let out_v = vcur;
            vcur += 1;
            FoldLoc {
                claims,
                lam_ch0,
                col_v,
                col_ch0,
                k_col,
                bridge_v,
                mu_ch0,
                row_v,
                row_ch0,
                k_row,
                out_v,
            }
        })
        .collect();
    assert_eq!(vals_rec.len(), vcur, "every stream value is accounted for");
    assert_eq!(chals.len(), ccur, "every squeeze is accounted for");
    for ((loc, cs), fp) in locs.iter().zip(fold_claims).zip(fold_proofs) {
        for (cl, c) in loc.claims.iter().zip(cs) {
            assert_eq!(
                &vals_rec[cl.row_low_v..cl.row_low_v + cl.row_low_n],
                &c.row.low[..],
                "row low on the stream"
            );
            assert_eq!(
                &vals_rec[cl.row_pt_v..cl.row_pt_v + cl.row_pt_n],
                &c.row.point[..],
                "row point on the stream"
            );
            assert_eq!(
                &vals_rec[cl.col_low_v..cl.col_low_v + cl.col_low_n],
                &c.col.low[..],
                "col low on the stream"
            );
            assert_eq!(
                &vals_rec[cl.col_pt_v..cl.col_pt_v + cl.col_pt_n],
                &c.col.point[..],
                "col point on the stream"
            );
            assert_eq!(vals_rec[cl.value_v], c.value, "claim value on the stream");
        }
        for (j, &(q1, qinf)) in fp.col_rounds.iter().enumerate() {
            assert_eq!(vals_rec[loc.col_v + 2 * j], q1, "col round q(1)");
            assert_eq!(vals_rec[loc.col_v + 2 * j + 1], qinf, "col round q(inf)");
        }
        assert_eq!(
            &vals_rec[loc.bridge_v..loc.bridge_v + loc.claims.len()],
            &fp.bridge[..],
            "the bridge on the stream"
        );
        for (j, &(q1, qinf)) in fp.row_rounds.iter().enumerate() {
            assert_eq!(vals_rec[loc.row_v + 2 * j], q1, "row round q(1)");
            assert_eq!(vals_rec[loc.row_v + 2 * j + 1], qinf, "row round q(inf)");
        }
        assert_eq!(vals_rec[loc.out_v], fp.value, "output value on the stream");
    }
    locs
}

/// Replay every fold's two endpoint identities from LOCATED words alone —
/// weights rebuilt through the verifier's own `Weight::eval`, the low fold
/// included — and return the located fold outputs (what the verifier's
/// accumulator must equal, surface for surface).
fn replay_fold_endpoints(
    locs: &[FoldLoc],
    vals_rec: &[F128],
    chals: &[F128],
) -> Vec<flock_core::matrix_fold::MatrixClaim> {
    use flock_core::matrix_fold::{MatrixClaim, Weight};
    let replay_rounds = |target: F128, base: usize, ch0: usize, n: usize| -> (F128, Vec<F128>) {
        let mut run = target;
        let mut rho = Vec::with_capacity(n);
        for j in 0..n {
            let (g1, gi) = (vals_rec[base + 2 * j], vals_rec[base + 2 * j + 1]);
            let r = chals[ch0 + j];
            let q0 = run + g1;
            run = gi * r * r + (q0 + g1 + gi) * r + q0;
            rho.push(r);
        }
        (run, rho)
    };
    locs.iter()
        .map(|loc| {
            let k = loc.claims.len();
            let lam: Vec<F128> = (0..k).map(|i| chals[loc.lam_ch0 + i]).collect();
            let target_c = loc
                .claims
                .iter()
                .zip(&lam)
                .fold(F128::ZERO, |acc, (cl, &l)| acc + l * vals_rec[cl.value_v]);
            let (run_c, rho_col) = replay_rounds(target_c, loc.col_v, loc.col_ch0, loc.k_col);
            let located = |low_v: usize, low_n: usize, pt_v: usize, pt_n: usize| -> Weight {
                Weight::low_eq(
                    vals_rec[low_v..low_v + low_n].to_vec(),
                    vals_rec[pt_v..pt_v + pt_n].to_vec(),
                )
            };
            let expect_c = loc
                .claims
                .iter()
                .zip(&lam)
                .enumerate()
                .fold(F128::ZERO, |acc, (i, (cl, &l))| {
                    let w = located(cl.col_low_v, cl.col_low_n, cl.col_pt_v, cl.col_pt_n);
                    acc + l * w.eval(&rho_col) * vals_rec[loc.bridge_v + i]
                });
            assert_eq!(run_c, expect_c, "col endpoint closes from located words");

            let mus: Vec<F128> = (0..k).map(|i| chals[loc.mu_ch0 + i]).collect();
            let target_r = (0..k)
                .zip(&mus)
                .fold(F128::ZERO, |acc, (i, &m)| acc + m * vals_rec[loc.bridge_v + i]);
            let (run_r, rho_row) = replay_rounds(target_r, loc.row_v, loc.row_ch0, loc.k_row);
            let w_mu = loc
                .claims
                .iter()
                .zip(&mus)
                .fold(F128::ZERO, |acc, (cl, &m)| {
                    let w = located(cl.row_low_v, cl.row_low_n, cl.row_pt_v, cl.row_pt_n);
                    acc + m * w.eval(&rho_row)
                });
            assert_eq!(
                run_r,
                w_mu * vals_rec[loc.out_v],
                "row endpoint closes from located words"
            );
            MatrixClaim {
                row: Weight::eq(rho_row),
                col: Weight::eq(rho_col),
                value: vals_rec[loc.out_v],
            }
        })
        .collect()
}

/// Emit the WHOLE fold region in-circuit: per group, the λ-combination of
/// the absorbed claim values, the col rounds (MergedRoundGate), the col
/// endpoint's weight evals (eq parts on the prefix slot; 64-wide lows
/// through 8 chained LeafEvalGate(8) rows with boundary-public hi-group
/// factors), then the μ side and the row endpoint — both endpoints as
/// zero-delta wires. The LAST fold's output value sits in the transcript
/// tail past the final squeeze (no chain wire) and enters as its own
/// input, bound by the row endpoint delta. Returns the per-fold publish
/// wires and the boundary-public records the checker validates.
#[allow(clippy::too_many_arguments)]
fn emit_fold_region(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    mrs: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    leslot: flock_core::circuit::builder::SlotId,
    locs: &[FoldLoc],
    sq: &[Vec<usize>],
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vmap: &[Option<usize>],
    chals: &[F128],
    vals_rec: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
) -> (Vec<FoldPub>, Vec<AlphaRec>) {
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    let chw = |fin: usize| -> Wire { outs[sq[fin][0]][0] };
    // seed · Π (1 + a_j + b_j) through the prefix slot, padded (zw, zw).
    let prefix = |sb: &mut ShapeBuilder, seed: Wire, fs: &[(Wire, Wire)]| -> Wire {
        let mut s = seed;
        for chunk in fs.chunks(pf_w) {
            let mut g_in = vec![s];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            s = sb.gate(pfslot, &g_in)[0];
        }
        s
    };
    // One weight eval at ρ: the low factor's MLE (seeded by the single
    // absorbed low word for eq weights, or folded through 8 LeafEval rows
    // for the 64-entry lows), times the eq-point prefix product over the
    // remaining coordinates.
    let mut alpha_recs: Vec<AlphaRec> = Vec::new();
    let emit_weight = |sb: &mut ShapeBuilder,
                       vals: &mut Vec<F128>,
                       recs: &mut Vec<AlphaRec>,
                       fi: usize,
                       row_side: bool,
                       low_v: usize,
                       low_n: usize,
                       pt_v: usize,
                       pt_n: usize,
                       rho_w: &[Wire],
                       rho_vals: &[F128]|
     -> Wire {
        let s = low_n.trailing_zeros() as usize;
        let seed = if low_n == 1 {
            wv(low_v)
        } else {
            assert_eq!(low_n, 64, "the lincheck low width");
            let mut acc = zw;
            for h in 0..8 {
                let mut a = F128::ONE;
                for b in 0..3 {
                    let r = rho_vals[3 + b];
                    a *= if (h >> b) & 1 == 1 { r } else { F128::ONE + r };
                }
                vals.push(a);
                // Record the PUBLIC index (not the input ordinal): with
                // other regions sharing the builder the two need not
                // coincide.
                recs.push((sb.public_len(), fi, row_side, h));
                let a_w = sb.public_input();
                let mut g_in: Vec<Wire> = (0..8).map(|j| wv(low_v + 8 * h + j)).collect();
                g_in.extend([rho_w[0], rho_w[1], rho_w[2]]);
                g_in.push(a_w);
                g_in.push(acc);
                acc = sb.gate(leslot, &g_in)[0];
            }
            acc
        };
        let fs: Vec<(Wire, Wire)> = (0..pt_n).map(|j| (wv(pt_v + j), rho_w[s + j])).collect();
        prefix(sb, seed, &fs)
    };

    let mut fold_pubs: Vec<FoldPub> = Vec::new();
    for (fi, loc) in locs.iter().enumerate() {
        let k = loc.claims.len();
        let lam_w: Vec<Wire> = (0..k).map(|i| chw(loc.lam_ch0 + i)).collect();
        let mut run_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            run_w = sb.gate(macs, &[run_w, lam_w[i], wv(cl.value_v)])[0];
        }
        let mut rho_col_w: Vec<Wire> = Vec::with_capacity(loc.k_col);
        for j in 0..loc.k_col {
            let r_w = chw(loc.col_ch0 + j);
            rho_col_w.push(r_w);
            run_w =
                sb.gate(mrs, &[run_w, wv(loc.col_v + 2 * j), wv(loc.col_v + 2 * j + 1), r_w])[0];
        }
        let rho_col_vals: Vec<F128> = (0..loc.k_col).map(|j| chals[loc.col_ch0 + j]).collect();
        let mut exp_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let w = emit_weight(
                sb,
                vals,
                &mut alpha_recs,
                fi,
                false,
                cl.col_low_v,
                cl.col_low_n,
                cl.col_pt_v,
                cl.col_pt_n,
                &rho_col_w,
                &rho_col_vals,
            );
            let t = sb.gate(macs, &[zw, w, wv(loc.bridge_v + i)])[0];
            exp_w = sb.gate(macs, &[exp_w, lam_w[i], t])[0];
        }
        // The col endpoint: running == expect, as a copy constraint.
        sb.connect(run_w, exp_w);

        let mu_w: Vec<Wire> = (0..k).map(|i| chw(loc.mu_ch0 + i)).collect();
        let mut run2_w = zw;
        for i in 0..k {
            run2_w = sb.gate(macs, &[run2_w, mu_w[i], wv(loc.bridge_v + i)])[0];
        }
        let mut rho_row_w: Vec<Wire> = Vec::with_capacity(loc.k_row);
        for j in 0..loc.k_row {
            let r_w = chw(loc.row_ch0 + j);
            rho_row_w.push(r_w);
            run2_w =
                sb.gate(mrs, &[run2_w, wv(loc.row_v + 2 * j), wv(loc.row_v + 2 * j + 1), r_w])[0];
        }
        let rho_row_vals: Vec<F128> = (0..loc.k_row).map(|j| chals[loc.row_ch0 + j]).collect();
        let mut wmu_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let w = emit_weight(
                sb,
                vals,
                &mut alpha_recs,
                fi,
                true,
                cl.row_low_v,
                cl.row_low_n,
                cl.row_pt_v,
                cl.row_pt_n,
                &rho_row_w,
                &rho_row_vals,
            );
            wmu_w = sb.gate(macs, &[wmu_w, mu_w[i], w])[0];
        }
        // The LAST fold's output value sits in the transcript tail past
        // the final squeeze — no chain wire (step 1's shape fact); it
        // enters as its own input, bound by the row endpoint delta.
        let value = if fi + 1 == locs.len() {
            vals.push(vals_rec[loc.out_v]);
            sb.input()
        } else {
            wv(loc.out_v)
        };
        // The row endpoint: running == weight·value, as a copy constraint
        // (this is also what binds the LAST fold's tail-input value).
        let rhs_w = sb.gate(macs, &[zw, wmu_w, value])[0];
        sb.connect(run2_w, rhs_w);
        fold_pubs.push(FoldPub {
            rho_col: rho_col_w,
            rho_row: rho_row_w,
            value,
        });
    }
    (fold_pubs, alpha_recs)
}

/// Walk the published fold blocks from `tail0`: both endpoint deltas zero
/// per fold, the accumulator claims rebuilt from the PUBLIC SEGMENT alone,
/// and every boundary-expanded low-fold eq public validated against the
/// PUBLISHED ρ coordinates. Returns the rebuilt claims for the caller's
/// accumulator reassembly.
fn check_fold_publics(
    public: &[F128],
    tail0: usize,
    locs: &[FoldLoc],
    alpha_recs: &[AlphaRec],
) -> Vec<flock_core::matrix_fold::MatrixClaim> {
    use flock_core::matrix_fold::{MatrixClaim, Weight};
    let mut p = tail0;
    let mut rebuilt: Vec<MatrixClaim> = Vec::new();
    for loc in locs {
        let rho_col = public[p..p + loc.k_col].to_vec();
        let rho_row = public[p + loc.k_col..p + loc.k_col + loc.k_row].to_vec();
        let value = public[p + loc.k_col + loc.k_row];
        rebuilt.push(MatrixClaim {
            row: Weight::eq(rho_row),
            col: Weight::eq(rho_col),
            value,
        });
        p += 1 + loc.k_col + loc.k_row;
    }
    for &(idx, fi, row_side, h) in alpha_recs {
        let base: usize = tail0
            + locs[..fi]
                .iter()
                .map(|l| 1 + l.k_col + l.k_row)
                .sum::<usize>();
        let rho = if row_side {
            &public[base + locs[fi].k_col..base + locs[fi].k_col + locs[fi].k_row]
        } else {
            &public[base..base + locs[fi].k_col]
        };
        let mut e = F128::ONE;
        for b in 0..3 {
            let r = rho[3 + b];
            e *= if (h >> b) & 1 == 1 { r } else { F128::ONE + r };
        }
        assert_eq!(
            public[idx],
            e,
            "boundary-expanded low-fold eq public (fold {fi}, h {h})"
        );
    }
    rebuilt
}

/// **MVP-11 step 2: the FULL fold region of a merge node, tape-pinned.**
///
/// `verify_aggregate_classes` for the two mixed children — bind, the
/// boolean folds (A and B for the blake3 type), the element folds (A and B
/// for the MacGate type — the first end-to-end exercise of the aggregate's
/// ELEMENT group, `gather_element`/`el_folds`/`discharge_element` included),
/// and the sigma fold, all under ONE challenger. The whole tape is pinned
/// op-for-op against the gathered claims' own shapes, every fold's surfaces
/// are held against the `AggregateProof` field-for-field, and every fold's
/// two endpoint identities close from LOCATED words alone — including the
/// boolean claims' LENGTH-64 LOW factors (the lagrange row weights and the
/// `z_partial` column weights), the one weight shape the sigma fold never
/// exercised. The five outputs reassemble the verifier's `Accumulator`
/// exactly, and all three groups discharge.
///
/// **Step 3 adds the CHILD-TAPE regions and CONNECTS them**: the same
/// outer circuit also carries each child's complete deferred verifier
/// (mvp10's assembly via the extracted [`emit_child_region`], instantiated
/// twice over shared slots, each checked by [`check_child_region`] against
/// its own native replicas), and the fold's absorbed claim surfaces are
/// copy-constrained to the child regions' assertion-emission wires: the
/// sigma claims fully (value = each child's deferred s_sigma stream word,
/// point = its GKR squeezes), the element and boolean claims' points
/// (chain squeeze wires) and the boolean z_partial lows (absorbed child
/// words). The matrix-eval values and the lagrange row lows stay the
/// boundary pattern — published, held by the checker against the
/// children's own assertions and against each child's PUBLISHED z_skip
/// (the SkipNodeGate/φ8 in-circuit derivation is the recorded upgrade).
/// The merge node's statement and its children's statements live in ONE
/// proof, and the fold no longer folds free inputs.
///
/// **PRIORS > 0 — the 4→1 shape.** Two more children's leaf folds become
/// PRIOR accumulators, so every fold group folds [inherited, inherited,
/// fresh, fresh] — the real merge arity, where each child arrives with its
/// subtree's accumulator. The bind byte is the prior COUNT; a boolean fold
/// group now MIXES low widths (inherited claims are pure eq, low [1]).
/// The inherited surfaces' lows bind to the constant 1 in-circuit; their
/// points and values publish and are checker-held against the priors' own
/// accumulators — the wire-to-wire connection arrives when merge outers
/// stack, since a prior's surface is exactly what a previous merge outer
/// publishes as its accumulator claim.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp11_merge_fold_region() {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const M11_MERGE_DOMAIN: &[u8] = b"flock-mvp11-merge-node-v0";

    let c0 = mvp11_child(0x4D32_0001);
    let c1 = mvp11_child(0x4D32_0002);
    // Two MORE children whose leaf folds become the PRIOR accumulators —
    // the 4→1 shape: in the real tree every child arrives with the
    // accumulator of its own subtree, so each fold group folds
    // [inherited, inherited, fresh, fresh].
    let c2 = mvp11_child(0x4D32_0003);
    let c3 = mvp11_child(0x4D32_0004);
    let (built0, built1) = (&c0.built, &c1.built);
    assert_eq!(
        built0.shape.circuit.digest(),
        built1.shape.circuit.digest(),
        "the children share one circuit"
    );
    assert_eq!(
        built0.shape.circuit.digest(),
        c2.built.shape.circuit.digest(),
        "the prior children share it too"
    );
    assert_eq!(
        built0.shape.circuit.digest(),
        c3.built.shape.circuit.digest(),
        "all four children, one circuit"
    );
    let registry = &built0.shape.registry;
    assert_eq!(registry.num_boolean(), 1, "one boolean type (blake3)");
    assert_eq!(registry.element_types().len(), 1, "one element type (mac)");
    let union0 = UnionInstance::new(registry, built0.shape.counts.clone());
    let union1 = UnionInstance::new(&built1.shape.registry, built1.shape.counts.clone());
    let bool_asserts = [
        c0.work.boolean.clone().expect("child 0 boolean work"),
        c1.work.boolean.clone().expect("child 1 boolean work"),
    ];
    let el_asserts = [
        (&union0, c0.work.element.clone().expect("child 0 element work")),
        (&union1, c1.work.element.clone().expect("child 1 element work")),
    ];
    let sigmas = [c0.sigma.clone(), c1.sigma.clone()];

    // The native merge: prove + verify the aggregate under one challenger
    // each, then discharge all three groups — matrix work read only here
    // (the fold prover) and in the discharges (the root), never by the
    // verifier this test arithmetises.
    let blake_r1cs = blake3::build_block_r1cs(7);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let el_ty = registry.element_types()[0]
        .element_type()
        .expect("the element slot's table");
    let el_mats = [(el_ty.a_0(), el_ty.b_0())];
    let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];

    // ---- the PRIOR accumulators: children 2 and 3's leaf folds ----
    // Each is an honest single-child aggregate (its own challenger, no
    // priors) — the accumulator a real child carries up from its subtree.
    // Prove + verify each and require agreement, as a parent would.
    const M11_LEAF_DOMAIN: &[u8] = b"flock-mvp11-leaf-fold-v0";
    let union2 = UnionInstance::new(&c2.built.shape.registry, c2.built.shape.counts.clone());
    let union3 = UnionInstance::new(&c3.built.shape.registry, c3.built.shape.counts.clone());
    let leaf_fold = |c: &MixedInner, union: &UnionInstance<'_>| -> aggregate::Accumulator {
        let ba = [c.work.boolean.clone().expect("leaf boolean work")];
        let ea = [(union, c.work.element.clone().expect("leaf element work"))];
        let sg = [c.sigma.clone()];
        let mut ch = FsChallenger::with_hash(M11_LEAF_DOMAIN, HashKind::Blake3);
        let (lp, la) = aggregate::prove_aggregate_classes(
            registry,
            &mats,
            &circs,
            &ba,
            &el_mats,
            &ea,
            Some((&built0.shape.circuit, &sg)),
            &[],
            &mut ch,
        )
        .expect("the leaf fold proves");
        let mut ch = FsChallenger::with_hash(M11_LEAF_DOMAIN, HashKind::Blake3);
        let lv = aggregate::verify_aggregate_classes(
            registry,
            &ba,
            &ea,
            Some((&built0.shape.circuit, &sg)),
            &[],
            &lp,
            &mut ch,
        )
        .expect("the leaf fold verifies");
        assert_eq!(la, lv, "leaf prover and verifier accumulators agree");
        lv
    };
    let acc_a = leaf_fold(&c2, &union2);
    let acc_b = leaf_fold(&c3, &union3);
    let priors = [&acc_a, &acc_b];
    let n_priors = priors.len();

    let mut chp = FsChallenger::with_hash(M11_MERGE_DOMAIN, HashKind::Blake3);
    let (agg, acc_p) = aggregate::prove_aggregate_classes(
        registry,
        &mats,
        &circs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        Some((&built0.shape.circuit, &sigmas)),
        &priors,
        &mut chp,
    )
    .expect("the merge-node fold proves");
    let mut rec =
        RecordingChallenger::new(FsChallenger::with_hash(M11_MERGE_DOMAIN, HashKind::Blake3));
    let acc_v = aggregate::verify_aggregate_classes(
        registry,
        &bool_asserts,
        &el_asserts,
        Some((&built0.shape.circuit, &sigmas)),
        &priors,
        &agg,
        &mut rec,
    )
    .expect("the merge-node fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_element(&el_mats),
        "the element group discharges"
    );
    assert!(
        acc_v.discharge_sigma(&built0.shape.circuit),
        "the sigma group discharges"
    );

    // The five folds' claim lists, in the verifier's own gather order:
    // the PRIORS' accumulator claims first (prior order), then one fresh
    // claim per child (child order) — [inherited, inherited, fresh,
    // fresh], the 4→1 shape. Built from the assertions' pub claim
    // constructors and the priors' own surfaces — the same data the
    // verifier gathers.
    let bc: Vec<_> = bool_asserts.iter().map(|a| a.claims(registry)).collect();
    let ec: Vec<_> = el_asserts.iter().map(|(u, a)| a.claims(u)).collect();
    let sig_a = acc_a.sigma.as_ref().expect("prior A carries sigma");
    let sig_b = acc_b.sigma.as_ref().expect("prior B carries sigma");
    let fold_claims: Vec<Vec<MatrixClaim>> = vec![
        vec![
            acc_a.per_type[0].0.clone(),
            acc_b.per_type[0].0.clone(),
            bc[0][0].0.clone(),
            bc[1][0].0.clone(),
        ],
        vec![
            acc_a.per_type[0].1.clone(),
            acc_b.per_type[0].1.clone(),
            bc[0][0].1.clone(),
            bc[1][0].1.clone(),
        ],
        vec![
            acc_a.per_element[0].0.clone(),
            acc_b.per_element[0].0.clone(),
            ec[0][0].0.clone(),
            ec[1][0].0.clone(),
        ],
        vec![
            acc_a.per_element[0].1.clone(),
            acc_b.per_element[0].1.clone(),
            ec[0][0].1.clone(),
            ec[1][0].1.clone(),
        ],
        vec![
            sig_a.1.clone(),
            sig_b.1.clone(),
            sigmas[0].claim(),
            sigmas[1].claim(),
        ],
    ];
    let fold_proofs: Vec<&FoldProof> = vec![
        &agg.folds[0].0,
        &agg.folds[0].1,
        &agg.el_folds[0].0,
        &agg.el_folds[0].1,
        agg.sigma_fold.as_ref().expect("the sigma fold rides along"),
    ];
    // The fresh boolean weights carry the length-64 lows; the INHERITED
    // claims are accumulator outputs — pure eq, low [1] — so a boolean
    // fold group now MIXES low widths (the shape priors > 0 adds). Element
    // and sigma are pure eq throughout. Pin what the machinery depends on.
    assert_eq!(fold_claims[0][0].row.low.len(), 1, "inherited claims are eq");
    assert_eq!(fold_claims[0][0].col.low.len(), 1, "inherited claims are eq");
    assert_eq!(fold_claims[0][n_priors].row.low.len(), 64, "lagrange low");
    assert_eq!(fold_claims[0][n_priors].col.low.len(), 64, "z_partial low");
    assert_eq!(fold_claims[2][0].row.low.len(), 1, "element claims are eq");
    assert_eq!(fold_claims[4][0].row.low.len(), 1, "sigma claims are eq");
    // Every claim in a group folds over the same variable counts —
    // inherited eq points span exactly what the fresh low⊗eq weights do.
    for cs in &fold_claims {
        for c in cs {
            assert_eq!(c.row.n_vars(), cs[0].row.n_vars(), "row vars agree");
            assert_eq!(c.col.n_vars(), cs[0].col.n_vars(), "col vars agree");
        }
    }

    // ---- the tape structure, pinned op-for-op ----
    // bind = label + registry digest + prior count, then the five folds in
    // aggregate order. Every finalizing op is a scalar squeeze, so the
    // challenge ordinal is the finalization ordinal — same as step 1.
    let t_shape = rec.shape();
    let ops = t_shape.ops();
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    want.extend(fold_region_ops(&fold_claims));
    assert_eq!(ops, want.as_slice(), "the merge tape is the expected shape");
    assert_eq!(rec.payloads()[0], registry.digest(), "bind: registry digest");
    assert_eq!(
        rec.payloads()[1],
        vec![n_priors as u8],
        "bind: the prior COUNT byte"
    );

    // ---- locate every fold's surfaces, and pin them field-for-field ----
    let locs = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);

    // ---- every fold's endpoints, replayed from LOCATED words alone ----
    // Weights are REBUILT from located stream words and evaluated through
    // the verifier's own `Weight::eval` — the low fold included, which is
    // what the boolean folds add over step 1's pure eq products.
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);

    // The five located outputs ARE the verifier's accumulator — the merge
    // node's public statement, reassembled surface by surface.
    assert_eq!(outs[0], acc_v.per_type[0].0, "boolean A accumulator");
    assert_eq!(outs[1], acc_v.per_type[0].1, "boolean B accumulator");
    assert_eq!(outs[2], acc_v.per_element[0].0, "element A accumulator");
    assert_eq!(outs[3], acc_v.per_element[0].1, "element B accumulator");
    let (sig_digest, sig_claim) = acc_v.sigma.as_ref().expect("sigma accumulated");
    assert_eq!(outs[4], *sig_claim, "sigma accumulator");
    assert_eq!(*sig_digest, built0.shape.circuit.digest(), "sigma key");

    // ---- the child tapes: each child's verification, recorded + parsed ----
    // ChildTape::new runs each child's RECORDING verify and re-asserts
    // mvp10's whole tape map on it; the regions below are mvp10's assembly
    // instantiated twice over shared slots.
    let t0 = ChildTape::new(&c0, DOMAIN);
    let t1 = ChildTape::new(&c1, DOMAIN);

    // ---- the in-circuit replay: TWO CHILD-TAPE REGIONS + the fold region,
    // in ONE outer circuit ----
    // Each child region is the complete deferred verifier of its child
    // (mvp10's assembly via the shared emitter). The fold region then
    // replays bind + all five folds on one b3 slot; every λ/μ/ρ is a chain
    // squeeze wire and every message/claim coordinate an absorbed stream
    // word. The rounds ride MergedRoundGate, the eq-point parts of the
    // weight evals ride PrefixGate, and the boolean claims' LENGTH-64 LOWS
    // fold through 8 chained LeafEvalGate(8) rows each — the group-
    // expansion factors eq(ρ[3..6], h) enter as boundary publics, checker-
    // validated against the PUBLISHED ρ coordinates (the alpha-expansion
    // trust class, mvp7's query-phase precedent). Ten endpoint zero-deltas
    // publish, and the five accumulator claims publish as the merge node's
    // statement — rebuilt from the public segment alone and discharged.
    let outer_stats = {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let stream = t_shape.stream_words(M11_MERGE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(
            stream.finalize_after.len(),
            fin_ops.len(),
            "finalize alignment"
        );
        assert_eq!(fin_ops.len(), chals.len(), "every finalizer is a scalar squeeze");
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();

        // The b3 slot carries all three chains (child0, child1, fold) plus
        // the children's query-phase openings — size the row capacity once.
        let b3_rows = t0.b3_rows + t1.b3_rows + trace.rows.len();
        let nu2 = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        let mut sb = ShapeBuilder::new(nu2);
        let mut cs = ChildSlots::new(&mut sb, nu2, t0.spread_w.max(t1.spread_w));
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let r0 = emit_child_region(&mut sb, &mut cs, &t0, &mut vals, &mut hints);
        let r1 = emit_child_region(&mut sb, &mut cs, &t1, &mut vals, &mut hints);
        // The fold region rides the SAME slots the child regions created:
        // rows, not columns.
        let b3s = cs.q.b3;
        let macs = cs.macs;
        let mrs = cs.mrs;
        let (pfslot, pf_w) = r0.pf;
        let leslot = cs
            .le
            .iter()
            .find(|&&(n, _)| n == 8)
            .map(|&(_, s)| s)
            .expect("the child regions created the 8-lane leaf-eval slot");

        let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [sb.public_input(), sb.public_input()];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(ops);
        let (chain_outs, ww) = emit_fs_chain(
            &mut sb,
            b3s,
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
        );
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();

        let (fold_pubs, alpha_recs) = emit_fold_region(
            &mut sb,
            macs,
            mrs,
            pfslot,
            pf_w,
            leslot,
            &locs,
            &trace.squeezes,
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );
        // ---- STEP 3's CONNECTS: the fold's absorbed claim surfaces ARE
        // the child regions' assertion-emission wires ----
        // Every surface the child region carries as a WIRE is now
        // copy-constrained to the fold's absorbed stream word: the sigma
        // claims fully (value = the child's own deferred s_sigma stream
        // word, point = its GKR squeezes), the element and boolean claims'
        // POINTS (the children's chain squeeze wires), and the boolean
        // claims' z_partial lows (absorbed child words, word for word).
        // What has no child wire stays the boundary pattern: the
        // matrix-eval VALUES (deferred proof fields — published below and
        // held against the children's own assertions) and the lagrange row
        // lows (published below; the checker rebuilds them from each
        // child's PUBLISHED z_skip — the SkipNodeGate/φ8 in-circuit
        // derivation is the recorded upgrade).
        let tapes = [&t0, &t1];
        let regions = [&r0, &r1];
        for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
            // Native pre-asserts (the method-note discipline): every wire
            // mapping below is first checked value-for-value against the
            // verifier's own assertion data.
            let nu_c = tk.sigma_native.nu;
            assert_eq!(
                &fold_claims[4][n_priors + k].row.point[..],
                &tk.sigma_native.rho[..nu_c],
                "sigma row point is the child's rho[..nu]"
            );
            assert_eq!(
                &fold_claims[4][n_priors + k].col.point[..],
                &tk.sigma_native.rho[nu_c..],
                "sigma col point is the child's rho[nu..]"
            );
            assert_eq!(
                fold_claims[4][n_priors + k].value, tk.sigma_native.value,
                "sigma value is the child's deferred evaluation"
            );
            let kappa = fold_claims[2][n_priors + k].row.point.len();
            assert_eq!(
                &fold_claims[2][n_priors + k].row.point[..],
                &tk.el_assert.r_con[..kappa],
                "element row point is r_con's head"
            );
            assert_eq!(
                &fold_claims[2][n_priors + k].col.point[..],
                &tk.el_assert.r_col[..kappa],
                "element col point is r_col's head"
            );
            assert_eq!(fold_claims[2][n_priors + k].value, tk.el_assert.evals[0].0);
            assert_eq!(fold_claims[3][n_priors + k].value, tk.el_assert.evals[0].1);
            let inner_b = fold_claims[0][n_priors + k].row.point.len();
            assert_eq!(
                &fold_claims[0][n_priors + k].row.point[..],
                &tk.bool_assert.x_inner_rest[..inner_b],
                "boolean row point is x_inner_rest's head"
            );
            assert_eq!(
                &fold_claims[0][n_priors + k].col.point[..],
                &tk.bool_assert.rr[..inner_b],
                "boolean col point is rr's head"
            );
            assert_eq!(
                &fold_claims[0][n_priors + k].col.low[..],
                &tk.bool_assert.z_partial[..],
                "boolean col low is z_partial"
            );
            assert_eq!(fold_claims[0][n_priors + k].value, tk.bool_assert.evals[0].0);
            assert_eq!(fold_claims[1][n_priors + k].value, tk.bool_assert.evals[0].1);

            // sigma: fully wire-to-wire (the eq lows are the constant 1).
            let cl = &locs[4].claims[n_priors + k];
            sb.connect(wv(cl.row_low_v), ow);
            sb.connect(wv(cl.col_low_v), ow);
            for j in 0..cl.row_pt_n {
                sb.connect(wv(cl.row_pt_v + j), rk.pt_w[j]);
            }
            for j in 0..cl.col_pt_n {
                sb.connect(wv(cl.col_pt_v + j), rk.pt_w[cl.row_pt_n + j]);
            }
            sb.connect(wv(cl.value_v), rk.sig_w);
            // element A/B: r_con = zc.r[ν..] (round order), r_col = the lc
            // rounds REVERSED — both chains' squeeze wires.
            for fi in [2, 3] {
                let cl = &locs[fi].claims[n_priors + k];
                sb.connect(wv(cl.row_low_v), ow);
                sb.connect(wv(cl.col_low_v), ow);
                for j in 0..cl.row_pt_n {
                    sb.connect(wv(cl.row_pt_v + j), rk.el_zc_rho_w[tk.n_log_i + j]);
                }
                let n_lc = rk.el_lc_rho_w.len();
                for j in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + j), rk.el_lc_rho_w[n_lc - 1 - j]);
                }
            }
            // boolean A/B: x_inner_rest follows the batch-major packing
            // (round 0 = the dim-6 var, rounds 1..1+ν = x_outer, the rest
            // = x_inner_rest[1..]), rr = the lc rounds REVERSED, and the
            // z_partial lows are the child's absorbed words.
            for fi in [0, 1] {
                let cl = &locs[fi].claims[n_priors + k];
                for j in 0..cl.row_pt_n {
                    let m = if j == 0 { 0 } else { tk.n_log_i + j };
                    sb.connect(wv(cl.row_pt_v + j), rk.b_mlv_w[m]);
                }
                let n_lc = rk.b_lc_w.len();
                for j in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + j), rk.b_lc_w[n_lc - 1 - j]);
                }
                for j in 0..cl.col_low_n {
                    sb.connect(wv(cl.col_low_v + j), rk.b_zpartial_w[j]);
                }
            }
            // The two boolean folds absorb ONE claim surface twice: fold
            // B's lagrange lows are the same words as fold A's — connected,
            // so the published copies below bind both.
            for j in 0..locs[0].claims[n_priors + k].row_low_n {
                sb.connect(
                    wv(locs[1].claims[n_priors + k].row_low_v + j),
                    wv(locs[0].claims[n_priors + k].row_low_v + j),
                );
            }
        }
        // The INHERITED claims (the priors' accumulator surfaces): pure
        // eq, so the low words bind to the constant 1 in-circuit; the
        // points and values publish below and the checker holds them
        // against the priors' own accumulators. The wire-to-wire
        // connection arrives when merge outers STACK — a prior's surface
        // is exactly what a previous merge outer PUBLISHES as its
        // accumulator claim (the step-2 publics).
        for p in 0..n_priors {
            for loc in &locs {
                let cl = &loc.claims[p];
                sb.connect(wv(cl.row_low_v), ow);
                sb.connect(wv(cl.col_low_v), ow);
            }
        }

        // Publishes AFTER every input is declared: per fold, the two
        // zero-deltas then the accumulator claim (ρ_col, ρ_row, value).
        let fold_pub_base = sb.public_len();
        for fp in &fold_pubs {
            for &w in &fp.rho_col {
                sb.publish(w);
            }
            for &w in &fp.rho_row {
                sb.publish(w);
            }
            sb.publish(fp.value);
        }
        // The value-binding publics: the fold's own absorbed words,
        // published so the checker can hold them against the children's
        // assertions (the matrix-eval values are deferred proof fields with
        // no child wire yet) and against each child's published z_skip
        // (the lagrange lows).
        for k in 0..2 {
            for j in 0..locs[0].claims[n_priors + k].row_low_n {
                sb.publish(wv(locs[0].claims[n_priors + k].row_low_v + j));
            }
            sb.publish(wv(locs[0].claims[n_priors + k].value_v));
            sb.publish(wv(locs[1].claims[n_priors + k].value_v));
            sb.publish(wv(locs[2].claims[n_priors + k].value_v));
            sb.publish(wv(locs[3].claims[n_priors + k].value_v));
        }
        // The inherited surfaces: per prior, per fold, the row and col
        // points then the value — published for the checker's walk against
        // the priors' accumulators.
        for p in 0..n_priors {
            for loc in &locs {
                let cl = &loc.claims[p];
                for j in 0..cl.row_pt_n {
                    sb.publish(wv(cl.row_pt_v + j));
                }
                for j in 0..cl.col_pt_n {
                    sb.publish(wv(cl.col_pt_v + j));
                }
                sb.publish(wv(cl.value_v));
            }
        }

        let shape2 = sb.finish().expect("the mvp11 merge circuit builds");
        let hint_refs: Vec<&dyn std::any::Any> =
            hints.iter().map(|h| h as &dyn std::any::Any).collect();
        let built2 = shape2.run(&vals, &hint_refs);

        // The two child regions' checker walks — the SAME helper mvp10 runs,
        // so each child's whole deferred-verifier statement (query phase,
        // GKR, element PIOP, multipoint, spine, residual, sigma emission +
        // discharge) is held against its own native replicas here too.
        let consumed0 = check_child_region(&built2.public, &t0, &r0);
        let consumed1 = check_child_region(&built2.public, &t1, &r1);
        assert!(
            r0.pub_base + consumed0 <= r1.pub_base && r1.pub_base + consumed1 <= fold_pub_base,
            "the three regions' public blocks are disjoint and ordered"
        );

        // The fold checker: walk the five published fold blocks — deltas
        // zero, claims rebuilt — then validate every boundary-expanded eq
        // public against the PUBLISHED ρ coordinates, reassemble the
        // Accumulator from the public segment alone, and discharge all
        // three groups.
        let tail_len: usize = locs.iter().map(|l| 1 + l.k_col + l.k_row).sum();
        let tail0 = fold_pub_base;
        let rebuilt = check_fold_publics(&built2.public, tail0, &locs, &alpha_recs);
        for (r, o) in rebuilt.iter().zip(&outs) {
            assert_eq!(r, o, "published fold output == located native output");
        }
        let acc_pub = aggregate::Accumulator {
            registry_digest: registry.digest(),
            per_type: vec![(rebuilt[0].clone(), rebuilt[1].clone())],
            per_element: vec![(rebuilt[2].clone(), rebuilt[3].clone())],
            sigma: Some((built0.shape.circuit.digest(), rebuilt[4].clone())),
        };
        assert_eq!(
            acc_pub, acc_v,
            "the Accumulator, reassembled from the public segment alone"
        );
        assert!(
            acc_pub.discharge(&mats)
                && acc_pub.discharge_element(&el_mats)
                && acc_pub.discharge_sigma(&built0.shape.circuit),
            "the public-segment accumulator discharges all three groups"
        );
        // The value-binding publics past the fold blocks: per child, the
        // boolean claims' 64 lagrange lows — rebuilt from the CHILD's own
        // published z_skip, closing the one surface the connects left to
        // the checker tier — then the four matrix-eval values held against
        // the children's assertions.
        {
            use flock_core::zerocheck::K_SKIP;
            use flock_core::zerocheck::multilinear::lagrange_weights_naive;
            let mut q = tail0 + tail_len;
            for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
                let zskip_pub =
                    built2.public[rk.pub_base + rk.n_query_pub + rk.n_tail - 1];
                let lam = lagrange_weights_naive(K_SKIP, zskip_pub);
                for (j, &l) in lam.iter().enumerate() {
                    assert_eq!(
                        built2.public[q + j],
                        l,
                        "child {k}: lagrange low {j} rebuilds from the published z_skip"
                    );
                }
                q += 64;
                assert_eq!(
                    built2.public[q],
                    tk.bool_assert.evals[0].0,
                    "child {k}: boolean A eval"
                );
                assert_eq!(
                    built2.public[q + 1],
                    tk.bool_assert.evals[0].1,
                    "child {k}: boolean B eval"
                );
                assert_eq!(
                    built2.public[q + 2],
                    tk.el_assert.evals[0].0,
                    "child {k}: element A eval"
                );
                assert_eq!(
                    built2.public[q + 3],
                    tk.el_assert.evals[0].1,
                    "child {k}: element B eval"
                );
                q += 4;
            }
            // The inherited surfaces against the priors' own accumulators
            // (fold_claims[fi][p] IS the prior's claim, cloned from
            // acc_a/acc_b above): row point, col point, value per fold.
            for p in 0..n_priors {
                for (fi, cs) in fold_claims.iter().enumerate() {
                    let want = &cs[p];
                    for (j, &x) in want.row.point.iter().enumerate() {
                        assert_eq!(
                            built2.public[q + j],
                            x,
                            "prior {p} fold {fi}: row coord {j}"
                        );
                    }
                    q += want.row.point.len();
                    for (j, &x) in want.col.point.iter().enumerate() {
                        assert_eq!(
                            built2.public[q + j],
                            x,
                            "prior {p} fold {fi}: col coord {j}"
                        );
                    }
                    q += want.col.point.len();
                    assert_eq!(
                        built2.public[q],
                        want.value,
                        "prior {p} fold {fi}: value"
                    );
                    q += 1;
                }
            }
            assert_eq!(
                q,
                built2.public.len(),
                "the value-binding publics are the very tail"
            );
        }

        // The outer proves and verifies over the circuit path.
        let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_ty2 = BitSpreadTable::new(t0.spread_w.max(t1.spread_w));
        let spread_r1cs2 = spread_ty2.build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        let mut el_ord: Vec<(usize, Vec<F128>)> = cs
            .element_slot_ids()
            .into_iter()
            .map(|sl| {
                let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                    SlotWitness::Element(z) => z.clone(),
                    other => panic!("element slot produced {other:?}"),
                };
                (shape2.registry_slot(sl), z)
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
            .collect();
        let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape2.registry_slot(cs.q.b3),
                UnionSlotProverInput::new(
                    blake3::generate_witness_batch_major_partial(
                        built2.rows::<Blake3Gate>(cs.q.b3),
                        nu2,
                    ),
                    b3_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.swap),
                UnionSlotProverInput::new(
                    SwapTable::generate_witness_batch_major(
                        built2.rows::<SwapGate>(cs.q.swap),
                        nu2,
                    ),
                    swap_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.spread),
                UnionSlotProverInput::new(
                    spread_ty2.generate_witness_batch_major(
                        built2.rows::<BitSpreadGate>(cs.q.spread),
                        nu2,
                    ),
                    spread_lc2,
                ),
            ),
        ];
        bslots.sort_by_key(|(i, _)| *i);
        let mut ch2 = FsChallenger::new(DOMAIN);
        let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &pcs2,
            bslots.into_iter().map(|(_, x)| x).collect(),
            el_inputs,
            &mut ch2,
        );
        let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (shape2.registry_slot(cs.q.b3), b3_lc2),
            (shape2.registry_slot(cs.q.swap), swap_lc2),
            (shape2.registry_slot(cs.q.spread), spread_lc2),
        ];
        lco.sort_by_key(|(i, _)| *i);
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lco.into_iter().map(|(_, c)| c).collect();
        let mut ch2 = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &lcs2,
            &ocommit,
            &oproof,
            &pcs2,
            &mut ch2,
        )
        .expect("the mvp11 merge circuit verifies");
        (
            b3_rows,
            nu2,
            union2.dense_m(),
            shape2.circuit.cells().mu(),
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0),
        )
    };

    println!(
        "\nMVP-11 MERGE NODE (2 CHILD-TAPE REGIONS + bind + 5 folds at 4->1, ONE outer)\n  \
         children: 2x the mvp10 assembly over SHARED slots (each the complete\n         \
         deferred verifier of its child, b3 rows {} + {})\n  \
         PRIORS: 2 leaf-fold accumulators — every fold group folds [inherited,\n         \
         inherited, fresh, fresh]; inherited eq lows bind to 1 in-circuit,\n         \
         points + values published and checker-held vs the priors' accumulators\n  \
         CONNECTED: every fresh claim's points bound to child chain wires (sigma\n         \
         fully, incl. its value = the child's s_sigma stream word; z_partial\n         \
         lows word-for-word); eval values + lagrange lows published, checker-\n         \
         held vs the children's assertions + each child's published z_skip\n  \
         folds: blake3 A/B ({} col + {} row rounds each, low-64 weights via LeafEval\n         \
         chains), mac A/B ({}+{} rounds, pure eq — FIRST element-group exercise),\n         \
         sigma ({}+{})\n  \
         fold tape: {} ops | {} stream values | {} squeezes — all 10 endpoints close\n  \
         from located words AND as published zero-deltas; the Accumulator reassembles\n  \
         from the public segment alone and discharges all three groups\n  \
         outer: total b3 rows {} | nu {} | dense_m {} | mu {} | proof {:.1} KiB\n",
        t0.b3_rows,
        t1.b3_rows,
        locs[0].k_col,
        locs[0].k_row,
        locs[2].k_col,
        locs[2].k_row,
        locs[4].k_col,
        locs[4].k_row,
        ops.len(),
        vals_rec.len(),
        chals.len(),
        outer_stats.0,
        outer_stats.1,
        outer_stats.2,
        outer_stats.3,
        outer_stats.4 as f64 / 1024.0,
    );
}

/// **MVP-11 at the REAL registry — the ~35-fold "swap-children" scale.**
///
/// The merge children are the real recursion node (the leaf outer,
/// [`build_leaf_outer`]): its registry carries the full gate-type census,
/// so `verify_aggregate_classes` runs one A/B fold pair per boolean type
/// and per element type, plus sigma — the merge node's fold region at its
/// first honest size. The whole tape pins through the WIDTH-DRIVEN helpers
/// unchanged, every endpoint closes from located words, and the region
/// replays IN-CIRCUIT with zero new machinery — the scale is rows, not
/// types. The accumulator (every group's A/B claims + sigma) reassembles
/// from the public segment alone and discharges against the real node's
/// own matrices and sigma table.
///
/// Scaffolding notes: the two children are the SAME leaf outer verified
/// deferred twice — `build_leaf_outer` is deterministic, so a second build
/// would yield byte-identical artifacts; claim distinctness adds nothing
/// at this step (the 4→1 test covers mixed inherited/fresh claims), the
/// registry scale is the content. The child-tape regions at this scale are
/// the composition phase, not this test.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp11_swap_children_fold_scale() {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const M11_SCALE_DOMAIN: &[u8] = b"flock-mvp11-merge-scale-v0";

    let lo = build_leaf_outer();
    let union_i = UnionInstance::new(&lo.shape.registry, lo.shape.counts.clone());
    let registry = &lo.shape.registry;
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (lo.b3_slot, lo.b3_r1cs.csc_lincheck_circuit()),
        (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
        (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.iter().map(|&(_, c)| c).collect();

    // The two children: the real node's deferred verify, run twice.
    let deferred = || {
        let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        let (_, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
            &union_i,
            &lo.shape.circuit,
            &lo.public,
            &lcs,
            &lo.commitment,
            &lo.proof,
            &lo.pcs,
            &mut ch,
        )
        .expect("the deferred verify accepts the real node");
        (
            work.boolean.expect("the real node's boolean matrix work"),
            work.element.expect("the real node's element matrix work"),
            sigma,
        )
    };
    let (ba0, ea0, sg0) = deferred();
    let (ba1, ea1, sg1) = deferred();
    let bool_asserts = [ba0, ba1];
    let el_asserts = [(&union_i, ea0), (&union_i, ea1)];
    let sigmas = [sg0, sg1];

    // The matrices + lincheck circuits per BOOLEAN type in registry order
    // (via the slot indices), and per ELEMENT type from the registry's own
    // table entries — everything the fold prover and the discharges read.
    let mut mats_ord = vec![
        (lo.b3_slot, (&lo.b3_r1cs.a_0, &lo.b3_r1cs.b_0)),
        (lo.swap_slot, (&lo.swap_r1cs.a_0, &lo.swap_r1cs.b_0)),
        (lo.spread_slot, (&lo.spread_r1cs.a_0, &lo.spread_r1cs.b_0)),
    ];
    mats_ord.sort_by_key(|&(i, _)| i);
    let mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
    let el_types: Vec<_> = registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<_> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();
    let n_bool = registry.num_boolean();
    let n_el = el_mats.len();
    assert_eq!(n_bool, 3, "the real node's boolean census: b3, swap, spread");
    assert!(n_el > 5, "the real node carries the element gate census");

    // The native fold: prove + record-verify + discharge all three groups.
    let mut chp = FsChallenger::with_hash(M11_SCALE_DOMAIN, HashKind::Blake3);
    let (agg, acc_p) = aggregate::prove_aggregate_classes(
        registry,
        &mats,
        &lcs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        Some((&lo.shape.circuit, &sigmas)),
        &[],
        &mut chp,
    )
    .expect("the scale fold proves");
    let mut rec =
        RecordingChallenger::new(FsChallenger::with_hash(M11_SCALE_DOMAIN, HashKind::Blake3));
    let acc_v = aggregate::verify_aggregate_classes(
        registry,
        &bool_asserts,
        &el_asserts,
        Some((&lo.shape.circuit, &sigmas)),
        &[],
        &agg,
        &mut rec,
    )
    .expect("the scale fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_element(&el_mats),
        "the element group discharges"
    );
    assert!(
        acc_v.discharge_sigma(&lo.shape.circuit),
        "the sigma group discharges"
    );

    // The fold groups in aggregate order: per boolean type A then B, per
    // element type A then B, then sigma.
    let bc: Vec<_> = bool_asserts.iter().map(|a| a.claims(registry)).collect();
    let ec: Vec<_> = el_asserts.iter().map(|(u, a)| a.claims(u)).collect();
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::new();
    for t in 0..n_bool {
        fold_claims.push(vec![bc[0][t].0.clone(), bc[1][t].0.clone()]);
        fold_claims.push(vec![bc[0][t].1.clone(), bc[1][t].1.clone()]);
    }
    for t in 0..n_el {
        fold_claims.push(vec![ec[0][t].0.clone(), ec[1][t].0.clone()]);
        fold_claims.push(vec![ec[0][t].1.clone(), ec[1][t].1.clone()]);
    }
    fold_claims.push(vec![sigmas[0].claim(), sigmas[1].claim()]);
    let mut fold_proofs: Vec<&FoldProof> = Vec::new();
    for t in 0..n_bool {
        fold_proofs.push(&agg.folds[t].0);
        fold_proofs.push(&agg.folds[t].1);
    }
    for t in 0..n_el {
        fold_proofs.push(&agg.el_folds[t].0);
        fold_proofs.push(&agg.el_folds[t].1);
    }
    fold_proofs.push(agg.sigma_fold.as_ref().expect("the sigma fold rides along"));
    let n_folds = fold_claims.len();
    let total_rounds: usize = fold_claims
        .iter()
        .map(|cs| cs[0].col.n_vars() + cs[0].row.n_vars())
        .sum();

    // ---- the tape, pinned through the width-driven helpers ----
    let t_shape = rec.shape();
    let ops = t_shape.ops();
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    want.extend(fold_region_ops(&fold_claims));
    assert_eq!(ops, want.as_slice(), "the scale tape is the expected shape");
    assert_eq!(rec.payloads()[0], registry.digest(), "bind: registry digest");
    assert_eq!(rec.payloads()[1], vec![0u8], "bind: prior count 0");
    let locs = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    // The located outputs ARE the verifier's accumulator, group for group.
    for t in 0..n_bool {
        assert_eq!(outs[2 * t], acc_v.per_type[t].0, "boolean type {t} A");
        assert_eq!(outs[2 * t + 1], acc_v.per_type[t].1, "boolean type {t} B");
    }
    for t in 0..n_el {
        assert_eq!(
            outs[2 * n_bool + 2 * t],
            acc_v.per_element[t].0,
            "element type {t} A"
        );
        assert_eq!(
            outs[2 * n_bool + 2 * t + 1],
            acc_v.per_element[t].1,
            "element type {t} B"
        );
    }
    let (sig_digest, sig_claim) = acc_v.sigma.as_ref().expect("sigma accumulated");
    assert_eq!(outs[n_folds - 1], *sig_claim, "sigma accumulator");
    assert_eq!(*sig_digest, lo.shape.circuit.digest(), "sigma key");

    // ---- the in-circuit replay: the whole ~35-fold region ----
    let outer_stats = {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let stream = t_shape.stream_words(M11_SCALE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(
            stream.finalize_after.len(),
            fin_ops.len(),
            "finalize alignment"
        );
        assert_eq!(fin_ops.len(), chals.len(), "every finalizer is a scalar squeeze");
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();

        let b3_rows = trace.rows.len();
        let nu2 = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        let mut sb = ShapeBuilder::new(nu2);
        let b3s = sb.slot(Blake3Gate { nu: nu2 });
        let macs = sb.slot(MacGate::new());
        let mrs = sb.slot(MergedRoundGate::new());
        let pf_w = 8usize;
        let pfslot = sb.slot(PrefixGate::new(pf_w));
        let leslot = sb.slot(LeafEvalGate::new(8));

        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [sb.public_input(), sb.public_input()];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(ops);
        let (chain_outs, ww) = emit_fs_chain(
            &mut sb,
            b3s,
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
        );
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();
        let (fold_pubs, alpha_recs) = emit_fold_region(
            &mut sb,
            macs,
            mrs,
            pfslot,
            pf_w,
            leslot,
            &locs,
            &trace.squeezes,
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );
        let fold_pub_base = sb.public_len();
        for fp in &fold_pubs {
            for &w in &fp.rho_col {
                sb.publish(w);
            }
            for &w in &fp.rho_row {
                sb.publish(w);
            }
            sb.publish(fp.value);
        }

        let shape2 = sb.finish().expect("the scale fold circuit builds");
        let built2 = shape2.run(&vals, &[]);

        let rebuilt = check_fold_publics(&built2.public, fold_pub_base, &locs, &alpha_recs);
        let tail_len: usize = locs.iter().map(|l| 1 + l.k_col + l.k_row).sum();
        assert_eq!(
            fold_pub_base + tail_len,
            built2.public.len(),
            "the fold publics are the tail"
        );
        let acc_pub = aggregate::Accumulator {
            registry_digest: registry.digest(),
            per_type: (0..n_bool)
                .map(|t| (rebuilt[2 * t].clone(), rebuilt[2 * t + 1].clone()))
                .collect(),
            per_element: (0..n_el)
                .map(|t| {
                    (
                        rebuilt[2 * n_bool + 2 * t].clone(),
                        rebuilt[2 * n_bool + 2 * t + 1].clone(),
                    )
                })
                .collect(),
            sigma: Some((lo.shape.circuit.digest(), rebuilt[n_folds - 1].clone())),
        };
        assert_eq!(
            acc_pub, acc_v,
            "the Accumulator, reassembled from the public segment alone"
        );
        assert!(
            acc_pub.discharge(&mats)
                && acc_pub.discharge_element(&el_mats)
                && acc_pub.discharge_sigma(&lo.shape.circuit),
            "the public-segment accumulator discharges all three groups"
        );

        // The outer proves and verifies over the circuit path.
        let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let mut el_ord: Vec<(usize, Vec<F128>)> = [macs, mrs, pfslot, leslot]
            .into_iter()
            .map(|sl| {
                let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                    SlotWitness::Element(z) => z.clone(),
                    other => panic!("element slot produced {other:?}"),
                };
                (shape2.registry_slot(sl), z)
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
            .collect();
        let mut ch2 = FsChallenger::new(DOMAIN);
        let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &pcs2,
            vec![UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(built2.rows::<Blake3Gate>(b3s), nu2),
                b3_lc2,
            )],
            el_inputs,
            &mut ch2,
        );
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![b3_lc2];
        let mut ch2 = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &lcs2,
            &ocommit,
            &oproof,
            &pcs2,
            &mut ch2,
        )
        .expect("the scale fold circuit verifies");
        (
            b3_rows,
            nu2,
            union2.dense_m(),
            shape2.circuit.cells().mu(),
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0),
        )
    };

    println!(
        "\nMVP-11 SCALE — the fold region at the REAL registry ({} folds, IN-CIRCUIT)\n  \
         children: the real recursion node (leaf outer, dense_m {} / mu {}) x2\n  \
         groups: {} boolean types (A/B, 64-wide lagrange/z_partial lows) + {} element\n         \
         types (A/B, pure eq) + sigma | {} sumcheck rounds total\n  \
         tape: {} ops | {} stream values | {} squeezes — every endpoint closes from\n  \
         located words AND as a published zero-delta; the {}-group Accumulator\n  \
         reassembles from the public segment alone and discharges all three groups\n  \
         outer: chain b3 rows {} | nu {} | dense_m {} | mu {} | proof {:.1} KiB\n",
        n_folds,
        lo.pcs.m,
        lo.shape.circuit.cells().mu(),
        n_bool,
        n_el,
        total_rounds,
        ops.len(),
        vals_rec.len(),
        chals.len(),
        n_folds,
        outer_stats.0,
        outer_stats.1,
        outer_stats.2,
        outer_stats.3,
        outer_stats.4 as f64 / 1024.0,
    );
}

/// **THE 2→1 RECURSION NODE.** Two DISTINCT real recursion nodes (seeded
/// leaf outers — one circuit, unrelated FS points) go in; ONE proof comes
/// out, carrying everything a parent needs:
///
/// - TWO REAL CHILD-TAPE REGIONS — each child's complete deferred verifier
///   (the swap assembly via [`emit_real_child_region`]) over SHARED slots,
/// - the FOLD REGION at the real registry (~35 folds via the width-driven
///   helpers), and
/// - THE CONNECTS: every fold claim's surfaces are copy-constrained to the
///   child regions' own assertion-emission wires — points to chain
///   squeezes, z_partial lows word-for-word, and (richer than the minimal
///   children) the matrix/element EVAL VALUES to the children's bound
///   advice publics. The lagrange row lows stay the boundary pattern:
///   published once per child, rebuilt by the checker from that child's
///   PUBLISHED z_skip.
///
/// The accumulator reassembles from the public segment alone, equals the
/// native verifier's, and discharges all three groups against the node
/// circuit's own matrices and sigma table. This outer IS the merge node:
/// its proof attests both children's verification AND the fold that
/// combined their claims. (It is not yet SELF-similar — normalization is
/// deliberately out of scope.)
/// Build a 2→1 RECURSION NODE over two children and return its artifacts
/// AS A [`LeafOuter`] (plus its output accumulator): the node's proof is
/// BLAKE3/BLAKE3-recursable and shaped exactly like a child input, so the
/// builder composes with ITSELF — `build_node_outer(&n0, &n1)` is the
/// level-2 node consuming its own outputs. The children must share one
/// circuit digest (the foldability key); their claims land at unrelated FS
/// points. Every tape pin, connect, and checker walk of the 2→1 milestone
/// lives inside — the builder IS the test.
fn build_node_outer(
    lo0: &LeafOuter,
    lo1: &LeafOuter,
) -> (LeafOuter, flock_core::aggregate::Accumulator) {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const M11_NODE_DOMAIN: &[u8] = b"flock-mvp11-two-to-one-v0";

    assert_eq!(
        lo0.shape.circuit.digest(),
        lo1.shape.circuit.digest(),
        "two children, ONE node circuit"
    );
    let registry = &lo0.shape.registry;
    let union0 = UnionInstance::new(registry, lo0.shape.counts.clone());
    let union1 = UnionInstance::new(&lo1.shape.registry, lo1.shape.counts.clone());
    let rt0 = RealTape::new(&lo0, DOMAIN);
    let rt1 = RealTape::new(&lo1, DOMAIN);
    assert_ne!(
        rt0.sigma_native.rho, rt1.sigma_native.rho,
        "distinct witnesses, distinct FS points"
    );

    // The matrices + lincheck circuits, registry order (lo0's copies —
    // one circuit, one registry).
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (lo0.b3_slot, lo0.b3_r1cs.csc_lincheck_circuit()),
        (lo0.swap_slot, lo0.swap_r1cs.csc_lincheck_circuit()),
        (lo0.spread_slot, lo0.spread_r1cs.csc_lincheck_circuit()),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.iter().map(|&(_, c)| c).collect();
    let mut mats_ord = vec![
        (lo0.b3_slot, (&lo0.b3_r1cs.a_0, &lo0.b3_r1cs.b_0)),
        (lo0.swap_slot, (&lo0.swap_r1cs.a_0, &lo0.swap_r1cs.b_0)),
        (lo0.spread_slot, (&lo0.spread_r1cs.a_0, &lo0.spread_r1cs.b_0)),
    ];
    mats_ord.sort_by_key(|&(i, _)| i);
    let mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
    let el_types: Vec<_> = registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<_> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();
    let n_bool = registry.num_boolean();
    let n_el = el_mats.len();

    // The native merge fold over the two children's assertions.
    let bool_asserts = [rt0.mat_assert.clone(), rt1.mat_assert.clone()];
    let el_asserts = [
        (&union0, rt0.el_assert.clone()),
        (&union1, rt1.el_assert.clone()),
    ];
    let sigmas = [rt0.sigma_native.clone(), rt1.sigma_native.clone()];
    let mut chp = FsChallenger::with_hash(M11_NODE_DOMAIN, HashKind::Blake3);
    let (agg, acc_p) = aggregate::prove_aggregate_classes(
        registry,
        &mats,
        &lcs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        Some((&lo0.shape.circuit, &sigmas)),
        &[],
        &mut chp,
    )
    .expect("the node fold proves");
    let mut rec =
        RecordingChallenger::new(FsChallenger::with_hash(M11_NODE_DOMAIN, HashKind::Blake3));
    let acc_v = aggregate::verify_aggregate_classes(
        registry,
        &bool_asserts,
        &el_asserts,
        Some((&lo0.shape.circuit, &sigmas)),
        &[],
        &agg,
        &mut rec,
    )
    .expect("the node fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_element(&el_mats),
        "the element group discharges"
    );
    assert!(
        acc_v.discharge_sigma(&lo0.shape.circuit),
        "the sigma group discharges"
    );

    // The fold groups in aggregate order, from the CHILDREN'S OWN
    // assertion data (the same constructors the verifier gathers with).
    let bc = [
        rt0.mat_assert.claims(registry),
        rt1.mat_assert.claims(registry),
    ];
    let ec = [rt0.el_assert.claims(&union0), rt1.el_assert.claims(&union1)];
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::new();
    for t in 0..n_bool {
        fold_claims.push(vec![bc[0][t].0.clone(), bc[1][t].0.clone()]);
        fold_claims.push(vec![bc[0][t].1.clone(), bc[1][t].1.clone()]);
    }
    for t in 0..n_el {
        fold_claims.push(vec![ec[0][t].0.clone(), ec[1][t].0.clone()]);
        fold_claims.push(vec![ec[0][t].1.clone(), ec[1][t].1.clone()]);
    }
    fold_claims.push(vec![sigmas[0].claim(), sigmas[1].claim()]);
    let mut fold_proofs: Vec<&FoldProof> = Vec::new();
    for t in 0..n_bool {
        fold_proofs.push(&agg.folds[t].0);
        fold_proofs.push(&agg.folds[t].1);
    }
    for t in 0..n_el {
        fold_proofs.push(&agg.el_folds[t].0);
        fold_proofs.push(&agg.el_folds[t].1);
    }
    fold_proofs.push(agg.sigma_fold.as_ref().expect("the sigma fold rides along"));
    let n_folds = fold_claims.len();

    // ---- the fold tape, pinned through the width-driven helpers ----
    let t_shape = rec.shape();
    let ops = t_shape.ops();
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    want.extend(fold_region_ops(&fold_claims));
    assert_eq!(ops, want.as_slice(), "the node tape is the expected shape");
    assert_eq!(rec.payloads()[0], registry.digest(), "bind: registry digest");
    assert_eq!(rec.payloads()[1], vec![0u8], "bind: prior count 0");
    let locs = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    for t in 0..n_bool {
        assert_eq!(outs[2 * t], acc_v.per_type[t].0, "boolean type {t} A");
        assert_eq!(outs[2 * t + 1], acc_v.per_type[t].1, "boolean type {t} B");
    }
    for t in 0..n_el {
        assert_eq!(
            outs[2 * n_bool + 2 * t],
            acc_v.per_element[t].0,
            "element type {t} A"
        );
        assert_eq!(
            outs[2 * n_bool + 2 * t + 1],
            acc_v.per_element[t].1,
            "element type {t} B"
        );
    }
    let (sig_digest, sig_claim) = acc_v.sigma.as_ref().expect("sigma accumulated");
    assert_eq!(outs[n_folds - 1], *sig_claim, "sigma accumulator");
    assert_eq!(*sig_digest, lo0.shape.circuit.digest(), "sigma key");

    // ---- ONE outer: two REAL child regions + the fold region ----
    let outer_stats = {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let stream = t_shape.stream_words(M11_NODE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(
            stream.finalize_after.len(),
            fin_ops.len(),
            "finalize alignment"
        );
        assert_eq!(fin_ops.len(), chals.len(), "every finalizer is a scalar squeeze");
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();

        let b3_rows = rt0.b3_rows + rt1.b3_rows + trace.rows.len();
        let nu2 = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        let mut sb = ShapeBuilder::new(nu2);
        let mut cs = ChildSlots::new(&mut sb, nu2, rt0.spread_w.max(rt1.spread_w));
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let r0 = emit_real_child_region(&mut sb, &mut cs, &rt0, &mut vals, &mut hints);
        let r1 = emit_real_child_region(&mut sb, &mut cs, &rt1, &mut vals, &mut hints);
        // The fold region rides the children's slots: rows, not columns.
        let (pfslot, pf_w) = r0.pf;
        let leslot = cs
            .le
            .iter()
            .find(|&&(n, _)| n == 8)
            .map(|&(_, s)| s)
            .expect("the child regions created the 8-lane leaf-eval slot");
        let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [sb.public_input(), sb.public_input()];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(t_shape.ops());
        let (chain_outs, ww) = emit_fs_chain(
            &mut sb,
            cs.q.b3,
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
        );
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();
        let (fold_pubs, alpha_recs) = emit_fold_region(
            &mut sb,
            cs.macs,
            cs.mrs,
            pfslot,
            pf_w,
            leslot,
            &locs,
            &trace.squeezes,
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );

        // ---- THE 2→1 CONNECTS: the fold's absorbed claim surfaces ARE
        // the real child regions' assertion-emission wires ----
        // Per child, per family: points to chain squeeze wires, z_partial
        // lows to absorbed child words, sigma fully (value = the child's
        // deferred s_sigma stream word), and — richer than the minimal
        // children — the matrix/element EVAL VALUES to the children's own
        // bound advice publics. Only the lagrange row lows stay the
        // boundary pattern (published below, rebuilt by the checker from
        // each child's PUBLISHED z_skip; SkipNodeGate/φ8 is the recorded
        // upgrade).
        let tapes = [&rt0, &rt1];
        let regions = [&r0, &r1];
        for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
            // Native pre-asserts (the method-note discipline).
            for t in 0..n_bool {
                let inner_t = fold_claims[2 * t][k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * t][k].row.point[..],
                    &tk.mat_assert.x_inner_rest[..inner_t],
                    "boolean type {t} row point is x_inner_rest's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][k].col.point[..],
                    &tk.mat_assert.rr[..inner_t],
                    "boolean type {t} col point is rr's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][k].col.low[..],
                    &tk.mat_assert.z_partial[..],
                    "boolean type {t} col low is z_partial"
                );
                assert_eq!(fold_claims[2 * t][k].value, tk.mat_assert.evals[t].0);
                assert_eq!(fold_claims[2 * t + 1][k].value, tk.mat_assert.evals[t].1);
            }
            for t in 0..n_el {
                let kappa = fold_claims[2 * n_bool + 2 * t][k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][k].row.point[..],
                    &tk.el_assert.r_con[..kappa],
                    "element type {t} row point is r_con's head"
                );
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][k].col.point[..],
                    &tk.el_assert.r_col[..kappa],
                    "element type {t} col point is r_col's head"
                );
                assert_eq!(fold_claims[2 * n_bool + 2 * t][k].value, tk.el_assert.evals[t].0);
                assert_eq!(
                    fold_claims[2 * n_bool + 2 * t + 1][k].value,
                    tk.el_assert.evals[t].1
                );
            }
            let nu_c = tk.sigma_native.nu;
            assert_eq!(
                &fold_claims[n_folds - 1][k].row.point[..],
                &tk.sigma_native.rho[..nu_c],
                "sigma row point is the child's rho[..nu]"
            );
            assert_eq!(
                &fold_claims[n_folds - 1][k].col.point[..],
                &tk.sigma_native.rho[nu_c..],
                "sigma col point is the child's rho[nu..]"
            );
            assert_eq!(fold_claims[n_folds - 1][k].value, tk.sigma_native.value);

            // boolean A/B per type: batch-major mlv mapping for the row
            // points, lc rounds REVERSED for the col points, z_partial
            // word-for-word, values to the mat_eval advice wires.
            for t in 0..n_bool {
                for fi in [2 * t, 2 * t + 1] {
                    let cl = &locs[fi].claims[k];
                    for j in 0..cl.row_pt_n {
                        let m = if j == 0 { 0 } else { tk.n_log_i + j };
                        sb.connect(wv(cl.row_pt_v + j), rk.b_mlv_w[m]);
                    }
                    let n_lc = rk.b_lc_w.len();
                    for j in 0..cl.col_pt_n {
                        sb.connect(wv(cl.col_pt_v + j), rk.b_lc_w[n_lc - 1 - j]);
                    }
                    for j in 0..cl.col_low_n {
                        sb.connect(wv(cl.col_low_v + j), rk.b_zpartial_w[j]);
                    }
                }
                sb.connect(wv(locs[2 * t].claims[k].value_v), rk.mat_eval_w[t].0);
                sb.connect(wv(locs[2 * t + 1].claims[k].value_v), rk.mat_eval_w[t].1);
                // ONE lagrange-low surface per child (lagrange(z_skip) is
                // type-independent): every boolean fold's lows connect to
                // fold 0's, and fold 0's publish below.
                if t > 0 {
                    for fi in [2 * t, 2 * t + 1] {
                        for j in 0..locs[0].claims[k].row_low_n {
                            sb.connect(
                                wv(locs[fi].claims[k].row_low_v + j),
                                wv(locs[0].claims[k].row_low_v + j),
                            );
                        }
                    }
                } else {
                    for j in 0..locs[0].claims[k].row_low_n {
                        sb.connect(
                            wv(locs[1].claims[k].row_low_v + j),
                            wv(locs[0].claims[k].row_low_v + j),
                        );
                    }
                }
            }
            // element A/B per type: r_con = zc.r[ν..] (round order), r_col
            // = the lc rounds REVERSED, values to the per-slot eval advice.
            for t in 0..n_el {
                for fi in [2 * n_bool + 2 * t, 2 * n_bool + 2 * t + 1] {
                    let cl = &locs[fi].claims[k];
                    sb.connect(wv(cl.row_low_v), ow);
                    sb.connect(wv(cl.col_low_v), ow);
                    for j in 0..cl.row_pt_n {
                        sb.connect(wv(cl.row_pt_v + j), rk.el_zc_rho_w[tk.n_log_i + j]);
                    }
                    let n_lc = rk.el_lc_rho_w.len();
                    for j in 0..cl.col_pt_n {
                        sb.connect(wv(cl.col_pt_v + j), rk.el_lc_rho_w[n_lc - 1 - j]);
                    }
                }
                sb.connect(
                    wv(locs[2 * n_bool + 2 * t].claims[k].value_v),
                    rk.el_eval_w[t].0,
                );
                sb.connect(
                    wv(locs[2 * n_bool + 2 * t + 1].claims[k].value_v),
                    rk.el_eval_w[t].1,
                );
            }
            // sigma: fully wire-to-wire.
            let cl = &locs[n_folds - 1].claims[k];
            sb.connect(wv(cl.row_low_v), ow);
            sb.connect(wv(cl.col_low_v), ow);
            for j in 0..cl.row_pt_n {
                sb.connect(wv(cl.row_pt_v + j), rk.pt_w[j]);
            }
            for j in 0..cl.col_pt_n {
                sb.connect(wv(cl.col_pt_v + j), rk.pt_w[cl.row_pt_n + j]);
            }
            sb.connect(wv(cl.value_v), rk.sig_w);
        }

        // Publishes: per fold, deltas + accumulator claim; then per child,
        // the lagrange-low surface (fold 0's words).
        let fold_pub_base = sb.public_len();
        for fp in &fold_pubs {
            for &w in &fp.rho_col {
                sb.publish(w);
            }
            for &w in &fp.rho_row {
                sb.publish(w);
            }
            sb.publish(fp.value);
        }
        for k in 0..2 {
            for j in 0..locs[0].claims[k].row_low_n {
                sb.publish(wv(locs[0].claims[k].row_low_v + j));
            }
        }

        let shape2 = sb.finish().expect("the 2->1 node circuit builds");
        assert!(
            shape2.circuit.cells().slots().len() <= 512,
            "the node's cell-slot budget regressed ({} slots)",
            shape2.circuit.cells().slots().len()
        );
        let hint_refs: Vec<&dyn std::any::Any> =
            hints.iter().map(|h| h as &dyn std::any::Any).collect();
        let built2 = shape2.run(&vals, &hint_refs);

        // The two child regions' checker walks — each child's whole
        // deferred-verifier statement held against its own replicas.
        let consumed0 = check_real_child_region(&built2.public, &rt0, &r0);
        let consumed1 = check_real_child_region(&built2.public, &rt1, &r1);
        assert!(
            r0.pub_base + consumed0 <= r1.pub_base && r1.pub_base + consumed1 <= fold_pub_base,
            "the three regions' public blocks are disjoint and ordered"
        );
        // The fold checker + the accumulator, reassembled from publics.
        let rebuilt = check_fold_publics(&built2.public, fold_pub_base, &locs, &alpha_recs);
        let tail_len: usize = locs.iter().map(|l| 1 + l.k_col + l.k_row).sum();
        let acc_pub = aggregate::Accumulator {
            registry_digest: registry.digest(),
            per_type: (0..n_bool)
                .map(|t| (rebuilt[2 * t].clone(), rebuilt[2 * t + 1].clone()))
                .collect(),
            per_element: (0..n_el)
                .map(|t| {
                    (
                        rebuilt[2 * n_bool + 2 * t].clone(),
                        rebuilt[2 * n_bool + 2 * t + 1].clone(),
                    )
                })
                .collect(),
            sigma: Some((lo0.shape.circuit.digest(), rebuilt[n_folds - 1].clone())),
        };
        assert_eq!(
            acc_pub, acc_v,
            "the Accumulator, reassembled from the public segment alone"
        );
        assert!(
            acc_pub.discharge(&mats)
                && acc_pub.discharge_element(&el_mats)
                && acc_pub.discharge_sigma(&lo0.shape.circuit),
            "the public-segment accumulator discharges all three groups"
        );
        // The lagrange-low publics: rebuilt from each child's PUBLISHED
        // z_skip — the one connect left at the checker tier.
        {
            use flock_core::zerocheck::K_SKIP;
            use flock_core::zerocheck::multilinear::lagrange_weights_naive;
            let mut q = fold_pub_base + tail_len;
            for (k, rk) in regions.iter().enumerate() {
                // z_skip sits just before the family-H re-exposure block.
                let zskip_pub = built2.public
                    [rk.pub_base + rk.n_query_pub + rk.n_tail - rk.n_fam_pub - 1];
                let lam = lagrange_weights_naive(K_SKIP, zskip_pub);
                for (j, &l) in lam.iter().enumerate() {
                    assert_eq!(
                        built2.public[q + j],
                        l,
                        "child {k}: lagrange low {j} rebuilds from the published z_skip"
                    );
                }
                q += 64;
            }
            assert_eq!(q, built2.public.len(), "the low publics are the very tail");
        }

        // The node proves and verifies over the circuit path.
        let union2 = UnionInstance::new(&shape2.registry, shape2.counts.clone());
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(6),
            // BLAKE3 for BOTH Merkle and FS: the node's proof must be
            // RECURSABLE — a parent replays this transcript in-circuit,
            // and each default diverges silently (the two recorded
            // gotchas, third occurrence).
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_ty2 = BitSpreadTable::new(rt0.spread_w.max(rt1.spread_w));
        let spread_r1cs2 = spread_ty2.build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape2.registry_slot(cs.q.b3),
                UnionSlotProverInput::new(
                    blake3::generate_witness_batch_major_partial(
                        built2.rows::<Blake3Gate>(cs.q.b3),
                        nu2,
                    ),
                    b3_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.swap),
                UnionSlotProverInput::new(
                    SwapTable::generate_witness_batch_major(
                        built2.rows::<SwapGate>(cs.q.swap),
                        nu2,
                    ),
                    swap_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.spread),
                UnionSlotProverInput::new(
                    spread_ty2.generate_witness_batch_major(
                        built2.rows::<BitSpreadGate>(cs.q.spread),
                        nu2,
                    ),
                    spread_lc2,
                ),
            ),
        ];
        bslots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = cs
            .element_slot_ids()
            .into_iter()
            .map(|sl| {
                let z = match &built2.witnesses[shape2.registry_slot(sl)] {
                    SlotWitness::Element(z) => z.clone(),
                    other => panic!("element slot produced {other:?}"),
                };
                (shape2.registry_slot(sl), z)
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(i, z)| live_element_input(z, shape2.counts[i], nu2))
            .collect();
        let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (shape2.registry_slot(cs.q.b3), b3_lc2),
            (shape2.registry_slot(cs.q.swap), swap_lc2),
            (shape2.registry_slot(cs.q.spread), spread_lc2),
        ];
        lco.sort_by_key(|(i, _)| *i);
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lco.into_iter().map(|(_, c)| c).collect();
        let t0p = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &pcs2,
            bslots.into_iter().map(|(_, x)| x).collect(),
            el_inputs,
            &mut ch2,
        );
        let prove_ms = t0p.elapsed().as_secs_f64() * 1e3;
        let t0v = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        verifier::verify_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &lcs2,
            &ocommit,
            &oproof,
            &pcs2,
            &mut ch2,
        )
        .expect("the 2->1 node verifies");
        let verify_ms = t0v.elapsed().as_secs_f64() * 1e3;
        let t0d = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        verifier::verify_ligerito_union_circuit_deferred(
            &union2,
            &shape2.circuit,
            &built2.public,
            &lcs2,
            &ocommit,
            &oproof,
            &pcs2,
            &mut ch2,
        )
        .expect("the 2->1 node verifies deferred");
        let deferred_ms = t0d.elapsed().as_secs_f64() * 1e3;
        let (b3_slot2, swap_slot2, spread_slot2) = (
            shape2.registry_slot(cs.q.b3),
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
        );
        println!(
            "\nTHE 2->1 RECURSION NODE (two children + {} folds, ONE proof)\n  \
             children: dense_m {} / mu {}, one circuit, distinct FS points\n  \
             regions: 2x the complete deferred verifier (swap assembly, shared slots)\n         \
             + the fold region; CONNECTED: all points, z_partial lows, sigma fully,\n         \
             and the matrix/element EVAL VALUES to the children's bound advice —\n         \
             lagrange lows published, checker-rebuilt from each child's z_skip\n  \
             outer: total b3 rows {} | nu {} | dense_m {} | mu {} \
             (cell slots: {} gate + {} public)\n  \
             prove {:.0} ms | verify {:.0} ms (DEFERRED {:.0} ms) | proof {:.1} KiB\n",
            n_folds,
            lo0.pcs.m,
            rt0.mu_i,
            b3_rows,
            nu2,
            union2.dense_m(),
            shape2.circuit.cells().mu(),
            shape2.circuit.cells().num_gate_slots(),
            shape2.circuit.cells().num_public_slots(),
            prove_ms,
            verify_ms,
            deferred_ms,
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        );
        (
            LeafOuter {
                public: built2.public.clone(),
                shape: shape2,
                proof: oproof,
                commitment: ocommit,
                pcs: pcs2,
                b3_r1cs: b3_r1cs2,
                swap_r1cs: swap_r1cs2,
                spread_r1cs: spread_r1cs2,
                b3_slot: b3_slot2,
                swap_slot: swap_slot2,
                spread_slot: spread_slot2,
            },
            acc_v,
        )
    };
    outer_stats
}

/// **THE 2→1 RECURSION NODE** — see [`build_node_outer`], which carries the
/// whole milestone (two real children's deferred verifiers + the 35-fold
/// region + the connects, one recursable proof out); this wrapper pins it
/// over two distinct leaf children.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp11_two_to_one_recursion_node() {
    let lo0 = build_leaf_outer_seeded(0x4D50_9B00);
    let lo1 = build_leaf_outer_seeded(0x4D50_9B01);
    build_node_outer(&lo0, &lo1);
}

/// **MVP-12: the node consumes its own output — the recursion TOWER.**
///
/// Four leaves → two 2→1 nodes → ONE level-2 node, built by calling
/// [`build_node_outer`] ON ITS OWN OUTPUTS. What this pins:
///
/// - The node circuit is SEED-INDEPENDENT (two nodes over different leaf
///   pairs share one digest — the foldability key at node level), and
/// - a node's proof PARSES AS A CHILD: `RealTape::new` re-asserts the
///   whole swap map on a node proof, and the level-2 build runs the same
///   emitters, connects and checkers over node-shaped children — 2→1
///   recursion composes.
/// - The SHAPE CENSUS (printed): level-2's outer vs level-1's — the
///   convergence data for pinning the normalization envelope.
///
/// TWO WALLS, hit concretely and deliberately left native (the
/// normalization worklist, recorded in the handoff):
/// - REGISTRY: the leaf outer's tables sit at nu 13, the node's at nu 16
///   (k_log differs), so a leaf-level accumulator cannot be a PRIOR of a
///   node-level fold — the accumulator chain needs one registry across
///   levels (pin nu*).
/// - SIGMA is digest-keyed: leaf-circuit sigma claims and node-circuit
///   sigma claims cannot fold together — the accumulator needs per-digest
///   sigma entries (bounded at 2 for the leaf/node two-circuit tree), or
///   full one-circuit normalization.
/// Until those land, the tree's root discharges ONE accumulator PER LEVEL
/// (all rebuilt from public segments): bounded-depth trees work today;
/// the walls are what stand between this and one accumulator total.
#[test]
#[ignore] // Heaviest — run with `-- --ignored`.
fn mvp12_recursion_tower() {
    // Four DISTINCT leaves, two per node.
    let l0 = build_leaf_outer_seeded(0x4D50_9B00);
    let l1 = build_leaf_outer_seeded(0x4D50_9B01);
    let l2 = build_leaf_outer_seeded(0x4D50_9B02);
    let l3 = build_leaf_outer_seeded(0x4D50_9B03);

    // Level 1: two 2→1 nodes.
    let (n0, acc0) = build_node_outer(&l0, &l1);
    let (n1, acc1) = build_node_outer(&l2, &l3);
    assert_eq!(
        n0.shape.circuit.digest(),
        n1.shape.circuit.digest(),
        "the NODE circuit is seed-independent — the tower's foldability key"
    );
    assert_ne!(
        n0.shape.circuit.digest(),
        l0.shape.circuit.digest(),
        "the node circuit is NOT yet the leaf circuit (normalization pending)"
    );

    // Level 2: the node consumes ITS OWN OUTPUTS — the same builder, the
    // same emitters, the same connects, over node-shaped children.
    let (n2, acc2) = build_node_outer(&n0, &n1);

    // The root's obligations, one accumulator per level (the walls above
    // are what collapse these to one): the leaf-level accumulators against
    // the LEAF registry's matrices + circuit, the node-level accumulator
    // against the NODE registry's.
    let leaf_mats = {
        let mut v = vec![
            (l0.b3_slot, (&l0.b3_r1cs.a_0, &l0.b3_r1cs.b_0)),
            (l0.swap_slot, (&l0.swap_r1cs.a_0, &l0.swap_r1cs.b_0)),
            (l0.spread_slot, (&l0.spread_r1cs.a_0, &l0.spread_r1cs.b_0)),
        ];
        v.sort_by_key(|&(i, _)| i);
        v.into_iter().map(|(_, m)| m).collect::<Vec<_>>()
    };
    let leaf_el_mats: Vec<_> = l0
        .shape
        .registry
        .element_types()
        .iter()
        .map(|s| {
            let t = s.element_type().expect("an element slot's table");
            (t.a_0(), t.b_0())
        })
        .collect();
    for (k, acc) in [&acc0, &acc1].into_iter().enumerate() {
        assert!(acc.discharge(&leaf_mats), "leaf-level acc {k}: boolean");
        assert!(
            acc.discharge_element(&leaf_el_mats),
            "leaf-level acc {k}: element"
        );
        assert!(
            acc.discharge_sigma(&l0.shape.circuit),
            "leaf-level acc {k}: sigma (keyed by the LEAF circuit)"
        );
    }
    let node_mats = {
        let mut v = vec![
            (n0.b3_slot, (&n0.b3_r1cs.a_0, &n0.b3_r1cs.b_0)),
            (n0.swap_slot, (&n0.swap_r1cs.a_0, &n0.swap_r1cs.b_0)),
            (n0.spread_slot, (&n0.spread_r1cs.a_0, &n0.spread_r1cs.b_0)),
        ];
        v.sort_by_key(|&(i, _)| i);
        v.into_iter().map(|(_, m)| m).collect::<Vec<_>>()
    };
    let node_el_mats: Vec<_> = n0
        .shape
        .registry
        .element_types()
        .iter()
        .map(|s| {
            let t = s.element_type().expect("an element slot's table");
            (t.a_0(), t.b_0())
        })
        .collect();
    assert!(acc2.discharge(&node_mats), "node-level acc: boolean");
    assert!(
        acc2.discharge_element(&node_el_mats),
        "node-level acc: element"
    );
    assert!(
        acc2.discharge_sigma(&n0.shape.circuit),
        "node-level acc: sigma (keyed by the NODE circuit)"
    );

    // ---- the SHAPE CENSUS: the convergence data for the envelope ----
    // If level-2's outer shape equals level-1's, the tower is already at
    // its fixed-point envelope and normalization is exact-shape pinning,
    // not shrinking.
    let (m1, mu1, pub1) = (
        n0.pcs.m,
        n0.shape.circuit.cells().mu(),
        n0.public.len(),
    );
    let (m2, mu2, pub2) = (
        n2.pcs.m,
        n2.shape.circuit.cells().mu(),
        n2.public.len(),
    );
    let converged = n2.shape.circuit.digest() == n0.shape.circuit.digest();
    println!(
        "\nMVP-12 RECURSION TOWER (4 leaves -> 2 nodes -> 1 level-2 node)\n  \
         level-1 node: dense_m {} | mu {} | publics {} | proof {:.1} KiB\n  \
         level-2 node: dense_m {} | mu {} | publics {} | proof {:.1} KiB\n  \
         level-2 digest == level-1 digest: {} (the normalization target:\n  \
         make this true — then ONE circuit serves every internal level)\n",
        m1,
        mu1,
        pub1,
        bincode::serialize(&n0.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        m2,
        mu2,
        pub2,
        bincode::serialize(&n2.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        converged,
    );
}
