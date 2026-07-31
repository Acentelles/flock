//! The circuit builder driving a **BLAKE3 chunk chain** end to end.
//!
//! This is the Fiat–Shamir chain's core structure in miniature: 16 BLAKE3
//! compressions whose chaining values thread row to row, `CHUNK_START` on the
//! first and `CHUNK_END` on the last, counter fixed — i.e. exactly one 1 KiB
//! BLAKE3 chunk. What the FS chain adds on top is the byte-packing glue that
//! places transcript bytes into `m` at arbitrary offsets; the chaining, the
//! flag pinning and the row layout are all here.
//!
//! It exercises, together and against a real prove/verify:
//!
//! - [`CircuitBuilder`] on a **boolean** slot (the element chain unit test
//!   covers the other class),
//! - `blake3::io_schema()`, including the packed `counter|block_len|flags`
//!   word being *wired to public cells* — which is what pins the flags per row
//!   position and lets one BLAKE3 table serve every compression flavour,
//! - `BuiltCircuit::rows::<G>()`, the read-back that hands a boolean slot's
//!   `&[Compression]` to `generate_witness_batch_major_partial`.
//!
//! BLAKE3 rather than SHA-256 on purpose: it is the settled hash for this
//! work, so validating the SHA path would validate one we do not use.

use flock_core::circuit::builder::{CircuitBuilder, GateType, SlotWitness, Wire};
use flock_core::field::F128;
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::schedule::TableType;
use flock_prover::union::UnionInstance;
use flock_prover::verifier;

const DOMAIN: &[u8] = b"flock-circuit-builder-v0";

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;

/// BLAKE3's IV, the chaining value a chunk starts from.
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

/// Four `u32`s into one committed 128-bit word: `lo` holds bits `[0,64)`, so
/// words 0,1 land in `lo` and 2,3 in `hi`.
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

fn unpack8(a: F128, b: F128) -> [u32; 8] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

/// The params word: `counter_lo | counter_hi | block_len | flags`, in that bit
/// order — so `lo` IS the 64-bit counter and `hi` carries `block_len` low,
/// `flags` high.
fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}

fn unpack_params(v: F128) -> (u64, u32, u32) {
    (v.lo, v.hi as u32, (v.hi >> 32) as u32)
}

/// One BLAKE3 compression as a circuit gate.
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = blake3::Compression;

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&blake3::build_block_r1cs(self.nu))
            .with_io_schema(blake3::io_schema())
    }

    fn eval(&self, inputs: &[F128]) -> (Vec<F128>, Self::Row) {
        // Schema In-order: cv0, cv1, m0..m3, params.
        let cv = unpack8(inputs[0], inputs[1]);
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = unpack_params(inputs[6]);

        let out = blake3::blake3_compress(&cv, &m, counter, block_len, flags);
        let out_lo: [u32; 8] = out[0..8].try_into().unwrap();
        let out_hi: [u32; 8] = out[8..16].try_into().unwrap();
        let (lo, hi) = (pack8(&out_lo), pack8(&out_hi));

        (
            vec![lo[0], lo[1], hi[0], hi[1]],
            (cv, m, counter, block_len, flags),
        )
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        // Boolean slots are bit-packed by `generate_witness_batch_major_partial`
        // in this crate, above the one the builder lives in.
        SlotWitness::DeferredToRows
    }
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

/// One BLAKE3 chunk (16 chained blocks) as a circuit: the IV and every message
/// block are public, the chunk's chaining value out is public, and every
/// intermediate CV is wired row to row.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn blake3_chunk_chain_through_the_builder() {
    let nu = 8usize; // BLAKE3 kappa = 14 ⇒ M = 22, the Ligerito floor
    let n_blocks = 16usize; // one 1 KiB chunk
    let mut rng = Rng(0xB1A3_0001);

    let messages: Vec<[u32; 16]> = (0..n_blocks)
        .map(|_| std::array::from_fn(|_| rng.next_u32()))
        .collect();

    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(Blake3Gate { nu });

    // The chunk's starting chaining value is public.
    let iv = pack8(&IV);
    let mut cv: [Wire; 2] = [b.public_value(iv[0]), b.public_value(iv[1])];

    for (i, m) in messages.iter().enumerate() {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i == n_blocks - 1 {
            flags |= CHUNK_END;
        }
        // Message words and the params word are public: pinning params per row
        // is what fixes the flags and the counter at that position.
        let m_w: Vec<Wire> = (0..4)
            .map(|j| b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())))
            .collect();
        let params = b.public_value(pack_params(0, 64, flags));

        let outs = b.gate(g, &[cv[0], cv[1], m_w[0], m_w[1], m_w[2], m_w[3], params]);
        // Schema Out-order: out_lo0, out_lo1, out_hi0, out_hi1. Chaining takes
        // out_lo; out_hi is unwired here (σ-fixed) — it only matters at a root.
        cv = [outs[0], outs[1]];
    }

    // The chunk's output chaining value is the circuit's public result.
    b.publish(cv[0]);
    b.publish(cv[1]);

    let built = b.finish().expect("builder produces a valid circuit");
    assert_eq!(built.counts, vec![n_blocks]);

    // The builder's rows must reproduce a plain native BLAKE3 chunk.
    let rows = built.rows::<Blake3Gate>(g);
    assert_eq!(rows.len(), n_blocks);
    let mut want_cv = IV;
    for (i, m) in messages.iter().enumerate() {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i == n_blocks - 1 {
            flags |= CHUNK_END;
        }
        assert_eq!(rows[i], (want_cv, *m, 0u64, 64u32, flags), "row {i}");
        let out = blake3::blake3_compress(&want_cv, m, 0, 64, flags);
        want_cv = out[0..8].try_into().unwrap();
    }
    // ...and the published result is that chunk's chaining value.
    let published = &built.public[built.public.len() - 2..];
    assert_eq!(unpack8(published[0], published[1]), want_cv);

    // ---- prove / verify ----
    let union = UnionInstance::new(&built.registry, built.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let r1cs = blake3::build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_jagged_union_circuit(
        &union,
        &built.circuit,
        &built.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            blake3::generate_witness_batch_major_partial(rows, nu),
            lc,
        )],
        Vec::new(),
        &mut ch,
    );

    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged_union_circuit(
        &union,
        &built.circuit,
        &built.public,
        &[lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("a builder-produced BLAKE3 chunk chain verifies");

    // A wrong claimed chunk output breaks the last wire equality — the wiring
    // is doing real work, not just decorating a satisfiable trace.
    let mut bad = built.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_jagged_union_circuit(
            &union,
            &built.circuit,
            &bad,
            &[lc],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered public output must be rejected"
    );
}
