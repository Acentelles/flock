//! Recording the Fiat–Shamir transcript's **shape**.
//!
//! The recursive verifier has to replay this protocol's Fiat–Shamir transcript
//! inside a circuit, which means every BLAKE3 compression of it becomes
//! committed rows. Laying those rows out needs an exact, ordered account of
//! what gets absorbed and squeezed — and writing that account by hand would
//! create a second description of the transcript that can silently drift from
//! the first.
//!
//! So it is not written by hand. [`RecordingChallenger`] decorates a real
//! [`Challenger`], delegates every call unchanged, and records the sequence.
//! Running the actual verifier under it yields the schedule *as a consequence
//! of the verifier's behaviour*, so there is nothing to keep in sync.
//!
//! ## Shape, not content
//!
//! A [`TranscriptOp`] records op kind and **lengths only** — never values. Two
//! runs over different witnesses, different counts, or different proofs must
//! produce the identical op sequence; that is what makes a fixed-topology
//! circuit possible at all. It is a property of the code, not a law, so it is
//! checked rather than assumed (see the shape-diff tests). Labels *are* part of
//! the shape: they are compile-time constants that partition the transcript
//! into protocol phases.
//!
//! Prover and verifier must produce the same transcript, so recording a prove
//! and recording a verify must yield equal shapes — a free differential.
//!
//! ## The delegation trap
//!
//! [`Challenger`] gives default bodies for `observe_f128_slice` and
//! `sample_f128_vec` that decompose into per-scalar calls, and [`FsChallenger`]
//! overrides both with genuinely different absorption (a `KIND_SLICE` tag and
//! one length prefix, not `n` scalar ops). A decorator that inherits those
//! defaults would therefore **change the transcript it is trying to observe**.
//! Every method here overrides and delegates for that reason, and
//! `recording_is_transparent` pins it.
//!
//! [`FsChallenger`]: crate::challenger::FsChallenger

use sha2::{Digest, Sha256};

use crate::challenger::Challenger;
use crate::field::F128;

/// One protocol-level transcript action, with values stripped.
///
/// Deliberately at *protocol* granularity rather than byte granularity: a
/// `Pow` is one op even though it absorbs a nonce and squeezes a state digest
/// internally, because that is the unit a circuit gadget will implement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptOp {
    /// `observe_label` — a domain-separation constant. Its bytes are shape.
    Label(Vec<u8>),
    /// `observe_f128`.
    ObserveScalar,
    /// `observe_f128_slice` of `n` elements.
    ObserveSlice(usize),
    /// `observe_bytes` of `len` bytes.
    ObserveBytes(usize),
    /// `sample_f128`.
    SqueezeScalar,
    /// `sample_f128_vec` of `n` elements.
    SqueezeSlice(usize),
    /// `grind_pow` / `verify_pow` at `bits`. Both sides absorb the nonce as
    /// `observe_bytes(8)` and take one state digest, so they share an op.
    Pow { bits: u32 },
}

impl TranscriptOp {
    /// Bytes this op absorbs: a fixed 16-byte header
    /// `[op][kind][0;6][len u64]`, then the payload zero-padded to a multiple
    /// of 16. Squeezed output is re-absorbed, so squeeze ops absorb too.
    ///
    /// **Why everything is 16-aligned.** Every observed value is an `F128` —
    /// 16 bytes, and exactly one 128-bit committed word. A recursion circuit
    /// replaying this transcript places those bytes into BLAKE3's `m` words,
    /// and its wires carry 128-bit words, so the placement is a *pure copy*
    /// iff each value starts at a multiple of 16. The former 1–2 byte tags and
    /// 8-byte length prefixes broke that (scalars landed at `2 + 18k`, so seven
    /// in eight straddled two `m` words), which would have cost a byte-shift
    /// packing gate and a boolean glue table. Alignment removes the problem at
    /// its source for ~15% more FS compressions.
    ///
    /// This is the byte layout the circuit reproduces, which is why it is
    /// cross-checked against the live challenger's own counter rather than
    /// trusted.
    pub fn absorbed_bytes(&self) -> usize {
        let pad16 = |n: usize| n.div_ceil(16) * 16;
        16 + match self {
            TranscriptOp::Label(l) => pad16(l.len()),
            TranscriptOp::ObserveScalar | TranscriptOp::SqueezeScalar => 16,
            TranscriptOp::ObserveSlice(n) | TranscriptOp::SqueezeSlice(n) => 16 * n,
            TranscriptOp::ObserveBytes(len) => pad16(*len),
            // The PoW nonce rides `observe_bytes(8)`.
            TranscriptOp::Pow { .. } => 16,
        }
    }

    /// Bytes of squeezed OUTPUT. Drives the XOF-output block count: each
    /// 64 bytes is one counter-mode compression, and those are mutually
    /// independent (unlike the finalizations, which serialize).
    pub fn squeezed_bytes(&self) -> usize {
        match self {
            TranscriptOp::SqueezeScalar => 16,
            TranscriptOp::SqueezeSlice(n) => 16 * n,
            TranscriptOp::Pow { .. } => 32, // the state digest the PoW binds to
            _ => 0,
        }
    }

    /// Whether this op finalizes the pending state. Finalizations are the
    /// transcript's serial depth: each squeeze's output is re-absorbed, so
    /// nothing after one can be computed before it.
    pub fn finalizes(&self) -> bool {
        matches!(
            self,
            TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) | TranscriptOp::Pow { .. }
        )
    }
}

/// An ordered account of one run's transcript, values stripped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptShape {
    ops: Vec<TranscriptOp>,
}

impl TranscriptShape {
    pub fn ops(&self) -> &[TranscriptOp] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Index of the first op where two shapes differ, or `None` if identical.
    /// Reported rather than a bare bool so a staticness failure names the op
    /// that broke it instead of just asserting that something did.
    pub fn first_difference(&self, other: &Self) -> Option<usize> {
        let n = self.ops.len().min(other.ops.len());
        (0..n)
            .find(|&i| self.ops[i] != other.ops[i])
            .or(if self.ops.len() == other.ops.len() {
                None
            } else {
                Some(n)
            })
    }

    /// Total bytes absorbed, excluding the domain separator absorbed at
    /// challenger construction (the recorder wraps an already-built
    /// challenger, so it never sees that).
    pub fn absorbed_bytes(&self) -> usize {
        self.ops.iter().map(TranscriptOp::absorbed_bytes).sum()
    }

    pub fn squeezed_bytes(&self) -> usize {
        self.ops.iter().map(TranscriptOp::squeezed_bytes).sum()
    }

    /// Serial depth in finalizations — the number that actually sizes the FS
    /// chain's critical path.
    pub fn finalizations(&self) -> usize {
        self.ops.iter().filter(|o| o.finalizes()).count()
    }

    /// Each squeeze addressed as `(enclosing label, ordinal within that
    /// label)` instead of by absolute index.
    ///
    /// Absolute indices renumber whenever anything upstream changes, which
    /// would make every challenge-to-consumer wire in the circuit shift for an
    /// unrelated edit. Phase-relative addressing is stable under insertions
    /// elsewhere in the transcript.
    pub fn squeeze_roles(&self) -> Vec<(Vec<u8>, usize)> {
        let mut out = Vec::new();
        let mut phase: Vec<u8> = Vec::new();
        let mut ordinal = 0usize;
        for op in &self.ops {
            match op {
                TranscriptOp::Label(l) => {
                    phase = l.clone();
                    ordinal = 0;
                }
                TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => {
                    out.push((phase.clone(), ordinal));
                    ordinal += 1;
                }
                _ => {}
            }
        }
        out
    }

    /// Digest of the shape, for pinning. A protocol change that moves the FS
    /// shape moves this, so it fails loudly and gets a deliberate re-pin —
    /// the same discipline as the proof-byte fixtures.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"flock-transcript-shape-v0");
        h.update((self.ops.len() as u64).to_le_bytes());
        for op in &self.ops {
            match op {
                TranscriptOp::Label(l) => {
                    h.update([0u8]);
                    h.update((l.len() as u64).to_le_bytes());
                    h.update(l);
                }
                TranscriptOp::ObserveScalar => h.update([1u8]),
                TranscriptOp::ObserveSlice(n) => {
                    h.update([2u8]);
                    h.update((*n as u64).to_le_bytes());
                }
                TranscriptOp::ObserveBytes(len) => {
                    h.update([3u8]);
                    h.update((*len as u64).to_le_bytes());
                }
                TranscriptOp::SqueezeScalar => h.update([4u8]),
                TranscriptOp::SqueezeSlice(n) => {
                    h.update([5u8]);
                    h.update((*n as u64).to_le_bytes());
                }
                TranscriptOp::Pow { bits } => {
                    h.update([6u8]);
                    h.update(bits.to_le_bytes());
                }
            }
        }
        h.finalize().into()
    }

    /// Hex digest, for fixture constants.
    pub fn digest_hex(&self) -> String {
        self.digest().iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// The BLAKE3 compression inventory of one transcript — the FS chain's actual
/// row count, broken out by flavour because they differ in flags, in counter,
/// and in whether they serialize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake3Inventory {
    /// Blocks compressed as the stream is absorbed. Sequential within a 1 KiB
    /// chunk, independent across chunks.
    pub absorb_blocks: usize,
    /// `PARENT` compressions that build the chunk tree during absorption:
    /// `C − popcount(C)` for `C` complete chunks.
    pub chunk_parents: usize,
    /// The pending block each finalization must compress — one per squeeze.
    pub finalize_blocks: usize,
    /// **The term a flat "one compression per squeeze" model misses.** A
    /// finalize is not local: it collapses the current chunk stack, so it costs
    /// `popcount(complete chunks)` `PARENT` compressions, the last of them
    /// `ROOT`. That grows as the transcript does, so late squeezes cost more
    /// than early ones.
    pub finalize_parents: usize,
    /// XOF output blocks past the first: the root compression already yields
    /// 64 bytes, so only longer squeezes need more. Counter-mode, hence
    /// mutually independent.
    pub xof_blocks: usize,
}

impl Blake3Inventory {
    pub fn total(&self) -> usize {
        self.absorb_blocks
            + self.chunk_parents
            + self.finalize_blocks
            + self.finalize_parents
            + self.xof_blocks
    }
}

impl TranscriptShape {
    /// Count the BLAKE3 compressions this transcript actually costs, by
    /// walking the recorded schedule and tracking the byte offset — which is
    /// all that is needed, since the chunk stack's depth at any point is
    /// `popcount(offset / 1024)`.
    ///
    /// Derived rather than estimated: the FS chain's row inventory is exactly
    /// this, and the flat `one per squeeze` approximation the hash-count bench
    /// uses undercounts it (see [`Blake3Inventory::finalize_parents`]).
    pub fn blake3_inventory(&self, domain_len: usize) -> Blake3Inventory {
        let mut inv = Blake3Inventory::default();
        // The domain header + padded domain is absorbed at construction.
        let mut offset = 16 + domain_len.div_ceil(16) * 16;

        // Complete chunks at a byte offset: a chunk stays "current" until more
        // data follows it, so an exact multiple of 1024 has not closed yet.
        let complete_chunks = |o: usize| o.saturating_sub(1) / 1024;

        let mut finalize_at = |o: usize, out_bytes: usize, inv: &mut Blake3Inventory| {
            let c = complete_chunks(o);
            inv.finalize_blocks += 1;
            inv.finalize_parents += c.count_ones() as usize;
            inv.xof_blocks += out_bytes.div_ceil(64).saturating_sub(1);
        };

        for op in &self.ops {
            match op {
                TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => {
                    // The header is absorbed, THEN the state is finalized, then
                    // the squeezed output is re-absorbed.
                    offset += 16;
                    finalize_at(offset, op.squeezed_bytes(), &mut inv);
                    offset += op.absorbed_bytes() - 16;
                }
                TranscriptOp::Pow { .. } => {
                    // `grind_pow` digests the state first, then absorbs the nonce.
                    finalize_at(offset, 32, &mut inv);
                    offset += op.absorbed_bytes();
                }
                _ => offset += op.absorbed_bytes(),
            }
        }

        // The live hasher compresses a block once it is full AND more input
        // arrives, so the final block waits for a finalize.
        inv.absorb_blocks = offset.saturating_sub(1) / 64;
        let c = complete_chunks(offset);
        inv.chunk_parents = c - (c.count_ones() as usize);
        inv
    }
}

/// One 128-bit word of the absorbed byte stream.
///
/// The 16-byte-aligned framing makes the stream a sequence of whole 128-bit
/// words, and BLAKE3 consumes it 64 bytes — i.e. exactly four of these — at a
/// time as one block's `m`. So each word maps to one `m` word of one row, and
/// the circuit's job per word is to decide *where it comes from*:
///
/// - [`Const`](StreamWord::Const) → a public cell (op headers, label bytes,
///   padding),
/// - [`Value`](StreamWord::Value) → the wire already holding that proof value,
///   by **pure copy** — the whole point of aligning the framing,
/// - [`Squeezed`](StreamWord::Squeezed) → the wire holding that challenge,
///   which the FS chain itself produced and the transcript re-absorbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamWord {
    Const(F128),
    /// The `i`-th observed value, counting `F128`s in observation order.
    Value(usize),
    /// The `i`-th squeezed challenge word, in squeeze order.
    Squeezed(usize),
}

impl TranscriptShape {
    /// The absorbed byte stream as 128-bit words, given the domain the
    /// challenger was built with.
    ///
    /// This is the circuit's placement map: word `k` of the result is `m` word
    /// `k % 4` of block `k / 4`. It is derived from the recorded shape and the
    /// framing constants in [`crate::challenger`], so there is one definition
    /// of the encoding, not two.
    pub fn stream_words(&self, domain: &[u8]) -> Vec<StreamWord> {
        use crate::challenger::{
            KIND_NONE, KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL, OP_OBSERVE,
            OP_SQUEEZE,
        };
        let header = |op: u8, kind: u8, len: u64| {
            StreamWord::Const(F128::new(op as u64 | ((kind as u64) << 8), len))
        };
        // Bytes, zero-padded to a multiple of 16, as little-endian words.
        let padded = |b: &[u8], out: &mut Vec<StreamWord>| {
            for c in b.chunks(16) {
                let mut w = [0u8; 16];
                w[..c.len()].copy_from_slice(c);
                out.push(StreamWord::Const(F128::new(
                    u64::from_le_bytes(w[..8].try_into().unwrap()),
                    u64::from_le_bytes(w[8..].try_into().unwrap()),
                )));
            }
        };

        let mut out = Vec::new();
        // The domain is absorbed at construction, before recording starts.
        out.push(header(OP_DOMAIN, KIND_NONE, domain.len() as u64));
        padded(domain, &mut out);

        let (mut values, mut challenges) = (0usize, 0usize);
        for op in &self.ops {
            match op {
                TranscriptOp::Label(l) => {
                    out.push(header(OP_LABEL, KIND_NONE, l.len() as u64));
                    padded(l, &mut out);
                }
                TranscriptOp::ObserveScalar => {
                    out.push(header(OP_OBSERVE, KIND_SCALAR, 1));
                    out.push(StreamWord::Value(values));
                    values += 1;
                }
                TranscriptOp::ObserveSlice(n) => {
                    out.push(header(OP_OBSERVE, KIND_SLICE, *n as u64));
                    for _ in 0..*n {
                        out.push(StreamWord::Value(values));
                        values += 1;
                    }
                }
                TranscriptOp::ObserveBytes(len) => {
                    out.push(header(OP_BYTES, KIND_NONE, *len as u64));
                    // Content is caller-supplied bytes; the circuit wires them
                    // from wherever they live. Recorded as zero padding here
                    // because the shape does not carry values.
                    for _ in 0..len.div_ceil(16) {
                        out.push(StreamWord::Const(F128::ZERO));
                    }
                }
                TranscriptOp::SqueezeScalar => {
                    out.push(header(OP_SQUEEZE, KIND_SCALAR, 1));
                    out.push(StreamWord::Squeezed(challenges));
                    challenges += 1;
                }
                TranscriptOp::SqueezeSlice(n) => {
                    out.push(header(OP_SQUEEZE, KIND_SLICE, *n as u64));
                    for _ in 0..*n {
                        out.push(StreamWord::Squeezed(challenges));
                        challenges += 1;
                    }
                }
                TranscriptOp::Pow { .. } => {
                    out.push(header(OP_BYTES, KIND_NONE, 8));
                    out.push(StreamWord::Const(F128::ZERO)); // the nonce, padded
                }
            }
        }
        out
    }
}

/// A [`Challenger`] decorator that records the transcript's shape while
/// delegating every call to `inner` unchanged.
///
/// Transparent by construction: it computes no challenge itself, so a proof
/// verifies under `RecordingChallenger<Ch>` exactly when it verifies under
/// `Ch`. See the module docs for why every defaulted trait method is
/// nonetheless overridden.
pub struct RecordingChallenger<Ch: Challenger> {
    inner: Ch,
    ops: Vec<TranscriptOp>,
}

impl<Ch: Challenger> RecordingChallenger<Ch> {
    pub fn new(inner: Ch) -> Self {
        Self {
            inner,
            ops: Vec::new(),
        }
    }

    /// The shape recorded so far.
    ///
    /// Callers recording a *verify* should confirm the verify actually ran to
    /// completion before using this: the verifier early-returns on rejection,
    /// so a rejected proof yields a silently TRUNCATED shape, which would
    /// generate a circuit constraining only a prefix of the transcript.
    /// Record against an honest proof.
    pub fn shape(&self) -> TranscriptShape {
        TranscriptShape {
            ops: self.ops.clone(),
        }
    }

    pub fn into_parts(self) -> (Ch, TranscriptShape) {
        let shape = TranscriptShape {
            ops: self.ops.clone(),
        };
        (self.inner, shape)
    }
}

impl<Ch: Challenger> Challenger for RecordingChallenger<Ch> {
    fn observe_label(&mut self, label: &[u8]) {
        self.ops.push(TranscriptOp::Label(label.to_vec()));
        self.inner.observe_label(label);
    }

    fn observe_f128(&mut self, value: F128) {
        self.ops.push(TranscriptOp::ObserveScalar);
        self.inner.observe_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.ops.push(TranscriptOp::ObserveSlice(values.len()));
        // Delegate the SLICE call — not `n` scalar calls (see module docs).
        self.inner.observe_f128_slice(values);
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.ops.push(TranscriptOp::ObserveBytes(bytes.len()));
        self.inner.observe_bytes(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        self.ops.push(TranscriptOp::SqueezeScalar);
        self.inner.sample_f128()
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.ops.push(TranscriptOp::SqueezeSlice(n));
        // Delegate the SLICE call — one squeeze, not `n` (see module docs).
        self.inner.sample_f128_vec(n)
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.ops.push(TranscriptOp::Pow { bits });
        self.inner.grind_pow(bits)
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.ops.push(TranscriptOp::Pow { bits });
        self.inner.verify_pow(nonce, bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::hash::HashKind;

    /// Drive a challenger through one op of every kind, returning the
    /// challenges it produced. Shared so the bare and decorated runs are
    /// driven by literally the same code.
    fn drive<Ch: Challenger>(ch: &mut Ch) -> Vec<F128> {
        let mut out = Vec::new();
        ch.observe_label(b"flock-record-test-v0");
        ch.observe_f128(F128::new(0xABCD, 0x1234));
        out.push(ch.sample_f128());
        ch.observe_f128_slice(&[F128::new(1, 2), F128::new(3, 4), F128::new(5, 6)]);
        out.extend(ch.sample_f128_vec(5));
        ch.observe_bytes(&[0xAA; 37]);
        let nonce = ch.grind_pow(0);
        assert!(ch.verify_pow(nonce, 0));
        out.push(ch.sample_f128());
        out
    }

    /// The decorator must not perturb the transcript it observes. This is the
    /// load-bearing test: `observe_f128_slice` and `sample_f128_vec` have
    /// default bodies that decompose into scalar calls, and inheriting either
    /// would silently change every challenge from that point on.
    #[test]
    fn recording_is_transparent() {
        for kind in [HashKind::Sha256, HashKind::Blake3] {
            let mut bare = FsChallenger::with_hash(b"transparency", kind);
            let expected = drive(&mut bare);

            let mut rec = RecordingChallenger::new(FsChallenger::with_hash(b"transparency", kind));
            let got = drive(&mut rec);

            assert_eq!(
                got, expected,
                "recording changed the challenge stream under {kind:?}"
            );
        }
    }

    #[test]
    fn shape_records_kinds_and_lengths_in_order() {
        let mut rec = RecordingChallenger::new(FsChallenger::new(b"shape"));
        let _ = drive(&mut rec);
        let shape = rec.shape();
        assert_eq!(
            shape.ops(),
            &[
                TranscriptOp::Label(b"flock-record-test-v0".to_vec()),
                TranscriptOp::ObserveScalar,
                TranscriptOp::SqueezeScalar,
                TranscriptOp::ObserveSlice(3),
                TranscriptOp::SqueezeSlice(5),
                TranscriptOp::ObserveBytes(37),
                TranscriptOp::Pow { bits: 0 },
                TranscriptOp::Pow { bits: 0 },
                TranscriptOp::SqueezeScalar,
            ]
        );
        // Four finalizing ops: two squeezes plus the two PoW state digests.
        assert_eq!(shape.finalizations(), 5);
        // Squeezes address by phase, not absolute index.
        assert_eq!(
            shape.squeeze_roles(),
            vec![
                (b"flock-record-test-v0".to_vec(), 0),
                (b"flock-record-test-v0".to_vec(), 1),
                (b"flock-record-test-v0".to_vec(), 2),
            ]
        );
    }

    /// The byte model in [`TranscriptOp::absorbed_bytes`] is what the circuit's
    /// packing gadgets will reproduce, so it is checked against the live
    /// challenger's own counter rather than trusted.
    #[cfg(feature = "hash-count")]
    #[test]
    fn absorbed_byte_model_matches_the_live_challenger() {
        let domain: &[u8] = b"bytes";
        let mut rec = RecordingChallenger::new(FsChallenger::new(domain));
        let _ = drive(&mut rec);
        let (inner, shape) = rec.into_parts();
        // The recorder wraps an already-constructed challenger, so the domain
        // separator (OP_DOMAIN ‖ len ‖ domain) is absorbed before recording
        // starts and is not part of the shape.
        let domain_bytes = (16 + domain.len().div_ceil(16) * 16) as u64;
        assert_eq!(
            shape.absorbed_bytes() as u64,
            inner.absorbed_bytes() - domain_bytes,
            "TranscriptOp::absorbed_bytes disagrees with FsChallenger"
        );
    }

    /// **The stream model is right**: reconstructing the absorbed bytes from a
    /// recorded shape and hashing them with plain BLAKE3 reproduces the
    /// challenge `FsChallenger` actually produced.
    ///
    /// This is the assumption the whole FS-chain circuit rests on — the
    /// circuit hashes the stream `stream_words` describes, so if that
    /// description is off by a byte the circuit proves the wrong transcript.
    /// Checked against the live challenger rather than derived.
    #[test]
    fn stream_words_reconstruct_what_the_challenger_absorbs() {
        let domain: &[u8] = b"flock-stream-model";
        let mut rec = RecordingChallenger::new(FsChallenger::with_hash(domain, HashKind::Blake3));

        // Absorb a spread of op kinds, then take one challenge. Everything
        // before the squeeze must be in the stream, byte for byte.
        let vals = [
            F128::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            F128::new(1, 2),
            F128::new(3, 4),
            F128::new(5, 6),
        ];
        rec.observe_label(b"phase-one");
        rec.observe_f128(vals[0]);
        rec.observe_f128_slice(&vals[1..4]);
        let got = rec.sample_f128();
        let shape = rec.shape();

        // Rebuild the stream, substituting the observed values.
        let words = shape.stream_words(domain);
        let mut bytes = Vec::new();
        for w in &words {
            let v = match *w {
                StreamWord::Const(c) => c,
                StreamWord::Value(i) => vals[i],
                // The squeeze's own output is re-absorbed AFTER it is produced,
                // so it is not part of the prefix the challenge derives from.
                StreamWord::Squeezed(_) => break,
            };
            bytes.extend_from_slice(&v.lo.to_le_bytes());
            bytes.extend_from_slice(&v.hi.to_le_bytes());
        }

        // `sample_f128` absorbs its header, then finalizes and takes 16 bytes.
        let mut h = ::blake3::Hasher::new();
        h.update(&bytes);
        let mut buf = [0u8; 16];
        h.finalize_xof().fill(&mut buf);
        let want = F128::new(
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        );

        assert_eq!(
            got, want,
            "the reconstructed stream is not what FsChallenger absorbed — the \
             FS-chain circuit would hash the wrong bytes"
        );
        // Every word is whole: the stream is a multiple of 16 bytes, so each
        // BLAKE3 block is exactly four stream words and no value straddles one.
        assert_eq!(bytes.len() % 16, 0);
    }

    #[test]
    fn first_difference_names_the_op() {
        let a = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar, TranscriptOp::SqueezeScalar],
        };
        let b = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar, TranscriptOp::SqueezeSlice(2)],
        };
        assert_eq!(a.first_difference(&a), None);
        assert_eq!(a.first_difference(&b), Some(1));
        // A shorter prefix differs at the point it runs out — the truncation
        // case an early-returning verifier would produce.
        let short = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar],
        };
        assert_eq!(a.first_difference(&short), Some(1));
    }
}
