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
    /// Bytes this op absorbs into the running state, per `FsChallenger`'s
    /// tagging (op byte, optional kind byte, optional `u64` length prefix,
    /// payload). Squeezed output is re-absorbed, so squeeze ops absorb too.
    ///
    /// This encodes the same byte layout the circuit's packing gadgets will
    /// have to reproduce, which is exactly why it is cross-checked against the
    /// live challenger's own byte counter rather than trusted.
    pub fn absorbed_bytes(&self) -> usize {
        match self {
            // OP_LABEL ‖ len ‖ label
            TranscriptOp::Label(l) => 1 + 8 + l.len(),
            // OP_OBSERVE ‖ KIND_SCALAR ‖ lo ‖ hi
            TranscriptOp::ObserveScalar => 2 + 16,
            // OP_OBSERVE ‖ KIND_SLICE ‖ len ‖ n×(lo ‖ hi)
            TranscriptOp::ObserveSlice(n) => 2 + 8 + 16 * n,
            // OP_BYTES ‖ len ‖ bytes
            TranscriptOp::ObserveBytes(len) => 1 + 8 + len,
            // OP_SQUEEZE ‖ KIND_SCALAR, then the 16 squeezed bytes re-absorbed
            TranscriptOp::SqueezeScalar => 2 + 16,
            // OP_SQUEEZE ‖ KIND_SLICE ‖ len, then 16n squeezed bytes re-absorbed
            TranscriptOp::SqueezeSlice(n) => 2 + 8 + 16 * n,
            // the nonce, absorbed through `observe_bytes`
            TranscriptOp::Pow { .. } => 1 + 8 + 8,
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
        let domain_bytes = (1 + 8 + domain.len()) as u64;
        assert_eq!(
            shape.absorbed_bytes() as u64,
            inner.absorbed_bytes() - domain_bytes,
            "TranscriptOp::absorbed_bytes disagrees with FsChallenger"
        );
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
