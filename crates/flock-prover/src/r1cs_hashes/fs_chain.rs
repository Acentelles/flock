//! The Fiat–Shamir chain: BLAKE3 over a transcript, with a finalize forked at
//! every squeeze.
//!
//! This is the witness generator for the FS chain's rows — the compression
//! sequence a recursion circuit has to reproduce. Each row is one
//! [`Compression`] of the shipped [`super::blake3`] table, so the FS chain adds
//! no table type; what it adds is *which* compressions, in what order, and how
//! their chaining values connect ([`FsChainTrace::links`]).
//!
//! ## Why the finalizes dominate
//!
//! BLAKE3 is a tree, not a sequential duplex. Absorbing is cheap — 16 blocks
//! chain into a 1 KiB chunk, chunk CVs merge pairwise — but a *squeeze* has to
//! finalize the whole tree at that moment: compress the pending block, then
//! collapse the current chunk stack with `PARENT` compressions, the last of
//! them `ROOT`. That costs `popcount(complete chunks)` merges and grows as the
//! transcript does, so late squeezes cost more than early ones. On the element
//! transcript the finalize merges alone are 208 of 698 rows — a flat
//! "one compression per squeeze" model undercounts by 42%
//! (`TranscriptShape::blake3_inventory`).
//!
//! ## Correctness
//!
//! Every finalize is checked against reference BLAKE3's XOF of the same prefix.
//! Getting the chunk stack subtly wrong would still produce a self-consistent
//! circuit — one proving a hash nobody computes — so this is differential
//! rather than structural.

use super::blake3::{Compression, blake3_compress};

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const BLOCK_BYTES: usize = 64;
const BLOCKS_PER_CHUNK: usize = 16;

pub const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Where a row's `cv` input comes from — the wiring the circuit must emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvSource {
    /// The BLAKE3 IV: a public constant, no wire.
    Iv,
    /// `out_lo` of an earlier row (chunk chaining, or a parent's left input).
    Row(usize),
}

/// One row plus where its chaining input came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub cv: CvSource,
    /// For `PARENT` rows, the row supplying the RIGHT half of the message.
    /// (`cv` supplies the left half; a parent's own `cv` input is the IV.)
    pub right: Option<usize>,
    /// For an XOF output block, the ROOT row whose `cv` and message it repeats
    /// — only the counter differs. A circuit wires those inputs to the same
    /// places and varies just the params word.
    pub repeats: Option<usize>,
}

/// The compression sequence for one transcript.
pub struct FsChainTrace {
    /// Every compression, in emission order — the slot's rows.
    pub rows: Vec<Compression>,
    /// Per row, where its inputs come from.
    pub links: Vec<Link>,
    /// Per squeeze, the rows whose outputs carry the challenge bytes. The
    /// first is the `ROOT` compression; any others are counter-mode XOF blocks
    /// and are mutually independent.
    pub squeezes: Vec<Vec<usize>>,
    /// For a row that compresses transcript bytes, the byte offset of its
    /// block; `None` for `PARENT` and XOF rows, whose message is chaining
    /// values rather than stream bytes.
    ///
    /// This is what lets a circuit wire a row's `m` back to the stream — and in
    /// particular wire a **re-absorbed challenge** from the row that produced
    /// it, instead of taking it on trust as a public constant. Without it the
    /// circuit would assert the challenges rather than derive them, which is
    /// the entire content of Fiat–Shamir.
    pub block_offsets: Vec<Option<usize>>,
}

/// Incremental BLAKE3 with forkable finalization.
pub struct FsChain {
    rows: Vec<Compression>,
    links: Vec<Link>,
    squeezes: Vec<Vec<usize>>,
    /// The current chunk's running chaining value, and the row that produced
    /// it (`None` at a chunk boundary, where it is the IV).
    chunk_cv: [u32; 8],
    chunk_cv_row: Option<usize>,
    chunk_counter: u64,
    blocks_in_chunk: usize,
    /// Completed subtree CVs, with the row that produced each.
    stack: Vec<([u32; 8], usize)>,
    buf: Vec<u8>,
    block_offsets: Vec<Option<usize>>,
    /// Byte offset of the pending block's first byte.
    buf_offset: usize,
    absorbed: usize,
}

fn words(block: &[u8]) -> [u32; 16] {
    let mut m = [0u32; 16];
    for (i, c) in block.chunks(4).enumerate() {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        m[i] = u32::from_le_bytes(w);
    }
    m
}

/// A parent node's message is `left ‖ right`.
fn parent_block(left: &[u32; 8], right: &[u32; 8]) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(left);
    m[8..].copy_from_slice(right);
    m
}

impl Default for FsChain {
    fn default() -> Self {
        Self::new()
    }
}

impl FsChain {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            links: Vec::new(),
            squeezes: Vec::new(),
            chunk_cv: IV,
            chunk_cv_row: None,
            chunk_counter: 0,
            blocks_in_chunk: 0,
            stack: Vec::new(),
            buf: Vec::with_capacity(BLOCK_BYTES),
            block_offsets: Vec::new(),
            buf_offset: 0,
            absorbed: 0,
        }
    }

    fn emit(&mut self, c: Compression, link: Link, offset: Option<usize>) -> usize {
        self.rows.push(c);
        self.links.push(link);
        self.block_offsets.push(offset);
        self.rows.len() - 1
    }

    /// Absorb transcript bytes. A block is compressed once it is full *and*
    /// more input arrives — the pending block always waits, because it may yet
    /// become a chunk's last (or the root's).
    pub fn absorb(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.buf.len() == BLOCK_BYTES {
                self.compress_pending(false);
            }
            self.buf.push(b);
            self.absorbed += 1;
        }
    }

    /// Compress the buffered block into the live state.
    fn compress_pending(&mut self, chunk_end: bool) {
        let mut flags = 0;
        if self.blocks_in_chunk == 0 {
            flags |= CHUNK_START;
        }
        if chunk_end || self.blocks_in_chunk == BLOCKS_PER_CHUNK - 1 {
            flags |= CHUNK_END;
        }
        let m = words(&self.buf);
        let cv = self.chunk_cv;
        let out = blake3_compress(&cv, &m, self.chunk_counter, self.buf.len() as u32, flags);
        let link = Link {
            cv: self.chunk_cv_row.map_or(CvSource::Iv, CvSource::Row),
            right: None,
            repeats: None,
        };
        let row = self.emit(
            (cv, m, self.chunk_counter, self.buf.len() as u32, flags),
            link,
            Some(self.buf_offset),
        );
        self.buf.clear();
        self.buf_offset = self.absorbed;

        let next_cv: [u32; 8] = out[..8].try_into().unwrap();
        if flags & CHUNK_END != 0 {
            self.chunk_counter += 1;
            self.blocks_in_chunk = 0;
            self.chunk_cv = IV;
            self.chunk_cv_row = None;
            self.push_subtree(next_cv, row);
        } else {
            self.blocks_in_chunk += 1;
            self.chunk_cv = next_cv;
            self.chunk_cv_row = Some(row);
        }
    }

    /// Add a completed chunk's CV, merging while the chunk count is even —
    /// BLAKE3's chunk-stack rule.
    fn push_subtree(&mut self, mut cv: [u32; 8], mut row: usize) {
        let mut total = self.chunk_counter;
        while total & 1 == 0 {
            let (left, left_row) = self.stack.pop().expect("stack underflow");
            let m = parent_block(&left, &cv);
            let out = blake3_compress(&IV, &m, 0, BLOCK_BYTES as u32, PARENT);
            let link = Link {
                cv: CvSource::Row(left_row),
                right: Some(row),
                repeats: None,
            };
            row = self.emit((IV, m, 0, BLOCK_BYTES as u32, PARENT), link, None);
            cv = out[..8].try_into().unwrap();
            total >>= 1;
        }
        self.stack.push((cv, row));
    }

    /// Fork a root finalization off the current state and take `out_bytes` of
    /// output, without disturbing the live chain.
    ///
    /// The rows this emits are extra — the live state keeps its pending block,
    /// because more transcript follows.
    pub fn finalize(&mut self, out_bytes: usize) -> Vec<u8> {
        let mut ids = Vec::new();

        // The pending block, as this fork's last chunk block. It is the ROOT
        // itself when no completed subtree remains to merge with.
        let mut flags = CHUNK_END;
        if self.blocks_in_chunk == 0 {
            flags |= CHUNK_START;
        }
        let root_here = self.stack.is_empty();
        if root_here {
            flags |= ROOT;
        }
        let m = words(&self.buf);
        let cv = self.chunk_cv;
        let blen = self.buf.len() as u32;
        let mut out = blake3_compress(&cv, &m, self.chunk_counter, blen, flags);
        let mut row = self.emit(
            (cv, m, self.chunk_counter, blen, flags),
            Link {
                cv: self.chunk_cv_row.map_or(CvSource::Iv, CvSource::Row),
                right: None,
                repeats: None,
            },
            Some(self.buf_offset),
        );
        let mut node: [u32; 8] = out[..8].try_into().unwrap();
        let (mut root_m, mut root_cv, mut root_counter, mut root_blen, mut root_flags) =
            (m, cv, self.chunk_counter, blen, flags);

        // Collapse the stack, top-down; the last merge is the root.
        for i in (0..self.stack.len()).rev() {
            let (left, left_row) = self.stack[i];
            let pm = parent_block(&left, &node);
            let mut pf = PARENT;
            if i == 0 {
                pf |= ROOT;
            }
            out = blake3_compress(&IV, &pm, 0, BLOCK_BYTES as u32, pf);
            row = self.emit(
                (IV, pm, 0, BLOCK_BYTES as u32, pf),
                Link {
                    cv: CvSource::Row(left_row),
                    right: Some(row),
                    repeats: None,
                },
                None,
            );
            node = out[..8].try_into().unwrap();
            (root_m, root_cv, root_counter, root_blen, root_flags) =
                (pm, IV, 0, BLOCK_BYTES as u32, pf);
        }
        ids.push(row);
        let root_row = row;

        // The root compression yields the first 64 output bytes; further blocks
        // re-run it at counter 1, 2, … — counter-mode, hence independent.
        let mut bytes = Vec::with_capacity(out_bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES);
        for w in out.iter() {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let mut ctr = 1u64;
        while bytes.len() < out_bytes {
            let o = blake3_compress(&root_cv, &root_m, root_counter + ctr, root_blen, root_flags);
            let r = self.emit(
                (root_cv, root_m, root_counter + ctr, root_blen, root_flags),
                Link {
                    cv: CvSource::Iv,
                    right: None,
                    repeats: Some(root_row),
                },
                None,
            );
            ids.push(r);
            for w in o.iter() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            ctr += 1;
        }
        bytes.truncate(out_bytes);
        self.squeezes.push(ids);
        bytes
    }

    pub fn finish(self) -> FsChainTrace {
        FsChainTrace {
            rows: self.rows,
            links: self.links,
            squeezes: self.squeezes,
            block_offsets: self.block_offsets,
        }
    }
}

/// Domain flag for sponge-chain absorb compressions (transcript-v2; sits
/// above BLAKE3's chunk bits). MUST equal the challenger's constant.
pub const CHAIN_ABSORB: u32 = 1 << 6;
/// Domain flag for sponge-chain squeeze/output compressions.
pub const CHAIN_SQUEEZE: u32 = 1 << 7;

/// The SPONGE-CHAINED transcript trace builder (transcript-v2): mirrors
/// [`flock_core::challenger::FsChallenger::with_chained_blake3`] row for
/// row — a sequential compression chain, no chunk tree, no per-squeeze
/// root forks. Emits [`FsChainTrace`] rows whose links are always plain
/// chaining (`right`/`repeats` never set); squeeze OUTPUT rows carry a
/// ZERO message block (`block_offsets = None`) — the circuit feeds shared
/// zero constants there.
///
/// Drop-in for [`FsChain`] in the tape constructors: same `absorb` /
/// `finalize` / `finish` surface, and `finalize` leaves the LIVE state
/// untouched exactly as the challenger's immutable squeeze does (the
/// pending partial block is flushed into a LOCAL fork row).
pub struct FsChainSponge {
    rows: Vec<Compression>,
    links: Vec<Link>,
    squeezes: Vec<Vec<usize>>,
    block_offsets: Vec<Option<usize>>,
    cv: [u32; 8],
    cv_row: Option<usize>,
    counter: u64,
    buf: Vec<u8>,
    buf_offset: usize,
    absorbed: usize,
}

impl Default for FsChainSponge {
    fn default() -> Self {
        Self::new()
    }
}

impl FsChainSponge {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            links: Vec::new(),
            squeezes: Vec::new(),
            block_offsets: Vec::new(),
            cv: IV,
            cv_row: None,
            counter: 0,
            buf: Vec::with_capacity(BLOCK_BYTES),
            buf_offset: 0,
            absorbed: 0,
        }
    }

    fn emit(&mut self, c: Compression, link: Link, offset: Option<usize>) -> usize {
        self.rows.push(c);
        self.links.push(link);
        self.block_offsets.push(offset);
        self.rows.len() - 1
    }

    fn cv_link(&self) -> Link {
        Link {
            cv: self.cv_row.map_or(CvSource::Iv, CvSource::Row),
            right: None,
            repeats: None,
        }
    }

    pub fn absorb(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.absorbed += bytes.len();
        while self.buf.len() >= BLOCK_BYTES {
            let m = words(&self.buf[..BLOCK_BYTES]);
            let out = blake3_compress(&self.cv, &m, self.counter, BLOCK_BYTES as u32, CHAIN_ABSORB);
            let link = self.cv_link();
            let row = self.emit(
                (self.cv, m, self.counter, BLOCK_BYTES as u32, CHAIN_ABSORB),
                link,
                Some(self.buf_offset),
            );
            self.cv = out[..8].try_into().expect("8 words");
            self.cv_row = Some(row);
            self.counter += 1;
            self.buf.drain(..BLOCK_BYTES);
            self.buf_offset += BLOCK_BYTES;
        }
    }

    /// Fork a squeeze off the current state: flush the pending partial block
    /// into a LOCAL row, then emit the 64-byte output rows (zero message,
    /// output index as counter, requested length bound in `blen`). The live
    /// chain keeps its pending bytes — the challenger's squeeze is an
    /// immutable read, and every `sample_*` absorbs a header first, so
    /// consecutive squeezes separate.
    pub fn finalize(&mut self, out_bytes: usize) -> Vec<u8> {
        let (fcv, fcv_row) = if self.buf.is_empty() {
            (self.cv, self.cv_row)
        } else {
            let m = words(&self.buf);
            let out = blake3_compress(
                &self.cv,
                &m,
                self.counter,
                self.buf.len() as u32,
                CHAIN_ABSORB,
            );
            let link = self.cv_link();
            let buf_for_row = (self.cv, m, self.counter, self.buf.len() as u32, CHAIN_ABSORB);
            let row = self.emit(buf_for_row, link, Some(self.buf_offset));
            (out[..8].try_into().expect("8 words"), Some(row))
        };
        let zero = [0u32; 16];
        let mut ids = Vec::new();
        let mut bytes = Vec::with_capacity(out_bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES);
        let mut j = 0u64;
        while bytes.len() < out_bytes {
            let o = blake3_compress(&fcv, &zero, j, out_bytes as u32, CHAIN_SQUEEZE);
            let row = self.emit(
                (fcv, zero, j, out_bytes as u32, CHAIN_SQUEEZE),
                Link {
                    cv: fcv_row.map_or(CvSource::Iv, CvSource::Row),
                    right: None,
                    repeats: None,
                },
                None,
            );
            ids.push(row);
            for w in o.iter() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            j += 1;
        }
        bytes.truncate(out_bytes);
        self.squeezes.push(ids);
        bytes
    }

    pub fn finish(self) -> FsChainTrace {
        FsChainTrace {
            rows: self.rows,
            links: self.links,
            squeezes: self.squeezes,
            block_offsets: self.block_offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SPONGE trace builder must equal the chained challenger byte for
    /// byte: same absorb schedule, same squeeze outputs. The challenger is
    /// the protocol; a divergent trace builder proves a transcript nobody
    /// hashes.
    #[test]
    fn sponge_finalize_matches_the_chained_challenger() {
        use flock_core::challenger::{Challenger, FsChallenger};
        use flock_core::field::F128;
        // Drive both through the SAME op schedule via the recording layer:
        // absorb framed values exactly as the challenger frames them.
        let mut ch = FsChallenger::with_chained_blake3(b"sponge-diff");
        let mut rec_bytes: Vec<u8> = Vec::new();
        // Reproduce the challenger's framing byte-for-byte using the
        // recording of a twin transcript.
        use flock_core::transcript_record::RecordingChallenger;
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(b"sponge-diff"));
        let mut squeezed_ch: Vec<Vec<u8>> = Vec::new();
        for i in 0..40u64 {
            let v = F128 { lo: i, hi: i.wrapping_mul(77) };
            ch.observe_f128(v);
            rec.observe_f128(v);
            if i % 3 == 0 {
                let a = ch.sample_f128();
                let b = rec.sample_f128();
                assert_eq!(a, b);
                let mut bs = Vec::new();
                bs.extend_from_slice(&a.lo.to_le_bytes());
                bs.extend_from_slice(&a.hi.to_le_bytes());
                squeezed_ch.push(bs);
            }
            if i % 7 == 0 {
                let vs_a = ch.sample_f128_vec(3);
                let vs_b = rec.sample_f128_vec(3);
                assert_eq!(vs_a, vs_b);
                let mut bs = Vec::new();
                for v2 in &vs_a {
                    bs.extend_from_slice(&v2.lo.to_le_bytes());
                    bs.extend_from_slice(&v2.hi.to_le_bytes());
                }
                squeezed_ch.push(bs);
            }
        }
        // Replay the recorded byte stream through the sponge trace builder;
        // every finalize must reproduce the challenger's squeezed bytes.
        let shape = rec.shape();
        let stream = shape.stream_words(b"sponge-diff");
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let fin_ops: Vec<_> = shape.ops().iter().filter(|o| o.finalizes()).collect();
        let mut chain = FsChainSponge::new();
        let mut at = 0usize;
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            let got = chain.finalize(fin_ops[k].squeezed_bytes());
            assert_eq!(got, squeezed_ch[k], "squeeze {k}");
        }
        assert_eq!(
            stream.finalize_after.len(),
            squeezed_ch.len(),
            "every squeeze checked"
        );
        let _ = rec_bytes;
    }

    /// Every finalize must equal reference BLAKE3's XOF of the same prefix.
    ///
    /// A subtly wrong chunk stack still yields a self-consistent circuit — one
    /// proving a hash nobody computes — so this is checked against the real
    /// implementation, at lengths that straddle every boundary the tree has:
    /// inside the first block, at block ends, at the 1 KiB chunk boundary, and
    /// across several chunks so the stack actually has depth.
    #[test]
    fn every_finalize_matches_reference_blake3() {
        let data: Vec<u8> = (0..70_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();

        let lengths = [
            0usize, 1, 63, 64, 65, 127, 128, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 5000,
            16_384, 16_385, 40_000, 65_536, 65_537,
        ];
        for &len in &lengths {
            for &out_bytes in &[16usize, 32, 64, 65, 3888] {
                let mut c = FsChain::new();
                c.absorb(&data[..len]);
                let got = c.finalize(out_bytes);

                let mut want = vec![0u8; out_bytes];
                let mut h = ::blake3::Hasher::new();
                h.update(&data[..len]);
                h.finalize_xof().fill(&mut want);

                assert_eq!(got, want, "len={len}, out_bytes={out_bytes}");
            }
        }
    }

    /// Forking a finalize must not disturb the live chain: absorbing more after
    /// a squeeze still hashes the whole prefix correctly.
    #[test]
    fn finalizing_does_not_disturb_the_live_chain() {
        let data: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        let mut c = FsChain::new();
        let stops = [100usize, 1024, 1100, 3000, 5000];
        let mut at = 0usize;
        for &s in &stops {
            c.absorb(&data[at..s]);
            at = s;
            let got = c.finalize(32);
            let mut want = [0u8; 32];
            ::blake3::Hasher::new()
                .update(&data[..s])
                .finalize_xof()
                .fill(&mut want);
            assert_eq!(got, want, "after absorbing {s} bytes");
        }
        let trace = c.finish();
        assert_eq!(trace.squeezes.len(), stops.len());
        assert_eq!(trace.rows.len(), trace.links.len());
    }

    /// The row count matches what `TranscriptShape::blake3_inventory` predicts
    /// from the schedule alone — the two derivations are independent.
    #[test]
    fn row_count_matches_the_derived_inventory() {
        // One squeeze of 16 bytes after 17,008 bytes, the element transcript's
        // shape: 265 absorb + 15 chunk parents, then the finalize.
        let data = vec![7u8; 17_008];
        let mut c = FsChain::new();
        c.absorb(&data);
        c.finalize(16);
        let t = c.finish();

        let complete_chunks = (17_008usize - 1) / 1024;
        let absorb = (17_008usize - 1) / 64;
        let chunk_parents = complete_chunks - complete_chunks.count_ones() as usize;
        let finalize = 1 + complete_chunks.count_ones() as usize;
        assert_eq!(t.rows.len(), absorb + chunk_parents + finalize);
    }
}
