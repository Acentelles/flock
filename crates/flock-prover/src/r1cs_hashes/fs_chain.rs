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
        }
    }

    fn emit(&mut self, c: Compression, link: Link) -> usize {
        self.rows.push(c);
        self.links.push(link);
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
        };
        let row = self.emit(
            (cv, m, self.chunk_counter, self.buf.len() as u32, flags),
            link,
        );
        self.buf.clear();

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
            };
            row = self.emit((IV, m, 0, BLOCK_BYTES as u32, PARENT), link);
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
            },
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
                },
            );
            node = out[..8].try_into().unwrap();
            (root_m, root_cv, root_counter, root_blen, root_flags) =
                (pm, IV, 0, BLOCK_BYTES as u32, pf);
        }
        ids.push(row);

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
                },
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
