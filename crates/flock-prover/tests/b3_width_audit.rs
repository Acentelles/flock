//! Width/density audit harness for the BLAKE3 R1CS encoding (the b3
//! reduction track, 2026-08-05). Three ignored probes:
//!
//! - `b3_density_profile`: nnz + per-round cascade depth of the REAL
//!   `build_matrices()` output.
//! - `b3_csc_fold_cost`: what the per-prove CSC lincheck fold actually
//!   costs at the current density (this number is what priced the
//!   Option D -> E move: 21M nnz folded in ~1.1 ms).
//! - `b3_lin_id_variants`: an independent simulator of the Option-D/E
//!   design space — per-G materialization knob, exact nnz counts. Its
//!   "full drop" row matched the landed encoder bit-for-bit (48,284,894
//!   nnz), and its "baseline" row preserves Option D's numbers now that
//!   the real encoder no longer materializes lin-ids.
//!
//! Run: `cargo test --release --test b3_width_audit -- --ignored --nocapture`

use flock_prover::r1cs_hashes::blake3::{G_BASE, USEFUL_BITS, build_matrices, g_block_bits};

#[test]
#[ignore]
fn b3_density_profile() {
    let (a, b) = build_matrices();
    let nnz = |m: &flock_core::r1cs::SparseBinaryMatrix| -> usize {
        m.rows.iter().map(|r| r.len()).sum()
    };
    let (na, nb) = (nnz(&a), nnz(&b));
    eprintln!(
        "total nnz: A {na} + B {nb} = {} ({} useful rows, avg {:.1}/row)",
        na + nb,
        USEFUL_BITS,
        (na + nb) as f64 / USEFUL_BITS as f64
    );

    // Per-round aux-row density: for each G, sum A+B nnz over its ADD
    // product rows (Option F: two fused adds + two 2-op adds; round 1's
    // column G's have narrower constant-c ADD_C1 groups).
    for r in 0..7 {
        let mut aux = 0usize;
        let mut n_rows = 0usize;
        let mut max_row = 0usize;
        for gi in 0..8 {
            let g = r * 8 + gi;
            for s in G_BASE[g]..G_BASE[g] + g_block_bits(g) {
                let w = a.rows[s].len() + b.rows[s].len();
                aux += w;
                n_rows += 1;
                max_row = max_row.max(w);
            }
        }
        eprintln!(
            "round {r}: aux nnz {aux} (avg {:.1}/row over {n_rows} rows, max {max_row})",
            aux as f64 / n_rows as f64,
        );
    }

    // Output rows (out_lo / out_hi): how deep is the final cascade read?
    let out_lo_nnz: usize = (256..512).map(|s| a.rows[s].len() + b.rows[s].len()).sum();
    let out_hi_nnz: usize = (1152..1408)
        .map(|s| a.rows[s].len() + b.rows[s].len())
        .sum();
    eprintln!(
        "out_lo nnz {out_lo_nnz} (avg {:.1}/bit), out_hi nnz {out_hi_nnz} (avg {:.1}/bit)",
        out_lo_nnz as f64 / 256.0,
        out_hi_nnz as f64 / 256.0
    );
    assert!(na + nb > 0);
}

/// Time the CSC fold the recursion node's b3 slot pays per prove.
#[test]
#[ignore]
fn b3_csc_fold_cost() {
    use flock_core::field::F128;
    use flock_core::lincheck::{CscCircuit, LincheckCircuit};
    use std::time::Instant;

    let (a, b) = build_matrices();
    let t0 = Instant::now();
    let csc = CscCircuit::from_matrices(&a, &b);
    eprintln!("CSC build: {:?}", t0.elapsed());

    let k = a.num_rows;
    let eq_inner: Vec<F128> = (0..k)
        .map(|i| F128::new((i as u64).wrapping_mul(0x9e3779b97f4a7c15) | 1, i as u64))
        .collect();
    let alpha = F128::new(0x1234_5678_9abc_def0, 7);

    // warm
    let _ = csc.fold_alpha_batched(alpha, &eq_inner);
    let t1 = Instant::now();
    let n = 5u32;
    for _ in 0..n {
        std::hint::black_box(csc.fold_alpha_batched(alpha, &eq_inner));
    }
    eprintln!("fold_alpha_batched: {:?}/call", t1.elapsed() / n);

    let t2 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(csc.fold_split(&eq_inner));
    }
    eprintln!("fold_split: {:?}/call", t2.elapsed() / n);
}

// ---------------------------------------------------------------------------
// Variant simulator: rebuild the Option-D cascade with a per-(round,G) knob
// for whether b_new/d_new are materialized, and count what the matrices
// would look like. Mirrors build_matrices()' logic (Word cascade, xor-dedup,
// carry rows) without touching the real encoder.
// ---------------------------------------------------------------------------

mod sim {
    use flock_prover::r1cs_hashes::blake3::{BLAKE3_IV, G_LANES, G_MSG_IDX, MSG_PERMUTATION};

    const WB: usize = 32;
    const CARRY: usize = 31;

    #[derive(Clone)]
    pub struct Word {
        pub bits: [Vec<usize>; WB],
    }

    fn dedup(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        let mut out = Vec::with_capacity(v.len());
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j < v.len() && v[j] == v[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                out.push(v[i]);
            }
            i = j;
        }
        out
    }

    impl Word {
        fn zero() -> Self {
            Self {
                bits: std::array::from_fn(|_| Vec::new()),
            }
        }
        fn from_slot(base: usize) -> Self {
            Self {
                bits: std::array::from_fn(|i| vec![base + i]),
            }
        }
        fn from_const(val: u32, zc: usize) -> Self {
            Self {
                bits: std::array::from_fn(|i| {
                    if (val >> i) & 1 == 1 {
                        vec![zc]
                    } else {
                        Vec::new()
                    }
                }),
            }
        }
        fn xor(&self, o: &Word) -> Word {
            let mut out = self.clone();
            for i in 0..WB {
                out.bits[i].extend(&o.bits[i]);
                out.bits[i] = dedup(std::mem::take(&mut out.bits[i]));
            }
            out
        }
        fn rotr(&self, n: usize) -> Word {
            Word {
                bits: std::array::from_fn(|i| self.bits[(i + n) % WB].clone()),
            }
        }
    }

    pub struct Counts {
        pub nnz: usize,
        pub max_row: usize,
        pub useful_bits: usize,
    }

    /// Simulate the encoding where `mat_bd[g] = false` drops G `g`'s
    /// b_new/d_new lin-id slots (their reads cascade instead). Slot indices
    /// are synthetic (allocation order); only counts matter.
    struct St {
        next_slot: usize,
        nnz: usize,
        max_row: usize,
    }
    impl St {
        fn alloc(&mut self, n: usize) -> usize {
            let s = self.next_slot;
            self.next_slot += n;
            s
        }
        fn track(&mut self, w: usize) {
            self.max_row = self.max_row.max(w);
        }
        fn add(&mut self, x: &Word, y: &Word) -> Word {
            let base = self.alloc(CARRY);
            for i in 0..CARRY {
                // A row: x.bits[i] + carry prefix; B row: y.bits[i] + prefix
                let a = x.bits[i].len() + i;
                let b = y.bits[i].len() + i;
                self.nnz += a + b;
                self.track(a);
                self.track(b);
            }
            // sum word
            let mut out = Word::zero();
            for i in 0..WB {
                let mut v = x.bits[i].clone();
                v.extend(&y.bits[i]);
                for j in 0..i.min(CARRY) {
                    v.push(base + j);
                }
                out.bits[i] = dedup(v);
            }
            out
        }
    }

    pub fn simulate(mat_bd: &[bool; 56]) -> Counts {
        let mut st = St {
            next_slot: 0,
            nnz: 0,
            max_row: 0,
        };
        // inputs: cv 256, m 512, params 128, const 1 — count their identity
        // rows (2 nnz each: [slot]·[zc]) like the real builder does.
        let zc = st.alloc(1);
        st.nnz += 2;
        let cv = st.alloc(256);
        let m = st.alloc(512);
        let params = st.alloc(128);
        st.nnz += 2 * (256 + 512 + 128);

        let msg_idx = {
            let mut perm: [usize; 16] = std::array::from_fn(|i| i);
            let mut out = [[[0usize; 2]; 8]; 7];
            for r in 0..7 {
                for g in 0..8 {
                    out[r][g][0] = perm[G_MSG_IDX[g][0]];
                    out[r][g][1] = perm[G_MSG_IDX[g][1]];
                }
                let mut next = [0usize; 16];
                for i in 0..16 {
                    next[i] = perm[MSG_PERMUTATION[i]];
                }
                perm = next;
            }
            out
        };

        let mut state: [Word; 16] = std::array::from_fn(|_| Word::zero());
        for w in 0..8 {
            state[w] = Word::from_slot(cv + 32 * w);
        }
        for i in 0..4 {
            state[8 + i] = Word::from_const(BLAKE3_IV[i], zc);
        }
        for i in 0..4 {
            state[12 + i] = Word::from_slot(params + 32 * i);
        }

        for r in 0..7 {
            for gi in 0..8 {
                let g = r * 8 + gi;
                let [la, lb, lc, ld] = G_LANES[gi];
                let [mxi, myi] = msg_idx[r][gi];
                let a0 = state[la].clone();
                let b0 = state[lb].clone();
                let c0 = state[lc].clone();
                let d0 = state[ld].clone();
                let mx = Word::from_slot(m + 32 * mxi);
                let my = Word::from_slot(m + 32 * myi);

                let tmp0 = st.add(&a0, &b0);
                let a1 = st.add(&tmp0, &mx);
                let d1 = d0.xor(&a1).rotr(16);
                let c1 = st.add(&c0, &d1);
                let b1 = b0.xor(&c1).rotr(12);
                let tmp1 = st.add(&a1, &b1);
                let a2 = st.add(&tmp1, &my);
                let d2 = d1.xor(&a2).rotr(8);
                let c2 = st.add(&c1, &d2);
                let b_new = b1.xor(&c2).rotr(7);

                state[la] = a2;
                state[lc] = c2;
                if mat_bd[g] {
                    let bs = st.alloc(WB);
                    let ds = st.alloc(WB);
                    for i in 0..WB {
                        st.nnz += b_new.bits[i].len() + 1; // A lin_func + B [zc]
                        st.track(b_new.bits[i].len());
                        st.nnz += d2.bits[i].len() + 1;
                        st.track(d2.bits[i].len());
                    }
                    state[lb] = Word::from_slot(bs);
                    state[ld] = Word::from_slot(ds);
                } else {
                    state[lb] = b_new;
                    state[ld] = d2;
                }
            }
        }

        // finalization rows
        for w in 0..8 {
            let lo = state[w].xor(&state[w + 8]);
            let cv_w = Word::from_slot(cv + 32 * w);
            let hi = state[w + 8].xor(&cv_w);
            let _ = st.alloc(64);
            for i in 0..WB {
                st.nnz += lo.bits[i].len() + 1;
                st.track(lo.bits[i].len());
                st.nnz += hi.bits[i].len() + 1;
                st.track(hi.bits[i].len());
            }
        }
        Counts {
            nnz: st.nnz,
            max_row: st.max_row,
            useful_bits: st.next_slot,
        }
    }
}

#[test]
#[ignore]
fn b3_lin_id_variants() {
    let report = |name: &str, mat: &[bool; 56]| {
        let t = std::time::Instant::now();
        let c = sim::simulate(mat);
        eprintln!(
            "{name}: useful_bits {} ({} word-cols), nnz {} ({:.1}M), max_row {}, sim {:?}",
            c.useful_bits,
            c.useful_bits.div_ceil(128),
            c.nnz,
            c.nnz as f64 / 1e6,
            c.max_row,
            t.elapsed()
        );
    };

    report("baseline (all materialized)", &[true; 56]);

    let mut drop_last = [true; 56];
    for g in 48..56 {
        drop_last[g] = false;
    }
    report("drop round 6", &drop_last);

    let mut drop_56 = [true; 56];
    for g in 40..56 {
        drop_56[g] = false;
    }
    report("drop rounds 5-6", &drop_56);

    let mut drop_456 = [true; 56];
    for g in 32..56 {
        drop_456[g] = false;
    }
    report("drop rounds 4-6", &drop_456);

    report("full drop", &[false; 56]);

    // alternate G's: materialize odd G's only
    let odd: [bool; 56] = std::array::from_fn(|g| g % 2 == 1);
    report("drop even G's", &odd);
}
