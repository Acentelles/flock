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

/// The L0 interleave for a content-sized commit: the embedded config's
/// own `initial_k` (6 everywhere except m29 Fast/Slim = 5 — the
/// recursion-node row-width choice). `prover_config_for` rejects a
/// mismatched batch, so every params site whose `m` is content-derived
/// must go through this.
fn pcs_batch_for(union: &UnionInstance, profile: LigeritoProfile) -> usize {
    flock_core::pcs::ligerito::embedded_initial_k_or_default(union.dense_m(), profile)
}

fn pcs_batch(union: &UnionInstance) -> usize {
    pcs_batch_for(union, LigeritoProfile::Fast)
}

/// `TOWER_PROFILE=slim` flips the RECURSION-PATH commits (the leaf's
/// workload inner, the leaf outer, the node outer) to Slim — rate 1/4,
/// roughly HALF the queries (m29: Σq 262 vs Fast's 491), so the
/// openings-dominated b3 trace shrinks with q while the doubled codeword
/// lands on the native NTT+Merkle side. Default Fast; the legacy mvp
/// tests stay Fast unconditionally.
fn tower_profile() -> LigeritoProfile {
    match std::env::var("TOWER_PROFILE").as_deref() {
        Ok("slim") => LigeritoProfile::Slim,
        _ => LigeritoProfile::Fast,
    }
}

/// The ENVELOPE dense floor `m*` (wall 2): every recursion-path OUTER —
/// leaf and node alike — commits at this size, so a node's children look
/// ONE shape regardless of level (an L1 node's leaf children carry the
/// same query geometry as an L2 node's node children).
///
/// OPT-IN via `TOWER_ENV_M` while the m* fork is open (measured
/// 2026-08-06 at slim): `TOWER_ENV_M=28` converges leaf+L1 geometry
/// (leaf prove 50→58, L1 = m28/nu 14/174.2 KiB, all green) but the
/// grown child proofs push L2's content over its 98%-full 2^21 boundary
/// → L2 dense_m 29, prove ~123→~157. The fork: m* = 29 (simple, slack,
/// ~+30 ms/node) vs m* = 28 (tight; needs the mac shave −8k words +
/// publics arithmetization −40k+ at the fixed point). No default until
/// the call is made.
fn envelope_floor_m() -> Option<usize> {
    if let Ok(v) = std::env::var("TOWER_ENV_M") {
        return v.parse().ok();
    }
    // Ron's call 2026-08-06: m* = 29 FIRST (the fixed point closes with
    // ~2x slack; every slim level commits m29), re-target the tight 28
    // later via the mac shave + publics diet — one deliberate re-pin.
    match tower_profile() {
        LigeritoProfile::Slim => Some(29),
        _ => None,
    }
}

/// A recursion-path OUTER's union instance, with the envelope floor
/// applied. Every instance over a leaf/node OUTER shape must come from
/// here — prover, verifier and tape recorder alike: the floor is
/// STATEMENT data, like the counts.
fn outer_union<'r>(
    registry: &'r flock_prover::schedule::Registry,
    counts: Vec<usize>,
) -> UnionInstance<'r> {
    let mut u = UnionInstance::new(registry, counts);
    if let Some(m) = envelope_floor_m() {
        u.set_dense_floor(m);
    }
    u
}

/// Wall 2's registry-geometry constants at the settled envelope (slim,
/// m* = 29): the UNION of the leaf-outer's and the node's type sets, at the
/// envelope maxima. Measured at the m29 fixed point (envelope_registry_diff
/// + the tower census, 2026-08-06):
///
/// - `spread_w` 19 = the max tree depth over the ENVELOPE's child ladders
///   (the m29 outer proof's L0; the registry shows it as io 20 — the
///   BitSpread schema is w+1 words). The leaf's own m22 inner needs only
///   12; every builder declares the envelope width and shallower ladders
///   leave the high outputs unread (the node already runs its shallow
///   levels over the one wide slot, so the emitters are width-generic by
///   existing use).
/// - `resid_pls` {12, 9, 6, 3, 0} = the m29 node ladder's suffix-fold
///   counts, chunk_log 3 throughout; the leaf's own {6, 3, 0} is a subset
///   and its two deep variants carry count 0.
/// - `nu` 15: the L2 mac slot MEASURED 27,405 rows at m29 (the recorded
///   17,975 was stale) — past 2^14, and the ~1.6k fused-dot shave cannot
///   close an 11k gap, so nu* = 14 waits for the mac diet and rides the
///   m* = 28 re-pin event.
/// - 17 types / 262 io words cross the 256 cell-slot boundary → c = 9,
///   mu = nu* + 9 = 24 tower-wide (measured ~+2 ms/node per mu step; the
///   chunked spread (−4 words) plus a resid-family consolidation are the
///   recorded way back under 256).
///
/// A ladder that drifts off these constants surfaces as a NEW slot at
/// emission time and hence a registry-digest mismatch — the failure is
/// loud, never silent.
struct EnvShape {
    nu: usize,
    spread_w: usize,
    resid_pls: [usize; 6],
    pf_w: usize,
    /// counts* — the ONE declared-count vector every envelope outer pads
    /// to (`ShapeBuilder::pad_slot_rows` before finish, so `shape.counts`
    /// and every union built from it carry counts* automatically). This is
    /// what makes the child's content — and hence num_lanes, the ladder,
    /// and the whole tape geometry a parent walks — LEVEL-INDEPENDENT.
    /// EXACT fixed-point values (Ron's framework: determinism over
    /// margin): elementwise max of the leaf/L1/L2 usage measured AT the
    /// padded envelope, iterated to closure. A usage that outgrows its cap
    /// fails the pad assert or the digest pin — loud, and the re-pin is
    /// deliberate. Boolean trio first (b3, swap, spread — NOTE: swap
    /// before spread here, matching the builders' declaration fields, NOT
    /// the registry print order), then the element types by cache key.
    counts_bool: [usize; 3],
    counts_el: [(usize, usize); 15],
    /// publics* — the ONE public-segment length every envelope outer pads
    /// to (published zeros appended after all real publics). The child's
    /// publics count is what a PARENT's walk consumes — H(publics) chain
    /// rows and the recombination's 8-lane folds both scale with it — so
    /// one count is what makes the L1 walk (leaf children) and the L2
    /// walk (node children) row-identical. The last [`ENV_APP_WORDS`] of
    /// them are the APPLICATION BLOCK (see [`env_app_base`]).
    publics: usize,
}

/// The APPLICATION STATEMENT's width in the envelope's public segment: the
/// hash-chain PoC's span `(h_start, h_end)`, eight 128-bit words.
const ENV_APP_WORDS: usize = 8;

/// The INHERITABLE ACCUMULATOR blocks. An outer publishes the accumulator
/// claims its parent will fold as PRIORS, and the parent connects to them
/// WIRE-TO-WIRE (`child_pub_w[base + ..]`) — so, exactly like the app
/// block, they have to sit where live public usage cannot move them.
/// Otherwise a first-level child and an internal child expose their claims
/// at different indices and no single parent circuit can read both.
///
/// TWO blocks, keyed by REGISTRY ROLE rather than by which fold produced
/// them — that is the distinction a parent actually cares about: MAIN
/// carries the ENVELOPE-registry claims (an internal node's own 2→1 fold),
/// CHAIN the LOWER-registry ones (a first-level node's chain fold; an
/// internal node's chain LANE). An outer with no claims of a role fills
/// that block with zeros. Each block is `[claims | zero padding]`, so a
/// shorter shape — a dev-size chain, a fold with fewer groups — rides the
/// same layout and a reader simply stops at its own group widths.
const ENV_ACC_CHAIN_WORDS: usize = 160;
const ENV_ACC_MAIN_WORDS: usize = 600;

/// The envelope's public TAIL, in order: `[.. body .. | pad | ACC_CHAIN |
/// ACC_MAIN | APP]`. Every base here is a CONSTANT of the envelope.
fn env_app_base(env: &EnvShape) -> usize {
    env.publics - ENV_APP_WORDS
}

fn env_acc_main_base(env: &EnvShape) -> usize {
    env_app_base(env) - ENV_ACC_MAIN_WORDS
}

fn env_acc_chain_base(env: &EnvShape) -> usize {
    env_acc_main_base(env) - ENV_ACC_CHAIN_WORDS
}

/// The reserved tail blocks an envelope outer hands to
/// [`pad_envelope_counts`] — published after the padding, each zero-filled
/// to its fixed width. Everything empty is the leaf/node outer's case.
#[derive(Default)]
struct EnvTail<'w> {
    /// Envelope-registry accumulator claims: this outer's own 2→1 fold.
    acc_main: &'w [Wire],
    /// Lower-registry accumulator claims: the FL's chain fold, or an
    /// internal node's chain LANE.
    acc_chain: &'w [Wire],
    /// The application statement.
    app: &'w [Wire],
}

/// `Some` exactly when the DEFAULT envelope is active: the registry
/// convergence below is pinned to m* = 29's measured geometry, so a
/// `TOWER_ENV_M` override other than 29 gets the dense floor only (an
/// experiment, not the envelope).
fn envelope_shape() -> Option<EnvShape> {
    (envelope_floor_m() == Some(29)).then(|| EnvShape {
        // 16, not 15: mac is ~97% per-child work (14,411 rows per child
        // against 921 shared, measured by MAC_CENSUS) and already sat at
        // 91% of 2^15 with two children, so any arity above 2 overflows it.
        // 2^16 fits three children (44k) and four (59k). Measured cost of
        // the step: +7.3 ms prove, zero proof bytes — the committed stack
        // is content-derived, so dense_m does not move.
        nu: 16,
        // 20 = the m32 FAST chain leaf's L0 depth (log_msg_cols 19 +
        // log_inv_rate 1), which the B-fast PoC's first-level node walks;
        // the m29 slim outer ladder needs only 19 and leaves the top
        // output unread.
        spread_w: 20,
        // Six variants: pl = Σ_{levels above} fold count, so the deepest is
        // the m32 FAST chain ladder's level-0 (six levels, 5×3 folds above
        // it) — the m29 slim outer ladder's five stop at 12 and ride the
        // rest at count 0.
        resid_pls: [15, 12, 9, 6, 3, 0],
        pf_w: 8,
        // Iterated at the padded envelope 2026-08-06 (probe + tower
        // census, elementwise max of leaf/node usage). Only b3, le8, pf8
        // and mac are content-geometry-sensitive; everything else hits its
        // cap exactly (registry-shaped) and skn/skc are the leaf's.
        counts_bool: [26200, 12250, 1060],
        counts_el: [
            (600, 49000), // mac — the nu* driver; watch the 2^15 ceiling
            (500, 1000),  // zcr
            (400, 900),   // mrs
            (0, 9000),    // spine
            (601, 300),   // assist
            (510, 64),    // skip-node (leaf-only usage)
            (511, 1),     // skip-close (leaf-only usage)
            (8, 4200),    // leaf-eval 8-lane
            (115, 900),   // resid pl 15 (the m32 chain ladder's level 0)
            (112, 1100),  // resid pl 12
            (109, 740),   // resid pl 9
            (106, 560),   // resid pl 6
            (103, 450),   // resid pl 3
            (100, 400),   // resid pl 0
            (318, 15000), // prefix w 8
        ],
        publics: 5300,
    })
}

/// Find-or-create a slot under this file's keyed-cache scheme (lanes /
/// 0 spine / 400 mrs / 500 zcr / 510 skn / 511 skc / 600 mac / 601 assist /
/// 100+pl resid / 310+w prefix). Every element-slot declaration on the
/// recursion path routes through this, so the envelope can pre-seed the
/// cache (fixing the declaration order registry-wide) while the
/// off-envelope path creates on first use, in the historical order,
/// byte-identically.
fn slot_cached<G>(
    sb: &mut ShapeBuilder,
    cache: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    key: usize,
    mk: impl FnOnce() -> G,
) -> flock_core::circuit::builder::SlotId
where
    G: GateType + Send + Sync + 'static,
    G::Row: Send + 'static,
    G::Hint: 'static,
{
    match cache.iter().find(|&&(k, _)| k == key) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(mk());
            cache.push((key, s));
            s
        }
    }
}

/// Declare the envelope's 17 table types in the ONE canonical order (wall
/// 2). `Registry::new` sorts class-major then k_log-descending with a
/// STABLE sort, so the declaration order here fixes every same-k_log
/// tie-break — the leaf-outer and node registries become the same sorted
/// type list, which together with nu* is registry-digest equality. Returns
/// the boolean trio; every element type pre-seeds `cache` under the keyed
/// scheme so both builders' demand sites hit the cache instead of
/// declaring. The order is the node's historical one with the leaf-only
/// types (SkipNode/SkipClose) appended inside their k_log group.
fn declare_envelope_slots(
    sb: &mut ShapeBuilder,
    nu: usize,
    cache: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    env: &EnvShape,
) -> CollapsedSlots {
    debug_assert_eq!(nu, env.nu, "the envelope declares at nu*");
    let q = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(env.spread_w),
            nu,
        }),
    };
    slot_cached(sb, cache, 600, MacGate::new);
    slot_cached(sb, cache, 500, ZcRoundGate::new);
    slot_cached(sb, cache, 400, MergedRoundGate::new);
    slot_cached(sb, cache, 0, SpineGate::new);
    slot_cached(sb, cache, 601, AssistLayerGate::new);
    slot_cached(sb, cache, 510, SkipNodeGate::new);
    slot_cached(sb, cache, 511, SkipCloseGate::new);
    slot_cached(sb, cache, 8, || LeafEvalGate::new(8));
    for &pl in &env.resid_pls {
        let lmc = pl + 3; // chunk_log 3 — both ladders' yr_log >= 3
        slot_cached(sb, cache, 100 + pl, || {
            ResidualGate::new(lmc, pl, 3, &sk_at_vks(lmc))
        });
    }
    slot_cached(sb, cache, 310 + env.pf_w, || PrefixGate::new(env.pf_w));
    q
}

/// Pad every envelope slot's declared count up to counts* (the counts pin:
/// one declared-count vector for every envelope outer, so the union content
/// a parent walks is level-independent). Call once per builder, AFTER all
/// emission, immediately before `finish()`.
///
/// A padding row is a REAL GATE with all-`zw` inputs (and a zero hint for
/// the hinted swap slot), so every mechanism sees an ordinary row by
/// construction: the boolean witness generators set the const bit the
/// lincheck's count binding demands (all-zero rows fail exactly there —
/// found the hard way), the element rows come out all-zero (the builder
/// tables are homogeneous), and the wiring covers the cells with genuine
/// gather-claimed gates. The outputs are deliberately unconsumed.
fn pad_envelope_counts(
    sb: &mut ShapeBuilder,
    q: &CollapsedSlots,
    cache: &[(usize, flock_core::circuit::builder::SlotId)],
    env: &EnvShape,
    zw: Wire,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    vals: &mut Vec<F128>,
    tail: &EnvTail,
) {
    let mut report: Vec<String> = Vec::new();
    let mut over: Vec<String> = Vec::new();
    let mut pad = |sb: &mut ShapeBuilder,
                   hints: &mut Vec<[u32; SLOT_WORDS]>,
                   over: &mut Vec<String>,
                   name: &str,
                   s: flock_core::circuit::builder::SlotId,
                   target: usize,
                   hinted: bool| {
        let live = sb.rows_in_slot(s);
        report.push(format!("{name} {live}/{target}"));
        if live > target {
            over.push(format!("{name} {live} > {target}"));
            return;
        }
        let ins = vec![zw; sb.slot_inputs(s)];
        for _ in live..target {
            if hinted {
                hints.push([0u32; SLOT_WORDS]);
                sb.gate_hinted(s, &ins);
            } else {
                sb.gate(s, &ins);
            }
        }
    };
    pad(sb, hints, &mut over, "b3", q.b3, env.counts_bool[0], false);
    pad(sb, hints, &mut over, "swap", q.swap, env.counts_bool[1], true);
    pad(sb, hints, &mut over, "spread", q.spread, env.counts_bool[2], false);
    for &(key, count) in &env.counts_el {
        let &(_, s) = cache
            .iter()
            .find(|&&(k, _)| k == key)
            .unwrap_or_else(|| panic!("envelope slot key {key} missing from the cache"));
        pad(sb, hints, &mut over, &format!("el{key}"), s, count, false);
    }
    // A slot the emission demanded but the envelope never declared: the
    // keyed cache created it on the fly, so this builder's registry carries
    // a type the other envelope outers do not — the digest diverges and
    // nothing else here would say so. Name it (the key IS the parameter:
    // 100 + pl for a residual variant, 310 + w for a prefix width).
    let stray: Vec<usize> = cache
        .iter()
        .map(|&(k, _)| k)
        .filter(|k| !env.counts_el.iter().any(|&(c, _)| c == *k))
        .collect();
    assert!(
        stray.is_empty(),
        "off-envelope slot keys {stray:?} — the emission needs types counts* does not declare"
    );
    // publics* (wall 4): the public segment pads to ONE length with
    // published zeros, appended after every real public — tail publics
    // shift no recorded block base, and a parent's walk (H(publics)
    // rows, recombination folds) sees the same segment length at every
    // level.
    // The TAIL blocks — the inheritable accumulator claims, then the
    // application statement — are published AFTER the padding, so each sits
    // at a constant of the envelope rather than at a function of this
    // outer's live usage. That is what lets a parent read a child's claims
    // and statement at ONE index whatever kind of child it walks. A block
    // this outer has no content for is zeros, built exactly as the padding
    // is.
    let body = env.publics - ENV_ACC_CHAIN_WORDS - ENV_ACC_MAIN_WORDS - ENV_APP_WORDS;
    let live_pub = sb.public_len();
    report.push(format!("publics {live_pub}/{body}"));
    if live_pub > body {
        over.push(format!("publics {live_pub} > {body}"));
    } else {
        for _ in live_pub..body {
            vals.push(F128::ZERO);
            sb.public_input();
        }
        for (name, w, width) in [
            ("acc_chain", tail.acc_chain, ENV_ACC_CHAIN_WORDS),
            ("acc_main", tail.acc_main, ENV_ACC_MAIN_WORDS),
            ("app", tail.app, ENV_APP_WORDS),
        ] {
            report.push(format!("{name} {}/{width}", w.len()));
            if w.len() > width {
                over.push(format!("{name} {} > {width}", w.len()));
                continue;
            }
            for &x in w {
                sb.publish(x);
            }
            for _ in w.len()..width {
                vals.push(F128::ZERO);
                sb.public_input();
            }
        }
    }
    // The live/cap census — the fixed-point iteration's data. One line per
    // build, so the tower prints leaf and node usage side by side.
    println!("  [counts* live/cap] {}", report.join(" | "));
    // An overshoot must be LOUD: a silent no-op would leave this outer's
    // counts above counts* and quietly break the level independence the
    // pin exists for. Growing usage re-pins counts* deliberately — the
    // full census above is the data for that re-pin.
    assert!(over.is_empty(), "counts* overshoot: {}", over.join(", "));
}

// ---------------------------------------------------------------------------
// THE BENCHMARK CONTRACT: what a proof costs ONLINE.
//
// ONLINE is per-STATEMENT work — everything a prover pays again for the
// next segment of the chain, the next pair of children:
//
//   walk    the circuit's evaluation over this statement (for a chain leaf
//           this IS the sequential hashing; reported apart from proving)
//   tapes   the child tape sources: recorded DEFERRED child verifies, the
//           production statement work (the pin/locate/replica scaffolding
//           around them in these tests is not this — it is per shape)
//   witgen  witness/trace generation and packing into the union's blocks
//   prove
//
// SETUP is per-SHAPE and cacheable, so it is timed separately and never
// folded into a per-proof number: the circuit emit+finish, the R1CS tables,
// the union and PCS params, the fill plan, the tape pins. A shape is
// statement-independent (the digest pins say so), so a production prover
// pays it once per level and then never again.
#[derive(Clone, Copy, Default)]
struct Online {
    setup_ms: f64,
    walk_ms: f64,
    tapes_ms: f64,
    witgen_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
}

impl Online {
    /// The per-proof online total: the prover's cost, walk included.
    fn total(&self) -> f64 {
        self.walk_ms + self.tapes_ms + self.witgen_ms + self.prove_ms
    }
}

fn median_of(runs: &[Online], f: impl Fn(&Online) -> f64) -> f64 {
    let mut v: Vec<f64> = runs.iter().map(&f).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn median_total(runs: &[Online]) -> f64 {
    median_of(runs, |o| o.total())
}

/// One stage's ONLINE line: per-phase medians, the total's median and
/// range, then the per-SHAPE setup for reference. Medians, not means —
/// the first run of any stage pays first-touch allocator costs that are
/// warmup, not marginal cost (the recorded L2 lesson).
fn report_stage(name: &str, runs: &[Online]) {
    let mut tot: Vec<f64> = runs.iter().map(|o| o.total()).collect();
    tot.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "    {name:9} walk {:6.1} + tapes {:5.1} + witgen {:5.1} + prove {:7.1} \
         = {:7.1} ms [{:.1}-{:.1}] | verify {:4.1} | (setup {:.0})",
        median_of(runs, |o| o.walk_ms),
        median_of(runs, |o| o.tapes_ms),
        median_of(runs, |o| o.witgen_ms),
        median_of(runs, |o| o.prove_ms),
        tot[tot.len() / 2],
        tot[0],
        tot[tot.len() - 1],
        median_of(runs, |o| o.verify_ms),
        median_of(runs, |o| o.setup_ms),
    );
}

const DOMAIN: &[u8] = b"flock-circuit-merkle-v0";

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;

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

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
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
        outputs.extend_from_slice(&[lo[0], lo[1], hi[0], hi[1]]);
        (cv, m, counter, block_len, flags)
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

    fn eval(&self, inputs: &[F128], hint: &Self::Hint, outputs: &mut Vec<F128>) -> Self::Row {
        let (o, row) = {
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
    };
        outputs.extend_from_slice(&o);
        row
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();

    std::hint::black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[lay.acc]]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
            None if trace.block_offsets[i].is_none() => {
                // A sponge-chain SQUEEZE output row (transcript-v2): zero
                // message block via the shared constant, chaining value
                // from the link.
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let z4 = cw(sb, vals, consts, F128::ZERO);
                (cv_in, [z4, z4, z4, z4])
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
                                let v = F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                );
                                let w = match &stream.words[wi] {
                                    // Domain labels are statement CONSTANTS
                                    // that repeat with every region — one
                                    // public per VALUE via the shared cache,
                                    // not one per occurrence (the census
                                    // found ~2.3k of the latter per child).
                                    StreamWord::Const(_) => cw(sb, vals, consts, v),
                                    StreamWord::Bytes { payload, .. }
                                        if pub_payloads.get(*payload).copied().unwrap_or(true) =>
                                    {
                                        vals.push(v);
                                        sb.public_input()
                                    }
                                    _ => {
                                        vals.push(v);
                                        sb.input()
                                    }
                                };
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

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[13], z[14], z[15], z[16], z[20]]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    /// `1/sks_vks[k]` (0 for 0), once per SLOT: `eval` used to invert these
    /// per ROW, and ~10 GF(2^128) inversions/row was ~85% of the whole
    /// online fill (measured 12 µs/row; the five residual slots were 9.2 of
    /// a child region's 10.8 ms).
    inv_sks: Vec<F128>,
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
            inv_sks: sks_vks.iter().map(|&v| inv(v)).collect(),
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

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        // A structural mirror of `new()`: same column cursor, same order.
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
        // The table's prefix rows read the ONE input wire (column 2 + pl,
        // which is also the empty-product seed) — eval mirrors the
        // constraint rather than shortcutting with literal ones, so the
        // counts* padding's all-zero inputs yield all-zero satisfying rows.
        let one_col = 2 + pl;
        let mut pr_v = z[one_col];
        for k in 0..pl {
            z[c] = z[1 + k] * (z[one_col] + z[s_col[k]] * self.inv_sks[k]);
            let pk = z[c];
            c += 1;
            z[c] = pr_v * (z[one_col] + pk);
            pr_v = z[c];
            c += 1;
        }
        let w_v: Vec<F128> = (0..yl)
            .map(|j| z[s_col[pl + j]] * self.inv_sks[pl + j])
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
        outputs.extend_from_slice(&outs);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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

/// [`live_element_input`] without the packed intermediate: the slot's rows
/// (from a DEFERRED run, [`CircuitWitness::take_rows_of`]) scatter straight
/// into the union block — the same `dst[(col << nu) + j] = row[col]` write
/// every element gate's `witness()` makes, minus the full-capacity buffer it
/// makes it into. `dst` arrives zeroed and a row shorter than the slot's
/// width leaves implicit zero columns, exactly as the packed path did.
fn live_element_input_from_rows(
    rows: Vec<Vec<F128>>,
    nu: usize,
) -> flock_prover::prover::UnionElementSlotInput<'static> {
    flock_prover::prover::UnionElementSlotInput::new(move |dst: &mut [F128]| {
        debug_assert!(rows.len() <= 1usize << nu);
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                dst[(col << nu) + j] = v;
            }
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

/// 2b stage 2: PrefixGate computes `seed * prod_j (1 + a_j + b_j)` — the
/// char-2 eq prefix of a packed-direct claim (seed = gamma, a = point,
/// b = fold challenges) or an OOD claim (seed = beta, a = z), and the eq
/// FACTORS of the close-out's per-position tensor (bit set → factor
/// `coord`, clear → `1 + coord`, pad → 1). The former SuffixGate/
/// PartialCombineGate/FinalDotGate close-out types are DISSOLVED (Round
/// 3): their tensor/combine/dot work rides prefix rows + the shared
/// MacGate — 51 schema words (each a cell slot AND a gather claim) for
/// ~30 rows of work became ~250 cheap rows and zero types.
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut pr = z[0];
        for j in 0..self.pl {
            z[c] = pr * (F128::ONE + z[1 + j] + z[1 + self.pl + j]);
            pr = z[c];
            c += 1;
        }
        outputs.extend_from_slice(&[pr]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
        let mut z = vec![F128::ZERO; 8];
        z[..4].copy_from_slice(&inputs[..4]);
        z[4] = (z[0] + z[2]) * z[3];
        z[5] = z[3] * z[3];
        z[6] = z[5] * z[2];
        z[7] = z[0] + z[1] + z[4] + z[6];
        outputs.extend_from_slice(&[z[7]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
        let mut z = vec![F128::ZERO; 5];
        z[..3].copy_from_slice(&inputs[..3]);
        z[3] = z[1] * z[2];
        z[4] = z[0] + z[3];
        outputs.extend_from_slice(&[z[4]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let sparse = flock_core::pcs::jagged::assist_sparse_transitions();
        let mut z = vec![F128::ZERO; 53];
        z[..AL_IN].copy_from_slice(&inputs[..AL_IN]);
        // z[8] is the ONE input wire the table's linear rows read — eval
        // must mirror the constraint, not shortcut it with a literal one:
        // the counts* padding rows run this eval on all-zero inputs, and a
        // literal would produce a row the zerocheck rejects.
        z[9] = z[4] * z[5];
        z[10] = z[8] + z[4] + z[5] + z[9];
        z[11] = z[4] + z[9];
        z[12] = z[5] + z[9];
        let eq4 = [10usize, 11, 12, 9];
        z[13] = z[6] * z[7];
        z[14] = z[8] + z[6] + z[7] + z[13];
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
        outputs.extend_from_slice(&z[AL_OUT0..AL_OUT0 + 4]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[9], z[14]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[7], z[10], z[13]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[10], z[11]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
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
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        
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
        outputs.extend_from_slice(&[z[8], z[13], z[14]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = flock_core::alloc_zeroed_vec::<F128>(self.ty.width() << nu);
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One packed-direct claim on the tape: its absorbed VALUE and gamma. The
/// POINT is not on the stream since merged-open v1 — it is transcript-derived
/// (gathers: the GKR's ρ_row + constant address bits; element claims: the
/// region PIOP's own challenges + the frozen prefix), and consumers rebuild
/// it from those wires and the verifier's native claims.
struct PdRec {
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
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v1") {
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
            if matches!(ops[cur.i], Op::ObserveScalar)
                && matches!(ops[cur.i + 1], Op::SqueezeScalar)
            {
                // A pd claim absorbs its VALUE only (merged-open v1); the
                // W-rounds that follow are [Obs, Obs, Squeeze] triplets, so
                // the lookahead disambiguates.
                let val_v = cur.v;
                cur.expect_obs_scalar();
                gammas.push(PdRec {
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
        log_batch_size: pcs_batch(&inner_union),
        profile: LigeritoProfile::Fast,
        num_lanes: inner_union.commit_lanes(pcs_batch(&inner_union)), // = 64 at full utilization
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
    let (geo, native_sums) = level_geometry(
        &levels,
        &lvl_src,
        &chals,
        HashKind::Blake3,
        &strat_scheds(&inner_params),
    );

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
    // merged-open v1: the points left the stream — pd_pts (the verifier's
    // own claim points) are the native reference; coordinates wire from the
    // element PIOP's round squeezes below.
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
    // Key 600: emit_residual_region's close-out already created the shared
    // MacGate slot (Round 3) — reuse it.
    let macslot = match leaf_slot.iter().find(|&&(k, _)| k == 600) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(MacGate::new());
            leaf_slot.push((600, s));
            s
        }
    };
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
    let k_cols_i = pd_pts[0].len() - n_log_i;
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
        let el_col_w = |i2: usize, j: usize| -> Wire {
            let coord = pd_pts[i2][n_log_i + j];
            if coord == F128::ZERO {
                zw
            } else if coord == F128::ONE {
                ow
            } else if i2 == 0 {
                chw(&outs, &trace.squeezes, piop.zc_rounds[n_log_i + j].fin)
            } else {
                let n_lc = piop.lc_rounds.len();
                chw(&outs, &trace.squeezes, piop.lc_rounds[n_lc - 1 - j].fin)
            }
        };
        for &i in members {
            let pd = &gammas[i];
            let gpd_w = chw(&outs, &trace.squeezes, pd.fin);
            let mut tail = ow;
            for r in 0..n_single {
                let y = r as u64;
                let factors: Vec<(Wire, Wire)> = (0..k_cols_i)
                    .map(|j| {
                        (
                            el_col_w(i, j),
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
                chw(&outs, &trace.squeezes, piop.zc_rounds[layer].fin)
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

    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
            .map(|g| (g.q, g.depth, g.c))
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

    fn eval(&self, inputs: &[F128], hint: &Self::Hint, outputs: &mut Vec<F128>) -> Self::Row {
        
        let row = SwapInput {
            bit_word: (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64),
            prev: unpack8(inputs[1], inputs[2]),
            sib: *hint,
        };
        let (left, right) = SwapTable::outputs(&row);
        let (lw, rw) = (digest_words(&left), digest_words(&right));
        outputs.extend_from_slice(&[lw[0], lw[1], rw[0], rw[1]]);
        row
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

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> u128 {
        let idx = (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64);
        outputs.extend((0..self.ty.depth).map(|l| F128::new(((idx >> l) & 1) as u64, 0)));
        idx
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

/// One opened Ligerito level's geometry. Legacy levels report it from the
/// proof itself; stratified levels carry the STATEMENT's schedule and the
/// proof is validated against it (`docs/stratified-queries.tex`: the
/// allocation is config authority, never proof-derived).
struct Lvl {
    q: usize,
    c: usize,
    depth: usize,
    /// The FOLD width `2^folds` — the lane-weight domain.
    lanes: usize,
    /// The COMMITTED width: `num_lanes` active lanes, which for a mixed
    /// union is an arbitrary integer `<= lanes` (the top lanes are
    /// definitionally zero and never encoded). Equal to `lanes` whenever
    /// the lane count happens to be a power of two.
    row_words: usize,
    /// The stratified schedule this level's config mandates. Every
    /// consumer (emit, residual, checker) maps query → (stratum depth,
    /// stratum, path slice) through this.
    sched: flock_core::pcs::stratified::LevelSchedule,
}

impl Lvl {
    /// Query `k`'s (terminal depth, stratum index).
    fn q_stratum(&self, k: usize) -> (usize, usize) {
        self.sched
            .query_strata()
            .nth(k)
            .expect("query index within schedule")
    }

    /// Query `k`'s siblings as a range into the level's flat path vec.
    fn path_range(&self, k: usize) -> std::ops::Range<usize> {
        let mut off = 0usize;
        for (i, (c, _)) in self.sched.query_strata().enumerate() {
            let len = self.depth - c;
            if i == k {
                return off..off + len;
            }
            off += len;
        }
        unreachable!("query index within schedule")
    }

    /// The position query `k` opens, from its squeezed word's low half:
    /// the low `depth − c_k` bits are sampled, the top bits ARE the
    /// stratum.
    fn q_pos(&self, k: usize, lo: u64) -> usize {
        let (c, stratum) = self.q_stratum(k);
        let lo_bits = self.depth - c;
        (stratum << lo_bits) | ((lo as usize) & ((1usize << lo_bits) - 1))
    }
}

/// The stratified schedules the inner proof's own config mandates — the
/// STATEMENT side of the query-phase geometry (None while the inner's
/// (m, profile) TOML is legacy). Derived from the same registry entry the
/// inner was proven under; never from the proof.
fn strat_scheds(params: &PcsParams) -> Vec<flock_core::pcs::stratified::LevelSchedule> {
    params
        .ligerito_verifier_config()
        .expect("registered config")
        .stratified
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
    scheds: &[flock_core::pcs::stratified::LevelSchedule],
) -> (Vec<Lvl>, Vec<F128>) {
    use flock_core::lincheck::build_eq_table;
    assert_eq!(scheds.len(), levels.len(), "one schedule per open level");
    let mut geo: Vec<Lvl> = Vec::new();
    let mut native_sums: Vec<F128> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let (cap, rows, paths) = lvl_src[li];
        let q = lvl.q_count;
        assert_eq!(rows.len(), q, "L{li}: one opened row per query");
        let sched = &scheds[li];
        // The proof is VALIDATED against the statement's schedule — never
        // the other way around.
        assert_eq!(sched.queries(), q, "L{li}: schedule owes the query count");
        let c = sched.cap_depth();
        assert_eq!(cap.len(), 1 << c, "L{li}: cap is the schedule's top layer");
        assert_eq!(
            paths.len(),
            sched.total_path_siblings(),
            "L{li}: flat paths sum the per-summand walks"
        );
        let depth = sched.log_block_len;
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
        let lv = Lvl {
            q,
            c,
            depth,
            lanes,
            row_words,
            sched: sched.clone(),
        };
        // Terminal layers above the cap, natively: layers[0] = the cap;
        // layers[j] = depth c − j. Legacy strata all sit AT the cap, so
        // only layers[0] exists and this is the old direct-cap check.
        let c_min = sched.summand_depths.last().copied().unwrap_or(c);
        let mut layers: Vec<Vec<[u8; 32]>> = vec![cap.to_vec()];
        for _ in 0..(c - c_min) {
            let next: Vec<[u8; 32]> = layers
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|p| core_merkle::hash_pair(&p[0], &p[1], hash))
                .collect();
            layers.push(next);
        }
        let mut sum = F128::ZERO;
        for (k, row) in rows.iter().enumerate() {
            let pos = lv.q_pos(k, chals[lvl.q_ch + k].lo);
            let (ck, _) = lv.q_stratum(k);
            let mut leaf_bytes = Vec::with_capacity(16 * lanes);
            for f in row {
                leaf_bytes.extend_from_slice(&f.lo.to_le_bytes());
                leaf_bytes.extend_from_slice(&f.hi.to_le_bytes());
            }
            let lh = core_merkle::hash_leaf(&leaf_bytes, hash);
            assert!(
                core_merkle::verify_merkle_proof_capped(
                    &layers[c - ck],
                    1 << depth,
                    &lh,
                    pos,
                    &paths[lv.path_range(k)],
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
        geo.push(lv);
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
        let (_cap, rows, paths) = lvl_src[li];
        let sqq = &sq[lvl.q_fin];
        let sqa = &sq[lvl.a_fin];
        // Terminals: the upper layers from the cap wires to the shallowest
        // summand (2^c − 2^c_min PARENT rows, NO root) — each query's path
        // stops at its stratum depth and connects to its schedule-constant
        // terminal wire; no hints beyond the proof's own siblings. A
        // top-summand terminal is the ABSORBED cap layer, and a direct
        // connect there puts a gate output into a class the FS chain's
        // absorb row consumes — opening → chain → squeeze → opening:
        // Cyclic. So the binding layer is layer 1: top-summand openings
        // hash ONE level further (below, in the query loop) and connect to
        // the DERIVED node, which collision resistance binds to the
        // absorbed pair. Hence at least one layer is always built.
        let c_min = g.sched.summand_depths.last().copied().unwrap_or(g.c);
        assert!(
            g.c > 0,
            "depth-0 top summand (q = 1) unsupported in-circuit"
        );
        let n_layers = (g.c - c_min).max(1);
        let mut layers_w: Vec<Vec<[Wire; 2]>> = vec![cap_w[li].clone()];
        for _ in 0..n_layers {
            let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
            let next: Vec<[Wire; 2]> = layers_w
                .last()
                .unwrap()
                .chunks(2)
                .map(|p| {
                    let out = sb.gate(
                        slots.b3,
                        &[iv[0], iv[1], p[0][0], p[0][1], p[1][0], p[1][1], params],
                    );
                    [out[0], out[1]]
                })
                .collect();
            layers_w.push(next);
        }
        // One PARENT params wire for the top-summand extension rows (the
        // `cw` helper is shadowed by the challenge wire inside the loop).
        let parent_params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
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
            let (ck, stratum) = g.q_stratum(k);
            let open_depth = g.depth - ck;
            let cv = emit_opening(
                sb,
                slots,
                iv,
                &leaf_w,
                cw,
                open_depth,
                0,
                Some(consts),
                vals,
            );
            hints.extend(paths[g.path_range(k)].iter().map(hash_to_digest));
            // Output-output connects: a multi-producer class with no gate
            // consumers — witgen asserts agreement, no dataflow cycle.
            let (bind, term) = if ck == g.c {
                // Top summand: hash one level past the cap with the
                // NEIGHBOUR cap word at constant direction (the stratum's
                // parity — no swap gate), and bind the DERIVED parent to
                // layer 1. Equality of the two layer-1 producers forces
                // cv == cap[stratum] by collision resistance, with every
                // edge forward.
                let sib = cap_w[li][stratum ^ 1];
                let (l, r) = if stratum & 1 == 0 {
                    (cv, sib)
                } else {
                    (sib, cv)
                };
                let out = sb.gate(
                    slots.b3,
                    &[iv[0], iv[1], l[0], l[1], r[0], r[1], parent_params],
                );
                ([out[0], out[1]], layers_w[1][stratum >> 1])
            } else {
                (cv, layers_w[g.c - ck][stratum])
            };
            sb.connect(bind[0], term[0]);
            sb.connect(bind[1], term[1]);
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
/// **CHUNKING (the mu-25 fix).** The ResidualGate instantiates at
/// `chunk_log = min(yr_log, 3)` — kappa 6 REGARDLESS of the proof's yr.
/// The real inner's yr = 32 otherwise pushed its schema to kappa 7-8. A
/// yr > 8 region runs as `2^(yr_log-3)` chunks of 8:
/// - the close-out claims' HIGH-bit eq factors ride the PREFIX SLOT
///   (seed = the claim's prefix product, factors = high coords vs the
///   chunk bits) — wire-bound, no new trust;
/// - the residual rows' high subset factor `sp_hi(h)` rides the CHECKER
///   tier (`awp = aw·sp_hi`, recomputed natively from the validated
///   position by `check_residual_publics` — the alpha-expansion trust
///   class; a wrong value fails the published accumulators).
/// Shapes with yr <= 8 take the single-chunk path BIT-IDENTICALLY.
/// The close-out itself (per-position eq tensors, the beta combines, the
/// yr dot) is prefix + MacGate rows since Round 3 — no dedicated types.
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
        // Reuse is sound exactly when the constructor parameters match, and
        // `pl` IS the parameter (`lmc = pl + chunk_log`, `sks` a function of
        // `lmc`; `chunk_log` is region-wide) — so the key is `100 + pl`, not
        // the level ordinal: two ladders of different depth land their
        // same-pl levels on ONE slot, which is what the envelope's
        // cross-side pre-seeding needs. Distinct per level within a ladder
        // (pl strictly decreases), so off-envelope this keys identically to
        // the old per-level scheme.
        let rslot = match leaf_slot.iter().find(|&&(k, _)| k == 100 + pl) {
            Some(&(_, s)) => s,
            None => {
                let s = sb.slot(ResidualGate::new(lmc, pl, chunk_log, &sks));
                leaf_slot.push((100 + pl, s));
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
            let pos = geo[li].q_pos(k, chals[lvl.q_ch + k].lo);
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
    // ROUND 3: the close-out's suffix/combine/dot arithmetic rides the
    // shared 4-word MacGate (cache key 600, the mvp8 convention) plus the
    // prefix slot — the SuffixGate/PartialCombineGate/FinalDotGate types
    // are DISSOLVED: 51 schema words (each a cell slot AND a gather claim)
    // bought ~30 rows of work; as mac/prefix rows the same work is ~250
    // live-prefix-cheap rows and zero types.
    let macs = match leaf_slot.iter().find(|&&(k, _)| k == 600) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(MacGate::new());
            leaf_slot.push((600, s));
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
    // accumulators: per position, ONE prefix row computes p·eq(coords, y)
    // (high bits chunk-shared, low bits per position; eq factor =
    // 1 + coord + [bit] in char 2) and ONE MacGate row accumulates it.
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
                for y in 0..chunk {
                    let factors: Vec<(Wire, Wire)> = coords[..chunk_log]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, if (y >> j) & 1 == 1 { ow } else { zw }))
                        .collect();
                    let py = prefix_chain(sb, ph, &factors);
                    let at2 = h * chunk + y;
                    evb_accs[at2] = sb.gate(macs, &[evb_accs[at2], py, ow])[0];
                }
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
    // beta-weighted residuals fold in per level (comb_y += beta·resid_y —
    // one MacGate row each), then the yr dot as one MAC chain.
    let mut comb = evb_accs;
    for (li, lvl) in levels.iter().enumerate() {
        let beta_w = chw(lvl.beta_fin);
        for y in 0..yr_len {
            comb[y] = sb.gate(macs, &[comb[y], beta_w, resid_pub[li][y]])[0];
        }
    }
    let mut inner_w = zw;
    for (yw, cb) in yr_wires.iter().zip(&comb) {
        inner_w = sb.gate(macs, &[inner_w, *yw, *cb])[0];
    }
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
                let pos = geo[li].q_pos(k, chals[lvl.q_ch + k].lo);
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
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
    let setup = blake3::Blake3Setup::batch_major_with_profile(n_blocks, tower_profile());
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
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (proof, commitment, _claim) =
        prover::prove_fast_ligerito_union(&union, &leaf_pcs, vec![slot], &mut ch);

    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
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
        while !matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-merged-open-v1") {
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
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v1") {
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
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

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
            level_geometry(&levels, &lvl_src, &chals, HashKind::Blake3, &strat_scheds(&leaf_pcs));

        let stream = t_shape.stream_words_duplex(DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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
        let nu_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
        let spread_own = geo.iter().map(|g| g.depth).max().unwrap().max(1);
        // Under the envelope the leaf takes the ENVELOPE geometry — nu*,
        // the node's spread width (its own trees are shallower; the high
        // outputs go unread) and, below, the full canonical type set with
        // the node-only types at count 0 — so its registry digest-equals
        // the node's (wall 2).
        let env = envelope_shape();
        let (nu, spread_w) = match &env {
            Some(e) => {
                assert!(
                    nu_content <= e.nu,
                    "leaf content nu {nu_content} exceeds the envelope nu* {}",
                    e.nu
                );
                assert!(
                    spread_own <= e.spread_w,
                    "leaf spread width {spread_own} exceeds the envelope's {}",
                    e.spread_w
                );
                (e.nu, e.spread_w)
            }
            None => (nu_content, spread_own),
        };

        let mut sb = ShapeBuilder::new(nu);
        let mut leaf_slot: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
        let slots = match &env {
            Some(e) => declare_envelope_slots(&mut sb, nu, &mut leaf_slot, e),
            None => CollapsedSlots {
                b3: sb.slot(Blake3Gate { nu }),
                swap: sb.slot(SwapGate { nu }),
                spread: sb.slot(BitSpreadGate {
                    ty: BitSpreadTable::new(spread_w),
                    nu,
                }),
            },
        };
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
        let mrslot = slot_cached(&mut sb, &mut leaf_slot, 400, MergedRoundGate::new);
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
        let spine = slot_cached(&mut sb, &mut leaf_slot, 0, SpineGate::new);
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
        let skslot = slot_cached(&mut sb, &mut leaf_slot, 510, SkipNodeGate::new);
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
        let scslot = slot_cached(&mut sb, &mut leaf_slot, 511, SkipCloseGate::new);
        let mut cin = vec![skc, skab];
        cin.extend_from_slice(&zpw);
        let cl = sb.gate(scslot, &cin);
        let (rc_w, seed_w) = (cl[0], cl[1]);
        let zslot = slot_cached(&mut sb, &mut leaf_slot, 500, ZcRoundGate::new);
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
        let alslot = slot_cached(&mut sb, &mut leaf_slot, 601, AssistLayerGate::new);
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
        // counts* + publics*: every envelope outer declares the ONE count
        // vector and the ONE public-segment length — shape.counts and
        // every union cloned from it carry them onward. The boundary
        // checks below walk the REAL segment end, so its pre-pad length
        // is recorded first.
        let prepad_publics = sb.public_len();
        if let Some(e) = &env {
            // A leaf outer folds nothing and carries no application, so
            // every reserved tail block is zeros — the layout is the
            // envelope's, not this builder's.
            pad_envelope_counts(
                &mut sb,
                &slots,
                &leaf_slot,
                e,
                zw,
                &mut hints,
                &mut vals,
                &EnvTail::default(),
            );
        }
        let shape = sb.finish().expect("valid leaf query-phase circuit");
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
        let built = shape.run(&vals, &hint_refs);

        // ---- boundary checks: alphas and the enforced sums.
        // The anchor-expect tail (sqrt-chain deltas + the claim==expect
        // delta) is appended after everything else; `plen` is the public
        // length BEFORE it, so every older from-the-end offset holds.
        // The sqrt-chain, anchor-expect, zc-round and T_m == anchor.v
        // identities are COPY CONSTRAINTS — no publics, no checker items.
        // The REAL segment's end: publics* zeros sit past it (envelope
        // only; equal to the full length otherwise).
        let plen = prepad_publics;
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
        let union_o = outer_union(&shape.registry, shape.counts.clone());
        let pf = tower_profile();
        let pcs_o = PcsParams {
            m: union_o.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union_o, pf),
            profile: pf,
            num_lanes: union_o.commit_lanes(pcs_batch_for(&union_o, pf)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs = blake3::build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let swap_r1cs = SwapTable::build_block_r1cs(nu);
        let swap_lc = swap_r1cs.csc_lincheck_circuit();
        let spread_ty = BitSpreadTable::new(spread_w);
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let b3_rows_l = built.rows::<Blake3Gate>(slots.b3).to_vec();
        let swap_rows_l = built.rows::<SwapGate>(slots.swap).to_vec();
        let spread_rows_l = built.rows::<BitSpreadGate>(slots.spread).to_vec();
        let els: Vec<Vec<F128>> = leaf_slot
            .iter()
            .map(|(_, sl)| match &built.witnesses[shape.registry_slot(*sl)] {
                SlotWitness::Element(z) => z.clone(),
                other => panic!("leaf-eval slot produced {other:?}"),
            })
            .collect();
        // Every element slot's block satisfies its table at the DECLARED
        // count — the earliest, name-carrying catch for an eval/table
        // drift (a literal where the table reads a wire — the counts*
        // padding found two; the zerocheck only says "the region sum is
        // off"). Cheap: one satisfies() pass per slot at build time.
        {
            let mut bad = Vec::new();
            for ((key, sl), z) in leaf_slot.iter().zip(&els) {
                let t = shape.registry_slot(*sl);
                if let Some(el) = shape.registry.types()[t].element_type() {
                    if !el.satisfies(z, nu, shape.counts[t]) {
                        bad.push(format!("key {key} (t{t}, count {})", shape.counts[t]));
                    }
                }
            }
            assert!(bad.is_empty(), "element slot unsatisfied at its declared count: {}", bad.join(", "));
        }
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
                    UnionSlotProverInput::in_place(
                        {
                            let r = b3_rows_l.clone();
                            move |dst| {
                                blake3::generate_witness_batch_major_partial_into(&r, nu, dst)
                            }
                        },
                        b3_lc,
                    ),
                ),
                (
                    shape.registry_slot(slots.swap),
                    UnionSlotProverInput::in_place(
                        {
                            let r = swap_rows_l.clone();
                            move |dst| SwapTable::generate_witness_batch_major_into(&r, dst)
                        },
                        swap_lc,
                    ),
                ),
                (
                    shape.registry_slot(slots.spread),
                    UnionSlotProverInput::in_place(
                        {
                            let r = spread_rows_l.clone();
                            let ty = BitSpreadTable::new(spread_w);
                            move |dst| ty.generate_witness_batch_major_into(&r, dst)
                        },
                        spread_lc,
                    ),
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
            let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
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
            let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
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
    /// The child cell space's public-slot count — the recombination's tail.
    n_pub_slots_c: usize,
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
    /// Derived pd claim points (merged-open v1), pinned order
    /// [element c, element lc, gathers in cell-slot order].
    pd_pts: Vec<Vec<F128>>,
}

impl<'p> RealTape<'p> {
    fn new(lo: &'p LeafOuter, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (lo.b3_slot, lo.b3_r1cs.csc_lincheck_circuit()),
            (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
            (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.into_iter().map(|(_, cc)| cc).collect();
        // ONE recorded DEFERRED verify serves both needs: it is
        // transcript-identical to the plain verify for honest proofs (so
        // the tape is unchanged), it skips the sigma discharge the plain
        // pass paid, and its exported assertions ARE the method-note
        // references (verifier-exported over formulas-written-twice).
        // This is also exactly what a production node runs per child —
        // the tape cost halved when the second pass dissolved.
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
        let (mat_assert, el_assert, sigma_native, claims) = {
            let (claims, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
                &union_i,
                &lo.shape.circuit,
                &lo.public,
                &lcs,
                &lo.commitment,
                &lo.proof,
                &lo.pcs,
                &mut rec,
            )
            .expect("the deferred verify accepts the leaf outer");
        assert!(claims.boolean.is_some(), "boolean claims from the real inner");
        assert!(claims.element.is_some(), "element claims from the real inner");
            (
                work.boolean.expect("a boolean PIOP ran"),
                work.element.expect("an element PIOP ran"),
                sigma,
                claims,
            )
        };
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops = t_shape.ops();
        let mut pub_payloads = bytes_payload_mask(ops);
        // Prefix sums over the op tape — the locate walks below call these
        // per feature and per ROUND (437 rounds at node scale), so a
        // rescan-per-call is quadratic in practice. One pass, O(1) lookups.
        let (pre_v, pre_c, pre_f) = {
            let mut pre_v = Vec::with_capacity(ops.len() + 1);
            let mut pre_c = Vec::with_capacity(ops.len() + 1);
            let mut pre_f = Vec::with_capacity(ops.len() + 1);
            let (mut v, mut c, mut f) = (0usize, 0usize, 0usize);
            pre_v.push(0);
            pre_c.push(0);
            pre_f.push(0);
            for op in ops {
                match op {
                    Op::SqueezeScalar => c += 1,
                    Op::SqueezeSlice(n) => c += n,
                    Op::ObserveScalar => v += 1,
                    Op::ObserveSlice(n) => v += n,
                    _ => {}
                }
                if op.finalizes() {
                    f += 1;
                }
                pre_v.push(v);
                pre_c.push(c);
                pre_f.push(f);
            }
            (pre_v, pre_c, pre_f)
        };
        let vc_at = |end: usize| -> (usize, usize) { (pre_v[end], pre_c[end]) };
        let fin_at = |end: usize| pre_f[end];

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
        let mo_l = find(b"flock-merged-open-v1");
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
        let (geo, native_sums) = level_geometry(
            &levels,
            &lvl_src,
            &chals,
            HashKind::Blake3,
            &strat_scheds(&lo.pcs),
        );
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
        let stream = t_shape.stream_words_duplex(domain);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let trace = {
            let mut chain = FsChainSponge::new();
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
        if std::env::var("B3_CENSUS").is_ok() {
            let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
            let blocks = trace.rows.len() - parents;
            eprintln!(
                "  [b3 census] chain {} (data blocks {} | parent/fork {}; absorbed {} B, {} squeezes) | H(publics) {} | openings+caps {} = {}",
                trace.rows.len(),
                blocks,
                parents,
                bytes.len(),
                trace.squeezes.len(),
                h_rows,
                b3_rows - trace.rows.len() - h_rows,
                b3_rows
            );
            for g in geo.iter() {
                eprintln!(
                    "    level: q {} depth {} row_words {} -> leaf {} + path {} + cap {}",
                    g.q,
                    g.depth,
                    g.row_words,
                    g.row_words.div_ceil(4) * g.q,
                    g.depth * g.q,
                    (1usize << g.c) - 1
                );
            }
            // CHAIN DECOMPOSITION + an independent row-count model of the
            // duplex discipline (transcript-v3), asserted against the
            // sponge trace: a squeeze row absorbs the pending partial
            // block as its MESSAGE, mutates cv, and has no header word.
            {
                let pad16 = |n: usize| n.div_ceil(16) * 16;
                let (mut hdr_w, mut pay_w, mut n_obs, mut n_sq) = (0usize, 0usize, 0usize, 0usize);
                // The domain header + padded domain are absorbed at
                // construction, ahead of the recorded ops.
                let (mut v3_rows, mut pend) = (0usize, 16 + pad16(domain.len()));
                for op in ops.iter() {
                    match op {
                        Op::Label(l) => {
                            hdr_w += 1;
                            pay_w += pad16(l.len()) / 16;
                            n_obs += 1;
                            pend += 16 + pad16(l.len());
                        }
                        Op::ObserveScalar => {
                            hdr_w += 1;
                            pay_w += 1;
                            n_obs += 1;
                            pend += 32;
                        }
                        Op::ObserveSlice(n) => {
                            hdr_w += 1;
                            pay_w += n;
                            n_obs += 1;
                            pend += 16 + 16 * n;
                        }
                        Op::ObserveBytes(len) => {
                            hdr_w += 1;
                            pay_w += pad16(*len) / 16;
                            n_obs += 1;
                            pend += 16 + pad16(*len);
                        }
                        Op::SqueezeScalar | Op::SqueezeSlice(_) | Op::Pow { .. } => {
                            n_sq += 1;
                            // v3: the squeeze row eats the pending partial
                            // block and emits output block 0; extra output
                            // blocks follow.
                            v3_rows += pend / 64;
                            v3_rows += 1 + (op.squeezed_bytes().div_ceil(64) - 1);
                            pend = 0;
                            if let Op::Pow { .. } = op {
                                // the nonce rides observe_bytes(8): header + word
                                pend += 32;
                            }
                        }
                    }
                    v3_rows += pend / 64;
                    pend %= 64;
                }
                if pend > 0 {
                    v3_rows += 1;
                }
                eprintln!(
                    "  [chain census] ops {} (obs {} / sq {}) | header words {} ({} B) | payload words {} | duplex rows {}",
                    ops.len(),
                    n_obs,
                    n_sq,
                    hdr_w,
                    16 * hdr_w,
                    pay_w,
                    trace.rows.len(),
                );
                assert_eq!(
                    v3_rows,
                    trace.rows.len(),
                    "the duplex row model diverged from the sponge trace"
                );
            }
        }
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

        // ---- the residual pairing's rotation (lane-major inners) ----
        // A pow2-lane inner (row_words == lanes — e.g. the m28-k4 slim node
        // whose 16-of-16 lanes make the commit exactly full) takes the
        // IDENTITY pairing, same as the native side's rotate gate and
        // ChildTape's conditional.
        let yr_len = lo.proof.pcs_open.inner.ligerito.final_proof.yr.len();
        let lane_major = geo[0].row_words < geo[0].lanes;
        let w_resid: Vec<RoundRec> = if lane_major {
            let k_rot = w_rounds.len() - levels[0].fold_fins.len();
            let mut v = w_rounds[k_rot..].to_vec();
            v.extend_from_slice(&w_rounds[..k_rot]);
            v
        } else {
            w_rounds.to_vec()
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
        // ROUND 4: the recombination + f == g, replayed from located words
        // (the emitter binds these; until it landed they rode only this
        // constructor's scaffolding verify).
        let n_pub_slots_c = pin_recombination(
            lo.shape.circuit.cells(),
            n_log_i,
            &lo.public,
            &lo.proof.wiring.gather,
            &gammas_i,
            2,
            &vals_rec,
            &gkr_rec.r_pt,
            gkr_rec.fgs_v,
        );
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
        // Derived pd points (merged-open v1: they left the stream): the
        // element pair from the verifier's own claims, the gathers from
        // gate_claim_point at the GKR's row point — the same derivation the
        // verifier itself performs. Pinned against the round challenges the
        // emitter wires below.
        let pd_pts_n: Vec<Vec<F128>> = {
            let cells = lo.shape.circuit.cells();
            let el = claims.element.as_ref().expect("element claims");
            let mut v = vec![el.c_point.clone(), el.lc_point.clone()];
            for i2 in 0..n_gather {
                v.push(cells.gate_claim_point(i2, &gkr_rec.r_pt[..cells.nu()]));
            }
            v
        };
        for pt in &pd_pts_n {
            assert_eq!(pt.len(), n_log_i + k_cols_i, "pd point split");
        }
        // The element claims' coordinate wires: rows = the element zc rounds
        // [..nu], c's cols = zc rounds [nu..] then prefix bits, lc's cols =
        // the lc rounds REVERSED then prefix bits — pinned value-for-value.
        {
            let e_rounds = piop_i.zc_rounds.len();
            for j in 0..n_log_i {
                assert_eq!(pd_pts_n[0][j], chals[piop_i.zc_rounds[j].ch], "c row {j}");
                assert_eq!(pd_pts_n[1][j], chals[piop_i.zc_rounds[j].ch], "lc row {j}");
            }
            for j in 0..e_rounds - n_log_i {
                assert_eq!(
                    pd_pts_n[0][n_log_i + j],
                    chals[piop_i.zc_rounds[n_log_i + j].ch],
                    "c col {j}"
                );
            }
            let n_lc = piop_i.lc_rounds.len();
            for j in 0..n_lc {
                assert_eq!(
                    pd_pts_n[1][n_log_i + j],
                    chals[piop_i.lc_rounds[n_lc - 1 - j].ch],
                    "lc col {j}"
                );
            }
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
            n_pub_slots_c,
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
            pd_pts: pd_pts_n,
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
    /// Labeled `public_len` checkpoints through the emission — the publics
    /// census (`PUB_CENSUS=1` on the node test prints the block sizes).
    census: Vec<(&'static str, usize, usize)>,
    /// The z_skip squeeze wire — see [`ChildRegion::zskip_w`].
    zskip_w: Wire,
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
    /// The child's PUBLIC SEGMENT as witness wires — the app-statement
    /// plumbing (hash-chain adjacency) reads through these.
    child_pub_w: Vec<Wire>,
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
    consts: &mut Vec<(F128, Wire)>,
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
    let mut cen: Vec<(&'static str, usize, usize)> =
        vec![("start", sb.public_len(), sb.rows_in_slot(cs.macs))];
    let iv_w = pack8(&flock_prover::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [sb.public_input(), sb.public_input()];
    let (outs, ww) = emit_fs_chain(
        sb,
        cs.q.b3,
        iv2,
        trace,
        stream,
        &rt.bytes,
        vals,
        consts,
        &rt.pub_payloads,
    );

    cen.push(("chain payloads + shared consts", sb.public_len(), sb.rows_in_slot(cs.macs)));
    if std::env::var("PUB_CENSUS").is_ok() {
        let pay_pub: usize = stream
            .words
            .iter()
            .enumerate()
            .filter(|(wi, w)| {
                matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. }
                    if rt.pub_payloads[*payload])
                    && ww[*wi].is_some()
            })
            .count();
        println!(
            "  [census probe] chain block: {} payload words public, {} cw consts",
            pay_pub,
            consts.len()
        );
    }
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
    // child's public words themselves are witness, bound here. The returned
    // wires ARE the child's public segment — the recombination folds them.
    let pub_w = {
        let pays = payload_words(stream);
        assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
        let dw = [
            ww[pays[4][0]].expect("digest word wired"),
            ww[pays[4][1]].expect("digest word wired"),
        ];
        emit_publics_hash(sb, cs.q, iv2, &rt.lo.public, dw, vals, consts)
    };
    cen.push(("H(publics) region", sb.public_len(), sb.rows_in_slot(cs.macs)));
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
        consts,
        hints,
    );

    cen.push(("query phase decl", sb.public_len(), sb.rows_in_slot(cs.macs)));
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


    cen.push(("zero/one/anchor consts", sb.public_len(), sb.rows_in_slot(cs.macs)));
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

    cen.push(("merged target + family-H advice", sb.public_len(), sb.rows_in_slot(cs.macs)));
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

    cen.push(("spine + residual advice", sb.public_len(), sb.rows_in_slot(cs.macs)));
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

    // ---- ROUND 4: the recombination + f == g, in-circuit ----
    let le8 = match cs.le.iter().find(|&&(n, _)| n == 8) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(LeafEvalGate::new(8));
            cs.le.push((8, s));
            s
        }
    };
    let gather_w: Vec<Wire> = (0..rt.n_gather)
        .map(|i| wv(gammas_i[2 + i].val_v))
        .collect();
    emit_recombination(
        sb,
        macs,
        le8,
        &pub_w,
        &gather_w,
        &pt_w,
        n_log_i,
        rt.n_pub_slots_c,
        f_w,
        g_w,
        zw,
        ow,
    );

    cen.push(("GKR advice (g0s, mask)", sb.public_len(), sb.rows_in_slot(cs.macs)));
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

    cen.push(("element PIOP advice", sb.public_len(), sb.rows_in_slot(cs.macs)));
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
                (t_vals_b[k2], cw(sb, vals, consts, t_vals_b[k2]))
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
        // Bilinearity. Since merged-open v1 the pd points are DERIVED, not
        // absorbed: a gather's column point is constant address bits (the
        // one-hot statement data — nothing to bind in-circuit), and the
        // element pair's coordinates are the region PIOP's own squeeze
        // wires, pinned against rt.pd_pts in the constructor.
        let mut w_st = zw;
        let el_col_w = |j: usize, i2: usize| -> Wire {
            let coord = rt.pd_pts[i2][n_log_i + j];
            if coord == F128::ZERO {
                zw
            } else if coord == F128::ONE {
                ow
            } else if i2 == 0 {
                outs[trace.squeezes[piop_i.zc_rounds[n_log_i + j].fin][0]][0]
            } else {
                let n_lc = piop_i.lc_rounds.len();
                outs[trace.squeezes[piop_i.lc_rounds[n_lc - 1 - j].fin][0]][0]
            }
        };
        for &i2 in members {
            let pd = &gammas_i[i2];
            let gpd_w = outs[trace.squeezes[pd.fin][0]][0];
            let z_col_n = &rt.pd_pts[i2][n_log_i..n_log_i + k_cols_i];
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
                assert!(i2 >= 2, "one-hot columns are gather claims");
                let e = eqc_w[rt.run_of[h]];
                w_st = sb.gate(macs, &[w_st, gpd_w, e])[0];
            } else {
                let z_col_w: Vec<Wire> = (0..k_cols_i).map(|j| el_col_w(j, i2)).collect();
                let d = eq_dot(sb, &z_col_w);
                w_st = sb.gate(macs, &[w_st, gpd_w, d])[0];
            }
        }
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i {
                if members[0] >= 2 {
                    pt_w[layer]
                } else {
                    outs[trace.squeezes[piop_i.zc_rounds[layer].fin][0]][0]
                }
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

    cen.push(("multipoint + anchor expect advice", sb.public_len(), sb.rows_in_slot(cs.macs)));
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

    cen.push(("assertion eval advice", sb.public_len(), sb.rows_in_slot(cs.macs)));
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
    cen.push(("TAIL: query alphas + native accs", sb.public_len(), sb.rows_in_slot(cs.macs)));
    sb.publish(t_final);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(*w);
        }
    }
    cen.push(("TAIL: chain ends + residual accs", sb.public_len(), sb.rows_in_slot(cs.macs)));
    sb.publish(inner_w);
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    cen.push(("TAIL: sigma + GKR point", sb.public_len(), sb.rows_in_slot(cs.macs)));
    sb.publish(el_zr);
    sb.publish(el_lcw);
    sb.publish(anc_w);
    for w in &mat_pub {
        sb.publish(*w);
    }
    for w in &ela_pub {
        sb.publish(*w);
    }
    cen.push(("TAIL: el ends + assertion publics", sb.public_len(), sb.rows_in_slot(cs.macs)));
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
        + n_fam_pub;
    cen.push(("TAIL: family-H re-exposure", sb.public_len(), sb.rows_in_slot(cs.macs)));
    RealRegion {
        pub_base,
        n_query_pub,
        n_tail,
        n_mat_pub: mat_pub.len(),
        n_fam_pub,
        census: cen,
        zskip_w: outs[trace.squeezes[rt.zskip_fin][0]][0],
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
        child_pub_w: pub_w,
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
    // The family-H re-exposure block: the words the rs_half / V_rs advice
    // reference, all published — validated here against the proof's own
    // fields and the located challenges.
    let mut fq = ela_base + r.n_ela_pub;
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
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let region = emit_real_child_region(&mut sb, &mut cs, &rt, &mut vals, &mut hints, &mut consts);
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
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
    let mut built2 = shape2.run(&vals, &hint_refs);
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
        log_batch_size: pcs_batch(&union2),
        profile: LigeritoProfile::Fast,
        num_lanes: union2.commit_lanes(pcs_batch(&union2)),
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
            let z = match std::mem::replace(
                &mut built2.witnesses[shape2.registry_slot(sl)],
                SlotWitness::DeferredToRows,
            ) {
                SlotWitness::Element(z) => z,
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
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
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
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
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
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
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

/// **Chain-PoC step 0: a SINGLE-SLOT BOOLEAN union takes wiring.** The
/// hash-chain application's leaf is one b3 slot whose rows chain
/// cv_out(i) → cv_in(i+1) by copy constraints — no element slot anywhere.
/// mvp10's minimal wired inner carried a MacGate, leaving open whether the
/// circuit transport NEEDS an element presence; `sha256_binary_tree_circuit`
/// (circuit_wiring.rs) already answers "no" for the PLAIN verify. This probe
/// pins the recursion-relevant remainder on the chain shape itself:
/// blake3/blake3 (the recursable config), the DEFERRED verify — boolean
/// matrix work + sigma assertion out, element work `None` — both discharging
/// natively, and the chain-link tamper dying on the wiring product rather
/// than anything softer.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn chain_probe_boolean_only_wired_union() {
    let n_blocks = 256usize;
    let nu = 8usize;
    let mut rng = Rng(0xC4A1_0001);
    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let iv = pack8(&IV);
    let mut cv = [b.public_value(iv[0]), b.public_value(iv[1])];
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
    }
    b.publish(cv[0]);
    b.publish(cv[1]);
    let built = b.finish().expect("the chain circuit builds");

    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    assert!(!union.has_element(), "one boolean slot, no element class");
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
        merkle_hash: HashKind::Blake3,
    };
    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let prove = |rows: &[blake3::Compression]| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(rows, nu),
                blake_lc,
            )],
            Vec::new(),
            &mut ch,
        )
    };
    let (proof, commitment, _) = prove(built.rows::<Blake3Gate>(hash));

    // The deferred verify — what a first-level node runs per chain child.
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (_claims, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the deferred verify accepts an honest boolean-only chain");
    let matrix = work
        .boolean
        .expect("the b3 slot yields boolean matrix work");
    assert!(work.element.is_none(), "no element class, no element work");
    matrix
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges against the b3 matrices");
    assert!(
        sigma.check(&built.shape.circuit),
        "the sigma assertion discharges against the chain's own sigma table"
    );

    // The chain-link tamper: row 17 re-witnessed from a cv one bit off.
    // Every gate still satisfies the b3 relation on its own row; only the
    // copy constraints around row 17 break — the wiring product must be
    // what catches it.
    let mut bad: Vec<blake3::Compression> = built.rows::<Blake3Gate>(hash).to_vec();
    bad[17].0[0] ^= 1;
    let (p, cm, _) = prove(&bad);
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    assert!(
        matches!(
            verifier::verify_ligerito_union_circuit(
                &union,
                &built.shape.circuit,
                &built.witness.public,
                &lcs,
                &cm,
                &p,
                &pcs_params,
                &mut ch,
            ),
            Err(flock_core::verifier::VerifyError::Wiring(
                flock_core::circuit::WiringError::Gkr(
                    flock_core::product_gkr::VerifyError::ProductMismatch
                )
            ))
        ),
        "a broken chain link must die on the wiring product"
    );
}

// ---------------------------------------------------------------------------
// The hash-chain PoC's LEAF (task 2): the message-chain proof.
//
// h_{i+1} = compress(IV, h_i, counter = 0, block_len = 64,
//                    CHUNK_START | CHUNK_END | ROOT)
//
// — the full 64-byte output fed back as the next MESSAGE block (Ron's call:
// no truncation to a 32-byte cv between steps), cv pinned to the IV, the
// single-block-root flag flavor, so one step reads as a standalone
// blake3-of-64-bytes. One b3 slot, rows chained out(i) → m(i+1) by copy
// constraints (task 1 pinned that boolean-only unions take wiring); the
// statement is 11 words: iv (2) + params (1) + h_start (4) declared, h_end
// (4) published last — DECLARATION-ordered, so the public tail IS h_end.

/// The chain statement's flag word: a standalone single-block root.
const CHAIN_FLAGS: u32 = CHUNK_START | CHUNK_END | ROOT;

/// The native reference: `n_blocks` message-chain steps from `h_start`.
fn native_chain(h_start: &[u32; 16], n_blocks: usize) -> [u32; 16] {
    let mut h = *h_start;
    for _ in 0..n_blocks {
        h = blake3::blake3_compress(&IV, &h, 0, 64, CHAIN_FLAGS);
    }
    h
}

/// The chain circuit alone (shared by the honest builder and the tamper
/// legs): one b3 slot, message-chain wiring, the 11-word statement.
/// The chain circuit's SHAPE, separated from the statement it runs on.
/// The shape does not depend on `h_start` — that is the digest-determinism
/// pin — so a chain prover builds this ONCE and pays only the per-segment
/// walk afterwards. The split is also what makes a leaf's ONLINE cost
/// measurable: the walk is per-statement (it computes the chain and
/// materialises the rows), the shape is not.
struct ChainShape {
    shape: flock_core::circuit::builder::CircuitShape,
    hash: flock_core::circuit::builder::SlotId,
    nu: usize,
}

fn build_chain_shape(n_blocks: usize) -> ChainShape {
    let nu = n_blocks.trailing_zeros() as usize;
    assert_eq!(1usize << nu, n_blocks, "block count is a power of two");
    let mut sb = ShapeBuilder::new(nu);
    let hash = sb.slot(Blake3Gate { nu });
    let cv = [sb.public_input(), sb.public_input()];
    let params = sb.public_input();
    let mut m: Vec<Wire> = (0..4).map(|_| sb.public_input()).collect();
    let mut out = Vec::new();
    for _ in 0..n_blocks {
        let mut hash_in = vec![cv[0], cv[1]];
        hash_in.extend_from_slice(&m);
        hash_in.push(params);
        out = sb.gate(hash, &hash_in);
        m = out.clone();
    }
    for w in &out {
        sb.publish(*w);
    }
    ChainShape {
        shape: sb.finish().expect("the chain circuit builds"),
        hash,
        nu,
    }
}

/// The statement a chain shape runs on, in declaration order: the IV pair,
/// the params word, then `h_start`. (`h_end` is PUBLISHED, so it is the
/// walk's output, not an input.)
fn chain_vals(h_start: &[u32; 16]) -> Vec<F128> {
    let iv = pack8(&IV);
    let mut v = vec![iv[0], iv[1], pack_params(0, 64, CHAIN_FLAGS)];
    v.extend((0..4).map(|j| pack4(h_start[4 * j..4 * j + 4].try_into().unwrap())));
    v
}

fn build_chain_circuit(
    h_start: &[u32; 16],
    n_blocks: usize,
) -> (
    flock_core::circuit::builder::BuiltCircuit,
    flock_core::circuit::builder::SlotId,
) {
    let cs = build_chain_shape(n_blocks);
    let witness = cs.shape.run(&chain_vals(h_start), &[]);
    (
        flock_core::circuit::builder::BuiltCircuit {
            shape: cs.shape,
            witness,
        },
        cs.hash,
    )
}

/// The chain-PoC leaf, end to end: FAST profile (the B-fast decision),
/// BLAKE3 for Merkle and FS (recursable), proven over the circuit path
/// with NO element slots, deferred-verified with both assertion families
/// discharged, and h_end cross-checked against the native chain. The
/// [`MixedInner`] embedding is deliberate: a chain proof is a circuit
/// proof, so [`ChildTape`] consumes it directly (element side `None`).
struct ChainProof {
    inner: MixedInner,
    h_start: [u32; 16],
    h_end: [u32; 16],
    /// What the leaf cost, split SETUP vs ONLINE — see [`Online`].
    t: Online,
}

/// A chain leaf. The SHAPE build is per-shape setup (statement-independent
/// — the digest pin), the WALK is per-statement and is the chain compute
/// itself, so it is reported apart from the proving phases.
fn build_chain_proof(h_start: [u32; 16], n_blocks: usize) -> ChainProof {
    let t_shape = std::time::Instant::now();
    let cs = build_chain_shape(n_blocks);
    let shape_ms = t_shape.elapsed().as_secs_f64() * 1e3;
    let (nu, hash) = (cs.nu, cs.hash);
    let t0 = std::time::Instant::now();
    let witness = cs.shape.run(&chain_vals(&h_start), &[]);
    let walk_ms = t0.elapsed().as_secs_f64() * 1e3;
    let built = flock_core::circuit::builder::BuiltCircuit {
        shape: cs.shape,
        witness,
    };

    let t_setup = std::time::Instant::now();
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    assert!(!union.has_element(), "a chain proof is boolean-only");
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
        merkle_hash: HashKind::Blake3,
    };
    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let setup_ms = shape_ms + t_setup.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    let wit = blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu);
    let witgen_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(wit, blake_lc)],
        Vec::new(),
        &mut ch,
    );
    let prove_ms = t2.elapsed().as_secs_f64() * 1e3;

    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (_claims, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the deferred verify accepts an honest chain proof");
    work.boolean
        .as_ref()
        .expect("the b3 slot yields boolean matrix work")
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges");
    assert!(work.element.is_none(), "no element class, no element work");
    assert!(sigma.check(&built.shape.circuit), "sigma discharges");

    // The statement is bound end to end: publics[3..7] are h_start, the
    // published tail is h_end, and h_end equals the native chain.
    let h_end = native_chain(&h_start, n_blocks);
    let public = &built.witness.public;
    for j in 0..4 {
        assert_eq!(
            public[3 + j],
            pack4(h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "public word {} is h_start[{}]",
            3 + j,
            j
        );
        assert_eq!(
            public[public.len() - 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "the published tail is the native h_end"
        );
    }

    ChainProof {
        inner: MixedInner {
            nu,
            built,
            proof,
            commitment,
            pcs: pcs_params,
            work,
            sigma,
        },
        h_start,
        h_end,
        t: Online {
            setup_ms,
            walk_ms,
            witgen_ms,
            prove_ms,
            ..Online::default()
        },
    }
}

/// **Task 2's pin: the message-chain leaf, honest + the tamper matrix.**
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn chain_proof_message_chain_roundtrip_and_tampers() {
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0002);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());

    // Honest: build_chain_proof internally deferred-verifies, discharges
    // both assertion families and cross-checks h_end against the native
    // chain. Determinism of the statement: a second build from the same
    // h_start yields the same h_end.
    let cp = build_chain_proof(h_start, n_blocks);
    assert_eq!(cp.h_end, native_chain(&h_start, n_blocks));
    assert_eq!(cp.inner.nu, 8);

    // The tamper legs run on a fresh circuit build (the honest one's rows,
    // modified), against the PLAIN verifier so every check is in force.
    let (built, hash) = build_chain_circuit(&h_start, n_blocks);
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let blake_r1cs = blake3::build_block_r1cs(cp.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let prove = |rows: &[blake3::Compression], public: &[F128]| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            public,
            &cp.inner.pcs,
            vec![UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(rows, cp.inner.nu),
                blake_lc,
            )],
            Vec::new(),
            &mut ch,
        )
    };
    let verify = |public: &[F128],
                  cm: &flock_core::pcs::commit::Commitment,
                  p: &flock_core::proof::R1csProofCircuitMerged| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            public,
            &lcs,
            cm,
            p,
            &cp.inner.pcs,
            &mut ch,
        )
    };

    // (a) A broken chain link: row 100 re-witnessed from a message one bit
    //     off. Its own b3 relation holds; the copy constraint out(99) ==
    //     m(100) breaks, and the wiring product is what must catch it.
    {
        let mut bad: Vec<blake3::Compression> = built.rows::<Blake3Gate>(hash).to_vec();
        bad[100].1[0] ^= 1;
        let (p, cm, _) = prove(&bad, &built.witness.public);
        assert!(
            matches!(
                verify(&built.witness.public, &cm, &p),
                Err(flock_core::verifier::VerifyError::Wiring(
                    flock_core::circuit::WiringError::Gkr(
                        flock_core::product_gkr::VerifyError::ProductMismatch
                    )
                ))
            ),
            "a broken chain link must die on the wiring product"
        );
    }

    // (b) A tampered STATEMENT word — h_start (public 3) and h_end (the
    //     tail): the honest proof must not verify against it, and a prover
    //     honestly proving the tampered statement must be rejected too.
    let plen = built.witness.public.len();
    for i in [3usize, plen - 1] {
        let mut bad = built.witness.public.clone();
        bad[i] += F128::ONE;
        assert!(
            verify(&bad, &cp.inner.commitment, &cp.inner.proof).is_err(),
            "statement word {i} must be bound to the transcript"
        );
        let (p, cm, _) = prove(built.rows::<Blake3Gate>(hash), &bad);
        assert!(
            verify(&bad, &cm, &p).is_err(),
            "statement word {i} must be enforced by the wiring"
        );
    }

    // (c') see chain_tape_regions_pinned for the tape-side continuation.
    // (c) Shape determinism — the foldability key. The chain circuit is
    //     h_start-INDEPENDENT (h_start only moves public VALUES), so every
    //     segment of a long chain proves against ONE circuit digest, and
    //     the accumulator folds their assertions under one sigma key.
    assert_eq!(cp.h_start, h_start);
    assert_eq!(
        cp.inner.built.shape.circuit.digest(),
        built.shape.circuit.digest(),
        "two builds from the same h_start agree"
    );
    let other_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let (other, _) = build_chain_circuit(&other_start, n_blocks);
    assert_eq!(
        cp.inner.built.shape.circuit.digest(),
        other.shape.circuit.digest(),
        "a DIFFERENT segment's chain circuit is digest-equal"
    );
    // The deferred work/sigma are verifier-exported references: both must
    // discharge against the REBUILT circuit too (same digest, same tables).
    cp.inner.work
        .boolean
        .as_ref()
        .expect("boolean matrix work travels with the chain proof")
        .check(&union, &lcs)
        .expect("the exported matrix work discharges against the rebuild");
    assert!(
        cp.inner.sigma.check(&built.shape.circuit),
        "the exported sigma discharges against the rebuild's sigma table"
    );
}

/// **Task 3: the chain tape.** [`ChildTape::new`] — the SAME constructor the
/// merge machinery instantiates per mixed child — walks the hash-chain
/// leaf's tape with the element side `None`. Its class-agnostic pins all
/// re-assert on the boolean-only shape: the duplex chain trace, the GKR
/// walk + masked input checks, rs×2, the pd census, the R=2+P schedule
/// replaying to anchor.v, the W rounds, the stratified query geometry, the
/// spine/residual natives, the recombination, and the full anchor-expect
/// replica. This test adds the chain-shape facts on top.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn chain_tape_regions_pinned() {
    let mut rng = Rng(0xC4A1_0003);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let n_blocks = 256usize;
    let cp = build_chain_proof(h_start, n_blocks);
    let ct = ChildTape::new(&cp.inner, DOMAIN);

    // The boolean-only shape facts.
    assert!(ct.el.is_none(), "no element PIOP region on a chain tape");
    assert!(ct.el_assert.is_none(), "no element assertion travels");
    assert_eq!(
        ct.n_pd,
        cp.inner.proof.wiring.gather.len(),
        "the pd claims are the wiring gathers ONLY"
    );
    assert!(ct.n_p > 0, "the gathers form scalar groups");
    assert_eq!(
        ct.sigma_native.value, cp.inner.sigma.value,
        "the tape's sigma reference is the deferred verify's"
    );
    assert_eq!(
        ct.bool_assert.target,
        cp.inner
            .work
            .boolean
            .as_ref()
            .expect("boolean work")
            .target,
        "the tape's boolean reference is the deferred verify's"
    );

    let union = UnionInstance::new(
        &cp.inner.built.shape.registry,
        cp.inner.built.shape.counts.clone(),
    );
    println!(
        "\nCHAIN TAPE (boolean-only wired leaf, message-chain)\n  \
         inner: nu {} | dense_m {} | pd claims {} (ALL gathers) | P {} | mu {}\n  \
         GKR layers {} | b3 rows (tape model) {} | L0 lanes {} x {} words\n",
        cp.inner.nu,
        union.dense_m(),
        ct.n_pd,
        ct.n_p,
        ct.mu_i,
        cp.inner.proof.wiring.gkr.layers.len(),
        ct.b3_rows,
        ct.geo[0].lanes,
        ct.geo[0].row_words,
    );
}

/// Bisect probe: ONE chain child region alone — emit, run, check.
#[test]
#[ignore]
fn chain_child_region_emits_alone() {
    let mut rng = Rng(0xC4A1_0005);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp = build_chain_proof(h_start, 256);
    let ct = ChildTape::new(&cp.inner, DOMAIN);
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let region = emit_child_region(&mut sb, &mut cs, &ct, &mut vals, &mut hints, &mut consts);
    let shape2 = sb.finish().expect("the chain child circuit builds");
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_child_region(&built2.public, &ct, &region);
    assert_eq!(
        region.pub_base + consumed,
        built2.public.len(),
        "the region's publics are the whole tail"
    );
}

/// The first-level node as a BUILDER: [`build_fl_node`]'s output. `lo` is
/// a real, RECURSABLE [`LeafOuter`] (BLAKE3 for both the FS chain and the
/// Merkle trees), so the internal-node machinery ([`RealTape`],
/// [`build_node_outer`]) consumes it exactly like a leaf outer; `acc` is
/// the folded chain accumulator the node carries up; `stmt_base` locates
/// the 8-word application-statement block (h_start, h_end) in `lo.public`.
struct FlNode {
    lo: LeafOuter,
    acc: flock_core::aggregate::Accumulator,
    stmt_base: usize,
    /// The published fold blocks' base: per group `[rho_col | rho_row |
    /// value]` — the accumulator claims a PARENT's lane fold connects to
    /// wire-to-wire (a prior's surface IS this published block).
    fold_pub_base: usize,
    h_start: [u32; 16],
    h_end: [u32; 16],
    /// What the FL cost, split SETUP vs ONLINE — see [`Online`].
    /// Everything else in the builder is pin/check scaffolding.
    t: Online,
}

/// **THE FIRST-LEVEL NODE.** Two ADJACENT chain proofs (the right segment
/// starts at the left's h_end) verified deferred in ONE outer circuit —
/// two chain-tape regions on shared slots — with their boolean + sigma
/// assertions folded 2→1 in-circuit (THREE fold groups; the chain class
/// has no element side), THE ADJACENCY as a wire-to-wire copy constraint
/// between the children's endpoint publics, and the combined span
/// (h_start_left, h_end_right) published as the node's own application
/// statement. The accumulator reassembles from the public segment alone
/// and discharges both groups. Every pin stays inside the builder (the
/// mvp9 precedent: the builder IS the test). Envelope snapping is the
/// scale step's job — this is the m22 dev shape.
fn build_fl_node(cp0: &ChainProof, cp1: &ChainProof) -> FlNode {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const FL_DOMAIN: &[u8] = b"flock-chain-fl-node-v0";

    // The right child CONTINUES the chain: its h_start IS the left's h_end.
    assert_eq!(cp1.h_start, cp0.h_end, "the segments are adjacent");
    assert_eq!(
        cp0.inner.built.shape.circuit.digest(),
        cp1.inner.built.shape.circuit.digest(),
        "one chain circuit digest, every segment"
    );

    let registry = &cp0.inner.built.shape.registry;
    assert_eq!(registry.num_boolean(), 1, "one boolean type (blake3)");
    assert!(
        registry.element_types().is_empty(),
        "the chain class has no element side"
    );
    let bool_asserts = [
        cp0.inner.work.boolean.clone().expect("left boolean work"),
        cp1.inner.work.boolean.clone().expect("right boolean work"),
    ];
    let sigmas = [cp0.inner.sigma.clone(), cp1.inner.sigma.clone()];

    // ---- the native fold: boolean + sigma, NO element groups, NO priors ----
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let el_mats: [flock_core::aggregate::ElementMatrices; 0] = [];
    let el_asserts: [(
        &UnionInstance<'_>,
        flock_core::element_r1cs::union::ElementAssertion,
    ); 0] = [];
    let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let mut chp = FsChallenger::with_chained_blake3(FL_DOMAIN);
    let (agg, acc_p) = aggregate::prove_aggregate_classes(
        registry,
        &mats,
        &circs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        Some((&cp0.inner.built.shape.circuit, &sigmas)),
        &[],
        &mut chp,
    )
    .expect("the first-level fold proves");
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(FL_DOMAIN));
    let acc_v = aggregate::verify_aggregate_classes(
        registry,
        &bool_asserts,
        &el_asserts,
        Some((&cp0.inner.built.shape.circuit, &sigmas)),
        &[],
        &agg,
        &mut rec,
    )
    .expect("the first-level fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.per_element.is_empty(), "no element group accumulated");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_sigma(&cp0.inner.built.shape.circuit),
        "the sigma group discharges against the ONE chain circuit"
    );

    // The three folds' claim lists — no priors, so [fresh, fresh] each.
    let n_priors = 0usize;
    let bc: Vec<_> = bool_asserts.iter().map(|a| a.claims(registry)).collect();
    let fold_claims: Vec<Vec<MatrixClaim>> = vec![
        vec![bc[0][0].0.clone(), bc[1][0].0.clone()],
        vec![bc[0][0].1.clone(), bc[1][0].1.clone()],
        vec![sigmas[0].claim(), sigmas[1].claim()],
    ];
    let fold_proofs: Vec<&FoldProof> = vec![
        &agg.folds[0].0,
        &agg.folds[0].1,
        agg.sigma_fold.as_ref().expect("the sigma fold rides along"),
    ];
    assert_eq!(fold_claims[0][0].row.low.len(), 64, "fresh lagrange low");
    assert_eq!(fold_claims[0][0].col.low.len(), 64, "fresh z_partial low");
    assert_eq!(fold_claims[2][0].row.low.len(), 1, "sigma claims are eq");

    // ---- the fold tape, pinned op-for-op ----
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
    assert_eq!(ops, want.as_slice(), "the first-level fold tape shape");
    assert_eq!(rec.payloads()[0], registry.digest(), "bind: registry digest");
    assert_eq!(rec.payloads()[1], vec![0u8], "bind: prior count 0");
    let locs = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    assert_eq!(outs[0], acc_v.per_type[0].0, "boolean A accumulator");
    assert_eq!(outs[1], acc_v.per_type[0].1, "boolean B accumulator");
    let (sig_digest, sig_claim) = acc_v.sigma.as_ref().expect("sigma accumulated");
    assert_eq!(outs[2], *sig_claim, "sigma accumulator");
    assert_eq!(
        *sig_digest,
        cp0.inner.built.shape.circuit.digest(),
        "sigma keys by the chain circuit digest"
    );

    // ---- the child tapes ----
    let t0 = ChildTape::new(&cp0.inner, DOMAIN);
    let t1 = ChildTape::new(&cp1.inner, DOMAIN);
    assert!(t0.el.is_none() && t1.el.is_none(), "chain children");
    let tape_verify_ms = t0.verify_ms + t1.verify_ms;

    // ---- the outer: TWO chain-tape regions + the fold region + adjacency ----
    {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let stream = t_shape.stream_words_duplex(FL_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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

        let b3_rows = t0.b3_rows + t1.b3_rows + trace.rows.len();
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        // THE ENVELOPE (task 7b): a first-level node is an internal node's
        // CHILD, so its proof must carry the same geometry every other
        // envelope outer does — nu*, the canonical type set at counts*, the
        // padded public segment and the m* dense floor. Then a parent's walk
        // over an FL child is row-identical to its walk over an internal
        // child, which is what makes ONE internal circuit serve every level.
        let env = envelope_shape();
        let nu2 = match &env {
            Some(e) => {
                assert!(
                    nu2_content <= e.nu,
                    "FL content nu {nu2_content} exceeds the envelope nu* {}",
                    e.nu
                );
                e.nu
            }
            None => nu2_content,
        };
        let t_build = std::time::Instant::now();
        let mut sb = ShapeBuilder::new(nu2);
        let spread_own2 = t0.spread_w.max(t1.spread_w);
        let (spread_w2, mut cs) = match &env {
            Some(e) => {
                assert!(
                    spread_own2 <= e.spread_w,
                    "chain-child ladder depth {spread_own2} exceeds the envelope spread width {}",
                    e.spread_w
                );
                (e.spread_w, ChildSlots::new_env(&mut sb, nu2, e))
            }
            None => (spread_own2, ChildSlots::new(&mut sb, nu2, spread_own2)),
        };
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        // The two chain-child regions are independent gate subgraphs (each
        // reads only its own tape's inputs; the fold region joins them
        // AFTER), so they are declared as islands and the fill plan
        // evaluates them concurrently. A cross-island read fails plan
        // compilation — the independence is checked, not assumed.
        let isl0 = sb.begin_island();
        let r0 = emit_child_region(&mut sb, &mut cs, &t0, &mut vals, &mut hints, &mut consts);
        sb.end_island(isl0);
        let isl1 = sb.begin_island();
        let r1 = emit_child_region(&mut sb, &mut cs, &t1, &mut vals, &mut hints, &mut consts);
        sb.end_island(isl1);
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

        // ---- the connects: fold surfaces == child-region wires ----
        use flock_core::field::PHI_8_TABLE;
        use flock_core::zerocheck::K_SKIP;
        use flock_core::zerocheck::multilinear::{
            lagrange_weights_naive, subspace_denominator_pair,
        };
        let lam_base = sb.public_len();
        let lam_w: Vec<Wire> = PHI_8_TABLE[..1 << K_SKIP]
            .iter()
            .map(|&v| {
                vals.push(v);
                sb.public_input()
            })
            .collect();
        vals.push(subspace_denominator_pair(K_SKIP).1);
        let deninv_w = sb.public_input();
        vals.push(F128::ZERO);
        let lag_zassert = sb.public_input();
        let tapes = [&t0, &t1];
        let regions = [&r0, &r1];
        for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
            assert_eq!(
                &fold_claims[0][n_priors + k].row.low[..],
                &lagrange_weights_naive(K_SKIP, tk.chals[tk.zskip_ch])[..],
                "child {k}: the fold's lagrange lows are the closed form"
            );
            let lows = emit_lagrange_lows(
                &mut sb,
                cs.macs,
                &lam_w,
                deninv_w,
                rk.zskip_w,
                tk.chals[tk.zskip_ch],
                &mut vals,
                zw,
                ow,
                lag_zassert,
            );
            for (j, &lw2) in lows.iter().enumerate() {
                sb.connect(lw2, wv(locs[0].claims[n_priors + k].row_low_v + j));
            }
            // Native pre-asserts, then the wire connects — sigma fully,
            // boolean points + z_partial lows.
            let nu_c = tk.sigma_native.nu;
            assert_eq!(
                &fold_claims[2][n_priors + k].row.point[..],
                &tk.sigma_native.rho[..nu_c],
                "sigma row point is the child's rho[..nu]"
            );
            assert_eq!(
                &fold_claims[2][n_priors + k].col.point[..],
                &tk.sigma_native.rho[nu_c..],
                "sigma col point is the child's rho[nu..]"
            );
            assert_eq!(
                fold_claims[2][n_priors + k].value, tk.sigma_native.value,
                "sigma value is the child's deferred evaluation"
            );
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
            let cl = &locs[2].claims[n_priors + k];
            sb.connect(wv(cl.row_low_v), ow);
            sb.connect(wv(cl.col_low_v), ow);
            for j in 0..cl.row_pt_n {
                sb.connect(wv(cl.row_pt_v + j), rk.pt_w[j]);
            }
            for j in 0..cl.col_pt_n {
                sb.connect(wv(cl.col_pt_v + j), rk.pt_w[cl.row_pt_n + j]);
            }
            sb.connect(wv(cl.value_v), rk.sig_w);
            // boolean A/B: batch-major x_inner_rest mapping, rr reversed,
            // z_partial word-for-word.
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
            // Fold B's lagrange lows are fold A's — one published copy
            // binds both.
            for j in 0..locs[0].claims[n_priors + k].row_low_n {
                sb.connect(
                    wv(locs[1].claims[n_priors + k].row_low_v + j),
                    wv(locs[0].claims[n_priors + k].row_low_v + j),
                );
            }
        }

        // ---- THE ADJACENCY: left h_end == right h_start, wire to wire ----
        // The chain statement is 11 words: [iv0, iv1, params, h_start x4 |
        // h_end x4 published last]. Both children's publics are witness
        // wires here, so adjacency is four copy constraints, and the node's
        // own application statement is the combined span.
        let p0 = r0.child_pub_w.len();
        let p1 = r1.child_pub_w.len();
        assert_eq!(p0, 11, "the chain statement is 11 words");
        assert_eq!(p1, 11, "both children share the statement shape");
        for j in 0..4 {
            sb.connect(r0.child_pub_w[p0 - 4 + j], r1.child_pub_w[3 + j]);
        }

        // THE INHERITABLE ACCUMULATOR: per fold the deltas + the claim
        // `[rho_col | rho_row | value]`. This is the surface a PARENT's
        // chain lane connects to as its priors, so under the envelope it
        // rides the reserved ACC_CHAIN block (the FL folds the CHAIN
        // registry) — a constant index, the same one at which an internal
        // child exposes its own lane's claims. Off-envelope it publishes
        // inline, as before.
        let mut acc_chain_w: Vec<Wire> = Vec::new();
        for fp in &fold_pubs {
            acc_chain_w.extend_from_slice(&fp.rho_col);
            acc_chain_w.extend_from_slice(&fp.rho_row);
            acc_chain_w.push(fp.value);
        }
        let fold_pub_base = match &env {
            Some(e) => env_acc_chain_base(e),
            None => {
                let b = sb.public_len();
                for &w in &acc_chain_w {
                    sb.publish(w);
                }
                b
            }
        };
        // The value-binding publics stay in the BODY: nothing above reads
        // them, they only bind the claim values this outer folded.
        for k in 0..2 {
            sb.publish(wv(locs[0].claims[n_priors + k].value_v));
            sb.publish(wv(locs[1].claims[n_priors + k].value_v));
        }
        // THE APPLICATION STATEMENT: the combined span (the left child's
        // h_start, the right child's h_end). counts* + publics*: an FL node
        // declares the same count vector and segment length every other
        // envelope outer does, and both the app block and the accumulator
        // claims ride the envelope's fixed TAIL.
        let app_w: Vec<Wire> = (0..4)
            .map(|j| r0.child_pub_w[3 + j])
            .chain((0..4).map(|j| r1.child_pub_w[p1 - 4 + j]))
            .collect();
        let stmt_base = match &env {
            Some(e) => {
                pad_envelope_counts(
                    &mut sb,
                    &cs.q,
                    &cs.env_cache(),
                    e,
                    zw,
                    &mut hints,
                    &mut vals,
                    &EnvTail {
                        acc_chain: &acc_chain_w,
                        app: &app_w,
                        ..EnvTail::default()
                    },
                );
                env_app_base(e)
            }
            None => {
                let b = sb.public_len();
                for &w in &app_w {
                    sb.publish(w);
                }
                b
            }
        };
        let shape2 = sb.finish().expect("the first-level node circuit builds");
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
        // THE INDEX-FILL RUNNER (setup), the node's path: compile the plan,
        // then pin it row-identical against the generic walk before the
        // online run trusts it. run() stays the differential oracle — this
        // pin is what keeps it one, now that the FL no longer walks in the
        // timed path either.
        let fill_plan = shape2.fill_plan();
        {
            let walk = shape2.run(&vals, &hint_refs);
            let fill = shape2.run_filled(&fill_plan, &vals, &hint_refs);
            assert_eq!(walk.public, fill.public, "fill plan: public segment");
            assert_eq!(walk.witnesses, fill.witnesses, "fill plan: slot witnesses");
            assert_eq!(
                walk.rows::<Blake3Gate>(cs.q.b3),
                fill.rows::<Blake3Gate>(cs.q.b3),
                "fill plan: b3 rows"
            );
            assert_eq!(
                walk.rows::<SwapGate>(cs.q.swap),
                fill.rows::<SwapGate>(cs.q.swap),
                "fill plan: swap rows"
            );
            assert_eq!(
                walk.rows::<BitSpreadGate>(cs.q.spread),
                fill.rows::<BitSpreadGate>(cs.q.spread),
                "fill plan: spread rows"
            );
        }
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
        let t_run = std::time::Instant::now();
        // DEFERRED: rows and publics only — the element witnesses are never
        // packed, and the assembly below feeds the prover from the rows.
        let mut built2 = shape2.run_filled_deferred(&fill_plan, &vals, &hint_refs);
        let run_ms = t_run.elapsed().as_secs_f64() * 1e3;

        // Child checkers (each child's whole deferred-verifier statement
        // against its own native replicas), then the fold checker + the
        // accumulator reassembled from publics, then the app statement.
        let consumed0 = check_child_region(&built2.public, &t0, &r0);
        let consumed1 = check_child_region(&built2.public, &t1, &r1);
        assert!(
            r0.pub_base + consumed0 <= r1.pub_base && r1.pub_base + consumed1 <= fold_pub_base,
            "the regions' public blocks are disjoint and ordered"
        );
        let rebuilt = check_fold_publics(&built2.public, fold_pub_base, &locs, &alpha_recs);
        for (r, o) in rebuilt.iter().zip(&outs) {
            assert_eq!(r, o, "published fold output == located native output");
        }
        let acc_pub = aggregate::Accumulator {
            registry_digest: registry.digest(),
            per_type: vec![(rebuilt[0].clone(), rebuilt[1].clone())],
            per_element: Vec::new(),
            sigma: Some((
                cp0.inner.built.shape.circuit.digest(),
                rebuilt[2].clone(),
            )),
        };
        assert_eq!(
            acc_pub, acc_v,
            "the Accumulator, reassembled from the public segment alone"
        );
        assert!(
            acc_pub.discharge(&mats)
                && acc_pub.discharge_sigma(&cp0.inner.built.shape.circuit),
            "the public-segment accumulator discharges both groups"
        );
        for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
            assert_eq!(built2.public[lam_base + i], v, "λ const {i}");
        }
        // THE APPLICATION STATEMENT: the published span is (h_start of the
        // left chain, h_end of the right) — the 512-step combined segment.
        for j in 0..4 {
            assert_eq!(
                built2.public[stmt_base + j],
                pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
                "node statement: h_start is the left child's"
            );
            assert_eq!(
                built2.public[stmt_base + 4 + j],
                pack4(cp1.h_end[4 * j..4 * j + 4].try_into().unwrap()),
                "node statement: h_end is the right child's"
            );
        }
        assert_eq!(
            cp1.h_end,
            native_chain(
                &cp0.h_start,
                cp0.inner.built.shape.counts[0] + cp1.inner.built.shape.counts[0]
            ),
            "the combined span IS the concatenated chain"
        );

        // The outer proves and verifies over the circuit path — BLAKE3 for
        // BOTH the Merkle trees and the FS chain, so the node is RECURSABLE
        // (an internal node's RealTape walks this transcript in-circuit;
        // the defaults diverge silently — both recorded gotchas).
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = tower_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: union2.commit_lanes(pcs_batch_for(&union2, pf)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_ty2 = BitSpreadTable::new(spread_w2);
        let spread_r1cs2 = spread_ty2.build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        // Everything from here to the prove is WITNESS ASSEMBLY — packing
        // the walk's rows into the union's slot inputs. It is per-statement
        // (online), so it gets its own timer rather than hiding inside the
        // shape build or the prove.
        let t_asm = std::time::Instant::now();
        // THE COPY-FREE ASSEMBLY, the node's path: the boolean drivers pack
        // straight into the union's slot blocks inside the prove (live rows
        // only under elide) — no capacity-sized intermediates, no memcpy.
        // The rows are hoisted to owned Vecs because the closures must be
        // Send and `built2.rows` hands out `dyn Any`-backed borrows.
        let b3_rows2 = built2.rows::<Blake3Gate>(cs.q.b3).to_vec();
        let swap_rows2 = built2.rows::<SwapGate>(cs.q.swap).to_vec();
        let spread_rows2 = built2.rows::<BitSpreadGate>(cs.q.spread).to_vec();
        let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape2.registry_slot(cs.q.b3),
                UnionSlotProverInput::in_place(
                    move |dst| {
                        blake3::generate_witness_batch_major_partial_into(&b3_rows2, nu2, dst)
                    },
                    b3_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.swap),
                UnionSlotProverInput::in_place(
                    move |dst| SwapTable::generate_witness_batch_major_into(&swap_rows2, dst),
                    swap_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.spread),
                UnionSlotProverInput::in_place(
                    move |dst| spread_ty2.generate_witness_batch_major_into(&spread_rows2, dst),
                    spread_lc2,
                ),
            ),
        ];
        bslots.sort_by_key(|(i, _)| *i);
        // Element inputs straight from the slots' rows: the run was
        // DEFERRED, so the full-capacity packed intermediate never exists —
        // the prove's in_place closure scatters the live rows directly.
        let mut el_ord: Vec<(usize, Vec<Vec<F128>>)> = cs
            .element_slot_ids()
            .into_iter()
            .map(|sl| {
                (
                    shape2.registry_slot(sl),
                    built2.take_rows_of::<Vec<F128>>(sl),
                )
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(_, rows)| live_element_input_from_rows(rows, nu2))
            .collect();
        let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (shape2.registry_slot(cs.q.b3), b3_lc2),
            (shape2.registry_slot(cs.q.swap), swap_lc2),
            (shape2.registry_slot(cs.q.spread), spread_lc2),
        ];
        lco.sort_by_key(|(i, _)| *i);
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lco.into_iter().map(|(_, c)| c).collect();
        let asm_ms = t_asm.elapsed().as_secs_f64() * 1e3;
        let t_prove = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
        let (oproof, ocommit, _) = prover::prove_fast_ligerito_union_circuit(
            &union2,
            &shape2.circuit,
            &built2.public,
            &pcs2,
            bslots.into_iter().map(|(_, x)| x).collect(),
            el_inputs,
            &mut ch2,
        );
        let prove_ms = t_prove.elapsed().as_secs_f64() * 1e3;
        let t_ver = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
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
        .expect("the first-level node verifies over the circuit path");
        let verify_ms2 = t_ver.elapsed().as_secs_f64() * 1e3;
        let (b3_ri, swap_ri, spread_ri) = (
            shape2.registry_slot(cs.q.b3),
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
        );
        FlNode {
            lo: LeafOuter {
                shape: shape2,
                public: built2.public,
                proof: oproof,
                commitment: ocommit,
                pcs: pcs2,
                b3_r1cs: b3_r1cs2,
                swap_r1cs: swap_r1cs2,
                spread_r1cs: spread_r1cs2,
                b3_slot: b3_ri,
                swap_slot: swap_ri,
                spread_slot: spread_ri,
            },
            acc: acc_pub,
            stmt_base,
            fold_pub_base,
            h_start: cp0.h_start,
            h_end: cp1.h_end,
            t: Online {
                setup_ms: build_ms,
                tapes_ms: tape_verify_ms,
                walk_ms: run_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms: verify_ms2,
            },
        }
    }
}

/// **The first-level node's pin, through the builder** (converted-first:
/// the test IS [`build_fl_node`]'s original body; every assert lives inside
/// the builder now, the wrapper re-checks the statement surface).
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn first_level_node_two_chains_fold_and_adjacency() {
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0004);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(h0, n_blocks);
    let cp1 = build_chain_proof(cp0.h_end, n_blocks);
    let fl = build_fl_node(&cp0, &cp1);
    assert_eq!(fl.h_start, cp0.h_start);
    assert_eq!(fl.h_end, cp1.h_end);
    for j in 0..4 {
        assert_eq!(
            fl.lo.public[fl.stmt_base + j],
            pack4(fl.h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "the statement block reads h_start out of the public segment"
        );
        assert_eq!(
            fl.lo.public[fl.stmt_base + 4 + j],
            pack4(fl.h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "the statement block reads h_end out of the public segment"
        );
    }
    println!(
        "\nFIRST-LEVEL NODE (two adjacent chain proofs, fold + adjacency)\n  \
         chain: {} + {} compressions | node statement: h_start .. H^{}(h_start)\n  \
         outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        n_blocks,
        n_blocks,
        2 * n_blocks,
        fl.lo.shape.circuit.cells().nu(),
        fl.lo.shape.circuit.cells().mu(),
        fl.lo.public.len(),
        bincode::serialize(&fl.lo.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
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

/// **THE RECOMBINATION (round 4), pinned natively.** The wiring verifier's
/// `ŵ(ρ) = Σ_gate eq_slot[ι]·gather[ι] + Σ_public eq_slot[ι]·⟨eq_row, slot⟩`
/// (`circuit.rs` verify_wiring_core) is the ONE check that reads the child's
/// publics — and, with `f_eval == g_eval` beside it, was enforced only by the
/// tape constructors' scaffolding-tier native verify, never by the parent's
/// statement. This replica recomputes both from LOCATED tape words — the
/// gather pd values, the child's public segment, the GKR squeeze point — so
/// the emission has a pinned reference for every wire it binds. Also pins the
/// pd-claim order the emitter indexes: `[element c, element lc, gathers in
/// cell-slot enumeration order]`.
///
/// Returns `num_public_slots` (the emitters derive everything else from the
/// gather count and `n_log_i`; `cells.nu()` is asserted against it here).
fn pin_recombination(
    cells: &flock_core::circuit::CellSpace,
    n_log_i: usize,
    public: &[F128],
    gather: &[F128],
    gammas: &[PdRec],
    n_el_pd: usize,
    vals_rec: &[F128],
    r_pt: &[F128],
    fgs_v: usize,
) -> usize {
    use flock_core::circuit::CellSlot;
    use flock_core::zerocheck::univariate_skip::build_eq;
    let (nu_c, c_bits) = (cells.nu(), cells.c_bits());
    assert_eq!(nu_c, n_log_i, "the cell space's row vars are the union's");
    assert_eq!(r_pt.len(), nu_c + c_bits, "ρ spans the cell space");
    assert_eq!(gather.len(), cells.num_gate_slots(), "one gather per gate slot");
    assert_eq!(
        gammas.len(),
        n_el_pd + gather.len(),
        "pd claims = the element (c, lc) pair, when the class exists, + the gathers"
    );
    for (i, g) in gather.iter().enumerate() {
        assert_eq!(
            vals_rec[gammas[n_el_pd + i].val_v],
            *g,
            "gather {i} is pd claim {} on the stream",
            n_el_pd + i
        );
    }
    let eq_row = build_eq(&r_pt[..nu_c]);
    let eq_slot = build_eq(&r_pt[nu_c..]);
    let mut acc = F128::ZERO;
    for (iota, slot) in cells.slots().iter().enumerate() {
        match *slot {
            CellSlot::Gate { .. } => acc += eq_slot[iota] * gather[iota],
            CellSlot::Public { s } => {
                let base = s << nu_c;
                let hi = ((base + (1usize << nu_c)).min(public.len())).saturating_sub(base);
                let mut v = F128::ZERO;
                for j in 0..hi {
                    v += eq_row[j] * public[base + j];
                }
                acc += eq_slot[iota] * v;
            }
            CellSlot::Pad => {}
        }
    }
    assert_eq!(
        acc,
        vals_rec[fgs_v],
        "the gathers + publics-MLE recombine to the absorbed f_eval"
    );
    assert_eq!(
        vals_rec[fgs_v],
        vals_rec[fgs_v + 1],
        "f_eval == g_eval on the stream"
    );
    cells.num_public_slots()
}

/// The eq table's `live` prefix over `point_w` wires (LSB-first —
/// `build_eq`'s convention), as MacGate rows: the DOUBLING build, one row
/// per node — `e·ρ` is `0 + e·ρ` and `e·(1+ρ)` is `e + e·ρ`, so both
/// children of a node are single MAC rows. Rows, not advice: every weight is
/// wire-bound to its squeeze. Ancestors of the live prefix are themselves a
/// prefix (low bits), so level `i` builds `min(2^i, live)` entries.
fn emit_eq_prefix(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    point_w: &[Wire],
    live: usize,
    zw: Wire,
    ow: Wire,
) -> Vec<Wire> {
    let live = live.max(1);
    let mut eq_w: Vec<Wire> = vec![ow];
    for (i, &rw) in point_w.iter().enumerate() {
        let half = 1usize << i;
        let width = (2 * half).min(live);
        let mut next = Vec::with_capacity(width);
        for x in 0..width.min(half) {
            next.push(sb.gate(macs, &[eq_w[x], eq_w[x], rw])[0]);
        }
        for x in half..width {
            next.push(sb.gate(macs, &[zw, eq_w[x - half], rw])[0]);
        }
        eq_w = next;
    }
    eq_w
}

/// **THE LAGRANGE ROW LOWS in-circuit (round 4).** The 64 weights
/// `L_i(z_skip) = Z_N(z)·(z + λ_i)^{-1}·den^{-1}` a merge fold's boolean
/// claims carry, derived from the child's z_skip WIRE instead of published
/// and checker-rebuilt: `t_i = z + λ_i` against the shared λ const wires,
/// `Z = Π t_i` (a MAC chain — no subspace recursion needed, the factors are
/// already wires), the inverses as ADVICE bound by `t_i·y_i = 1` rows
/// (witness, not publics), and `w_i = (Z·den^{-1})·y_i`. The caller connects
/// each `w_i` to the fold's absorbed low word.
///
/// `z` on a node has no inverse witness — the ≈2^-121 completeness caveat a
/// fixed-topology circuit carries in its soundness accounting instead of a
/// branch (`lagrange_weights_on_coset`'s own posture, same constant).
#[allow(clippy::too_many_arguments)]
fn emit_lagrange_lows(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    lam_w: &[Wire],
    deninv_w: Wire,
    zskip_w: Wire,
    z_native: F128,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> Vec<Wire> {
    use flock_core::field::PHI_8_TABLE;
    let ell = lam_w.len();
    let mut t_w = Vec::with_capacity(ell);
    let mut z_acc = ow;
    for &lw2 in lam_w {
        let t = sb.gate(macs, &[zskip_w, lw2, ow])[0];
        z_acc = sb.gate(macs, &[zw, z_acc, t])[0];
        t_w.push(t);
    }
    let scale = sb.gate(macs, &[zw, z_acc, deninv_w])[0];
    (0..ell)
        .map(|i| {
            let ti = z_native + PHI_8_TABLE[i];
            assert!(!ti.is_zero(), "z_skip on a φ8 node (≈2^-121)");
            vals.push(ti.inv());
            let y = sb.input();
            // 1 + t·y == 0 (char 2), into the dedicated assert-zero anchor —
            // connecting a producer into the ubiquitous `ow` class is the
            // recorded Cyclic trap.
            let delta = sb.gate(macs, &[ow, t_w[i], y])[0];
            sb.connect(delta, zassert);
            sb.gate(macs, &[zw, scale, y])[0]
        })
        .collect()
}

/// **THE RECOMBINATION in-circuit (round 4).** Rebuild `ŵ(ρ)` from the
/// absorbed gather pd wires and the H region's publics wires, CONNECT it to
/// the absorbed `f_eval`, and connect `f == g` — the two `verify_wiring_core`
/// checks that until now rode only the tape constructors' scaffolding
/// verify. The publics half is the recorded design: the H region's wires
/// feed 8-lane LeafEval folds at `ρ_row[..3]` (the "leaf arithmetic joins
/// the openings" pattern) with hi-group eq weights from the doubling build;
/// the gate half is an eq_slot-weighted MAC chain over the gather wires.
/// Zero new publics, inputs, or slot types — the checker walks are
/// untouched. Dataflow is acyclic: ρ wires come from chain rows BEFORE the
/// `(f, g, s_σ)` absorb, and `f_w` feeds only LATER chain rows.
#[allow(clippy::too_many_arguments)]
fn emit_recombination(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    le8: flock_core::circuit::builder::SlotId,
    pub_w: &[Wire],
    gather_w: &[Wire],
    pt_w: &[Wire],
    nu_c: usize,
    n_pub_slots: usize,
    f_w: Wire,
    g_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    sb.connect(f_w, g_w);
    let rows = 1usize << nu_c;
    assert_eq!(
        pub_w.chunks(rows).count(),
        n_pub_slots,
        "public slots tile the child's segment"
    );
    let eq_slot_w = emit_eq_prefix(sb, macs, &pt_w[nu_c..], gather_w.len() + n_pub_slots, zw, ow);
    let max_chunks = pub_w
        .chunks(rows)
        .map(|s| s.len().div_ceil(8))
        .max()
        .expect("a circuit child has publics");
    let eq_hi_w = emit_eq_prefix(sb, macs, &pt_w[3..nu_c], max_chunks, zw, ow);
    let mut acc = zw;
    for (i, &gw2) in gather_w.iter().enumerate() {
        acc = sb.gate(macs, &[acc, eq_slot_w[i], gw2])[0];
    }
    for (s, spub) in pub_w.chunks(rows).enumerate() {
        let mut v = zw;
        for (h, chunk) in spub.chunks(8).enumerate() {
            let mut a_in: Vec<Wire> = chunk.to_vec();
            a_in.resize(8, zw);
            a_in.extend_from_slice(&pt_w[..3]);
            a_in.push(eq_hi_w[h]);
            a_in.push(v);
            v = sb.gate(le8, &a_in)[0];
        }
        acc = sb.gate(macs, &[acc, eq_slot_w[gather_w.len() + s], v])[0];
    }
    sb.connect(acc, f_w);
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
    /// The recording verify's wall time — the ONLINE tape-source cost (the
    /// pins/locates around it are per-shape scaffolding).
    verify_ms: f64,
    // located regions. `el` is `None` for a BOOLEAN-ONLY circuit inner
    // (the hash-chain leaf) — the element PIOP region does not exist on
    // its tape, and the `el_*`/`a_sum_n`/`b_sum_n` natives below are
    // meaningful iff `el.is_some()`.
    gkr: GkrRec,
    el: Option<ElPiopRec>,
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
    /// The child cell space's public-slot count — the recombination's tail.
    n_pub_slots_c: usize,
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
    el_assert: Option<flock_core::element_r1cs::union::ElementAssertion>,
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
    /// Derived pd claim points (merged-open v1) — see [`RealTape::pd_pts`].
    pd_pts: Vec<Vec<F128>>,
}

impl<'p> ChildTape<'p> {
    fn new(inner: &'p MixedInner, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let built = &inner.built;
        let proof = &inner.proof;
        let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
        let blake_r1cs = blake3::build_block_r1cs(inner.nu);
        let blake_lc = blake_r1cs.csc_lincheck_circuit();
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
        let t_v = std::time::Instant::now();
        let all_claims = verifier::verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &inner.commitment,
            proof,
            &inner.pcs,
            &mut rec,
        )
        .expect("the mixed circuit inner verifies");
        let verify_ms = t_v.elapsed().as_secs_f64() * 1e3;
        let native_claims = all_claims
            .boolean
            .clone()
            .expect("the boolean class yields the RS (ab, c) claims");
        let bool_assert = inner.work.boolean.clone().expect("boolean matrix work");
        // The element side is OPTIONAL: a boolean-only circuit inner (the
        // hash-chain leaf) has no element class, and its tape carries no
        // element PIOP region. The union is the authority.
        let has_el = union.has_element();
        let el_assert = inner.work.element.clone();
        assert_eq!(
            el_assert.is_some(),
            has_el,
            "element work travels iff the union has an element class"
        );
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
        let gkr_l = find(b"flock-product-gkr-batched-v0");
        let mo_l = find(b"flock-merged-open-v1");
        let rs_l = find(b"flock-ring-switch-v0");
        let mp_l = find(b"flock-multipoint-twisted-v1");
        let fa_l = find(b"flock-frobenius-assist-v0");
        assert_eq!(zc_l.len(), 1, "one boolean zerocheck");
        assert_eq!(lc_l.len(), 1, "one boolean lincheck");
        if has_el {
            assert_eq!(elzc_l.len(), 1, "one element zerocheck");
            assert_eq!(el_l.len(), 1, "one element lincheck region");
            assert!(elzc_l[0] < el_l[0], "element zc before element lc");
            assert!(lc_l[0] < el_l[0], "boolean PIOP before element PIOP");
            assert!(el_l[0] < gkr_l[0], "element PIOP before the wiring GKR");
        } else {
            assert!(
                elzc_l.is_empty() && el_l.is_empty(),
                "a boolean-only tape carries NO element region"
            );
            assert!(lc_l[0] < gkr_l[0], "boolean PIOP before the wiring GKR");
        }
        assert_eq!(gkr_l.len(), 1, "one batched wiring GKR");
        assert_eq!(mo_l.len(), 1, "one merged open");
        assert_eq!(rs_l.len(), 2, "rs x 2 — one ab/c pair for the boolean class");
        assert_eq!(mp_l.len(), 1, "one multipoint region");
        assert_eq!(fa_l.len(), 1, "one anchor region");
        assert!(zc_l[0] < lc_l[0], "boolean zc before boolean lc");
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
        assert_eq!(
            proof.element.is_some(),
            has_el,
            "the element proof section mirrors the union's classes"
        );
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

        // ---- the ELEMENT PIOP region, located (mixed inners only) ----
        // Shape, per `parse_open_levels`' element branch: [tau slice |
        // tau_len rounds | ea, eb, ec | lc label | alpha | lc rounds].
        let el_rec = has_el.then(|| {
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
        });

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
            // Packed-direct claims (merged-open v1): [ObserveScalar(value),
            // SqueezeScalar(gamma)] each — the W rounds that follow are
            // [Obs, Obs, Squeeze] triplets, so the lookahead disambiguates.
            let mut pd_recs: Vec<usize> = Vec::new(); // value index
            while matches!(ops[i], Op::ObserveScalar)
                && matches!(ops[i + 1], Op::SqueezeScalar)
            {
                let (pv, _) = vc_at(i);
                i += 2;
                pd_recs.push(pv);
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
        // The pd claims are the element class's two (c, lc) — when the
        // class exists — plus one per wiring GATHER; every gather value is
        // absorbed, in proof order.
        let n_el_pd = if has_el { 2 } else { 0 };
        assert_eq!(
            pd_recs.len(),
            n_el_pd + proof.wiring.gather.len(),
            "pd claims = element (c, lc) + the wiring gathers"
        );
        let pd_vals: Vec<F128> = pd_recs.iter().map(|&pv| vals_rec[pv]).collect();
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
        let stream = t_shape.stream_words_duplex(domain);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let trace = {
            let mut chain = FsChainSponge::new();
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
        assert_eq!(
            piop_o.is_some(),
            has_el,
            "the parser sees the element PIOP iff the class exists"
        );
        assert_eq!(
            gammas_o.len(),
            pd_recs.len(),
            "the parser and the region walk agree on the pd claims"
        );
        let (geo, native_sums) = level_geometry(
            &levels,
            &lvl_src,
            &chals,
            HashKind::Blake3,
            &strat_scheds(&inner.pcs),
        );
        let b3_rows = trace.rows.len()
            + h_rows
            + geo
                .iter()
                .map(|g| (g.row_words.div_ceil(4) + g.depth) * g.q + (1usize << g.c) - 1)
                .sum::<usize>();
        if std::env::var("B3_CENSUS").is_ok() {
            let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
            let blocks = trace.rows.len() - parents;
            eprintln!(
                "  [b3 census] chain {} (data blocks {} | parent/fork {}; absorbed {} B, {} squeezes) | H(publics) {} | openings+caps {} = {}",
                trace.rows.len(),
                blocks,
                parents,
                bytes.len(),
                trace.squeezes.len(),
                h_rows,
                b3_rows - trace.rows.len() - h_rows,
                b3_rows
            );
            for g in geo.iter() {
                eprintln!(
                    "    level: q {} depth {} row_words {} -> leaf {} + path {} + cap {}",
                    g.q,
                    g.depth,
                    g.row_words,
                    g.row_words.div_ceil(4) * g.q,
                    g.depth * g.q,
                    (1usize << g.c) - 1
                );
            }
            // CHAIN DECOMPOSITION + an independent row-count model of the
            // duplex discipline (transcript-v3), asserted against the
            // sponge trace: a squeeze row absorbs the pending partial
            // block as its MESSAGE, mutates cv, and has no header word.
            {
                let pad16 = |n: usize| n.div_ceil(16) * 16;
                let (mut hdr_w, mut pay_w, mut n_obs, mut n_sq) = (0usize, 0usize, 0usize, 0usize);
                // The domain header + padded domain are absorbed at
                // construction, ahead of the recorded ops.
                let (mut v3_rows, mut pend) = (0usize, 16 + pad16(domain.len()));
                for op in ops.iter() {
                    match op {
                        Op::Label(l) => {
                            hdr_w += 1;
                            pay_w += pad16(l.len()) / 16;
                            n_obs += 1;
                            pend += 16 + pad16(l.len());
                        }
                        Op::ObserveScalar => {
                            hdr_w += 1;
                            pay_w += 1;
                            n_obs += 1;
                            pend += 32;
                        }
                        Op::ObserveSlice(n) => {
                            hdr_w += 1;
                            pay_w += n;
                            n_obs += 1;
                            pend += 16 + 16 * n;
                        }
                        Op::ObserveBytes(len) => {
                            hdr_w += 1;
                            pay_w += pad16(*len) / 16;
                            n_obs += 1;
                            pend += 16 + pad16(*len);
                        }
                        Op::SqueezeScalar | Op::SqueezeSlice(_) | Op::Pow { .. } => {
                            n_sq += 1;
                            // v3: the squeeze row eats the pending partial
                            // block and emits output block 0; extra output
                            // blocks follow.
                            v3_rows += pend / 64;
                            v3_rows += 1 + (op.squeezed_bytes().div_ceil(64) - 1);
                            pend = 0;
                            if let Op::Pow { .. } = op {
                                // the nonce rides observe_bytes(8): header + word
                                pend += 32;
                            }
                        }
                    }
                    v3_rows += pend / 64;
                    pend %= 64;
                }
                if pend > 0 {
                    v3_rows += 1;
                }
                eprintln!(
                    "  [chain census] ops {} (obs {} / sq {}) | header words {} ({} B) | payload words {} | duplex rows {}",
                    ops.len(),
                    n_obs,
                    n_sq,
                    hdr_w,
                    16 * hdr_w,
                    pay_w,
                    trace.rows.len(),
                );
                assert_eq!(
                    v3_rows,
                    trace.rows.len(),
                    "the duplex row model diverged from the sponge trace"
                );
            }
        }
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

        // ---- the element PIOP's native chain + strip sums (mixed only) ----
        let (el_g0, el_run_n, a_sum_n, b_sum_n) = if let Some(el_rec) = &el_rec {
            let el_assert = el_assert.as_ref().expect("element assertion");
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
            (el_g0, el_run_n, a_sum_n, b_sum_n)
        } else {
            (Vec::new(), F128::ZERO, F128::ZERO, F128::ZERO)
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
        // ROUND 4: the recombination + f == g, replayed from located words
        // (the emitter binds these; until it landed they rode only this
        // constructor's scaffolding verify).
        let n_pub_slots_c = pin_recombination(
            inner.built.shape.circuit.cells(),
            n_log_i,
            &inner.built.witness.public,
            &inner.proof.wiring.gather,
            &gammas_o,
            n_el_pd,
            &vals_rec,
            &gkr_rec.r_pt,
            gkr_rec.fgs_v,
        );
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
            if let Some(el_rec) = &el_rec {
                let el_assert = el_assert.as_ref().expect("element assertion");
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
        // Derived pd points (merged-open v1) — see RealTape's twin.
        let pd_pts_n: Vec<Vec<F128>> = {
            let cells = inner.built.shape.circuit.cells();
            let mut v: Vec<Vec<F128>> = if has_el {
                let el = all_claims.element.as_ref().expect("element claims");
                vec![el.c_point.clone(), el.lc_point.clone()]
            } else {
                Vec::new()
            };
            for i2 in 0..gammas_o.len() - n_el_pd {
                v.push(cells.gate_claim_point(i2, &gkr_rec.r_pt[..cells.nu()]));
            }
            v
        };
        for pt in &pd_pts_n {
            assert_eq!(pt.len(), n_log_i + k_cols_i, "pd point split");
        }
        if let Some(el_rec) = &el_rec {
            let e_rounds = el_rec.zc_rounds.len();
            for j in 0..n_log_i {
                assert_eq!(pd_pts_n[0][j], chals[el_rec.zc_rounds[j].2], "c row {j}");
            }
            for j in 0..e_rounds - n_log_i {
                assert_eq!(
                    pd_pts_n[0][n_log_i + j],
                    chals[el_rec.zc_rounds[n_log_i + j].2],
                    "c col {j}"
                );
            }
            let n_lc = el_rec.lc_rounds.len();
            for j in 0..n_lc {
                assert_eq!(
                    pd_pts_n[1][n_log_i + j],
                    chals[el_rec.lc_rounds[n_lc - 1 - j].2],
                    "lc col {j}"
                );
            }
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
            verify_ms,
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
            n_pub_slots_c,
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
            pd_pts: pd_pts_n,
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
    /// The residual region's keyed slot cache (`emit_residual_region`'s
    /// `leaf_slot`). Key scheme: `600` = the shared MacGate (pre-seeded,
    /// so close-out rows land on `macs` instead of a duplicate type);
    /// `100 + pl` = the ResidualGate at that suffix-fold count (pl is the
    /// type's real parameter — see `emit_residual_region`); `310 + width`
    /// = the shared PrefixGate at that width (the eq/prefix-product rows —
    /// NOT a residual gate, it merely lives in this cache).
    resid: Vec<(usize, flock_core::circuit::builder::SlotId)>,
    /// The leaf-only skip types the ENVELOPE cross-declares at count 0
    /// (wall 2) — no node emission touches them, but the element prover
    /// input assembly must still cover their registry slots. Empty
    /// off-envelope.
    skips: Vec<flock_core::circuit::builder::SlotId>,
}

impl ChildSlots {
    fn new(sb: &mut ShapeBuilder, nu2: usize, spread_w: usize) -> Self {
        let macs = sb.slot(MacGate::new());
        ChildSlots {
            q: CollapsedSlots {
                b3: sb.slot(Blake3Gate { nu: nu2 }),
                swap: sb.slot(SwapGate { nu: nu2 }),
                spread: sb.slot(BitSpreadGate {
                    ty: BitSpreadTable::new(spread_w),
                    nu: nu2,
                }),
            },
            macs,
            zcr: sb.slot(ZcRoundGate::new()),
            mrs: sb.slot(MergedRoundGate::new()),
            spine: sb.slot(SpineGate::new()),
            alslot: sb.slot(AssistLayerGate::new()),
            le: Vec::new(),
            // Key 600 pre-seeds the SHARED MacGate into the residual cache:
            // emit_residual_region's close-out rows land on the same slot
            // instead of registering a duplicate type.
            resid: vec![(600, macs)],
            skips: Vec::new(),
        }
    }

    /// The ENVELOPE constructor (wall 2): the same canonical declaration
    /// order [`declare_envelope_slots`] gives the leaf builder — including
    /// the leaf-only SkipNode/SkipClose types this node never rows (count
    /// 0) — so the node's registry digest-equals the leaf's. Every keyed
    /// entry pre-seeds the demand caches; emission that would need a slot
    /// OUTSIDE the envelope set creates a new type and fails the digest
    /// pin loudly.
    fn new_env(sb: &mut ShapeBuilder, nu2: usize, env: &EnvShape) -> Self {
        let mut cache: Vec<(usize, flock_core::circuit::builder::SlotId)> = Vec::new();
        let q = declare_envelope_slots(sb, nu2, &mut cache, env);
        let take = |k: usize| {
            cache
                .iter()
                .find(|&&(c, _)| c == k)
                .expect("an envelope slot")
                .1
        };
        ChildSlots {
            q,
            macs: take(600),
            zcr: take(500),
            mrs: take(400),
            spine: take(0),
            alslot: take(601),
            le: vec![(8, take(8))],
            // The residual-region cache inherits every entry in its key
            // namespaces: the shared mac (600), the five resid variants
            // (100 + pl) and the prefix slot (310 + w).
            resid: cache
                .iter()
                .filter(|&&(k, _)| k == 600 || (100..200).contains(&k) || (310..400).contains(&k))
                .cloned()
                .collect(),
            skips: vec![take(510), take(511)],
        }
    }

    /// The keyed cache view `pad_envelope_counts` consumes — envelope path
    /// only (`new_env`; the skips are positional [skn, skc]).
    fn env_cache(&self) -> Vec<(usize, flock_core::circuit::builder::SlotId)> {
        let mut v = vec![
            (600, self.macs),
            (500, self.zcr),
            (400, self.mrs),
            (0, self.spine),
            (601, self.alslot),
            (510, self.skips[0]),
            (511, self.skips[1]),
        ];
        v.extend(self.le.iter().map(|&(n, s)| (n, s)));
        v.extend(self.resid.iter().filter(|&&(k, _)| k != 600).cloned());
        v
    }

    /// Every element-class slot, for the outer prover's slot inputs.
    fn element_slot_ids(&self) -> Vec<flock_core::circuit::builder::SlotId> {
        let mut v = vec![self.macs, self.zcr, self.mrs, self.spine, self.alslot];
        v.extend(self.le.iter().map(|&(_, s)| s));
        // Key 600 is the SHARED MacGate seed (already listed as `macs`).
        v.extend(self.resid.iter().filter(|&&(k, _)| k != 600).map(|&(_, s)| s));
        v.extend(self.skips.iter().copied());
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
    /// The z_skip squeeze wire — the merge assemblies derive the lagrange
    /// row lows from it IN-CIRCUIT (no publish, no checker rebuild).
    zskip_w: Wire,
    /// The residual close-out's prefix slot (and width) — reusable by a
    /// caller emitting more prefix products into the same builder.
    pf: (flock_core::circuit::builder::SlotId, usize),
    /// The child's PUBLIC SEGMENT as witness wires (the H(publics) region's
    /// inputs, in the child's declaration order). Application-statement
    /// plumbing — the hash-chain adjacency connect — reads through these.
    child_pub_w: Vec<Wire>,
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
    consts: &mut Vec<(F128, Wire)>,
) -> ChildRegion {
    let trace = &ct.trace;
    let stream = &ct.stream;
    let chals = &ct.chals[..];
    let levels = &ct.levels[..];
    let geo = &ct.geo[..];
    let w_rounds = &ct.w_rounds[..];
    let mp_o = &ct.mp_o;
    let inner_pd2 = &ct.inner_pd2;
    // `None` for a boolean-only (chain) child: the element PIOP emission,
    // its two publics and its ChildRegion wires all vanish together.
    let el_rec = ct.el.as_ref();
    let n_el_pd = if el_rec.is_some() { 2 } else { 0 };
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
    let (outs, ww) = emit_fs_chain(
        sb,
        cs.q.b3,
        iv2,
        trace,
        stream,
        &ct.bytes,
        vals,
        consts,
        &ct.pub_payloads,
    );
    // ---- ROUND 2: the H(publics) region (v2 statement binding) ----
    // The returned wires ARE the child's public segment — the recombination
    // folds them.
    let pub_w = {
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
            consts,
        )
    };
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
        consts,
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

    // ---- ROUND 4: the recombination + f == g, in-circuit ----
    let le8 = match cs.le.iter().find(|&&(n, _)| n == 8) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(LeafEvalGate::new(8));
            cs.le.push((8, s));
            s
        }
    };
    let gather_w: Vec<Wire> = (0..ct.n_pd - n_el_pd)
        .map(|i| wv(ct.gammas_o[n_el_pd + i].val_v))
        .collect();
    emit_recombination(
        sb,
        macs,
        le8,
        &pub_w,
        &gather_w,
        &pt_w,
        n_log_i,
        ct.n_pub_slots_c,
        f_w,
        g_w,
        zw,
        ow,
    );

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

    // ---- the ELEMENT PIOP rounds in-circuit (mixed children only) ----
    // Zerocheck rounds are ZcRoundGate rows (tau slice wires as eq weights,
    // g0 advice + zero deltas); lincheck rounds are MergedRoundGate rows.
    // The entry is DERIVED: va = ea + a_sum, vb = eb + b_sum, entry =
    // va + alpha·vb — only the two constant-strip sums are advice.
    let el_pub = el_rec.map(|el_rec| {
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
        (el_zr, el_lcw)
    });

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
                (t_vals_b[k2], cw(sb, vals, consts, t_vals_b[k2]))
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
                            // merged-open v1: derived coordinates — element
                            // claims wire from the region PIOP's squeezes
                            // (pinned in ChildTape::new), gather coords are
                            // constant address bits.
                            let coord = ct.pd_pts[i2][n_log_i + jj];
                            let cw2 = if coord == F128::ZERO {
                                zw
                            } else if coord == F128::ONE {
                                ow
                            } else {
                                // Non-constant col coords occur only on the
                                // element pair (gather coords are address
                                // bits) — a boolean-only child never lands
                                // here.
                                let el_rec = el_rec.expect("element pd claim");
                                if i2 == 0 {
                                    outs[trace.squeezes[el_rec.zc_rounds[n_log_i + jj].1][0]][0]
                                } else {
                                    let n_lc = el_rec.lc_rounds.len();
                                    outs[trace.squeezes[el_rec.lc_rounds[n_lc - 1 - jj].1][0]][0]
                                }
                            };
                            (cw2, if (y >> jj) & 1 == 1 { ow } else { zw })
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
                if members[0] >= n_el_pd {
                    pt_w[layer]
                } else {
                    let el_rec = el_rec.expect("element pd claim");
                    outs[trace.squeezes[el_rec.zc_rounds[layer].1][0]][0]
                }
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
    if let Some((el_zr, el_lcw)) = el_pub {
        sb.publish(el_zr);
        sb.publish(el_lcw);
    }
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
    let n_tail = 2 + n_el_pd + 4 + levels.len() * ct.yr_len + 1 + 1 + ct.mu_i;
    let n_query_pub: usize =
        levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    ChildRegion {
        pub_base,
        n_query_pub,
        n_tail,
        sig_w,
        pt_w,
        el_zc_rho_w: el_rec
            .map(|el_rec| {
                el_rec
                    .zc_rounds
                    .iter()
                    .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
                    .collect()
            })
            .unwrap_or_default(),
        el_lc_rho_w: el_rec
            .map(|el_rec| {
                el_rec
                    .lc_rounds
                    .iter()
                    .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
                    .collect()
            })
            .unwrap_or_default(),
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: ct
            .lc_rounds_b
            .iter()
            .map(|&(_, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: (0..64).map(|i| wv(ct.zp_v + i)).collect(),
        zskip_w: outs[trace.squeezes[ct.zskip_fin][0]][0],
        pf: (pfslot, pf_w),
        child_pub_w: pub_w,
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
    let mp_base = if let Some(el_assert) = &ct.el_assert {
        assert_eq!(
            public[el_base],
            ct.el_run_n,
            "the element zc chain ends at the native running claim"
        );
        // THE INDEPENDENT CLOSE: the in-circuit lincheck chain ends exactly
        // at the native ElementAssertion's target.
        assert_eq!(
            public[el_base + 1],
            el_assert.target,
            "the element lc chain ends at the native assertion's target"
        );
        el_base + 2
    } else {
        el_base
    };
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
    // `num_lanes` ACTIVE lanes — an arbitrary integer (47 here; 61 before
    // the b3 lin-id drop narrowed the stack), NOT a whole number of
    // blocks, and narrower than the fold width.
    assert_eq!(ct.geo[0].row_words, 47, "the mixed inner's active lane count");
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
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let region = emit_child_region(&mut sb, &mut cs, &ct, &mut vals, &mut hints, &mut consts);
    let shape2 = sb.finish().expect("the mvp10 chain circuit builds");
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
        hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
    let mut built2 = shape2.run(&vals, &hint_refs);
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
        log_batch_size: pcs_batch(&union2),
        profile: LigeritoProfile::Fast,
        num_lanes: union2.commit_lanes(pcs_batch(&union2)),
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
            let z = match std::mem::replace(
                &mut built2.witnesses[shape2.registry_slot(sl)],
                SlotWitness::DeferredToRows,
            ) {
                SlotWitness::Element(z) => z,
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
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
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
    let mut chp = FsChallenger::with_chained_blake3(M11_DOMAIN);
    let (fp, out_p) = matrix_fold::prove_fold(&m_sig, &combs, &claims, &mut chp);
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_DOMAIN));
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
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let stream = t_shape.stream_words_duplex(M11_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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
        let mut built2 = shape2.run(&vals, &[]);

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
            log_batch_size: pcs_batch(&union2),
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(pcs_batch(&union2)),
            merkle_hash: Default::default(),
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let mut el_ord: Vec<(usize, Vec<F128>)> = [macs, mrs, pfslot]
            .into_iter()
            .map(|sl| {
                let z = match std::mem::replace(
                    &mut built2.witnesses[shape2.registry_slot(sl)],
                    SlotWitness::DeferredToRows,
                ) {
                    SlotWitness::Element(z) => z,
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
        let mut ch = FsChallenger::with_chained_blake3(M11_LEAF_DOMAIN);
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
        let mut ch = FsChallenger::with_chained_blake3(M11_LEAF_DOMAIN);
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

    let mut chp = FsChallenger::with_chained_blake3(M11_MERGE_DOMAIN);
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
        RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_MERGE_DOMAIN));
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
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let stream = t_shape.stream_words_duplex(M11_MERGE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let r0 = emit_child_region(&mut sb, &mut cs, &t0, &mut vals, &mut hints, &mut consts);
        let r1 = emit_child_region(&mut sb, &mut cs, &t1, &mut vals, &mut hints, &mut consts);
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
        // The lagrange-low constants, shared by both children: the 64 φ8
        // nodes and the subspace denominator inverse — statement constants
        // the checker validates below (the ONE public surface the
        // in-circuit derivation adds).
        use flock_core::field::PHI_8_TABLE;
        use flock_core::zerocheck::K_SKIP;
        use flock_core::zerocheck::multilinear::{
            lagrange_weights_naive, subspace_denominator_pair,
        };
        let lam_base = sb.public_len();
        let lam_w: Vec<Wire> = PHI_8_TABLE[..1 << K_SKIP]
            .iter()
            .map(|&v| {
                vals.push(v);
                sb.public_input()
            })
            .collect();
        vals.push(subspace_denominator_pair(K_SKIP).1);
        let deninv_w = sb.public_input();
        // The lows' assert-zero anchor: producers only, no consumer edges.
        vals.push(F128::ZERO);
        let lag_zassert = sb.public_input();
        let tapes = [&t0, &t1];
        let regions = [&r0, &r1];
        for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
            // The lagrange row lows, IN-CIRCUIT from the child's z_skip wire
            // (native pre-assert first: the fold's absorbed lows ARE the closed
            // form at the located z_skip).
            assert_eq!(
                &fold_claims[0][n_priors + k].row.low[..],
                &lagrange_weights_naive(K_SKIP, tk.chals[tk.zskip_ch])[..],
                "child {k}: the fold's lagrange lows are the closed form"
            );
            let lows = emit_lagrange_lows(
                &mut sb,
                cs.macs,
                &lam_w,
                deninv_w,
                rk.zskip_w,
                tk.chals[tk.zskip_ch],
                &mut vals,
                zw,
                ow,
                lag_zassert,
            );
            for (j, &lw2) in lows.iter().enumerate() {
                sb.connect(lw2, wv(locs[0].claims[n_priors + k].row_low_v + j));
            }
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
                &tk.el_assert.as_ref().expect("mixed child").r_con[..kappa],
                "element row point is r_con's head"
            );
            assert_eq!(
                &fold_claims[2][n_priors + k].col.point[..],
                &tk.el_assert.as_ref().expect("mixed child").r_col[..kappa],
                "element col point is r_col's head"
            );
            assert_eq!(fold_claims[2][n_priors + k].value, tk.el_assert.as_ref().expect("mixed child").evals[0].0);
            assert_eq!(fold_claims[3][n_priors + k].value, tk.el_assert.as_ref().expect("mixed child").evals[0].1);
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
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
        let mut built2 = shape2.run(&vals, &hint_refs);

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
            for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                assert_eq!(built2.public[lam_base + i], v, "λ const {i}");
            }
            assert_eq!(
                built2.public[lam_base + (1 << K_SKIP)],
                subspace_denominator_pair(K_SKIP).1,
                "the subspace denominator inverse const"
            );
            assert_eq!(
                built2.public[lam_base + (1 << K_SKIP) + 1],
                F128::ZERO,
                "the lows' assert-zero anchor"
            );
            let mut q = tail0 + tail_len;
            for (k, tk) in tapes.iter().enumerate() {
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
                    tk.el_assert.as_ref().expect("mixed child").evals[0].0,
                    "child {k}: element A eval"
                );
                assert_eq!(
                    built2.public[q + 3],
                    tk.el_assert.as_ref().expect("mixed child").evals[0].1,
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
            log_batch_size: pcs_batch(&union2),
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(pcs_batch(&union2)),
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
                let z = match std::mem::replace(
                    &mut built2.witnesses[shape2.registry_slot(sl)],
                    SlotWitness::DeferredToRows,
                ) {
                    SlotWitness::Element(z) => z,
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
    let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
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
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
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
    let mut chp = FsChallenger::with_chained_blake3(M11_SCALE_DOMAIN);
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
        RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_SCALE_DOMAIN));
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
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let stream = t_shape.stream_words_duplex(M11_SCALE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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
        let mut built2 = shape2.run(&vals, &[]);

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
            log_batch_size: pcs_batch(&union2),
            profile: LigeritoProfile::Fast,
            num_lanes: union2.commit_lanes(pcs_batch(&union2)),
            merkle_hash: Default::default(),
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let mut el_ord: Vec<(usize, Vec<F128>)> = [macs, mrs, pfslot, leslot]
            .into_iter()
            .map(|sl| {
                let z = match std::mem::replace(
                    &mut built2.witnesses[shape2.registry_slot(sl)],
                    SlotWitness::DeferredToRows,
                ) {
                    SlotWitness::Element(z) => z,
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
/// The PRODUCTION per-proof tape cost of one child: the recorded deferred
/// verify alone — the tape (op sequence + values + challenges) and the
/// assertion references in one pass. Everything else `RealTape::new` does
/// (pins, locates, native replicas) is SHAPE-STABLE index work a real node
/// precomputes at setup; the value/hint fill from a fresh tape is index
/// copies on top of this. Union + lcs construction is included
/// (conservative — a node would cache both).
fn record_child_verify(lo: &LeafOuter, domain: &'static [u8]) {
    use flock_core::transcript_record::RecordingChallenger;
    let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (lo.b3_slot, lo.b3_r1cs.csc_lincheck_circuit()),
        (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
        (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, cc)| cc).collect();
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
    verifier::verify_ligerito_union_circuit_deferred(
        &union_i,
        &lo.shape.circuit,
        &lo.public,
        &lcs,
        &lo.commitment,
        &lo.proof,
        &lo.pcs,
        &mut rec,
    )
    .expect("the child verifies (recorded)");
}


fn build_node_outer(
    lo0: &LeafOuter,
    lo1: &LeafOuter,
) -> (LeafOuter, flock_core::aggregate::Accumulator, Online) {
    let (lo, acc, t, _, _) = build_node_outer_app(&[lo0, lo1], None, None);
    (lo, acc, t)
}

/// A LOWER-registry accumulator lane riding through an internal node
/// (task 6): the two children each carry an accumulator over a registry
/// that is NOT the fold's own (the chain registry at the first level), so
/// it cannot join the node's fold as a prior — it folds in its OWN
/// priors-only aggregate, whose prior surfaces connect WIRE-TO-WIRE to
/// the children's published accumulator claims (`claims_base` locates
/// them; a prior's surface IS what the child published).
struct ChainLane<'a> {
    registry: &'a flock_prover::schedule::Registry,
    mats: &'a [flock_core::aggregate::TypeMatrices<'a>],
    circs: &'a [&'a dyn flock_core::lincheck::LincheckCircuit],
    /// The lane's sigma table owner (the chain circuit).
    circuit: &'a flock_core::circuit::Circuit,
    priors: &'a [&'a flock_core::aggregate::Accumulator],
    /// The published `[rho_col | rho_row | value]` fold blocks' base in
    /// EACH child's public segment (every child shares the layout).
    claims_base: usize,
}

/// [`build_node_outer`] with the APPLICATION-STATEMENT plumbing: when the
/// children carry an app block (`app_stmt` = its offset in their public
/// segments — the hash-chain span (h_start, h_end), 8 words), the node
/// connects left.h_end == right.h_start wire-to-wire and publishes the
/// combined span as its OWN app block, returning that block's offset — so
/// the output feeds the next level with the same plumbing.
fn build_node_outer_app(
    los: &[&LeafOuter],
    app_stmt: Option<usize>,
    lane: Option<ChainLane<'_>>,
) -> (
    LeafOuter,
    flock_core::aggregate::Accumulator,
    Online,
    Option<usize>,
    Option<flock_core::aggregate::Accumulator>,
) {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const M11_NODE_DOMAIN: &[u8] = b"flock-mvp11-two-to-one-v0";

    // ARITY IS A KNOB: the node folds `k = los.len()` children in one
    // proof. Commit and open are FLOOR-bound — they cost the same whatever
    // k is, as long as the content stays under 2^(m*-7) — so every child
    // past the first rides that toll for free, and a k-ary layer needs
    // 1/(k-1) as many nodes. What does scale with k is the per-child
    // region: mac is ~97% per-child work, which is why nu* is 16.
    let n_kids = los.len();
    assert!(n_kids >= 2, "a node folds at least two children");
    let lo0 = los[0];
    for lo in los {
        assert_eq!(
            lo.shape.circuit.digest(),
            lo0.shape.circuit.digest(),
            "every child, ONE node circuit"
        );
    }
    let registry = &lo0.shape.registry;
    let unions: Vec<UnionInstance> = los
        .iter()
        .map(|lo| outer_union(&lo.shape.registry, lo.shape.counts.clone()))
        .collect();
    let t_tapes = std::time::Instant::now();
    // The children's tapes are independent statement work — build them
    // concurrently (each is a recording verify + the region pins).
    let rts: Vec<RealTape> = {
        use rayon::prelude::*;
        los.par_iter().map(|lo| RealTape::new(lo, DOMAIN)).collect()
    };
    let tape_setup_ms = t_tapes.elapsed().as_secs_f64() * 1e3;
    for i in 1..n_kids {
        assert_ne!(
            rts[0].sigma_native.rho, rts[i].sigma_native.rho,
            "distinct witnesses, distinct FS points"
        );
    }

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

    // The native merge fold over every child's assertions.
    let bool_asserts: Vec<_> = rts.iter().map(|rt| rt.mat_assert.clone()).collect();
    let el_asserts: Vec<_> = rts
        .iter()
        .zip(&unions)
        .map(|(rt, u)| (u, rt.el_assert.clone()))
        .collect();
    let sigmas: Vec<_> = rts.iter().map(|rt| rt.sigma_native.clone()).collect();
    let mut chp = FsChallenger::with_chained_blake3(M11_NODE_DOMAIN);
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
        RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_NODE_DOMAIN));
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
    let bc: Vec<_> = rts.iter().map(|rt| rt.mat_assert.claims(registry)).collect();
    let ec: Vec<_> = rts
        .iter()
        .zip(&unions)
        .map(|(rt, u)| rt.el_assert.claims(u))
        .collect();
    // One group per (type, side), each carrying ONE claim per child — the
    // fold machinery is claim-count-generic, so arity enters here only as
    // the length of these vectors.
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::new();
    for t in 0..n_bool {
        fold_claims.push((0..n_kids).map(|i| bc[i][t].0.clone()).collect());
        fold_claims.push((0..n_kids).map(|i| bc[i][t].1.clone()).collect());
    }
    for t in 0..n_el {
        fold_claims.push((0..n_kids).map(|i| ec[i][t].0.clone()).collect());
        fold_claims.push((0..n_kids).map(|i| ec[i][t].1.clone()).collect());
    }
    fold_claims.push(sigmas.iter().map(|s| s.claim()).collect());
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

    // ---- the LANE (task 6): the children's LOWER-registry accumulators
    // fold PRIORS-ONLY — natively here, in-circuit below. 3 groups
    // (bool A/B + sigma) × [priorL, priorR], no fresh claims. ----
    const LANE_DOMAIN: &[u8] = b"flock-chain-lane-v0";
    let lane_native = lane.as_ref().map(|ln| {
        let el_asserts_l: [(
            &UnionInstance<'_>,
            flock_core::element_r1cs::union::ElementAssertion,
        ); 0] = [];
        let mut chp = FsChallenger::with_chained_blake3(LANE_DOMAIN);
        let (lagg, lacc_p) = aggregate::prove_aggregate_classes(
            ln.registry,
            ln.mats,
            ln.circs,
            &[],
            &[],
            &el_asserts_l,
            Some((ln.circuit, &[])),
            ln.priors,
            &mut chp,
        )
        .expect("the lane fold proves");
        let mut lrec =
            RecordingChallenger::new(FsChallenger::with_chained_blake3(LANE_DOMAIN));
        let lacc_v = aggregate::verify_aggregate_classes(
            ln.registry,
            &[],
            &el_asserts_l,
            Some((ln.circuit, &[])),
            ln.priors,
            &lagg,
            &mut lrec,
        )
        .expect("the lane fold verifies");
        assert_eq!(lacc_p, lacc_v, "lane prover and verifier agree");
        let lclaims: Vec<Vec<MatrixClaim>> = vec![
            ln.priors.iter().map(|p| p.per_type[0].0.clone()).collect(),
            ln.priors.iter().map(|p| p.per_type[0].1.clone()).collect(),
            ln.priors
                .iter()
                .map(|p| p.sigma.as_ref().expect("lane prior sigma").1.clone())
                .collect(),
        ];
        let lproofs: Vec<&FoldProof> = vec![
            &lagg.folds[0].0,
            &lagg.folds[0].1,
            lagg.sigma_fold.as_ref().expect("lane sigma fold"),
        ];
        let lops: Vec<Op> = lrec.shape().ops().to_vec();
        let lvals: Vec<F128> = lrec.values().to_vec();
        let lchals: Vec<F128> = lrec.challenges().to_vec();
        let mut want: Vec<Op> = vec![
            Op::Label(b"flock-aggregate-v0".to_vec()),
            Op::ObserveBytes(32),
            Op::ObserveBytes(1),
        ];
        want.extend(fold_region_ops(&lclaims));
        assert_eq!(lops, want, "the lane tape shape");
        assert_eq!(lrec.payloads()[0], ln.registry.digest(), "lane registry digest");
        assert_eq!(
            lrec.payloads()[1],
            vec![ln.priors.len() as u8],
            "lane prior count"
        );
        let llocs = locate_and_pin_folds(&lclaims, &lproofs, &lvals, &lchals);
        let louts = replay_fold_endpoints(&llocs, &lvals, &lchals);
        assert_eq!(louts[0], lacc_v.per_type[0].0, "lane boolean A");
        assert_eq!(louts[1], lacc_v.per_type[0].1, "lane boolean B");
        let (ld, lc2) = lacc_v.sigma.as_ref().expect("lane sigma out");
        assert_eq!(louts[2], *lc2, "lane sigma accumulator");
        assert_eq!(*ld, ln.circuit.digest(), "lane sigma keys by the chain circuit");
        let lstream = lrec.shape().stream_words_duplex(LANE_DOMAIN);
        let lbytes = lstream.to_bytes(lrec.values(), lrec.payloads());
        (lacc_v, llocs, lstream, lbytes, lops, lchals, lvals)
    });

    // ---- ONE outer: two REAL child regions + the fold region ----
    let outer_stats = {
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChainSponge;

        let stream = t_shape.stream_words_duplex(M11_NODE_DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChainSponge::new();
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

        let b3_rows = rts.iter().map(|rt| rt.b3_rows).sum::<usize>() + trace.rows.len();
        // MEASURED AND REJECTED (2026-08-05): over-provisioning nu by one
        // bit to re-engage the pay-per-live arms. Boolean committed area was
        // then CAPACITY-shaped (M_bool 31→32 doubled the boolean stack; the
        // open went 84→190 ms and level-1 prove 260→390). RE-MEASURED
        // 2026-08-05 post-stratified/slim/live via TOWER_NU_BUMP=1 (slim L2,
        // steady): +1 nu costs only +7.3 ms prove (wiring +2.0 at μ+1, open
        // +2.4, element PIOP +0.7, witgen +0.4; commit and boolean zc+lc
        // FLAT, dense_m HELD at 28 — the committed stack is content-derived)
        // — the old doubling no longer reproduces, so a nu-14 squeeze buys
        // only ~5-7 ms and no proof bytes. The knob stays as the capacity-
        // sensitivity probe.
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7)
            + std::env::var("TOWER_NU_BUMP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
        // Under the envelope the node pins nu* and the canonical type set
        // (wall 2); the TOWER_NU_BUMP capacity probe is an off-envelope
        // knob — the pin wins.
        let env = envelope_shape();
        let nu2 = match &env {
            Some(e) => {
                assert!(
                    nu2_content <= e.nu,
                    "node content nu {nu2_content} exceeds the envelope nu* {}",
                    e.nu
                );
                e.nu
            }
            None => nu2_content,
        };
        let mut sb = ShapeBuilder::new(nu2);
        // Under the envelope the DECLARED width is the envelope's (the max
        // over child kinds at the fixed point); a shallower child ladder
        // rides the wide slot with its high outputs unread, and one that
        // exceeds it fails here. The witness tables below build at
        // `spread_w2`, so it must be the DECLARED width.
        let spread_own2 = rts.iter().map(|rt| rt.spread_w).max().expect("a child");
        let (spread_w2, mut cs) = match &env {
            Some(e) => {
                assert!(
                    spread_own2 <= e.spread_w,
                    "child ladder depth {spread_own2} exceeds the envelope spread width {}",
                    e.spread_w
                );
                (e.spread_w, ChildSlots::new_env(&mut sb, nu2, e))
            }
            None => (spread_own2, ChildSlots::new(&mut sb, nu2, spread_own2)),
        };
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        // The two child regions are independent gate subgraphs (each reads
        // only its own tape's inputs; the fold region joins them AFTER) —
        // declared as islands so the online phase evaluates them in
        // parallel.
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let mac_c0_start = sb.rows_in_slot(cs.macs);
        let mut mac_marks: Vec<usize> = Vec::with_capacity(n_kids);
        let regions: Vec<RealRegion> = rts
            .iter()
            .map(|rt| {
                let isl = sb.begin_island();
                let r =
                    emit_real_child_region(&mut sb, &mut cs, rt, &mut vals, &mut hints, &mut consts);
                sb.end_island(isl);
                mac_marks.push(sb.rows_in_slot(cs.macs));
                r
            })
            .collect();
        let r0 = &regions[0];
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
        let mac_after_fold = sb.rows_in_slot(cs.macs);

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
        // The lagrange-low constants, shared by both children: the 64 φ8
        // nodes and the subspace denominator inverse — statement constants
        // the checker validates below (the ONE public surface the
        // in-circuit derivation adds).
        use flock_core::field::PHI_8_TABLE;
        use flock_core::zerocheck::K_SKIP;
        use flock_core::zerocheck::multilinear::{
            lagrange_weights_naive, subspace_denominator_pair,
        };
        let lam_base = sb.public_len();
        let lam_w: Vec<Wire> = PHI_8_TABLE[..1 << K_SKIP]
            .iter()
            .map(|&v| {
                vals.push(v);
                sb.public_input()
            })
            .collect();
        vals.push(subspace_denominator_pair(K_SKIP).1);
        let deninv_w = sb.public_input();
        // The lows' assert-zero anchor: producers only, no consumer edges.
        vals.push(F128::ZERO);
        let lag_zassert = sb.public_input();
        for (k, (tk, rk)) in rts.iter().zip(&regions).enumerate() {
            // The lagrange row lows, IN-CIRCUIT from the child's z_skip wire
            // (native pre-assert first: the fold's absorbed lows ARE the closed
            // form at the located z_skip).
            assert_eq!(
                &fold_claims[0][k].row.low[..],
                &lagrange_weights_naive(K_SKIP, tk.chals[tk.zskip_ch])[..],
                "child {k}: the fold's lagrange lows are the closed form"
            );
            let lows = emit_lagrange_lows(
                &mut sb,
                cs.macs,
                &lam_w,
                deninv_w,
                rk.zskip_w,
                tk.chals[tk.zskip_ch],
                &mut vals,
                zw,
                ow,
                lag_zassert,
            );
            for (j, &lw2) in lows.iter().enumerate() {
                sb.connect(lw2, wv(locs[0].claims[k].row_low_v + j));
            }
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

        // Publishes: per fold, deltas + accumulator claim. This is the
        // ENVELOPE-registry surface a parent inherits, so under the
        // envelope it rides the reserved ACC_MAIN block at a constant
        // index; off-envelope it publishes inline, as before.
        let mut acc_main_w: Vec<Wire> = Vec::new();
        for fp in &fold_pubs {
            acc_main_w.extend_from_slice(&fp.rho_col);
            acc_main_w.extend_from_slice(&fp.rho_row);
            acc_main_w.push(fp.value);
        }
        let fold_pub_base = match &env {
            Some(e) => env_acc_main_base(e),
            None => {
                let b = sb.public_len();
                for &w in &acc_main_w {
                    sb.publish(w);
                }
                b
            }
        };
        // ---- the APPLICATION STATEMENT (hash-chain adjacency) ----
        // When the children carry an app block: left.h_end == right.h_start
        // as four copy constraints (both children's publics are witness
        // wires here), and the combined span publishes as THIS node's block.
        // The adjacency connects happen here; the PUBLISH of the combined
        // span moves to the envelope's fixed tail block (below, with the
        // padding) so its offset is level-independent. Off-envelope it
        // publishes inline, exactly as before.
        // ADJACENCY CHAINS ACROSS EVERY CONSECUTIVE PAIR: child i's h_end
        // is child i+1's h_start, so the node's own span is the first
        // child's h_start and the last child's h_end — the same statement
        // whatever the arity.
        let app_w: Option<Vec<Wire>> = app_stmt.map(|off| {
            for w in regions.windows(2) {
                for j in 0..4 {
                    sb.connect(w[0].child_pub_w[off + 4 + j], w[1].child_pub_w[off + j]);
                }
            }
            let last = &regions[n_kids - 1];
            (0..4)
                .map(|j| regions[0].child_pub_w[off + j])
                .chain((0..4).map(|j| last.child_pub_w[off + 4 + j]))
                .collect()
        });
        let app_inline = match &env {
            Some(_) => None,
            None => app_w.as_ref().map(|v| {
                let base = sb.public_len();
                for &w in v {
                    sb.publish(w);
                }
                base
            }),
        };
        // ---- the LANE fold region, in-circuit: priors-only, every prior
        // surface WIRED to the child's published accumulator claim (a
        // prior's surface IS what the child published — the child_pub_w
        // words at claims_base, layout [rho_col | rho_row | value] per
        // group), lows to the constant 1. Its own chain block rides the
        // shared b3 slot; the fold rows the shared mac/mrs/prefix slots.
        let lane_pub = lane_native.as_ref().map(|ln2| {
            let (_, llocs, lstream, lbytes, lops, lchals, lvals) = ln2;
            let lane_ref = lane.as_ref().expect("lane native implies lane");
            let mut lchain = FsChainSponge::new();
            let mut at = 0usize;
            let lfin: Vec<_> = lops.iter().filter(|o| o.finalizes()).collect();
            assert_eq!(lstream.finalize_after.len(), lfin.len(), "lane finalize alignment");
            for (k, &upto) in lstream.finalize_after.iter().enumerate() {
                lchain.absorb(&lbytes[at * 16..upto * 16]);
                at = upto;
                lchain.finalize(lfin[k].squeezed_bytes());
            }
            lchain.absorb(&lbytes[at * 16..]);
            let ltrace = lchain.finish();
            let lpub_payloads = bytes_payload_mask(lops);
            let (lchain_outs, lww) = emit_fs_chain(
                &mut sb,
                cs.q.b3,
                iv2,
                &ltrace,
                lstream,
                lbytes,
                &mut vals,
                &mut consts,
                &lpub_payloads,
            );
            let mut lvmap: Vec<Option<usize>> = Vec::new();
            for (wi, w) in lstream.words.iter().enumerate() {
                if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
                    if lvmap.len() <= vi {
                        lvmap.resize(vi + 1, None);
                    }
                    lvmap[vi] = Some(wi);
                }
            }
            let lwv =
                |vi: usize| -> Wire { lww[lvmap[vi].expect("lane word")].expect("lane wired") };
            let (lfold_pubs, lalpha_recs) = emit_fold_region(
                &mut sb,
                cs.macs,
                cs.mrs,
                pfslot,
                pf_w,
                leslot,
                llocs,
                &ltrace.squeezes,
                &lchain_outs,
                &lww,
                &lvmap,
                lchals,
                lvals,
                &mut vals,
                zw,
                ow,
            );
            for (k, rk) in regions.iter().enumerate() {
                let mut off = lane_ref.claims_base;
                for loc in llocs {
                    let cl = &loc.claims[k];
                    for j in 0..cl.col_pt_n {
                        sb.connect(lwv(cl.col_pt_v + j), rk.child_pub_w[off + j]);
                    }
                    for j in 0..cl.row_pt_n {
                        sb.connect(lwv(cl.row_pt_v + j), rk.child_pub_w[off + loc.k_col + j]);
                    }
                    sb.connect(
                        lwv(cl.value_v),
                        rk.child_pub_w[off + loc.k_col + loc.k_row],
                    );
                    sb.connect(lwv(cl.row_low_v), ow);
                    sb.connect(lwv(cl.col_low_v), ow);
                    off += loc.k_col + loc.k_row + 1;
                }
            }
            // The lane's claims are the LOWER-registry surface a parent
            // inherits: under the envelope they ride the reserved
            // ACC_CHAIN block — the same constant index at which an FL
            // child exposes its own chain fold.
            let mut lane_w: Vec<Wire> = Vec::new();
            for fp in &lfold_pubs {
                lane_w.extend_from_slice(&fp.rho_col);
                lane_w.extend_from_slice(&fp.rho_row);
                lane_w.push(fp.value);
            }
            let lane_words = lane_w.len();
            let lane_pub_base = match &env {
                Some(e) => env_acc_chain_base(e),
                None => {
                    let b = sb.public_len();
                    for &w in &lane_w {
                        sb.publish(w);
                    }
                    b
                }
            };
            (lane_pub_base, lane_words, lalpha_recs, lane_w)
        });

        if std::env::var("MAC_CENSUS").is_ok() {
            let mac_total = sb.rows_in_slot(cs.macs);
            println!("\nMAC ROW CENSUS (shared mac slot; child 0 labels, child 1 same shape):");
            for w in r0.census.windows(2) {
                if w[1].2 != w[0].2 {
                    println!("  {:42} {:6}", w[1].0, w[1].2 - w[0].2);
                }
            }
            let mut prev = mac_c0_start;
            for (i, &mk) in mac_marks.iter().enumerate() {
                println!("  {:42} {:6}", format!("= child {i} region"), mk - prev);
                prev = mk;
            }
            println!("  {:42} {:6}", "fold region", mac_after_fold - prev);
            println!("  {:42} {:6}", "lagrange lows + tail", mac_total - mac_after_fold);
            println!("  {:42} {:6}", "TOTAL", mac_total);
        }
        if std::env::var("PUB_CENSUS").is_ok() {
            println!("\nPUBLICS CENSUS (child 0; child 1 same shape):");
            for w in r0.census.windows(2) {
                println!("  {:38} {:6}", w[1].0, w[1].1 - w[0].1);
            }
            let child = r0.census.last().unwrap().1 - r0.census[0].1;
            println!("  {:38} {:6}", "= CHILD TOTAL", child);
            let tail_len: usize = locs.iter().map(|l| 1 + l.k_col + l.k_row).sum();
            println!("  {:38} {:6}", "lagrange consts", 66usize);
            println!("  {:38} {:6}", "fold region publics", tail_len);
            println!("  {:38} {:6}", "TOTAL (2 children + shared)", sb.public_len());
        }
        let build_ms = t_tapes.elapsed().as_secs_f64() * 1e3 - tape_setup_ms;
        let t_build2 = std::time::Instant::now();
        // counts* + publics*: the node declares the same count vector and
        // segment length the leaf does. The tail-anchor assert below walks
        // the REAL segment end, recorded pre-pad.
        let prepad_publics2 = sb.public_len();
        let app_base = match &env {
            Some(e) => {
                let empty: Vec<Wire> = Vec::new();
                pad_envelope_counts(
                    &mut sb,
                    &cs.q,
                    &cs.env_cache(),
                    e,
                    zw,
                    &mut hints,
                    &mut vals,
                    &EnvTail {
                        acc_main: &acc_main_w,
                        acc_chain: lane_pub.as_ref().map(|(_, _, _, w)| w).unwrap_or(&empty),
                        app: app_w.as_deref().unwrap_or(&empty),
                    },
                );
                app_w.as_ref().map(|_| env_app_base(e))
            }
            None => app_inline,
        };
        let shape2 = sb.finish().expect("the 2->1 node circuit builds");
        assert!(
            shape2.circuit.cells().slots().len() <= 512,
            "the node's cell-slot budget regressed ({} slots)",
            shape2.circuit.cells().slots().len()
        );
        // ROUND-3 DATA (NODE_CENSUS=1): per-type schema words — each one a
        // cell slot AND a gather claim — plus live rows and utilization,
        // the consolidation pass's worklist.
        if std::env::var("NODE_CENSUS").is_ok() {
            let mut lab: Vec<(usize, String)> = vec![
                (shape2.registry_slot(cs.q.b3), "b3".to_string()),
                (shape2.registry_slot(cs.q.swap), "swap".to_string()),
                (shape2.registry_slot(cs.q.spread), "spread".to_string()),
                (shape2.registry_slot(cs.macs), "mac".to_string()),
                (shape2.registry_slot(cs.zcr), "zcr".to_string()),
                (shape2.registry_slot(cs.mrs), "mrs".to_string()),
                (shape2.registry_slot(cs.spine), "spine".to_string()),
                (shape2.registry_slot(cs.alslot), "assist".to_string()),
            ];
            for &(n, s) in &cs.le {
                lab.push((shape2.registry_slot(s), format!("le{n}")));
            }
            for &(k, s) in &cs.resid {
                // Decode the cache's key scheme (see ChildSlots::resid);
                // the shared mac (600) is already labeled above.
                let name = match k {
                    600 => continue,
                    k if k >= 310 => format!("pf{}", k - 310),
                    k => format!("resid{}", k - 100),
                };
                lab.push((shape2.registry_slot(s), name));
            }
            println!("  NODE TYPE CENSUS (io = cell slots = gather claims):");
            let (mut area_b, mut area_e) = (0usize, 0usize);
            for (t, ty) in shape2.registry.types().iter().enumerate() {
                let name = lab
                    .iter()
                    .find(|(i, _)| *i == t)
                    .map(|(_, s)| s.as_str())
                    .unwrap_or("?");
                // Mirrors UnionInstance::used_cols: word-columns that carry
                // data (a boolean type's GF(2) columns bit-pack 128/word; an
                // element type's useful_bits is element_cols * 128).
                let used_cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
                let area = shape2.counts[t] * used_cols;
                let native = match ty.class {
                    flock_core::schedule::TableClass::Boolean => {
                        area_b += area;
                        format!("GF(2)     {:6} bit-cols", ty.useful_bits)
                    }
                    _ => {
                        area_e += area;
                        format!("GF(2^128) {:6} el-cols ", ty.useful_bits / 128)
                    }
                };
                println!(
                    "    t{t:2} {name:>8} | {native} = {used_cols:3} word-cols | io {:3} | \
                     rows {:6} ({:3}%) | area {area:9} words",
                    ty.io_schema.len(),
                    shape2.counts[t],
                    (100 * shape2.counts[t]) >> nu2,
                );
            }
            println!(
                "    class areas: GF(2) {area_b} + GF(2^128) {area_e} = dense {} words",
                area_b + area_e,
            );
        }
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> =
            hints.iter().map(|h| h as &(dyn std::any::Any + Sync)).collect();
        let build_ms = build_ms + t_build2.elapsed().as_secs_f64() * 1e3;
        // THE INDEX-FILL RUNNER (setup): compile the fill plan, then pin it
        // row-identical against the generic walk before the online loop
        // trusts it — publics, the three boolean row stores, and every
        // element slot's packed witness, field for field. The walk stays the
        // differential oracle; only the plan runs in the timed loop.
        let t_plan = std::time::Instant::now();
        let fill_plan = shape2.fill_plan();
        let build_ms = build_ms + t_plan.elapsed().as_secs_f64() * 1e3;
        {
            let walk = shape2.run(&vals, &hint_refs);
            let fill = shape2.run_filled(&fill_plan, &vals, &hint_refs);
            assert_eq!(walk.public, fill.public, "fill plan: public segment");
            assert_eq!(walk.witnesses, fill.witnesses, "fill plan: slot witnesses");
            assert_eq!(
                walk.rows::<Blake3Gate>(cs.q.b3),
                fill.rows::<Blake3Gate>(cs.q.b3),
                "fill plan: b3 rows"
            );
            assert_eq!(
                walk.rows::<SwapGate>(cs.q.swap),
                fill.rows::<SwapGate>(cs.q.swap),
                "fill plan: swap rows"
            );
            assert_eq!(
                walk.rows::<BitSpreadGate>(cs.q.spread),
                fill.rows::<BitSpreadGate>(cs.q.spread),
                "fill plan: spread rows"
            );
        }
        // The node proves and verifies over the circuit path. Union, PCS
        // params and the R1CS tables are per-SHAPE — offline, ahead of the
        // online loop.
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = tower_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: union2.commit_lanes(pcs_batch_for(&union2, pf)),
            // BLAKE3 for BOTH Merkle and FS: the node's proof must be
            // RECURSABLE — a parent replays this transcript in-circuit,
            // and each default diverges silently (the two recorded
            // gotchas, third occurrence).
            merkle_hash: HashKind::Blake3,
        };
        let t_r1cs = std::time::Instant::now();
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_r1cs2 = BitSpreadTable::new(spread_w2).build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        let build_ms = build_ms + t_r1cs.elapsed().as_secs_f64() * 1e3;
        // TOWER_STEADY=N re-runs the ONLINE phases (trace + asm + prove +
        // verify) N extra times over the SAME built shape: the offline
        // setup (circuit, R1CS, PCS params, warmed pools) is paid once, so
        // iterations after the first print the steady-state online cost.
        let steady_reps: usize = std::env::var("TOWER_STEADY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut steady_left = steady_reps;
        let (built2, oproof, ocommit, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms) =
            loop {
        // Tapes are statement work: re-run them each online iteration
        // (results discarded — identical by determinism) so the printed
        // number is the steady-state cost, not the first-touch one.
        // The ONLINE tape cost: two recorded deferred child verifies (the
        // production statement work). The pin/locate scaffolding ran once
        // above (tape_setup_ms) — its indices are shape-stable.
        let tapes_ms = {
            let t = std::time::Instant::now();
            {
                use rayon::prelude::*;
                los.par_iter().for_each(|lo| record_child_verify(lo, DOMAIN));
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        let t_trace = std::time::Instant::now();
        // DEFERRED: rows and publics only — the element witnesses are never
        // packed; the assembly below feeds the prover from the rows.
        let mut built2 = shape2.run_filled_deferred(&fill_plan, &vals, &hint_refs);
        let trace_ms = t_trace.elapsed().as_secs_f64() * 1e3;

        // The two child regions' checker walks — each child's whole
        // deferred-verifier statement held against its own replicas.
        let consumed: Vec<usize> = rts
            .iter()
            .zip(&regions)
            .map(|(rt, r)| check_real_child_region(&built2.public, rt, r))
            .collect();
        for i in 0..n_kids {
            let end = regions[i].pub_base + consumed[i];
            let next = if i + 1 < n_kids {
                regions[i + 1].pub_base
            } else {
                fold_pub_base
            };
            assert!(
                end <= next,
                "child {i}'s public block overruns the next region"
            );
        }
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
        // The lagrange-low constants: the one public surface the in-circuit
        // derivation adds — validated against the verifier's own values.
        {
            for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                assert_eq!(built2.public[lam_base + i], v, "λ const {i}");
            }
            assert_eq!(
                built2.public[lam_base + (1 << K_SKIP)],
                subspace_denominator_pair(K_SKIP).1,
                "the subspace denominator inverse const"
            );
            assert_eq!(
                built2.public[lam_base + (1 << K_SKIP) + 1],
                F128::ZERO,
                "the lows' assert-zero anchor"
            );
            // OFF-ENVELOPE the REAL segment ends at the last publish block:
            // the lane's fold blocks when a lane rides (its emitter also
            // declares its own boundary publics before them), else the
            // node's fold blocks + the app block. UNDER the envelope those
            // blocks moved to the reserved tail, so the body simply has to
            // fit — which `pad_envelope_counts` asserts — and the tail
            // layout is checked where it matters, by rebuilding both
            // accumulators at their CONSTANT bases below.
            if env.is_none() {
                let seg_end = lane_pub
                    .as_ref()
                    .map(|&(b, w, _, _)| b + w)
                    .unwrap_or(
                        fold_pub_base
                            + tail_len
                            + if app_inline.is_some() { ENV_APP_WORDS } else { 0 },
                    );
                assert_eq!(
                    seg_end, prepad_publics2,
                    "the last publish block ends the REAL segment"
                );
            }
            // The LANE accumulator, reassembled from the public segment
            // alone — the parent-facing statement of the lower registry.
            if let (Some((lpb, _, lar, _)), Some((lacc_n, llocs, ..))) =
                (lane_pub.as_ref(), lane_native.as_ref())
            {
                let lrebuilt = check_fold_publics(&built2.public, *lpb, llocs, lar);
                let lane_ref = lane.as_ref().expect("lane");
                let lacc_pub2 = aggregate::Accumulator {
                    registry_digest: lane_ref.registry.digest(),
                    per_type: vec![(lrebuilt[0].clone(), lrebuilt[1].clone())],
                    per_element: Vec::new(),
                    sigma: Some((lane_ref.circuit.digest(), lrebuilt[2].clone())),
                };
                assert_eq!(
                    &lacc_pub2, lacc_n,
                    "the LANE accumulator, reassembled from publics alone"
                );
            }
        }

        let t_asm = std::time::Instant::now();
        // Recreated per online iteration — the spread closure consumes it.
        let spread_ty2 = BitSpreadTable::new(spread_w2);
        // The copy-free assembly path: the boolean drivers pack straight
        // into the union slot blocks inside the prove (live rows only under
        // elide) — no intermediate capacity-sized buffers, no memcpy. The
        // rows are hoisted to owned Vecs because the closures must be Send
        // and `built2.rows` hands out `dyn Any`-backed borrows.
        let b3_rows2 = built2.rows::<Blake3Gate>(cs.q.b3).to_vec();
        let swap_rows2 = built2.rows::<SwapGate>(cs.q.swap).to_vec();
        let spread_rows2 = built2.rows::<BitSpreadGate>(cs.q.spread).to_vec();
        let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape2.registry_slot(cs.q.b3),
                UnionSlotProverInput::in_place(
                    move |dst| {
                        blake3::generate_witness_batch_major_partial_into(&b3_rows2, nu2, dst)
                    },
                    b3_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.swap),
                UnionSlotProverInput::in_place(
                    move |dst| SwapTable::generate_witness_batch_major_into(&swap_rows2, dst),
                    swap_lc2,
                ),
            ),
            (
                shape2.registry_slot(cs.q.spread),
                UnionSlotProverInput::in_place(
                    move |dst| spread_ty2.generate_witness_batch_major_into(&spread_rows2, dst),
                    spread_lc2,
                ),
            ),
        ];
        bslots.sort_by_key(|(i, _)| *i);
        // Element inputs straight from the slots' rows: the run was
        // DEFERRED, so the full-capacity packed intermediate never exists —
        // the prove's in_place closure scatters the live rows directly.
        let mut el_ord: Vec<(usize, Vec<Vec<F128>>)> = cs
            .element_slot_ids()
            .into_iter()
            .map(|sl| {
                (
                    shape2.registry_slot(sl),
                    built2.take_rows_of::<Vec<F128>>(sl),
                )
            })
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(_, rows)| live_element_input_from_rows(rows, nu2))
            .collect();
        let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (shape2.registry_slot(cs.q.b3), b3_lc2),
            (shape2.registry_slot(cs.q.swap), swap_lc2),
            (shape2.registry_slot(cs.q.spread), spread_lc2),
        ];
        lco.sort_by_key(|(i, _)| *i);
        let lcs2: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lco.into_iter().map(|(_, c)| c).collect();
        let asm_ms = t_asm.elapsed().as_secs_f64() * 1e3;
        let t0p = std::time::Instant::now();
        let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
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
        let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
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
        let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
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
        println!(
            "\nTHE 2->1 RECURSION NODE (two children + {} folds, ONE proof)\n  \
             children: dense_m {} / mu {}, one circuit, distinct FS points\n  \
             regions: 2x the complete deferred verifier (swap assembly, shared slots)\n         \
             + the fold region; CONNECTED: all points, z_partial lows, sigma fully,\n         \
             and the matrix/element EVAL VALUES to the children's bound advice —\n         \
             lagrange lows DERIVED in-circuit from each child's z_skip wire\n  \
             outer: total b3 rows {} | nu {} | dense_m {} | mu {} \
             (cell slots: {} gate + {} public)\n  \
             PER PROOF (online): child tapes {:.0} + witgen/trace {:.0} + witness asm {:.0} + prove {:.0} \
             = {:.0} ms | verify {:.0} ms (DEFERRED {:.0} ms) | proof {:.1} KiB\n  \
             SETUP: circuit build (per SHAPE, cacheable) {:.0} ms | tape pins+locates (shape-stable) {:.0} ms\n",
            n_folds,
            lo0.pcs.m,
            rts[0].mu_i,
            b3_rows,
            nu2,
            union2.dense_m(),
            shape2.circuit.cells().mu(),
            shape2.circuit.cells().num_gate_slots(),
            shape2.circuit.cells().num_public_slots(),
            tapes_ms,
            trace_ms,
            asm_ms,
            prove_ms,
            tapes_ms + trace_ms + asm_ms + prove_ms,
            verify_ms,
            deferred_ms,
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
            build_ms,
            tape_setup_ms,
        );
        if steady_left > 0 {
            steady_left -= 1;
            continue;
        }
        break (built2, oproof, ocommit, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms);
        };
        let (b3_slot2, swap_slot2, spread_slot2) = (
            shape2.registry_slot(cs.q.b3),
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
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
            Online {
                setup_ms: build_ms,
                walk_ms: trace_ms,
                tapes_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms,
            },
            app_base,
            lane_native.map(|(a, ..)| a),
        )
    };
    outer_stats
}

/// **Task 5: THE INTERNAL NODE carries the chain statement.** Four chain
/// segments → two first-level nodes → ONE internal node, built by
/// [`build_node_outer_app`]'s own machinery over the FL [`LeafOuter`]s
/// (RealTape walks an FL tape here for the first time): the FL-level
/// adjacency (fl0.h_end == fl1.h_start) is checked wire-to-wire at the
/// internal level through the children's witness publics, and the combined
/// span publishes as the internal node's own app block == the native
/// H^1024(h_start). Accumulators are per-level at the dev shape — the
/// cross-level threading (chain accs as PRIORS of the internal fold) is
/// task 6.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn internal_node_over_two_fl_nodes() {
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0006);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(h0, n_blocks);
    let cp1 = build_chain_proof(cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cp2.h_end, n_blocks);
    let fl0 = build_fl_node(&cp0, &cp1);
    let fl1 = build_fl_node(&cp2, &cp3);
    assert_eq!(
        fl0.lo.shape.circuit.digest(),
        fl1.lo.shape.circuit.digest(),
        "one first-level circuit digest — the FL shape is data-independent"
    );
    assert_eq!(fl0.stmt_base, fl1.stmt_base, "one statement offset");
    assert_eq!(fl1.h_start, fl0.h_end, "the FL spans are adjacent");

    let (node, acc, _t, app, _lane) =
        build_node_outer_app(&[&fl0.lo, &fl1.lo], Some(fl0.stmt_base), None);
    let app = app.expect("the internal node carries the app block");
    for j in 0..4 {
        assert_eq!(
            node.public[app + j],
            pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "internal statement: h_start is the whole span's start"
        );
        assert_eq!(
            node.public[app + 4 + j],
            pack4(cp3.h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "internal statement: h_end is the whole span's end"
        );
    }
    assert_eq!(
        cp3.h_end,
        native_chain(&cp0.h_start, 4 * n_blocks),
        "the internal span IS the 1024-step chain"
    );
    // Per-level accumulators at the dev shape: the internal node's own acc
    // keys sigma by the FL circuit digest; the chain-level accs live in the
    // FlNodes. (Task 6 threads them as priors.)
    let (sig_digest, _) = acc.sigma.as_ref().expect("the node accumulated sigma");
    assert_eq!(
        *sig_digest,
        fl0.lo.shape.circuit.digest(),
        "the internal accumulator keys by the FL circuit"
    );
    // **TASK 7b's PIN: an FL node and an internal node are ONE ENVELOPE.**
    // Same registry digest (wall 2), same declared count vector (counts*),
    // same public-segment length with the app block at the same fixed
    // offset (publics*) — so a parent's walk cannot tell an FL child from an
    // internal child, which is what makes one internal circuit serve every
    // level above the first.
    if envelope_shape().is_some() {
        assert_eq!(
            fl0.lo.shape.registry.digest(),
            node.shape.registry.digest(),
            "FL and internal share ONE envelope registry"
        );
        assert_eq!(
            fl0.lo.shape.counts, node.shape.counts,
            "FL and internal declare ONE count vector"
        );
        assert_eq!(
            fl0.lo.public.len(),
            node.public.len(),
            "FL and internal share ONE public-segment length"
        );
        assert_eq!(
            app, fl0.stmt_base,
            "the app block sits at the envelope's fixed offset for BOTH child kinds"
        );
        assert_eq!(fl0.lo.pcs.m, node.pcs.m, "one dense floor m*");
    }
    println!(
        "\nINTERNAL NODE over two first-level nodes (app-statement plumbed)\n  \
         span: H^{}(h_start) | internal outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        4 * n_blocks,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        node.public.len(),
        bincode::serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **THE 3-ARY INTERNAL NODE.** Commit and open are FLOOR-bound — they
/// cost the same whatever the arity, as long as content stays under
/// `2^(m*-7)` — so they are a per-node toll that every child past the first
/// rides for free, and a k-ary layer needs `1/(k-1)` as many nodes. Six
/// chain segments → three first-level nodes → ONE internal node folding all
/// three, lane included (three priors, not two).
///
/// The prerequisite is `nu* = 16`: mac is ~97% per-child work (14,411 rows
/// per child against 921 shared), so three children need ~44k rows against
/// 2^15's 32,768.
#[test]
#[ignore] // Heavy — six chain proofs, three FLs, one 3-ary node.
fn internal_node_three_ary() {
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_000C);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..6 {
        let cp = build_chain_proof(h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fls: Vec<FlNode> = (0..3).map(|i| build_fl_node(&cps[2 * i], &cps[2 * i + 1])).collect();
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let priors: Vec<&flock_core::aggregate::Accumulator> = fls.iter().map(|f| &f.acc).collect();
    let kids: Vec<&LeafOuter> = fls.iter().map(|f| &f.lo).collect();

    let (node, acc, t, app, lane_acc) = build_node_outer_app(
        &kids,
        Some(fls[0].stmt_base),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: &cps[0].inner.built.shape.circuit,
            priors: &priors,
            claims_base: fls[0].fold_pub_base,
        }),
    );
    let app = app.expect("the app block rode");
    let lane_acc = lane_acc.expect("the lane rode");

    // The statement spans all six segments, and both accumulators discharge.
    let h_end = native_chain(&h0, 6 * n_blocks);
    for j in 0..4 {
        assert_eq!(
            node.public[app + j],
            pack4(h0[4 * j..4 * j + 4].try_into().unwrap()),
            "3-ary node h_start"
        );
        assert_eq!(
            node.public[app + 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "3-ary node h_end == H^N(h_start)"
        );
    }
    assert!(
        lane_acc.discharge(&chain_mats) && lane_acc.discharge_sigma(&cps[0].inner.built.shape.circuit),
        "the 3-prior chain lane discharges"
    );
    // The accumulator holds claims about the CHILDREN's tables, so it
    // discharges against THEIR matrices — which are the node's own only
    // because the envelope pins one nu and one registry.
    let ch = &fls[0].lo;
    let el_types: Vec<_> = ch
        .shape
        .registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<_> = el_types.iter().map(|ty| (ty.a_0(), ty.b_0())).collect();
    let mut mats_ord = vec![
        (ch.b3_slot, (&ch.b3_r1cs.a_0, &ch.b3_r1cs.b_0)),
        (ch.swap_slot, (&ch.swap_r1cs.a_0, &ch.swap_r1cs.b_0)),
        (ch.spread_slot, (&ch.spread_r1cs.a_0, &ch.spread_r1cs.b_0)),
    ];
    mats_ord.sort_by_key(|&(i, _)| i);
    let mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
    assert!(
        acc.discharge(&mats) && acc.discharge_element(&el_mats)
            && acc.discharge_sigma(&fls[0].lo.shape.circuit),
        "the 3-ary node's own accumulator discharges all three groups"
    );
    println!(
        "\n3-ARY INTERNAL NODE (three FL children in ONE proof)\n  \
         span H^{}(h_start) | nu {} | mu {} | publics {} | proof {:.1} KiB\n  \
         ONLINE: walk {:.1} + tapes {:.1} + witgen {:.1} + prove {:.1} = {:.1} ms | verify {:.1}\n  \
         per-leaf internal share (k=3): {:.1} ms vs 2-ary's C/2\n",
        6 * n_blocks,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        node.public.len(),
        bincode::serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        t.walk_ms,
        t.tapes_ms,
        t.witgen_ms,
        t.prove_ms,
        t.total(),
        t.verify_ms,
        t.total() / 4.0,
    );
}

/// **TASK 7b's HEADLINE: ONE INTERNAL CIRCUIT, EVERY LEVEL.** Eight chain
/// segments → four first-level nodes → two internal nodes → one THIRD-level
/// node, and the level-3 circuit digest EQUALS the level-2 one: once the FL
/// node is envelope-shaped (nu*, counts*, publics*, m*, the app block at
/// `env_app_base`), a parent's walk cannot tell an FL child from an internal
/// child, so the same circuit serves both — the tower is depth-unbounded in
/// SHAPE.
///
/// **The chain LANE is threaded across BOTH levels here**, which is the
/// point of the reserved `ACC_CHAIN` block: a first-level child publishes
/// its own chain fold there and an internal child publishes its LANE's fold
/// there, at the SAME constant index — so the level-3 lane connects to its
/// internal children exactly as the level-2 lane connects to its FL
/// children, and the two circuits stay identical.
///
/// STILL OPEN (the fork in the handoff): each internal node's MAIN
/// (envelope-registry) accumulator is not yet inherited by its parent, so a
/// tower deeper than two levels is not yet sound end to end — the level-2
/// main accumulators would have to be folded as priors of the level-3 main
/// fold, which needs the FL-vs-internal sigma key pair. Their claims now
/// have a fixed home (`ACC_MAIN`) waiting for it. This test pins the SHAPE
/// and the lane mechanism.
#[test]
#[ignore] // Heavy — eight chain proofs and seven outers.
fn chain_tower_three_levels_one_internal_digest() {
    // The cross-level connects read the children's claims and statement at
    // CONSTANT indices, and those constants exist only under the envelope —
    // off-envelope every builder publishes inline, where the offsets depend
    // on live usage and an FL child and an internal child genuinely differ.
    let Some(env) = envelope_shape() else {
        println!(
            "\nTHREE-LEVEL CHAIN TOWER: skipped — the fixed-offset tail blocks the \
             cross-level\n  connects need exist only under the envelope \
             (TOWER_PROFILE=slim)\n"
        );
        return;
    };
    // 256: the registered Ligerito configs floor at m22, and a 128-block
    // chain commits at m21.
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_000B);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..8 {
        let cp = build_chain_proof(h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fls: Vec<FlNode> = (0..4).map(|i| build_fl_node(&cps[2 * i], &cps[2 * i + 1])).collect();
    let app_fl = fls[0].stmt_base;
    for f in &fls {
        assert_eq!(f.stmt_base, app_fl, "one FL app offset");
        assert_eq!(
            f.lo.shape.circuit.digest(),
            fls[0].lo.shape.circuit.digest(),
            "one FL circuit digest"
        );
    }
    // The lane's registry materials — the CHAIN side, shared by every level.
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_circuit = &cps[0].inner.built.shape.circuit;
    let acc_base = fls[0].fold_pub_base;
    assert_eq!(
        acc_base,
        env_acc_chain_base(&env),
        "an FL publishes its chain claims in the reserved ACC_CHAIN block"
    );

    // Level 2: the chain lane's priors are the FL children's chain
    // accumulators, read at the FL's ACC_CHAIN block.
    let (n0, _a0, _t0, app0, lane0) = build_node_outer_app(
        &[&fls[0].lo, &fls[1].lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            priors: &[&fls[0].acc, &fls[1].acc],
            claims_base: acc_base,
        }),
    );
    let (n1, _a1, _t1, app1, lane1) = build_node_outer_app(
        &[&fls[2].lo, &fls[3].lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            priors: &[&fls[2].acc, &fls[3].acc],
            claims_base: acc_base,
        }),
    );
    let app0 = app0.expect("level-2 app block");
    assert_eq!(app0, app1.expect("level-2 app block"), "one L2 app offset");
    assert_eq!(
        n0.shape.circuit.digest(),
        n1.shape.circuit.digest(),
        "one level-2 circuit digest"
    );
    let (lane0, lane1) = (lane0.expect("L2 lane"), lane1.expect("L2 lane"));

    // THE STEP THAT MATTERS: an internal node over two INTERNAL children,
    // its lane inheriting THEIR lane accumulators — published at the same
    // ACC_CHAIN index the FL children used, which is why one circuit reads
    // both kinds.
    let (n2, _a2, _t2, app2, lane2) = build_node_outer_app(
        &[&n0, &n1],
        Some(app0),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            priors: &[&lane0, &lane1],
            claims_base: acc_base,
        }),
    );
    let app2 = app2.expect("level-3 app block");
    let lane2 = lane2.expect("L3 lane");
    assert!(
        lane2.discharge(&chain_mats) && lane2.discharge_sigma(chain_circuit),
        "the level-3 chain lane discharges against the chain tables — \
         eight leaves' claims in one accumulator"
    );
    assert_eq!(
        n2.shape.circuit.digest(),
        n0.shape.circuit.digest(),
        "ONE internal circuit digest at level 3 as at level 2 — depth-unbounded shape"
    );
    assert_eq!(app2, app0, "the app block never moves");
    assert_eq!(app2, env_app_base(&env), "and it is the envelope's own tail");

    // The statement rode all three levels: the root span is the whole chain.
    let h_end = native_chain(&h0, 8 * n_blocks);
    for j in 0..4 {
        assert_eq!(
            n2.public[app2 + j],
            pack4(h0[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_start"
        );
        assert_eq!(
            n2.public[app2 + 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_end == H^N(h_start)"
        );
    }
    println!(
        "\nTHREE-LEVEL CHAIN TOWER (8 chains -> 4 FL -> 2 internal -> 1)\n  \
         span H^{}(h_start) | L2 digest == L3 digest | app at fixed offset {}\n  \
         internal outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        8 * n_blocks,
        app2,
        n2.shape.circuit.cells().nu(),
        n2.shape.circuit.cells().mu(),
        n2.public.len(),
        bincode::serialize(&n2.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **Task 6: THE CHAIN TOWER, END TO END, WITH THE LANE.** Four chain
/// segments → two first-level nodes → one internal node; the chain-level
/// accumulators ride the internal node as a PRIORS-ONLY LANE (their
/// registry differs from the FL fold's, so they cannot join it), with the
/// prior surfaces connected WIRE-TO-WIRE to the children's published
/// accumulator claims — mvp11's recorded prediction ("a prior's surface
/// IS what a previous outer publishes") landing. The ROOT then discharges
/// BOTH lanes — the chain lane against the chain b3 matrices + the chain
/// circuit's sigma table, the FL lane against the FL mats/element
/// types/digest — and reads the statement h_end == H^1024(h_start). Plus
/// the tamper matrix: a tampered FL statement word, a tampered lane
/// prior, and a tampered internal app word all die.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn chain_tower_e2e_with_lane() {
    use flock_core::aggregate;

    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0007);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(h0, n_blocks);
    let cp1 = build_chain_proof(cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cp2.h_end, n_blocks);
    let fl0 = build_fl_node(&cp0, &cp1);
    let fl1 = build_fl_node(&cp2, &cp3);
    assert_eq!(fl0.fold_pub_base, fl1.fold_pub_base, "one fold-block layout");

    // The lane's registry materials — the CHAIN side.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let (node, acc, _t, app, lane_acc) =
        build_node_outer_app(&[&fl0.lo, &fl1.lo], Some(fl0.stmt_base), Some(lane));
    let app = app.expect("the app block rode");
    let lane_acc = lane_acc.expect("the lane rode");

    // ---- THE ROOT ----
    // (1) The statement: the whole span, out of the internal node's publics.
    for j in 0..4 {
        assert_eq!(
            node.public[app + j],
            pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
        );
        assert_eq!(
            node.public[app + 4 + j],
            pack4(cp3.h_end[4 * j..4 * j + 4].try_into().unwrap()),
        );
    }
    assert_eq!(cp3.h_end, native_chain(&cp0.h_start, 4 * n_blocks));
    // (2) The CHAIN lane discharges: boolean vs the chain b3 matrices,
    // sigma vs the chain circuit's own (masked) sigma table.
    assert!(lane_acc.discharge(&chain_mats), "chain-lane boolean discharges");
    assert!(lane_acc.per_element.is_empty(), "the chain lane has no element group");
    assert!(
        lane_acc.discharge_sigma(&cp0.inner.built.shape.circuit),
        "chain-lane sigma discharges against the chain circuit"
    );
    // (3) The FL lane discharges: boolean vs the FL b3/swap/spread mats
    // (registry order), element vs the FL element types, sigma vs the FL
    // circuit digest's table.
    let mut mats_ord = vec![
        (fl0.lo.b3_slot, (&fl0.lo.b3_r1cs.a_0, &fl0.lo.b3_r1cs.b_0)),
        (
            fl0.lo.swap_slot,
            (&fl0.lo.swap_r1cs.a_0, &fl0.lo.swap_r1cs.b_0),
        ),
        (
            fl0.lo.spread_slot,
            (&fl0.lo.spread_r1cs.a_0, &fl0.lo.spread_r1cs.b_0),
        ),
    ];
    mats_ord.sort_by_key(|&(i, _)| i);
    let fl_mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
    assert!(acc.discharge(&fl_mats), "FL-lane boolean discharges");
    let fl_el_mats: Vec<_> = fl0
        .lo
        .shape
        .registry
        .element_types()
        .iter()
        .map(|t| {
            let e = t.element_type().expect("element table");
            (e.a_0(), e.b_0())
        })
        .collect();
    assert!(
        acc.discharge_element(&fl_el_mats),
        "FL-lane element discharges"
    );
    assert!(
        acc.discharge_sigma(&fl0.lo.shape.circuit),
        "FL-lane sigma discharges"
    );

    // ---- the tamper matrix ----
    // (a) A tampered FL STATEMENT word (its h_end): the FL proof must not
    //     verify against it — the adjacency data is statement-bound.
    {
        let union_f = outer_union(&fl0.lo.shape.registry, fl0.lo.shape.counts.clone());
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (fl0.lo.b3_slot, fl0.lo.b3_r1cs.csc_lincheck_circuit()),
            (fl0.lo.swap_slot, fl0.lo.swap_r1cs.csc_lincheck_circuit()),
            (fl0.lo.spread_slot, fl0.lo.spread_r1cs.csc_lincheck_circuit()),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs_f: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.into_iter().map(|(_, c)| c).collect();
        let mut bad = fl0.lo.public.clone();
        bad[fl0.stmt_base + 4] += F128::ONE;
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        assert!(
            verifier::verify_ligerito_union_circuit(
                &union_f,
                &fl0.lo.shape.circuit,
                &bad,
                &lcs_f,
                &fl0.lo.commitment,
                &fl0.lo.proof,
                &fl0.lo.pcs,
                &mut ch,
            )
            .is_err(),
            "a tampered FL h_end must be rejected"
        );
    }
    // (b) A tampered LANE PRIOR: the fold proof no longer matches.
    {
        let mut bad_acc = fl0.acc.clone();
        bad_acc.per_type[0].0.value += F128::ONE;
        let el_asserts_l: [(
            &UnionInstance<'_>,
            flock_core::element_r1cs::union::ElementAssertion,
        ); 0] = [];
        let mut chp = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        let (lagg, _) = aggregate::prove_aggregate_classes(
            chain_registry,
            &chain_mats,
            &chain_circs,
            &[],
            &[],
            &el_asserts_l,
            Some((&cp0.inner.built.shape.circuit, &[])),
            &[&fl0.acc, &fl1.acc],
            &mut chp,
        )
        .expect("honest lane fold proves");
        let mut ch = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        assert!(
            aggregate::verify_aggregate_classes(
                chain_registry,
                &[],
                &el_asserts_l,
                Some((&cp0.inner.built.shape.circuit, &[])),
                &[&bad_acc, &fl1.acc],
                &lagg,
                &mut ch,
            )
            .is_err(),
            "a tampered inherited claim must be rejected by the lane fold"
        );
    }
    // (c) A tampered INTERNAL app word: the internal proof's statement is
    //     bound the same way.
    {
        let union_n = outer_union(&node.shape.registry, node.shape.counts.clone());
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (node.b3_slot, node.b3_r1cs.csc_lincheck_circuit()),
            (node.swap_slot, node.swap_r1cs.csc_lincheck_circuit()),
            (node.spread_slot, node.spread_r1cs.csc_lincheck_circuit()),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs_n: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.into_iter().map(|(_, c)| c).collect();
        let mut bad = node.public.clone();
        bad[app + 7] += F128::ONE;
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        assert!(
            verifier::verify_ligerito_union_circuit(
                &union_n,
                &node.shape.circuit,
                &bad,
                &lcs_n,
                &node.commitment,
                &node.proof,
                &node.pcs,
                &mut ch,
            )
            .is_err(),
            "a tampered internal h_end must be rejected"
        );
    }

    println!(
        "\nCHAIN TOWER E2E (4 chains -> 2 FL -> 1 internal, lane threaded)\n  \
         root statement: h_end == H^{}(h_start) | both lanes discharge | tampers die\n  \
         internal outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        4 * n_blocks,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        node.public.len(),
        bincode::serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// Attribution probe for the chain leaf's prove: build the SAME m32 leaf
/// `LEAF_RUNS` times (fresh statement each run) and print the phase split;
/// run with `PCS_TRACE=1` for the prover's own internal breakdown. The
/// batch bench (`blake3_proof`, no wiring) is the comparison floor.
#[test]
#[ignore] // Measurement probe — run explicitly with --nocapture.
fn chain_leaf_prove_probe() {
    let n_blocks: usize = std::env::var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let runs: usize = std::env::var("LEAF_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let mut rng = Rng(0xC4A1_0009);
    for r in 0..runs {
        let h: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let cp = build_chain_proof(h, n_blocks);
        println!(
            "RUN {r}: setup {:.0} ms | walk {:.0} | witgen {:.0} | prove {:.0} ms",
            cp.t.setup_ms, cp.t.walk_ms, cp.t.witgen_ms, cp.t.prove_ms
        );
    }
    // The CONTROL: a UNION batch proof (same table, same size, NO wiring)
    // — separates the union-transport tax from the wiring tax. Same
    // blake3/blake3 config as the chain leaf. CAVEAT: the control's prove
    // TOTAL includes a ~570 ms prebuilt-witness copy (this probe hands the
    // union a materialized batch witness — the recorded prebuilt-driver
    // lesson); compare the PCS_TRACE ITEMIZED phases, not the total.
    let nu = n_blocks.trailing_zeros() as usize;
    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let registry = flock_prover::schedule::Registry::new(
        vec![TableType::from_block_r1cs(&blake_r1cs).with_io_schema(blake3::io_schema())],
        nu,
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    let pcs = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: pcs_batch(&union),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch(&union)),
        merkle_hash: HashKind::Blake3,
    };
    for r in 0..runs {
        let inputs: Vec<blake3::Compression> = (0..n_blocks)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, 0u64, 64u32, CHAIN_FLAGS)
            })
            .collect();
        let wit = blake3::generate_witness_batch_major(&inputs, nu);
        let t = std::time::Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (p, _, _) = prover::prove_fast_ligerito_union(
            &union,
            &pcs,
            vec![UnionSlotProverInput::new(wit, blake_lc)],
            &mut ch,
        );
        println!(
            "CONTROL {r} (union batch, no wiring): prove {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        std::hint::black_box(&p);
    }
}

/// **Task 7a: THE M32 HEADLINE.** The chain tower at the THROUGHPUT-OPTIMAL
/// leaf size: 4 chain segments of `CHAIN_BLOCKS` (default 2^18 = 262,144)
/// compressions each — fast profile, ~16.8 MB hashed per leaf — through
/// two first-level nodes and one internal node with the lane, timed per
/// phase. The statement: h_end == H^(4·2^18)(h_start) ≈ one million
/// sequential compressions, proven and folded to one recursable proof.
/// Warm-box numbers; the cold certification wants the reboot + probe
/// ritual first (the recorded discipline).
#[test]
#[ignore] // The headline measurement — run explicitly with --nocapture.
fn chain_tower_m32_headline() {
    let n_blocks: usize = std::env::var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let mut rng = Rng(0xC4A1_0008);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());

    // The sequential phase (the VDF delay): the chain values themselves.
    let t0 = std::time::Instant::now();
    let h_all = native_chain(&h0, 4 * n_blocks);
    let chain_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The leaves, timed individually (parallelizable in deployment).
    let mut _leaf_ms: Vec<f64> = Vec::new();
    let mut mk = |start: [u32; 16]| -> ChainProof {
        let t = std::time::Instant::now();
        let cp = build_chain_proof(start, n_blocks);
        _leaf_ms.push(t.elapsed().as_secs_f64() * 1e3);
        cp
    };
    let cp0 = mk(h0);
    let cp1 = mk(cp0.h_end);
    let cp2 = mk(cp1.h_end);
    let cp3 = mk(cp2.h_end);
    assert_eq!(cp3.h_end, h_all, "the four segments ARE the chain");

    let t_fl = std::time::Instant::now();
    let fl0 = build_fl_node(&cp0, &cp1);
    let fl1 = build_fl_node(&cp2, &cp3);
    let _fl_ms = t_fl.elapsed().as_secs_f64() * 1e3 / 2.0;

    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let t_in = std::time::Instant::now();
    let (node, acc, nt, app, lane_acc) =
        build_node_outer_app(&[&fl0.lo, &fl1.lo], Some(fl0.stmt_base), Some(lane));
    let _internal_ms = t_in.elapsed().as_secs_f64() * 1e3;
    let app = app.expect("app block");
    let lane_acc = lane_acc.expect("lane");

    // The root.
    let t_root = std::time::Instant::now();
    for j in 0..4 {
        assert_eq!(
            node.public[app + 4 + j],
            pack4(h_all[4 * j..4 * j + 4].try_into().unwrap()),
            "root statement: h_end == H^(4·{n_blocks})(h_start)"
        );
    }
    assert!(lane_acc.discharge(&chain_mats) && lane_acc.discharge_sigma(&cp0.inner.built.shape.circuit));
    let mut mats_ord = vec![
        (fl0.lo.b3_slot, (&fl0.lo.b3_r1cs.a_0, &fl0.lo.b3_r1cs.b_0)),
        (fl0.lo.swap_slot, (&fl0.lo.swap_r1cs.a_0, &fl0.lo.swap_r1cs.b_0)),
        (
            fl0.lo.spread_slot,
            (&fl0.lo.spread_r1cs.a_0, &fl0.lo.spread_r1cs.b_0),
        ),
    ];
    mats_ord.sort_by_key(|&(i, _)| i);
    let fl_mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
    let fl_el_mats: Vec<_> = fl0
        .lo
        .shape
        .registry
        .element_types()
        .iter()
        .map(|t| {
            let e = t.element_type().expect("element table");
            (e.a_0(), e.b_0())
        })
        .collect();
    assert!(
        acc.discharge(&fl_mats)
            && acc.discharge_element(&fl_el_mats)
            && acc.discharge_sigma(&fl0.lo.shape.circuit)
    );
    let root_ms = t_root.elapsed().as_secs_f64() * 1e3;

    let total_compr = 4 * n_blocks;
    // ---- SETUP vs ONLINE, per the contract on `Online` ----
    let leaves: Vec<Online> = [&cp0, &cp1, &cp2, &cp3].iter().map(|c| c.t).collect();
    let fl_t: Vec<Online> = [&fl0, &fl1].iter().map(|f| f.t).collect();
    let leaf_on = median_total(&leaves);
    let fl_on = median_total(&fl_t);
    let internal_on = nt.total();
    // A balanced tree over L leaves carries L/2 first-level nodes and
    // L/2 − 1 internal ones, so a leaf's amortised share tends to
    // leaf + FL/2 + internal/2; at four leaves the internal share is /4.
    let per_leaf_online = leaf_on + fl_on / 2.0 + internal_on / 4.0;
    println!(
        "\nCHAIN TOWER M32 HEADLINE (warm box — cold cert. owed post-reboot)\n  \
         {} compressions/leaf x 4 leaves = {} total ({:.1} MB hashed)\n  \
         sequential chain compute (the VDF delay, inherent): {:.0} ms\n  \
         ONLINE per proof (setup is per-SHAPE and excluded — see `Online`):",
        n_blocks,
        total_compr,
        (total_compr * 64) as f64 / 1e6,
        chain_ms,
    );
    report_stage("leaf", &leaves);
    report_stage("FL", &fl_t);
    report_stage("internal", std::slice::from_ref(&nt));
    println!(
        "    root (both lanes + statement): {:.1} ms\n  \
         PER-LEAF ONLINE (leaf + FL/2 + internal/4): {:.0} ms -> {:.0}k compressions/sec\n  \
         internal outer: nu {} | mu {} | proof {:.1} KiB\n",
        root_ms,
        per_leaf_online,
        n_blocks as f64 / per_leaf_online,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        bincode::serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **THE ONLINE BENCH: leaf, first-level node, internal node.** One number
/// per stage, measuring only what a prover pays PER STATEMENT — the walk,
/// the child tape sources, witness assembly, and the prove. Per-SHAPE setup
/// (circuit emit+finish, R1CS tables, PCS params, the fill plan, the tape
/// pins) is timed but reported apart and never folded into a per-proof
/// number: a shape is statement-independent, so a production prover builds
/// it once per level and reuses it for every segment.
///
/// Each stage is measured by re-running its builder `BENCH_RUNS` times over
/// FIXED inputs and taking per-phase MEDIANS — the first run of any stage
/// pays first-touch allocator costs that are warmup, not marginal cost.
/// The builders' pin/locate/replica scaffolding runs on every iteration and
/// costs wall time, but it is not inside any timer here.
///
/// Knobs: `BENCH_RUNS` (default 3), `CHAIN_BLOCKS` (default 256 — set
/// 262144 for the m32 production leaf), `TOWER_PROFILE=slim` for the
/// envelope. BOX DISCIPLINE: run the stability probe first and reboot if it
/// is far out of band — this box's benchmarks self-corrupt under sustained
/// load, and nothing here can tell you that happened.
#[test]
#[ignore] // Benchmark — run explicitly with --nocapture.
fn tower_online_bench() {
    let runs: usize = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let n_blocks: usize = std::env::var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut rng = Rng(0xC4A1_00BE);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());

    // MEASUREMENT HYGIENE: each stage runs with only ITS OWN inputs
    // resident. An m32 chain proof and an FL node are both large, and
    // holding the whole tower alive while timing one stage inflates it
    // through allocator and pool pressure — the leaf's spread read
    // 639-1183 ms when the bench built everything up front. A production
    // prover drops a child once it has been folded, so the stages are
    // ordered to do the same.

    // ---- LEAF: nothing else is alive yet ----
    let leaf: Vec<Online> = (0..runs)
        .map(|_| build_chain_proof(h0, n_blocks).t)
        .collect();

    // ---- FL: two chain children and nothing more ----
    let cp0 = build_chain_proof(h0, n_blocks);
    let cp1 = build_chain_proof(cp0.h_end, n_blocks);
    let fl: Vec<Online> = (0..runs).map(|_| build_fl_node(&cp0, &cp1).t).collect();

    // ---- INTERNAL: two FL children plus cp0 (the lane's chain materials).
    // The right pair is built in a scope so it is dropped before timing.
    let fl0 = build_fl_node(&cp0, &cp1);
    let fl1 = {
        let cp2 = build_chain_proof(cp1.h_end, n_blocks);
        let cp3 = build_chain_proof(cp2.h_end, n_blocks);
        build_fl_node(&cp2, &cp3)
    };
    drop(cp1);
    // The lane is what production carries: the children's chain
    // accumulators fold in a priors-only aggregate of their own.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let internal: Vec<Online> = (0..runs)
        .map(|_| {
            build_node_outer_app(
                &[&fl0.lo, &fl1.lo],
                Some(fl0.stmt_base),
                Some(ChainLane {
                    registry: chain_registry,
                    mats: &chain_mats,
                    circs: &chain_circs,
                    circuit: &cp0.inner.built.shape.circuit,
                    priors: &[&fl0.acc, &fl1.acc],
                    claims_base: fl0.fold_pub_base,
                }),
            )
            .2
        })
        .collect();

    let (leaf_on, fl_on, int_on) = (
        median_total(&leaf),
        median_total(&fl),
        median_total(&internal),
    );
    // A balanced tree over L leaves carries L/2 first-level nodes and
    // L/2 − 1 internal ones, so a leaf's amortised share tends to
    // leaf + FL/2 + internal/2 as the tree deepens.
    let per_leaf = leaf_on + fl_on / 2.0 + int_on / 2.0;
    println!(
        "\nONLINE BENCH — {n_blocks} compressions/leaf, {runs} runs/stage, profile {:?}\n  \
         per-proof ONLINE (setup is per-SHAPE, shown for reference only):",
        tower_profile(),
    );
    report_stage("leaf", &leaf);
    report_stage("FL", &fl);
    report_stage("internal", &internal);
    println!(
        "  AMORTISED per leaf (leaf + FL/2 + internal/2): {:.0} ms \
         -> {:.0}k compressions/sec\n  \
         the leaf's walk IS the chain compute — the application's own \
         sequential work, not proving\n",
        per_leaf,
        n_blocks as f64 / per_leaf,
    );
}

/// Isolate the RecordingChallenger's overhead on the tape construction's
/// exact workload: the deferred verify of an L1-node child, bare vs
/// recorded, `L2_RUNS` runs each (default 10). The domain must be the
/// node builder's own (the proof was proven under it).
#[test]
#[ignore] // Benchmark — run explicitly with `-- --ignored --nocapture`.
fn recording_overhead_probe() {
    use flock_core::transcript_record::RecordingChallenger;
    let runs: usize = std::env::var("L2_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let l0 = build_leaf_outer_seeded(0x4D50_9B00);
    let l1 = build_leaf_outer_seeded(0x4D50_9B01);
    let (n0, _acc, _) = build_node_outer(&l0, &l1);

    let union_i = outer_union(&n0.shape.registry, n0.shape.counts.clone());
    let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (n0.b3_slot, n0.b3_r1cs.csc_lincheck_circuit()),
        (n0.swap_slot, n0.swap_r1cs.csc_lincheck_circuit()),
        (n0.spread_slot, n0.spread_r1cs.csc_lincheck_circuit()),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
        lcs_ord.into_iter().map(|(_, cc)| cc).collect();

    let mut bare_ms: Vec<f64> = Vec::new();
    let mut rec_ms: Vec<f64> = Vec::new();
    for _ in 0..runs {
        let t = std::time::Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verifier::verify_ligerito_union_circuit_deferred(
            &union_i, &n0.shape.circuit, &n0.public, &lcs, &n0.commitment, &n0.proof,
            &n0.pcs, &mut ch,
        )
        .expect("bare deferred verify accepts");
        bare_ms.push(t.elapsed().as_secs_f64() * 1e3);

        let t = std::time::Instant::now();
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
        verifier::verify_ligerito_union_circuit_deferred(
            &union_i, &n0.shape.circuit, &n0.public, &lcs, &n0.commitment, &n0.proof,
            &n0.pcs, &mut rec,
        )
        .expect("recorded deferred verify accepts");
        rec_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    bare_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rec_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "deferred verify of an L1 node over {runs} runs: bare median {:.2} ms \
         [{:.2}-{:.2}] | recorded median {:.2} ms [{:.2}-{:.2}] | wrapper {:.2} ms",
        bare_ms[runs / 2],
        bare_ms[0],
        bare_ms[runs - 1],
        rec_ms[runs / 2],
        rec_ms[0],
        rec_ms[runs - 1],
        rec_ms[runs / 2] - bare_ms[runs / 2],
    );
}

/// L2-ONLY BENCHMARK. Build the tower's scaffolding once — four leaves,
/// two level-1 nodes, none of it timed or reported — then rebuild + prove
/// the LEVEL-2 node `L2_RUNS` times (default 5). One machine-readable
/// `RUN l2_prove <ms>` line per run (stability_probe's convention) plus a
/// median summary; the node's own breakdown lines above each RUN carry
/// tapes/trace/witness splits. Profile via `TOWER_PROFILE` as usual:
///
///   L2_RUNS=10 TOWER_PROFILE=slim cargo test --release -p flock-prover \
///     --test circuit_merkle l2_node_bench -- --ignored --nocapture
#[test]
#[ignore] // Benchmark — run explicitly with `-- --ignored --nocapture`.
fn l2_node_bench() {
    let runs: usize = std::env::var("L2_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let l0 = build_leaf_outer_seeded(0x4D50_9B00);
    let l1 = build_leaf_outer_seeded(0x4D50_9B01);
    let l2 = build_leaf_outer_seeded(0x4D50_9B02);
    let l3 = build_leaf_outer_seeded(0x4D50_9B03);
    let (n0, _acc0, _) = build_node_outer(&l0, &l1);
    let (n1, _acc1, _) = build_node_outer(&l2, &l3);

    let runs_t: Vec<Online> = (0..runs)
        .map(|_| {
            let (_n2, _acc2, t) = build_node_outer(&n0, &n1);
            t
        })
        .collect();
    println!("\nL2 NODE — {runs} runs, per-proof ONLINE (setup is per-SHAPE):");
    report_stage("l2 node", &runs_t);
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
    let (n0, acc0, _) = build_node_outer(&l0, &l1);
    let (n1, acc1, _) = build_node_outer(&l2, &l3);
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
    let (n2, acc2, _) = build_node_outer(&n0, &n1);

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
    // The envelope's nu* driver: per-level max type count (nu must cover
    // the largest slot at EVERY level, so the tower's nu* = the max here).
    for (name, n) in [("level-1", &n0), ("level-2", &n2)] {
        let (t_max, c_max) = n
            .shape
            .counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(t, c)| (t, *c))
            .unwrap();
        println!(
            "  {name} nu {} | max count {} at t{} ({}/{} io {}) | counts {:?}",
            n.shape.registry.nu(),
            c_max,
            t_max,
            n.shape.registry.types()[t_max].useful_bits,
            if n.shape.registry.types()[t_max].is_element() { "el" } else { "bool" },
            n.shape.registry.types()[t_max].io_schema.len(),
            n.shape.counts,
        );
    }
}

/// RECONNAISSANCE for the envelope's registry convergence (wall 2): the
/// leaf outer's registry vs the node's, type by type — the exact diff the
/// shared envelope registry must union. Informational; run with
/// `-- --ignored --nocapture`.
#[test]
#[ignore]
fn envelope_registry_diff() {
    let lo = build_leaf_outer_seeded(0x4D50_9B00);
    let l1 = build_leaf_outer_seeded(0x4D50_9B01);
    let (n0, _acc, _) = build_node_outer(&lo, &l1);
    for (name, shape) in [("LEAF OUTER", &lo.shape), ("NODE", &n0.shape)] {
        println!(
            "\n{name}: nu {} | m_total {} | dense_m {} | mu {} | {} types",
            shape.registry.nu(),
            shape.registry.m_total(),
            outer_union(&shape.registry, shape.counts.clone()).dense_m(),
            shape.circuit.cells().mu(),
            shape.registry.types().len(),
        );
        for (t, ty) in shape.registry.types().iter().enumerate() {
            println!(
                "  t{t:2} | {} | k_log {:2} | useful_bits {:6} | io {:3} | count {:6}",
                if ty.is_element() { "el  " } else { "bool" },
                ty.k_log,
                ty.useful_bits,
                ty.io_schema.len(),
                shape.counts[t],
            );
        }
    }

    // ---- WALL 2, the pin: under the envelope the two registries are ONE ----
    if envelope_shape().is_some() {
        assert_eq!(
            lo.shape.registry.digest(),
            n0.shape.registry.digest(),
            "WALL 2: the leaf-outer and node registries digest-equal under the envelope"
        );
        println!("\nWALL 2 PIN: leaf and node registry digests are EQUAL");

        // ---- the payoff: a LEAF-level accumulator is a valid PRIOR of a
        // NODE-level fold. check_priors demands exactly the registry digest
        // and both groups' widths, so wall 2 is what lets an accumulator
        // cross levels. Sigma stays per-level here: wall 3 keys sigma by
        // CIRCUIT digest and the leaf/node circuits still differ — the
        // leaf-level fold carries the matrix + element groups only (the
        // sigma-free shape the aggregate supports), and the node fold adds
        // its own sigma fresh. The count-0 element types ride BOTH folds:
        // the leaf's assertions cover its two count-0 deep resids, the
        // node's cover its count-0 skips — this is the zero-count pin in
        // the fold/assertion machinery.
        use flock_core::aggregate;
        const PRIOR_DOMAIN: &[u8] = b"flock-envelope-prior-probe-v0";
        let registry = &lo.shape.registry;
        let (rt0, rt1) = (RealTape::new(&lo, DOMAIN), RealTape::new(&l1, DOMAIN));
        let u0 = outer_union(registry, lo.shape.counts.clone());
        let u1 = outer_union(&l1.shape.registry, l1.shape.counts.clone());
        let mut mats_ord = vec![
            (lo.b3_slot, (&lo.b3_r1cs.a_0, &lo.b3_r1cs.b_0)),
            (lo.swap_slot, (&lo.swap_r1cs.a_0, &lo.swap_r1cs.b_0)),
            (lo.spread_slot, (&lo.spread_r1cs.a_0, &lo.spread_r1cs.b_0)),
        ];
        mats_ord.sort_by_key(|&(i, _)| i);
        let mats: Vec<_> = mats_ord.iter().map(|&(_, m)| m).collect();
        let mut lcs_ord: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
            (lo.b3_slot, lo.b3_r1cs.csc_lincheck_circuit()),
            (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
            (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
            lcs_ord.iter().map(|&(_, c)| c).collect();
        let el_types: Vec<_> = registry
            .element_types()
            .iter()
            .map(|s| s.element_type().expect("an element slot's table"))
            .collect();
        let el_mats: Vec<_> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();

        // The leaf-level fold: both leaves' assertions, no sigma, no priors.
        let bool_asserts = [rt0.mat_assert.clone(), rt1.mat_assert.clone()];
        let el_asserts = [(&u0, rt0.el_assert.clone()), (&u1, rt1.el_assert.clone())];
        let mut chp = FsChallenger::with_chained_blake3(PRIOR_DOMAIN);
        let (agg_l, acc_leaf) = aggregate::prove_aggregate_classes(
            registry,
            &mats,
            &lcs,
            &bool_asserts,
            &el_mats,
            &el_asserts,
            None,
            &[],
            &mut chp,
        )
        .expect("the leaf-level fold proves");
        let mut chv = FsChallenger::with_chained_blake3(PRIOR_DOMAIN);
        let acc_leaf_v = aggregate::verify_aggregate_classes(
            registry,
            &bool_asserts,
            &el_asserts,
            None,
            &[],
            &agg_l,
            &mut chv,
        )
        .expect("the leaf-level fold verifies");
        assert_eq!(acc_leaf, acc_leaf_v, "leaf fold accumulators agree");

        // The node-level fold: n0's own assertions + the leaf accumulator
        // as a PRIOR — accepted precisely because the registries digest-
        // equal, then folded and discharged against ONE set of matrices.
        let rtn = RealTape::new(&n0, DOMAIN);
        let un = outer_union(&n0.shape.registry, n0.shape.counts.clone());
        let n_bool = [rtn.mat_assert.clone()];
        let n_el = [(&un, rtn.el_assert.clone())];
        let n_sigmas = [rtn.sigma_native.clone()];
        let mut chp2 = FsChallenger::with_chained_blake3(PRIOR_DOMAIN);
        let (agg_n, acc_node) = aggregate::prove_aggregate_classes(
            registry,
            &mats,
            &lcs,
            &n_bool,
            &el_mats,
            &n_el,
            Some((&n0.shape.circuit, &n_sigmas)),
            &[&acc_leaf],
            &mut chp2,
        )
        .expect("a leaf accumulator is a valid node-fold prior (wall 2)");
        let mut chv2 = FsChallenger::with_chained_blake3(PRIOR_DOMAIN);
        let acc_node_v = aggregate::verify_aggregate_classes(
            registry,
            &n_bool,
            &n_el,
            Some((&n0.shape.circuit, &n_sigmas)),
            &[&acc_leaf],
            &agg_n,
            &mut chv2,
        )
        .expect("the cross-level fold verifies");
        assert_eq!(acc_node, acc_node_v, "cross-level accumulators agree");
        assert!(acc_node.discharge(&mats), "cross-level boolean discharge");
        assert!(
            acc_node.discharge_element(&el_mats),
            "cross-level element discharge"
        );
        assert!(
            acc_node.discharge_sigma(&n0.shape.circuit),
            "the node's own sigma discharges"
        );
        // Negative control: a tampered prior digest is rejected by
        // check_priors before any fold work runs.
        let mut bad = acc_leaf.clone();
        bad.registry_digest[0] ^= 1;
        let mut chb = FsChallenger::with_chained_blake3(PRIOR_DOMAIN);
        assert!(
            aggregate::prove_aggregate_classes(
                registry,
                &mats,
                &lcs,
                &n_bool,
                &el_mats,
                &n_el,
                Some((&n0.shape.circuit, &n_sigmas)),
                &[&bad],
                &mut chb,
            )
            .is_err(),
            "a mismatched prior registry digest is rejected"
        );
        println!(
            "WALL 2 PRIOR: a leaf accumulator crossed into a node fold, \
             folded with fresh node claims and discharged; tampered digest rejected"
        );
    }
}
