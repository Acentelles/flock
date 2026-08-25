//! **The recursion tower**: chain leaves folded 2->1 up to a converging
//! spine — the production chain100/chain128 pipeline.
//!
//! [`TowerConfig`] names the two production configurations: the LEAF (the
//! application's BLAKE3 hash-chain segment) proves under the rate-1/2 Fast
//! twin of the tower's security level, and every OUTER (FL / internal /
//! spine) proves under the Slim twin at the m* = 29 / nu* = 14 envelope,
//! always on. The pipeline is three builders:
//!
//! - [`build_chain_proof`] — the LEAF: a Ligerito proof of one chain
//!   segment (the workload).
//! - [`build_fl_node_k`] — the FIRST-LEVEL node: adjacent chain leaves
//!   verified in-circuit (tape replay), their claims folded and proven as
//!   one envelope outer.
//! - [`build_node_outer_app`] — INTERNAL and SPINE nodes: 2->1 recursion
//!   over envelope outers, with the chain-lane accumulator riding along
//!   and the spine inheriting its base's accumulator toward the converged
//!   fixed point (`chain_spine_converges` gates the ONE-digest property).
//!
//! Bench knobs (`CHAIN_BLOCKS`, `BENCH_RUNS`, `TOWER_STEADY`, and the
//! test-only `TOWER_CONFIG=chain100`) live in the `#[test]` harness; the
//! production geometry is typed, never env-var-driven.

use crate::challenger::FsChallenger;
use crate::prover::{self, UnionSlotProverInput};
use crate::r1cs_hashes::blake3;
use crate::r1cs_hashes::merkle_r1cs::SLOT_WORDS;
#[cfg(test)]
use crate::r1cs_hashes::merkle_r1cs::{ChunkPathInput, MerkleTreeLayout, blake3_spec};
use crate::schedule::TableType;
use crate::union::UnionInstance;
#[cfg(test)]
use flock_core::circuit::builder::CircuitBuilder;
use flock_core::circuit::builder::{GateType, ShapeBuilder, SlotWitness, Wire};
use flock_core::field::{F128, F256};
use flock_core::matrix_fold::{MatrixClaim, Weight};
use flock_core::merkle::{self as core_merkle, HashKind};
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::verifier;

/// The L0 interleave for a content-sized commit: the embedded config's
/// own `initial_k` (6 everywhere except m29 Fast/Slim = 5 — the
/// recursion-node row-width choice). `prover_config_for` rejects a
/// mismatched batch, so every params site whose `m` is content-derived
/// must go through this.
fn pcs_batch_for(union: &UnionInstance, profile: LigeritoProfile) -> usize {
    flock_core::pcs::ligerito::embedded_initial_k_or_default(union.dense_m(), profile)
}

/// The two production recursion towers. The LEAF (the application's chain
/// segment — the workload inner proof) proves under the rate-1/2 Fast twin
/// of the tower's security level; the OUTERS (FL / internal / spine) prove
/// under the Slim twin at the m* = 29 / nu* = 14 envelope, always ON.
///
/// Fast leaf + Slim outers is deliberate. The leaf keeps the SAME tape
/// structure as Fast (the FL/node tape walkers are level-blind), while its
/// query count follows the tower's security level: a 100-bit recursion
/// carries a 100-bit leaf (Fast100, 448q — a 128-bit leaf under a 100-bit
/// recursion balloons the FL's replayed transcript past the arity-2
/// envelope), and the 128-bit aggressive recursion carries the aggressive
/// Fast128 leaf (rate-1/2 on the rate+2 ladder; m32: Σq 675 → 527).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TowerConfig {
    /// The 100-bit tower: Fast100 leaf, Slim100 outers.
    Chain100,
    /// The 128-bit tower on the aggressive rate ladder: Fast128 leaf,
    /// Slim128 outers.
    Chain128,
}

impl TowerConfig {
    /// The chain leaf's WORKLOAD inner-proof profile (rate 1/2).
    pub fn leaf_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain100 => LigeritoProfile::Fast100,
            TowerConfig::Chain128 => LigeritoProfile::Fast128,
        }
    }
    /// The recursion-path OUTER profile (rate 1/4, envelope-ON).
    pub fn outer_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain100 => LigeritoProfile::Slim100,
            TowerConfig::Chain128 => LigeritoProfile::Slim128,
        }
    }
}

/// Bench/test knob: which production tower the ignored tower tests and
/// benches exercise. `TOWER_CONFIG=chain100` selects [`TowerConfig::Chain100`];
/// the default is the 128-bit production tower.
#[cfg(test)]
fn test_config() -> TowerConfig {
    match std::env::var("TOWER_CONFIG").as_deref() {
        Ok("chain100") => TowerConfig::Chain100,
        _ => TowerConfig::Chain128,
    }
}

fn tower_fold_grinding(cfg: TowerConfig) -> flock_core::matrix_fold::FoldGrinding {
    let profile = cfg.outer_profile();
    PcsParams {
        m: 22,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: 5,
        profile,
        num_lanes: None,
        merkle_hash: HashKind::Blake3,
    }
    .matrix_fold_grinding()
}

/// The ENVELOPE dense floor `m*` (wall 2): every recursion-path OUTER —
/// leaf and node alike — commits at this size, so a node's children look
/// ONE shape regardless of level (an L1 node's leaf children carry the
/// same query geometry as an L2 node's node children).
///
/// Ron's call 2026-08-06: m* = 29 (the fixed point closes with ~2x slack;
/// every Slim level commits m29). Re-targeting the tight m* = 28 (needs
/// the mac shave −8k words + publics arithmetization −40k+ at the fixed
/// point) is a deliberate future re-pin; `envelope_content_probe` is the
/// instrument that sizes it.
const ENVELOPE_FLOOR_M: usize = 29;

/// A recursion-path OUTER's union instance, with the envelope floor
/// applied. Every instance over a leaf/node OUTER shape must come from
/// here — prover, verifier and tape recorder alike: the floor is
/// STATEMENT data, like the counts.
fn outer_union<'r>(
    registry: &'r crate::schedule::Registry,
    counts: Vec<usize>,
) -> UnionInstance<'r> {
    let mut u = UnionInstance::new(registry, counts);
    u.set_dense_floor(ENVELOPE_FLOOR_M);
    u
}

/// Wall 2's registry-geometry constants at the settled envelope (slim,
/// m* = 29): the UNION of the leaf-outer's and the node's type sets, at the
/// envelope maxima. Measured at the m29 fixed point (envelope_registry_diff
/// + the tower census, 2026-08-06):
///
/// - `spread_w` 20 covers the m32 FAST chain leaf's L0 depth; the m29 Slim
///   outer ladder needs 19 and leaves the high output unread.
/// - Extension-field residual work uses three reusable gates, independent
///   of prefix length (the base-field per-prefix `ResidualGate` family died
///   in the stage-3 registry diet).
/// - `nu` 14: each of the two independent child verifiers has its own
///   identical BLAKE slot, and the consolidated extension-field residual
///   gates keep every physical slot below 2^14 rows.
/// - 29 table types occupy 511 gate slots; with one public slot, the cell
///   address needs 9 bits and `mu = nu + 9 = 23` tower-wide.
///
/// A ladder that drifts off these constants surfaces as a NEW slot at
/// emission time and hence a registry-digest mismatch — the failure is
/// loud, never silent.
struct EnvShape {
    nu: usize,
    spread_w: usize,
    pf_w: usize,
    /// Historical counts* oracle values. Shipped envelope proofs use
    /// unconditional free counts, so these values no longer pad rows or
    /// determine the circuit digest; `counts_el` remains the canonical
    /// element-slot key list and retains the old cap census for
    /// comparison.
    counts_el: [(usize, usize); 15],
    /// publics* — the ONE public-segment length every envelope outer pads
    /// to (published zeros appended after all real publics). The child's
    /// publics count is what a PARENT's walk consumes — H(publics) chain
    /// rows and the recombination's 8-lane folds both scale with it — so
    /// one count is what makes the L1 walk (leaf children) and the L2
    /// walk (node children) row-identical. The last [`ENV_APP_WORDS`] of
    /// them are the APPLICATION BLOCK (see [`env_app_base`]).
    publics: usize,
    /// lanes* — the pinned committed lane count (see [`outer_lanes`]): the
    /// one aggregate of a child's layout that stays circuit structure
    /// under FREE COUNTS.
    lanes: usize,
}

/// The APPLICATION STATEMENT's width in the envelope's public segment: the
/// hash-chain PoC's span `(h_start, h_end)`, eight 128-bit words.
const ENV_APP_WORDS: usize = 8;

/// Steady-repetition override: how many EXTRA times a builder re-runs its
/// ONLINE phases (tapes + walk + witgen + prove + verify) over the
/// once-built shape, collecting one [`Online`] record per iteration. The
/// bench sets this per stage so a 5-run median costs ONE ~3-5 s setup
/// instead of five — the per-shape setup was ~96% of the bench's wall
/// clock. `usize::MAX` = unset (the `TOWER_STEADY` env knob applies).
static STEADY_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

fn steady_reps() -> usize {
    let ov = STEADY_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov != usize::MAX {
        return ov;
    }
    std::env::var("TOWER_STEADY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

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
// F256 verification adds extension-field table claims to the shared
// registry. A live node uses 1,028 words; keep a fixed-shape margin so the
// same block carries the accumulator at every recursive level.
const ENV_ACC_MAIN_WORDS: usize = 1152;

/// THE PASSENGER (wall 3): one sigma-shaped and one jagged-shaped entry,
/// same layout as the ACC_MAIN keyed slots. A spine node's node-slot
/// inherits an entry keyed by its child's OWN child — which matches at
/// every steady level but once, at the first steady node over a base
/// node. That single ORPHAN cannot fold (its key names a circuit no slot
/// of this node names), so it rides here, re-published child to parent by
/// a gated copy, until the root discharges it against the base circuit's
/// own tables. Zeros when empty, which is every node but that one and its
/// ancestors.
const ENV_PASS_WORDS: usize = 96;

/// FREE COUNTS ARE UNCONDITIONAL (the count win shipped, 2026-08-09): under
/// the envelope, children declare their own per-type row counts — the
/// heights reach a parent only as folded claims on the jagged layout,
/// discharged at the root — and only the LANE COUNT stays pinned
/// (`EnvShape::lanes`). The former count-padding switches are retired;
/// `counts_el` remains only as the key list + historical census.
fn outer_lanes(union: &UnionInstance, log_batch_size: usize) -> Option<usize> {
    let content = union.commit_lanes(log_batch_size);
    let env = envelope_shape();
    let c = content.unwrap_or(1usize << log_batch_size);
    assert!(
        c <= env.lanes,
        "content lanes {c} exceed the lane pin {}",
        env.lanes
    );
    Some(env.lanes)
}

fn env_app_base(env: &EnvShape) -> usize {
    env.publics - ENV_APP_WORDS
}

fn env_pass_base(env: &EnvShape) -> usize {
    env_app_base(env) - ENV_PASS_WORDS
}

fn env_acc_main_base(env: &EnvShape) -> usize {
    env_pass_base(env) - ENV_ACC_MAIN_WORDS
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
    /// The PASSENGER: entries this node could not fold and did not drop.
    pass: &'w [Wire],
    /// The application statement.
    app: &'w [Wire],
}

/// The fixed envelope shape — always on, no override: the registry
/// convergence below is pinned to m* = 29's measured geometry.
fn envelope_shape() -> EnvShape {
    EnvShape {
        // The two-child envelope fits at 14 after consolidating the F256
        // residual tables and assigning each independent child verifier to
        // its own identical BLAKE slot. Every physical slot remains below
        // 2^14 rows, while 512 cell slots give mu = 14 + 9 = 23.
        nu: 14,
        // 20 = the m32 FAST chain leaf's L0 depth (log_msg_cols 19 +
        // log_inv_rate 1), which the B-fast PoC's first-level node walks;
        // the m29 slim outer ladder needs only 19 and leaves the top
        // output unread.
        spread_w: 20,
        // Six variants: pl = Σ_{levels above} fold count, so the deepest is
        // the m32 FAST chain ladder's level-0 (six levels, 5×3 folds above
        // it) — the m29 slim outer ladder's five stop at 12 and ride the
        // rest at count 0.
        pf_w: 8,
        // Iterated at the padded envelope 2026-08-06 (probe + tower
        // census, elementwise max of leaf/node usage). Only b3, le8, pf8
        // and mac are content-geometry-sensitive; everything else hits its
        // cap exactly (registry-shaped).
        // The 4th entry is the fused PoW-mask slot (one row per grinding
        // site). It is a historical oracle cap only: free counts are
        // unconditional, and the strict-Slim m29 spine exercises the live
        // count and fixed envelope layout end to end.
        // BLAKE is the only boolean family whose live count exceeds 2^14.
        // The two independent child regions use identical slots while the
        // shipped free-count path records each actual prefix.
        counts_el: [
            (600, 49000), // mac — the nu* driver; watch the 2^15 ceiling
            (602, 8000),  // fold/recombination MACs, split from verifier arithmetic
            (500, 1000),  // zcr
            // mrs — 1000, was 900: wall 3's steady spine node runs the
            // extra keyed slot's rounds (measured 949 live).
            (400, 1000),
            (0, 9000),    // spine
            (700, 9000),  // extension-field Ligerito spine
            (701, 15000), // extension-field multiply-accumulate
            (601, 300),   // assist
            (8, 4200),    // leaf-eval 8-lane
            (808, 4200),  // extension-field leaf evaluation
            (318, 15000), // prefix w 8
            // The extension residual relation is decomposed into three
            // shared tables: one normalized-weight row and one accumulator
            // row per query, plus one three-factor prefix row per later
            // Ligerito level. These caps are the sums of the former six
            // per-prefix variants at the envelope maxima above.
            (880, 4150),   // normalized W_0..W_18 chain
            (881, 12690),  // three-factor extension prefix
            (882, 4150),   // eight-way residual accumulation
            (1008, 15000), // extension prefix w 8
        ],
        // Preserve the existing public body while enlarging ACC_MAIN.
        publics: 5684,
        // The committed lane count — the ONE piece of a child's layout that
        // stays circuit structure (the parent hashes `num_lanes`-word
        // leaves), so it is pinned while everything count-shaped rides the
        // jagged claims. 24 covers every envelope member's content-derived
        // count at min-one-row. F256 raises the largest live content to 25
        // lanes; 31 stays below `2^initial_k = 32`, so children remain
        // lane-major.
        lanes: 31,
    }
}

/// Find-or-create a slot under this file's keyed-cache scheme
/// (0 spine / 8 leaf-eval / 400 mrs / 500 zcr / 600 mac / 601 assist /
/// 602 fold-mac / 700 spine256 / 701 mac256 / 808 leaf-eval256 /
/// 880 resid-weights / 881 resid-prefix3 / 882 resid-acc /
/// 310+w prefix / 1000+w prefix256). Every element-slot declaration on the
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

/// Declare the envelope's 29 table types in one canonical order.
/// `Registry::new` sorts class-major then k_log-descending with a
/// STABLE sort, so the declaration order here fixes every same-k_log
/// tie-break — the leaf-outer and node registries become the same sorted
/// type list, which together with nu* is registry-digest equality. Returns
/// the six boolean slots; every element type pre-seeds `cache` under the keyed
/// scheme so both builders' demand sites hit the cache instead of
/// declaring. The order is the node's historical one.
fn declare_envelope_slots(
    sb: &mut ShapeBuilder,
    nu: usize,
    cache: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    env: &EnvShape,
) -> CollapsedSlots {
    debug_assert_eq!(nu, env.nu, "the envelope declares at nu*");
    let q = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        b3_alt: Some(sb.slot(Blake3Gate { nu })),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(env.spread_w),
            nu,
        }),
        pow: sb.slot(PowMaskGate { nu }),
        family: Some(sb.slot(FamilyTransposeTileGate { nu })),
    };
    slot_cached(sb, cache, 600, MacGate::new);
    slot_cached(sb, cache, 602, MacGate::new);
    slot_cached(sb, cache, 500, ZcRoundGate::new);
    slot_cached(sb, cache, 400, MergedRoundGate::new);
    slot_cached(sb, cache, 0, SpineGate::new);
    slot_cached(sb, cache, 700, SpineGate256::new);
    slot_cached(sb, cache, 701, MacGate256::new);
    slot_cached(sb, cache, 601, AssistLayerGate::new);
    slot_cached(sb, cache, 8, || LeafEvalGate::new(8));
    slot_cached(sb, cache, 808, || LeafEvalGate256::new(8));
    slot_cached(sb, cache, 880, ResidualWeightsGate256::new);
    slot_cached(sb, cache, 881, ResidualPrefix3Gate256::new);
    slot_cached(sb, cache, 882, ResidualAccGate256::new);
    slot_cached(sb, cache, 310 + env.pf_w, || PrefixGate::new(env.pf_w));
    slot_cached(sb, cache, 1000 + env.pf_w, || PrefixGate256::new(env.pf_w));
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
    consts: &mut Vec<(F128, Wire)>,
    tail: &EnvTail,
) {
    // FREE COUNTS ARE THE DEFAULT (the count win): the ROW padding is
    // skipped — children declare their own counts, min-one-row keeps every
    // type live, and the heights reach a parent only as jagged claims.
    // The tail blocks and the public segment still pad, so the layout a
    // parent reads is unchanged. The historical caps remain in `counts_el`
    // as the slot-declaration key list, but row-count padding is retired.
    let mut report: Vec<String> = Vec::new();
    let mut over: Vec<String> = Vec::new();
    let mut pad = |sb: &mut ShapeBuilder,
                   hints: &mut Vec<[u32; SLOT_WORDS]>,
                   over: &mut Vec<String>,
                   name: &str,
                   s: flock_core::circuit::builder::SlotId,
                   target: usize,
                   hinted: bool,
                   fixed_inputs: Option<&[Wire]>| {
        let live = sb.rows_in_slot(s);
        report.push(format!("{name} {live}/{target}"));
        if live > target {
            over.push(format!("{name} {live} > {target}"));
            return;
        }
        let ins = fixed_inputs
            .map(<[Wire]>::to_vec)
            .unwrap_or_else(|| vec![zw; sb.slot_inputs(s)]);
        assert_eq!(
            ins.len(),
            sb.slot_inputs(s),
            "padding input arity for {name}"
        );
        for _ in live..target {
            if hinted {
                hints.push([0u32; SLOT_WORDS]);
                sb.gate_hinted(s, &ins);
            } else {
                sb.gate(s, &ins);
            }
        }
    };
    // In free-count mode a live slot's target IS its live count, while an
    // empty declared slot pads only to ONE ROW — never to the old cap.
    // That is the whole pin the run structure needs: `assist_boundaries`
    // merges columns only when they are EMPTY, so every non-empty column is
    // a singleton run and the run count is registry-derived EXCEPT through
    // the predicate `n_t > 0`. Keep every type non-empty and the counts
    // become pure values.
    let floor1 = |sb: &ShapeBuilder, s| sb.rows_in_slot(s).max(1);
    let t_b3 = floor1(sb, q.b3);
    let b3_alt = q.b3_alt.expect("the envelope declares two BLAKE slots");
    let t_b3_alt = floor1(sb, b3_alt);
    let t_swap = floor1(sb, q.swap);
    let t_spread = floor1(sb, q.spread);
    let t_pow = floor1(sb, q.pow);
    pad(sb, hints, &mut over, "b3", q.b3, t_b3, false, None);
    pad(sb, hints, &mut over, "b3b", b3_alt, t_b3_alt, false, None);
    pad(sb, hints, &mut over, "swap", q.swap, t_swap, true, None);
    pad(
        sb, hints, &mut over, "spread", q.spread, t_spread, false, None,
    );
    let pow_check = cw(sb, vals, consts, F128::new(0, 1u64 << 63));
    let pow_inputs = [zw, zw, zw, pow_check];
    pad(
        sb,
        hints,
        &mut over,
        "pow",
        q.pow,
        t_pow,
        false,
        Some(&pow_inputs),
    );
    let family = q.family.expect("the envelope declares family H");
    let t_family = floor1(sb, family);
    pad(
        sb, hints, &mut over, "family", family, t_family, false, None,
    );
    for &(key, count) in &env.counts_el {
        let &(_, s) = cache
            .iter()
            .find(|&&(k, _)| k == key)
            .unwrap_or_else(|| panic!("envelope slot key {key} missing from the cache"));
        let _ = count;
        let target = floor1(sb, s);
        pad(
            sb,
            hints,
            &mut over,
            &format!("el{key}"),
            s,
            target,
            false,
            None,
        );
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
    let body =
        env.publics - ENV_ACC_CHAIN_WORDS - ENV_ACC_MAIN_WORDS - ENV_PASS_WORDS - ENV_APP_WORDS;
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
            ("pass", tail.pass, ENV_PASS_WORDS),
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
    // The live/target census: target is the live count (or one
    // schema-preserving dummy row).
    println!("  [envelope rows live/target] {}", report.join(" | "));
    // Overshoot is a real failure: the public segment or a tail block
    // outgrew the envelope's fixed layout.
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
// Populated by the pub builders; READ only by the in-file `#[test]` benches
// (`tower_online_bench` and friends), so the lib unit sees the fields unread.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Default)]
struct Online {
    setup_ms: f64,
    walk_ms: f64,
    tapes_ms: f64,
    witgen_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    /// ONE timer around the whole online span (walk through prove), not the
    /// phase sum: everything between the phases — the union/PCS param
    /// construction, buffer drops, allocator work — lands here and nowhere
    /// else. `0.0` where a stage has not been wired for it.
    wall_ms: f64,
}

#[cfg(test)]
impl Online {
    /// The per-proof online total. The MEASURED wall where a stage supplies
    /// it, the phase sum otherwise — a sum can only be a lower bound.
    fn total(&self) -> f64 {
        if self.wall_ms > 0.0 {
            self.wall_ms
        } else {
            self.walk_ms + self.tapes_ms + self.witgen_ms + self.prove_ms
        }
    }

    /// What the phases add up to — printed beside the wall so the gap between
    /// them is visible rather than assumed away.
    fn summed(&self) -> f64 {
        self.walk_ms + self.tapes_ms + self.witgen_ms + self.prove_ms
    }
}

#[cfg(test)]
fn median_of(runs: &[Online], f: impl Fn(&Online) -> f64) -> f64 {
    let mut v: Vec<f64> = runs.iter().map(&f).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(test)]
fn median_total(runs: &[Online]) -> f64 {
    median_of(runs, |o| o.total())
}

/// One stage's ONLINE line: per-phase medians, the total's median and
/// range, then the per-SHAPE setup for reference. Medians, not means —
/// the first run of any stage pays first-touch allocator costs that are
/// warmup, not marginal cost (the recorded L2 lesson).
/// **WHERE A PROOF'S BYTES ARE.** Serialized size per component, so that any
/// shrinking effort is steered by the census rather than by intuition about
/// which piece looks big. Sizes are `bincode` lengths of the sub-structures;
/// they sum to slightly less than the whole (the outer struct's own tags).
///
/// The interesting ratio is proof bytes vs what they cost a PARENT: the
/// parent replays the child's transcript through its b3 slot at one
/// compression per 64 bytes, so at the measured ~6.1 µs per b3 row a KiB of
/// child proof is ~0.1 ms of parent per child.
#[cfg(test)]
fn proof_census(label: &str, p: &flock_core::proof::R1csProofCircuitMerged, pcs: &PcsParams) {
    // Per-level stratified schedules, and the siblings a path emits ABOVE
    // the cap layer. The cap is the whole layer at the DEEPEST summand's
    // depth c1, so every query can be checked with d - c1 siblings — since
    // truncation, that is all the prover emits. MEASURED from the proof
    // (not recomputed from the schedule), so `redundant` certifies the
    // truncation stays landed: anything past q·(d − c1) is waste.
    if let Ok(cfg) = pcs.ligerito_prover_config() {
        let lig = &p.pcs_open.inner.ligerito;
        let r = lig.recursive_caps.len();
        assert_eq!(cfg.stratified.len(), r + 1, "one schedule per open level");
        let level_paths = |lvl: usize| -> usize {
            if lvl == 0 {
                lig.initial_proof.merkle_proof.len()
            } else if lvl < r {
                lig.recursive_proofs[lvl - 1].merkle_proof.len()
            } else {
                lig.final_proof.merkle_proof.len()
            }
        };
        let (mut waste, mut emitted) = (0usize, 0usize);
        let mut per_level: Vec<String> = Vec::new();
        for (lvl, sch) in cfg.stratified.iter().enumerate() {
            let c1 = sch.cap_depth();
            let e = level_paths(lvl);
            let w = e - sch.queries() * (sch.log_block_len - c1);
            per_level.push(format!(
                "L{lvl}: q={} depths={:?} cap={c1} sibs={e} redundant={w}",
                sch.queries(),
                sch.summand_depths,
            ));
            waste += w;
            emitted += e;
        }
        println!(
            "\n  STRATIFIED PATHS — {label}\n    {}\n    \
             emitted {emitted} siblings, {waste} redundant above the cap \
             ({:.1} KiB of {:.1} KiB, {:.0}%)",
            per_level.join("\n    "),
            waste as f64 * 32.0 / 1024.0,
            emitted as f64 * 32.0 / 1024.0,
            100.0 * waste as f64 / emitted as f64,
        );
    }
    proof_census_inner(label, p)
}

#[cfg(test)]
fn proof_census_inner(label: &str, p: &flock_core::proof::R1csProofCircuitMerged) {
    let sz = |b: Result<Vec<u8>, _>| b.map(|v| v.len()).unwrap_or(0) as f64 / 1024.0;
    let total = sz(bincode::serialize(p));
    let lig = &p.pcs_open.inner.ligerito;
    let rows = |v: &Vec<flock_core::pcs::ligerito::RecursiveProof>| -> (f64, f64) {
        (
            v.iter()
                .map(|r| sz(bincode::serialize(&r.opened_rows)))
                .sum(),
            v.iter()
                .map(|r| sz(bincode::serialize(&r.merkle_proof)))
                .sum(),
        )
    };
    let (rec_rows, rec_paths) = rows(&lig.recursive_proofs);
    let l0_rows = sz(bincode::serialize(&lig.initial_proof.opened_rows));
    let l0_paths = sz(bincode::serialize(&lig.initial_proof.merkle_proof));
    println!(
        "\n  PROOF CENSUS — {label}: {total:.1} KiB\n\
         \x20   boolean PIOP        {:6.1}\n\
         \x20   element PIOP        {:6.1}\n\
         \x20   wiring              {:6.1}\n\
         \x20   merged rounds       {:6.1}\n\
         \x20   ring switches       {:6.1}\n\
         \x20   multipoint values   {:6.1}   (128 per rs claim)\n\
         \x20   multipoint rounds   {:6.1}\n\
         \x20   multipoint anchor   {:6.1}\n\
         \x20   inner: L0 rows      {:6.1}\n\
         \x20   inner: L0 paths     {:6.1}\n\
         \x20   inner: rec rows     {:6.1}\n\
         \x20   inner: rec paths    {:6.1}\n\
         \x20   inner: caps         {:6.1}   (L0 {:.1} + rec {:.1})\n\
         \x20   inner: final block  {:6.1}\n\
         \x20   inner: sumcheck     {:6.1}",
        sz(bincode::serialize(&p.boolean)),
        sz(bincode::serialize(&p.element)),
        sz(bincode::serialize(&p.wiring)),
        sz(bincode::serialize(&p.pcs_open.merged_rounds)),
        sz(bincode::serialize(&p.pcs_open.ring_switches)),
        sz(bincode::serialize(&p.pcs_open.frobenius.values)),
        sz(bincode::serialize(&p.pcs_open.frobenius.rounds)),
        sz(bincode::serialize(&p.pcs_open.frobenius.anchor)),
        l0_rows,
        l0_paths,
        rec_rows,
        rec_paths,
        sz(bincode::serialize(&lig.initial_cap)) + sz(bincode::serialize(&lig.recursive_caps)),
        sz(bincode::serialize(&lig.initial_cap)),
        sz(bincode::serialize(&lig.recursive_caps)),
        sz(bincode::serialize(&lig.final_proof)),
        sz(bincode::serialize(&lig.sumcheck_transcript)),
    );
}

#[cfg(test)]
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
    // Where a stage measures its wall directly, print what the phases add up
    // to beside it: the difference is real per-proof cost that no phase timer
    // owns, and quoting the sum alone hides it.
    if runs.iter().any(|o| o.wall_ms > 0.0) {
        let (wall, summed) = (
            median_of(runs, |o| o.wall_ms),
            median_of(runs, |o| o.summed()),
        );
        println!(
            "    {:9} MEASURED wall {:7.1} ms vs phase sum {:7.1} ({:+.1} unaccounted)",
            "",
            wall,
            summed,
            wall - summed,
        );
    }
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
#[cfg(test)]
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

#[cfg(test)]
struct Rng(u64);
#[cfg(test)]
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
/// `tests/circuit_builder.rs`; duplicated rather than shared because a lib
/// module cannot import from the crate's `tests/` binaries.)
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
#[cfg(test)]
struct MerklePathGate {
    layout: MerkleTreeLayout,
    nu: usize,
}

#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
struct Tree {
    data: Vec<u8>,
    flat: Vec<[u8; 32]>,
    root: [u8; 32],
    depth: usize,
    leaf_bytes: usize,
}

#[cfg(test)]
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
#[cfg(test)]
fn table_index(pos: usize, _depth: usize) -> u128 {
    pos as u128
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

/// Extension-field version of the opened-row evaluation. Each extension
/// value is represented by its `(c0, c1)` base-field wires. A product
///
/// ```text
/// (a0 + a1 u)(b0 + b1 u),  u^2 = u + x^-1,
/// ```
///
/// is constrained with the three Karatsuba products `a0 b0`, `a1 b1`, and
/// `(a0+a1)(b0+b1)`; the two output limbs are linear combinations of those
/// products. L0 words enter as `(word, 0)`, while recursive commitment rows
/// already contain adjacent `(c0, c1)` words.
struct LeafEvalGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    lay: LeafLayout256,
}

#[derive(Clone, Copy)]
struct LeafLayout256 {
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

impl LeafLayout256 {
    fn new(lanes: usize) -> Self {
        assert!(lanes.is_power_of_two() && lanes >= 2);
        let vars = lanes.trailing_zeros() as usize;
        let v = 2 * lanes;
        let alpha = v + 2 * vars;
        let prev = alpha + 2;
        let fold = prev + 2;
        let t = fold + 5 * (lanes - 1);
        let acc = t + 3;
        let k = t + 5;
        Self {
            lanes,
            vars,
            v,
            alpha,
            prev,
            fold,
            n_in: fold,
            t,
            acc,
            k,
            kappa: k.next_power_of_two().trailing_zeros().max(2) as usize,
        }
    }

    fn base(&self, l: usize) -> usize {
        (1..l).fold(self.fold, |acc, k| acc + 5 * (self.lanes >> k))
    }

    fn prev_pair(&self, l: usize, j: usize) -> [usize; 2] {
        if l == 1 {
            [2 * j, 2 * j + 1]
        } else {
            let base = self.base(l - 1) + 5 * j;
            [base + 3, base + 4]
        }
    }

    fn y(&self) -> [usize; 2] {
        let base = self.base(self.vars);
        [base + 3, base + 4]
    }
}

/// Emit `out = add + a*b` over F256, returning the next unused column.
/// The five emitted columns are the three Karatsuba products and two limbs.
fn build_mac256(
    b: &mut flock_core::element_r1cs::ElementTableBuilder,
    at: usize,
    add: Option<[usize; 2]>,
    a: [usize; 2],
    rhs: [usize; 2],
) -> usize {
    let one = F128::ONE;
    let nr = flock_core::field::gf2_256::QUADRATIC_NONRESIDUE;
    b.mult(at, a[0], rhs[0]);
    b.mult(at + 1, a[1], rhs[1]);
    b.mult_lin(
        at + 2,
        &[(a[0], one), (a[1], one)],
        &[(rhs[0], one), (rhs[1], one)],
    );
    let mut c0 = vec![(at, one), (at + 1, nr)];
    let mut c1 = vec![(at + 2, one), (at, one)];
    if let Some(add) = add {
        c0.push((add[0], one));
        c1.push((add[1], one));
    }
    b.linear(at + 3, &c0);
    b.linear(at + 4, &c1);
    at + 5
}

fn eval_mac256(add: F256, a: F256, b: F256) -> F256 {
    add + a * b
}

impl LeafEvalGate256 {
    fn new(lanes: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let lay = LeafLayout256::new(lanes);
        let mut b = ElementTableBuilder::new(lay.kappa);
        for c in 0..lay.n_in {
            b.free_wire(c);
        }
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let left = lay.prev_pair(l, 2 * i);
                let right = lay.prev_pair(l, 2 * i + 1);
                let challenge = [lay.v + 2 * (l - 1), lay.v + 2 * (l - 1) + 1];
                let at = lay.base(l) + 5 * i;
                let nr = flock_core::field::gf2_256::QUADRATIC_NONRESIDUE;
                b.mult_lin(
                    at,
                    &[(left[0], one), (right[0], one)],
                    &[(challenge[0], one)],
                );
                b.mult_lin(
                    at + 1,
                    &[(left[1], one), (right[1], one)],
                    &[(challenge[1], one)],
                );
                b.mult_lin(
                    at + 2,
                    &[
                        (left[0], one),
                        (right[0], one),
                        (left[1], one),
                        (right[1], one),
                    ],
                    &[(challenge[0], one), (challenge[1], one)],
                );
                b.linear(at + 3, &[(left[0], one), (at, one), (at + 1, nr)]);
                b.linear(at + 4, &[(left[1], one), (at + 2, one), (at, one)]);
            }
        }
        build_mac256(
            &mut b,
            lay.t,
            Some([lay.prev, lay.prev + 1]),
            [lay.alpha, lay.alpha + 1],
            lay.y(),
        );
        Self {
            ty: std::sync::Arc::new(b.build().expect("extension leaf-eval block is valid")),
            lay,
        }
    }
}

impl GateType for LeafEvalGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.lay.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.lay.acc));
        schema.push(IoWord::output(self.lay.acc + 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let lay = self.lay;
        let mut z = vec![F128::ZERO; lay.k];
        z[..lay.n_in].copy_from_slice(&inputs[..lay.n_in]);
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let lp = lay.prev_pair(l, 2 * i);
                let rp = lay.prev_pair(l, 2 * i + 1);
                let left = F256::new(z[lp[0]], z[lp[1]]);
                let right = F256::new(z[rp[0]], z[rp[1]]);
                let r = F256::new(z[lay.v + 2 * (l - 1)], z[lay.v + 2 * (l - 1) + 1]);
                let out = eval_mac256(left, left + right, r);
                let at = lay.base(l) + 5 * i;
                let p0 = (left.c0 + right.c0) * r.c0;
                let p1 = (left.c1 + right.c1) * r.c1;
                let p2 = (left.c0 + right.c0 + left.c1 + right.c1) * (r.c0 + r.c1);
                z[at] = p0;
                z[at + 1] = p1;
                z[at + 2] = p2;
                z[at + 3] = out.c0;
                z[at + 4] = out.c1;
            }
        }
        let y = lay.y();
        let alpha = F256::new(z[lay.alpha], z[lay.alpha + 1]);
        let yv = F256::new(z[y[0]], z[y[1]]);
        let prev = F256::new(z[lay.prev], z[lay.prev + 1]);
        let p0 = alpha.c0 * yv.c0;
        let p1 = alpha.c1 * yv.c1;
        let p2 = (alpha.c0 + alpha.c1) * (yv.c0 + yv.c1);
        z[lay.t] = p0;
        z[lay.t + 1] = p1;
        z[lay.t + 2] = p2;
        let acc = prev + alpha * yv;
        z[lay.acc] = acc.c0;
        z[lay.acc + 1] = acc.c1;
        outputs.extend_from_slice(&z[lay.acc..lay.acc + 2]);
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

// ---------------------------------------------------------------------------
// MVP-4: the vertical slice
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MVP-5: every level's query phase
// ---------------------------------------------------------------------------

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
fn cw(
    sb: &mut ShapeBuilder,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    v: F128,
) -> Wire {
    match consts.iter().find(|&&(x, _)| x == v) {
        Some(&(_, w)) => w,
        None => {
            vals.push(v);
            let w = sb.fixed_public_input(v);
            consts.push((v, w));
            w
        }
    }
}

/// Which byte payloads of a tape stay PUBLIC under the witness/public
/// split: every `observe_bytes` payload — the STATEMENT surfaces (registry
/// digest, counts, caps, a child's circuit digest + public words) and
/// nothing else. PoW nonces share the payload counter but remain private
/// witnesses constrained by the fused BLAKE3 and bit-spread rows.
fn bytes_payload_mask(ops: &[flock_core::transcript_record::TranscriptOp]) -> Vec<bool> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut v = Vec::new();
    for op in ops {
        match op {
            Op::ObserveBytes(_) => v.push(true),
            Op::Pow { .. } | Op::LegacyPow { .. } => v.push(false),
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
///
/// **The FORKED transcript needs nothing from this function.** A circuit-bound
/// union proof always forks (the wiring argument runs on its own chain), but
/// [`merge_chain`] presents both chains as ONE — rows spliced at the fork
/// point, indices remapped — so this loop only ever sees a linear trace. The
/// fork's whole in-circuit footprint arrives through `cross`: four words that
/// are ALIASES of earlier squeeze outputs rather than declared inputs.
///
/// Still open, and cheaper still: the child chain could CONTINUE from the
/// fork-point CV under a domain byte instead of seed-squeeze-then-absorb.
/// That drops the two seed rows and takes the fork's cost to ~one row.
fn emit_fs_chain(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_core::transcript_record::Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    pub_payloads: &[bool],
    cross: &[Option<(usize, usize)>],
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    emit_fs_chain_partitioned(
        sb,
        b3,
        None,
        iv,
        trace,
        stream,
        bytes,
        vals,
        consts,
        pub_payloads,
        cross,
    )
}

/// As [`emit_fs_chain`], with rows at and after `primary_rows` emitted into
/// a second slot carrying the same BLAKE3 relation. Wires may cross the slot
/// boundary normally; the circuit's copy constraints preserve the chain.
fn emit_fs_chain_partitioned(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    alternate: Option<(flock_core::circuit::builder::SlotId, usize)>,
    iv: [Wire; 2],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_core::transcript_record::Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    pub_payloads: &[bool],
    cross: &[Option<(usize, usize)>],
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    use crate::r1cs_hashes::fs_chain::CvSource;
    use flock_core::transcript_record::StreamWord;
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let b3_row = match alternate {
            Some((slot, primary_rows)) if i >= primary_rows => slot,
            _ => b3,
        };
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = cw(sb, vals, consts, pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(b3_row, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                let left = match link.cv {
                    CvSource::Row(_) => [outs[l][0], outs[l][1]],
                    CvSource::RowHi(_) => [outs[l][2], outs[l][3]],
                    CvSource::Iv => unreachable!(),
                };
                (iv, [left[0], left[1], outs[right][0], outs[right][1]])
            }
            None if trace.block_offsets[i].is_none() => {
                // A sponge-chain SQUEEZE output row (transcript-v2): zero
                // message block via the shared constant, chaining value
                // from the link.
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let z4 = cw(sb, vals, consts, F128::ZERO);
                (cv_in, [z4, z4, z4, z4])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = trace.block_word_counts[i];
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        cw(sb, vals, consts, F128::ZERO)
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            // A CROSS-LINK word is not an input at all: it IS
                            // an earlier squeeze's output (the fork's seed on
                            // the child side, the child's closing digest on
                            // the parent's). Aliasing the wire is the whole
                            // in-circuit cost of the fork — zero extra rows,
                            // and the link is unforgeable because the row that
                            // produced it is the same row the challenge came
                            // from.
                            None if cross.get(wi).copied().flatten().is_some() => {
                                let (row, half) = cross[wi].unwrap();
                                assert!(row < i, "cross-link word {wi} reads row {row} >= {i}");
                                let w = outs[row][half];
                                word_wire[wi] = Some(w);
                                w
                            }
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
        outs.push(sb.gate(b3_row, &g_in));
    }
    (outs, word_wire)
}

/// Splice every fork's ops inline at its fork position and drop the `Merge`
/// markers — the FLAT view of a forked transcript.
///
/// This is the whole locator story. Every region walker in this file resolves
/// its position by `find(label)` over a flat op list and by counting
/// value/challenge/finalize ops up to an index; a fork's ops sit inline at the
/// fork slot (the recorder splices values, payloads and challenges at the
/// fork-time bases for exactly this reason), so on the flattened view every
/// label is found and every ordinal is the GLOBAL one. No walker changes, no
/// chain index anywhere.
fn flatten_ops(
    ops: &[flock_core::transcript_record::TranscriptOp],
) -> Vec<flock_core::transcript_record::TranscriptOp> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            Op::Forked { ops: child, .. } => out.extend(flatten_ops(child)),
            Op::Merge { .. } => {}
            other => out.push(other.clone()),
        }
    }
    out
}

/// A forked transcript presented to the circuit as ONE chain.
///
/// The fork is two independent BLAKE3 chains, but the emitter, the region
/// locators and every `trace.squeezes[fin]` site want a single linear
/// numbering. So the child's rows are SPLICED into the parent's at the fork
/// point — after the two seed squeezes, before the parent's post-fork absorbs
/// — and every row index is remapped. The result is indistinguishable from an
/// unforked trace except for four wires:
///
/// - the child's two opening seed words ARE the parent's two seed-squeeze
///   outputs, and
/// - the parent's two merge words ARE the child's two closing-squeeze outputs.
///
/// [`MergedChain::cross`] carries those four as (row, half) aliases, so they
/// cost no gates and no rows: the emitter wires them instead of declaring
/// inputs. Splicing at the fork point is what makes that possible — both
/// sources are already emitted when their consumer's row comes up.
struct MergedChain {
    /// Parent words then child words, in the same order as `trace`'s rows.
    stream: flock_core::transcript_record::Stream,
    bytes: Vec<u8>,
    trace: crate::r1cs_hashes::fs_chain::FsChainTrace,
    /// Per merged word: `Some((row, half))` iff the word is a cross-link.
    cross: Vec<Option<(usize, usize)>>,
}

/// Build the merged view from a recorded shape's ops and its parent stream.
/// With no fork this is the identity (the trace the sites built by hand).
fn merge_chain(
    ops: &[flock_core::transcript_record::TranscriptOp],
    stream: &flock_core::transcript_record::Stream,
    values: &[F128],
    payloads: &[Vec<u8>],
) -> MergedChain {
    use crate::r1cs_hashes::fs_chain::{CvSource, Link, trace_duplex_forked};
    use flock_core::transcript_record::StreamWord;
    let chains = trace_duplex_forked(ops, stream, values, payloads);
    let parent_bytes = stream.to_bytes(values, payloads);
    if chains.children.is_empty() {
        let cross = vec![None; stream.words.len()];
        return MergedChain {
            stream: stream.clone(),
            bytes: parent_bytes,
            trace: chains.parent,
            cross,
        };
    }
    let p = &chains.parent;
    let n_ch = chains.children.len();
    // Every fork's splice point, in parent-row order. The forks a proof takes
    // are SEQUENTIAL (the wiring's opens and closes before the opening phase
    // starts its own), so the splits are strictly increasing and each child
    // occupies one contiguous run of merged rows. `last_seed` is the second
    // of the pair — `seed_squeeze` names the first — which is exactly where
    // the flattened ops put the child.
    let last_seed: Vec<usize> = chains.children.iter().map(|c| c.seed_squeeze + 1).collect();
    let splits: Vec<usize> = last_seed
        .iter()
        .map(|&k| p.squeezes[k].iter().copied().max().unwrap() + 1)
        .collect();
    assert!(
        splits.windows(2).all(|w| w[0] <= w[1]) && last_seed.windows(2).all(|w| w[0] < w[1]),
        "forks must be sequential — nested or interleaved forks are not supported"
    );
    let ncs: Vec<usize> = chains.children.iter().map(|c| c.trace.rows.len()).collect();
    // A parent row shifts by every child spliced at or before it; child `i`
    // starts after its split plus every earlier child's rows.
    let pmap = |r: usize| {
        r + (0..n_ch)
            .filter(|&j| splits[j] <= r)
            .map(|j| ncs[j])
            .sum::<usize>()
    };
    let child_base: Vec<usize> = (0..n_ch)
        .map(|i| splits[i] + ncs[..i].iter().sum::<usize>())
        .collect();
    let cmap = |i: usize| {
        let base = child_base[i];
        move |r: usize| base + r
    };
    let remap = |l: &Link, f: &dyn Fn(usize) -> usize| Link {
        cv: match l.cv {
            CvSource::Iv => CvSource::Iv,
            CvSource::Row(r) => CvSource::Row(f(r)),
            CvSource::RowHi(r) => CvSource::RowHi(f(r)),
        },
        right: l.right.map(f),
        repeats: l.repeats.map(f),
    };

    // Child streams and their word/byte offsets in the merged view.
    let cstreams: Vec<&flock_core::transcript_record::Stream> =
        stream.forks.iter().map(|f| &f.stream).collect();
    let woffs: Vec<usize> = (0..n_ch)
        .map(|i| stream.words.len() + cstreams[..i].iter().map(|s| s.words.len()).sum::<usize>())
        .collect();
    debug_assert_eq!(
        parent_bytes.len(),
        stream.words.len() * 16,
        "words are 16 bytes"
    );

    // Splice: parent up to split 0, child 0, parent to split 1, child 1, ...
    let mut rows = Vec::new();
    let mut links: Vec<Link> = Vec::new();
    let mut block_offsets = Vec::new();
    let mut block_word_counts = Vec::new();
    let mut at = 0usize;
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        rows.extend_from_slice(&p.rows[at..splits[i]]);
        links.extend(p.links[at..splits[i]].iter().map(|l| remap(l, &pmap)));
        block_offsets.extend_from_slice(&p.block_offsets[at..splits[i]]);
        block_word_counts.extend_from_slice(&p.block_word_counts[at..splits[i]]);
        rows.extend_from_slice(&c.trace.rows);
        links.extend(c.trace.links.iter().map(|l| remap(l, &cm)));
        block_offsets.extend(
            c.trace
                .block_offsets
                .iter()
                .map(|o| o.map(|b| b + woffs[i] * 16)),
        );
        block_word_counts.extend_from_slice(&c.trace.block_word_counts);
        at = splits[i];
    }
    rows.extend_from_slice(&p.rows[at..]);
    links.extend(p.links[at..].iter().map(|l| remap(l, &pmap)));
    block_offsets.extend_from_slice(&p.block_offsets[at..]);
    block_word_counts.extend_from_slice(&p.block_word_counts[at..]);

    // The same splice on the squeeze list, the words and the finalize points.
    let mut squeezes: Vec<Vec<usize>> = Vec::new();
    let mut squeeze_words: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut words = stream.words.clone();
    let mut finalize_after: Vec<usize> = Vec::new();
    let mut bytes = parent_bytes;
    let mut at = 0usize;
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        squeezes.extend(
            p.squeezes[at..=last_seed[i]]
                .iter()
                .map(|s| s.iter().copied().map(pmap).collect::<Vec<_>>()),
        );
        squeeze_words.extend(
            p.squeeze_words[at..=last_seed[i]]
                .iter()
                .map(|s| s.iter().map(|&(r, w)| (pmap(r), w)).collect::<Vec<_>>()),
        );
        squeezes.extend(
            c.trace
                .squeezes
                .iter()
                .map(|s| s.iter().copied().map(&cm).collect::<Vec<_>>()),
        );
        squeeze_words.extend(
            c.trace
                .squeeze_words
                .iter()
                .map(|s| s.iter().map(|&(r, w)| (cm(r), w)).collect::<Vec<_>>()),
        );
        finalize_after.extend_from_slice(&stream.finalize_after[at..=last_seed[i]]);
        finalize_after.extend(cstreams[i].finalize_after.iter().map(|w| w + woffs[i]));
        words.extend(cstreams[i].words.iter().cloned());
        bytes.extend_from_slice(&cstreams[i].to_bytes(values, payloads));
        at = last_seed[i] + 1;
    }
    squeezes.extend(
        p.squeezes[at..]
            .iter()
            .map(|s| s.iter().copied().map(pmap).collect::<Vec<_>>()),
    );
    squeeze_words.extend(
        p.squeeze_words[at..]
            .iter()
            .map(|s| s.iter().map(|&(r, w)| (pmap(r), w)).collect::<Vec<_>>()),
    );
    finalize_after.extend_from_slice(&stream.finalize_after[at..]);

    // The four cross-links. Each side's pair is two CONSECUTIVE
    // `ObserveScalar`s, and the walk emits [header, value] per observe — so
    // the second word sits two after the first.
    //
    // Each link is CHECKED, not just placed. Getting a cross index wrong is
    // the one mistake nothing downstream would notice: challenges replay
    // identically whether a word is aliased to its squeeze or declared as a
    // witness input that happens to hold the same value — and the second is a
    // soundness hole, because it leaves the wiring's chain unbound. So the
    // row is compressed here and matched against the recorded value.
    let mut cross = vec![None; words.len()];
    let mut link_word = |wi: usize, row: usize| {
        let StreamWord::Value(vi) = words[wi] else {
            panic!("cross-link word {wi} is not an observed value");
        };
        let (cv, m, counter, blen, flags) = rows[row];
        let out = crate::r1cs_hashes::blake3::blake3_compress(&cv, &m, counter, blen, flags);
        let mut b = [0u8; 16];
        for (i, w) in out[..4].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(
            F128::new(
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..].try_into().unwrap()),
            ),
            values[vi],
            "cross-link word {wi} does not carry row {row}'s squeeze output"
        );
        cross[wi] = Some((row, 0));
    };
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        for (k, half) in [(c.seed_squeeze, 0usize), (c.seed_squeeze + 1, 1)] {
            link_word(
                woffs[i] + c.child_seed_word + 2 * half,
                pmap(p.squeezes[k][0]),
            );
        }
        for (k, half) in [(c.digest_squeeze, 0usize), (c.digest_squeeze + 1, 1)] {
            link_word(c.parent_digest_word + 2 * half, cm(c.trace.squeezes[k][0]));
        }
    }

    assert_eq!(
        cross.iter().filter(|c| c.is_some()).count(),
        4 * n_ch,
        "each fork contributes exactly four cross-link words"
    );
    MergedChain {
        stream: flock_core::transcript_record::Stream {
            words,
            finalize_after,
            forks: Vec::new(),
        },
        bytes,
        trace: crate::r1cs_hashes::fs_chain::FsChainTrace {
            rows,
            links,
            squeezes,
            squeeze_words,
            block_offsets,
            block_word_counts,
        },
        cross,
    }
}

/// THE SPLICE DIFFERENTIAL: every challenge the recorder produced must
/// fall back out of the merged chain at its flattened finalize ordinal.
///
/// This is what makes [`merge_chain`] trustworthy rather than merely
/// plausible. A single match requires the row order, the index remapping, the
/// squeeze ordering AND the byte-offset shift to all be right at once, and the
/// child's own challenges sit in the middle of the sequence — the run walks
/// straight through the fork. Cheap enough to leave on at every real shape.
fn assert_chain_replays(
    ops: &[flock_core::transcript_record::TranscriptOp],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    chals: &[F128],
) {
    use flock_core::transcript_record::TranscriptOp as Op;
    let (mut fin, mut ch, mut checked) = (0usize, 0usize, 0usize);
    for op in ops {
        let n = match op {
            Op::SqueezeScalar => 1,
            Op::SqueezeSlice(n) => *n,
            _ => 0,
        };
        for j in 0..n {
            let (row, word) = trace.squeeze_words[fin][j];
            let (cv, m, counter, blen, flags) = trace.rows[row];
            let out = crate::r1cs_hashes::blake3::blake3_compress(&cv, &m, counter, blen, flags);
            let mut b = [0u8; 16];
            for (i, w) in out[word * 4..word * 4 + 4].iter().enumerate() {
                b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            assert_eq!(
                F128::new(
                    u64::from_le_bytes(b[..8].try_into().unwrap()),
                    u64::from_le_bytes(b[8..].try_into().unwrap()),
                ),
                chals[ch + j],
                "the merged chain diverges at finalize {fin}, word {j} (challenge {})",
                ch + j,
            );
            checked += 1;
        }
        if op.finalizes() {
            fin += 1;
        }
        match op {
            Op::SqueezeScalar => ch += 1,
            Op::SqueezeSlice(n) => ch += n,
            _ => {}
        }
    }
    assert!(checked > 0, "no scalar squeezes to replay");
    assert_eq!(fin, trace.squeezes.len(), "finalize count vs squeeze rows");
    assert_eq!(ch, chals.len(), "challenge count vs the recorded list");
}

/// Independent row count for the duplex transcript, including every fork as
/// its own IV-rooted chain.  It deliberately derives absorption from the
/// serialized stream rather than from [`FsChainTrace`].
fn duplex_row_count_model(
    ops: &[flock_core::transcript_record::TranscriptOp],
    stream: &flock_core::transcript_record::Stream,
) -> usize {
    use flock_core::transcript_record::TranscriptOp as Op;

    let mut pending_pow = None;
    let mut finals: Vec<(&Op, Option<u32>)> = Vec::new();
    for op in ops {
        match op {
            Op::Pow { bits } => {
                assert!(
                    pending_pow.replace(*bits).is_none(),
                    "nested fused PoW markers"
                );
            }
            op if op.finalizes() => finals.push((op, pending_pow.take())),
            Op::Forked { .. } => {}
            _ => assert!(pending_pow.is_none(), "fused PoW must precede its squeeze"),
        }
    }
    assert!(pending_pow.is_none(), "fused PoW marker without a squeeze");
    assert_eq!(finals.len(), stream.finalize_after.len());

    let (mut rows, mut at, mut pending) = (0usize, 0usize, 0usize);
    for (k, &upto) in stream.finalize_after.iter().enumerate() {
        pending += 16 * (upto - at);
        at = upto;
        let (op, pow_bits) = finals[k];
        let words = op.squeezed_bytes() / 16;
        if pow_bits.is_some() {
            // Retain the final (possibly full) block for the fused row.
            rows += pending.saturating_sub(1) / 64;
            rows += 1 + words.saturating_sub(3).div_ceil(4);
        } else {
            // Ordinary absorb drains every full block before the squeeze.
            rows += pending / 64;
            rows += 1 + words.saturating_sub(4).div_ceil(4);
        }
        pending = 0;
    }
    pending += 16 * (stream.words.len() - at);
    rows += pending / 64;

    let children: Vec<_> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Forked { label, ops } => Some((label, ops)),
            _ => None,
        })
        .collect();
    assert_eq!(children.len(), stream.forks.len());
    for ((label, child_ops), child_stream) in children.into_iter().zip(&stream.forks) {
        assert_eq!(label, &child_stream.label);
        rows += duplex_row_count_model(child_ops, &child_stream.stream);
    }
    rows
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

/// The Ligerito sumcheck spine over F256. The batching coefficient `beta`
/// remains a base-field scalar, while the quadratic, messages, fold
/// challenge, running target, and outputs are pairs of base-field wires.
struct SpineGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

const SP256_IN: usize = 17;
const SP256_K: usize = 50;

impl SpineGate256 {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let pair = |at| [at, at + 1];
        let (c, qb, qa, tr, u0, u2, y, beta, r) = (
            pair(0),
            pair(2),
            pair(4),
            pair(6),
            pair(8),
            pair(10),
            pair(12),
            14,
            pair(15),
        );
        let (pc, pb, pa, pt) = (pair(17), pair(19), pair(21), pair(23));
        let (co, bo, ao, tro) = (pair(25), pair(27), pair(29), pair(31));
        let mut b = ElementTableBuilder::new(6);
        for w in 0..SP256_IN {
            b.free_wire(w);
        }
        b.mult(pc[0], beta, u0[0]);
        b.mult(pc[1], beta, u0[1]);
        b.mult_lin(pb[0], &[(y[0], one), (u2[0], one)], &[(beta, one)]);
        b.mult_lin(pb[1], &[(y[1], one), (u2[1], one)], &[(beta, one)]);
        b.mult(pa[0], beta, u2[0]);
        b.mult(pa[1], beta, u2[1]);
        b.mult(pt[0], beta, y[0]);
        b.mult(pt[1], beta, y[1]);
        for (out, lhs, rhs) in [(co, c, pc), (bo, qb, pb), (ao, qa, pa), (tro, tr, pt)] {
            b.linear(out[0], &[(lhs[0], one), (rhs[0], one)]);
            b.linear(out[1], &[(lhs[1], one), (rhs[1], one)]);
        }
        let r2_at = 33;
        build_mac256(&mut b, r2_at, None, r, r);
        let r2 = pair(r2_at + 3);
        let rb_at = 38;
        build_mac256(&mut b, rb_at, None, r, bo);
        let rb = pair(rb_at + 3);
        let ra_at = 43;
        build_mac256(&mut b, ra_at, None, r2, ao);
        let ra = pair(ra_at + 3);
        b.linear(48, &[(co[0], one), (rb[0], one), (ra[0], one)]);
        b.linear(49, &[(co[1], one), (rb[1], one), (ra[1], one)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("extension spine gate is valid")),
        }
    }
}

impl GateType for SpineGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..SP256_IN).map(IoWord::input).collect();
        for o in [25, 26, 27, 28, 29, 30, 31, 32, 48, 49] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let get = |z: &[F128], at| F256::new(z[at], z[at + 1]);
        let put = |z: &mut [F128], at, v: F256| {
            z[at] = v.c0;
            z[at + 1] = v.c1;
        };
        let mut z = vec![F128::ZERO; SP256_K];
        z[..SP256_IN].copy_from_slice(&inputs[..SP256_IN]);
        let (c, b, a, tr, u0, u2, y, beta, r) = (
            get(&z, 0),
            get(&z, 2),
            get(&z, 4),
            get(&z, 6),
            get(&z, 8),
            get(&z, 10),
            get(&z, 12),
            z[14],
            get(&z, 15),
        );
        let (pc, pb, pa, pt) = (u0 * beta, (y + u2) * beta, u2 * beta, y * beta);
        put(&mut z, 17, pc);
        put(&mut z, 19, pb);
        put(&mut z, 21, pa);
        put(&mut z, 23, pt);
        let (co, bo, ao, tro) = (c + pc, b + pb, a + pa, tr + pt);
        put(&mut z, 25, co);
        put(&mut z, 27, bo);
        put(&mut z, 29, ao);
        put(&mut z, 31, tro);
        let r2 = r * r;
        let rb = r * bo;
        let ra = r2 * ao;
        for (at, x, lhs, rhs) in [(33, r2, r, r), (38, rb, r, bo), (43, ra, r2, ao)] {
            let p0 = lhs.c0 * rhs.c0;
            let p1 = lhs.c1 * rhs.c1;
            let p2 = (lhs.c0 + lhs.c1) * (rhs.c0 + rhs.c1);
            z[at] = p0;
            z[at + 1] = p1;
            z[at + 2] = p2;
            put(&mut z, at + 3, x);
        }
        let to = co + rb + ra;
        put(&mut z, 48, to);
        outputs.extend_from_slice(&[
            co.c0, co.c1, bo.c0, bo.c1, ao.c0, ao.c1, tro.c0, tro.c1, to.c0, to.c1,
        ]);
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

fn emit_spine256(
    sb: &mut ShapeBuilder,
    slot: flock_core::circuit::builder::SlotId,
    c: [Wire; 2],
    b: [Wire; 2],
    a: [Wire; 2],
    tr: [Wire; 2],
    u0: [Wire; 2],
    u2: [Wire; 2],
    y: [Wire; 2],
    beta: Wire,
    r: [Wire; 2],
) -> [[Wire; 2]; 5] {
    let inputs: Vec<Wire> = [c, b, a, tr, u0, u2, y]
        .into_iter()
        .flatten()
        .chain([beta])
        .chain(r)
        .collect();
    let out = sb.gate(slot, &inputs);
    [
        [out[0], out[1]],
        [out[2], out[3]],
        [out[4], out[5]],
        [out[6], out[7]],
        [out[8], out[9]],
    ]
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

/// Residual-basis accumulation for extension-valued fold challenges. The
/// novel-basis chain and query weights are base-field values; only the
/// products involving later fold challenges and the running accumulators
/// need two limbs.
struct ResidualWeightsGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    coeffs: Vec<F128>,
}

impl ResidualWeightsGate256 {
    /// Sized to the deepest walked ladder, like `spread_w`: the
    /// m32 FAST chain leaf's L0 needs `pl 15 + yr_log 4 = 19` (the residual
    /// domain is 16 entries = two 8-chunks; the chunk-high extension reads
    /// `weights[pl + yr_log - 1]`). The m29 outer ladders stay below this;
    /// anything deeper fails the `lmc` assert loudly at build.
    const N_WEIGHTS: usize = 19;

    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let sks = sk_at_vks(Self::N_WEIGHTS);
        let coeffs: Vec<F128> = (0..Self::N_WEIGHTS - 1)
            .map(|k| {
                assert_ne!(sks[k + 1], F128::ZERO, "novel-basis normalizer is nonzero");
                sks[k] * sks[k] * sks[k + 1].inv()
            })
            .collect();
        // in: W_0=q, one. out: W_1..W_18.
        let mut b = ElementTableBuilder::new(5);
        b.free_wire(0).free_wire(1);
        let mut prev = 0;
        for (j, &d) in coeffs.iter().enumerate() {
            let out = 2 + j;
            b.mult_lin(out, &[(prev, d)], &[(prev, o), (1, o)]);
            prev = out;
        }
        Self {
            ty: std::sync::Arc::new(b.build().expect("normalized residual weights gate")),
            coeffs,
        }
    }
}

impl GateType for ResidualWeightsGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema = vec![IoWord::input(0), IoWord::input(1)];
        schema.extend((2..2 + self.coeffs.len()).map(IoWord::output));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 2 + self.coeffs.len()];
        z[..2].copy_from_slice(&inputs[..2]);
        let mut prev = 0;
        for (j, &d) in self.coeffs.iter().enumerate() {
            let out = 2 + j;
            z[out] = d * z[prev] * (z[prev] + z[1]);
            prev = out;
        }
        outputs.extend_from_slice(&z[2..]);
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

/// Three consecutive F256 residual-prefix factors. Ligerito introduces
/// three post-introduction fold challenges per level, so every active
/// residual prefix is a chain of this one relation:
///
/// `P' = P product_i (1 + R_i (1 + W_i))`, for `i=0,1,2`.
struct ResidualPrefix3Gate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    out: [usize; 2],
}

impl ResidualPrefix3Gate256 {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let nr = flock_core::field::QUADRATIC_NONRESIDUE;
        // in: prefix pair, three challenge pairs, three base weights, one.
        let (n_in, one) = (12usize, 11usize);
        let mut b = ElementTableBuilder::new(6);
        for col in 0..n_in {
            b.free_wire(col);
        }
        let mut c = n_in;
        let mut pr = [0, 1];
        for i in 0..3 {
            let r = [2 + 2 * i, 2 + 2 * i + 1];
            let w = 8 + i;
            b.mult_lin(c, &[(r[0], o)], &[(one, o), (w, o)]);
            b.mult_lin(c + 1, &[(r[1], o)], &[(one, o), (w, o)]);
            let pk = [c, c + 1];
            c += 2;
            b.mult_lin(c, &[(pr[0], o)], &[(one, o), (pk[0], o)]);
            b.mult(c + 1, pr[1], pk[1]);
            b.mult_lin(
                c + 2,
                &[(pr[0], o), (pr[1], o)],
                &[(one, o), (pk[0], o), (pk[1], o)],
            );
            b.linear(c + 3, &[(c, o), (c + 1, nr)]);
            b.linear(c + 4, &[(c + 2, o), (c, o)]);
            pr = [c + 3, c + 4];
            c += 5;
        }
        assert_eq!(c, 33, "three residual-prefix factors use 33 columns");
        Self {
            ty: std::sync::Arc::new(b.build().expect("three-factor residual prefix gate")),
            out: pr,
        }
    }
}

impl GateType for ResidualPrefix3Gate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..12).map(IoWord::input).collect();
        schema.push(IoWord::output(self.out[0]));
        schema.push(IoWord::output(self.out[1]));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 33];
        z[..12].copy_from_slice(&inputs[..12]);
        let mut c = 12;
        let mut pr = F256::new(z[0], z[1]);
        for i in 0..3 {
            let r = F256::new(z[2 + 2 * i], z[2 + 2 * i + 1]);
            let pk = r * (z[11] + z[8 + i]);
            z[c] = pk.c0;
            z[c + 1] = pk.c1;
            c += 2;
            let factor = F256::new(z[11] + pk.c0, pk.c1);
            let product = pr * factor;
            z[c] = pr.c0 * factor.c0;
            z[c + 1] = pr.c1 * factor.c1;
            z[c + 2] = (pr.c0 + pr.c1) * (factor.c0 + factor.c1);
            z[c + 3] = product.c0;
            z[c + 4] = product.c1;
            pr = product;
            c += 5;
        }
        outputs.extend_from_slice(&[pr.c0, pr.c1]);
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

/// Add one residual query to all eight low-coordinate accumulators:
/// `acc_y' = acc_y + aw * prefix * product_{j:y_j=1} W_j`.
struct ResidualAccGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    acc_out: [[usize; 2]; 8],
}

impl ResidualAccGate256 {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: aw, prefix pair, three low weights, eight accumulator pairs.
        let (n_in, acc0) = (22usize, 6usize);
        let mut b = ElementTableBuilder::new(6);
        for col in 0..n_in {
            b.free_wire(col);
        }
        let mut c = n_in;
        b.mult(c, 0, 1).mult(c + 1, 0, 2);
        let t = [c, c + 1];
        c += 2;
        b.mult(c, 3, 4).mult(c + 1, 3, 5).mult(c + 2, 4, 5);
        b.mult(c + 3, c, 5);
        let weights = [
            None,
            Some(3),
            Some(4),
            Some(c),
            Some(5),
            Some(c + 1),
            Some(c + 2),
            Some(c + 3),
        ];
        c += 4;
        let mut contributions = [t; 8];
        for y in 1..8 {
            let w = weights[y].expect("a nonzero subset has a weight");
            b.mult(c, t[0], w).mult(c + 1, t[1], w);
            contributions[y] = [c, c + 1];
            c += 2;
        }
        let mut acc_out = [[0usize; 2]; 8];
        for y in 0..8 {
            b.linear(c, &[(acc0 + 2 * y, o), (contributions[y][0], o)]);
            b.linear(c + 1, &[(acc0 + 2 * y + 1, o), (contributions[y][1], o)]);
            acc_out[y] = [c, c + 1];
            c += 2;
        }
        assert_eq!(c, 58, "the residual accumulator uses 58 columns");
        Self {
            ty: std::sync::Arc::new(b.build().expect("residual accumulator gate")),
            acc_out,
        }
    }
}

impl GateType for ResidualAccGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..22).map(IoWord::input).collect();
        for out in self.acc_out {
            schema.push(IoWord::output(out[0]));
            schema.push(IoWord::output(out[1]));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 58];
        z[..22].copy_from_slice(&inputs[..22]);
        let prefix = F256::new(z[1], z[2]);
        let t = prefix * z[0];
        z[22] = t.c0;
        z[23] = t.c1;
        let low = [z[3], z[4], z[5]];
        z[24] = low[0] * low[1];
        z[25] = low[0] * low[2];
        z[26] = low[1] * low[2];
        z[27] = z[24] * low[2];
        let weights = [
            F128::ONE,
            low[0],
            low[1],
            z[24],
            low[2],
            z[25],
            z[26],
            z[27],
        ];
        let mut c = 28;
        let mut contributions = [t; 8];
        for y in 1..8 {
            contributions[y] = t * weights[y];
            z[c] = contributions[y].c0;
            z[c + 1] = contributions[y].c1;
            c += 2;
        }
        for y in 0..8 {
            z[c] = z[6 + 2 * y] + contributions[y].c0;
            z[c + 1] = z[6 + 2 * y + 1] + contributions[y].c1;
            outputs.push(z[c]);
            outputs.push(z[c + 1]);
            c += 2;
        }
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

/// [`live_element_input`] without the packed intermediate: the slot's rows
/// (from a DEFERRED run, [`CircuitWitness::take_rows_of`]) scatter straight
/// into the union block — the same `dst[(col << nu) + j] = row[col]` write
/// every element gate's `witness()` makes, minus the full-capacity buffer it
/// makes it into. `dst` arrives zeroed and a row shorter than the slot's
/// width leaves implicit zero columns, exactly as the packed path did.
fn live_element_input_from_rows(
    rows: Vec<Vec<F128>>,
    nu: usize,
) -> crate::prover::UnionElementSlotInput<'static> {
    crate::prover::UnionElementSlotInput::new(move |dst: &mut [F128]| {
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

/// Extension-field prefix product `seed * product_j (1 + a_j + b_j)`.
/// Every value occupies two base-field wires; `one` and `zero` inputs make
/// the constant extension element `(1, 0)` explicit and preserve the
/// all-zero padding-row convention.
struct PrefixGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    pl: usize,
    n_in: usize,
    out: [usize; 2],
    k: usize,
}

impl PrefixGate256 {
    fn new(pl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let n_in = 4 + 4 * pl; // seed pair, a pairs, b pairs, one, zero
        let one = n_in - 2;
        let zero = n_in - 1;
        let mut b = ElementTableBuilder::new(gate_kappa(n_in + 5 * pl));
        for w in 0..n_in {
            b.free_wire(w);
        }
        let mut c = n_in;
        let mut pr = [0, 1];
        for j in 0..pl {
            let a = [2 + 2 * j, 2 + 2 * j + 1];
            let bs = 2 + 2 * pl + 2 * j;
            let factor0 = vec![(one, o), (a[0], o), (bs, o)];
            let factor1 = vec![(zero, o), (a[1], o), (bs + 1, o)];
            let nr = flock_core::field::QUADRATIC_NONRESIDUE;
            b.mult_lin(c, &[(pr[0], o)], &factor0);
            b.mult_lin(c + 1, &[(pr[1], o)], &factor1);
            b.mult_lin(
                c + 2,
                &[(pr[0], o), (pr[1], o)],
                &[
                    (one, o),
                    (zero, o),
                    (a[0], o),
                    (a[1], o),
                    (bs, o),
                    (bs + 1, o),
                ],
            );
            b.linear(c + 3, &[(c, o), (c + 1, nr)]);
            b.linear(c + 4, &[(c + 2, o), (c, o)]);
            pr = [c + 3, c + 4];
            c += 5;
        }
        Self {
            ty: std::sync::Arc::new(b.build().expect("extension prefix gate")),
            pl,
            n_in,
            out: pr,
            k: c,
        }
    }
}

impl GateType for PrefixGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.out[0]));
        schema.push(IoWord::output(self.out[1]));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let one = self.n_in - 2;
        let zero = self.n_in - 1;
        let mut c = self.n_in;
        let mut pr = F256::new(z[0], z[1]);
        for j in 0..self.pl {
            let a = F256::new(z[2 + 2 * j], z[2 + 2 * j + 1]);
            let bs = 2 + 2 * self.pl + 2 * j;
            let factor = F256::new(z[one], z[zero]) + a + F256::new(z[bs], z[bs + 1]);
            let product = pr * factor;
            z[c] = pr.c0 * factor.c0;
            z[c + 1] = pr.c1 * factor.c1;
            z[c + 2] = (pr.c0 + pr.c1) * (factor.c0 + factor.c1);
            z[c + 3] = product.c0;
            z[c + 4] = product.c1;
            pr = product;
            c += 5;
        }
        outputs.extend_from_slice(&[pr.c0, pr.c1]);
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

/// Extension-field multiply-accumulate `out = acc + x*y`.
struct MacGate256 {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

impl MacGate256 {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let mut b = ElementTableBuilder::new(4);
        for w in 0..6 {
            b.free_wire(w);
        }
        build_mac256(&mut b, 6, Some([0, 1]), [2, 3], [4, 5]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("extension mac gate")),
        }
    }
}

impl GateType for MacGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..6).map(IoWord::input).collect();
        schema.push(IoWord::output(9));
        schema.push(IoWord::output(10));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let acc = F256::new(inputs[0], inputs[1]);
        let x = F256::new(inputs[2], inputs[3]);
        let y = F256::new(inputs[4], inputs[5]);
        let product = x * y;
        let out = acc + product;
        let mut z = vec![F128::ZERO; 11];
        z[..6].copy_from_slice(&inputs[..6]);
        z[6] = x.c0 * y.c0;
        z[7] = x.c1 * y.c1;
        z[8] = (x.c0 + x.c1) * (y.c0 + y.c1);
        z[9] = out.c0;
        z[10] = out.c1;
        outputs.extend_from_slice(&[out.c0, out.c1]);
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

fn emit_mac256(
    sb: &mut ShapeBuilder,
    slot: flock_core::circuit::builder::SlotId,
    acc: [Wire; 2],
    x: [Wire; 2],
    y: [Wire; 2],
) -> [Wire; 2] {
    let out = sb.gate(slot, &[acc[0], acc[1], x[0], x[1], y[0], y[1]]);
    [out[0], out[1]]
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

/// One packed-direct claim on the tape: its absorbed VALUE and gamma. The
/// POINT is not on the stream since merged-open v1 — it is transcript-derived
/// (gathers: the GKR's ρ_row + constant address bits; element claims: the
/// region PIOP's own challenges + the frozen prefix), and consumers rebuild
/// it from those wires and the verifier's native claims.
struct PdRec {
    val_v: usize,
    fin: usize,
    ch: usize,
    /// Word offset inside the vector squeeze shared by all batch coefficients.
    squeeze_offset: usize,
}

#[inline]
fn squeeze_word_wire(
    outs: &[Vec<Wire>],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    fin: usize,
    offset: usize,
) -> Wire {
    let (row, word) = trace.squeeze_words[fin][offset];
    outs[row][word]
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
    /// Additional L0 OOD claims batched into the initial sumcheck before its
    /// first message. Nonempty only on level 0.
    initial_ood: Vec<InitialOodRec>,
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

/// An L0 OOD claim has no separate intro quadratic: its equality basis and
/// target are combined before the initial sumcheck message is emitted.
struct InitialOodRec {
    z_fin: usize,
    z_ch: usize,
    z_len: usize,
    y_v: usize,
    beta_fin: usize,
    beta_ch: usize,
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

        fn expect_obs_f256(&mut self) {
            assert!(
                matches!(self.ops[self.i], Op::ObserveSlice(2)),
                "op {}: expected ObserveSlice(2), got {:?}",
                self.i,
                self.ops[self.i]
            );
            self.bump();
        }

        /// PoW finalizes the transcript and absorbs a nonce but creates no
        /// field challenge or scalar message.  PIOP locators call this before
        /// every protected squeeze; the generic circuit relation constrains
        /// the skipped operation separately.
        fn skip_pows(&mut self) {
            while matches!(self.ops[self.i], Op::Pow { .. }) {
                self.bump();
            }
        }
    }

    // The opening protocol begins at its domain label. Cap byte lengths are
    // not structural delimiters: under strict profiles a later recursive cap
    // can have the same length as L0, so a last-matching-length heuristic can
    // enter the tape at the wrong level.
    let label = ops
        .iter()
        .position(
            |o| matches!(o, Op::Label(l) if l.as_slice() == b"flock-ligerito-basis-f256-split-v0"),
        )
        .expect("Ligerito opening label");
    assert!(
        matches!(ops.get(label + 1), Some(Op::ObserveScalar)),
        "opening target"
    );
    let start = label + 2;
    assert!(
        matches!(ops.get(start), Some(Op::ObserveBytes(n)) if *n == cap0_bytes),
        "L0 cap"
    );
    // The merged intake runs every ring switch, absorbs every packed-direct
    // value, then protects and samples ONE coefficient vector in claim order
    // (RS first, PD second).  Consequently every PD coefficient below names
    // both a challenge ordinal and a word offset in that shared finalization.
    let mut cur = Cur {
        ops,
        i: 0,
        fin: 0,
        ch: 0,
        v: 0,
    };
    let mut gammas: Vec<PdRec> = Vec::new();
    let mut rounds: Vec<RoundRec> = Vec::new();
    let mut mp: Option<MpRec> = None;
    let mut inner_pd: Option<InnerPd> = None;
    let mut piop: Option<PiopRec> = None;
    let mut in_pd = false;
    let mut intake_rs = 0usize;
    let mut intake_pd_vals: Vec<usize> = Vec::new();
    while cur.i < start {
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-zc-v0") {
            cur.bump();
            cur.skip_pows();
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
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "zc rho");
                zc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
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
            cur.skip_pows();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "alpha");
            let (alpha_fin, alpha_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut lc_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "lc rho");
                lc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
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
            intake_rs = 0;
            intake_pd_vals.clear();
            cur.bump();
            continue;
        }
        if in_pd {
            // Ring-switched claims front the intake on boolean-bearing
            // tapes: [label, s_hat_v slice, r_dprime slice] each, then the
            // bare gamma squeezes — walk over them (mvp9 pins them
            // separately).
            if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
                intake_rs += 1;
                cur.bump(); // label
                cur.bump(); // s_hat_v slice
                cur.skip_pows();
                cur.bump(); // r_dprime slice
                continue;
            }
            if matches!(ops[cur.i], Op::Pow { .. }) {
                // A ring-switch or packed-direct batch-coefficient witness.
                cur.bump();
                continue;
            }
            if matches!(ops[cur.i], Op::ObserveScalar) {
                intake_pd_vals.push(cur.v);
                cur.expect_obs_scalar();
                continue;
            }
            if let Op::SqueezeSlice(n) = ops[cur.i] {
                assert_eq!(
                    n,
                    intake_rs + intake_pd_vals.len(),
                    "one coefficient per merged claim"
                );
                gammas.extend(intake_pd_vals.iter().enumerate().map(|(j, &val_v)| PdRec {
                    val_v,
                    fin: cur.fin,
                    ch: cur.ch + intake_rs + j,
                    squeeze_offset: intake_rs + j,
                }));
                cur.bump();
            } else {
                panic!("merged batching vector, got {:?}", ops[cur.i]);
            }
            in_pd = false;
            // The merged W-rounds follow the intake immediately: one
            // [ObserveScalar x2, SqueezeScalar] triplet per dense variable,
            // running until the multipoint label — count-free, so boolean
            // tapes (no packed-direct claims) parse identically.
            while matches!(ops[cur.i], Op::ObserveScalar)
                && matches!(ops[cur.i + 1], Op::ObserveScalar)
            {
                let mut squeeze_i = cur.i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
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
            cur.skip_pows();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint gamma");
            let (gamma_fin, gamma_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut mp_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
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
                cur.skip_pows();
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
            assert!(
                matches!(ops[cur.i], Op::SqueezeSlice(1)),
                "inner gamma vector"
            );
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
    let mut initial_ood = Vec::new();
    while matches!(cur.ops[cur.i], Op::SqueezeSlice(_)) {
        let z_len = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => n,
            _ => unreachable!(),
        };
        let (z_fin, z_ch) = (cur.fin, cur.ch);
        cur.bump();
        let y_v = cur.v;
        cur.expect_obs_scalar();
        cur.skip_pows();
        assert!(
            matches!(cur.ops[cur.i], Op::SqueezeScalar),
            "L0 OOD beta at op {}: context {:?}",
            cur.i,
            &cur.ops[cur.i.saturating_sub(4)..(cur.i + 4).min(cur.ops.len())]
        );
        initial_ood.push(InitialOodRec {
            z_fin,
            z_ch,
            z_len,
            y_v,
            beta_fin: cur.fin,
            beta_ch: cur.ch,
        });
        cur.bump();
    }
    let start_v = cur.v;
    cur.expect_obs_f256(); // sumcheck start msg u_0
    cur.expect_obs_f256(); // ... u_2

    let mut levels = Vec::new();
    let mut yr_v = 0usize;
    for li in 0..=r {
        // Fold batch: one double-width squeeze and two F256 message absorbs
        // per round. Fold grinding is zero in the F256 protocol.
        let mut fold_fins = Vec::new();
        let mut fold_chs = Vec::new();
        let mut fold_msg_vs = Vec::new();
        loop {
            match cur.ops[cur.i] {
                Op::Pow { .. } if matches!(cur.ops.get(cur.i + 1), Some(Op::SqueezeSlice(2))) => {
                    cur.bump()
                }
                Op::SqueezeSlice(2) => {
                    fold_fins.push(cur.fin);
                    fold_chs.push(cur.ch);
                    cur.bump();
                    fold_msg_vs.push(cur.v);
                    cur.expect_obs_f256();
                    cur.expect_obs_f256();
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
                cur.expect_obs_scalar(); // base-field y
                let intro_v = cur.v;
                cur.expect_obs_f256(); // intro u_0
                cur.expect_obs_f256(); // intro u_2
                cur.skip_pows();
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
                cur.expect_obs_scalar();
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
        cur.skip_pows();
        let (a_fin, a_ch, a_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected alpha squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let intro_v = cur.v;
        if li < r {
            cur.expect_obs_f256(); // intro u_0
            cur.expect_obs_f256(); // intro u_2
        }
        cur.skip_pows();
        assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "beta");
        let (beta_fin, beta_ch) = (cur.fin, cur.ch);
        cur.bump();
        levels.push(OpenLevel {
            initial_ood: if li == 0 {
                std::mem::take(&mut initial_ood)
            } else {
                Vec::new()
            },
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
    (start_v, piop, gammas, rounds, mp, inner_pd, yr_v, levels)
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

use crate::r1cs_hashes::merkle_glue::{
    BitSpreadInput, BitSpreadTable, FamilyTransposeTileInput, FamilyTransposeTileTable,
    PowMaskInput, PowMaskTable, SwapInput, SwapTable,
};

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
    type Row = BitSpreadInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.ty.build_block_r1cs(self.nu))
            .with_io_schema(self.ty.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> BitSpreadInput {
        let raw = |i: usize| (inputs[i].lo as u128) | ((inputs[i].hi as u128) << 64);
        let word = raw(0);
        let zero_mask = raw(1);
        debug_assert_eq!(inputs[2], F128::ZERO);
        let position_mask = raw(3);
        let position_prefix = raw(4);
        outputs.extend((0..self.ty.depth).map(|l| F128::new(((word >> l) & 1) as u64, 0)));
        outputs.push(F128::new(
            ((word & position_mask) ^ position_prefix) as u64,
            (((word & position_mask) ^ position_prefix) >> 64) as u64,
        ));
        BitSpreadInput {
            word,
            zero_mask,
            position_mask,
            position_prefix,
        }
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The fused PoW mask row: predicate prefix + nonce width in ONE 4-word
/// row — see [`PowMaskTable`] for the layout and the repurposed-high-half
/// trick that makes it fit 512 bits.
struct PowMaskGate {
    nu: usize,
}

/// One wired 8x8 tile of the family-H tensor-algebra transpose.  The boolean
/// relation binds the tile selector as well as all eight source and output
/// words; the element layer only has to accumulate the resulting partial dot
/// products.
struct FamilyTransposeTileGate {
    nu: usize,
}

impl GateType for FamilyTransposeTileGate {
    type Row = FamilyTransposeTileInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&FamilyTransposeTileTable::build_block_r1cs(self.nu))
            .with_io_schema(FamilyTransposeTileTable::io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let rows: [F128; 8] = inputs[..8].try_into().expect("eight transpose rows");
        debug_assert_eq!(inputs[8].hi, 0, "the tile selector fits one byte");
        debug_assert_eq!(inputs[8].lo >> 8, 0, "the tile selector fits one byte");
        let row = FamilyTransposeTileInput {
            rows,
            selector: inputs[8].lo as u8,
        };
        outputs.extend_from_slice(&FamilyTransposeTileTable::outputs(&row));
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

impl GateType for PowMaskGate {
    type Row = PowMaskInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&PowMaskTable.build_block_r1cs(self.nu))
            .with_io_schema(PowMaskTable.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), _outputs: &mut Vec<F128>) -> PowMaskInput {
        let w = |i: usize| (inputs[i].lo as u128) | ((inputs[i].hi as u128) << 64);
        debug_assert_eq!(inputs[3], F128::new(0, 1u64 << 63));
        PowMaskInput {
            pred: w(0),
            nonce: w(1),
            mask: w(2),
        }
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The three slots a collapsed opening writes into, plus the fused PoW mask
/// slot the grinding checks ride: one 4-word [`PowMaskTable`] row carries a
/// whole check (prefix mask AND nonce width) — on the deep Merkle-index
/// slot the same check paid two 16-word rows and two bit relocations.
#[derive(Clone, Copy)]
struct CollapsedSlots {
    b3: flock_core::circuit::builder::SlotId,
    /// A second identical compression table for recursion shapes whose two
    /// independent child-verifier workloads each fit a smaller row domain.
    /// The Slim envelope and strict Fast nodes use distinct slots; smaller
    /// standalone and compatibility-profile circuits retain one slot.
    b3_alt: Option<flock_core::circuit::builder::SlotId>,
    swap: flock_core::circuit::builder::SlotId,
    spread: flock_core::circuit::builder::SlotId,
    pow: flock_core::circuit::builder::SlotId,
    /// Present on recursive verifier circuits.  Smaller query-only fixtures
    /// do not declare the family-H transpose relation.
    family: Option<flock_core::circuit::builder::SlotId>,
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
    /// Number of committed F128 words. Recursive codewords are also base
    /// field rows; their extra coordinate bit is included in `folds`.
    raw_row_words: usize,
    /// The stratified schedule this level's config mandates. Every
    /// consumer (emit, residual, checker) maps query → (stratum depth,
    /// stratum, path slice) through this.
    sched: flock_core::pcs::stratified::LevelSchedule,
    /// The tree's layers from the cap upward, folded natively by
    /// `level_geometry`: entry `i` is the depth-`(c − i)` layer, entry 0
    /// the cap itself — [`Self::full_path`]'s sibling sources.
    cap_layers: Vec<Vec<[u8; 32]>>,
}

/// Map actual sumcheck-fold order back to the transcript point's natural
/// coordinate order. Partial L0 lane grids bind the high lane coordinates
/// first; full grids and every later level use ordinary low-to-high order.
fn l0_ood_z_index(
    z_len: usize,
    initial_k: usize,
    committed_row_words: usize,
    fold_order: usize,
) -> usize {
    if committed_row_words == 1usize << initial_k {
        fold_order
    } else {
        let log_msg_cols = z_len - initial_k;
        if fold_order < initial_k {
            log_msg_cols + fold_order
        } else {
            fold_order - initial_k
        }
    }
}

impl Lvl {
    /// Query `k`'s (terminal depth, stratum index).
    fn q_stratum(&self, k: usize) -> (usize, usize) {
        self.sched
            .query_strata()
            .nth(k)
            .expect("query index within schedule")
    }

    /// Query `k`'s PROOF siblings as a range into the level's flat path
    /// vec — uniformly `d − c` per query since paths truncate at the cap.
    /// The climb to a shallower summand's stratum terminal needs `c − c_k`
    /// more siblings, all folds of the cap: [`Self::full_path`] synthesizes
    /// them.
    fn path_range(&self, k: usize) -> std::ops::Range<usize> {
        let len = self.depth - self.c;
        k * len..(k + 1) * len
    }

    /// Query `k`'s FULL climb siblings — the truncated proof slice extended
    /// up to its stratum terminal at depth `c_k` with the cap-fold siblings
    /// the proof no longer carries (`self.cap_layers`, folded natively by
    /// `level_geometry`). Advice either way: the synthesized entries feed
    /// the same hint stream, and the constant-stratum terminal connect is
    /// what binds the climb.
    fn full_path(&self, k: usize, pos: usize, paths: &[[u8; 32]]) -> Vec<[u8; 32]> {
        let (ck, _) = self.q_stratum(k);
        let mut sibs: Vec<[u8; 32]> = paths[self.path_range(k)].to_vec();
        // Path entry j is the sibling at depth d − j; the proof stops at
        // the cap (j < d − c), the tail (depths c down to c_k + 1) folds
        // out of the cap. Its indices carry SQUEEZED bits below the
        // stratum, so this is witgen data, never circuit wiring.
        for j in (self.depth - self.c)..(self.depth - ck) {
            let m = self.depth - j;
            sibs.push(self.cap_layers[self.c - m][(pos >> j) ^ 1]);
        }
        sibs
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

/// Exact BLAKE compression rows emitted by [`emit_query_phase`]. Each level
/// materializes only the cap layers down to its shallowest configured
/// stratum. A query hashes its committed row, climbs to that stratum, and
/// top-stratum queries take one additional edge so the opening binds to a
/// derived cap-layer node without creating a transcript cycle.
fn level_query_phase_b3_rows(g: &Lvl) -> (usize, usize, usize) {
    let c_min = g.sched.summand_depths.last().copied().unwrap_or(g.c);
    let n_layers = (g.c - c_min).max(1);
    let cap_rows = (1..=n_layers).map(|j| 1usize << (g.c - j)).sum();
    let leaf_rows = g.raw_row_words.div_ceil(4) * g.q;
    let path_rows = (0..g.q)
        .map(|k| {
            let (ck, _) = g.q_stratum(k);
            (g.depth - ck) + usize::from(ck == g.c)
        })
        .sum();
    (leaf_rows, path_rows, cap_rows)
}

fn query_phase_b3_rows(geo: &[Lvl]) -> usize {
    geo.iter()
        .map(|g| {
            let (leaf, path, cap) = level_query_phase_b3_rows(g);
            leaf + path + cap
        })
        .sum()
}

/// Place `extra` identical-relation rows on two existing slots as evenly as
/// their current loads permit. Returns `(extra_on_a, resulting_max_load)`.
fn balance_extra_rows(a: usize, b: usize, extra: usize) -> (usize, usize) {
    let target = (a + b + extra).div_ceil(2).max(a).max(b);
    let on_a = target.saturating_sub(a).min(extra);
    (on_a, (a + on_a).max(b + extra - on_a))
}

/// The tree's layers from the cap upward, natively: entry `i` is the
/// depth-`(c − i)` layer, entry 0 the cap itself — the sibling sources for
/// [`Lvl::full_path`]'s synthesized tail. `n_layers` is clamped to at
/// least 1 so entry 0 exists even for a single-summand schedule (which
/// never indexes past it).
fn native_cap_layers(cap: &[[u8; 32]], n_layers: usize, hash: HashKind) -> Vec<Vec<[u8; 32]>> {
    let mut layers: Vec<Vec<[u8; 32]>> = vec![cap.to_vec()];
    for _ in 1..n_layers.max(1) {
        let next: Vec<[u8; 32]> = layers
            .last()
            .unwrap()
            .chunks_exact(2)
            .map(|p| core_merkle::hash_pair(&p[0], &p[1], hash))
            .collect();
        layers.push(next);
    }
    layers
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
) -> (Vec<Lvl>, Vec<F256>) {
    use flock_core::lincheck::build_eq_table;
    assert_eq!(scheds.len(), levels.len(), "one schedule per open level");
    let mut geo: Vec<Lvl> = Vec::new();
    let mut native_sums: Vec<F256> = Vec::new();
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
        let raw_row_words = rows[0].len();
        let row_words = raw_row_words;
        assert!(
            row_words >= 1 && row_words <= lanes,
            "L{li}: opened width {row_words} must fit the fold width {lanes}"
        );
        let fold_vals: Vec<F256> = lvl
            .fold_chs
            .iter()
            .map(|&i| F256::new(chals[i], chals[i + 1]))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let mut eqv = vec![F256::ONE];
        for &v in &fold_vals {
            let old = eqv.len();
            eqv.resize(2 * old, F256::ZERO);
            for i in 0..old {
                let x = eqv[i];
                eqv[i + old] = x * v;
                eqv[i] = x * (F256::ONE + v);
            }
        }
        let aw = build_eq_table(&alpha_vals);
        let c_min = sched.summand_depths.last().copied().unwrap_or(c);
        let lv = Lvl {
            q,
            c,
            depth,
            lanes,
            row_words,
            raw_row_words,
            sched: sched.clone(),
            cap_layers: native_cap_layers(cap, c - c_min, hash),
        };
        // Paths truncate at the cap, so every query verifies directly
        // against the absorbed layer — no terminal-layer rebuild; the
        // stratum needs no enforcement because `q_pos` derives the index
        // itself with the stratum in the top bits.
        let mut sum = F256::ZERO;
        for (k, row) in rows.iter().enumerate() {
            let pos = lv.q_pos(k, chals[lvl.q_ch + k].lo);
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
                    &paths[lv.path_range(k)],
                    hash,
                ),
                "L{li} query {k}: capped path verifies natively"
            );
            let dot = row
                .iter()
                .zip(eqv.iter())
                .map(|(&x, &e)| F256::from(x) * e)
                .fold(F256::ZERO, |a, v| a + v);
            sum += aw[k] * dot;
        }
        native_sums.push(sum);
        geo.push(lv);
    }
    (geo, native_sums)
}

fn replay_ligerito_spine256(
    levels: &[OpenLevel],
    values: &[F128],
    challenges: &[F128],
    start_v: usize,
    initial_target: F128,
    enforced_sums: &[F256],
) -> F256 {
    let msg = |at: usize| {
        (
            F256::new(values[at], values[at + 1]),
            F256::new(values[at + 2], values[at + 3]),
        )
    };
    let quad = |at: usize, target: F256| {
        let (u0, u2) = msg(at);
        (u0, target + u2, u2)
    };
    let eval = |q: (F256, F256, F256), r: F256| q.0 + r * q.1 + r * r * q.2;

    let mut target = F256::from(initial_target);
    for od in &levels[0].initial_ood {
        target += F256::from(challenges[od.beta_ch] * values[od.y_v]);
    }
    let mut q = quad(start_v, target);
    for (li, level) in levels.iter().enumerate() {
        for (j, &mv) in level.fold_msg_vs.iter().enumerate() {
            let ch = level.fold_chs[j];
            target = eval(q, F256::new(challenges[ch], challenges[ch + 1]));
            q = quad(mv, target);
        }
        if li + 1 < levels.len() {
            for od in &level.ood {
                let y = F256::from(values[od.y_v]);
                let iq = quad(od.intro_v, y);
                let beta = challenges[od.beta_ch];
                q.0 += iq.0 * beta;
                q.1 += iq.1 * beta;
                q.2 += iq.2 * beta;
                target += y * beta;
            }
            let iq = quad(level.intro_v, enforced_sums[li]);
            let beta = challenges[level.beta_ch];
            q.0 += iq.0 * beta;
            q.1 += iq.1 * beta;
            q.2 += iq.2 * beta;
            target += enforced_sums[li] * beta;
        } else {
            target += enforced_sums[li] * challenges[level.beta_ch];
        }
    }
    target
}

fn observed_f256(values: &[F128], start: usize, len: usize) -> Vec<F256> {
    (0..len)
        .map(|i| F256::new(values[start + 2 * i], values[start + 2 * i + 1]))
        .collect()
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
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    outs: &[Vec<Wire>],
    chals: &[F128],
    cap_w: &[Vec<[Wire; 2]>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
) -> (Vec<Vec<Wire>>, Vec<[Wire; 2]>, Vec<Vec<Wire>>) {
    use flock_core::lincheck::build_eq_table;
    let mut to_publish: Vec<Vec<Wire>> = Vec::new();
    let mut level_accs: Vec<[Wire; 2]> = Vec::new();
    let mut query_positions: Vec<Vec<Wire>> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let g = &geo[li];
        let (_cap, rows, paths) = lvl_src[li];
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
        let a_wires: Vec<Wire> = (0..lvl.a_count)
            .map(|j| squeeze_word_wire(outs, trace, lvl.a_fin, j))
            .collect();
        // v: this level's fold challenges, chain outputs, wired straight in.
        let v_wires: Vec<[Wire; 2]> = lvl
            .fold_fins
            .iter()
            .map(|&f| {
                [
                    squeeze_word_wire(outs, trace, f, 0),
                    squeeze_word_wire(outs, trace, f, 1),
                ]
            })
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        // The hi-group weights of the leaf-eval split: eq over the native
        // values of the fold challenges past the 8-lane gate's three.
        let le_vars = g.lanes.min(8).trailing_zeros() as usize;
        let le_groups = g.lanes >> le_vars;
        let hw = {
            let v_hi: Vec<F256> = lvl.fold_chs[le_vars..]
                .iter()
                .map(|&i| F256::new(chals[i], chals[i + 1]))
                .collect();
            let mut table = vec![F256::ONE];
            for r in v_hi {
                let old = table.len();
                table.resize(2 * old, F256::ZERO);
                for i in 0..old {
                    let x = table[i];
                    table[i + old] = x * r;
                    table[i] = x * (F256::ONE + r);
                }
            }
            table
        };
        let zero = cw(sb, vals, consts, F128::ZERO);
        let mut acc = [zero, zero];
        let mut level_positions = Vec::with_capacity(g.q);
        // Zero wire for the fold's known-zero top lanes (only declared when
        // the committed row is narrower than the fold).
        let pad_w = if g.row_words < g.lanes {
            Some([zero, zero])
        } else {
            None
        };
        for k in 0..g.q {
            vals.extend_from_slice(&rows[k]);
            let leaf_w: Vec<Wire> = (0..g.raw_row_words).map(|_| sb.input()).collect();
            let cw = squeeze_word_wire(outs, trace, lvl.q_fin, k);
            let (ck, stratum) = g.q_stratum(k);
            let open_depth = g.depth - ck;
            let (cv, position_w) = emit_opening(
                sb,
                slots,
                iv,
                &leaf_w,
                cw,
                open_depth,
                0,
                stratum << open_depth,
                Some(consts),
                vals,
            );
            level_positions.push(position_w);
            // The proof's siblings truncate at the cap; the climb to the
            // stratum terminal still runs the full `d − c_k` rows, so
            // witgen reconstitutes the cap-fold tail as extra hints.
            let pos = g.q_pos(k, chals[lvl.q_ch + k].lo);
            hints.extend(g.full_path(k, pos, paths).iter().map(hash_to_digest));
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
            let mut fold_w: Vec<[Wire; 2]> = leaf_w.iter().map(|&w| [w, zero]).collect();
            fold_w.resize(g.lanes, pad_w.unwrap_or(fold_w[0]));
            let lanes = g.lanes.min(8);
            for h in 0..le_groups {
                let mut a_in: Vec<Wire> = fold_w[lanes * h..lanes * (h + 1)]
                    .iter()
                    .flat_map(|p| *p)
                    .collect();
                a_in.extend(v_wires[..le_vars].iter().flat_map(|p| *p));
                let weight = hw[h] * aw[k];
                vals.push(weight.c0);
                vals.push(weight.c1);
                a_in.push(sb.input());
                a_in.push(sb.input());
                a_in.extend_from_slice(&acc);
                let out = sb.gate(leafeval[li], &a_in);
                acc = [out[0], out[1]];
            }
        }
        to_publish.push(a_wires);
        level_accs.push(acc);
        query_positions.push(level_positions);
    }
    (to_publish, level_accs, query_positions)
}

/// Emit the RESIDUAL region — the third shared piece of the deferred
/// verifier, after the FS chain and the query phase. Per query, the shared
/// extension-field residual gates derive the normalized `W_k` ladder once,
/// multiply the later-level challenges into a prefix three at a time, and
/// update each eight-position accumulator chunk. Together these compute
/// `induce_sumcheck_evaluate_at_residual` (the `next_s` chain from a
/// boundary-bound q_field, a prefix over the LATER levels' fold wires,
/// suffix subset products over the `2^yr` residual positions); the
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
/// **CHUNKING.** The accumulator gate has eight positions (`chunk_log=3`)
/// at kappa 6 regardless of the proof's residual size. The real inner's
/// yr = 32 would otherwise push its schema to kappa 7-8. A yr > 8 region
/// runs as `2^(yr_log-3)` chunks of 8:
/// - the close-out claims' HIGH-bit eq factors ride the PREFIX SLOT
///   (seed = the claim's prefix product, factors = high coords vs the
///   chunk bits) — wire-bound, no new trust;
/// - the residual rows' high subset factor `sp_hi(h)` rides the CHECKER
///   tier (`awp = aw·sp_hi`, recomputed natively from the validated
///   position by `check_residual_publics` — the alpha-expansion trust
///   class; a wrong value fails the published accumulators).
/// The smallest supported residual domain, yr = 8, takes one chunk.
/// The close-out itself (per-position eq tensors, the beta combines, the
/// yr dot) is prefix + MacGate rows since Round 3 — no dedicated types.
#[allow(clippy::too_many_arguments)]
fn emit_residual_region(
    sb: &mut ShapeBuilder,
    leaf_slot: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    levels: &[OpenLevel],
    geo: &[Lvl],
    alpha_wires: &[Vec<Wire>],
    query_positions: &[Vec<Wire>],
    w_rounds: &[RoundRec],
    inner_pd_fin: usize,
    yr_wires: &[[Wire; 2]],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    outs: &[Vec<Wire>],
    zw: Wire,
    ow: Wire,
) -> (
    Vec<Vec<[Wire; 2]>>,
    [Wire; 2],
    (flock_core::circuit::builder::SlotId, usize),
) {
    let yr_len = yr_wires.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let chunk_log = yr_log.min(3);
    assert_eq!(
        chunk_log, 3,
        "the shared F256 residual gates operate on eight-position chunks"
    );
    let chunk = 1usize << chunk_log;
    let n_chunks = 1usize << (yr_log - chunk_log);
    let chw = |fin: usize| -> [Wire; 2] {
        [
            squeeze_word_wire(outs, trace, fin, 0),
            squeeze_word_wire(outs, trace, fin, 1),
        ]
    };
    let base_chw = |fin: usize| -> Wire { squeeze_word_wire(outs, trace, fin, 0) };
    let mut resid_pub: Vec<Vec<[Wire; 2]>> = Vec::new();
    let weights_slot = slot_cached(sb, leaf_slot, 880, ResidualWeightsGate256::new);
    let prefix3_slot = slot_cached(sb, leaf_slot, 881, ResidualPrefix3Gate256::new);
    let acc_slot = slot_cached(sb, leaf_slot, 882, ResidualAccGate256::new);
    let scalar_macs = slot_cached(sb, leaf_slot, 600, MacGate::new);
    assert_eq!(alpha_wires.len(), levels.len());
    assert_eq!(query_positions.len(), levels.len());
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len() - 1).sum();
        // The deepest weight this level touches is `weights[pl + yr_log - 1]`
        // (the chunk-high extension below), not `pl + chunk_log`: the old
        // bound under-checked whenever the residual domain has more than one
        // 8-chunk, so the m32 walk overran the gate instead of failing here.
        let lmc = pl + yr_log;
        assert!(
            lmc <= ResidualWeightsGate256::N_WEIGHTS,
            "the residual ladder needs {lmc} normalized weights"
        );
        assert_eq!(pl % 3, 0, "one Ligerito level contributes three folds");
        let ris_w: Vec<[Wire; 2]> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f)))
            .collect();
        assert_eq!(alpha_wires[li].len(), lvl.a_count);
        assert_eq!(query_positions[li].len(), geo[li].q);
        // Expand eq(alpha, k) from the transcript challenge wires. For each
        // prior weight x and next coordinate r, the low/high children are
        // x(1+r) and xr. Both are MacGate rows, so no alpha-derived advice
        // enters the residual relation.
        let mut aw = vec![ow];
        for &r in &alpha_wires[li] {
            let old = aw;
            let mut next = Vec::with_capacity(2 * old.len());
            for &x in &old {
                next.push(sb.gate(scalar_macs, &[x, x, r])[0]);
            }
            for &x in &old {
                next.push(sb.gate(scalar_macs, &[zw, x, r])[0]);
            }
            aw = next;
        }
        assert!(aw.len() >= geo[li].q, "alpha tensor covers every query");
        let mut accs: Vec<[Wire; 2]> = (0..yr_len).map(|_| [zw, zw]).collect();
        for k in 0..geo[li].q {
            let qf = query_positions[li][k];
            let w_tail = sb.gate(weights_slot, &[qf, ow]);
            let mut weights = Vec::with_capacity(ResidualWeightsGate256::N_WEIGHTS);
            weights.push(qf);
            weights.extend(w_tail);
            let mut prefix = [ow, zw];
            for at in (0..pl).step_by(3) {
                let mut g_in = vec![prefix[0], prefix[1]];
                g_in.extend(ris_w[at..at + 3].iter().flat_map(|p| *p));
                g_in.extend_from_slice(&weights[at..at + 3]);
                g_in.push(ow);
                let out = sb.gate(prefix3_slot, &g_in);
                prefix = [out[0], out[1]];
            }
            let low_weights = &weights[pl..pl + 3];
            for h in 0..n_chunks {
                // Extend the transcript-derived alpha weight by the high
                // residual subset selected by this chunk. The relevant W_j
                // values are outputs of ResidualWeightsGate256, so this
                // replaces the former free aw*sp_hi advice.
                let mut awp = aw[k];
                for j in 0..(yr_log - chunk_log) {
                    if (h >> j) & 1 == 1 {
                        awp = sb.gate(scalar_macs, &[zw, awp, weights[pl + chunk_log + j]])[0];
                    }
                }
                let mut g_in = vec![awp, prefix[0], prefix[1]];
                g_in.extend_from_slice(low_weights);
                g_in.extend(accs[h * chunk..(h + 1) * chunk].iter().flat_map(|p| *p));
                let out = sb.gate(acc_slot, &g_in);
                for (dst, src) in accs[h * chunk..(h + 1) * chunk]
                    .iter_mut()
                    .zip(out.chunks_exact(2))
                {
                    *dst = [src[0], src[1]];
                }
            }
        }
        resid_pub.push(accs);
    }
    // The close-out. The ligerito layer sees ONE packed-direct claim:
    // (rho, q_eval) with gamma'; rho's coords are the W-round squeezes —
    // chain wires. The OOD claims are the same shape, seed = beta, point =
    // the squeezed z.
    let total_fold_count: usize = levels.iter().map(|l| l.fold_fins.len()).sum();
    let pl_full = levels[0].fold_fins.len()
        + levels[1..]
            .iter()
            .map(|l| l.fold_fins.len() - 1)
            .sum::<usize>();
    let ris_full: Vec<[Wire; 2]> = levels[0]
        .fold_fins
        .iter()
        .map(|&f| chw(f))
        .chain(
            levels[1..]
                .iter()
                .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f))),
        )
        .collect();
    // ROUND 3: the close-out's suffix/combine/dot arithmetic rides the
    // shared 4-word MacGate (cache key 600, the mvp8 convention) plus the
    // prefix slot — the SuffixGate/PartialCombineGate/FinalDotGate types
    // are DISSOLVED: 51 schema words (each a cell slot AND a gather claim)
    // bought ~30 rows of work; as mac/prefix rows the same work is ~250
    // live-prefix-cheap rows and zero types.
    let macs = match leaf_slot.iter().find(|&&(k, _)| k == 701) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(MacGate256::new());
            leaf_slot.push((701, s));
            s
        }
    };
    let pf_w = total_fold_count.min(8);
    let pfslot = match leaf_slot.iter().find(|&&(k, _)| k == 1000 + pf_w) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PrefixGate256::new(pf_w));
            leaf_slot.push((1000 + pf_w, s));
            s
        }
    };
    // Other deferred-verifier arithmetic is base-field valued and reuses
    // the original prefix type. Return that slot to the caller.
    let base_pfslot = match leaf_slot.iter().find(|&&(k, _)| k == 310 + pf_w) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PrefixGate::new(pf_w));
            leaf_slot.push((310 + pf_w, s));
            s
        }
    };
    // Seed-chained prefix product: any factor list, `pf_w` per row.
    let prefix_chain =
        |sb: &mut ShapeBuilder, seed: [Wire; 2], factors: &[([Wire; 2], [Wire; 2])]| -> [Wire; 2] {
            let mut s = seed;
            for chunk_f in factors.chunks(pf_w) {
                let mut g_in = vec![s[0], s[1]];
                for (a, _) in chunk_f {
                    g_in.extend_from_slice(a);
                }
                g_in.extend(std::iter::repeat_n(zw, 2 * (pf_w - chunk_f.len())));
                for (_, b) in chunk_f {
                    g_in.extend_from_slice(b);
                }
                g_in.extend(std::iter::repeat_n(zw, 2 * (pf_w - chunk_f.len())));
                g_in.push(ow);
                g_in.push(zw);
                let out = sb.gate(pfslot, &g_in);
                s = [out[0], out[1]];
            }
            s
        };
    // A split commitment adds one coordinate variable per recursive level.
    // Folding that bit at r contributes phi(r) = 1 + r(1 + u). Express it
    // as the prefix factor 1 + r + r*u so the same F256 product gate binds
    // the coordinate transport used by the native verifier.
    let coordinate_factors = |sb: &mut ShapeBuilder, start_level: usize| {
        levels[start_level.max(1)..]
            .iter()
            .map(|level| {
                let r = chw(level.fold_fins[0]);
                let ru = emit_mac256(sb, macs, [zw, zw], r, [zw, ow]);
                (r, ru)
            })
            .collect::<Vec<_>>()
    };
    let mut evb_accs: Vec<[Wire; 2]> = (0..yr_len).map(|_| [zw, zw]).collect();
    // Fold one claim (prefix product p at full-yl coord wires) into the
    // accumulators: per position, ONE prefix row computes p·eq(coords, y)
    // (high bits chunk-shared, low bits per position; eq factor =
    // 1 + coord + [bit] in char 2) and ONE MacGate row accumulates it.
    let apply_suffix =
        |sb: &mut ShapeBuilder, evb_accs: &mut [[Wire; 2]], p: [Wire; 2], coords: &[[Wire; 2]]| {
            assert_eq!(coords.len(), yr_log, "the claim tail spans yr");
            for h in 0..n_chunks {
                let ph = if n_chunks == 1 {
                    p
                } else {
                    let factors: Vec<([Wire; 2], [Wire; 2])> = coords[chunk_log..]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, [if (h >> j) & 1 == 1 { ow } else { zw }, zw]))
                        .collect();
                    prefix_chain(sb, p, &factors)
                };
                for y in 0..chunk {
                    let factors: Vec<([Wire; 2], [Wire; 2])> = coords[..chunk_log]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, [if (y >> j) & 1 == 1 { ow } else { zw }, zw]))
                        .collect();
                    let py = prefix_chain(sb, ph, &factors);
                    let at2 = h * chunk + y;
                    evb_accs[at2] = emit_mac256(sb, macs, evb_accs[at2], py, [ow, zw]);
                }
            }
        };
    {
        assert_eq!(
            w_rounds.len(),
            pl_full + yr_log,
            "rho spans the dense domain"
        );
        let mut factors: Vec<([Wire; 2], [Wire; 2])> = w_rounds[..pl_full]
            .iter()
            .map(|rr| [base_chw(rr.fin), zw])
            .zip(ris_full.iter().copied())
            .collect();
        factors.extend(coordinate_factors(sb, 0));
        let pw = prefix_chain(sb, [base_chw(inner_pd_fin), zw], &factors);
        let coords: Vec<[Wire; 2]> = w_rounds[pl_full..]
            .iter()
            .map(|rr| [base_chw(rr.fin), zw])
            .collect();
        apply_suffix(sb, &mut evb_accs, pw, &coords);
    }
    for od in &levels[0].initial_ood {
        let folded = od.z_len - yr_log;
        assert_eq!(folded, ris_full.len(), "L0 OOD spans every fold");
        let initial_k = levels[0].fold_fins.len();
        let z_index = |j| l0_ood_z_index(od.z_len, initial_k, geo[0].row_words, j);
        let mut factors: Vec<([Wire; 2], [Wire; 2])> = (0..folded)
            .map(|j| {
                (
                    [squeeze_word_wire(outs, trace, od.z_fin, z_index(j)), zw],
                    ris_full[j],
                )
            })
            .collect();
        factors.extend(coordinate_factors(sb, 0));
        let pw = prefix_chain(sb, [base_chw(od.beta_fin), zw], &factors);
        let coords: Vec<[Wire; 2]> = (0..yr_log)
            .map(|j| {
                [
                    squeeze_word_wire(outs, trace, od.z_fin, z_index(folded + j)),
                    zw,
                ]
            })
            .collect();
        apply_suffix(sb, &mut evb_accs, pw, &coords);
    }
    for (li, lvl) in levels.iter().enumerate() {
        for od in &lvl.ood {
            let folded = od.z_len - yr_log;
            let later: Vec<[Wire; 2]> = levels[li + 1]
                .fold_fins
                .iter()
                .map(|&f| chw(f))
                .chain(
                    levels[li + 2..]
                        .iter()
                        .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f))),
                )
                .collect();
            assert_eq!(later.len(), folded, "OOD prefix = later folds");
            let mut factors: Vec<([Wire; 2], [Wire; 2])> = (0..folded)
                .map(|j| ([squeeze_word_wire(outs, trace, od.z_fin, j), zw], later[j]))
                .collect();
            factors.extend(coordinate_factors(sb, li + 2));
            let pw = prefix_chain(sb, [base_chw(od.beta_fin), zw], &factors);
            let coords: Vec<[Wire; 2]> = (0..yr_log)
                .map(|j| {
                    let jj = folded + j;
                    [squeeze_word_wire(outs, trace, od.z_fin, jj), zw]
                })
                .collect();
            apply_suffix(sb, &mut evb_accs, pw, &coords);
        }
    }
    // beta-weighted residuals fold in per level (comb_y += beta·resid_y —
    // one MacGate row each), then the yr dot as one MAC chain.
    let mut comb = evb_accs;
    for (li, lvl) in levels.iter().enumerate() {
        let coordinate = coordinate_factors(sb, li + 1);
        let beta_w = prefix_chain(sb, [base_chw(lvl.beta_fin), zw], &coordinate);
        for y in 0..yr_len {
            comb[y] = emit_mac256(sb, macs, comb[y], beta_w, resid_pub[li][y]);
        }
    }
    let mut inner_w = [zw, zw];
    for (yw, cb) in yr_wires.iter().zip(&comb) {
        inner_w = emit_mac256(sb, macs, inner_w, *yw, *cb);
    }
    (resid_pub, inner_w, (base_pfslot, pf_w))
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
    yr_vals: &[F256],
    chals: &[F128],
) -> F256 {
    use flock_core::lincheck::build_eq_table;
    let yr_len = yr_vals.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let mut at = at;
    let mut resid_native: Vec<Vec<F256>> = vec![vec![F256::ZERO; yr_len]; levels.len()];
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len() - 1).sum();
        let lmc = pl + yr_log;
        let sks = sk_at_vks(lmc);
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let ris: Vec<F256> = levels[li + 1..]
            .iter()
            .flat_map(|l| {
                l.fold_chs
                    .iter()
                    .skip(1)
                    .map(|&i| F256::new(chals[i], chals[i + 1]))
            })
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        for y in 0..yr_len {
            let mut sum = F256::ZERO;
            for k in 0..geo[li].q {
                let pos = geo[li].q_pos(k, chals[lvl.q_ch + k].lo);
                let mut sk = Vec::with_capacity(lmc);
                if lmc > 0 {
                    sk.push(F128::new(pos as u64, 0));
                    for j in 1..lmc {
                        sk.push(sk[j - 1] * sk[j - 1] + sks[j - 1] * sk[j - 1]);
                    }
                }
                let mut prod = F256::ONE;
                for j in 0..pl {
                    prod *= F256::ONE + ris[j] * (F128::ONE + sk[j] * inv(sks[j]));
                }
                for j in 0..yr_log {
                    if (y >> j) & 1 == 1 {
                        prod *= sk[pl + j] * inv(sks[pl + j]);
                    }
                }
                sum += aw[k] * prod;
            }
            assert_eq!(
                F256::new(public[at], public[at + 1]),
                sum,
                "L{li} residual y={y}"
            );
            resid_native[li][y] = sum;
            at += 2;
        }
    }
    // evb + combine, natively: gamma-weighted char-2 eq products, then the
    // yr dot.
    let ris_v: Vec<F256> = levels[0]
        .fold_chs
        .iter()
        .map(|&i| F256::new(chals[i], chals[i + 1]))
        .chain(levels[1..].iter().flat_map(|l| {
            l.fold_chs
                .iter()
                .skip(1)
                .map(|&i| F256::new(chals[i], chals[i + 1]))
        }))
        .collect();
    let pl_full = ris_v.len();
    let coordinate_scale = |start_level: usize| {
        levels[start_level.max(1)..]
            .iter()
            .fold(F256::ONE, |acc, level| {
                let at = level.fold_chs[0];
                let r = F256::new(chals[at], chals[at + 1]);
                acc * (F256::ONE + r * F256::new(F128::ONE, F128::ONE))
            })
    };
    let mut inner_n = F256::ZERO;
    for y in 0..yr_len {
        let mut evb = F256::from(chals[inner_pd_ch]);
        for j in 0..pl_full {
            evb *= F256::ONE + F256::from(chals[w_rounds[j].ch]) + ris_v[j];
        }
        for j in 0..yr_log {
            evb *= if (y >> j) & 1 == 1 {
                F256::from(chals[w_rounds[pl_full + j].ch])
            } else {
                F256::from(F128::ONE + chals[w_rounds[pl_full + j].ch])
            };
        }
        evb *= coordinate_scale(0);
        let mut comb = evb;
        for od in &levels[0].initial_ood {
            let folded = od.z_len - yr_log;
            assert_eq!(folded, pl_full, "L0 OOD spans every fold");
            let initial_k = levels[0].fold_chs.len();
            let z_index = |j| l0_ood_z_index(od.z_len, initial_k, geo[0].row_words, j);
            let mut t = F256::from(chals[od.beta_ch]);
            for j in 0..folded {
                t *= F256::ONE + F256::from(chals[od.z_ch + z_index(j)]) + ris_v[j];
            }
            for j in 0..yr_log {
                t *= if (y >> j) & 1 == 1 {
                    F256::from(chals[od.z_ch + z_index(folded + j)])
                } else {
                    F256::from(F128::ONE + chals[od.z_ch + z_index(folded + j)])
                };
            }
            t *= coordinate_scale(0);
            comb += t;
        }
        for (li, lvl) in levels.iter().enumerate() {
            comb += resid_native[li][y] * chals[lvl.beta_ch] * coordinate_scale(li + 1);
            for od in &lvl.ood {
                let folded = od.z_len - yr_log;
                let later: Vec<F256> = levels[li + 1]
                    .fold_chs
                    .iter()
                    .map(|&i| F256::new(chals[i], chals[i + 1]))
                    .chain(levels[li + 2..].iter().flat_map(|l| {
                        l.fold_chs
                            .iter()
                            .skip(1)
                            .map(|&i| F256::new(chals[i], chals[i + 1]))
                    }))
                    .collect();
                let mut t = F256::from(chals[od.beta_ch]);
                for j in 0..folded {
                    t *= F256::ONE + F256::from(chals[od.z_ch + j]) + later[j];
                }
                for j in 0..yr_log {
                    t *= if (y >> j) & 1 == 1 {
                        F256::from(chals[od.z_ch + folded + j])
                    } else {
                        F256::from(F128::ONE + chals[od.z_ch + folded + j])
                    };
                }
                t *= coordinate_scale(li + 2);
                comb += t;
            }
        }
        inner_n += yr_vals[y] * comb;
    }
    assert_eq!(
        F256::new(public[at], public[at + 1]),
        inner_n,
        "the close-out inner"
    );
    inner_n
}

/// BLAKE3 serializes its output words little-endian, while "leading bits"
/// means most-significant-bit first within each serialized byte. Return the
/// circuit-word mask whose set bits are exactly that prefix.
fn pow_leading_zero_mask(bits: u32) -> F128 {
    assert!(bits <= 128, "the fused predicate is one F128 word");
    let mut mask = 0u128;
    for k in 0..bits as usize {
        let serialized_bit = 8 * (k / 8) + (7 - k % 8);
        mask |= 1u128 << serialized_bit;
    }
    F128::new(mask as u64, (mask >> 64) as u64)
}

/// Arithmetize every grinding operation in a recorded verifier transcript.
///
/// The fused BLAKE3 row has already bound the nonce to the transcript,
/// advanced the state and produced the protected challenge. This helper adds
/// only the selected-zero relations
///
/// ```text
/// prefix_bits(predicate_word, lambda) = 0^lambda
/// nonce[64..128] = 0.
/// ```
///
/// The selected-zero equations are rows of the shared PoW-mask table, whose
/// mask and check inputs are statement constants. A zero-bit operation instead
/// enforces the canonical nonce 0.
fn emit_pow_checks(
    sb: &mut ShapeBuilder,
    _b3: flock_core::circuit::builder::SlotId,
    pow: flock_core::circuit::builder::SlotId,
    _iv: [Wire; 2],
    pows: &[([Wire; 2], u32)],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    let check_word = (!pows.is_empty()).then(|| cw(sb, vals, consts, F128::new(0, 1u64 << 63)));
    for &([predicate, nonce], bits) in pows {
        // One fused PowMask row per check: the prefix cells mask the
        // predicate and the mask word's wire-bound high half pins the
        // nonce to 64 bits — the transcript stream allocates a whole F128
        // word to the 8-byte nonce, and this is what keeps the remaining
        // eight bytes padding rather than an extra grinding knob for a
        // malicious recursive prover.
        assert!(
            bits <= 64,
            "the PowMask row's prefix cells cover the low mask half"
        );
        if bits == 0 {
            // Canonical zero nonce: the nonce rides BOTH input words — the
            // prefix cells pin its low half under the all-ones low mask,
            // the structural high-half cells pin the rest.  All 128 bits,
            // so a disabled site cannot become a grinding knob either.
            let ones = cw(sb, vals, consts, F128::new(u64::MAX, 0));
            let _ = sb.gate(
                pow,
                &[nonce, nonce, ones, check_word.expect("nonempty PoW list")],
            );
        } else {
            let mask_w = cw(sb, vals, consts, pow_leading_zero_mask(bits));
            let _ = sb.gate(
                pow,
                &[
                    predicate,
                    nonce,
                    mask_w,
                    check_word.expect("nonempty PoW list"),
                ],
            );
        }
    }
}

/// Locate fused PoW predicate and nonce wires on an arbitrary recorded tape
/// and constrain every native fused verification call in-circuit.
#[allow(clippy::too_many_arguments)]
fn emit_recorded_pow_checks(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    spread: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    ops: &[flock_core::transcript_record::TranscriptOp],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_core::transcript_record::Stream,
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    use flock_core::transcript_record::TranscriptOp as Op;
    let (mut fin, mut pay) = (0usize, 0usize);
    let mut pows = Vec::new();
    for op in ops {
        if let Op::Pow { bits } = op {
            pows.push((fin, pay, *bits));
        }
        if op.finalizes() {
            fin += 1;
        }
        if matches!(
            op,
            Op::ObserveBytes(_) | Op::Pow { .. } | Op::LegacyPow { .. }
        ) {
            pay += 1;
        }
    }
    let checks: Vec<([Wire; 2], u32)> = pows
        .into_iter()
        .map(|(fin, pay, bits)| {
            let sq = &trace.squeezes[fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pay))
                .expect("pow nonce stream word");
            (
                [outs[sq[0]][1], ww[wi].expect("pow nonce wired")],
                bits,
            )
        })
        .collect();
    emit_pow_checks(sb, b3, spread, iv, &checks, vals, consts);
}

#[test]
fn fused_pow_masks_match_raw_compression() {
    let cv: [u32; 8] = std::array::from_fn(|i| 0x1020_3040u32.wrapping_mul(i as u32 + 1));
    for pending_words in 0..4 {
        let pending_len = 16 * pending_words;
        for nonce in 0..64u64 {
            for bits in [1u32, 2, 7, 8, 9, 13, 16, 17, 31, 64, 127, 128] {
                let mut block = [0u8; 64];
                for (i, b) in block[..pending_len].iter_mut().enumerate() {
                    *b = (17 * i + 9) as u8;
                }
                block[pending_len..pending_len + 8].copy_from_slice(&nonce.to_le_bytes());
                let message: [u32; 16] = std::array::from_fn(|i| {
                    u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
                });
                let out = blake3::blake3_compress(
                    &cv,
                    &message,
                    flock_core::challenger::pow_squeeze_counter(bits, pending_len + 16),
                    64,
                    crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
                );
                let mut predicate = [0u8; 16];
                for (i, word) in out[4..8].iter().enumerate() {
                    predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
                let predicate_word = u128::from_le_bytes(predicate);
                let mask = pow_leading_zero_mask(bits);
                let mask = (mask.lo as u128) | ((mask.hi as u128) << 64);
                let circuit_accepts = predicate_word & mask == 0;
                let native_accepts =
                    (0..bits as usize).all(|k| predicate[k / 8] & (1 << (7 - k % 8)) == 0);
                assert_eq!(
                    circuit_accepts, native_accepts,
                    "pending words {pending_words}, nonce {nonce}, lambda {bits}"
                );
            }
        }
    }

    // Pin the other half of the gadget on the fused PowMask row: a
    // nonzero-bit PoW permits only a 64-bit nonce, and a zero-bit site
    // permits only the canonical nonce zero.  The prefix checks live in the
    // R1CS; the nonce-width check is the mask input word's WIRE BINDING
    // (the word must equal the statement's mask constant, whose high half
    // is zero), so the pin models both.
    let ty = PowMaskTable;
    let r1cs = ty.build_block_r1cs(0);
    let accepted = |pred: u128, nonce: u128, mask: u128| {
        let [z, _, _] = ty.build_witness(PowMaskInput { pred, nonce, mask });
        let word2 = (256..384).fold(0u128, |acc, i| acc | ((z[i] as u128) << (i - 256)));
        r1cs.satisfies(&z) && word2 == mask
    };
    // The nonce width, under a clearing predicate.
    assert!(accepted(0, 42, 0b11));
    assert!(!accepted(0, (1u128 << 100) | 42, 0b11));
    // The prefix itself.
    assert!(!accepted(0b10, 42, 0b11));
    // The canonical zero-bit shape: the nonce as both words, all-ones low mask.
    assert!(accepted(0, 0, u64::MAX as u128));
    assert!(!accepted(1, 1, u64::MAX as u128));
    assert!(!accepted(1u128 << 100, 1u128 << 100, u64::MAX as u128));
}

#[test]
fn recursive_pow_relation_accepts_valid_and_rejects_invalid_nonce() {
    let bits = 6u32;
    let mut state_digest = [0u8; 32];
    for (i, b) in state_digest.iter_mut().enumerate() {
        *b = (29 * i + 3) as u8;
    }
    let cv: [u32; 8] = std::array::from_fn(|i| {
        u32::from_le_bytes(state_digest[4 * i..4 * i + 4].try_into().unwrap())
    });
    let fused = |nonce: u64| {
        let mut block = [0u8; 64];
        block[..8].copy_from_slice(&nonce.to_le_bytes());
        let message: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        });
        blake3::blake3_compress(
            &cv,
            &message,
            flock_core::challenger::pow_squeeze_counter(bits, 16),
            64,
            crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
        )
    };
    let accepts = |nonce: u64| {
        let out = fused(nonce);
        let mut predicate = [0u8; 16];
        for (i, word) in out[4..8].iter().enumerate() {
            predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        predicate[0] & 0b1111_1100 == 0
    };
    let good = (0..u64::MAX)
        .find(|&n| accepts(n))
        .expect("a six-bit nonce exists");
    let bad = (good + 1..u64::MAX)
        .find(|&n| !accepts(n))
        .expect("a neighboring invalid nonce exists");

    let build = |nonce: u64, circuit_bits: u32| {
        // BLAKE3 has k_log=15; nu=7 places this focused union at the
        // smallest embedded security-config size m=22.
        let nu = 7usize;
        let mut sb = ShapeBuilder::new(nu);
        let b3 = sb.slot(Blake3Gate { nu });
        let spread = sb.slot(PowMaskGate { nu });
        let mut vals = Vec::new();
        let digest_v = [
            F128::new(
                u64::from_le_bytes(state_digest[..8].try_into().unwrap()),
                u64::from_le_bytes(state_digest[8..16].try_into().unwrap()),
            ),
            F128::new(
                u64::from_le_bytes(state_digest[16..24].try_into().unwrap()),
                u64::from_le_bytes(state_digest[24..].try_into().unwrap()),
            ),
        ];
        vals.extend_from_slice(&[digest_v[0], digest_v[1], F128::new(nonce, 0)]);
        let digest_w = [sb.input(), sb.input()];
        let nonce_w = sb.input();
        let mut consts = Vec::new();
        let zero = cw(&mut sb, &mut vals, &mut consts, F128::ZERO);
        let params = cw(
            &mut sb,
            &mut vals,
            &mut consts,
            pack_params(
                flock_core::challenger::pow_squeeze_counter(circuit_bits, 16),
                64,
                crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
            ),
        );
        let h = sb.gate(
            b3,
            &[digest_w[0], digest_w[1], nonce_w, zero, zero, zero, params],
        );
        emit_pow_checks(
            &mut sb,
            b3,
            spread,
            digest_w,
            &[([h[1], nonce_w], circuit_bits)],
            &mut vals,
            &mut consts,
        );
        let shape = sb.finish().expect("the focused PoW circuit builds");
        let built = shape.run(&vals, &[]);
        (nu, b3, spread, shape, built)
    };

    let (nu, good_b3, good_spread, good_shape, good_built) = build(good, bits);
    let (_, bad_b3, bad_spread, bad_shape, bad_built) = build(bad, bits);
    assert_eq!(good_shape.circuit.digest(), bad_shape.circuit.digest());
    let (_, _, _, downgraded_shape, downgraded_built) = build(0, 0);
    assert_ne!(
        good_shape.circuit.digest(),
        downgraded_shape.circuit.digest(),
        "changing the PoW difficulty changes digest-bound counter/mask constants"
    );
    assert!(!good_shape.circuit.check_public(&downgraded_built.public));
    assert!(
        good_built
            .rows::<PowMaskGate>(good_spread)
            .iter()
            .all(|r| r.pred & r.mask == 0 && r.nonce >> 64 == 0),
        "the valid witness satisfies every fused PoW row"
    );
    assert!(
        bad_built
            .rows::<PowMaskGate>(bad_spread)
            .iter()
            .any(|r| r.pred & r.mask != 0 || r.nonce >> 64 != 0),
        "the invalid nonce reaches a failing in-circuit prefix row"
    );

    let prove = |shape: &flock_core::circuit::builder::CircuitShape,
                 built: &flock_core::circuit::builder::CircuitWitness,
                 b3_slot,
                 spread_slot| {
        let mut union = UnionInstance::new(&shape.registry, shape.counts.clone());
        union.set_dense_floor(22);
        let pcs = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: pcs_batch_for(&union, LigeritoProfile::Fast),
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(pcs_batch_for(&union, LigeritoProfile::Fast)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs = blake3::build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let spread_ty = PowMaskTable;
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let mut slots = vec![
            (
                shape.registry_slot(b3_slot),
                UnionSlotProverInput::new(
                    blake3::generate_witness_batch_major_partial(
                        built.rows::<Blake3Gate>(b3_slot),
                        nu,
                    ),
                    b3_lc,
                ),
            ),
            (
                shape.registry_slot(spread_slot),
                UnionSlotProverInput::new(
                    spread_ty
                        .generate_witness_batch_major(built.rows::<PowMaskGate>(spread_slot), nu),
                    spread_lc,
                ),
            ),
        ];
        slots.sort_by_key(|(i, _)| *i);
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs,
            slots.into_iter().map(|(_, s)| s).collect(),
            Vec::new(),
            &mut ch,
        );
        let mut lcs = vec![
            (shape.registry_slot(b3_slot), b3_lc),
            (shape.registry_slot(spread_slot), spread_lc),
        ];
        lcs.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = lcs
            .into_iter()
            .map(|(_, lc)| lc as &dyn flock_core::lincheck::LincheckCircuit)
            .collect();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs,
            &mut ch,
        )
    };

    prove(&good_shape, &good_built, good_b3, good_spread)
        .expect("a valid grinding witness proves and verifies");
    let bad_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove(&bad_shape, &bad_built, bad_b3, bad_spread)
    }));
    assert!(
        match bad_result {
            Ok(result) => result.is_err(),
            Err(_) => true,
        },
        "an invalid grinding witness must not yield an accepted recursive proof"
    );
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
    position_prefix: usize,
    mut consts: Option<&mut Vec<(F128, Wire)>>,
    pubs: &mut Vec<F128>,
) -> ([Wire; 2], Wire) {
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
    let zero_w = shared(sb, pubs, F128::ZERO);
    let pad_w = if leaf_w.len().is_multiple_of(4) {
        None
    } else {
        Some(zero_w)
    };

    // The index word's bits, one per level.
    // Its zero mask is empty: this row only relocates bits.  Grinding rows
    // below reuse the same table with nonzero masks to enforce predicates.
    let position_bits = depth - cap_depth;
    let position_mask = if position_bits == 128 {
        u128::MAX
    } else {
        (1u128 << position_bits) - 1
    };
    assert_eq!(
        position_prefix & position_mask as usize,
        0,
        "the fixed stratum and sampled low bits must be disjoint"
    );
    let mask_w = shared(
        sb,
        pubs,
        F128::new(position_mask as u64, (position_mask >> 64) as u64),
    );
    let prefix_w = shared(sb, pubs, F128::new(position_prefix as u64, 0));
    let spread = sb.gate(s.spread, &[index_w, zero_w, zero_w, mask_w, prefix_w]);
    let bits = &spread[..spread.len() - 1];
    let position_w = *spread.last().expect("spread emits the derived position");

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
        let out = sb.gate(s.b3, &[cv[0], cv[1], mw(0), mw(1), mw(2), mw(3), params]);
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
    (cv, position_w)
}

/// The leaf outer's artifacts, returned by [`build_leaf_outer`] so the
/// recursion swap can consume the proof as ITS inner: the circuit shape
/// (owning registry + counts — `UnionInstance::new(&shape.registry,
/// shape.counts.clone())` reconstructs the instance), the public segment,
/// the BLAKE3/BLAKE3 circuit proof, and the boolean tables whose lincheck
/// circuits a verifier needs (in registry order via the `*_slot` indices).
pub struct LeafOuter {
    shape: flock_core::circuit::builder::CircuitShape,
    public: Vec<F128>,
    proof: flock_core::proof::R1csProofCircuitMerged,
    commitment: flock_core::pcs::Commitment,
    pcs: PcsParams,
    b3_r1cs: flock_core::r1cs::BlockR1cs,
    swap_r1cs: flock_core::r1cs::BlockR1cs,
    spread_r1cs: flock_core::r1cs::BlockR1cs,
    pow_r1cs: flock_core::r1cs::BlockR1cs,
    family_r1cs: flock_core::r1cs::BlockR1cs,
    b3_slots: Vec<usize>,
    swap_slot: usize,
    spread_slot: usize,
    pow_slot: usize,
    family_slot: usize,
}

fn leaf_boolean_lcs(lo: &LeafOuter) -> Vec<&dyn flock_core::lincheck::LincheckCircuit> {
    let mut ordered: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
        (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        (lo.pow_slot, lo.pow_r1cs.csc_lincheck_circuit()),
        (lo.family_slot, lo.family_r1cs.csc_lincheck_circuit()),
    ];
    ordered.extend(lo.b3_slots.iter().map(|&slot| {
        (
            slot,
            lo.b3_r1cs.csc_lincheck_circuit() as &dyn flock_core::lincheck::LincheckCircuit,
        )
    }));
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, circuit)| circuit).collect()
}

fn leaf_boolean_mats(
    lo: &LeafOuter,
) -> Vec<(
    &flock_core::r1cs::SparseBinaryMatrix,
    &flock_core::r1cs::SparseBinaryMatrix,
)> {
    let mut ordered = vec![
        (lo.swap_slot, (&lo.swap_r1cs.a_0, &lo.swap_r1cs.b_0)),
        (lo.spread_slot, (&lo.spread_r1cs.a_0, &lo.spread_r1cs.b_0)),
        (lo.pow_slot, (&lo.pow_r1cs.a_0, &lo.pow_r1cs.b_0)),
        (lo.family_slot, (&lo.family_r1cs.a_0, &lo.family_r1cs.b_0)),
    ];
    ordered.extend(
        lo.b3_slots
            .iter()
            .map(|&slot| (slot, (&lo.b3_r1cs.a_0, &lo.b3_r1cs.b_0))),
    );
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, matrices)| matrices).collect()
}

// **THE SWAP, step 1 — mvp9's outer becomes the inner.** The leaf-outer
// circuit proof (the first real recursion node, BLAKE3/BLAKE3 from the
// shared builder) is natively verified under a RecordingChallenger and its
// tape walked by the SAME machinery mvp10's assembly consumes:
// parse_open_levels, the region label map, level_geometry (native capped
// paths + enforced-sum replicas per level), and the R=2 + P multipoint
// schedule replayed to the anchor's claimed v — pinned before any
// assembly, the step-1 pattern every phase ran. What it establishes about
// the REAL inner's shape: the element PIOP parses at multi-slot scale, the
// packed-direct claims are the element (c, lc) pair plus every wiring
// gather, the R=2 + P>0 schedule holds, and the committed lane count is
// once more an arbitrary integer.
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
    trace: crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: flock_core::transcript_record::Stream,
    bytes: Vec<u8>,
    /// The fork's four cross-link wires ([`MergedChain::cross`]).
    cross: Vec<Option<(usize, usize)>>,
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
    native_sums: Vec<F256>,
    /// The grinding ops: (fin ordinal, payload ordinal, bits).
    pows: Vec<(usize, usize, u32)>,
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
    /// beta term, bound through the digest-keyed circuit-structure table.
    eps_n: Vec<F128>,
    /// (g_v, ch, fin) per boolean lc round — messages feed the in-circuit
    /// lincheck replay.
    lc_rounds_b: Vec<(usize, usize, usize)>,
    zskip_ch: usize,
    zskip_fin: usize,
    zp_v: usize,
    /// The rs regions: (s_hat_v ordinal, r_dprime fin, r_dprime ch), plus
    /// the two rs gammas' `(fin, word offset)` and challenge ordinals — the
    /// family-H circuit. Both coefficients share one vector squeeze.
    rs_recs: Vec<(usize, usize, usize)>,
    rs_gam_fins: Vec<(usize, usize)>,
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
    native_target: F128,
    native_running: F128,
    t_final_n: F256,
    anc_end_n: F128,
    mid_n: F128,
    live_n: F128,
    mu_i: usize,
    // anchor-expect geometry — statement constants of the real inner
    n_log_i: usize,
    k_cols_i: usize,
    m_mp2: usize,
    bounds_i: Vec<(u64, u64, u32)>,
    #[allow(dead_code)]
    // The layout's run→column map — the eqc_w era's consumer; kept as shape data.
    run_of: Vec<usize>,
    x_ab_n: Vec<F128>,
    x_c_n: Vec<F128>,
    groups_ix: Vec<Vec<usize>>,
    /// Derived pd claim points (merged-open v1), pinned order
    /// [element c, element lc, gathers in cell-slot order].
    pd_pts: Vec<Vec<F128>>,
    /// The deferred verify's jagged-layout export (the count win) — the
    /// independent reference for the W-value publics, tied to the native
    /// expect replica in the constructor.
    jag: flock_core::matrix_fold::JaggedAssertion,
}

impl<'p> RealTape<'p> {
    fn new(lo: &'p LeafOuter, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

        let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
        let lcs = leaf_boolean_lcs(lo);
        // ONE recorded DEFERRED verify serves both needs: it is
        // transcript-identical to the plain verify for honest proofs (so
        // the tape is unchanged), it skips the sigma discharge the plain
        // pass paid, and its exported assertions ARE the method-note
        // references (verifier-exported over formulas-written-twice).
        // This is also exactly what a production node runs per child —
        // the tape cost halved when the second pass dissolved.
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
        let (mat_assert, el_assert, sigma_native, jag_assert, claims) = {
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
            assert!(
                claims.boolean.is_some(),
                "boolean claims from the real inner"
            );
            assert!(
                claims.element.is_some(),
                "element claims from the real inner"
            );
            (
                work.boolean.expect("a boolean PIOP ran"),
                work.element.expect("an element PIOP ran"),
                sigma,
                work.jagged,
                claims,
            )
        };
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops = flatten_ops(t_shape.ops());
        let mut pub_payloads = bytes_payload_mask(&ops);
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
            for op in &ops {
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
            (
                zc_l.len(),
                lc_l.len(),
                elzc_l.len(),
                el_l.len(),
                gkr_l.len()
            ),
            (1, 1, 1, 1, 1),
            "one region each"
        );
        assert_eq!(
            (mo_l.len(), rs_l.len(), mp_l.len(), fa_l.len()),
            (1, 2, 1, 1)
        );
        // THE FORKED ORDER. The wiring argument runs on its own chain, and
        // the flattened view splices it in at the fork point — so the GKR
        // region now PRECEDES the boolean PIOP instead of following the
        // element's. Everything downstream of the merge is unmoved.
        assert!(
            gkr_l[0] < zc_l[0],
            "the wiring fork precedes the boolean PIOP"
        );
        assert!(zc_l[0] < lc_l[0] && lc_l[0] < elzc_l[0] && elzc_l[0] < el_l[0]);
        assert!(el_l[0] < mo_l[0]);
        assert!(mo_l[0] < rs_l[0] && rs_l[1] < mp_l[0] && mp_l[0] < fa_l[0]);

        // parse_open_levels + level_geometry — the assembly's own walkers,
        // unchanged, on the real-inner tape.
        let lig = &lo.proof.pcs_open.inner.ligerito;
        let r = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v_i, piop_i, gammas_i, w_rounds, mp_i, inner_pd_i, yr_v_i, levels) =
            parse_open_levels(&ops, 32 * lig.initial_cap.len(), r);
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
        assert!(
            geo[0].row_words <= geo[0].lanes,
            "committed width fits the fold"
        );

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
            tm, vals_rec[mp_i.anchor_v],
            "T0 folds to the anchor's claimed v"
        );

        // ---- the wiring GKR walk (the mvp10 walker, real-inner layers) ----
        // Records every ordinal the transcription wires against and replays
        // the whole layer recursion natively in lockstep, input checks
        // included — the rhs consuming the DEFERRED s_sigma from the proof.
        let gkr_rec = {
            let gkr = &lo.proof.wiring.gkr;
            let mut i = gkr_l[0] + 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
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
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
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
                    let mut rho_i = i + 2;
                    while matches!(ops[rho_i], Op::Pow { .. }) {
                        rho_i += 1;
                    }
                    assert!(matches!(ops[rho_i], Op::SqueezeScalar), "round rho");
                    let (_, rc2) = vc_at(rho_i);
                    let rho = chals[rc2];
                    rrecs.push((gv, fin_at(rho_i)));
                    i = rho_i + 1;
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
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
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
        // The transcript is FORKED (the wiring runs on its own chain);
        // `merge_chain` splices the child's rows in at the fork point and
        // hands back one linear numbering plus the four cross-link wires.
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = merge_chain(
            t_shape.ops(),
            &t_shape.stream_words_duplex(domain),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, &chals);
        let b3_rows = trace.rows.len() + h_rows + query_phase_b3_rows(&geo);
        if std::env::var("B3_CENSUS").is_ok() {
            let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
            let blocks = trace.rows.len() - parents;
            let mut pow_by_bits = std::collections::BTreeMap::<u32, usize>::new();
            for op in &ops {
                if let Op::Pow { bits } = op {
                    *pow_by_bits.entry(*bits).or_default() += 1;
                }
            }
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
            eprintln!(
                "  [pow census] {} checks by bits {:?}",
                pow_by_bits.values().sum::<usize>(),
                pow_by_bits
            );
            for g in geo.iter() {
                let (leaf, path, cap) = level_query_phase_b3_rows(g);
                eprintln!(
                    "    level: q {} depth {} row_words {} -> leaf {} + path {} + cap {}",
                    g.q, g.depth, g.raw_row_words, leaf, path, cap,
                );
            }
            // CHAIN DECOMPOSITION + an independent row-count model of the
            // duplex discipline (transcript-v3), asserted against the
            // sponge trace: a squeeze row absorbs the pending partial
            // block as its MESSAGE, mutates cv, and has no header word.
            {
                let pad16 = |n: usize| n.div_ceil(16) * 16;
                let (mut hdr_w, mut pay_w, mut n_obs, mut n_sq) = (0usize, 0usize, 0usize, 0usize);
                for op in ops.iter() {
                    match op {
                        Op::Label(l) => {
                            hdr_w += 1;
                            pay_w += pad16(l.len()) / 16;
                            n_obs += 1;
                        }
                        Op::ObserveScalar => {
                            hdr_w += 1;
                            pay_w += 1;
                            n_obs += 1;
                        }
                        Op::ObserveSlice(n) => {
                            hdr_w += 1;
                            pay_w += n;
                            n_obs += 1;
                        }
                        Op::ObserveBytes(len) => {
                            hdr_w += 1;
                            pay_w += pad16(*len) / 16;
                            n_obs += 1;
                        }
                        Op::Forked { .. } | Op::Merge { .. } => {}
                        Op::Pow { .. } => {
                            pay_w += 1;
                        }
                        Op::LegacyPow { .. } => {
                            n_sq += 1;
                        }
                        Op::SqueezeScalar | Op::SqueezeSlice(_) => {
                            n_sq += 1;
                        }
                    }
                }
                let v3_rows =
                    duplex_row_count_model(t_shape.ops(), &t_shape.stream_words_duplex(domain));
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
            for op in &ops {
                if let Op::Pow { bits } = op {
                    out.push((fin, pay, *bits));
                }
                if op.finalizes() {
                    fin += 1;
                }
                match op {
                    Op::ObserveBytes(_) | Op::Pow { .. } | Op::LegacyPow { .. } => pay += 1,
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
                while matches!(ops[i2], Op::Pow { .. }) {
                    i2 += 1;
                }
                assert!(matches!(ops[i2], Op::SqueezeSlice(7)), "r_dprime");
                recs.push((sv, fin_at(i2), vc_at(i2).1));
                i2 += 1;
            }
            // All PD values follow the RS regions. One PoW then protects one
            // vector squeeze in claim order: RS[0..2], PD[0..P].
            for pd in &gammas_i {
                assert!(
                    matches!(ops[i2], Op::ObserveScalar),
                    "pd value before batch vector"
                );
                assert_eq!(vc_at(i2).0, pd.val_v, "pd intake order");
                i2 += 1;
            }
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(
                matches!(ops[i2], Op::SqueezeSlice(n) if n == 2 + gammas_i.len()),
                "mixed coefficient vector"
            );
            let base_ch = vc_at(i2).1;
            let fin = fin_at(i2);
            let gchs = vec![base_ch, base_ch + 1];
            let gfins = vec![(fin, 0), (fin, 1)];
            (recs, gchs, gfins)
        };
        // Native differential replay of the two-halves target and V. The
        // recursive circuit independently computes the RS, packed-direct,
        // and group parts below; these values no longer discharge soundness.
        let (native_target, native_running) = {
            use flock_core::pcs::ring_switch as rsw;
            use flock_core::zerocheck::univariate_skip::build_eq;
            let gs: Vec<F128> = rs_gam_ch2.iter().map(|&ch| chals[ch]).collect();
            let mut rs_half = F128::ZERO;
            let mut coeffs: Vec<Vec<F128>> = Vec::new();
            for (k, &(sv, _, rc)) in rs_recs2.iter().enumerate() {
                let shv = &vals_rec[sv..sv + 128];
                let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
                let eq = build_eq(&rdp);
                rs_half += gs[k] * rsw::inner_product(&rsw::tensor_algebra_transpose(shv), &eq);
                let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
                coeffs.push(rsw::linearized_coefficients(&rsw::build_fold_byte_table(
                    &scaled,
                )));
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
            (target, running)
        };

        // ---- the spine's native quad replay ----
        let t_final_n = replay_ligerito_spine256(
            &levels,
            &vals_rec,
            &chals,
            start_v_i,
            chals[inner_pd_i.ch] * vals_rec[inner_pd_i.q_v],
            &native_sums,
        );

        // ---- the residual pairing's rotation (lane-major inners) ----
        // A pow2-lane inner (row_words == lanes — e.g. the m28-k4 slim node
        // whose 16-of-16 lanes make the commit exactly full) takes the
        // IDENTITY pairing, same as the native side's rotate gate and
        // ChildTape's conditional.
        let yr_len = lo.proof.pcs_open.inner.ligerito.final_proof.yr.len() / 2;
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
            el_assert.alpha, chals[piop_i.alpha_ch],
            "the located alpha is the assertion's"
        );
        let (a_sum_n, b_sum_n) = {
            let slots_el = flock_core::element_r1cs::union::region_slots(&union_i);
            let nu_i = union_i.n_log();
            let mut a_sum = F128::ZERO;
            let mut b_sum = F128::ZERO;
            for s in &slots_el {
                let kappa = s.ty.kappa();
                let eq_con =
                    flock_core::zerocheck::univariate_skip::build_eq(&el_assert.r_con[..kappa]);
                let prefix = s.layout.region_prefix(nu_i);
                let mut w = F128::ONE;
                for (j, &x) in el_assert.r_con[kappa..].iter().enumerate() {
                    w *= if (prefix >> j) & 1 == 1 {
                        x
                    } else {
                        F128::ONE + x
                    };
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
            // A grinded zerocheck inserts one `Pow` immediately before each
            // protected squeeze.  The generic PoW locator below emits and
            // binds its BLAKE3 predicate for *every* such op; this PIOP
            // locator only needs to step past them before naming the squeeze
            // wires which feed the arithmetic replay.
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
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
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeScalar), "z_skip");
            let zskip = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            let mut zc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
                let mut squeeze_i = i2 + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                zc_r.push((vc_at(squeeze_i).1, fin_at(squeeze_i)));
                i2 = squeeze_i + 1;
            }
            // The zerocheck finals (v_a, v_b, ...) — the lincheck entry's
            // absorbed operands.
            let (zcf, _) = vc_at(i2);
            while matches!(ops[i2], Op::ObserveScalar) {
                i2 += 1;
            }
            assert_eq!(i2, lc_l[0], "the zerocheck runs straight into the lincheck");
            i2 += 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeScalar), "lc alpha");
            let lc_alpha = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            // The const-pin beta squeezes, one per pinned boolean type.
            let mut betas: Vec<(usize, usize)> = Vec::new();
            loop {
                while matches!(ops[i2], Op::Pow { .. }) {
                    i2 += 1;
                }
                if !matches!(ops[i2], Op::SqueezeScalar) {
                    break;
                }
                betas.push((vc_at(i2).1, fin_at(i2)));
                i2 += 1;
            }
            // (g_v, ch, fin) per lc round — the message ordinals feed the
            // round-0 in-circuit lincheck replay.
            let mut lc_r: Vec<(usize, usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
                let mut squeeze_i = i2 + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                lc_r.push((vc_at(i2).0, vc_at(squeeze_i).1, fin_at(squeeze_i)));
                i2 = squeeze_i + 1;
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
                    chals[zc_rounds_b[m].0], x,
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
                mat_assert.alpha, chals[bl_alpha.0],
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
        assert!(
            lc_rounds_b.len() <= 1 + k_cols_i,
            "lc rounds fit the col bits"
        );
        // The boolean lincheck ENTRY, natively: target0 = α·v_a + v_b +
        // Σ β_t·eq_prefix_sum(x_outer, n_t), with x_outer the zc mlv rows
        // (batch-major: rounds 1..1+ν) — replayed through the located lc
        // rounds it must end at the deferred MatrixAssertion's own target
        // (the method-note discipline; this pre-assert is what licenses the
        // in-circuit replay's wire map).
        let (eps_n, entry_n) = {
            let x_outer_n: Vec<F128> = (0..n_log_i).map(|j| chals[zc_rounds_b[1 + j].0]).collect();
            let pinned: Vec<usize> = mat_assert
                .betas
                .iter()
                .enumerate()
                .filter_map(|(t, b)| b.map(|_| t))
                .collect();
            assert_eq!(pinned.len(), betas_b.len(), "one squeeze per const pin");
            let mut eps = Vec::with_capacity(betas_b.len());
            let mut entry = mat_assert.alpha * vals_rec[zc_finals_v] + vals_rec[zc_finals_v + 1];
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
                    let za = if layer < n_log_i {
                        z_row[layer]
                    } else {
                        F128::ZERO
                    };
                    let rb = if layer < m_mp2 {
                        point_n[layer]
                    } else {
                        F128::ZERO
                    };
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
                    // The count win's tie at real-inner scale: the RS raw W
                    // the region now publishes equals the deferred export's
                    // claim value.
                    assert_eq!(
                        jag_assert.rs[si].value, w_n,
                        "RS raw W == exported jagged claim {si}"
                    );
                    let coeff = if si == 0 {
                        g_at_n
                    } else {
                        gpow_n[128] * g_at_n
                    };
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
                    // The group's exported decomposition recombines to the
                    // same raw group W, member for member.
                    let (combo, dense) = &jag_assert.groups[g_ix];
                    let mut raw = combo.as_ref().map_or(F128::ZERO, |c| c.value);
                    let mut d_it = dense.iter();
                    for &i2 in members {
                        let hot = pd_pts_n[i2][n_log_i..]
                            .iter()
                            .all(|&x| x == F128::ZERO || x == F128::ONE);
                        if hot {
                            continue;
                        }
                        let (g, c) = d_it.next().expect("a dense entry per non-hot member");
                        assert_eq!(*g, chals[gammas_i[i2].ch], "dense member γ_pd");
                        raw += *g * c.value;
                    }
                    assert!(d_it.next().is_none(), "every dense entry consumed");
                    assert_eq!(
                        raw, w_n,
                        "group {g_ix} raw W == exported jagged decomposition"
                    );
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
            cross,
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
            rs_gam_fins: rs_gam_fin2,
            mat_assert,
            el_assert,
            sigma_native,
            z_ix,
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
            run_of,
            x_ab_n,
            x_c_n,
            groups_ix,
            pd_pts: pd_pts_n,
            jag: jag_assert,
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
    n_ela_pub: usize,
    /// Labeled `public_len` checkpoints through the emission — the publics
    /// census (`PUB_CENSUS=1` on the node test prints the block sizes).
    census: Vec<(&'static str, usize, usize)>,
    /// The jagged assertion's value wires (the count win), in emission
    /// order: rs claims, then per group the combo and its dense members —
    /// the fresh-claim surfaces a merge fold connects to.
    jag_w: Vec<Wire>,
    /// The claims' IDENTITY wires (the points-connect): σ shared, and per
    /// claim (jag_w order) the row wires — Eq: z_col coordinate wires
    /// (constant coords ride zw/ow); Combo: the γ_pd coefficient wires in
    /// term order (addresses are registry constants, bound by the fold
    /// side's shared constant publics).
    jag_sig_w: Vec<Wire>,
    jag_row_w: Vec<Vec<Wire>>,
    /// The z_skip squeeze wire — see [`ChildRegion::zskip_w`].
    zskip_w: Wire,
    /// Every fresh claim in `sigma_native.claims()` as `(row, col, value)`
    /// wires, in accumulator order.
    structure_claim_w: Vec<(Vec<Wire>, Vec<Wire>, Wire)>,
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
    /// The child's own CIRCUIT DIGEST, absorbed by its statement binding
    /// (payload 3) — two public words. This is the KEY its sigma and
    /// jagged claims fold under, and the spine's match-gate compares it
    /// against the key an inherited entry was published with.
    #[allow(dead_code)]
    cd_w: [Wire; 2],
}

/// Decompose the polynomial-basis trace-dual table into one geometric row and
/// seven exceptional entries.  If `d_t = moore_inverse()[t]`, then
/// `d_t = g0 * ratio^t` for every `t >= 7`; the low seven corrections are the
/// only effect of the low terms in GHASH's defining polynomial.  Frobenius
/// powers preserve this form, which lets the circuit evaluate every inverse-
/// Moore row with one prefix product and seven MACs instead of wiring a
/// 128-word constant row.
fn family_h_dual_decomposition() -> (F128, F128, [F128; 7]) {
    use std::sync::OnceLock;
    static DECOMP: OnceLock<(F128, F128, [F128; 7])> = OnceLock::new();
    *DECOMP.get_or_init(|| {
        let minv = flock_core::pcs::ring_switch::moore_inverse();
        let d = &minv[..128];
        let ratio = d[8] * d[7].inv();
        let ratio_inv = ratio.inv();
        let mut g0 = d[7];
        for _ in 0..7 {
            g0 *= ratio_inv;
        }
        let mut corrections = [F128::ZERO; 7];
        let mut g = g0;
        for t in 0..128 {
            if t < 7 {
                corrections[t] = d[t] + g;
            } else {
                assert_eq!(d[t], g, "the GHASH dual basis is geometric above t=6");
            }
            g *= ratio;
        }
        (g0, ratio, corrections)
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_family_h(
    sb: &mut ShapeBuilder,
    tile: flock_core::circuit::builder::SlotId,
    macs: flock_core::circuit::builder::SlotId,
    fold_macs: flock_core::circuit::builder::SlotId,
    spine: flock_core::circuit::builder::SlotId,
    spine256: flock_core::circuit::builder::SlotId,
    mac256: flock_core::circuit::builder::SlotId,
    row_capacity: usize,
    shv: &[Vec<Wire>; 2],
    values: &[Vec<Wire>; 2],
    r_dprime: &[Vec<Wire>; 2],
    gamma: [Wire; 2],
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    zw: Wire,
    ow: Wire,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> (Wire, Wire) {
    assert_eq!(pf_w, 8, "the envelope family-H prefix width is eight");
    for k in 0..2 {
        assert_eq!(shv[k].len(), 128, "one full tensor-algebra row table");
        assert_eq!(values[k].len(), 128, "one Frobenius value per Moore row");
        assert_eq!(
            r_dprime[k].len(),
            7,
            "the ring-switch suffix has seven bits"
        );
    }
    let prefix = |sb: &mut ShapeBuilder, seed: Wire, a: &[Wire], b: &[Wire]| {
        assert!(a.len() <= pf_w && b.len() <= pf_w);
        let mut input = Vec::with_capacity(2 + 2 * pf_w);
        input.push(seed);
        input.extend_from_slice(a);
        input.extend(std::iter::repeat_n(zw, pf_w - a.len()));
        input.extend_from_slice(b);
        input.extend(std::iter::repeat_n(zw, pf_w - b.len()));
        input.push(ow);
        sb.gate(pfslot, &input)[0]
    };
    let spine_mac = |sb: &mut ShapeBuilder, acc: Wire, x: Wire, y: Wire| {
        sb.gate(spine, &[zw, zw, zw, acc, zw, zw, x, y, zw])[3]
    };
    let spine_square =
        |sb: &mut ShapeBuilder, x: Wire| sb.gate(spine, &[zw, zw, ow, zw, zw, zw, zw, zw, x])[4];

    // Equality weights are shared by the transpose dot and the seven sparse
    // corrections in every inverse-Moore coefficient.
    let mut eq_w: [Vec<Wire>; 2] = std::array::from_fn(|_| Vec::with_capacity(128));
    for k in 0..2 {
        for t in 0..128 {
            let bits: Vec<Wire> = (0..7)
                .map(|j| if (t >> j) & 1 == 1 { ow } else { zw })
                .collect();
            eq_w[k].push(prefix(sb, ow, &r_dprime[k], &bits));
        }
    }

    // The transpose is tiled so the boolean relation needs only 17 wired
    // words.  Dot-product linearity lets us accumulate each partial output
    // directly, without materializing the 128 transposed words in the element
    // layer.  Claim 0 uses the main MAC slot and claim 1 the fold MAC slot;
    // this split keeps both below 2^14 rows at the two-child fixed point.
    let mut rs_half = zw;
    for k in 0..2 {
        let dot_slot = if k == 0 { macs } else { fold_macs };
        let mut dot = zw;
        for destination_byte in 0..16 {
            let rows = &shv[k][8 * destination_byte..8 * destination_byte + 8];
            for source_byte in 0..16 {
                let selector = cw(
                    sb,
                    vals,
                    consts,
                    F128::new((source_byte | (destination_byte << 4)) as u64, 0),
                );
                let mut input = rows.to_vec();
                input.push(selector);
                let partial = sb.gate(tile, &input);
                for c in 0..8 {
                    dot = sb.gate(dot_slot, &[dot, partial[c], eq_w[k][8 * source_byte + c]])[0];
                }
            }
        }
        rs_half = sb.gate(macs, &[rs_half, gamma[k], dot])[0];
    }

    // c_j/gamma is the MLE of inverse-Moore row j at r_dprime.  The trace-
    // dual table is geometric except at indices 0..6, so one prefix product
    // plus seven correction MACs computes it.  Constants are fixed publics:
    // changing any of them changes the circuit digest.
    let (mut g0_j, mut ratio_j, mut corrections_j) = family_h_dual_decomposition();
    let minv = flock_core::pcs::ring_switch::moore_inverse();
    // Only the eight orbit seeds are fixed publics.  Successive Frobenius
    // powers are derived by the existing spine's squaring cell, saving more
    // than one thousand public constants at a two-child node.
    let mut g0_j_w = cw(sb, vals, consts, g0_j);
    let mut corrections_j_w: [Wire; 7] =
        std::array::from_fn(|t| cw(sb, vals, consts, corrections_j[t]));
    let mut coeff_w: [Vec<Wire>; 2] = std::array::from_fn(|_| Vec::with_capacity(128));
    for j in 0..128 {
        let mut ratio_pows = [F128::ZERO; 7];
        ratio_pows[0] = ratio_j;
        for q in 1..7 {
            ratio_pows[q] = ratio_pows[q - 1] * ratio_pows[q - 1];
        }
        for k in 0..2 {
            let scaled_r: Vec<Wire> = (0..7)
                .map(|q| {
                    let factor = cw(sb, vals, consts, ratio_pows[q]);
                    spine_mac(sb, zw, r_dprime[k][q], factor)
                })
                .collect();
            let mut mle = prefix(sb, g0_j_w, &r_dprime[k], &scaled_r);
            for t in 0..7 {
                mle = sb.gate(macs, &[mle, corrections_j_w[t], eq_w[k][t]])[0];
            }
            coeff_w[k].push(sb.gate(macs, &[zw, gamma[k], mle])[0]);
        }

        // Pin the closed form to the native matrix on every row.  This is a
        // shape-time assertion, not witness checking, and catches a basis or
        // field-polynomial change before it can silently alter family H.
        let mut gp = g0_j;
        for t in 0..128 {
            let got = gp + if t < 7 { corrections_j[t] } else { F128::ZERO };
            assert_eq!(got, minv[j * 128 + t], "inverse-Moore row {j}, entry {t}");
            gp *= ratio_j;
        }
        g0_j *= g0_j;
        ratio_j *= ratio_j;
        for d in &mut corrections_j {
            *d *= *d;
        }
        if j + 1 < 128 {
            g0_j_w = spine_square(sb, g0_j_w);
            for d in &mut corrections_j_w {
                *d = spine_square(sb, *d);
            }
        }
    }

    // Pair the two RS claims in one F256 squaring chain.  After j extension-
    // field squarings of a+b*u, the second component is b^(2^j) and the first
    // is a^(2^j)+K_j*b^(2^j), with K_{j+1}=K_j^2+NR.  One base-field MAC
    // recovers a^(2^j). The squarings fill the residual region's narrower
    // F256 MAC relation first, then spill into the equivalent Ligerito-spine
    // multiplication only at that slot's physical row limit. This keeps both
    // existing slots in bounds without adding a table type.
    let mut vrs = zw;
    let mut k_j = F128::ZERO;
    for j in 0..128 {
        let mut pair = [values[0][j], values[1][j]];
        for _ in 0..j {
            pair = if sb.rows_in_slot(mac256) < row_capacity {
                emit_mac256(sb, mac256, [zw, zw], pair, pair)
            } else {
                emit_spine256(
                    sb,
                    spine256,
                    [zw, zw],
                    [zw, zw],
                    [ow, zw],
                    [zw, zw],
                    [zw, zw],
                    [zw, zw],
                    [zw, zw],
                    zw,
                    pair,
                )[4]
            };
        }
        let kw = cw(sb, vals, consts, k_j);
        let p0 = sb.gate(macs, &[pair[0], kw, pair[1]])[0];
        vrs = sb.gate(macs, &[vrs, coeff_w[0][j], p0])[0];
        vrs = sb.gate(macs, &[vrs, coeff_w[1][j], pair[1]])[0];
        k_j = k_j * k_j + flock_core::field::QUADRATIC_NONRESIDUE;
    }
    (rs_half, vrs)
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
    b3_slot: flock_core::circuit::builder::SlotId,
    rt: &RealTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    consts: &mut Vec<(F128, Wire)>,
) -> RealRegion {
    let child_q = CollapsedSlots {
        b3: b3_slot,
        ..cs.q
    };
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
            match cs.le.iter().find(|(n, _)| *n == 800 + lanes) {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(LeafEvalGate256::new(lanes));
                    cs.le.push((800 + lanes, sl));
                    sl
                }
            }
        })
        .collect();
    let mut cen: Vec<(&'static str, usize, usize)> =
        vec![("start", sb.public_len(), sb.rows_in_slot(cs.macs))];
    let iv_w = pack8(&crate::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [
        sb.fixed_public_input(iv_w[0]),
        sb.fixed_public_input(iv_w[1]),
    ];
    let (outs, ww) = emit_fs_chain(
        sb,
        b3_slot,
        iv2,
        trace,
        stream,
        &rt.bytes,
        vals,
        consts,
        &rt.pub_payloads,
        &rt.cross,
    );

    cen.push((
        "chain payloads + shared consts",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
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
    // The PoW grinding wires: [predicate word, nonce word] per op.
    let pow_wires: Vec<[Wire; 2]> = rt
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
            [outs[sq[0]][1], nw]
        })
        .collect();
    let pow_checks: Vec<([Wire; 2], u32)> = pow_wires
        .iter()
        .zip(&rt.pows)
        .map(|(&w, &(_, _, bits))| (w, bits))
        .collect();
    emit_pow_checks(sb, b3_slot, cs.q.pow, iv2, &pow_checks, vals, consts);

    // ---- ROUND 2: the H(publics) region (v2 statement binding) ----
    // Payload 4 of the circuit binding is the 32-byte publics digest; the
    // child's public words themselves are witness, bound here. The returned
    // wires ARE the child's public segment — the recombination folds them.
    // Payload 3 is the child's CIRCUIT DIGEST (`bind_statement_circuit`'s
    // order: registry, counts, cap, circuit, publics) — the FOLD KEY this
    // child's claims belong under (wall 3), exported so the fold region's
    // absorbed group digest binds to the circuit actually verified here.
    let pays = payload_words(stream);
    assert_eq!(pays[3].len(), 2, "the circuit digest payload is 32 bytes");
    let cd_w = [
        ww[pays[3][0]].expect("circuit digest word wired"),
        ww[pays[3][1]].expect("circuit digest word wired"),
    ];
    let pub_w = {
        assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
        let dw = [
            ww[pays[4][0]].expect("digest word wired"),
            ww[pays[4][1]].expect("digest word wired"),
        ];
        emit_publics_hash(sb, child_q, iv2, &rt.lo.public, dw, vals, consts)
    };
    cen.push((
        "H(publics) region",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    let cap_w = cap_wires(stream, &ww, &rt.cap_pays);
    let (to_publish, level_accs, query_positions) = emit_query_phase(
        sb,
        child_q,
        iv2,
        &leafeval,
        levels,
        geo,
        &rt.lvl_src,
        trace,
        &outs,
        chals,
        &cap_w,
        vals,
        consts,
        hints,
    );

    cen.push((
        "query phase decl",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
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
    let spine256 = cs.spine256;
    // The assert-zero anchor: a dedicated zero public NO gate consumes,
    // so the zero-delta outputs connected into its class add no
    // dataflow edges (connecting them to the ubiquitous `zw` creates
    // cycles — the acyclicity check draws producer→consumer edges).
    vals.push(F128::ZERO);
    let zassert = sb.public_input();

    cen.push((
        "zero/one/anchor consts",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // The ligerito SPINE: start gamma'·q_eval, eval/build per fold,
    // intro-folds consuming the query phase's accumulator wires.
    let gpw = outs[trace.squeezes[inner_pd_i.fin][0]][0];
    let z2 = [zw, zw];
    let tw0 = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        z2,
        z2,
        [wv(inner_pd_i.q_v), zw],
        gpw,
        z2,
    );
    let mut tsp = tw0[3];
    for od in &levels[0].initial_ood {
        let bw = outs[trace.squeezes[od.beta_fin][0]][0];
        tsp = emit_spine256(
            sb,
            spine256,
            z2,
            z2,
            z2,
            tsp,
            z2,
            z2,
            [wv(od.y_v), zw],
            bw,
            z2,
        )[3];
    }
    let st = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        [wv(rt.start_v_i), wv(rt.start_v_i + 1)],
        [wv(rt.start_v_i + 2), wv(rt.start_v_i + 3)],
        tsp,
        ow,
        z2,
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = [
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 0),
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 1),
            ];
            let ev = emit_spine256(sb, spine256, qc, qb, qa, z2, z2, z2, z2, zw, rw);
            tsp = ev[4];
            let bld = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                z2,
                [wv(mv), wv(mv + 1)],
                [wv(mv + 2), wv(mv + 3)],
                tsp,
                ow,
                z2,
            );
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r {
            for od in &lvl.ood {
                let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                let f = emit_spine256(
                    sb,
                    spine256,
                    qc,
                    qb,
                    qa,
                    tsp,
                    [wv(od.intro_v), wv(od.intro_v + 1)],
                    [wv(od.intro_v + 2), wv(od.intro_v + 3)],
                    [wv(od.y_v), zw],
                    bw,
                    z2,
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            }
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                qc,
                qb,
                qa,
                tsp,
                [wv(lvl.intro_v), wv(lvl.intro_v + 1)],
                [wv(lvl.intro_v + 2), wv(lvl.intro_v + 3)],
                level_accs[li],
                bw,
                z2,
            );
            (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                tsp,
                z2,
                z2,
                level_accs[li],
                bw,
                z2,
            );
            tsp = f[3];
        }
    }
    let t_final = tsp;

    // The RESIDUAL region via the shared emitter (lane-major rotation).
    let yr_wires: Vec<[Wire; 2]> = (0..rt.yr_len)
        .map(|y| [wv(rt.yr_v_i + 2 * y), wv(rt.yr_v_i + 2 * y + 1)])
        .collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        &to_publish,
        &query_positions,
        &rt.w_resid,
        inner_pd_i.fin,
        &yr_wires,
        trace,
        &outs,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: inner == t_r as a copy constraint.
    sb.connect(inner_w[0], t_final[0]);
    sb.connect(inner_w[1], t_final[1]);

    // The complete family-H relation.  All inputs below are already bound
    // transcript or proof wires; no target/V advice and no native checker are
    // part of the recursive statement anymore.
    let shv_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let sv = rt.rs_recs[k].0;
        (0..128).map(|i| wv(sv + i)).collect()
    });
    let value_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        mp_i.val_vs[128 * k..128 * (k + 1)]
            .iter()
            .map(|&vi| wv(vi))
            .collect()
    });
    let rdp_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let fin = rt.rs_recs[k].1;
        (0..7)
            .map(|j| squeeze_word_wire(&outs, trace, fin, j))
            .collect()
    });
    let gamma_w: [Wire; 2] = std::array::from_fn(|k| {
        let (fin, offset) = rt.rs_gam_fins[k];
        squeeze_word_wire(&outs, trace, fin, offset)
    });
    let (rsh_w, vrs_w) = emit_family_h(
        sb,
        cs.q.family.expect("family-H slot"),
        cs.macs,
        cs.fold_macs,
        cs.spine,
        cs.spine256,
        cs.resid
            .iter()
            .find(|&&(key, _)| key == 701)
            .expect("the child slots declare an F256 MAC slot")
            .1,
        1usize << cs.nu,
        &shv_w,
        &value_w,
        &rdp_w,
        gamma_w,
        pfslot,
        pf_w,
        zw,
        ow,
        vals,
        consts,
    );

    let mut pdh_w = zw;
    for pd in gammas_i {
        let gw = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
        pdh_w = sb.gate(cs.macs, &[pdh_w, gw, wv(pd.val_v)])[0];
    }
    let tgt_w = sb.gate(cs.macs, &[rsh_w, ow, pdh_w])[0];
    let mut runw = tgt_w;
    for rr in w_rounds {
        let r_w = outs[trace.squeezes[rr.fin][0]][0];
        runw = sb.gate(mrslot, &[runw, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let mut vgrp_w = zw;
    for &vi in &mp_i.val_vs[256..] {
        vgrp_w = sb.gate(cs.macs, &[vgrp_w, ow, wv(vi)])[0];
    }
    let v_w = sb.gate(cs.macs, &[vrs_w, ow, vgrp_w])[0];
    let rhs_v_w = sb.gate(cs.macs, &[zw, wv(inner_pd_i.q_v), v_w])[0];
    sb.connect(runw, rhs_v_w);

    cen.push((
        "family-H + merged boundary",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));

    cen.push((
        "spine + residual advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
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
    assert_eq!(
        pt_w.len(),
        rt.mu_i,
        "the GKR point spans the inner cell space"
    );
    // M̂(ρ) / livê(ρ), bound through the digest-keyed
    // circuit-structure claims folded by the parent.
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
        cs.fold_macs,
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

    cen.push((
        "GKR advice (g0s, mask)",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // ---- the MULTI-SLOT element PIOP (general strip) ----
    let mut el_zr = zw;
    for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
        let t_w = squeeze_word_wire(&outs, trace, piop_i.tau_fin, k);
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        vals.push(rt.el_g0[k]);
        let g0w = sb.input();
        let o = sb.gate(
            zcr,
            &[el_zr, wv(rr.g_v), wv(rr.g_v + 1), t_w, rho_w, g0w, ow],
        );
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

    cen.push((
        "element PIOP advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
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
    assert_eq!(
        mp_sig_w.len(),
        2 * (m_mp2 + 1),
        "sigma spans the anchor layers"
    );

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
    // THE COUNT WIN: everything from here to `connect(anc_w, expect_w)`
    // used to be the W side of the anchor expect — per-run boundary eq
    // products with the child's jagged run boundaries baked as ow/zw
    // (`eqc_w`, THE one site where counts were circuit structure) plus the
    // eq-table dots consuming them (~7.4k rows per region, 6.8% of a
    // node's committed words). All deleted: each statement's raw W arrives
    // as a PUBLISHED CLAIM VALUE on the jagged layout table — the deferred
    // verify's own export, keyed by the child digest — checker-held here
    // and discharged at the ROOT of the accumulation tree. The claim's
    // points are wires this region already carries (σ = the anchor round
    // squeezes, z_cols = statement point wires, γ_pd = squeezes); nothing
    // count-shaped remains in the circuit.
    let mut jag_w: Vec<Wire> = Vec::new();
    // The claims' IDENTITY wires (the points-connect): per claim, in
    // jag_w order, the row wires the merge fold's absorbed words connect
    // to; σ is mp_sig_w, shared.
    let mut jag_row_w: Vec<Vec<Wire>> = Vec::new();
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        vals.push(rt.jag.rs[si].value);
        let w_st = sb.input();
        jag_w.push(w_st);
        jag_row_w.push(xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect());
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
        // The γ-baked one-hot combo (all this group's gather claims) is ONE
        // published value; each dense (element) member publishes its raw eq
        // value with γ_pd applied by a MAC on the squeeze wire — the
        // exported decomposition reassembled in wires.
        let (combo, dense) = &rt.jag.groups[g_ix];
        let hots: Vec<bool> = members
            .iter()
            .map(|&i2| {
                rt.pd_pts[i2][n_log_i..n_log_i + k_cols_i]
                    .iter()
                    .all(|&x| x == F128::ZERO || x == F128::ONE)
            })
            .collect();
        let mut w_st = match combo {
            Some(c) => {
                vals.push(c.value);
                let w = sb.input();
                jag_w.push(w);
                // The combo's identity: the hot members' γ_pd squeeze
                // wires, member order == the assertion's term order.
                let gws: Vec<Wire> = members
                    .iter()
                    .zip(&hots)
                    .filter(|&(_, &h)| h)
                    .map(|(&i2, _)| {
                        let pd = &gammas_i[i2];
                        squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset)
                    })
                    .collect();
                if let flock_core::matrix_fold::JaggedRowWeight::Combo(t) = &c.row {
                    assert_eq!(t.len(), gws.len(), "combo terms == hot members");
                }
                jag_row_w.push(gws);
                w
            }
            None => zw,
        };
        let mut d_it = dense.iter();
        for (&i2, &hot) in members.iter().zip(&hots) {
            if hot {
                assert!(i2 >= 2, "one-hot columns are gather claims");
                continue;
            }
            let (_, c) = d_it.next().expect("a dense entry per non-hot member");
            let pd = &gammas_i[i2];
            let gpd_w = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
            vals.push(c.value);
            let d_w = sb.input();
            jag_w.push(d_w);
            // The dense claim's identity: its z_col coordinate wires —
            // constant coords ride zw/ow, the rest the element PIOP's own
            // squeeze wires (the mapping the constructor pinned).
            jag_row_w.push(
                (0..k_cols_i)
                    .map(|jj| {
                        let coord = rt.pd_pts[i2][n_log_i + jj];
                        if coord == F128::ZERO {
                            zw
                        } else if coord == F128::ONE {
                            ow
                        } else if i2 == 0 {
                            outs[trace.squeezes[piop_i.zc_rounds[n_log_i + jj].fin][0]][0]
                        } else {
                            let n_lc = piop_i.lc_rounds.len();
                            outs[trace.squeezes[piop_i.lc_rounds[n_lc - 1 - jj].fin][0]][0]
                        }
                    })
                    .collect(),
            );
            w_st = sb.gate(macs, &[w_st, gpd_w, d_w])[0];
        }
        assert!(d_it.next().is_none(), "every dense entry consumed");
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
    if std::env::var("ASSIST_CENSUS").is_ok() {
        eprintln!(
            "ASSIST CENSUS  runs {} of {} cols (k {}), m+1 {} — W side is {} PUBLISHED \
             claim values (the count win); the eqc_w/eq_dot machinery is gone",
            rt.bounds_i.len(),
            1usize << k_cols_i,
            k_cols_i,
            m_mp2 + 1,
            jag_w.len(),
        );
    }

    cen.push((
        "multipoint + anchor expect advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
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
    // the const-pin betas + their structure-table-bound prefix values,
    // the z_partial
    // words — and the ~20-row BOOLEAN LINCHECK REPLAY, so the published
    // chain end IS the equation's bound target: entry = α·v_a + v_b +
    // Σ β_t·eps_t from absorbed finals and squeeze wires, rounds through
    // the shared MergedRoundGate slot.
    let inner_b = rt.mat_assert.x_inner_rest.len();
    let mat_x_inner_w: Vec<Wire> = (0..inner_b)
        .map(|j| {
            let m = if j == 0 { 0 } else { n_log_i + j };
            mlv_pw[m].1
        })
        .collect();
    mat_pub.extend_from_slice(&mat_x_inner_w);
    for j in 0..n_log_i {
        mat_pub.push(mlv_pw[1 + j].1);
    }
    let mat_rr_w: Vec<Wire> = lc_pw.iter().map(|&(_, w)| w).collect();
    let zpartial_ws: Vec<Wire> = (0..64).map(|i| wv(rt.zp_v + i)).collect();
    let va_b = wv(rt.zc_finals_v);
    let vb_b = wv(rt.zc_finals_v + 1);
    let mut lcb_w = sb.gate(cs.macs, &[vb_b, bl_alpha_w, va_b])[0];
    let mut eps_wires = Vec::with_capacity(rt.betas_b.len());
    let mut beta_wires = vec![None; rt.lo.shape.registry.num_boolean()];
    for (k, &(_, bfin)) in rt.betas_b.iter().enumerate() {
        let bw = outs[trace.squeezes[bfin][0]][0];
        let type_index = rt.sigma_native.boolean_pins[k].0;
        beta_wires[type_index] = Some(bw);
        vals.push(rt.eps_n[k]);
        let ew = sb.public_input();
        eps_wires.push(ew);
        lcb_w = sb.gate(cs.macs, &[lcb_w, bw, ew])[0];
        mat_pub.push(bw);
        mat_pub.push(ew);
    }
    for &(g_v, _, fin) in &rt.lc_rounds_b {
        let rw = outs[trace.squeezes[fin][0]][0];
        lcb_w = sb.gate(mrslot, &[lcb_w, wv(g_v), wv(g_v + 1), rw])[0];
    }
    emit_boolean_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &rt.lo.shape.registry,
        bl_alpha_w,
        &mat_x_inner_w,
        &mat_rr_w,
        &zpartial_ws,
        &beta_wires,
        &mat_eval_w,
        lcb_w,
        zw,
        ow,
    );
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
    let inner_union = UnionInstance::new(&rt.lo.shape.registry, rt.lo.shape.counts.clone());
    let el_r_con_w: Vec<Wire> = piop_i.zc_rounds[n_log_i..]
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let el_r_col_w: Vec<Wire> = piop_i
        .lc_rounds
        .iter()
        .rev()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    emit_element_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &inner_union,
        el_alpha_w,
        &el_r_con_w,
        &el_r_col_w,
        wv(gammas_i[rt.z_ix].val_v),
        &el_eval_w,
        el_lcw,
        zw,
        ow,
    );

    cen.push((
        "assertion eval advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // ---- the publishes, in the swap's recorded order ----
    let pub_base = sb.public_len();
    for a_wires in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    for w in &level_accs {
        sb.publish(w[0]);
        sb.publish(w[1]);
    }
    cen.push((
        "TAIL: query alphas + native accs",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(t_final[0]);
    sb.publish(t_final[1]);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(w[0]);
            sb.publish(w[1]);
        }
    }
    cen.push((
        "TAIL: chain ends + residual accs",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(inner_w[0]);
    sb.publish(inner_w[1]);
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: sigma + GKR point",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(el_zr);
    sb.publish(el_lcw);
    sb.publish(anc_w);
    for w in &mat_pub {
        sb.publish(*w);
    }
    for w in &ela_pub {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: el ends + assertion publics",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // Family H is now internal arithmetic.  Its source words are already
    // bound where they enter the transcript/proof stream, so no duplicate
    // public re-exposure or checker-only advice remains.
    // ---- the JAGGED ASSERTION emission (the count win) ----
    // Raw W claim values in emission order (rs, then per group combo +
    // dense members), checker-held against the deferred export — the
    // fresh-claim surfaces a merge fold connects to.
    for w in &jag_w {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: jagged claim values",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));

    let n_query_pub: usize = 2 * levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    let n_tail = 4
        + 2 * levels.len() * rt.yr_len
        + 2
        + 1
        + rt.mu_i
        + 2
        + 1
        + mat_pub.len()
        + ela_pub.len()
        + jag_w.len();
    let el_zc_rho_w: Vec<Wire> = piop_i
        .zc_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let boolean_values: Vec<(usize, Wire)> = rt
        .sigma_native
        .boolean_pins
        .iter()
        .map(|(t, _, _)| *t)
        .zip(eps_wires)
        .collect();
    let structure_claim_w = circuit_structure_claim_wires(
        &rt.sigma_native,
        &pt_w,
        mid_w,
        live_w,
        sig_w,
        &mlv_pw[1..1 + n_log_i]
            .iter()
            .map(|&(_, w)| w)
            .collect::<Vec<_>>(),
        &boolean_values,
        Some(&el_zc_rho_w[n_log_i..]),
        Some((asum_w, bsum_w)),
        zw,
        ow,
    );
    RealRegion {
        pub_base,
        n_query_pub,
        n_tail,
        n_mat_pub: mat_pub.len(),
        census: cen,
        jag_w,
        jag_sig_w: mp_sig_w.clone(),
        jag_row_w,
        zskip_w: outs[trace.squeezes[rt.zskip_fin][0]][0],
        n_ela_pub: ela_pub.len(),
        structure_claim_w,
        pt_w,
        el_zc_rho_w,
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
        cd_w,
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
            F256::new(public[at2 + 2 * li], public[at2 + 2 * li + 1]),
            *want,
            "L{li} enforced sum matches the native replica"
        );
    }
    let sp_base = at2 + 2 * rt.native_sums.len();
    assert_eq!(
        F256::new(public[sp_base], public[sp_base + 1]),
        rt.t_final_n,
        "the spine's t_r matches the native replay"
    );
    assert_eq!(
        public[sp_base + 2],
        rt.native_target,
        "the computed target is the native two-halves combination"
    );
    assert_eq!(
        public[sp_base + 3],
        rt.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    let inner_n = check_residual_publics(
        public,
        sp_base + 4,
        &rt.levels,
        &rt.geo,
        &rt.w_resid,
        rt.inner_pd_i.ch,
        &observed_f256(&rt.vals_rec, rt.yr_v_i, rt.yr_len),
        chals,
    );
    assert_eq!(
        inner_n, rt.t_final_n,
        "inner == t_r: the real-inner statement closes"
    );
    // The GKR/element/multipoint/anchor identities are COPY CONSTRAINTS —
    // no publics, no checker items; the proof itself carries them.
    let sig_base = sp_base + 4 + 2 * rt.levels.len() * rt.yr_len + 2;
    assert_eq!(
        public[sig_base], rt.lo.proof.wiring.gkr.s_sigma_eval,
        "the emitted sigma value is the proof's deferred evaluation"
    );
    let sa = flock_core::circuit::SigmaAssertion {
        rho: public[sig_base + 1..sig_base + 1 + rt.mu_i].to_vec(),
        nu: rt.lo.shape.circuit.cells().nu(),
        base_bits: rt.sigma_native.base_bits,
        masked_id_value: rt.mid_n,
        live_value: rt.live_n,
        value: public[sig_base],
        boolean_pins: rt.sigma_native.boolean_pins.clone(),
        element_constants: rt.sigma_native.element_constants.clone(),
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
        public[el_base], rt.el_run_n,
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
        public[mat_base], rt.mat_assert.alpha,
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
        public[mq], rt.mat_assert.target,
        "the in-circuit boolean lc replay ends at the assertion's target"
    );
    assert_eq!(
        mq + 1,
        mat_base + r.n_mat_pub,
        "the mat block walk is complete"
    );
    let ela_base = mat_base + r.n_mat_pub;
    assert_eq!(
        public[ela_base], rt.el_assert.alpha,
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
    let mut fq = ela_base + r.n_ela_pub;
    // The jagged assertion's value surfaces (the count win), in emission
    // order — each the deferred export's own raw claim value; the full
    // claims discharge against the child's layout at the root.
    {
        let mut expect_vals: Vec<F128> = rt.jag.rs.iter().map(|c| c.value).collect();
        for (combo, dense) in &rt.jag.groups {
            if let Some(c) = combo {
                expect_vals.push(c.value);
            }
            for (_, c) in dense {
                expect_vals.push(c.value);
            }
        }
        for (j, want) in expect_vals.iter().enumerate() {
            assert_eq!(
                public[fq + j],
                *want,
                "jagged claim value {j} matches the deferred export"
            );
        }
        fq += expect_vals.len();
    }
    assert_eq!(
        fq,
        r.pub_base + r.n_query_pub + r.n_tail,
        "the jagged publics close the region's tail"
    );
    r.n_query_pub + r.n_tail
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
        log_batch_size: pcs_batch_for(&union, LigeritoProfile::Fast),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch_for(&union, LigeritoProfile::Fast)),
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
#[derive(Clone)]
struct ChainShape {
    shape: flock_core::circuit::builder::CircuitShape,
    hash: flock_core::circuit::builder::SlotId,
    nu: usize,
}

/// The chain SHAPE per n_blocks, cached process-wide: the emission+finish
/// (~1.4 s at m32) is statement-independent — that is the digest pin — so
/// the tower's material proofs CLONE the cached shape (Registry + Circuit
/// memcpy, ~an order of magnitude cheaper) instead of re-emitting it.
/// `build_chain_proof`'s setup_ms honestly reflects whichever it paid.
fn chain_shape_cached(n_blocks: usize) -> std::sync::Arc<ChainShape> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Cache = Mutex<Vec<(usize, Arc<ChainShape>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, s)) = g.iter().find(|(k, _)| *k == n_blocks) {
        return s.clone();
    }
    let s = Arc::new(build_chain_shape(n_blocks));
    g.push((n_blocks, s.clone()));
    s
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

#[cfg(test)]
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
pub struct ChainProof {
    inner: MixedInner,
    h_start: [u32; 16],
    h_end: [u32; 16],
    /// What the leaf cost, split SETUP vs ONLINE — see [`Online`]. The
    /// LAST online iteration under steady repetition. Read by the in-file
    /// `#[test]` benches only.
    #[cfg_attr(not(test), allow(dead_code))]
    t: Online,
    /// One record per online iteration (1 + steady_reps of them).
    #[cfg_attr(not(test), allow(dead_code))]
    onlines: Vec<Online>,
}

/// A chain leaf. The SHAPE build is per-shape setup (statement-independent
/// — the digest pin), the WALK is per-statement and is the chain compute
/// itself, so it is reported apart from the proving phases.
pub fn build_chain_proof(cfg: TowerConfig, h_start: [u32; 16], n_blocks: usize) -> ChainProof {
    let t_shape = std::time::Instant::now();
    let cs: ChainShape = chain_shape_cached(n_blocks).as_ref().clone();
    let shape_ms = t_shape.elapsed().as_secs_f64() * 1e3;
    let (nu, hash) = (cs.nu, cs.hash);
    let t_setup = std::time::Instant::now();
    let blake_r1cs = chain_blake_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let setup_ms = shape_ms + t_setup.elapsed().as_secs_f64() * 1e3;

    // ONLINE, `1 + steady_reps()` iterations over the ONE shape: walk (the
    // chain compute itself), witgen, prove. Identical inputs, so every
    // iteration's outputs match and the last one ships.
    let reps = 1 + steady_reps();
    let mut onlines: Vec<Online> = Vec::with_capacity(reps);
    let mut fin = None;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let witness = cs.shape.run(&chain_vals(&h_start), &[]);
        let walk_ms = t0.elapsed().as_secs_f64() * 1e3;
        let union = UnionInstance::new(&cs.shape.registry, cs.shape.counts.clone());
        assert!(!union.has_element(), "a chain proof is boolean-only");
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            // The chain leaf's WORKLOAD inner: the tower's security level,
            // rate 1/2 (a 100-bit recursion carries a 100-bit leaf). The
            // batch is keyed by the SAME profile as the params — the old
            // Fast-keyed batch only worked because the Fast twins share
            // initial_k at these m's.
            profile: cfg.leaf_profile(),
            log_batch_size: pcs_batch_for(&union, cfg.leaf_profile()),
            num_lanes: union.commit_lanes(pcs_batch_for(&union, cfg.leaf_profile())),
            merkle_hash: HashKind::Blake3,
        };
        let t1 = std::time::Instant::now();
        let wit =
            blake3::generate_witness_batch_major_partial(witness.rows::<Blake3Gate>(hash), nu);
        let witgen_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = std::time::Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &cs.shape.circuit,
            &witness.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(wit, blake_lc)],
            Vec::new(),
            &mut ch,
        );
        let prove_ms = t2.elapsed().as_secs_f64() * 1e3;
        // `t0` opened before the walk, so this is the whole online span in
        // ONE timer — the honest per-leaf number, against which the phase
        // sum is only a lower bound.
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        onlines.push(Online {
            setup_ms,
            walk_ms,
            witgen_ms,
            prove_ms,
            wall_ms,
            ..Online::default()
        });
        fin = Some((witness, proof, commitment, pcs_params));
    }
    let (witness, proof, commitment, pcs_params) = fin.expect("one online iteration at least");
    let built = flock_core::circuit::builder::BuiltCircuit {
        shape: cs.shape,
        witness,
    };
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());

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
        t: *onlines.last().expect("one online iteration"),
        onlines,
    }
}

/// **Task 2's pin: the message-chain leaf, honest + the tamper matrix.**
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn chain_proof_message_chain_roundtrip_and_tampers() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0002);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());

    // Honest: build_chain_proof internally deferred-verifies, discharges
    // both assertion families and cross-checks h_end against the native
    // chain. Determinism of the statement: a second build from the same
    // h_start yields the same h_end.
    let cp = build_chain_proof(cfg, h_start, n_blocks);
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
    cp.inner
        .work
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
    let cfg = test_config();
    let mut rng = Rng(0xC4A1_0003);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let n_blocks = 256usize;
    let cp = build_chain_proof(cfg, h_start, n_blocks);
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
        cp.inner.work.boolean.as_ref().expect("boolean work").target,
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
    let cfg = test_config();
    let mut rng = Rng(0xC4A1_0005);
    let h_start: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp = build_chain_proof(cfg, h_start, 256);
    let ct = ChildTape::new(&cp.inner, DOMAIN);
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let b3_slot = cs.q.b3;
    let region = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        &ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape2 = sb.finish().expect("the chain child circuit builds");
    let hint_refs: Vec<&(dyn std::any::Any + Sync)> = hints
        .iter()
        .map(|h| h as &(dyn std::any::Any + Sync))
        .collect();
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
/// [`build_node_outer_app`]) consumes it exactly like a leaf outer; `acc` is
/// the folded chain accumulator the node carries up; `stmt_base` locates
/// the 8-word application-statement block (h_start, h_end) in `lo.public`.
// Several fields are read only by the in-file `#[test]` benches; the lib
// unit sees them write-only.
#[cfg_attr(not(test), allow(dead_code))]
pub struct FlNode {
    lo: LeafOuter,
    acc: flock_core::aggregate::Accumulator,
    stmt_base: usize,
    /// The published fold blocks' base: per group `[rho_col | rho_row |
    /// value]` — the accumulator claims a PARENT's lane fold connects to
    /// wire-to-wire (a prior's surface IS this published block).
    fold_pub_base: usize,
    h_start: [u32; 16],
    h_end: [u32; 16],
    /// What the FL cost, split SETUP vs ONLINE — see [`Online`]. The LAST
    /// online iteration under steady repetition; everything else in the
    /// builder is pin/check scaffolding.
    t: Online,
    /// One record per online iteration (1 + steady_reps of them).
    onlines: Vec<Online>,
}

/// **THE FIRST-LEVEL NODE.** k ADJACENT chain proofs (each segment starts
/// at the previous one's h_end) verified deferred in ONE outer circuit —
/// k chain-tape regions on shared slots — with their boolean + sigma
/// assertions folded k→1 in-circuit (THREE fold groups; the chain class
/// has no element side), THE ADJACENCY as a wire-to-wire copy constraint
/// per seam between the children's endpoint publics, and the combined span
/// (first h_start, last h_end) published as the node's own application
/// statement. The accumulator reassembles from the public segment alone
/// and discharges both groups. Every pin stays inside the builder (the
/// mvp9 precedent: the builder IS the test). Envelope snapping is the
/// scale step's job — this is the m22 dev shape.
/// The chain layout's jagged params — the count win's per-digest table
/// owner for the lane, rebuilt exactly as the opening verifier reads it.
#[cfg(test)]
fn chain_jagged_params(cp: &ChainProof) -> flock_core::pcs::jagged::JaggedParams {
    let u = UnionInstance::new(
        &cp.inner.built.shape.registry,
        cp.inner.built.shape.counts.clone(),
    );
    flock_core::pcs::jagged::JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        cp.inner.commitment.params.m - flock_core::pcs::LOG_PACKING,
    )
}

/// The chain BLAKE3 block R1CS per nu, cached process-wide: the ~21M-nnz
/// base is identical for every chain proof and every FL's chain-side fold
/// materials, and the tower bench used to build ten of them. Serves the
/// borrow-only sites; callers that STORE an R1CS (LeafOuter) still build
/// their own.
fn chain_blake_r1cs(nu: usize) -> std::sync::Arc<flock_core::r1cs::BlockR1cs> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Cache = Mutex<Vec<(usize, Arc<flock_core::r1cs::BlockR1cs>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, r)) = g.iter().find(|(k, _)| *k == nu) {
        return r.clone();
    }
    let r = Arc::new(blake3::build_block_r1cs(nu));
    g.push((nu, r.clone()));
    r
}

/// The FL's per-statement tape source, bare: ONE recording deferred verify
/// of a chain child. The pin/locate scaffolding is per-shape and lives in
/// [`ChildTape::new`]; this is what an online iteration re-pays (results
/// discarded — identical by determinism).
fn record_chain_child_verify(
    cp: &ChainProof,
    blake_lc: &dyn flock_core::lincheck::LincheckCircuit,
) {
    use flock_core::transcript_record::RecordingChallenger;
    let inner = &cp.inner;
    let union = UnionInstance::new(
        &inner.built.shape.registry,
        inner.built.shape.counts.clone(),
    );
    let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
    verifier::verify_ligerito_union_circuit_deferred(
        &union,
        &inner.built.shape.circuit,
        &inner.built.witness.public,
        &lcs,
        &inner.commitment,
        &inner.proof,
        &inner.pcs,
        &mut rec,
    )
    .expect("the chain child verifies (recorded)");
}

pub fn build_fl_node(cfg: TowerConfig, cp0: &ChainProof, cp1: &ChainProof) -> FlNode {
    build_fl_node_k(cfg, &[cp0, cp1])
}

/// The 2-ary first-level node: two adjacent chain proofs verified deferred
/// in ONE outer, their assertions folded 2→1 per group, adjacency as one
/// four-word seam, the app statement the combined span. The `cps` slice is
/// the arity LEVER, but today it is pinned to exactly two children — the
/// split-BLAKE slot assignment (`ChildSlots::new_env` sets `b3_alt` for
/// child 1 only) has no slots for a third child.
pub fn build_fl_node_k(cfg: TowerConfig, cps: &[&ChainProof]) -> FlNode {
    use flock_core::aggregate;
    use flock_core::matrix_fold::{FoldProof, MatrixClaim};
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

    const FL_DOMAIN: &[u8] = b"flock-chain-fl-node-v0";

    let k_ary = cps.len();
    assert_eq!(
        k_ary, 2,
        "split-BLAKE recursion supports exactly two children"
    );
    let cp0 = cps[0];
    let cp_last = cps[k_ary - 1];
    // Each child CONTINUES the chain: its h_start IS the previous h_end.
    for pair in cps.windows(2) {
        assert_eq!(pair[1].h_start, pair[0].h_end, "the segments are adjacent");
    }
    for cp in &cps[1..] {
        assert_eq!(
            cp0.inner.built.shape.circuit.digest(),
            cp.inner.built.shape.circuit.digest(),
            "one chain circuit digest, every segment"
        );
    }

    let registry = &cp0.inner.built.shape.registry;
    assert_eq!(registry.num_boolean(), 1, "one boolean type (blake3)");
    assert!(
        registry.element_types().is_empty(),
        "the chain class has no element side"
    );
    let bool_asserts: Vec<_> = cps
        .iter()
        .map(|cp| cp.inner.work.boolean.clone().expect("child boolean work"))
        .collect();
    let sigmas: Vec<_> = cps.iter().map(|cp| cp.inner.sigma.clone()).collect();

    // ---- the native fold: boolean + sigma, NO element groups, NO priors ----
    let blake_r1cs = chain_blake_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let el_mats: [flock_core::aggregate::ElementMatrices; 0] = [];
    let el_asserts: [(
        &UnionInstance<'_>,
        flock_core::element_r1cs::union::ElementAssertion,
    ); 0] = [];
    let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    // THE JAGGED GROUP (the count win): the chain children's W-claims fold
    // under the chain digest — the layout is a shape constant of the ONE
    // chain circuit, rebuilt here exactly as the opening verifier reads it.
    let chain_digest = cp0.inner.built.shape.circuit.digest();
    let chain_union_j = UnionInstance::new(registry, cp0.inner.built.shape.counts.clone());
    let chain_params_j = flock_core::pcs::jagged::JaggedParams::from_heights(
        &chain_union_j.jagged_heights(),
        chain_union_j.n_log(),
        cp0.inner.commitment.params.m - flock_core::pcs::LOG_PACKING,
    );
    let jags: Vec<_> = cps.iter().map(|cp| &cp.inner.work.jagged).collect();
    let jagged_p: Vec<aggregate::JaggedKeyProve<'_>> =
        vec![(chain_digest, &chain_params_j, jags.to_vec())];
    let jagged_v: Vec<aggregate::JaggedKeyVerify<'_>> = vec![(chain_digest, jags.to_vec())];
    let mut chp = FsChallenger::with_chained_blake3(FL_DOMAIN);
    let (agg, acc_p) = aggregate::prove_aggregate_classes_with_grinding(
        registry,
        &mats,
        &circs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        &[(&cp0.inner.built.shape.circuit, sigmas.iter().collect())],
        &jagged_p,
        &[],
        tower_fold_grinding(cfg),
        &mut chp,
    )
    .expect("the first-level fold proves");
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(FL_DOMAIN));
    let acc_v = aggregate::verify_aggregate_classes_with_grinding(
        registry,
        &bool_asserts,
        &el_asserts,
        &[(&cp0.inner.built.shape.circuit, sigmas.iter().collect())],
        &jagged_v,
        &[],
        &agg,
        tower_fold_grinding(cfg),
        &mut rec,
    )
    .expect("the first-level fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.per_element.is_empty(), "no element group accumulated");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_sigma(&[&cp0.inner.built.shape.circuit]),
        "the sigma group discharges against the ONE chain circuit"
    );
    assert_eq!(acc_v.jagged.len(), 1, "one jagged key: the chain layout");
    assert!(
        acc_v.discharge_jagged(&[(chain_digest, &chain_params_j)]),
        "the folded jagged entry discharges against the chain layout"
    );

    // The three folds' claim lists — no priors, so [fresh; k] each.
    let n_priors = 0usize;
    let bc: Vec<_> = bool_asserts.iter().map(|a| a.claims(registry)).collect();
    let fold_claims: Vec<Vec<MatrixClaim>> = vec![
        bc.iter().map(|c| c[0].0.clone()).collect(),
        bc.iter().map(|c| c[0].1.clone()).collect(),
        sigmas.iter().flat_map(|s| s.claims()).collect(),
    ];
    let fold_proofs: Vec<&FoldProof> = vec![&agg.folds[0].0, &agg.folds[0].1, &agg.sigma_folds[0]];
    assert_eq!(fold_claims[0][0].row.low.len(), 64, "fresh lagrange low");
    assert_eq!(fold_claims[0][0].col.low.len(), 64, "fresh z_partial low");
    assert_eq!(fold_claims[2][0].row.low.len(), 1, "sigma claims are eq");

    // ---- the fold tape, pinned op-for-op ----
    let t_shape = rec.shape();
    let ops = flatten_ops(t_shape.ops());
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    let n_uni = fold_claims.len() - 1;
    want.extend(fold_region_ops(cfg, &fold_claims[..n_uni]));
    // The sigma group binds per key now (wall 3): its label + digest
    // precede the fold, exactly as the jagged groups bind.
    want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
    want.push(Op::ObserveBytes(32));
    want.extend(fold_region_ops(cfg, &fold_claims[n_uni..]));
    // The jagged group rides the SAME tape after the uniform folds.
    let jagged_keys: Vec<([u8; 32], Vec<flock_core::matrix_fold::JaggedClaim>)> = vec![(
        chain_digest,
        jags.iter()
            .flat_map(|a| a.claims().into_iter().cloned())
            .collect(),
    )];
    want.extend(jagged_fold_region_ops(cfg, &jagged_keys));
    assert_eq!(ops, want.as_slice(), "the first-level fold tape shape");
    assert_eq!(
        rec.payloads()[0],
        registry.digest(),
        "bind: registry digest"
    );
    assert_eq!(rec.payloads()[1], vec![0u8], "bind: prior count 0");
    let (locs, vcur, ccur) = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let jfps: Vec<&flock_core::matrix_fold::FoldProof> = agg.jagged_folds.iter().collect();
    let jlocs = locate_and_pin_jagged_folds(
        &jagged_keys,
        &jfps,
        vals_rec,
        chals,
        rec.payloads(),
        &labeled_bytes_payloads(&ops, b"flock-aggregate-jagged-v0"),
        vcur,
        ccur,
    );
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    assert_eq!(outs[0], acc_v.per_type[0].0, "boolean A accumulator");
    assert_eq!(outs[1], acc_v.per_type[0].1, "boolean B accumulator");
    let (sig_digest, sig_claim) = acc_v.sigma.first().expect("sigma accumulated");
    assert_eq!(outs[2], *sig_claim, "sigma accumulator");
    assert_eq!(
        *sig_digest,
        cp0.inner.built.shape.circuit.digest(),
        "sigma keys by the chain circuit digest"
    );
    let jouts = replay_jagged_fold_endpoints(&jlocs, vals_rec, chals);
    assert_eq!(
        jouts[0], acc_v.jagged[0].1,
        "the jagged entry from located words"
    );

    // ---- the child tapes ----
    let tapes: Vec<ChildTape> = cps
        .iter()
        .map(|cp| ChildTape::new(&cp.inner, DOMAIN))
        .collect();
    assert!(tapes.iter().all(|t| t.el.is_none()), "chain children");

    // ---- the outer: k chain-tape regions + the fold region + adjacency ----
    {
        use crate::prover::UnionElementSlotInput;

        // The transcript is FORKED (the wiring runs on its own chain);
        // `merge_chain` splices the child's rows in at the fork point and
        // hands back one linear numbering plus the four cross-link wires.
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = merge_chain(
            t_shape.ops(),
            &t_shape.stream_words_duplex(FL_DOMAIN),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, chals);

        let env = envelope_shape();
        let split_b3 = tapes.len() == 2;
        let (fold_b3_primary_rows, b3_rows) = if split_b3 {
            let a = tapes[0].b3_rows;
            let b = tapes[1].b3_rows;
            let unsplit = (a + trace.rows.len()).max(b);
            let (on_a, balanced) = balance_extra_rows(a, b, trace.rows.len());
            if unsplit > (1usize << env.nu) {
                (Some(on_a), balanced)
            } else {
                (None, unsplit)
            }
        } else {
            (
                None,
                tapes.iter().map(|t| t.b3_rows).sum::<usize>() + trace.rows.len(),
            )
        };
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        // THE ENVELOPE (task 7b): a first-level node is an internal node's
        // CHILD, so its proof must carry the same geometry every other
        // envelope outer does — nu*, the canonical type set at counts*, the
        // padded public segment and the m* dense floor. Then a parent's walk
        // over an FL child is row-identical to its walk over an internal
        // child, which is what makes ONE internal circuit serve every level.
        assert!(
            nu2_content <= env.nu,
            "FL content nu {nu2_content} exceeds the envelope nu* {}",
            env.nu
        );
        let nu2 = env.nu;
        let t_build = std::time::Instant::now();
        let mut sb = ShapeBuilder::new(nu2);
        let spread_own2 = tapes.iter().map(|t| t.spread_w).max().expect("children");
        assert!(
            spread_own2 <= env.spread_w,
            "chain-child ladder depth {spread_own2} exceeds the envelope spread width {}",
            env.spread_w
        );
        let spread_w2 = env.spread_w;
        let mut cs = ChildSlots::new_env(&mut sb, nu2, &env);
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        // The chain-child regions are independent gate subgraphs (each
        // reads only its own tape's inputs; the fold region joins them
        // AFTER), so they are declared as islands and the fill plan
        // evaluates them concurrently. A cross-island read fails plan
        // compilation — the independence is checked, not assumed.
        let regions: Vec<_> = tapes
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let isl = sb.begin_island();
                let b3_slot = match (i, cs.q.b3_alt) {
                    (0, _) => cs.q.b3,
                    (1, Some(slot)) => slot,
                    (_, None) => cs.q.b3,
                    _ => panic!("split-BLAKE recursion supports exactly two children"),
                };
                let r = emit_child_region(
                    &mut sb,
                    &mut cs,
                    b3_slot,
                    t,
                    &mut vals,
                    &mut hints,
                    &mut consts,
                );
                sb.end_island(isl);
                r
            })
            .collect();
        let b3s = cs.q.b3;
        let macs = cs.macs;
        let mrs = cs.mrs;
        let (pfslot, pf_w) = regions[0].pf;
        let leslot = cs
            .le
            .iter()
            .find(|&&(n, _)| n == 8)
            .map(|&(_, s)| s)
            .expect("the child regions created the 8-lane leaf-eval slot");

        let iv_w = pack8(&crate::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [
            sb.fixed_public_input(iv_w[0]),
            sb.fixed_public_input(iv_w[1]),
        ];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(&ops);
        let (chain_outs, ww) = emit_fs_chain_partitioned(
            &mut sb,
            b3s,
            fold_b3_primary_rows.map(|n| {
                (
                    cs.q.b3_alt
                        .expect("a balanced fold chain needs the second BLAKE slot"),
                    n,
                )
            }),
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
            &cross,
        );
        emit_recorded_pow_checks(
            &mut sb,
            b3s,
            cs.q.pow,
            iv2,
            &ops,
            &trace,
            &stream,
            &chain_outs,
            &ww,
            &mut vals,
            &mut consts,
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
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
            false, // the jagged group follows on the same tape
        );
        let jfold_pubs = emit_jagged_fold_region(
            &mut sb,
            macs,
            mrs,
            pfslot,
            pf_w,
            &jlocs,
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );
        // THE POINTS-CONNECT (the count win's identity bind): every
        // absorbed claim surface in the jagged fold is a child-region
        // wire — the VALUE (the wire the anchor expect consumed), σ (the
        // child's anchor round squeezes), the row identities (z_col
        // wires / γ_pd squeezes / zw-ow constants), and the structural
        // words (tags, the shape header, Combo addresses) pinned to
        // shared constant publics the checker validates. With identity
        // AND value bound, the folded entry provably says "Ĵ at the
        // identity the child's verification determined equals the value
        // its anchor expect consumed" — a cooked-identity substitution
        // has nowhere to live.
        let mut jag_const_rec: Vec<(F128, usize)> = Vec::new();
        {
            let mut jag_consts: Vec<(F128, Wire)> = Vec::new();
            let mut cw_j = |sb: &mut ShapeBuilder,
                            vals: &mut Vec<F128>,
                            rec2: &mut Vec<(F128, usize)>,
                            v: F128|
             -> Wire {
                if let Some(&(_, w)) = jag_consts.iter().find(|&&(x, _)| x == v) {
                    return w;
                }
                vals.push(v);
                rec2.push((v, sb.public_len()));
                let w = sb.public_input();
                jag_consts.push((v, w));
                w
            };
            let loc = &jlocs[0];
            let mut ci = 0usize;
            for rk in &regions {
                for (li, &jw) in rk.jag_w.iter().enumerate() {
                    let cl = &loc.claims[ci];
                    sb.connect(wv(cl.val_v), jw);
                    for j in 0..loc.n_col {
                        sb.connect(wv(cl.col_v + j), rk.jag_sig_w[j]);
                    }
                    if cl.terms.is_empty() {
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut jag_const_rec,
                            F128::new(0, cl.row_pt.1 as u64),
                        );
                        sb.connect(wv(cl.row_scale_v - 1), tag);
                        // A FRESH claim is live: its zero-claim scale is 1.
                        sb.connect(wv(cl.row_scale_v), ow);
                        for j in 0..cl.row_pt.1 {
                            sb.connect(wv(cl.row_pt.0 + j), rk.jag_row_w[li][j]);
                        }
                    } else {
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut jag_const_rec,
                            F128::new(1, cl.terms.len() as u64),
                        );
                        sb.connect(wv(cl.terms[0].0 - 1), tag);
                        for (tj, &(cv, addr)) in cl.terms.iter().enumerate() {
                            sb.connect(wv(cv), rk.jag_row_w[li][tj]);
                            let aw = cw_j(
                                &mut sb,
                                &mut vals,
                                &mut jag_const_rec,
                                F128::new(addr as u64, 0),
                            );
                            sb.connect(wv(cv + 1), aw);
                        }
                    }
                    ci += 1;
                }
            }
            assert_eq!(ci, loc.claims.len(), "every jagged claim connected");
            // The group's shape header word binds too.
            let header_v = loc.hdr_v;
            let hw = cw_j(
                &mut sb,
                &mut vals,
                &mut jag_const_rec,
                F128::new(loc.k_row as u64, loc.claims.len() as u64),
            );
            sb.connect(wv(header_v), hw);
        }

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
            // Native pre-asserts, then the wire connects — all static
            // Product-GKR evaluations, boolean points + z_partial lows.
            let native_structure = tk.sigma_native.claims();
            for (j, claim) in native_structure.iter().enumerate() {
                assert_eq!(
                    &fold_claims[2][n_priors + native_structure.len() * k + j],
                    claim,
                    "circuit-structure claim {j}"
                );
            }
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
            assert_eq!(
                fold_claims[0][n_priors + k].value,
                tk.bool_assert.evals[0].0
            );
            assert_eq!(
                fold_claims[1][n_priors + k].value,
                tk.bool_assert.evals[0].1
            );

            // Circuit structure: every native claim is fully wire-bound.
            for (j, (row_w, col_w, value_w)) in rk.structure_claim_w.iter().enumerate() {
                let cl = &locs[2].claims[n_priors + rk.structure_claim_w.len() * k + j];
                sb.connect(wv(cl.row_low_v), ow);
                sb.connect(wv(cl.col_low_v), ow);
                assert_eq!(cl.row_pt_n, row_w.len());
                assert_eq!(cl.col_pt_n, col_w.len());
                for (j, &w) in row_w.iter().enumerate() {
                    sb.connect(wv(cl.row_pt_v + j), w);
                }
                for (j, &w) in col_w.iter().enumerate() {
                    sb.connect(wv(cl.col_pt_v + j), w);
                }
                sb.connect(wv(cl.value_v), *value_w);
            }
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
            assert_eq!(rk.mat_eval_w.len(), 1, "chain child Boolean type count");
            sb.connect(wv(locs[0].claims[n_priors + k].value_v), rk.mat_eval_w[0].0);
            sb.connect(wv(locs[1].claims[n_priors + k].value_v), rk.mat_eval_w[0].1);
            // Fold B's lagrange lows are fold A's — one published copy
            // binds both.
            for j in 0..locs[0].claims[n_priors + k].row_low_n {
                sb.connect(
                    wv(locs[1].claims[n_priors + k].row_low_v + j),
                    wv(locs[0].claims[n_priors + k].row_low_v + j),
                );
            }
        }

        // ---- THE ADJACENCY: each h_end == the next h_start, wire to wire ----
        // The chain statement is 11 words: [iv0, iv1, params, h_start x4 |
        // h_end x4 published last]. The children's publics are witness
        // wires here, so adjacency is four copy constraints per seam, and
        // the node's own application statement is the combined span.
        for rk in &regions {
            assert_eq!(rk.child_pub_w.len(), 11, "the chain statement is 11 words");
        }
        for pair in regions.windows(2) {
            for j in 0..4 {
                sb.connect(pair[0].child_pub_w[11 - 4 + j], pair[1].child_pub_w[3 + j]);
            }
        }

        // THE INHERITABLE ACCUMULATOR: per fold the deltas + the claim
        // `[rho_col | rho_row | value]`. This is the surface a PARENT's
        // chain lane connects to as its priors, so under the envelope it
        // rides the reserved ACC_CHAIN block (the FL folds the CHAIN
        // registry) — a constant index, the same one at which an internal
        // child exposes its own lane's claims. Off-envelope it publishes
        // inline, as before.
        let mut acc_chain_w: Vec<Wire> = Vec::new();
        for fp in fold_pubs.iter().chain(&jfold_pubs) {
            acc_chain_w.push(fp.live);
            acc_chain_w.extend_from_slice(&fp.rho_col);
            acc_chain_w.extend_from_slice(&fp.rho_row);
            acc_chain_w.push(fp.value);
        }
        let fold_pub_base = env_acc_chain_base(&env);
        // The value-binding publics stay in the BODY: nothing above reads
        // them, they only bind the claim values this outer folded.
        for k in 0..k_ary {
            sb.publish(wv(locs[0].claims[n_priors + k].value_v));
            sb.publish(wv(locs[1].claims[n_priors + k].value_v));
        }
        // THE APPLICATION STATEMENT: the combined span (the first child's
        // h_start, the last child's h_end). counts* + publics*: an FL node
        // declares the same count vector and segment length every other
        // envelope outer does, and both the app block and the accumulator
        // claims ride the envelope's fixed TAIL.
        let app_w: Vec<Wire> = (0..4)
            .map(|j| regions[0].child_pub_w[3 + j])
            .chain((0..4).map(|j| regions[k_ary - 1].child_pub_w[11 - 4 + j]))
            .collect();
        let stmt_base = {
            pad_envelope_counts(
                &mut sb,
                &cs.q,
                &cs.env_cache(),
                &env,
                zw,
                &mut hints,
                &mut vals,
                &mut consts,
                &EnvTail {
                    acc_chain: &acc_chain_w,
                    app: &app_w,
                    ..EnvTail::default()
                },
            );
            env_app_base(&env)
        };
        let shape2 = sb.finish().expect("the first-level node circuit builds");
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> = hints
            .iter()
            .map(|h| h as &(dyn std::any::Any + Sync))
            .collect();
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
            if let Some(slot) = cs.q.b3_alt {
                assert_eq!(
                    walk.rows::<Blake3Gate>(slot),
                    fill.rows::<Blake3Gate>(slot),
                    "fill plan: second b3 rows"
                );
            }
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
            assert_eq!(
                walk.rows::<PowMaskGate>(cs.q.pow),
                fill.rows::<PowMaskGate>(cs.q.pow),
                "fill plan: pow rows"
            );
            let family_slot = cs.q.family.expect("family-H slot");
            assert_eq!(
                walk.rows::<FamilyTransposeTileGate>(family_slot),
                fill.rows::<FamilyTransposeTileGate>(family_slot),
                "fill plan: family-H rows"
            );
        }
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
        // Per-SHAPE prover materials, hoisted above the online loop — BLAKE3
        // for BOTH the Merkle trees and the FS chain, so the node is
        // RECURSABLE (both recorded gotchas).
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = cfg.outer_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: outer_lanes(&union2, pcs_batch_for(&union2, pf)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs2 = blake3::build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_r1cs2 = BitSpreadTable::new(spread_w2).build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        let pow_r1cs2 = PowMaskTable.build_block_r1cs(nu2);
        let pow_lc2 = pow_r1cs2.csc_lincheck_circuit();
        let family_slot = cs.q.family.expect("family-H slot");
        let family_r1cs2 = FamilyTransposeTileTable::build_block_r1cs(nu2);
        let family_lc2 = family_r1cs2.csc_lincheck_circuit();
        // ONLINE, `1 + steady_reps()` iterations over the ONE shape: tapes
        // (the recording verifies, re-run with results discarded — identical
        // by determinism), the walk (fill plan), witness assembly, prove,
        // verify. The checker asserts re-run too — they read publics only.
        let reps = 1 + steady_reps();
        let mut onlines: Vec<Online> = Vec::with_capacity(reps);
        let mut fin = None;
        for _ in 0..reps {
            let t_tapes = std::time::Instant::now();
            for cp in cps {
                record_chain_child_verify(cp, blake_lc);
            }
            let tapes_ms_i = t_tapes.elapsed().as_secs_f64() * 1e3;
            let t_run = std::time::Instant::now();
            // DEFERRED: rows and publics only — the element witnesses are never
            // packed, and the assembly below feeds the prover from the rows.
            let mut built2 = shape2.run_filled_deferred(&fill_plan, &vals, &hint_refs);
            let run_ms = t_run.elapsed().as_secs_f64() * 1e3;

            // Child checkers (each child's whole deferred-verifier statement
            // against its own native replicas), then the fold checker + the
            // accumulator reassembled from publics, then the app statement.
            let mut region_end = 0usize;
            for (tk, rk) in tapes.iter().zip(&regions) {
                let consumed = check_child_region(&built2.public, tk, rk);
                assert!(
                    region_end <= rk.pub_base && rk.pub_base + consumed <= fold_pub_base,
                    "the regions' public blocks are disjoint and ordered"
                );
                region_end = rk.pub_base + consumed;
            }
            // ACC_CHAIN keeps the un-keyed entry layout: the lane's registry
            // role has ONE key (the chain circuit), so nothing to disambiguate.
            let (rebuilt, _, _) = check_fold_publics(
                &built2.public,
                fold_pub_base,
                &locs,
                &alpha_recs,
                locs.len(),
            );
            for (r, o) in rebuilt.iter().zip(&outs) {
                assert_eq!(r, o, "published fold output == located native output");
            }
            let jag_pub_at =
                fold_pub_base + locs.iter().map(|l| 2 + l.k_col + l.k_row).sum::<usize>();
            let (jrebuilt, _, _) =
                check_jagged_fold_publics(&built2.public, jag_pub_at, &jlocs, false);
            assert_eq!(
                jrebuilt[0], jouts[0],
                "published jagged entry == located native"
            );
            let acc_pub = aggregate::Accumulator {
                registry_digest: registry.digest(),
                per_type: vec![(rebuilt[0].clone(), rebuilt[1].clone())],
                per_element: Vec::new(),
                sigma: vec![(cp0.inner.built.shape.circuit.digest(), rebuilt[2].clone())],
                jagged: vec![(chain_digest, jrebuilt[0].clone())],
            };
            assert_eq!(
                acc_pub, acc_v,
                "the Accumulator, reassembled from the public segment alone"
            );
            assert!(
                acc_pub.discharge(&mats)
                    && acc_pub.discharge_sigma(&[&cp0.inner.built.shape.circuit])
                    && acc_pub.discharge_jagged(&[(chain_digest, &chain_params_j)]),
                "the public-segment accumulator discharges all three groups"
            );
            for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                assert_eq!(built2.public[lam_base + i], v, "λ const {i}");
            }
            for &(v, idx) in &jag_const_rec {
                assert_eq!(built2.public[idx], v, "jagged shared constant public");
            }
            // THE APPLICATION STATEMENT: the published span is (h_start of the
            // first chain, h_end of the last) — the combined segment.
            for j in 0..4 {
                assert_eq!(
                    built2.public[stmt_base + j],
                    pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
                    "node statement: h_start is the first child's"
                );
                assert_eq!(
                    built2.public[stmt_base + 4 + j],
                    pack4(cp_last.h_end[4 * j..4 * j + 4].try_into().unwrap()),
                    "node statement: h_end is the last child's"
                );
            }
            assert_eq!(
                cp_last.h_end,
                native_chain(
                    &cp0.h_start,
                    cps.iter().map(|cp| cp.inner.built.shape.counts[0]).sum(),
                ),
                "the combined span IS the concatenated chain"
            );

            // Everything from here to the prove is WITNESS ASSEMBLY — packing
            // the walk's rows into the union's slot inputs. It is per-statement
            // (online), so it gets its own timer rather than hiding inside the
            // shape build or the prove.
            // Recreated per online iteration — the spread closure consumes it.
            let spread_ty2 = BitSpreadTable::new(spread_w2);
            let pow_ty2 = PowMaskTable;
            let t_asm = std::time::Instant::now();
            // THE COPY-FREE ASSEMBLY, the node's path: the boolean drivers pack
            // straight into the union's slot blocks inside the prove (live rows
            // only under elide) — no capacity-sized intermediates, no memcpy.
            // The rows are hoisted to owned Vecs because the closures must be
            // Send and `built2.rows` hands out `dyn Any`-backed borrows.
            let b3_declared: Vec<_> = std::iter::once(cs.q.b3).chain(cs.q.b3_alt).collect();
            let b3_rows2: Vec<_> = b3_declared
                .iter()
                .map(|&s| (s, built2.rows::<Blake3Gate>(s).to_vec()))
                .collect();
            let swap_rows2 = built2.rows::<SwapGate>(cs.q.swap).to_vec();
            let spread_rows2 = built2.rows::<BitSpreadGate>(cs.q.spread).to_vec();
            let pow_rows2 = built2.rows::<PowMaskGate>(cs.q.pow).to_vec();
            let family_rows2 = built2.rows::<FamilyTransposeTileGate>(family_slot).to_vec();
            let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
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
                (
                    shape2.registry_slot(cs.q.pow),
                    UnionSlotProverInput::in_place(
                        move |dst| pow_ty2.generate_witness_batch_major_into(&pow_rows2, dst),
                        pow_lc2,
                    ),
                ),
                (
                    shape2.registry_slot(family_slot),
                    UnionSlotProverInput::in_place(
                        move |dst| {
                            FamilyTransposeTileTable::generate_witness_batch_major_into(
                                &family_rows2,
                                dst,
                            )
                        },
                        family_lc2,
                    ),
                ),
            ];
            bslots.extend(b3_rows2.into_iter().map(|(s, rows)| {
                (
                    shape2.registry_slot(s),
                    UnionSlotProverInput::in_place(
                        move |dst| {
                            blake3::generate_witness_batch_major_partial_into(&rows, nu2, dst)
                        },
                        b3_lc2,
                    ),
                )
            }));
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
                (shape2.registry_slot(cs.q.swap), swap_lc2),
                (shape2.registry_slot(cs.q.spread), spread_lc2),
                (shape2.registry_slot(cs.q.pow), pow_lc2),
                (shape2.registry_slot(family_slot), family_lc2),
            ];
            lco.extend(b3_declared.iter().map(|&s| {
                (
                    shape2.registry_slot(s),
                    b3_lc2 as &dyn flock_core::lincheck::LincheckCircuit,
                )
            }));
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
            onlines.push(Online {
                setup_ms: build_ms,
                tapes_ms: tapes_ms_i,
                walk_ms: run_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms: verify_ms2,
                wall_ms: 0.0,
            });
            fin = Some((built2, oproof, ocommit, acc_pub));
        }
        let (built2, oproof, ocommit, acc_pub) = fin.expect("one online iteration");
        let (swap_ri, spread_ri, pow_ri, family_ri) = (
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
            shape2.registry_slot(cs.q.pow),
            shape2.registry_slot(family_slot),
        );
        let b3_ris = std::iter::once(cs.q.b3)
            .chain(cs.q.b3_alt)
            .map(|s| shape2.registry_slot(s))
            .collect();
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
                pow_r1cs: pow_r1cs2,
                family_r1cs: family_r1cs2,
                b3_slots: b3_ris,
                swap_slot: swap_ri,
                spread_slot: spread_ri,
                pow_slot: pow_ri,
                family_slot: family_ri,
            },
            acc: acc_pub,
            stmt_base,
            fold_pub_base,
            h_start: cp0.h_start,
            h_end: cp_last.h_end,
            t: *onlines.last().expect("one online iteration"),
            onlines,
        }
    }
}

/// **The first-level node's pin, through the builder** (converted-first:
/// the test IS [`build_fl_node`]'s original body; every assert lives inside
/// the builder now, the wrapper re-checks the statement surface).
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn first_level_node_two_chains_fold_and_adjacency() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0004);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let fl = build_fl_node(cfg, &cp0, &cp1);
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
        bincode::serialize(&fl.lo.proof)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
    );
}

/// **THE ENVELOPE CONTENT PROBE** — the m\* headroom question: the FL's and
/// the internal node's UNFLOORED content (dense_words / content dense_m)
/// under free counts, against the m\*28 cap (2^21 packed words). The
/// per-type breakdown (used_cols × rows, descending) is the diet map if the
/// gap needs closing. `CHAIN_BLOCKS` sizes the leaves (the real question is
/// 262144 = m32); `TOWER_PROFILE=slim` for the envelope.
#[test]
#[ignore] // Heavy at m32 — four chain proofs, two FLs, one node.
fn envelope_content_probe() {
    let cfg = test_config();
    let n_blocks: usize = std::env::var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut rng = Rng(0xC4A1_00CE);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..4 {
        let cp = build_chain_proof(cfg, h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fl0 = build_fl_node(cfg, &cps[0], &cps[1]);
    let fl1 = build_fl_node(cfg, &cps[2], &cps[3]);
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cps[0]);
    let node = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: &cps[0].inner.built.shape.circuit,
            params: &chain_jp,
            priors: &[&fl0.acc, &fl1.acc],
            claims_base: fl0.fold_pub_base,
        }),
        None,
    );
    println!(
        "\nENVELOPE CONTENT PROBE — {n_blocks} compressions/leaf, profile {:?}\n  \
         m28 cap = {} words | m29 cap = {} words",
        cfg.outer_profile(),
        1usize << (28 - 7),
        1usize << (29 - 7),
    );
    for (name, lo) in [("FL", &fl0.lo), ("internal", &node.lo)] {
        let u = UnionInstance::new(&lo.shape.registry, lo.shape.counts.clone());
        let dw = u.dense_words();
        println!(
            "  {name}: dense_words {dw} = {:.1}% of m28 cap | content dense_m {} | floored m {}",
            100.0 * dw as f64 / (1u64 << 21) as f64,
            u.dense_m(),
            outer_union(&lo.shape.registry, lo.shape.counts.clone()).dense_m(),
        );
        // The diet map: per-type committed words, descending.
        let mut per: Vec<(usize, usize, usize, usize)> = lo
            .shape
            .registry
            .types()
            .iter()
            .zip(&lo.shape.counts)
            .enumerate()
            .map(|(i, (ty, &n_t))| {
                let cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
                (cols * n_t, i, cols, n_t)
            })
            .collect();
        per.sort_by_key(|p| std::cmp::Reverse(p.0));
        for &(words, i, cols, rows) in per.iter().take(8) {
            println!(
                "    type {i:2}: {words:>8} words ({cols:3} cols x {rows:6} rows) = {:.1}%",
                100.0 * words as f64 / dw as f64
            );
        }
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
    assert_eq!(
        gather.len(),
        cells.num_gate_slots(),
        "one gather per gate slot"
    );
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
        acc, vals_rec[fgs_v],
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

/// Wire identities for every claim emitted by `SigmaAssertion::claims`, in
/// exactly the same order. The accumulator's circuit-structure table binds
/// Product-GKR's masked-ID/live/sigma evaluations, Boolean count-prefix
/// values, and element affine-constant strips under one child digest.
#[allow(clippy::too_many_arguments)]
fn circuit_structure_claim_wires(
    sigma: &flock_core::circuit::SigmaAssertion,
    gkr_point: &[Wire],
    masked_id_w: Wire,
    live_w: Wire,
    sigma_w: Wire,
    boolean_point: &[Wire],
    boolean_values: &[(usize, Wire)],
    element_point: Option<&[Wire]>,
    element_values: Option<(Wire, Wire)>,
    zw: Wire,
    ow: Wire,
) -> Vec<(Vec<Wire>, Vec<Wire>, Wire)> {
    let bit = |b: usize| if b == 0 { zw } else { ow };
    let selector = |plane: usize| -> [Wire; 3] {
        [bit(plane & 1), bit((plane >> 1) & 1), bit((plane >> 2) & 1)]
    };
    let mut base_point = gkr_point[sigma.nu..].to_vec();
    base_point.resize(sigma.base_bits, zw);
    let mut out = Vec::new();
    for (plane, value_w) in [(0, masked_id_w), (1, live_w), (2, sigma_w)] {
        let mut col = base_point.clone();
        col.extend_from_slice(&selector(plane));
        out.push((gkr_point[..sigma.nu].to_vec(), col, value_w));
    }
    assert_eq!(
        boolean_values.len(),
        sigma.boolean_pins.len(),
        "Boolean pin wires"
    );
    for ((type_index, point, _), (wire_type, value_w)) in sigma
        .boolean_pins
        .iter()
        .zip(boolean_values.iter().copied())
    {
        assert_eq!(*type_index, wire_type, "Boolean pin slot order");
        assert_eq!(point.len(), boolean_point.len(), "Boolean pin point width");
        let mut col: Vec<Wire> = (0..sigma.base_bits)
            .map(|j| bit((type_index >> j) & 1))
            .collect();
        col.extend_from_slice(&selector(5));
        out.push((boolean_point.to_vec(), col, value_w));
    }
    if let Some((point, _, _)) = &sigma.element_constants {
        let point_w = element_point.expect("element structure point wires");
        assert_eq!(point.len(), point_w.len(), "element constant point width");
        let (a_w, b_w) = element_values.expect("element constant value wires");
        for (plane, value_w) in [(3, a_w), (4, b_w)] {
            let mut col = point_w.to_vec();
            col.resize(sigma.base_bits, zw);
            col.extend_from_slice(&selector(plane));
            out.push((vec![zw; sigma.nu], col, value_w));
        }
    }
    assert_eq!(
        out.len(),
        sigma.claims().len(),
        "structure claim wire count"
    );
    out
}

/// `seed * product_j eq(left[j], right[j])`, chunked through the shared
/// prefix-product gate.  `PrefixGate` uses the characteristic-two identity
/// `eq(a,b) = 1 + a + b` for Boolean `b`; every right-hand coordinate used
/// here is either a fixed prefix bit or a Fiat--Shamir wire constrained by
/// the transcript circuit.
fn emit_eq_product(
    sb: &mut ShapeBuilder,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    seed: Wire,
    left: &[Wire],
    right: &[Wire],
    zw: Wire,
    ow: Wire,
) -> Wire {
    assert_eq!(left.len(), right.len(), "eq-product arity");
    let mut acc = seed;
    for (aa, bb) in left.chunks(pf_w).zip(right.chunks(pf_w)) {
        let mut inputs = vec![acc];
        inputs.extend_from_slice(aa);
        inputs.extend(std::iter::repeat_n(zw, pf_w - aa.len()));
        inputs.extend_from_slice(bb);
        inputs.extend(std::iter::repeat_n(zw, pf_w - bb.len()));
        inputs.push(ow);
        acc = sb.gate(pfslot, &inputs)[0];
    }
    acc
}

fn prefix_bit_wires(bits: usize, n: usize, zw: Wire, ow: Wire) -> Vec<Wire> {
    (0..n)
        .map(|j| if (bits >> j) & 1 == 0 { zw } else { ow })
        .collect()
}

/// The fourth output of `SpineGate` is the same `acc + x*y` primitive as
/// `MacGate`.  Assertion checks use this existing, lower-occupancy slot so
/// they do not push the recursion envelope's main MAC slot over `2^nu`.
fn assertion_mac(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    acc: Wire,
    x: Wire,
    y: Wire,
    zw: Wire,
) -> Wire {
    sb.gate(spine, &[zw, zw, zw, acc, zw, zw, x, y, zw])[3]
}

/// Enforce the scalar-only half of `MatrixAssertion::check_reported`.
/// The matrix evaluations themselves are fold claims and eventually
/// discharge against the digest-keyed matrices; this relation binds those
/// values to the transcript-derived lincheck target inside the recursive
/// circuit.
#[allow(clippy::too_many_arguments)]
fn emit_boolean_reported_check(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    registry: &crate::schedule::Registry,
    alpha_w: Wire,
    x_inner_w: &[Wire],
    rr_w: &[Wire],
    z_partial_w: &[Wire],
    beta_w: &[Option<Wire>],
    eval_w: &[(Wire, Wire)],
    target_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    use flock_core::zerocheck::K_SKIP;

    assert_eq!(eval_w.len(), registry.num_boolean(), "Boolean eval count");
    assert_eq!(beta_w.len(), registry.num_boolean(), "Boolean beta count");
    assert_eq!(z_partial_w.len(), 1usize << K_SKIP, "Boolean low weight");
    let mut acc = zw;
    for (t, ((ty, layout), &(va_w, vb_w))) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(eval_w)
        .enumerate()
    {
        let inner = ty.k_log - K_SKIP;
        assert!(inner <= x_inner_w.len() && inner <= rr_w.len());
        let row_prefix = prefix_bit_wires(layout.prefix, x_inner_w.len() - inner, zw, ow);
        let col_prefix = prefix_bit_wires(layout.prefix, rr_w.len() - inner, zw, ow);
        let w_t = emit_eq_product(
            sb,
            pfslot,
            pf_w,
            ow,
            &x_inner_w[inner..],
            &row_prefix,
            zw,
            ow,
        );
        let p_t = emit_eq_product(sb, pfslot, pf_w, ow, &rr_w[inner..], &col_prefix, zw, ow);
        let ab = assertion_mac(sb, spine, vb_w, alpha_w, va_w, zw);
        let wp = assertion_mac(sb, spine, zw, w_t, p_t, zw);
        let term = assertion_mac(sb, spine, zw, wp, ab, zw);
        acc = assertion_mac(sb, spine, acc, term, ow, zw);

        match (ty.const_pin, beta_w[t]) {
            (Some(col), Some(beta)) => {
                let high = col >> K_SKIP;
                let high_bits = prefix_bit_wires(high, inner, zw, ow);
                let w_col = emit_eq_product(
                    sb,
                    pfslot,
                    pf_w,
                    z_partial_w[col & ((1usize << K_SKIP) - 1)],
                    &rr_w[..inner],
                    &high_bits,
                    zw,
                    ow,
                );
                let pin = assertion_mac(sb, spine, zw, p_t, beta, zw);
                acc = assertion_mac(sb, spine, acc, pin, w_col, zw);
            }
            (None, None) => {}
            _ => panic!("const-pin challenge schedule does not match the registry"),
        }
    }
    sb.connect(acc, target_w);
}

/// Enforce the scalar-only half of `ElementAssertion::check_reported`.
/// As on the Boolean side, the per-slot A/B values are separately folded
/// against static matrices; this equation binds their weighted combination
/// to the transcript-derived element-lincheck target.
#[allow(clippy::too_many_arguments)]
fn emit_element_reported_check(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    union: &UnionInstance<'_>,
    alpha_w: Wire,
    r_con_w: &[Wire],
    r_col_w: &[Wire],
    z_eval_w: Wire,
    eval_w: &[(Wire, Wire)],
    target_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    let layouts = union.element_slot_layout();
    assert_eq!(eval_w.len(), layouts.len(), "element eval count");
    let nu = union.n_log();
    let mut acc = zw;
    for (layout, &(va_w, vb_w)) in layouts.iter().zip(eval_w) {
        let kappa = layout.kappa;
        assert!(kappa <= r_con_w.len() && kappa <= r_col_w.len());
        let bits = prefix_bit_wires(layout.region_prefix(nu), r_con_w.len() - kappa, zw, ow);
        let w_r = emit_eq_product(sb, pfslot, pf_w, ow, &r_con_w[kappa..], &bits, zw, ow);
        let w_col = emit_eq_product(sb, pfslot, pf_w, ow, &r_col_w[kappa..], &bits, zw, ow);
        let ab = assertion_mac(sb, spine, va_w, alpha_w, vb_w, zw);
        let wp = assertion_mac(sb, spine, zw, w_r, w_col, zw);
        let term = assertion_mac(sb, spine, zw, wp, ab, zw);
        acc = assertion_mac(sb, spine, acc, term, ow, zw);
    }
    let rhs = assertion_mac(sb, spine, zw, acc, z_eval_w, zw);
    sb.connect(rhs, target_w);
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
    let eq_slot_w = emit_eq_prefix(
        sb,
        macs,
        &pt_w[nu_c..],
        gather_w.len() + n_pub_slots,
        zw,
        ow,
    );
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
    trace: crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: flock_core::transcript_record::Stream,
    bytes: Vec<u8>,
    /// The fork's four cross-link wires ([`MergedChain::cross`]).
    cross: Vec<Option<(usize, usize)>>,
    b3_rows: usize,
    spread_w: usize,
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
    native_sums: Vec<F256>,
    n_pd: usize,
    /// Packed-direct claim carrying the element lincheck's `z_eval`.
    z_ix: Option<usize>,
    /// The child cell space's public-slot count — the recombination's tail.
    n_pub_slots_c: usize,
    n_p: usize,
    // the boolean PIOP's round ordinals, located with fins ((ch, fin) pairs)
    zc_rounds_b: Vec<(usize, usize)>,
    outer_b: (usize, usize),
    bl_alpha: (usize, usize),
    betas_b: Vec<(usize, usize)>,
    zc_finals_v: usize,
    lc_msg_vs: Vec<usize>,
    lc_rounds_b: Vec<(usize, usize)>,
    eps_n: Vec<F128>,
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
    /// The two ring-switch regions: `(s_hat_v, r_dprime finalization,
    /// r_dprime challenge)`, plus each batching coefficient's location in
    /// their shared vector squeeze. These are the family-H source wires.
    rs_recs: Vec<(usize, usize, usize)>,
    rs_gam_fins: Vec<(usize, usize)>,
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
    t_final_n: F256,
    anc_end_n: F128,
    mid_n: F128,
    live_n: F128,
    mu_i: usize,
    // anchor-expect geometry — statement constants of the inner shape
    n_log_i: usize,
    k_cols_i: usize,
    m_mp2: usize,
    bounds_i: Vec<(u64, u64, u32)>,
    #[allow(dead_code)] // Run start columns — the run-weight era's consumer; kept as shape data.
    run_y0: Vec<usize>,
    #[allow(dead_code)] // The complement run — likewise.
    comp_ix: usize,
    x_ab_n: Vec<F128>,
    x_c_n: Vec<F128>,
    groups_ix: Vec<Vec<usize>>,
    /// Derived pd claim points (merged-open v1) — see [`RealTape::pd_pts`].
    pd_pts: Vec<Vec<F128>>,
    /// The deferred verify's jagged-layout export (the count win): the
    /// independent reference for the W-value publics the region publishes
    /// instead of rebuilding — tied member-for-member to the native expect
    /// replica in the constructor.
    jag: flock_core::matrix_fold::JaggedAssertion,
}

impl<'p> ChildTape<'p> {
    fn new(inner: &'p MixedInner, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

        let built = &inner.built;
        let proof = &inner.proof;
        let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
        let blake_r1cs = blake3::build_block_r1cs(inner.nu);
        let blake_lc = blake_r1cs.csc_lincheck_circuit();
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
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
        let ops: Vec<Op> = flatten_ops(t_shape.ops()).to_vec();
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
        } else {
            assert!(
                elzc_l.is_empty() && el_l.is_empty(),
                "a boolean-only tape carries NO element region"
            );
        }
        // THE FORKED ORDER: the wiring argument's chain is spliced in at the
        // fork point, so its region precedes the boolean PIOP's.
        assert!(
            gkr_l[0] < zc_l[0],
            "the wiring fork precedes the boolean PIOP"
        );
        assert_eq!(gkr_l.len(), 1, "one batched wiring GKR");
        assert_eq!(mo_l.len(), 1, "one merged open");
        assert_eq!(
            rs_l.len(),
            2,
            "rs x 2 — one ab/c pair for the boolean class"
        );
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
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
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
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
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
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
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
                    let mut rho_i = i + 2;
                    while matches!(ops[rho_i], Op::Pow { .. }) {
                        rho_i += 1;
                    }
                    assert!(matches!(ops[rho_i], Op::SqueezeScalar), "round rho");
                    let (_, rc2) = vc_at(rho_i);
                    let rho = chals[rc2];
                    rrecs.push((gv, fin_at(rho_i)));
                    i = rho_i + 1;
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
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
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
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
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
                let mut squeeze_i = i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                assert!(matches!(ops[squeeze_i], Op::SqueezeScalar), "el zc rho");
                zc_rounds.push((gv, fin_at(squeeze_i), vc_at(squeeze_i).1));
                i = squeeze_i + 1;
            }
            let (eab_v, _) = vc_at(i);
            for _ in 0..3 {
                assert!(matches!(ops[i], Op::ObserveScalar), "el zc final");
                i += 1;
            }
            assert_eq!(i, el_l[0], "the lc label follows the finals");
            i += 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(matches!(ops[i], Op::SqueezeScalar), "el lc alpha");
            let (alpha_fin, alpha_ch) = (fin_at(i), vc_at(i).1);
            i += 1;
            let mut lc_rounds = Vec::new();
            while matches!(ops[i], Op::ObserveScalar) && matches!(ops[i + 1], Op::ObserveScalar) {
                let mut squeeze_i = i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                let (gv, _) = vc_at(i);
                lc_rounds.push((gv, fin_at(squeeze_i), vc_at(squeeze_i).1));
                i = squeeze_i + 1;
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

        // ---- the merged open: rs x 2, then PD values, then one coefficient vector ----
        let (pd_recs, mp_val_v, rs_recs, rs_gam_ch, rs_gam_fins) = {
            let mut i = mo_l[0] + 1;
            let mut rs_recs: Vec<(usize, usize, usize)> = Vec::new();
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
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeSlice(7)), "r_dprime");
                rs_recs.push((sv, fin_at(i), vc_at(i).1));
                i += 1;
            }
            // Packed-direct claims contribute just their values.  Their
            // coefficients share the vector squeeze with both RS claims.
            let mut pd_recs: Vec<usize> = Vec::new(); // value index
            while matches!(ops[i], Op::ObserveScalar) {
                let (pv, _) = vc_at(i);
                i += 1;
                pd_recs.push(pv);
            }
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(
                matches!(ops[i], Op::SqueezeSlice(n) if n == 2 + pd_recs.len()),
                "mixed coefficient vector"
            );
            let rs_gam_ch = vc_at(i).1;
            let rs_gam_fin = fin_at(i);
            i += 1;
            // W rounds until the multipoint label.
            let mut w_rounds = 0usize;
            while matches!(ops[i], Op::ObserveScalar) {
                assert!(matches!(ops[i + 1], Op::ObserveScalar), "w round pair");
                i += 2;
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeScalar), "w round squeeze");
                i += 1;
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
            (
                pd_recs,
                mv,
                rs_recs,
                rs_gam_ch,
                vec![(rs_gam_fin, 0), (rs_gam_fin, 1)],
            )
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
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
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
            while matches!(ops[i], Op::ObserveScalar) && matches!(ops[i + 1], Op::ObserveScalar) {
                let (gv, _) = vc_at(i);
                i += 2;
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                if !matches!(ops[i], Op::SqueezeScalar) {
                    break;
                }
                let (_, rc) = vc_at(i);
                let (g1, gi) = (vals_rec[gv], vals_rec[gv + 1]);
                let r = chals[rc];
                let g0 = t + g1;
                t = g0 + (g1 + g0 + gi) * r + gi * r * r;
                i += 1;
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
        let mut ga_i = gkr_l[0] + 1;
        while matches!(ops[ga_i], Op::Pow { .. }) {
            ga_i += 1;
        }
        assert!(matches!(ops[ga_i], Op::SqueezeScalar), "GKR fingerprint");
        let ga_fin = fin_at(ga_i);
        let (_, ga_c) = vc_at(ga_i);
        let mut mp_i = mp_l[0] + 1;
        while matches!(ops[mp_i], Op::ObserveScalar) {
            mp_i += 1;
        }
        while matches!(ops[mp_i], Op::Pow { .. }) {
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
        // The transcript is FORKED (the wiring runs on its own chain);
        // `merge_chain` splices the child's rows in at the fork point and
        // hands back one linear numbering plus the four cross-link wires.
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = merge_chain(
            t_shape.ops(),
            &t_shape.stream_words_duplex(domain),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, &chals);

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
        let b3_rows = trace.rows.len() + h_rows + query_phase_b3_rows(&geo);
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
                let (leaf, path, cap) = level_query_phase_b3_rows(g);
                eprintln!(
                    "    level: q {} depth {} row_words {} -> leaf {} + path {} + cap {}",
                    g.q, g.depth, g.raw_row_words, leaf, path, cap,
                );
            }
            // CHAIN DECOMPOSITION + an independent row-count model of the
            // duplex discipline (transcript-v3), asserted against the
            // sponge trace: a squeeze row absorbs the pending partial
            // block as its MESSAGE, mutates cv, and has no header word.
            {
                let pad16 = |n: usize| n.div_ceil(16) * 16;
                let (mut hdr_w, mut pay_w, mut n_obs, mut n_sq) = (0usize, 0usize, 0usize, 0usize);
                for op in ops.iter() {
                    match op {
                        Op::Label(l) => {
                            hdr_w += 1;
                            pay_w += pad16(l.len()) / 16;
                            n_obs += 1;
                        }
                        Op::ObserveScalar => {
                            hdr_w += 1;
                            pay_w += 1;
                            n_obs += 1;
                        }
                        Op::ObserveSlice(n) => {
                            hdr_w += 1;
                            pay_w += n;
                            n_obs += 1;
                        }
                        Op::ObserveBytes(len) => {
                            hdr_w += 1;
                            pay_w += pad16(*len) / 16;
                            n_obs += 1;
                        }
                        Op::Forked { .. } | Op::Merge { .. } => {}
                        Op::Pow { .. } => {
                            pay_w += 1;
                        }
                        Op::LegacyPow { .. } => {
                            n_sq += 1;
                        }
                        Op::SqueezeScalar | Op::SqueezeSlice(_) => {
                            n_sq += 1;
                        }
                    }
                }
                let v3_rows =
                    duplex_row_count_model(t_shape.ops(), &t_shape.stream_words_duplex(domain));
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
            for (k, &(sv, _, rc)) in rs_recs.iter().enumerate() {
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
        let t_final_n = replay_ligerito_spine256(
            &levels,
            &vals_rec,
            &chals,
            start_v,
            chals[inner_pd2.ch] * vals_rec[inner_pd2.q_v],
            &native_sums,
        );

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
        let (
            zc_rounds_b,
            (zskip_ch, zskip_fin),
            (outer_ch_b, outer_fin_b),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_msg_vs,
            lc_rounds_b,
            zp_v,
        ) = {
            let mut i2 = zc_l[0] + 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_skip slice");
            i2 += 1;
            assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_outer slice");
            let outer = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_ab");
            i2 += 1;
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_c");
            i2 += 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeScalar), "z_skip");
            let zskip = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            let mut zc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
                let mut squeeze_i = i2 + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                zc_r.push((vc_at(squeeze_i).1, fin_at(squeeze_i)));
                i2 = squeeze_i + 1;
            }
            let (zcf, _) = vc_at(i2);
            while matches!(ops[i2], Op::ObserveScalar) {
                i2 += 1;
            }
            assert_eq!(i2, lc_l[0], "the zerocheck runs straight into the lincheck");
            i2 += 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeScalar), "lc alpha");
            let lc_alpha = (vc_at(i2).1, fin_at(i2));
            i2 += 1;
            let mut betas = Vec::new();
            loop {
                while matches!(ops[i2], Op::Pow { .. }) {
                    i2 += 1;
                }
                if !matches!(ops[i2], Op::SqueezeScalar) {
                    break;
                }
                betas.push((vc_at(i2).1, fin_at(i2)));
                i2 += 1;
            }
            let mut lc_msgs = Vec::new();
            let mut lc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
                let mut squeeze_i = i2 + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                lc_msgs.push(vc_at(i2).0);
                lc_r.push((vc_at(squeeze_i).1, fin_at(squeeze_i)));
                i2 = squeeze_i + 1;
            }
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "z_partial slice");
            let (zp, _) = vc_at(i2);
            (zc_r, zskip, outer, lc_alpha, betas, zcf, lc_msgs, lc_r, zp)
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
                    chals[zc_rounds_b[m].0], x,
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
            assert_eq!(
                chals[bl_alpha.0], bool_assert.alpha,
                "the located Boolean lincheck alpha"
            );
            let pinned: Vec<usize> = bool_assert
                .betas
                .iter()
                .enumerate()
                .filter_map(|(t, beta)| beta.map(|_| t))
                .collect();
            assert_eq!(pinned.len(), betas_b.len(), "one beta per const pin");
            for (k, &t) in pinned.iter().enumerate() {
                assert_eq!(
                    chals[betas_b[k].0],
                    bool_assert.betas[t].expect("pinned beta"),
                    "Boolean const-pin beta {k}"
                );
            }
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
        let eps_n: Vec<F128> = bool_assert.pin_evals.iter().flatten().copied().collect();
        assert_eq!(eps_n.len(), betas_b.len(), "one prefix value per beta");
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
                    let rb = if layer < m_mp2 {
                        point_n[layer]
                    } else {
                        F128::ZERO
                    };
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
                    // The count win's tie: the RAW per-statement W the
                    // region now PUBLISHES equals the deferred export's
                    // claim value — the verifier-exported reference, not a
                    // formula written twice.
                    assert_eq!(
                        inner.work.jagged.rs[si].value, w_n,
                        "RS raw W == exported jagged claim {si}"
                    );
                    let coeff = if si == 0 {
                        g_at_n
                    } else {
                        gpow_n[128] * g_at_n
                    };
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
                    // The group's exported decomposition — the γ-baked
                    // one-hot combo plus γ-outside dense members — must
                    // recombine to the same raw group W, member for member.
                    let (combo, dense) = &inner.work.jagged.groups[g_ix];
                    let mut raw = combo.as_ref().map_or(F128::ZERO, |c| c.value);
                    let mut d_it = dense.iter();
                    for &i2 in members {
                        let hot = pd_pts_n[i2][n_log_i..]
                            .iter()
                            .all(|&x| x == F128::ZERO || x == F128::ONE);
                        if hot {
                            continue;
                        }
                        let (g, c) = d_it.next().expect("a dense entry per non-hot member");
                        assert_eq!(*g, chals[gammas_o[i2].ch], "dense member γ_pd");
                        raw += *g * c.value;
                    }
                    assert!(d_it.next().is_none(), "every dense entry consumed");
                    assert_eq!(
                        raw, w_n,
                        "group {g_ix} raw W == exported jagged decomposition"
                    );
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
        let yr_len = proof.pcs_open.inner.ligerito.final_proof.yr.len() / 2;
        let lane_major = geo[0].row_words < geo[0].lanes;
        let w_resid: Vec<RoundRec> = if lane_major {
            let k_rot = w_rounds.len() - levels[0].fold_fins.len();
            let mut v = w_rounds[k_rot..].to_vec();
            v.extend_from_slice(&w_rounds[..k_rot]);
            v
        } else {
            w_rounds.to_vec()
        };

        let z_ix = el_assert.as_ref().map(|assertion| {
            gammas_o
                .iter()
                .position(|pd| vals_rec[pd.val_v] == assertion.z_eval)
                .expect("element z_eval is an absorbed packed-direct value")
        });

        ChildTape {
            inner,
            vals_rec,
            chals,
            pub_payloads,
            cap_pays,
            trace,
            stream,
            bytes,
            cross,
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
            z_ix,
            n_pub_slots_c,
            n_p,
            zc_rounds_b,
            outer_b: (outer_ch_b, outer_fin_b),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_msg_vs,
            lc_rounds_b,
            eps_n,
            zskip_ch,
            zskip_fin,
            zp_v,
            ga_c,
            ga_fin,
            mg_c,
            mg_fin,
            rs_recs,
            rs_gam_fins,
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
            jag: inner.work.jagged.clone(),
        }
    }
}

/// The gate slots a child-tape region emits into. Created ONCE by the outer
/// test and shared by every region in the builder — the mvp11 merge outer
/// instantiates two child regions (and the fold region) over shared slots.
/// The recursion envelope and strict Fast nodes place the two independent
/// child BLAKE workloads in identical slots; the other families still add
/// rows, not columns. The `le`/`resid` caches fill on demand during emission;
/// cache hits require same-shape children (the keyed constructor parameters
/// must match, which the merge test asserts by requiring one shared circuit).
struct ChildSlots {
    nu: usize,
    q: CollapsedSlots,
    macs: flock_core::circuit::builder::SlotId,
    fold_macs: flock_core::circuit::builder::SlotId,
    zcr: flock_core::circuit::builder::SlotId,
    mrs: flock_core::circuit::builder::SlotId,
    spine: flock_core::circuit::builder::SlotId,
    spine256: flock_core::circuit::builder::SlotId,
    alslot: flock_core::circuit::builder::SlotId,
    le: Vec<(usize, flock_core::circuit::builder::SlotId)>,
    /// The residual region's keyed slot cache (`emit_residual_region`'s
    /// `leaf_slot`). Key scheme: `600` = the shared MacGate (pre-seeded,
    /// so close-out rows land on `macs` instead of a duplicate type);
    /// `701` = the shared extension-field MAC; `100 + pl` = the base-field
    /// ResidualGate at that suffix-fold count; `310 + width` and
    /// `1000 + width` = base/extension prefix gates; and `880..=882` = the
    /// three shared extension-field residual relations.
    resid: Vec<(usize, flock_core::circuit::builder::SlotId)>,
}

impl ChildSlots {
    #[cfg(test)]
    fn new(sb: &mut ShapeBuilder, nu2: usize, spread_w: usize) -> Self {
        Self::new_with_b3_split(sb, nu2, spread_w, false)
    }

    #[cfg(test)]
    fn new_with_b3_split(
        sb: &mut ShapeBuilder,
        nu2: usize,
        spread_w: usize,
        split_b3: bool,
    ) -> Self {
        let macs = sb.slot(MacGate::new());
        let fold_macs = sb.slot(MacGate::new());
        let mac256 = sb.slot(MacGate256::new());
        let b3 = sb.slot(Blake3Gate { nu: nu2 });
        let b3_alt = split_b3.then(|| sb.slot(Blake3Gate { nu: nu2 }));
        ChildSlots {
            nu: nu2,
            q: CollapsedSlots {
                b3,
                b3_alt,
                swap: sb.slot(SwapGate { nu: nu2 }),
                spread: sb.slot(BitSpreadGate {
                    ty: BitSpreadTable::new(spread_w),
                    nu: nu2,
                }),
                pow: sb.slot(PowMaskGate { nu: nu2 }),
                family: Some(sb.slot(FamilyTransposeTileGate { nu: nu2 })),
            },
            macs,
            fold_macs,
            zcr: sb.slot(ZcRoundGate::new()),
            mrs: sb.slot(MergedRoundGate::new()),
            spine: sb.slot(SpineGate::new()),
            spine256: sb.slot(SpineGate256::new()),
            alslot: sb.slot(AssistLayerGate::new()),
            le: Vec::new(),
            // Key 600 pre-seeds the SHARED MacGate into the residual cache:
            // emit_residual_region's close-out rows land on the same slot
            // instead of registering a duplicate type.
            resid: vec![(600, macs), (701, mac256)],
        }
    }

    /// The ENVELOPE constructor (wall 2): the same canonical declaration
    /// order [`declare_envelope_slots`] gives every envelope outer, so all
    /// their registry digests agree. Every keyed entry pre-seeds the
    /// demand caches; emission that would need a slot OUTSIDE the envelope
    /// set creates a new type and fails the digest pin loudly.
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
            nu: nu2,
            q,
            macs: take(600),
            fold_macs: take(602),
            zcr: take(500),
            mrs: take(400),
            spine: take(0),
            spine256: take(700),
            alslot: take(601),
            le: vec![(8, take(8)), (808, take(808))],
            // The residual-region cache inherits every entry in its key
            // namespaces: the shared macs, base residual variants, the
            // three shared F256 residual relations, and both prefix slots.
            resid: cache
                .iter()
                .filter(|&&(k, _)| {
                    matches!(k, 600 | 701)
                        || (100..200).contains(&k)
                        || (310..400).contains(&k)
                        || (880..=882).contains(&k)
                        || (1000..1100).contains(&k)
                })
                .cloned()
                .collect(),
        }
    }

    /// The keyed cache view `pad_envelope_counts` consumes — envelope path
    /// only (`new_env`).
    fn env_cache(&self) -> Vec<(usize, flock_core::circuit::builder::SlotId)> {
        let mut v = vec![
            (600, self.macs),
            (602, self.fold_macs),
            (500, self.zcr),
            (400, self.mrs),
            (0, self.spine),
            (700, self.spine256),
            (601, self.alslot),
        ];
        v.extend(self.le.iter().map(|&(n, s)| (n, s)));
        v.extend(self.resid.iter().filter(|&&(k, _)| k != 600).cloned());
        v
    }

    /// Every element-class slot, for the outer prover's slot inputs.
    fn element_slot_ids(&self) -> Vec<flock_core::circuit::builder::SlotId> {
        let mut v = vec![
            self.macs,
            self.fold_macs,
            self.zcr,
            self.mrs,
            self.spine,
            self.spine256,
            self.alslot,
        ];
        v.extend(self.le.iter().map(|&(_, s)| s));
        // Key 600 is the SHARED MacGate seed (already listed as `macs`).
        v.extend(
            self.resid
                .iter()
                .filter(|&&(k, _)| k != 600)
                .map(|&(_, s)| s),
        );
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
    structure_claim_w: Vec<(Vec<Wire>, Vec<Wire>, Wire)>,
    /// The jagged assertion's value wires (the count win), in emission
    /// order: rs claims, then per group the combo and its dense members —
    /// the fresh-claim surfaces a merge fold connects to.
    jag_w: Vec<Wire>,
    /// The claims' IDENTITY wires (the points-connect): σ — the anchor
    /// round squeezes, shared by every claim of the region — and per claim
    /// (jag_w order) the row wires: Eq claims carry z_col coordinate wires
    /// (constant coords ride zw/ow), Combo claims carry the γ_pd
    /// coefficient wires in term order (addresses are registry constants,
    /// bound by the fold side's shared constant publics).
    jag_sig_w: Vec<Wire>,
    jag_row_w: Vec<Vec<Wire>>,
    /// The boolean MatrixAssertion's wires: the zc mlv round rhos (round
    /// order — [dim6 | x_outer | x_inner_rest]), the lc round rhos (round
    /// order — rr is these reversed), and the absorbed z_partial words.
    b_mlv_w: Vec<Wire>,
    b_lc_w: Vec<Wire>,
    b_zpartial_w: Vec<Wire>,
    /// Reported matrix evaluations, constrained by the in-circuit scalar
    /// closure and connected to the aggregate fold's fresh claim values.
    mat_eval_w: Vec<(Wire, Wire)>,
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
    b3_slot: flock_core::circuit::builder::SlotId,
    ct: &ChildTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    consts: &mut Vec<(F128, Wire)>,
) -> ChildRegion {
    let child_q = CollapsedSlots {
        b3: b3_slot,
        ..cs.q
    };
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
    let _n_runs = ct.bounds_i.len();

    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match cs.le.iter().find(|(n, _)| *n == 800 + lanes) {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(LeafEvalGate256::new(lanes));
                    cs.le.push((800 + lanes, sl));
                    sl
                }
            }
        })
        .collect();
    let iv_w = pack8(&crate::r1cs_hashes::fs_chain::IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [
        sb.fixed_public_input(iv_w[0]),
        sb.fixed_public_input(iv_w[1]),
    ];
    let (outs, ww) = emit_fs_chain(
        sb,
        b3_slot,
        iv2,
        trace,
        stream,
        &ct.bytes,
        vals,
        consts,
        &ct.pub_payloads,
        &ct.cross,
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
            child_q,
            iv2,
            &ct.inner.built.witness.public,
            dw,
            vals,
            consts,
        )
    };
    let cap_w = cap_wires(stream, &ww, &ct.cap_pays);
    let (to_publish, level_accs, query_positions) = emit_query_phase(
        sb,
        child_q,
        iv2,
        &leafeval,
        levels,
        geo,
        &ct.lvl_src,
        trace,
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
    assert_eq!(
        pt_w.len(),
        ct.mu_i,
        "the GKR point spans the inner cell space"
    );
    // The input checks under the LIVE-IDENTITY padding: M̂(ρ) and livê(ρ),
    // bound through the digest-keyed circuit-structure claims folded by the
    // parent.
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
        cs.fold_macs,
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

    // ---- the LIGERITO SPINE ----
    let spine = cs.spine;
    let spine256 = cs.spine256;
    let gpw = outs[trace.squeezes[inner_pd2.fin][0]][0];
    let z2 = [zw, zw];
    let tw0 = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        z2,
        z2,
        [wv(inner_pd2.q_v), zw],
        gpw,
        z2,
    );
    let mut tsp = tw0[3];
    for od in &levels[0].initial_ood {
        let bw = outs[trace.squeezes[od.beta_fin][0]][0];
        tsp = emit_spine256(
            sb,
            spine256,
            z2,
            z2,
            z2,
            tsp,
            z2,
            z2,
            [wv(od.y_v), zw],
            bw,
            z2,
        )[3];
    }
    let st = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        [wv(ct.start_v), wv(ct.start_v + 1)],
        [wv(ct.start_v + 2), wv(ct.start_v + 3)],
        tsp,
        ow,
        z2,
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = [
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 0),
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 1),
            ];
            let ev = emit_spine256(sb, spine256, qc, qb, qa, z2, z2, z2, z2, zw, rw);
            tsp = ev[4];
            let bld = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                z2,
                [wv(mv), wv(mv + 1)],
                [wv(mv + 2), wv(mv + 3)],
                tsp,
                ow,
                z2,
            );
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r_lvl {
            for od in &lvl.ood {
                let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                let f = emit_spine256(
                    sb,
                    spine256,
                    qc,
                    qb,
                    qa,
                    tsp,
                    [wv(od.intro_v), wv(od.intro_v + 1)],
                    [wv(od.intro_v + 2), wv(od.intro_v + 3)],
                    [wv(od.y_v), zw],
                    bw,
                    z2,
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            }
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                qc,
                qb,
                qa,
                tsp,
                [wv(lvl.intro_v), wv(lvl.intro_v + 1)],
                [wv(lvl.intro_v + 2), wv(lvl.intro_v + 3)],
                level_accs[li],
                bw,
                z2,
            );
            (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                tsp,
                z2,
                z2,
                level_accs[li],
                bw,
                z2,
            );
            tsp = f[3];
        }
    }
    let t_final = tsp;

    // ---- the RESIDUAL region (shared emitter) ----
    let yr_wires: Vec<[Wire; 2]> = (0..ct.yr_len)
        .map(|y| [wv(ct.yr_v2 + 2 * y), wv(ct.yr_v2 + 2 * y + 1)])
        .collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        &to_publish,
        &query_positions,
        &ct.w_resid,
        inner_pd2.fin,
        &yr_wires,
        trace,
        &outs,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: the residual side's inner and the spine's
    // t_r are the same statement scalar — a copy constraint, not a
    // checker item (both stay published as test cross-checks).
    sb.connect(inner_w[0], t_final[0]);
    sb.connect(inner_w[1], t_final[1]);

    // ---- FAMILY H + the merged intake boundary ----
    // The transpose/equality dot products and inverse-Moore/Frobenius
    // recombination are all recursive-circuit arithmetic. Every source is
    // an existing transcript/proof wire; the native target is retained only
    // as a published test oracle below.
    let shv_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let sv = ct.rs_recs[k].0;
        (0..128).map(|i| wv(sv + i)).collect()
    });
    let value_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        mp_o.val_vs[128 * k..128 * (k + 1)]
            .iter()
            .map(|&vi| wv(vi))
            .collect()
    });
    let rdp_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let fin = ct.rs_recs[k].1;
        (0..7)
            .map(|j| squeeze_word_wire(&outs, trace, fin, j))
            .collect()
    });
    let gamma_w: [Wire; 2] = std::array::from_fn(|k| {
        let (fin, offset) = ct.rs_gam_fins[k];
        squeeze_word_wire(&outs, trace, fin, offset)
    });
    let (rsh_w, vrs_w) = emit_family_h(
        sb,
        cs.q.family.expect("family-H slot"),
        cs.macs,
        cs.fold_macs,
        cs.spine,
        cs.spine256,
        cs.resid
            .iter()
            .find(|&&(key, _)| key == 701)
            .expect("the child slots declare an F256 MAC slot")
            .1,
        1usize << cs.nu,
        &shv_w,
        &value_w,
        &rdp_w,
        gamma_w,
        pfslot,
        pf_w,
        zw,
        ow,
        vals,
        consts,
    );
    let mut pdh_w = zw;
    for pd in &ct.gammas_o {
        let gw = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
        pdh_w = sb.gate(macs, &[pdh_w, gw, wv(pd.val_v)])[0];
    }
    let tgt_w = sb.gate(macs, &[rsh_w, ow, pdh_w])[0];
    let mut runw = tgt_w;
    for rr in w_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        runw = sb.gate(mrs, &[runw, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    let mut vgrp_w = zw;
    for &vi in &mp_o.val_vs[256..] {
        vgrp_w = sb.gate(macs, &[vgrp_w, ow, wv(vi)])[0];
    }
    let v_w = sb.gate(macs, &[vrs_w, ow, vgrp_w])[0];
    let rhs_v_w = sb.gate(macs, &[zw, wv(inner_pd2.q_v), v_w])[0];
    sb.connect(runw, rhs_v_w);

    // ---- the ELEMENT PIOP rounds in-circuit (mixed children only) ----
    // Zerocheck rounds are ZcRoundGate rows (tau slice wires as eq weights,
    // g0 advice + zero deltas); lincheck rounds are MergedRoundGate rows.
    // The entry is DERIVED: va = ea + a_sum, vb = eb + b_sum, entry =
    // va + alpha·vb — only the two constant-strip sums are advice.
    let el_pub = el_rec.map(|el_rec| {
        let mut el_zr = zw;
        for (k, &(gv, rfin, _)) in el_rec.zc_rounds.iter().enumerate() {
            let t_w = squeeze_word_wire(&outs, trace, el_rec.tau_fin, k);
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
        (el_zr, el_lcw, asum_w, bsum_w, el_alpha_w)
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
    // THE COUNT WIN: the counts used to enter the parent's circuit HERE —
    // per-run boundary eq products with the jagged run boundaries (the
    // prefix sums of the child's per-type heights) baked as ow/zw, then
    // per-statement run-weight enumerations consuming them. All of it is
    // gone: each statement's raw W arrives as a PUBLISHED CLAIM VALUE on
    // the jagged layout table (the deferred verify's own export, keyed by
    // the child digest), checker-held here and discharged at the ROOT of
    // the accumulation tree — the eps discipline, ported. The claim's
    // points are wires this region already carries (σ = the anchor round
    // squeezes, z_cols = statement point wires, γ_pd = squeezes); nothing
    // count-shaped remains in the circuit.
    let mut jag_w: Vec<Wire> = Vec::new();
    // The claims' IDENTITY wires (the points-connect): σ shared per
    // region, and per claim — in jag_w order — the row-identity wires the
    // merge fold's absorbed words connect to.
    let mut jag_row_w: Vec<Vec<Wire>> = Vec::new();
    // Per RS statement: the published w, the DP, the coefficient.
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        vals.push(ct.jag.rs[si].value);
        let w_st = sb.input();
        jag_w.push(w_st);
        jag_row_w.push(xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect());
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
    // Per group: the γ-baked one-hot combo publishes as ONE value; each
    // dense (element) member publishes its raw eq value with its γ_pd
    // applied by a MAC on the squeeze wire — the exported decomposition,
    // reassembled in wires. Coefficient γ^{256+k}·e_at as before.
    for (g_ix, members) in ct.groups_ix.iter().enumerate() {
        let (combo, dense) = &ct.jag.groups[g_ix];
        let hots: Vec<bool> = members
            .iter()
            .map(|&i2| {
                ct.pd_pts[i2][n_log_i..]
                    .iter()
                    .all(|&x| x == F128::ZERO || x == F128::ONE)
            })
            .collect();
        let mut w_st = match combo {
            Some(c) => {
                vals.push(c.value);
                let w = sb.input();
                jag_w.push(w);
                // The combo's identity: the hot members' γ_pd squeeze
                // wires, in member order == the assertion's term order.
                let gws: Vec<Wire> = members
                    .iter()
                    .zip(&hots)
                    .filter(|&(_, &h)| h)
                    .map(|(&i2, _)| {
                        let pd = &ct.gammas_o[i2];
                        squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset)
                    })
                    .collect();
                if let flock_core::matrix_fold::JaggedRowWeight::Combo(t) = &c.row {
                    assert_eq!(t.len(), gws.len(), "combo terms == hot members");
                }
                jag_row_w.push(gws);
                w
            }
            None => zw,
        };
        let mut d_it = dense.iter();
        for (&i2, &hot) in members.iter().zip(&hots) {
            if hot {
                continue;
            }
            let (_, c) = d_it.next().expect("a dense entry per non-hot member");
            let pd = &ct.gammas_o[i2];
            let gpd_w = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
            vals.push(c.value);
            let d_w = sb.input();
            jag_w.push(d_w);
            // The dense claim's identity: its z_col coordinate wires —
            // constant coords ride zw/ow, the rest are the element PIOP's
            // own squeeze wires (the mapping the constructor pinned).
            jag_row_w.push(
                (0..k_cols_i)
                    .map(|jj| {
                        let coord = ct.pd_pts[i2][n_log_i + jj];
                        if coord == F128::ZERO {
                            zw
                        } else if coord == F128::ONE {
                            ow
                        } else {
                            let el_rec = el_rec.expect("element pd claim");
                            if i2 == 0 {
                                outs[trace.squeezes[el_rec.zc_rounds[n_log_i + jj].1][0]][0]
                            } else {
                                let n_lc = el_rec.lc_rounds.len();
                                outs[trace.squeezes[el_rec.lc_rounds[n_lc - 1 - jj].1][0]][0]
                            }
                        }
                    })
                    .collect(),
            );
            w_st = sb.gate(macs, &[w_st, gpd_w, d_w])[0];
        }
        assert!(d_it.next().is_none(), "every dense entry consumed");
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

    // The aggregate verifier's scalar closures are part of the recursive
    // relation too.  The reported A/B values below become fold claims; these
    // equations prevent a prover from choosing values that discharge against
    // the matrices but do not reproduce the child verifier's running target.
    let mat_eval_w: Vec<(Wire, Wire)> = ct
        .bool_assert
        .evals
        .iter()
        .map(|&(a, b)| {
            vals.push(a);
            let aw = sb.input();
            vals.push(b);
            let bw = sb.input();
            (aw, bw)
        })
        .collect();
    let mat_alpha_w = outs[trace.squeezes[ct.bl_alpha.1][0]][0];
    let mat_x_inner_w: Vec<Wire> = (0..ct.bool_assert.x_inner_rest.len())
        .map(|j| {
            let m = if j == 0 { 0 } else { n_log_i + j };
            mlv_pw[m].1
        })
        .collect();
    let mat_rr_w: Vec<Wire> = lc_pw.iter().map(|&(_, w)| w).collect();
    let zpartial_ws: Vec<Wire> = (0..64).map(|i| wv(ct.zp_v + i)).collect();
    let mut beta_wires = vec![None; ct.inner.built.shape.registry.num_boolean()];
    let mut pin_wires = Vec::with_capacity(ct.sigma_native.boolean_pins.len());
    let mut lcb_w = assertion_mac(
        sb,
        spine,
        wv(ct.zc_finals_v + 1),
        mat_alpha_w,
        wv(ct.zc_finals_v),
        zw,
    );
    for (k, (type_index, _, value)) in ct.sigma_native.boolean_pins.iter().enumerate() {
        let beta = outs[trace.squeezes[ct.betas_b[k].1][0]][0];
        beta_wires[*type_index] = Some(beta);
        vals.push(*value);
        let eps_w = sb.input();
        assert_eq!(
            *value, ct.eps_n[k],
            "static pin claim equals lincheck prefix"
        );
        pin_wires.push((*type_index, eps_w));
        lcb_w = assertion_mac(sb, spine, lcb_w, beta, eps_w, zw);
    }
    for (&g_v, &(_, fin)) in ct.lc_msg_vs.iter().zip(&ct.lc_rounds_b) {
        let rho_w = outs[trace.squeezes[fin][0]][0];
        lcb_w = sb.gate(mrs, &[lcb_w, wv(g_v), wv(g_v + 1), rho_w])[0];
    }
    emit_boolean_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &ct.inner.built.shape.registry,
        mat_alpha_w,
        &mat_x_inner_w,
        &mat_rr_w,
        &zpartial_ws,
        &beta_wires,
        &mat_eval_w,
        lcb_w,
        zw,
        ow,
    );

    let el_zc_rho_w: Vec<Wire> = el_rec
        .map(|el_rec| {
            el_rec
                .zc_rounds
                .iter()
                .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
                .collect()
        })
        .unwrap_or_default();
    if let (Some(el_assert), Some((_, el_lcw, _, _, el_alpha_w))) = (&ct.el_assert, el_pub) {
        let el_eval_w: Vec<(Wire, Wire)> = el_assert
            .evals
            .iter()
            .map(|&(a, b)| {
                vals.push(a);
                let aw = sb.input();
                vals.push(b);
                let bw = sb.input();
                (aw, bw)
            })
            .collect();
        let inner_union = UnionInstance::new(
            &ct.inner.built.shape.registry,
            ct.inner.built.shape.counts.clone(),
        );
        let el_r_col_w: Vec<Wire> = el_rec
            .expect("element transcript")
            .lc_rounds
            .iter()
            .rev()
            .map(|&(_, fin, _)| outs[trace.squeezes[fin][0]][0])
            .collect();
        emit_element_reported_check(
            sb,
            spine,
            pfslot,
            pf_w,
            &inner_union,
            el_alpha_w,
            &el_zc_rho_w[n_log_i..],
            &el_r_col_w,
            wv(ct.gammas_o[ct.z_ix.expect("element z_eval index")].val_v),
            &el_eval_w,
            el_lcw,
            zw,
            ow,
        );
    }
    let element_values = el_pub.map(|(_, _, a, b, _)| (a, b));
    let element_point = el_rec.map(|_| &el_zc_rho_w[n_log_i..]);
    let boolean_point: Vec<Wire> = mlv_pw[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
    let structure_claim_w = circuit_structure_claim_wires(
        &ct.sigma_native,
        &pt_w,
        mid_w,
        live_w,
        sig_w,
        &boolean_point,
        &pin_wires,
        element_point,
        element_values,
        zw,
        ow,
    );

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
        sb.publish(w[0]);
        sb.publish(w[1]);
    }
    sb.publish(ga_w);
    sb.publish(mg_w);
    if let Some((el_zr, el_lcw, _, _, _)) = el_pub {
        sb.publish(el_zr);
        sb.publish(el_lcw);
    }
    sb.publish(anc_w);
    sb.publish(t_final[0]);
    sb.publish(t_final[1]);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(w[0]);
            sb.publish(w[1]);
        }
    }
    sb.publish(inner_w[0]);
    sb.publish(inner_w[1]);
    // ---- the SIGMA ASSERTION emission (route B, in-circuit) ----
    // The claim exits as bound publics: the value is the deferred s_sigma
    // stream word — the SAME wire the rhs input check just consumed — and
    // the point is the GKR's own accumulated squeeze wires.
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    // ---- the JAGGED ASSERTION emission (the count win) ----
    // Raw W claim values in emission order (rs, then per group combo +
    // dense members), checker-held against the deferred export.
    for w in &jag_w {
        sb.publish(*w);
    }
    let n_tail = 2 + n_el_pd + 5 + 2 * levels.len() * ct.yr_len + 2 + 1 + ct.mu_i + jag_w.len();
    let n_query_pub: usize = 2 * levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    ChildRegion {
        pub_base,
        n_query_pub,
        n_tail,
        structure_claim_w,
        jag_w,
        jag_sig_w: mp_sig_w.clone(),
        jag_row_w,
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: ct
            .lc_rounds_b
            .iter()
            .map(|&(_, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: (0..64).map(|i| wv(ct.zp_v + i)).collect(),
        mat_eval_w,
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
                F256::new(public[at + 2 * li], public[at + 2 * li + 1]),
                *want,
                "L{li} enforced sum matches the native replica"
            );
        }
        assert_eq!(
            at + 2 * ct.native_sums.len(),
            r.pub_base + r.n_query_pub,
            "the query publics walk consumed its whole block"
        );
    }
    let base2 = r.pub_base + r.n_query_pub;
    assert_eq!(
        public[base2], chals[ct.ga_c],
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
            public[el_base], ct.el_run_n,
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
        public[mp_base], ct.anc_end_n,
        "the anchor rounds end at the native claim"
    );
    // THE LIGERITO CLOSE: the in-circuit spine reaches the native t_r.
    assert_eq!(
        F256::new(public[mp_base + 1], public[mp_base + 2]),
        ct.t_final_n,
        "the spine's final t_r matches the native replay"
    );
    // The merged intake is fully constrained; its publications retain the
    // native replay as a test oracle.
    assert_eq!(
        public[mp_base + 3],
        ct.native_target,
        "the computed RS target matches the native gamma-combination"
    );
    assert_eq!(
        public[mp_base + 4],
        ct.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    // The residual region against the shared native replica — and THE
    // CLOSURE: the residual-side inner and the spine's t_r are the same
    // statement scalar, both held against published circuit outputs.
    let inner_n = check_residual_publics(
        public,
        mp_base + 5,
        &ct.levels,
        &ct.geo,
        &ct.w_resid,
        ct.inner_pd2.ch,
        &observed_f256(&ct.vals_rec, ct.yr_v2, ct.yr_len),
        chals,
    );
    assert_eq!(
        inner_n, ct.t_final_n,
        "inner == t_r: the mixed statement closes"
    );
    // The sigma assertion, as the accumulator would read it: the value and
    // the mu point coordinates, matched against the native claim.
    let sig_base = mp_base + 5 + 2 * ct.levels.len() * ct.yr_len + 2;
    assert_eq!(
        public[sig_base], ct.inner.proof.wiring.gkr.s_sigma_eval,
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
            base_bits: ct.sigma_native.base_bits,
            masked_id_value: ct.mid_n,
            live_value: ct.live_n,
            value: public[sig_base],
            boolean_pins: ct.sigma_native.boolean_pins.clone(),
            element_constants: ct.sigma_native.element_constants.clone(),
        };
        assert_eq!(sa.rho, ct.sigma_native.rho, "the emitted sigma point");
        assert_eq!(sa.value, ct.sigma_native.value, "the emitted sigma value");
        assert_eq!(sa.nu, ct.sigma_native.nu, "the emitted sigma split");
        assert!(
            sa.check(&ct.inner.built.shape.circuit),
            "the emitted sigma assertion discharges against the inner circuit"
        );
    }
    // The jagged assertion's value surfaces (the count win), in emission
    // order — rs claims, then per group the combo and its dense members —
    // each the deferred export's own raw claim value. The full claims
    // (points included) discharge against the child's layout, so the
    // published values are exactly what a merge fold's fresh-claim
    // surfaces connect to.
    {
        let jag_base = sig_base + 1 + ct.mu_i;
        let mut expect_vals: Vec<F128> = ct.jag.rs.iter().map(|c| c.value).collect();
        for (combo, dense) in &ct.jag.groups {
            if let Some(c) = combo {
                expect_vals.push(c.value);
            }
            for (_, c) in dense {
                expect_vals.push(c.value);
            }
        }
        for (j, want) in expect_vals.iter().enumerate() {
            assert_eq!(
                public[jag_base + j],
                *want,
                "jagged claim value {j} matches the deferred export"
            );
        }
        assert_eq!(
            jag_base + expect_vals.len(),
            r.pub_base + r.n_query_pub + r.n_tail,
            "the jagged publics close the region's tail"
        );
    }
    r.n_query_pub + r.n_tail
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
            b3_alt: None,
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(depth),
                nu,
            }),
            pow: sb.slot(PowMaskGate { nu }),
            family: None,
        };
        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&IV);
        vals.extend_from_slice(&iv_w);
        let iv = [
            sb.fixed_public_input(iv_w[0]),
            sb.fixed_public_input(iv_w[1]),
        ];
        let leaf = tree.leaf(pos);
        let leaf_w: Vec<Wire> = (0..words)
            .map(|w| {
                vals.push(leaf_word(leaf, 16 * w));
                sb.public_input()
            })
            .collect();
        vals.push(F128::new(pos as u64, 0));
        let idx_w = sb.public_input();
        let (root, _) = emit_opening(
            &mut sb, slots, iv, &leaf_w, idx_w, depth, 0, 0, None, &mut vals,
        );
        sb.publish(root[0]);
        sb.publish(root[1]);
        let shape = sb.finish().expect("the opening circuit builds");
        let hints: Vec<[u32; SLOT_WORDS]> = tree.siblings(pos);
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> = hints
            .iter()
            .map(|h| h as &(dyn std::any::Any + Sync))
            .collect();
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
    /// The entry's LIVE word — the zero-claim scale (wall 3): a real
    /// entry publishes 1 (ow); an absent one is all zeros, which decodes
    /// as the zero claim. Fold outputs are always real.
    live: Wire,
    rho_col: Vec<Wire>,
    rho_row: Vec<Wire>,
    value: Wire,
}

/// The fold region's op tape for a claim-list set: per group, the
/// matrix-fold label, every claim's four weight slices + value, the
/// lambdas, col rounds, bridge, mus, row rounds, and the output value.
/// Width-driven, so mixed low widths and any claim count pin themselves.
fn fold_region_ops(
    cfg: TowerConfig,
    fold_claims: &[Vec<flock_core::matrix_fold::MatrixClaim>],
) -> Vec<flock_core::transcript_record::TranscriptOp> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut want: Vec<Op> = Vec::new();
    let grinding = tower_fold_grinding(cfg);
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
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len())); // lambdas
        for _ in 0..cs[0].col.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        for _ in 0..cs.len() {
            want.push(Op::ObserveScalar); // bridge
        }
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len())); // mus
        for _ in 0..cs[0].row.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.push(Op::ObserveScalar); // the output value
    }
    want
}

/// Locate every fold's surfaces on the value/challenge streams (counters
/// start at 0 — the bind prefix carries only byte payloads) and pin them
/// field-for-field against the gathered claims and the `FoldProof`s.
/// Returns the counters alongside, so JAGGED groups on the same tape can
/// continue the walk ([`locate_and_pin_jagged_folds`]); callers with no
/// jagged groups assert exhaustion themselves via
/// [`assert_fold_tape_exhausted`].
fn locate_and_pin_folds(
    fold_claims: &[Vec<flock_core::matrix_fold::MatrixClaim>],
    fold_proofs: &[&flock_core::matrix_fold::FoldProof],
    vals_rec: &[F128],
    _chals: &[F128],
) -> (Vec<FoldLoc>, usize, usize) {
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
    (locs, vcur, ccur)
}

/// Map every challenge ordinal to the transcript finalization and output-word
/// offset that emitted it. Grinding adds finalizations without challenges,
/// while vector squeezes emit several challenge words from one finalization.
fn challenge_word_locs(ops: &[flock_core::transcript_record::TranscriptOp]) -> Vec<(usize, usize)> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut out = Vec::new();
    let mut fin = 0usize;
    for op in ops {
        match op {
            Op::SqueezeScalar => out.push((fin, 0)),
            Op::SqueezeSlice(n) => out.extend((0..*n).map(|offset| (fin, offset))),
            _ => {}
        }
        if op.finalizes() {
            fin += 1;
        }
    }
    out
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
            let expect_c =
                loc.claims
                    .iter()
                    .zip(&lam)
                    .enumerate()
                    .fold(F128::ZERO, |acc, (i, (cl, &l))| {
                        let w = located(cl.col_low_v, cl.col_low_n, cl.col_pt_v, cl.col_pt_n);
                        acc + l * w.eval(&rho_col) * vals_rec[loc.bridge_v + i]
                    });
            assert_eq!(run_c, expect_c, "col endpoint closes from located words");

            let mus: Vec<F128> = (0..k).map(|i| chals[loc.mu_ch0 + i]).collect();
            let target_r = (0..k).zip(&mus).fold(F128::ZERO, |acc, (i, &m)| {
                acc + m * vals_rec[loc.bridge_v + i]
            });
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
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    challenge_locs: &[(usize, usize)],
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vmap: &[Option<usize>],
    chals: &[F128],
    vals_rec: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    tail_input_last: bool,
) -> (Vec<FoldPub>, Vec<AlphaRec>) {
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    let chw = |ch: usize| -> Wire {
        let (fin, offset) = challenge_locs[ch];
        squeeze_word_wire(outs, trace, fin, offset)
    };
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
            run_w = sb.gate(
                mrs,
                &[run_w, wv(loc.col_v + 2 * j), wv(loc.col_v + 2 * j + 1), r_w],
            )[0];
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
            run2_w = sb.gate(
                mrs,
                &[
                    run2_w,
                    wv(loc.row_v + 2 * j),
                    wv(loc.row_v + 2 * j + 1),
                    r_w,
                ],
            )[0];
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
        // enters as its own input, bound by the row endpoint delta. When
        // JAGGED groups follow on the same tape (`tail_input_last =
        // false`), their absorbs flush this word and it has a chain wire
        // like any other — the tail treatment moves to the last jagged
        // group.
        let value = if tail_input_last && fi + 1 == locs.len() {
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
            live: ow,
            rho_col: rho_col_w,
            rho_row: rho_row_w,
            value,
        });
    }
    (fold_pubs, alpha_recs)
}

/// Read ONE published accumulator entry at `p`, advancing it.
///
/// Entry layout (wall 3): `[key | live | rho_col | rho_row | value]` — the
/// two KEY words only for the keyed groups (sigma, jagged), where the
/// entry names the circuit whose table it is about; the registry-keyed
/// matrix groups carry none. The LIVE word is the zero-claim scale, so a
/// block of zeros decodes as the zero claim: weights identically zero,
/// value zero, true about every table. That is what a DEAD SLOT is, and
/// why a base node and a steady node can be read at the same offsets.
fn read_acc_entry(
    public: &[F128],
    p: &mut usize,
    keyed: bool,
    k_col: usize,
    k_row: usize,
) -> ([F128; 2], flock_core::matrix_fold::MatrixClaim) {
    use flock_core::matrix_fold::{MatrixClaim, Weight};
    let key = if keyed {
        *p += 2;
        [public[*p - 2], public[*p - 1]]
    } else {
        [F128::ZERO; 2]
    };
    let live = public[*p];
    let rho_col = public[*p + 1..*p + 1 + k_col].to_vec();
    let rho_row = public[*p + 1 + k_col..*p + 1 + k_col + k_row].to_vec();
    let value = public[*p + 1 + k_col + k_row];
    *p += 2 + k_col + k_row;
    (
        key,
        MatrixClaim {
            row: Weight::low_eq(vec![live], rho_row),
            col: Weight::low_eq(vec![live], rho_col),
            value,
        },
    )
}

/// Walk the published fold blocks from `tail0`: both endpoint deltas zero
/// per fold, the accumulator claims rebuilt from the PUBLIC SEGMENT alone,
/// and every boundary-expanded low-fold eq public validated against the
/// PUBLISHED ρ coordinates. `locs[keyed_from..]` are the KEYED groups (the
/// sigma slots ride the uniform tape's tail). Returns the rebuilt claims,
/// their keys, and the offset just past the last entry.
fn check_fold_publics(
    public: &[F128],
    tail0: usize,
    locs: &[FoldLoc],
    alpha_recs: &[AlphaRec],
    keyed_from: usize,
) -> (
    Vec<flock_core::matrix_fold::MatrixClaim>,
    Vec<[F128; 2]>,
    usize,
) {
    use flock_core::matrix_fold::MatrixClaim;
    let width = |i: usize, l: &FoldLoc| 2 + l.k_col + l.k_row + if i >= keyed_from { 2 } else { 0 };
    let mut p = tail0;
    let mut rebuilt: Vec<MatrixClaim> = Vec::new();
    let mut keys: Vec<[F128; 2]> = Vec::new();
    for (i, loc) in locs.iter().enumerate() {
        let (k, c) = read_acc_entry(public, &mut p, i >= keyed_from, loc.k_col, loc.k_row);
        if i >= keyed_from {
            keys.push(k);
        }
        rebuilt.push(c);
    }
    for &(idx, fi, row_side, h) in alpha_recs {
        let base: usize = tail0
            + locs[..fi]
                .iter()
                .enumerate()
                .map(|(i, l)| width(i, l))
                .sum::<usize>()
            + if fi >= keyed_from { 3 } else { 1 }; // past key + live
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
            public[idx], e,
            "boundary-expanded low-fold eq public (fold {fi}, h {h})"
        );
    }
    (rebuilt, keys, p)
}

// ---------------------------------------------------------------------------
// The JAGGED fold groups on the merge tape (the count win) — the five
// helpers' siblings for the layout-table folds, which ride the SAME
// aggregate challenger AFTER the uniform folds (option a, Ron's call).
// mvp11_jagged_fold_tape is the standalone template these extract.
// ---------------------------------------------------------------------------

/// One absorbed JAGGED claim's stream ordinals: the tagged row weight and
/// the col point + value. `terms` empty ⇔ an Eq row (the tag pins which).
struct JClaimLoc {
    /// Eq rows: the SCALE word's ordinal (the zero-claim scale — `1` for a
    /// fresh claim, the inherited entry's live word otherwise). Combo rows:
    /// unused (0).
    row_scale_v: usize,
    /// Eq rows: (point ordinal, len). Combo rows: unused (0, 0).
    row_pt: (usize, usize),
    /// Combo rows: (coeff ordinal, address) per term — the address WORD
    /// sits at coeff ordinal + 1, pinned to its REGISTRY constant.
    terms: Vec<(usize, u32)>,
    col_v: usize,
    val_v: usize,
}

/// One jagged fold group's located surfaces — [`FoldLoc`]'s sibling.
struct JaggedFoldLoc {
    /// The group's `(k_row, n_claims)` shape header word.
    hdr_v: usize,
    claims: Vec<JClaimLoc>,
    lam_ch0: usize,
    col_v: usize,
    col_ch0: usize,
    n_col: usize,
    bridge_v: usize,
    mu_ch0: usize,
    row_v: usize,
    row_ch0: usize,
    k_row: usize,
    out_v: usize,
}

/// The jagged groups' op tape: per key, the group label + digest payload,
/// then the jagged fold's ops — the label, the shape header, the tagged
/// variable-width claim blocks, and the two sumchecks. Width-driven.
fn jagged_fold_region_ops(
    cfg: TowerConfig,
    keys: &[([u8; 32], Vec<flock_core::matrix_fold::JaggedClaim>)],
) -> Vec<flock_core::transcript_record::TranscriptOp> {
    use flock_core::matrix_fold::JaggedRowWeight;
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut want: Vec<Op> = Vec::new();
    let grinding = tower_fold_grinding(cfg);
    for (_, cs) in keys {
        let n_col = cs[0].col.len();
        let k_row = cs
            .iter()
            .find_map(|c| match &c.row {
                JaggedRowWeight::Eq(_, p) => Some(p.len()),
                JaggedRowWeight::Combo(_) => None,
            })
            .expect("every jagged key carries at least one Eq claim");
        want.push(Op::Label(b"flock-aggregate-jagged-v0".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.push(Op::Label(b"flock-jagged-fold-v0".to_vec()));
        want.push(Op::ObserveScalar); // the (k_row, n_claims) shape header
        for c in cs {
            match &c.row {
                JaggedRowWeight::Eq(_, p) => {
                    // tag, then the zero-claim SCALE, then the point.
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveSlice(p.len()));
                }
                JaggedRowWeight::Combo(t) => {
                    want.push(Op::ObserveScalar);
                    for _ in t {
                        want.extend([Op::ObserveScalar, Op::ObserveScalar]);
                    }
                }
            }
            want.push(Op::ObserveSlice(n_col));
            want.push(Op::ObserveScalar);
        }
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len()));
        for _ in 0..n_col {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.extend(std::iter::repeat_n(Op::ObserveScalar, cs.len()));
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len()));
        for _ in 0..k_row {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.push(Op::ObserveScalar);
    }
    want
}

/// Payload ordinals of `ObserveBytes` operations immediately following a
/// particular label. `Pow` also contributes one payload (its nonce), so fixed
/// payload offsets are invalid as soon as grinding is enabled.
fn labeled_bytes_payloads(
    ops: &[flock_core::transcript_record::TranscriptOp],
    label: &[u8],
) -> Vec<usize> {
    use flock_core::transcript_record::TranscriptOp as Op;
    let mut out = Vec::new();
    let mut payload = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if matches!(op, Op::ObserveBytes(_))
            && i > 0
            && matches!(&ops[i - 1], Op::Label(l) if l.as_slice() == label)
        {
            out.push(payload);
        }
        if matches!(
            op,
            Op::ObserveBytes(_) | Op::Pow { .. } | Op::LegacyPow { .. }
        ) {
            payload += 1;
        }
    }
    out
}

/// Locate + pin the jagged groups AFTER the uniform folds — the value and
/// challenge counters CONTINUE from the callers' (which is why
/// [`locate_and_pin_folds`] hands its counters back), and the digest
/// payloads continue after bind's two. Everything pins field-for-field:
/// the shape header, every tagged weight (Combo ADDRESS words against
/// their registry constants), col points, values, rounds, bridge, output.
#[allow(clippy::too_many_arguments)]
fn locate_and_pin_jagged_folds(
    keys: &[([u8; 32], Vec<flock_core::matrix_fold::JaggedClaim>)],
    fps: &[&flock_core::matrix_fold::FoldProof],
    vals_rec: &[F128],
    chals: &[F128],
    payloads: &[Vec<u8>],
    digest_payloads: &[usize],
    mut vcur: usize,
    mut ccur: usize,
) -> Vec<JaggedFoldLoc> {
    use flock_core::matrix_fold::JaggedRowWeight;
    assert_eq!(keys.len(), fps.len(), "one fold per jagged key");
    assert_eq!(
        keys.len(),
        digest_payloads.len(),
        "one digest payload per jagged key"
    );
    let locs: Vec<JaggedFoldLoc> = keys
        .iter()
        .zip(fps)
        .zip(digest_payloads)
        .map(|(((digest, cs), fp), &digest_payload)| {
            assert_eq!(
                payloads[digest_payload],
                digest.to_vec(),
                "the group's digest payload"
            );
            let n_col = cs[0].col.len();
            let k_row = cs
                .iter()
                .find_map(|c| match &c.row {
                    JaggedRowWeight::Eq(_, p) => Some(p.len()),
                    JaggedRowWeight::Combo(_) => None,
                })
                .expect("every jagged key carries at least one Eq claim");
            assert_eq!(
                vals_rec[vcur],
                F128::new(k_row as u64, cs.len() as u64),
                "the group's shape header word"
            );
            let hdr_v = vcur;
            vcur += 1;
            let claims: Vec<JClaimLoc> = cs
                .iter()
                .map(|c| {
                    let tag_v = vcur;
                    let (row_pt, terms) = match &c.row {
                        JaggedRowWeight::Eq(scale, p) => {
                            assert_eq!(vals_rec[tag_v], F128::new(0, p.len() as u64), "eq row tag");
                            assert_eq!(vals_rec[tag_v + 1], *scale, "eq row SCALE on the stream");
                            assert_eq!(
                                &vals_rec[tag_v + 2..tag_v + 2 + p.len()],
                                &p[..],
                                "eq row point on the stream"
                            );
                            vcur = tag_v + 2 + p.len();
                            ((tag_v + 2, p.len()), Vec::new())
                        }
                        JaggedRowWeight::Combo(t) => {
                            assert_eq!(
                                vals_rec[tag_v],
                                F128::new(1, t.len() as u64),
                                "combo row tag"
                            );
                            let mut terms = Vec::with_capacity(t.len());
                            for (j, &(coeff, addr)) in t.iter().enumerate() {
                                let cv = tag_v + 1 + 2 * j;
                                assert_eq!(vals_rec[cv], coeff, "combo coeff on the stream");
                                assert_eq!(
                                    vals_rec[cv + 1],
                                    F128::new(addr as u64, 0),
                                    "combo ADDRESS word == the registry constant"
                                );
                                terms.push((cv, addr));
                            }
                            vcur = tag_v + 1 + 2 * t.len();
                            ((0, 0), terms)
                        }
                    };
                    let col_v = vcur;
                    assert_eq!(
                        &vals_rec[col_v..col_v + n_col],
                        &c.col[..],
                        "col point (σ) on the stream"
                    );
                    let val_v = col_v + n_col;
                    assert_eq!(vals_rec[val_v], c.value, "claim value on the stream");
                    vcur = val_v + 1;
                    JClaimLoc {
                        row_scale_v: tag_v + 1,
                        row_pt,
                        terms,
                        col_v,
                        val_v,
                    }
                })
                .collect();
            let lam_ch0 = ccur;
            ccur += cs.len();
            let col_v = vcur;
            let col_ch0 = ccur;
            vcur += 2 * n_col;
            ccur += n_col;
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
            for (j, &(q1, qinf)) in fp.col_rounds.iter().enumerate() {
                assert_eq!(vals_rec[col_v + 2 * j], q1, "jagged col round q(1)");
                assert_eq!(vals_rec[col_v + 2 * j + 1], qinf, "jagged col round q(inf)");
            }
            assert_eq!(
                &vals_rec[bridge_v..bridge_v + cs.len()],
                &fp.bridge[..],
                "the jagged bridge on the stream"
            );
            for (j, &(q1, qinf)) in fp.row_rounds.iter().enumerate() {
                assert_eq!(vals_rec[row_v + 2 * j], q1, "jagged row round q(1)");
                assert_eq!(vals_rec[row_v + 2 * j + 1], qinf, "jagged row round q(inf)");
            }
            assert_eq!(
                vals_rec[out_v], fp.value,
                "jagged output value on the stream"
            );
            JaggedFoldLoc {
                hdr_v,
                claims,
                lam_ch0,
                col_v,
                col_ch0,
                n_col,
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
    locs
}

/// Replay the jagged folds' endpoint identities from LOCATED words alone
/// and return the located entries — [`replay_fold_endpoints`]'s sibling.
fn replay_jagged_fold_endpoints(
    locs: &[JaggedFoldLoc],
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
    let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
    locs.iter()
        .map(|loc| {
            let k = loc.claims.len();
            let lam: Vec<F128> = (0..k).map(|i| chals[loc.lam_ch0 + i]).collect();
            let target_c = loc
                .claims
                .iter()
                .zip(&lam)
                .fold(F128::ZERO, |acc, (cl, &l)| acc + l * vals_rec[cl.val_v]);
            let (run_c, rho_col) = replay_rounds(target_c, loc.col_v, loc.col_ch0, loc.n_col);
            let expect_c =
                loc.claims
                    .iter()
                    .zip(&lam)
                    .enumerate()
                    .fold(F128::ZERO, |acc, (i, (cl, &l))| {
                        let w = (0..loc.n_col).fold(F128::ONE, |w, j| {
                            w * (F128::ONE + vals_rec[cl.col_v + j] + rho_col[j])
                        });
                        acc + l * w * vals_rec[loc.bridge_v + i]
                    });
            assert_eq!(
                run_c, expect_c,
                "jagged col endpoint closes from located words"
            );

            let mus: Vec<F128> = (0..k).map(|i| chals[loc.mu_ch0 + i]).collect();
            let target_r = (0..k).zip(&mus).fold(F128::ZERO, |acc, (i, &m)| {
                acc + m * vals_rec[loc.bridge_v + i]
            });
            let (run_r, rho_row) = replay_rounds(target_r, loc.row_v, loc.row_ch0, loc.k_row);
            let w_mu = loc
                .claims
                .iter()
                .zip(&mus)
                .fold(F128::ZERO, |acc, (cl, &m)| {
                    let rw = if cl.terms.is_empty() {
                        // The eq product SEEDED by the zero-claim scale.
                        (0..cl.row_pt.1).fold(vals_rec[cl.row_scale_v], |w, j| {
                            w * (F128::ONE + vals_rec[cl.row_pt.0 + j] + rho_row[j])
                        })
                    } else {
                        cl.terms.iter().fold(F128::ZERO, |a, &(cv, addr)| {
                            let e = rho_row.iter().enumerate().fold(F128::ONE, |e, (l, &r)| {
                                e * (F128::ONE + bit((addr >> l) & 1 == 1) + r)
                            });
                            a + vals_rec[cv] * e
                        })
                    };
                    acc + m * rw
                });
            assert_eq!(
                run_r,
                w_mu * vals_rec[loc.out_v],
                "jagged row endpoint closes from located words"
            );
            MatrixClaim {
                row: Weight::eq(rho_row),
                col: Weight::eq(rho_col),
                value: vals_rec[loc.out_v],
            }
        })
        .collect()
}

/// Emit the jagged fold groups in-circuit — [`emit_fold_region`]'s sibling
/// (mvp11_jagged_fold_tape's replay, extracted): MergedRoundGate rounds,
/// PrefixGate eq products for the weight evals (a Combo row's ADDRESS bits
/// bake as ow/zw — registry constants, count-independent — with its
/// coefficients as absorbed stream wires), both endpoints as COPY
/// CONSTRAINTS, the entries returned for publishing. The LAST group's
/// output value takes the tail-input treatment ([`emit_fold_region`] must
/// then run with `tail_input_last = false`).
#[allow(clippy::too_many_arguments)]
fn emit_jagged_fold_region(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    mrs: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    locs: &[JaggedFoldLoc],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    challenge_locs: &[(usize, usize)],
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vmap: &[Option<usize>],
    vals_rec: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
) -> Vec<FoldPub> {
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    let chw = |ch: usize| -> Wire {
        let (fin, offset) = challenge_locs[ch];
        squeeze_word_wire(outs, trace, fin, offset)
    };
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
    let mut fold_pubs: Vec<FoldPub> = Vec::new();
    for (fi, loc) in locs.iter().enumerate() {
        let k = loc.claims.len();
        let lam_w: Vec<Wire> = (0..k).map(|i| chw(loc.lam_ch0 + i)).collect();
        let mut run_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            run_w = sb.gate(macs, &[run_w, lam_w[i], wv(cl.val_v)])[0];
        }
        let mut rho_col_w: Vec<Wire> = Vec::with_capacity(loc.n_col);
        for j in 0..loc.n_col {
            let r_w = chw(loc.col_ch0 + j);
            rho_col_w.push(r_w);
            run_w = sb.gate(
                mrs,
                &[run_w, wv(loc.col_v + 2 * j), wv(loc.col_v + 2 * j + 1), r_w],
            )[0];
        }
        let mut exp_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let fs: Vec<(Wire, Wire)> = (0..loc.n_col)
                .map(|j| (wv(cl.col_v + j), rho_col_w[j]))
                .collect();
            let cw = prefix(sb, ow, &fs);
            let t = sb.gate(macs, &[zw, cw, wv(loc.bridge_v + i)])[0];
            exp_w = sb.gate(macs, &[exp_w, lam_w[i], t])[0];
        }
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
            run2_w = sb.gate(
                mrs,
                &[
                    run2_w,
                    wv(loc.row_v + 2 * j),
                    wv(loc.row_v + 2 * j + 1),
                    r_w,
                ],
            )[0];
        }
        let mut wmu_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let rw = if cl.terms.is_empty() {
                let fs: Vec<(Wire, Wire)> = (0..cl.row_pt.1)
                    .map(|j| (wv(cl.row_pt.0 + j), rho_row_w[j]))
                    .collect();
                // SEEDED by the claim's own scale wire (the zero-claim
                // form) rather than the constant 1 — free, the prefix
                // chain already takes a seed.
                prefix(sb, wv(cl.row_scale_v), &fs)
            } else {
                let mut acc = zw;
                for &(cv, addr) in &cl.terms {
                    let fs: Vec<(Wire, Wire)> = rho_row_w
                        .iter()
                        .enumerate()
                        .map(|(l, &r)| (r, if (addr >> l) & 1 == 1 { ow } else { zw }))
                        .collect();
                    let e = prefix(sb, ow, &fs);
                    acc = sb.gate(macs, &[acc, wv(cv), e])[0];
                }
                acc
            };
            wmu_w = sb.gate(macs, &[wmu_w, mu_w[i], rw])[0];
        }
        let value = if fi + 1 == locs.len() {
            vals.push(vals_rec[loc.out_v]);
            sb.input()
        } else {
            wv(loc.out_v)
        };
        let rhs_w = sb.gate(macs, &[zw, wmu_w, value])[0];
        sb.connect(run2_w, rhs_w);
        fold_pubs.push(FoldPub {
            live: ow,
            rho_col: rho_col_w,
            rho_row: rho_row_w,
            value,
        });
    }
    fold_pubs
}

/// Walk the published jagged entries from `at` — [`check_fold_publics`]'s
/// sibling (no boundary publics: jagged lows are trivially 1). The jagged
/// group is KEYED, so under the spine layout every entry leads with its
/// key; `keyed = false` is the ACC_CHAIN layout, which the lane's
/// single-key registry role leaves as it was. Returns the rebuilt entries,
/// their keys, and the offset just past the last one.
fn check_jagged_fold_publics(
    public: &[F128],
    at: usize,
    locs: &[JaggedFoldLoc],
    keyed: bool,
) -> (
    Vec<flock_core::matrix_fold::MatrixClaim>,
    Vec<[F128; 2]>,
    usize,
) {
    let mut p = at;
    let mut out = Vec::with_capacity(locs.len());
    let mut keys = Vec::new();
    for loc in locs {
        let (k, c) = read_acc_entry(public, &mut p, keyed, loc.n_col, loc.k_row);
        if keyed {
            keys.push(k);
        }
        out.push(c);
    }
    (out, keys, p)
}

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
    let lcs = leaf_boolean_lcs(lo);
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

/// A node's PUBLISHED ACC_MAIN block, entry for entry — the surface a
/// spine parent inherits, which is not quite the accumulator the fold
/// returns: the keyed groups have a fixed number of SLOTS (one per child
/// role), and a slot this node had no fold for is present as a DEAD entry
/// (zero key, zero claim). That fixed shape is the whole point — a base
/// node and a steady node publish the same layout, so ONE parent circuit
/// reads either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainBlock {
    per_type: Vec<(MatrixClaim, MatrixClaim)>,
    per_element: Vec<(MatrixClaim, MatrixClaim)>,
    /// Slot order: 0 is the FL-child slot, 1 the NODE-child slot.
    sigma: Vec<([F128; 2], MatrixClaim)>,
    jagged: Vec<([F128; 2], MatrixClaim)>,
    /// The PASSENGER, same two slots: (sigma-shaped, jagged-shaped).
    passenger: Vec<([F128; 2], MatrixClaim)>,
}

/// The two keyed slots every node publishes: the fresh FL child's, then
/// the node child's.
const N_KEY_SLOTS: usize = 2;

/// A circuit digest as the two field words a transcript absorbs it as —
/// the form the published keys and the match-gate compare in.
fn digest_f128(d: &[u8; 32]) -> [F128; 2] {
    let w = |o: usize| {
        F128::new(
            u64::from_le_bytes(d[o..o + 8].try_into().unwrap()),
            u64::from_le_bytes(d[o + 8..o + 16].try_into().unwrap()),
        )
    };
    [w(0), w(16)]
}

/// A claim scaled by a BIT: `1` returns it unchanged, `0` returns the zero
/// claim at the same POINTS — weights identically zero, value zero. The
/// points stay because in-circuit they are the child's published words and
/// only the lows and the value pass through the gate.
fn gate_claim(c: &MatrixClaim, live: bool) -> MatrixClaim {
    if live {
        return c.clone();
    }
    MatrixClaim {
        row: Weight::low_eq(vec![F128::ZERO], c.row.point.clone()),
        col: Weight::low_eq(vec![F128::ZERO], c.col.point.clone()),
        value: F128::ZERO,
    }
}

/// `true` when an entry's LIVE word is nonzero — a claim that is about
/// something.
fn entry_live(c: &MatrixClaim) -> bool {
    c.row.low[0] != F128::ZERO
}

/// THE SPINE (wall 3): the node child's published block riding in as this
/// node's MAIN-fold prior. `node_child` is that child's index in `los`
/// (the steady shape: 1 — child 0 is the fresh FL).
pub struct SpineIn<'a> {
    node_child: usize,
    prior: &'a MainBlock,
    /// THE ADVERSARIAL LEG: re-witness the match-gate's mac rows as a
    /// cheating prover's world — the mismatched slot CLAIMS its digests
    /// match and folds the orphan live — with every row self-satisfying,
    /// then assert the proof dies on exactly the wiring product. The
    /// builder's honest asserts all still run (the publics are untouched).
    forge: bool,
}

/// Everything [`build_node_outer_app`] hands back.
// Read only by the in-file `#[test]` benches; the lib unit sees the fields
// write-only.
#[cfg_attr(not(test), allow(dead_code))]
pub struct NodeOut {
    lo: LeafOuter,
    /// The MAIN fold's accumulator — LIVE entries only, the thing a root
    /// discharges.
    acc: flock_core::aggregate::Accumulator,
    /// The LAST online iteration (steady state under repetition).
    online: Online,
    /// One record per online iteration (1 + steady_reps of them) — the
    /// bench's medians come from here, one setup for all of them.
    onlines: Vec<Online>,
    app_base: Option<usize>,
    lane_acc: Option<flock_core::aggregate::Accumulator>,
    /// The published ACC_MAIN + passenger blocks — what a spine parent
    /// inherits.
    block: MainBlock,
}

/// A LOWER-registry accumulator lane riding through an internal node
/// (task 6): the two children each carry an accumulator over a registry
/// that is NOT the fold's own (the chain registry at the first level), so
/// it cannot join the node's fold as a prior — it folds in its OWN
/// priors-only aggregate, whose prior surfaces connect WIRE-TO-WIRE to
/// the children's published accumulator claims (`claims_base` locates
/// them; a prior's surface IS what the child published).
pub struct ChainLane<'a> {
    registry: &'a crate::schedule::Registry,
    mats: &'a [flock_core::aggregate::TypeMatrices<'a>],
    circs: &'a [&'a dyn flock_core::lincheck::LincheckCircuit],
    /// The lane's sigma table owner (the chain circuit).
    circuit: &'a flock_core::circuit::Circuit,
    /// The lane's jagged table owner (the chain LAYOUT — the count win's
    /// per-digest key, inherited priors-only through internal nodes).
    params: &'a flock_core::pcs::jagged::JaggedParams,
    priors: &'a [&'a flock_core::aggregate::Accumulator],
    /// The published `[rho_col | rho_row | value]` fold blocks' base in
    /// EACH child's public segment (every child shares the layout).
    claims_base: usize,
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
/// builder composes with ITSELF — `build_node_outer_app(&[&n0, &n1], ..)` is the
/// level-2 node consuming its own outputs. The children must share one
/// circuit digest (the foldability key); their claims land at unrelated FS
/// points. Every tape pin, connect, and checker walk of the 2→1 milestone
/// lives inside — the builder IS the test.
///
/// APPLICATION-STATEMENT plumbing: when the children carry an app block
/// (`app_stmt` = its offset in their public segments — the hash-chain span
/// (h_start, h_end), 8 words), the node connects left.h_end ==
/// right.h_start wire-to-wire and publishes the combined span as its OWN
/// app block, returning that block's offset — so the output feeds the next
/// level with the same plumbing.
pub fn build_node_outer_app(
    cfg: TowerConfig,
    los: &[&LeafOuter],
    app_stmt: Option<usize>,
    lane: Option<ChainLane<'_>>,
    spine: Option<SpineIn<'_>>,
) -> NodeOut {
    use flock_core::aggregate;
    use flock_core::matrix_fold::FoldProof;
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
    let forge_match = spine.as_ref().is_some_and(|sp| sp.forge);
    let lo0 = los[0];
    // MIXED DIGESTS ARE THE SPINE (wall 3): a steady node's children are a
    // FRESH FL and the PREVIOUS NODE, which are different circuits. They
    // still share the registry, the publics length and the lane count (the
    // wall-2 envelope), which is what makes ONE child region walk either —
    // only the fold KEYS differ, one slot per child. Without a spine the
    // old rule stands: one key, so one digest.
    if spine.is_none() {
        for lo in los {
            assert_eq!(
                lo.shape.circuit.digest(),
                lo0.shape.circuit.digest(),
                "a fresh-only node folds every child under ONE key"
            );
        }
    } else {
        assert_eq!(n_kids, 2, "the spine's steady node is 2->1");
    }
    for lo in los {
        assert_eq!(
            lo.shape.registry.digest(),
            lo0.shape.registry.digest(),
            "every child, ONE envelope registry"
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
    let lcs = leaf_boolean_lcs(lo0);
    let mats = leaf_boolean_mats(lo0);
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
    // THE KEYED GROUPS, per child SHAPE (wall 3): a fresh-only node has ONE
    // key (every child is the same circuit); the SPINE has one SLOT PER
    // CHILD, because its children are different circuits and claims about
    // different permutations — different layouts — cannot fold together.
    // The layout is a shape constant of the child circuit, so the key that
    // names the circuit names the table.
    let key_circuits: Vec<&flock_core::circuit::Circuit> = match &spine {
        None => vec![&lo0.shape.circuit],
        Some(_) => los.iter().map(|lo| &lo.shape.circuit).collect(),
    };
    let n_keys = key_circuits.len();
    let key_digests: Vec<[u8; 32]> = key_circuits.iter().map(|c| c.digest()).collect();
    // Which children's FRESH claims ride each key: all of them under one
    // key, or child j under slot j.
    let key_kids: Vec<Vec<usize>> = match &spine {
        None => vec![(0..n_kids).collect()],
        Some(_) => (0..n_kids).map(|i| vec![i]).collect(),
    };
    let params_j: Vec<flock_core::pcs::jagged::JaggedParams> = (0..n_keys)
        .map(|j| {
            let i = key_kids[j][0];
            flock_core::pcs::jagged::JaggedParams::from_heights(
                &unions[i].jagged_heights(),
                unions[i].n_log(),
                los[i].commitment.params.m - flock_core::pcs::LOG_PACKING,
            )
        })
        .collect();
    let jags: Vec<&flock_core::matrix_fold::JaggedAssertion> =
        rts.iter().map(|rt| &rt.jag).collect();
    let jagged_p: Vec<aggregate::JaggedKeyProve<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_digests[j],
                &params_j[j],
                key_kids[j].iter().map(|&i| jags[i]).collect(),
            )
        })
        .collect();
    let jagged_v: Vec<aggregate::JaggedKeyVerify<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_digests[j],
                key_kids[j].iter().map(|&i| jags[i]).collect(),
            )
        })
        .collect();
    let sigma_keys: Vec<aggregate::SigmaKey<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_circuits[j],
                key_kids[j].iter().map(|&i| &sigmas[i]).collect(),
            )
        })
        .collect();
    // THE PRIOR (the spine): the node child's published block, normalized
    // to this node's slots — an inherited entry whose published key names
    // the slot's circuit folds; one that does not is GATED to the zero
    // claim and its live original becomes an ORPHAN, which the passenger
    // carries rather than drops.
    let prior_acc: Option<aggregate::Accumulator> = spine.as_ref().map(|sp| {
        let p = sp.prior;
        assert_eq!(p.sigma.len(), N_KEY_SLOTS, "the prior's sigma slots");
        assert_eq!(p.jagged.len(), N_KEY_SLOTS, "the prior's jagged slots");
        let want: Vec<[F128; 2]> = key_digests.iter().map(digest_f128).collect();
        let norm = |slots: &[([F128; 2], MatrixClaim)]| -> Vec<([u8; 32], MatrixClaim)> {
            slots
                .iter()
                .enumerate()
                .map(|(j, (k, c))| {
                    let hit = *k == want[j];
                    // The FL slot is the SAME shape at every level — its
                    // key is wired equal in-circuit, so a miss here is a
                    // broken spine, not a case the passenger covers.
                    assert!(j > 0 || hit, "the FL slot's inherited key must match");
                    (key_digests[j], gate_claim(c, hit))
                })
                .collect()
        };
        aggregate::Accumulator {
            registry_digest: registry.digest(),
            per_type: p.per_type.clone(),
            per_element: p.per_element.clone(),
            sigma: norm(&p.sigma),
            jagged: norm(&p.jagged),
        }
    });
    let priors: Vec<&aggregate::Accumulator> = prior_acc.iter().collect();
    let mut chp = FsChallenger::with_chained_blake3(M11_NODE_DOMAIN);
    let (agg, acc_p) = aggregate::prove_aggregate_classes_with_grinding(
        registry,
        &mats,
        &lcs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        &sigma_keys,
        &jagged_p,
        &priors,
        tower_fold_grinding(cfg),
        &mut chp,
    )
    .expect("the node fold proves");
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_NODE_DOMAIN));
    let acc_v = aggregate::verify_aggregate_classes_with_grinding(
        registry,
        &bool_asserts,
        &el_asserts,
        &sigma_keys,
        &jagged_v,
        &priors,
        &agg,
        tower_fold_grinding(cfg),
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
        acc_v.discharge_sigma(&key_circuits),
        "the sigma group discharges"
    );
    assert_eq!(acc_v.jagged.len(), n_keys, "one jagged entry per key");
    let jag_tables: Vec<([u8; 32], &flock_core::pcs::jagged::JaggedParams)> = (0..n_keys)
        .map(|j| (key_digests[j], &params_j[j]))
        .collect();
    assert!(
        acc_v.discharge_jagged(&jag_tables),
        "the folded jagged entries discharge against their children's layouts"
    );

    // The fold groups in aggregate order, from the CHILDREN'S OWN
    // assertion data (the same constructors the verifier gathers with).
    let bc: Vec<_> = rts
        .iter()
        .map(|rt| rt.mat_assert.claims(registry))
        .collect();
    let ec: Vec<_> = rts
        .iter()
        .zip(&unions)
        .map(|(rt, u)| rt.el_assert.claims(u))
        .collect();
    // One group per (type, side): the PRIOR's claim first when a spine
    // rides (`gather`'s order — priors, then assertions), then one per
    // child. The fold machinery is claim-count-generic, so both arity and
    // the prior enter here only as the length of these vectors.
    let pri = prior_acc.as_ref();
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::new();
    for t in 0..n_bool {
        for side in 0..2 {
            let mut g: Vec<MatrixClaim> = pri
                .map(|p| {
                    if side == 0 {
                        p.per_type[t].0.clone()
                    } else {
                        p.per_type[t].1.clone()
                    }
                })
                .into_iter()
                .collect();
            g.extend((0..n_kids).map(|i| {
                if side == 0 {
                    bc[i][t].0.clone()
                } else {
                    bc[i][t].1.clone()
                }
            }));
            fold_claims.push(g);
        }
    }
    for t in 0..n_el {
        for side in 0..2 {
            let mut g: Vec<MatrixClaim> = pri
                .map(|p| {
                    if side == 0 {
                        p.per_element[t].0.clone()
                    } else {
                        p.per_element[t].1.clone()
                    }
                })
                .into_iter()
                .collect();
            g.extend((0..n_kids).map(|i| {
                if side == 0 {
                    ec[i][t].0.clone()
                } else {
                    ec[i][t].1.clone()
                }
            }));
            fold_claims.push(g);
        }
    }
    // The SIGMA slots close the uniform tape, one per key.
    let n_uni = fold_claims.len();
    for j in 0..n_keys {
        let mut g: Vec<MatrixClaim> = pri.map(|p| p.sigma[j].1.clone()).into_iter().collect();
        g.extend(key_kids[j].iter().flat_map(|&i| sigmas[i].claims()));
        fold_claims.push(g);
    }
    let mut fold_proofs: Vec<&FoldProof> = Vec::new();
    for t in 0..n_bool {
        fold_proofs.push(&agg.folds[t].0);
        fold_proofs.push(&agg.folds[t].1);
    }
    for t in 0..n_el {
        fold_proofs.push(&agg.el_folds[t].0);
        fold_proofs.push(&agg.el_folds[t].1);
    }
    fold_proofs.extend(agg.sigma_folds.iter());
    let n_folds = fold_claims.len();

    // ---- the fold tape, pinned through the width-driven helpers ----
    let t_shape = rec.shape();
    let ops = flatten_ops(t_shape.ops());
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    want.extend(fold_region_ops(cfg, &fold_claims[..n_uni]));
    // The sigma group binds per key (wall 3): its label + digest precede
    // each key's fold, exactly as the jagged groups bind.
    for j in 0..n_keys {
        want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.extend(fold_region_ops(cfg, &fold_claims[n_uni + j..n_uni + j + 1]));
    }
    // The jagged groups ride the SAME tape after the uniform folds — the
    // prior's (gated) entry first, then that key's children's claims.
    let jagged_keys: Vec<([u8; 32], Vec<flock_core::matrix_fold::JaggedClaim>)> = (0..n_keys)
        .map(|j| {
            let mut cs: Vec<flock_core::matrix_fold::JaggedClaim> = pri
                .map(|p| {
                    flock_core::matrix_fold::JaggedClaim::from_folded(&p.jagged[j].1)
                        .expect("an inherited jagged entry is scaled plain eq")
                })
                .into_iter()
                .collect();
            cs.extend(
                key_kids[j]
                    .iter()
                    .flat_map(|&i| jags[i].claims().into_iter().cloned()),
            );
            (key_digests[j], cs)
        })
        .collect();
    want.extend(jagged_fold_region_ops(cfg, &jagged_keys));
    assert_eq!(ops, want.as_slice(), "the node tape is the expected shape");
    assert_eq!(
        rec.payloads()[0],
        registry.digest(),
        "bind: registry digest"
    );
    assert_eq!(
        rec.payloads()[1],
        vec![priors.len() as u8],
        "bind: prior count"
    );
    let sigma_payloads = labeled_bytes_payloads(&ops, b"flock-aggregate-sigma-v1");
    let jagged_payloads = labeled_bytes_payloads(&ops, b"flock-aggregate-jagged-v0");
    assert_eq!(
        sigma_payloads.len(),
        n_keys,
        "one sigma digest payload per key"
    );
    assert_eq!(
        jagged_payloads.len(),
        n_keys,
        "one jagged digest payload per key"
    );
    for j in 0..n_keys {
        assert_eq!(
            rec.payloads()[sigma_payloads[j]],
            key_digests[j].to_vec(),
            "the sigma slot {j} key payload"
        );
    }
    let (locs, vcur, ccur) = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let jfps: Vec<&FoldProof> = agg.jagged_folds.iter().collect();
    let jlocs = locate_and_pin_jagged_folds(
        &jagged_keys,
        &jfps,
        vals_rec,
        chals,
        rec.payloads(),
        &jagged_payloads,
        vcur,
        ccur,
    );
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
    for j in 0..n_keys {
        let (d, c) = &acc_v.sigma[j];
        assert_eq!(outs[n_uni + j], *c, "sigma slot {j} accumulator");
        assert_eq!(*d, key_digests[j], "sigma slot {j} key");
    }
    let jouts = replay_jagged_fold_endpoints(&jlocs, vals_rec, chals);
    for j in 0..n_keys {
        assert_eq!(
            jouts[j], acc_v.jagged[j].1,
            "the jagged slot {j} entry from located words"
        );
    }

    // ---- the LANE (task 6): the children's LOWER-registry accumulators
    // fold PRIORS-ONLY — natively here, in-circuit below. 3 groups
    // (bool A/B + sigma) × [priorL, priorR], no fresh claims. ----
    const LANE_DOMAIN: &[u8] = b"flock-chain-lane-v0";
    let lane_native = lane.as_ref().map(|ln| {
        let el_asserts_l: [(
            &UnionInstance<'_>,
            flock_core::element_r1cs::union::ElementAssertion,
        ); 0] = [];
        // The jagged key rides PRIORS-ONLY through the lane, exactly like
        // the lane's other groups: the FL children's chain-keyed entries
        // fold with no fresh claims.
        let ljagged_p: Vec<aggregate::JaggedKeyProve<'_>> =
            vec![(ln.circuit.digest(), ln.params, Vec::new())];
        let ljagged_v: Vec<aggregate::JaggedKeyVerify<'_>> =
            vec![(ln.circuit.digest(), Vec::new())];
        let mut chp = FsChallenger::with_chained_blake3(LANE_DOMAIN);
        let (lagg, lacc_p) = aggregate::prove_aggregate_classes_with_grinding(
            ln.registry,
            ln.mats,
            ln.circs,
            &[],
            &[],
            &el_asserts_l,
            &[(ln.circuit, Vec::new())],
            &ljagged_p,
            ln.priors,
            tower_fold_grinding(cfg),
            &mut chp,
        )
        .expect("the lane fold proves");
        let mut lrec = RecordingChallenger::new(FsChallenger::with_chained_blake3(LANE_DOMAIN));
        let lacc_v = aggregate::verify_aggregate_classes_with_grinding(
            ln.registry,
            &[],
            &el_asserts_l,
            &[(ln.circuit, Vec::new())],
            &ljagged_v,
            ln.priors,
            &lagg,
            tower_fold_grinding(cfg),
            &mut lrec,
        )
        .expect("the lane fold verifies");
        assert_eq!(lacc_p, lacc_v, "lane prover and verifier agree");
        assert_eq!(
            lacc_v.jagged.len(),
            1,
            "the lane carries the chain jagged key"
        );
        assert!(
            lacc_v.discharge_jagged(&[(ln.circuit.digest(), ln.params)]),
            "the lane's folded jagged entry discharges against the chain layout"
        );
        let lclaims: Vec<Vec<MatrixClaim>> = vec![
            ln.priors.iter().map(|p| p.per_type[0].0.clone()).collect(),
            ln.priors.iter().map(|p| p.per_type[0].1.clone()).collect(),
            ln.priors
                .iter()
                .map(|p| p.sigma.first().expect("lane prior sigma").1.clone())
                .collect(),
        ];
        let lproofs: Vec<&FoldProof> =
            vec![&lagg.folds[0].0, &lagg.folds[0].1, &lagg.sigma_folds[0]];
        let lops: Vec<Op> = flatten_ops(lrec.shape().ops());
        let lvals: Vec<F128> = lrec.values().to_vec();
        let lchals: Vec<F128> = lrec.challenges().to_vec();
        let mut want: Vec<Op> = vec![
            Op::Label(b"flock-aggregate-v0".to_vec()),
            Op::ObserveBytes(32),
            Op::ObserveBytes(1),
        ];
        let n_uni_l = lclaims.len() - 1;
        want.extend(fold_region_ops(cfg, &lclaims[..n_uni_l]));
        want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.extend(fold_region_ops(cfg, &lclaims[n_uni_l..]));
        // The inherited jagged claims (the priors' chain-keyed entries,
        // plain eq by construction) ride the same tape after.
        let ljagged_keys: Vec<([u8; 32], Vec<flock_core::matrix_fold::JaggedClaim>)> = vec![(
            ln.circuit.digest(),
            ln.priors
                .iter()
                .flat_map(|p| p.jagged.iter())
                .filter(|(d, _)| *d == ln.circuit.digest())
                .map(|(_, c)| {
                    flock_core::matrix_fold::JaggedClaim::from_folded(c)
                        .expect("prior jagged entries are plain eq")
                })
                .collect(),
        )];
        want.extend(jagged_fold_region_ops(cfg, &ljagged_keys));
        assert_eq!(lops, want, "the lane tape shape");
        assert_eq!(
            lrec.payloads()[0],
            ln.registry.digest(),
            "lane registry digest"
        );
        assert_eq!(
            lrec.payloads()[1],
            vec![ln.priors.len() as u8],
            "lane prior count"
        );
        let (llocs, lvcur, lccur) = locate_and_pin_folds(&lclaims, &lproofs, &lvals, &lchals);
        let ljfps: Vec<&FoldProof> = lagg.jagged_folds.iter().collect();
        let ljlocs = locate_and_pin_jagged_folds(
            &ljagged_keys,
            &ljfps,
            &lvals,
            &lchals,
            lrec.payloads(),
            &labeled_bytes_payloads(&lops, b"flock-aggregate-jagged-v0"),
            lvcur,
            lccur,
        );
        let louts = replay_fold_endpoints(&llocs, &lvals, &lchals);
        assert_eq!(louts[0], lacc_v.per_type[0].0, "lane boolean A");
        assert_eq!(louts[1], lacc_v.per_type[0].1, "lane boolean B");
        let (ld, lc2) = lacc_v.sigma.first().expect("lane sigma out");
        assert_eq!(louts[2], *lc2, "lane sigma accumulator");
        assert_eq!(
            *ld,
            ln.circuit.digest(),
            "lane sigma keys by the chain circuit"
        );
        let ljouts = replay_jagged_fold_endpoints(&ljlocs, &lvals, &lchals);
        assert_eq!(
            ljouts[0], lacc_v.jagged[0].1,
            "lane jagged entry from located words"
        );
        let lstream = lrec.shape().stream_words_duplex(LANE_DOMAIN);
        let lbytes = lstream.to_bytes(lrec.values(), lrec.payloads());
        (lacc_v, llocs, ljlocs, lstream, lbytes, lops, lchals, lvals)
    });

    // ---- ONE outer: two REAL child regions + the fold region ----

    {
        use crate::prover::UnionElementSlotInput;

        // The transcript is FORKED (the wiring runs on its own chain);
        // `merge_chain` splices the child's rows in at the fork point and
        // hands back one linear numbering plus the four cross-link wires.
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = merge_chain(
            t_shape.ops(),
            &t_shape.stream_words_duplex(M11_NODE_DOMAIN),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, chals);

        let env = envelope_shape();
        let split_b3 = n_kids == 2;
        let (fold_b3_primary_rows, b3_rows) = if split_b3 {
            let a = rts[0].b3_rows;
            let b = rts[1].b3_rows;
            let unsplit = (a + trace.rows.len()).max(b);
            let (on_a, balanced) = balance_extra_rows(a, b, trace.rows.len());
            if unsplit > (1usize << env.nu) {
                (Some(on_a), balanced)
            } else {
                (None, unsplit)
            }
        } else {
            (
                None,
                rts.iter().map(|rt| rt.b3_rows).sum::<usize>() + trace.rows.len(),
            )
        };
        if std::env::var("B3_CENSUS").is_ok() {
            let fold_pows = ops
                .iter()
                .filter(|op| matches!(op, Op::Pow { bits } if *bits != 0))
                .count();
            eprintln!(
                "  [node pow census] child checks {:?} | fold checks {} | standalone BLAKE rows 0",
                rts.iter().map(|rt| rt.pows.len()).collect::<Vec<_>>(),
                fold_pows,
            );
        }
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        // The node pins the envelope's nu* and canonical type set (wall 2).
        assert!(
            nu2_content <= env.nu,
            "node content nu {nu2_content} exceeds the envelope nu* {}",
            env.nu
        );
        let nu2 = env.nu;
        let mut sb = ShapeBuilder::new(nu2);
        // The DECLARED width is the envelope's (the max over child kinds at
        // the fixed point); a shallower child ladder rides the wide slot
        // with its high outputs unread, and one that exceeds it fails here.
        // The witness tables below build at `spread_w2`, so it must be the
        // DECLARED width.
        let spread_own2 = rts.iter().map(|rt| rt.spread_w).max().expect("a child");
        assert!(
            spread_own2 <= env.spread_w,
            "child ladder depth {spread_own2} exceeds the envelope spread width {}",
            env.spread_w
        );
        let spread_w2 = env.spread_w;
        let mut cs = ChildSlots::new_env(&mut sb, nu2, &env);
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
            .enumerate()
            .map(|(i, rt)| {
                let isl = sb.begin_island();
                let b3_slot = match (i, cs.q.b3_alt) {
                    (0, _) => cs.q.b3,
                    (1, Some(slot)) => slot,
                    (_, None) => cs.q.b3,
                    _ => panic!("split-BLAKE recursion supports exactly two children"),
                };
                let r = emit_real_child_region(
                    &mut sb,
                    &mut cs,
                    b3_slot,
                    rt,
                    &mut vals,
                    &mut hints,
                    &mut consts,
                );
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
        let iv_w = pack8(&crate::r1cs_hashes::fs_chain::IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [
            sb.fixed_public_input(iv_w[0]),
            sb.fixed_public_input(iv_w[1]),
        ];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(&flatten_ops(t_shape.ops()));
        let (chain_outs, ww) = emit_fs_chain_partitioned(
            &mut sb,
            cs.q.b3,
            fold_b3_primary_rows.map(|n| {
                (
                    cs.q.b3_alt
                        .expect("a balanced fold chain needs the second BLAKE slot"),
                    n,
                )
            }),
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
            &cross,
        );
        emit_recorded_pow_checks(
            &mut sb,
            cs.q.b3,
            cs.q.pow,
            iv2,
            &ops,
            &trace,
            &stream,
            &chain_outs,
            &ww,
            &mut vals,
            &mut consts,
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
            cs.fold_macs,
            cs.mrs,
            pfslot,
            pf_w,
            leslot,
            &locs,
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
            false, // the jagged group follows on the same tape
        );
        let jfold_pubs = emit_jagged_fold_region(
            &mut sb,
            cs.fold_macs,
            cs.mrs,
            pfslot,
            pf_w,
            &jlocs,
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );
        // ---- THE SPINE (wall 3): the node child's published ACC_MAIN
        // block IS this node's prior, read wire-to-wire out of that
        // child's public segment at the envelope's constant offset — the
        // lane's `claims_base` machinery, now on the MAIN fold. ----
        //
        // The registry-keyed matrix entries ride straight in (lows to the
        // child's live word, piece 1's wiring). The KEYED slots go through
        // the MATCH-GATE: slot `j` folds claims about child `j`'s tables,
        // and the entry inherited at slot `j` names whatever circuit the
        // CHILD's own slot `j` was about. Slot 0 (the fresh FL slot) is the
        // same FL shape at every level, so its key is a hard CONNECT — a
        // mismatch is a broken spine, not a case. Slot 1 (the node slot)
        // genuinely mismatches exactly once, at the first steady node over
        // a base node, and there the entry is gated to the zero claim and
        // its live original rides the PASSENGER instead of being dropped.
        //
        // `g = live · match` scales the claim's lows AND its value: a
        // gated-off entry must claim ZERO, not its old value about a
        // weight that is now zero.
        // The keyed slots' entry widths — the same for a live slot and a
        // dead one, which is what makes the layout readable at a constant
        // offset. `[key(2) | live | rho_col | rho_row | value]`.
        let sig_ent = 4 + locs[n_uni].k_col + locs[n_uni].k_row;
        let jag_ent = 4 + jlocs[0].n_col + jlocs[0].k_row;
        // The spine gadget's mac rows, bracketed for the ADVERSARIAL leg:
        // 26 rows exactly — 13 per keyed slot-1 gate (two is-eq gadgets of
        // 4 rows + m, g, gv, nm, h), the FL slot emitting none (its key is
        // a hard connect).
        let mac_spine0 = sb.rows_in_slot(cs.macs);
        let spine_w = spine.as_ref().map(|sp| {
            let e = &env;
            let rk = &regions[sp.node_child];
            let cp = |i: usize| rk.child_pub_w[i];
            // The assert-zero anchor for the gadget below: producers only,
            // no consumer edges (the lagrange lows' pattern).
            vals.push(F128::ZERO);
            let za = sb.public_input();
            // eq(a, b) as a BIT: d = a + b, an advice inverse w, and
            // z = 1 + d·w with z·d == 0 — z is 1 exactly when d is 0 (to
            // claim z = 1 with d ≠ 0 a prover needs w = 0, and then
            // z·d = d ≠ 0 fails the assert).
            let is_eq =
                |sb: &mut ShapeBuilder, vals: &mut Vec<F128>, a: Wire, b: Wire, d: F128| -> Wire {
                    let d_w = sb.gate(cs.macs, &[a, b, ow])[0];
                    vals.push(d.inv());
                    let inv_w = sb.input();
                    let p_w = sb.gate(cs.macs, &[zw, d_w, inv_w])[0];
                    let z_w = sb.gate(cs.macs, &[ow, p_w, ow])[0];
                    let chk = sb.gate(cs.macs, &[zw, z_w, d_w])[0];
                    sb.connect(chk, za);
                    z_w
                };
            // Walk the child's block exactly as this node publishes its
            // own — the layouts coincide, which is the shape fact the
            // whole spine rests on.
            let mut off = env_acc_main_base(e);
            let uni_off: Vec<usize> = (0..n_uni)
                .map(|i| {
                    let o = off;
                    off += 2 + locs[i].k_col + locs[i].k_row;
                    o
                })
                .collect();
            let sig_off: Vec<usize> = (0..N_KEY_SLOTS)
                .map(|_| {
                    let o = off;
                    off += sig_ent;
                    o
                })
                .collect();
            let jag_off: Vec<usize> = (0..N_KEY_SLOTS)
                .map(|_| {
                    let o = off;
                    off += jag_ent;
                    o
                })
                .collect();
            assert!(
                off - env_acc_main_base(e) <= ENV_ACC_MAIN_WORDS,
                "the prior's ACC_MAIN block overruns its reserved width"
            );
            // One keyed slot: the published key against this node's own,
            // then the gate. Returns (g, gated value, orphan gate h).
            let slot = |sb: &mut ShapeBuilder,
                        vals: &mut Vec<F128>,
                        o: usize,
                        j: usize,
                        ent: &([F128; 2], MatrixClaim),
                        k_col: usize,
                        k_row: usize|
             -> (Wire, Wire, Wire) {
                let live_w = cp(o + 2);
                let val_w = cp(o + 3 + k_col + k_row);
                let want = digest_f128(&key_digests[j]);
                if j == 0 {
                    // The FL slot: one shape at every level, so the key is
                    // an EQUALITY, wired, not a case.
                    sb.connect(cp(o), regions[j].cd_w[0]);
                    sb.connect(cp(o + 1), regions[j].cd_w[1]);
                    assert_eq!(ent.0, want, "the FL slot's inherited key is the FL circuit");
                    return (live_w, val_w, zw);
                }
                let m0 = is_eq(sb, vals, cp(o), regions[j].cd_w[0], ent.0[0] + want[0]);
                let m1 = is_eq(sb, vals, cp(o + 1), regions[j].cd_w[1], ent.0[1] + want[1]);
                let m_w = sb.gate(cs.macs, &[zw, m0, m1])[0];
                let g_w = sb.gate(cs.macs, &[zw, live_w, m_w])[0];
                let gv_w = sb.gate(cs.macs, &[zw, g_w, val_w])[0];
                // h = live · (1 + match) — the ORPHAN gate: live exactly
                // when this entry could not fold and must ride on.
                let nm_w = sb.gate(cs.macs, &[ow, m_w, ow])[0];
                let h_w = sb.gate(cs.macs, &[zw, live_w, nm_w])[0];
                (g_w, gv_w, h_w)
            };
            let sig: Vec<(Wire, Wire, Wire)> = (0..N_KEY_SLOTS)
                .map(|j| {
                    slot(
                        &mut sb,
                        &mut vals,
                        sig_off[j],
                        j,
                        &sp.prior.sigma[j],
                        locs[n_uni].k_col,
                        locs[n_uni].k_row,
                    )
                })
                .collect();
            let jag: Vec<(Wire, Wire, Wire)> = (0..N_KEY_SLOTS)
                .map(|j| {
                    slot(
                        &mut sb,
                        &mut vals,
                        jag_off[j],
                        j,
                        &sp.prior.jagged[j],
                        jlocs[0].n_col,
                        jlocs[0].k_row,
                    )
                })
                .collect();
            (uni_off, sig_off, jag_off, sig, jag)
        });
        if spine.is_some() {
            assert_eq!(
                sb.rows_in_slot(cs.macs) - mac_spine0,
                26,
                "the spine gadget's mac-row census"
            );
        }
        // THE POINTS-CONNECT (the count win's identity bind): value, σ,
        // row identities, and the structural words — see build_fl_node's
        // block for the argument; this is the same bind at node scale.
        let mut jag_const_rec: Vec<(F128, usize)> = Vec::new();
        {
            let mut jag_consts: Vec<(F128, Wire)> = Vec::new();
            let mut cw_j = |sb: &mut ShapeBuilder,
                            vals: &mut Vec<F128>,
                            rec2: &mut Vec<(F128, usize)>,
                            v: F128|
             -> Wire {
                if let Some(&(_, w)) = jag_consts.iter().find(|&&(x, _)| x == v) {
                    return w;
                }
                vals.push(v);
                rec2.push((v, sb.public_len()));
                let w = sb.public_input();
                jag_consts.push((v, w));
                w
            };
            for (gi, loc) in jlocs.iter().enumerate() {
                let mut ci = 0usize;
                // The INHERITED claim leads the group (aggregate's gather
                // order): its scale is the gate, its value the gated one,
                // and its points are the child's published words.
                if let Some((_, _, jag_off, _, jag)) = &spine_w {
                    let cl = &loc.claims[0];
                    let o = jag_off[gi];
                    let rk = &regions[spine.as_ref().unwrap().node_child];
                    let tag = cw_j(
                        &mut sb,
                        &mut vals,
                        &mut jag_const_rec,
                        F128::new(0, cl.row_pt.1 as u64),
                    );
                    sb.connect(wv(cl.row_scale_v - 1), tag);
                    sb.connect(wv(cl.row_scale_v), jag[gi].0);
                    for j in 0..loc.n_col {
                        sb.connect(wv(cl.col_v + j), rk.child_pub_w[o + 3 + j]);
                    }
                    for j in 0..cl.row_pt.1 {
                        sb.connect(wv(cl.row_pt.0 + j), rk.child_pub_w[o + 3 + loc.n_col + j]);
                    }
                    sb.connect(wv(cl.val_v), jag[gi].1);
                    ci = 1;
                }
                for &ki in &key_kids[gi] {
                    let rk = &regions[ki];
                    for (li, &jw) in rk.jag_w.iter().enumerate() {
                        let cl = &loc.claims[ci];
                        sb.connect(wv(cl.val_v), jw);
                        for j in 0..loc.n_col {
                            sb.connect(wv(cl.col_v + j), rk.jag_sig_w[j]);
                        }
                        if cl.terms.is_empty() {
                            let tag = cw_j(
                                &mut sb,
                                &mut vals,
                                &mut jag_const_rec,
                                F128::new(0, cl.row_pt.1 as u64),
                            );
                            sb.connect(wv(cl.row_scale_v - 1), tag);
                            // A FRESH claim is live: its scale is 1.
                            sb.connect(wv(cl.row_scale_v), ow);
                            for j in 0..cl.row_pt.1 {
                                sb.connect(wv(cl.row_pt.0 + j), rk.jag_row_w[li][j]);
                            }
                        } else {
                            let tag = cw_j(
                                &mut sb,
                                &mut vals,
                                &mut jag_const_rec,
                                F128::new(1, cl.terms.len() as u64),
                            );
                            sb.connect(wv(cl.terms[0].0 - 1), tag);
                            for (tj, &(cv, addr)) in cl.terms.iter().enumerate() {
                                sb.connect(wv(cv), rk.jag_row_w[li][tj]);
                                let aw = cw_j(
                                    &mut sb,
                                    &mut vals,
                                    &mut jag_const_rec,
                                    F128::new(addr as u64, 0),
                                );
                                sb.connect(wv(cv + 1), aw);
                            }
                        }
                        ci += 1;
                    }
                }
                assert_eq!(ci, loc.claims.len(), "every jagged claim connected");
                let header_v = loc.hdr_v;
                let hw = cw_j(
                    &mut sb,
                    &mut vals,
                    &mut jag_const_rec,
                    F128::new(loc.k_row as u64, loc.claims.len() as u64),
                );
                sb.connect(wv(header_v), hw);
            }
            // THE FOLD KEY IS THE CIRCUIT VERIFIED: each group's absorbed
            // digest payload connects to the child region's own statement
            // digest, so a slot cannot fold claims about a circuit this
            // node did not verify.
            let pays_n = payload_words(&stream);
            for j in 0..n_keys {
                for p in [sigma_payloads[j], jagged_payloads[j]] {
                    assert_eq!(pays_n[p].len(), 2, "a group key payload is 32 bytes");
                    for (b, &kw) in pays_n[p].iter().enumerate() {
                        sb.connect(ww[kw].expect("key payload wired"), regions[j].cd_w[b]);
                    }
                }
            }
        }
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
        // THE PRIOR's uniform surfaces (the spine): claim 0 of every group.
        // The registry-keyed matrix entries ride in exactly as the lane's
        // priors do — LOWS to the child's live word, points and value
        // straight through — and the sigma slots ride the same wiring with
        // the MATCH-GATE's outputs in place of live and value.
        let cj = if let Some((uni_off, _, _, sig, _)) = &spine_w {
            let rk = &regions[spine.as_ref().unwrap().node_child];
            for (i, loc) in locs.iter().enumerate().take(n_uni) {
                let cl = &loc.claims[0];
                let o = uni_off[i];
                assert_eq!(
                    cl.row_low_n, 1,
                    "an inherited claim's lows are its live word"
                );
                sb.connect(wv(cl.row_low_v), rk.child_pub_w[o]);
                sb.connect(wv(cl.col_low_v), rk.child_pub_w[o]);
                for j in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + j), rk.child_pub_w[o + 1 + j]);
                }
                for j in 0..cl.row_pt_n {
                    sb.connect(wv(cl.row_pt_v + j), rk.child_pub_w[o + 1 + loc.k_col + j]);
                }
                sb.connect(
                    wv(cl.value_v),
                    rk.child_pub_w[o + 1 + loc.k_col + loc.k_row],
                );
            }
            for j in 0..n_keys {
                let loc = &locs[n_uni + j];
                let cl = &loc.claims[0];
                let o = spine_w.as_ref().unwrap().1[j];
                sb.connect(wv(cl.row_low_v), sig[j].0);
                sb.connect(wv(cl.col_low_v), sig[j].0);
                for i2 in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + i2), rk.child_pub_w[o + 3 + i2]);
                }
                for i2 in 0..cl.row_pt_n {
                    sb.connect(wv(cl.row_pt_v + i2), rk.child_pub_w[o + 3 + loc.k_col + i2]);
                }
                sb.connect(wv(cl.value_v), sig[j].1);
            }
            1
        } else {
            0
        };
        for (k, (tk, rk)) in rts.iter().zip(&regions).enumerate() {
            // The lagrange row lows, IN-CIRCUIT from the child's z_skip wire
            // (native pre-assert first: the fold's absorbed lows ARE the closed
            // form at the located z_skip).
            assert_eq!(
                &fold_claims[0][cj + k].row.low[..],
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
                sb.connect(lw2, wv(locs[0].claims[cj + k].row_low_v + j));
            }
            // Native pre-asserts (the method-note discipline).
            for t in 0..n_bool {
                let inner_t = fold_claims[2 * t][cj + k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * t][cj + k].row.point[..],
                    &tk.mat_assert.x_inner_rest[..inner_t],
                    "boolean type {t} row point is x_inner_rest's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][cj + k].col.point[..],
                    &tk.mat_assert.rr[..inner_t],
                    "boolean type {t} col point is rr's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][cj + k].col.low[..],
                    &tk.mat_assert.z_partial[..],
                    "boolean type {t} col low is z_partial"
                );
                assert_eq!(fold_claims[2 * t][cj + k].value, tk.mat_assert.evals[t].0);
                assert_eq!(
                    fold_claims[2 * t + 1][cj + k].value,
                    tk.mat_assert.evals[t].1
                );
            }
            for t in 0..n_el {
                let kappa = fold_claims[2 * n_bool + 2 * t][cj + k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][cj + k].row.point[..],
                    &tk.el_assert.r_con[..kappa],
                    "element type {t} row point is r_con's head"
                );
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][cj + k].col.point[..],
                    &tk.el_assert.r_col[..kappa],
                    "element type {t} col point is r_col's head"
                );
                assert_eq!(
                    fold_claims[2 * n_bool + 2 * t][cj + k].value,
                    tk.el_assert.evals[t].0
                );
                assert_eq!(
                    fold_claims[2 * n_bool + 2 * t + 1][cj + k].value,
                    tk.el_assert.evals[t].1
                );
            }
            let sfi0 = if spine.is_some() { n_uni + k } else { n_uni };
            let n_structure = tk.sigma_native.claims().len();
            let sk0 = if spine.is_some() {
                cj
            } else {
                cj + n_structure * k
            };
            for (j, claim) in tk.sigma_native.claims().iter().enumerate() {
                assert_eq!(&fold_claims[sfi0][sk0 + j], claim);
            }

            // boolean A/B per type: batch-major mlv mapping for the row
            // points, lc rounds REVERSED for the col points, z_partial
            // word-for-word, values to the mat_eval advice wires.
            for t in 0..n_bool {
                for fi in [2 * t, 2 * t + 1] {
                    let cl = &locs[fi].claims[cj + k];
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
                sb.connect(wv(locs[2 * t].claims[cj + k].value_v), rk.mat_eval_w[t].0);
                sb.connect(
                    wv(locs[2 * t + 1].claims[cj + k].value_v),
                    rk.mat_eval_w[t].1,
                );
                // ONE lagrange-low surface per child (lagrange(z_skip) is
                // type-independent): every boolean fold's lows connect to
                // fold 0's, and fold 0's publish below.
                if t > 0 {
                    for fi in [2 * t, 2 * t + 1] {
                        for j in 0..locs[0].claims[cj + k].row_low_n {
                            sb.connect(
                                wv(locs[fi].claims[cj + k].row_low_v + j),
                                wv(locs[0].claims[cj + k].row_low_v + j),
                            );
                        }
                    }
                } else {
                    for j in 0..locs[0].claims[cj + k].row_low_n {
                        sb.connect(
                            wv(locs[1].claims[cj + k].row_low_v + j),
                            wv(locs[0].claims[cj + k].row_low_v + j),
                        );
                    }
                }
            }
            // element A/B per type: r_con = zc.r[ν..] (round order), r_col
            // = the lc rounds REVERSED, values to the per-slot eval advice.
            for t in 0..n_el {
                for fi in [2 * n_bool + 2 * t, 2 * n_bool + 2 * t + 1] {
                    let cl = &locs[fi].claims[cj + k];
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
                    wv(locs[2 * n_bool + 2 * t].claims[cj + k].value_v),
                    rk.el_eval_w[t].0,
                );
                sb.connect(
                    wv(locs[2 * n_bool + 2 * t + 1].claims[cj + k].value_v),
                    rk.el_eval_w[t].1,
                );
            }
            // Circuit structure: every static helper evaluation rides the
            // child's key slot. A fresh-only node folds every child under
            // slot 0.
            let sfi = if spine.is_some() { n_uni + k } else { n_uni };
            let sk = if spine.is_some() {
                cj
            } else {
                cj + rk.structure_claim_w.len() * k
            };
            for (j, (row_w, col_w, value_w)) in rk.structure_claim_w.iter().enumerate() {
                let cl = &locs[sfi].claims[sk + j];
                sb.connect(wv(cl.row_low_v), ow);
                sb.connect(wv(cl.col_low_v), ow);
                assert_eq!(cl.row_pt_n, row_w.len());
                assert_eq!(cl.col_pt_n, col_w.len());
                for (j, &w) in row_w.iter().enumerate() {
                    sb.connect(wv(cl.row_pt_v + j), w);
                }
                for (j, &w) in col_w.iter().enumerate() {
                    sb.connect(wv(cl.col_pt_v + j), w);
                }
                sb.connect(wv(cl.value_v), *value_w);
            }
        }

        // Publishes: per fold, the accumulator claim [live | rho_col |
        // rho_row | value] (endpoint identities are copy constraints,
        // nothing published). This is the
        // ENVELOPE-registry surface a parent inherits, so under the
        // envelope it rides the reserved ACC_MAIN block at a constant
        // index; off-envelope it publishes inline, as before.
        // THE SPINE LAYOUT: the registry-keyed matrix entries, then the
        // sigma SLOTS, then the jagged SLOTS — `N_KEY_SLOTS` of each,
        // whatever this node's fold actually had, each leading with the
        // KEY it is about. A fresh-only node has one live slot per family
        // and publishes the other DEAD (all zeros), which decodes as the
        // zero claim, so a base node and a steady node are read at the
        // same offsets by one parent circuit.
        let key_pay = payload_words(&stream);
        let key_wires = |p: usize| -> [Wire; 2] {
            [
                ww[key_pay[p][0]].expect("key payload wired"),
                ww[key_pay[p][1]].expect("key payload wired"),
            ]
        };
        let mut acc_main_w: Vec<Wire> = Vec::new();
        let push_entry = |w: &mut Vec<Wire>,
                          key: Option<[Wire; 2]>,
                          fp: Option<&FoldPub>,
                          k_col: usize,
                          k_row: usize| {
            if let Some(k) = key {
                w.extend_from_slice(&k);
            }
            match fp {
                Some(fp) => {
                    w.push(fp.live);
                    w.extend_from_slice(&fp.rho_col);
                    w.extend_from_slice(&fp.rho_row);
                    w.push(fp.value);
                }
                None => w.extend(std::iter::repeat_n(zw, 2 + k_col + k_row)),
            }
        };
        for fp in fold_pubs.iter().take(n_uni) {
            push_entry(&mut acc_main_w, None, Some(fp), 0, 0);
        }
        for j in 0..N_KEY_SLOTS {
            let live = (j < n_keys).then(|| (key_wires(sigma_payloads[j]), &fold_pubs[n_uni + j]));
            push_entry(
                &mut acc_main_w,
                Some(live.map(|(k, _)| k).unwrap_or([zw, zw])),
                live.map(|(_, fp)| fp),
                locs[n_uni].k_col,
                locs[n_uni].k_row,
            );
        }
        for j in 0..N_KEY_SLOTS {
            let live = (j < n_keys).then(|| (key_wires(jagged_payloads[j]), &jfold_pubs[j]));
            push_entry(
                &mut acc_main_w,
                Some(live.map(|(k, _)| k).unwrap_or([zw, zw])),
                live.map(|(_, fp)| fp),
                jlocs[0].n_col,
                jlocs[0].k_row,
            );
        }
        // THE PASSENGER: `out = child's passenger + h · (the orphaned
        // entry)`, word for word. `h` is live for exactly one node of a
        // spine (the first steady one over a base), and there the child's
        // own passenger is empty — so the sum is a SELECT that can never
        // silently drop a live claim: two live terms garble each other and
        // the root's discharge rejects, which is the safe direction.
        let pass_w: Vec<Wire> = match &spine_w {
            Some((_, sig_off, jag_off, sig, jag)) => {
                let e = &env;
                let rk = &regions[spine.as_ref().unwrap().node_child];
                let pb = env_pass_base(e);
                let mut out = Vec::with_capacity(sig_ent + jag_ent);
                for (base, o, h, width) in [
                    (pb, sig_off[1], sig[1].2, sig_ent),
                    (pb + sig_ent, jag_off[1], jag[1].2, jag_ent),
                ] {
                    for w in 0..width {
                        out.push(
                            sb.gate(
                                cs.fold_macs,
                                &[rk.child_pub_w[base + w], h, rk.child_pub_w[o + w]],
                            )[0],
                        );
                    }
                }
                out
            }
            _ => Vec::new(),
        };
        let fold_pub_base = env_acc_main_base(&env);
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
        // The publish of the combined span rides the envelope's fixed tail
        // block (below, with the padding), never inline.
        let app_inline: Option<usize> = None;
        // ---- the LANE fold region, in-circuit: priors-only, every prior
        // surface WIRED to the child's published accumulator claim (a
        // prior's surface IS what the child published — the child_pub_w
        // words at claims_base, layout [rho_col | rho_row | value] per
        // group), lows to the constant 1. Its own chain block rides the
        // shared b3 slot; the fold rows the shared mac/mrs/prefix slots.
        let lane_pub = lane_native.as_ref().map(|ln2| {
            let (_, llocs, ljlocs, lstream, lbytes, lops, lchals, lvals) = ln2;
            let lane_ref = lane.as_ref().expect("lane native implies lane");
            // Use the protocol tracer rather than a manual finalize loop:
            // Secure fold tapes contain fused `Pow`+squeeze operations whose
            // compression counter differs from an ordinary squeeze.
            let ltrace = crate::r1cs_hashes::fs_chain::trace_duplex(lstream, lbytes, lops);
            assert_chain_replays(lops, &ltrace, lchals);
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
                &[],
            );
            emit_recorded_pow_checks(
                &mut sb,
                cs.q.b3,
                cs.q.pow,
                iv2,
                lops,
                &ltrace,
                lstream,
                &lchain_outs,
                &lww,
                &mut vals,
                &mut consts,
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
                cs.fold_macs,
                cs.mrs,
                pfslot,
                pf_w,
                leslot,
                llocs,
                &ltrace,
                &challenge_word_locs(lops),
                &lchain_outs,
                &lww,
                &lvmap,
                lchals,
                lvals,
                &mut vals,
                zw,
                ow,
                false, // the jagged group follows on the lane tape
            );
            let ljfold_pubs = emit_jagged_fold_region(
                &mut sb,
                cs.fold_macs,
                cs.mrs,
                pfslot,
                pf_w,
                ljlocs,
                &ltrace,
                &challenge_word_locs(lops),
                &lchain_outs,
                &lww,
                &lvmap,
                lvals,
                &mut vals,
                zw,
                ow,
            );
            for (k, rk) in regions.iter().enumerate() {
                let mut off = lane_ref.claims_base;
                for loc in llocs {
                    let cl = &loc.claims[k];
                    // [live | rho_col | rho_row | value]: the LOWS connect
                    // to the child's LIVE word (the zero-claim scale) —
                    // a real entry carries 1, an absent one decodes zero.
                    sb.connect(lwv(cl.row_low_v), rk.child_pub_w[off]);
                    sb.connect(lwv(cl.col_low_v), rk.child_pub_w[off]);
                    for j in 0..cl.col_pt_n {
                        sb.connect(lwv(cl.col_pt_v + j), rk.child_pub_w[off + 1 + j]);
                    }
                    for j in 0..cl.row_pt_n {
                        sb.connect(
                            lwv(cl.row_pt_v + j),
                            rk.child_pub_w[off + 1 + loc.k_col + j],
                        );
                    }
                    sb.connect(
                        lwv(cl.value_v),
                        rk.child_pub_w[off + 1 + loc.k_col + loc.k_row],
                    );
                    off += loc.k_col + loc.k_row + 2;
                }
                // The inherited JAGGED prior: child k's published entry —
                // the block right after the uniform groups in its
                // ACC_CHAIN layout — connects to the lane's absorbed claim
                // surfaces wire-to-wire, exactly like the groups above.
                for loc in ljlocs {
                    let cl = &loc.claims[k];
                    assert!(cl.terms.is_empty(), "inherited jagged claims are plain eq");
                    // The Eq-SCALE: an inherited jagged claim's scale IS the
                    // child's live word, exactly as the uniform groups' lows
                    // are — the zero-claim gate, in the wiring.
                    sb.connect(lwv(cl.row_scale_v), rk.child_pub_w[off]);
                    for j in 0..loc.n_col {
                        sb.connect(lwv(cl.col_v + j), rk.child_pub_w[off + 1 + j]);
                    }
                    for j in 0..cl.row_pt.1 {
                        sb.connect(
                            lwv(cl.row_pt.0 + j),
                            rk.child_pub_w[off + 1 + loc.n_col + j],
                        );
                    }
                    sb.connect(
                        lwv(cl.val_v),
                        rk.child_pub_w[off + 1 + loc.n_col + loc.k_row],
                    );
                    off += loc.n_col + loc.k_row + 2;
                }
            }
            // The lane's structural words (claim tags + the shape header)
            // pin to shared constant publics, like the main fold's — the
            // identities themselves are wire-bound above.
            let mut lane_const_rec: Vec<(F128, usize)> = Vec::new();
            {
                let mut jc: Vec<(F128, Wire)> = Vec::new();
                let mut cw_j = |sb: &mut ShapeBuilder,
                                vals: &mut Vec<F128>,
                                rec2: &mut Vec<(F128, usize)>,
                                v: F128|
                 -> Wire {
                    if let Some(&(_, w)) = jc.iter().find(|&&(x, _)| x == v) {
                        return w;
                    }
                    vals.push(v);
                    rec2.push((v, sb.public_len()));
                    let w = sb.public_input();
                    jc.push((v, w));
                    w
                };
                for loc in ljlocs {
                    for cl in &loc.claims {
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut lane_const_rec,
                            F128::new(0, cl.row_pt.1 as u64),
                        );
                        sb.connect(lwv(cl.row_scale_v - 1), tag);
                    }
                    let header_v = loc.hdr_v;
                    let hw = cw_j(
                        &mut sb,
                        &mut vals,
                        &mut lane_const_rec,
                        F128::new(loc.k_row as u64, loc.claims.len() as u64),
                    );
                    sb.connect(lwv(header_v), hw);
                }
            }
            // The lane's claims are the LOWER-registry surface a parent
            // inherits: under the envelope they ride the reserved
            // ACC_CHAIN block — the same constant index at which an FL
            // child exposes its own chain fold.
            let mut lane_w: Vec<Wire> = Vec::new();
            for fp in lfold_pubs.iter().chain(&ljfold_pubs) {
                lane_w.push(fp.live);
                lane_w.extend_from_slice(&fp.rho_col);
                lane_w.extend_from_slice(&fp.rho_row);
                lane_w.push(fp.value);
            }
            let lane_words = lane_w.len();
            let lane_pub_base = env_acc_chain_base(&env);
            (
                lane_pub_base,
                lane_words,
                lalpha_recs,
                lane_w,
                lane_const_rec,
            )
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
            println!(
                "  {:42} {:6}",
                "lagrange lows + tail",
                mac_total - mac_after_fold
            );
            println!("  {:42} {:6}", "TOTAL", mac_total);
        }
        if std::env::var("B3_CENSUS").is_ok() {
            eprintln!(
                "  [node emitted BLAKE rows] primary {} | secondary {} | model max {}",
                sb.rows_in_slot(cs.q.b3),
                cs.q.b3_alt.map(|slot| sb.rows_in_slot(slot)).unwrap_or(0),
                b3_rows,
            );
        }
        if std::env::var("PUB_CENSUS").is_ok() {
            println!("\nPUBLICS CENSUS (child 0; child 1 same shape):");
            for w in r0.census.windows(2) {
                println!("  {:38} {:6}", w[1].0, w[1].1 - w[0].1);
            }
            let child = r0.census.last().unwrap().1 - r0.census[0].1;
            println!("  {:38} {:6}", "= CHILD TOTAL", child);
            let tail_len: usize = locs.iter().map(|l| 2 + l.k_col + l.k_row).sum();
            println!("  {:38} {:6}", "lagrange consts", 66usize);
            println!("  {:38} {:6}", "fold region publics", tail_len);
            println!(
                "  {:38} {:6}",
                "TOTAL (2 children + shared)",
                sb.public_len()
            );
        }
        let build_ms = t_tapes.elapsed().as_secs_f64() * 1e3 - tape_setup_ms;
        let t_build2 = std::time::Instant::now();
        // publics*: the node pads to the same public-segment length the
        // leaf does (free counts: the count VECTORS deliberately differ —
        // see the assert_ne below the builders).
        let prepad_publics2 = sb.public_len();
        let app_base = {
            let _ = app_inline;
            let empty: Vec<Wire> = Vec::new();
            pad_envelope_counts(
                &mut sb,
                &cs.q,
                &cs.env_cache(),
                &env,
                zw,
                &mut hints,
                &mut vals,
                &mut consts,
                &EnvTail {
                    acc_main: &acc_main_w,
                    acc_chain: lane_pub.as_ref().map(|(_, _, _, w, _)| w).unwrap_or(&empty),
                    pass: &pass_w,
                    app: app_w.as_deref().unwrap_or(&empty),
                },
            );
            app_w.as_ref().map(|_| env_app_base(&env))
        };
        let shape2 = sb.finish().expect("the 2->1 node circuit builds");
        // The two-limb Ligerito verifier plus the split BLAKE table stays
        // below 512 cell slots, which pins the optimized mu=23 boundary.
        assert!(
            shape2.circuit.cells().slots().len() <= 512,
            "the F256 node's cell-slot budget regressed ({} slots)",
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
                (
                    shape2.registry_slot(cs.q.family.expect("family-H slot")),
                    "family-h".to_string(),
                ),
                (shape2.registry_slot(cs.macs), "mac".to_string()),
                (shape2.registry_slot(cs.fold_macs), "fold-mac".to_string()),
                (shape2.registry_slot(cs.zcr), "zcr".to_string()),
                (shape2.registry_slot(cs.mrs), "mrs".to_string()),
                (shape2.registry_slot(cs.spine), "spine".to_string()),
                (shape2.registry_slot(cs.spine256), "spine256".to_string()),
                (shape2.registry_slot(cs.alslot), "assist".to_string()),
            ];
            if let Some(slot) = cs.q.b3_alt {
                lab.push((shape2.registry_slot(slot), "b3b".to_string()));
            }
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
        let hint_refs: Vec<&(dyn std::any::Any + Sync)> = hints
            .iter()
            .map(|h| h as &(dyn std::any::Any + Sync))
            .collect();
        let build_ms = build_ms + t_build2.elapsed().as_secs_f64() * 1e3;
        // THE INDEX-FILL RUNNER (setup): compile the fill plan, then pin it
        // row-identical against the generic walk before the online loop
        // trusts it — publics, every boolean row store, and every
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
            if let Some(slot) = cs.q.b3_alt {
                assert_eq!(
                    walk.rows::<Blake3Gate>(slot),
                    fill.rows::<Blake3Gate>(slot),
                    "fill plan: second b3 rows"
                );
            }
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
            assert_eq!(
                walk.rows::<PowMaskGate>(cs.q.pow),
                fill.rows::<PowMaskGate>(cs.q.pow),
                "fill plan: pow rows"
            );
            let family_slot = cs.q.family.expect("family-H slot");
            assert_eq!(
                walk.rows::<FamilyTransposeTileGate>(family_slot),
                fill.rows::<FamilyTransposeTileGate>(family_slot),
                "fill plan: family-H rows"
            );
        }
        // The node proves and verifies over the circuit path. Union, PCS
        // params and the R1CS tables are per-SHAPE — offline, ahead of the
        // online loop.
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = cfg.outer_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: outer_lanes(&union2, pcs_batch_for(&union2, pf)),
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
        let pow_r1cs2 = PowMaskTable.build_block_r1cs(nu2);
        let pow_lc2 = pow_r1cs2.csc_lincheck_circuit();
        let family_slot = cs.q.family.expect("family-H slot");
        let family_r1cs2 = FamilyTransposeTileTable::build_block_r1cs(nu2);
        let family_lc2 = family_r1cs2.csc_lincheck_circuit();
        let build_ms = build_ms + t_r1cs.elapsed().as_secs_f64() * 1e3;
        // TOWER_STEADY=N (or the bench's STEADY_OVERRIDE) re-runs the ONLINE
        // phases (tapes + trace + asm + prove + verify) N extra times over
        // the SAME built shape: the offline setup (circuit, R1CS, PCS
        // params, warmed pools) is paid once, so iterations after the first
        // give the steady-state online cost. Every iteration's record lands
        // in `onlines` (NodeOut carries them; `online` stays the last).
        let mut steady_left = steady_reps();
        let mut onlines: Vec<Online> = Vec::with_capacity(steady_left + 1);
        let (built2, oproof, ocommit, block_pub, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms) = loop {
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
                    los.par_iter()
                        .for_each(|lo| record_child_verify(lo, DOMAIN));
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
            // The fold checker + the accumulator, reassembled from publics —
            // and THE BLOCK, which is the accumulator plus the dead slots and
            // the passenger: what a spine parent reads.
            let (rebuilt, sig_keys, mut p_at) =
                check_fold_publics(&built2.public, fold_pub_base, &locs, &alpha_recs, n_uni);
            let mut sigma_slots: Vec<([F128; 2], MatrixClaim)> = sig_keys
                .iter()
                .zip(&rebuilt[n_uni..])
                .map(|(k, c)| (*k, c.clone()))
                .collect();
            for _ in n_keys..N_KEY_SLOTS {
                sigma_slots.push(read_acc_entry(
                    &built2.public,
                    &mut p_at,
                    true,
                    locs[n_uni].k_col,
                    locs[n_uni].k_row,
                ));
            }
            let (jrebuilt, jag_keys, mut p_at) =
                check_jagged_fold_publics(&built2.public, p_at, &jlocs, true);
            let mut jagged_slots: Vec<([F128; 2], MatrixClaim)> = jag_keys
                .iter()
                .zip(&jrebuilt)
                .map(|(k, c)| (*k, c.clone()))
                .collect();
            for _ in n_keys..N_KEY_SLOTS {
                jagged_slots.push(read_acc_entry(
                    &built2.public,
                    &mut p_at,
                    true,
                    jlocs[0].n_col,
                    jlocs[0].k_row,
                ));
            }
            for j in 0..n_keys {
                assert_eq!(
                    jrebuilt[j], jouts[j],
                    "published jagged slot {j} == located native"
                );
                assert_eq!(
                    sigma_slots[j].0,
                    digest_f128(&key_digests[j]),
                    "the published sigma key names child {j}'s circuit"
                );
                assert_eq!(
                    jagged_slots[j].0,
                    digest_f128(&key_digests[j]),
                    "the published jagged key names child {j}'s layout"
                );
            }
            let tail_len = p_at - fold_pub_base;
            let passenger: Vec<([F128; 2], MatrixClaim)> = {
                let mut q = env_pass_base(&env);
                vec![
                    read_acc_entry(
                        &built2.public,
                        &mut q,
                        true,
                        locs[n_uni].k_col,
                        locs[n_uni].k_row,
                    ),
                    read_acc_entry(&built2.public, &mut q, true, jlocs[0].n_col, jlocs[0].k_row),
                ]
            };
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
                sigma: (0..n_keys)
                    .map(|j| (key_digests[j], rebuilt[n_uni + j].clone()))
                    .collect(),
                jagged: (0..n_keys)
                    .map(|j| (key_digests[j], jrebuilt[j].clone()))
                    .collect(),
            };
            assert_eq!(
                acc_pub, acc_v,
                "the Accumulator, reassembled from the public segment alone"
            );
            assert!(
                acc_pub.discharge(&mats)
                    && acc_pub.discharge_element(&el_mats)
                    && acc_pub.discharge_sigma(&key_circuits)
                    && acc_pub.discharge_jagged(&jag_tables),
                "the public-segment accumulator discharges all four groups"
            );
            // THE PASSENGER, natively: the child's own, unless this node is
            // the one whose node slot could not fold — then the orphan itself.
            // The two are never both live in a spine, and the in-circuit form
            // is their SUM, so this select and that sum agree.
            if !passenger.is_empty() {
                let dead = |k_col: usize, k_row: usize| {
                    (
                        [F128::ZERO; 2],
                        MatrixClaim {
                            row: Weight::low_eq(vec![F128::ZERO], vec![F128::ZERO; k_row]),
                            col: Weight::low_eq(vec![F128::ZERO], vec![F128::ZERO; k_col]),
                            value: F128::ZERO,
                        },
                    )
                };
                let want: Vec<([F128; 2], MatrixClaim)> = match &spine {
                    None => vec![
                        dead(locs[n_uni].k_col, locs[n_uni].k_row),
                        dead(jlocs[0].n_col, jlocs[0].k_row),
                    ],
                    Some(sp) => {
                        let slot1 = digest_f128(&key_digests[1]);
                        [&sp.prior.sigma[1], &sp.prior.jagged[1]]
                            .iter()
                            .enumerate()
                            .map(|(t, ent)| {
                                let carried = sp.prior.passenger[t].clone();
                                if ent.0 != slot1 && entry_live(&ent.1) {
                                    assert!(
                                        !entry_live(&carried.1),
                                        "a spine orphans ONCE: the passenger was already full"
                                    );
                                    (*ent).clone()
                                } else {
                                    carried
                                }
                            })
                            .collect()
                    }
                };
                assert_eq!(
                    passenger, want,
                    "the published passenger is the child's, plus this node's orphan"
                );
            }
            let block_pub = MainBlock {
                per_type: acc_pub.per_type.clone(),
                per_element: acc_pub.per_element.clone(),
                sigma: sigma_slots,
                jagged: jagged_slots,
                passenger,
            };
            // The lagrange-low constants: the one public surface the in-circuit
            // derivation adds — validated against the verifier's own values.
            {
                for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                    assert_eq!(built2.public[lam_base + i], v, "λ const {i}");
                }
                for &(v, idx) in &jag_const_rec {
                    assert_eq!(built2.public[idx], v, "jagged shared constant public");
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
                // UNDER the envelope the publish blocks live on the reserved
                // tail, so the body simply has to fit — which
                // `pad_envelope_counts` asserts — and the tail layout is
                // checked where it matters, by rebuilding both accumulators
                // at their CONSTANT bases below.
                let _ = (tail_len, prepad_publics2);
                // The LANE accumulator, reassembled from the public segment
                // alone — the parent-facing statement of the lower registry.
                if let (Some((lpb, _, lar, _, lrec)), Some((lacc_n, llocs, ljlocs, ..))) =
                    (lane_pub.as_ref(), lane_native.as_ref())
                {
                    let (lrebuilt, _, _) =
                        check_fold_publics(&built2.public, *lpb, llocs, lar, llocs.len());
                    let lu_len: usize = llocs.iter().map(|l| 2 + l.k_col + l.k_row).sum();
                    let (ljrebuilt, _, _) =
                        check_jagged_fold_publics(&built2.public, *lpb + lu_len, ljlocs, false);
                    let lane_ref = lane.as_ref().expect("lane");
                    let lacc_pub2 = aggregate::Accumulator {
                        registry_digest: lane_ref.registry.digest(),
                        per_type: vec![(lrebuilt[0].clone(), lrebuilt[1].clone())],
                        per_element: Vec::new(),
                        sigma: vec![(lane_ref.circuit.digest(), lrebuilt[2].clone())],
                        jagged: vec![(lane_ref.circuit.digest(), ljrebuilt[0].clone())],
                    };
                    assert_eq!(
                        &lacc_pub2, lacc_n,
                        "the LANE accumulator, reassembled from publics alone"
                    );
                    for &(v, idx) in lrec {
                        assert_eq!(built2.public[idx], v, "lane jagged constant public");
                    }
                }
            }

            let t_asm = std::time::Instant::now();
            // Recreated per online iteration — the spread closure consumes it.
            let spread_ty2 = BitSpreadTable::new(spread_w2);
            let pow_ty2 = PowMaskTable;
            // The copy-free assembly path: the boolean drivers pack straight
            // into the union slot blocks inside the prove (live rows only under
            // elide) — no intermediate capacity-sized buffers, no memcpy. The
            // rows are hoisted to owned Vecs because the closures must be Send
            // and `built2.rows` hands out `dyn Any`-backed borrows.
            let b3_declared: Vec<_> = std::iter::once(cs.q.b3).chain(cs.q.b3_alt).collect();
            let b3_rows2: Vec<_> = b3_declared
                .iter()
                .map(|&slot| (slot, built2.rows::<Blake3Gate>(slot).to_vec()))
                .collect();
            let swap_rows2 = built2.rows::<SwapGate>(cs.q.swap).to_vec();
            let spread_rows2 = built2.rows::<BitSpreadGate>(cs.q.spread).to_vec();
            let pow_rows2 = built2.rows::<PowMaskGate>(cs.q.pow).to_vec();
            let family_rows2 = built2.rows::<FamilyTransposeTileGate>(family_slot).to_vec();
            let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
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
                (
                    shape2.registry_slot(cs.q.pow),
                    UnionSlotProverInput::in_place(
                        move |dst| pow_ty2.generate_witness_batch_major_into(&pow_rows2, dst),
                        pow_lc2,
                    ),
                ),
                (
                    shape2.registry_slot(family_slot),
                    UnionSlotProverInput::in_place(
                        move |dst| {
                            FamilyTransposeTileTable::generate_witness_batch_major_into(
                                &family_rows2,
                                dst,
                            )
                        },
                        family_lc2,
                    ),
                ),
            ];
            bslots.extend(b3_rows2.into_iter().map(|(slot, rows)| {
                (
                    shape2.registry_slot(slot),
                    UnionSlotProverInput::in_place(
                        move |dst| {
                            blake3::generate_witness_batch_major_partial_into(&rows, nu2, dst)
                        },
                        b3_lc2,
                    ),
                )
            }));
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
            // THE MATCH-GATE FORGERY (the adversarial leg): re-witness the
            // spine gadget's 26 mac rows as the world a cheating prover wants —
            // the advice inverse set to 0 so both is-eq gadgets CLAIM the
            // mismatched digests are equal (z = 1), the gate then folding the
            // orphan LIVE (m = 1, g = live, gv = value) and waving the
            // passenger off (h = 0). Every forged row still satisfies the mac
            // relation (t = x·y, out = acc + t), so the element PIOP holds;
            // what cannot be reconciled are the COPY CONSTRAINTS — chk = z·d =
            // d ≠ 0 sits in the assert-zero anchor's class, and g/gv/h sit in
            // the classes of the fold tape's honest absorbed words (the native
            // fold folded the ZERO claim) and the passenger sum. The wiring
            // product is what must kill it — the same tier the chain-link
            // tamper pinned.
            if forge_match {
                let mac_ri = shape2.registry_slot(cs.macs);
                let rows = &mut el_ord
                    .iter_mut()
                    .find(|(i, _)| *i == mac_ri)
                    .expect("the mac slot's rows")
                    .1;
                let (zero, one) = (F128::ZERO, F128::ONE);
                for blk in 0..2 {
                    // sigma's node-slot gate, then jagged's — 13 rows each:
                    // [d p z chk] x2 digest words, then m, g, gv, nm, h.
                    let s = mac_spine0 + 13 * blk;
                    for w in 0..2 {
                        let b = s + 4 * w;
                        let d = rows[b][4];
                        assert_ne!(d, zero, "the forged slot genuinely mismatches");
                        rows[b + 1] = vec![zero, d, zero, zero, zero];
                        rows[b + 2] = vec![one, zero, one, zero, one];
                        rows[b + 3] = vec![zero, one, d, d, d];
                    }
                    let live = rows[s + 9][1];
                    let val = rows[s + 10][2];
                    assert_eq!(live, one, "the orphaned entry is live");
                    rows[s + 8] = vec![zero, one, one, one, one];
                    rows[s + 9] = vec![zero, live, one, live, live];
                    rows[s + 10] = vec![zero, live, val, live * val, live * val];
                    rows[s + 11] = vec![one, one, one, one, zero];
                    rows[s + 12] = vec![zero, live, zero, zero, zero];
                }
            }
            let el_inputs: Vec<UnionElementSlotInput> = el_ord
                .into_iter()
                .map(|(_, rows)| live_element_input_from_rows(rows, nu2))
                .collect();
            let mut lco: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
                (shape2.registry_slot(cs.q.swap), swap_lc2),
                (shape2.registry_slot(cs.q.spread), spread_lc2),
                (shape2.registry_slot(cs.q.pow), pow_lc2),
                (shape2.registry_slot(family_slot), family_lc2),
            ];
            lco.extend(b3_declared.iter().map(|&slot| {
                (
                    shape2.registry_slot(slot),
                    b3_lc2 as &dyn flock_core::lincheck::LincheckCircuit,
                )
            }));
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
            let vres = verifier::verify_ligerito_union_circuit(
                &union2,
                &shape2.circuit,
                &built2.public,
                &lcs2,
                &ocommit,
                &oproof,
                &pcs2,
                &mut ch2,
            );
            if forge_match {
                // The forged world's rows all satisfy their relations — only
                // the wiring product can object, and it MUST.
                assert!(
                    matches!(
                        vres,
                        Err(flock_core::verifier::VerifyError::Wiring(
                            flock_core::circuit::WiringError::Gkr(
                                flock_core::product_gkr::VerifyError::ProductMismatch
                            )
                        ))
                    ),
                    "a forged live fold of a mismatched entry must die on the wiring product"
                );
            } else {
                vres.expect("the 2->1 node verifies");
            }
            let verify_ms = t0v.elapsed().as_secs_f64() * 1e3;
            let deferred_ms = if forge_match {
                0.0
            } else {
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
                t0d.elapsed().as_secs_f64() * 1e3
            };
            let b3_live: Vec<usize> = std::iter::once(cs.q.b3)
                .chain(cs.q.b3_alt)
                .map(|slot| shape2.counts[shape2.registry_slot(slot)])
                .collect();
            let b3_live_total: usize = b3_live.iter().sum();
            println!(
                "\nTHE 2->1 RECURSION NODE (two children + {} folds, ONE proof)\n  \
             children: dense_m {} / mu {}, one circuit, distinct FS points\n  \
             regions: 2x the complete deferred verifier (swap assembly, shared slots)\n         \
             + the fold region; CONNECTED: all points, z_partial lows, sigma fully,\n         \
             and the matrix/element EVAL VALUES to the children's bound advice —\n         \
             lagrange lows DERIVED in-circuit from each child's z_skip wire\n  \
             outer: BLAKE rows {} across {:?} | nu {} | dense_m {} | mu {} \
             (cell slots: {} gate + {} public)\n  \
             PER PROOF (online): child tapes {:.0} + witgen/trace {:.0} + witness asm {:.0} + prove {:.0} \
             = {:.0} ms | verify {:.0} ms (DEFERRED {:.0} ms) | proof {:.1} KiB\n  \
             SETUP: circuit build (per SHAPE, cacheable) {:.0} ms | tape pins+locates (shape-stable) {:.0} ms\n",
                n_folds,
                lo0.pcs.m,
                rts[0].mu_i,
                b3_live_total,
                b3_live,
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
            onlines.push(Online {
                setup_ms: build_ms,
                walk_ms: trace_ms,
                tapes_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms,
                wall_ms: 0.0,
            });
            if steady_left > 0 {
                steady_left -= 1;
                continue;
            }
            break (
                built2, oproof, ocommit, block_pub, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms,
            );
        };
        let (swap_slot2, spread_slot2, pow_slot2, family_slot2) = (
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
            shape2.registry_slot(cs.q.pow),
            shape2.registry_slot(family_slot),
        );
        let b3_slots2 = std::iter::once(cs.q.b3)
            .chain(cs.q.b3_alt)
            .map(|slot| shape2.registry_slot(slot))
            .collect();
        NodeOut {
            lo: LeafOuter {
                public: built2.public.clone(),
                shape: shape2,
                proof: oproof,
                commitment: ocommit,
                pcs: pcs2,
                b3_r1cs: b3_r1cs2,
                swap_r1cs: swap_r1cs2,
                spread_r1cs: spread_r1cs2,
                pow_r1cs: pow_r1cs2,
                family_r1cs: family_r1cs2,
                b3_slots: b3_slots2,
                swap_slot: swap_slot2,
                spread_slot: spread_slot2,
                pow_slot: pow_slot2,
                family_slot: family_slot2,
            },
            acc: acc_v,
            online: Online {
                setup_ms: build_ms,
                walk_ms: trace_ms,
                tapes_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms,
                wall_ms: 0.0,
            },
            onlines,
            app_base,
            lane_acc: lane_native.map(|(a, ..)| a),
            block: block_pub,
        }
    }
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
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0006);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    assert_eq!(
        fl0.lo.shape.circuit.digest(),
        fl1.lo.shape.circuit.digest(),
        "one first-level circuit digest — the FL shape is data-independent"
    );
    assert_eq!(fl0.stmt_base, fl1.stmt_base, "one statement offset");
    assert_eq!(fl1.h_start, fl0.h_end, "the FL spans are adjacent");

    let out = build_node_outer_app(cfg, &[&fl0.lo, &fl1.lo], Some(fl0.stmt_base), None, None);
    let (node, acc, app) = (out.lo, out.acc, out.app_base);
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
    let (sig_digest, _) = acc.sigma.first().expect("the node accumulated sigma");
    assert_eq!(
        *sig_digest,
        fl0.lo.shape.circuit.digest(),
        "the internal accumulator keys by the FL circuit"
    );
    // **TASK 7b's PIN, amended by the COUNT WIN: an FL node and an
    // internal node are ONE ENVELOPE.** Same registry digest (wall 2),
    // same public-segment length with the app block at the same fixed
    // offset (publics*), same PINNED lane count (lanes*) — so a parent's
    // walk cannot tell an FL child from an internal child. Under FREE
    // COUNTS the declared count vectors deliberately DIFFER: the heights
    // are data now, reaching a parent only as jagged claims, and the
    // parent's circuit never reads them.
    {
        assert_eq!(
            fl0.lo.shape.registry.digest(),
            node.shape.registry.digest(),
            "FL and internal share ONE envelope registry"
        );
        assert_ne!(
            fl0.lo.shape.counts, node.shape.counts,
            "free counts: the FL and internal declare their OWN counts"
        );
        assert_eq!(
            fl0.lo.pcs.num_lanes, node.pcs.num_lanes,
            "ONE lane count (lanes* — pinned, the layout's structural residue)"
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
        bincode::serialize(&node.proof)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
    );
}

/// One node's own JAGGED LAYOUT — the table its published claims are
/// about, keyed by its circuit digest. Heights are a shape constant of
/// that circuit, which is why the key names the table.
#[cfg(test)]
fn node_jagged_params(lo: &LeafOuter) -> flock_core::pcs::jagged::JaggedParams {
    let u = outer_union(&lo.shape.registry, lo.shape.counts.clone());
    flock_core::pcs::jagged::JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        lo.commitment.params.m - flock_core::pcs::LOG_PACKING,
    )
}

/// **WALL 3: THE SPINE CONVERGES.** Eight chain segments → four FLs → a
/// BASE node (two FLs, fresh-only) → node_2 (a fresh FL + the base) →
/// node_3 (a fresh FL + node_2) — and `D(node_2) == D(node_3)`: ONE steady
/// shape from level 3 on, at any depth. That is the completeness wall
/// coming down. What makes it work:
///
/// * every node's MAIN fold inherits its node child's published
///   accumulator as a PRIOR, so nothing is dropped at depth > 2 (the gap
///   this arc opened against);
/// * the keyed groups have one SLOT PER CHILD ROLE — the FL slot and the
///   node slot — so a base node (one live key) and a steady node (two)
///   publish the SAME layout, dead slots being zeros that decode as the
///   zero claim;
/// * the node slot's inherited entry is MATCH-GATED against the key it was
///   published with. It matches at every steady level but one: node_3's
///   node slot inherits node_2's, which is keyed by the BASE circuit. That
///   single orphan is gated to zero in the fold and rides the PASSENGER to
///   the root, where it discharges against the base's own tables.
///
/// The chain LANE rides all three levels unchanged, and the app statement
/// spans the whole chain: the spine grows by prepending a fresh FL, so the
/// fresh child is always the earlier segment.
///
/// Ends with the MATCH-GATE ADVERSARIAL MATRIX: (a) a forged node_3 whose
/// gadget rows claim the mismatched digests match and fold the orphan live
/// — self-satisfying rows, so it must die on exactly the wiring product;
/// (b) a dropped passenger and (c) a forged entry key, both statement
/// tampers the proofs refuse.
#[test]
#[ignore] // Heavy — eight chain proofs and eight outers.
fn chain_spine_converges() {
    let cfg = test_config();
    use flock_core::aggregate;

    let env = envelope_shape();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_5B1E);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..8 {
        let cp = build_chain_proof(cfg, h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fls: Vec<FlNode> = (0..4)
        .map(|i| build_fl_node(cfg, &cps[2 * i], &cps[2 * i + 1]))
        .collect();
    let app_fl = fls[0].stmt_base;
    assert_eq!(
        app_fl,
        env_app_base(&env),
        "the FL's app block is the envelope's"
    );
    for f in &fls {
        assert_eq!(f.stmt_base, app_fl, "one FL app offset");
        assert_eq!(
            f.lo.shape.circuit.digest(),
            fls[0].lo.shape.circuit.digest(),
            "one FL circuit digest"
        );
    }
    // The lane's chain-side materials, shared by every level.
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_circuit = &cps[0].inner.built.shape.circuit;
    let chain_jp = chain_jagged_params(&cps[0]);
    let acc_base = fls[0].fold_pub_base;
    assert_eq!(
        acc_base,
        env_acc_chain_base(&env),
        "the FL's ACC_CHAIN block"
    );

    // THE BASE: fresh-only over the LAST two FLs. The spine grows by
    // prepending, so the base covers the tail of the chain.
    let base = build_node_outer_app(
        cfg,
        &[&fls[2].lo, &fls[3].lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[2].acc, &fls[3].acc],
            claims_base: acc_base,
        }),
        None,
    );
    let app_n = base.app_base.expect("the base's app block");
    assert_eq!(app_n, app_fl, "one app offset, FL and node alike");
    let base_lane = base.lane_acc.clone().expect("the base's lane");
    assert_eq!(
        base.block.sigma.len(),
        N_KEY_SLOTS,
        "the base publishes both slots"
    );
    assert!(
        !entry_live(&base.block.sigma[1].1) && !entry_live(&base.block.jagged[1].1),
        "a fresh-only node's NODE slot is dead"
    );
    assert!(
        base.block.passenger.iter().all(|(_, c)| !entry_live(c)),
        "the base carries no passenger"
    );

    // node_2: a fresh FL + the base. Its node slot's inherited entry is
    // the base's DEAD one, and its own node-slot output is keyed by the
    // BASE circuit — the entry that will orphan one level up.
    let n2 = build_node_outer_app(
        cfg,
        &[&fls[1].lo, &base.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[1].acc, &base_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &base.block,
            forge: false,
        }),
    );
    let n2_lane = n2.lane_acc.clone().expect("node_2's lane");
    assert!(
        n2.block.passenger.iter().all(|(_, c)| !entry_live(c)),
        "node_2 orphans nothing — the base's node slot was already dead"
    );
    assert_eq!(
        n2.block.sigma[1].0,
        digest_f128(&base.lo.shape.circuit.digest()),
        "node_2's node slot is keyed by the BASE circuit"
    );
    assert_ne!(
        base.lo.shape.circuit.digest(),
        n2.lo.shape.circuit.digest(),
        "THE transitional mismatch: the base and the steady node are \
         different shapes, so node_3's node slot cannot fold what node_2 \
         published there"
    );

    // node_3: a fresh FL + node_2 — the STEADY node, and the one that
    // orphans. Its node slot names node_2's circuit, so the entry it
    // inherits (keyed by the base's) cannot fold and rides the passenger.
    let n3 = build_node_outer_app(
        cfg,
        &[&fls[0].lo, &n2.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[0].acc, &n2_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &n2.block,
            forge: false,
        }),
    );

    // ---- THE CONVERGENCE ----
    if n3.lo.shape.circuit.digest() != n2.lo.shape.circuit.digest() {
        println!("  SPINE DIGEST MISMATCH — per-slot rows (node_2 vs node_3):");
        for (t, (a, b)) in n2
            .lo
            .shape
            .counts
            .iter()
            .zip(&n3.lo.shape.counts)
            .enumerate()
        {
            if a != b {
                println!("    type {t}: n2 {a} vs n3 {b}");
            }
        }
        println!(
            "    publics {} vs {} | lanes {:?} vs {:?} | dense_m {} vs {}",
            n2.lo.public.len(),
            n3.lo.public.len(),
            n2.lo.pcs.num_lanes,
            n3.lo.pcs.num_lanes,
            n2.lo.pcs.m,
            n3.lo.pcs.m,
        );
        let (w2, w3) = (n2.lo.shape.circuit.wires(), n3.lo.shape.circuit.wires());
        println!(
            "    wire classes: {} vs {} ({} differ)",
            w2.len(),
            w3.len(),
            w2.iter().zip(w3).filter(|(a, b)| a != b).count()
        );
    }
    assert_eq!(
        n3.lo.shape.circuit.digest(),
        n2.lo.shape.circuit.digest(),
        "ONE steady spine shape: node_2 == node_3, at any depth"
    );

    // ---- THE ROOT ----
    // (1) the steady accumulator: two keyed slots, the FL's and node_2's.
    assert_eq!(n3.acc.sigma.len(), N_KEY_SLOTS, "the root's sigma slots");
    assert_eq!(
        n3.acc.sigma[0].0,
        fls[0].lo.shape.circuit.digest(),
        "FL slot key"
    );
    assert_eq!(
        n3.acc.sigma[1].0,
        n2.lo.shape.circuit.digest(),
        "node slot key"
    );
    // (2) THE PASSENGER: node_2's node-slot entries, keyed by the BASE
    // circuit — the only orphan a spine ever makes — against the base's
    // own tables.
    let pass = &n3.block.passenger;
    let base_d = base.lo.shape.circuit.digest();
    assert_eq!(
        pass[0].0,
        digest_f128(&base_d),
        "the passenger names the base"
    );
    assert_eq!(
        pass[1].0,
        digest_f128(&base_d),
        "the passenger names the base"
    );
    assert!(
        entry_live(&pass[0].1) && entry_live(&pass[1].1),
        "the orphan boarded"
    );
    let base_jp = node_jagged_params(&base.lo);
    let pass_acc = aggregate::Accumulator {
        registry_digest: n3.acc.registry_digest,
        per_type: Vec::new(),
        per_element: Vec::new(),
        sigma: vec![(base_d, pass[0].1.clone())],
        jagged: vec![(base_d, pass[1].1.clone())],
    };
    assert!(
        pass_acc.discharge_sigma(&[&base.lo.shape.circuit]),
        "the passenger's sigma claim discharges against the BASE circuit's wiring"
    );
    assert!(
        pass_acc.discharge_jagged(&[(base_d, &base_jp)]),
        "the passenger's jagged claim discharges against the BASE circuit's layout"
    );
    // (3) the chain lane: eight leaves' claims in one accumulator.
    let lane3 = n3.lane_acc.clone().expect("the root's lane");
    assert!(
        lane3.discharge(&chain_mats) && lane3.discharge_sigma(&[chain_circuit]),
        "the root chain lane discharges against the chain tables"
    );
    // (4) the statement: the span is the whole chain.
    let h_end = native_chain(&h0, 8 * n_blocks);
    for j in 0..4 {
        assert_eq!(
            n3.lo.public[app_fl + j],
            pack4(h0[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_start"
        );
        assert_eq!(
            n3.lo.public[app_fl + 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_end == H^N(h_start)"
        );
    }
    // ---- THE MATCH-GATE ADVERSARIAL MATRIX (the owed soundness leg) ----
    // A statement-tier verify helper, the e2e tamper legs' assembly.
    let verify_with = |lo: &LeafOuter, publics: &[F128]| -> bool {
        let u = outer_union(&lo.shape.registry, lo.shape.counts.clone());
        let lcs = leaf_boolean_lcs(lo);
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &u,
            &lo.shape.circuit,
            publics,
            &lcs,
            &lo.commitment,
            &lo.proof,
            &lo.pcs,
            &mut ch,
        )
        .is_ok()
    };
    // (a) THE FORGED LIVE FOLD — the load-bearing leg. A cheating node_3
    // re-witnesses the match-gate to claim the D_base entry MATCHES and
    // folds the orphan live (no passenger). Every forged gate row is
    // self-satisfying, so only the copy constraints can object; the
    // builder asserts the proof dies on exactly Wiring/Gkr/ProductMismatch
    // (the assert lives inside build_node_outer_app, forge: true).
    build_node_outer_app(
        cfg,
        &[&fls[0].lo, &n2.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[0].acc, &n2_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &n2.block,
            forge: true,
        }),
    );
    // (b) THE PASSENGER DROP: zero the boarded orphan's live word — "no
    // orphan ever rode". The passenger is STATEMENT, so the honest proof
    // refuses the doctored segment.
    {
        let pass_base = env_pass_base(&env);
        assert_eq!(
            n3.lo.public[pass_base + 2],
            F128::ONE,
            "the orphan's sigma entry rides live"
        );
        let mut bad = n3.lo.public.clone();
        bad[pass_base + 2] = F128::ZERO;
        assert!(
            !verify_with(&n3.lo, &bad),
            "a dropped passenger must be rejected"
        );
    }
    // (c) THE FORGED CHILD KEY: node_2's published node-slot entry claims
    // to be keyed by the STEADY circuit instead of the base's — the lie
    // that would let node_3 fold it without a mismatch. The key words are
    // statement, so node_2's own proof refuses.
    {
        let uni_w =
            |c: &flock_core::matrix_fold::MatrixClaim| 2 + c.col.point.len() + c.row.point.len();
        let mut key_at = env_acc_main_base(&env);
        for (a, b) in n2.block.per_type.iter().chain(n2.block.per_element.iter()) {
            key_at += uni_w(a) + uni_w(b);
        }
        let s0 = &n2.block.sigma[0].1;
        key_at += 4 + s0.col.point.len() + s0.row.point.len(); // past the FL slot
        assert_eq!(
            n2.lo.public[key_at],
            digest_f128(&base_d)[0],
            "the offset arithmetic found the node slot's key"
        );
        let mut bad = n2.lo.public.clone();
        bad[key_at] = digest_f128(&n2.lo.shape.circuit.digest())[0];
        assert!(
            !verify_with(&n2.lo, &bad),
            "a forged entry key must be rejected"
        );
    }
    println!(
        "\nTHE SPINE CONVERGES (8 chains -> 4 FL -> base -> node_2 -> node_3)\n  \
         span H^{}(h_start) | D(node_2) == D(node_3) | 4 shapes total\n  \
         ONE steady accumulator (sigma+jagged x 2 slots) + a 2-entry passenger\n  \
         + the chain lane, all discharged at the root\n  \
         MATCH-GATE ADVERSARIAL MATRIX: forged live fold dies on the wiring\n  \
         product; a dropped passenger and a forged entry key die on the statement\n  \
         steady outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        8 * n_blocks,
        n3.lo.shape.circuit.cells().nu(),
        n3.lo.shape.circuit.cells().mu(),
        n3.lo.public.len(),
        bincode::serialize(&n3.lo.proof)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
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
    let cfg = test_config();
    use flock_core::aggregate;

    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0007);
    let h0: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    assert_eq!(
        fl0.fold_pub_base, fl1.fold_pub_base,
        "one fold-block layout"
    );

    // The lane's registry materials — the CHAIN side.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        params: &chain_jp,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let out = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(lane),
        None,
    );
    let (node, acc) = (out.lo, out.acc);
    let app = out.app_base.expect("the app block rode");
    let lane_acc = out.lane_acc.expect("the lane rode");

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
    assert!(
        lane_acc.discharge(&chain_mats),
        "chain-lane boolean discharges"
    );
    assert!(
        lane_acc.per_element.is_empty(),
        "the chain lane has no element group"
    );
    assert!(
        lane_acc.discharge_sigma(&[&cp0.inner.built.shape.circuit]),
        "chain-lane sigma discharges against the chain circuit"
    );
    // (3) The FL lane discharges: boolean vs the FL b3/swap/spread mats
    // (registry order), element vs the FL element types, sigma vs the FL
    // circuit digest's table.
    let fl_mats = leaf_boolean_mats(&fl0.lo);
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
        acc.discharge_sigma(&[&fl0.lo.shape.circuit]),
        "FL-lane sigma discharges"
    );

    // ---- the tamper matrix ----
    // (a) A tampered FL STATEMENT word (its h_end): the FL proof must not
    //     verify against it — the adjacency data is statement-bound.
    {
        let union_f = outer_union(&fl0.lo.shape.registry, fl0.lo.shape.counts.clone());
        let lcs_f = leaf_boolean_lcs(&fl0.lo);
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
        let jagged_pt: Vec<aggregate::JaggedKeyProve<'_>> = vec![(
            cp0.inner.built.shape.circuit.digest(),
            &chain_jp,
            Vec::new(),
        )];
        let jagged_vt: Vec<aggregate::JaggedKeyVerify<'_>> =
            vec![(cp0.inner.built.shape.circuit.digest(), Vec::new())];
        let mut chp = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        let (lagg, _) = aggregate::prove_aggregate_classes_with_grinding(
            chain_registry,
            &chain_mats,
            &chain_circs,
            &[],
            &[],
            &el_asserts_l,
            &[(&cp0.inner.built.shape.circuit, Vec::new())],
            &jagged_pt,
            &[&fl0.acc, &fl1.acc],
            tower_fold_grinding(cfg),
            &mut chp,
        )
        .expect("honest lane fold proves");
        let mut ch = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        assert!(
            aggregate::verify_aggregate_classes_with_grinding(
                chain_registry,
                &[],
                &el_asserts_l,
                &[(&cp0.inner.built.shape.circuit, Vec::new())],
                &jagged_vt,
                &[&bad_acc, &fl1.acc],
                &lagg,
                tower_fold_grinding(cfg),
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
        let lcs_n = leaf_boolean_lcs(&node);
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
        bincode::serialize(&node.proof)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
    );
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
    let cfg = test_config();
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
        let cp = build_chain_proof(cfg, start, n_blocks);
        _leaf_ms.push(t.elapsed().as_secs_f64() * 1e3);
        cp
    };
    let cp0 = mk(h0);
    let cp1 = mk(cp0.h_end);
    let cp2 = mk(cp1.h_end);
    let cp3 = mk(cp2.h_end);
    assert_eq!(cp3.h_end, h_all, "the four segments ARE the chain");

    let t_fl = std::time::Instant::now();
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    let _fl_ms = t_fl.elapsed().as_secs_f64() * 1e3 / 2.0;

    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = blake3::build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        params: &chain_jp,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let t_in = std::time::Instant::now();
    let out = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(lane),
        None,
    );
    let (node, acc, nt) = (out.lo, out.acc, out.online);
    let _internal_ms = t_in.elapsed().as_secs_f64() * 1e3;
    let app = out.app_base.expect("app block");
    let lane_acc = out.lane_acc.expect("lane");

    // The root.
    let t_root = std::time::Instant::now();
    for j in 0..4 {
        assert_eq!(
            node.public[app + 4 + j],
            pack4(h_all[4 * j..4 * j + 4].try_into().unwrap()),
            "root statement: h_end == H^(4·{n_blocks})(h_start)"
        );
    }
    assert!(
        lane_acc.discharge(&chain_mats)
            && lane_acc.discharge_sigma(&[&cp0.inner.built.shape.circuit])
    );
    let fl_mats = leaf_boolean_mats(&fl0.lo);
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
            && acc.discharge_sigma(&[&fl0.lo.shape.circuit])
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
        "\nCHAIN TOWER M32 HEADLINE (warm box; per-stage timing lives in tower_online_bench)\n  \
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
        bincode::serialize(&node.proof)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
    );
    proof_census("internal node", &node.proof, &node.pcs);
    proof_census("chain leaf (m32 Fast)", &cp0.inner.proof, &cp0.inner.pcs);
    proof_census("FL node", &fl0.lo.proof, &fl0.lo.pcs);
}

/// **THE ONLINE BENCH: leaf, first-level node, internal node.** One number
/// per stage, measuring only what a prover pays PER STATEMENT — the walk,
/// the child tape sources, witness assembly, and the prove. Per-SHAPE setup
/// (circuit emit+finish, R1CS tables, PCS params, the fill plan, the tape
/// pins) is timed but reported apart and never folded into a per-proof
/// number: a shape is statement-independent, so a production prover builds
/// it once per level and reuses it for every segment.
///
/// Each stage is measured by ONE builder call whose online phases repeat
/// `BENCH_RUNS` times over FIXED inputs (STEADY_OVERRIDE — the setup is
/// paid once), taking per-phase MEDIANS — the first iteration pays
/// first-touch allocator costs that are warmup, not marginal cost.
///
/// Knobs: `BENCH_RUNS` (default 3), `CHAIN_BLOCKS` (default 256 — set
/// 262144 for the m32 production leaf), `TOWER_PROFILE=slim` for the
/// envelope. BOX DISCIPLINE: run the stability probe first and reboot if it
/// is far out of band — this box's benchmarks self-corrupt under sustained
/// load, and nothing here can tell you that happened.
#[test]
#[ignore] // Benchmark — run explicitly with --nocapture.
fn tower_online_bench() {
    let cfg = test_config();
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
    //
    // ONE BUILDER CALL PER STAGE (STEADY_OVERRIDE): the per-shape setup —
    // circuit emission, tape pins, R1CS, PCS params — is paid once and the
    // online phases repeat `runs` times inside the builder. The old
    // per-iteration rebuild spent ~96% of the bench's wall clock re-doing
    // byte-identical setup. The node arms therefore run as BLOCKS seconds
    // apart instead of the old minutes-apart interleave; box drift over
    // seconds is far below what the interleave guarded against.
    use std::sync::atomic::Ordering;
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)

    // ---- LEAF: nothing else is alive; the measured proof BECOMES cp0 ----
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let leaf = cp0.onlines.clone();
    STEADY_OVERRIDE.store(0, Ordering::Relaxed);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);

    // ---- FL: two chain children and nothing more. The measured FL is the
    // spine's FRESH child — the EARLIEST segments, since a spine PREPENDS —
    // so the measured leaf and FL become the tower's own materials. ----
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)
    let fresh = build_fl_node(cfg, &cp0, &cp1);
    let fl = fresh.onlines.clone();
    STEADY_OVERRIDE.store(0, Ordering::Relaxed);

    // ---- the rest of the tower: four more segments, two more FLs. The
    // INTERNAL arm's node doubles as the spine's BASE child (identical
    // children shape — the old separate base build was pure waste), so the
    // whole tower is 6 chain proofs, 3 FLs and 2 node builds where it was
    // 10, 5 and 3. The segment pairs are scoped so they drop once folded —
    // what production carries.
    let (fl0, fl1) = {
        let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
        let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
        let cp4 = build_chain_proof(cfg, cp3.h_end, n_blocks);
        let cp5 = build_chain_proof(cfg, cp4.h_end, n_blocks);
        (
            build_fl_node(cfg, &cp2, &cp3),
            build_fl_node(cfg, &cp4, &cp5),
        )
    };
    drop(cp1);
    // The lane is what production carries: the children's chain
    // accumulators fold in a priors-only aggregate of their own.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = chain_blake_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    // ---- INTERNAL (= the spine's base) then SPINE, one call each ----
    // Both arms' online iterations run back to back inside their builder
    // call (seconds apart, not the old minutes-apart interleave), from
    // materials that are ALL resident before either starts.
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)
    let base = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fresh.stmt_base),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: &cp0.inner.built.shape.circuit,
            params: &chain_jp,
            priors: &[&fl0.acc, &fl1.acc],
            claims_base: fresh.fold_pub_base,
        }),
        None,
    );
    let internal = base.onlines.clone();
    // The steady spine node: the fresh FL (the segments BEFORE the base's)
    // plus the base as the node child whose accumulator it inherits —
    // built exactly as the convergence test builds node_2.
    let spine: Vec<Online> = {
        let lane = base.lane_acc.clone().expect("the base's lane");
        build_node_outer_app(
            cfg,
            &[&fresh.lo, &base.lo],
            Some(fresh.stmt_base),
            Some(ChainLane {
                registry: chain_registry,
                mats: &chain_mats,
                circs: &chain_circs,
                circuit: &cp0.inner.built.shape.circuit,
                params: &chain_jp,
                priors: &[&fresh.acc, &lane],
                claims_base: fresh.fold_pub_base,
            }),
            Some(SpineIn {
                node_child: 1,
                prior: &base.block,
                forge: false,
            }),
        )
        .onlines
    };
    STEADY_OVERRIDE.store(usize::MAX, Ordering::Relaxed);

    // SHAPE WARMUP IS SETUP, NOT MARGINAL COST. A stage's first online
    // iteration primes the zero/scratch pools and faults in the allocator
    // arena for its buffer size classes — one-time per-shape state that a
    // production prover reaches once and keeps. Left inside the samples it
    // skews whichever arm runs FIRST (both node arms share size classes,
    // so the later arm inherits the earlier one's warmth: the internal-vs-
    // spine delta once read −7% from ordering alone). So iteration 0 is
    // reported on the setup tier and every per-proof number below is a
    // median over the STEADY iterations only.
    let steady = |runs: &[Online]| -> Vec<Online> {
        if runs.len() > 1 {
            runs[1..].to_vec()
        } else {
            runs.to_vec()
        }
    };
    let warmup = |runs: &[Online]| -> Option<f64> { (runs.len() > 1).then(|| runs[0].total()) };
    let warms = [
        ("leaf", warmup(&leaf)),
        ("FL", warmup(&fl)),
        ("internal", warmup(&internal)),
        ("spine", warmup(&spine)),
    ];
    let (leaf, fl, internal, spine) = (
        steady(&leaf),
        steady(&fl),
        steady(&internal),
        steady(&spine),
    );
    let (leaf_on, fl_on, int_on) = (
        median_total(&leaf),
        median_total(&fl),
        median_total(&internal),
    );
    // ANY binary tree over L leaves carries L/2 first-level nodes and
    // L/2 − 1 nodes above them — the count is tree-shape-indifferent — so
    // a leaf's amortised share tends to leaf + FL/2 + node/2 whichever
    // shape the tower uses. The SPINE's node is the honest one to divide
    // by: it is what every level above 2 runs.
    let node_on = if spine.is_empty() {
        int_on
    } else {
        median_total(&spine)
    };
    let per_leaf = leaf_on + fl_on / 2.0 + node_on / 2.0;
    println!(
        "\nONLINE BENCH — {n_blocks} compressions/leaf, {runs} runs/stage, profile {:?}\n  \
         per-proof ONLINE (setup is per-SHAPE, shown for reference only):",
        cfg,
    );
    for (name, w) in warms {
        if let Some(ms) = w {
            println!("    {name:9} shape warmup (setup tier, dropped from medians): {ms:.1} ms");
        }
    }
    report_stage("leaf", &leaf);
    report_stage("FL", &fl);
    report_stage("internal", &internal);
    if !spine.is_empty() {
        report_stage("spine (steady)", &spine);
        println!(
            "  the spine's node costs {:+.1}% against the fresh-only \
             internal — the tree's node COUNT is unchanged, so this delta \
             IS wall 3's whole price",
            100.0 * (node_on - int_on) / int_on,
        );
    }
    println!(
        "  AMORTISED per leaf (leaf + FL/2 + node/2): {:.0} ms \
         -> {:.0}k compressions/sec\n  \
         the leaf's walk IS the chain compute — the application's own \
         sequential work, not proving\n",
        per_leaf,
        n_blocks as f64 / per_leaf,
    );
}
