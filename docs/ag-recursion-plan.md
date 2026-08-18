# AG-skip in the recursion tower — plan

Status: PLAN (2026-08-18, branch `ag-union`). Goal: the tower's provers run the
AG-skip boolean zerocheck (union-AG measured −21% prove at m32) and the
recursion circuit replays those proofs.

ENDGAME (Ron, 2026-08-18): the RS/φ8 univariate skip will EVENTUALLY be
deprecated and removed — AG becomes the only skip basis. Not immediate, but
it re-weights this plan: AG-everywhere (Phase C) is the critical path rather
than optional; the in-circuit AG lows (Phase D) are the *successor* of
`emit_lagrange_lows`, not an optional upgrade; migration knobs should be
flip-in-place + delete, not permanent parallel API; and Phase F lists the
removal blockers (x86 + CUDA AG kernels chief among them).

## Status + next-session entry point (updated 2026-08-18)

DONE: **Phase A complete** (commit d6a0eb1 — circuit-AG entries
`R1csProofCircuitMergedAg` / `prove_fast_ligerito_union_circuit_ag` /
`verify_ligerito_union_circuit_ag{,_deferred}`; fused PoW predicate aligned
to the PowMask convention; `cross_class_circuit_ag_roundtrip` green).
**Phase B slice 1 done** (5a35c4e — `chain_leaf_ag_roundtrip`: the real
chain shape proves under AG, RS 17.6 vs AG 12.3 ms at m22 dev size).

NEXT — **Phase B remainder**, all in `crates/flock-prover/src/tower.rs`
(line refs at 5a35c4e):

1. `MixedInner.proof` becomes a flavor enum (`Rs(R1csProofCircuitMerged) |
   Ag(R1csProofCircuitMergedAg)`), with accessors for the shared fields
   (lincheck / wiring / pcs_open) so the non-zerocheck pins stay one code
   path. `build_chain_proof` proves AG behind a private flavor switch
   (flip-in-place — no public TowerConfig change).
2. `ChildTape::new` AG arm:
   - native verify under the recorder → `verify_ligerito_union_circuit_ag`
     (11451); assertions still from `inner.work`.
   - region anchor `find(b"flock-ag-skip-v1")` instead of
     `flock-zerocheck-v0` (11494).
   - round-1 pins: `ObserveSlice(158)`/`(64)` against `bp.ag.round1_ab/_c`
     (11553-11583).
   - the locator walk (12284-12354): ONE `SqueezeSlice` (r_outer — RS has
     two: r_skip + r_outer); then in place of `[Pow] + SqueezeScalar`
     z_skip: `Label("flock-ag-skip-r1-point") + SqueezeScalar ×2 +
     Label("flock-ag-skip-r1-nonce") + ObserveBytes(4)`. The mlv loop code
     is unchanged. After `z_partial`, the lincheck fresh-skip is the same
     5-op AG shape (label + 2 squeezes + label + ObserveBytes(4)).
   - the `.phi8()` pin (12392) → native point pin: rebuild the 32-byte seed
     from the two located seed chals (`ag_skip::r1_seed` layout: s0.lo,
     s0.hi, s1.lo, s1.hi LE), run `evaluation_point_from_nonce_pow(seed,
     proof.r1_nonce, kind, R1_POW_BITS)`, compare to
     `bool_assert.z_skip` (`SkipPoint::Ag`).
   - GOTCHA: the nonce payload rides the STREAM WORDS (`ww`), not
     `vals_rec` (ObserveBytes contributes no vals entries);
     `bytes_payload_mask` marks it public by default — desired here.
3. `ChildRegion`: for AG children skip `emit_lagrange_lows` (FL connects at
   10201-10215) and instead publish (seed₂, nonce, point₅) via the
   `FoldPub`/`AlphaRec` publics pattern (14558/14244); extend the FL
   checker (sibling of `check_fold_publics`, 14770) to recompute the point
   from (seed, nonce) and `base_evaluation_functional(point)` against the
   fold row-lows. The fold tape itself is unchanged (64-wide lows both
   bases). Publics count (5684 at tower.rs:362) moves — it is shape, not a
   pinned digest.
4. c-point baked constants: the 7 ghash inner constants →
   `ag_skip::friendly_challenges()` (13504-13511) for AG children.
5. Tests: FL node over AG leaves (twin of the existing FL roundtrip),
   shape-diff, capacity asserts, then `tower_online_bench` with the AG
   leaf (`CHAIN_BLOCKS=262144`; expect ~−100 ms/leaf at m32).

Memory track: `recursion-track.md` (machine-local) mirrors this and adds
session gotchas. The two survey reports' full maps are summarized in the
section below; anything deeper re-derives quickly from the line refs.

## What the survey established (tower @ `ag-union` tip)

1. **The boolean zerocheck's arithmetic is NOT wired in-circuit today.** The
   tower binds a child's boolean zerocheck transcript-positionally (the FS
   chain rows force the observed proof words) and consumes only four surfaces:
   the `r_outer` squeeze words (→ c-claim point wires), the `z_skip` squeeze
   wire (→ `emit_lagrange_lows` → fold row-lows, tower.rs:11210 + call sites
   10201/16655), the finals `v_a, v_b` (→ lincheck-entry replay), and
   `z_partial`. The Λ-interpolation, round checks, and the ring-switch
   `claim_check` are never arithmetized. **Consequence: AG's 222-coordinate
   round-1 costs zero new in-circuit arithmetic under the current posture.**
2. **Proof shapes**: all three levels prove `R1csProofCircuitMerged` via
   `prove_fast_ligerito_union_circuit`; leaf = 1 boolean type, no element;
   FL/internal/spine = the envelope registry (6 boolean + 15 element types).
   Tapes record `verify_ligerito_union_circuit_deferred` under a
   `RecordingChallenger`. There is no AG circuit-flavor entry yet.
3. **The fold region is already flavor-generic**: `SkipPoint::weights(6)` is
   64-wide in both bases, so `MatrixClaim` widths, the fold tape shape, and
   `emit_weight` are unchanged. Only the row-lows *derivation* (today:
   `emit_lagrange_lows` from the one `zskip_w` wire, ~260 MAC rows) is
   φ8-specific.
4. **Capacity is not the constraint**: AG round-1 adds +94 observed words ≈
   +23.5 BLAKE3 compressions per child region (~0.25% of a region; b3 slots
   have ~7k rows headroom each). An in-circuit base-functional derivation
   would be ~1.0–1.5k MAC rows per child vs the mac slot's ~49k.
5. **The AG points have no transcript word.** RS gives one located squeeze
   wire; AG's `r₁` and the lincheck's fresh skip are decoded from
   (seed = 2 squeezes, nonce = ObserveBytes(4)) through a STANDALONE hash
   (outside the duplex sponge) — the existing `Op::Pow`/PowMask machinery
   never sees the fused PoW. Binding the point is the one real design
   decision.

## The design decision: how the recursion binds an AG point

**Tier 0 — checker-published (recommended start).** Per AG child, publish
(seed₂, nonce, point₅) in the outer proof's public segment; the fold row-lows
already ride the fold tape as observed words. The consumer's native checker
(the same tier as `check_fold_publics` / `AlphaRec`) re-derives
`evaluation_point_from_nonce_pow(seed, nonce) == point` (one hash + one
attempt, ~1 µs) and `base_evaluation_functional(point) == row-lows`, and the
`.phi8()`-style native pins become point pins. Zero new gates, zero new table
types (the registry-diet lesson: type COUNT costs ~20% node time), ~8 extra
publics/child (publics count 5684 moves — it is shape, not a pinned digest).
This is exactly the documented pre-in-circuit boundary posture the stale
comments at tower.rs:16571-16580 describe.

**Tier 1 — in-circuit upgrade (later, if the exit contract wants it).**
(a) on-curve predicate for the advice point (each Artin–Schreier equation
`z² + z = rhs` is one square + add; base equation a handful of mults);
(b) `emit_ag_lows`: base functional from point wires — x-powers ≤ 31,
4 y-powers, 8 z-monomials, per-push MACs, one denominator inverse via the
existing advice-inverse/zassert idiom (~1.0–1.5k MAC rows);
(c) standalone-hash binding of H(seed‖nonce) via the `emit_opening`-style
Blake3 rows + a PowMask row + XOF-stream binding of x/slot/choice bits.
Decode-canonicity (which AS root) either pinned via linear functionals
(Frobenius-chain idiom exists) or compensated with ~5 extra PoW bits.

## Phases

**Phase A — native entries + hygiene (small).**
- `R1csProofCircuitMergedAg` + `prove_fast_ligerito_union_circuit_ag`
  (aarch64) + `verify_ligerito_union_circuit_ag{,_deferred}` — thin over the
  already-flavored shared bodies (`prove_union_with_binding_zc`,
  `verify_union_piops`/`BooleanPiopRef`). Lift the boolean-only assert for the
  mixed/circuit AG arms (element region is flavor-independent; the AG flavor
  already forces honest-zero witness mode).
- **Align the fused PoW predicate with the PowMask convention** (MSB-first
  leading bits on the hash's serialized bytes, replacing low-LE-bits) NOW,
  while the AG transcript is unshipped — Tier 0 doesn't need it, Tier 1 does,
  and it's free today, frozen later.
- Circuit-AG roundtrip tests (leaf-shaped: blake3 circuit + wiring).

**Phase B — leaf-AG (the first payoff: workload −21% at m32).**
- `TowerConfig` grows the AG flavor for the leaf (decision: new variants
  `Chain128Ag`… vs a `leaf_zc` field).
- `build_chain_proof` proves circuit-AG; `ChildTape` gets the AG walk
  (anchor `flock-ag-skip-v1`; no r_skip slice; ObserveSlice(158)+(64);
  r₁ = Label + 2 Squeeze + Label + ObserveBytes(4); AG fresh-skip 5-op shape;
  round-1 pins typed to `AgProof`), tower.rs:12288-12300 / 11560-11582 /
  11494.
- `ChildRegion`: `zskip_w` → the Tier-0 surface (publish seed/nonce/point,
  skip `emit_lagrange_lows` for AG children, checker recomputes lows +
  point derivation); `.phi8()` pin at 12392 → point pin.
- c-point baked constants: the 7 ghash inner constants →
  `ag_skip::friendly_challenges()` (tower.rs:13504-13511).
- FL-over-AG-leaves roundtrip, shape-diff, capacity asserts, then
  tower_online_bench (expect leaf ~−100 ms at m32 ≈ +12-16% throughput).

**Phase C — outers-AG (envelope nodes).**
- Same items on `RealTape`/`RealRegion` (6211, 6859-6875, 6959, 8719,
  8222-8227, 10178-10212) + mixed-class-with-element AG entries.
- Envelope re-checks (nu* ≤ 14 per b3 slot, m* = 29 content, publics count),
  internal/spine digest equality, `chain_spine_converges` re-run.
- Expect node prove −10-15% (zerocheck share at m29).

**Phase D — Tier-1 upgrade** (in-circuit lows + on-curve + hash binding).
Under the deprecation endgame this is SCHEDULED, not optional: when RS is
removed, `emit_lagrange_lows` dies and `emit_ag_lows` is its mainline
replacement — going to Tier 0 permanently would be a posture REGRESSION
(the in-circuit lows were the recorded upgrade over the checker boundary).
Tier 0 remains the right FIRST landing; D closes the loop before removal.

**Phase E — audit + docs.** Extend the audit's recursive-agreement section:
AG rows are checker-tier obligations (like the fold publics), not PowMask
rows; friendly constants ≠ 1 is free in-circuit (baked constants); update
`docs/local/recursion-handoff.md` censuses and the memory track.

**Phase F — RS deprecation prerequisites** (the removal blockers, so they
can be scheduled early rather than discovered late):
1. **x86 AG round-1 kernel.** The AG prover's round-1 is aarch64-NEON SLP
   only; RS removal without an AVX-512 port kills x86 proving entirely
   (x86 VERIFY already works — `verify_ag`/`verify_with_grinding` are
   arch-independent).
2. **CUDA AG round-1 kernel.** The GPU prover runs the full RS zerocheck
   on-device (`cuda-ghash/zerocheck_round1/2/tail.cuh`, z_skip→lincheck
   hand-off resident); RS removal needs the AG twin (the mlv tail carries
   over — it is shape-identical — but round-1 over the genus-95 product
   code is a new kernel + vectors).
3. Recursion: delete the RS locator arms + `emit_lagrange_lows` + the φ8
   fused Pow+squeeze sites (keep the AG arms structurally parallel from
   Phase B so this is arm-deletion, not surgery).
4. Profiles/grinding: the RS rows of the audit schedule retire;
   `ZerocheckGrinding::skip_bits` / `LincheckGrinding::skip_bits` collapse
   to the AG accounting; the ungrinded direct route decides whether it
   gains the fused nonce or stays no-claim.
5. Transcript/fixture retirement: every RS byte pin (m6 merged fixtures,
   mixed-class pins, chain/Merkle/keccak3/sha2 relations — all currently
   RS), proof-IO version bump, and the parallel `*Ag` structs/entries
   renamed to primary as the RS structs are deleted.
6. `SkipPoint::Phi8` and `phi8()` die; `SkipPoint` may collapse back to a
   plain `EvaluationPoint` (claim types simplify).

## Open decisions (for Ron)

1. Tier 0 vs Tier 1 to start — recommend Tier 0 first, with Tier 1 (Phase D)
   scheduled before RS removal (see endgame note).
2. TowerConfig: under the endgame, do NOT grow the public config — keep
   Chain100/Chain128 and flip the flavor in place per phase (a private
   accessor / test-only knob during migration), so the eventual state has
   no residual flavor API to delete.
3. Proof struct: parallel `R1csProofCircuitMergedAg` during migration (no
   RS wire churn now); renamed to primary when the RS structs are deleted
   in Phase F. Structure all tower locator/region code as PARALLEL ARMS so
   RS removal is arm-deletion.
4. Scope order leaf-first — recommend yes; Phase C is on the deprecation
   critical path (no longer optional-until-measured).
5. PoW-convention alignment in Phase A — recommend yes (mandatory before
   the AG transcript becomes the only one).
6. Phase F sequencing: the x86 and CUDA AG round-1 kernels are the
   long-lead items — start them as soon as AG-everywhere (Phase C) is
   validated, independent of Phase D.
