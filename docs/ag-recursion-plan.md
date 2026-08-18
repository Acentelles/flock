# AG-skip in the recursion tower — plan

Status: PLAN (2026-08-18, branch `ag-union`). Goal: the tower's provers run the
AG-skip boolean zerocheck (union-AG measured −21% prove at m32) and the
recursion circuit replays those proofs.

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

**Phase D — optional Tier-1 upgrade** (in-circuit lows + on-curve + hash
binding), driven by the recursion exit contract, not performance.

**Phase E — audit + docs.** Extend the audit's recursive-agreement section:
AG rows are checker-tier obligations (like the fold publics), not PowMask
rows; friendly constants ≠ 1 is free in-circuit (baked constants); update
`docs/local/recursion-handoff.md` censuses and the memory track.

## Open decisions (for Ron)

1. Tier 0 vs Tier 1 to start — recommend Tier 0.
2. TowerConfig shape for the flavor — variants vs field.
3. Proof struct: separate `R1csProofCircuitMergedAg` (recommended — matches
   the existing Ag pattern, keeps RS byte-compat) vs an enum field.
4. Scope order leaf-first — recommend yes (Phase B is where the workload
   payoff is; Phase C is optional-until-measured).
5. PoW-convention alignment in Phase A — recommend yes.
