# Two recursion variants: 100-bit and 128-bit

Original review context: Ron's 2026-08-12 `pow-mask-slot` session, based on
`min/recursion-128bit` at `4603c0f`. The resolution annotations and validation
status below describe the rebased `min/recursion-128bit` working tree as of
2026-08-13.

Ron wants both security levels runnable side by side while the 128-bit work
matures: **the 100-bit variant is the configuration shipped before your
branch**, and **the 128-bit variant is the configuration you are building**.
This note records how the variants are realized, the evidence they reproduce
the old cost point, their validation status on both recursion tracks, and
three pre-existing issues the chain-track runs surfaced — all three verified
against your unmodified tip, none introduced by our commits.

## The variants

| | 100-bit | 128-bit |
| --- | --- | --- |
| leaf/node track (mvp) | `TOWER_PROFILE=fast100` | default `TOWER_PROFILE=fast` |
| chain track / spine | `TOWER_PROFILE=slim100` (envelope on, m29) | `TOWER_PROFILE=slim` (envelope on, m29) |
| in code | `PcsParams.profile = Fast100 / Slim100` | `= Fast / Slim` |

The right-hand column is the strict component-security configuration. Its
Johnson query, two-point OOD, Flock-paper Appendix C.3 batching, and F256 MCA
terms each clear 128 bits. `Secure` remains the separate historical 120-bit
unique-decoding profile used as an additional regression target for every
algebraic-grinding family; its name does not make it the 128-bit profile.

`Fast100` and `Slim100` are new `LigeritoProfile` variants that are `Fast`
and `Slim` in every respect — the same Johnson accounting, two-point OOD,
Flock-paper Appendix C.3 grinding schedule, transcript shape, and the m28/m29
`initial_k` exceptions — except the consistency-query term targets the
profile's own `security_bits()` (100) instead of
`LIST_DECODING_QUERY_TARGET_BITS`. The Johnson query floor in
`LigeritoSecurityConfig::validate` is keyed off `analysis_version`
(`query128` vs `query100`), so each config family validates against its own
target; a boundary test pins the `Fast100` floor and the schedule equality.
Both variants live in one binary and use the same split-F256 transcript shape
at proof-format v19; their public query schedules remain different.

**They reproduce the pre-branch schedules exactly.** The canonical generator
at target 100 re-derives, byte-for-byte, the counts your Part 3 replaced:

- `m27_fast100`: per-level `[218, 106, 71, 53]`, Σq 448 (your audit's own
  "before" example);
- `m29_fast100`: Σq 491; `m29_slim100`: Σq 262 — the exact numbers in
  `circuit_merkle.rs`'s profile comment.

This doubles as a proof that the old Fast/Slim were 100-bit query targets.
Note the 100-bit variants are *slightly stronger* than the literal
pre-branch state: they inherit the two-point OOD binding and the C.3
algebraic grinding (a handful of extra rows, strictly more soundness).

## Validation status

The original measurements were made on `pow-mask-slot`. The rebased branch
carries the same fused PoW-mask design (one four-word row per grinding check)
at `f8093ec`, plus the soundness repair described in
`128-bit-grinding-audit.md`.

- `flock-core --lib` (incl. the config suite over all 70 embedded TOMLs) and
  the full `flock-prover` suite: green.
- 100-bit leaf/node: mvp11 node + mvp12 tower green under `fast100`.
- 128-bit leaf/node: mvp11 node + mvp12 tower green under strict `fast`.
- 100-bit chain track: `first_level_node_two_chains_fold_and_adjacency`,
  `chain_tower_e2e_with_lane`, and — notably — **`chain_spine_converges`
  green under `slim100` + the m29 envelope**. The envelope's fixed point
  held with zero re-pins, since it was iterated against exactly these
  schedules. This is the first spine run since your Ligerito parts landed.
- 128-bit chain track: strict-Slim m29 spine and chain tower are green. The
  historical Secure chain tower is also green as a compatibility regression;
  see the 2026-08-13 resolution below.

Historical same-run mvp11 comparison from Ron's review (medians of 10 online
reps, M4 Max; `secure` is the 120-bit compatibility profile, not the strict
128-bit column above):

| | 100-bit (`fast100`) | 128-bit (`secure`) |
| --- | ---: | ---: |
| online prove | 117 ms | 146 ms (+25%) |
| outer proof | 292.0 KiB | 566.7 KiB (+94%) |
| BLAKE rows / capacity | 24,827 at nu 15 / mu 23 | 38,489 at nu 16 / mu 24 |

## Issues found on `4603c0f` and current resolution

**1. MVP-7 failed to parse its own tape (`LegacyPow`).**
`mvp7_real_query_phase` dies at "op 272: expected the next cap absorb, got
`LegacyPow { bits: 9 }`" — its inner still records legacy PoW ops the
post-fusion parser never learned. Looks like mvp7 was not converted with
the rest of the mvp ladder. Repro:
`cargo test --release -p flock-prover --test circuit_merkle mvp7_real_query_phase -- --ignored --exact`.

Resolved upstream by converting MVP-7 to the fused transcript path. The exact
reproduction command is green after the rebase.

**2. The `initial_ood` walk broke on the new Slim schedule ("L0 OOD beta").**
`parse_open_levels`'s L0 OOD loop panics at the `SqueezeScalar` expectation
when walking a strict-128 slim tape — which blocks **every** envelope run
(`TOWER_PROFILE=slim` + `TOWER_ENV_M=29`), and with it the whole spine, at
the 128-bit schedule. Diagnostic that should localize it quickly: the SAME
parser walks old-schedule slim tapes fine (`slim100` spine passes end to
end), and fast/fast100 tapes fine — so it is something the new slim counts
change about the L0 region's op order, not slim's query-phase PoW or rate
per se. The diagnostic at that point dumped its surrounding op context.
Repro:
`TOWER_ENV_M=29 TOWER_PROFILE=slim cargo test --release -p flock-prover --test circuit_merkle chain_spine_converges -- --ignored --exact`.

Resolved by anchoring `parse_open_levels` at the
`flock-ligerito-basis-v0` protocol label, then checking the target and L0 cap
that follow it. The former "last cap with this byte length" heuristic could
select a later equal-sized recursive cap. The exact strict-Slim reproduction
now passes through the converged spine.

**3. The Secure chain tower failed in witness generation.**
`chain_tower_e2e_with_lane` under `TOWER_PROFILE=secure` panics with "a
connected wire disagrees with the gate output that produces it (slot 0
[= b3], class root …)" in `builder.rs`. This is your audit's "recursive
verification of the chain/Merkle wrapper proofs is out of scope" carve-out
made concrete: the chain-track tape walkers were never exercised against
grinding transcripts, and somewhere the FL/node emitters wire a chain row
under a pre-grinding shape assumption. Repro:
`TOWER_PROFILE=secure cargo test --release -p flock-prover --test circuit_merkle chain_tower_e2e_with_lane -- --ignored --exact`.

Resolved by replacing the chain lane's manual ordinary-finalize replay with
the canonical PoW-aware `fs_chain::trace_duplex`. Under Secure, the manual
loop ignored the fused PoW compression counter and therefore supplied the
recursive fold endpoint with a different challenge from the native verifier.
The exact reproduction now passes, including both lane discharges and tamper
tests.

## Chain-128 closure

1. The strict-Slim parser is fixed and its converged m29 spine passes.
2. The Secure chain lane uses the PoW-aware trace and passes end to end.
3. No count-cap re-iteration is required in the shipped design: free counts
   are unconditional. The strict-Slim spine validates the live PoW count,
   pinned lane count, fixed public layout, and steady circuit digest. The old
   `counts_bool`/`counts_el` values remain only the retired padding oracle and
   slot-declaration key list.

The 2026-08-13 validation matrix also passed strict Fast `mvp11`/`mvp12`,
Fast100 and Secure `mvp11`/`mvp12`, strict Slim and Slim100 chain towers,
Slim100's converged spine, `mvp7`, the full active `flock-core` suite (498
passed, 22 ignored), and the full active `flock-prover` suite and integrations
with no failures. Ron's three ignored
proof-byte pin tests were also run explicitly. The branch's deliberate v19
Ligerito protocol changes moved the fixture digests; the new values
were identical across two print runs, were documented at the fixtures, and
the pin tests now pass normally.

## Rebased branch map

- `f8093ec` — the fused PoW mask row (one four-word row per grinding check;
  isolated grinding overhead 11 → 6 ms, +4 → +2 BLAKE rows). One subtlety
  relevant to your recursive PoW relation: the nonce-width rejection lives
  in the mask word's WIRE BINDING (word 2 must equal the statement's mask
  constant, whose high half is zero), not in the R1CS alone.
- `2959d88` — merge of the Ligerito parts 1–3; the new Ligerito
  `Pow` sites ride the fused row automatically via the generic tape walk.
- `3f943eb` — `Fast100`.
- `cfcfe16` — `Slim100`, spine validation, and the parser diagnostic that
  exposed issue 2.
- `be75c25` — Ron's original review report. The current working-tree fixes are
  recorded above and in `128-bit-grinding-audit.md`.

---

Also, see https://claude.ai/code/artifact/70a216b1-ecd9-4839-b1ce-cbbca24a3618 for an audit of our branch in Fable 5.
