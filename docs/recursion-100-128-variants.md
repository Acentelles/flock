# Two recursion variants: 100-bit and 128-bit

To: Min (rkm0959). From: Ron's session, 2026-08-12, branch `pow-mask-slot`
(sits on your `min/recursion-128bit` @ `4603c0f`, zero-conflict merge).

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
| leaf/node track (mvp) | `TOWER_PROFILE=fast100` | `TOWER_PROFILE=secure` |
| chain track / spine | `TOWER_PROFILE=slim100` (envelope on, m29) | blocked — see issues 2 and 3 |
| in code | `PcsParams.profile = Fast100 / Slim100` | `= Secure` |

`Fast100` and `Slim100` are new `LigeritoProfile` variants that are `Fast`
and `Slim` in every respect — the same Johnson accounting, two-point OOD,
Appendix C.3 grinding schedule, transcript shape, and the m28/m29
`initial_k` exceptions — except the consistency-query term targets the
profile's own `security_bits()` (100) instead of
`LIST_DECODING_QUERY_TARGET_BITS`. The Johnson query floor in
`LigeritoSecurityConfig::validate` is keyed off `analysis_version`
(`query128` vs `query100`), so each config family validates against its own
target; a boundary test pins the `Fast100` floor and the schedule equality.
No proof-format change: both variants live in one binary at v18.

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

Everything below on `pow-mask-slot` (which also carries the fused PoW mask
row — one 4-word row per grinding check; see commit `41a3a17`):

- `flock-core --lib` (incl. the config suite over all 70 embedded TOMLs) and
  the full `flock-prover` suite: green.
- 100-bit leaf/node: mvp11 node + mvp12 tower green under `fast100`.
- 128-bit leaf/node: mvp11 node + mvp12 tower green under `secure`.
- 100-bit chain track: `first_level_node_two_chains_fold_and_adjacency`,
  `chain_tower_e2e_with_lane`, and — notably — **`chain_spine_converges`
  green under `slim100` + the m29 envelope**. The envelope's fixed point
  held with zero re-pins, since it was iterated against exactly these
  schedules. This is the first spine run since your Ligerito parts landed.
- 128-bit chain track: blocked by issues 2 and 3 below.

Same-run mvp11 node comparison (medians of 10 online reps, M4 Max):

| | 100-bit (`fast100`) | 128-bit (`secure`) |
| --- | ---: | ---: |
| online prove | 117 ms | 146 ms (+25%) |
| outer proof | 292.0 KiB | 566.7 KiB (+94%) |
| BLAKE rows / capacity | 24,827 at nu 15 / mu 23 | 38,489 at nu 16 / mu 24 |

## Issues found (all pre-existing on `4603c0f`)

**1. mvp7 fails to parse its own tape (`LegacyPow`).**
`mvp7_real_query_phase` dies at "op 272: expected the next cap absorb, got
`LegacyPow { bits: 9 }`" — its inner still records legacy PoW ops the
post-fusion parser never learned. Looks like mvp7 was not converted with
the rest of the mvp ladder. Repro:
`cargo test --release -p flock-prover --test circuit_merkle mvp7_real_query_phase -- --ignored --exact`.

**2. The `initial_ood` walk breaks on the NEW slim schedule ("L0 OOD beta").**
`parse_open_levels`'s L0 OOD loop panics at the `SqueezeScalar` expectation
when walking a strict-128 slim tape — which blocks **every** envelope run
(`TOWER_PROFILE=slim` + `TOWER_ENV_M=29`), and with it the whole spine, at
the 128-bit schedule. Diagnostic that should localize it quickly: the SAME
parser walks old-schedule slim tapes fine (`slim100` spine passes end to
end), and fast/fast100 tapes fine — so it is something the new slim counts
change about the L0 region's op order, not slim's query-phase PoW or rate
per se. The assert now dumps its surrounding op context (we instrumented
it). Repro:
`TOWER_ENV_M=29 TOWER_PROFILE=slim cargo test --release -p flock-prover --test circuit_merkle chain_spine_converges -- --ignored --exact`.

**3. The Secure chain tower dies in witgen (first-ever run).**
`chain_tower_e2e_with_lane` under `TOWER_PROFILE=secure` panics with "a
connected wire disagrees with the gate output that produces it (slot 0
[= b3], class root …)" in `builder.rs`. This is your audit's "recursive
verification of the chain/Merkle wrapper proofs is out of scope" carve-out
made concrete: the chain-track tape walkers were never exercised against
grinding transcripts, and somewhere the FL/node emitters wire a chain row
under a pre-grinding shape assumption. Repro:
`TOWER_PROFILE=secure cargo test --release -p flock-prover --test circuit_merkle chain_tower_e2e_with_lane -- --ignored --exact`.

## What chain-128 needs

1. Issue 2 fixed — unblocks the envelope at the new slim schedule.
2. Issue 3 fixed — unblocks Secure (and presumably strict-128-slim) chain
   recursion.
3. An envelope re-iteration: Part 3 changed slim's counts, so the m29
   envelope's `counts_bool`/`counts_el`/`publics` fixed point moves. Our
   fused-PoW slot added a fourth boolean count (`counts_bool[3]`, currently
   an estimated 4096 cap) that should be pinned in the same pass.

## Branch map

- `41a3a17` — the fused PoW mask row (one 4-word row per grinding check;
  isolated grinding overhead 11 → 6 ms, +4 → +2 BLAKE rows). One subtlety
  relevant to your recursive PoW relation: the nonce-width rejection lives
  in the mask word's WIRE BINDING (word 2 must equal the statement's mask
  constant, whose high half is zero), not in the R1CS alone.
- `e9769ab` — merge of your Ligerito parts 1–3 (clean; the new Ligerito
  `Pow` sites ride the fused row automatically via the generic tape walk).
- `fc694b3` — `Fast100`.
- HEAD — `Slim100` + the spine validation + the instrumented parser assert.
