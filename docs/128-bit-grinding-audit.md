# 128-bit grinding audit

Status: implementation audit and independent re-review of the current
`min/recursion-128bit` working tree. The non-Ligerito audit was completed on
2026-08-11; Ligerito two-point OOD binding, Appendix C.3 algebraic grinding,
and the 128-bit list-decoding query schedule were reviewed on 2026-08-12.

This is the authoritative reviewer-facing summary of the grinding milestone.
It records what was implemented, why each bit count is sufficient, how the
prover/native verifier/recursive R1CS agree, and how to reproduce the test
evidence. It is intentionally self-contained so a reviewer does not need
branch-local planning or implementation-history documents.

The review target is **per-part 128-bit computational security** for the
algebraic challenge sites, including the completed Ligerito list-decoding
Parts 1 through 3. The Ligerito mutually correlated agreement (MCA) term
remains an explicit later milestone. As requested, the following are not
review blockers and are not used to qualify the conclusion:

- an eventual global soundness ledger or union bound;
- inactive legacy APIs; and
- recursive verification of the public chain/Merkle wrapper proofs.

## Executive conclusion

**Go/no-go result: go for the F256/MCA milestone.** No missing or under-grinded
algebraic challenge was found in the active production or recursion paths,
including Appendix C.3's claim batching, quadratic fold/sumcheck, and queried
consistency batching. Two-point OOD binding and the list-decoding consistency
query term are also complete. This is not a claim that Ligerito as a whole is
already a 128-bit component: its MCA/proximity-gap term remains.

All active non-Ligerito algebraic challenge families found in the current
production prover/verifier paths have Secure-profile grinding:

- Boolean and element zerocheck/lincheck;
- ring switching, opening batching, merged sumcheck, multipoint and anchor;
- the Product-GKR wiring/permutation argument;
- dense, element, sigma and jagged accumulator folds; and
- the public chain and Merkle-path wrapper arguments.

The prover and native verifier use identical schedules, reject malformed nonce
shapes before challenge replay, and bind each nonce immediately before the
protected randomness. The active recursive Flock-proof circuit checks every
recorded child-proof and accumulator-fold `Pow` with native BLAKE3 arithmetic,
64-bit nonce constraints, and the requested leading-zero predicate.

The current permutation/copy-constraint argument is the active batched
Product-GKR path. It is included in this conclusion. The older standalone
`permutation.rs` API is inactive and outside the requested review boundary.

## How to review this milestone

A reviewer can follow the work in this order:

1. Check the work-factor rule and exact native PoW relation below.
2. Check the degree and bit-count table in "Implemented schedules."
3. Follow the code map in "Production coverage matrix."
4. Compare the transcript order in "Prover / verifier agreement."
5. Inspect the recursive relation and generic tape walker.
6. Run the commands in "Test evidence."

No trust in the prover's claimed nonce validity or in a native pre-check is
part of the recursive soundness argument: the R1CS relation itself recomputes
and constrains every recorded nonzero PoW.

## Soundness rule

Let `q = 2^128`. For a nonzero degree-`D` polynomial evaluated at fresh field
randomness, Schwartz--Zippel gives

```text
Pr[false acceptance at this site] <= D / q.
```

Under the roadmap's work-normalized random-oracle model, a `lambda`-bit grind
changes the computational term to approximately

```text
D / 2^(128 + lambda).
```

The code centralizes the strict local rule as

```text
bits_for_degree(D) = floor(log2 D) + 1,   D >= 1,
bits_for_degree(0) = 0.
```

Thus a linear event uses one bit, a quadratic event two, degree 3 also two,
and degree 7 three. This is computational amplification, not an improvement
to the information-theoretic bound against an unbounded adversary.

Equivalently, finding a nonce that satisfies both the PoW predicate and a
degree-`D` bad-challenge condition takes expected work approximately

```text
2^(128 + lambda) / D.
```

The extra `+1` at powers of two is intentional because the target is a strict
inequality. For example,

```text
D = 1, lambda = 1:  1 / 2^129       = 2^-129
D = 2, lambda = 2:  2 / 2^130       = 2^-129
D = 3, lambda = 2:  3 / 2^130       < 2^-128
D = 7, lambda = 3:  7 / 2^131       < 2^-128
D = 256, lambda = 9: 256 / 2^137    = 2^-129.
```

Implementation: `grinding_bits_for_degree` in
[`challenger.rs`](../crates/flock-core/src/challenger.rs).

## Exact PoW relation

For chained transcript state `(cv, pending)`, the prover finds a 64-bit nonce
`w` such that the domain-separated fused compression

```text
O = Compress(cv, pending || w_le || 0^64 || padding,
             pow_squeeze_counter(lambda, message_len), 64, CHAIN_SQUEEZE)
prefix_lambda(O_1) = 0^lambda.
```

Here `O_i` are 128-bit output words. The protected scalar challenge is `O_0`,
the predicate is the disjoint word `O_1`, and the next chaining value is the
high half `O_2 || O_3`. At zero bits the canonical nonce is zero.

Under the random-function/XOF assumption for disjoint words of the custom
BLAKE3 compression, searching for `w` is the work amplifier without biasing
the protected field challenge. The nonce and the entire preceding transcript
are inputs to the same operation.

The recursive relation in `circuit_merkle.rs` wires this compression as the
ordinary challenge-squeeze row, constrains the nonce to 64 bits, and
constrains the selected predicate bits to zero. A scalar `Pow` therefore adds
no BLAKE row beyond the squeeze already needed for its challenge. It adds:

- bit-spread rows for nonce width and leading-zero constraints.

Native and recursive implementations agree on byte order: `w` is
little-endian, BLAKE3 output bytes use their native serialization, and
"leading bits" means most-significant-bit first within each serialized byte.
The differential test covers masks from 1 through 128 bits, including byte and
128-bit boundaries. See `grinding-hash-fusion-design.md` for the full
transition and code map.

## Implemented schedules

All bit counts below are sufficient for the strict local rule.

| Family / challenge | degree bound | Secure bits |
| --- | ---: | ---: |
| Boolean zerocheck initial equality point | `m` | `bits_for_degree(m)` |
| Boolean zerocheck skip point | `2^(K_SKIP+1)-1` | `K_SKIP+1` |
| Boolean zerocheck ordinary round | 2 | 2 |
| Boolean lincheck batching/pins | 1 | 1 |
| Boolean lincheck ordinary round | 2 | 2 |
| Boolean lincheck final skip | `2^k-1` | `k` |
| Element zerocheck initial equality point | `m_words` | `bits_for_degree(m_words)` |
| Element zerocheck/lincheck ordinary round | 2 | 2 |
| Element lincheck batching | 1 | 1 |
| Ring-switch point in `F128^7` | at most 7 | 3 |
| Whole mixed opening coefficient vector | total degree 1 | one `Pow(1)` |
| Dense merged-sumcheck round | 2 | 2 |
| Multipoint coefficient `gamma` | `K-1` | `bits_for_degree(K-1)` |
| Multipoint / Frobenius-anchor round | 2 | 2 |
| Product-GKR fingerprint `(alpha,beta)` | live entries `L-1` | `bits_for_degree(L-1)` |
| Product-GKR layer batching / close | 1 | 1 |
| Product-GKR product-sumcheck round | 2 | 2 |
| Fold coefficient vector `lambda` or `mu` | total degree 1 | one `Pow(1)` |
| Fold column or row sumcheck round | 2 | 2 |
| Chain packed-position vector, dimension `d` | at most `d` | `bits_for_degree(d)` |
| Chain initial `(tau,alpha)` | at most `max(n,1)` | `bits_for_degree(max(n,1))` |
| Chain shift round | 2 | 2 |
| Merkle packed-position vector, dimension `d` | at most `d` | `bits_for_degree(d)` |
| Merkle initial `(tau,alpha)` | at most `max(n,path_log+1)` | corresponding rule |
| Merkle shift round | effective verifier degree 3 | 2 |
| Ligerito scalar claim batching | list union `L_max` | `floor(log2 L_max)+1` |
| Ligerito quadratic fold/sumcheck round | list union `2 L_max` | `floor(log2(2 L_max))+1` |
| Ligerito queried-consistency batching | `L_max ceil(log2 Q)` | `floor(log2(L_max ceil(log2 Q)))+1` |

For Product-GKR the fingerprint degree is `L-1`, not `L`: the top homogeneous
term cancels because the live identity tags are permuted. For the Merkle shift,
the verifier accepts cubic interpolation even where the honest message may be
quadratic, so the malicious-proof degree is conservatively 3.

### Ligerito Part 3: list-decoding query schedule

At a Johnson-regime level with code rate `rho = 2^(-r)` and fixed slack
`eta = 0.02`, the proximity radius and consistency-query error are

```text
gamma = 1 - sqrt(rho) - eta,
epsilon_query <= (1 - gamma)^Q.
```

Writing

```text
b_per_query = log2(1 / (1 - gamma)),
lambda_query = the existing query-phase PoW bits,
```

the work-normalized security contribution is

```text
b_query = Q * b_per_query + lambda_query.
```

Part 3 chooses the smallest integer query count satisfying the **strict**
local target

```text
Q_min = floor((128 - lambda_query) / b_per_query) + 1,
Q_min * b_per_query + lambda_query > 128.
```

Fast uses `lambda_query = 0`, so its raw query error is strictly below
`2^-128`. Slim retains its existing 16-bit query-phase PoW and therefore
requires the raw query term to exceed 112 bits; the combined work-normalized
term is strictly above 128 bits. The PoW is verified by both the native
verifier and the recursive R1CS relation. Secure is a unique-decoding profile,
so its older 120-bit policy is not changed by this list-decoding milestone.

The canonical Johnson counts by inverse-rate exponent are:

| `r` in `rho = 2^-r` | Fast `Q` | Slim `Q` | Slim delivered bits |
| ---: | ---: | ---: | ---: |
| 1 | 279 | -- | -- |
| 2 | 136 | 119 | 128.267 |
| 3 | 91 | 79 | 128.228 |
| 4 | 68 | 60 | 129.338 |
| 5 | 55 | 48 | 128.578 |
| 6 | 46 | 41 | 130.221 |

Only the rates used by a profile's recursion ladder appear in its TOML. For
the representative `m27_fast` ladder this changes the per-level schedule from
`[218, 106, 71, 53]` (448 total) to `[279, 136, 91, 68]` (574 total). The
generator and validator recompute the bound from exact floating-point
formulas; the rounded `expected_eps_query_bits` fields are diagnostics, not
trusted security inputs. A boundary test proves that every generated Johnson
level clears 128 bits and that removing one query makes it fail or meet, but
not strictly clear, the target. Because queried-consistency batching has
degree `ceil(log2 Q)`, eight generated levels cross a power-of-two boundary;
their consistency-batching PoW increases by one bit as a derived consequence.

### Degree justifications by error family

The table uses the following concrete bad-event polynomials.

**Random equality-point reduction.** Converting a false table identity into a
scalar claim produces

```text
E(r) = sum_x eq(r, x) * f(x).
```

For `d` sampled coordinates, `eq(r,x)` is multilinear and `E` has total degree
at most `d`. This gives the Boolean `m` and element `m_words` initial bounds.

**Ordinary sumcheck.** Once the prover has observed/sent the current round
message, false acceptance at the new challenge is the zero set of the
difference between the claimed and required univariate identities. The
Boolean, element, merged-opening, multipoint, anchor and fold round
polynomials are products of at most two multilinear factors, hence degree at
most two. The chain shift has the same bound. The current Merkle wrapper
verifier accepts a degree-three interpolation, so its malicious-proof bound is
three.

**Optimized Boolean skip checks.** With `ell = 2^K_SKIP`, the zerocheck
combined skip polynomial has degree below `2 * ell`, hence degree at most
`2^(K_SKIP+1)-1`. The lincheck closing interpolation has degree below `ell`,
hence at most `2^k-1` for its actual skip width `k`.

**Linear batching.** Boolean/element lincheck batching, Product-GKR layer
batching/close, opening coefficient vectors and matrix-fold coefficient
vectors reduce to a nonzero polynomial such as

```text
E(alpha) = E_0 + alpha * E_1
```

or `sum_i alpha_i E_i`; its total degree is one.

**Ring switching.** After the claimed `s_hat_v` is bound and its public claim
relation checked, a false bridge to the packed witness leaves a nonzero
multilinear discrepancy evaluated at `r'' in F128^7`. Its total degree is at
most seven.

**Product-GKR fingerprint.** With `L` live entries the initial error is

```text
prod_x (f_x + alpha * id_x       + beta)
+
prod_x (g_x + alpha * id_sigma(x) + beta).
```

In characteristic two the two total-degree-`L` homogeneous terms are equal
and cancel because `sigma` permutes the live tags. The remaining degree is at
most `L-1`.

**Multipoint batching.** If the claimed value discrepancies are `e_j`, the
bad batching event is

```text
T(gamma) = sum_(j=0)^(K-1) gamma^j * e_j = 0,
```

which has degree at most `K-1` when the discrepancy vector is nonzero.

### Multipoint boundary

The multipoint relation uses

```text
T(gamma) = sum_(j=0)^(K-1) gamma^j e_j,
K = 128 n_RS + n_groups.
```

The previous fixed one-bit schedule was insufficient. The implementation now
derives the schedule from `K-1`: `K=256` needs 8 bits and `K=257` needs 9.
The common mixed route has two RS claims and at least one scalar group, hence
uses at least 9 bits.

### Why one PoW can protect a vector squeeze

Several optimized sites sample a whole vector after one PoW. This is sound
when the bad event is a nonzero multivariate polynomial of bounded **total**
degree in that vector; the number of coordinates is not itself the degree.

- Opening batching checks `sum_i gamma_i E_i = 0`, total degree one.
- Matrix folding checks `sum_i lambda_i E_i = 0` and later
  `sum_i mu_i E'_i = 0`, each total degree one.
- The Boolean/element initial equality-point conversions have total degree at
  most their stated dimension, so their dynamic bit count does depend on that
  dimension.

This is why opening and fold coefficient vectors use one `Pow(1)`, while the
initial equality points use `bits_for_degree(dimension)`.

### Challenge-dependent denominator audit

The Convention-A verifiers reconstruct a missing endpoint through equations
of the form

```text
g(0) = (running + t * g(1)) / (1 + t).
```

The exceptional event `t = 1` is itself a degree-one bad set. It does not
create an unprotected site:

- Boolean zerocheck's seven protocol-fixed inner coordinates are not one;
  every sampled outer coordinate comes from the already-grinded initial
  vector.
- Every element-zerocheck `tau_i` comes from its grinded initial vector.
- Product-GKR's `t` coordinates are prior protected round or layer-close
  challenges.

For each initial vector, the union of its sampled exceptional hyperplanes has
degree at most the dimension used by `bits_for_degree`. Thus that exceptional
family, treated as its own algebraic part, also has a strict sub-`2^-128`
work-normalized term.

## Production coverage matrix

This table maps each implemented family to its policy, active native entry
points, proof witness, and recursive handling. `Secure` selects the PIOP,
Product-GKR and PCS policies through `PcsParams`; the recursion tower selects
the matching fold policy through `tower_fold_grinding()`. `Fast` and `Slim`
retain their old transcript shape.

| Family | Policy / proof data | Active prover and verifier | Recursive R1CS |
| --- | --- | --- | --- |
| Boolean zerocheck | `ZerocheckGrinding`; `ZerocheckProof.grinding_nonces` | `zerocheck.rs`; normal and union calls in `prover.rs` / `verifier.rs` | generic child-tape `Pow` checks |
| Boolean lincheck | `LincheckGrinding`; `LincheckProof.grinding_nonces` | `lincheck.rs`, `lincheck/union.rs` | generic child-tape `Pow` checks |
| Element zerocheck/lincheck | `element_r1cs::Grinding`; both subproof nonce vectors | `element_r1cs/{zerocheck,lincheck,union}.rs` | generic child-tape `Pow` checks plus element verifier arithmetic |
| Product-GKR permutation | `BatchedGrinding`; `ProductGkrBatchedProof.grinding_nonces` | `circuit::prove_wiring_with_grinding`; matching ordinary/deferred verification | generic child-tape `Pow` checks plus GKR arithmetic |
| Ring switch / opening batching / merged rounds | `OpeningGrinding`; ring-switch nonce and exact nonce vectors | `pcs.rs`, `pcs/ring_switch.rs` | generic child-tape `Pow` checks plus opening arithmetic |
| Multipoint / anchor | `MultipointGrinding`; gamma, round and anchor nonces | active twisted multipoint functions in `pcs/jagged.rs` | generic child-tape `Pow` checks plus multipoint/anchor arithmetic |
| Dense, element and sigma folds | `FoldGrinding`; `FoldProof.grinding_nonces` | `matrix_fold.rs`, `aggregate.rs` | generic fold-tape `Pow` checks plus fold arithmetic |
| Jagged folds | same `FoldGrinding` and proof type | jagged fold entry points in `matrix_fold.rs`, `aggregate.rs` | same generic fold-tape handling |
| Ligerito Appendix C.3 | per-level claim/fold/consistency schedules; three nonce vectors | `pcs/ligerito.rs`, dense and succinct verifiers | generic child-tape `Pow` checks plus Ligerito arithmetic |
| Ligerito consistency queries | per-level `queries` and query-phase `grinding_bits` | `pcs/ligerito.rs`, dense and succinct verifiers | query sampling, openings, and query-phase `Pow` replayed from the child tape |

Key reviewer entry points:

- policy selection: [`pcs/commit.rs`](../crates/flock-core/src/pcs/commit.rs);
- native PoW and degree helper:
  [`challenger.rs`](../crates/flock-core/src/challenger.rs);
- Boolean PIOPs: [`zerocheck.rs`](../crates/flock-core/src/zerocheck.rs),
  [`lincheck.rs`](../crates/flock-core/src/lincheck.rs), and
  [`lincheck/union.rs`](../crates/flock-core/src/lincheck/union.rs);
- element PIOP: [`element_r1cs.rs`](../crates/flock-core/src/element_r1cs.rs)
  and [`element_r1cs/`](../crates/flock-core/src/element_r1cs);
- permutation/copy constraints:
  [`product_gkr.rs`](../crates/flock-core/src/product_gkr.rs) and
  [`circuit.rs`](../crates/flock-core/src/circuit.rs);
- PCS transport: [`pcs.rs`](../crates/flock-core/src/pcs.rs),
  [`pcs/ring_switch.rs`](../crates/flock-core/src/pcs/ring_switch.rs), and
  [`pcs/jagged.rs`](../crates/flock-core/src/pcs/jagged.rs);
- accumulation: [`matrix_fold.rs`](../crates/flock-core/src/matrix_fold.rs)
  and [`aggregate.rs`](../crates/flock-core/src/aggregate.rs);
- production plumbing: [`prover.rs`](../crates/flock-prover/src/prover.rs)
  and [`verifier.rs`](../crates/flock-core/src/verifier.rs); and
- recursive recording and R1CS replay:
  [`transcript_record.rs`](../crates/flock-core/src/transcript_record.rs) and
  [`circuit_merkle.rs`](../crates/flock-prover/tests/circuit_merkle.rs).

The native chain and Merkle-path wrappers also carry grinding for their own
packed-position and shift arguments. Their native implementations are recorded
in the schedule table for completeness. Recursive verification of those
wrapper arguments is explicitly outside this milestone's review scope.

### Active-challenge inventory result

The re-review searched every active `sample_f128` / `sample_f128_vec` in the
production verifier families above and followed its caller into the normal,
mixed/union, deferred, and recursive routes. Every algebraic squeeze is either
immediately preceded by its policy's PoW or is a previously protected
challenge reused by a later check.

Raw samples used only to create transcript fork seeds or bind transcript state
do not independently test a polynomial identity; the algebraic challenges
inside the resulting child transcript remain individually protected.
Ligerito OOD and Appendix C.3 algebraic randomness have since been audited.
The list-decoding query term has also been raised to a strict 128-bit
work-normalized target. Only the MCA/proximity term remains for the next
stage.

## Transcript and proof-shape improvements

### Opening coefficients

The old merged opening used a separate `Pow(1)` and scalar squeeze for every
claim. A representative child had 199 such claims. The new order is

```text
run all ring switches;
observe all packed-direct values;
Pow(1);
sample_f128_vec(n_RS + n_PD).
```

The discrepancy is total degree one in the whole coefficient vector, so one
PoW is sufficient. Ring-switch outputs are scaled after the shared vector is
sampled. This removes 198 PoW witnesses and 396 serialized finalizations per
representative child (198 PoWs plus 198 scalar-to-vector squeeze savings).

### Matrix-fold coefficients

Dense, element, sigma and jagged folds similarly use one vector squeeze for
all `lambda` coefficients and one for all `mu` coefficients. The recursive
reader maps every challenge ordinal to `(finalization, output-word offset)`;
it no longer assumes one challenge per finalization.

### Proof format

The incompatible additions and transcript changes now place the aggregate
proof format at v18. Product-GKR, matrix fold, chain and Merkle shift proofs carry
transcript-ordered nonce vectors; chain/Merkle wrappers carry their
packed-position nonce; opening batching has one nonce and one vector squeeze.
Version 16 added two-point Ligerito OOD data; version 17 adds the Appendix C.3
claim- and consistency-batching nonce vectors. Version 18 selects the larger
strict-128 Johnson query schedules; the Rust proof structure is unchanged,
but v17 caps, authentication paths, and transcript challenges cannot be
replayed under the new public schedule.

## Prover / verifier / recursive-circuit agreement

For each enabled family the native verifier:

1. derives the public policy from the PCS profile;
2. checks the exact expected nonce count (or canonical optional scalar);
3. observes the same proof messages as the prover;
4. verifies the nonce before sampling protected randomness; and
5. replays the same arithmetic relation.

The load-bearing transcript order is always

```text
observe the prover message(s) that fix the bad-event polynomial;
verify/grind Pow(lambda);
sample the protected scalar or vector;
use that sample in the verifier equation.
```

The family-specific nonce orders are:

| Family | Nonce order |
| --- | --- |
| Boolean zerocheck | initial vector; univariate skip; every ordinary round |
| Boolean lincheck | `alpha`; pinned-table `beta`s in slot order; every ordinary round; final skip |
| Element PIOP | initial `tau`; every zerocheck round; lincheck `alpha`; every lincheck round |
| Product-GKR | fingerprint; for each layer: `lambda`, its rounds, close |
| Outer PCS transport | each ring switch; shared opening coefficient vector; merged rounds; multipoint gamma; multipoint rounds; anchor rounds |
| Matrix fold | column coefficient vector; column rounds; row coefficient vector; row rounds |
| Ligerito | each OOD `beta`; every fold challenge; per-level consistency `alpha`; per-level glue `beta` |

Proofs with vector nonce fields are checked for their **exact** expected
length before replay. Optional scalar nonce fields use a canonical zero when
their site is disabled, so they cannot silently become an extra transcript
grinding knob.

For active recursive Flock proofs, `RecordingChallenger` records every PoW.
The recursive circuit uses the generic PoW relation for all child-tape sites,
including Product-GKR and PCS transport. Accumulator fold tapes are also
replayed in-circuit, including both coefficient-vector PoWs and every round
PoW. Hand parsers now tolerate PoW payload insertion and vector squeezes by
deriving payload/challenge locations from the op tape.

The Secure tower test consumes a recursive node proof as a child, so this is
not only a one-level parse check.

The recursive PoW implementation has two independent tests:

- a native/circuit differential test checks the exact BLAKE3 block,
  serialization, and prefix masks; and
- a focused R1CS proof accepts a valid nonce and rejects a neighboring invalid
  nonce.

The Secure node then exercises the generic relation at real Boolean, element,
Product-GKR, PCS-transport and fold sites. The Secure tower additionally
proves composability by consuming a recursive node proof as a child.

## Historical pre-fusion BLAKE and performance census

This section preserves the measurement that motivated fusion. It is not the
current implementation; the current isolated results are recorded near the
end of this document and in `grinding-hash-fusion-design.md`.

Commands:

```text
B3_CENSUS=1 TOWER_PROFILE={fast,secure} cargo test --release \
  -p flock-prover --test circuit_merkle \
  mvp11_two_to_one_recursion_node -- --ignored --nocapture
```

Historical per-child BLAKE census:

| component | Fast | Secure | delta |
| --- | ---: | ---: | ---: |
| Fiat--Shamir transcript chain | 1,707 | 2,438 | +731 |
| `H(child_publics)` | 216 | 432 | +216 |
| Ligerito leaves, paths and caps | 9,266 | 15,668 | +6,402 |
| standalone nonzero-PoW checks | 15 | 445 | +430 |
| **total per child** | **11,204** | **18,983** | **+7,779** |

The two-child fold region contributes 2,229 rows in Fast and 3,249 in Secure.
That pre-fusion Secure node had 1,400 standalone PoW BLAKE rows in total:
`2 * 445` child checks plus 510 fold checks.

| node metric | Fast | Secure | change |
| --- | ---: | ---: | ---: |
| online proving (three-run median) | 232 ms | 350 ms | +50.9% |
| native verification | 10 ms | 13 ms | +3 ms |
| outer proof | 291.5 KiB | 589.2 KiB | +102.1% |
| BLAKE rows | 24,637 | 41,215 | +67.3% |
| BLAKE slot `nu` | 15 | 16 | one capacity bit |
| circuit cell `mu` | 23 | 24 | one capacity bit |

These are full profile comparisons, not an isolated grinding benchmark.
Secure changes the child Ligerito geometry from 448 to 748 total queries; the
12,804 extra Ligerito opening rows across two children remain the dominant
increase. The new algebraic families plus fold grinding are partly offset by
opening-vector consolidation: relative to the earlier Secure snapshot of
40,035 rows, the completed node is 41,215 rows, a net increase of 1,180.

One Secure child records 449 PoW operations: 445 nonzero and four canonical
zero-bit sites. The distribution is

```text
{0: 4, 1: 52, 2: 380, 3: 4, 4: 2, 5: 3, 6: 1, 7: 1, 9: 1, 18: 1}.
```

Product-GKR accounts for 276 of the nonzero child sites. This high count is
expected from its per-layer quadratic rounds; each costs one recursive BLAKE
row, while the one 18-bit fingerprint grind dominates native nonce-search
work for this measured child.

## Test evidence

The final review reran the following commands from the repository root:

```sh
cargo test -p flock-core
cargo test -p flock-prover

cargo test -p flock-prover --test verifier_roundtrip \
  secure_profile_grinds_boolean_piops -- --ignored --nocapture
cargo test -p flock-prover --test union_element \
  secure_profile_grinds_element_piops -- --ignored --nocapture

cargo test -p flock-prover proof_io::tests -- --ignored

TOWER_PROFILE=secure cargo test --release -p flock-prover \
  --test circuit_merkle mvp11_two_to_one_recursion_node \
  -- --ignored --nocapture
TOWER_PROFILE=secure cargo test --release -p flock-prover \
  --test circuit_merkle mvp12_recursion_tower \
  -- --ignored --nocapture
```

Results:

- `flock-core`: 476 unit tests passed, followed by all 11 active integration
  tests; no failures.
- `flock-prover`: 74 active library tests and every active integration suite
  passed; no failures.
- Secure Boolean and Secure element/mixed production roundtrips passed.
- All three current v18 proof-bundle serialization/verification roundtrips
  passed (the earlier non-Ligerito checkpoint used v15).
- Secure `mvp11_two_to_one_recursion_node` passed with 41,215 BLAKE rows and a
  589.2 KiB outer proof in the post-rebase recorded run.
- Secure `mvp12_recursion_tower` passed, including a recursive proof consumed
  by another recursive verifier.

Focused evidence in the full suites additionally covers:

- invalid and malformed nonces for Boolean, element, Product-GKR, ring switch,
  dense fold, jagged fold and multipoint/anchor proofs;
- the `K=256/257` multipoint schedule boundary;
- standalone sigma and jagged recursive fold-tape replay after vector
  squeezing; and
- native/circuit PoW agreement plus valid/invalid recursive nonce behavior.

The 2026-08-12 Ligerito Part 2 review additionally ran:

```sh
cargo test -p flock-core --lib
cargo test -p flock-prover
cargo test -p flock-prover proof_io::tests -- --ignored --nocapture
cargo test -p flock-prover --test circuit_merkle \
  mvp10_circuit_inner_tape -- --ignored --exact --nocapture
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

All passed. The core run had 483 active tests and 22 ignored. The focused
Ligerito mutation test checks both dense and succinct verifiers, invalid
claim/consistency nonces, and missing/extra nonce vectors. The config suite
rejects each under-sized Appendix C.3 schedule and checks all 42 embedded
TOMLs against canonical generator output. The recursive tests prove that the
new `Pow` operations and protected arithmetic are replayed inside R1CS.

The 2026-08-12 Ligerito Part 3 review additionally ran:

```sh
cargo test -p flock-core --lib
cargo test -p flock-prover
cargo test -p flock-prover proof_io::tests -- --ignored --nocapture
cargo test -p flock-prover --test circuit_merkle \
  mvp10_circuit_inner_tape -- --ignored --exact --nocapture
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

All passed. The core run had 485 active tests and 22 ignored. The full prover
suite and all three v18 proof-container round trips passed. Focused config
tests check all 28 Fast/Slim Johnson configurations against canonical
derivation, check the strict one-query boundary, and reject a coherently
tampered under-target schedule. A real `m22_fast` native round trip and both
debug and release recursive verifier paths passed.

### Part 3 isolated performance

The release comparison holds the fused grinding implementation and two-point
OOD design fixed, changing only the Johnson query counts. Command:

```sh
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

| representative `m27_fast` metric | Part 2 | Part 3 | change |
| --- | ---: | ---: | ---: |
| total consistency queries | 448 | 574 | +28.1% |
| child BLAKE rows | 8,207 | 10,656 | +29.8% |
| child proof | 323.5 KiB | 434.4 KiB | +34.3% |
| child prove median | 81 ms | 86 ms | +6.2% |
| child native verify | 6 ms | 7 ms | +1 ms |
| outer BLAKE rows | 11,183 | 14,840 | +32.7% |
| outer proof | 253.4 KiB | 313.8 KiB | +23.8% |
| outer prove | 135 ms | 147 ms | +8.9% |
| outer native verify | 9 ms | 10 ms | +1 ms |

The child proving ranges overlapped (`76--92 ms` before and `85--101 ms`
after), so the timing delta is indicative rather than a stable microbenchmark.
The outer circuit stayed at `nu = 14`, `mu = 22`; Part 3 did not cross a
capacity boundary.

## Handoff to the next milestone

There is no remaining in-scope algebraic grinding implementation blocker.
Ligerito now runs in the Johnson list-decoding regime with two-point OOD
binding, and its Appendix C.3 algebraic terms are individually below
`2^-128` after grinding. Its consistency-query schedule also strictly clears
128 bits, including Slim's retained 16-bit query-phase grind. The next
security work is to move the MCA/correlated-agreement arithmetic to the
planned 256-bit quadratic-extension field after checking its bound and
recursive cost.

The global ledger, inactive legacy APIs, and recursive chain/Merkle wrapper
verification remain deliberate exclusions of this review, not defects that
block beginning Ligerito.

The isolated benchmark was completed after rebasing onto the fresh
`recursion_circuit` branch. Holding the original 448-query Johnson/OOD
Ligerito geometry fixed, the 2-to-1 recursion node changed from 24,637 to
27,559 BLAKE rows (+11.9%), from 291.5 to 302.0 KiB (+3.6%), and from a
three-run median of 232 to 257 ms online proving (+10.8%). The corresponding
level-2 tower changes were 27,600 to 32,400 rows (+17.4%) and 299.0 to 316.2
KiB (+5.8%). Both isolated recursive tests passed. The temporary config
substitution was removed after measurement. The component census and
Secure/UDR comparison are recorded in the historical-performance section
above.

The grinding verifier was subsequently fused with its protected
Fiat--Shamir squeeze. Repeating the same isolated experiment reduced the
Secure-grinding node from 27,559 to 24,455 BLAKE rows and from 302.0 to 295.2
KiB. Against an optimized no-new-grinding control (24,451 rows, 291.5 KiB),
the remaining grinding-only overhead is four BLAKE rows (+0.016%) and 3.7 KiB
(+1.3%). A final same-shape rerun measured three-run online medians of 217
versus 208 ms (+4.3%) and prove-component medians of 203 versus 195 ms
(+4.1%). The four-leaf recursion tower passed as well. The exact fused
transition and its security argument are documented in
`grinding-hash-fusion-design.md`.

## Code-quality notes

Good foundations now include a central degree-to-bits helper, exact nonce
shape checks, canonical disabled fields, vectorized linear batching, one
generic recursive PoW relation, and native/circuit differential tests. The
largest maintenance risk remains the hand-written transcript parser in
`circuit_merkle.rs`; deriving challenge and payload maps from the op tape has
removed two classes of fixed-offset bugs, but a generated verifier transcript
IR would be safer long term.

Two small documentation-only inaccuracies were found during the final review:

- a source comment in `lincheck.rs` says a quadratic round with two grinding
  bits gives `2^-130`; the correct term is `2^-129`. The implementation and
  the tables in this document are correct;
- the `tower_profile` comment in `circuit_merkle.rs` still says Secure selects
  only Boolean-zerocheck grinding, while it now selects all policies above.

There is also one policy-plumbing cleanup opportunity: the recursion test's
`tower_fold_grinding()` duplicates the same Secure/Fast/Slim choice exposed as
`PcsParams::matrix_fold_grinding()`. They currently agree, but using one source
would make future policy changes harder to drift.

## Reviewer checklist

A reviewer should be able to answer yes to all of the following after
following the code map and tests:

- Is every false polynomial fixed in the transcript before its PoW and
  challenge?
- Does its stated total-degree bound imply the configured bit count?
- Do prover and verifier derive dynamic counts from the same public shape?
- Does the verifier reject missing, extra, invalid, and noncanonical nonces?
- Does the recursive tape expose the exact pre-PoW transcript digest and nonce
  word to `emit_pow_checks`?
- Are the subsequent arithmetic challenge wires outputs of the post-nonce
  transcript state?
- Are vector-squeeze word offsets derived rather than assumed?
- Do the Secure production node and tower tests pass?
- Does every Johnson query level satisfy
  `Q * log2(1/(1-gamma)) + lambda_query > 128` exactly?

The 2026-08-11 non-Ligerito re-review and the 2026-08-12 Ligerito Parts 2 and
3 re-reviews answered yes to each applicable item.
