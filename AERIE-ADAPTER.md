# Aerie private-salt HashToPoint adapter plan

| Field | Value |
|---|---|
| Base | upstream `succinctlabs/flock` at `8790722` |
| Branch | `aerie/private-salt-adapter`, local only until fork hosting is decided |
| Consumer | `aerie` `crates/falcon-hash-circuit::backend::BinaryHashBackend` |
| Governing specs | aerie `specs/falcon-v1-private-salt-hash.md`, decision record `specs/falcon-v1-private-salt-backend.md` |

## 1. What needs no fork change

- **Transcript seam.** `Challenger::observe_bytes` is a real, length-prefixed
  absorb on `FsChallenger` (`flock-core/src/challenger.rs:343`). The joint
  Section 7 composition is: construct `FsChallenger`, `observe_bytes` the
  outer aerie transcript digest before any flock message, run the flock
  protocol, then absorb the serialized flock proof back into the outer
  transcript. No trait or type changes. The `grind_pow` PoW hook must be
  accounted in the composed Fiat-Shamir analysis (aerie spec Section 9).
- **Arbitrary-point openings.** `pcs/ring_switch.rs::prove` takes any
  `F128` coordinate vector; `prove_batched` shares one witness bit-scan
  across points (two fingerprint repetitions = one scan). The `s_hat_v`
  message is the 128 per-basis-coordinate partial evaluations, which is
  the packed-leaf reconstruction the aerie fingerprint consumes.
- **Field identity.** `F128` is `GF(2^128)` under `x^128 + x^7 + x^2 + x + 1`
  with `theta_b = x^b`, identical to aerie's `K` (`falcon-hash-circuit::gf128`).
  No basis conversion anywhere on the boundary.

## 2. Reusable machinery inventory

| Piece | Where | Reuse |
|---|---|---|
| Keccak-f R1CS encoder, 24 rounds inline, implicit intermediate states | `r1cs_hashes/keccak.rs`, `K_LOG = 16`, 42,560 useful bits per permutation | as is for each sponge permutation |
| 3-wide Keccak packing (about 97% PCS utilization) | `r1cs_hashes/keccak3.rs` | preferred once the record circuit shape is fixed |
| Chain glue: consecutive-instance equality via a shift sumcheck over aligned I/O slots | `r1cs_hashes/chain_common.rs` (`ChainLayout`) | sponge chaining, with the absorption amendment below |
| Block-diagonal repeated instances, `RowMajor`/`BatchMajor` layouts | `flock-core/src/r1cs.rs` | the per-record repetition axis |
| Jagged multi-table composition | `flock-core/src/pcs/jagged.rs` | hash blocks, slot blocks, and the `Z_H` table under one PCS |

## 3. The gap: the HashToPoint record circuit

The upstream chain relation is `state_24[i] == state_0[i+1]`. HashToPoint
needs three amendments.

### 3.1 Absorption with a witness salt

The first absorbed block is `salt || hpk || 0x00 || 0x00 || m[..]` padded;
later absorbed blocks XOR public message bytes into the running state.
`hpk`, the domain bytes, and the message are public inputs; the 320 salt
bits are witness wires in the first block's `state_0` region. The chain
shift relation gains public XOR constants (fold into the selector or the
pin rows); the salt wires make block 1's input region partially witness,
which changes its self-loop pin rows from constants to committed wires.

### 3.2 The candidate slot relation

Constraint mechanics, from `flock-core/src/r1cs.rs`: a row `j` enforces
`(A_j z) & (B_j z) = C_j z`. Under `C = I` every row defines its wire; an
empty row forces its wire to zero; `c_0` is a real matrix, so a
general-`C` row expresses a pure constraint without consuming a wire; the
constant-one wire is pinned through `const_pin`. Over `GF(2)` the witness
is a bit vector by type, so Booleanity is free, and every full-adder
carry is one row via `maj(a, b, c) = (a xor c)(b xor c) xor c`:

```text
row:  (a_i xor c_i)(M_i xor c_i) = c_{i+1} xor c_i.
```

Per-slot design (this supersedes the aerie census model v0 for this
backend; model v0 counts Booleanity rows that GF(2) gets for free):

- committed wires: word `x` (16), accept flag and its comparison chain
  (17), residue `a` (14), quotient `q` as 3-bit binary (3), the `3q`
  mini-adder carries (3), main adder carries (16), centering flag and
  its chain (15), counter carries (10), write gate (1): about 95;
- `M = 12,289 q` is linear: `12,289 = 3 * 2^12 + 1`, so
  `M = (3q << 12) xor q` with disjoint bit ranges; only the 4-bit `3q`
  adder costs carries;
- `x = a + M` is enforced by defining `a_i := x_i xor M_i xor c_i` with
  the carries as maj rows; the forced conditions (no overflow, the `a`
  range, `q <= 5`) are self-referential rows `(expr ^ w)(1) = w`,
  enforced by the zerocheck itself with `C_0 = I` intact;
- accept comparison `x < 61,445` as the carry-out of `x + 4,091`
  (16 maj rows); ordered stable compaction as a 10-bit running counter
  (10 maj rows) plus one write-gate AND per slot.

Implemented in `r1cs_hashes/hash_to_point_slots.rs` at the satisfies
level: `K_LOG = 13`, 68 slots per block (one rate block), slot stride
100 with 99 wires (16 word bits, 3 quotient bits, 16 + 3 + 16 + 14 + 14
chain products, materialized accept and centering flags, 10 counter
products, 5 forced-zero pins), `USEFUL_BITS = 6,816` of 8,192 (83%
utilization). A nine-squeeze-block record costs about 61k slot wires
against 426k Keccak wires, so the slot layer is about 12.5% of the hash
cost. Witness generation evaluates the rows themselves (consistent with
the matrices by construction) and asserts the integer reference
semantics; tests cover the boundary battery, cross-block counter
chaining, tamper rejection, and the soundness demonstration that a
wrong quotient satisfies every definitional row and is caught only by
the forced-zero pins. The aerie E0 census model v0 should be revised
against these measured counts.

### 3.3 The `Z_H` output region and copy link

A dedicated aligned region per record holding the `512 x 15` packed
target bits, linked to the slot blocks by a SCATTER argument rather
than R1CS multiplexer rows (which would cost about `612 x 512` selector
products per record). The counter's monotone increments make the
scatter a stable permutation, and one identity makes it one sumcheck:

- Slot side: in characteristic 2,
  `beta^count = prod_b (1 + count_b (beta^(2^b) + 1))`, LINEAR in each
  committed counter bit. With a bit-combining challenge `gamma`, the
  slot fingerprint
  `S(beta) = sum_s gate_s * val_s(gamma) * beta^(count_s)` is a product
  sumcheck over slots with 12 multilinear factors (the write gate, ten
  counter factors, the gamma-combined value), so degree 12 per round.
  The gate is `acc_s * (1 xor cnt_bit_9)`: honest records accept more
  than 512 of their 612 candidates (the battery test pins this), so
  writes must stop at the 512th acceptance, and with the counter below
  1,024 the condition `count < 512` is one bit.
- Dense side: the power sum `sum_j Z[j] beta^j` equals
  `prod_b (1 + beta^(2^b)) * MLE_Z(x)` at the derived point
  `x_b = beta^(2^b) / (1 + beta^(2^b))` (defined whenever
  `beta^(2^b) != 1`), i.e. ONE extra MLE opening of the `Z_H` region at
  a public point.

Equality of the two sides at a post-commitment `beta` binds the dense
table to the gated slot outputs with soundness about `611 / 2^128` per
`beta` (the difference is a univariate of degree at most 611 in
`beta`); the `gamma` combination adds the usual bit-plane union term.
Stability and ordering need no extra argument: counters are forced by
the R1CS to increment exactly on accepted slots from a structural zero,
so gated slots hit indices `0..512` in order.

COMPLETE at the protocol level in
`r1cs_hashes/hash_to_point_scatter.rs`: the degree-12 product sumcheck,
both identities, the factor tables, AND the full terminal discharge.
The slot encoder moved to a PLANE-MAJOR layout (`K_LOG = 17`, each wire
class in its own 1,024-slot aligned plane, 101 planes, 79% utilization)
precisely so every class table is a sub-cube of the witness: its MLE at
`r` is an ordinary opening at `(plane bits, r)`, which the batched
ring-switch openings already support (a test pins the sub-cube
identity). The counter prefix parities discharge through one
rho-batched 2-factor sumcheck whose weight is the transparent
greater-than multilinear `GT(r, y)`, evaluable in `O(n)`; the prover
sends the ten per-bit claims, and the rho-combination binds them
(degree 9 in rho). The parity sources are the accept plane for bit 0
and carry-out plane `H_{b-1}` for bit b (the stored products are carry
OUTS; an off-by-one here was caught by the terminal reconstruction
test). Load-bearing simplification: XOR of 0/1 values IS the field sum
in characteristic 2, so no symbolic bit ever needs materializing.
Remaining: route the roughly 35 sub-cube openings and the discharge
through the real batched PCS opening against the slot commitment
(plumbing, no new protocol content).
`C_H` for the aerie fingerprint is then either a second small
commitment of the region or a sub-cube opening; decide by measurement
(work item 4).

## 4. Work items, in order

1. Bucket descriptors: fixed shapes per (absorption blocks, squeeze
   blocks) per aerie spec Section 4, as repeated block-diagonal
   descriptors.
2. Sponge-chain amendment: DONE in `hash_to_point_sponge.rs`, and
   simpler than planned: records stride four 3-wide `keccak3` blocks
   (twelve-permutation capacity, ten live; permutation `e` at block
   `e % 4`, sub-keccak `e / 4`), so with a post-commitment `delta` every
   chaining constraint is a transparent-weighted SUB-CUBE OPENING pair
   (no chain shift sumcheck):
   interior edges `IN_e = OUT_(e-1)`, the absorption edge plus the
   verifier-computed public MLE of the second message blocks, and the
   record-start pinning with the 320 salt bits subtracted as two scaled
   aligned sub-cubes. The witness's physical state layout is lane-major
   = FIPS byte order, so byte ranges are sub-cubes directly; a sha3
   differential pins the simulation and padding. The WORD LINKAGE
   (`hash_to_point_link.rs`) closes the last gap: the slot words equal
   the squeeze stream via four-challenge weighted sums over aligned
   piece decompositions, with the big-endian byte swap a single
   address-bit complement (`gamma` weights `(gamma^8, 1)` on position
   bit 3) and the x-planes sixteen-aligned so the bit combination is
   one tensor. `prove_hash_to_point`/`verify_hash_to_point` drive both
   lanes plus the linkage under one transcript. The linkage claims are
   FOLDED into the lanes' single batched openings (each lane split into
   `*_core` + `open_*` taking extra points; the composed driver runs
   both cores, samples the linkage, then opens each lane once): two
   Ligerito openings total, proof bytes 1,420,280 -> 849,536 at 32
   records and 1,470,752 -> 876,104 at 64 (about -40%), and the
   packed-witness clones leave the composed path. The full composition
   roundtrips (`full_hash_to_point_roundtrips`, 32 records, with
   linkage-value and public-message tamper rejection). Landing it
   required a fork fix in `pcs/ligerito.rs`: the succinct verifier's
   final-level `yr` binding samples `alpha_last`/`beta_last`, but the
   prover never consumed them, so the challengers desynced for any
   protocol stage appended AFTER an opening (invisible upstream because
   opens were always the terminal transcript consumer). The prover's
   terminal branch now mirrors both samples.
3. Slot-relation encoder (Section 3.2): DONE through the generic
   proving path in `r1cs_hashes/hash_to_point_slots.rs`, restructured
   to ONE RECORD PER BLOCK (`K_LOG = 16`, 612 slots, 93% utilization):
   the counter chains across the whole record inside the block and its
   zero reset is structural (no counter-input wires), so cross-block
   counter chaining machinery is unnecessary. `SlotSetup` wraps
   `prover::prove_ligerito` and `verifier::verify_ligerito`; the
   roundtrip test proves 64 records (39,168 slots, `m = 22`) and
   verifies, with domain separation rejecting. The forced-zero pins
   became SELF-REFERENTIAL rows `(expr ^ w)(1) = w`, satisfiable
   exactly when `expr = 0`, so the zerocheck enforces them with
   `C_0 = I` intact and no claim-level machinery (the generic verifier
   converts the c-claim to a z-claim assuming identity `C`, so empty-C
   constraint rows would NOT verify). Remaining: the write-gate wire
   for the scatter and the full-record differential against
   `falcon::preprocess::hash_to_point_public_raw`.
4. `Z_H` region plus copy link (Section 3.3): COMPLETE against real
   commitments. The `Z_H` table lives as eight aligned planes INSIDE the
   slot-witness commitment (free-input wires bound by the scatter), so
   there is no second commitment and no small-`m` Ligerito problem. The
   multi-record scatter runs once over the `(record, slot)` domain with
   a transparent delta-power record factor (13 factors); the dense side
   is one sub-cube opening at the mixed power point (delta over record
   bits, beta over index bits, gamma over coordinate bits).
   `hash_to_point_record::{prove_record, verify_record}` assemble the
   slot R1CS (zerocheck + lincheck through the generic path), the
   scatter, the `Z_H` binding identity, the fifteen aerie-fingerprint
   sub-cube openings (their theta-weighted sum is `MLE_K(Z_H, r)`), and
   ONE batched opening carrying the base `[ab, c]` quirky claims plus
   all 57 multilinear claims through the new `LowBinding` seam.
   Roundtrip and tamper tests pass at 32 records (`m = 22`).

   Post-bench optimization pass (first host numbers: 28-31 ms/record,
   flat 15 ms verify, ~400 KB log-growth proofs): materializing the ten
   counter-AFTER bits as planes made every counter tap O(1) instead of
   O(slot) AND deleted the GT discharge entirely, because on gated
   slots `beta^(count-before) = beta^(-1) beta^(count-after)` and
   count-after lives in same-slot planes, so the counter factors are
   plain sub-cube openings at the scatter point (the binding identity
   gains one factor of beta). Claim values now evaluate over gathered
   sub-cubes instead of the dense witness. Same-machine A/B at 32
   records: prove 4,611 -> 1,231 ms; witness generation needs a
   dependency-aware order for the counter family (H(s) taps CNT(s-1),
   CNT(s) taps H(s)). The remaining prover dominator is the degree-13
   scatter sumcheck (42%), ordinary sumcheck engineering.

   Post-optimization host anchors (M3 Max, quiet host, 2026-08-30,
   `hash_to_point_record_bench`; Keccak sponge lane NOT included):

   | records | prove_ms | verify_ms | proof_bytes |
   |---:|---:|---:|---:|
   | 32 | 197.6 | 10.7 | 397,912 |
   | 64 | 308.4 | 10.9 | 410,776 |
   | 128 | 764.5 | 11.6 | 421,672 |
   | 256 | 1,564.0 | 12.0 | 442,376 |

   About 6.1 ms/record above 64 records, verifier flat, bytes
   logarithmic (about 520 KB extrapolated at 16,384 records, below the
   640 KB salt array the profile removes). Honest accounting: at this
   rate the bridge lane extrapolates to about 100 s at 16,384 records,
   about 50-100x the spec's Keccak-lane estimate, so the bridge lane is
   currently the profile's prover bottleneck. Known unexploited levers:
   the scatter sumcheck still has standard optimizations untouched
   (gate sparsity, factoring the transparent delta factor out of the
   inner product). The single-threaded custom stages were parallelized
   2026-08-30 (rayon): witness generation per block, the scatter round
   evaluation as a one-pass-per-index chunked reduction (each factor
   pair loaded once and extended to all degree + 1 points),
   opening-value gathers per claim, sponge traces per record, and the
   linkage gathers. Same-session managed-env stage ratios at 32
   records: scatter sumcheck about 9x, witness generation about 5x,
   opening values about 3x.

   COMPLETE-relation host anchors (M3 Max, quiet host, 2026-08-31,
   `hash_to_point_full_bench`, post parallelization + linkage fold-in;
   sponge lane + slot relation + scatter + `Z_H` + fingerprint + word
   linkage, salt private; NOT a Falcon aggregate-signature result):

   | records | prove_ms | verify_ms | proof_bytes |
   |---:|---:|---:|---:|
   | 32 | 139.6 | 22.2 | 849,536 |
   | 64 | 169.1 | 23.1 | 876,104 |
   | 128 | 258.3 | 24.3 | 903,456 |
   | 256 | 417.5 | 25.1 | 944,272 |

   Marginal cost about 1.3 ms/record with a ~100 ms floor; the
   complete relation at 256 records is 3.7x faster than the record
   lane ALONE was before this pass. Verifier flat at 22-25 ms. Linear
   extrapolation to 16,384 records: about 21 s prove (a floor: PCS
   stages grow superlinearly), 10-20x the spec's Keccak-lane estimate,
   down from 50-100x. Bytes grow 27-41 KB per doubling, about 1.2 MB
   extrapolated at 16,384: ABOVE the 640 KB salt array the profile
   removes, so on wire bytes alone the profile is currently net
   negative at target scale; the byte levers are Ligerito config
   (query counts, rate) rather than more claim folding. Remaining
   prover levers: PCS config tuning for these two instance shapes,
   then the generic zerocheck/lincheck stages, now the dominant host
   cost share.
5. Implement aerie's `BinaryHashBackend` for the assembled system: the
   flock-side engine is `prove_record`/`verify_record` (commitment,
   relation minus the sponge lane, fingerprint value at a
   challenger-sampled point; in production the challenger is the joint
   aerie transcript seeded through `observe_bytes`). The thin aerie-side
   trait impl is type conversion only (same field, same polynomial).
6. E2 host benchmarks:
   `cargo run --release --example hash_to_point_record_bench [N...]`
   emits TSV for the record lane alone (coverage banner: no sponge).
   `cargo run --release --example hash_to_point_full_bench [N...]`
   benches the COMPLETE relation (sponge + slots + scatter + `Z_H` +
   fingerprint + word linkage; salt private). Still outside it: the
   aerie-side dual-commitment consistency and the composed Section 7
   transcript; never present either as a Falcon aggregate-signature
   result.

## 5. Open questions

- Whether `keccak.rs` (65% utilization, simpler) or `keccak3.rs` (97%)
  hosts the record circuit first; start with `keccak.rs`, migrate after
  item 3's census.
- Where the slot relation lives: extra regions inside the Keccak blocks
  versus a separate jagged table family; the jagged path looks cleaner
  and keeps the Keccak walkers untouched.
- PoW grinding policy (`grind_pow`) under the joint transcript.
- The parallel default-stack test run is marginal: an upstream rayon
  worker sits close to the 2 MB default, and adding tests can shift
  co-scheduling enough to abort it (verified: base parallel passes,
  parallel with the slot tests sometimes aborts, serial and
  `RUST_MIN_STACK=512MB` parallel both pass). Run the suite with
  `RUST_MIN_STACK=536870912`, as aerie already does.
