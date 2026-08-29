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

PROTOTYPED in `r1cs_hashes/hash_to_point_scatter.rs`: the degree-12
product sumcheck, both identities, the factor tables from a real record
witness, and the dense side as one MLE opening, with tamper tests. A
load-bearing simplification found on the way: XOR of 0/1 values IS the
field sum in characteristic 2, so the MLE of any XOR-of-wires table is
the F128-linear combination of wire MLEs; the symbolic counter and
residue bits never need materializing, and every factor terminal is a
public-weighted linear functional of the committed witness. The
remaining integration is discharging those terminals through the
packed-direct/lincheck opening layer instead of direct evaluation.
`C_H` for the aerie fingerprint is then either a second small
commitment of the region or a sub-cube opening; decide by measurement
(work item 4).

## 4. Work items, in order

1. Bucket descriptors: fixed shapes per (absorption blocks, squeeze
   blocks) per aerie spec Section 4, as repeated block-diagonal
   descriptors.
2. Sponge-chain amendment: public-XOR absorption plus the witness salt
   block (Section 3.1).
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
4. `Z_H` region plus copy link (Section 3.3): the scatter argument is
   PROTOTYPED (sumcheck, identities, factor tables, tamper tests);
   remaining: terminal discharge through the packed-direct/lincheck
   opening layer, and the dedicated-commitment versus sub-cube-opening
   decision by measured proof bytes and verifier time.
5. Implement aerie's `BinaryHashBackend` for the assembled system, with
   the `observe_bytes` seam and the fingerprint opening via
   `ring_switch::prove_batched`.
6. E2 host benchmarks at the aerie spec's `N_T` sweep.

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
