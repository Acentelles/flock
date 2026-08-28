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
  the carries as maj rows; the no-overflow condition `c_16 = 0` and the
  range flags `[a < 12,289] = 1`, `[u = (a > 6,144)]` are claim-level
  pins: their complement wires live in a small public region the
  verifier checks by opening, exactly the pattern the Keccak encoder
  uses for its public endpoints;
- accept comparison `x < 61,445` as the carry-out of `x + 4,091`
  (16 maj rows); ordered stable compaction as a 10-bit running counter
  (10 maj rows) plus one write-gate AND per slot.

Ballpark 95 rows and wires per slot; a nine-block record has 612 slots,
so about 58k slot rows and wires per record on top of the ten Keccak
permutations (each 38,400 chi ANDs). Measured numbers land with work
item 3; the aerie E0 census model v0 should be revised against them.

### 3.3 The `Z_H` output region and copy link

A dedicated aligned region per record (the `state_24` slot pattern:
byte-contiguous, `2^region_log`-aligned, zero-padded) holding the
`512 x 15` packed target bits. The compaction write gates pin the region
to the accepted candidates' `(a, u)` bits, which is the aerie spec's
Section 5.2 copy link. `C_H` for the fingerprint is then either a second
small commitment of this region alone or a sub-cube opening of the main
witness at Boolean prefix coordinates selecting the region; decide by
measurement (work item 4).

## 4. Work items, in order

1. Bucket descriptors: fixed shapes per (absorption blocks, squeeze
   blocks) per aerie spec Section 4, as repeated block-diagonal
   descriptors.
2. Sponge-chain amendment: public-XOR absorption plus the witness salt
   block (Section 3.1).
3. Slot-relation encoder (Section 3.2) with witness generation from
   aerie's `falcon-hash-circuit::trace` and a differential test against
   `falcon::preprocess::hash_to_point_public_raw`; report the measured
   E0 census, replacing aerie's model v0.
4. `Z_H` region plus copy link (Section 3.3); pick dedicated commitment
   versus sub-cube opening by measured proof bytes and verifier time.
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
