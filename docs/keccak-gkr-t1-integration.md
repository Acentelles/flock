# T1: Keccak-via-GKR integration plan (sponge lane)

Goal: prove the private-salt sponge Keccak with GKR instead of R1CS, committing
only each permutation's boundary (state_0, state_24) rather than the 24 round
states. Target: remove enough Flock CPU-work to cross three seconds at 16,384
signatures. See the aerie plan `specs/techniques/private-salt-flock-sub-three-second-2026-09-11.md`.

## Prototype state (B1, this branch base ed0c252)

`r1cs_hashes/keccak_gkr.rs`: `prove_layer` reduces a claim `MLE(L_round, rho)=v`
to a claim on `MLE(L_prev, r'')` with a degree-3 chi sumcheck
(`eq * (a0 + a2 + a1 a2)`) plus a batched linear reduction (`V * L`), Fiat-Shamir
via the flock challenger. The test `one_layer_reduction_is_exact_and_the_chain_closes`
runs the 24-layer chain for ONE permutation and prints its wall time. No
verifier, no batching over permutations, no sponge wiring.

## The integration crux: batch over the instance axis

The compact profile has 16,384 records x 10 live permutations = 163,840
permutations. Running the per-permutation chain 163,840 times is the wrong
shape. The layer reduction must be batched: one sumcheck for layer i over all
permutations at once, with variables (instance axis || 1600-bit state axis).
The eq and V/L tables gain an instance dimension; the round structure is shared.
This amortizes the per-round Fiat-Shamir and keeps the sumcheck arithmetic
proportional to total bits, not per-instance overhead x instances.

## Feasibility gate (decide before building the full path)

GKR removes the commit and opening of the Keccak intermediate rounds (measured
by the span split: hybrid.commit + hybrid.pcs_open share attributable to the
sponge). It adds the batched layer sumchecks over the round trace (uncommitted:
no Merkle, no RS, no opening). Net win = removed(commit+open of rounds) minus
added(batched GKR sumcheck). The prototype per-permutation chain time x 163,840
is the UNBATCHED upper bound on the added cost; batching lowers it. Promote only
if the projected net clears the ~2,816 ms CPU-work (~400 ms wall) the P0 census
requires. This is the "roughly 20x kernel" bar from the earlier route-C hand-off.

## Milestones

- M1. Batched layer reduction over N permutations; measure added CPU-work per
  layer and project the 24-layer, 163,840-permutation total. Gate: projected
  added cost < removed commit+open. (Kernel: SIMD the chi degree-3 round and the
  linear reduction; the univariate skip applies to the first state round.)
- M2. Boundary wiring: state_0 is the padded input block
  (r || hpk || 00 || 00 || m absorbed into the rate), state_24 the output; the
  sponge chains 10 permutations with absorb between. Commit only the boundaries
  plus the absorbed message bits; link state_24 of perm k to the absorb of
  perm k+1 and the final squeeze to the target.
- M3. Verifier: replay the layer reductions and check the boundary commitment
  openings; Fiat-Shamir over the 24-layer chain and the instance batch.
- M4. Replace the R1CS Keccak in `hash_to_point_sponge` with the GKR path behind
  a feature; keep the R1CS path as the differential reference.
- M5. 16K measurement on the host (user-owned): confirm the committed root drops
  and the net wall win is >= 400 ms before promotion.

## Risks

- Kernel speed (M1) is make-or-break; if the batched sumcheck cost approaches the
  removed commit+open, T1 does not reach three seconds alone.
- Boundary/absorb wiring must preserve exact Keccak sponge semantics and the
  record fingerprint / bridge linkage.
- No composed soundness claim until the verifier and Fiat-Shamir are complete
  and the extraction argument covers the GKR chain.
