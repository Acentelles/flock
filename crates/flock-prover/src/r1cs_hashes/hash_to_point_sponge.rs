//! The sponge lane of the aerie private-salt HashToPoint relation.
//!
//! Proves, per record, the SHAKE256 sponge over the framed input
//! `salt || hpk || 0x00 || 0x00 || message` for the default bucket (two
//! absorption blocks, nine squeeze blocks, ten Keccak-f permutations):
//! the Keccak permutations through the existing walker-circuit R1CS, and
//! the sponge chaining as TRANSPARENT-WEIGHTED SUB-CUBE OPENINGS with no
//! new sumchecks:
//!
//! - records stride four 3-wide keccak3 blocks (twelve-permutation
//!   capacity, ten live), so a within-record permutation address is a
//!   Boolean bit-field: permutation `e` sits at block `e % 4`,
//!   sub-keccak `e / 4`, state slots `2 (e / 4) + out`;
//! - with a post-commitment challenge `delta`, each edge class `e` gives
//!   one delta-power-batched opening pair: `IN_e` (the `state_0` slot at
//!   offset bits `e`) must equal `OUT_(e-1)` (the `state_24` slot);
//! - the one absorption edge (`e = 1`) adds the verifier-computed public
//!   MLE of the second message blocks;
//! - the record-start states (`e = 0`) pin everything except the salt:
//!   the 320 salt bits split into aligned 256- and 64-bit sub-cubes whose
//!   scaled openings subtract from the full-slot opening, leaving the
//!   public frame the verifier computes itself.
//!
//! The witness's PHYSICAL state layout is lane-major, which is exactly
//! the FIPS byte order, so byte ranges are sub-cubes directly; only the
//! simulation converts to the walker's logical index
//! (`j = (p % 64) * 25 + p / 64`), pinned by a sha3 differential test.
//!
//! Everything rides ONE batched opening on the Keccak commitment through
//! the `LowBinding` seam, appended to the base `[ab, c]` claims. The word
//! linkage to the slot blocks (that the candidate words equal the squeeze
//! stream) is the companion argument in `hash_to_point_record`'s lane;
//! see `AERIE-ADAPTER.md`.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck;
use flock_core::pcs::{self, Commitment, LowBinding};
use flock_core::zerocheck;

use super::hash_to_point_record::flock_claim_shape;
use super::keccak::{self, STATE_BITS, State};
use super::keccak3::{self, KeccakLincheckCircuit, KeccakSetup};

/// Live permutations per record in the default bucket: two absorption
/// blocks plus eight further squeezes (the first squeeze block is the
/// second absorption permutation's output).
pub const LIVE_PERMS: usize = 10;
/// Blocks per record under the 3-wide encoder, a power of two so the
/// within-record block offset is a Boolean bit-field.
pub const BLOCKS_PER_RECORD: usize = 4;
/// Permutation capacity per record: three sub-keccaks in each of the
/// four blocks (ten live, two padding).
pub const PERM_CAPACITY: usize = keccak3::N_SUB * BLOCKS_PER_RECORD;
pub const RATE_BYTES: usize = 136;
pub const SALT_BYTES: usize = 40;
pub const SALT_BITS: usize = 8 * SALT_BYTES;
/// Message lengths that make exactly two absorption blocks: the frame
/// prefix is 106 bytes, so the frame must exceed one rate block and fit,
/// with the two SHAKE padding bits, in two.
pub const MESSAGE_BYTES: core::ops::RangeInclusive<usize> = 31..=165;

/// One record's sponge inputs; the salt is the private part.
#[derive(Clone, Debug)]
pub struct SpongeRecord {
    pub salt: [u8; SALT_BYTES],
    pub hpk: [u8; 64],
    pub message: Vec<u8>,
}

/// The public part, all the verifier needs.
#[derive(Clone, Debug)]
pub struct SpongePublic {
    pub hpk: [u8; 64],
    pub message: Vec<u8>,
}

fn state_from_physical_bytes(bytes: &[u8; 200]) -> State {
    let mut state = [false; STATE_BITS];
    for p in 0..STATE_BITS {
        let logical = (p % 64) * 25 + p / 64;
        state[logical] = (bytes[p / 8] >> (p % 8)) & 1 == 1;
    }
    state
}

fn state_to_physical_bytes(state: &State) -> [u8; 200] {
    let mut bytes = [0_u8; 200];
    for p in 0..STATE_BITS {
        let logical = (p % 64) * 25 + p / 64;
        if state[logical] {
            bytes[p / 8] |= 1 << (p % 8);
        }
    }
    bytes
}

/// The two absorption blocks of the framed input, SHAKE-padded.
pub fn framed_blocks(record: &SpongeRecord) -> ([u8; RATE_BYTES], [u8; RATE_BYTES]) {
    assert!(
        MESSAGE_BYTES.contains(&record.message.len()),
        "the default bucket needs exactly two absorption blocks"
    );
    let mut frame = Vec::with_capacity(106 + record.message.len());
    frame.extend_from_slice(&record.salt);
    frame.extend_from_slice(&record.hpk);
    frame.extend_from_slice(&[0x00, 0x00]);
    frame.extend_from_slice(&record.message);
    let mut block1 = [0_u8; RATE_BYTES];
    block1.copy_from_slice(&frame[..RATE_BYTES]);
    let mut block2 = [0_u8; RATE_BYTES];
    block2[..frame.len() - RATE_BYTES].copy_from_slice(&frame[RATE_BYTES..]);
    // SHAKE pad10*1: domain 0x1F after the message, 0x80 on the last byte.
    block2[frame.len() - RATE_BYTES] ^= 0x1F;
    block2[RATE_BYTES - 1] ^= 0x80;
    (block1, block2)
}

/// The ten live `state_0` states plus the 612 squeezed big-endian words.
pub fn sponge_trace(record: &SpongeRecord) -> ([State; LIVE_PERMS], Vec<u16>) {
    let (block1, block2) = framed_blocks(record);
    let mut states = [[false; STATE_BITS]; LIVE_PERMS];
    let mut bytes = [0_u8; 200];
    bytes[..RATE_BYTES].copy_from_slice(&block1);
    states[0] = state_from_physical_bytes(&bytes);

    let mut words = Vec::with_capacity(612);
    let mut state = states[0];
    for instance in 1..LIVE_PERMS {
        keccak::keccak_f(&mut state);
        let mut out = state_to_physical_bytes(&state);
        if instance == 1 {
            for (byte, &xor) in out.iter_mut().zip(block2.iter()) {
                *byte ^= xor;
            }
        }
        states[instance] = state_from_physical_bytes(&out);
        state = states[instance];
        // Squeeze block q is read from state_24 of instance 1 + q; here
        // that is states[instance]'s pre-image... the squeeze words come
        // from the OUTPUT of instances 1..=9, i.e. state_0 of the NEXT
        // instance for 2..=9 plus the final output. Collect from the
        // output directly:
        if instance >= 2 {
            let squeezed = state_to_physical_bytes(&states[instance]);
            for w in 0..68 {
                words.push(u16::from_be_bytes([squeezed[2 * w], squeezed[2 * w + 1]]));
            }
        }
    }
    // The last squeeze block: the output of the tenth permutation.
    keccak_f_output_words(&mut state, &mut words);
    (states, words)
}

fn keccak_f_output_words(state: &mut State, words: &mut Vec<u16>) {
    keccak::keccak_f(state);
    let squeezed = state_to_physical_bytes(state);
    for w in 0..68 {
        words.push(u16::from_be_bytes([squeezed[2 * w], squeezed[2 * w + 1]]));
    }
}

/// Prover state: the Keccak family sized for `records * 16` instances.
pub struct SpongeSetup {
    pub records: usize,
    pub keccak: KeccakSetup,
}

impl SpongeSetup {
    pub fn new(records: usize) -> Self {
        assert!(records.is_power_of_two(), "records must be a power of two");
        Self {
            records,
            keccak: KeccakSetup::new(records * PERM_CAPACITY),
        }
    }

    fn record_vars(&self) -> usize {
        self.keccak.n_blocks_log() - 2
    }
}

/// The sponge-lane proof: the Keccak R1CS plus the chaining openings.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpongeProof {
    pub commitment: Commitment,
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    /// Claimed opening values in [`sponge_points`]' fixed order.
    pub opening_values: Vec<F128>,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

/// The fixed claim order: `IN_e` for `e in 0..10`, `OUT_e` for
/// `e in 0..10`, then the two salt sub-cubes of the record starts.
fn sponge_points(record_vars: usize, delta: F128, r: &[F128]) -> Option<Vec<Vec<F128>>> {
    assert_eq!(r.len(), 11);
    let mut scale = F128::ONE;
    let mut rec_coords = Vec::with_capacity(record_vars);
    let delta_powers = {
        let mut powers = Vec::with_capacity(record_vars);
        let mut current = delta;
        for _ in 0..record_vars {
            powers.push(current);
            current *= current;
        }
        powers
    };
    for i in 0..record_vars {
        let weight = delta_powers[record_vars - 1 - i];
        let denominator = F128::ONE + weight;
        if denominator == F128::ZERO {
            return None;
        }
        scale *= denominator;
        rec_coords.push(weight * denominator.inv());
    }
    // NOTE: unlike the Z_H power point, the edge claims want the PLAIN
    // delta-power weighted sum, so the derived coords and scale apply to
    // every claim uniformly and cancel in the equality checks; the public
    // terms are computed with raw delta powers and divided by the scale.
    let _ = scale;

    let boolean = |bit: bool| if bit { F128::ONE } else { F128::ZERO };
    let point = |e: usize, out: bool, offset_high: &[F128], low: &[F128]| -> Vec<F128> {
        let mut p = rec_coords.clone();
        // Permutation e sits at block e % 4, sub-keccak e / 4.
        let blk = e % BLOCKS_PER_RECORD;
        for j in (0..2).rev() {
            p.push(boolean((blk >> j) & 1 == 1));
        }
        // Slot bits (6, MSB-first): sub-keccak state slot 2 (e / 4) + out.
        let slot = 2 * (e / BLOCKS_PER_RECORD) + usize::from(out);
        for j in (0..6).rev() {
            p.push(boolean((slot >> j) & 1 == 1));
        }
        p.extend_from_slice(offset_high);
        p.extend_from_slice(low);
        p
    };
    let mut points = Vec::with_capacity(22);
    for e in 0..LIVE_PERMS {
        points.push(point(e, false, &[], r));
    }
    for e in 0..LIVE_PERMS {
        points.push(point(e, true, &[], r));
    }
    // Salt sub-cubes of the record-start state: physical bits 0..256 and
    // 256..320 of the state_0 slot.
    let z = F128::ZERO;
    points.push(point(0, false, &[z, z, z], &r[3..]));
    points.push(point(0, false, &[z, z, F128::ONE, z, z], &r[5..]));
    Some(points)
}

/// The verifier-computed public terms: the delta-weighted 11-variable
/// region MLEs of (a) the second absorption blocks and (b) the record
/// start frames with the salt bits zeroed.
fn public_terms(
    publics: &[SpongePublic],
    record_vars: usize,
    delta: F128,
    r: &[F128],
) -> (F128, F128) {
    let weights = {
        // eq tensor over the 2048-bit slot, MSB-first over 11 vars.
        let mut w = vec![F128::ONE];
        for &coord in r {
            let mut next = Vec::with_capacity(2 * w.len());
            for &x in &w {
                next.push(x * (F128::ONE + coord));
                next.push(x * coord);
            }
            w = next;
        }
        w
    };
    let records = 1 << record_vars;
    let mut xor_term = F128::ZERO;
    let mut frame_term = F128::ZERO;
    let mut delta_power = F128::ONE;
    for record in 0..records {
        if let Some(public) = publics.get(record) {
            let stand_in = SpongeRecord {
                salt: [0_u8; SALT_BYTES],
                hpk: public.hpk,
                message: public.message.clone(),
            };
            let (block1, block2) = framed_blocks(&stand_in);
            let mut xor_sum = F128::ZERO;
            let mut frame_sum = F128::ZERO;
            for byte in 0..RATE_BYTES {
                for bit in 0..8 {
                    let p = 8 * byte + bit;
                    if (block2[byte] >> bit) & 1 == 1 {
                        xor_sum += weights[p];
                    }
                    // The frame's public part: everything past the salt.
                    if p >= SALT_BITS && (block1[byte] >> bit) & 1 == 1 {
                        frame_sum += weights[p];
                    }
                }
            }
            xor_term += delta_power * xor_sum;
            frame_term += delta_power * frame_sum;
        }
        delta_power *= delta;
    }
    (xor_term, frame_term)
}

/// The scale relating the derived-coordinate openings to raw delta-power
/// sums: `prod_b (1 + delta^(2^b))`.
fn record_scale(record_vars: usize, delta: F128) -> Option<F128> {
    let mut scale = F128::ONE;
    let mut power = delta;
    for _ in 0..record_vars {
        let denominator = F128::ONE + power;
        if denominator == F128::ZERO {
            return None;
        }
        scale *= denominator;
        power *= power;
    }
    Some(scale)
}

/// Salt sub-cube scales: the full-slot eq weights restricted to the two
/// aligned salt pieces factor into these public prefixes.
fn salt_scales(r: &[F128]) -> (F128, F128) {
    let one = F128::ONE;
    let s256 = (one + r[0]) * (one + r[1]) * (one + r[2]);
    let s64 = (one + r[0]) * (one + r[1]) * r[2] * (one + r[3]) * (one + r[4]);
    (s256, s64)
}

/// Prover-side leftovers a cross-lane argument needs to open this
/// commitment again: the packed witness and the commit-time data.
pub struct LaneArtifacts {
    pub z_packed: Vec<F128>,
    pub prover_data: pcs::ProverData,
    pub commitment: Commitment,
}

/// Everything the sponge lane produces before its batched opening: the
/// commit-and-R1CS core, the chaining claim set, and the squeeze words.
/// Cross-lane arguments read `fast.z_packed` for their claim values and
/// fold their points into [`open_sponge`].
pub struct SpongeCore {
    pub fast: crate::prover::ProveCore,
    pub points: Vec<Vec<F128>>,
    pub opening_values: Vec<F128>,
    pub all_words: Vec<Vec<u16>>,
}

/// The standalone sponge lane: core plus an immediate opening with no
/// cross-lane claims. Transcript-identical to the composed path with an
/// empty extra set.
pub fn prove_sponge<Ch: Challenger>(
    setup: &SpongeSetup,
    records: &[SpongeRecord],
    challenger: &mut Ch,
) -> (SpongeProof, Vec<Vec<u16>>, LaneArtifacts) {
    let core = prove_sponge_core(setup, records, challenger);
    let words = core.all_words.clone();
    let z_packed = core.fast.z_packed.clone();
    let (proof, prover_data) = open_sponge(setup, core, &[], &[], challenger);
    let artifacts = LaneArtifacts {
        z_packed,
        prover_data,
        commitment: proof.commitment.clone(),
    };
    (proof, words, artifacts)
}

/// The flat keccak3 input-state list, block-major within each record:
/// `initial_states[3 (4 rec + blk) + sub]` is permutation `4 sub + blk`
/// of record `rec`, with zero states in the two pad positions.
pub fn sponge_initial_states(traces: &[[State; LIVE_PERMS]]) -> Vec<State> {
    let zero_state: State = [false; STATE_BITS];
    let mut initial_states = Vec::with_capacity(traces.len() * PERM_CAPACITY);
    for states in traces {
        for blk in 0..BLOCKS_PER_RECORD {
            for sub in 0..keccak3::N_SUB {
                let e = BLOCKS_PER_RECORD * sub + blk;
                initial_states.push(if e < LIVE_PERMS { states[e] } else { zero_state });
            }
        }
    }
    initial_states
}

pub fn prove_sponge_core<Ch: Challenger>(
    setup: &SpongeSetup,
    records: &[SpongeRecord],
    challenger: &mut Ch,
) -> SpongeCore {
    use rayon::prelude::*;
    assert_eq!(records.len(), setup.records);
    let record_vars = setup.record_vars();
    let trace = std::env::var("RECORD_TRACE").is_ok();
    let mut stage = std::time::Instant::now();
    let mut lap = |label: &str| {
        if trace {
            eprintln!(
                "  [prove_sponge] {label}: {:7.1} ms",
                stage.elapsed().as_secs_f64() * 1e3
            );
        }
        stage = std::time::Instant::now();
    };
    let traces: Vec<([State; LIVE_PERMS], Vec<u16>)> =
        records.par_iter().map(sponge_trace).collect();
    let mut state_lists = Vec::with_capacity(records.len());
    let mut all_words = Vec::with_capacity(records.len());
    for (states, words) in traces {
        state_lists.push(states);
        all_words.push(words);
    }
    // Blocks past the live records are the witness builder's all-zero
    // padding triples.
    let initial_states = sponge_initial_states(&state_lists);
    lap("sponge traces");

    let (z_packed, a_packed, b_packed, z_lincheck) =
        keccak3::generate_witness_with_ab_packed_and_lincheck(
            &initial_states,
            setup.keccak.n_blocks_log(),
        );
    lap("keccak witness");
    let core = crate::prover::prove_fast_core(
        &setup.keccak.r1cs,
        &setup.keccak.pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        &KeccakLincheckCircuit,
        challenger,
    );
    lap("keccak core (commit + zerocheck + lincheck)");

    challenger.observe_label(b"aerie-sponge-challenges-v0");
    let delta = challenger.sample_f128();
    let r = challenger.sample_f128_vec(11);

    let points = sponge_points(record_vars, delta, &r).expect("derived coords defined");
    let opening_values = gather_eval_many(&core.z_packed, &points);
    lap("opening values");

    SpongeCore {
        fast: core,
        points,
        opening_values,
        all_words,
    }
}

/// The sponge lane's single batched opening: `[ab, c]` quirky, the
/// chaining claims, and any cross-lane multilinear `extra_points`
/// (appended after the lane's own claims, values carried by the caller).
/// Returns the prover data for callers that keep artifacts.
pub fn open_sponge<Ch: Challenger>(
    setup: &SpongeSetup,
    core: SpongeCore,
    extra_points: &[Vec<F128>],
    extra_values: &[F128],
    challenger: &mut Ch,
) -> (SpongeProof, pcs::ProverData) {
    // The sponge lane keeps per-claim ring switches: its bit domain is
    // three variables LARGER than the record lane's (m = log2(4N) + 17,
    // already 29 at 1024 records), so the closure's dense weight table
    // and folded witness blow past memory exactly where the lane's
    // claim count (~32) makes the closure least valuable. The staged
    // family-form closure for this lane is documented in the profile
    // note as the 16K follow-up.
    assert_eq!(extra_points.len(), extra_values.len());
    let _ = extra_values;
    let fast = core.fast;
    let ab = fast.ab.clone();
    let c = fast.c.clone();
    let mut x_fulls: Vec<Vec<F128>> = vec![
        {
            let mut v = ab.point.x_inner_rest.clone();
            v.extend_from_slice(&ab.point.x_outer);
            v
        },
        {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        },
    ];
    for point in core.points.iter().chain(extra_points) {
        let (_low, x_outer) = flock_claim_shape(point);
        x_fulls.push(x_outer);
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pre_ab: Option<&[F128]> = fast.s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(fast.s_hat_v_c.as_slice());
    let mut precomputed: Vec<Option<&[F128]>> = vec![pre_ab, pre_c];
    precomputed.extend(std::iter::repeat_n(
        None,
        core.points.len() + extra_points.len(),
    ));
    let lig_config = setup
        .keccak
        .pcs_params
        .ligerito_prover_config()
        .expect("Ligerito config");
    let padding = setup.keccak.r1cs.padding_spec();
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        fast.z_packed,
        &fast.prover_data,
        &fast.commitment,
        &x_refs,
        &precomputed,
        &[],
        &padding,
        &lig_config,
        challenger,
    );

    (
        SpongeProof {
            commitment: fast.commitment,
            zerocheck: fast.zc_proof,
            lincheck: fast.lc_proof,
            opening_values: core.opening_values,
            pcs_open,
        },
        fast.prover_data,
    )
}

/// Gather-and-fold bit-MLE evaluation of the packed witness at a point.
pub fn gather_eval(z_packed: &[F128], point: &[F128]) -> F128 {
    let (fixed, free) = split_boolean(point);
    gather_eval_split(z_packed, fixed, &free)
}

/// Evaluate MANY points of one packed witness, sharing work across
/// points whose FREE coordinates agree (the common case: claim
/// families differ only in Boolean offsets — chaining edges, plane
/// selectors, aligned linkage pieces). Per family this builds the eq
/// TENSOR over the free coordinates and the relative-address table
/// ONCE; each member point is then one bit-gated additive pass with no
/// per-point layer materialization and no per-point fold. Exactly
/// equal to [`gather_eval`] per point (the tensor dot IS the
/// multilinear fold, and field addition regroups exactly).
pub fn gather_eval_many(z_packed: &[F128], points: &[Vec<F128>]) -> Vec<F128> {
    use rayon::prelude::*;
    // Group point indices by their free-coordinate signature.
    let mut split: Vec<(usize, Vec<(usize, F128)>)> = Vec::with_capacity(points.len());
    for point in points {
        split.push(split_boolean(point));
    }
    let mut groups: Vec<(Vec<(usize, F128)>, Vec<usize>)> = Vec::new();
    for (index, (_, free)) in split.iter().enumerate() {
        if let Some((_, members)) = groups.iter_mut().find(|(sig, _)| sig == free) {
            members.push(index);
        } else {
            groups.push((free.clone(), vec![index]));
        }
    }
    let mut values = vec![F128::ZERO; points.len()];
    for (free, members) in &groups {
        let free_vars = free.len();
        let size = 1_usize << free_vars;
        // Eq tensor over the free coordinates, MSB-first (matching the
        // fold order of gather_eval_split: last free coordinate binds
        // the lowest tensor bit).
        let mut tensor = vec![F128::ONE];
        for &(_, coord) in free {
            let mut next = Vec::with_capacity(2 * tensor.len());
            for &value in &tensor {
                next.push(value * (F128::ONE + coord));
                next.push(value * coord);
            }
            tensor = next;
        }
        // Relative addresses: bit j of the tensor index (MSB-first over
        // the free list) maps to the free coordinate's address bit.
        let rel_addr: Vec<usize> = (0..size)
            .map(|index| {
                let mut address = 0_usize;
                for (j, &(bit, _)) in free.iter().enumerate() {
                    if (index >> (free_vars - 1 - j)) & 1 == 1 {
                        address |= 1 << bit;
                    }
                }
                address
            })
            .collect();
        let member_values: Vec<F128> = members
            .par_iter()
            .map(|&point_index| {
                let fixed = split[point_index].0;
                let mut sum = F128::ZERO;
                for (offset, &weight) in rel_addr.iter().zip(&tensor) {
                    if crate::chain::read_packed_bit(z_packed, fixed | offset) {
                        sum += weight;
                    }
                }
                sum
            })
            .collect();
        for (&point_index, value) in members.iter().zip(member_values) {
            values[point_index] = value;
        }
    }
    values
}

/// Split a point into its Boolean-fixed address offset and free coords
/// (with their address-bit positions), for the packed gather evaluation.
fn split_boolean(point: &[F128]) -> (usize, Vec<(usize, F128)>) {
    let vars = point.len();
    let mut fixed = 0_usize;
    let mut free = Vec::new();
    for (i, &coord) in point.iter().enumerate() {
        let bit = vars - 1 - i;
        if coord == F128::ZERO {
        } else if coord == F128::ONE {
            fixed |= 1 << bit;
        } else {
            free.push((bit, coord));
        }
    }
    (fixed, free)
}

/// Gather-and-fold bit-MLE evaluation over the PACKED witness.
fn gather_eval_split(z_packed: &[F128], fixed: usize, free: &[(usize, F128)]) -> F128 {
    let free_vars = free.len();
    let mut layer = vec![F128::ZERO; 1 << free_vars];
    for (index, slot) in layer.iter_mut().enumerate() {
        let mut address = fixed;
        for (j, &(bit, _)) in free.iter().enumerate() {
            if (index >> (free_vars - 1 - j)) & 1 == 1 {
                address |= 1 << bit;
            }
        }
        if crate::chain::read_packed_bit(z_packed, address) {
            *slot = F128::ONE;
        }
    }
    for &(_, coord) in free {
        let half = layer.len() / 2;
        for i in 0..half {
            let low = layer[i];
            layer[i] = low + coord * (low + layer[half + i]);
        }
        layer.truncate(half);
    }
    layer[0]
}

/// Verify the sponge lane against the public inputs.
/// The verifier-side counterpart of [`SpongeCore`]: the replayed base
/// claims and the lane's own opening points, ready for the batch verify.
pub struct SpongeVerifyCore {
    pub ab: flock_core::proof::ZClaim,
    pub c: flock_core::proof::ZClaim,
    pub points: Vec<Vec<F128>>,
}

pub fn verify_sponge<Ch: Challenger>(
    setup: &SpongeSetup,
    publics: &[SpongePublic],
    proof: &SpongeProof,
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    let core = verify_sponge_core(setup, publics, proof, challenger)?;
    verify_sponge_open(setup, proof, core, &[], &[], challenger)
}

/// Everything except the batched opening: base R1CS replay, the sponge
/// challenges, and the chaining/pinning checks over the claimed values.
pub fn verify_sponge_core<Ch: Challenger>(
    setup: &SpongeSetup,
    publics: &[SpongePublic],
    proof: &SpongeProof,
    challenger: &mut Ch,
) -> Result<SpongeVerifyCore, &'static str> {
    assert_eq!(publics.len(), setup.records);
    let record_vars = setup.record_vars();
    let (ab, c) = flock_core::verifier::verify_core(
        &setup.keccak.r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        &KeccakLincheckCircuit,
        challenger,
    )
    .map_err(|_| "keccak R1CS verification failed")?;

    challenger.observe_label(b"aerie-sponge-challenges-v0");
    let delta = challenger.sample_f128();
    let r = challenger.sample_f128_vec(11);

    if proof.opening_values.len() != 22 {
        return Err("wrong opening value count");
    }
    let values = &proof.opening_values;
    let scale = record_scale(record_vars, delta).ok_or("degenerate delta")?;
    let (xor_term, frame_term) = public_terms(publics, record_vars, delta, &r);

    // Interior edges: IN_e == OUT_(e-1) for e in 2..=9.
    for e in 2..LIVE_PERMS {
        if values[e] != values[10 + e - 1] {
            return Err("a sponge chain edge does not hold");
        }
    }
    // The absorption edge: IN_1 == OUT_0 + public second blocks. The
    // openings carry the derived-coordinate scale; the public term is a
    // raw delta-power sum, so it is divided by the scale.
    if scale * values[1] != scale * values[10] + xor_term {
        return Err("the absorption edge does not hold");
    }
    // Record starts: the full slot minus the scaled salt sub-cubes must
    // equal the public frame.
    let (s256, s64) = salt_scales(&r);
    if scale * (values[0] + s256 * values[20] + s64 * values[21]) != frame_term {
        return Err("a record-start state does not pin to its public frame");
    }

    let points = sponge_points(record_vars, delta, &r).ok_or("degenerate delta")?;
    Ok(SpongeVerifyCore { ab, c, points })
}

/// Verify the lane's single batched opening: `[ab, c]` quirky, the
/// chaining claims, and any cross-lane `extra_points`/`extra_values`
/// (in the same order the prover folded them in).
pub fn verify_sponge_open<Ch: Challenger>(
    setup: &SpongeSetup,
    proof: &SpongeProof,
    core: SpongeVerifyCore,
    extra_points: &[Vec<F128>],
    extra_values: &[F128],
    challenger: &mut Ch,
) -> Result<(), &'static str> {
    if extra_points.len() != extra_values.len() {
        return Err("extra claim points and values disagree");
    }
    let SpongeVerifyCore { ab, c, points } = core;
    let mut claim_values = vec![ab.value, c.value];
    claim_values.extend_from_slice(&proof.opening_values);
    claim_values.extend_from_slice(extra_values);
    let mut bindings = vec![
        LowBinding::Quirky {
            z_skip: ab.point.z_skip,
        },
        LowBinding::Quirky {
            z_skip: c.point.z_skip,
        },
    ];
    let mut x_fulls: Vec<Vec<F128>> = vec![
        {
            let mut v = ab.point.x_inner_rest.clone();
            v.extend_from_slice(&ab.point.x_outer);
            v
        },
        {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        },
    ];
    for point in points.iter().chain(extra_points) {
        let (x_low, x_outer) = flock_claim_shape(point);
        bindings.push(LowBinding::Multilinear { x_low });
        x_fulls.push(x_outer);
    }
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_config = setup
        .keccak
        .pcs_params
        .ligerito_verifier_config()
        .expect("verifier config");
    pcs::verify_opening_batch_ligerito_mixed_bound(
        &proof.commitment,
        &claim_values,
        &bindings,
        &x_refs,
        &[],
        &proof.pcs_open,
        &lig_config,
        challenger,
    )
    .map_err(|_| "batched opening verification failed")
}

#[cfg(test)]
mod tests {
    use flock_core::challenger::FsChallenger;

    use super::*;

    fn test_record(seed: u8) -> SpongeRecord {
        SpongeRecord {
            salt: [seed.wrapping_add(1); SALT_BYTES],
            hpk: [seed.wrapping_mul(3).wrapping_add(7); 64],
            message: (0..33).map(|i| seed.wrapping_add(i)).collect(),
        }
    }

    #[test]
    fn sponge_trace_matches_the_sha3_stream() {
        // The convention pin: the walker-state simulation must reproduce
        // the independent sha3 XOF words bit for bit, which fixes the
        // logical/physical state mapping and the SHAKE padding.
        use sha3::Shake256;
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        for seed in [0_u8, 5, 99] {
            let record = test_record(seed);
            let (_states, words) = sponge_trace(&record);
            assert_eq!(words.len(), 612);

            let mut hasher = Shake256::default();
            hasher.update(&record.salt);
            hasher.update(&record.hpk);
            hasher.update(&[0x00, 0x00]);
            hasher.update(&record.message);
            let mut reader = hasher.finalize_xof();
            for (index, &word) in words.iter().enumerate() {
                let mut bytes = [0_u8; 2];
                reader.read(&mut bytes);
                assert_eq!(word, u16::from_be_bytes(bytes), "word {index}, seed {seed}");
            }
        }
    }

    #[test]
    fn sponge_proof_roundtrips_and_rejects_wrong_publics() {
        let records = 8;
        let setup = SpongeSetup::new(records);
        let inputs: Vec<SpongeRecord> = (0..records as u8).map(test_record).collect();
        let publics: Vec<SpongePublic> = inputs
            .iter()
            .map(|record| SpongePublic {
                hpk: record.hpk,
                message: record.message.clone(),
            })
            .collect();

        let mut prover_challenger = FsChallenger::new(b"aerie-sponge-proof");
        let (proof, words, _artifacts) = prove_sponge(&setup, &inputs, &mut prover_challenger);
        assert_eq!(words.len(), records);

        let mut verifier_challenger = FsChallenger::new(b"aerie-sponge-proof");
        verify_sponge(&setup, &publics, &proof, &mut verifier_challenger)
            .expect("honest sponge verifies");

        // A different public message must break the frame or xor checks.
        let mut wrong_publics = publics.clone();
        wrong_publics[3].message[5] ^= 1;
        let mut fresh = FsChallenger::new(b"aerie-sponge-proof");
        assert!(verify_sponge(&setup, &wrong_publics, &proof, &mut fresh).is_err());

        // A different hpk likewise (it lives in the pinned frame).
        let mut wrong_publics = publics.clone();
        wrong_publics[0].hpk[10] ^= 1;
        let mut fresh = FsChallenger::new(b"aerie-sponge-proof");
        assert!(verify_sponge(&setup, &wrong_publics, &proof, &mut fresh).is_err());

        // A tampered edge opening rejects.
        let mut wrong = proof.clone();
        wrong.opening_values[4] += F128::ONE;
        let mut fresh = FsChallenger::new(b"aerie-sponge-proof");
        assert!(verify_sponge(&setup, &publics, &wrong, &mut fresh).is_err());
    }
}
