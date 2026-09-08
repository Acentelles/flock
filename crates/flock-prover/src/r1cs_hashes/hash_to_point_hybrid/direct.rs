//! Direct generation of the existing compact witness, without an intermediate
//! Keccak3 domain or a second simulation to discover permutation input states.
use super::super::{hash_to_point_slots as slots, hash_to_point_sponge as sponge, keccak};
use super::circuit::{Setup, Witness, copy_word_rows};
use super::layout::{CONST, K, copy64, product_position, record_position, state_position};
use flock_core::{field::F128, r1cs::word_apply::Program};
use rayon::prelude::*;

/// The compact sponge rows and candidate words, before record completion.
/// The private witness field prevents confusing this layout with SpongeWitness.
pub struct CompactSpongeWitness {
    pub all_words: Vec<Vec<u16>>,
    witness: Witness,
}

#[inline]
fn write64(words: &mut [F128], bit: usize, value: u64) {
    debug_assert_eq!(bit % 64, 0);
    let word = &mut words[bit / 128];
    if bit.is_multiple_of(128) {
        word.lo = value;
    } else {
        word.hi = value;
    }
}

/// Write one complete permutation's original state and AND rows directly at
/// its compact addresses. The final lanes feed the next sponge permutation.
fn fill_permutation(
    lanes: &mut keccak::Lanes,
    permutation: usize,
    z: &mut [F128],
    a: &mut [F128],
    b: &mut [F128],
) {
    let input_base = state_position(permutation, 0);
    let output_base = state_position(permutation, 1);
    let product_base = product_position(permutation);
    for (lane, &value) in lanes.iter().enumerate() {
        write64(z, input_base + lane * 64, value);
        write64(a, input_base + lane * 64, value);
        write64(b, input_base + lane * 64, u64::MAX);
    }
    for round in 0..24 {
        let mut phi = *lanes;
        keccak::theta_lanes(&mut phi);
        let phi = keccak::rho_pi_lanes(&phi);
        for y in 0..5 {
            for x in 0..5 {
                let lane = x + 5 * y;
                let left = !phi[(x + 1) % 5 + 5 * y];
                let right = phi[(x + 2) % 5 + 5 * y];
                let product = left & right;
                let bit = product_base + round * 1600 + lane * 64;
                write64(z, bit, product);
                write64(a, bit, left);
                write64(b, bit, right);
                lanes[lane] = phi[lane] ^ product;
            }
        }
        keccak::iota_lanes(lanes, round);
    }
    for (lane, &value) in lanes.iter().enumerate() {
        write64(z, output_base + lane * 64, value);
        write64(a, output_base + lane * 64, value);
        write64(b, output_base + lane * 64, u64::MAX);
    }
}

/// Generate all ten live SHAKE permutations once, directly in the unchanged
/// hybrid layout. Candidate words come from the same committed output lanes.
/// This function performs no commitment or transcript operation.
pub fn compact_sponge_witness(
    setup: &Setup,
    records: &[sponge::SpongeRecord],
) -> CompactSpongeWitness {
    let _span = tracing::info_span!("hybrid.direct_sponge_witness").entered();
    assert_eq!(records.len(), 1 << setup.record_vars());
    let packed_len = records.len() * K / 128;
    let mut z = vec![F128::ZERO; packed_len];
    let mut a = vec![F128::ZERO; packed_len];
    let mut b = vec![F128::ZERO; packed_len];
    let all_words = z
        .par_chunks_mut(K / 128)
        .zip(a.par_chunks_mut(K / 128))
        .zip(b.par_chunks_mut(K / 128))
        .zip(records.par_iter())
        .map(|(((z, a), b), record)| {
            let (first, second) = sponge::framed_blocks(record);
            let mut lanes = [0u64; 25];
            for (lane, bytes) in lanes.iter_mut().zip(first.chunks_exact(8)) {
                *lane = u64::from_le_bytes(bytes.try_into().unwrap());
            }
            write64(z, CONST, 1);
            write64(a, CONST, 1);
            write64(b, CONST, 1);
            let mut words = Vec::with_capacity(slots::SLOTS);
            for permutation in 0..sponge::LIVE_PERMS {
                if permutation == 1 {
                    for (lane, bytes) in lanes.iter_mut().zip(second.chunks_exact(8)) {
                        *lane ^= u64::from_le_bytes(bytes.try_into().unwrap());
                    }
                }
                fill_permutation(&mut lanes, permutation, z, a, b);
                if permutation >= 1 {
                    for &lane in &lanes[..sponge::RATE_BYTES / 8] {
                        for bytes in lane.to_le_bytes().chunks_exact(2) {
                            words.push(u16::from_be_bytes([bytes[0], bytes[1]]));
                        }
                    }
                }
            }
            debug_assert_eq!(words.len(), slots::SLOTS);
            words
        })
        .collect();
    CompactSpongeWitness {
        all_words,
        witness: Witness { z, a, b },
    }
}

/// Complete the record rows in already compact sponge buffers. The arbitrary
/// supplied record bits and matrices are evaluated exactly as by `assemble`;
/// cross-circuit copy rows always read the actual sponge, not record inputs.
pub fn assemble_direct(setup: &Setup, sponge: CompactSpongeWitness, record: Vec<F128>) -> Witness {
    let _span = tracing::info_span!("hybrid.direct_assembly").entered();
    let CompactSpongeWitness {
        witness: Witness {
            mut z,
            mut a,
            mut b,
        },
        all_words,
    } = sponge;
    let records = 1 << setup.record_vars();
    assert_eq!(z.len(), records * K / 128);
    assert_eq!(record.len(), records * slots::K / 128);
    drop(all_words);
    let program_span = tracing::info_span!("hybrid.record_programs").entered();
    let programs = Program::new(&setup.slots.r1cs.a_0).zip(Program::new(&setup.slots.r1cs.b_0));
    let fallback = programs.is_none().then(|| {
        (
            setup.slots.r1cs.apply_a_packed(&record),
            setup.slots.r1cs.apply_b_packed(&record),
        )
    });
    drop(program_span);
    let _span = tracing::info_span!("hybrid.direct_record_assembly").entered();
    z.par_chunks_mut(K / 128)
        .zip(a.par_chunks_mut(K / 128))
        .zip(b.par_chunks_mut(K / 128))
        .enumerate()
        .with_min_len(8)
        .for_each_init(
            || {
                (
                    vec![F128::ZERO; slots::K / 128],
                    vec![F128::ZERO; slots::K / 128],
                )
            },
            |(local_a, local_b), (rec, ((z, a), b))| {
                let start = rec * slots::K / 128;
                let record = &record[start..start + slots::K / 128];
                let (record_a, record_b) = if let Some((pa, pb)) = &programs {
                    pa.apply_block(record, local_a);
                    pb.apply_block(record, local_b);
                    (local_a.as_slice(), local_b.as_slice())
                } else {
                    let (a, b) = fallback.as_ref().unwrap();
                    (
                        &a[start..start + slots::K / 128],
                        &b[start..start + slots::K / 128],
                    )
                };
                for old in (0..slots::K).step_by(64) {
                    if let Some(new) = record_position(old) {
                        copy64(record, old, z, new);
                        copy64(record_a, old, a, new);
                        copy64(record_b, old, b, new);
                    }
                }
                copy_word_rows(z, a);
            },
        );
    Witness { z, a, b }
}
