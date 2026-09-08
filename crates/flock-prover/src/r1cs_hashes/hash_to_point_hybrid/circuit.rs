use super::super::{hash_to_point_slots as slots, hash_to_point_sponge as sponge, keccak3};
use super::layout::*;
use flock_core::{field::F128, lincheck::LincheckCircuit, pcs::PcsParams, r1cs::BlockR1cs};
use rayon::prelude::*;

/// The matrices in `r1cs` are storage stubs for a fixed composite walker.
/// The transcript additionally binds the canonical composition descriptor,
/// the complete slot circuit, both address maps and every word-copy edge.
pub struct Setup {
    pub r1cs: BlockR1cs,
    pub params: PcsParams,
    pub slots: slots::SlotSetup,
    pub descriptor: [u8; 32],
}
impl Setup {
    pub fn new(records: usize) -> Self {
        assert!(records.is_power_of_two() && records >= 8);
        let slots = slots::SlotSetup::new(records);
        let r1cs = super::super::common::build_block_r1cs_empty_stub(
            records.trailing_zeros() as usize,
            K_LOG,
            6,
            END,
        );
        let params = PcsParams {
            m: r1cs.m,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: Default::default(),
            merkle_hash: Default::default(),
        };
        let mut h = blake3::Hasher::new();
        h.update(b"aerie/hybrid-keccak-f1600-24-slot-v1/three-256-slot-segments/trim-state-padding/copy-big-endian-words");
        h.update(&r1cs.statement_digest());
        h.update(&slots.r1cs.statement_digest());
        for old in 0..slots::K {
            h.update(&(record_position(old).unwrap_or(usize::MAX) as u64).to_le_bytes());
        }
        for block in 0..4 {
            for old in 0..keccak3::K {
                h.update(&(sponge_position(block, old).unwrap_or(usize::MAX) as u64).to_le_bytes());
            }
        }
        for slot in 0..slots::SLOTS {
            for bit in 0..16 {
                h.update(&(word_source(slot, bit) as u64).to_le_bytes());
            }
        }
        let descriptor = *h.finalize().as_bytes();
        Self {
            r1cs,
            params,
            slots,
            descriptor,
        }
    }
    pub fn record_vars(&self) -> usize {
        self.r1cs.m - K_LOG
    }
    pub fn bind<Ch: flock_core::challenger::Challenger>(&self, ch: &mut Ch) {
        ch.observe_label(b"aerie-hybrid-compact-circuit-v1");
        ch.observe_bytes(&self.descriptor);
    }
}
impl LincheckCircuit for Setup {
    fn n_cols(&self) -> usize {
        K
    }
    fn const_pin_col(&self) -> Option<usize> {
        Some(CONST)
    }
    fn fold_alpha_batched(&self, alpha: F128, weights: &[F128]) -> Vec<F128> {
        assert_eq!(weights.len(), K);
        let mut result = vec![F128::ZERO; K];
        result[CONST] = (alpha + F128::ONE) * weights[CONST];
        for block in 0..4 {
            let rows: Vec<_> = (0..keccak3::K)
                .map(|old| {
                    if old == keccak3::Z_CONST {
                        F128::ZERO
                    } else {
                        sponge_position(block, old).map_or(F128::ZERO, |p| weights[p])
                    }
                })
                .collect();
            for (old, value) in keccak3::KeccakLincheckCircuit
                .fold_alpha_batched(alpha, &rows)
                .into_iter()
                .enumerate()
            {
                if let Some(p) = sponge_position(block, old) {
                    result[p] += value;
                } else {
                    assert_eq!(value, F128::ZERO, "unmapped Keccak column {old}");
                }
            }
        }
        let rows: Vec<_> = (0..slots::K)
            .map(|old| {
                if old == slots::Z_CONST_POS {
                    F128::ZERO
                } else {
                    record_position(old).map_or(F128::ZERO, |p| weights[p])
                }
            })
            .collect();
        for (old, value) in self
            .slots
            .r1cs
            .sparse_lincheck_circuit()
            .fold_alpha_batched(alpha, &rows)
            .into_iter()
            .enumerate()
        {
            if let Some(p) = record_position(old) {
                result[p] += value;
            } else {
                assert_eq!(value, F128::ZERO, "unmapped slot column {old}");
            }
        }
        // Replace each vacuous word-input row with z_SHAKE * 1 = z_word.
        for slot in 0..slots::SLOTS {
            for bit in 0..16 {
                let row = record_position(slots::word_position(slot, bit)).unwrap();
                let weight = alpha * weights[row];
                result[row] += weight;
                result[word_source(slot, bit)] += weight;
            }
        }
        result
    }
}

pub struct Witness {
    pub z: Vec<F128>,
    pub a: Vec<F128>,
    pub b: Vec<F128>,
}

/// Move original packed witnesses into one compact domain. A/B are actual
/// gate products, including the new cross-circuit word-copy rows.
pub fn assemble(setup: &Setup, sponge: sponge::SpongeWitness, record: Vec<F128>) -> Witness {
    let _span = tracing::info_span!("hybrid.assemble").entered();
    let records = 1 << setup.record_vars();
    assert_eq!(sponge.z_packed.len(), records * K / 128);
    assert_eq!(record.len(), records * slots::K / 128);
    let apply_span = tracing::info_span!("hybrid.record_apply_ab").entered();
    let record_a = setup.slots.r1cs.apply_a_packed(&record);
    let record_b = setup.slots.r1cs.apply_b_packed(&record);
    drop(apply_span);
    let _span = tracing::info_span!("hybrid.relocate_witness").entered();
    let mut z = vec![F128::ZERO; records * K / 128];
    let mut a = z.clone();
    let mut b = z.clone();
    z.par_chunks_mut(K / 128)
        .zip(a.par_chunks_mut(K / 128))
        .zip(b.par_chunks_mut(K / 128))
        .enumerate()
        .for_each(|(rec, ((z, a), b))| {
            for block in 0..4 {
                for old in (0..keccak3::K).step_by(64) {
                    if let Some(new) = sponge_position(block, old) {
                        let src = rec * K + block * keccak3::K + old;
                        copy64(&sponge.z_packed, src, z, new);
                        copy64(&sponge.a_packed, src, a, new);
                        copy64(&sponge.b_packed, src, b, new);
                    }
                }
            }
            for old in (0..slots::K).step_by(64) {
                if let Some(new) = record_position(old) {
                    let src = rec * slots::K + old;
                    copy64(&record, src, z, new);
                    copy64(&record_a, src, a, new);
                    copy64(&record_b, src, b, new);
                }
            }
            for slot in 0..slots::SLOTS {
                for bit_index in 0..16 {
                    set_bit(
                        a,
                        record_position(slots::word_position(slot, bit_index)).unwrap(),
                        bit(z, word_source(slot, bit_index)),
                    );
                }
            }
        });
    Witness { z, a, b }
}
