//! Compact, record-major layout. Every retained range is 64-bit aligned.
use super::super::{hash_to_point_slots as slots, keccak3};
use flock_core::field::F128;

pub const K_LOG: usize = 19;
pub const K: usize = 1 << K_LOG;
// Only 612 slots are live. Two 256-slot segments and one 128-slot segment
// retain every live row/column and reclaim the final 128 zero slots.
pub const TARGET: usize = 2 * 32768 + 16384;
pub const CONST: usize = TARGET + 8192;
// State slots are full aligned subcubes, including their constrained-zero
// padding. The base also aligns the first sixteen slots into a Boolean face.
pub const STATES: usize = 3 * 32768;
pub const STATE_STRIDE: usize = 2048;
pub const T_BASE: usize = STATES + 20 * STATE_STRIDE;
pub const END: usize = T_BASE + 10 * 24 * 1600;
const _: () = assert!(slots::SLOTS <= 640);
const _: () = assert!(CONST + 128 <= STATES);
const _: () = assert!(END <= K);

pub fn record_position(old: usize) -> Option<usize> {
    if old == slots::Z_CONST_POS {
        return Some(CONST);
    }
    let (plane, slot) = (old / 1024, old % 1024);
    if plane < 112 && slot < 640 {
        let (base, width, offset) = if slot < 512 {
            ((slot / 256) * 32768, 256, slot % 256)
        } else {
            (2 * 32768, 128, slot - 512)
        };
        Some(base + plane * width + offset)
    } else if (112..120).contains(&plane) {
        Some(TARGET + (plane - 112) * 1024 + slot)
    } else {
        None
    }
}

pub fn sponge_position(block: usize, old: usize) -> Option<usize> {
    if old == keccak3::Z_CONST {
        return Some(CONST);
    }
    if old < 6 * 2048 {
        let slot = old / 2048;
        let permutation = 4 * (slot / 2) + block;
        let bit = old % 2048;
        (permutation < 10 && bit < 1600)
            .then_some(STATES + (2 * permutation + slot % 2) * STATE_STRIDE + bit)
    } else if (keccak3::T_PACKED_BIT_BASE..keccak3::USEFUL_BITS).contains(&old) {
        let offset = old - keccak3::T_PACKED_BIT_BASE;
        let permutation = 4 * (offset / 38400) + block;
        (permutation < 10).then_some(T_BASE + permutation * 38400 + offset % 38400)
    } else {
        None
    }
}

/// Candidate words are big-endian; state bytes are little-endian lane bytes.
pub fn word_source(slot: usize, bit: usize) -> usize {
    assert!(slot < slots::SLOTS && bit < 16);
    STATES + (2 * (1 + slot / 68) + 1) * STATE_STRIDE + 16 * (slot % 68) + (bit ^ 8)
}

pub fn bit(words: &[F128], p: usize) -> bool {
    let w = words[p / 128];
    (if p % 128 < 64 { w.lo } else { w.hi }) >> (p % 64) & 1 != 0
}
pub fn set_bit(words: &mut [F128], p: usize, value: bool) {
    let w = &mut words[p / 128];
    let limb = if p % 128 < 64 { &mut w.lo } else { &mut w.hi };
    *limb = (*limb & !(1 << (p % 64))) | (u64::from(value) << (p % 64));
}
pub fn copy64(src: &[F128], from: usize, dst: &mut [F128], to: usize) {
    debug_assert_eq!(from % 64, 0);
    debug_assert_eq!(to % 64, 0);
    let w = src[from / 128];
    let v = if from % 128 == 0 { w.lo } else { w.hi };
    let w = &mut dst[to / 128];
    if to % 128 == 0 {
        w.lo = v;
    } else {
        w.hi = v;
    }
}

#[derive(Clone)]
pub struct Fragment {
    pub point: Vec<F128>,
    pub weight: F128,
}

/// A source subcube with a coordinate permutation into the compact layout.
fn cube(
    out: &mut Vec<Fragment>,
    point: &[F128],
    old_vars: usize,
    old_base: usize,
    new_base: usize,
    axes: &[(usize, usize)],
) {
    let records = point.len() - old_vars;
    let mut dest: Vec<_> = point[..records]
        .iter()
        .copied()
        .chain((0..K_LOG).rev().map(|i| {
            if new_base >> i & 1 == 0 {
                F128::ZERO
            } else {
                F128::ONE
            }
        }))
        .collect();
    let mut free = 0usize;
    for &(old, new) in axes {
        free |= 1 << old;
        dest[records + K_LOG - 1 - new] = point[records + old_vars - 1 - old];
    }
    let mut weight = F128::ONE;
    for i in 0..old_vars {
        if free >> i & 1 == 0 {
            let r = point[records + old_vars - 1 - i];
            weight *= if old_base >> i & 1 == 0 {
                F128::ONE + r
            } else {
                r
            };
        }
    }
    if weight != F128::ZERO {
        out.push(Fragment {
            point: dest,
            weight,
        });
    }
}

/// Exactly reconstruct the old record MLE, including a possibly nonzero
/// fifteenth target bit and the shared constant, from compact subcubes.
pub fn record_fragments(point: &[F128]) -> Vec<Fragment> {
    let mut out = Vec::new();
    for segment in 0..3 {
        let slot_log = if segment < 2 { 8 } else { 7 };
        for (plane, log) in [(0, 6), (64, 5), (96, 4)] {
            let axes: Vec<_> = (0..slot_log)
                .map(|i| (i, i))
                .chain((0..log).map(|i| (10 + i, slot_log + i)))
                .collect();
            cube(
                &mut out,
                point,
                17,
                (plane << 10) + (segment << 8),
                (segment << 15) + (plane << slot_log),
                &axes,
            );
        }
    }
    cube(
        &mut out,
        point,
        17,
        112 << 10,
        TARGET,
        &(0..13).map(|i| (i, i)).collect::<Vec<_>>(),
    );
    // The compact cell zero is constrained to zero; the old constant is
    // supplied separately at CONST, so the first cube needs no subtraction.
    cube(&mut out, point, 17, 0, CONST, &[]);
    out
}

/// Only state subcubes are queried by the SHAKE framing/chain relation.
/// Each aligned state includes the 448 zero-padding cells. These have empty
/// A/B rows in both layouts, so C = I constrains them to zero. Opening the
/// whole slot reconstructs the original MLE without splitting its live bits.
pub fn sponge_fragments(point: &[F128]) -> Vec<Fragment> {
    let mut out = Vec::new();
    for permutation in 0..10 {
        for output in 0..2 {
            let old = (permutation % 4) * keccak3::K + (2 * (permutation / 4) + output) * 2048;
            let new = STATES + (2 * permutation + output) * STATE_STRIDE;
            cube(
                &mut out,
                point,
                19,
                old,
                new,
                &(0..11).map(|i| (i, i)).collect::<Vec<_>>(),
            );
        }
    }
    out
}
