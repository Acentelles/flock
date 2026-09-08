//! Compact, record-major layout. Every retained range is 64-bit aligned.
use super::super::{hash_to_point_slots as slots, keccak3};
use flock_core::field::F128;

pub const K_LOG: usize = 19;
pub const K: usize = 1 << K_LOG;
pub const ALIGNED_STATES: bool = cfg!(feature = "aligned-hybrid");
pub const PROFILE: &str = if ALIGNED_STATES {
    "aligned-state-v2"
} else {
    "compact-state-v1"
};
pub const DESCRIPTOR_DOMAIN: &[u8] = if ALIGNED_STATES {
    b"aerie/hybrid-keccak-f1600-24-slot-v2/256-256-128-slot-segments/aligned-state-subcubes/copy-big-endian-words"
} else {
    b"aerie/hybrid-keccak-f1600-24-slot-v1/three-256-slot-segments/trim-state-padding/copy-big-endian-words"
};
pub const TRANSCRIPT_DOMAIN: &[u8] = if ALIGNED_STATES {
    b"aerie-hybrid-compact-circuit-v2"
} else {
    b"aerie-hybrid-compact-circuit-v1"
};
pub const TARGET: usize = if ALIGNED_STATES { 81920 } else { 3 * 32768 };
pub const CONST: usize = TARGET + 8192;
pub const KECCAK: usize = CONST + 128;
pub const PERM: usize = 2 * 1600 + 24 * 1600;
pub const STATES: usize = 3 * 32768;
pub const STATE_STRIDE: usize = 2048;
pub const T_BASE: usize = STATES + 20 * STATE_STRIDE;
pub const END: usize = if ALIGNED_STATES {
    T_BASE + 10 * 24 * 1600
} else {
    KECCAK + 10 * PERM
};
const _: () = assert!(!ALIGNED_STATES || slots::SLOTS <= 640);
const _: () = assert!(!ALIGNED_STATES || CONST + 128 <= STATES);
const _: () = assert!(END <= K);

#[inline]
pub fn state_position(permutation: usize, output: usize) -> usize {
    if ALIGNED_STATES {
        STATES + (2 * permutation + output) * STATE_STRIDE
    } else {
        KECCAK + permutation * PERM + output * 1600
    }
}

#[inline]
pub fn product_position(permutation: usize) -> usize {
    if ALIGNED_STATES {
        T_BASE + permutation * 38400
    } else {
        KECCAK + permutation * PERM + 3200
    }
}

pub fn record_position(old: usize) -> Option<usize> {
    if old == slots::Z_CONST_POS {
        return Some(CONST);
    }
    let (plane, slot) = (old / 1024, old % 1024);
    let retained_slots = if ALIGNED_STATES { 640 } else { 768 };
    if plane < 112 && slot < retained_slots {
        let (base, width, offset) = if ALIGNED_STATES && slot >= 512 {
            (2 * 32768, 128, slot - 512)
        } else {
            ((slot / 256) * 32768, 256, slot % 256)
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
        (permutation < 10 && bit < 1600).then_some(state_position(permutation, slot % 2) + bit)
    } else if (keccak3::T_PACKED_BIT_BASE..keccak3::USEFUL_BITS).contains(&old) {
        let offset = old - keccak3::T_PACKED_BIT_BASE;
        let permutation = 4 * (offset / 38400) + block;
        (permutation < 10).then_some(product_position(permutation) + offset % 38400)
    } else {
        None
    }
}

/// Candidate words are big-endian; state bytes are little-endian lane bytes.
pub fn word_source(slot: usize, bit: usize) -> usize {
    assert!(slot < slots::SLOTS && bit < 16);
    state_position(1 + slot / 68, 1) + 16 * (slot % 68) + (bit ^ 8)
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
        let slot_log = if ALIGNED_STATES && segment == 2 { 7 } else { 8 };
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
/// In the aligned profile the original constrained-zero state padding is
/// retained, so each full 2048-bit slot is opened as one subcube.
pub fn sponge_fragments(point: &[F128]) -> Vec<Fragment> {
    let mut out = Vec::new();
    for permutation in 0..10 {
        for output in 0..2 {
            let old = (permutation % 4) * keccak3::K + (2 * (permutation / 4) + output) * 2048;
            let new = state_position(permutation, output);
            if ALIGNED_STATES {
                cube(
                    &mut out,
                    point,
                    19,
                    old,
                    new,
                    &(0..11).map(|i| (i, i)).collect::<Vec<_>>(),
                );
                continue;
            }
            let mut offset = 0usize;
            while offset < 1600 {
                let remaining = 1600usize - offset;
                let mut log = (usize::BITS - 1 - remaining.leading_zeros()) as usize;
                while (old + offset) % (1 << log) != 0 || (new + offset) % (1 << log) != 0 {
                    log -= 1;
                }
                cube(
                    &mut out,
                    point,
                    19,
                    old + offset,
                    new + offset,
                    &(0..log).map(|i| (i, i)).collect::<Vec<_>>(),
                );
                offset += 1 << log;
            }
        }
    }
    out
}
