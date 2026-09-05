//! Byte-field reduction for the ARM first-round kernel.
//!
//! Both tables contain polynomial remainders modulo 0x11b. For every
//! 16-bit polynomial p, reduce(p) equals its low byte XOR the two table
//! entries selected by its high-byte nibbles. No field encoding changes.
use core::arch::aarch64::*;

const fn reduction_table(shift: u32) -> [u8; 16] {
    let mut out = [0; 16];
    let mut i = 0;
    while i < 16 {
        let mut p = (i as u16) << shift;
        let mut bit = 15;
        while bit >= 8 {
            if p & (1 << bit) != 0 {
                p ^= 0x11b << (bit - 8);
            }
            bit -= 1;
        }
        out[i] = p as u8;
        i += 1;
    }
    out
}

const R8: [u8; 16] = reduction_table(8);
const R12: [u8; 16] = reduction_table(12);

// The remainder map is GF(2)-linear. Split the high byte into its two
// nibbles, look up their remainders, and XOR with the untouched low byte.
/// Reduce 16 interleaved 16-bit polynomials to their byte-field values.
///
/// # Safety
/// Requires aarch64 NEON, which is guaranteed by the parent module's cfg.
#[inline(always)]
pub unsafe fn gf8_reduce_vec16(c0: uint8x16_t, c1: uint8x16_t) -> uint8x16_t {
    unsafe {
        let lo = vuzp1q_u8(c0, c1);
        let hi = vuzp2q_u8(c0, c1);
        let r0 = vqtbl1q_u8(vld1q_u8(R8.as_ptr()), vandq_u8(hi, vdupq_n_u8(15)));
        let r1 = vqtbl1q_u8(vld1q_u8(R12.as_ptr()), vshrq_n_u8::<4>(hi));
        veorq_u8(lo, veorq_u8(r0, r1))
    }
}

#[inline(always)]
/// Multiply 16 byte-field pairs without changing their representation.
///
/// # Safety
/// Requires aarch64 NEON, including its baseline byte polynomial multiply.
pub unsafe fn gf8_mul_vec16(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    unsafe {
        let lo = vreinterpretq_u8_p16(vmull_p8(
            vreinterpret_p8_u8(vget_low_u8(a)),
            vreinterpret_p8_u8(vget_low_u8(b)),
        ));
        let hi = vreinterpretq_u8_p16(vmull_p8(
            vreinterpret_p8_u8(vget_high_u8(a)),
            vreinterpret_p8_u8(vget_high_u8(b)),
        ));
        gf8_reduce_vec16(lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial_remainder(mut p: u16) -> u8 {
        for bit in (8..16).rev() {
            if p & (1 << bit) != 0 {
                p ^= 0x11b << (bit - 8);
            }
        }
        p as u8
    }

    #[test]
    fn nibble_round1_reducer_matches_every_u16_polynomial() {
        for start in (0..65536u32).step_by(16) {
            let mut input = [0u16; 16];
            for (i, p) in input.iter_mut().enumerate() {
                *p = (start + i as u32) as u16;
            }
            let mut out = [0u8; 16];
            unsafe {
                let lo = vld1q_u8(input.as_ptr().cast());
                let hi = vld1q_u8(input.as_ptr().add(8).cast());
                vst1q_u8(out.as_mut_ptr(), gf8_reduce_vec16(lo, hi));
            }
            for i in 0..16 {
                assert_eq!(out[i], polynomial_remainder(input[i]), "p={}", input[i]);
            }
        }
    }

    #[test]
    fn nibble_round1_multiplier_matches_every_byte_pair() {
        for a in 0..256u16 {
            for b in (0..256u16).step_by(16) {
                let aa = [a as u8; 16];
                let mut bb = [0u8; 16];
                for (i, x) in bb.iter_mut().enumerate() {
                    *x = (b + i as u16) as u8;
                }
                let mut out = [0u8; 16];
                unsafe {
                    vst1q_u8(
                        out.as_mut_ptr(),
                        gf8_mul_vec16(vld1q_u8(aa.as_ptr()), vld1q_u8(bb.as_ptr())),
                    );
                }
                for i in 0..16 {
                    let mut product = 0u16;
                    for bit in 0..8 {
                        if bb[i] & (1 << bit) != 0 {
                            product ^= a << bit;
                        }
                    }
                    assert_eq!(out[i], polynomial_remainder(product));
                }
            }
        }
    }
}
