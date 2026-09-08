//! Lifetime of packed A/B storage across the zerocheck round-2 boundary.

use crate::field::F128;

#[cfg(test)]
mod tests;

pub(super) enum PackedInputs<'a> {
    Borrowed(&'a [u8], &'a [u8]),
    Owned(Vec<F128>, Vec<F128>),
}

impl PackedInputs<'_> {
    pub(super) fn bytes(&self) -> (&[u8], &[u8]) {
        match self {
            Self::Borrowed(a, b) => (a, b),
            Self::Owned(a, b) => {
                // SAFETY: F128 consists only of two u64 fields; the initialized
                // Vec storage can be read as bytes for the duration of this
                // borrow. This is the same native-byte view used by the R1CS
                // prover's existing borrowed zerocheck path.
                unsafe {
                    (
                        std::slice::from_raw_parts(
                            a.as_ptr().cast::<u8>(),
                            core::mem::size_of_val(a.as_slice()),
                        ),
                        std::slice::from_raw_parts(
                            b.as_ptr().cast::<u8>(),
                            core::mem::size_of_val(b.as_slice()),
                        ),
                    )
                }
            }
        }
    }

    /// Called only after the last round-2 read of the packed inputs.
    /// At K_SKIP=6 their F128 length equals the first tail output length.
    pub(super) fn into_tail_scratch(self, n_in: usize) -> (Vec<F128>, Vec<F128>) {
        match self {
            Self::Borrowed(_, _) => {
                if n_in >= 1024 {
                    (
                        crate::scratch::take_f128(n_in / 2),
                        crate::scratch::take_f128(n_in / 2),
                    )
                } else {
                    (Vec::new(), Vec::new())
                }
            }
            Self::Owned(a, b) => {
                assert_eq!(a.len(), n_in / 2, "packed A/tail output shape");
                assert_eq!(b.len(), n_in / 2, "packed B/tail output shape");
                if n_in >= 1024 {
                    // Both outputs are overwritten by the same first fused
                    // fold before reading. No clear, copy or allocation.
                    (a, b)
                } else {
                    // Small instances never enter the fused ping-pong path.
                    // Return these dead inputs once, within the proving call.
                    crate::scratch::give_f128(a);
                    crate::scratch::give_f128(b);
                    (Vec::new(), Vec::new())
                }
            }
        }
    }
}
