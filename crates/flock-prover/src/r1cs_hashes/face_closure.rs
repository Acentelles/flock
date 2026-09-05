//! Reduce complete Boolean faces of MLE claims before ring switching.
//!
//! All original points and values are transcript-bound before challenges.
//! Within each family of identical non-Boolean coordinates, disjoint complete
//! Boolean faces are paired deterministically. Multilinearity gives
//! `Z(rho, r) = sum_b eq(rho, b) Z(b, r)` without a witness-wide sumcheck.
//! Sparse families are partitioned into smaller faces; no missing corner is
//! invented. The result is authenticated by the caller's ordinary PCS opening.
//!
//! For a face with a nonzero vector of claim errors, the combined error is a
//! nonzero multilinear polynomial of degree at most the face dimension. Its
//! vanishing probability is at most that dimension / |F|. A union bound over
//! faces and the PCS binding error applies. This is a protocol change, not a
//! byte-identical arithmetic optimization.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct ClosedClaim {
    pub point: Vec<F128>,
    pub value: F128,
}

#[derive(Default)]
struct Family {
    signature: Vec<(usize, F128)>,
    // Cube key: (varying coordinates, Boolean offset). Values are the
    // original claim indices, including every corner exactly once.
    cubes: BTreeMap<(u64, u64), Vec<usize>>,
}

/// Shared prover/verifier reduction. Work and memory depend on claim count
/// and arity, never the committed domain. Input order determines family order;
/// cube order is deterministic. Duplicate points must have identical values.
pub fn close_faces<Ch: Challenger>(
    points: &[Vec<F128>],
    values: &[F128],
    challenger: &mut Ch,
) -> Result<Vec<ClosedClaim>, &'static str> {
    if points.len() != values.len() {
        return Err("face claim count mismatch");
    }
    let vars = points.first().map_or(0, Vec::len);
    if vars > 63 || points.iter().any(|p| p.len() != vars) {
        return Err("invalid face claim arity");
    }
    let mut families: Vec<Family> = Vec::new();
    for (index, point) in points.iter().enumerate() {
        let mut signature = Vec::new();
        let mut offset = 0_u64;
        for (axis, &coord) in point.iter().enumerate() {
            if coord == F128::ONE {
                offset |= 1 << axis;
            } else if coord != F128::ZERO {
                signature.push((axis, coord));
            }
        }
        let family_index = if let Some(i) = families.iter().position(|f| f.signature == signature) {
            i
        } else {
            families.push(Family {
                signature,
                ..Family::default()
            });
            families.len() - 1
        };
        let family = &mut families[family_index];
        if let Some(previous) = family.cubes.get(&(0, offset)) {
            if values[previous[0]] != values[index] {
                return Err("conflicting duplicate face claim");
            }
        } else {
            family.cubes.insert((0, offset), vec![index]);
        }
    }

    challenger.observe_label(b"flock-complete-face-closure-v1");
    challenger.observe_bytes(&(vars as u64).to_le_bytes());
    challenger.observe_bytes(&(points.len() as u64).to_le_bytes());
    for (point, &value) in points.iter().zip(values) {
        challenger.observe_f128_slice(point);
        challenger.observe_f128(value);
    }

    let mut result = Vec::new();
    for mut family in families {
        // Repeat until no further equal-shaped faces can be paired. This is
        // a deterministic partition, not a claim of minimal face cover.
        loop {
            let before = family.cubes.len();
            for axis in 0..vars {
                let bit = 1_u64 << axis;
                let keys: Vec<_> = family.cubes.keys().copied().collect();
                for (mask, offset) in keys {
                    if (mask | offset) & bit != 0 {
                        continue;
                    }
                    let other = (mask, offset | bit);
                    if !family.cubes.contains_key(&(mask, offset))
                        || !family.cubes.contains_key(&other)
                    {
                        continue;
                    }
                    let mut vertices = family.cubes.remove(&(mask, offset)).unwrap();
                    vertices.extend(family.cubes.remove(&other).unwrap());
                    let old = family.cubes.insert((mask | bit, offset), vertices);
                    debug_assert!(old.is_none());
                }
            }
            if family.cubes.len() == before {
                break;
            }
        }
        for ((mask, _), vertices) in family.cubes {
            let mut point = points[vertices[0]].clone();
            let axes: Vec<_> = (0..vars).filter(|axis| mask & (1 << axis) != 0).collect();
            debug_assert_eq!(vertices.len(), 1 << axes.len());
            for &axis in &axes {
                point[axis] = challenger.sample_f128();
            }
            let value = vertices.into_iter().fold(F128::ZERO, |sum, index| {
                let weight = axes.iter().fold(F128::ONE, |weight, &axis| {
                    weight * (point[axis] + F128::ONE + points[index][axis])
                });
                sum + weight * values[index]
            });
            result.push(ClosedClaim { point, value });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    fn evaluate(bits: &[bool], point: &[F128]) -> F128 {
        let mut table: Vec<_> = bits
            .iter()
            .map(|&b| if b { F128::ONE } else { F128::ZERO })
            .collect();
        for &r in point {
            let half = table.len() / 2;
            for i in 0..half {
                table[i] = table[i] + r * (table[i] + table[i + half]);
            }
            table.truncate(half);
        }
        table[0]
    }

    fn claims(corners: &[usize]) -> (Vec<bool>, Vec<Vec<F128>>, Vec<F128>) {
        let bits: Vec<_> = (0..128).map(|i| (i * 17 + (i >> 1) * 9) % 11 < 5).collect();
        let points: Vec<Vec<F128>> = corners
            .iter()
            .map(|&corner| {
                (0..7)
                    .map(|axis| {
                        if [0, 2, 4].contains(&axis) {
                            let bit = [0, 2, 4].iter().position(|&x| x == axis).unwrap();
                            if corner & (1 << bit) != 0 {
                                F128::ONE
                            } else {
                                F128::ZERO
                            }
                        } else {
                            F128 {
                                lo: 71 + axis as u64,
                                hi: 5,
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let values = points.iter().map(|p| evaluate(&bits, p)).collect();
        (bits, points, values)
    }

    #[test]
    fn complete_and_incomplete_faces_match_direct_evaluation() {
        for (corners, expected) in [
            (vec![0, 1, 2, 3, 4, 5, 6, 7], 1),
            (vec![0, 1, 2], 2),
            (vec![0, 0, 1, 4], 2),
        ] {
            let (bits, points, values) = claims(&corners);
            let mut prover = FsChallenger::new(b"face-test");
            let closed = close_faces(&points, &values, &mut prover).unwrap();
            assert_eq!(closed.len(), expected);
            for claim in &closed {
                assert_eq!(claim.value, evaluate(&bits, &claim.point));
            }
            let mut verifier = FsChallenger::new(b"face-test");
            let replay = close_faces(&points, &values, &mut verifier).unwrap();
            for (a, b) in closed.iter().zip(replay) {
                assert_eq!(a.point, b.point);
                assert_eq!(a.value, b.value);
            }
            assert_eq!(prover.sample_f128(), verifier.sample_f128());
        }
    }

    #[test]
    fn binds_all_values_and_rejects_conflicting_duplicates() {
        let (bits, points, mut values) = claims(&[0, 1, 2, 3]);
        values[3] += F128::ONE;
        let mut transcript = FsChallenger::new(b"face-test");
        let closed = close_faces(&points, &values, &mut transcript).unwrap();
        assert_ne!(closed[0].value, evaluate(&bits, &closed[0].point));
        let (_, points, mut values) = claims(&[0, 0]);
        values[1] += F128::ONE;
        assert!(close_faces(&points, &values, &mut transcript).is_err());
        assert!(close_faces(&points, &[], &mut transcript).is_err());
    }
}
