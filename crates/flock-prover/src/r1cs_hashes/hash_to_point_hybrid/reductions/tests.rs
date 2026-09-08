use super::*;
use flock_core::challenger::FsChallenger;

fn digest(f: impl FnOnce(&mut FsChallenger)) -> [F128; 2] {
    let mut ch = FsChallenger::new(b"hybrid-reduction-ledger-tests");
    f(&mut ch);
    [ch.sample_f128(), ch.sample_f128()]
}

#[test]
fn hybrid_reduction_statement_ledger_binds_and_checks_all_parameters() {
    let mut setup = Setup::new(8);
    let commitment = Commitment {
        root: [37; 32],
        params: setup.params.clone(),
    };
    let baseline = digest(|ch| absorb_statement_ledger(&setup, &commitment, ch).unwrap());
    let mut altered = commitment.clone();
    altered.root[0] ^= 1;
    assert_ne!(
        baseline,
        digest(|ch| absorb_statement_ledger(&setup, &altered, ch).unwrap())
    );
    setup.descriptor[0] ^= 1;
    assert_ne!(
        baseline,
        digest(|ch| absorb_statement_ledger(&setup, &commitment, ch).unwrap())
    );
    setup.descriptor[0] ^= 1;
    for which in 0..5 {
        let mut wrong = commitment.clone();
        match which {
            0 => wrong.params.m = usize::MAX,
            1 => wrong.params.log_inv_rate += 1,
            2 => wrong.params.log_batch_size += 1,
            3 => wrong.params.profile = pcs::ligerito::LigeritoProfile::Secure,
            _ => wrong.params.merkle_hash = flock_core::hash::HashKind::Blake3,
        }
        let mut ch = FsChallenger::new(b"hybrid-reduction-ledger-tests");
        assert_eq!(
            absorb_statement_ledger(&setup, &wrong, &mut ch),
            Err("hybrid commitment parameters")
        );
        assert_eq!([ch.sample_f128(), ch.sample_f128()], digest(|_| {}));
    }
    assert_eq!(
        digest(|ch| bind_root(&setup, &commitment, ch)),
        digest(|ch| {
            setup.bind(ch);
            flock_core::proof::bind_statement(ch, &setup.r1cs, &commitment);
        }),
    );
}

fn identity_fragments(point: &[F128]) -> Vec<Fragment> {
    vec![Fragment {
        point: point.to_vec(),
        weight: F128::ONE,
    }]
}

#[test]
fn hybrid_reduction_ledgers_bind_returned_values_points_weights_and_order() {
    let mut random = FsChallenger::new(b"hybrid-reduction-arbitrary-ledger");
    let points: Vec<_> = (0..3).map(|_| random.sample_f128_vec(5)).collect();
    let values = random.sample_f128_vec(3);
    let ledger =
        RelationLedger::checked(points.clone(), identity_fragments, &values, &values).unwrap();
    let baseline = digest(|ch| ledger.absorb(ch));
    for which in 0..6 {
        let mut wrong = ledger.clone();
        match which {
            0 => {
                wrong.originals[0] += F128::ONE;
                wrong.values[0] += F128::ONE;
            }
            1 => wrong.source_points[0][0] += F128::ONE,
            2 => wrong.groups[0][0].point[0] += F128::ONE,
            3 => wrong.groups[0][0].weight += F128::ONE,
            4 => {
                wrong.source_points.swap(0, 1);
                wrong.groups.swap(0, 1);
                wrong.originals.swap(0, 1);
                wrong.values.swap(0, 1);
            }
            _ => {
                wrong.source_points[0].push(F128::ZERO);
            }
        }
        assert_ne!(baseline, digest(|ch| wrong.absorb(ch)), "mutation {which}");
    }
    let mut invalid = values.clone();
    invalid[1] += F128::ONE;
    assert!(
        RelationLedger::checked(points.clone(), identity_fragments, &values, &invalid).is_err()
    );
    assert!(
        RelationLedger::checked(points.clone(), identity_fragments, &values, &values[..2]).is_err()
    );
    assert!(
        RelationLedger::checked(points[..2].to_vec(), identity_fragments, &values, &values)
            .is_err()
    );

    let claim = ZClaim {
        point: lincheck::QuirkyPoint {
            z_skip: random.sample_f128(),
            x_inner_rest: random.sample_f128_vec(2),
            x_outer: random.sample_f128_vec(3),
        },
        value: random.sample_f128(),
    };
    let bound = digest(|ch| quirky(ch, &claim));
    for which in 0..4 {
        let mut wrong = claim.clone();
        match which {
            0 => wrong.point.z_skip += F128::ONE,
            1 => wrong.point.x_inner_rest[0] += F128::ONE,
            2 => wrong.point.x_outer[0] += F128::ONE,
            _ => wrong.value += F128::ONE,
        }
        assert_ne!(bound, digest(|ch| quirky(ch, &wrong)));
    }
}

fn publics(inputs: &[sponge::SpongeRecord]) -> Vec<sponge::SpongePublic> {
    inputs
        .iter()
        .map(|r| sponge::SpongePublic {
            hpk: r.hpk,
            message: r.message.clone(),
        })
        .collect()
}

fn verify_split(
    setup: &Setup,
    publics: &[sponge::SpongePublic],
    proof: &Proof,
) -> Result<[F128; 2], &'static str> {
    let mut ch = FsChallenger::new(b"hybrid-reduction-proof-identity");
    bind_root(setup, &proof.commitment, &mut ch);
    let circuit = verify_circuit_after_root(setup, publics, proof, &mut ch)?;
    let record = verify_record(setup, proof, &mut ch)?;
    let core = recombine_verifier(proof, circuit, record)?;
    verify_open(setup, proof, core, &mut ch)?;
    Ok([ch.sample_f128(), ch.sample_f128()])
}

#[test]
fn hybrid_reduction_split_preserves_full_proof_ledgers_and_transcript() {
    let inputs = super::super::tests::fixtures(32);
    let setup = Setup::new(inputs.len());
    let publics = publics(&inputs);
    let prepared = || commit(&setup, super::super::tests::direct_witness(&setup, &inputs));
    let mut reference_ch = FsChallenger::new(b"hybrid-reduction-proof-identity");
    let core = prove_core(&setup, prepared(), &mut reference_ch);
    let reference = open(&setup, core, &mut reference_ch);
    let reference_bytes = bincode::serialize(&reference).unwrap();
    let continuation = [reference_ch.sample_f128(), reference_ch.sample_f128()];
    for packed in [false, true] {
        let Prepared {
            witness: Witness { z, a, b },
            commitment,
            data,
        } = prepared();
        let z_pointer = z.as_ptr();
        let mut ch = FsChallenger::new(b"hybrid-reduction-proof-identity");
        bind_root(&setup, &commitment, &mut ch);
        let circuit = prove_circuit_after_root(&setup, &z, a, b, Default::default(), &mut ch);
        let mut seen_point = Vec::new();
        let record = prove_record(&setup, &z, packed, &mut ch, |p| seen_point = p.to_vec());
        assert_eq!(seen_point.as_slice(), record.r_fp());
        let circuit_digest = digest(|ledger_ch| circuit.absorb_ledger(ledger_ch));
        let record_digest = digest(|ledger_ch| record.absorb_ledger(ledger_ch));
        let core = recombine(z, commitment, data, circuit, record).unwrap();
        assert_eq!(core.fast.z_packed.as_ptr(), z_pointer);
        let proof = open(&setup, core, &mut ch);
        assert_eq!(bincode::serialize(&proof).unwrap(), reference_bytes);
        assert_eq!([ch.sample_f128(), ch.sample_f128()], continuation);

        let mut verifier = FsChallenger::new(b"hybrid-reduction-proof-identity");
        bind_root(&setup, &proof.commitment, &mut verifier);
        let vc = verify_circuit_after_root(&setup, &publics, &proof, &mut verifier).unwrap();
        let vr = verify_record(&setup, &proof, &mut verifier).unwrap();
        assert_eq!(
            digest(|ledger_ch| vc.absorb_ledger(ledger_ch)),
            circuit_digest
        );
        assert_eq!(
            digest(|ledger_ch| vr.absorb_ledger(ledger_ch)),
            record_digest
        );
        let core = recombine_verifier(&proof, vc, vr).unwrap();
        verify_open(&setup, &proof, core, &mut verifier).unwrap();
        assert_eq!(
            [verifier.sample_f128(), verifier.sample_f128()],
            continuation
        );

        // The unchanged verifier also accepts the split-produced proof.
        let mut legacy = FsChallenger::new(b"hybrid-reduction-proof-identity");
        let core = verify_core(&setup, &publics, &proof, &mut legacy).unwrap();
        verify_open(&setup, &proof, core, &mut legacy).unwrap();
        assert_eq!([legacy.sample_f128(), legacy.sample_f128()], continuation);
        if !packed {
            // Both pending reductions must replay the same proof input, even
            // when another allocation happens to contain identical bytes.
            let other = proof.clone();
            let mut verifier = FsChallenger::new(b"hybrid-reduction-proof-identity");
            bind_root(&setup, &proof.commitment, &mut verifier);
            let circuit =
                verify_circuit_after_root(&setup, &publics, &proof, &mut verifier).unwrap();
            let record = verify_record(&setup, &other, &mut verifier).unwrap();
            assert!(matches!(
                recombine_verifier(&proof, circuit, record),
                Err("hybrid reduction proof source")
            ));
        }
    }

    assert_eq!(
        verify_split(&setup, &publics, &reference).unwrap(),
        continuation
    );
    for which in 0..8 {
        let mut wrong = reference.clone();
        match which {
            0 => wrong.sponge_values[0] += F128::ONE,
            1 => wrong.record_values[0] += F128::ONE,
            2 => wrong.fingerprint += F128::ONE,
            3 => wrong.fragment_values[0] += F128::ONE,
            4 => {
                wrong.fragment_values.pop();
            }
            5 => wrong.fragment_values.push(F128::ZERO),
            6 => wrong.record_values[44] += F128::ONE,
            _ => *wrong.record_values.last_mut().unwrap() += F128::ONE,
        }
        assert!(
            verify_split(&setup, &publics, &wrong).is_err(),
            "mutation {which}"
        );
    }
}

#[test]
fn hybrid_reduction_record_borrow_matches_arbitrary_packed_witness() {
    let setup = Setup::new(8);
    let mut source = FsChallenger::new(b"hybrid-reduction-record-arbitrary");
    let z = source.sample_f128_vec(1 << (setup.r1cs.m - 7));
    let mut scalar_ch = FsChallenger::new(b"hybrid-reduction-record-arbitrary-proof");
    let mut packed_ch = scalar_ch.clone();
    let scalar = prove_record(&setup, &z, false, &mut scalar_ch, |_| {});
    let packed = prove_record(&setup, &z, true, &mut packed_ch, |_| {});
    assert_eq!(
        bincode::serialize(&scalar.scatter).unwrap(),
        bincode::serialize(&packed.scatter).unwrap()
    );
    assert_eq!(scalar.record.originals, packed.record.originals);
    assert_eq!(scalar.record.values, packed.record.values);
    assert_eq!(scalar.r_fp(), packed.r_fp());
    assert_eq!(scalar.fingerprint(), packed.fingerprint());
    assert_eq!(
        digest(|ch| scalar.absorb_ledger(ch)),
        digest(|ch| packed.absorb_ledger(ch))
    );
    assert_eq!(scalar_ch.sample_f128(), packed_ch.sample_f128());
}
