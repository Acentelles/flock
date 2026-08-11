//! GPU prove -> Rust verify roundtrip. Needs a Blackwell GPU (sm_120):
//!   cargo test -p flock-cuda-ffi --release --features gpu -- --ignored --nocapture
//!
//! The CUDA prover (cuda-ghash/prove_ffi.cu) returns a flat little-endian
//! stream; `parse_proof` mirrors its FfiWriter layout exactly and rebuilds the
//! typed `R1csProofLigerito`, which the ordinary Rust verifier then checks.
#![cfg(feature = "gpu")]

use flock_prover::challenger::FsChallenger;
use flock_prover::field::{F8, F128};
use flock_prover::lincheck::{self, LincheckProof};
use flock_prover::ntt::AdditiveNttGf8;
use flock_prover::pcs::ligerito::{FinalProof, LigeritoProof, RecursiveProof, SumcheckMessage};
use flock_prover::pcs::ring_switch::RingSwitchProof;
use flock_prover::pcs::{BatchOpeningProofLigerito, Commitment, PcsParams};
use flock_prover::proof::R1csProofLigerito;
use flock_prover::r1cs::SparseBinaryMatrix;
use flock_prover::r1cs_hashes::blake3 as b3;
use flock_prover::verifier;
use flock_prover::zerocheck::{ZerocheckProof, K_SKIP};

const DOMAIN: &[u8] = b"flock-lig-r1cs-v0";

#[repr(C)]
struct ProveParams {
    m: i32,
    statement_digest: *const u8,
    domain: *const u8,
    domain_len: u32,
    a_col_ptr: *const u32,
    a_rows: *const u32,
    a_nnz: u32,
    b_col_ptr: *const u32,
    b_rows: *const u32,
    b_nnz: u32,
    const_pin_col: i32,
    useful_bits: i32,
    k_log: i32,
    zc_mcol: *const u8,
    zc_f8mul: *const u8,
    initial_k: i32,
    num_levels: i32,
    log_inv_rates: *const i32,
    recursive_ks: *const i32,
    queries: *const i32,
    grinding_bits: *const i32,
    fold_grinding_bits: *const i32,
    ood_samples: *const i32,
    recursive_steps: i32,
}

unsafe extern "C" {
    fn flock_cuda_prove_blake3(p: *const ProveParams, out: *mut *mut u8, out_len: *mut usize) -> i32;
    fn flock_cuda_free(p: *mut u8);
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_link_smoke() {
    let n = flock_cuda_ffi::gpu::device_count();
    assert!(n > 0, "no CUDA device visible (got {n})");
}

// `lincheck.rs::csc_from_rows` twin (same as dump_lincheck_vectors).
fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

// Zerocheck round-1 kernel tables (same as dump_zerocheck_full_vectors).
fn zc_tables() -> (Vec<u8>, Vec<u8>) {
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let mut mcol = vec![0u8; 64 * 64];
    for s in 0..64 {
        let mut col = vec![F8::ZERO; 64];
        col[s] = F8(1);
        ntt_s.inverse(&mut col);
        ntt_l.forward(&mut col);
        for i in 0..64 {
            mcol[s * 64 + i] = col[i].0;
        }
    }
    let mut f8mul = vec![0u8; 256 * 256];
    for x in 0..256usize {
        for y in 0..256usize {
            f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
        }
    }
    (mcol, f8mul)
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Reader<'a> {
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        v
    }
    fn f128(&mut self) -> F128 {
        let lo = self.u64();
        let hi = self.u64();
        F128 { lo, hi }
    }
    fn f128s(&mut self) -> Vec<F128> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.f128()).collect()
    }
    fn hash(&mut self) -> [u8; 32] {
        let h: [u8; 32] = self.b[self.o..self.o + 32].try_into().unwrap();
        self.o += 32;
        h
    }
    fn hashes(&mut self) -> Vec<[u8; 32]> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.hash()).collect()
    }
    fn rows(&mut self) -> Vec<Vec<F128>> {
        let n_rows = self.u64() as usize;
        let row_len = self.u64() as usize;
        (0..n_rows)
            .map(|_| (0..row_len).map(|_| self.f128()).collect())
            .collect()
    }
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_prove_rust_verify_roundtrip() {
    let n_blocks_log = 8usize; // m = 22: the smallest fast config
    let r1cs = b3::build_block_r1cs(n_blocks_log);
    let m = r1cs.m;
    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        merkle_hash: Default::default(),
    };
    let cfg = pcs_params
        .ligerito_prover_config()
        .expect("m22 fast ligerito config");

    let digest = r1cs.statement_digest();
    let (a_cp, a_rw) = csc_from_rows(&r1cs.a_0);
    let (b_cp, b_rw) = csc_from_rows(&r1cs.b_0);
    let (mcol, f8mul) = zc_tables();

    let to_i32 = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    let log_inv_rates = to_i32(&cfg.log_inv_rates);
    let recursive_ks = to_i32(&cfg.recursive_ks);
    let queries = to_i32(&cfg.queries);
    let grinding_bits = to_i32(&cfg.grinding_bits);
    let fold_grinding_bits = to_i32(&cfg.fold_grinding_bits);
    let ood_samples = to_i32(&cfg.ood_samples);
    let num_levels = log_inv_rates.len() as i32;
    let r_steps = cfg.recursive_steps;

    let params = ProveParams {
        m: m as i32,
        statement_digest: digest.as_ptr(),
        domain: DOMAIN.as_ptr(),
        domain_len: DOMAIN.len() as u32,
        a_col_ptr: a_cp.as_ptr(),
        a_rows: a_rw.as_ptr(),
        a_nnz: a_rw.len() as u32,
        b_col_ptr: b_cp.as_ptr(),
        b_rows: b_rw.as_ptr(),
        b_nnz: b_rw.len() as u32,
        const_pin_col: r1cs.const_pin.map_or(-1, |c| c as i32),
        useful_bits: r1cs.useful_bits as i32,
        k_log: r1cs.k_log as i32,
        zc_mcol: mcol.as_ptr(),
        zc_f8mul: f8mul.as_ptr(),
        initial_k: cfg.initial_k as i32,
        num_levels,
        log_inv_rates: log_inv_rates.as_ptr(),
        recursive_ks: recursive_ks.as_ptr(),
        queries: queries.as_ptr(),
        grinding_bits: grinding_bits.as_ptr(),
        fold_grinding_bits: fold_grinding_bits.as_ptr(),
        ood_samples: ood_samples.as_ptr(),
        recursive_steps: r_steps as i32,
    };

    let mut out: *mut u8 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { flock_cuda_prove_blake3(&params, &mut out, &mut out_len) };
    assert_eq!(rc, 0, "CUDA prover returned error {rc}");
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };

    // ---- parse the flat stream (must mirror prove_ffi.cu::FfiWriter) ----
    let mut r = Reader { b: &bytes, o: 0 };
    let root = r.hash();
    let round1_ab = r.f128s();
    let round1_c = r.f128s();
    let n_mlv = r.u64() as usize;
    let multilinear_rounds: Vec<(F128, F128)> = (0..n_mlv).map(|_| (r.f128(), r.f128())).collect();
    let final_a_eval = r.f128();
    let final_b_eval = r.f128();
    let final_c_eval = r.f128();
    let n_lc = r.u64() as usize;
    let lc_rounds: Vec<(F128, F128)> = (0..n_lc).map(|_| (r.f128(), r.f128())).collect();
    let z_partial = r.f128s();
    let shat_ab = r.f128s();
    let shat_c = r.f128s();
    let recursive_roots = r.hashes();
    let n_opens = r.u64() as usize;
    assert_eq!(n_opens, r_steps + 1, "level opens = r+1");
    let mut opens: Vec<(Vec<Vec<F128>>, Vec<[u8; 32]>)> = (0..n_opens)
        .map(|_| {
            let rows = r.rows();
            let proof = r.hashes();
            (rows, proof)
        })
        .collect();
    let yr = r.f128s();
    let n_sc = r.u64() as usize;
    let sumcheck_transcript: Vec<SumcheckMessage> = (0..n_sc)
        .map(|_| SumcheckMessage {
            u_0: r.f128(),
            u_2: r.f128(),
        })
        .collect();
    let ood_values = r.f128s();
    let grinding_nonces: Vec<u64> = { let n = r.u64() as usize; (0..n).map(|_| r.u64()).collect() };
    let fold_grinding_nonces: Vec<u64> = { let n = r.u64() as usize; (0..n).map(|_| r.u64()).collect() };
    assert_eq!(r.o, bytes.len(), "trailing bytes in FFI stream");

    let (initial_rows, initial_mp) = opens.remove(0);
    let (final_rows, final_mp) = opens.pop().expect("final open");
    let recursive_proofs: Vec<RecursiveProof> = opens
        .into_iter()
        .map(|(rows, mp)| RecursiveProof {
            opened_rows: rows,
            merkle_proof: mp,
        })
        .collect();

    let proof = R1csProofLigerito {
        zerocheck: ZerocheckProof {
            round1_ab,
            round1_c,
            multilinear_rounds,
            final_a_eval,
            final_b_eval,
            final_c_eval,
        },
        lincheck: LincheckProof {
            rounds: lc_rounds,
            z_partial,
        },
        pcs_open: BatchOpeningProofLigerito {
            ring_switches: vec![
                RingSwitchProof { s_hat_v: shat_ab },
                RingSwitchProof { s_hat_v: shat_c },
            ],
            ligerito: LigeritoProof {
                initial_root: root,
                initial_proof: RecursiveProof {
                    opened_rows: initial_rows,
                    merkle_proof: initial_mp,
                },
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: final_rows,
                    merkle_proof: final_mp,
                },
                sumcheck_transcript,
                grinding_nonces,
                ood_values,
                fold_grinding_nonces,
            },
        },
    };
    let commitment = Commitment {
        root,
        params: pcs_params.clone(),
    };

    let lc_circuit =
        lincheck::SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("Rust verifier rejected the GPU proof: {e:?}"));
    println!(
        "GPU proof verified: m={m}, ab claim value {:016x}:{:016x}",
        claim.ab.value.hi, claim.ab.value.lo
    );

    // Tamper: flip one bit of the final-level clear polynomial -> reject.
    let mut bad = proof.clone();
    bad.pcs_open.ligerito.final_proof.yr[0].lo ^= 1;
    let mut ch_t = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito(&r1cs, &commitment, &bad, &lc_circuit, &pcs_params, &mut ch_t)
            .is_err(),
        "verifier accepted a tampered GPU proof"
    );

    // Tamper: corrupt one zerocheck round message -> transcript replay rejects.
    let mut bad = proof.clone();
    bad.zerocheck.multilinear_rounds[0].0.hi ^= 1;
    let mut ch_t = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito(&r1cs, &commitment, &bad, &lc_circuit, &pcs_params, &mut ch_t)
            .is_err(),
        "verifier accepted a zerocheck-tampered GPU proof"
    );
}
