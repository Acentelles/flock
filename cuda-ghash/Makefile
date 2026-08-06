NVCC      ?= nvcc
ARCH      ?= sm_120
NVCCFLAGS  = -O3 -gencode arch=compute_120,code=$(ARCH) -lineinfo -std=c++17
VECTORS    = vectors.bin

.PHONY: all test bench bench_ext4 sass vectors clean
all: test_f128 bench_f128 bench_ext4

# KoalaBear/BabyBear degree-4 extension (SP1 bb31 port) for comparison.
bench_ext4: bench_ext4.cu monty31.cuh
	$(NVCC) $(NVCCFLAGS) bench_ext4.cu -o $@

# SP1's actual current KoalaBear kb31 extension (vendored under sp1/, needs c++20).
bench_kb31: bench_kb31.cu sp1/fields/kb31_extension_t.cuh
	$(NVCC) -O3 -gencode arch=compute_120,code=sm_120 -std=c++20 -Isp1 bench_kb31.cu -o $@

# Real-world sumcheck inner product: GHASH vs KoalaBear, compute-only vs memory-fed.
bench_sumcheck: bench_sumcheck.cu f128.cuh sp1/fields/kb31_extension_t.cuh
	$(NVCC) -O3 -gencode arch=compute_120,code=sm_120 -std=c++20 -Isp1 bench_sumcheck.cu -o $@

test_f128: test_f128.cu f128.cuh
	$(NVCC) $(NVCCFLAGS) test_f128.cu -o $@

bench_f128: bench_f128.cu f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_f128.cu -o $@

# Interleaved additive-NTT (PCS commit, P2) — bit-for-bit vs the flare oracle.
test_commit_ntt: test_commit_ntt.cu ntt_f128.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_commit_ntt.cu -o $@

# Throughput baseline for the P2 NTT kernel (on-device codeword, no oracle).
bench_commit_ntt: bench_commit_ntt.cu ntt_f128.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_commit_ntt.cu -o $@

# a·b sumcheck (PCS open / Ligerito) — round messages + folds bit-for-bit vs the flock oracle.
test_sumcheck_ab: test_sumcheck_ab.cu sumcheck_ab.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) test_sumcheck_ab.cu -o $@

# Throughput for the a·b sumcheck cascade (on-device a,b, no oracle).
bench_sumcheck_ab: bench_sumcheck_ab.cu sumcheck_ab.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_sumcheck_ab.cu -o $@

# Lincheck prover (zerocheck→PCS reduction) — comb_vec (CSC fold) + z_vec
# (partial fold) + top-bit sumcheck rounds + claim bit-for-bit vs the flock oracle.
test_lincheck: test_lincheck.cu lincheck.cuh phi8_table.cuh ntt_host.hpp f128.cuh challenger.hpp
	$(NVCC) $(NVCCFLAGS) test_lincheck.cu -o $@

# Throughput for the lincheck kernels (on-device inputs, no oracle).
bench_lincheck: bench_lincheck.cu lincheck.cuh phi8_table.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_lincheck.cu -o $@

# BLAKE3 witness generation (S4 GPU target) — per-block R1CS trace (z/a/b) +
# lincheck stripe transpose, bit-for-bit vs the flock generator.
test_blake3_witness: test_blake3_witness.cu blake3_witness.cuh
	$(NVCC) $(NVCCFLAGS) test_blake3_witness.cu -o $@

# Throughput for the BLAKE3 witness kernels (random inputs, no oracle).
bench_blake3_witness: bench_blake3_witness.cu blake3_witness.cuh
	$(NVCC) $(NVCCFLAGS) bench_blake3_witness.cu -o $@

# Zerocheck round-1 (univariate-skip URM message) — round1_ab/round1_c
# bit-for-bit vs the real optimized prove_packed round-1.
test_zerocheck_round1: test_zerocheck_round1.cu zerocheck_round1.cuh phi8_table.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zerocheck_round1.cu -o $@

# Throughput for the zerocheck round-1 kernel (random inputs, no oracle).
bench_zerocheck_round1: bench_zerocheck_round1.cu zerocheck_round1.cuh phi8_table.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_zerocheck_round1.cu -o $@

# CPU-structured round-1 (shift-reduce + convert-table). Host ref + GPU oracle + bench.
test_zc_round1_cpustyle: test_zc_round1_cpustyle.cu phi8_table.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zc_round1_cpustyle.cu -o $@
test_zc_challenger: test_zc_challenger.cu zc_challenger_device.cuh challenger.hpp sha256.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zc_challenger.cu -o $@
test_zc_cpustyle_gpu: test_zc_cpustyle_gpu.cu zerocheck_round1_cpustyle.cuh zerocheck_round1.cuh phi8_table.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zc_cpustyle_gpu.cu -o $@
bench_zc_cpustyle: bench_zc_cpustyle.cu zerocheck_round1_cpustyle.cuh zerocheck_round1.cuh phi8_table.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_zc_cpustyle.cu -o $@

# Zerocheck multilinear sumcheck tail (eq-weighted deg-2 message + fold) —
# bit-for-bit vs round_pair_naive + fold_in_place_pair.
test_zerocheck_tail: test_zerocheck_tail.cu zerocheck_tail.cuh sumcheck_ab.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zerocheck_tail.cu -o $@

# Zerocheck round-2 (fold-at-z via UniSkipFoldTable + first mlv message) —
# bit-for-bit vs uni_skip_fold_and_round_pair_optimized_packed.
test_zerocheck_round2: test_zerocheck_round2.cu zerocheck_round2.cuh zerocheck_tail.cuh sumcheck_ab.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zerocheck_round2.cu -o $@

# Full zerocheck prove_packed orchestration — end-to-end transcript bit-for-bit.
test_zerocheck_full: test_zerocheck_full.cu zerocheck_round1.cuh zerocheck_round1_cpustyle.cuh zc_challenger_device.cuh sha256.cuh zerocheck_round2.cuh zerocheck_tail.cuh sumcheck_ab.cuh challenger.hpp phi8_table.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_zerocheck_full.cu -o $@

# Induced basis builder (PCS open / Ligerito, step 4) — basis_poly + enforced_sum
# bit-for-bit vs the real ligerito::induce_sumcheck_poly oracle.
test_induce: test_induce.cu induce_sumcheck.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_induce.cu -o $@

# introduce_new + glue (PCS open / Ligerito, step 5) — message {u_0,u_2}, h_new,
# and glued basis bit-for-bit vs the real SumcheckProver oracle.
test_introduce_glue: test_introduce_glue.cu introduce_glue.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) test_introduce_glue.cu -o $@

# Merkle tree (PCS commit, P3) — root bit-for-bit vs the flare oracle.
test_commit_merkle: test_commit_merkle.cu merkle.cuh sha256.cuh
	$(NVCC) $(NVCCFLAGS) test_commit_merkle.cu -o $@

# Throughput for the P3 Merkle kernel (on-device leaves, no oracle).
bench_merkle: bench_merkle.cu merkle.cuh sha256.cuh
	$(NVCC) $(NVCCFLAGS) bench_merkle.cu -o $@

# Regenerate the commit oracle (CMT1): NTT + Merkle golden vectors.
# Override sizes:  make commit_vectors CMT_ARGS="24 1 5"
CMT_ARGS ?= 16 1 5
commit_vectors:
	cd .. && cargo run --release --quiet --bin dump_commit_vectors -- cuda-ghash/commit_vectors.bin $(CMT_ARGS)

# Build + run the NTT oracle check end-to-end.
commit_ntt: test_commit_ntt commit_vectors
	./test_commit_ntt commit_vectors.bin

# Regenerate the a·b sumcheck oracle (SMC1): a,b + per-round messages + final.
# Override size:  make sumcheck_vectors SMC_ARGS="22"   (log_len)
SMC_ARGS ?= 12
sumcheck_vectors:
	cd .. && cargo run --release --quiet --bin dump_sumcheck_vectors -- cuda-ghash/sumcheck_vectors.bin $(SMC_ARGS)

# Build + run the a·b sumcheck oracle check end-to-end (hand-replicated formula).
sumcheck_ab: test_sumcheck_ab sumcheck_vectors
	./test_sumcheck_ab sumcheck_vectors.bin

# Same check, but the oracle is sourced from the REAL Ligerito SumcheckProver
# (src/pcs/ligerito.rs) — validates the kernel against the code the prover runs.
ligerito_sumcheck_vectors:
	cd .. && cargo run --release --quiet --bin dump_ligerito_sumcheck_vectors -- cuda-ghash/sumcheck_vectors.bin $(SMC_ARGS)
sumcheck_ab_ligerito: test_sumcheck_ab ligerito_sumcheck_vectors
	./test_sumcheck_ab sumcheck_vectors.bin

# Regenerate the lincheck oracle (LNCK): witness + CSC base matrices + quirky
# claim point + golden comb_vec / z_vec / sumcheck rounds / (r_inner_skip, w).
# Override:  make lincheck_vectors LC_ARGS="14 8 2 256"  (m k_log k_skip useful_bits)
LC_ARGS ?= 10 4 2 16
lincheck_vectors:
	cd .. && cargo run --release --quiet --bin dump_lincheck_vectors -- cuda-ghash/lincheck_vectors.bin $(LC_ARGS)

# Build + run the lincheck oracle check end-to-end.
lincheck: test_lincheck lincheck_vectors
	./test_lincheck lincheck_vectors.bin

# Regenerate the BLAKE3 witness oracle (B3WT): random Compressions + golden
# z/a/b/z_lincheck.  Override:  make blake3_witness_vectors B3_ARGS="24 5"  (n_blocks n_blocks_log)
B3_ARGS ?= 24 5
blake3_witness_vectors:
	cd .. && cargo run --release --quiet --bin dump_blake3_witness_vectors -- cuda-ghash/blake3_witness_vectors.bin $(B3_ARGS)

# Build + run the BLAKE3 witness oracle check end-to-end.
blake3_witness: test_blake3_witness blake3_witness_vectors
	./test_blake3_witness blake3_witness_vectors.bin

# Dump the REAL BLAKE3 R1CS lincheck CSC matrices for bench_ligerito's lincheck phase.
blake3_lincheck_matrices:
	cd .. && cargo run --release --quiet --bin dump_blake3_lincheck_matrices -- cuda-ghash/blake3_lincheck_matrices.bin

# Regenerate the zerocheck round-1 oracle (ZCR1): a/b/c + r + M + golden round1.
# Override:  make zerocheck_round1_vectors ZC_ARGS="20 14 8000"  (m k_log useful_bits)
ZC_ARGS ?= 15
zerocheck_round1_vectors:
	cd .. && cargo run --release --quiet --bin dump_zerocheck_round1_vectors -- cuda-ghash/zerocheck_round1_vectors.bin $(ZC_ARGS)

# Build + run the zerocheck round-1 oracle check end-to-end.
zerocheck_round1: test_zerocheck_round1 zerocheck_round1_vectors
	./test_zerocheck_round1 zerocheck_round1_vectors.bin

# Regenerate the zerocheck sumcheck-tail oracle (ZTAL). Override:  make ztail_vectors ZTAIL_ARGS="20"
ZTAIL_ARGS ?= 16
ztail_vectors:
	cd .. && cargo run --release --quiet --bin dump_zerocheck_tail_vectors -- cuda-ghash/zerocheck_tail_vectors.bin $(ZTAIL_ARGS)

# Build + run the zerocheck sumcheck-tail oracle check end-to-end.
zerocheck_tail: test_zerocheck_tail ztail_vectors
	./test_zerocheck_tail zerocheck_tail_vectors.bin

# Regenerate the zerocheck round-2 oracle (ZR2). Override:  make zr2_vectors ZR2_ARGS="20"
ZR2_ARGS ?= 15
zr2_vectors:
	cd .. && cargo run --release --quiet --bin dump_zerocheck_round2_vectors -- cuda-ghash/zerocheck_round2_vectors.bin $(ZR2_ARGS)

# Build + run the zerocheck round-2 oracle check end-to-end.
zerocheck_round2: test_zerocheck_round2 zr2_vectors
	./test_zerocheck_round2 zerocheck_round2_vectors.bin

# Regenerate the full zerocheck transcript oracle (ZCFV). Override:  make zcfull_vectors ZCFULL_ARGS="20"
ZCFULL_ARGS ?= 15
zcfull_vectors:
	cd .. && cargo run --release --quiet --bin dump_zerocheck_full_vectors -- cuda-ghash/zerocheck_full_vectors.bin $(ZCFULL_ARGS)

# Build + run the full zerocheck orchestration check end-to-end.
zerocheck_full: test_zerocheck_full zcfull_vectors
	./test_zerocheck_full zerocheck_full_vectors.bin

# Regenerate the induce oracle (INDC) from the real ligerito::induce_sumcheck_poly.
# Override:  make induce_vectors INDUCE_ARGS="14 3 16"   (log_msg_cols v_len n_queries)
INDUCE_ARGS ?= 10 2 8
induce_vectors:
	cd .. && cargo run --release --quiet --bin dump_induce_vectors -- cuda-ghash/induce_vectors.bin $(INDUCE_ARGS)

# Build + run the induce oracle check end-to-end.
induce: test_induce induce_vectors
	./test_induce induce_vectors.bin

# Regenerate the introduce+glue oracle (INGL) from the real SumcheckProver.
# Override:  make introduce_glue_vectors IGL_ARGS="18"   (log_len)
IGL_ARGS ?= 12
introduce_glue_vectors:
	cd .. && cargo run --release --quiet --bin dump_introduce_glue_vectors -- cuda-ghash/introduce_glue_vectors.bin $(IGL_ARGS)

# Build + run the introduce+glue oracle check end-to-end.
introduce_glue: test_introduce_glue introduce_glue_vectors
	./test_introduce_glue introduce_glue_vectors.bin

# Regenerate bit-for-bit reference vectors from the real flare F128::mul.
vectors:
	cd .. && cargo run --release --quiet --bin dump_ghash_vectors -- cuda-ghash/$(VECTORS)

test: test_f128 vectors
	./test_f128 $(VECTORS)

bench: bench_f128
	./bench_f128

# Confirm the hot loops really use native CLMAD SASS.
sass: bench_f128.cu f128.cuh
	$(NVCC) $(NVCCFLAGS) -cubin bench_f128.cu -o bench_f128.cubin
	@echo "CLMAD instructions in the binary:"; \
	cuobjdump -sass bench_f128.cubin | grep -ic clmad; \
	cuobjdump -sass bench_f128.cubin | grep -i clmad | head -8

clean:
	rm -f test_f128 bench_f128 *.cubin $(VECTORS)

# Full multilinear sumcheck (round poly + fold, log N rounds) GHASH vs KoalaBear.
bench_full_sumcheck: bench_full_sumcheck.cu f128.cuh sp1/fields/kb31_extension_t.cuh
	$(NVCC) -O3 -gencode arch=compute_120,code=sm_120 -std=c++20 -Isp1 bench_full_sumcheck.cu -o $@

# Composed device-resident SumcheckProver (PCS open / Ligerito, step 6) —
# full transcript + final f bit-for-bit vs the real SumcheckProver.
test_sumcheck_prover: test_sumcheck_prover.cu sumcheck_ab.cuh introduce_glue.cuh f128.cuh
	$(NVCC) $(NVCCFLAGS) test_sumcheck_prover.cu -o $@
SCP_ARGS ?= 14
sumcheck_prover_vectors:
	cd .. && cargo run --release --quiet --bin dump_sumcheck_prover_vectors -- cuda-ghash/sumcheck_prover_vectors.bin $(SCP_ARGS)
sumcheck_prover: test_sumcheck_prover sumcheck_prover_vectors
	./test_sumcheck_prover sumcheck_prover_vectors.bin

# Throughput for the induce accumulation kernel (on-device inputs, no oracle).
bench_induce: bench_induce.cu induce_sumcheck.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_induce.cu -o $@

# Host Fiat-Shamir challenger (PCS open / Ligerito, step 6) — transcript bit-for-bit
# vs the real FsChallenger. Pure host C++ (no CUDA).
test_challenger: test_challenger.cpp challenger.hpp
	$(CXX) -O2 -std=c++17 test_challenger.cpp -o $@
CXX ?= g++
challenger_vectors:
	cd .. && cargo run --release --quiet --bin dump_challenger_vectors -- cuda-ghash/challenger_vectors.bin
challenger: test_challenger challenger_vectors
	./test_challenger challenger_vectors.bin

# Host Merkle multi-proof / query opening (PCS open / Ligerito, step 6) —
# byte-for-bit vs the real merkle::merkle_multi_proof. Pure host C++.
test_merkle_open: test_merkle_open.cpp merkle_open.hpp merkle_open_device.cuh
	$(NVCC) $(NVCCFLAGS) -x cu test_merkle_open.cpp -o $@
merkle_open_vectors:
	cd .. && cargo run --release --quiet --bin dump_merkle_open_vectors -- cuda-ghash/merkle_open_vectors.bin $(MKOP_ARGS)
MKOP_ARGS ?= 14 50
merkle_open: test_merkle_open merkle_open_vectors
	./test_merkle_open merkle_open_vectors.bin

# Ligerito L0 initial-sumcheck orchestrator (PCS open / Ligerito, step 6) —
# host FsChallenger drives the device-resident sumcheck; transcript bit-for-bit
# vs the real FsChallenger + SumcheckProver.
test_ligerito_l0: test_ligerito_l0.cu sumcheck_ab.cuh challenger.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_ligerito_l0.cu -o $@
L0_ARGS ?= 14 4 0
ligerito_l0_vectors:
	cd .. && cargo run --release --quiet --bin dump_ligerito_l0_vectors -- cuda-ghash/ligerito_l0_vectors.bin $(L0_ARGS)
ligerito_l0: test_ligerito_l0 ligerito_l0_vectors
	./test_ligerito_l0 ligerito_l0_vectors.bin

# Full Ligerito open prover benchmark (prove-only, no oracle).
bench_keccak3_gpu: bench_keccak3_gpu.cu keccak3_witness.cuh zerocheck_round1_cpustyle.cuh zerocheck_round1.cuh ntt_f128.cuh merkle.cuh merkle_open.hpp induce_sumcheck.cuh introduce_glue.cuh sumcheck_ab.cuh challenger.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_keccak3_gpu.cu -o $@
bench_ligerito: bench_ligerito.cu zerocheck_round1_cpustyle.cuh zerocheck_round1.cuh zerocheck_round2.cuh zerocheck_tail.cuh zc_challenger_device.cuh sha256.cuh ntt_f128.cuh merkle.cuh merkle_open.hpp induce_sumcheck.cuh introduce_glue.cuh sumcheck_ab.cuh lincheck.cuh blake3_witness.cuh challenger.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) bench_ligerito.cu -o $@

# Transpose-NTT induce (GPU fast path) — basis bit-for-bit vs induce_sumcheck_poly_via_ntt.
test_transpose_induce: test_transpose_induce.cu ntt_transpose.cuh induce_sumcheck.cuh ntt_host.hpp f128.cuh
	$(NVCC) $(NVCCFLAGS) test_transpose_induce.cu -o $@

# ---- CI aggregates (.github/workflows/cuda.yml) ----------------------------

# Every test binary with a build rule. Compiling needs nvcc but no GPU, so the
# hosted CI leg uses this to catch header/API bit-rot cheaply. Parallel-safe.
TEST_BINS = test_f128 test_commit_ntt test_sumcheck_ab test_lincheck \
            test_blake3_witness test_zerocheck_round1 test_zc_round1_cpustyle \
            test_zc_challenger test_zc_cpustyle_gpu test_zerocheck_tail \
            test_zerocheck_round2 test_zerocheck_full test_induce \
            test_introduce_glue test_commit_merkle test_sumcheck_prover \
            test_challenger test_merkle_open test_ligerito_l0 \
            test_transpose_induce

.PHONY: build-tests run-tests
build-tests: $(TEST_BINS)

# Every vector-validated end-to-end target: dump the oracle via the Rust
# dump_* bin, then replay it on device. Needs a Blackwell GPU (clmad) plus
# stable Rust. Run serially — sumcheck_ab and sumcheck_ab_ligerito share
# sumcheck_vectors.bin, so this target must not be invoked with -j.
run-tests: test commit_ntt sumcheck_ab sumcheck_ab_ligerito lincheck \
           blake3_witness zerocheck_round1 zerocheck_tail zerocheck_round2 \
           zerocheck_full induce introduce_glue sumcheck_prover challenger \
           merkle_open ligerito_l0
