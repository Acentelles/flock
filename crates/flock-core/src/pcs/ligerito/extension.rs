//! Quadratic-extension folds with base-field code-switch commitments.
//!
//! A folded extension table `f = f_0 + u f_1` is never committed as an
//! extension-field word. At every code switch it becomes the base-field table
//!
//! `g(b, x) = f_b(x)`,
//!
//! with the coordinate bit `b` as the least-significant multilinear variable.
//! The current linear claim is transported by replacing a basis `B(x)` with
//! `u^b B(x)`. Consequently each recursive level spends one fold round on the
//! coordinate bit and removes only `k - 1` variables from the extension table.

use super::*;
use rayon::prelude::*;

/// Split extension values into the base-field table `g(b, x)`, with adjacent
/// `(b=0, b=1)` values for every `x`.
pub(super) fn split_coordinates(values: &[F256]) -> Vec<F128> {
    let mut split = vec![F128::ZERO; 2 * values.len()];
    split
        .par_chunks_exact_mut(2)
        .zip(values.par_iter())
        .for_each(|(out, value)| {
            out[0] = value.c0;
            out[1] = value.c1;
        });
    split
}

/// Transport an extension-valued basis across the coordinate split. For each
/// old basis value `B(x)`, the new pair is `(B(x), u B(x))`.
pub(super) fn split_basis(values: &[F256]) -> Vec<F256> {
    let mut split = vec![F256::ZERO; 2 * values.len()];
    split
        .par_chunks_exact_mut(2)
        .zip(values.par_iter())
        .for_each(|(out, &value)| {
            out[0] = value;
            out[1] = F256::U * value;
        });
    split
}

/// Inner product after the coordinate split. This is also the final residual
/// check: the proof exposes only F128 coordinate words while the basis and
/// running claim remain in F256.
pub(super) fn split_inner_product(words: &[F128], basis: &[F256]) -> F256 {
    assert_eq!(words.len(), basis.len());
    words
        .par_iter()
        .zip(basis.par_iter())
        .map(|(&word, &weight)| weight * word)
        .reduce(|| F256::ZERO, |a, b| a + b)
}

fn build_eq_table256(point: &[F256]) -> Vec<F256> {
    let mut table = vec![F256::ONE];
    for &r in point {
        let old = table.len();
        table.resize(2 * old, F256::ZERO);
        for i in 0..old {
            let v = table[i];
            table[i + old] = v * r;
            table[i] = v * (F256::ONE + r);
        }
    }
    table
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RoundQuad256 {
    c: F256,
    b: F256,
    a: F256,
}

impl RoundQuad256 {
    pub(super) fn from_msg(msg: SumcheckMessage256, claim: F256) -> Self {
        Self {
            c: msg.u_0,
            b: claim + msg.u_2,
            a: msg.u_2,
        }
    }

    pub(super) fn eval(self, r: F256) -> F256 {
        self.c + r * self.b + (r * r) * self.a
    }

    pub(super) fn fold(self, rhs: Self, alpha: F128) -> Self {
        Self {
            c: self.c + rhs.c * alpha,
            b: self.b + rhs.b * alpha,
            a: self.a + rhs.a * alpha,
        }
    }
}

#[inline]
pub(super) fn observe_message<Ch: Challenger>(challenger: &mut Ch, msg: SumcheckMessage256) {
    challenger.observe_f256(msg.u_0);
    challenger.observe_f256(msg.u_2);
}

fn round_msg(f: &[F256], b: &[F256]) -> SumcheckMessage256 {
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_power_of_two() && f.len() >= 2);
    let half = f.len() / 2;
    let (u_0, u_2) = (0..half)
        .into_par_iter()
        .map(|j| {
            let (f0, f1) = (f[2 * j], f[2 * j + 1]);
            let (b0, b1) = (b[2 * j], b[2 * j + 1]);
            (f0 * b0, (f0 + f1) * (b0 + b1))
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage256 { u_0, u_2 }
}

fn round_msg_blocked(f: &[F256], b: &[F256], d: usize) -> SumcheckMessage256 {
    if d == 1 || f.len() == d {
        return round_msg(f, b);
    }
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(2 * d));
    let (u_0, u_2) = (0..f.len() / (2 * d))
        .into_par_iter()
        .map(|j| {
            let mut u0 = F256::ZERO;
            let mut u2 = F256::ZERO;
            let lo = 2 * j * d;
            let hi = lo + d;
            for k in 0..d {
                let (f0, f1) = (f[lo + k], f[hi + k]);
                let (b0, b1) = (b[lo + k], b[hi + k]);
                u0 += f0 * b0;
                u2 += (f0 + f1) * (b0 + b1);
            }
            (u0, u2)
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage256 { u_0, u_2 }
}

#[inline]
fn next_round_msg(f: &[F256], b: &[F256], d: usize) -> SumcheckMessage256 {
    if d > 1 && f.len() > d {
        round_msg_blocked(f, b, d)
    } else {
        round_msg(f, b)
    }
}

fn fold_extension(values: &[F256], r: F256, d: usize) -> Vec<F256> {
    let half = values.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|o| {
            let (block, within) = (o / d, o % d);
            let lo = 2 * block * d + within;
            let hi = lo + d;
            values[lo] + r * (values[hi] + values[lo])
        })
        .collect()
}

pub(super) fn evaluate_dense_at_residual(
    basis: &[F128],
    point_prefix: &[F256],
    residual_log: usize,
) -> Vec<F256> {
    let mut values: Vec<F256> = basis.iter().copied().map(F256::from).collect();
    for &r in point_prefix {
        values = fold_extension(&values, r, 1);
    }
    assert_eq!(values.len(), 1usize << residual_log);
    values
}

fn fold_base(values: &[F128], r: F256, d: usize) -> Vec<F256> {
    let half = values.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|o| {
            let (block, within) = (o / d, o % d);
            let lo = 2 * block * d + within;
            let hi = lo + d;
            F256::from(values[lo]) + r * (values[hi] + values[lo])
        })
        .collect()
}

fn fold_base_fill(
    f: &[F128],
    fill: BasisWindowFn<'_>,
    r: F256,
    d: usize,
) -> (Vec<F256>, Vec<F256>) {
    let half = f.len() / 2;
    let mut nf = vec![F256::ZERO; half];
    let mut nb = vec![F256::ZERO; half];
    let chunk = 2048usize.min(d);
    nf.par_chunks_mut(chunk)
        .zip(nb.par_chunks_mut(chunk))
        .enumerate()
        .for_each_init(
            || (vec![F128::ZERO; chunk], vec![F128::ZERO; chunk]),
            |(blo, bhi), (ci, (fo, bo))| {
                let o = ci * chunk;
                let (block, within) = (o / d, o % d);
                let lo = 2 * block * d + within;
                let hi = lo + d;
                let len = fo.len();
                fill(&mut blo[..len], lo);
                fill(&mut bhi[..len], hi);
                for k in 0..len {
                    fo[k] = F256::from(f[lo + k]) + r * (f[hi + k] + f[lo + k]);
                    bo[k] = F256::from(blo[k]) + r * (bhi[k] + blo[k]);
                }
            },
        );
    (nf, nb)
}

struct VirtualEqTerm256 {
    coords: Vec<F128>,
    scale: F256,
    lo: Vec<F128>,
    hi: Vec<F128>,
    n_lo: usize,
}

impl VirtualEqTerm256 {
    fn from_base(term: VirtualEqTerm) -> Self {
        Self {
            coords: term.coords,
            scale: F256::from(term.scale),
            lo: term.lo,
            hi: term.hi,
            n_lo: term.n_lo,
        }
    }

    fn rebuild(&mut self) {
        self.n_lo = self.coords.len() / 2;
        self.lo =
            crate::pcs::ring_switch::build_eq_scaled_parallel(&self.coords[..self.n_lo], F128::ONE);
        self.hi =
            crate::pcs::ring_switch::build_eq_scaled_parallel(&self.coords[self.n_lo..], F128::ONE);
    }

    fn fold_coord(&mut self, p: usize, r: F256) {
        self.scale *= F256::ONE + F256::from(self.coords[p]) + r;
        self.coords.remove(p);
        self.rebuild();
    }

    fn len(&self) -> usize {
        1usize << self.coords.len()
    }

    fn value_at(&self, u: usize) -> F256 {
        let mask = (1usize << self.n_lo) - 1;
        self.scale * (self.lo[u & mask] * self.hi[u >> self.n_lo])
    }

    fn add_to(&self, out: &mut [F256], g0: usize) {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot += self.value_at(g0 + i);
        }
    }
}

pub(super) struct VirtualEqBasis256 {
    terms: Vec<VirtualEqTerm256>,
}

impl VirtualEqBasis256 {
    pub(super) fn from_base(value: VirtualEqBasis) -> Self {
        Self {
            terms: value
                .terms
                .into_iter()
                .map(VirtualEqTerm256::from_base)
                .collect(),
        }
    }

    pub(super) fn fold_coord(&mut self, p: usize, r: F256) {
        for term in &mut self.terms {
            term.fold_coord(p, r);
        }
    }

    fn len(&self) -> usize {
        self.terms[0].len()
    }

    fn fill(&self, out: &mut [F256], g0: usize) {
        out.fill(F256::ZERO);
        for term in &self.terms {
            term.add_to(out, g0);
        }
    }

    fn materialize(&self) -> Vec<F256> {
        let mut out = vec![F256::ZERO; self.len()];
        out.par_chunks_mut(1 << 12)
            .enumerate()
            .for_each(|(i, chunk)| self.fill(chunk, i << 12));
        out
    }
}

enum PendingBasis {
    Extension(Vec<F256>),
}

pub(super) struct SumcheckProver256 {
    initial_f: Option<Vec<F128>>,
    initial_b: Option<Vec<F128>>,
    f: Vec<F256>,
    combined_basis: Vec<F256>,
    transcript: Vec<SumcheckMessage256>,
    pending: Option<PendingBasis>,
}

impl SumcheckProver256 {
    pub(super) fn new(f: Vec<F128>, b: Option<Vec<F128>>, first: SumcheckMessage) -> Self {
        Self {
            initial_f: Some(f),
            initial_b: b,
            f: Vec::new(),
            combined_basis: Vec::new(),
            transcript: vec![SumcheckMessage256 {
                u_0: F256::from(first.u_0),
                u_2: F256::from(first.u_2),
            }],
            pending: None,
        }
    }

    pub(super) fn first_fold_materialized(&mut self, r: F256, d: usize) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        let b = self.initial_b.take().expect("materialized basis missing");
        self.f = fold_base(&f, r, d);
        self.combined_basis = fold_base(&b, r, d);
        let msg = next_round_msg(&self.f, &self.combined_basis, d);
        self.transcript.push(msg);
        msg
    }

    pub(super) fn first_fold_jit(
        &mut self,
        r: F256,
        d: usize,
        fill: BasisWindowFn<'_>,
    ) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        (self.f, self.combined_basis) = fold_base_fill(&f, fill, r, d);
        let msg = next_round_msg(&self.f, &self.combined_basis, d);
        self.transcript.push(msg);
        msg
    }

    pub(super) fn first_fold_virtual(
        &mut self,
        r: F256,
        d: usize,
        basis: &VirtualEqBasis256,
    ) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        self.f = fold_base(&f, r, d);
        let msg = if self.f.len() == d {
            self.combined_basis = basis.materialize();
            round_msg(&self.f, &self.combined_basis)
        } else {
            round_msg_virtual(&self.f, basis, d)
        };
        self.transcript.push(msg);
        msg
    }

    pub(super) fn fold_materialized(&mut self, r: F256, d: usize) -> SumcheckMessage256 {
        self.f = fold_extension(&self.f, r, d);
        self.combined_basis = fold_extension(&self.combined_basis, r, d);
        let msg = next_round_msg(&self.f, &self.combined_basis, d);
        self.transcript.push(msg);
        msg
    }

    pub(super) fn fold_virtual(
        &mut self,
        r: F256,
        d: usize,
        basis: &VirtualEqBasis256,
    ) -> SumcheckMessage256 {
        self.f = fold_extension(&self.f, r, d);
        let msg = if self.f.len() == d {
            self.combined_basis = basis.materialize();
            round_msg(&self.f, &self.combined_basis)
        } else {
            round_msg_virtual(&self.f, basis, d)
        };
        self.transcript.push(msg);
        msg
    }

    pub(super) fn fold(&mut self, r: F256) -> SumcheckMessage256 {
        self.fold_materialized(r, 1)
    }

    /// Replace the just-produced next-round message with the message for the
    /// coordinate-split table. No transcript item is added: the code switch is
    /// a representation change at the same sumcheck boundary.
    pub(super) fn code_switch_and_replace_message(&mut self) -> SumcheckMessage256 {
        assert!(self.pending.is_none());
        let words = split_coordinates(&self.f);
        self.f = words.into_iter().map(F256::from).collect();
        self.combined_basis = split_basis(&self.combined_basis);
        let msg = round_msg(&self.f, &self.combined_basis);
        *self
            .transcript
            .last_mut()
            .expect("a fold message must precede a code switch") = msg;
        msg
    }

    fn introduce_extension(&mut self, basis: Vec<F256>, claim: F256) -> SumcheckMessage256 {
        assert_eq!(basis.len(), self.f.len());
        let msg = round_msg(&self.f, &basis);
        debug_assert_eq!(
            basis
                .iter()
                .zip(&self.f)
                .fold(F256::ZERO, |acc, (&b, &f)| acc + b * f),
            claim
        );
        self.transcript.push(msg);
        self.pending = Some(PendingBasis::Extension(basis));
        msg
    }

    /// Introduce a base-field MLE claim on the currently split table. The
    /// answer is one F128 value because every table word is in the subfield.
    pub(super) fn introduce_ood_with_eval(
        &mut self,
        basis: Vec<F128>,
    ) -> (SumcheckMessage256, F128) {
        assert_eq!(basis.len(), self.f.len());
        let mut answer = F128::ZERO;
        for (&f, &b) in self.f.iter().zip(&basis) {
            assert_eq!(f.c1, F128::ZERO, "OOD table must be base-valued");
            answer += f.c0 * b;
        }
        let ext_basis = basis.into_iter().map(F256::from).collect();
        let msg = self.introduce_extension(ext_basis, F256::from(answer));
        (msg, answer)
    }

    /// Introduce a claim stated on the extension table immediately before its
    /// coordinate split. The `u^b` weight transports it to the current table.
    pub(super) fn introduce_presplit_basis(
        &mut self,
        basis: Vec<F128>,
        claim: F256,
    ) -> SumcheckMessage256 {
        let basis_ext: Vec<F256> = basis.into_iter().map(F256::from).collect();
        self.introduce_extension(split_basis(&basis_ext), claim)
    }

    pub(super) fn glue(&mut self, beta: F128) {
        let pending = self.pending.take().expect("glue without introduce");
        match pending {
            PendingBasis::Extension(basis) => {
                self.combined_basis
                    .par_iter_mut()
                    .zip(basis.par_iter())
                    .for_each(|(dst, &src)| *dst += src * beta);
            }
        }
    }

    pub(super) fn f(&self) -> &[F256] {
        &self.f
    }

    pub(super) fn transcript(&self) -> &[SumcheckMessage256] {
        &self.transcript
    }
}

fn round_msg_virtual(f: &[F256], basis: &VirtualEqBasis256, d: usize) -> SumcheckMessage256 {
    let mut b = vec![F256::ZERO; f.len()];
    basis.fill(&mut b, 0);
    round_msg_blocked(f, &b, d)
}

fn induced_basis(
    log_msg_cols: usize,
    log_inv_rate: usize,
    queries: &[usize],
    alpha: &[F128],
) -> Vec<F128> {
    let empty_rows = vec![Vec::new(); queries.len()];
    induce_sumcheck_poly_auto(
        log_msg_cols,
        log_inv_rate,
        &eval_sk_at_vks(log_msg_cols),
        &empty_rows,
        &[],
        queries,
        alpha,
    )
    .0
}

fn base_table(values: &[F256]) -> Vec<F128> {
    values
        .iter()
        .map(|value| {
            assert_eq!(value.c1, F128::ZERO, "committed words must be in F128");
            value.c0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recursive_prover_with_basis_impl<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    mut b_initial: Vec<F128>,
    mut target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    l0_num_lanes: usize,
    l0_lane_major: bool,
    l0_jit_basis: Option<BasisWindowFn<'_>>,
    l0_virtual_basis: Option<VirtualEqBasis>,
    mut first_msg: Option<SumcheckMessage>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    assert!(r >= 1);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(config.fold_grinding_bits.iter().all(|&bits| bits == 0));
    assert!(config.recursive_ks.iter().all(|&k| k >= 2));

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    assert_eq!(l0_codeword.len(), block_len_0 * l0_num_lanes);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);
    let fold_block = if l0_lane_major {
        1usize << log_msg_cols_0
    } else {
        1
    };

    challenger.observe_label(b"flock-ligerito-basis-f256-split-v0");
    challenger.observe_f128(target);
    let strat = |level: usize| &config.stratified[level];
    let cap_depth = |level: usize| config.stratified[level].cap_depth();
    let initial_cap = merkle::cap_layer(l0_tree, block_len_0, cap_depth(0)).to_vec();
    challenger.observe_bytes(initial_cap.as_flattened());

    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;
    let ood_count = |level: usize| config.ood_samples[level];
    let l0_row = |q: usize| &l0_codeword[q * l0_num_lanes..(q + 1) * l0_num_lanes];

    let mut ood_values = Vec::new();
    let mut claim_batch_grinding_nonces = Vec::new();
    let mut consistency_batch_grinding_nonces = Vec::new();
    let mut grinding_nonces = Vec::new();

    let factored = l0_jit_basis.is_some() || l0_virtual_basis.is_some();
    let mut virtual_basis = l0_virtual_basis;
    let mut jit_ood_basis: Option<VirtualEqBasis> = None;
    for _ in 0..ood_count(0) {
        let z = challenger.sample_f128_vec(log_n);
        let (ood_msg, y, eq_z) = if factored {
            let (msg, y) = round_msg_and_eval_eq_point_blocked(&packed_witness, &z, fold_block);
            (msg, y, None)
        } else {
            let eq_z = build_eq_table(&z);
            let (msg, y) = round_msg_and_eval_blocked(&packed_witness, &eq_z, fold_block);
            (msg, y, Some(eq_z))
        };
        challenger.observe_f128(y);
        ood_values.push(y);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(0));
        claim_batch_grinding_nonces.push(nonce);
        target += beta * y;
        if let Some(msg) = first_msg.as_mut() {
            msg.u_0 += beta * ood_msg.u_0;
            msg.u_2 += beta * ood_msg.u_2;
        }
        if let Some(vb) = virtual_basis.as_mut() {
            vb.add_term(z, beta);
        } else if factored {
            if let Some(vb) = jit_ood_basis.as_mut() {
                vb.add_term(z, beta);
            } else {
                jit_ood_basis = Some(VirtualEqBasis::new(z, beta));
            }
        } else {
            let eq_z = eq_z.expect("materialized OOD basis");
            b_initial
                .par_iter_mut()
                .zip(eq_z.par_iter())
                .for_each(|(dst, &src)| *dst += beta * src);
        }
    }

    let first = match first_msg {
        Some(msg) => msg,
        None => {
            assert!(!factored, "factored L0 needs its precomputed first message");
            round_msg_and_eval_blocked(&packed_witness, &b_initial, fold_block).0
        }
    };
    let materialized = (!factored).then_some(b_initial);
    let mut sumcheck = SumcheckProver256::new(packed_witness, materialized, first);
    observe_message(challenger, sumcheck.transcript()[0]);

    let mut virtual_basis = virtual_basis.map(VirtualEqBasis256::from_base);
    let mut jit = l0_jit_basis;
    let mut lane_challenges = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        let challenge = challenger.sample_f256();
        if let Some(vb) = virtual_basis.as_mut() {
            vb.fold_coord(fold_block.trailing_zeros() as usize, challenge);
        }
        let _pre_switch = if let Some(vb) = virtual_basis.as_ref() {
            if j == 0 {
                sumcheck.first_fold_virtual(challenge, fold_block, vb)
            } else {
                sumcheck.fold_virtual(challenge, fold_block, vb)
            }
        } else if j == 0 {
            if let Some(fill) = jit.take() {
                match jit_ood_basis.as_ref() {
                    Some(ood) => {
                        let combined = |out: &mut [F128], offset: usize| {
                            fill(out, offset);
                            ood.add_to(out, offset);
                        };
                        sumcheck.first_fold_jit(challenge, fold_block, &combined)
                    }
                    None => sumcheck.first_fold_jit(challenge, fold_block, fill),
                }
            } else {
                sumcheck.first_fold_materialized(challenge, fold_block)
            }
        } else if virtual_basis.is_some() {
            unreachable!()
        } else {
            sumcheck.fold_materialized(challenge, fold_block)
        };
        let msg = if j + 1 == initial_k {
            sumcheck.code_switch_and_replace_message()
        } else {
            _pre_switch
        };
        observe_message(challenger, msg);
        lane_challenges.push(challenge);
    }

    let n1 = log_n - initial_k;
    let mut current_split_dim = n1 + 1;
    let commit_split = |values: &[F256], level: usize, split_dim: usize| {
        let log_lanes = config.recursive_ks[level - 1];
        assert!(split_dim >= log_lanes);
        let log_cols = split_dim - log_lanes;
        let log_rate = config.log_inv_rates[level];
        let ntt = AdditiveNttF128::standard(log_cols + log_rate);
        ligero_commit(
            &base_table(values),
            log_cols,
            log_lanes,
            log_rate,
            &ntt,
            config.merkle_hash,
        )
    };

    let mut previous = commit_split(sumcheck.f(), 1, current_split_dim);
    let mut recursive_caps = vec![previous.cap(cap_depth(1)).to_vec()];
    challenger.observe_bytes(recursive_caps[0].as_flattened());

    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(current_split_dim);
        let (msg, y) = sumcheck.introduce_ood_with_eval(build_eq_table(&z));
        challenger.observe_f128(y);
        ood_values.push(y);
        observe_message(challenger, msg);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(1));
        claim_batch_grinding_nonces.push(nonce);
        sumcheck.glue(beta);
    }

    let (nonce, queries_0) = grind_and_sample_queries(
        challenger,
        config.grinding_bits[0] as u32,
        block_len_0,
        config.queries[0],
        strat(0),
    );
    grinding_nonces.push(nonce);
    let (nonce, alpha_0) =
        challenger.grind_pow_and_sample_f128_vec(consistency_bits(1), ceil_log2(config.queries[0]));
    consistency_batch_grinding_nonces.push(nonce);
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| l0_row(q).to_vec()).collect();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_paths_for(l0_tree, block_len_0, &queries_0, strat(0)),
    };
    let basis_0 = induced_basis(n1, log_inv_rate_0, &queries_0, &alpha_0);
    let enforced_0 = induce_enforced_sum(&opened_rows_0, &lane_challenges, &alpha_0);
    let msg = sumcheck.introduce_presplit_basis(basis_0, enforced_0);
    observe_message(challenger, msg);
    let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(1));
    claim_batch_grinding_nonces.push(nonce);
    sumcheck.glue(beta);

    let mut recursive_proofs = Vec::new();
    for i in 0..r {
        let k = config.recursive_ks[i];
        assert!(current_split_dim >= k);
        let mut level_challenges = Vec::with_capacity(k);
        for j in 0..k {
            let challenge = challenger.sample_f256();
            let pre_switch = sumcheck.fold(challenge);
            let msg = if j + 1 == k && i + 1 != r {
                sumcheck.code_switch_and_replace_message()
            } else {
                pre_switch
            };
            observe_message(challenger, msg);
            level_challenges.push(challenge);
        }
        let extension_dim = current_split_dim - k;
        let level = i + 1;

        if i + 1 == r {
            let yr = split_coordinates(sumcheck.f());
            for &value in &yr {
                challenger.observe_f128(value);
            }
            let (nonce, queries) = grind_and_sample_queries(
                challenger,
                config.grinding_bits[level] as u32,
                previous.block_len,
                config.queries[level],
                strat(level),
            );
            grinding_nonces.push(nonce);
            let (nonce, _) = challenger.grind_pow_and_sample_f128_vec(
                consistency_bits(level),
                ceil_log2(config.queries[level]),
            );
            consistency_batch_grinding_nonces.push(nonce);
            let (nonce, _) = challenger.grind_pow_and_sample_f128(claim_bits(level));
            claim_batch_grinding_nonces.push(nonce);
            let opened_rows = queries.iter().map(|&q| previous.row(q).to_vec()).collect();
            return LigeritoProof {
                initial_cap,
                initial_proof,
                recursive_caps,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows,
                    merkle_proof: merkle_paths_for(
                        &previous.tree,
                        previous.block_len,
                        &queries,
                        strat(level),
                    ),
                },
                sumcheck_transcript: Vec::new(),
                sumcheck_transcript_f256: sumcheck.transcript().to_vec(),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces: Vec::new(),
                claim_batch_grinding_nonces,
                consistency_batch_grinding_nonces,
            };
        }

        current_split_dim = extension_dim + 1;
        let next_level = i + 2;
        let next = commit_split(sumcheck.f(), next_level, current_split_dim);
        let cap = next.cap(cap_depth(next_level)).to_vec();
        challenger.observe_bytes(cap.as_flattened());
        recursive_caps.push(cap);

        for _ in 0..ood_count(next_level) {
            let z = challenger.sample_f128_vec(current_split_dim);
            let (msg, y) = sumcheck.introduce_ood_with_eval(build_eq_table(&z));
            challenger.observe_f128(y);
            ood_values.push(y);
            observe_message(challenger, msg);
            let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(next_level));
            claim_batch_grinding_nonces.push(nonce);
            sumcheck.glue(beta);
        }

        let (nonce, queries) = grind_and_sample_queries(
            challenger,
            config.grinding_bits[level] as u32,
            previous.block_len,
            config.queries[level],
            strat(level),
        );
        grinding_nonces.push(nonce);
        let (nonce, alpha) = challenger.grind_pow_and_sample_f128_vec(
            consistency_bits(next_level),
            ceil_log2(config.queries[level]),
        );
        consistency_batch_grinding_nonces.push(nonce);
        let opened_rows: Vec<Vec<F128>> =
            queries.iter().map(|&q| previous.row(q).to_vec()).collect();
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows.clone(),
            merkle_proof: merkle_paths_for(
                &previous.tree,
                previous.block_len,
                &queries,
                strat(level),
            ),
        });
        let basis = induced_basis(extension_dim, config.log_inv_rates[level], &queries, &alpha);
        let enforced = induce_enforced_sum(&opened_rows, &level_challenges, &alpha);
        let msg = sumcheck.introduce_presplit_basis(basis, enforced);
        observe_message(challenger, msg);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(next_level));
        claim_batch_grinding_nonces.push(nonce);
        sumcheck.glue(beta);
        previous = next;
    }
    unreachable!()
}

fn coordinate_fold_factor(challenge: F256) -> F256 {
    F256::ONE + challenge * (F256::ONE + F256::U)
}

fn induced_basis_at_residual(
    log_msg_cols: usize,
    queries: &[usize],
    alpha: &[F128],
    fixed: &[F256],
    residual_log: usize,
) -> Vec<F256> {
    assert_eq!(fixed.len() + residual_log, log_msg_cols);
    let sks_vks = eval_sk_at_vks(log_msg_cols);
    let inv: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();
    let weights = build_eq_table(alpha);
    let mut per_query = Vec::with_capacity(queries.len());
    for (&query, &weight) in queries.iter().zip(&weights) {
        let mut w = vec![F128::ZERO; log_msg_cols];
        if log_msg_cols > 0 {
            w[0] = F128::new(query as u64, 0);
            for k in 1..log_msg_cols {
                w[k] = next_s(w[k - 1], sks_vks[k - 1]);
            }
            for k in 0..log_msg_cols {
                w[k] *= inv[k];
            }
        }
        let prefix = fixed.iter().zip(&w).fold(F256::ONE, |acc, (&p, &wk)| {
            acc * (F256::ONE + p * (F128::ONE + wk))
        });
        per_query.push((weight, prefix, w[fixed.len()..].to_vec()));
    }
    (0..1usize << residual_log)
        .map(|y| {
            per_query
                .iter()
                .map(|&(weight, prefix, ref suffix)| {
                    let tail = suffix.iter().enumerate().fold(F256::ONE, |acc, (j, &wk)| {
                        if (y >> j) & 1 == 0 {
                            acc
                        } else {
                            acc * F256::from(wk)
                        }
                    });
                    prefix * tail * weight
                })
                .fold(F256::ZERO, |a, b| a + b)
        })
        .collect()
}

#[derive(Clone)]
struct OodResidualContext {
    point: Vec<F128>,
    beta: F128,
    /// `None` is an L0 base-table claim. `Some(s)` is a claim on the split
    /// commitment consumed by recursive level `s`.
    split_level: Option<usize>,
}

#[derive(Clone)]
struct ConsistencyResidualContext {
    log_cols: usize,
    queries: Vec<usize>,
    alpha: Vec<F128>,
    beta: F128,
    /// First recursive split level whose coordinate weight applies.
    start_level: usize,
}

fn residual_original_challenges(
    initial: &[F256],
    levels: &[Vec<F256>],
    start_level: usize,
) -> Vec<F256> {
    let initial_len = if start_level == 0 { initial.len() } else { 0 };
    let recursive_len: usize = levels[start_level..]
        .iter()
        .map(|level| level.len().saturating_sub(1))
        .sum();
    let mut out = Vec::with_capacity(initial_len + recursive_len);
    if start_level == 0 {
        out.extend_from_slice(initial);
    }
    for level in &levels[start_level..] {
        out.extend_from_slice(&level[1..]);
    }
    out
}

fn coordinate_factor_product(levels: &[Vec<F256>], start_level: usize) -> F256 {
    levels[start_level..].iter().fold(F256::ONE, |acc, level| {
        acc * coordinate_fold_factor(level[0])
    })
}

fn eq_residual(point: &[F128], fixed: &[F256], residual_log: usize, scale: F256) -> Vec<F256> {
    assert_eq!(fixed.len() + residual_log, point.len());
    let prefix = point[..fixed.len()]
        .iter()
        .zip(fixed)
        .fold(scale, |acc, (&z, &r)| acc * (F256::ONE + F256::from(z) + r));
    (0..1usize << residual_log)
        .map(|y| {
            point[fixed.len()..]
                .iter()
                .enumerate()
                .fold(prefix, |acc, (j, &z)| {
                    acc * if (y >> j) & 1 == 1 { z } else { F128::ONE + z }
                })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_verifier_with_basis_succinct<Ch, F>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    log_n: usize,
    target: F128,
    expected_initial_cap: &[Hash],
    l0_num_lanes: usize,
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F256], usize) -> Vec<F256>,
{
    let initial_k = config.initial_k;
    let rounds = config.recursive_steps;
    if rounds < 1
        || proof.initial_cap != expected_initial_cap
        || !proof.sumcheck_transcript.is_empty()
        || proof.sumcheck_transcript_f256.is_empty()
        || !proof.fold_grinding_nonces.is_empty()
        || config.fold_grinding_bits.iter().any(|&bits| bits != 0)
        || config.recursive_ks.iter().any(|&k| k < 2)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-f256-split-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(proof.initial_cap.as_flattened());
    let log_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_cols_0 + config.log_inv_rates[0]);
    let strat = |level: usize| &config.stratified[level];
    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;
    let ood_count = |level: usize| config.ood_samples[level];

    let mut tx = 0usize;
    let mut ood_index = 0usize;
    let mut claim_nonce = 0usize;
    let mut consistency_nonce = 0usize;
    let mut query_nonce = 0usize;
    let mut claim = F256::from(target);
    let mut ood_contexts = Vec::new();
    let lane_major = l0_num_lanes < 1usize << initial_k;

    for _ in 0..ood_count(0) {
        let mut point = challenger.sample_f128_vec(log_n);
        if lane_major {
            // The reused integer-lane L0 commitment folds the high variables
            // first.  Its residual coordinate order is therefore
            // `[high variables | low variables]`, matching the rotation used
            // for the opening basis in `pcs::verify_opening_batch_*`.
            point.rotate_left(log_n - initial_k);
        }
        let Some(&y) = proof.ood_values.get(ood_index) else {
            return false;
        };
        ood_index += 1;
        challenger.observe_f128(y);
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(0)) else {
            return false;
        };
        claim_nonce += 1;
        claim += F256::from(beta * y);
        ood_contexts.push(OodResidualContext {
            point,
            beta,
            split_level: None,
        });
    }

    let Some(&first) = proof.sumcheck_transcript_f256.get(tx) else {
        return false;
    };
    tx += 1;
    observe_message(challenger, first);
    let mut quad = RoundQuad256::from_msg(first, claim);
    let mut initial_challenges = Vec::with_capacity(initial_k);
    for _ in 0..initial_k {
        let challenge = challenger.sample_f256();
        claim = quad.eval(challenge);
        initial_challenges.push(challenge);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        quad = RoundQuad256::from_msg(msg, claim);
    }

    let mut current_split_dim = log_n - initial_k + 1;
    let Some(cap_1) = proof.recursive_caps.first() else {
        return false;
    };
    challenger.observe_bytes(cap_1.as_flattened());
    for _ in 0..ood_count(1) {
        let point = challenger.sample_f128_vec(current_split_dim);
        let Some(&y) = proof.ood_values.get(ood_index) else {
            return false;
        };
        ood_index += 1;
        challenger.observe_f128(y);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        let intro = RoundQuad256::from_msg(msg, F256::from(y));
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(1)) else {
            return false;
        };
        claim_nonce += 1;
        quad = quad.fold(intro, beta);
        claim += F256::from(beta * y);
        ood_contexts.push(OodResidualContext {
            point,
            beta,
            split_level: Some(0),
        });
    }

    let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
        return false;
    };
    let Some(queries_0) = verify_and_sample_queries(
        challenger,
        nonce,
        config.grinding_bits[0] as u32,
        block_len_0,
        config.queries[0],
        strat(0),
    ) else {
        return false;
    };
    query_nonce += 1;
    let Some(&nonce) = proof
        .consistency_batch_grinding_nonces
        .get(consistency_nonce)
    else {
        return false;
    };
    let Some(alpha_0) = challenger.verify_pow_and_sample_f128_vec(
        nonce,
        consistency_bits(1),
        ceil_log2(config.queries[0]),
    ) else {
        return false;
    };
    consistency_nonce += 1;
    if !verify_level_opens(
        &proof.initial_cap,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        l0_num_lanes,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
        strat(0),
    ) {
        return false;
    }
    let enforced_0 = induce_enforced_sum(
        &proof.initial_proof.opened_rows,
        &initial_challenges,
        &alpha_0,
    );
    let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
        return false;
    };
    tx += 1;
    observe_message(challenger, msg);
    let intro = RoundQuad256::from_msg(msg, enforced_0);
    let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
        return false;
    };
    let Some(beta_0) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(1)) else {
        return false;
    };
    claim_nonce += 1;
    quad = quad.fold(intro, beta_0);
    claim += enforced_0 * beta_0;
    let mut consistency_contexts = vec![ConsistencyResidualContext {
        log_cols: log_n - initial_k,
        queries: queries_0,
        alpha: alpha_0,
        beta: beta_0,
        start_level: 0,
    }];

    let mut level_challenges: Vec<Vec<F256>> = Vec::with_capacity(rounds);
    let mut previous_cap = cap_1.as_slice();
    let mut previous_log_lanes = config.recursive_ks[0];
    if current_split_dim < previous_log_lanes {
        return false;
    }
    let mut previous_log_cols = current_split_dim - previous_log_lanes;
    let mut previous_rate = config.log_inv_rates[1];
    let mut cap_index = 1usize;
    let mut proof_index = 0usize;

    for i in 0..rounds {
        let k = config.recursive_ks[i];
        if current_split_dim < k {
            return false;
        }
        let mut challenges = Vec::with_capacity(k);
        for _ in 0..k {
            let challenge = challenger.sample_f256();
            claim = quad.eval(challenge);
            challenges.push(challenge);
            let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
                return false;
            };
            tx += 1;
            observe_message(challenger, msg);
            quad = RoundQuad256::from_msg(msg, claim);
        }
        level_challenges.push(challenges);
        let extension_dim = current_split_dim - k;
        let level = i + 1;

        if i + 1 == rounds {
            if tx != proof.sumcheck_transcript_f256.len()
                || proof.final_proof.yr.len() != 2usize << extension_dim
            {
                return false;
            }
            for &value in &proof.final_proof.yr {
                challenger.observe_f128(value);
            }
            let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
                return false;
            };
            let block_len = 1usize << (previous_log_cols + previous_rate);
            let Some(queries) = verify_and_sample_queries(
                challenger,
                nonce,
                config.grinding_bits[level] as u32,
                block_len,
                config.queries[level],
                strat(level),
            ) else {
                return false;
            };
            query_nonce += 1;
            let Some(&nonce) = proof
                .consistency_batch_grinding_nonces
                .get(consistency_nonce)
            else {
                return false;
            };
            let Some(alpha) = challenger.verify_pow_and_sample_f128_vec(
                nonce,
                consistency_bits(level),
                ceil_log2(config.queries[level]),
            ) else {
                return false;
            };
            consistency_nonce += 1;
            if !verify_level_opens(
                previous_cap,
                block_len,
                &queries,
                &proof.final_proof.opened_rows,
                1usize << previous_log_lanes,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
                strat(level),
            ) {
                return false;
            }
            let enforced =
                induce_enforced_sum(&proof.final_proof.opened_rows, &level_challenges[i], &alpha);
            let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
                return false;
            };
            let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level)) else {
                return false;
            };
            claim_nonce += 1;
            claim += enforced * beta;
            consistency_contexts.push(ConsistencyResidualContext {
                log_cols: extension_dim,
                queries,
                alpha,
                beta,
                start_level: rounds,
            });

            let original_challenges =
                residual_original_challenges(&initial_challenges, &level_challenges, 0);
            if original_challenges.len() + extension_dim != log_n {
                return false;
            }
            let mut residual = eval_b_residual(&original_challenges, extension_dim);
            if residual.len() != 1usize << extension_dim {
                return false;
            }
            let initial_coordinate_scale = coordinate_factor_product(&level_challenges, 0);
            for value in &mut residual {
                *value *= initial_coordinate_scale;
            }

            for context in &ood_contexts {
                let (fixed, coordinate_scale) = match context.split_level {
                    None => (
                        original_challenges.clone(),
                        coordinate_factor_product(&level_challenges, 0),
                    ),
                    Some(split_level) => {
                        let mut fixed = level_challenges[split_level].clone();
                        for later in &level_challenges[split_level + 1..] {
                            fixed.extend_from_slice(&later[1..]);
                        }
                        (
                            fixed,
                            coordinate_factor_product(&level_challenges, split_level + 1),
                        )
                    }
                };
                let values = eq_residual(
                    &context.point,
                    &fixed,
                    extension_dim,
                    coordinate_scale * context.beta,
                );
                for (dst, value) in residual.iter_mut().zip(values) {
                    *dst += value;
                }
            }

            for context in &consistency_contexts {
                let fixed = if context.start_level == 0 {
                    residual_original_challenges(&[], &level_challenges, context.start_level)
                } else {
                    residual_original_challenges(&[], &level_challenges, context.start_level)
                };
                let values = induced_basis_at_residual(
                    context.log_cols,
                    &context.queries,
                    &context.alpha,
                    &fixed,
                    extension_dim,
                );
                let scale = coordinate_factor_product(&level_challenges, context.start_level)
                    * context.beta;
                for (dst, value) in residual.iter_mut().zip(values) {
                    *dst += value * scale;
                }
            }

            let final_basis = split_basis(&residual);
            let residual_claim = split_inner_product(&proof.final_proof.yr, &final_basis);
            return residual_claim == claim
                && ood_index == proof.ood_values.len()
                && query_nonce == proof.grinding_nonces.len()
                && claim_nonce == proof.claim_batch_grinding_nonces.len()
                && consistency_nonce == proof.consistency_batch_grinding_nonces.len()
                && cap_index == proof.recursive_caps.len()
                && proof_index == proof.recursive_proofs.len();
        }

        current_split_dim = extension_dim + 1;
        let Some(cap) = proof.recursive_caps.get(cap_index) else {
            return false;
        };
        cap_index += 1;
        challenger.observe_bytes(cap.as_flattened());
        for _ in 0..ood_count(level + 1) {
            let point = challenger.sample_f128_vec(current_split_dim);
            let Some(&y) = proof.ood_values.get(ood_index) else {
                return false;
            };
            ood_index += 1;
            challenger.observe_f128(y);
            let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
                return false;
            };
            tx += 1;
            observe_message(challenger, msg);
            let intro = RoundQuad256::from_msg(msg, F256::from(y));
            let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
                return false;
            };
            let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level + 1))
            else {
                return false;
            };
            claim_nonce += 1;
            quad = quad.fold(intro, beta);
            claim += F256::from(beta * y);
            ood_contexts.push(OodResidualContext {
                point,
                beta,
                split_level: Some(i + 1),
            });
        }

        let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
            return false;
        };
        let block_len = 1usize << (previous_log_cols + previous_rate);
        let Some(queries) = verify_and_sample_queries(
            challenger,
            nonce,
            config.grinding_bits[level] as u32,
            block_len,
            config.queries[level],
            strat(level),
        ) else {
            return false;
        };
        query_nonce += 1;
        let Some(&nonce) = proof
            .consistency_batch_grinding_nonces
            .get(consistency_nonce)
        else {
            return false;
        };
        let Some(alpha) = challenger.verify_pow_and_sample_f128_vec(
            nonce,
            consistency_bits(level + 1),
            ceil_log2(config.queries[level]),
        ) else {
            return false;
        };
        consistency_nonce += 1;
        let Some(opening) = proof.recursive_proofs.get(proof_index) else {
            return false;
        };
        proof_index += 1;
        if !verify_level_opens(
            previous_cap,
            block_len,
            &queries,
            &opening.opened_rows,
            1usize << previous_log_lanes,
            &opening.merkle_proof,
            config.merkle_hash,
            strat(level),
        ) {
            return false;
        }
        let enforced = induce_enforced_sum(&opening.opened_rows, &level_challenges[i], &alpha);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        let intro = RoundQuad256::from_msg(msg, enforced);
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level + 1)) else {
            return false;
        };
        claim_nonce += 1;
        quad = quad.fold(intro, beta);
        claim += enforced * beta;
        consistency_contexts.push(ConsistencyResidualContext {
            log_cols: extension_dim,
            queries,
            alpha,
            beta,
            start_level: i + 1,
        });

        previous_cap = cap;
        previous_log_lanes = config.recursive_ks[i + 1];
        if current_split_dim < previous_log_lanes {
            return false;
        }
        previous_log_cols = current_split_dim - previous_log_lanes;
        previous_rate = config.log_inv_rates[i + 2];
    }
    false
}

/// Evaluate queried base-field rows at extension-valued lane challenges and
/// batch the queried columns by the base-field `alpha` point.
pub(super) fn induce_enforced_sum(
    opened_rows: &[Vec<F128>],
    lane_challenges: &[F256],
    alpha: &[F128],
) -> F256 {
    let lane_weights = build_eq_table256(lane_challenges);
    let row_weights = build_eq_table(alpha);
    opened_rows
        .par_iter()
        .zip(row_weights.par_iter())
        .map(|(row, &row_weight)| {
            // The reused L0 commitment may contain only the live lanes of a
            // non-power-of-two packed batch.  The omitted logical lanes are
            // zero, so pairing the committed prefix with the corresponding
            // equality weights is exactly the padded dot product.  Recursive
            // commitments are power-of-two and therefore use every weight.
            assert!(row.len() <= lane_weights.len());
            row.iter()
                .zip(&lane_weights)
                .fold(F256::ZERO, |acc, (&word, &weight)| acc + weight * word)
                * row_weight
        })
        .reduce(|| F256::ZERO, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::{Challenger, RandomChallenger};

    fn random_f256(challenger: &mut RandomChallenger) -> F256 {
        F256::new(challenger.sample_f128(), challenger.sample_f128())
    }

    #[test]
    fn coordinate_split_preserves_every_linear_claim() {
        let mut rng = RandomChallenger::new(0xC001_D1A7);
        for log_n in 0..8 {
            let n = 1usize << log_n;
            let values: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let basis: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let expected = values
                .iter()
                .zip(&basis)
                .fold(F256::ZERO, |acc, (&f, &b)| acc + f * b);
            let words = split_coordinates(&values);
            let weights = split_basis(&basis);
            assert_eq!(split_inner_product(&words, &weights), expected);
        }
    }

    #[test]
    fn virtual_eq_basis_matches_dense_extension_folds() {
        let mut rng = RandomChallenger::new(0xF256_BA51);
        for log_n in 6..11 {
            for initial_k in 1..=4 {
                let log_cols = log_n - initial_k;
                let block = 1usize << log_cols;
                let point: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let gamma = rng.sample_f128();
                let point_ood: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let beta = rng.sample_f128();
                let mut dense: Vec<F256> = build_eq_table(&point)
                    .into_iter()
                    .zip(build_eq_table(&point_ood))
                    .map(|(value, ood)| F256::from(gamma * value + beta * ood))
                    .collect();
                let mut base = VirtualEqBasis::new(point, gamma);
                base.add_term(point_ood, beta);
                let mut virtual_basis = VirtualEqBasis256::from_base(base);
                for _ in 0..initial_k {
                    let r = random_f256(&mut rng);
                    dense = fold_extension(&dense, r, block);
                    virtual_basis.fold_coord(log_cols, r);
                    assert_eq!(
                        virtual_basis.materialize(),
                        dense,
                        "log_n={log_n}, initial_k={initial_k}"
                    );
                }
            }
        }
    }

    #[test]
    fn split_ood_answer_is_a_single_base_field_value() {
        let mut rng = RandomChallenger::new(0x00D5_0127);
        let values: Vec<F256> = (0..8).map(|_| random_f256(&mut rng)).collect();
        let words = split_coordinates(&values);
        let z: Vec<F128> = (0..4).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let answer = words
            .iter()
            .zip(eq)
            .fold(F128::ZERO, |acc, (&word, weight)| acc + word * weight);
        // The committed MLE is base-valued even though its coordinate-weighted
        // claim reconstructs an extension value.
        assert_eq!(F256::from(answer).c1, F128::ZERO);
    }

    #[test]
    fn queried_consistency_uses_all_coordinate_rows() {
        let mut rng = RandomChallenger::new(0xC05E_157E);
        let lane_point: Vec<F256> = (0..3).map(|_| random_f256(&mut rng)).collect();
        let alpha: Vec<F128> = (0..2).map(|_| rng.sample_f128()).collect();
        let rows: Vec<Vec<F128>> = (0..4)
            .map(|_| (0..8).map(|_| rng.sample_f128()).collect())
            .collect();
        let lane_weights = build_eq_table256(&lane_point);
        let row_weights = build_eq_table(&alpha);
        let expected = rows
            .iter()
            .zip(row_weights)
            .fold(F256::ZERO, |outer, (row, row_weight)| {
                outer
                    + row
                        .iter()
                        .zip(&lane_weights)
                        .fold(F256::ZERO, |inner, (&word, &weight)| inner + weight * word)
                        * row_weight
            });
        assert_eq!(induce_enforced_sum(&rows, &lane_point, &alpha), expected);
    }

    #[test]
    fn each_split_level_removes_k_minus_one_original_variables() {
        for dimension in 4..20 {
            for k in 2..=dimension {
                let committed_dimension = dimension + 1;
                let after_folds = committed_dimension - k;
                assert_eq!(after_folds, dimension - (k - 1));
            }
        }
    }
}
