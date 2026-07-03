// 31-bit Montgomery prime field + degree-4 binomial extension on CUDA.
//
// Faithful port of SP1's GPU extension multiply (succinctlabs/sp1-gpu,
// cuda/fields/bb31_extension_t.cuh): schoolbook 16 base mults + 6 `xW` wrap
// terms = 22 base-field multiplies, binomial reduction product[i+j-D] += ai*bj*W.
//
// Instantiated for both 31-bit STARK fields:
//   BabyBear  p = 2^31 - 2^27 + 1 = 0x78000001,  X^4 - 11   (SP1 GPU's field)
//   KoalaBear p = 2^31 - 2^24 + 1 = 0x7f000001,  X^4 -  3   (Plonky3, asked about)
// The two differ only in (p, W); instruction count is identical.
#pragma once
#include <cstdint>

typedef unsigned int       u32;
typedef unsigned long long u64;

// Element stored in Montgomery form (v = canonical * 2^32 mod P).
// MU = -P^{-1} mod 2^32 and WM = (W * 2^32) mod P are precomputed on the host
// (see consts.cpp) and passed as literal template params: computing them via
// constexpr *inside* the template segfaults nvcc 13.3's cudafe++ front-end.
template<u32 P, u32 MU, u32 WM>
struct Monty31 {
    u32 v;

    // Montgomery reduction: returns (x * 2^-32) mod P for x < P*2^32.
    static __host__ __device__ __forceinline__ u32 reduce(u64 x) {
        u32 m = (u32)x * MU;            // low 32 bits of x*MU
        u64 t = x + (u64)m * P;         // now divisible by 2^32
        u32 r = (u32)(t >> 32);         // < 2P
        return r >= P ? r - P : r;
    }

    static __host__ __device__ __forceinline__ Monty31 zero() { Monty31 r; r.v = 0; return r; }
    static __host__ __device__ __forceinline__ Monty31 W()    { Monty31 r; r.v = WM; return r; }
    static __host__ __device__ __forceinline__ Monty31 from_canonical(u32 a) {
        Monty31 r; r.v = (u32)(((u64)(a % P) << 32) % P); return r;
    }
    __host__ __device__ __forceinline__ u32 to_canonical() const { return reduce((u64)v); }

    __host__ __device__ __forceinline__ Monty31 operator*(Monty31 b) const {
        Monty31 r; r.v = reduce((u64)v * b.v); return r;
    }
    __host__ __device__ __forceinline__ Monty31 operator+(Monty31 b) const {
        u32 s = v + b.v; Monty31 r; r.v = (s >= P) ? s - P : s; return r;
    }
    __host__ __device__ __forceinline__ Monty31 operator-(Monty31 b) const {
        Monty31 r; r.v = (v >= b.v) ? v - b.v : v + P - b.v; return r;
    }
};

// Degree-4 binomial extension  Fp[X]/(X^4 - W).  Verbatim SP1 schoolbook.
template<class F>
struct Ext4 {
    F c[4];

    __host__ __device__ __forceinline__ Ext4 operator*(const Ext4& b) const {
        F prod[4] = {F::zero(), F::zero(), F::zero(), F::zero()};
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                F t = c[i] * b.c[j];
                if (i + j >= 4) prod[i + j - 4] = prod[i + j - 4] + t * F::W();
                else            prod[i + j]     = prod[i + j]     + t;
            }
        }
        Ext4 r;
#pragma unroll
        for (int k = 0; k < 4; k++) r.c[k] = prod[k];
        return r;
    }

    __host__ __device__ __forceinline__ Ext4 operator+(const Ext4& b) const {
        Ext4 r;
#pragma unroll
        for (int k = 0; k < 4; k++) r.c[k] = c[k] + b.c[k];
        return r;
    }

    static __host__ __device__ __forceinline__ Ext4 from_canonical(u32 a, u32 b, u32 cc, u32 d) {
        Ext4 r;
        r.c[0] = F::from_canonical(a); r.c[1] = F::from_canonical(b);
        r.c[2] = F::from_canonical(cc); r.c[3] = F::from_canonical(d);
        return r;
    }
};

// Concrete scalar fields.
typedef Monty31<0x78000001u, 0x77ffffffu, 939524073u> BabyBear;   // SP1 GPU's field, W=11
typedef Monty31<0x7f000001u, 0x7effffffu, 100663290u> KoalaBear;  // Plonky3 field, W=3

// Multiply by the (small) extension constant W. Montgomery form is linear in
// the integer, so x*k is a k-fold add chain — no Montgomery multiply needed.
// Generic fallback uses a full multiply; the two concrete fields specialize.
template<class F> __host__ __device__ __forceinline__ F mul_by_W(F x) { return x * F::W(); }
__host__ __device__ __forceinline__ KoalaBear mul_by_W(KoalaBear x) {   // *3
    KoalaBear x2 = x + x; return x2 + x;
}
__host__ __device__ __forceinline__ BabyBear mul_by_W(BabyBear x) {     // *11 = 8+2+1
    BabyBear x2 = x + x, x4 = x2 + x2, x8 = x4 + x4; return (x8 + x2) + x;
}

// ---------------------------------------------------------------------------
// Karatsuba multiply for Fp4 via the Fp2 tower:  Fp4 = Fp2[X]/(X^2 - u),
// Fp2 = Fp[u]/(u^2 - W).  9 general mults + 4 (xW add-chains), vs schoolbook 22.
// ---------------------------------------------------------------------------
template<class F> struct Fp2 { F a, b; };  // a + b*u,  u^2 = W

template<class F> __host__ __device__ __forceinline__ Fp2<F> fp2_mul(Fp2<F> x, Fp2<F> y) {
    F t0 = x.a * y.a;
    F t1 = x.b * y.b;
    F t2 = (x.a + x.b) * (y.a + y.b);
    F real = t0 + mul_by_W(t1);       // ac + W*bd
    F imag = (t2 - t0) - t1;          // ad + bc
    return Fp2<F>{real, imag};
}

template<class F>
__host__ __device__ __forceinline__ Ext4<F> ext4_karatsuba(const Ext4<F>& a, const Ext4<F>& b) {
    // a = P + Q X,  P = a0 + a2 u, Q = a1 + a3 u  (X^2 = u)
    Fp2<F> P{a.c[0], a.c[2]}, Q{a.c[1], a.c[3]};
    Fp2<F> R{b.c[0], b.c[2]}, S{b.c[1], b.c[3]};
    Fp2<F> T0 = fp2_mul(P, R);
    Fp2<F> T1 = fp2_mul(Q, S);
    Fp2<F> T2 = fp2_mul(Fp2<F>{P.a + Q.a, P.b + Q.b}, Fp2<F>{R.a + S.a, R.b + S.b});
    // real = PR + u*QS = (T0.a + W*T1.b) + (T0.b + T1.a) u
    F real_a = T0.a + mul_by_W(T1.b);
    F real_b = T0.b + T1.a;
    // imag = PS + QR = T2 - T0 - T1
    F imag_a = (T2.a - T0.a) - T1.a;
    F imag_b = (T2.b - T0.b) - T1.b;
    Ext4<F> r;
    r.c[0] = real_a; r.c[2] = real_b;   // X^0, X^2
    r.c[1] = imag_a; r.c[3] = imag_b;   // X^1, X^3
    return r;
}

// Concrete extension fields.
typedef Ext4<BabyBear>  BabyBear4;
typedef Ext4<KoalaBear> KoalaBear4;

// ---------------------------------------------------------------------------
// Independent (non-Montgomery) reference for validation.
// ---------------------------------------------------------------------------
struct Ref4 { u32 c[4]; };
__host__ __forceinline__ Ref4 ref_mul(const Ref4& a, const Ref4& b, u32 P, u32 W) {
    u64 prod[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            u64 t = ((u64)a.c[i] * b.c[j]) % P;
            if (i + j >= 4) prod[i + j - 4] = (prod[i + j - 4] + t * W) % P;
            else            prod[i + j]     = (prod[i + j]     + t) % P;
        }
    Ref4 r; for (int k = 0; k < 4; k++) r.c[k] = (u32)prod[k]; return r;
}
