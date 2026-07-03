// Benchmark + validation for the KoalaBear / BabyBear degree-4 extension
// multiply (faithful port of SP1's bb31_extension_t), to compare against the
// GHASH GF(2^128) multiply on the same GPU. Same methodology as bench_f128.cu.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "monty31.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

static const u64 GOLD = 0x9E3779B97F4A7C15ull;

// File-scope SplitMix64 (a local struct referencing a function param crashes
// nvcc 13.3 cudafe++; it's also ill-formed C++).
struct SplitMix { u64 s; u64 nx(){ s += GOLD; u64 z=s;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31);} };

// V=0 schoolbook (SP1), V=1 karatsuba.
template<int V, class F> __device__ __forceinline__ Ext4<F> ext_mul(const Ext4<F>& a, const Ext4<F>& b) {
    if (V == 0) return a * b; else return ext4_karatsuba(a, b);
}

// ---- validation: device single-mul vs independent host reference ----
template<int V, class EXT>
__global__ void k_mul_once(const EXT* a, const EXT* b, EXT* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ext_mul<V>(a[i], b[i]);
}

template<int V, class EXT>
bool validate(u32 P, u32 W, const char* name) {
    const int n = 100000;
    SplitMix rng{0x1234};
    std::vector<EXT> ha(n), hb(n), ho(n);
    std::vector<Ref4> ra(n), rb(n);
    for (int i = 0; i < n; i++) {
        u32 a0=rng.nx()%P,a1=rng.nx()%P,a2=rng.nx()%P,a3=rng.nx()%P;
        u32 b0=rng.nx()%P,b1=rng.nx()%P,b2=rng.nx()%P,b3=rng.nx()%P;
        ha[i]=EXT::from_canonical(a0,a1,a2,a3); hb[i]=EXT::from_canonical(b0,b1,b2,b3);
        ra[i]={a0,a1,a2,a3}; rb[i]={b0,b1,b2,b3};
    }
    EXT *da,*db,*dout; size_t bytes=n*sizeof(EXT);
    CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&db,bytes)); CK(cudaMalloc(&dout,bytes));
    CK(cudaMemcpy(da,ha.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,hb.data(),bytes,cudaMemcpyHostToDevice));
    k_mul_once<V,EXT><<<(n+255)/256,256>>>(da,db,dout,n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ho.data(),dout,bytes,cudaMemcpyDeviceToHost));
    int bad=0;
    for (int i=0;i<n;i++){
        Ref4 want=ref_mul(ra[i],rb[i],P,W);
        for(int k=0;k<4;k++) if(ho[i].c[k].to_canonical()!=want.c[k]) { bad++; break; }
    }
    cudaFree(da);cudaFree(db);cudaFree(dout);
    printf("  %-22s validation: %s (%d/%d)\n", name, bad?"FAIL":"OK", n-bad, n);
    return bad==0;
}

// ---- throughput: 4x ILP per thread x full grid ----
template<class EXT, int V>
__global__ void k_tp(EXT delta, int iters, EXT* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    EXT a0 = EXT::from_canonical(tid*2u+1, 0xABCDu, 0x11u, 0x77u);
    EXT a1 = EXT::from_canonical(0xDEADu, tid+3u, 0x22u, 0x55u);
    EXT a2 = EXT::from_canonical(0xBEEFu, 0x99u, tid+5u, 0x33u);
    EXT a3 = EXT::from_canonical(0xF00Du, 0x44u, 0x66u, tid+7u);
    EXT b  = EXT::from_canonical(0x1234u, 0x5678u, 0x9ABCu, 0xDEF0u);
    for (int i = 0; i < iters; i++) {
        a0 = ext_mul<V>(a0,b); a1 = ext_mul<V>(a1,b);
        a2 = ext_mul<V>(a2,b); a3 = ext_mul<V>(a3,b);
        b = b + delta;                  // vary b so nothing hoists
    }
    out[tid] = (a0 + a1) + (a2 + a3);
}

template<class EXT, int V>
__global__ void k_lat(EXT a, EXT b, EXT delta, int iters, EXT* out) {
    for (int i = 0; i < iters; i++) { a = ext_mul<V>(a,b); b = b + delta; }
    out[0] = a;
}

struct Timer { cudaEvent_t s,e; Timer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~Timer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(){cudaEventRecord(s);} float stop(){cudaEventRecord(e);cudaEventSynchronize(e);
    float ms;cudaEventElapsedTime(&ms,s,e);return ms;} };

template<class F> static float best(F fn){ fn();CK(cudaDeviceSynchronize());
    Timer t;float b=1e30f;for(int r=0;r<3;r++){t.start();fn();float m=t.stop();CK(cudaGetLastError());if(m<b)b=m;}return b; }

template<class EXT, int V>
void bench(int blocks,int tpb,EXT* out,const char* name,double nmul){
    long threads=(long)blocks*tpb;
    EXT delta = EXT::from_canonical(7,11,13,17);
    // throughput
    const long target=2'000'000'000L; int iters=(int)(target/(threads*4)); if(iters<1)iters=1;
    double ops=(double)threads*iters*4;
    float ms=best([&]{ k_tp<EXT,V><<<blocks,tpb>>>(delta,iters,out); });
    printf("  %-22s throughput: %8.2f GExt4Mul/s  (%6.1f Gbasemul/s)  %.3f ns/op(agg)\n",
           name, ops/(ms*1e6), ops*nmul/(ms*1e6), ms*1e6/ops);
    // latency
    const int li=4'000'000;
    EXT a0=EXT::from_canonical(3,5,7,9), b0=EXT::from_canonical(2,4,6,8);
    float lm=best([&]{ k_lat<EXT,V><<<1,1>>>(a0,b0,delta,li,out); });
    printf("  %-22s latency:    %8.2f ns/op  [%d iters]\n", name, lm*1e6/li, li);
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    int sm=p.multiProcessorCount; int tpb=256, blocks=sm*32; long threads=(long)blocks*tpb;
    printf("Device: %s | %d SMs | %ld threads\n\n", p.name, sm, threads);

    printf("== Validation vs independent reference ==\n");
    bool ok=true;
    ok &= validate<0,BabyBear4>(0x78000001u,11u,"BabyBear schoolbook");
    ok &= validate<1,BabyBear4>(0x78000001u,11u,"BabyBear karatsuba");
    ok &= validate<0,KoalaBear4>(0x7f000001u,3u,"KoalaBear schoolbook");
    ok &= validate<1,KoalaBear4>(0x7f000001u,3u,"KoalaBear karatsuba");
    if(!ok){ printf("\n*** VALIDATION FAILED ***\n"); return 1; }

    void* outv; CK(cudaMalloc(&outv, threads*sizeof(KoalaBear4)));
    printf("\n== Degree-4 extension multiply (schoolbook=22 mults, karatsuba=13) ==\n");
    bench<BabyBear4,0>(blocks,tpb,(BabyBear4*)outv,"BabyBear schoolbook",22);
    bench<BabyBear4,1>(blocks,tpb,(BabyBear4*)outv,"BabyBear karatsuba",13);
    bench<KoalaBear4,0>(blocks,tpb,(KoalaBear4*)outv,"KoalaBear schoolbook",22);
    bench<KoalaBear4,1>(blocks,tpb,(KoalaBear4*)outv,"KoalaBear karatsuba",13);

    KoalaBear4 h; CK(cudaMemcpy(&h,outv,sizeof(h),cudaMemcpyDeviceToHost));
    printf("\nchecksum c0 = %u\n", h.c[0].to_canonical());
    return 0;
}
