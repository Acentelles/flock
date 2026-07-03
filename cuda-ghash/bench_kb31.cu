// Benchmark SP1's *actual* current KoalaBear degree-4 extension multiply
// (succinctlabs/sp1, sp1-gpu/crates/sys/include/fields/kb31_extension_t.cuh):
// fused 16-mul schoolbook with lazy/batched Montgomery reduction, mul.wide.u32.
//
// Compares against my hand-ports (monty31.cuh: schoolbook & karatsuba) and the
// GHASH numbers. Same harness (4x ILP throughput, single-thread latency).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "fields/kb31_extension_t.cuh"   // -Isp1

typedef unsigned int u32; typedef unsigned long long u64;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static const u64 GOLD = 0x9E3779B97F4A7C15ull;
static const u32 KBP = 0x7f000001u, KBW = 3u;

// Independent (non-Montgomery) reference for KoalaBear Fp4, poly X^4 - 3.
struct Ref4 { u32 c[4]; };
__host__ Ref4 ref_mul(const Ref4& a, const Ref4& b) {
    u64 p[4] = {0,0,0,0};
    for (int i=0;i<4;i++) for (int j=0;j<4;j++){
        u64 t = ((u64)a.c[i]*b.c[j]) % KBP;
        if (i+j>=4) p[i+j-4] = (p[i+j-4] + t*KBW) % KBP;
        else        p[i+j]   = (p[i+j]   + t) % KBP;
    }
    Ref4 r; for(int k=0;k<4;k++) r.c[k]=(u32)p[k]; return r;
}

// Multiply and write the result back as canonical u32[4] (as_canonical_u32 is
// device-only, so do the conversion here rather than on the host).
__global__ void k_mul_once(const kb31_extension_t* a, const kb31_extension_t* b,
                           u32* canon, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ kb31_extension_t r=a[i]*b[i];
        for(int k=0;k<4;k++) canon[i*4+k]=r.value[k].as_canonical_u32(); }
}

template<int K> __global__ void k_tp(kb31_extension_t delta, int iters, kb31_extension_t* o){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    kb31_extension_t a[K];
    #pragma unroll
    for(int k=0;k<K;k++) a[k]=kb31_extension_t((int)((t*7u+k*13u+1u)%KBP),(int)((k*101u+5u)%KBP),
                                               (int)((t+k*3u+7u)%KBP),(int)((k*17u+9u)%KBP));
    kb31_extension_t b((int)((t+1u)%KBP), 0x5678, 0x1234, 0x4321);
    for(int i=0;i<iters;i++){
        #pragma unroll
        for(int k=0;k<K;k++) a[k]=a[k]*b;
        b=b+delta;
    }
    kb31_extension_t s=a[0];
    #pragma unroll
    for(int k=1;k<K;k++) s=s+a[k];
    o[t]=s;
}

__global__ void k_lat(kb31_extension_t a, kb31_extension_t b, kb31_extension_t delta, int iters, kb31_extension_t* o){
    for(int i=0;i<iters;i++){ a=a*b; b=b+delta; }
    o[0]=a;
}

struct Timer{cudaEvent_t s,e;Timer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    void start(){cudaEventRecord(s);} float stop(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}};
template<class F> float best(F fn){fn();CK(cudaDeviceSynchronize());Timer t;float b=1e30f;
    for(int r=0;r<3;r++){t.start();fn();float m=t.stop();CK(cudaGetLastError());if(m<b)b=m;}return b;}

bool validate(int n, kb31_extension_t* scratch){
    std::vector<kb31_extension_t> ha(n,kb31_extension_t(0,0,0,0)),hb(n,kb31_extension_t(0,0,0,0));
    std::vector<u32> hc(n*4); std::vector<Ref4> ra(n),rb(n);
    u64 st=0xABCDEF;
    auto rnd=[&](){st+=GOLD;u64 z=st;z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;return (u32)((z^(z>>31))%KBP);};
    for(int i=0;i<n;i++){
        u32 a0=rnd(),a1=rnd(),a2=rnd(),a3=rnd(),b0=rnd(),b1=rnd(),b2=rnd(),b3=rnd();
        ha[i]=kb31_extension_t((int)a0,(int)a1,(int)a2,(int)a3);
        hb[i]=kb31_extension_t((int)b0,(int)b1,(int)b2,(int)b3);
        ra[i]={a0,a1,a2,a3}; rb[i]={b0,b1,b2,b3};
    }
    kb31_extension_t *da,*db; u32* dc; size_t bytes=n*sizeof(kb31_extension_t);
    CK(cudaMalloc(&da,bytes));CK(cudaMalloc(&db,bytes));CK(cudaMalloc(&dc,n*4*sizeof(u32)));
    CK(cudaMemcpy(da,ha.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,hb.data(),bytes,cudaMemcpyHostToDevice));
    k_mul_once<<<(n+255)/256,256>>>(da,db,dc,n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hc.data(),dc,n*4*sizeof(u32),cudaMemcpyDeviceToHost));
    int bad=0;
    for(int i=0;i<n;i++){ Ref4 w=ref_mul(ra[i],rb[i]);
        for(int k=0;k<4;k++) if(hc[i*4+k]!=w.c[k]){bad++;break;} }
    cudaFree(da);cudaFree(db);cudaFree(dc);
    printf("  validation vs independent ref: %s (%d/%d)\n", bad?"FAIL":"OK", n-bad, n);
    return bad==0;
}

template<int K> void tp(int blk,int tpb,long thr,kb31_extension_t* o){
    kb31_extension_t delta(7,11,13,17);
    long target=2'000'000'000L; int iters=(int)(target/(thr*K)); if(iters<1)iters=1;
    double ops=(double)thr*iters*K;
    float ms=best([&]{k_tp<K><<<blk,tpb>>>(delta,iters,o);});
    printf("  throughput ILP=%-2d: %8.2f GMul/s   %.3f ns/op(agg)\n",K,ops/(ms*1e6),ms*1e6/ops);
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    int sm=p.multiProcessorCount,tpb=256,blk=sm*32; long thr=(long)blk*tpb;
    printf("Device: %s | %d SMs | %ld threads\n",p.name,sm,thr);
    printf("== SP1 current KoalaBear Fp4 (kb31_extension_t, fused schoolbook + lazy reduction) ==\n");
    kb31_extension_t* o; CK(cudaMalloc(&o,thr*sizeof(kb31_extension_t)));
    if(!validate(100000,o)) { printf("*** VALIDATION FAILED ***\n"); return 1; }
    tp<4>(blk,tpb,thr,o); tp<8>(blk,tpb,thr,o); tp<16>(blk,tpb,thr,o);
    kb31_extension_t a0(3,5,7,9), b0(2,4,6,8), delta(7,11,13,17);
    int li=4'000'000;
    float lm=best([&]{k_lat<<<1,1>>>(a0,b0,delta,li,o);});
    printf("  latency (single thread): %8.2f ns/op  [%d iters]\n", lm*1e6/li, li);
    kb31_extension_t h(0,0,0,0); CK(cudaMemcpy(&h,o,sizeof(h),cudaMemcpyDeviceToHost));
    printf("  checksum c0(montgomery raw)=%u\n", h.value[0].val);
    return 0;
}
