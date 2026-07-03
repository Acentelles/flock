// Real-world sumcheck inner product  S = sum_i a(i)*b(i)  compared between
// GHASH GF(2^128) (deferred-reduction Karatsuba) and SP1 KoalaBear Fp4.
//
// Two regimes:
//   compute-only : operands synthesized in registers  -> pure multiply ceiling
//   memory-fed   : a(i),b(i) streamed from VRAM (32 B/term) -> realistic roofline
//
// Each element is 16 bytes in both fields, so the memory cost per term is
// identical -> a fair apples-to-apples streaming comparison.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "f128.cuh"                      // GHASH
#include "fields/kb31_extension_t.cuh"   // SP1 KoalaBear (-Isp1, c++20)

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
typedef unsigned int u32;
static const u64 GOLD=0x9E3779B97F4A7C15ull;
static const u32 KBP=0x7f000001u;
__host__ __device__ u64 sm64(u64 z){z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;return z^(z>>31);}

// ---------- fills ----------
__global__ void ghash_fill(F128* a,F128* b,long N){ long i=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x;
  for(;i<N;i+=s){ a[i]=F128{sm64(i*2+1),sm64(i*2+2)}; b[i]=F128{sm64(i*2+3),sm64(i*2+4)}; } }
__global__ void kb_fill(kb31_extension_t* a,kb31_extension_t* b,long N){ long i=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x;
  for(;i<N;i+=s){ a[i]=kb31_extension_t((int)(sm64(i*2+1)%KBP),(int)(sm64(i*2+5)%KBP),(int)(sm64(i*2+9)%KBP),(int)(sm64(i*2+13)%KBP));
                  b[i]=kb31_extension_t((int)(sm64(i*2+3)%KBP),(int)(sm64(i*2+7)%KBP),(int)(sm64(i*2+11)%KBP),(int)(sm64(i*2+15)%KBP)); } }

// ---------- GHASH dot ----------
__global__ void ghash_mem(const F128* a,const F128* b,long N,int R,F256* part){
  long t=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x; F256 acc{0,0,0,0};
  for(int r=0;r<R;r++) for(long i=t;i<N;i+=s){ f256_xor(acc, mul_unreduced_karatsuba(a[i],b[i])); }
  part[t]=acc;
}
__global__ void ghash_compute(long T,F256* part){
  long t=blockIdx.x*blockDim.x+threadIdx.x; F256 acc{0,0,0,0};
  F128 a{0x9E3779B9u*(u64)t+1,0x1234u^t}, b{0xC2B2AE35u*(u64)t+7,0x5678u^t};
  for(long i=0;i<T;i++){ f256_xor(acc, mul_unreduced_karatsuba(a,b)); a.lo+=GOLD; a.hi+=0xC2B2AE3527D4EB2Full; b.lo+=0xD1B54A32D192ED03ull; b.hi+=GOLD; }
  part[t]=acc;
}

// ---------- KoalaBear dot ----------
__global__ void kb_mem(const kb31_extension_t* a,const kb31_extension_t* b,long N,int R,kb31_extension_t* part){
  long t=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x; kb31_extension_t acc=kb31_extension_t::zero();
  for(int r=0;r<R;r++) for(long i=t;i<N;i+=s){ acc=acc+a[i]*b[i]; }
  part[t]=acc;
}
__global__ void kb_compute(long T,kb31_extension_t* part){
  long t=blockIdx.x*blockDim.x+threadIdx.x; kb31_extension_t acc=kb31_extension_t::zero();
  kb31_extension_t a((int)((t*7u+1u)%KBP),(int)((t+5u)%KBP),11,17), b((int)((t*3u+7u)%KBP),13,(int)((t+9u)%KBP),19),
                   da(7,11,13,17), db(19,23,29,31);
  for(long i=0;i<T;i++){ acc=acc+a*b; a=a+da; b=b+db; }
  part[t]=acc;
}

struct Timer{cudaEvent_t s,e;Timer(){cudaEventCreate(&s);cudaEventCreate(&e);} void start(){cudaEventRecord(s);}
  float stop(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}};
template<class F> float best(F fn){fn();CK(cudaDeviceSynchronize());Timer t;float b=1e30f;
  for(int r=0;r<3;r++){t.start();fn();float m=t.stop();CK(cudaGetLastError());if(m<b)b=m;}return b;}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int sm=p.multiProcessorCount,tpb=256,blk=sm*32; long thr=(long)blk*tpb;
  double bw_peak = 1792.0; // RTX 5090: GDDR7 28 Gbps x 512-bit / 8 = 1792 GB/s
  printf("Device: %s | %d SMs | peak VRAM BW ~%.0f GB/s\n",p.name,sm,bw_peak);
  printf("Per term: 2 x 16B operand loads = 32 B  ->  memory caps terms/s at ~%.1f G/s\n\n", bw_peak/32.0);

  const long N=1L<<26;                 // 67.1M terms; arrays 1 GiB each
  void *dA,*dB; CK(cudaMalloc(&dA,N*16)); CK(cudaMalloc(&dB,N*16));
  void* dpart; CK(cudaMalloc(&dpart, thr*sizeof(F256)));
  const int Rmem=8; const long Tc=768;
  double mem_terms=(double)N*Rmem, comp_terms=(double)thr*Tc;

  // ---- GHASH ----
  ghash_fill<<<blk,tpb>>>((F128*)dA,(F128*)dB,N); CK(cudaDeviceSynchronize());
  float gm=best([&]{ ghash_mem<<<blk,tpb>>>((F128*)dA,(F128*)dB,N,Rmem,(F256*)dpart); });
  float gc=best([&]{ ghash_compute<<<blk,tpb>>>(Tc,(F256*)dpart); });
  printf("GHASH GF(2^128)  (deferred Karatsuba, 6 CLMAD/term)\n");
  printf("  compute-only : %7.2f G-terms/s\n", comp_terms/(gc*1e6));
  printf("  memory-fed   : %7.2f G-terms/s   (%6.0f GB/s)\n\n", mem_terms/(gm*1e6), mem_terms*32/(gm*1e6));

  // ---- KoalaBear ----
  kb_fill<<<blk,tpb>>>((kb31_extension_t*)dA,(kb31_extension_t*)dB,N); CK(cudaDeviceSynchronize());
  float km=best([&]{ kb_mem<<<blk,tpb>>>((kb31_extension_t*)dA,(kb31_extension_t*)dB,N,Rmem,(kb31_extension_t*)dpart); });
  float kc=best([&]{ kb_compute<<<blk,tpb>>>(Tc,(kb31_extension_t*)dpart); });
  printf("KoalaBear Fp4    (SP1 kb31, mul+add)\n");
  printf("  compute-only : %7.2f G-terms/s\n", comp_terms/(kc*1e6));
  printf("  memory-fed   : %7.2f G-terms/s   (%6.0f GB/s)\n", mem_terms/(km*1e6), mem_terms*32/(km*1e6));
  return 0;
}
