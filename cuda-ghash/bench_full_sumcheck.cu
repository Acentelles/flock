// Full multilinear sumcheck of  S = sum_{x in {0,1}^n} a(x)*b(x).
// Per round: degree-2 round poly at {0, 1, infinity}, data-dependent challenge,
// then fold both tables in half. log2(N) rounds, tables halving each round.
// Engines: GHASH reduce-per-term, GHASH deferred-reduction, KoalaBear (SP1 kb31).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "f128.cuh"
#include "fields/kb31_extension_t.cuh"

typedef unsigned int u32;
typedef kb31_extension_t KB;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static const u32 KBP=0x7f000001u;
__host__ __device__ u64 sm64(u64 z){z+=0x9E3779B97F4A7C15ull;z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;return z^(z>>31);}

// ============================ GHASH ============================
__global__ void g_fill(F128*A,F128*B,long N){ long i=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x;
  for(;i<N;i+=s){A[i]=F128{sm64(i*2+1),sm64(i*2+2)};B[i]=F128{sm64(i*2+3),sm64(i*2+4)};} }

// reduce-per-term round poly
__global__ void g_round(const F128*A,const F128*B,long half,F128*part){
  __shared__ F128 s0[256],s1[256],s2[256];
  long t=blockIdx.x*blockDim.x+threadIdx.x,st=(long)gridDim.x*blockDim.x;
  F128 e0{0,0},e1{0,0},ei{0,0};
  for(long i=t;i<half;i+=st){ F128 al=A[i],ah=A[i+half],bl=B[i],bh=B[i+half];
    e0=f128_add(e0,ghash_mul_karatsuba(al,bl)); e1=f128_add(e1,ghash_mul_karatsuba(ah,bh));
    F128 dA{ah.lo^al.lo,ah.hi^al.hi},dB{bh.lo^bl.lo,bh.hi^bl.hi};
    ei=f128_add(ei,ghash_mul_karatsuba(dA,dB)); }
  int x=threadIdx.x; s0[x]=e0;s1[x]=e1;s2[x]=ei; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(x<s){s0[x]=f128_add(s0[x],s0[x+s]);s1[x]=f128_add(s1[x],s1[x+s]);s2[x]=f128_add(s2[x],s2[x+s]);} __syncthreads(); }
  if(x==0){part[blockIdx.x*3]=s0[0];part[blockIdx.x*3+1]=s1[0];part[blockIdx.x*3+2]=s2[0];}
}
// deferred round poly: accumulate unreduced 256-bit, reduce 3 sums once
__global__ void g_round_def(const F128*A,const F128*B,long half,F256*part){
  __shared__ F256 s0[256],s1[256],s2[256];
  long t=blockIdx.x*blockDim.x+threadIdx.x,st=(long)gridDim.x*blockDim.x;
  F256 e0{0,0,0,0},e1{0,0,0,0},ei{0,0,0,0};
  for(long i=t;i<half;i+=st){ F128 al=A[i],ah=A[i+half],bl=B[i],bh=B[i+half];
    f256_xor(e0,mul_unreduced_karatsuba(al,bl)); f256_xor(e1,mul_unreduced_karatsuba(ah,bh));
    F128 dA{ah.lo^al.lo,ah.hi^al.hi},dB{bh.lo^bl.lo,bh.hi^bl.hi};
    f256_xor(ei,mul_unreduced_karatsuba(dA,dB)); }
  int x=threadIdx.x; s0[x]=e0;s1[x]=e1;s2[x]=ei; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(x<s){f256_xor(s0[x],s0[x+s]);f256_xor(s1[x],s1[x+s]);f256_xor(s2[x],s2[x+s]);} __syncthreads(); }
  if(x==0){part[blockIdx.x*3]=s0[0];part[blockIdx.x*3+1]=s1[0];part[blockIdx.x*3+2]=s2[0];}
}
__global__ void g_combine(const F128*part,int blocks,F128*r){
  F128 e0{0,0},e1{0,0},ei{0,0};
  for(int b=0;b<blocks;b++){e0=f128_add(e0,part[b*3]);e1=f128_add(e1,part[b*3+1]);ei=f128_add(ei,part[b*3+2]);}
  u64 h=sm64(e0.lo^e1.hi^ei.lo); r[0]=F128{h,sm64(h^e0.hi^e1.lo^ei.hi)};
}
__global__ void g_combine_def(const F256*part,int blocks,F128*r){
  F256 a0{0,0,0,0},a1{0,0,0,0},ai{0,0,0,0};
  for(int b=0;b<blocks;b++){f256_xor(a0,part[b*3]);f256_xor(a1,part[b*3+1]);f256_xor(ai,part[b*3+2]);}
  F128 e0=f256_reduce(a0),e1=f256_reduce(a1),ei=f256_reduce(ai);
  u64 h=sm64(e0.lo^e1.hi^ei.lo); r[0]=F128{h,sm64(h^e0.hi^e1.lo^ei.hi)};
}
__global__ void g_fold(F128*A,F128*B,long half,const F128*rp){
  F128 r=rp[0]; long t=blockIdx.x*blockDim.x+threadIdx.x,st=(long)gridDim.x*blockDim.x;
  for(long i=t;i<half;i+=st){ F128 dA{A[i+half].lo^A[i].lo,A[i+half].hi^A[i].hi},dB{B[i+half].lo^B[i].lo,B[i+half].hi^B[i].hi};
    A[i]=f128_add(A[i],ghash_mul_karatsuba(r,dA)); B[i]=f128_add(B[i],ghash_mul_karatsuba(r,dB)); }
}

// ============================ KoalaBear ============================
__global__ void kb_fill(KB*A,KB*B,long N){ long i=blockIdx.x*blockDim.x+threadIdx.x,s=(long)gridDim.x*blockDim.x;
  for(;i<N;i+=s){A[i]=KB((int)(sm64(i*2+1)%KBP),(int)(sm64(i*2+5)%KBP),(int)(sm64(i*2+9)%KBP),(int)(sm64(i*2+13)%KBP));
                 B[i]=KB((int)(sm64(i*2+3)%KBP),(int)(sm64(i*2+7)%KBP),(int)(sm64(i*2+11)%KBP),(int)(sm64(i*2+15)%KBP));} }
__global__ void kb_round(const KB*A,const KB*B,long half,KB*part){
  __shared__ KB s0[256],s1[256],s2[256];
  long t=blockIdx.x*blockDim.x+threadIdx.x,st=(long)gridDim.x*blockDim.x;
  KB e0=KB::zero(),e1=KB::zero(),ei=KB::zero();
  for(long i=t;i<half;i+=st){ KB al=A[i],ah=A[i+half],bl=B[i],bh=B[i+half];
    e0=e0+al*bl; e1=e1+ah*bh; ei=ei+(ah-al)*(bh-bl); }
  int x=threadIdx.x; s0[x]=e0;s1[x]=e1;s2[x]=ei; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(x<s){s0[x]=s0[x]+s0[x+s];s1[x]=s1[x]+s1[x+s];s2[x]=s2[x]+s2[x+s];} __syncthreads(); }
  if(x==0){part[blockIdx.x*3]=s0[0];part[blockIdx.x*3+1]=s1[0];part[blockIdx.x*3+2]=s2[0];}
}
__global__ void kb_combine(const KB*part,int blocks,KB*r){
  KB e0=KB::zero(),e1=KB::zero(),ei=KB::zero();
  for(int b=0;b<blocks;b++){e0=e0+part[b*3];e1=e1+part[b*3+1];ei=ei+part[b*3+2];}
  u64 h=sm64(e0.value[0].val^e1.value[1].val^ei.value[2].val);
  r[0]=KB((int)(h%KBP),(int)(sm64(h)%KBP),(int)(sm64(h+1)%KBP),(int)(sm64(h+2)%KBP));
}
__global__ void kb_fold(KB*A,KB*B,long half,const KB*rp){
  KB r=rp[0]; long t=blockIdx.x*blockDim.x+threadIdx.x,st=(long)gridDim.x*blockDim.x;
  for(long i=t;i<half;i+=st){ A[i]=A[i]+r*(A[i+half]-A[i]); B[i]=B[i]+r*(B[i+half]-B[i]); }
}

struct Timer{cudaEvent_t s,e;Timer(){cudaEventCreate(&s);cudaEventCreate(&e);} void start(){cudaEventRecord(s);}
  float stop(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}};

int main(int argc,char**argv){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int sm=p.multiProcessorCount,tpb=256,blk=sm*4;
  const int n=argc>1?atoi(argv[1]):24; const long N=1L<<n;
  size_t need=(size_t)N*16*2;
  printf("Device: %s | full sumcheck of sum a(x)b(x), N=2^%d=%ld, %d rounds\n",p.name,n,N,n);
  printf("Two tables need %.2f GB VRAM (%.1f GB free)\n\n", need/1e9, (double)p.totalGlobalMem/1e9);
  void *dA=0,*dB=0,*part,*rdev;
  cudaError_t ea=cudaMalloc(&dA,N*16), eb=cudaMalloc(&dB,N*16);
  if(ea||eb){ printf("*** out of memory for 2^%d (need %.1f GB) — try fewer variables ***\n",n,need/1e9); return 2; }
  CK(cudaMalloc(&part,(size_t)blk*3*32)); CK(cudaMalloc(&rdev,32));
  double total_mults=5.0*(double)(N-1);  // ~5 field mults per index, sum of halves = N-1

  auto time_engine=[&](const char* name,int engine){
    float best=1e30f;
    for(int rep=0;rep<3;rep++){
      if(engine<2) g_fill<<<blk,tpb>>>((F128*)dA,(F128*)dB,N); else kb_fill<<<blk,tpb>>>((KB*)dA,(KB*)dB,N);
      CK(cudaDeviceSynchronize());
      Timer T; T.start();
      for(long M=N;M>1;M>>=1){ long half=M>>1;
        if(engine==0){ g_round<<<blk,tpb>>>((F128*)dA,(F128*)dB,half,(F128*)part); g_combine<<<1,1>>>((F128*)part,blk,(F128*)rdev); g_fold<<<blk,tpb>>>((F128*)dA,(F128*)dB,half,(F128*)rdev); }
        else if(engine==1){ g_round_def<<<blk,tpb>>>((F128*)dA,(F128*)dB,half,(F256*)part); g_combine_def<<<1,1>>>((F256*)part,blk,(F128*)rdev); g_fold<<<blk,tpb>>>((F128*)dA,(F128*)dB,half,(F128*)rdev); }
        else { kb_round<<<blk,tpb>>>((KB*)dA,(KB*)dB,half,(KB*)part); kb_combine<<<1,1>>>((KB*)part,blk,(KB*)rdev); kb_fold<<<blk,tpb>>>((KB*)dA,(KB*)dB,half,(KB*)rdev); }
      }
      float ms=T.stop(); CK(cudaGetLastError()); if(ms<best)best=ms;
    }
    printf("  %-28s %8.3f ms   (%.2f G-mult/s effective)\n",name,best,total_mults/(best*1e6));
  };
  time_engine("GHASH reduce-per-term",0);
  time_engine("GHASH deferred-reduction",1);
  time_engine("KoalaBear Fp4 (SP1 kb31)",2);
  printf("\n(~%.0fM field mults total across all rounds; round 1 alone = %ld indices x 5)\n", total_mults/1e6, N/2);
  return 0;
}
