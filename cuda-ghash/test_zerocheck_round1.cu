// Bit-for-bit validation of the CUDA zerocheck round-1 (univariate-skip URM,
// CANONICAL form) against the flock oracle from dump_zerocheck_round1_vectors.rs
// (ZCR1) — golden = the real optimized prove_packed round-1.
//
// Build:  make test_zerocheck_round1
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "zerocheck_round1.cuh"
#include "phi8_table.cuh"
#include "ntt_host.hpp"

#define CK(x) do { cudaError_t e=(x); if(e){ printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

static uint32_t rd_u32(FILE* f){ uint32_t v=0; if(fread(&v,4,1,f)!=1){printf("short u32\n");exit(1);} return v; }
static F128 rd_f128(FILE* f){ u64 v[2]; if(fread(v,8,2,f)!=2){printf("short f128\n");exit(1);} return F128{v[0],v[1]}; }
static bool eqf(F128 a, F128 b){ return a.lo==b.lo && a.hi==b.hi; }

static std::vector<F128> build_eq_host(const std::vector<F128>& r){
    const F128 ONE{1,0};
    std::vector<F128> t; t.reserve((size_t)1<<r.size()); t.push_back(ONE);
    for(size_t j=0;j<r.size();j++){ F128 rj=r[j], omr=f128_add_hd(ONE,rj); size_t len=(size_t)1<<j; t.resize(2*len);
        for(size_t x=0;x<len;x++){ F128 v=t[x]; t[x+len]=f128_mul_hd(v,rj); t[x]=f128_mul_hd(v,omr);} }
    return t;
}

int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"zerocheck_round1_vectors.bin";
    FILE* f = fopen(path,"rb");
    if(!f){ printf("cannot open %s\n", path); return 1; }
    if(rd_u32(f)!=0x5A435231u){ printf("bad magic\n"); return 1; }
    int m=(int)rd_u32(f), k_skip=(int)rd_u32(f), k_log=(int)rd_u32(f), useful_bits=(int)rd_u32(f);
    long long rows = 1LL << (m - k_skip);
    std::vector<F128> r(m); for(auto&v:r) v=rd_f128(f);
    std::vector<uint8_t> mcol(64*64), f8mul((size_t)256*256);
    if(fread(mcol.data(),1,mcol.size(),f)!=mcol.size()){printf("short M\n");return 1;}
    if(fread(f8mul.data(),1,f8mul.size(),f)!=f8mul.size()){printf("short f8mul\n");return 1;}
    size_t pb = (size_t)1 << (m - 3);
    std::vector<uint8_t> a(pb), b(pb), c(pb);
    if(fread(a.data(),1,pb,f)!=pb||fread(b.data(),1,pb,f)!=pb||fread(c.data(),1,pb,f)!=pb){printf("short abc\n");return 1;}
    std::vector<F128> g_ab(64), g_c(64);
    for(auto&v:g_ab) v=rd_f128(f);
    for(auto&v:g_c) v=rd_f128(f);
    fclose(f);

    printf("ZCR1: m=%d k_skip=%d k_log=%d useful_bits=%d rows=%lld\n", m, k_skip, k_log, useful_bits, rows);

    std::vector<F128> r_rest(r.begin()+k_skip, r.end());
    std::vector<F128> eq_full = build_eq_host(r_rest);
    if((long long)eq_full.size()!=rows){ printf("eq size mismatch\n"); return 1; }

    zc_round1_upload_tables(mcol.data(), f8mul.data(), PHI_8_TABLE);

    uint8_t *d_a,*d_b,*d_c; F128 *d_eq,*d_ab,*d_c_out;
    CK(cudaMalloc(&d_a,pb)); CK(cudaMalloc(&d_b,pb)); CK(cudaMalloc(&d_c,pb));
    CK(cudaMalloc(&d_eq,rows*sizeof(F128)));
    CK(cudaMalloc(&d_ab,64*sizeof(F128))); CK(cudaMalloc(&d_c_out,64*sizeof(F128)));
    CK(cudaMemcpy(d_a,a.data(),pb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b,b.data(),pb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c,c.data(),pb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_eq,eq_full.data(),rows*sizeof(F128),cudaMemcpyHostToDevice));

    launch_zc_round1(d_a, d_b, d_c, d_eq, rows, d_ab, d_c_out);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<F128> ab(64), cc(64);
    CK(cudaMemcpy(ab.data(),d_ab,64*sizeof(F128),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cc.data(),d_c_out,64*sizeof(F128),cudaMemcpyDeviceToHost));
    for(int i=0;i<64;i++){
        if(!eqf(ab[i],g_ab[i])){ printf("AB FAIL [%d]\n",i); return 1; }
        if(!eqf(cc[i],g_c[i])){ printf("C FAIL [%d]\n",i); return 1; }
    }
    printf("ZEROCHECK ROUND-1 OK (canonical): round1_ab[64] + round1_c[64] match flock bit-for-bit\n");
    return 0;
}
