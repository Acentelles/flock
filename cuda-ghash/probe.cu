#include <cstdio>
#include "fields/kb31_extension_t.cuh"
__global__ void probe(){
  kb31_extension_t a(5,7,9,11);
  printf("roundtrip a=(5,7,9,11) -> (%u,%u,%u,%u)\n",
    a.value[0].as_canonical_u32(),a.value[1].as_canonical_u32(),a.value[2].as_canonical_u32(),a.value[3].as_canonical_u32());
  kb31_extension_t one(1,0,0,0);
  kb31_extension_t c=a*one;
  printf("a*one -> (%u,%u,%u,%u)  [expect 5,7,9,11]\n",
    c.value[0].as_canonical_u32(),c.value[1].as_canonical_u32(),c.value[2].as_canonical_u32(),c.value[3].as_canonical_u32());
  kb31_extension_t X(0,1,0,0), X2=X*X;
  printf("X*X -> (%u,%u,%u,%u)  [expect 0,0,1,0]\n",
    X2.value[0].as_canonical_u32(),X2.value[1].as_canonical_u32(),X2.value[2].as_canonical_u32(),X2.value[3].as_canonical_u32());
  kb31_extension_t X3(0,0,0,1), X4=X3*X;
  printf("X^3*X=X^4 -> (%u,%u,%u,%u)  [expect 3,0,0,0 since W=3]\n",
    X4.value[0].as_canonical_u32(),X4.value[1].as_canonical_u32(),X4.value[2].as_canonical_u32(),X4.value[3].as_canonical_u32());
}
int main(){ probe<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
