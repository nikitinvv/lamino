#ifndef LAMUSFFT_CUH
#define LAMUSFFT_CUH

#include <cufft.h>

enum dir {
  TOMO_FWD,
  TOMO_ADJ
};

class lamusfft {
  bool is_free = false;

  size_t m0,m1,m2;
  float mu0,mu1,mu2;

  float2 *f;
  float2 *g;
  float2 *ff;
  float2 *gg;
  float2 *f0;
  float2 *g0;
  float *theta;

  float *x;
  float *y;
  float *z;

  float2 *fdee;

  cufftHandle plan3d;
  cufftHandle plan2d;
  
  dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3;

public:
  size_t n0,n1,n2;  
  size_t det;
  size_t ntheta; // number of angles
  float phi;
  float gamma;

  lamusfft(size_t n0, size_t n1, size_t n2, size_t det, size_t ntheta, float phi, float gamma, float eps);
  ~lamusfft();
  void fwd(size_t g, size_t f, size_t theta);
  void adj(size_t f, size_t g, size_t theta);
  void free();
};

#endif
