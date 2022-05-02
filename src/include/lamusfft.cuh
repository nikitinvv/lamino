#ifndef LAMUSFFT_CUH
#define LAMUSFFT_CUH

#include <cufft.h>

enum dir {
  TOMO_FWD,
  TOMO_ADJ
};

class lamusfft {
  bool is_free = false;

  size_t m, mz;
  float mu,muz;

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
  size_t n, nz; 
  size_t detw, deth;
  size_t ntheta; // number of angles
  float phi;
  
  lamusfft(size_t n, size_t nz, size_t detw , size_t deth, size_t ntheta, float phi, float eps);
  ~lamusfft();
  void fwd(size_t g, size_t f, size_t theta);
  void adj(size_t f, size_t g, size_t theta);
  void free();
};

#endif
