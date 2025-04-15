#include <stdio.h>

#include "lamusfft.cuh"
#include "kernels.cu"
#include "shift.cu"

lamusfft::lamusfft(size_t n0, size_t n1, size_t n2, size_t detw, size_t deth, size_t ntheta, float phi,float alpha,float eps)
    : n0(n0), n1(n1), n2(n2), detw(detw), deth(deth), ntheta(ntheta), phi(phi), alpha(alpha) {
  mu0 = -log(eps) / (2 * n0 * n0);
  mu1 = -log(eps) / (2 * n1 * n1);
  mu2 = -log(eps) / (2 * n2 * n2);
  m0 = ceil(2 * n0 * 1 / PI * sqrt(-mu0 * log(eps) + (mu0 * n0) * (mu0 * n0) / 4));
  m1 = ceil(2 * n1 * 1 / PI * sqrt(-mu1 * log(eps) + (mu1 * n1) * (mu1 * n1) / 4));
  m2 = ceil(2 * n2 * 1 / PI * sqrt(-mu2 * log(eps) + (mu2 * n2) * (mu2 * n2) / 4));
  fprintf(stderr,"interp radius in USFFT: %d\n",m0);
  cudaMalloc((void **)&fdee,
             (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));

  cudaMalloc((void **)&x, detw * deth * ntheta * sizeof(float));
  cudaMalloc((void **)&y, detw * deth * ntheta * sizeof(float));
  cudaMalloc((void **)&z, detw * deth * ntheta * sizeof(float));
  
  int ffts[3];
  int idist;
  int inembed[3];
  // fft 2d
  ffts[0] = 2 * n2;
  ffts[1] = 2 * n1;
  ffts[2] = 2 * n0;
  idist = (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)* (2 * n2 + 2 * m2);
  inembed[0] = 2 * n2 + 2 * m2; // Note the order is reverse!
  inembed[1] = 2 * n1 + 2 * m1;
  inembed[2] = 2 * n0 + 2 * m0;
  cufftPlanMany(&plan3d, 3, ffts, inembed, 1, idist, inembed, 1, idist,
                CUFFT_C2C, 1);
  
  // fft 2d
  ffts[0] = deth;
  ffts[1] = detw;
  idist = detw*deth;
  inembed[0] = deth;
  inembed[1] = detw;
  cufftPlanMany(&plan2d, 2, ffts, inembed, 1, idist, inembed, 1, idist,
                CUFFT_C2C, ntheta);
  
  BS3d = dim3(16, 16, 4);

  GS3d0 = dim3(ceil(detw / (float)BS3d.x), ceil(deth / (float)BS3d.y),
                ceil(ntheta / (float)BS3d.z));
  GS3d1 = dim3(ceil(n0 / (float)BS3d.x), ceil(n1 / (float)BS3d.y),
                ceil(n2 / (float)BS3d.z));
  GS3d2 = dim3(ceil(2*n0 / (float)BS3d.x), 
               ceil(2*n1 / (float)BS3d.y),
               ceil(2*n2 / (float)BS3d.z));                
  GS3d3 = dim3(ceil((2 * n0 + 2 * m0) / (float)BS3d.x),
               ceil((2 * n1 + 2 * m1) / (float)BS3d.y), 
               ceil((2 * n2 + 2 * m2) / (float)BS3d.z));  
}

// destructor, memory deallocation
lamusfft::~lamusfft() { free(); }

void lamusfft::free() {
  if (!is_free) {
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cufftDestroy(plan3d);
    cufftDestroy(plan3d);
    cufftDestroy(plan2d);
    is_free = true;
  }
}

void lamusfft::fwd(size_t g_, size_t f_, size_t theta_) {
  
  f = (float2*)f_;
  g = (float2*)g_;
  theta = (float*)theta_;
  
  cudaMemset(fdee, 0, (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));
  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi,alpha, detw, deth, ntheta);

  divker <<<GS3d1, BS3d>>> (fdee, f, mu0, mu1, mu2, n0, n1, n2, m0, m1,m2, TOMO_FWD);  
  
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  
  cufftExecC2C(plan3d, (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)].x,
                (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)].x, CUFFT_FORWARD);
  
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  
  wrap <<<GS3d3, BS3d>>> (fdee, n0, n1, n2, m0, m1, m2, TOMO_FWD);
  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m0, m1, m2, mu0, mu1, mu2, n0, n1, n2, detw,deth, ntheta, TOMO_FWD);
  
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);
}

void lamusfft::adj(size_t f_, size_t g_, size_t theta_) {

  f = (float2*)f_;
  g = (float2*)g_;
  theta = (float*)theta_;
  cudaMemset(fdee, 0, (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));

  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi,alpha, detw, deth, ntheta);

  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);

  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m0, m1, m2, mu0, mu1, mu2, n0, n1, n2, detw, deth, ntheta, TOMO_ADJ);
  wrap <<<GS3d3, BS3d>>> (fdee, n0, n1, n2, m0, m1, m2, TOMO_ADJ);

  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  cufftExecC2C(plan3d, (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)],
                (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)], CUFFT_INVERSE);
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  

  divker <<<GS3d1, BS3d>>> (fdee, f, mu0, mu1, mu2, n0, n1, n2, m0,m1,m2, TOMO_ADJ);
}
