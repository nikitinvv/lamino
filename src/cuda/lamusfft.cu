#include <stdio.h>

#include "lamusfft.cuh"
#include "kernels.cu"
#include "shift.cu"

lamusfft::lamusfft(size_t n0, size_t n1, size_t n2, size_t det, size_t ntheta, float phi)
    : n0(n0), n1(n1), n2(n2), det(det), ntheta(ntheta), phi(phi) {
  float eps = 1e-3;
  mu0 = -log(eps) / (2 * n0 * n0);
  mu1 = -log(eps) / (2 * n1 * n1);
  mu2 = -log(eps) / (2 * n2 * n2);
  m0 = ceil(2 * n0 * 1 / PI * sqrt(-mu0 * log(eps) + (mu0 * n0) * (mu0 * n0) / 4));
  m1 = ceil(2 * n1 * 1 / PI * sqrt(-mu1 * log(eps) + (mu1 * n1) * (mu1 * n1) / 4));
  m2 = ceil(2 * n2 * 1 / PI * sqrt(-mu2 * log(eps) + (mu2 * n2) * (mu2 * n2) / 4));
  cudaMalloc((void **)&f, n0 * n1 * n2 * sizeof(float2));
  cudaMalloc((void **)&g, det * det * ntheta * sizeof(float2));
  cudaMalloc((void **)&fde, 2 * n0 * 2 * n1 * 2 * n2 * sizeof(float2));
  cudaMalloc((void **)&fdee,
             (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));

  cudaMalloc((void **)&x, det * det * ntheta * sizeof(float));
  cudaMalloc((void **)&y, det * det * ntheta * sizeof(float));
  cudaMalloc((void **)&z, det * det * ntheta * sizeof(float));
  cudaMalloc((void **)&theta, ntheta * sizeof(float));
  
  int ffts[3];
  int idist;
  int odist;
  int inembed[3];
  int onembed[3];
  // fft 2d
  ffts[0] = 2 * n2;
  ffts[1] = 2 * n1;
  ffts[2] = 2 * n0;
  idist = 2 * n0 * 2 * n1 * 2 * n2;
  odist = (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)* (2 * n2 + 2 * m2);
  inembed[0] = 2 * n2; // Note the order is reverse!
  inembed[1] = 2 * n1;
  inembed[2] = 2 * n0;
  onembed[0] = 2 * n2 + 2 * m2;
  onembed[1] = 2 * n1 + 2 * m1;
  onembed[2] = 2 * n0 + 2 * m0;
  cufftPlanMany(&plan3dfwd, 3, ffts, inembed, 1, idist, onembed, 1, odist,
                CUFFT_C2C, 1);
  cufftPlanMany(&plan3dadj, 3, ffts, onembed, 1, odist, inembed, 1, idist,
                CUFFT_C2C, 1);

  // fft 2d
  ffts[0] = det;
  ffts[1] = det;
  idist = det*det;
  odist = det*det;
  inembed[0] = det;
  inembed[1] = det;
  onembed[0] = det;
  onembed[1] = det;
  cufftPlanMany(&plan2d, 2, ffts, inembed, 1, idist, onembed, 1, odist,
                CUFFT_C2C, ntheta);
  
  BS3d = dim3(16, 16, 4);

  GS3d0 = dim3(ceil(det / (float)BS3d.x), ceil(det / (float)BS3d.y),
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
    cudaFree(f);
    cudaFree(g);
    cudaFree(fde);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cufftDestroy(plan3dfwd);
    cufftDestroy(plan3dadj);
    cufftDestroy(plan2d);
    is_free = true;
  }
}

void lamusfft::fwd(size_t g_, size_t f_, size_t theta_) {
  cudaMemcpy(f, (float2 *)f_, n0 * n1 * n2 * sizeof(float2), cudaMemcpyDefault);
  cudaMemcpy(theta, (float *)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
  cudaMemset(fde, 0, 2 * n0 * 2 * n1 * 2 * n2 * sizeof(float2));
  cudaMemset(fdee, 0, (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));

  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi, det, ntheta);

  divker <<<GS3d1, BS3d>>> (fde, f, mu0, mu1, mu2, n0, n1, n2, TOMO_FWD);  
  fftshiftc3d <<<GS3d2, BS3d>>> (fde, 2 * n0, 2 * n1, 2 * n2);
  cufftExecC2C(plan3dfwd, (cufftComplex *)fde,
                (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)], CUFFT_FORWARD);
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  

  wrap <<<GS3d3, BS3d>>> (fdee, n0, n1, n2, m0, m1, m2, TOMO_FWD);
  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m0, m1, m2, mu0, mu1, mu2, n0, n1, n2, det, ntheta, TOMO_FWD);
  
  fftshiftc2d <<<GS3d0, BS3d>>> (g, det, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, det, ntheta);

  cudaMemcpy((float2 *)g_, g, det * det * ntheta * sizeof(float2), cudaMemcpyDefault);
}

void lamusfft::adj(size_t f_, size_t g_, size_t theta_) {
  cudaMemcpy(g, (float2 *)g_, det * det * ntheta * sizeof(float2), cudaMemcpyDefault);
  cudaMemset(fde, 0, 2 * n0 * 2 * n1 * 2 * n2 * sizeof(float2));
  cudaMemset(fdee, 0, (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (2 * n2 + 2 * m2) * sizeof(float2));

  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi, det, ntheta);

  fftshiftc2d <<<GS3d0, BS3d>>> (g, det, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, det, ntheta);

  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m0, m1, m2, mu0, mu1, mu2, n0, n1, n2, det, ntheta, TOMO_ADJ);
  wrap <<<GS3d3, BS3d>>> (fdee, n0, n1, n2, m0, m1, m2, TOMO_ADJ);

  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n0 + 2 * m0, 2 * n1 +2 * m1, 2 * n2 +2 * m2);
  cufftExecC2C(plan3dadj, (cufftComplex *)&fdee[m0 + m1 * (2 * n0 + 2 * m0) + m2 * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)],
                (cufftComplex *)fde, CUFFT_INVERSE);
  fftshiftc3d <<<GS3d2, BS3d>>> (fde, 2 * n0, 2 * n1, 2 * n2);
  

  divker <<<GS3d1, BS3d>>> (fde, f, mu0, mu1, mu2, n0, n1, n2, TOMO_ADJ);
  
  cudaMemcpy((float2 *)f_, f, n0 * n1 * n2 * sizeof(float2),
              cudaMemcpyDefault);
}
