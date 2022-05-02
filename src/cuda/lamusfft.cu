#include <stdio.h>

#include "lamusfft.cuh"
#include "kernels.cu"
#include "shift.cu"

lamusfft::lamusfft(size_t n, size_t nz, size_t detw, size_t deth, size_t ntheta, float phi, float eps)
    : n(n), nz(nz), detw(detw), deth(deth), ntheta(ntheta), phi(phi) {
  mu = -log(eps) / (2 * n * n);  
  muz = -log(eps) / (2 * nz * nz);
  
  m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));
  mz = ceil(2 * nz * 1 / PI * sqrt(-muz * log(eps) + (muz * nz) * (muz * nz) / 4));
  cudaMalloc((void **)&f, n * n * nz * sizeof(float2));
  cudaMalloc((void **)&g, detw * deth * ntheta * sizeof(float2));
  cudaMalloc((void **)&fdee,
             (2 * n + 2 * m) * (2 * n + 2 * m) * (2 * nz + 2 * mz) * sizeof(float2));

  cudaMalloc((void **)&x, detw * deth * ntheta * sizeof(float));
  cudaMalloc((void **)&y, detw * deth * ntheta * sizeof(float));
  cudaMalloc((void **)&z, detw * deth * ntheta * sizeof(float));
  cudaMalloc((void **)&theta, ntheta * sizeof(float));
  
  int ffts[3];
  int idist;
  int inembed[3];
  // fft 2d
  ffts[0] = 2 * nz;
  ffts[1] = 2 * n;
  ffts[2] = 2 * n;
  idist = (2 * nz + 2 * mz) * (2 * n + 2 * m)* (2 * n + 2 * m);
  inembed[0] = 2 * nz + 2 * mz; // Note the order is reverse!
  inembed[1] = 2 * n + 2 * m;
  inembed[2] = 2 * n + 2 * m;
  fprintf(stderr,"%d \n",cufftPlanMany(&plan3d, 3, ffts, inembed, 1, idist, inembed, 1, idist,
                CUFFT_C2C, 1));
  
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
  GS3d1 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y),
                ceil(nz / (float)BS3d.z));
  GS3d2 = dim3(ceil(2*n / (float)BS3d.x), 
               ceil(2*n / (float)BS3d.y),
               ceil(2*nz / (float)BS3d.z));                
  GS3d3 = dim3(ceil((2 * n + 2 * m) / (float)BS3d.x),
               ceil((2 * n + 2 * m) / (float)BS3d.y), 
               ceil((2 * nz + 2 * mz) / (float)BS3d.z));  
}

// destructor, memory deallocation
lamusfft::~lamusfft() { free(); }

void lamusfft::free() {
  if (!is_free) {
    cudaFree(f);
    cudaFree(g);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cufftDestroy(plan3d);
    cufftDestroy(plan2d);
    is_free = true;
  }
}

void lamusfft::fwd(size_t g_, size_t f_, size_t theta_) {
  cudaMemcpy(f, (float2 *)f_, n * n * nz * sizeof(float2), cudaMemcpyDefault);
  cudaMemcpy(theta, (float *)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
  cudaMemset(fdee, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * (2 * nz + 2 * mz) * sizeof(float2));

  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi, detw, deth, ntheta);

  divker <<<GS3d1, BS3d>>> (fdee, f, mu, muz, n, nz, m, mz, TOMO_FWD);  
  
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n + 2 * m, 2 * nz +2 * mz);
  
  cufftExecC2C(plan3d, (cufftComplex *)&fdee[m + m * (2 * n + 2 * m) + mz * (2 * n + 2 * m) * (2 * n + 2 * m)].x,
                (cufftComplex *)&fdee[m + m * (2 * n + 2 * m) + mz * (2 * n + 2 * m) * (2 * n + 2 * m)].x, CUFFT_FORWARD);
  
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n + 2 * m, 2 * nz +2 * mz);
  
  wrap <<<GS3d3, BS3d>>> (fdee, n, nz, m, mz, TOMO_FWD);
  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m, mz, mu, muz, n, nz, detw, deth, ntheta, TOMO_FWD);
  
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);

  cudaMemcpy((float2 *)g_, g, detw * deth * ntheta * sizeof(float2), cudaMemcpyDefault);
}

void lamusfft::adj(size_t f_, size_t g_, size_t theta_) {
  cudaMemcpy(g, (float2 *)g_, detw * deth * ntheta * sizeof(float2), cudaMemcpyDefault);
  cudaMemset(fdee, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * (2 * nz + 2 * mz) * sizeof(float2));

  takexyz <<<GS3d0, BS3d>>> (x, y, z, theta, phi, detw, deth, ntheta);

  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
  fftshiftc2d <<<GS3d0, BS3d>>> (g, detw, deth, ntheta);

  gather <<<GS3d0, BS3d>>> (g, fdee, x, y, z, m, mz, mu, muz, n, nz, detw, deth, ntheta, TOMO_ADJ);
  wrap <<<GS3d3, BS3d>>> (fdee, n, nz, m, mz, TOMO_ADJ);

  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n + 2 * m, 2 * nz +2 * mz);
  cufftExecC2C(plan3d, (cufftComplex *)&fdee[m + m * (2 * n + 2 * m) + mz * (2 * n + 2 * m) * (2 * n + 2 * m)],
                (cufftComplex *)&fdee[m + m * (2 * n + 2 * m) + mz * (2 * n + 2 * m) * (2 * n + 2 * m)], CUFFT_INVERSE);
  fftshiftc3d <<<GS3d3, BS3d>>> (fdee, 2 * n + 2 * m,  2 * nz +2 * mz);
  

  divker <<<GS3d1, BS3d>>> (fdee, f, mu, muz, n, nz, m, mz, TOMO_ADJ);
  
  cudaMemcpy((float2 *)f_, f, n * n * nz * sizeof(float2),
              cudaMemcpyDefault);
}
