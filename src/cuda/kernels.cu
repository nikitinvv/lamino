#define PI 3.1415926535897932384626433

// Divide by phi
void __global__ divker(float2 *g, float2 *f, float mu0, float mu1, float mu2, int n0, int n1, int n2, int m0, int m1, int m2, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  float ker = __expf(
    -mu0 * (tx - n0 / 2) * (tx - n0 / 2)
    -mu1 * (ty - n1 / 2) * (ty - n1 / 2)
    -mu2 * (tz - n2 / 2) * (tz - n2 / 2)
  );
  int f_ind = (
    + tx
    + ty * n0
    + tz * n0 * n1 
  );
  int g_ind = (
    + (tx + n0 / 2+m0)
    + (ty + n1 / 2+m1) * (2 * n0 +2*m0)
    + (tz + n2 / 2+m2) * (2 * n0 +2*m0)  * (2 * n1 +2*m1)
  );
  if (direction == TOMO_FWD){
    g[g_ind].x = f[f_ind].x / ker / (8 * n0 * n1 * n2);
    g[g_ind].y = f[f_ind].y / ker / (8 * n0 * n1 * n2);
  } else {
    f[f_ind].x = g[g_ind].x / ker / (8 * n0 * n1 * n2);
    f[f_ind].y = g[g_ind].y / ker / (8 * n0 * n1 * n2);
  }
}

void __global__ circ(float2 *f, float r, int N, int Nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= N || ty >= N || tz >= Nz)
    return;
  int id0 = tx + ty * N + tz * N * N;
  float x = (tx - N / 2) / float(N);
  float y = (ty - N / 2) / float(N);
  int lam = (4 * x * x + 4 * y * y) < 1 - r;
  f[id0].x *= lam;
  f[id0].y *= lam;
}

void __global__ takexyz(float *x, float *y, float *z, float *theta, float phi, int detw, int deth, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= detw || ty >= deth || tz >= ntheta)
    return;
  int id = tx + ty * detw + tz*detw*deth;
  x[id] = (tx - detw / 2) / (float)detw * __cosf(theta[tz]) + (ty - deth / 2) / (float)deth * __sinf(theta[tz])*__cosf(phi);
  y[id] = (tx - detw / 2) / (float)detw * __sinf(theta[tz]) - (ty - deth / 2) / (float)deth * __cosf(theta[tz])*__cosf(phi);
  z[id] = (ty - deth / 2) / (float)deth *__sinf(phi);
  if (x[id] >= 0.5f)
    x[id] = 0.5f - 1e-5;
  if (y[id] >= 0.5f)
    y[id] = 0.5f - 1e-5;    
  if (z[id] >= 0.5f)
    z[id] = 0.5f - 1e-5;
  if (x[id] < -0.5f)
    x[id] = -0.5f + 1e-5;
  if (y[id] < -0.5f)
    y[id] = -0.5f + 1e-5;    
  if (z[id] < -0.5f)
    z[id] = -0.5f + 1e-5;        
}

void __global__ wrap(float2 *f, int n0, int n1, int n2, int m0, int m1, int m2, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n0 + 2 * m0 || 
      ty >= 2 * n1 + 2 * m1 || 
      tz >= 2 * n2 + 2 * m2)
    return;
  if (tx < m0 || tx >= 2 * n0 + m0 || 
      ty < m1 || ty >= 2 * n1 + m1 || 
      tz < m2 || tz >= 2 * n2 + m2) {
    int tx0 = (tx - m0 + 2 * n0) % (2 * n0);
    int ty0 = (ty - m1 + 2 * n1) % (2 * n1);
    int tz0 = (tz - m2 + 2 * n2) % (2 * n2);
    int id1 = (
      + tx
      + ty * (2 * n0 + 2 * m0)
      + tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)
    );
    int id2 = (
      + tx0 + m0
      + (ty0 + m1) * (2 * n0 + 2 * m0)
      + (tz0 + m2) * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1)
    );
    if (direction == TOMO_FWD) {
      f[id1].x = f[id2].x;
      f[id1].y = f[id2].y;
    } else {
      atomicAdd(&f[id2].x, f[id1].x);
      atomicAdd(&f[id2].y, f[id1].y);
    }
  }
}

void __global__ gather(float2 *g, float2 *f, float *x, float *y, float *z, int m0, int m1, int m2, 
  float mu0, float mu1, float mu2, int n0, int n1, int n2, int detw, int deth,  int ntheta, dir direction) {                        
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= detw || ty >= deth || tz >= ntheta)
    return;

  float2 g0;
  float x0 = x[tx + ty * detw + tz * detw * deth];
  float y0 = y[tx + ty * detw + tz * detw * deth];
  float z0 = z[tx + ty * detw + tz * detw * deth];
  int g_ind = (
    + tx
    + ty * detw
    + tz * detw * deth
  );
  if (direction == TOMO_FWD) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x / sqrtf(detw*deth);
    g0.y = g[g_ind].y / sqrtf(detw*deth);
  }
  for (int i2 = 0; i2 < 2 * m2 + 1; i2++) 
  {
    int ell2 = floorf(2 * n2 * z0) - m2 + i2;
    for (int i1 = 0; i1 < 2 * m1 + 1; i1++) 
    {
      int ell1 = floorf(2 * n1 * y0) - m1 + i1;
      for (int i0 = 0; i0 < 2 * m0 + 1; i0++) 
      {
        int ell0 = floorf(2 * n0 * x0) - m0 + i0;
        float w0 = ell0 / (float)(2 * n0) - x0;
        float w1 = ell1 / (float)(2 * n1) - y0;
        float w2 = ell2 / (float)(2 * n2) - z0;
        float w = (
          sqrtf(PI)*sqrtf(PI)*sqrtf(PI) / (sqrtf(mu0 * mu1 * mu2))
          * __expf(-PI * PI / mu0 * (w0 * w0) - PI * PI / mu1 * (w1 * w1)- PI * PI / mu2 * (w2 * w2))
        );
        int f_ind = (
          + n0 + m0 + ell0
          + (2 * n0 + 2 * m0) * (n1 + m1 + ell1)
          + (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * (n2 + m2 + ell2)
        );
        if (direction == TOMO_FWD) {
            g0.x += w * f[f_ind].x;
            g0.y += w * f[f_ind].y;
        } else 
        {
          float *fx = &(f[f_ind].x);
          float *fy = &(f[f_ind].y);
          atomicAdd(fx, w * g0.x);
          atomicAdd(fy, w * g0.y);
        }
      }
    }
  }
  if (direction == TOMO_FWD){
    g[g_ind].x = g0.x / sqrtf(detw*deth);
    g[g_ind].y = g0.y / sqrtf(detw*deth);
  }
}
