#define PI 3.1415926535897932384626433

// Divide by phi
void __global__ divker(float2 *g, float2 *f, float mu, float muz, int n, int nz, int m, int mz, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  float ker = __expf(
    -mu * (tx - n / 2) * (tx - n / 2)
    -mu * (ty - n / 2) * (ty - n / 2)
    -muz * (tz - nz / 2) * (tz - nz / 2)
  );
  int f_ind = (
    + tx
    + ty * n
    + tz * n * n
  );
  int g_ind = (
    + (tx + n / 2+m)
    + (ty + n / 2+m) * (2 * n +2*m)
    + (tz + nz / 2+mz) * (2 * n +2*m)  * (2 * n +2*m)
  );
  if (direction == TOMO_FWD){
    g[g_ind].x = f[f_ind].x / ker / (8 * n * n * nz);
    g[g_ind].y = f[f_ind].y / ker / (8 * n * n * nz);
  } else {
    f[f_ind].x = g[g_ind].x / ker / (8 * n * n * nz);
    f[f_ind].y = g[g_ind].y / ker / (8 * n * n * nz);
  }
}


void __global__ takexyz(float *x, float *y, float *z, float *theta, float phi, int detw, int deth,  int ntheta) {
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

void __global__ wrap(float2 *f, int n, int nz, int m, int mz, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n + 2 * m || 
      ty >= 2 * n + 2 * m || 
      tz >= 2 * nz + 2 * mz)
    return;
  if (tx < m || tx >= 2 * n + m || 
      ty < m || ty >= 2 * n + m || 
      tz < mz || tz >= 2 * nz + mz) {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int tz0 = (tz - mz + 2 * nz) % (2 * nz);
    int id1 = (
      + tx
      + ty * (2 * n + 2 * m)
      + tz * (2 * n + 2 * m) * (2 * n + 2 * m)
    );
    int id2 = (
      + tx0 + m
      + (ty0 + m) * (2 * n + 2 * m)
      + (tz0 + mz) * (2 * n + 2 * m) * (2 * n + 2 * m)
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

void __global__ gather(float2 *g, float2 *f, float *x, float *y, float *z, int m, int mz, 
  float mu, float muz, int n, int nz, int detw, int deth,  int ntheta, dir direction) {                        
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
  for (int i2 = 0; i2 < 2 * mz + 1; i2++) 
  {
    int ell2 = floorf(2 * nz * z0) - mz + i2;
    for (int i1 = 0; i1 < 2 * m + 1; i1++) 
    {
      int ell1 = floorf(2 * n * y0) - m + i1;
      for (int i0 = 0; i0 < 2 * m + 1; i0++) 
      {
        int ell0 = floorf(2 * n * x0) - m + i0;
        float w0 = ell0 / (float)(2 * n) - x0;
        float w1 = ell1 / (float)(2 * n) - y0;
        float w2 = ell2 / (float)(2 * nz) - z0;
        float w = (
          sqrtf(PI)*sqrtf(PI)*sqrtf(PI) / (sqrtf(mu * mu * muz))
          * __expf(-PI * PI / mu * (w0 * w0) - PI * PI / mu * (w1 * w1)- PI * PI / muz * (w2 * w2))
        );
        int f_ind = (
          + n + m + ell0
          + (2 * n + 2 * m) * (n + m + ell1)
          + (2 * n + 2 * m) * (2 * n + 2 * m) * (nz + mz + ell2)
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
