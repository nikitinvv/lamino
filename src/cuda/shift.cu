void __global__ fftshiftc2d(float2 *f, int det, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= det || ty >= det || tz >= ntheta)
    return;
  int g = (1 - 2 * ((tx + 1) % 2))*(1 - 2 * ((ty + 1) % 2));
  int f_ind = tx + ty * det  + tz * det * det;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ fftshiftc3d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2))* (1 - 2 * ((tz + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1].x *= g;
  f[tx + ty * n0 + tz * n0 * n1].y *= g; 
}
