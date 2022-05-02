"""
CUDA Raw kernels for computing back-projection to orthogonal slices
"""

import cupy as cp

source = """
extern "C" {    
    void __global__ fwd(float *data, float *f, float *theta, float phi, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= ntheta || tz >= deth)
            return;
        float x = 0;
        float y = 0;
        float z = 0;
        int xr = 0;
        int yr = 0;
        int zr = 0;
        
        float theta0 = theta[ty];                
        float data0 = 0;
        float R[9] = {};
        for (int t = 0; t<n; t++)
        {
            R[0] =  __cosf(theta0);              R[1] =  __sinf(theta0);               R[2] = 0;
            R[3] = -__sinf(theta0)*__sinf(phi);  R[4] =  __cosf(theta0)*__sinf(phi);   R[5] = __cosf(phi);
            R[6] =  __sinf(theta0)*__cosf(phi);  R[7] = -__cosf(theta0)*__cosf(phi);   R[8] = __sinf(phi);
            x = R[0]*(tx-n/2)+R[3]*(t-n/2)+R[6]*(tz-deth/2) + n/2;
            y = R[1]*(tx-n/2)+R[4]*(t-n/2)+R[7]*(tz-deth/2) + n/2;
            z = R[2]*(tx-n/2)+R[5]*(t-n/2)+R[8]*(tz-deth/2) + nz/2;      
            xr = (int)x;
            yr = (int)y;
            zr = (int)z;
            
            // linear interp            
            if ((xr >= 0) & (xr < n - 1) & (yr >= 0) & (yr < n - 1) & (zr >= 0) & (zr < nz - 1))
            {
                x = x-xr;
                y = y-yr;
                z = z-zr;
                data0 +=f[xr+0+(yr+0)*n+(zr+0)*n*n]*(1-x)*(1-y)*(1-z)+
                        f[xr+1+(yr+0)*n+(zr+0)*n*n]*(0+x)*(1-y)*(1-z)+
                        f[xr+0+(yr+1)*n+(zr+0)*n*n]*(1-x)*(0+y)*(1-z)+
                        f[xr+1+(yr+1)*n+(zr+0)*n*n]*(0+x)*(0+y)*(1-z)+
                        f[xr+0+(yr+0)*n+(zr+1)*n*n]*(1-x)*(1-y)*(0+z)+
                        f[xr+1+(yr+0)*n+(zr+1)*n*n]*(0+x)*(1-y)*(0+z)+
                        f[xr+0+(yr+1)*n+(zr+1)*n*n]*(1-x)*(0+y)*(0+z)+
                        f[xr+1+(yr+1)*n+(zr+1)*n*n]*(0+x)*(0+y)*(0+z);
            }
        }
        data[tx + tz * n + ty * n * deth] = data0*n;        
    }    

    void __global__ adj(float *f, float *data, float *theta, float phi, int s_z, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        float R[6] = {};
        
        for (int t = 0; t<ntheta; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2) + n/2;
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(tz+s_z) + deth/2;//s_z==nz/2 in the nonchunk case, st_z-heightz else
            
            ur = (int)u;
            vr = (int)v;            
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < deth - 1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+(vr+0)*n+t*n*deth]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*deth]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*deth]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*deth]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0*n;        
    }    
}
"""

module = cp.RawModule(code=source)
fwd_kernel = module.get_function('fwd')
adj_kernel = module.get_function('adj')

def fwd(data, f, theta, phi):
    [nz, n] = f.shape[:2]
    [ntheta,deth] = data.shape[:2]
    fwd_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32+0.5)), deth), (32, 32, 1),
                  (data, f, theta, cp.float32(phi), n, nz, deth, ntheta))
    return data

def adj(f, data, theta, phi, s_z):
    [nz, n] = f.shape[:2]
    [ntheta,deth] = data.shape[:2]
    adj_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nz), (32, 32, 1),
                  (f,data, theta, cp.float32(phi), s_z, n, nz, deth, ntheta))
    return data

