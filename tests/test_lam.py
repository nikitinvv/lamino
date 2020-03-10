import numpy as np
import lamcg as lcg
import cupy as cp
import dxchange
n0 = 512
n1 = 512
n2 = 512
det = 512
ntheta = 180
phi = np.pi/3
theta = np.linspace(0,2*np.pi,ntheta,endpoint=False).astype('float32')
f = np.zeros([n0,n1,n2]).astype('complex64')
f[n0//4:3*n0//4,n1//4:3*n1//4,n2//4:3*n2//4]=1

f=-dxchange.read_tiff('deltachip-512.tiff').astype('complex64')
f[384:]=0
f=np.roll(f,64,axis=0)
with lcg.SolverLam(n0, n1, n2, det, ntheta, phi) as slv:
    f_gpu = cp.array(f)
    theta_gpu = cp.array(theta)    
    data_gpu = slv.fwd_lam(f_gpu,theta_gpu)
    dxchange.write_tiff_stack(data_gpu.get().real,'data/r',overwrite=True)

    init_gpu = cp.zeros([n0,n1,n2],dtype='complex64')
    rec = slv.cg_lam(data_gpu,init_gpu,theta_gpu,128)    
    dxchange.write_tiff_stack(rec.get().real,'rec128/r',overwrite=True)
    

   