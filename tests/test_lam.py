import numpy as np
import lamcg as lcg
import cupy as cp
import dxchange

n0 = 256
n1 = 256
n2 = 256
det = 256
ntheta = 180
phi = np.pi/3
theta = np.linspace(0,2*np.pi,ntheta,endpoint=False).astype('float32')
f=-dxchange.read_tiff('delta-chip-256.tiff').astype('complex64')
with lcg.SolverLam(n0, n1, n2, det, ntheta, phi,1e-1) as slv:
    f_gpu = cp.array(f)
    theta_gpu = cp.array(theta)    
    data_gpu = slv.fwd_lam(f_gpu,theta_gpu)
    dxchange.write_tiff_stack(data_gpu.get().real,'data/r',overwrite=True)

    init_gpu = cp.zeros([n0,n1,n2],dtype='complex64')
    rec = slv.cg_lam(data_gpu,init_gpu,theta_gpu,4,True)    
    dxchange.write_tiff_stack(rec.get().real,'rec128/r',overwrite=True)
    


   