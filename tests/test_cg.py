import numpy as np
import lamcg as lcg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import dxchange
import cupy
# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

n0 = 70
n1 = 10000
n2 = 200
detw = 4096
deth = 700
ntheta = 128
phi = np.pi/2-2/180*np.pi
theta = np.linspace(-45/180*np.pi,45/180*np.pi,ntheta,endpoint=True).astype('float32')
f = np.ones([n0,n1,n2]).astype('complex64')
# f[n0//8:3*n0//8,n1//4:3*n1//4,n2//4:3*n2//4]=1


with lcg.SolverLam(n0, n1, n2, detw, deth, ntheta, phi, 1e-2) as slv:

    g = slv.fwd_lam(f,theta)
    rec = slv.cg_lam(g, f*0, theta, 128, True)
    dxchange.write_tiff_stack(rec.real, 'rec/r', overwrite=True)