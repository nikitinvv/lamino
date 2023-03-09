import numpy as np
import lamcg as lcg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import dxchange
import time
n0 = 256
n1 = 128
n2 = 200
detw = 400
deth = 512
ntheta = 128
phi = np.pi/2-2/180*np.pi
theta = np.linspace(-45/180*np.pi,45/180*np.pi,ntheta,endpoint=True).astype('float32')
f = np.ones([n0,n1,n2]).astype('complex64')
# f[n0//8:3*n0//8,n1//4:3*n1//4,n2//4:3*n2//4]=1


with lcg.SolverLam(n0, n1, n2, detw, deth, ntheta, phi,1e-2) as slv:
    t = time.time()
    g_gpu = slv.fwd_lam(f,theta)
    print(time.time()-t)
    
    dxchange.write_tiff(g_gpu.real, 'data/r', overwrite=True)
    
    t = time.time()
    ff_gpu = slv.adj_lam(g_gpu, theta)
    print(time.time()-t)
    print('Adj test 2')
    print(np.sum(f*np.conj(ff_gpu)))
    print(np.sum(g_gpu*np.conj(g_gpu)))
    gg_gpu = slv.fwd_lam(ff_gpu,theta)
    print('Norm test')
    r = 1/ntheta/deth/detw/(n0*n1*n2)**(1/3)
    g_gpu*=r
    gg_gpu*=r*r
    print(np.sum(g_gpu*np.conj(gg_gpu))/np.sum(gg_gpu*np.conj(gg_gpu)))    
    