import numpy as np
import lamcg as lcg
import dxchange

n0 = 70
n1 = 1000
n2 = 200
detw = 512
deth = 256
ntheta = 32
phi = np.pi/2-2/180*np.pi
niter = 32
theta = np.linspace(-20/180*np.pi, 20/180*np.pi, ntheta, endpoint=True).astype('float32')
f = np.ones([n2,n1,n0],dtype='complex64')#-dxchange.read_tiff('delta-chip-256.tiff')
with lcg.SolverLam(n0, n1, n2, detw, deth, ntheta, phi) as slv:
    data = slv.fwd_lam(f, theta)
    print(data.shape)
    dxchange.write_tiff(data.real, 'data/r', overwrite=True)
    #init = np.zeros([n0, n1, n2], dtype='float32')
    #rec = slv.cg_lam(data, init, theta, niter, dbg=True)
    #dxchange.write_tiff_stack(rec, 'rec/r', overwrite=True)
