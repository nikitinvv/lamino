import numpy as np
import lamcg as lcg
import dxchange
import time
n = 2448
deth = 2048
phi = (90-20)/180*np.pi
nz = int(np.ceil((deth/np.sin(phi))/4))*4

print(f'{deth=},{n=},{nz=}')

ntheta = 1500
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = np.zeros([nz,n,n],dtype='float32')
with lcg.SolverLamLinear(n, 16, deth, 16) as slv:
    t0 = time.time()
    # data2 = slv.fwd_lam(f, theta, phi)            
    data2 = np.zeros([ntheta,deth,n],dtype='float32')
    t1 = time.time()
    frec2 = slv.inv_lam(data2, theta, phi, nz)
    t2 = time.time()
    print(t1-t0,t2-t1)