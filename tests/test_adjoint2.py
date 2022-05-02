import numpy as np
import lamcg as lcg
import matplotlib.pyplot as plt
import dxchange

n = 256
deth = 128
phi = (90-20)/180*np.pi
nz = int(np.ceil((deth/np.sin(phi))/4))*4

ntheta = 360
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = -dxchange.read_tiff('delta-chip-256.tiff')[128-nz//2:128+nz//2]

with lcg.SolverLamLinear(n, nz, deth, ntheta) as slv:
    data2 = slv.fwd_lam(f, theta, phi)    
    frec2 = slv.adj_lam(data2, theta, phi, nz)
    print(np.sum(f*np.conj(frec2)))
    print(np.sum(data2*np.conj(data2)))

with lcg.SolverLam(n, nz, n, deth, ntheta, phi) as slv:
    data2 = slv.fwd_lam(f, theta)    
    frec2 = slv.adj_lam(data2, theta)
    print(np.sum(f*np.conj(frec2)))
    print(np.sum(data2*np.conj(data2)))
    