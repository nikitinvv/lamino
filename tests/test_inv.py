import numpy as np
import lamcg as lcg
import dxchange

n = 256
deth = 256
phi = (90)/180*np.pi
nz = int(np.ceil((deth/np.sin(phi))/4))*4

print(f'{deth=},{n=},{nz=}')

ntheta = 360
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = -dxchange.read_tiff('delta-chip-256.tiff')[128-nz//2:128+nz//2]

with lcg.SolverLam(n, nz, n, deth, ntheta, phi) as slv:
    data3 = slv.fwd_lam(f, theta)
    print(np.linalg.norm(data3))
    dxchange.write_tiff_stack(data3, 'data3/r', overwrite=True)
    frec3 = slv.inv_lam(data3, theta)
    print(np.linalg.norm(frec3))
    dxchange.write_tiff_stack(frec3, 'rec3/r', overwrite=True)


        
    
