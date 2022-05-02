import numpy as np
import lamcg as lcg
import dxchange

n = 256
deth = 256-16
phi = (90-20)/180*np.pi
nz = int(np.ceil((deth/np.sin(phi))/4))*4

print(f'{deth=},{n=},{nz=}')

ntheta = 360
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = -dxchange.read_tiff('delta-chip-256.tiff')[128-nz//2:128+nz//2]
with lcg.SolverLamLinear(n, nz, deth, ntheta//2) as slv:
        data2 = slv.fwd_lam(f, theta,phi)            
        dxchange.write_tiff_stack(data2, 'data2/r', overwrite=True)

with lcg.SolverLam(n, nz, n, deth, ntheta, phi) as slv2:
    data3 = slv2.fwd_lam(f, theta)
    dxchange.write_tiff_stack(data3, 'data3/r', overwrite=True)
    rec3 = slv2.inv_lam(data3, theta)
    dxchange.write_tiff_stack(rec3, 'rec3/r', overwrite=True)

with lcg.SolverLamLinear(n, 16, deth, 32) as slv:    
        frec2 = slv.inv_lam(data2, theta, phi, nz)
        dxchange.write_tiff_stack(frec2, 'rec2/r', overwrite=True)
        frec3 = slv.inv_lam(data3, theta, phi, nz)
        dxchange.write_tiff_stack(frec3, 'rec4/r', overwrite=True)
        print(np.linalg.norm(frec2))
        print(np.linalg.norm(frec3))