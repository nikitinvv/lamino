"""Module for tomography."""

import cupy as cp
import numpy as np
from lamcg.kernels import fwd,adj
from cupyx.scipy.fft import rfft, irfft

class SolverLamLinear():
    """Base class for laminography solvers using the direct line integration with linear interpolation on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    n : int
        Object size in x, detector width
    nz : int
        Object size in z.    
    deth : int
        Detector height.
    ntheta : int
        The number of projections.
    """
    def __init__(self, n, nz, deth, ntheta):
        self.n = n
        self.nz = nz
        self.deth = deth
        self.ntheta = ntheta

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        pass

    def fwd_lam(self,u,theta,phi):
        if u.shape[0]!=self.nz:
            print('fwd operator doesnt allow chunks in z')
            exit()
        data = np.zeros([len(theta), self.deth, self.n], dtype='float32')    

        data_gpu = cp.zeros([self.ntheta, self.deth, self.n], dtype='float32')
        theta_gpu = cp.zeros([self.ntheta], dtype='float32')                
        u_gpu = cp.asarray(u)
        
        for it in range(int(np.ceil(len(theta)/self.ntheta))):
            st_t = it*self.ntheta
            end_t = (it+1)*self.ntheta
            print(f'{st_t=},{end_t=}')
            
            sht = data[st_t:end_t].shape[0]
                
            data_gpu[:sht] = cp.asarray(data[st_t:end_t])
            data_gpu[sht:] = 0
            theta_gpu[:sht] = cp.asarray(theta[st_t:end_t])        
            
            fwd(data_gpu,u_gpu,theta_gpu,phi)
            
            data[st_t:end_t] = data_gpu[:sht].get()
        return data

    def adj_lam(self,data,theta,phi,heightz):
        
        u = np.zeros([heightz, self.n, self.n], dtype='float32')    
        
        data_gpu = cp.zeros([self.ntheta, self.deth, self.n], dtype='float32')
        u_gpu = cp.zeros([self.nz, self.n, self.n], dtype='float32')
        theta_gpu = cp.zeros([self.ntheta], dtype='float32')                
        
        for it in range(0,int(np.ceil(len(theta)/self.ntheta))):
            st_t = it*self.ntheta
            end_t = (it+1)*self.ntheta
            print(f'{st_t=},{end_t=}')
            for iz in range(int(np.ceil(u.shape[0]/self.nz))):                                            
                st_z = iz*self.nz
                end_z = (iz+1)*self.nz
                print(f'{st_z=},{end_z=}')

                sht = data[st_t:end_t].shape[0]
                shz = u[st_z:end_z].shape[0]

                data_gpu[:sht] = cp.asarray(data[st_t:end_t])
                data_gpu[sht:] = 0
                theta_gpu[:sht] = cp.asarray(theta[st_t:end_t])                      
                u_gpu[:shz] = cp.asarray(u[st_z:end_z])                
                
                adj(u_gpu,data_gpu,theta_gpu,phi,st_z-heightz//2)
                
                u[st_z:end_z] = u_gpu[:shz].get()
        return u
    
    
    def inv_lam(self,data,theta,phi,heightz=0):
        """Inverse Laminography transform (L^*W)"""
        if heightz == 0:
            heightz = self.nz

        data = self.fbp_filter_center(data)
        obj = self.adj_lam(data,theta,phi,heightz)
        return obj

    def fbp_filter_center(self, data, sh=0):
        """FBP filtering of projections"""
        
        data = cp.asarray(data)
        ne = 3*self.n//2
        
        t = cp.fft.rfftfreq(ne).astype('float32')
        # if self.args.gridrec_filter == 'parzen':
        #     w = t * (1 - t * 2)**3  
        # elif self.args.gridrec_filter == 'shepp':
            # w = t * cp.sinc(t)  
        # elif self.args.gridrec_filter == 'ramp':
        w = t          
        # w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.det/2))  # center fix
        # w = w*cp.exp(-2*cp.pi*1j*t*(-0.5))  # center fix
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')        
        data = irfft(w*rfft(data, axis=2), axis=2)
        data = cp.ascontiguousarray(data[:, :, ne//2-self.n//2:ne//2+self.n//2])
        
        return data.get()
        
    