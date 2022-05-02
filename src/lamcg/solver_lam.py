"""Module for tomography."""

import cupy as cp
import numpy as np
from lamcg.lamusfft import lamusfft
from cupyx.scipy.fft import rfft, irfft

class SolverLam(lamusfft):
    """Base class for laminography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    n : int
        Object size in x.
    nz : int
        Object size in z.    
    detw : int
        Detector width.
    deth : int
        Detector height.
    ntheta : int
        The number of projections.
    phi : float
        Tilt angle for laminography
    eps : float
        Accuracy for the USFFT computation. Default: 1e-3.
    """

    def __init__(self, n, nz, detw, deth, ntheta, phi, eps=1e-3):
        """Please see help(SolverLam) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(n, nz, detw, deth, ntheta, phi, eps)  

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_lam(self, u, theta):
        """Laminography transform (L)"""
        res = cp.zeros([self.ntheta, self.deth, self.detw], dtype='complex64')

        u_gpu = cp.asarray(u.astype('complex64'))
        theta_gpu = cp.asarray(theta)

        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u_gpu.data.ptr, theta_gpu.data.ptr)

        if(np.isrealobj(u)):
            res = res.real
        if(isinstance(u, np.ndarray)):
            res = res.get()
        return res


    def fbp_filter_center(self, data, sh=0):
        """FBP filtering of projections"""
        
        ne = 3*self.detw//2
        
        t = cp.fft.rfftfreq(ne).astype('float32')
        # if self.args.gridrec_filter == 'parzen':
            # w = t * (1 - t * 2)**3  
        # elif self.args.gridrec_filter == 'shepp':
            # w = t * cp.sinc(t)  
        # elif self.args.gridrec_filter == 'ramp':
        w = t          
        #w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.det/2))  # center fix
                
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.detw//2, ne//2-self.detw//2)), mode='edge')
        #self.cl_rec.filter(data,w,cp.cuda.get_current_stream())
        data = irfft(w*rfft(data, axis=2), axis=2)
        data = data[:, :, ne//2-self.detw//2:ne//2+self.detw//2]            

        return data

    def adj_lam(self, data, theta):
        """Adjoint Laminography transform (L^*)"""
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64')

        data_gpu = cp.asarray(data.astype('complex64'))
        theta_gpu = cp.asarray(theta)



        # C++ wrapper, send pointers to GPU arrays
        self.adj(res.data.ptr, data_gpu.data.ptr, theta_gpu.data.ptr)

        if(cp.isrealobj(data)):
            res = res.real
        if(isinstance(data, np.ndarray)):
            res = res.get()
        return res
    
    def inv_lam(self,data,theta):
        data_gpu = cp.asarray(data.astype('float32'))
        data_gpu = self.fbp_filter_center(data_gpu)
        obj = self.adj_lam(data_gpu,theta)
        return obj.real.get()         
    
    # def line_search(self, minf, gamma, Lu, Ld):
    #     """Line search for the step sizes gamma"""
    #     while(minf(Lu)-minf(Lu+gamma*Ld) < 0):
    #         gamma *= 0.5
    #     return gamma

    # def cg_lam(self, data0, u0, theta0, titer, dbg=False):
    #     """CG solver for ||Lu-data||_2"""
    #     u = cp.asarray(u0)
    #     theta = cp.asarray(theta0)
    #     data = cp.asarray(data0)

    #     # minimization functional
    #     def minf(Lu):
    #         f = cp.linalg.norm(Lu-data)**2
    #         return f
    #     for i in range(titer):
    #         Lu = self.fwd_lam(u, theta)
    #         grad = self.adj_lam(Lu-data, theta) * 1 / \
    #             self.ntheta/self.n0/self.n1/self.n2

    #         if i == 0:
    #             d = -grad
    #         else:
    #             d = -grad+cp.linalg.norm(grad)**2 / \
    #                 (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
    #         # line search
    #         Ld = self.fwd_lam(d, theta)
    #         gamma = 0.5*self.line_search(minf, 1, Lu, Ld)
    #         grad0 = grad
    #         # update step
    #         u = u + gamma*d
    #         # check convergence
    #         if (dbg == True):
    #             print("%4d, %.3e, %.7e" %
    #                   (i, gamma, minf(Lu)))

    #     if(isinstance(u0, np.ndarray)):
    #         u = u.get()
    #     return u