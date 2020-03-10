"""Module for tomography."""

import cupy as cp
import numpy as np
from lamcg.lamusfft import lamusfft


class SolverLam(lamusfft):
    """Base class for laminography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    """

    def __init__(self, n0, n1, n2, det, ntheta, phi):
        """Please see help(SolverLam) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(n2, n1, n0, det, ntheta, phi)# reorder sizes
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_lam(self, u, theta):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.det, self.det], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u.data.ptr, theta.data.ptr)        
        return res

    def adj_lam(self, data, theta):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.n2, self.n1, self.n0], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data.data.ptr, theta.data.ptr)
        return res    
    def adj_lam(self, data, theta):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.n2, self.n1, self.n0], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data.data.ptr, theta.data.ptr)
        return res            
    
    def line_search(self, minf, gamma, Lu, Ld):
        """Line search for the step sizes gamma"""
        while(minf(Lu)-minf(Lu+gamma*Ld) < 0):
            gamma *= 0.5
        return gamma        

    def cg_lam(self, data, u, theta, titer):
        """CG solver for ||Lu-data||_2"""
        # minimization functional
        def minf(Lu):
            f = cp.linalg.norm(Lu-data)**2
            return f
        for i in range(titer):
            Lu = self.fwd_lam(u,theta)
            grad = self.adj_lam(Lu-data, theta) * 1/self.ntheta/self.n0/self.n1/self.n2
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Ld = self.fwd_lam(d, theta)
            gamma = 0.5*self.line_search(minf, 1, Lu, Ld)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Lu)))
        return u        