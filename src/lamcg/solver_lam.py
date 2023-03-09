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
    n0 : int
        Object size in x.
    n1 : int
        Object size in y.
    n2 : int
        Object size in z.
    det : int
        Detector size in one dimension.
    ntheta : int
        The number of projections.
    eps : float
        Accuracy for the USFFT computation. Default: 1e-3.
    """

    def __init__(self, n0, n1, n2, detw, deth, ntheta, phi, eps=1e-3):
        """Please see help(SolverLam) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(n2, n1, n0, detw,deth, ntheta, phi, eps)  # reorder sizes

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_lam(self, u, theta, gpu_arrays=False):
        """Laminography transform (L)"""
        res = cp.zeros([self.ntheta, self.deth, self.detw], dtype='complex64')

        u_gpu = cp.asarray(u.copy())
        theta_gpu = cp.asarray(theta)

        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u_gpu.data.ptr, theta_gpu.data.ptr)
        if(isinstance(u, np.ndarray)):
            res = res.get()
        return res

    def adj_lam(self, data, theta):
        """Adjoint Laminography transform (L^*)"""
        res = cp.zeros([self.n2, self.n1, self.n0], dtype='complex64')

        data_gpu = cp.asarray(data.copy())
        theta_gpu = cp.asarray(theta)

        # C++ wrapper, send pointers to GPU arrays
        self.adj(res.data.ptr, data_gpu.data.ptr, theta_gpu.data.ptr)
        if(isinstance(data, np.ndarray)):
            res = res.get()
        return res

    def fwd_reg(self, u):
        """Forward operator for regularization (J)"""
        res = np.tile(u*0, (3, 1, 1, 1))
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    def adj_reg(self, gr):
        """Adjoint operator for regularization (J*)"""
        res = gr[0]*0
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        return -res

    def solve_reg(self, u, mu, tau, alpha):
        """Solution of the L1 problem by soft-thresholding"""
        z = self.fwd_reg(u)+mu/tau
        za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/(za[za > alpha/tau])
        return z

    def line_search(self, minf, gamma, Lu, Ld):
        """Line search for the step sizes gamma"""
        while(minf(Lu)-minf(Lu+gamma*Ld) < 0):
            gamma *= 0.5
        return gamma

    def line_search_ext(self, minf, gamma, Lu, Ld, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Lu, gu)-minf(Lu+gamma*Ld, gu+gamma*gd) < 0):
            gamma *= 0.5
        return gamma

    def cg_lam(self, data0, u0, theta0, titer, dbg=False):
        """CG solver for ||Lu-data||_2"""
        u = cp.asarray(u0)
        theta = cp.asarray(theta0)
        data = cp.asarray(data0)

        # minimization functional
        def minf(Lu):
            f = cp.linalg.norm(Lu-data)**2
            return f
        for i in range(titer):
            Lu = self.fwd_lam(u, theta)
            grad = self.adj_lam(Lu-data, theta) * 1 / \
                self.ntheta/self.deth/self.detw/(self.n0*self.n1*self.n2)**(1/3)

            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            # Ld = self.fwd_lam(d, theta)
            gamma = 0.5#*self.line_search(minf, 8, Lu, Ld)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg == True):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Lu)))
        if(isinstance(u0, np.ndarray)):
            u = u.get()
        return u

    def cg_lam_ext(self, data0, u0, theta0, titer, tau=0, xi0=None, dbg=False):
        """CG solver for ||Lu-data||_2+tau||Ju-xi||_2"""

        u = cp.asarray(u0)
        theta = cp.asarray(theta0)
        data = cp.asarray(data0)
        xi = cp.asarray(xi0)

        def minf(Lu, gu):
            return cp.linalg.norm(Lu-data)**2 + tau*cp.linalg.norm(gu-xi)**2

        for i in range(titer):
            Lu = self.fwd_lam(u, theta)
            gu = self.fwd_reg(u)
            grad = (self.adj_lam(Lu-data, theta) * 1/self.ntheta/self.n0/self.n1/self.n2 +
                    tau*self.adj_reg(gu-xi)/2)/max(tau, 1)  # normalized gradient
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Ld = self.fwd_lam(d, theta)
            gd = self.fwd_reg(d)
            gamma = 0.5*self.line_search_ext(minf, 1, Lu, Ld, gu, gd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg == True):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Lu, gu)))

        if(isinstance(u0, np.ndarray)):
            u = u.get()
        return u
