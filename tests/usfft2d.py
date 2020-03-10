import numpy as np


def eq2us_formula(x, f):
    """
    Direct computation of DFT from equally-spaced grid to unequally-spaced grid
    x - unequally-spaced grid 
    f - function on a regular grid
    """

    [N0, N1] = f.shape
    xeq0 = np.arange(-N0//2, N0//2)
    xeq1 = np.arange(-N1//2, N1//2)
    F = np.zeros(x.shape[0], dtype=complex)
    for k in range(x.shape[0]):
        for i0 in range(N0):
            for i1 in range(N1):
                F[k] = F[k]+f[i0, i1] * \
                    np.exp(-2*np.pi*1j*(xeq0[i0]*x[k, 0]+xeq1[i1]*x[k, 1]))
    return F


def us2eq_formula(x, f, N):
    """
    Direct computation of DFT from unequally-spaced grid to equally-spaced grid
    x - unequally-spaced grid 
    f - function on the grid x
    """
    [N0, N1] = N
    xeq0 = np.arange(-N0//2, N0//2)
    xeq1 = np.arange(-N1//2, N1//2)
    F = np.zeros([N0, N1], dtype=complex)
    for k in range(x.shape[0]):
        for i0 in range(N0):
            for i1 in range(N1):
                F[i0, i1] = F[i0, i1]+f[k] * \
                    np.exp(-2*np.pi*1j*(xeq0[i0]*x[k, 0]+xeq1[i1]*x[k, 1]))
    return F


def eq2us(x, f, eps, N):
    """
    USFFT from equally-spaced grid to unequally-spaced grid
    x - unequally-spaced grid 
    f - function on a regular grid of size N
    eps - accuracy of computing USFFT
    """

    # parameters for the USFFT transform
    [N0, N1] = N
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    M0 = np.int(np.ceil(2*N0*Te1))
    M1 = np.int(np.ceil(2*N1*Te2))

    # smearing kernel (phi0)
    phi0 = np.zeros((2*N0, 2*N1))
    [xeq0, xeq1] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2]
    phi0[N0//2:N0//2+N0, N1//2:N1//2+N1] = np.exp(-mu0*xeq0**2-mu1*xeq1**2)

    # FFT and compesantion for smearing
    fe = np.zeros([2*N0, 2*N1], dtype=complex)
    fe[N0//2:N0//2+N0, N1//2:N1//2+N1] = f / \
        (2*N0*2*N1)/phi0[N0//2:N0//2+N0, N1//2:N1//2+N1]
    Fe0 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fe)))

    # wrapping array Fe0
    [idx, idy] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    Fe = np.zeros([2*N0+2*M0, 2*N1+2*M1], dtype=complex)
    Fe[idx+M0, idy+M1] = Fe0[idx0, idy0]

    # smearing operation (Fe=f*phi)
    F = np.zeros(x.shape[0], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int(np.floor(2*N0*x[k, 0]))
        ell1 = np.int(np.floor(2*N1*x[k, 1]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                F[k] += Fe[N0+ell0+i0, N1+ell1+i1] * \
                    np.pi/(np.sqrt(mu0*mu1))*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k, 0])**2
                                                     - np.pi**2/mu1*((ell1-M1+i1)/(2*N1)-x[k, 1])**2))
    return F


def us2eq(x, f, eps, N):
    """
    USFFT from unequally-spaced grid to equally-spaced grid
    x - unequally-spaced grid 
    f - function on the grid x
    eps - accuracy of computing USFFT
    """
    [N0, N1] = N
    # parameters for the USFFT transform
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    M0 = np.int(np.ceil(2*N0*Te1))
    M1 = np.int(np.ceil(2*N1*Te2))

    # smearing kernel (phi0)
    phi0 = np.zeros((2*N0, 2*N1))
    [xeq0, xeq1] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2]
    phi0[N0//2:N0//2+N0, N1//2:N1//2+N1] = np.exp(-mu0*xeq0**2-mu1*xeq1**2)

    # smearing operation (G=f*phia)
    G = np.zeros([2*N0+2*M0, 2*N1+2*M1], dtype=complex)
    for k in range(x.shape[0]):
        ell0 = np.int(np.floor(2*N0*x[k, 0]))
        ell1 = np.int(np.floor(2*N1*x[k, 1]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                phia = np.pi/(np.sqrt(mu0*mu1))*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k, 0])**2
                                                        - np.pi**2/mu1*((ell1-M1+i1)/(2*N1)-x[k, 1])**2))
                G[N0+ell0+i0, N1+ell1+i1] += f[k]*phia

    # wrapping array G
    [idx, idy] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    # accumulate by indexes (with possible index intersections)
    G = np.bincount(np.ndarray.flatten(idy0+idx0*(2*N1)), weights=np.real(np.ndarray.flatten(G))) +\
        1j*np.bincount(np.ndarray.flatten(idy0+idx0*(2*N1)),
                       weights=np.imag(np.ndarray.flatten(G)))
    G = np.reshape(G, [2*N0, 2*N1])

    # FFT and compesantion for smearing
    F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(G)))
    F = F[N0//2:N0//2+N0, N1//2:N1//2+N1]\
        / phi0[N0//2:N0//2+N0, N1//2:N1//2+N1]/(2*N0*2*N1)

    return F


def test():
    # accuracy of USFFT
    eps = 1e-12
    # size of the equally-spaced grid
    N = [32, 32]
    # unequally-spaced grid
    M = 4
    x = (np.random.random([M, 2])-0.5)
    # function on equally-spaced grid
    f = np.random.random(N)+1j*np.random.random(N)

    # compute Fourier transform from equally-spaced grid to unequally-spaced grid
    # direct summation
    G = eq2us(x, f, eps, N)
    # USFFT
    G_formula = eq2us_formula(x, f)

    # function on unequally-spaced grid
    f = np.random.random(M)+1j*np.random.random(M)
    # compute Fourier transform from equally-spaced grid to unequally-spaced grid
    # direct summation
    F_formula = us2eq_formula(x, f, N)
    # USFFT
    F = us2eq(x, f, eps, N)

    # normalized errors
    print(np.linalg.norm(G-G_formula))#/np.linalg.norm(G))
    print(np.linalg.norm(F-F_formula))#/np.linalg.norm(F))


if __name__ == '__main__':
    test()
