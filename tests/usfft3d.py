import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def eq2us_formula(x, f):
    """
    Direct computation of DFT from equally-spaced grid to unequally-spaced grid
    x - unequally-spaced grid 
    f - function on a regular grid
    """

    [N0, N1, N2] = f.shape
    xeq0 = np.arange(-N0//2, N0//2)
    xeq1 = np.arange(-N1//2, N1//2)
    xeq2 = np.arange(-N2//2, N2//2)
    F = np.zeros(x.shape[0], dtype=complex)
    for k in range(x.shape[0]):
        for i0 in range(N0):
            for i1 in range(N1):
                for i2 in range(N2):
                    F[k] = F[k]+f[i0, i1, i2] * \
                        np.exp(-2*np.pi*1j *
                               (xeq0[i0]*x[k, 0]+xeq1[i1]*x[k, 1]+xeq2[i2]*x[k, 2]))
    return F


def us2eq_formula(x, f, N):
    """
    Direct computation of DFT from unequally-spaced grid to equally-spaced grid
    x - unequally-spaced grid 
    f - function on the grid x
    """
    print(f.shape)
    [N0, N1, N2] = N
    xeq0 = np.arange(-N0//2, N0//2)
    xeq1 = np.arange(-N1//2, N1//2)
    xeq2 = np.arange(-N2//2, N2//2)
    F = np.zeros([N0, N1, N2], dtype=complex)
    for k in range(x.shape[0]):
        for i0 in range(N0):
            for i1 in range(N1):
                for i2 in range(N2):
                    F[i0, i1, i2] = F[i0, i1, i2]+f[k] * \
                        np.exp(-2*np.pi*1j *
                               (xeq0[i0]*x[k, 0]+xeq1[i1]*x[k, 1]+xeq2[i2]*x[k, 2]))
    return F


def eq2us(x, f, eps, N):
    """
    USFFT from equally-spaced grid to unequally-spaced grid
    x - unequally-spaced grid 
    f - function on a regular grid of size N
    eps - accuracy of computing USFFT
    """
    print(N)
    print(f.shape)
    print(x.shape)
    # parameters for the USFFT transform
    [N0, N1, N2] = N
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    mu2 = -np.log(eps)/(2*N2**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    Te3 = 1/np.pi*np.sqrt(-mu2*np.log(eps)+(mu2*N2)**2/4)
    M0 = np.int(np.ceil(2*N0*Te1))
    M1 = np.int(np.ceil(2*N1*Te2))
    M2 = np.int(np.ceil(2*N2*Te3))

    # smearing kernel (theta0)
    theta0 = np.zeros((2*N0, 2*N1, 2*N2))
    [xeq0, xeq1, xeq2] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2, -N2//2:N2//2]
    theta0[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2 +
         N2] = np.exp(-mu0*xeq0**2-mu1*xeq1**2-mu2*xeq2**2)

    # FFT and compesantion for smearing
    fe = np.zeros([2*N0, 2*N1, 2*N2], dtype=complex)
    fe[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2] = f / \
        (2*N0*2*N1*2*N2)/theta0[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2]
    Fe0 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(fe)))

    # wrapping array Fe0
    [idx, idy, idz] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1, -M2:2*N2+M2]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    idz0 = np.mod(idz+2*N2, 2*N2)
    Fe = np.zeros([2*N0+2*M0, 2*N1+2*M1, 2*N2+2*M2], dtype=complex)
    Fe[idx+M0, idy+M1, idz+M2] = Fe0[idx0, idy0, idz0]

    # smearing operation (Fe=f*theta)
    F = np.zeros(x.shape[0], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int(np.floor(2*N0*x[k, 0]))
        ell1 = np.int(np.floor(2*N1*x[k, 1]))
        ell2 = np.int(np.floor(2*N2*x[k, 2]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                for i2 in range(2*M2+1):
                    if(N0+ell0+i0<2*N0+2*M0 and N0+ell0+i0>=0 and N1+ell1+i1<2*N1+2*M1 and N1+ell1+i1>=0 and N2+ell2+i2<2*N2+2*M2 and N2+ell2+i2>=0):
                        F[k] += Fe[N0+ell0+i0, N1+ell1+i1, N2+ell2+i2] * \
                            np.sqrt(np.pi)**3/np.sqrt(mu0*mu1*mu2)*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k, 0])**2- np.pi**2/mu1 *((ell1-M1+i1) / (2*N1)-x[k, 1])**2- np.pi**2/mu2*((ell2-M2+i2)/(2*N2)-x[k, 2])**2))
    return F


def us2eq(x, f, eps, N):
    """
    USFFT from unequally-spaced grid to equally-spaced grid
    x - unequally-spaced grid 
    f - function on the grid x
    eps - accuracy of computing USFFT
    """
    # parameters for the USFFT transform
    [N0, N1, N2] = N
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    mu2 = -np.log(eps)/(2*N2**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    Te3 = 1/np.pi*np.sqrt(-mu2*np.log(eps)+(mu2*N2)**2/4)
    M0 = np.int(np.ceil(2*N0*Te1))
    M1 = np.int(np.ceil(2*N1*Te2))
    M2 = np.int(np.ceil(2*N2*Te3))

    # smearing kernel (theta0)
    theta0 = np.zeros((2*N0, 2*N1, 2*N2))
    [xeq0, xeq1, xeq2] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2, -N2//2:N2//2]
    theta0[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2 +
         N2] = np.exp(-mu0*xeq0**2-mu1*xeq1**2-mu2*xeq2**2)

    # smearing operation (G=f*thetaa)
    G = np.zeros([2*N0+2*M0, 2*N1+2*M1, 2*N2+2*M2], dtype=complex)
    for k in range(x.shape[0]):
        ell0 = np.int(np.floor(2*N0*x[k, 0]))
        ell1 = np.int(np.floor(2*N1*x[k, 1]))
        ell2 = np.int(np.floor(2*N2*x[k, 2]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                for i2 in range(2*M2+1):
                    if(N0+ell0+i0<2*N0+2*M0 and N0+ell0+i0>=0 and N1+ell1+i1<2*N1+2*M1 and N1+ell1+i1>=0 and N2+ell2+i2<2*N2+2*M2 and N2+ell2+i2>=0):
                        thetaa = np.sqrt(np.pi)**3/np.sqrt(mu0*mu1*mu2)*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k, 0])**2
                                                                            - np.pi**2/mu1 *
                                                                            ((ell1-M1+i1) /
                                                                            (2*N1)-x[k, 1])**2
                                                                            - np.pi**2/mu2*((ell2-M2+i2)/(2*N2)-x[k, 2])**2))
                        G[N0+ell0+i0, N1+ell1+i1, N2+ell2+i2] += f[k]*thetaa

    # wrapping array G
    [idx, idy, idz] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1, -M2:2*N2+M2]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    idz0 = np.mod(idz+2*N2, 2*N2)
    # accumulate by indexes (with possible index intersections)
    G = np.bincount(np.ndarray.flatten(idz0+idy0*(2*N2)+idx0*(2*N1*2*N2)), weights=np.real(np.ndarray.flatten(G))) +\
        1j*np.bincount(np.ndarray.flatten(idz0+idy0*(2*N2)+idx0*(2*N1*2*N2)),
                       weights=np.imag(np.ndarray.flatten(G)))
    G = np.reshape(G, [2*N0, 2*N1, 2*N2])

    # FFT and compesantion for smearing
    F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(G)))
    F = F[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2]\
        / theta0[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2]/(2*N0*2*N1*2*N2)

    return F


def fwd_laminography(f, theta, phi, detx, dety, N):
    [ku, kv] = np.meshgrid(np.arange(-detx//2, detx//2) /
                           detx, np.arange(-dety//2, dety//2)/dety)
    ku = np.ndarray.flatten(ku)
    kv = np.ndarray.flatten(kv)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    
    x = np.zeros([len(theta), detx*dety, 3])
    for itheta in range(len(theta)):
        x[itheta, :, 0] = ku*np.cos(theta[itheta])+kv*np.sin(theta[itheta])*np.cos(phi)
        x[itheta, :, 1] = ku*np.sin(theta[itheta])-kv*np.cos(theta[itheta])*np.cos(phi)
        x[itheta, :, 2] = kv*np.sin(phi)
        ax.scatter(x[itheta, :, 0], x[itheta, :, 1], x[itheta, :, 2], c='blue')        
        plt.pause(0.03)
    plt.show()
    x = np.reshape(x, [len(theta)*detx*dety, 3])    
    F = eq2us(x, f, 1e-4, N)
    F = F.reshape([len(theta), detx,dety])
    res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F,axes=(1,2)),axes=(1,2),norm="ortho"),axes=(1,2))
    return res

def adj_laminography(f, theta, phi, detx, dety, N):
    F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f,axes=(1,2)),axes=(1,2),norm="ortho"),axes=(1,2))
    [ku, kv] = np.meshgrid(np.arange(-detx//2, detx//2) /
                           detx, np.arange(-dety//2, dety//2)/dety)
    ku = np.ndarray.flatten(ku)
    kv = np.ndarray.flatten(kv)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    
    x = np.zeros([len(theta), detx*dety, 3])
    for itheta in range(len(theta)):
        x[itheta, :, 0] = ku*np.cos(theta[itheta])+kv*np.sin(theta[itheta])*np.cos(phi)
        x[itheta, :, 1] = ku*np.sin(theta[itheta])-kv*np.cos(theta[itheta])*np.cos(phi)
        x[itheta, :, 2] = kv*np.sin(phi)
        ax.scatter(x[itheta, :, 0], x[itheta, :, 1], x[itheta, :, 2], c='blue')
        # plt.show()
        plt.pause(0.03)
    plt.show()
    x = np.reshape(x, [len(theta)*detx*dety, 3])    
    F = np.ndarray.flatten(F)    
    res = us2eq(-x, F, 1e-4, N)    
    return res    


def test():
    # accuracy of USFFT
    eps = 1e-12
    # size of the equally-spaced grid
    N = [8, 8, 8]
    # unequally-spaced grid
    M = 16
    x = (np.random.random([M, 3])-0.5)
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
    print(np.linalg.norm(G-G_formula)/np.linalg.norm(G))
    print(np.linalg.norm(F-F_formula)/np.linalg.norm(F))
    
    f = np.random.random(N)+1j*np.random.random(N)
    R=fwd_laminography(f, np.linspace(0, np.pi*2, 8), np.pi/6, 8, 8, N)
    fn=adj_laminography(R, np.linspace(0, np.pi*2, 8), np.pi/6, 8, 8, N)
    print(np.sum(R*np.conj(R)))
    print(np.sum(f*np.conj(fn)))
    

if __name__ == '__main__':
    test()
