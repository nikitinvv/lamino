import numpy as np
import lamcg as lcg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def eq2us(x, f, eps, N):
    """
    USFFT from equally-spaced grid to unequally-spaced grid
    x - unequally-spaced grid 
    f - function on a regular grid of size N
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
    M0 = np.int32(np.ceil(2*N0*Te1))
    M1 = np.int32(np.ceil(2*N1*Te2))
    M2 = np.int32(np.ceil(2*N2*Te3))
    
    # smearing kernel (ker)
    ker = np.zeros((2*N0, 2*N1, 2*N2))
    [xeq0, xeq1, xeq2] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2, -N2//2:N2//2]
    ker[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2 +
         N2] = np.exp(-mu0*xeq0**2-mu1*xeq1**2-mu2*xeq2**2)

    # FFT and compesantion for smearing
    fe = np.zeros([2*N0, 2*N1, 2*N2], dtype=complex)
    fe[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2] = f / \
        (2*N0*2*N1*2*N2)/ker[N0//2:N0//2+N0, N1//2:N1//2+N1, N2//2:N2//2+N2]
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
        ell0 = np.int32(np.floor(2*N0*x[k, 0]))
        ell1 = np.int32(np.floor(2*N1*x[k, 1]))
        ell2 = np.int32(np.floor(2*N2*x[k, 2]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                for i2 in range(2*M2+1):
                    F[k] += Fe[N0+ell0+i0, N1+ell1+i1, N2+ell2+i2] * np.sqrt(np.pi)**3/np.sqrt(mu0*mu1*mu2)*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k, 0])**2-np.pi**2/mu1*((ell1-M1+i1)/(2*N1)-x[k, 1])**2-np.pi**2/mu2*((ell2-M2+i2)/(2*N2)-x[k, 2])**2))
    return F


def eq2us1d(x, f, eps, N):
    # parameters for the USFFT transform
    N0 = N
    mu0 = -np.log(eps)/(2*N0**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    M0 = np.int32(np.ceil(2*N0*Te1))
    
    # smearing kernel (ker)
    ker = np.zeros([2*N0,1,1])
    xeq0 = np.arange(-N0//2,N0//2)
    ker[N0//2:N0//2+N0] = np.exp(-mu0*xeq0**2)[:,np.newaxis,np.newaxis]

    # FFT and compesantion for smearing
    fe = np.zeros([2*N0,f.shape[1],f.shape[2]], dtype=complex)
    fe[N0//2:N0//2+N0,:] = f/(2*N0)/(ker[N0//2:N0//2+N0])
    Fe0 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(fe,axes=0),axis=0),axes=0)
    
    # wrapping array Fe0
    idx = np.arange(-M0,2*N0+M0)
    idx0 = np.mod(idx+2*N0, 2*N0)
    Fe = np.zeros([2*N0+2*M0,*Fe0.shape[1:]], dtype=complex)
    Fe[idx+M0] = Fe0[idx0]    

    # smearing operation (Fe=f*theta)
    F = np.zeros([x.shape[0],*f.shape[1:]], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int32(np.floor(2*N0*x[k]))        
        for i0 in range(2*M0+1):
            F[k] += Fe[N0+ell0+i0] * np.sqrt(np.pi/mu0)*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k])**2))
    return F


def eq2us2d(x, s, f, eps, N):
    # parameters for the USFFT transform
    [N0, N1] = N
    mu0 = -np.log(eps)/(2*N0**2)
    mu1 = -np.log(eps)/(2*N1**2)
    Te1 = 1/np.pi*np.sqrt(-mu0*np.log(eps)+(mu0*N0)**2/4)
    Te2 = 1/np.pi*np.sqrt(-mu1*np.log(eps)+(mu1*N1)**2/4)
    M0 = np.int32(np.ceil(2*N0*Te1))
    M1 = np.int32(np.ceil(2*N1*Te2))
    
    # smearing kernel (ker)
    ker = np.zeros((2*N0, 2*N1))
    [xeq0, xeq1] = np.mgrid[-N0//2:N0//2, -N1//2:N1//2]
    ker[N0//2:N0//2+N0, N1//2:N1//2+N1] = np.exp(-mu0*xeq0**2-mu1*xeq1**2)
    # FFT and compesantion for smearing
    fe = np.zeros([f.shape[0], 2*N0, 2*N1], dtype=complex)
    fe[:, N0//2:N0//2+N0, N1//2:N1//2+N1] = f / (2*N0*2*N1)/ker[N0//2:N0//2+N0, N1//2:N1//2+N1]
    Fe0 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fe,axes=[1,2])),axes=[1,2])
    
    # wrapping array Fe0
    [idx, idy] = np.mgrid[-M0:2*N0+M0, -M1:2*N1+M1]
    idx0 = np.mod(idx+2*N0, 2*N0)
    idy0 = np.mod(idy+2*N1, 2*N1)
    Fe = np.zeros([f.shape[0],2*N0+2*M0, 2*N1+2*M1], dtype=complex)
    Fe[:, idx+M0, idy+M1] = Fe0[:, idx0, idy0]
    
    # smearing operation (Fe=f*theta)
    F = np.zeros([x.shape[0]], dtype=complex)
    for k in range(x.shape[0]):
        F[k] = 0
        ell0 = np.int32(np.floor(2*N0*x[k, 0]))
        ell1 = np.int32(np.floor(2*N1*x[k, 1]))
        for i0 in range(2*M0+1):
            for i1 in range(2*M1+1):
                F[k] += Fe[s[k],N0+ell0+i0, N1+ell1+i1] * np.pi/np.sqrt(mu0*mu1)*(np.exp(-np.pi**2/mu0*((ell0-M0+i0)/(2*N0)-x[k,0])**2-np.pi**2/mu1*((ell1-M1+i1)/(2*N1)-x[k,1])**2))
    return F



def fwd_laminography(f, theta, phi, det, N):
    [ku, kv] = np.meshgrid(np.arange(-det//2, det//2) /
                           det, np.arange(-det//2, det//2)/det)
    ku = np.ndarray.flatten(ku)
    kv = np.ndarray.flatten(kv)
    
    x = np.zeros([len(theta), det*det, 3])
    for itheta in range(len(theta)):
        x[itheta, :, 2] = ku*np.cos(theta[itheta])+kv*np.sin(theta[itheta])*np.cos(phi)
        x[itheta, :, 1] = kv*np.sin(theta[itheta])-kv*np.cos(theta[itheta])*np.cos(phi)
        x[itheta, :, 0] = kv*np.sin(phi)
    x[x>=0.5] = 0.5 - 1e-5
    x[x<-0.5] = -0.5 + 1e-5

    F0 = eq2us(np.reshape(x, [len(theta)*det*det, 3]),f,1e-3,N)    
    
    
    F = eq2us1d(x[0, ::det, 0],f,1e-3,N[0])    
    s = np.tile(np.arange(det*det)//(det),ntheta)
    F = eq2us2d(x[:, :, 1:3].reshape([x.shape[0]*x.shape[1], 2]),s,F,1e-3,[N[1],N[2]]).flatten()    
    
    F = F.reshape([len(theta), det,det])
    F0 = F0.reshape([len(theta), det,det])
    
    
    plt.subplot(2,3,1)
    plt.imshow(np.abs(F[0]))
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.imshow(np.abs(F0[0]))
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.imshow(np.abs(F0[0]-F[0]))    
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.imshow(np.abs(F[1]))
    plt.subplot(2,3,5)
    plt.imshow(np.abs(F0[1]))
    plt.subplot(2,3,6)
    plt.imshow(np.abs(F0[1]-F[1]))    
    
    
    print(np.linalg.norm(F0))
    print(np.linalg.norm(F))
    print(np.linalg.norm(F)/np.linalg.norm(F0))
    # print(s)
    exit()

    
    x = np.reshape(x, [len(theta)*det*det, 3])    
    F = eq2us(x, f, 1e-3, N)
    
    
    F = F.reshape([len(theta), det,det])
    res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F,axes=(1,2)),axes=(1,2),norm="ortho"),axes=(1,2))
    return res


n0 = 8
n1 = 8
n2 = 8
det = 8
ntheta = 8
phi = np.pi/2-30/180*np.pi
theta = np.linspace(1,2*np.pi,ntheta,endpoint=False).astype('float32')
f = np.zeros([n0,n1,n2]).astype('complex64')
f = np.random.random([n0,n1,n2])+1j*np.random.random([n0,n1,n2])
# f[n0//8:3*n0//8,n1//4:3*n1//4,n2//4:3*n2//4]=1

g = fwd_laminography(f, theta, phi, det, [n0,n1,n2])    
ff = adj_laminography(g, theta, phi, det, [n0,n1,n2])
print(np.sum(f*np.conj(ff)))
print(np.sum(g*np.conj(g)))
