import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def fastReflectionSuppression(img, h, lmbda=0, epsilon=0.00001, mu=1):
    '''
    args: img: 3 dimensional image with values from 0-255
    h:
    lmbda: (lambda)
    epsilon: 
    mu:
    '''
    img = img.astype(float)
    img /= 255

    T = getT(getRHS(img, epsilon, h), mu, lmbda, epsilon)
    
    constrastStretching(T)
    return T


def constrastStretching(ar: np.array, new_min: float = 0, new_max: float = 1):
    '''
    Linear contrast stretching
    '''
    ar -= ar.min()
    ar /= ar.max()
    ar *= (new_max - new_min) + new_min

def getT(rhs, mu,lmbd, epsilon):
    '''
    Obtain T from RHS of eq7 with DCTs
    '''
    T = np.zeros_like(rhs).astype(float)
    M, N = rhs.shape[:2]
    for i in range(3):
        m = np.cos((np.pi * np.arange(M)) / M)
        n = np.cos((np.pi * np.arange(N)) / N)
        K = 2 * (np.add.outer(m, n) - 2)

        denom = mu * (K ** 2) - lmbd * K + epsilon    

        u = dct2D(rhs[..., i])

        u = u / denom
        u = idct2D(u)

        T[..., i] = u
        
    return T


def getRHS(image, epsilon, h):
    '''
    computes right-hand side of equation (7)
    '''
    channels = image.shape[-1]
    laplacians = np.zeros(shape=image.shape)
    for c in range(channels):
        lapl1 = laplacian(image[..., c], h=h)
        temp = divergence(gradient(lapl1))
        laplacians[:, :, c] = temp
    # L(...) + \epsilon * Y
    rhs = laplacians + epsilon * image
    return rhs

def laplacian(f, h):
    '''
    compute the divergence of gradient after thresholding with h
    '''
    grad = gradient(f)
    norm = np.linalg.norm(grad, axis=-1)
    
    mask = (norm < h)[..., np.newaxis].repeat(2, axis=-1)
    grad[mask] = 0
    
    # and compute its divergence by summing the second-order gradients
    laplacian = divergence(grad)
    
    return laplacian

def gradient(A: np.ndarray):
    """
    returns nparray tuple of gradient in dir0, dir1
    """
    
    rows, cols = A.shape
    
    grad_x = np.zeros_like(A)
    grad_x[:, 0: cols - 1] = np.diff(A, axis=1)

    grad_y = np.zeros_like(A)
    grad_y[0:rows - 1, :] = np.diff(A, axis=0)

    B = np.concatenate((grad_x[..., np.newaxis], grad_y[..., np.newaxis]), axis=-1)

    return B

def divergence(gradient):
    '''
    returns divergence of gradient as a 2D array of tuples
    '''
    B = np.zeros(gradient.shape[:2])
    T = gradient[..., 0]
    
    T1 = np.zeros(gradient.shape[:2])
    T1[:, 1:] = T[:, :-1]
    B += T - T1
    
    T = gradient[..., 1]
    T1 = np.zeros(gradient.shape[:2])
    T1[1:, :] = T[:-1, :]
    B += T - T1
    return B


def dct2D(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2D(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
