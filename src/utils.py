import numpy as np
import cv2
from tqdm import tqdm

#all funtions that have been implemented for previous questions

def gaussKernel(k, sigma):
    ret = np.zeros((k, k))
    center = k // 2
    for i in range(k):
        for j in range(k):
            ret[i][j] = np.exp( ((i-center)**2 + (j-center)**2) / (-2 * sigma) ) / (2 * np.pi * sigma ** 2)

    return ret / np.sum(ret) 

def padImage(img, margin):
    ret = np.zeros((img.shape[0] + margin * 2, img.shape[1] + margin * 2))
    ret[margin:img.shape[0]+margin, margin:img.shape[1]+margin] = img
    return ret.astype(int)

def apply2DKernel(img, kernel):
    k = kernel.shape[0]
    ret = np.zeros_like(img, dtype=int)
    padded = padImage(img, k // 2)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            ret[i][j] = np.sum(padded[i:i+k,j:j+k] * kernel)

    return ret.astype(int)
    

def bilateralFilter(img, k, sigma_s, sigma_r):
    ret = np.zeros_like(img)
    img = padImage(img, k//2)
    ss = 2 * (sigma_s ** 2)
    sr = 2 * (sigma_r ** 2)
    t1 = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            t1[i, j] = ((i - k//2)**2 + (j - k//2) ** 2) / ss

    for i in tqdm(range(ret.shape[0])):
        for j in range(ret.shape[1]):
            t =  ((img[i : i+k, j : j+k] - img[i + k//2, j + k//2]) ** 2) / sr
            t += t1
            t = np.exp(-t)

            ret[i, j] = np.sum(img[i:i+k, j:j+k] * t, axis =(0, 1)) / np.sum(t, axis=(0,1))

    return ret.astype(int)

def bilateralFilterColor(img, k, sigma_s, sigma_r):
    ret = np.copy(img)
    R = bilateralFilter(img[..., 0], k, sigma_s, sigma_r)
    G = bilateralFilter(img[..., 1], k, sigma_s, sigma_r)
    B = bilateralFilter(img[..., 2], k, sigma_s, sigma_r)

    ret[..., 0] = R
    ret[..., 1] = G
    ret[..., 2] = B

    return ret

def adaptiveThreshold(img, k, maxValue, thresh, c):
    '''adaptive threshold, k = blockSize'''
    print(c)
    ret = np.zeros_like(img, dtype=int)
    img = padImage(img, k//2 + 1).astype(int)

    pref = np.copy(img)
    pref = np.add.accumulate(img, axis=0)
    pref = np.add.accumulate(pref, axis=1)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            cur = pref[i, j] + pref[i+k, j+k] - pref[i+k, j] - pref[i, j+k] - c
            ret[i][j] = img[i + k//2, j+k//2] * (k ** 2) >= cur * (100 - thresh) / 100
    ret *= maxValue
    return ret.astype(np.uint8)


def DFT2D(img):
    X,Y = img.shape
    ret = np.zeros((X,Y),dtype=complex)
    for i in tqdm(range(X)):
        for j in range(Y):
            tot = 0.0
            for x in range(X):
                for y in range(Y):
                    e = np.exp(- 2j * np.pi * (float(i * x) / X + float(j * y) / Y))
                    tot +=  img[x, y] * e
            ret[i, j] = tot
    return ret

def FFT1D(a):
    '''recursive funtion to get fast fourier transform in 1D'''
    n = len(a)
    if n <= 1:
        return a
    else:
        res_eve = FFT1D(a[0::2])
        res_odd = FFT1D(a[1::2])
        w = np.exp((-2j *np.pi*np.arange(n // 2))/n)
        return np.append(res_eve + w * res_odd, res_eve - w * res_odd)

def FFT2D(img):
    '''apply fft to img'''
    return np.apply_along_axis(FFT1D, 1, np.apply_along_axis(FFT1D, 0, img))

def invFFT2D(data):
    '''get inverse FFT'''
    fstar = FFT2D(np.conj(data)) / (data.shape[0]*data.shape[1])
    return np.conj(fstar)

def dilate(img, se):
    m255 = False
    img = img.astype(np.float32)
    if np.max(img) > 1 :
        m255=True
        img /= 255

    ret = np.zeros_like(img)
    img = padImage(img, se.shape[0] // 2)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            cur = img[i:i+se.shape[0], j:j+se.shape[1]]
            curr = (cur == se)
            curr[se == 0] = 0
            if curr.any():
                ret[i, j] = 1
    if m255:
        ret *=255
    return ret.astype(np.uint8)
    

def erode(img, se):
    m255 = False
    img = img.astype(np.float32)
    if np.max(img) > 1 :
        m255=True
        img /= 255

    ret = np.zeros_like(img)
    img = padImage(img, se.shape[0] // 2)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            cur = img[i:i+se.shape[0], j:j+se.shape[1]]
            curr = (cur == se)
            curr[se == 0] = 1
            if curr.all():
                ret[i, j] = 1
    if m255:
        ret *=255
    return ret.astype(np.uint8)

def opening(img, se):
    return dilate(erode(img, se), se)


def closing(img, se):
    return erode(dilate(img, se), se)


def floodFill(id, curI, curJ, color):
    xsize, ysize = id.shape
    orig_value = id[curI, curJ]
    
    stack = set(((curI, curJ),))
    connects = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]
    while stack:
        i, j = stack.pop()

        if id[i, j] == orig_value:
            id[i, j] = color
            for ii, jj in connects:
                if i + ii >= 0 and j+jj >=0 and i +ii < xsize and j+jj < ysize:
                    stack.add((i+ii, j+jj))

def getRegions(img, bg=True):
    ret = np.zeros((img.shape[0], img.shape[1],))
    id = np.ones_like(img).astype(int) * (-1) 
    id[img == 0] = 0
    curCol = 2
    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            if id[i, j] == -1:
                if bg:
                    if i == 0 or j ==0 or i == img.shape[0]-1 or j == img.shape[1] -1 :
                        floodFill(id, i, j, 1)
                        continue
                floodFill(id, i, j, curCol)
                curCol+=1
    print(curCol)
    return id

def threshold(img, thresh, inverse=False, maxVal=255):
    ret = np.zeros_like(img)
    ret = img >= thresh
    if inverse:
        ret = 1-ret
        
    return ret*maxVal

def getOtsuThresh(img):
    hist = getHistogram(img)
    sum = np.sum(np.arange(256) * hist)
    cur = 0
    q1 = 0
    q2 = 0
    maxVar = 0
    curVar = 0
    ret = 0
    N = img.shape[0] * img.shape[1]
    for i in range(256):
        q1 += hist[i]
        if q1 == 0:
            continue
        q2 = N - q1
        if q2 == 0:
            break
        cur += float(i * hist[i])

        m1 = cur/ q1
        m2 = (sum - cur)/ q2

        curVar = q1 *q2 * ((m1-m2)**2)
        if curVar > maxVar:
            maxVar = curVar
            ret = i
    return ret

def getHistogram(img):
    ret = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[img[i][j]] += 1
    return ret

def randomColor():
    return np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])

def colorIdimg(id):
    ret = np.zeros((id.shape[0], id.shape[1], 3))
    black = np.zeros(3)
    white = np.ones(3) * 255

    numColors = np.max(id)

    ret[id == 0, :] = black
    ret[id == 1, :] = white

    print(id[11, 10])

    for i in tqdm(range(numColors - 1)):
        cur = i + 2
        ret[id == cur] = randomColor()

    return ret.astype(np.uint8)
    

def showRegions(regions, selectedRegions):
    ret = np.zeros_like(regions)
    for i in selectedRegions:
        ret[regions == i] = 1
    return ret

def fillHoles(img, maxHoleSize):
    inv = 1-img
    ret = np.copy(img)
    regions = getRegions(inv)
    for i in np.unique(regions):
        if np.sum(regions == i) <= maxHoleSize: ret[regions == i] = 1

    return ret