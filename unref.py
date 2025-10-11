
# Modified from WF_Diff/basicsr/metrics/psnr_ssim.py
import cv2
import numpy as np
from skimage import transform
from scipy import ndimage
import math
import argparse
import os


def getImageFiles(folder:str)->list:
    return sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.bmp', '.jpeg'))])


def getUCIQE(img):
    img_bgr =img

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    # if nargin == 1:                                 # According to training result mentioned in the paper:
    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val


def getUCIQE2(img):
    # HSV-based UCIQE
    # Deprecated
    image = img
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    # 求亮度对比值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe



def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    RG = R - G
    YB = (R+G)/2 -B
    K = R.shape[0]*R.shape[1]
    RG1 = RG.reshape(1,K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanRG = np.sum(RG1)/N
    deltaRG = np.sqrt(np.sum((RG1-meanRG)**2)/N)

    YB1 = YB.reshape(1,K)
    YB1 = np.sort(YB1)
    alphaL = 0.1
    alphaR = 0.1
    YB1 = YB1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB)**2)/N)
    uicm = -0.0268*np.sqrt(meanRG**2+meanYB**2)+ 0.1586*np.sqrt(deltaYB**2+deltaRG**2)
    return uicm

def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m-m%patchez+patchez)
        y = int(n-n%patchez+patchez)
        R = transform.resize(R,(x,y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))
    m = R.shape[0]
    n = R.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    AMEER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = R[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEER = 1/(k1*k2) *np.abs(AMEER)
    AMEEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = G[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEG = 1/(k1*k2) *np.abs(AMEEG)
    AMEEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = B[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEB = 1/(k1*k2) *np.abs(AMEEB)
    uiconm = AMEER +AMEEG +AMEEB
    return uiconm

def _uism(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    hx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    hy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest')+ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest')+ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest')+ndimage.convolve(B, hy, mode='nearest'))
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))
    m = SobelR.shape[0]
    n = SobelR.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    EMER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelR[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max/Min)
    EMER = 2/(k1*k2)*np.abs(EMER)

    EMEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelG[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max/Min)
    EMEG = 2/(k1*k2)*np.abs(EMEG)
    EMEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelB[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max/Min)
    EMEB = 2/(k1*k2)*np.abs(EMEB)
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism


def getUIQM(cv2BGR_img:np.ndarray)->float:
    x = cv2BGR_img
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

def main():
    parser = argparse.ArgumentParser(description='Compute UIQM and UCIQE of a given folder')
    parser.add_argument('--folder','-f', required=True, type=str, help='Directory of images')
    # output file
    parser.add_argument('--output','-o', required=True, type=str, help='Output file')
    args = parser.parse_args()

    files = getImageFiles(args.folder)
    count, UIQM_total, UCIQE_total = 0, 0, 0
    for file in files:
        img_BGR = cv2.imread(os.path.join(args.folder, file))
        uiqm = getUIQM(img_BGR)
        uciqe = getUCIQE(img_BGR)
        print(f"{file} UIQM: {uiqm} UCIQE: {uciqe}")
        UIQM_total += uiqm
        UCIQE_total += uciqe
        count += 1
        with open(args.output, 'a') as f:
            f.write(f"{file} UIQM: {uiqm} UCIQE: {uciqe} \n")
    print(f"Average UIQM: {UIQM_total/count} Average UCIQE: {UCIQE_total/count}")
    with open(args.output, 'a') as f:
        f.write(f"Average UIQM: {UIQM_total/count} Average UCIQE: {UCIQE_total/count}\n")


if __name__ == "__main__": 
    main()