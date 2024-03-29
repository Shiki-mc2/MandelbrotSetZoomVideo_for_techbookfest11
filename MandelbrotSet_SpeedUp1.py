# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import cm
from numba import jit
from numba import uint8, uint16, uint32, float64

@jit((uint16, uint16, float64, float64, float64, uint32, uint32, uint8[:,:]))
def CalMandelbrotSet(row, col, x, y, r, N, R, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    nmg = np.zeros((row,col)  ,dtype=np.int32)
    
    xmin = x - r * (col-1)/(row-1)
    ymax = y + r
    dpp  = 2*r/(row-1)
    R2   = R*R
    
    for i in range(row):
        for j in range(col):
            z = 0.0
            c = xmin + dpp*j + 1j*(ymax - dpp*i)

            for k in range(N):
                z = z**2 + c
                if (z * z.conjugate()).real > R2: 
                    img[i,j,:] = colorlist[k]
                    nmg[i,j]   = k
                    break
            else:
                nmg[i,j] = N
    return img, nmg

def main():
    row     = 540
    col     = 720
    
    x       = -1.26222
    y       = -0.04592
    r       = 0.0001
    
    N       = 1000
    R       = 2.0

    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)

    img, nmg = CalMandelbrotSet(row, col, x, y, r, N, R, colorlist)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_SpeedUp1.png",img)
    print("r=%e n min=%5d n max=%5d" % (r, np.min(nmg), np.max(nmg)))

if __name__ == "__main__":
    main()
