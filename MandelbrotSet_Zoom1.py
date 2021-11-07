# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import cm

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
    fps     = 30
    
    zoom    = 0.982
    n_frame = 240
    
    x   = 0.45
    y   = 0.35
    r   = 2.00
    
    N   = 200
    R   = 2.0
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)
    
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    video  = cv2.VideoWriter("MandelbrotSet_Zoom1.mp4", fourcc, fps, (col, row))
    
    for i in range(n_frame):
        img, nmg = CalMandelbrotSet(row, col, x, y, r*(zoom**i), N, R, colorlist)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video.write(img)
        print("%3d, min n = %5d, max n = %5d" % (i, np.min(nmg), np.max(nmg)))
    video.release()

if __name__ == "__main__":
    main()