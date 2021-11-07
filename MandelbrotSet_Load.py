# -*- coding: utf-8 -*-
import pickle

import numpy as np
from matplotlib import cm
import cv2
from numba import jit

@jit
def CalMandelbrotSet_dat(row, col, N, nmg, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    
    for i in range(row):
        for j in range(col):
            if nmg[i,j] < N:
                img[i,j,:] = colorlist[nmg[i,j]]
    return img

def main():
    fname = "MandelbrotSet_Save.dat"
    with open(fname,"rb") as frb:
            [row, col, x, y, r, N, R, nmg] = pickle.load(frb)
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.hsv(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)
    
    img = CalMandelbrotSet_dat(row, col, N, nmg, colorlist)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_Load.png",img)
    print("min n = %d" % np.min(nmg), "max n = %d" % np.max(nmg))
    
if __name__ == "__main__":
    main()