# -*- coding: utf-8 -*-
import os
import csv
import time
import subprocess as cmd

import numpy as np
from   matplotlib import cm
import cv2
from   mpmath import mp

def CalMandelbrotSet(row, col, x, y, r, N, R, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    nmg = np.zeros((row,col)  ,dtype=np.int32)
    
    xmin = x - r * (col-1)/(row-1)
    ymax = y + r
    dpp  = 2*r/(row-1)
    
    fname = "cpp_%d_%04d.csv" % (int(time.time()), np.random.randint(10000))
    print(fname)

    prec = int(max(mp.log(1/dpp)/mp.log(2),49) + 3)
    print("prec=",prec)
    cmdline = ("CalMandelbrotSet %d %d %d %s %s %s %d %s %s" %
               (row, col, prec, xmin, ymax, dpp, N, R, fname))
    cmd.run(cmdline, shell=True)
    with open(fname,"r") as fr:
        cr = csv.reader(fr)
        for i, lines in enumerate(cr):
            for j , value in enumerate(lines):
                k = int(value)
                if k < N:
                    img[i,j,:] = colorlist[k]
                nmg[i,j]   = k
                
    os.remove(fname)
    return img, nmg

def main():
    row     = 540
    col     = 720
    
    mp.dps = 30
    x       = mp.mpf("-1.26222162762384535")
    y       = mp.mpf("-0.04591700163513884")
    r       = mp.mpf("0.000000000000005")
    
    N       = 1000
    R       = 2.0
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)
    
    img, nmg = CalMandelbrotSet(row, col, x, y, r, N, R, colorlist)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_MultiplePrecision2.png",img)
    print("r=%e n min=%5d n max=%5d" % (r, np.min(nmg), np.max(nmg)))

if __name__ == "__main__":
    main()