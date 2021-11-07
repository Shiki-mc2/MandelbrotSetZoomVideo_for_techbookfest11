# -*- coding: utf-8 -*-
import numpy as np
import cv2
from   mpmath import mp

import MandelbrotSet as mb

def main():
    row     = 540
    col     = 720
    
    mp.dps  = 30
    x       = mp.mpf("0.444953")
    y       = mp.mpf("0.345580")
    r       = mp.mpf("0.00003")
    
    N       = 1000
    R       = 2.0
    
    mode    = "jit"
    
    griddim  = ( 8, 8)
    blockdim = (16,16)
    sub_dvi  = 4
    
    color_set = [
            ( 0.00, 0.00, 0.50),
            ( 0.00, 0.50, 1.00),
            ( 1.00, 1.00, 1.00),
            ( 1.00, 1.00, 0.00),
            ( 1.00, 0.00, 0.00),
                 ]
    restart_idx = 1
    color_set.append(color_set[restart_idx])
    colorlist = mb.GenerateGradation(color_set, 16, 1.1, N, restart_idx)

    img, nmg = mb.CalMandelbrotSet(row, col, x, y, r, N, R, colorlist, mode=mode,
                                   griddim = griddim, blockdim = blockdim,
                                   sub_dvi = sub_dvi)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_OneShot.png",img)
    
    print("min n = %d" % np.min(nmg), "max n = %d" % np.max(nmg))
    
    with open("MandelbrotSet_OneShot.csv","w") as fw:
        fw.write("ID,row,col,x,y,r,N,R,minL,maxL\n")
        fw.write("%03d,%d,%d,%s,%s,%13.8e,%d,%f,%d,%d\n" %
                 (0, row, col, x, y, r, N, R, np.min(nmg), np.max(nmg)))
    
if __name__ == "__main__":
    main()