# -*- coding: utf-8 -*-
import sys
import pickle

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
    r       = mp.mpf("2.0")
    
    N       = 1000
    R       = 2.0
    
    mode    = "jit"
    
    zoom    = 0.8
    i_zoom  = range(50)
    
    is_load = True
    is_save = True
    
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

    if len(sys.argv) > 1:
        i_zoom = [int(sys.argv[1])]
        
    if len(sys.argv) > 2:
        mode = sys.argv[2]

    row_zoom = int(round(row/zoom))
    col_zoom = int(round(col/zoom))
    
    for i in i_zoom:
        r_zoom   = r*zoom**i
        fname = "MandelbrotSet_%04d.dat" % i
        
        img, nmg = mb.CalMandelbrotSet(
            row_zoom, col_zoom, x, y, r_zoom, N, R, colorlist, mode=mode,
            griddim = griddim, blockdim = blockdim, sub_dvi = sub_dvi,
            fname = is_load and fname or "")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite("MandelbrotSet_%04d.png" % i,img)
        
        if is_save:
            with open(fname,"wb") as fwb:
                pickle.dump([row_zoom, col_zoom, x, y, r_zoom, N, R, nmg], fwb)
        
        print("id=%3d, r=%8.3e min n = %5d, max n = %5d" %
              (i, r_zoom, np.min(nmg), np.max(nmg)))
        
        with open("MandelbrotSet_%04d.csv" % i,"w") as fw:
            fw.write("ID,row,col,x,y,r,N,R,minL,maxL\n")
            fw.write("%03d,%d,%d,%s,%s,%13.8e,%d,%f,%d,%d\n" %
                     (i, row, col, x, y, r_zoom, N, R, np.min(nmg), np.max(nmg)))
    
if __name__ == "__main__":
    main()