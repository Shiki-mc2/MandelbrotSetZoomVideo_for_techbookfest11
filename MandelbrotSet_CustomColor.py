# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numba import jit

@jit
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

def GenerateGradation(color_set, init_step, inc_rate, N, restart_idx = 0):
    colorlist = np.zeros((N,3))
    offset = 0
    j = 0
    
    grad = []
    n_grad = len(color_set) - 1
    for i in range(n_grad):
        grad.append(LinearSegmentedColormap.from_list('custom_cmap',
                    [(0.0,color_set[i]),(1.0,color_set[i+1])] ))
    
    for i in range(N):
        if i - offset == int(init_step):
            offset +=int(init_step)
            init_step = init_step*inc_rate
            j =(j + 1) % ( n_grad )
            if j == 0:
                j = restart_idx
        colorlist[i,:] = np.array(grad[j]((i - offset)/init_step)[:3])*255
    colorlist = np.array(colorlist,dtype=np.uint8)
    return colorlist
    
def main():
    row     = 540
    col     = 720
    
    x       = -0.4
    y       =  0.0
    r       =  1.6
    
    N       = 1000
    R       = 2.0
    
    color_set = [
            ( 0.10, 0.00, 0.00),
            ( 1.00, 0.00, 0.00),
            ( 1.00, 1.00, 0.00),
            ( 0.00, 1.00, 0.00),
            ( 0.00, 1.00, 1.00),
            ( 0.00, 0.00, 1.00),
            ( 1.00, 0.00, 1.00),
                 ]
    restart_idx = 1
    init_step   = 16
    inc_rate    = 1.1
    
    color_set.append(color_set[restart_idx])
    colorlist = GenerateGradation(color_set, init_step, inc_rate, N,
                                  restart_idx = restart_idx)
    
    img, nmg = CalMandelbrotSet(row, col, x, y, r, N, R, colorlist)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_CustomColor.png",img)
    print("r=%e n min=%5d n max=%5d" % (r, np.min(nmg), np.max(nmg)))

if __name__ == "__main__":
    main()