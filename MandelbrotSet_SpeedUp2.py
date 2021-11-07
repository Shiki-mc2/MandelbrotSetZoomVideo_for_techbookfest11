# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import cm
from numba import cuda
from numba import uint8,uint16,uint32,float64

@cuda.jit((uint16,uint16,uint16,uint16,float64,float64,float64,uint32,float64,
           uint8[:,:,:],uint32[:,:],uint8[:,:]))
def CalMandelbrotSet_CUDA(row_sta, row_end, col_sta, col_end, xmin, ymax, dpp, N, R2,
                          img, nmg, colorlist):
    row_g , col_g = cuda.grid(2) 
    
    row_d = cuda.gridDim.x * cuda.blockDim.x;
    col_d = cuda.gridDim.y * cuda.blockDim.y;
    
    for i in range(row_sta + row_g, row_end, row_d):
        for j in range(col_sta + col_g, col_end, col_d):
            i_sub = i - row_sta
            j_sub = j - col_sta
            z = complex(0.0)
            c = xmin + dpp*j + 1j*(ymax - dpp*i)
            
            for k in range(N):
                z = z**2 + c
                if (z * z.conjugate()).real > R2:
                    img[i_sub,j_sub,0] = colorlist[k,0]
                    img[i_sub,j_sub,1] = colorlist[k,1]
                    img[i_sub,j_sub,2] = colorlist[k,2]
                    nmg[i_sub,j_sub]   = k
                    break
            else:
                nmg[i_sub,j_sub] = N
           
def CalMandelbrotSet(row, col, x, y, r, N, R, colorlist, griddim, blockdim, sub_dvi):
    img = np.zeros((row,col,3),dtype=np.uint8)
    nmg = np.zeros((row,col)  ,dtype=np.int32)

    xmin = x - r * (col-1)/(row-1)
    ymax = y + r
    dpp  = 2*r/(row-1)
    R2   = R*R
    
    row_unit = row//sub_dvi
    col_unit = col//sub_dvi
    
    for i in range(sub_dvi):
        row_sta = i * row_unit
        if i == sub_dvi - 1:
            row_end = row
        else:
            row_end = row_sta + row_unit

        for j in range(sub_dvi):
            col_sta = j * col_unit
            if j == sub_dvi - 1:
                col_end = col
            else:
                col_end = col_sta + col_unit

            img_sub = np.zeros((row_end-row_sta,col_end-col_sta,3),dtype=np.uint8)
            nmg_sub = np.zeros((row_end-row_sta,col_end-col_sta),  dtype=np.int32)
            
            d_img   = cuda.to_device(img_sub)
            d_nmg   = cuda.to_device(nmg_sub)
            
            CalMandelbrotSet_CUDA[griddim, blockdim](
                row_sta, row_end, col_sta, col_end, xmin, ymax, dpp, N, R2,
                d_img, d_nmg, colorlist)
            
            d_img.copy_to_host(img_sub)
            d_nmg.copy_to_host(nmg_sub)
    
            img[row_sta:row_end, col_sta:col_end,:] = img_sub[:,:,:]
            nmg[row_sta:row_end, col_sta:col_end]   = nmg_sub[:,:]

    return img, nmg

def main():
    row     = 1080
    col     = 1440

    x       = -1.2622216276238
    y       = -0.0459170016351
    r       =  0.00000000001
    
    N       = 1000
    R       = 2.0
    
    griddim  = ( 4, 4)
    blockdim = (16,16)
    sub_dvi  = 4
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)
            
    img, nmg = CalMandelbrotSet(row, col, x, y, r, N, R, colorlist,
                                griddim, blockdim, sub_dvi)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_SpeedUp2.png",img)
    print("r=%e n min=%5d n max=%5d" % (r, np.min(nmg), np.max(nmg)))

if __name__ == "__main__":
    main()
        