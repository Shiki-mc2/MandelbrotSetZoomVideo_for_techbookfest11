# -*- coding: utf-8 -*-

import numpy as np
import cv2

from mpmath import mp

import MandelbrotSet as mb

def MargeAndPop(row, col, row_zoom, col_zoom, imgs):
    ridx1 = (row_zoom - row)//2
    ridx2 = (row_zoom + row)//2
    cidx1 = (col_zoom - col)//2
    cidx2 = (col_zoom + col)//2
    imgs[1][1][:] = (imgs[1][1][:]*0.5 + 
                     cv2.resize(imgs[0][1][ridx1:ridx2,cidx1:cidx2],
                                (col_zoom, row_zoom))*0.5)
    imgs.pop(0)

def ZoomIn(isInit, fpz, r, zoom, imgs, row, col, row_zoom, col_zoom, video):
    for i in range(1-isInit, fpz+1):
        img = []
        r_zoom = r*(zoom**(i/fpz))
        for j in range(len(imgs)):
            if imgs[j][0] > r_zoom or (isInit and j == 0):
                r_ratio = r_zoom / imgs[j][0]
                row_half = (row_zoom - 1) / 2
                col_half = (col_zoom - 1) / 2
                ridx1 = int(round((1 - r_ratio)*row_half))
                ridx2 = int(round((1 + r_ratio)*row_half))+1
                cidx1 = int(round((1 - r_ratio)*col_half))
                cidx2 = int(round((1 + r_ratio)*col_half))+1
                if len(img) == 0:
                    img = cv2.resize(imgs[j][1][ridx1:ridx2,cidx1:cidx2],(col, row),
                                     interpolation = cv2.INTER_AREA)
                else:                        
                    rimg = cv2.resize(imgs[j][1][ridx1:ridx2,cidx1:cidx2],(col, row),
                                      interpolation = cv2.INTER_AREA)
                    img[:] = img[:]*0.3 + rimg[:]*0.7
            else:
                r_ratio = imgs[j][0] / r_zoom
                row_half = (row - 1) / 2
                col_half = (col - 1) / 2
                ridx1 = int(round((1 - r_ratio)*row_half))
                ridx2 = int(round((1 + r_ratio)*row_half))+1
                cidx1 = int(round((1 - r_ratio)*col_half))
                cidx2 = int(round((1 + r_ratio)*col_half))+1
                
                row_sub = ridx2 - ridx1
                col_sub = cidx2 - cidx1
                rimg = cv2.resize(imgs[j][1],(col_sub,row_sub),
                                  interpolation = cv2.INTER_AREA)
                img[ridx1:ridx2,cidx1:cidx2] = (
                    img[ridx1:ridx2,cidx1:cidx2]*0.3+rimg*0.7)

        if video:
            vimg = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            video.write(vimg)

def main():
    row     = 540
    col     = 720
    fps     = 30
    
    mp.dps  = 30
    x       = mp.mpf("0.444953")
    y       = mp.mpf("0.345580")
    r       = mp.mpf("2.00")
    
    N       = 1000
    R       = 2.0
    
    mode    = "jit"

    zoom    = 0.8
    n_zoom  = 50
    fpz     = 10
    overlap = 3
    
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
    
    row_zoom = int(round(row/zoom))
    col_zoom = int(round(col/zoom))
    
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    video  = cv2.VideoWriter("MandelbrotSet_video.mp4", fourcc, fps, (col, row))
    
    imgs = []
    fw = open("MandelbrotSet_video.csv","w")
    fw.write("ID,row,col,x,y,r,N,R,minL,maxL\n")
    for i in range(n_zoom + overlap - 1):
        if i < n_zoom:
            if len(imgs) > overlap:
                MargeAndPop(row, col, row_zoom, col_zoom, imgs)
                
            r_zoom = r*zoom**(len(imgs))
            fname = "MandelbrotSet_%04d.dat" % i
            img, nmg = mb.CalMandelbrotSet(
                row_zoom, col_zoom, x, y, r_zoom, N, R, colorlist, mode=mode,
                griddim = griddim, blockdim = blockdim, sub_dvi = sub_dvi,
                fname=fname)

            fw.write("%03d,%d,%d,%s,%s,%13.8e,%d,%f,%d,%d\n" % 
                     (i, row_zoom, col_zoom, x, y, r_zoom, N, R,
                      np.min(nmg), np.max(nmg)))
            print("i=%d r=%e n min=%5d n max=%5d" % 
                  (i, r_zoom, np.min(nmg), np.max(nmg)))
            imgs.append([r_zoom, img.copy()])
        else:
            MargeAndPop(row, col, row_zoom, col_zoom, imgs)
        
        if i < overlap:
            continue
        
        ZoomIn(i <= overlap, fpz, r, zoom, imgs, row, col, row_zoom, col_zoom, video)
        r *= zoom
        
    fw.close()
    video.release()

if __name__ == "__main__":
    main()