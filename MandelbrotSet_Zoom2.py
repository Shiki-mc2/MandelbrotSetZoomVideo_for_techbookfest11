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
                    img  = cv2.resize(imgs[j][1][ridx1:ridx2,cidx1:cidx2],(col, row), 
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

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video.write(img)

def main():
    row     = 540
    col     = 720
    fps     = 30
    
    zoom    = 0.8
    n_zoom  = 20
    fpz     = 12
    overlap = 3
    
    x       = 0.45
    y       = 0.35
    r       = 2.00
    
    N       = 200
    R       = 2.0
    
    row_zoom = int(round(row/zoom))
    col_zoom = int(round(col/zoom))
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)
    
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    video  = cv2.VideoWriter("MandelbrotSet_Zoom2.mp4", fourcc, fps, (col, row))
    
    imgs = []
    for i in range(n_zoom + overlap - 1):
        if i < n_zoom:
            if len(imgs) > overlap:
                MargeAndPop(row, col, row_zoom, col_zoom, imgs)
                
            r_zoom = r*zoom**(len(imgs))
            img, nmg = CalMandelbrotSet(row_zoom, col_zoom, x, y, r_zoom, N, R,
                                        colorlist)
            print("%3d, min n = %5d, max n = %5d" % (i, np.min(nmg), np.max(nmg)))
            imgs.append([r_zoom, img.copy()])
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite("MandelbrotSet_Zoom2_%02d.png" % i, img)
        else:
            MargeAndPop(row, col, row_zoom, col_zoom, imgs)
        
        if i < overlap:
            continue
        
        ZoomIn(i <= overlap, fpz, r, zoom, imgs, row, col, row_zoom, col_zoom, video)
        r *= zoom

    video.release()

if __name__ == "__main__":
    main()
    
