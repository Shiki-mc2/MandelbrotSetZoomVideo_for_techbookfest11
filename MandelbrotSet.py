# -*- coding: utf-8 -*-

import os
import time
import csv
import subprocess as cmd
import pickle

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numba import jit, cuda
from numba import uint8, uint16, uint32, float64
from mpmath import mp

def CalMandelbrotSet_normal(row, col, x, y, r, N, R, colorlist):
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

@jit((uint16, uint16, float64, float64, float64, uint32, float64, uint8[:,:]))
def CalMandelbrotSet_jit(row, col, x, y, r, N, R, colorlist):
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

def CalMandelbrotSet_mpf(row, col, x, y, r, N, R, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    nmg = np.zeros((row,col)  ,dtype=np.int32)
    
    xmin = x - r * (col-1)/(row-1)
    ymax = y + r
    dpp  = 2*r/(row-1)
    
    fname = "mpf_%d_%04d.csv" % (int(time.time()), np.random.randint(10000))

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

@jit
def CalMandelbrotSet_dat(row, col, N, nmg, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    
    for i in range(row):
        for j in range(col):
            if nmg[i,j] < N:
                img[i,j,:] = colorlist[nmg[i,j]]
    return img

def CalMandelbrotSet(row, col, x, y, r, N, R, colorlist, mode="normal",
                     griddim = (8,8), blockdim = (16,16), sub_dvi = 4, fname=""):
    if os.path.isfile(fname):
        with open(fname,"rb") as frb:
            print("load : %s" % fname)
            [row_, col_, x_, y_, r_, N_, R_, nmg] = pickle.load(frb)
            if(row != row_ or col != col_ or x != x_ or y != y_ or r != r or 
               N != N_ or R != R_):
                print("Warning, Args data and load data is mismatch")
                print("Args : row=%d col=%d x=%f y=%f r=%f N=%d R=%f" %
                      (row,  col,  x,  y,  r,  N,  R ))
                print("load : row=%d col=%d x=%f y=%f r=%f N=%d R=%f" %
                      (row_, col_, x_, y_, r_, N_, R_))
        img = CalMandelbrotSet_dat(row, col, N, nmg, colorlist)
        return img, nmg
    
    if mode != "mpf":
        x = float(x)
        y = float(y)
        r = float(r)
        
    if mode == "normal":
        return CalMandelbrotSet_normal(row, col, x, y, r, N, R, colorlist)
    elif mode == "jit":
        return CalMandelbrotSet_jit(row, col, x, y, r, N, R, colorlist)
    elif mode == "gpu":
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
                
                d_img       = cuda.to_device(img_sub)
                d_nmg       = cuda.to_device(nmg_sub)
                
                CalMandelbrotSet_CUDA[griddim, blockdim](
                    row_sta, row_end, col_sta, col_end, xmin, ymax, dpp, N, R2,
                    d_img, d_nmg, colorlist)
                
                d_img.copy_to_host(img_sub)
                d_nmg.copy_to_host(nmg_sub)
        
                img[row_sta:row_end, col_sta:col_end,:] = img_sub[:,:,:]
                nmg[row_sta:row_end, col_sta:col_end]   = nmg_sub[:,:]
    
        return img, nmg
    elif mode == "mpf":
        return CalMandelbrotSet_mpf(row, col, x, y, r, N, R, colorlist)

def GenerateGradation(color_set, base_cycle, inc, n, restart_idx):
    colorlist = np.zeros((n,3))
    offset = 0
    j = 0
    
    grad = []
    n_grad = len(color_set) - 1
    for i in range(n_grad):
        grad.append(LinearSegmentedColormap.from_list(
            'custom_cmap',[(0.0,color_set[i]),(1.0,color_set[i+1])] ))
    
    for i in range(n):
        if i - offset >= base_cycle:
            offset += base_cycle
            base_cycle = base_cycle*inc
            j =(j + 1) % ( n_grad )
            if j == 0:
                j = restart_idx
        colorlist[i,:] = np.array(grad[j]((i - offset)/base_cycle)[:3])*255
    colorlist = np.array(colorlist,dtype=np.uint8)
    return colorlist