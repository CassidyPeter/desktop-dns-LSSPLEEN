# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np

def read_grid(casename):

    Cax = 0.06301079
    grid = {} 
    blk = {}
    
    path = os.path.join(os.getcwd(),casename)
    blockdims = os.path.join(path,'blockdims.txt')
    bijk = np.loadtxt(blockdims,dtype=np.int32)
    #print(len(np.shape(bijk)))
    if( len(np.shape(bijk))==1 ):
        NB = 1
    else:
        NB,_ = np.shape(bijk)

    for ib in range(NB):
        blk[ib] = {}
        if(NB==1):
            ni,nj,nk = bijk[:]
        else:
            ni,nj,nk = bijk[ib,:]
        
        grid_file = 'grid_' + str(ib+1) + '.txt'        
        grid_file_path = os.path.join(path,grid_file)        
        
        grid = np.loadtxt(grid_file_path)
        # print(np.shape(grid))
        
        x=np.reshape(grid[:,0],[ni,nj],order='F')
        y=np.reshape(grid[:,1],[ni,nj],order='F')

        # z=np.reshape( ,[ni,nk],order='F'
        # print(np.shape(x))
        
        blk[ib]['x'] = x
        blk[ib]['y'] = y

        # blk[ib]['z'] = z


    ss_blade_blocks = [3,5,7]
    ps_blade_blocks = [4]
    # ss
    for i,ib in enumerate(ss_blade_blocks):
        if i==0:
            addition = 0
        if i>0:
            addition = blk[ss_blade_blocks[i-1]-1]['s'][-1]
        x_edge = blk[ib-1]['x'][:, -1]*Cax
        y_edge = blk[ib-1]['y'][:, -1]*Cax
        if ib==7: # Last block is flipped for some reason
            x_edge = x_edge[::-1]
            y_edge = y_edge[::-1]
        dx = np.gradient(x_edge)
        dy = np.gradient(y_edge)
        s = np.cumsum(np.sqrt(dx**2 + dy**2)) + addition # Add surface length to previous block end length
        blk[ib-1]['s'] = s  # Store cumulative surface length


    # ps - ps_blade_blocks = 4
    x_edge = blk[4-1]['x'][:, -1]*Cax
    y_edge = blk[4-1]['y'][:, -1]*Cax
    dx = np.gradient(x_edge)
    dy = np.gradient(y_edge)
    s = np.cumsum(np.sqrt(dx**2 + dy**2))
    
    blk[4-1]['s'] = s  # Store cumulative surface length

    return  blk
    
