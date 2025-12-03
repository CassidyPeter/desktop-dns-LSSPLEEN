# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np
import matplotlib.pyplot as plt


def plot_flo_LL(prop,geom,plot_var,caxis):

    
    plt.figure(1)
    plt.axis('equal')


    for ib in range(len(geom)):
        x=geom[ib]['x']
        y=geom[ib]['y']
        y2=geom[ib]['y'] + 0.65
        y3=geom[ib]['y'] + 0.65*2
        f = prop[ib][plot_var]
        if(len(np.shape(f))==2):        
           # plt.pcolormesh(x,y,f,shading='gouraud')
           h1 = plt.pcolormesh(x, y, f, shading='gouraud')
           h2 = plt.pcolormesh(x, y2, f, shading='gouraud')
           h3 = plt.pcolormesh(x, y3, f, shading='gouraud')
        else:
           # plt.pcolormesh(x,y,f[:, :, 0],shading='gouraud')
           h1 = plt.pcolormesh(x, y, f[:, :, 0], shading='gouraud')
           h2 = plt.pcolormesh(x, y2, f[:, :, 0], shading='gouraud')
           h3 = plt.pcolormesh(x, y3, f[:, :, 0], shading='gouraud')

        h1.set_clim(caxis)
        h2.set_clim(caxis)
        h3.set_clim(caxis)
        
        # plt.clim(caxis)
    plt.colorbar()  
    plt.title(plot_var)        

    return plt




