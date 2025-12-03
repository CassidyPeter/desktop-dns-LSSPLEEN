# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np
import matplotlib.pyplot as plt


def plot_flo_surface(prop,geom,plot_var,caxis,side):

    if side=='ss':
        blocks = [2,4,6]
    elif side == 'ps':
        blocks = [3]
    
    plt.figure(1)
    plt.axis('equal')


    # for ib in range(len(geom)):
    for ib in blocks:
        x=geom[ib]['x'][:, -1, :]
        y=geom[ib]['y'][:, -1, :]
        f = prop[ib][plot_var][:, -1, :]
        h1 = plt.pcolormesh(x, y, f, shading='gouraud')

        h1.set_clim(caxis)
        
        # plt.clim(caxis)
    plt.colorbar()  
    plt.title(plot_var)        

    return plt




