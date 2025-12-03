# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np
import matplotlib.pyplot as plt
# import scienceplots


def plot_flo(prop,geom,plot_var,caxis):

    # with plt.style.context(['science', 'notebook']):
    plt.figure(1)
    plt.axis('equal')


    for ib in range(len(geom)):
    # for ib in [2,3,4,6,8]:
        x=geom[ib]['x']
        y=geom[ib]['y']
        y2=geom[ib]['y'] + 0.97
        y3=geom[ib]['y'] + 0.97*2
        f = prop[ib][plot_var]
        if(len(np.shape(f))==2):        
           # plt.pcolormesh(x,y,f,shading='gouraud')
           h1 = plt.pcolormesh(x, y, f, shading='gouraud')
           # h2 = plt.pcolormesh(x, y2, f, shading='gouraud')
           # h3 = plt.pcolormesh(x, y3, f, shading='gouraud')
        else:
           # plt.pcolormesh(x,y,f[:, :, 0],shading='gouraud')
           h1 = plt.pcolormesh(x, y, f[:, :, 0], shading='gouraud')
           # h2 = plt.pcolormesh(x, y2, f[:, :, 0], shading='gouraud')
           # h3 = plt.pcolormesh(x, y3, f[:, :, 0], shading='gouraud')

        h1.set_clim(caxis)
        plt.set_cmap('seismic')
        
        # h2.set_clim(caxis)
        # h3.set_clim(caxis)
        
        # plt.clim(caxis)
    cbar = plt.colorbar()  
    if plot_var=='vortz':
        cbar.ax.set_title(r'$\Omega_z \ \mathrm{[s^{-1}]}$', pad=10)
    plt.title(plot_var)     
    plt.xlabel('x/Cax')
    plt.ylabel('y/Cax')

    return plt

