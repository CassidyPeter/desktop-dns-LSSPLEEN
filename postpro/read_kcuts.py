# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import timeit

from meshing.read_case import *
from .grad import *

def read_kcuts(casename,nfiles,plot_var,caxis,DPI):
    
    
    flo = {}
        
    
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)
    
    # unpack case
    blk = case['blk']
      
    # get solver version
    version = case['solver']['version']    

    # get gas props
    gam     = case['gas']['gamma']
    cp      = case['gas']['cp']
    mu_ref  = case['gas']['mu_ref']
    mu_tref = case['gas']['mu_tref']
    mu_cref = case['gas']['mu_cref']
    pr      = case['gas']['pr']
    cv = cp/gam
    rgas = cp-cv     
    
    
    # read kslice file
    kfile = os.path.join(path,casename,'kslice_time.txt')
    ktime = np.loadtxt(kfile,dtype={'names': ('ncut', 'time', 'k'),'formats': ('i', 'f', 'i')})
    
    nfmax = len(ktime['ncut'])
        
    if(len(nfiles)==1):
       if (nfiles[0] > nfmax) or (nfiles[0]==0):
           nfile = ktime['ncut']
           time =  ktime['time']
       elif nfiles[0] < 0 :
           nfnow = np.abs(nfiles)-1
           nfile = ktime['ncut'][nfnow]
           time =  ktime['time'][nfnow]
       else:    
           nfile = ktime['ncut'][-nfiles[0]:]
           time  = ktime['time'][-nfiles[0]:]
       
    else:
       nfile = ktime['ncut'][nfiles[0]-1:nfiles[1]]
       time =  ktime['time'][nfiles[0]-1:nfiles[1]]
    
    nfiles = len(time)
    nt = 0

    for ncut in nfile:  
        start_ncut = timeit.default_timer()
        nt = nt + 1
        for ib in range(len(blk)):
            
            x = blk[ib]['x']
            y = blk[ib]['y']
            flo[ib] = {}
              
            ni,nj = np.shape(blk[ib]['x'])
            nk = 1
      
            ro = np.zeros([ni,nj])
            ru = np.zeros([ni,nj])
            rv = np.zeros([ni,nj])
            rw = np.zeros([ni,nj])
            Et = np.zeros([ni,nj])
         
        
            if version == 'cpu':
                flow_name = 'kcu2_' + str(ib+1) + '_' + str(ncut)
                ind_name =  'nod2_' + str(ib+1) + '_' + str(ncut)
                
                flow_file = os.path.join(path,casename,flow_name)
                ind_file = os.path.join(path,casename,ind_name)
                
                f = open(flow_file,'rb')
                g = open(ind_file,'rb')
                
                q   = np.fromfile(f,dtype='float64',count=-1)
                ind = np.fromfile(g,dtype='uint32',count=-1)
                
                f.close()
                g.close
                
                N = np.int(len(q)/5)
                M = np.int(len(ind)/3)
                
                q = np.reshape(q,[5,N],order='F') # make sure to reshape with fortran rule!
                ind = np.reshape(ind,[3,M],order='F') # make sure to reshape with fortran rule!
                for n in range(M):
                    i = ind[0,n]-1
                    j = ind[1,n]-1
                    ro[i,j] = q[0,n]
                    ru[i,j] = q[1,n]
                    rv[i,j] = q[2,n]
                    rw[i,j] = q[3,n]
                    Et[i,j] = q[4,n]
                    
            elif version == 'gpu':
                flow_name = 'kcut_' + str(ib+1) + '_' + str(ncut)
                #print(flow_name)
                
                flow_file = os.path.join(path,casename,flow_name)
                
                f = open(flow_file,'rb')
                
                q   = np.fromfile(f,dtype='float64',count=ni*nj*5)
                
                f.close()
                            
                q = np.reshape(q,[5,ni,nj],order='F') # make sure to reshape with fortran rule!
                ro = q[0,:,:]
                ru = q[1,:,:]
                rv = q[2,:,:]
                rw = q[3,:,:]
                Et = q[4,:,:]
                
            
            # get derived quantities
            u = ru/ro
            v = rv/ro
            w = rw/ro
            p = (gam-1.0)*(Et - 0.5*(u*u + v*v + w*w)*ro)
            T = p/(ro*rgas)
            mu = (mu_ref)*( ( mu_tref + mu_cref )/( T + mu_cref ) )*((T/mu_tref)**1.5)  
            alpha = np.arctan2(v,u)*180.0/np.pi
            s = cp*np.log(T/300) - rgas*np.log(p/1e5)
            vel = np.sqrt(u*u + v*v + w*w)
            mach = vel/np.sqrt(gam*rgas*T)
            To = T*(1.0 + (gam-1)*0.5*mach*mach)
            po = p*((To/T)**(gam/(gam-1.0)))
            c = np.sqrt(gam*rgas*T)
            
            dudx,dudy = grad(u,x,y)
            dvdx,dvdy = grad(v,x,y)
            dpdx,dpdy = grad(p,x,y)
            drdx,drdy = grad(ro,x,y)
            
            vortz = dvdx - dudy
        
        
            # get 1-d wave amplitudes
            lam1 = u - c
            lam2 = u
            lam5 = u + c
            
            L_1 = lam1*(dpdx - ro*c*dudx)
            L_2 = lam2*(c*c*drdx - dpdx)
            L_5 = lam5*(dpdx + ro*c*dudx)
            
            flo[ib]['ro'] = ro
            flo[ib]['ru'] = ru
            flo[ib]['rv'] = rv
            flo[ib]['rw'] = rw
            flo[ib]['Et'] = Et
        
            flo[ib]['p'] = p
            flo[ib]['u'] = u
            flo[ib]['v'] = v
            flo[ib]['w'] = w
            flo[ib]['T'] = T
            flo[ib]['s'] = s
            flo[ib]['vortz'] = vortz
            
            flo[ib]['L_1'] = L_1
            flo[ib]['L_2'] = L_2
            flo[ib]['L_5'] = L_5
            
        with plt.style.context(['science', 'notebook']):
            # plot flow
            plt.figure(1)
            plt.axis('equal')
                
    
            plt.set_cmap('seismic')
            
            # for ib in range(len(blk)):
            for ib in [2,3,4,5,6,7,8]:
            # for ib in [4]:
                x=blk[ib]['x']
                y=blk[ib]['y']       
                # y2=blk[ib]['y'] + 0.97
                # y3=blk[ib]['y'] - 0.97
                h1 = plt.pcolormesh(x,y, flo[ib][plot_var],shading='gouraud')
                # h2 = plt.pcolormesh(x,y2, flo[ib][plot_var],shading='gouraud')
                # h3 = plt.pcolormesh(x,y3, flo[ib][plot_var],shading='gouraud')
                
                # plt.clim(caxis)
                h1.set_clim(caxis)
                # h2.set_clim(caxis)
                # h3.set_clim(caxis)
            
            plt.set_cmap('seismic')
            cbar = plt.colorbar()
            if plot_var=='vortz':
                cbar.ax.set_title(r'$\Omega_z \ \mathrm{[s^{-1}]}$', pad=10)
            plt.xlabel('x/Cax')
            plt.ylabel('y/Cax')
            plt.title(plot_var)
            
            plt.axis([0, 1.3, -1, 0.2]) # Blade closeup
            plot_file = os.path.join(path,casename,'Vortz_Blade', 'kcut_'+str(nt)+'_Blade.png')             
            plt.savefig(plot_file,dpi=DPI)
    
            
            # plt.axis([0.63, 0.8, -0.13, 0.02]) # SS BL CLOSEUP LIKE RIGHT ON THE LSB
            # plot_file = os.path.join(path,casename,'kcut_'+str(nt)+'_SSBLCLOSE.png')     
            plt.axis([0.65, 1, -0.6, 0]) # SS BL closeup
            plot_file = os.path.join(path,casename,'Vortz_SSBL', 'kcut_'+str(nt)+'_SSBL.png') 
            plt.savefig(plot_file,dpi=DPI)
            
            plt.close()
            stop_ncut = timeit.default_timer()
            print('Time for Kcut', ncut, ': ', round((stop_ncut - start_ncut)/60,2), 'mins')  

    
    return flo,blk
