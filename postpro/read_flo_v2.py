# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge
# Updated by PC in August 2025 to add Q-Criterion and Lambda-2 and enable 3D data to be read in

import os
import numpy as np
from meshing.read_case import *
from .grad import *
from .grad3d import *
from .area import *

def read_flo_v2(casename):
    
    
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

    # spanwise resolution
    nk = case['solver']['nk']
    if nk > 1:
        Lz = case['solver']['span']
        dz = Lz / (nk-1) if nk > 1 else 1.0

    
    for ib in range(len(blk)):
        
        x = blk[ib]['x']
        y = blk[ib]['y']
        flo[ib] = {}
          
        ni,nj = np.shape(blk[ib]['x'])

        if(nk>1):
           ro = np.zeros([ni,nj,nk])
           ru = np.zeros([ni,nj,nk])
           rv = np.zeros([ni,nj,nk])
           rw = np.zeros([ni,nj,nk])
           Et = np.zeros([ni,nj,nk])
        else:
           ro = np.zeros([ni,nj])
           ru = np.zeros([ni,nj])
           rv = np.zeros([ni,nj])
           rw = np.zeros([ni,nj])
           Et = np.zeros([ni,nj])
                    
        if version == 'gpu':
        
            flow_name = 'flow_' + str(ib+1)
            rans_name = 'rans_' + str(ib+1)
            
            flow_file = os.path.join(path,casename,flow_name)
            rans_file = os.path.join(path,casename,rans_name)
            
            f = open(flow_file,'rb')
            q   = np.fromfile(f,dtype='float64',count=ni*nj*nk*5)
            f.close()

            if(os.path.exists(rans_file)): 
                f = open(rans_file,'rb')
                v   = np.fromfile(f,dtype='float64',count=ni*nj)
                f.close()
                mut_model = np.reshape(v,[ni,nj,1],'F')
            else:
                mut_model = np.zeros([ni,nj,1])
                        
            q = np.reshape(q,[5,ni,nj,nk],order='F') # make sure to reshape with fortran rule!
            ro = q[0,:,:,:]
            ru = q[1,:,:,:]
            rv = q[2,:,:,:]
            rw = q[3,:,:,:]
            Et = q[4,:,:,:]
            
        
        # derived quantities
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
        
        
        # dudx = np.zeros([ni,nj,nk])
        # dudy = np.zeros([ni,nj,nk])
        
        # dvdx = np.zeros([ni,nj,nk])
        # dvdy = np.zeros([ni,nj,nk])

        # velocity gradients
        dudx = np.zeros_like(ro)
        dudy = np.zeros_like(ro)
        dvdx = np.zeros_like(ro)
        dvdy = np.zeros_like(ro)
        dwdx = np.zeros_like(ro)
        dwdy = np.zeros_like(ro)

        # compute x,y derivatives slice-by-slice
        for k in range(nk):
            dudx[:,:,k], dudy[:,:,k] = grad(u[:,:,k], x, y)
            dvdx[:,:,k], dvdy[:,:,k] = grad(v[:,:,k], x, y)
            dwdx[:,:,k], dwdy[:,:,k] = grad(w[:,:,k], x, y)

        
        # z-derivatives by central difference in k
        dudz = np.gradient(u, dz, axis=2, edge_order=2) if nk > 1 else np.zeros_like(u)
        dvdz = np.gradient(v, dz, axis=2, edge_order=2) if nk > 1 else np.zeros_like(v)
        dwdz = np.gradient(w, dz, axis=2, edge_order=2) if nk > 1 else np.zeros_like(w)
        
        # Q-criterion and lambda2
        Q = np.zeros_like(ro)
        lambda2 = np.zeros_like(ro)



        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    gradU = np.array([
                        [dudx[i,j,k], dudy[i,j,k], dudz[i,j,k]],
                        [dvdx[i,j,k], dvdy[i,j,k], dvdz[i,j,k]],
                        [dwdx[i,j,k], dwdy[i,j,k], dwdz[i,j,k]]
                    ])
                    S = 0.5 * (gradU + gradU.T)
                    Omega = 0.5 * (gradU - gradU.T)
                    Q[i,j,k] = 0.5 * (np.sum(Omega**2) - np.sum(S**2))
                    S2_Om2 = S @ S + Omega @ Omega
                    eigs = np.linalg.eigvalsh(S2_Om2)
                    lambda2[i,j,k] = eigs[1]  # second largest eigenvalue




        
        
        vortz = dvdx - dudy
        a = area(x,y)

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

        flo[ib]['po'] = po
        flo[ib]['To'] = To
        flo[ib]['mach'] = mach
        flo[ib]['mu'] = mu
        flo[ib]['mut_model'] = mut_model
        flo[ib]['area'] = a

        flo[ib]['alpha'] = alpha
        flo[ib]['vel'] = vel

        flo[ib]['Q'] = Q
        flo[ib]['lambda2'] = lambda2


    return flo,blk
