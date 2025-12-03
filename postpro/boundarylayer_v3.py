# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge
# Modified extensively by Peter Cassidy
# Revised by Gemini to use delta_99 and local profile-based Ue

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from meshing.read_case import *
from .read_flo import *
from .grad import *
from .read_2d_mean import *
from scipy import spatial

def boundarylayer_v3(casename,*args):
    
    flo = {}
        
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)
    
    # get flow and geom
    if(len(args) > 0):
        nfiles = args[0]
        #print(nfiles)
        prop,blk,_ = read_2d_mean(casename,args[0])
    else:
        prop,blk = read_flo(casename)
        
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
    
    
    # patch info
    next_block = case['next_block']
    next_patch = case['next_patch']

    Nb = len(blk) 
    
    # initialize arrays
    IPS   = np.asarray([],dtype=float)
    ISS   = np.asarray([],dtype=float)
    volbl = np.asarray([],dtype=float)
    x0    = np.asarray([],dtype=float)
    y0    = np.asarray([],dtype=float)
    d0    = np.asarray([],dtype=float)
    d1    = np.asarray([],dtype=float)
    d2    = np.asarray([],dtype=float)
    d3    = np.asarray([],dtype=float)
    d1i   = np.asarray([],dtype=float)
    d2i   = np.asarray([],dtype=float)
    d3i   = np.asarray([],dtype=float)
    d4    = np.asarray([],dtype=float)
    d5    = np.asarray([],dtype=float)
    d6    = np.asarray([],dtype=float)
    d7    = np.asarray([],dtype=float)
    d8    = np.asarray([],dtype=float)
    d9    = np.asarray([],dtype=float)
    d10   = np.asarray([],dtype=float)
    d11   = np.asarray([],dtype=float)
    d12   = np.asarray([],dtype=float)
    yplus = np.asarray([],dtype=float)
    xplus = np.asarray([],dtype=float)
    zplus = np.asarray([],dtype=float)
    tplus = np.asarray([],dtype=float)
    tauw  = np.asarray([],dtype=float)
    Retheta=np.asarray([],dtype=float)
    Retau = np.asarray([],dtype=float)
    Res   = np.asarray([],dtype=float)
    cf    = np.asarray([],dtype=float)
    XX    = np.asarray([],dtype=float)
    YY    = np.asarray([],dtype=float)
    SS    = np.asarray([],dtype=float)
    Ue    = np.asarray([],dtype=float) # BL edge velocity
    Pe    = np.asarray([],dtype=float) # BL edge pressure
    beta  = np.asarray([],dtype=float) # Clauser parameter
    K     = np.asarray([],dtype=float) # pressure gradient parameter
    lamb  = np.asarray([],dtype=float) # Thwaite's criterion
    uplus_profile = []
    yplus_profile = []
    r_profile = []
    vel_profile = []
    BL_bool = {} # for loss breakdown masks
    
    
        
    ns = -1
    # Loop through just blocks containing BL and in specific order, SS 2,4,6 -> PS 3
    # for nb in range(Nb):
    for nb in [2,4,6,3]:

        BL_bool[nb] = []
        
        x = blk[nb]['x']
        y = blk[nb]['y']
        
        ro = prop[nb]['ro']
        mu = prop[nb]['mu']
        u  = prop[nb]['u']
        v  = prop[nb]['v']
        w  = prop[nb]['w']
        p  = prop[nb]['p']
        po = prop[nb]['po']
        To = prop[nb]['To']
        
        fspan = os.path.join(path,casename,'span_'+str(nb+1)+'.txt')
        span = np.loadtxt(fspan)
    
        try:
            mut = prop[nb]['mut_model']
        except:
            mut = np.zeros(np.shape(mu))
            
        try:
            delz = span[1,0]-span[0,0]
        except:
            delz = 0.0
        
        try:
            pr = prop[nb]['turb_production']
        except:
            pr = np.zeros(np.shape(mu))

        try:
            diss_strain = prop[nb]['dissipation_strain']
        except:
            diss_strain = np.zeros(np.shape(mu))
            
        if(len(np.shape(ro))==3):
            ro = np.mean(ro, axis=2)
            mu = np.mean(mu, axis=2)
            u  = np.mean(u,  axis=2)
            v  = np.mean(v,  axis=2)
            w  = np.mean(w,  axis=2)
            p  = np.mean(p,  axis=2)
            po = np.mean(po, axis=2)
            To = np.mean(To, axis=2)
            pr = np.mean(pr, axis=2)
            diss_strain = np.mean(diss_strain, axis=2)
            mut= np.mean(mut,axis=2)
            
            
        # All this wall code basically makes wall True for the blocks around the blade [2,3,4,6]
        im_wall = (next_block[nb]['im'] == 0 and next_patch[nb]['im'] >= 3)
        ip_wall = (next_block[nb]['ip'] == 0 and next_patch[nb]['ip'] >= 3)
        jm_wall = (next_block[nb]['jm'] == 0 and next_patch[nb]['jm'] >= 3)
        jp_wall = (next_block[nb]['jp'] == 0 and next_patch[nb]['jp'] >= 3)
        
        wall = im_wall or ip_wall or jm_wall or jp_wall

        
        if(wall):
        
            # transpose matrices so that wall is always on jm boundary
            if(im_wall):
                x  = x.T
                y  = y.T
                ro = ro.T
                mu = mu.T
                u  = u.T
                v  = v.T
                w  = w.T
                p  = p.T
                po = po.T
                To = To.T
                pr = pr.T
                diss_strain = diss_strain.T
                mut= mut.T
            
            if(ip_wall):
                x  = x[::-1,:]
                y  = y[::-1,:]
                ro = ro[::-1,:]
                mu = mu[::-1,:]
                u  = u[::-1,:]
                v  = v[::-1,:]
                w  = w[::-1,:]
                p  = p[::-1,:]
                po = po[::-1,:]
                To = To[::-1,:]
                pr = pr[::-1,:]  
                diss_strain = diss_strain[::-1,:]
                mut= mut[::-1,:]
                
                x  = x.T
                y  = y.T
                ro = ro.T
                mu = mu.T
                u  = u.T
                v  = v.T
                w  = w.T
                p  = p.T
                po = po.T
                To = To.T
                pr = pr.T  
                diss_strain = diss_strain.T
                mut= mut.T

            if(jp_wall): # basically all blocks are of jp_wall type
                x  = x[:,::-1]
                y  = y[:,::-1]
                ro = ro[:,::-1]
                mu = mu[:,::-1]
                u  = u[:,::-1]
                v  = v[:,::-1]
                w  = w[:,::-1]
                p  = p[:,::-1]
                po = po[:,::-1]
                To = To[:,::-1]
                pr = pr[:,::-1]  
                diss_strain = diss_strain[:,::-1]
                mut= mut[:,::-1]

                if nb==6:
                    x  = x[::-1,:]
                    y  = y[::-1,:]
                    ro = ro[::-1,:]
                    mu = mu[::-1,:]
                    u  = u[::-1,:]
                    v  = v[::-1,:]
                    w  = w[::-1,:]
                    p  = p[::-1,:]
                    po = po[::-1,:]
                    To = To[::-1,:]
                    pr = pr[::-1,:]  
                    diss_strain = diss_strain[::-1,:]
                    mut= mut[::-1,:]
                
            
            ni,nj=np.shape(x)
            
            # --- START OF REVISION ---
            # REMOVED isentropic uinf calculation.
            # 'uinf' and 'ruinf' will now be calculated INSIDE the i-loop
            # based on the maximum tangential velocity of each profile.
            # --- END OF REVISION ---
            
            edgej = np.zeros(ni)
            
            if nb==3:
                ns = -1
            for i in range(ni):
                ns = ns + 1

                # Compute local tangents
                if(i==0):
                    dxt = (x[i+1,0]-x[i,0])    
                    dyt = (y[i+1,0]-y[i,0])    
                elif(i==ni-1):
                    dxt = (x[i,0]-x[i-1,0])    
                    dyt = (y[i,0]-y[i-1,0])    
                else:
                    dxt = (x[i+1,0]-x[i-1,0])*0.5    
                    dyt = (y[i+1,0]-y[i-1,0])*0.5      
                
                # Removed the old block for guessing b.layer height ('Del')
                
                print('nb:',nb,'i:',i,'ns:',ns)
                
                ydist = y[i,3]-y[i,0]
                xdist = x[i,3]-x[i,0]
                

                IPS=np.append(IPS,(ydist < 0))
                ISS=np.append(ISS,(ydist > 0))
                        
                
                yprof = y[i,:]
                xprof = x[i,:]
                uprof = u[i,:]
                vprof = v[i,:]
                wprof = w[i,:]
                roprof = ro[i,:]
                turbprof = pr[i,:]/ro[i,:]
                dissprof = diss_strain[i,:]/ro[i,:]
                muprof = mu[i,:]
                mutprof = mut[i,:]
                pprof = p[i,:]
                
                nprof = 10000
                
                xd = xprof-xprof[0] # displacement from wall for each profile point
                yd = yprof-yprof[0] # displacement from wall for each profile point

                # wall normal coordinate using r projected onto unit normal rprof(j)=abs(r dot nhat)
                rprof = np.abs(-xd*dyt + yd*dxt)/np.sqrt(dxt*dxt + dyt*dyt)
                # Streamwise tangential velocity using u,v projected into unit tangent velprof(j)=u dot that
                velprof =(uprof*dxt + vprof*dyt)/np.sqrt(dxt*dxt + dyt*dyt)
                
                # --- START OF REVISION ---
                # Define local edge properties based on MAX TANGENTIAL VELOCITY
                # This replaces the problematic isentropic calculation
                
                # Handle separated flow (velprof can be negative)
                if np.max(velprof) > 0.0:
                    uinf_local_idx = np.argmax(velprof)
                    uinf_local   = velprof[uinf_local_idx]
                    roinf_local  = roprof[uinf_local_idx]
                    ruinf_local  = roinf_local * uinf_local
                else:
                    # Profile is fully separated or zero
                    uinf_local   = 0.001
                    roinf_local  = roprof[0]
                    ruinf_local  = roinf_local * uinf_local
                
                # Put limit where uinf is very small
                if uinf_local <= 0.001: uinf_local = 0.001
                
                # --- END OF REVISION ---
                
                dvel = velprof[2]  
                dyy  = rprof[2]   
            
                muw = muprof[0]
                row = roprof[0]
                tw = muw*dvel/dyy #muw*velprof[2]/rprof[2]   
                ut = np.sqrt(np.abs(tw)/row)
                yp = ut*row*rprof[1]/muw
                xp = yp*np.sqrt(dxt*dxt + dyt*dyt)/rprof[1]
                zp = yp*delz/rprof[1]
                tp = row*ut**2 /muw

                yp_profile = ut*row*rprof[:]/muw
                up_profile = velprof[:]/ut
                
                
                yplus=np.append(yplus,yp)
                xplus=np.append(xplus,xp)
                zplus=np.append(zplus,zp)
                tplus=np.append(tplus,tp)

                yplus_profile.append(yp_profile)
                uplus_profile.append(up_profile)
                r_profile.append(rprof)
                vel_profile.append(velprof)
                
                tauw=np.append(tauw,tw)
                # cf=np.append(cf,tw/(0.5*ruinf[i]*uinf[i])) # old method
                cf=np.append(cf,tw/(0.5*roinf_local*uinf_local**2)) # proper method
                # cf=np.append(cf,tw/(0.5*1.2048*10.33**2)) # to try match Argo and MISES Cf (uses constants)
                XX=np.append(XX,xprof[0])
                YY=np.append(YY,yprof[0])

                # Surface length s (confusingly SS is actually surface length for both ss and ps, but I kept in the same style as XX,YY)
                if(ns>0):
                    SS=np.append(SS, np.sqrt((XX[-1]-XX[-2])**2 + (YY[-1]-YY[-2])**2)+SS[-1])
                else:
                    SS=np.append(SS,0)
                
                # Ensure unique points for interpolation
                rprof_unique, unique_idx = np.unique(rprof, return_index=True)
                
                # If fewer than 4 unique points, cubic interpolation fails. Use linear.
                interp_kind = 'cubic' if len(rprof_unique) >= 4 else 'linear'

                ri = np.linspace(np.min(rprof_unique),np.max(rprof_unique),nprof)
                
                f = interp1d(rprof_unique, yprof[unique_idx], kind=interp_kind)
                yi =  f(ri)
                
                f = interp1d(rprof_unique, xprof[unique_idx], kind=interp_kind)
                xi =  f(ri)
                
                f = interp1d(rprof_unique, uprof[unique_idx], kind=interp_kind)
                ui =  f(ri)
    
                f = interp1d(rprof_unique, vprof[unique_idx], kind=interp_kind)
                vi =  f(ri)
    
                f = interp1d(rprof_unique, wprof[unique_idx], kind=interp_kind)
                wi =  f(ri)
                
                f = interp1d(rprof_unique, muprof[unique_idx], kind=interp_kind)
                mui =  f(ri)
    
                f = interp1d(rprof_unique, mutprof[unique_idx], kind=interp_kind)
                muti = f(ri)
    
                f = interp1d(rprof_unique, roprof[unique_idx], kind=interp_kind)
                roi =  f(ri)
    
                f = interp1d(rprof_unique, turbprof[unique_idx], kind=interp_kind)
                turbi =f(ri)

                f = interp1d(rprof_unique, dissprof[unique_idx], kind=interp_kind)
                dissi =f(ri)

                f = interp1d(rprof_unique, pprof[unique_idx], kind=interp_kind)
                pi =f(ri)
                
                # --- START OF REVISION ---
                # Interpolate the TANGENTIAL velocity profile
                f = interp1d(rprof_unique, velprof[unique_idx], kind=interp_kind)
                velprofi = f(ri)

                # This is the total velocity magnitude, used for dissipation calcs
                vmagi = np.sqrt(ui*ui + vi*vi + wi*wi)

                # REVISED BOUNDARY LAYER EDGE DEFINITION
                # Use the interpolated TANGENTIAL velocity (velprofi)
                # and the local TANGENTIAL Ue (uinf_local)
                ii = velprofi < (0.99 * uinf_local)
                
                # Handle edge cases
                if not np.any(ii):
                    # All points are >= 0.99*Ue. Set BL thickness to the first grid point.
                    ii = np.zeros_like(ri, dtype=bool)
                    ii[0] = True 
                    # print(f"Warning: Profile {ns} (nb={nb}, i={i}) starts at or above 0.99*Ue.")
                elif np.all(ii):
                    # All points are < 0.99*Ue. Use the whole profile.
                    # This *shouldn't* happen now, as uinf_local is max(velprof)
                    # print(f"Warning: Profile {ns} (nb={nb}, i={i}) never reaches 0.99*Ue.")
                    pass
                # --- END OF REVISION ---
                
                edgen = np.argmax(ri*ii)
                edgej[i] = np.argmin(abs(rprof-ri[edgen]))
                
                    
                dx = xi[1:]-xi[:-1]
                dy = yi[1:]-yi[:-1]

                # Wall normal spacing  
                dely = np.abs(-dx*dyt + dy*dxt)/np.sqrt(dxt*dxt + dyt*dyt) 
                if(edgen > (len(dely)-1)):  
                    edgen = len(dely)-1          
                del_now = np.sum(dely[:edgen])


                # recompute bl edge estimate
                if(del_now > np.sum(dely)):
                    edgen = np.argmax(ri*ii)
                    edgej[i] = np.argmin(abs(rprof-ri[edgen]))

                
                # This 'delv' uses total magnitude, which is correct for dissipation
                delv = vmagi[1:]-vmagi[:-1]
                    
                uav = (ui[1:]+ui[:-1])*0.5
                vav = (vi[1:]+vi[:-1])*0.5
                wav = (wi[1:]+wi[:-1])*0.5
                                        
                turbav =(turbi[1:]+turbi[:-1])*0.5
                dissav =(dissi[1:]+dissi[:-1])*0.5 # dissipation due to mean strain from 2d mean prop, not calced here
                ronow  =  (roi[1:] + roi[:-1])*0.5
                munow  =  (mui[1:] + mui[:-1])*0.5
                mutnow = (muti[1:]+ muti[:-1])*0.5

                
                # This 'vnow' is tangential velocity, which is correct for integral params
                vnow = (uav*dxt + vav*dyt)/np.sqrt(dxt*dxt + dyt*dyt) 
                if ns==0:
                    print('uav', uav, 'dxt', dxt, 'vav', vav, 'dyt', dyt)
                    
                diss_av = (munow/ronow)*(delv/dely)*(delv/dely) # 
                diss_model = (munow*mutnow/ronow)*(delv/dely)*(delv/dely) # modelled dissipation using turbulent viscosity mu_t

                # --- START OF REVISION ---
                # Cap vnow at the local tangential freestream velocity
                vnow[vnow > uinf_local] = uinf_local 
                # --- END OF REVISION ---

                x0=np.append(x0,xi[edgen])
                y0=np.append(y0,yi[edgen])
                
                # --- START OF REVISION ---
                # Store the locally defined Ue and Pe
                Ue=np.append(Ue, uinf_local)
                Pe=np.append(Pe, pi[edgen])
                # --- END OF REVISION ---

                # BL masks for loss breakdown (BL edge is found with interped high resolution y profile, so finding closest block coord)
                search = np.array([x[i,:], y[i,:]])
                dists = (search[0] - xi[edgen])**2 + (search[1] - yi[edgen])**2
                closest_idx = np.argmin(dists)
                mask = np.arange(search.shape[1]) <= closest_idx
                BL_bool[nb].append(mask[::-1])

                
                # --- START OF REVISION ---
                # Normalize integral parameters with local, tangential Ue/Roe
                vn    = vnow/uinf_local
                rvn   = ronow*vnow/ruinf_local
                rvdef = 1.0 - rvn
                vdef  = 1.0 - vn 
                v2def = 1.0 - vn*vn
                # --- END OF REVISION ---

                # Data all normalised to axial chord
                d0=np.append(d0,np.sum(dely[:edgen])) # delta99
                d1=np.append(d1,np.sum(rvdef[:edgen]*dely[:edgen])) # # Displacement thickness
                d2=np.append(d2,np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen])) # Momentum thickness
                d3=np.append(d3,np.sum(rvn[:edgen]*v2def[:edgen]*dely[:edgen])) # Energy thickness

                
                d1i=np.append(d1i,np.sum(vdef[:edgen]*dely[:edgen])) # ? inviscid?
                d2i=np.append(d2i,np.sum(vn[:edgen]*vdef[:edgen]*dely[:edgen])) # ? inviscid?
                d3i=np.append(d3i,np.sum(vn[:edgen]*v2def[:edgen]*dely[:edgen])) # ? inviscid?
                
                # --- START OF REVISION ---
                # Normalize dissipation with local, tangential Ue
                d9=np.append(d9,    np.sum(diss_av[:edgen]/(uinf_local**3) *dely[:edgen])) # Dissipation due to mean strain - normalised to 1/Ue3^3
                d10=np.append(d10, np.sum(turbav[:edgen]/(uinf_local**3) *dely[:edgen])) # Turbulence production - normalised to 1/Ue3^3
                # --- END OF REVISION ---
                
                d11=np.append(d11, np.sum((turbav[:edgen] + diss_av[:edgen])*dely[:edgen])) # Dissipation coefficient cd using actual dissipation
                d12=np.append(d12, np.sum((turbav[:edgen] + diss_model[:edgen])*dely[:edgen])) # Dissipation coefficient cd using turbulent viscosity modelled dissipation?
                
                # --- START OF REVISION ---
                # Calculate Reynolds numbers with local edge properties
                # Note: ronow[edgen] and munow[edgen] may be out of bounds if edgen is at the end
                # Use roinf_local and mu_at_edge instead for robustness
                mu_at_edge = mui[edgen] if edgen < len(mui) else mui[-1]
                
                Retheta=np.append(Retheta, uinf_local * np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen]) * roinf_local/mu_at_edge)
                Retau=np.append(Retau, ut * np.sum(dely[:edgen]) * roinf_local/mu_at_edge)
                Res=np.append(Res, uinf_local * SS[ns] * roinf_local/mu_at_edge)

                beta=np.append(beta, np.sum(rvdef[:edgen]*dely[:edgen]) / tw) # Clauser parameter
                K=np.append(K, mu_at_edge/(roinf_local*uinf_local**2)) # Pressure gradient parameter (acceleration parameter)
                lamb=np.append(lamb, np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen])**2 * mu_at_edge/roinf_local) # Thwaites criterion
                # --- END OF REVISION ---
                
        BL_bool[nb] = np.stack(BL_bool[nb]) # stack to avoid a tuple of arrays
        if nb==6:
            BL_bool[nb] = BL_bool[nb][::-1,:] # flip for last SS block
                
                
    xLE = min(XX)
    xTE = max(XX)
    cax = xTE-xLE
    #
    xn = (XX-xLE)/cax
    
    bl = {}
    bl['ss'] = {}
    bl['ps'] = {}

    
    ISS = ISS > 0
    IPS = IPS > 0
    

    # ISS = (ISS>0) & (xn>0.05) & (xn<0.95)
    # IPS = (IPS>0) & (xn>0.05) & (xn<0.95)

    # Trim LE and TE where BL calcs become weird - use percentage of surface length
    ISS = (ISS > 0) & (SS < 0.996 * SS[ISS].max()) & (SS > 0.02 * SS[ISS].max())
    IPS = (IPS > 0) & (SS < 0.835 * SS[IPS].max()) & (SS > 0.02 * SS[IPS].max())


    bl['Mask'] = BL_bool # Mask for loss breakdown
    
    # --- START OF REVISION ---
    # Need to handle potential divide-by-zero if d2 is 0
    H = np.zeros_like(d1)
    valid_H = (d2 != 0)
    H[valid_H] = d1[valid_H] / d2[valid_H] # Shape factor
    # --- END OF REVISION ---
        
    yplus_profile = np.array(yplus_profile)
    uplus_profile = np.array(uplus_profile)
    r_profile = np.array(r_profile)
    vel_profile = np.array(vel_profile)

    # Gradients and final calcs for Clauser / acceleration param / Thwaites
    # Need to check for empty arrays from trimming before gradient
    if np.any(ISS):
        beta[ISS] = beta[ISS] * np.gradient(Pe[ISS], SS[ISS])
        K[ISS] = K[ISS] * np.gradient(Ue[ISS], SS[ISS])
        lamb[ISS] = lamb[ISS] * np.gradient(Ue[ISS], SS[ISS])
    if np.any(IPS):
        beta[IPS] = beta[IPS] * np.gradient(Pe[IPS], SS[IPS])
        K[IPS] = K[IPS] * np.gradient(Ue[IPS], SS[IPS])
        lamb[IPS] = lamb[IPS] * np.gradient(Ue[IPS], SS[IPS])
    
    bl['ss']['x'   ] = XX[ISS]
    bl['ss']['y'   ] = YY[ISS]
    bl['ss']['s'   ] = SS[ISS]
    bl['ss']['d0'  ] = d0[ISS]
    bl['ss']['d1'  ] = d1[ISS]
    bl['ss']['d2'  ] = d2[ISS]
    bl['ss']['d3'  ] = d3[ISS]
    bl['ss']['d9'  ] = d9[ISS]
    bl['ss']['d10'  ] = d10[ISS]
    bl['ss']['d11'  ] = d11[ISS]
    bl['ss']['d12'  ] = d12[ISS]
    bl['ss']['Retheta'  ] = Retheta[ISS]
    bl['ss']['Retau'  ] = Retau[ISS]
    bl['ss']['Res'  ] = Res[ISS]
    bl['ss']['H'   ] = H[ISS]
    bl['ss']['cf'  ] = cf[ISS]
    bl['ss']['yplus'  ] = yplus[ISS]
    bl['ss']['xplus'  ] = xplus[ISS]
    bl['ss']['zplus'  ] = zplus[ISS]
    bl['ss']['tplus'  ] = tplus[ISS]
    bl['ss']['x0'  ] = x0[ISS] # coords for edge of BL
    bl['ss']['y0'  ] = y0[ISS] # coords for edge of BL
    bl['ss']['beta'  ] = beta[ISS]
    bl['ss']['K'  ] = K[ISS]
    bl['ss']['lamb'  ] = lamb[ISS]
    
    bl['ss']['yplus_profile'  ] = yplus_profile[ISS,:]
    bl['ss']['uplus_profile'  ] = uplus_profile[ISS,:]
    bl['ss']['r_profile'  ] = r_profile[ISS,:]
    bl['ss']['vel_profile'  ] = vel_profile[ISS,:]
    
    bl['ps']['x'   ] = XX[IPS]
    bl['ps']['y'   ] = YY[IPS]
    bl['ps']['s'   ] = SS[IPS]
    bl['ps']['d0'  ] = d0[IPS]
    bl['ps']['d1'  ] = d1[IPS]
    bl['ps']['d2'  ] = d2[IPS]
    bl['ps']['d3'  ] = d3[IPS]
    bl['ps']['d9'  ] = d9[IPS]
    bl['ps']['d10'  ] = d10[IPS]
    bl['ps']['d11'  ] = d11[IPS]
    bl['ps']['d12'  ] = d12[IPS]
    bl['ps']['Retheta'  ] = Retheta[IPS]
    bl['ps']['Retau'  ] = Retau[IPS]
    bl['ps']['Res'  ] = Res[IPS]
    bl['ps']['H'   ] = H[IPS]
    bl['ps']['cf'  ] = cf[IPS]
    bl['ps']['yplus'  ] = yplus[IPS]
    bl['ps']['xplus'  ] = xplus[IPS]
    bl['ps']['zplus'  ] = zplus[IPS]
    bl['ps']['tplus'  ] = tplus[IPS]
    bl['ps']['x0'  ] = x0[IPS]
    bl['ps']['y0'  ] = y0[IPS]
    bl['ps']['beta'  ] = beta[IPS]
    bl['ps']['K'  ] = K[IPS]
    bl['ps']['lamb'  ] = lamb[IPS]

    bl['ps']['yplus_profile'  ] = yplus_profile[IPS,:]
    bl['ps']['uplus_profile'  ] = uplus_profile[IPS,:]
    bl['ps']['r_profile'  ] = r_profile[IPS,:]
    bl['ps']['vel_profile'  ] = vel_profile[IPS,:]
    
    return bl