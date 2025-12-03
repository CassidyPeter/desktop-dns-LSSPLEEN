# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge
# Modified extensively by Peter Cassidy

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from meshing.read_case import *
from .read_flo import *
from .grad import *
from .read_2d_mean import *
from scipy import spatial

def boundarylayer_v2(casename,*args):
    
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
            vel = np.sqrt(u*u + v*v + w*w)
        
            psurf  = p[:,0]
            Tosurf = To[:,0]
            pomax  = np.max(po,axis=1)
            umax   = np.max(vel,axis=1)    
            
            psurf[psurf>pomax]=pomax[psurf>pomax]
        
            minf = np.sqrt((((psurf/pomax)**(-(gam-1.0)/gam))-1.0)*2.0/(gam-1.0))
            Tinf = Tosurf*((psurf/pomax)**((gam-1.0)/gam))
            uinf = minf*np.sqrt(gam*rgas*Tinf)
        
            ie = uinf > umax
            uinf[ie] = umax[ie]
            
            # put limit where uinf is very small
            uinf[uinf<=0.001]=0.001
            
            roinf = psurf/(rgas*Tinf)
            ruinf = roinf*uinf
            
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
                
                # guess b.layer height
                if(ns>0):
                   if(d1[ns-1]>0.0 and d2[ns-1]>0.0):
                      hprev = d1[ns-1]/d2[ns-1]
                   else:
                      hprev = 2.5   
                   if(hprev<0):
                       hprev = 2.50
                   Del = d2[ns-1]*(3.15 + 1.72/(hprev-1.0)) + d1[ns-1] # BL thickness correlation from 1973 paper (see Drela paper on VII) 
                   # print('hprev:',hprev,'del:',d0[ns-1],'disp:',d1[ns-1],'mom:',d2[ns-1])      
                   # print('nb:',nb,'i:',i,'ns:',ns)
                else: # first point on surface
                   Del = 1.0e-4
                   # print('nb:',nb,'i:',i,'ns:',ns)
                # print('hprev:',hprev,'del:',d0[ns-1],'disp:',d1[ns-1],'mom:',d2[ns-1])      
                
                # print('nb:',nb,'i:',i,'ns:',ns)
                # print('Del:', Del)
                
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
                #rprof = np.sqrt(xd*xd + yd*yd)
                #velprof = np.sqrt(uprof*uprof + vprof*vprof + wprof*wprof)

                # wall normal coordinate using r projected onto unit normal rprof(j)=abs(r dot nhat)
                rprof = np.abs(-xd*dyt + yd*dxt)/np.sqrt(dxt*dxt + dyt*dyt)
                # Streamwise tangential velocity using u,v projected into unit tangent velprof(j)=u dot that
                velprof =(uprof*dxt + vprof*dyt)/np.sqrt(dxt*dxt + dyt*dyt)
                
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
                # cf=np.append(cf,tw/(0.5*ruinf[i]*uinf[i])) # proper method - uses calculated ro and U
                cf=np.append(cf,tw/(0.5*1.2048*10.33**2)) # to try match Argo and MISES Cf (uses constants)
                XX=np.append(XX,xprof[0])
                YY=np.append(YY,yprof[0])

                # Surface length s (confusingly SS is actually surface length for both ss and ps, but I kept in the same style as XX,YY)
                if(ns>0):
                    SS=np.append(SS, np.sqrt((XX[-1]-XX[-2])**2 + (YY[-1]-YY[-2])**2)+SS[-1])
                else:
                    SS=np.append(SS,0)
                
                ri = np.linspace(np.min(rprof),np.max(rprof),nprof)
                
                f = interp1d(rprof, yprof, kind='cubic')
                yi =  f(ri)
                
                f = interp1d(rprof, xprof, kind='cubic')
                xi =  f(ri)
                
                f = interp1d(rprof, uprof, kind='cubic')
                ui =  f(ri)
        
                f = interp1d(rprof, vprof, kind='cubic')
                vi =  f(ri)
        
                f = interp1d(rprof, wprof, kind='cubic')
                wi =  f(ri)
                
                f = interp1d(rprof, muprof, kind='cubic')
                mui =  f(ri)
        
                f = interp1d(rprof, mutprof, kind='cubic')
                muti = f(ri)
        
                f = interp1d(rprof, roprof, kind='cubic')
                roi =  f(ri)
        
                f = interp1d(rprof, turbprof, kind='cubic')
                turbi =f(ri)

                f = interp1d(rprof, dissprof, kind='cubic')
                dissi =f(ri)

                f = interp1d(rprof, pprof, kind='cubic')
                pi =f(ri)
        
                vmagi = np.sqrt(ui*ui + vi*vi + wi*wi)

                # This is Andy's method of getting ii using a LE correction but doesn't work for slow forming LPT BLs
                # if(ns<100): # correction near LE
                #    ii = vmagi < uinf[i]*0.99999
                # else:
                #    ii = ri < Del

                # New version that just picks smallest BL edge - but from testing, the 0.99uinf is never correct
                ii_vel = vmagi < 0.99999 * uinf[i]   # δ99 velocity criterion - for near to LE
                ii_del = ri < Del                 # correlation criterion - for everywhere else
                # Pick whichever gives the smaller boundary layer edge
                ii = np.logical_and(ii_vel, ii_del)
                  
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
                if(del_now > Del):
                   edgen = np.argmax(ri*ii)
                   edgej[i] = np.argmin(abs(rprof-ri[edgen]))

                
                delv = vmagi[1:]-vmagi[:-1]
                 
                uav = (ui[1:]+ui[:-1])*0.5
                vav = (vi[1:]+vi[:-1])*0.5
                wav = (wi[1:]+wi[:-1])*0.5
                                     
                turbav =(turbi[1:]+turbi[:-1])*0.5
                dissav =(dissi[1:]+dissi[:-1])*0.5 # dissipation due to mean strain from 2d mean prop, not calced here
                ronow  =  (roi[1:] + roi[:-1])*0.5
                munow  =  (mui[1:] + mui[:-1])*0.5
                mutnow = (muti[1:]+ muti[:-1])*0.5

                
                vnow = (uav*dxt + vav*dyt)/np.sqrt(dxt*dxt + dyt*dyt) 
                # if ns==0:
                #     print('uav', uav, 'dxt', dxt, 'vav', vav, 'dyt', dyt)
                    # vav is negative
                    
                diss_av = (munow/ronow)*(delv/dely)*(delv/dely) # 
                diss_model = (munow*mutnow/ronow)*(delv/dely)*(delv/dely) # modelled dissipation using turbulent viscosity mu_t
  
                vnow[vnow>uinf[i]]=uinf[i] 
  
                x0=np.append(x0,xi[edgen])
                y0=np.append(y0,yi[edgen])

                Ue=np.append(Ue,uinf[i])
                Pe=np.append(Pe,pi[edgen])

                # BL masks for loss breakdown (BL edge is found with interped high resolution y profile, so finding closest block coord)
                search = np.array([x[i,:], y[i,:]])
                dists = (search[0] - xi[edgen])**2 + (search[1] - yi[edgen])**2
                closest_idx = np.argmin(dists)
                mask = np.arange(search.shape[1]) <= closest_idx
                BL_bool[nb].append(mask[::-1])

                
                vn    = vnow/uinf[i]
                rvn   = ronow*vnow/ruinf[i]
                rvdef = 1.0 - rvn
                vdef  = 1.0 - vn 
                v2def = 1.0 - vn*vn

                # Data all normalised to axial chord
                d0=np.append(d0,np.sum(dely[:edgen])) # delta99
                d1=np.append(d1,np.sum(rvdef[:edgen]*dely[:edgen])) # # Displacement thickness
                d2=np.append(d2,np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen])) # Momentum thickness
                d3=np.append(d3,np.sum(rvn[:edgen]*v2def[:edgen]*dely[:edgen])) # Energy thickness
                # if ns==0:
                    # print('d1:', d1, 'd2:', d2, 'rvn[:edgen]', rvn[:edgen], 'vdef[:edgen]', vdef[:edgen], 'rvdef[:edgen]', rvdef[:edgen])
                    # rvn is issue! negative for some reason
                    # print('ronow:', ronow, 'vnow:', vnow, 'ruinf[i]:', ruinf[i])
                    # ruinf is quite high, and vnow is negative (because uav and vav are kinda flipped sign from each other in first few points)
                
                d1i=np.append(d1i,np.sum(vdef[:edgen]*dely[:edgen])) # ? inviscid?
                d2i=np.append(d2i,np.sum(vn[:edgen]*vdef[:edgen]*dely[:edgen])) # ? inviscid?
                d3i=np.append(d3i,np.sum(vn[:edgen]*v2def[:edgen]*dely[:edgen])) # ? inviscid?
                
                # d9=np.append(d9,np.sum(diss_av[:edgen]/(ronow[edgen]*uinf[i]**3) *dely[:edgen])) # Dissipation due to mean strain normalised to 1/(roe Ue3^3)
                # d10=np.append(d10,np.sum(turbav[:edgen]/(ronow[edgen]*uinf[i]**3) *dely[:edgen])) # Turbulence production normalised to 1/(roe Ue3^3)
                d9=np.append(d9,   np.sum(diss_av[:edgen]/(uinf[i]**3) *dely[:edgen])) # Dissipation due to mean strain - normalised to 1/Ue3^3
                d10=np.append(d10, np.sum(turbav[:edgen]/(uinf[i]**3) *dely[:edgen])) # Turbulence production - normalised to 1/Ue3^3
                d11=np.append(d11, np.sum((turbav[:edgen] + diss_av[:edgen])*dely[:edgen])) # Dissipation coefficient cd using actual dissipation
                d12=np.append(d12, np.sum((turbav[:edgen] + diss_model[:edgen])*dely[:edgen])) # Dissipation coefficient cd using turbulent viscosity modelled dissipation?
                Retheta=np.append(Retheta, uinf[i]*np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen])*ronow[edgen]/munow[edgen])
                Retau=np.append(Retau, ut*np.sum(dely[:edgen])*ronow[edgen]/munow[edgen])
                Res=np.append(Res, uinf[i]*SS[ns]*ronow[edgen]/munow[edgen])

                beta=np.append(beta, np.sum(rvdef[:edgen]*dely[:edgen]) / tw) # Clauser parameter
                K=np.append(K, munow[edgen]/(ronow[edgen]*uinf[i]**2)) # Pressure gradient parameter (acceleration parameter)
                lamb=np.append(lamb, np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen])**2 * munow[edgen]/ronow[edgen]) # Thwaites criterion
                
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

    H = np.divide(d1, d2) # Shape factor
        
    yplus_profile = np.array(yplus_profile)
    uplus_profile = np.array(uplus_profile)
    r_profile = np.array(r_profile)
    vel_profile = np.array(vel_profile)

    # Gradients and final calcs for Clauser / acceleration param / Thwaites
    beta[ISS] = beta[ISS] * np.gradient(Pe[ISS], SS[ISS])
    K[ISS] = K[ISS] * np.gradient(Ue[ISS], SS[ISS])
    lamb[ISS] = lamb[ISS] * np.gradient(Ue[ISS], SS[ISS])
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


