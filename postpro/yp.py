# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np

def yp(prop,geom,inlet,exit):

    inlet_blocks = inlet['blocks']
    exit_blocks = exit['blocks']
    
    xinlet = inlet['x']
    xwake = exit['x']

    poin = 0
    pin = 0
    pitch = 0
    mass = 0

    for i in inlet_blocks:
        x = geom[i-1]['x'][:,0]
        if(i==inlet_blocks[0]): ii = np.argmin(np.abs(x-xinlet))
        maxy = np.max(geom[i-1]['y'][ii,:])
        miny = np.min(geom[i-1]['y'][ii,:])
        pitch = pitch + (maxy-miny)
    
        xx = np.squeeze(geom[i-1]['x'])
        yy = np.squeeze(geom[i-1]['y'])
        po = np.squeeze(prop[i-1]['po'])
        p =  np.squeeze(prop[i-1]['p'])
        ro = np.squeeze(prop[i-1]['ro'])
        u =  np.squeeze(prop[i-1]['u'])
        v =  np.squeeze(prop[i-1]['v'])
        ru = ro*u
        rv = ro*v
        
        dy   = -yy[ii,:-1]+yy[ii,1:]
        dx   = -xx[ii,:-1]+xx[ii,1:]
        ruav = (ru[ii,:-1]+ru[ii,1:])*0.5
        rvav = (rv[ii,:-1]+rv[ii,1:])*0.5
        poav = (po[ii,:-1]+po[ii,1:])*0.5
        pav = (p[ii,:-1]+p[ii,1:])*0.5
        dm = ruav*dy + rvav*dx
        mass = mass + np.sum(dm)
        poin = poin + np.sum(poav*dm)
        pin = pin + np.sum(pav*dy)
    poin = poin/mass
    pin = pin/pitch

    ywake = []    
    Yp = [] 
    uup = []
    vvp = []
    wwp = []
    uvp = []
    poex = 0
    pex = 0
    pitch = 0
    mex = 0
    TE_x = 0.9958*1.002 # hacky TE coords
    TE_y = -0.55875*1.002 # hacky TE coords
    alpha_2 = 61.5 # outlet metal angle
    cax = 0.06301079 # axial chord


    
    for i in exit_blocks:
        x = geom[i-1]['x'][:,1]
        if(i==exit_blocks[0]):
           ii = np.argmin(np.abs(x-xwake))
           y0 = geom[i-1]['y'][ii,0]

        maxy = np.max(geom[i-1]['y'][ii,:])
        miny = np.min(geom[i-1]['y'][ii,:])
        pitch = pitch + (maxy-miny)
        y = geom[i-1]['y'][ii,:]-y0

        xx = np.squeeze(geom[i-1]['x'])
        yy = np.squeeze(geom[i-1]['y'])
        po = np.squeeze(prop[i-1]['po'])
        p =  np.squeeze(prop[i-1]['p'])
        ro = np.squeeze(prop[i-1]['ro'])
        u =  np.squeeze(prop[i-1]['u'])
        v =  np.squeeze(prop[i-1]['v'])
        ruu =np.squeeze(prop[i-1]['ruu'])
        rvv =np.squeeze(prop[i-1]['rvv'])
        rww =np.squeeze(prop[i-1]['rww'])
        ruv =np.squeeze(prop[i-1]['ruv'])
        
        ru = ro*u
        rv = ro*v

        # Re stresses
        uu = ruu / ro
        vv = rvv / ro
        ww = rww / ro
        uv = ruv / ro

        dy   = -yy[ii,:-1]+yy[ii,1:]
        dx   = -xx[ii,:-1]+xx[ii,1:]
        ruav = (ru[ii,:-1]+ru[ii,1:])*0.5
        rvav = (rv[ii,:-1]+rv[ii,1:])*0.5
        poav = (po[ii,:-1]+po[ii,1:])*0.5
        pav = (p[ii,:-1]+p[ii,1:])*0.5
        dm = ruav*dy + rvav*dx
        mex = mex + np.sum(dm)
        poex = poex + np.sum(poav*dm)
        pex = pex + np.sum(pav*dy)

        if(y[0]<0): 
           ywake = np.append(yy[ii,:],ywake)
           Yp = np.append(po[ii,:],Yp) # effectively poexit non mass averaged
           uup = np.append(uu[ii,:],uup)
           vvp = np.append(vv[ii,:],vvp)
           wwp = np.append(ww[ii,:],wwp)
           uvp = np.append(uv[ii,:],uvp)
        else:
           ywake = np.append(ywake,yy[ii,:])
           Yp = np.append(Yp,po[ii,:]) # effectively poexit non mass averaged
           uup = np.append(uup, uu[ii,:])
           vvp = np.append(vvp, vv[ii,:])
           wwp = np.append(wwp, ww[ii,:])
           uvp = np.append(uvp, uv[ii,:])

    poex = poex/mex
    pex = pex/pitch

    # Centre wake coords on TE aligned streamline
    dy = (xwake-1) * np.tan(np.deg2rad(alpha_2))
    wake_centre_x = (TE_x + (xwake-1))/pitch
    wake_centre_y = (TE_y - dy)/pitch

    
    # identify if compresssor or turbine
    if(pex > pin): # compressor
        dyn = (poin - pin)
    else: # turbine
        dyn = (poin - pex)


    # Re stress profiles

    

    # wake dictionary
    wake = {}
    wake['y']  = ywake/pitch + abs(wake_centre_y)
    wake['yp'] = (poin-Yp)/dyn # Stagnation pressure loss coefficient
    wake['xi'] = 1 - (1 - (pex/Yp)**(0.4/1.4)) / (1 - (pex/poin)**(0.4/1.4)) # Kinetic energy loss coefficient (less sensitive to compressibility)


    # Re stress dictionary
    ReStress = {}
    ReStress['uu'] = uup
    ReStress['vv'] = vvp
    ReStress['ww'] = wwp
    ReStress['uv'] = uvp
    
    # performance dictionary
    perf = {}
    perf['wake'] = wake
    perf['ReStress'] = ReStress
    perf['yp'] = (poin-poex)/dyn
    perf['xi'] = 1 - (1 - (pex/poex)**(0.4/1.4)) / (1 - (pex/poin)**(0.4/1.4))
    perf['mass in'] = mass
    perf['mass out'] = mass
    perf['poin'] = poin
    perf['pin'] = pin
    perf['poex'] = poex
    perf['pex'] = pex
    perf['dyn'] = dyn



    #print(poin,poex,pin,pex,dyn,mass,mex)

    return perf


