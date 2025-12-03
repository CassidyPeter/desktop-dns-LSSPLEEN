# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
import matplotlib.pyplot as plt

from meshing.read_case import *


def plot_monitor(casename,maw=2500):

    # get current path
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)

    # get inlet target values
    v_in = case['bcs']['vin']
    a_in = case['bcs']['alpha']
    po_in = case['bcs']['Poin']
    To_in = case['bcs']['Toin']
    
    # get gas props
    gam     = case['gas']['gamma']
    cp      = case['gas']['cp']
    mu_ref  = case['gas']['mu_ref']
    mu_tref = case['gas']['mu_tref']
    mu_cref = case['gas']['mu_cref']
    pr      = case['gas']['pr']
    cv = cp/gam
    rgas = cp-cv     
   
    # read monitor file
    f = os.path.join(path,casename,'monitor.txt')
    
    
    monitor = np.loadtxt(f,dtype={'names': ('iter', 'time', 'ro', 'ru', 'rv', 'rw', 'Et'),'formats': ('i','f','f','f','f','f','f')})

 
    time = monitor['time']
    ro = monitor['ro']
    ru = monitor['ru']
    rv = monitor['rv']
    rw = monitor['rw']
    Et = monitor['Et']
    
    
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
    
    
    s_in = cp*np.log(To_in/300) - rgas*np.log(po_in/1e5)
    
    prop = {}
    prop['time']=monitor['time']
    prop['ro'] = ro
    prop['ru'] = ru
    prop['rv'] = rv
    prop['rw'] = rw
    prop['Et'] = Et
    prop['p'] = p
    prop['u'] = u
    prop['v'] = v
    prop['w'] = w
    prop['T'] = T
    prop['s'] = s
    prop['po'] = po
    prop['To'] = To
    prop['mach'] = mach
    prop['vel'] = vel
    prop['alpha'] = alpha
    prop['timesteps']=monitor['time'].size * monitor['iter'][0]
    
    target = {}     
    target['vin'] = v_in*np.ones(len(time))
    target['ain'] = a_in*np.ones(len(time))
    target['sin'] = s_in*np.ones(len(time))
    target['poin'] = po_in*np.ones(len(time))


    window_size = maw
    weights = np.ones(window_size) / window_size
    
    plt.figure(1)

    plt.subplot(2,2,1)
    tempplt = np.convolve(prop['vel'], weights, mode='valid')
    prop['lastvel'] = tempplt[-1]
    plt.plot(time,vel,'-k.',time,target['vin'],time[0:len(tempplt)],tempplt)
    plt.xlabel('time (s)')
    plt.ylabel('inlet velocity (m/s)')
    plt.legend(['monitor','target','moving average'])
    
    plt.subplot(2,2,2)
    tempplt = np.convolve(prop['alpha'], weights, mode='valid')
    prop['lastalpha'] = tempplt[-1]
    plt.plot(time,alpha,'-k.',time,target['ain'],time[0:len(tempplt)],tempplt)
    plt.xlabel('time (s)')
    plt.ylabel('inlet flow angle (deg)')
    plt.legend(['monitor','target','moving average'])
    
    plt.subplot(2,2,3)
    plt.plot(time,np.exp(-(s-s_in)/rgas),'-k.',time,np.ones(len(time)))
    plt.xlabel('time (s)')
    plt.ylabel('inlet entropy exp(-s/R)')
    plt.legend(['monitor','target'])
        
    # plt.subplot(2,2,4)
    # plt.plot(time, p - p[-1],'-k.')
    # plt.xlabel('time (s)')
    # plt.ylabel('inlet pressure fluctuation (Pa)')


    plt.subplot(2,2,4)
    tempplt = np.convolve(prop['po'], weights, mode='valid')
    prop['lastpo'] = tempplt[-1]
    plt.plot(time, po,'-k.',time,target['poin'],time[0:len(tempplt)],tempplt)
    plt.xlabel('time (s)')
    plt.ylabel('inlet total pressure (Pa)')
    plt.legend(['monitor','target','moving average'])
    
    plt.show()  

    return prop, target




