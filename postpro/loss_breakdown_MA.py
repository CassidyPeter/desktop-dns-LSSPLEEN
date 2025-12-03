# Peter Cassidy October 2025
# Mass-averaged version based on recommendation of Soham & Dom

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
# import scienceplots

from .read_case import *
from .read_2d_mean import *
from .read_flo import *
from .boundarylayer_v2 import *
from .read_profile_TE import *


def loss_breakdown_MA(casename, nmean, entropy_term):


    flo = {}

    path = os.getcwd()

    cols=['#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00']

    # get case details
    case = read_case(casename)

    # get flow and geom
    # Kinda inefficient as we call read_2d_mean here as well as in boundarylayer_v2, but fuck it cba 
    prop,blk,_ = read_2d_mean(casename,[nmean])
    bl = boundarylayer_v2(casename,[nmean])

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

    
    Nb = len(blk) 


    losses = {'Total': 0.0, 'PS_BL': 0.0, 'SS_BL': 0.0, 'Wake': 0.0, 'Passage': 0.0}
    losses_MA = {'Total': 0.0, 'PS_BL': 0.0, 'SS_BL': 0.0, 'Wake': 0.0, 'Passage': 0.0}
    mass_fluxes = {'PS_BL': 0.0, 'SS_BL': 0.0, 'Wake': 0.0, 'Passage': 0.0}
    domain_min = -0.2
    domain_max = 1.2
    
    # Wake shape - tuned for high lift case
    xprof,yprof,pitch,stag,xLE,yLE,xTE,yTE=read_profile_TE('geom/LS_SPLEEN_PROFILE_10k.txt',True)
    Cax = xTE-xLE
    # xTE_Norm = 0.9958 * Cax # hacky TE by observation (Andy's code takes last x position as TE....)
    # yTE_Norm = -0.55875*Cax
    xTE_Norm = 0.9958*1.002 * Cax # hacky TE by observation (Andy's code takes last x position as TE....)
    yTE_Norm = -0.55875*1.002*Cax
    wake_width = 2e-3 # in mm
    wake_corner_1_x = (xTE_Norm + np.sin(np.deg2rad(61.5)) * (wake_width+0.2e-3)) / Cax # upper right
    wake_corner_1_y = (yTE_Norm + np.cos(np.deg2rad(61.5)) * (wake_width+0.2e-3)) / Cax
    wake_corner_2_x = (xTE_Norm - np.sin(np.deg2rad(61.5)) * (wake_width-0.2e-3)) / Cax # upper left
    wake_corner_2_y = (yTE_Norm - np.cos(np.deg2rad(61.5)) * (wake_width-0.2e-3)) / Cax
    wake_corner_3_y = - 1.3
    wake_corner_3_x = wake_corner_1_x + ((1.3-0.55875) * np.cos(np.deg2rad(61.5-19))) # lower right - HL
    wake_corner_4_y = - 1.3
    wake_corner_4_x = wake_corner_2_x + ((1.3-0.55875) * np.cos(np.deg2rad(61.5-7))) # lower left - HL
    wake_coords = np.array([
        (wake_corner_1_x, wake_corner_1_y),
        (wake_corner_2_x, wake_corner_2_y),
        (wake_corner_4_x, wake_corner_4_y),
        (wake_corner_3_x, wake_corner_3_y)
    ])
    wake_path = Path(wake_coords)
    

    # prop2,geom2,_=read_2d_mean(casename,[10])
    # plot_flo(prop2,geom2,'s',[-0.05, 0.1])
    # plot_flo(prop2,geom2,'Dsi',[-2, 2])
    # plot_flo(prop2,geom2,'Dsi',[-5, 5])
    
    
    plt.figure(1)
    plt.axis('equal')
    for ib in range(Nb):
            x = blk[ib]['x']
            y = blk[ib]['y']
            a = prop[ib]['area']
            # f = prop[ib]['Dsi'] # Total irreversible entropy
            f = prop[ib][entropy_term] # Using entropy term defined as input parameter
            ru = prop[ib]['ru'] # rho * u
    
            # Flow quantities are stored n nodes of mesh, so get finite volume cells for each point
            fav = 0.25 * (f[:-1,:-1] + f[1:,:-1] + f[:-1,1:] + f[1:,1:])
            xav = 0.25 * (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:])
            yav = 0.25 * (y[:-1,:-1] + y[1:,:-1] + y[:-1,1:] + y[1:,1:])
            ruav = 0.25 * (ru[:-1,:-1] + ru[1:,:-1] + ru[:-1,1:] + ru[1:,1:])
            aav = a   # already cell-based, same shape as fav
        
    
            if ib in [2,4,6]: # Suction Side block
                mask_ss = bl['Mask'][ib]
                mask_ps = np.full(x.shape, False)
    
            elif ib==3: # Pressure Side block
                mask_ps = bl['Mask'][ib]
                mask_ss = np.full(x.shape, False)
    
            else: # all other blocks just given Falsess
                mask_ss = np.full(x.shape, False)
                mask_ps = np.full(x.shape, False)
    
            
            # Boolean mask, give True for cell if any of surrounding nodes are true. Converts from node to cell-centered
            mask_ss_cell = (mask_ss[:-1,:-1] | mask_ss[1:,:-1]  | mask_ss[:-1,1:]  | mask_ss[1:,1:]) 
            mask_ps_cell = (mask_ps[:-1,:-1] | mask_ps[1:,:-1]  | mask_ps[:-1,1:]  | mask_ps[1:,1:])
    
            # Wake mask
            wake_pts = np.vstack((x.ravel(), y.ravel())).T
            mask_flat = wake_path.contains_points(wake_pts)
            mask_wake = mask_flat.reshape(x.shape)
            mask_wake_cell = (mask_wake[:-1,:-1] | mask_wake[1:,:-1]  | mask_wake[:-1,1:]  | mask_wake[1:,1:])
        
    
            # Update BL masks to not include area near TE reserved for wake sub-region
            mask_ps_cell = mask_ps_cell & ~(mask_ps_cell & mask_wake_cell)
            mask_ss_cell = mask_ss_cell & ~(mask_ss_cell & mask_wake_cell)
    
            # Update wake mask to constrain
            mask_wake_cell = mask_wake_cell & (xav<domain_max)
    
            # Combine all masks to subtract and get passage
            mask_passage = ~(mask_ps_cell | mask_ss_cell | mask_wake_cell) & (xav<domain_max) & (xav>domain_min)
            
            
            # losses['PS_BL'] += np.sum(fav[mask_ps_cell] * aav[mask_ps_cell])
            # losses['SS_BL'] += np.sum(fav[mask_ss_cell] * aav[mask_ss_cell])
            # losses['Wake'] += np.sum(fav[mask_wake_cell] * aav[mask_wake_cell])
            # losses['Passage'] += np.sum(fav[mask_passage] * aav[mask_passage])

            # ---- Integrate entropy generation, mass-weighted ----
            mdot_ps = np.sum(ruav[mask_ps_cell] * aav[mask_ps_cell])
            mdot_ss = np.sum(ruav[mask_ss_cell] * aav[mask_ss_cell])
            mdot_wake = np.sum(ruav[mask_wake_cell] * aav[mask_wake_cell])
            mdot_pass = np.sum(ruav[mask_passage] * aav[mask_passage])
            
            loss_ps = np.sum(fav[mask_ps_cell] * ruav[mask_ps_cell] * aav[mask_ps_cell])
            loss_ss = np.sum(fav[mask_ss_cell] * ruav[mask_ss_cell] * aav[mask_ss_cell])
            loss_wake = np.sum(fav[mask_wake_cell] * ruav[mask_wake_cell] * aav[mask_wake_cell])
            loss_pass = np.sum(fav[mask_passage] * ruav[mask_passage] * aav[mask_passage])
            
            # Accumulate per-block totals
            losses['PS_BL'] += loss_ps
            losses['SS_BL'] += loss_ss
            losses['Wake'] += loss_wake
            losses['Passage'] += loss_pass
            
            mass_fluxes['PS_BL'] += mdot_ps
            mass_fluxes['SS_BL'] += mdot_ss
            mass_fluxes['Wake'] += mdot_wake
            mass_fluxes['Passage'] += mdot_pass

        
            # Plot masks
            # plt.scatter(xav[mask_passage], yav[mask_passage],s=1, c=cols[1])
            # plt.scatter(xav[mask_ps_cell], yav[mask_ps_cell], s=3, c=cols[0])
            # plt.scatter(xav[mask_ss_cell], yav[mask_ss_cell], s=3, c=cols[2])
            # plt.scatter(xav[mask_wake_cell], yav[mask_wake_cell], s=3, c=cols[3])
            
            plt.contourf(xav, yav, mask_passage.astype(float), levels=[0.5, 1.5], colors=cols[1])
            plt.contourf(xav, yav, mask_ps_cell.astype(float), levels=[0.5, 1.5], colors=cols[0])
            plt.contourf(xav, yav, mask_ss_cell.astype(float), levels=[0.5, 1.5], colors=cols[2])
            plt.contourf(xav, yav, mask_wake_cell.astype(float), levels=[0.5, 1.5], colors=cols[3])
            
    
            plt.contour(xav, yav, mask_passage.astype(float), levels=[0.5], colors=cols[1])
            plt.contour(xav, yav, mask_ps_cell.astype(float), levels=[0.5], colors=cols[0])
            plt.contour(xav, yav, mask_ss_cell.astype(float), levels=[0.5], colors=cols[2])
            plt.contour(xav, yav, mask_wake_cell.astype(float), levels=[0.5], colors=cols[3])
            # plt.pcolormesh(xav, yav, mask_wake_cell.astype(float), cmap='Greys', shading='auto')

        
    losses['Total'] = losses['PS_BL'] + losses['SS_BL'] + losses['Wake'] + losses['Passage']
    plt.xlim(domain_min, domain_max)




    # Normalize by total mass flux to get true mass-averaged values
    for key in ['PS_BL', 'SS_BL', 'Wake', 'Passage']:
            losses_MA[key] = losses[key] / mass_fluxes[key]
    losses_MA['Total'] = losses_MA['PS_BL'] + losses_MA['SS_BL'] + losses_MA['Wake'] + losses_MA['Passage']
    
    # Total as weighted mean
    # total_mdot = sum(mass_fluxes.values())
    # losses['Total'] = (
    #     sum(losses[k] * mass_fluxes[k] for k in ['PS_BL', 'SS_BL', 'Wake', 'Passage'])
    #     / total_mdot
    # )

    return losses, losses_MA, plt





