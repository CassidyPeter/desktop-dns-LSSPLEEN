# Peter Cassidy September 2025

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import scienceplots

from .read_case import *
from .read_2d_mean import *
from .read_flo import *
from .boundarylayer_v2 import *
from .read_profile_TE import *


def loss_breakdown_debug(casename, *args):


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
    if (len(args) > 0):
        nfiles = args[0]
        prop,blk,_ = read_2d_mean(casename,args[0])
        bl = boundarylayer_v2(casename,args[0])
    else:
        prop,blk = read_flo(casename)
        bl = boundarylayer_v2(casename)

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
    domain_min = -0.25
    domain_max = 1.25
    
    # Wake shape
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
    wake_corner_3_x = wake_corner_1_x + ((1.3-0.55875) * np.cos(np.deg2rad(61.5-19))) # lower right
    wake_corner_4_y = - 1.3
    wake_corner_4_x = wake_corner_2_x + ((1.3-0.55875) * np.cos(np.deg2rad(61.5-7))) # lower left
    wake_coords = np.array([
        (wake_corner_1_x, wake_corner_1_y),
        (wake_corner_2_x, wake_corner_2_y),
        (wake_corner_4_x, wake_corner_4_y),
        (wake_corner_3_x, wake_corner_3_y)
    ])
    wake_path = Path(wake_coords)
    
    
    for ib in range(Nb):
            x = blk[ib]['x']
            y = blk[ib]['y']
            a = prop[ib]['area']
            f = prop[ib]['Dsi']
    
            # Flow quantities are stored n nodes of mesh, so get finite volume cells for each point
            fav = 0.25 * (f[:-1,:-1] + f[1:,:-1] + f[:-1,1:] + f[1:,1:])
            xav = 0.25 * (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:])
            yav = 0.25 * (y[:-1,:-1] + y[1:,:-1] + y[:-1,1:] + y[1:,1:])
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
            
            
            losses['PS_BL'] += np.sum(fav[mask_ps_cell] * aav[mask_ps_cell])
            losses['SS_BL'] += np.sum(fav[mask_ss_cell] * aav[mask_ss_cell])
            losses['Wake'] += np.sum(fav[mask_wake_cell] * aav[mask_wake_cell])
            losses['Passage'] += np.sum(fav[mask_passage] * aav[mask_passage])
    


            # debug negative passage fav
            # 1) Basic stats
            print("f (node) min/max/mean:", np.nanmin(f), np.nanmax(f), np.nanmean(f))
            print("fav (cell) min/max/mean:", np.nanmin(fav), np.nanmax(fav), np.nanmean(fav))
            print("aav min/max:", np.nanmin(aav), np.nanmax(aav))
            print("mask_passage count:", np.count_nonzero(mask_passage))
            print("mask_ps_cell count:", np.count_nonzero(mask_ps_cell), " mask_ss_cell:", np.count_nonzero(mask_ss_cell),
                  " mask_wake_cell:", np.count_nonzero(mask_wake_cell))
            
            # 2) NaN/Inf checks
            if np.isnan(f).any() or np.isinf(f).any():
                print("WARNING: f has NaN or Inf.  NaN count:", np.count_nonzero(np.isnan(f)))
            if np.isnan(fav).any() or np.isinf(fav).any():
                print("WARNING: fav has NaN or Inf.  NaN count:", np.count_nonzero(np.isnan(fav)))
            
            # 3) Per-region stats (area-weighted and plain)
            def region_stats(mask_cell, name):
                vals = fav[mask_cell]
                areas = aav[mask_cell]
                print(vals.size)
                if vals.size == 0:
                    print(f"{name}: empty")
                    return
                weighted = np.sum(vals * areas) / np.sum(areas)
                print(f"{name} count: {vals.size}, min/mean/max: {np.nanmin(vals):.4e}, {np.nanmean(vals):.4e}, {np.nanmax(vals):.4e}, area-weighted mean: {weighted:.4e}")
            
            region_stats(mask_ps_cell, "PS_BL")
            region_stats(mask_ss_cell, "SS_BL")
            region_stats(mask_wake_cell, "Wake")
            region_stats(mask_passage, "Passage")
            
            # 4) Find the worst negative cells in the passage (largest negative area-weighted contribution)
            if np.count_nonzero(mask_passage):
                vals = fav[mask_passage].ravel()
                areas = aav[mask_passage].ravel()
                # Sort by contribution (most negative first)
                contrib = vals * areas
                idx_flat = np.argsort(contrib)  # ascending (largest negative at front)
                worst_idx = idx_flat[:10]  # top 10 worst
                # Map these back to cell indices
                passage_indices = np.argwhere(mask_passage)
                print("Top negative contributors (value, area, contrib, i, j):")
                for k in worst_idx:
                    i, j = passage_indices[k]
                    print(vals[k], areas[k], contrib[k], "cell:", i, j, " coords:", xav[i,j], yav[i,j])
            
            # 5) Optional: inspect node values surrounding the worst cell to see if a node is strongly negative
            if np.count_nonzero(mask_passage):
                # pick worst cell index i,j from above loop
                i_w, j_w = passage_indices[worst_idx[0]]
                # nodes are at i_w or i_w+1 etc (since fav used f[:-1,:-1] etc)
                nodes_vals = f[i_w:i_w+2, j_w:j_w+2]
                print("Surrounding node f values for worst cell:", nodes_vals)


    
        
    losses['Passage'] = -losses['Passage'] # so uhhhh not sure why I have to flip this, the Dsi values in passage are negative for some reason...
    losses['Total'] = losses['PS_BL'] + losses['SS_BL'] + losses['Wake'] + losses['Passage']
    plt.xlim(domain_min, domain_max)

    return losses, plt