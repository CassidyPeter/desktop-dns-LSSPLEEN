# Peter Cassidy
# Exports blocks to VTK StructuredGrid format for Paraview

import numpy as np
import pyvista as pv
from postpro.read_case import *

def export_vtk(casename, prop, geom):

    case = read_case(casename)
    Cax = 0.06301079

    # Spanwise spacing
    nk = case['solver']['nk']
    Lz = case['solver']['span']
    dz = (Lz/(nk-1)) / Cax

    for ib in range(len(geom)):
    # for ib in [2,3,4,6,8]:
        print(f"Exporting block {ib+1}/{len(geom)}")
    
        # Physical 2D coordinates of the block
        x = geom[ib]['x']   # (ni, nj)
        y = geom[ib]['y']   # (ni, nj)
        ni, nj = x.shape
        # nk = prop[ib]['Q'].shape[2]  # assume all 3D vars same shape
    
        # Spanwise coordinates
        z = np.arange(nk) * dz
    
        # Extrude into 3D
        X = np.repeat(x[:, :, None], nk, axis=2)
        Y = np.repeat(y[:, :, None], nk, axis=2)
        Z = np.repeat(z[None, None, :], ni, axis=0)
        Z = np.repeat(Z, nj, axis=1)
    
        # Build StructuredGrid
        grid = pv.StructuredGrid(X, Y, Z)
        # print(grid)
        del prop[ib]['mut_model']
        del prop[ib]['area']
    
        # Attach fields
        for key, arr in prop[ib].items():
            if arr.ndim == 3:  # already 3D
                grid[key] = arr.ravel(order="F").astype(np.float32)
            elif arr.ndim == 2:  # extrude along z
                arr3d = np.repeat(arr[:, :, None], nk, axis=2)
                grid[key] = arr3d.ravel(order="F").astype(np.float32)
    
        # Save block
        fname = f"{casename}/block{ib+1}.vtk"
        grid.save(fname)
        print(f" -> Saved {fname}")
