import os
import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
import timeit

from postpro.read_case import *
from postpro.grad import *
from postpro.read_profile import *

def POD_v2(casename,nfiles_req,blocks,dt,savedata=True):
    ### POD analysis of raw KCut blocks - transformed to be grid-aligned
    print('Starting POD analysis')

    POD = {}
    flo = {}
    path = os.getcwd()
    
    # get case details
    try:
        case = read_case(casename)
    except FileNotFoundError:
        print(f"Error: Case directory not found at {casename}")
        exit()
    except Exception as e:
        print(f"Error reading case: {e}")
        exit()
    
    
    # unpack case
    blk = case['blk']
    
    # get solver version
    version = case['solver']['version']
    
    # get gas props
    gam = case['gas']['gamma']
    cp = case['gas']['cp']
    mu_ref = case['gas']['mu_ref']
    mu_tref = case['gas']['mu_tref']
    mu_cref = case['gas']['mu_cref']
    pr = case['gas']['pr']
    cv = cp/gam
    rgas = cp-cv
    
    cax = 0.06301079
    try:
        x_prof,y_prof,pitch,stag=read_profile('geom/LS_SPLEEN_PROFILE_10k.txt',True)
    except FileNotFoundError:
        print("Warning: Profile file 'geom/LS_SPLEEN_PROFILE_10k.txt' not found. Continue without.")
        
    
    # read kslice file
    kfile = os.path.join(path,casename,'kslice_time.txt')
    try:
        ktime = np.loadtxt(kfile,dtype={'names': ('ncut', 'time', 'k'),'formats': ('i', 'f', 'i')})
    except FileNotFoundError:
        print(f"Error: kslice_time.txt not found at {kfile}")
        exit()
    
    nfmax = len(ktime['ncut'])
    
    if(len(nfiles_req)==1):
        if (nfiles_req[0] > nfmax) or (nfiles_req[0]==0):
            nfile = ktime['ncut']
            time =  ktime['time']
        elif nfiles_req[0] < 0 :
            nfnow = np.abs(nfiles_req)-1
            nfile = ktime['ncut'][nfnow]
            time =  ktime['time'][nfnow]
        else:
            nfile = ktime['ncut'][-nfiles_req[0]:]
            time  = ktime['time'][-nfiles_req[0]:]
    
    else:
        nfile = ktime['ncut'][nfiles_req[0]-1:nfiles_req[1]]
        time =  ktime['time'][nfiles_req[0]-1:nfiles_req[1]]
    
    nfiles = len(time) # This is the actual number of files (Nt)
    nt = 0
    
    x_all = None # nIxnJ (where nI is block 4 nI + block6 nI)
    y_all = None # nIxnJ (where nI is block 4 nI + block6 nI)
    u_all = [] # (nIxnJ)xnT (where nI and nJ have been made long skinny, nT is number of kcuts)
    v_all = [] # (nIxnJ)xnT (where nI and nJ have been made long skinny, nT is number of kcuts)
    w_all = [] # (nIxnJ)xnT (where nI and nJ have been made long skinny, nT is number of kcuts) # new
    print(f"Reading {nfiles} snapshots...")
    for idx, ncut in enumerate(nfile):
        start_ncut = timeit.default_timer()
        nt = nt + 1
    
        x_blocks = []
        y_blocks = []
        u_blocks = []
        v_blocks = []
        w_blocks = [] # new
    
        # for ib in range(len(blk)):
        for ib in blocks:
    
            x = blk[ib]['x']*cax
            y = blk[ib]['y']*cax
    
            flo[ib] = {}
    
            ni,nj = np.shape(blk[ib]['x'])
            nk = 1
    
            ro = np.zeros([ni,nj])
            ru = np.zeros([ni,nj])
            rv = np.zeros([ni,nj])
            rw = np.zeros([ni,nj])
            Et = np.zeros([ni,nj])
    
            flow_name = 'kcut_' + str(ib+1) + '_' + str(ncut)
            flow_file = os.path.join(path,casename,flow_name)
            
            try:
                f = open(flow_file,'rb')
            except FileNotFoundError:
                print(f"Warning: Snapshot file {flow_file} not found. Skipping.")
                continue
                
            q = np.fromfile(f,dtype='float64',count=ni*nj*5)
            f.close()
            
            if q.size != ni*nj*5:
                print(f"Warning: Snapshot file {flow_file} has unexpected size. Skipping.")
                continue
    
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
    
            # flip last SS block
            if ib==6:
                x = x[::-1,:]
                y = y[::-1,:]
                u = u[::-1,:]
                v = v[::-1,:]
                w = w[::-1,:]
    
    
            x_blocks.append(x)
            y_blocks.append(y)
            u_blocks.append(u)
            v_blocks.append(v)
            w_blocks.append(w) # new
    
        # --- combine blocks along i-direction ---
        if not x_blocks: # Check if list is empty (e.g., if files were skipped)
            print(f"No data loaded for snapshot {ncut}, skipping.")
            nt -= 1
            continue
            
        x_combined = np.concatenate(x_blocks, axis=0)
        y_combined = np.concatenate(y_blocks, axis=0)
        u_combined = np.concatenate(u_blocks, axis=0)
        v_combined = np.concatenate(v_blocks, axis=0)
        w_combined = np.concatenate(w_blocks, axis=0) # new
    
        if idx == 0: # save geometry once
            x_all = x_combined
            y_all = y_combined
    
        # reshape into long-skinny column
        u_col = u_combined.reshape(-1,1,order='F')
        v_col = v_combined.reshape(-1,1,order='F')
        w_col = w_combined.reshape(-1,1,order='F') # new
    
        u_all.append(u_col)
        v_all.append(v_col)
        w_all.append(w_col) # new
        
        # print(f"Loaded snapshot {idx+1}/{nfiles} (ncut={ncut})")
    
    # Check if any data was loaded
    if not u_all:
        print("Error: No data was successfully loaded. Exiting.")
        exit()
    
    # stack into arrays: (nI*nJ) x ncuts
    u_all = np.hstack(u_all)
    v_all = np.hstack(v_all)
    w_all = np.hstack(w_all) # new
    
    print("Data loading complete.")
    
    # ==============================================================================
    # === NEW SECTION: TRANSFORM GRID AND VELOCITIES ===
    # ==============================================================================
    
    print("Calculating grid transformation...")
    
    # Get grid dimensions
    Ni, Nj = x_all.shape
    NxNy = Ni * Nj
    Nt = u_all.shape[1]
    
    # Calculate grid partial derivatives
    # Using more accurate central differences (except boundaries)
    dy_di, dy_dj = np.gradient(y_all, edge_order=2)
    dx_di, dx_dj = np.gradient(x_all, edge_order=2)
    
    
    # Define basis vectors based on user's description
    Nx = dy_dj
    Ny = -dx_dj
    Tx = Ny 
    Ty = -Nx
    
    # Normalization magnitude
    Mag = np.sqrt(Tx**2 + Ty**2)
    Mag[Mag < 1e-12] = 1.0 
    
    # Normalize basis vectors (shape (Ni, Nj))
    Tx_norm = Tx / Mag
    Ty_norm = Ty / Mag
    Nx_norm = Nx / Mag
    Ny_norm = Ny / Mag
    
    print("Applying transformation to all velocity snapshots...")
    
    # Reshape velocities to 3D fields (Ni, Nj, Nt)
    u_field = u_all.reshape(Ni, Nj, Nt, order='F')
    v_field = v_all.reshape(Ni, Nj, Nt, order='F')
    w_field = w_all.reshape(Ni, Nj, Nt, order='F') # new
    
    # Reshape basis vectors for broadcasting (Ni, Nj, 1)
    Tx_3d = Tx_norm[:, :, np.newaxis]
    Ty_3d = Ty_norm[:, :, np.newaxis]
    Nx_3d = Nx_norm[:, :, np.newaxis]
    Ny_3d = Ny_norm[:, :, np.newaxis]
    
    # Apply transformation (projection)
    u_transformed = u_field * Tx_3d + v_field * Ty_3d
    v_transformed = u_field * Nx_3d + v_field * Ny_3d
    w_transformed = w_field
    
    # Reshape transformed velocities back to (NxNy, Nt)
    u_all = u_transformed.reshape(NxNy, Nt, order='F')
    v_all = v_transformed.reshape(NxNy, Nt, order='F')
    w_all = w_transformed.reshape(NxNy, Nt, order='F') # new
    
    print("Creating new rectangular 'box' grid for calculation...")
    # Save original grid for plotting
    x_all_original = np.copy(x_all)
    y_all_original = np.copy(y_all)
    
    # Create new "box" coordinates (simple index grid)
    i_coords = np.arange(Ni)
    j_coords = np.arange(Nj)
    x_all_new, y_all_new = np.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Overwrite x_all and y_all with the new index grid
    # The gradient calls later will now use this new grid
    x_all = x_all_new.astype(float)
    y_all = y_all_new.astype(float)
    
    # --- NEW CODE: Create physical 'plotting' grid (x_plot, y_plot) ---
    # This grid represents the true physical geometry for visualization.
    # The POD calculation will still use the (i, j) index grid (x_all, y_all).
    print("Creating new physical 'plotting' grid (x_plot, y_plot)...")
    
    # --- 1. Calculate y_plot (distance from wall) ---
    # This also handles the "upside down" issue.
    # Wall is at j=Nj-1, Far-field is at j=0
    # Calculate segment lengths along j-lines (moving away from wall)
    dx_j = np.diff(x_all_original, axis=1) # Shape (Ni, Nj-1)
    dy_j = np.diff(y_all_original, axis=1) # Shape (Ni, Nj-1)
    ds_j = np.sqrt(dx_j**2 + dy_j**2)      # ds_j[i, k] is dist between (i,k) and (i,k+1)
    
    # We need to sum backwards from the wall at j=Nj-1
    # ds_j_rev[i, 0] is the segment (Nj-2) to (Nj-1) [closest to wall]
    ds_j_rev = ds_j[:, ::-1] 
    
    # Cumulative sum of reversed segments
    # cum_dist_rev[i, 0] = dist(Nj-2, Nj-1)
    # cum_dist_rev[i, 1] = dist(Nj-2, Nj-1) + dist(Nj-3, Nj-2)
    cum_dist_rev = np.cumsum(ds_j_rev, axis=1) # Shape (Ni, Nj-1)
    
    # Create y_plot (Ni, Nj)
    y_plot_rev = np.zeros((Ni, Nj))
    y_plot_rev[:, 1:] = cum_dist_rev # y_plot_rev[i, 0] = 0 (this will be the wall point's new index)
                                    # y_plot_rev[i, Nj-1] = total dist (far-field point's new index)
    
    # Reverse j-axis back to match original data order (j=0 is far-field, j=Nj-1 is wall)
    # y_plot[i, Nj-1] = 0 (wall)
    # y_plot[i, 0]   = total dist (far-field)
    y_plot = y_plot_rev[:, ::-1]
    
    # --- 2. Calculate x_plot (normalized arc length s/SL) ---
    # *** s/SL BUG FIX IS HERE ***
    SL_total_unscaled = 1.491030896539091 # Total blade surface length (UNSCALED)
    SL_total_scaled = SL_total_unscaled * cax # Apply scaling
    # *** END FIX ***
    
    x_wall = x_all_original[:, -1] # Wall coords (at j=Nj-1)
    y_wall = y_all_original[:, -1]
    
    # Calculate segment lengths along the wall (in the i-direction)
    dx_wall = np.diff(x_wall)
    dy_wall = np.diff(y_wall)
    ds_wall = np.sqrt(dx_wall**2 + dy_wall**2)
    
    # Cumulative sum to get arc length *of this block*
    s_block = np.zeros(Ni)
    s_block[1:] = np.cumsum(ds_wall)
    
    # Total arc length of this block
    S_block_total = s_block[-1]
    
    # Find the starting arc length (offset)
    # We know s_abs = s_block + s_offset
    # We also know s_abs[-1] = SL_total_scaled (since s/SL=1 at i=Ni-1)
    # SL_total_scaled = s_block[-1] + s_offset
    # s_offset = SL_total_scaled - s_block[-1]
    s_offset = SL_total_scaled - S_block_total
    
    # Create the 1D array of absolute arc length and normalize it
    s_abs_1d = s_block + s_offset
    s_over_SL_1d = s_abs_1d / SL_total_scaled
    
    # Tile to create 2D (Ni, Nj) array for plotting
    x_plot = np.tile(s_over_SL_1d[:, np.newaxis], (1, Nj))
    
    
    print("... 'x_plot' and 'y_plot' created for plotting.")
    print(f"  y_plot wall (j={Nj-1}) min/max: {y_plot[:,-1].min():.2e}, {y_plot[:,-1].max():.2e}")
    print(f"  y_plot far-field (j=0) min/max: {y_plot[:,0].min():.3f}, {y_plot[:,0].max():.3f}")
    print(f"  x_plot (s/SL) start/end: {x_plot[0,0]:.4f}, {x_plot[-1,0]:.4f}")
    # --- END NEW CODE ---
    
    print("Transformation complete. Starting POD...")
    
    # ==============================================================================
    # === END NEW SECTION ===
    # ==============================================================================
    
    
    # print("x_all:", x_all.shape) # Will be (864, 144)
    # print("y_all:", y_all.shape) # Will be (864, 144)
    # print("u_all:", u_all.shape) # Will be (124416, 560)
    # print("v_all:", v_all.shape) # Will be (124416, 560) 
    
    # --- POD processing ---
    
    Nx = x_all.shape[0] # This is Ni
    Ny = x_all.shape[1] # This is Nj
    # Nt = u_all.shape[1] # Already defined
    # NxNy = Nx*Ny # Already defined
    
    
    MU = np.mean(u_all,axis=1,keepdims=True)
    MV = np.mean(v_all,axis=1,keepdims=True)
    MW = np.mean(w_all,axis=1,keepdims=True) # new
    
    Up = u_all - MU
    Vp = v_all - MV
    Wp = w_all - MW
    
    # UV
    UV = np.vstack([Up,Vp])
    print("Shape of UV matrix:", UV.shape)
    
    # Correlation matrix
    print("Calculating Correlation matrix C (UV.T @ UV)...")
    C = UV.T @ UV
    print("Shape of C:", C.shape)
    
    # Eigen decomposition
    print("Calculating Eigendecomposition...")
    Lam, Chi = np.linalg.eig(C)
    
    # Ensure eigenvalues and eigenvectors are real (they should be for a real, symmetric matrix)
    Lam = np.real(Lam) # Energy of each mode
    Chi = np.real(Chi) # Temporal coefficients
    
    # Sort eigenvalues/modes descending
    idx = np.argsort(Lam)[::-1]
    Lam = Lam[idx]
    Chi = Chi[:,idx]
    
    # POD spatial modes
    print("Calculating POD spatial modes (Phi = UV @ Chi)...")
    Phi = UV @ Chi
    Phi_u = Phi[:NxNy, :]
    Phi_v = Phi[NxNy:, :]
    
    # Re stresses
    uu_real = np.mean(Up * Up, axis=1)
    vv_real = np.mean(Vp * Vp, axis=1)
    uv_real = np.mean(Up * Vp, axis=1)
    
    uu_real = uu_real.reshape(Nx, Ny, order='F')
    vv_real = vv_real.reshape(Nx, Ny, order='F')
    uv_real = uv_real.reshape(Nx, Ny, order='F')
    
    # RMS fields
    uRMS = np.sqrt(np.mean(Up**2, axis=1)).reshape(Nx,Ny, order='F')
    vRMS = np.sqrt(np.mean(Vp**2, axis=1)).reshape(Nx,Ny, order='F')
    
    cum_lam = np.cumsum(Lam)/np.sum(Lam)
    
    print("Phi_u shape:", Phi_u.shape)
    print("Phi_v shape:", Phi_v.shape)
    print("Eigenvalues (Lam) shape:", Lam.shape)
    print("uRMS shape:", uRMS.shape)
    print("...POD core calculation finished.")
    
    # Normalise POD consistently with Re stresses
    # Note: This normalization might need review. 
    # A common one is to make Phi orthonormal: Phi_norm = Phi / np.sqrt(Lam[np.newaxis,:])
    # But let's follow the script's original logic for now.
    Phi_u_norm = Phi_u / np.sqrt(Nt)
    Phi_v_norm = Phi_v / np.sqrt(Nt)
    
    # Reshape to physical space
    Phi_u_norm = Phi_u_norm.reshape(Nx,Ny,Nt,order='F')
    Phi_v_norm = Phi_v_norm.reshape(Nx,Ny,Nt,order='F')
    
    # Modal Re stresses
    uu_k = Phi_u_norm * Phi_u_norm
    vv_k = Phi_v_norm * Phi_v_norm
    uv_k = Phi_u_norm * Phi_v_norm
    
    # Mean velocity gradients
    MU_field = MU.reshape(Nx,Ny,order='F')
    MV_field = MV.reshape(Nx,Ny,order='F')
    
    print("Calculating TKE production terms...")
    # Take gradients
    # CRITICAL: The spacing is now based on the *new* index grid
    # x_all[:,0] is [0, 1, 2, ..., Nx-1] (spacing=1)
    # y_all[0,:] is [0, 1, 2, ..., Ny-1] (spacing=1)
    # These gradients are d/di and d/dj
    dmuxm, dmuym = np.gradient(MU_field, x_all[:,0], y_all[0,:], edge_order=2); dmuym, dmuxm = dmuym, dmuxm;
    dmvxm, dmvym = np.gradient(MV_field, x_all[:,0], y_all[0,:], edge_order=2); dmvym, dmvxm = dmvym, dmvxm;
    
    # Filter out large boundary gradients
    # This filter might be too aggressive or not aggressive enough for d/di, d/dj
    # Let's be cautious and widen it, or remove it, as these are index gradients now.
    # Let's try removing it, as index gradients shouldn't be large.
    # dmuym[np.abs(dmuym) > 30000] = 0
    # dmuxm[np.abs(dmuxm) > 30000] = 0
    # dmvym[np.abs(dmvym) > 30000] = 0
    # dmvxm[np.abs(dmvxm) > 30000] = 0
    
    # Fix random nans to help later
    dmuxm = np.nan_to_num(dmuxm, 0.0)
    dmuym = np.nan_to_num(dmuym, 0.0)
    dmvxm = np.nan_to_num(dmvxm, 0.0)
    dmvym = np.nan_to_num(dmvym, 0.0)
    
    
    # Expand mean gradients along mode dimension
    dmux3 = np.repeat(dmuxm[:, :, np.newaxis], Nt, axis=2)
    dmuy3 = np.repeat(dmuym[:, :, np.newaxis], Nt, axis=2)
    dmvx3 = np.repeat(dmvxm[:, :, np.newaxis], Nt, axis=2)
    dmvy3 = np.repeat(dmvym[:, :, np.newaxis], Nt, axis=2) # <-- Corrected typo here
    
    # TKE production for each mode
    # This production is now in the transformed (i,j) coordinate system
    uup = uu_k * dmux3
    uvp = uv_k * dmuy3
    vup = uv_k * dmvx3
    vvp = vv_k * dmvy3
    
    Plp = uup + uvp + vup + vvp        # production field (Nx, Ny, Nt)
    PlpCum = np.sum(Plp, axis=2)        # cumulative over modes (Nx, Ny)
    
    # Integrate TKE production over domain for each mode
    # The integration is now over d_i and d_j, which is just summation
    # np.trapz(data, x) is sum(data * dx)
    # Since dx=1 here, np.trapz(data, x_all[:,0]) is equivalent to np.sum(data, axis=0)
    # Let's use np.sum for clarity that we are on an index grid.
    
    # Integrate first in i (axis=0), then in j (axis=0 of result)
    UUxp = np.sum(uup, axis=0)   # shape (Ny, Nt)
    UUp  = np.sum(UUxp, axis=0) # shape (Nt,)
    
    UVxp = np.sum(uvp, axis=0)
    UVp  = np.sum(UVxp, axis=0)
    
    VUxp = np.sum(vup, axis=0)
    VUp  = np.sum(VUxp, axis=0)
    
    VVxp = np.sum(vvp, axis=0)
    VVp  = np.sum(VVxp, axis=0)
    
    # Production per mode
    PLp = UUp + UVp + VUp + VVp    # (Nt,)
    
    print("Script finished.")
    print(f"Total TKE Production per mode (first 5): {PLp[:5]}")
    print(f"Cumulative energy (first 5 modes): {cum_lam[:5]}")

    POD['x_all'] = x_all               # KCut blocks x coordinates
    POD['y_all'] = y_all               # KCut blocks y coordinates
    POD['x_plot'] = x_plot             # grid-aligned x coordinates
    POD['y_plot'] = y_plot             # grid-aligned y coordinates
    POD['x_prof'] = x_prof             # blade profile x coordinates
    POD['y_prof'] = y_prof             # blade profile y coordinates
    POD['Phi_u_norm'] = Phi_u_norm     # u POD modes
    POD['Phi_v_norm'] = Phi_v_norm     # v POD modes
    POD['Up'] = Up                     # u prime
    POD['Vp'] = Vp                     # v prime
    POD['uRMS'] = uRMS                 # u RMS field
    POD['vRMS'] = vRMS                 # v RMS field
    POD['MU'] = MU                     # u mean
    POD['MV'] = MV                     # v mean   
    POD['Nx'] = Nx                     # No. points in x direction
    POD['Ny'] = Ny                     # No. points in y direction
    POD['Nt'] = Nt                     # No. temporal snapshots
    POD['Lam'] = Lam                   # energy in each POD mode
    POD['Chi'] = Chi                   # temporal coefficients
    POD['cum_lam'] = cum_lam           # cumulative energies
    POD['PLp'] = PLp                   # TKE production per mode (Nt,)
    POD['UUp'] = UUp                   # TKE production - streamwise (Nt,)
    POD['UVp'] = UVp                   # TKE production - shear (Nt,) 
    POD['VUp'] = VUp                   # TKE production - shear (Nt,) 
    POD['VVp'] = VVp                   # TKE production - normal (Nt,) 
    POD['Plp'] = Plp                   # TKE production field (Nx,Ny,Nt)
    POD['PlpCum'] = PlpCum             # Cumulative TKE production field over all modes (Nx, Ny) 
    POD['uup'] = uup                   # TKE production field - streamwise (Nx,Ny,Nt)
    POD['uvp'] = uvp                   # TKE production field - shear (Nx,Ny,Nt) 
    POD['vup'] = vup                   # TKE production field - shear (Nx,Ny,Nt) 
    POD['vvp'] = vvp                   # TKE production field - normal (Nx,Ny,Nt) 
    POD['time'] = time                 # resolved time from all snapshots
    
    
    if savedata:
        print("\nSaving data for animation script...")
        # save the fluctuating velocity 'Up', the plotting grids, and dimensions
        # Note: 'Up' is large, this file might be several GBs.
        OUTPUTNAME = os.path.join(casename,'animation_data.npz')
        try:
            np.savez(OUTPUTNAME, 
                     Up=Up, 
                     Vp=Vp,
                     Wp=Wp,
                     x_plot=x_plot, 
                     y_plot=y_plot, 
                     Nx=Nx, 
                     Ny=Ny,
                     Phi_u_norm=Phi_u_norm,
                     Phi_v_norm=Phi_v_norm,
                     a_t=Chi)
            print("... Data successfully saved to 'animation_data.npz'.")
        except Exception as e:
            print(f"Error saving data: {e}")
    

    return POD


