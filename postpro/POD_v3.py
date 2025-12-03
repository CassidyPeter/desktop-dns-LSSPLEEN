import os
import numpy as np
import matplotlib.pyplot as plt
import timeit

from postpro.read_case import *
from postpro.grad import *
from postpro.read_profile import *

def POD_v3(casename, nfiles_req, blocks, dt, savedata=True, transform_grid=True):
    ### POD analysis of raw KCut blocks
    # transform_grid: If True, aligns velocities to grid and calculates s/SL. 
    #                 If False, keeps Cartesian global frame.
    
    print(f'Starting POD analysis (Transform Grid: {transform_grid})')

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
    
    nfiles = len(time) 
    nt = 0
    
    x_all = None 
    y_all = None 
    u_all = [] 
    v_all = [] 
    w_all = [] 
    print(f"Reading {nfiles} snapshots...")
    for idx, ncut in enumerate(nfile):
        start_ncut = timeit.default_timer()
        nt = nt + 1
    
        x_blocks = []
        y_blocks = []
        u_blocks = []
        v_blocks = []
        w_blocks = [] 
    
        for ib in blocks:
    
            x = blk[ib]['x']*cax
            y = blk[ib]['y']*cax
    
            flo[ib] = {}
    
            ni,nj = np.shape(blk[ib]['x'])
            
            q = np.zeros([ni*nj*5]) # Pre-allocate logic replaced by direct read below to match your snippet
    
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
    
            q = np.reshape(q,[5,ni,nj],order='F') 
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
            w_blocks.append(w) 
    
        # --- combine blocks along i-direction ---
        if not x_blocks: 
            print(f"No data loaded for snapshot {ncut}, skipping.")
            nt -= 1
            continue
            
        x_combined = np.concatenate(x_blocks, axis=0)
        y_combined = np.concatenate(y_blocks, axis=0)
        u_combined = np.concatenate(u_blocks, axis=0)
        v_combined = np.concatenate(v_blocks, axis=0)
        w_combined = np.concatenate(w_blocks, axis=0) 
    
        if idx == 0: 
            x_all = x_combined
            y_all = y_combined
    
        # reshape into long-skinny column
        u_col = u_combined.reshape(-1,1,order='F')
        v_col = v_combined.reshape(-1,1,order='F')
        w_col = w_combined.reshape(-1,1,order='F') 
    
        u_all.append(u_col)
        v_all.append(v_col)
        w_all.append(w_col) 
    
    # Check if any data was loaded
    if not u_all:
        print("Error: No data was successfully loaded. Exiting.")
        exit()
    
    # stack into arrays: (nI*nJ) x ncuts
    u_all = np.hstack(u_all)
    v_all = np.hstack(v_all)
    w_all = np.hstack(w_all) 
    
    print("Data loading complete.")

    # Save original physical grid before any potential overwriting
    x_all_original = np.copy(x_all)
    y_all_original = np.copy(y_all)
    
    # Get grid dimensions
    Ni, Nj = x_all.shape
    NxNy = Ni * Nj
    Nt = u_all.shape[1]

    # ==============================================================================
    # === GRID TRANSFORMATION SECTION ===
    # ==============================================================================
    
    if transform_grid:
        print("Calculating grid transformation (Grid Aligned)...")
        
        # Calculate grid partial derivatives
        # Using more accurate central differences (except boundaries)
        dy_di, dy_dj = np.gradient(y_all, edge_order=2)
        dx_di, dx_dj = np.gradient(x_all, edge_order=2)
        
        # # Define basis vectors based on user's description
        # Nx = dy_dj
        # Ny = -dx_dj
        # Tx = Ny 
        # Ty = -Nx
        
        # # Normalization magnitude
        # Mag = np.sqrt(Tx**2 + Ty**2)
        # Mag[Mag < 1e-12] = 1.0 

        # --- FIX: Calculate Tangent Basis (Tx, Ty) directly from i-lines ---
        # This ensures alignment with the surface regardless of orthogonality
        Mag_T = np.sqrt(dx_di**2 + dy_di**2)
        Tx = dx_di / Mag_T
        Ty = dy_di / Mag_T
    
        # --- Calculate Normal Basis (Nx, Ny) directly from j-lines ---
        # This points "out" from the wall (assuming j increases away from wall)
        Mag_N = np.sqrt(dx_dj**2 + dy_dj**2)
        Nx = dx_dj / Mag_N
        Ny = dy_dj / Mag_N
    
        # Check for near-zero magnitudes (singularities)
        Tx[Mag_T < 1e-12] = 1.0; Ty[Mag_T < 1e-12] = 0.0
        Nx[Mag_N < 1e-12] = 0.0; Ny[Mag_N < 1e-12] = 1.0
    
        # Optional: Orthogonalize N relative to T? 
        # For separation bubble analysis, it is usually better to strictly 
        # align T with the grid lines (i-direction) so u_trans is exactly "flow parallel to surface".
        # The code above does exactly that.
        # --- END OF FIX

        # # Normalize basis vectors (shape (Ni, Nj))
        # Tx_norm = Tx / Mag
        # Ty_norm = Ty / Mag
        # Nx_norm = Nx / Mag
        # Ny_norm = Ny / Mag
        Tx_norm = Tx
        Ty_norm = Ty
        Nx_norm = Nx
        Ny_norm = Ny

        del Nx, Ny # free up so they can be used for # of grid points in i and j directions (confusing I know)
        
        print("Applying transformation to all velocity snapshots...")
        
        # Reshape velocities to 3D fields (Ni, Nj, Nt)
        u_field = u_all.reshape(Ni, Nj, Nt, order='F')
        v_field = v_all.reshape(Ni, Nj, Nt, order='F')
        w_field = w_all.reshape(Ni, Nj, Nt, order='F') 
        
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
        w_all = w_transformed.reshape(NxNy, Nt, order='F') 
        
        print("Creating new rectangular 'box' grid for calculation...")
        
        # Create new "box" coordinates (simple index grid)
        i_coords = np.arange(Ni)
        j_coords = np.arange(Nj)
        x_all_new, y_all_new = np.meshgrid(i_coords, j_coords, indexing='ij')
        
        # Overwrite x_all and y_all with the new index grid
        # The gradient calls later will now use this new grid
        x_all = x_all_new.astype(float)
        y_all = y_all_new.astype(float)
        
        # --- Create physical 'plotting' grid (x_plot, y_plot) ---
        print("Creating new physical 'plotting' grid (x_plot, y_plot)...")
        
        # --- 1. Calculate y_plot (distance from wall) ---
        dx_j = np.diff(x_all_original, axis=1) # Shape (Ni, Nj-1)
        dy_j = np.diff(y_all_original, axis=1) # Shape (Ni, Nj-1)
        ds_j = np.sqrt(dx_j**2 + dy_j**2)      
        
        ds_j_rev = ds_j[:, ::-1] 
        cum_dist_rev = np.cumsum(ds_j_rev, axis=1) 
        
        y_plot_rev = np.zeros((Ni, Nj))
        y_plot_rev[:, 1:] = cum_dist_rev 
        y_plot = y_plot_rev[:, ::-1]
        
        # --- 2. Calculate x_plot (normalized arc length s/SL) ---
        SL_total_unscaled = 1.491030896539091 
        SL_total_scaled = SL_total_unscaled * cax 
        
        x_wall = x_all_original[:, -1] # Wall coords (at j=Nj-1)
        y_wall = y_all_original[:, -1]
        
        dx_wall = np.diff(x_wall)
        dy_wall = np.diff(y_wall)
        ds_wall = np.sqrt(dx_wall**2 + dy_wall**2)
        
        s_block = np.zeros(Ni)
        s_block[1:] = np.cumsum(ds_wall)
        
        S_block_total = s_block[-1]
        s_offset = SL_total_scaled - S_block_total
        s_abs_1d = s_block + s_offset
        s_over_SL_1d = s_abs_1d / SL_total_scaled
        
        # Tile to create 2D (Ni, Nj) array for plotting
        x_plot = np.tile(s_over_SL_1d[:, np.newaxis], (1, Nj))
        
        print("... 'x_plot' and 'y_plot' created for plotting (s/SL and Dist).")

    else:
        # --- NO TRANSFORMATION ---
        print("Skipping Grid Transformation. Keeping data in Domain Frame.")
        
        # Velocities remain in u_all, v_all as loaded (Cartesian)
        
        # x_all and y_all remain physical (Cartesian)
        # Note: This means TKE gradients later will use physical units if grid is rectilinear,
        # or might be inaccurate if grid is curvilinear and np.gradient expects 1D axes.
        
        # Set plotting grids to physical coordinates
        x_plot = x_all_original
        y_plot = y_all_original
        
        print("... 'x_plot' and 'y_plot' set to physical coordinates.")

    
    print("Preparation complete. Starting POD...")
    
    # ==============================================================================
    # === POD Processing ===
    # ==============================================================================
    
    Nx = x_all.shape[0] # This is Ni
    Ny = x_all.shape[1] # This is Nj
    
    MU = np.mean(u_all,axis=1,keepdims=True)
    MV = np.mean(v_all,axis=1,keepdims=True)
    MW = np.mean(w_all,axis=1,keepdims=True) 
    
    Up = u_all - MU
    Vp = v_all - MV
    Wp = w_all - MW
    
    # UV
    UV = np.vstack([Up,Vp])
    
    # Correlation matrix
    print("Calculating Correlation matrix C...")
    C = UV.T @ UV
    
    # Eigen decomposition
    print("Calculating Eigendecomposition...")
    Lam, Chi = np.linalg.eig(C)
    
    Lam = np.real(Lam) 
    Chi = np.real(Chi) 
    
    idx = np.argsort(Lam)[::-1]
    Lam = Lam[idx]
    Chi = Chi[:,idx]
    
    # POD spatial modes
    print("Calculating POD spatial modes...")
    Phi = UV @ Chi
    Phi_u = Phi[:NxNy, :]
    Phi_v = Phi[NxNy:, :]
    
    # Re stresses
    uu_real = np.mean(Up * Up, axis=1).reshape(Nx, Ny, order='F')
    vv_real = np.mean(Vp * Vp, axis=1).reshape(Nx, Ny, order='F')
    uv_real = np.mean(Up * Vp, axis=1).reshape(Nx, Ny, order='F')
    
    # RMS fields
    uRMS = np.sqrt(np.mean(Up**2, axis=1)).reshape(Nx,Ny, order='F')
    vRMS = np.sqrt(np.mean(Vp**2, axis=1)).reshape(Nx,Ny, order='F')
    
    cum_lam = np.cumsum(Lam)/np.sum(Lam)
    
    # Normalise POD consistently 
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
    
    # Handle Gradients depending on grid type
    if transform_grid:
        # We are on the index "box" grid (0,1,2..). x_all[:,0] is 1D array of integers.
        dmuxm, dmuym = np.gradient(MU_field, x_all[:,0], y_all[0,:], edge_order=2); dmuym, dmuxm = dmuym, dmuxm;
        dmvxm, dmvym = np.gradient(MV_field, x_all[:,0], y_all[0,:], edge_order=2); dmvym, dmvxm = dmvym, dmvxm;
    else:
        # We are on physical grid. Since it's likely curvilinear, passing 1D slices 
        # (x_all[:,0]) to np.gradient is technically inaccurate for internal points 
        # if the grid is skewed/curved.
        # Fallback: We calculate gradients w.r.t Index, then normally we would multiply by metrics.
        # For simplicity in this script, we will calculate gradients w.r.t index to avoid crashing,
        # but warn the user that TKE terms are in "per-index" units, not "per-meter".
        print("Warning: TKE gradients calculated w.r.t grid index because grid is curvilinear.")
        # Create dummy index grid just for gradient function stability
        xi_dummy = np.arange(Nx)
        eta_dummy = np.arange(Ny)
        dmuxm, dmuym = np.gradient(MU_field, xi_dummy, eta_dummy, edge_order=2); dmuym, dmuxm = dmuym, dmuxm;
        dmvxm, dmvym = np.gradient(MV_field, xi_dummy, eta_dummy, edge_order=2); dmvym, dmvxm = dmvym, dmvxm;
    
    
    dmuxm = np.nan_to_num(dmuxm, 0.0)
    dmuym = np.nan_to_num(dmuym, 0.0)
    dmvxm = np.nan_to_num(dmvxm, 0.0)
    dmvym = np.nan_to_num(dmvym, 0.0)
    
    # Expand mean gradients along mode dimension
    dmux3 = np.repeat(dmuxm[:, :, np.newaxis], Nt, axis=2)
    dmuy3 = np.repeat(dmuym[:, :, np.newaxis], Nt, axis=2)
    dmvx3 = np.repeat(dmvxm[:, :, np.newaxis], Nt, axis=2)
    dmvy3 = np.repeat(dmvym[:, :, np.newaxis], Nt, axis=2) 
    
    # TKE production for each mode
    uup = uu_k * dmux3
    uvp = uv_k * dmuy3
    vup = uv_k * dmvx3
    vvp = vv_k * dmvy3
    
    Plp = uup + uvp + vup + vvp        
    PlpCum = np.sum(Plp, axis=2)        
    
    # Integrate TKE production
    # Note: sum assumes dx=1 (index space).
    UUxp = np.sum(uup, axis=0)  
    UUp  = np.sum(UUxp, axis=0) 
    
    UVxp = np.sum(uvp, axis=0)
    UVp  = np.sum(UVxp, axis=0)
    
    VUxp = np.sum(vup, axis=0)
    VUp  = np.sum(VUxp, axis=0)
    
    VVxp = np.sum(vvp, axis=0)
    VVp  = np.sum(VVxp, axis=0)
    
    PLp = UUp + UVp + VUp + VVp    
    
    print("Script finished.")
    
    POD['x_all'] = x_all               
    POD['y_all'] = y_all               
    POD['x_plot'] = x_plot             
    POD['y_plot'] = y_plot             
    POD['x_prof'] = x_prof             
    POD['y_prof'] = y_prof             
    POD['Phi_u_norm'] = Phi_u_norm     
    POD['Phi_v_norm'] = Phi_v_norm     
    POD['Up'] = Up                     
    POD['Vp'] = Vp                     
    POD['uRMS'] = uRMS                 
    POD['vRMS'] = vRMS                 
    POD['MU'] = MU                     
    POD['MV'] = MV                     
    POD['Nx'] = Nx                     
    POD['Ny'] = Ny                     
    POD['Nt'] = Nt                     
    POD['Lam'] = Lam                   
    POD['Chi'] = Chi                   
    POD['cum_lam'] = cum_lam           
    POD['PLp'] = PLp                   
    POD['UUp'] = UUp                   
    POD['UVp'] = UVp                   
    POD['VUp'] = VUp                   
    POD['VVp'] = VVp                   
    POD['Plp'] = Plp                   
    POD['PlpCum'] = PlpCum             
    POD['uup'] = uup                   
    POD['uvp'] = uvp                   
    POD['vup'] = vup                   
    POD['vvp'] = vvp                   
    POD['time'] = time                 
    
    
    if savedata:
        print("\nSaving data for animation script...")
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