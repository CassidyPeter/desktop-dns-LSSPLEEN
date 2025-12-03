import os
import matplotlib.pyplot as plt
# import scienceplots
import timeit
import numpy as np
import scipy.fft
import scipy.signal
import scipy.special
import warnings

from postpro.read_case import *
from postpro.grad import *
from postpro.read_profile import *

def SPOD_v1(casename, nfiles_req, blocks, dt=1.0, window='hamming', weight=None, n_overlap=None, n_fft=None, 
         method='fast', conflvl=0.95, save_fft=False, save_dir='results'):
    """
    Spectral Proper Orthogonal Decomposition (SPOD) in Python.
    
    Parameters
    ----------
    X : numpy.ndarray
        Data matrix. First dimension must be time. 
        Shape: (n_time, n_space) 
    dt : float
        Time step between snapshots.
    window : str, int, or numpy.ndarray
        Window parameter. 
        - If 'hamming', uses Hamming window.
        - If int, uses Hamming window of length `window`.
        - If array, uses that specific window.
    weight : numpy.ndarray, optional
        Spatial inner product weight. Shape must match spatial dimension of X.
        Defaults to uniform weighting (1.0).
    n_overlap : int, optional
        Number of overlapping snapshots between blocks. Defaults to 50% of window length.
    n_fft : int, optional
        Number of snapshots per block (window length). 
        Defaults to 2**floor(log2(nt/10)).
    method : str
        'fast' keeps everything in memory. 
    conflvl : float
        Confidence interval level (0-1). Default 0.95.
        
    Returns
    -------
    L : numpy.ndarray
        Modal energy spectra. Shape: (n_freq, n_blocks)
    P : numpy.ndarray
        SPOD modes. Shape: (n_freq, n_space, n_modes)
    f : numpy.ndarray
        Frequency vector.
    Lc : numpy.ndarray
        Confidence intervals for L. Shape: (n_freq, n_blocks, 2) [Lower, Upper]
    A : numpy.ndarray
        Expansion coefficients. Shape: (n_freq, n_blocks, n_blocks)
    """

    print('Starting SPOD analysis')

    SPOD = {}
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
    
    # UV long skinny matrix
    UV = np.vstack([Up,Vp])
    print("Shape of UV matrix:", UV.shape)






    # Need to fix, X input is first dimension time, then extra dimensions are spatial (1D, 2D...). How to incorporate 2 velocity components? Stack?
    # X = np.transpose(UV)
    # print("Shape of transposed UV matrix:", UV.shape)
                  
    # 1. Dimensions and Preprocessing
    nt, nx = X.shape
    
    # Check for complex data
    is_real_x = np.iscomplexobj(X) == False
    
    # 2. Parse Window and Spectral Parameters
    if n_fft is None:
        if isinstance(window, int):
            n_fft = window
        elif isinstance(window, np.ndarray):
            n_fft = len(window)
        else:
            # Default heuristic from MATLAB code
            n_fft = 2**int(np.floor(np.log2(nt/10)))
            
    if n_overlap is None:
        n_overlap = int(np.floor(n_fft / 2))
        
    # Create Window vector
    if isinstance(window, str):
        if window.lower() == 'hamming':
            win_weight = scipy.signal.windows.hamming(n_fft)
    elif isinstance(window, int):
         win_weight = scipy.signal.windows.hamming(n_fft)
    elif isinstance(window, np.ndarray):
        win_weight = window
        if len(win_weight) != n_fft:
            raise ValueError("Provided window length does not match n_fft")
    else:
        win_weight = scipy.signal.windows.hamming(n_fft)
        
    # Normalize window to unit energy/power (matching MATLAB code)
    win_weight = win_weight / np.sqrt(np.sum(win_weight**2))
    n_tapers = 1 # Multitaper not fully implemented in this basic port, defaulting to 1
    
    # Determine number of blocks
    n_blocks = int(np.floor((nt - n_overlap) / (n_fft - n_overlap)))
    
    # 3. Weights
    if weight is None:
        weight = np.ones(nx)
    else:
        if weight.size != nx:
            raise ValueError(f"Weight size {weight.size} does not match spatial dim {nx}")
        weight = weight.flatten()

    # 4. Frequency Axis
    if is_real_x:
        # One-sided spectrum for real data
        f = np.arange(0, int(np.ceil(n_fft/2)) + 1) / n_fft / dt
        n_freq = len(f)
    else:
        # Two-sided spectrum
        f = np.arange(0, n_fft) / n_fft / dt
        if n_fft % 2 == 0:
            f[int(n_fft/2):] = f[int(n_fft/2):] - 1/dt
        else:
            f[int((n_fft+1)/2):] = f[int((n_fft+1)/2):] - 1/dt
        n_freq = len(f)

    # 5. Calculate Temporal DFT (The "Q" Matrix)
    # We will store Q_hat in memory: (n_freq, n_space, n_blocks)
    print(f"Calculating Temporal DFT. Blocks: {n_blocks}, FFT Size: {n_fft}")
    
    Q_hat = np.zeros((n_freq, nx, n_blocks), dtype=np.complex128)
    
    # Remove global mean (optional, but standard practice in the MATLAB script if blockwise not set)
    # Here we assume user has pre-processed or wants global mean removed.
    x_mean = np.mean(X, axis=0)
    
    for i_blk in range(n_blocks):
        # Get time indices
        offset = min(i_blk * (n_fft - n_overlap) + n_fft, nt) - n_fft
        time_idx = np.arange(n_fft) + offset
        
        # Extract block
        # Note: data is (Time, Space)
        blk = X[time_idx, :] - x_mean
        
        # Windowing (Broadcasting window across space)
        # win_weight is (n_fft,), blk is (n_fft, nx)
        blk_win = blk * win_weight[:, np.newaxis]
        
        # FFT
        # MATLAB loads column-wise, Python is row-major, but we act on time axis (0)
        blk_hat = np.fft.fft(blk_win, axis=0) * np.sqrt(dt) # Scaled by sqrt(dt) per MATLAB code
        
        # Slice frequencies
        if is_real_x:
            # Take positive half
            blk_hat = blk_hat[:n_freq, :]
        else:
            blk_hat = blk_hat[:n_freq, :]
            
        # Store (Transpose to Space x Block for easier SPOD later? 
        # MATLAB stores Q_hat as (Freq, Space, Block). We will match that.
        Q_hat[:, :, i_blk] = blk_hat

    # 6. SPOD Calculation (Loop over frequencies)
    print("Calculating SPOD modes...")
    
    L = np.zeros((n_freq, n_blocks))
    P = np.zeros((n_freq, nx, n_blocks), dtype=np.complex128)
    A = np.zeros((n_freq, n_blocks, n_blocks), dtype=np.complex128)
    
    # Number of independent realizations
    n_indep = n_blocks # Assuming no multitaper
    
    # Loop frequencies
    for i_freq in range(n_freq):
        # Q_hat_f: (Space, Block)
        Q_hat_f = Q_hat[i_freq, :, :]
        
        # Weighted CSD Matrix M = Q^H * W * Q
        # Optimization: apply weight to one Q first
        # (Space, Block)^H @ ((Space, 1) * (Space, Block))
        # (Block, Space) @ (Space, Block) -> (Block, Block)
        
        M = Q_hat_f.conj().T @ (Q_hat_f * weight[:, np.newaxis]) / n_indep
        
        # Eigenvalue decomposition
        # M is Hermitian, so we can use eigh
        Lambda, Theta = np.linalg.eigh(M)
        
        # Sort descending (eigh returns ascending)
        idx = np.argsort(Lambda)[::-1]
        Lambda = Lambda[idx]
        Theta = Theta[:, idx]
        
        # Calculate Modes: Psi = Q * Theta * Lambda^(-1/2)
        # (Space, Block) @ (Block, Block) @ (Block, Block diagonal)
        Psi = (Q_hat_f @ Theta) * (1.0 / np.sqrt(Lambda * n_indep))
        
        # Store
        P[i_freq, :, :] = Psi
        L[i_freq, :] = np.abs(Lambda)
        
        # Expansion Coefficients
        # A = sqrt(n_blocks * Lambda) * Theta'
        A[i_freq, :, :] = np.diag(np.sqrt(n_blocks * Lambda)) @ Theta.conj().T

        # Adjust for one-sided spectrum energy
        if is_real_x:
            if i_freq != 0 and i_freq != (n_freq - 1):
                L[i_freq, :] = 2 * L[i_freq, :]

    # 7. Confidence Intervals (Chi-Squared)
    Lc = np.zeros((n_freq, n_blocks, 2))
    if conflvl > 0:
        # Degrees of freedom = 2 * n_blocks (for complex Gaussian)
        dof = 2 * n_blocks
        chi2_lower = scipy.special.gammaincinv(n_blocks, conflvl) * 2 # Approx mapping to MATLAB's use
        # A simpler standard chi2 usage:
        # Lower bound: 2*n_blks / chi2_inv(1-alpha)
        # MATLAB uses gammaincinv which is inverse incomplete gamma.
        # Let's stick to standard Chi2 distribution from scipy.stats for clarity implies:
        from scipy.stats import chi2
        
        xi2_upper = chi2.ppf(1 - (1-conflvl)/2, dof) # This might vary slightly from MATLAB implementation
        xi2_lower = chi2.ppf((1-conflvl)/2, dof)
        
        # Using MATLAB exact formula translation:
        # xi2_upper = 2 * gammaincinv(1-conf, nBlks)
        # Python gammaincinv matches MATLAB's parameter order roughly but check docs.
        # Let's trust the chi2 ppf logic as robust standard.
        
        # To match MATLAB exactly:
        xi2_upper_m = 2 * scipy.special.gammaincinv(n_blocks, 1 - conflvl)
        xi2_lower_m = 2 * scipy.special.gammaincinv(n_blocks, conflvl)
        
        Lc[:, :, 0] = L * 2 * n_indep / xi2_lower_m
        Lc[:, :, 1] = L * 2 * n_indep / xi2_upper_m
        
    print("Script finished.")

    SPOD['x_all'] = x_all               # KCut blocks x coordinates
    SPOD['y_all'] = y_all               # KCut blocks y coordinates
    SPOD['x_plot'] = x_plot             # grid-aligned x coordinates
    SPOD['y_plot'] = y_plot             # grid-aligned y coordinates
    SPOD['Nx'] = Nx                     # No. points in x direction
    SPOD['Ny'] = Ny                     # No. points in y direction
    SPOD['Nt'] = nt                     # No. temporal snapshots
    SPOD['L'] = L                       # 
    SPOD['P'] = P                       # 
    SPOD['f'] = f                       # 
    SPOD['Lc'] = Lc                     # 
    SPOD['A'] = A                       # 
    

    return SPOD



