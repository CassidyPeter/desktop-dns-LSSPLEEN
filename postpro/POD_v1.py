import os
import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
import timeit

from postpro.read_case import *
from postpro.grad import *
from postpro.read_profile import *

def POD_v1(casename,nfiles,blocks):
    ### POD analysis of raw KCut blocks (not transformed to be grid-aligned)
    print('Starting POD analysis')
    
    flo = {}
    POD = {}
        
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)
    
    # unpack case
    blk = case['blk']
      
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
    
    cax = 0.06301079
    x_prof,y_prof,pitch,stag=read_profile('geom/LS_SPLEEN_PROFILE_10k.txt',True)
    
    # read kslice file
    kfile = os.path.join(path,casename,'kslice_time.txt')
    ktime = np.loadtxt(kfile,dtype={'names': ('ncut', 'time', 'k'),'formats': ('i', 'f', 'i')})
    
    nfmax = len(ktime['ncut'])
    
    if(len(nfiles)==1):
       if (nfiles[0] > nfmax) or (nfiles[0]==0):
           nfile = ktime['ncut']
           time =  ktime['time']
       elif nfiles[0] < 0 :
           nfnow = np.abs(nfiles)-1
           nfile = ktime['ncut'][nfnow]
           time =  ktime['time'][nfnow]
       else:    
           nfile = ktime['ncut'][-nfiles[0]:]
           time  = ktime['time'][-nfiles[0]:]
       
    else:
       nfile = ktime['ncut'][nfiles[0]-1:nfiles[1]]
       time =  ktime['time'][nfiles[0]-1:nfiles[1]]
    
    nfiles = len(time)
    nt = 0
    
    x_all = None # nIxnJ (where nI is block 4 nI + block6 nI)
    y_all = None # nIxnJ (where nI is block 4 nI + block6 nI)
    u_all = [] # (nIxnJ)xnT (where nI and nJ have been made long skinny, nT is number of kcuts)
    v_all = [] # (nIxnJ)xnT (where nI and nJ have been made long skinny, nT is number of kcuts)
    
    for idx, ncut in enumerate(nfile):  
        start_ncut = timeit.default_timer()
        nt = nt + 1
    
        x_blocks = []
        y_blocks = []
        u_blocks = []
        v_blocks = []
      
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
            
            f = open(flow_file,'rb')
            
            q   = np.fromfile(f,dtype='float64',count=ni*nj*5)
            
            f.close()
                        
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
                x  = x[::-1,:]
                y  = y[::-1,:]
                u  = u[::-1,:]
                v  = v[::-1,:]
                w  = w[::-1,:]
    
            
            x_blocks.append(x)
            y_blocks.append(y)
            u_blocks.append(u)
            v_blocks.append(v)
    
        # --- combine blocks along i-direction ---
        x_combined = np.concatenate(x_blocks, axis=0)
        y_combined = np.concatenate(y_blocks, axis=0)
        u_combined = np.concatenate(u_blocks, axis=0)
        v_combined = np.concatenate(v_blocks, axis=0)
            
        if idx == 0:  # save geometry once
            x_all = x_combined
            y_all = y_combined
    
        # reshape into long-skinny column
        u_col = u_combined.reshape(-1,1,order='F')
        v_col = v_combined.reshape(-1,1,order='F')
        
        u_all.append(u_col)
        v_all.append(v_col)
    print('Read in KCut files successfully')
    
    # stack into arrays: (nI*nJ) x ncuts
    u_all = np.hstack(u_all)
    v_all = np.hstack(v_all)
    
    # print("x_all:", x_all.shape)
    # print("y_all:", y_all.shape)
    # print("u_all:", u_all.shape)
    # print("v_all:", v_all.shape)
    
    Nx = x_all.shape[0]
    Ny = x_all.shape[1]
    Nt = u_all.shape[1]
    NxNy = Nx*Ny
    
    
    MU = np.mean(u_all,axis=1,keepdims=True)
    MV = np.mean(v_all,axis=1,keepdims=True)
    
    Up = u_all - MU
    Vp = v_all - MV
    
    # UV
    UV = np.vstack([Up,Vp])
    print('Performing POD')
    # Correlation matrix
    C = UV.T @ UV
    
    # Eigen decomposition
    Lam, Chi = np.linalg.eig(C)
    
    # Sort eigenvalues/modes descending
    idx = np.argsort(Lam)[::-1]
    Lam = Lam[idx]
    Chi = Chi[:,idx]
    
    # POD spatial modes
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
    
    # Normalise POD consistently with Re stresses
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
    
    # Take gradients
    dmuxm, dmuym = np.gradient(MU_field, x_all[:,0], y_all[0,:], edge_order=2); dmuym, dmuxm = dmuym, dmuxm;
    dmvxm, dmvym = np.gradient(MV_field, x_all[:,0], y_all[0,:], edge_order=2); dmvym, dmvxm = dmvym, dmvxm;
    
    # Filter out large boundary gradients
    dmuym[np.abs(dmuym) > 30000] = 0
    dmuxm[np.abs(dmuxm) > 30000] = 0
    dmvym[np.abs(dmvym) > 30000] = 0
    dmvxm[np.abs(dmvxm) > 30000] = 0
    
    # Fix random nans to help later
    dmuxm = np.nan_to_num(dmuxm, 0.0)
    dmvxm = np.nan_to_num(dmvxm, 0.0)
    
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
    
    Plp = uup + uvp + vup + vvp         # production field (Nx, Ny, Nt)
    PlpCum = np.sum(Plp, axis=2)        # cumulative over modes (Nx, Ny)
    
    # Integrate TKE production over domain for each mode
    # Integrate first in x, then in y
    UUxp = np.trapz(uup, x_all[:,0], axis=0)   # shape (Ny, Nt)
    UUp  = np.trapz(UUxp, y_all[0,:], axis=0) # shape (Nt,)
    
    UVxp = np.trapz(uvp, x_all[:,0], axis=0)
    UVp  = np.trapz(UVxp, y_all[0,:], axis=0)
    
    VUxp = np.trapz(vup, x_all[:,0], axis=0)
    VUp  = np.trapz(VUxp, y_all[0,:], axis=0)
    
    VVxp = np.trapz(vvp, x_all[:,0], axis=0)
    VVp  = np.trapz(VVxp, y_all[0,:], axis=0)
    
    # Production per mode
    PLp = UUp + UVp + VUp + VVp   # (Nt,) # should this be negative as the definition is also negative?

    POD['x_all'] = x_all               # KCut blocks x coordinates
    POD['y_all'] = y_all               # KCut blocks y coordinates
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
    

    

    return POD


## Backup plotting code for OG blade POD
## Sum of modes
# # %matplotlib qt
# %matplotlib inline

# limit = 500

# fig1, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
# ax.plot(POD['Lam'][0:limit-1]/np.sum(POD['Lam']),'k.-',linewidth=1)
# ax.set_ylabel('POD Eigenvalue');
# ax.grid(which='both')
# # ax.set_xlabel('Mode k')

# ax1.plot(POD['cum_lam'][0:limit-1],'k.-',linewidth=1)
# # ax1.set_xlabel('Mode k'); 
# ax1.set_ylabel('Cumulative energy');
# # ax1.set_xscale('log')
# ax1.grid(which='both')

# ax2.plot(np.cumsum(POD['PLp'][0:limit-1]),'k.-', linewidth=1)
# ax2.set_xlabel('Mode k'); ax2.set_ylabel('Cumulative PTKE');
# # ax2.set_xscale('log')
# ax2.grid(which='both')

# plt.savefig((casename + r"\ModesSum.png"),dpi=750) 


## POD modes fields
# # %matplotlib qt
# %matplotlib inline

# modes = [1,2,3,4]
# # modes = [1,4,6,8]
# range = [-0.5, 0.5]

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True, sharey=True)
# plt.set_cmap('seismic')
# for idx, ax in enumerate(axes.flat):
#     im = ax.pcolormesh(x_all, y_all, Phi_u_norm[:,:,modes[idx]-1].reshape(Nx,Ny,order='F'), shading='gouraud')
#     im.set_clim(range)
#     # ax.plot(xprof*cax,yprof*cax,'-k')
#     ax.set_title(r'$\phi_u \ Mode:$' + str(modes[idx]))
#     # ax.axis('equal')
#     ax.axis([0.035, 0.07, -0.037, 0.003]) # large SS
#     # ax.axis([0.035, 0.055, -0.014, 0.002]) # focused LSB
#     if idx in [2,3]:
#         ax.set_xlabel('x [m]')
#     if idx in [0,2]:
#         ax.set_ylabel('y [m]')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.ax.set_title(r'$\phi_u$', pad=10)

# plt.show()
# # plt.savefig((casename + r"\Phiu.png"),dpi=750) 


## Field reconstruction
# Low-rank reconstruction
# # %matplotlib qt
# %matplotlib inline

# # Define the cumulative mode counts for reconstruction
# reconstructions = [1, 5, 100]
# clim = [-1, 1]

# # Time snapshot for comparison
# t_idx = 1000

# # Temporal coefficients
# # a_t = Chi * np.sqrt(Lam)
# a_t = Chi

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True, sharey=True)
# plt.set_cmap('seismic')

# ax = axes.flat[0]
# u_orig = Up[:, t_idx].reshape(Nx, Ny, order='F')
# im = ax.pcolormesh(x_all, y_all, u_orig, shading='gouraud')
# im.set_clim(clim)
# ax.plot(xprof*cax, yprof*cax, '-k')
# ax.set_title(f'Original field (t={t_idx})')
# ax.axis([0.035, 0.055, -0.014, 0.002])
# ax.set_ylabel('y [m]')


# for i, n_modes in enumerate(reconstructions):
#     ax = axes.flat[i+1]
    
#     # POD reconstruction: U' ≈ Φ_u_norm * a_t
#     # Compute fluctuation field at chosen time
#     # u_recon_fluc = np.sum(Phi_u_norm[:, :, :n_modes] * a_t[t_idx, :n_modes], axis=2)
#     u_recon_fluc = np.tensordot(Phi_u_norm[:, :, :n_modes], a_t[t_idx, :n_modes], axes=([2], [0]))

#     # Purely for plotting - normalised to max of original field
#     u_recon_vis = u_recon_fluc / np.max(np.abs(u_recon_fluc)) * np.max(np.abs(u_orig))


    
#     im = ax.pcolormesh(x_all, y_all, u_recon_vis, shading='gouraud')
#     im.set_clim(clim)
#     # im.set_clim(-0.05,0.05)
#     ax.plot(xprof*cax, yprof*cax, '-k')
#     ax.set_title(f'First {n_modes} modes')
#     ax.axis([0.035, 0.055, -0.014, 0.002])
    
#     # Axis labels
#     if i in [1,2]:
#         ax.set_xlabel('x [m]')
#     if i == 1:
#         ax.set_ylabel('y [m]')

# # Shared colorbar
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.ax.set_title(r"$u'$", pad=10)

# plt.show()
# # plt.savefig((casename + r"\Phiu_recon.png"), dpi=750)



## PTKE
# # %matplotlib qt
# %matplotlib inline

# modes = [1,2,3,4]
# # modes = [1,4,6,8]
# Uout = 17.5
# range = [-round(10/Uout**3,3), round(10/Uout**3,3)]
# range = [-0.04, 0.04]

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True, sharey=True)

# for idx, ax in enumerate(axes.flat):
#     im = ax.pcolormesh(x_all, y_all, Plp[:,:,modes[idx]-1].reshape(Nx,Ny,order='F')/Uout**3, shading='gouraud')
#     im.set_clim(range)
#     ax.plot(xprof*cax,yprof*cax,'-k')
#     ax.set_title(r'$P_{TKE} \ Mode:$' + str(modes[idx]))
#     # ax.axis('equal')
#     # ax.axis([0.035, 0.07, -0.037, 0.003]) # large SS
#     ax.axis([0.035, 0.055, -0.014, 0.002]) # focused LSB
#     if idx in [2,3]:
#         ax.set_xlabel('x [m]')
#     if idx in [0,2]:
#         ax.set_ylabel('y [m]')
    
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.ax.set_title(r'$P_{TKE}/U_{out}^3$', pad=10)

# plt.show()
# # plt.savefig((casename + r"\PTKEAll.png"),dpi=750) 



## Overall loss terms
# # %matplotlib qt
# %matplotlib inline

# dx = x_all[1,0] - x_all[0,0] # spacing in x
# dy = y_all[0,1] - y_all[0,0] # spacing in y

# # Spatial cumulative energy transfer
# spatial_sum = np.sum(uup + uvp + vup + vvp, axis=(0,1))
# spatial_integral = spatial_sum * dx * dy
# spatial_cumulative = np.cumsum(spatial_integral)

# # Modewise cumulative energy transfer
# modewise_cumulative = np.cumsum(PLp)

# plt.figure(1)

# # How much energy transfer have I captured by integrating up to mode k in physical space?
# plt.plot(spatial_cumulative, linewidth=2, label='Spatial cumulative energy transfer')

# # How much energy transfer have I captured by including mode k in POD mode space?
# plt.plot(modewise_cumulative, linewidth=2, label='Modewise cumulative energy transfer')
# plt.xscale('log')
# plt.grid(which='both')
# plt.xlabel('Mode k'); plt.ylabel('Energy');
# plt.legend(loc='upper left')
# plt.show()

# LOSS = np.sum(PLp)


## POD coefficients
# # %matplotlib qt
# %matplotlib inline

# modes = [0,1,4,5]
# # modes = [0,1,2,3]

# fig1, (ax, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True)
# ax.plot(Chi[:,modes[0]]); ax.set_title('Mode ' + str(modes[0]+1));
# ax1.plot(Chi[:,modes[1]]); ax1.set_title('Mode ' + str(modes[1]+1));
# ax2.plot(Chi[:,modes[2]]); ax2.set_title('Mode ' + str(modes[2]+1));
# ax3.plot(Chi[:,modes[3]]); ax3.set_title('Mode ' + str(modes[3]+1));
# ax3.set_xlabel('Snapshot')



