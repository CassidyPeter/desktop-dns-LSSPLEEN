# Copyright (c) Andrew Wheeler 2023, University of Cambridge, all rights reserved.
# Modified to include vectorized Q-criterion and Lambda-2 for 3D (extruded z)
# Modified by Peter Cassidy (use regular read_flo.py for typical 2D slice cases)

import os
import numpy as np
from meshing.read_case import *
from .grad import *
from .area import *

def read_flo_v3(casename):

    flo = {}
    path = os.getcwd()

    # ----------------------------
    # Case + constants
    # ----------------------------
    case = read_case(casename)
    blk = case['blk']
    version = case['solver']['version']

    gam     = case['gas']['gamma']
    cp      = case['gas']['cp']
    mu_ref  = case['gas']['mu_ref']
    mu_tref = case['gas']['mu_tref']
    mu_cref = case['gas']['mu_cref']
    pr      = case['gas']['pr']
    cv = cp/gam
    rgas = cp - cv

    nk = case['solver']['nk']

    # Spanwise spacing (uniform, extruded z)
    # Try to get dz from case, otherwise default to 1.0 if nk==1 else 1.0/(nk-1)
    if nk > 1:
        Lz = case['solver']['span']
        if Lz is None:
            dz = 1.0/(nk-1) if nk > 1 else 1.0
        else:
            dz = Lz/(nk-1)

    for ib in range(len(blk)):
    # for ib in [2,3,4,6,8]:

        x = blk[ib]['x']  # (ni, nj)
        y = blk[ib]['y']  # (ni, nj)
        flo[ib] = {}

        ni, nj = np.shape(x)

        # Allocate
        if nk > 1:
            ro = np.zeros((ni, nj, nk))
            ru = np.zeros((ni, nj, nk))
            rv = np.zeros((ni, nj, nk))
            rw = np.zeros((ni, nj, nk))
            Et = np.zeros((ni, nj, nk))
        else:
            ro = np.zeros((ni, nj))
            ru = np.zeros((ni, nj))
            rv = np.zeros((ni, nj))
            rw = np.zeros((ni, nj))
            Et = np.zeros((ni, nj))

        # ----------------------------
        # Read fields
        # ----------------------------
        mut_model = np.zeros((ni, nj, 1))
        if version == 'gpu':
            flow_name = 'flow_' + str(ib + 1)
            rans_name = 'rans_' + str(ib + 1)

            flow_file = os.path.join(path, casename, flow_name)
            rans_file = os.path.join(path, casename, rans_name)

            with open(flow_file, 'rb') as f:
                q = np.fromfile(f, dtype='float64', count=ni * nj * nk * 5)

            if os.path.exists(rans_file):
                with open(rans_file, 'rb') as f:
                    v = np.fromfile(f, dtype='float64', count=ni * nj)
                mut_model = np.reshape(v, (ni, nj, 1), order='F')

            q = np.reshape(q, (5, ni, nj, nk), order='F')  # Fortran order
            ro = q[0, :, :, :]
            ru = q[1, :, :, :]
            rv = q[2, :, :, :]
            rw = q[3, :, :, :]
            Et = q[4, :, :, :]

        # ----------------------------
        # Derived thermodynamic quantities
        # ----------------------------
        u = ru / ro
        v = rv / ro
        w = rw / ro

        kin = 0.5 * (u*u + v*v + w*w)
        p = (gam - 1.0) * (Et - ro * kin)
        T = p / (ro * rgas)
        mu = (mu_ref) * ((mu_tref + mu_cref) / (T + mu_cref)) * ((T / mu_tref)**1.5)
        alpha = np.arctan2(v, u) * (180.0 / np.pi)
        s = cp * np.log(T / 300.0) - rgas * np.log(p / 1e5)
        vel = np.sqrt(u*u + v*v + w*w)
        mach = vel / np.sqrt(gam * rgas * T)
        To = T * (1.0 + 0.5 * (gam - 1.0) * mach*mach)
        po = p * ((To / T)**(gam / (gam - 1.0)))

        # ----------------------------
        # Velocity gradients
        #  - (x,y): curvilinear 2D through provided grad()
        #  - (z): finite differences along k with spacing dz
        # ----------------------------
        dudx = np.zeros_like(ro)
        dudy = np.zeros_like(ro)
        dvdx = np.zeros_like(ro)
        dvdy = np.zeros_like(ro)
        dwdx = np.zeros_like(ro)
        dwdy = np.zeros_like(ro)

        if nk > 1:
            for k in range(nk):
                dudx[:, :, k], dudy[:, :, k] = grad(u[:, :, k], x, y)
                dvdx[:, :, k], dvdy[:, :, k] = grad(v[:, :, k], x, y)
                dwdx[:, :, k], dwdy[:, :, k] = grad(w[:, :, k], x, y)
            dudz = np.gradient(u, dz, axis=2, edge_order=2)
            dvdz = np.gradient(v, dz, axis=2, edge_order=2)
            dwdz = np.gradient(w, dz, axis=2, edge_order=2)
        else:
            # purely 2D case
            dudx[:, :], dudy[:, :] = grad(u, x, y)
            dvdx[:, :], dvdy[:, :] = grad(v, x, y)
            dwdx[:, :], dwdy[:, :] = grad(w, x, y)
            dudz = np.zeros_like(u)
            dvdz = np.zeros_like(v)
            dwdz = np.zeros_like(w)

        # ----------------------------
        # Q-criterion (fully vectorized, no big matrices)
        # S = 0.5 (A + A^T), Ω = 0.5 (A - A^T), A = ∇u
        # ||S||^2 = Sxx^2 + Syy^2 + Szz^2 + 2*(Sxy^2 + Sxz^2 + Syz^2)
        # ||Ω||^2 = 2*(Ωxy^2 + Ωxz^2 + Ωyz^2)
        # Q = 0.5 (||Ω||^2 - ||S||^2)
        # ----------------------------
        Sxx = dudx
        Syy = dvdy
        Szz = dwdz
        Sxy = 0.5 * (dudy + dvdx)
        Sxz = 0.5 * (dudz + dwdx)
        Syz = 0.5 * (dvdz + dwdy)

        Omxy = 0.5 * (dudy - dvdx)
        Omxz = 0.5 * (dudz - dwdx)
        Omyz = 0.5 * (dvdz - dwdy)

        S_norm2 = (Sxx*Sxx + Syy*Syy + Szz*Szz) + 2.0 * (Sxy*Sxy + Sxz*Sxz + Syz*Syz)
        Om_norm2 = 2.0 * (Omxy*Omxy + Omxz*Omxz + Omyz*Omyz)
        Q = 0.5 * (Om_norm2 - S_norm2)

        # ----------------------------
        # Lambda-2 (vectorized in k-chunks to cap memory)
        # Build M = S@S + Ω@Ω (symmetric 3x3) component-wise, then
        # do batched eigvalsh on chunks: returns 3 eigenvalues sorted.
        # λ2 = middle eigenvalue.
        # ----------------------------
        lambda2 = np.empty_like(ro)

        # Components of A = ∇u for convenience
        Axx, Axy, Axz = dudx, dudy, dudz
        Ayx, Ayy, Ayz = dvdx, dvdy, dvdz
        Azx, Azy, Azz = dwdx, dwdy, dwdz

        # S and Ω components already computed above.
        # Compute M = S@S + Ω@Ω without assembling full stacks of A
        # First write S and Ω matrices (conceptually):
        # S = [[Sxx, Sxy, Sxz],
        #      [Sxy, Syy, Syz],
        #      [Sxz, Syz, Szz]]
        # Ω = [[  0, Omxy, Omxz],
        #      [-Omxy,   0, Omyz],
        #      [-Omxz,-Omyz,  0]]

        # Helper to compute M_ij = Σ_k (S_{ik} S_{jk} + Ω_{ik} Ω_{jk})
        def M_components():
            # S^2 components (S_ik * S_kj)
            S2_xx = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz
            S2_yy = Sxy*Sxy + Syy*Syy + Syz*Syz
            S2_zz = Sxz*Sxz + Syz*Syz + Szz*Szz
            S2_xy = Sxx*Sxy + Sxy*Syy + Sxz*Syz
            S2_xz = Sxx*Sxz + Sxy*Syz + Sxz*Szz
            S2_yz = Sxy*Sxz + Syy*Syz + Syz*Szz

            # Ω^2 components (Om_ik * Om_kj)
            # Note: Om_yx = -Omxy, Om_zx = -Omxz, Om_zy = -Omyz
            Om2_xx = 0*0    + Omxy*(-Omxy) + Omxz*(-Omxz)  # = -Omxy*Omxy - Omxz*Omxz
            Om2_yy = (-Omxy)*Omxy + 0*0    + Omyz*(-Omyz)  # = -Omxy*Omxy - Omyz*Omyz
            Om2_zz = (-Omxz)*Omxz + (-Omyz)*Omyz + 0*0    # = -Omxz*Omxz - Omyz*Omyz
            Om2_xy = 0*Omxy + Omxy*0    + Omxz*(-Omyz)  # = -Omxz*Omyz
            Om2_xz = 0*Omxz + Omxy*Omyz + Omxz*0         # = +Omxy*Omyz
            Om2_yz = (-Omxy)*Omxz + 0*Omyz + Omyz*0         # = -Omxy*Omxz

            # M = S^2 + Ω^2
            Mxx = S2_xx + Om2_xx
            Myy = S2_yy + Om2_yy
            Mzz = S2_zz + Om2_zz
            Mxy = S2_xy + Om2_xy
            Mxz = S2_xz + Om2_xz
            Myz = S2_yz + Om2_yz

            return Mxx, Myy, Mzz, Mxy, Mxz, Myz
        # INCORRECT CODE
        # def M_components():
        #     # Diagonals
        #     Mxx = (Sxx*Sxx + Sxy*Sxy + Sxz*Sxz) + (0*0 + Omxy*Omxy + Omxz*Omxz)
        #     Myy = (Sxy*Sxy + Syy*Syy + Syz*Syz) + (Omxy*Omxy + 0*0 + Omyz*Omyz)
        #     Mzz = (Sxz*Sxz + Syz*Syz + Szz*Szz) + (Omxz*Omxz + Omyz*Omyz + 0*0)
        #     # Off-diagonals (symmetric)
        #     Mxy = (Sxx*Sxy + Sxy*Syy + Sxz*Syz) + (0*Omxy + Omxy*0 + Omxz*Omyz)   # = Omxz*Omyz + ...
        #     Mxz = (Sxx*Sxz + Sxy*Syz + Sxz*Szz) + (0*Omxz + Omxy*(-Omyz) + Omxz*0) # = -Omxy*Omyz + ...
        #     Myz = (Sxy*Sxz + Syy*Syz + Syz*Szz) + ((-Omxy)*Omxz + 0*Omyz + Omyz*0) # = -Omxy*Omxz + ...
        #     return Mxx, Myy, Mzz, Mxy, Mxz, Myz

        Mxx, Myy, Mzz, Mxy, Mxz, Myz = M_components()

        # Chunk over k to keep peak memory low while using fast LAPACK underneath
        # Tune chunk size if needed
        max_k_chunk = 16 if nk > 1 else 1
        # max_k_chunk = 64 # increase max_k_chunk if RAM to spare for more speed!
        for k0 in range(0, nk, max_k_chunk):
            k1 = min(nk, k0 + max_k_chunk)

            # assemble symmetric matrices for the chunk: shape (ni, nj, kc, 3, 3)
            kc = k1 - k0
            M = np.empty((ni, nj, kc, 3, 3), dtype=Mxx.dtype)

            # Fill symmetric components
            M[..., 0, 0] = Mxx[:, :, k0:k1]
            M[..., 1, 1] = Myy[:, :, k0:k1]
            M[..., 2, 2] = Mzz[:, :, k0:k1]

            M[..., 0, 1] = M[..., 1, 0] = Mxy[:, :, k0:k1]
            M[..., 0, 2] = M[..., 2, 0] = Mxz[:, :, k0:k1]
            M[..., 1, 2] = M[..., 2, 1] = Myz[:, :, k0:k1]

            # Flatten leading dims for batched eig (N, 3, 3)
            M_flat = M.reshape(-1, 3, 3)
            eigs = np.linalg.eigvalsh(M_flat)  # (N, 3), ascending
            eigs = eigs.reshape(ni, nj, kc, 3)

            # Middle eigenvalue = λ2
            lambda2[:, :, k0:k1] = eigs[:, :, :, 1]

        # ----------------------------
        # Other outputs
        # ----------------------------
        vortz = dvdx - dudy  # 2D vorticity component
        a = area(x, y)

        # Save
        flo[ib]['ro'] = ro
        flo[ib]['ru'] = ru
        flo[ib]['rv'] = rv
        flo[ib]['rw'] = rw
        flo[ib]['Et'] = Et

        flo[ib]['p'] = p
        flo[ib]['u'] = u
        flo[ib]['v'] = v
        flo[ib]['w'] = w
        flo[ib]['T'] = T
        flo[ib]['s'] = s
        flo[ib]['vortz'] = vortz

        flo[ib]['po'] = po
        flo[ib]['To'] = To
        flo[ib]['mach'] = mach
        flo[ib]['mu'] = mu
        flo[ib]['mut_model'] = mut_model
        flo[ib]['area'] = a

        flo[ib]['alpha'] = alpha
        flo[ib]['vel'] = vel

        flo[ib]['Q'] = Q
        flo[ib]['lambda2'] = lambda2

    return flo, blk
