# Peter Cassidy

import numpy as np

# def grad3d(f, x, y, z):
#     dfi = np.gradient(f, axis=0, edge_order=2)
#     dfj = np.gradient(f, axis=1, edge_order=2)
#     dfk = np.gradient(f, axis=2, edge_order=2)

#     dxi = np.gradient(x, axis=0, edge_order=2)
#     dxj = np.gradient(x, axis=1, edge_order=2)
#     dxk = np.gradient(x, axis=2, edge_order=2)

#     dyi = np.gradient(y, axis=0, edge_order=2)
#     dyj = np.gradient(y, axis=1, edge_order=2)
#     dyk = np.gradient(y, axis=2, edge_order=2)

#     dzi = np.gradient(z, axis=0, edge_order=2)
#     dzj = np.gradient(z, axis=1, edge_order=2)
#     dzk = np.gradient(z, axis=2, edge_order=2)

#     J = (dxi * (dyj * dzk - dyk * dzj) -
#          dxj * (dyi * dzk - dyk * dzi) +
#          dxk * (dyi * dzj - dyj * dzi))

#     dfx_dx = (dfi * (dyj * dzk - dyk * dzj) +
#               dfj * (dyk * dxi - dzi * dxk) +
#               dfk * (dyi * dxj - dyj * dxi)) / J

#     dfx_dy = (dfi * (dzj * dxk - dxj * dzk) +
#               dfj * (dxi * dzk - dzi * dxk) +
#               dfk * (dxj * dzi - dzj * dxi)) / J

#     dfx_dz = (dfi * (dxj * dyk - dyj * dxk) +
#               dfj * (dyi * dxk - dxi * dyk) +
#               dfk * (dxi * dyj - dyi * dxj)) / J

#     return dfx_dx, dfx_dy, dfx_dz

def grad3d(f, x, y, z):
    """
    Compute gradient of scalar field f in curvilinear coordinates.
    Assumes f, x, y, z are 3D arrays with shape (ni, nj, nk).
    Returns: dfdx, dfdy, dfdz
    """
    dfi = np.gradient(f, axis=0, edge_order=2)
    dfj = np.gradient(f, axis=1, edge_order=2)
    dfk = np.gradient(f, axis=2, edge_order=2)

    dxi = np.gradient(x, axis=0, edge_order=2)
    dxj = np.gradient(x, axis=1, edge_order=2)
    dxk = np.gradient(x, axis=2, edge_order=2)

    dyi = np.gradient(y, axis=0, edge_order=2)
    dyj = np.gradient(y, axis=1, edge_order=2)
    dyk = np.gradient(y, axis=2, edge_order=2)

    dzi = np.gradient(z, axis=0, edge_order=2)
    dzj = np.gradient(z, axis=1, edge_order=2)
    dzk = np.gradient(z, axis=2, edge_order=2)

    # Jacobian and inverse metric coefficients
    J = (dxi * (dyj * dzk - dyk * dzj) -
         dxj * (dyi * dzk - dyk * dzi) +
         dxk * (dyi * dzj - dyj * dzi))

    dfdx = (dfi * (dyj * dzk - dyk * dzj) +
            dfj * (dyk * dzi - dyi * dzk) +
            dfk * (dyi * dzj - dyj * dzi)) / J

    dfdy = (dfi * (dzj * dxk - dzk * dxj) +
            dfj * (dzk * dxi - dzi * dxk) +
            dfk * (dzi * dxj - dzj * dxi)) / J

    dfdz = (dfi * (dxj * dyk - dxk * dyj) +
            dfj * (dxk * dyi - dxi * dyk) +
            dfk * (dxi * dyj - dxj * dyi)) / J

    return dfdx, dfdy, dfdz
