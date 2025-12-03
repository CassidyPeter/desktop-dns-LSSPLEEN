import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# --- Function to compute transition points ---
def compute_transition(s, H):
    """
    Compute transition location based on the max shape factor gradient, and also the midpoint shape factor (mid of laminar and turb).
    Returns (trans_grad, trans_midp)
    """
    # Normalize s
    s_norm = s / s[-1]
    
    # Restrict to mid surface range (avoid LE/TE)
    mask = (s_norm > 0.3) & (s_norm < 0.9)
    s_sub = s_norm[mask]
    H_sub = H[mask]
    
    # Smooth H
    win = 31 if len(s_sub) >= 31 else (len(s_sub)//2)*2+1
    H_smooth = savgol_filter(H_sub, window_length=win, polyorder=3)
    
    # --- Max |dH/ds| transition ---
    dHds = np.gradient(H_smooth, s_sub)
    dHds_smooth = savgol_filter(dHds, 31, 3)
    trans_grad = s_sub[np.nanargmax(np.abs(dHds_smooth))]
    
    # --- Midpoint threshold transition ---
    navg = 20 # average first or last navg points to get lam or turb H
    H_lam = np.median(H_smooth[:navg])
    H_turb = np.median(H_smooth[-navg:])
    H_thresh = 0.5 * (H_lam + H_turb)
    trans_midp = s_sub[np.nanargmin(np.abs(H_smooth - H_thresh))]
    
    return trans_grad, trans_midp

def compute_sep_reatt(s, Cf):
    """
    Compute separation and reattachment locations based on Cf zero-crossings.
    Returns (sep_loc, reatt_loc). If no crossings found, returns 0.
    """
    # Normalize s
    s_norm = s / s[-1]

    # Restrict to analysis region
    mask = (s_norm >= 0.1) & (s_norm <= 0.9)
    s_sub = s_norm[mask]
    Cf_sub = Cf[mask]

    # Smooth Cf to reduce noise (optional)
    win = 31 if len(s_sub) >= 31 else (len(s_sub)//2)*2+1
    Cf_smooth = savgol_filter(Cf_sub, window_length=win, polyorder=3)

    # Identify zero crossings: Cf goes from + to - (sep) and - to + (reatt)
    sign_changes = np.sign(Cf_smooth)
    zero_crossings = np.where(np.diff(sign_changes) != 0)[0]

    # Default: no crossing
    sep_loc, reatt_loc = 0.0, 0.0

    if len(zero_crossings) > 0:
        # Compute crossing points linearly between s[i] and s[i+1]
        for i in zero_crossings:
            if Cf_smooth[i] > 0 and Cf_smooth[i+1] < 0:
                # separation point
                frac = -Cf_smooth[i] / (Cf_smooth[i+1] - Cf_smooth[i])
                sep_loc = s_sub[i] + frac * (s_sub[i+1] - s_sub[i])
            elif Cf_smooth[i] < 0 and Cf_smooth[i+1] > 0:
                # reattachment point
                frac = -Cf_smooth[i] / (Cf_smooth[i+1] - Cf_smooth[i])
                reatt_loc = s_sub[i] + frac * (s_sub[i+1] - s_sub[i])

    return sep_loc, reatt_loc


### Code to develop transition detection
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter, find_peaks

# side = 'ss'
# # side = 'ps'

# s = bl[side]['s'] / bl[side]['s'][-1]
# H = bl[side]['H']

# # restrict to 0.3-0.9s/SL
# mask = (s > 0.3) & (s < 0.9)
# s_sub = s[mask]
# H_sub = H[mask]

# # smoothing parameters
# win = 31 if len(s_sub) >= 31 else (len(s_sub)//2)*2+1
# poly = 3

# H_smooth = savgol_filter(H_sub, window_length=win, polyorder=poly)

# # first and second derivatives using smoothed H
# dHds = np.gradient(H_smooth, s_sub)
# d2Hds2 = np.gradient(dHds, s_sub)

# # Indicator 1: max |dH/ds|
# i_tr_grad = np.nanargmax(np.abs(dHds))
# trans_grad = s_sub[i_tr_grad]

# # Indicator 2: threshold method: use averages of ends to get midpoint of lam and turb values
# navg = 20
# navg = min(navg, len(H_smooth)//4)
# H_lam = np.median(H_smooth[:navg])
# H_turb = np.median(H_smooth[-navg:])
# H_thresh = 0.5*(H_lam + H_turb)
# i_tr_thresh = np.nanargmin(np.abs(H_smooth - H_thresh))
# trans_thresh = s_sub[i_tr_thresh]

# # Indicator 3: inflection point(s): zero crossings of second derivative (basically max |dH/ds|)
# # find indices where sign(d2Hds2) changes
# sign_changes = np.where(np.sign(d2Hds2[:-1]) * np.sign(d2Hds2[1:]) < 0)[0]
# # convert to midpoints
# inflect_idxs = (sign_changes + 0.5).astype(int)
# # if none found, fallback to max curvature magnitude
# if len(inflect_idxs) == 0:
#     i_tr_inflect = np.nanargmax(np.abs(d2Hds2))
# else:
#     # choose the inflection nearest (in s) to the gradient-based transition
#     inflect_positions = s_sub[inflect_idxs]
#     i_choice = np.nanargmin(np.abs(inflect_positions - trans_grad))
#     i_tr_inflect = inflect_idxs[i_choice]

# trans_inflect = float(s_sub[i_tr_inflect])

# # Indicator 4: estimate reattachment: find local maxima in H (separated hump)
# # look for peaks in H_smooth
# peaks, props = find_peaks(H_smooth, prominence=(np.std(H_smooth)*0.5))
# reattach_idx = None
# if len(peaks) > 0:
#     # choose first significant peak upstream of the steep drop (i_tr_grad)
#     upstream_peaks = peaks[peaks < i_tr_grad]
#     if upstream_peaks.size > 0:
#         reattach_idx = upstream_peaks[-1]  # last upstream peak
#     else:
#         # otherwise take largest peak
#         reattach_idx = peaks[np.nanargmax(H_smooth[peaks])]

# if reattach_idx is not None:
#     reattach_s = float(s_sub[reattach_idx])
# else:
#     reattach_s = None

# print("H_lam, H_turb, H_thresh:", round(H_lam,2), round(H_turb,2), round(H_thresh,2))
# print("trans_grad (max |dH/ds|)   = {:.4f}".format(trans_grad))
# print("trans_thresh (H midpoint)  = {:.4f}".format(trans_thresh))
# print("trans_inflect (inflection) = {:.4f}".format(trans_inflect))
# # print("reattachment (H peak)      = {:.4f}".format(reattach_s))

# # Plot
# plt.figure(figsize=(9,5))
# plt.plot(s_sub, H_sub, 'k--', alpha=0.3, label='H raw (subset)')
# plt.plot(s_sub, H_smooth, 'b', lw=2, label='H smoothed')
# plt.plot(s_sub, dHds*0.2 + np.mean(H_smooth), label='scaled dH/ds', alpha=0.8)  # scaled for visualization

# plt.axvline(trans_grad, color='r', linestyle='--', label='trans_grad (max |dH/ds|)')
# plt.axvline(trans_thresh, color='g', linestyle='--', label='trans_thresh (H midpoint)')
# plt.axvline(trans_inflect, color='m', linestyle='--', label='trans_inflect (inflection)')
# if reattach_s is not None:
#     plt.axvline(reattach_s, color='orange', linestyle=':', label='reattachment (H peak upstream)')

# plt.axhline(H_thresh, color='gray', linestyle=':', label='H threshold')
# plt.xlabel('s / SL')
# plt.ylabel('H')
# plt.legend()
# plt.title('Transition using shape factor')
# plt.grid(True)
# plt.show()