# Peter Cassidy

import numpy as np
import os
from scipy.signal import welch, windows

def process_probe_v2(casename, probe, range_start, export=False):
    for idx in range(len(probe)):
        vel = probe[idx]['vel'][range_start:]
        # vel = probe[idx]['u'][range_start:]
        dt = probe[idx]['time'][1] - probe[idx]['time'][0]

        # Compute integral length scale
        Fs = int(1/(dt)) # sampling frequency
        L = len(vel) # number of samples
        t = np.arange(L)/Fs # Time vector
        T = L * dt # Averaging time
        aFData = np.fft.fft(vel-np.mean(vel))
        n = L // 2 # Half-spectrum
        aFreq = Fs * (np.arange(1,n+1)/n) # Frequency array
        aFMag = np.abs(aFData[0:n]/L) # Single sided normalised magnitude

        # # Energy density spectrum
        Power = np.abs(np.fft.fft(vel-np.mean(vel)))**2/L
        Energy = Power / Fs
        # print(np.sqrt(np.trapz(Energy[1:n+1],aFreq))/np.mean(vel)) # turbulence intensity check from spectrum!

        fs = int(1/(dt)) # sampling frequency
        n = len(vel)
        
        # --- Case 1: Default windowing (MATLAB uses default values when [] is passed)
        f, pxx = welch(vel-np.mean(vel), fs=fs)
        
        # --- Case 2: Hamming window with 50% overlap ---
        window = windows.hamming(int(np.floor(n / 8)))
        noverlap = int(len(window) / 2)
        f2, pxx2 = welch(vel-np.mean(vel), fs=fs, window=window, noverlap=noverlap, nfft=n)
        
        # --- Case 3: Fixed window length of 5000 samples ---
        window = 5000
        f3, pxx3 = welch(vel-np.mean(vel), fs=fs, nperseg=window)

        # Compute integral length scale (macro)
        uprime = np.std(vel)
        uprimesquared = uprime**2
        ubar = np.mean(vel)
        tu = uprime/ubar
        Ef0 = np.mean(Energy[1:100]) # Using index 1 to 99 (Python is 0 indexed)
        integral_length_scale = Ef0 * ubar / (4*uprimesquared)


        # Compute dissipation length scale (micro)
        Y = []
        for i in range(len(aFreq)):
            Y.append(aFreq[i] * aFreq[i] * Energy[i])
        Z = np.trapz(Y, x=aFreq)
        Z *= 2 * np.pi**2 / (ubar**2 * uprimesquared)
        dissipation_length_scale = np.sqrt(1/Z)


        probe[idx]['T'] = T # Averaging time
        probe[idx]['dt'] = dt # probe timestep
        probe[idx]['L'] = L # number of samples
        probe[idx]['ils'] = round(integral_length_scale*1000,2)
        probe[idx]['dls'] = round(dissipation_length_scale*1000,2)
        probe[idx]['tu'] = round(tu*100,2)
        probe[idx]['aFreq'] = aFreq # frequency array
        probe[idx]['Energy'] = Energy[1:n + 1] # Energy density (indexed with half spectrum)
        probe[idx]['f'] = f
        probe[idx]['pxx'] = pxx
        probe[idx]['f2'] = f2
        probe[idx]['pxx2'] = pxx2
        probe[idx]['f3'] = f3
        probe[idx]['pxx3'] = pxx3

        if export:
            output_file = os.path.join(casename, 'ProbeData', 'ProbeData'+str(idx)+'.csv')
            if not(os.path.isdir(os.path.join(casename, 'ProbeData'))):
                os.mkdir(os.path.join(casename, 'ProbeData'))
            np.savetxt(output_file, vel, delimiter=',', header='vel', comments='') # Save to CSV
        

    return probe
