   
---

# Desktop DNS

---

Desktop DNS is a toolkit for pre- and post-processing Direct Numerical Simulations of turbomachinery flows. Starting with a blade profile, the toolkit enables users to mesh, simulate and analyze the blade.  
Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


Small adjustments made to meshing for high-turning LPTs, such as the low-speed scaled version of the SPLEEN LPT linear cascade as found in Pastorino, G. et al. (2025) ‘Aerodynamic scaling of a high-speed low-pressure turbine cascade, optimized for boundary layer similarity in a low-speed test rig’, in Turbomachinery, Fluid Dynamics and Thermodynamics. 16th European Turbomachinery Conference (ETC16), Hannover, Germany.

Additional post-processing code: finds non-equilibrium regions of the boundary layer (mismatch in turbulence production and dissipation), calculates Reynolds stress wake profiles, finds separation/reattachment/transition(shape factor) locations, calculates law of the wall profiles, and applies proper orthogonal decomposition (POD) to midspan temporal snapshots (Kcuts) with grid-aligned frame transformation. Animations can also be created of fluctuating components and POD reconstructions using _n_ modes. Spectrograms are computed for the FFT amplitude of temporal coefficients.
 
Heavily WIP and may not readily extend to other geometries

---
