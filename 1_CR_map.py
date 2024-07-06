import numpy as np
import scipy as sp
import LibppGam as ppG
import LibproCR as pCR
import matplotlib.pyplot as plt
import time
from astropy.io import fits

mp=pCR.mp # eV -> Proton mass

# Record the starting time
start_time=time.time()

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Parameters for injection spectrum
alpha=4.23 # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=np.array([R, L, alpha, xiSNR, u0])

# Compute the coefficients
q_n=pCR.compute_coefficients(pCR.func_gSNR_YUK04,zeta_n,R)

# Define grid for cosmic-ray distribution
rg=np.linspace(0.0,R,501)    # pc
zg=np.linspace(0.0,L,41)     # pc
E=np.logspace(10.0,14.0,81) # eV 
fE=pCR.func_fE(pars_prop,zeta_n,q_n,E,rg,zg) # eV^-1 cm^-3
fE[fE<0.0]=0.0

# Comput the cosmic-ray flux
vp=np.sqrt((E+mp)**2-mp**2)*3.0e10/(E+mp)
jE=fE*vp[:,np.newaxis,np.newaxis]*1.0e9 # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray distribution
CR_time=time.time()

# Compute the cross-section from Kafexhiu's code (numpy deos not work)
Eg=np.logspace(1,2,11)
dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
for i in range(len(E)):
    for j in range(len(Eg)):
        dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
qg_Geant4=1.8*sp.integrate.trapezoid(jE[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

pCR.plot_gSNR(zeta_n,q_n,rg,R)
pCR.plot_jE_p_LOC(pars_prop,zeta_n,q_n)
pCR.plot_jE_rz(fE,rg,zg)
pCR.plot_emissivity_LOC(qg_Geant4,Eg,rg,zg)

# Load gas density
hdul=fits.open('samples_densities_hpixr.fits')
print(hdul.info())

rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins
drs=np.diff(rs)*3.086e21
rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # Edges of radial bins
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()
ngas=2.0*samples_H2+samples_HI # cm^-3

# Interpolate cosmic-ray distribution on healpix-r grid as gas
N_sample, N_rs, N_pix=samples_HI.shape
NSIDE=int(np.sqrt(N_pix/12))
qg_Geant4_healpixr=pCR.get_healpix_interp(qg_Geant4,Eg,rg,zg,rs,NSIDE) # GeV^-1 s^-1

# Compute the diffuse emission in all gas samples
gamma_map=np.sum(ngas[:,np.newaxis,:,:]*qg_Geant4_healpixr[np.newaxis,:,:,:]*drs[np.newaxis,np.newaxis,:,np.newaxis],axis=2) # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray map
end_time=time.time()

# Calculate the elapsed time
elapsed_time_CR=CR_time-start_time
elapsed_time_gamma=end_time-CR_time

print("Cosmic-ray computing time:                 ", elapsed_time_CR, "seconds")
print("Gamma-ray computing time in %d energy bin: " % len(Eg), elapsed_time_gamma, "seconds")

# Save the gamma-ray maps in a .npz file
np.savez('gamma_map.npz', Eg=Eg, gamma_map=gamma_map)

