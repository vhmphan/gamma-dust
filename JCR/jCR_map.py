import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as jnp
import LibjaxCR as jCR
import LibpltCR as fCR
import LibppGam as ppG
import LibproCR as pCR
import time
import scipy as sp
import numpy as np
from astropy.io import fits

# Record the starting time
start_time=time.time()

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)
zeta_n=jnp.array(zeta_n)

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Position of solar system from the gas map (see Soding et al. 2024)
Rsol=8178.0 # pc

# Parameters for injection spectrum
alpha=4.23 # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=jnp.array([R, L, alpha, xiSNR, u0])

# Define grid for cosmic-ray distribution
rg=jnp.linspace(0.0,R,501)    # pc
zg=jnp.linspace(0.0,L,41)     # pc
E=jnp.logspace(10.0,14.0,81) # eV 
jE=jCR.func_jE_YUK04(pars_prop,zeta_n,E,rg,zg)

# Record the time finishing computing cosmic-ray distribution
CR_time=time.time()

# # Compute the cross-section from Kafexhiu's code (numpy deos not work)
# Eg=np.logspace(1,2,2)
# dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
# for i in range(len(E)):
#     for j in range(len(Eg)):
#         dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
# qg_Geant4=1.8*sp.integrate.trapezoid(jE[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

# # Load gas density
# hdul=fits.open('../samples_densities_hpixr.fits')
# rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # kpc -> Edges of radial bins
# drs=np.diff(rs)*3.086e21 # cm -> Radial bin width for line-of-sight integration
# rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # pc -> Centres of radial bins for interpolating cosmic-ray distribution
# samples_HI=(hdul[3].data).T # cm^-3
# samples_H2=(hdul[4].data).T # cm^-3
# hdul.close()
# ngas=2.0*samples_H2+samples_HI # cm^-3

# # Interpolate gamma-ray emissivity on healpix-r grid as gas
# N_sample, N_rs, N_pix=samples_HI.shape
# NSIDE=int(np.sqrt(N_pix/12))
# qg_Geant4_healpixr=pCR.get_healpix_interp(qg_Geant4,Eg,rg,zg,rs,NSIDE,Rsol) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

# # Compute the diffuse emission in all gas samples
# ngas=jnp.array(ngas)
# qg_Geant4=jnp.array(qg_Geant4)
# drs=jnp.array(drs)
# gamma_map=jnp.sum(ngas[:,jnp.newaxis,:,:]*qg_Geant4_healpixr[jnp.newaxis,:,:,:]*drs[jnp.newaxis,jnp.newaxis,:,jnp.newaxis],axis=2) # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray map
end_time=time.time()

# Calculate the computing time
elapsed_time_CR=CR_time-start_time
# elapsed_time_gamma=end_time-CR_time

print("Cosmic-ray computing time:                 ", elapsed_time_CR, "seconds")
# print("Gamma-ray computing time in %2d energy bin: " % len(Eg), elapsed_time_gamma, "seconds")

fCR.plot_jEp_GAL(np.array(jE),rg,zg)