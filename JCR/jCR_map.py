import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as jnp
import LibjaxCR as jCR
import LibpltCR as fCR
import LibproCR_jax as pCR
import time
import scipy as sp
import numpy as np
from astropy.io import fits
import healpy as hp

def load_gas():

    # Position of solar system from the gas map (see Soding et al. 2024)
    Rsol=8178.0 # pc

    hdul=fits.open('../samples_densities_hpixr.fits')
    rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # kpc -> Edges of radial bins
    drs=np.diff(rs)*3.086e21 # cm -> Radial bin width for line-of-sight integration
    rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # pc -> Centres of radial bins for interpolating cosmic-ray distribution
    samples_HI=(hdul[3].data).T # cm^-3
    samples_H2=(hdul[4].data).T # cm^-3
    hdul.close()
    ngas=2.0*samples_H2+samples_HI # cm^-3

    N_sample, N_rs, N_pix=samples_HI.shape
    NSIDE=int(np.sqrt(N_pix/12))

    # Angles for all pixels
    thetas, phis=hp.pix2ang(NSIDE,jnp.arange(N_pix),nest=True,lonlat=False)

    # Points for interpolation
    ls=phis[jnp.newaxis, :]
    bs=jnp.pi/2.0-thetas[jnp.newaxis, :]
    rs=rs[:, jnp.newaxis]

    xs=-rs*jnp.cos(ls)*jnp.cos(bs)+Rsol
    ys=-rs*jnp.sin(ls)*jnp.cos(bs)
    zs=rs*jnp.sin(bs)

    points_intr=(jnp.sqrt(xs**2+ys**2),jnp.abs(zs))

    return jnp.array(ngas), jnp.array(drs), points_intr


# Record the starting time
start_time=time.time()

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)
zeta_n=jnp.array(zeta_n)

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

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

# Compute the cross-section from Kafexhiu's code (numpy does not work)
Eg=jnp.logspace(1,2,2)

start_time=time.time()

dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=load_gas()

start_time=time.time()
N=1
for i in range(N):

    # Compute cosmic-ray flux
    jE=jCR.func_jE_YUK04(pars_prop,zeta_n,E,rg,zg)

    # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
    qg_Geant4=1.8*jnp.trapezoid(jE[:,jnp.newaxis,:,:]*dXSdEg_Geant4[:,:,jnp.newaxis,jnp.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 

    # Interpolate gamma-ray emissivity on healpix-r grid as gas
    qg_Geant4_healpixr=jCR.get_healpix_interp(qg_Geant4,rg,zg,points_intr) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity

    # Compute the diffuse emission in all gas samples
    gamma_map=jCR.func_gamma_map(ngas,qg_Geant4_healpixr,drs) # GeV^-1 cm^-2 s^-1

# Record the time finishing computing cosmic-ray map
end_time=time.time()

# Calculate the computing time
print("Computing time in %2d cosmic-ray parameters and %2d gamma-ray energy bin: " % (N,len(Eg)), end_time-start_time, "seconds")

# # Save the gamma-ray maps in a .npz file
# np.savez('gamma_map.npz', Eg=np.array(Eg), gamma_map=np.array(gamma_map))

# fCR.plot_jEp_GAL(np.array(jE),rg,zg)