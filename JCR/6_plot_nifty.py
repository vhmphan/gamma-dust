# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Demonstration of the non-parametric correlated field model in NIFTy.re

# The Model
import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("text",usetex=True)

import jax.random as jr
from jax import numpy as jnp
import jax.scipy as jsp

import LibjaxCR as jCR

import scipy as sp
import nifty8.re as jft

import healpy as hp
import h5py
import numpy as np
import time
from astropy.io import fits

from healpy.newvisufunc import projview, newprojplot

start_time = time.time()

# Plot gamma-ray sky maps
def plot_gamma_map(gamma_map_theta, gamma_map_mean, dorm):

    fig=plt.figure(figsize=(18, 5))

    projview(
        np.log10(gamma_map_theta[0,0,:]), 
        title=r'Sample mean gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\phi_{\rm fit}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=131
    )

    projview(
        np.log10(gamma_map_mean[0,0,:]), 
        title=r'Diffuse gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\left[\phi_{\rm data}(E_\gamma)\right]\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=132
    )

    projview(
        np.abs(np.log10(gamma_map_theta[0,0,:]/gamma_map_mean[0,0,:])), 
        title=r'Ratio',
        coord=["G"], cmap='magma',
        # min=0.0, max=0.0,
        # cbar_ticks=[0, 0.05, 0.1],
        nest=True, 
        unit=r'$\log_{10}\left[\phi_{\rm fit}(E_\gamma)/\phi_{\rm data}(E_\gamma)\right]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=133
    )

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

    plt.savefig('Results_nifty/fg_gamma-map_QGSJET_%s.png' % dorm, dpi=300)
    plt.close()

def plot_gamma_l(gamma_map_theta, gamma_map_mean, dorl, dorm):

    l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
    l=np.where(l<0,l+360,l)

    if(dorl=='disk'):
        mask=(np.abs(b)<=10.0) 
    else:
        mask=(np.abs(b)>10.0) 

    gamma_map_theta=np.array(gamma_map_theta[0,0,:])
    gamma_map_mean=np.array(gamma_map_mean[0,0,:])

    gamma_map_theta[mask]=np.nan
    gamma_map_mean[mask]=np.nan

    # Load or create a HEALPix map (for example, using a random map here)
    nside=64
    npix=hp.nside2npix(nside)

    # Define the Cartesian grid
    nlon, nlat=720, 360  # Resolution of the Cartesian grid
    lon=np.linspace(-180,180,nlon)  # Longitude from -180 to 180 degrees
    lat=np.linspace(-90,90,nlat)  # Latitude from -90 to 90 degrees
    lon_grid, lat_grid=np.meshgrid(lon,lat)

    # Convert Cartesian coordinates to theta and phi
    theta=np.radians(90-lat_grid)  # Co-latitude in radians
    phi=np.radians(lon_grid)  # Longitude in radians

    # Get HEALPix pixel indices for each (theta, phi)
    healpix_indices=hp.ang2pix(nside, theta, phi, nest=True)

    # Create the new Cartesian skymap
    gamma_map_theta=gamma_map_theta[healpix_indices][:,::-1]
    gamma_map_mean=gamma_map_mean[healpix_indices][:,::-1]

    fs=22

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    ax.plot(lon,np.nansum(gamma_map_mean, axis=0),'k-',linewidth=3,label=r'${\rm Data}$')
    ax.plot(lon,np.nansum(gamma_map_theta, axis=0),'g:',linewidth=3,label=r'${\rm Fit}$')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'$l \,{\rm (degree)}$',fontsize=fs)
    ax.set_ylabel(r'$\int \phi(E_\gamma, l, b){\rm d}b\, ({\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2})$',fontsize=fs)
    for label_axd in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_axd.set_fontsize(fs)
    ax.set_xlim(-180,180)
    if(dorl=='disk'):
        ax.set_ylim(0.0,1.0e-5)
        ax.set_title(r'Gamma-ray intensity integrated over $|b|> 10^\circ$',fontsize=fs)
    else:
        ax.set_ylim(0.0,5.0e-5)
        ax.set_title(r'Gamma-ray intensity integrated over $|b|\leq 10^\circ$',fontsize=fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig("Results_nifty/fg_Ig_lon_%s_%s.png" % (dorl, dorm))
    plt.close()

def plot_gSNR(rG, gSNR_truth, gSNR_sample, dorm):
    # Plotting gSNR and power spectrum
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot mock data
    ax[0].plot(rG, gSNR_truth, c='C5', alpha=1.0, label='Mock data')

    # # Test Bessel expansion
    # num_zeros=100
    # zeta_n=sp.special.jn_zeros(0, num_zeros)

    # for i in range(n_samples):
    #     ax[0].plot(rG, np.sum(q_n_sample[i,:,np.newaxis]*sp.special.j0(zeta_n[:,np.newaxis]*rG/20000.0), axis=0), 'k:')

    # Plot samples and mean 
    for i in range(n_samples):
        ax[0].plot(rG, gSNR_sample[i,:], 'k:', alpha=0.5)

    ax[0].plot(rG, np.mean(gSNR_sample, axis=0), c='red', alpha=1)
    # ax[0].fill_between(rG, np.mean(gSNR_sample, axis=0)+np.std(gSNR_sample, axis=0), np.mean(gSNR_sample, axis=0)-np.std(gSNR_sample, axis=0), color='salmon', alpha=0.2)

    ax[0].set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
    ax[0].set_xlabel(r'$R\, {\rm (pc)}$')

    ax[0].legend(loc='upper right', prop={"size":fs})
    # ax[0].set_xlim(8000.0,8200.0)
    # ax[0].set_ylim(0.0,9.0e-9)
    # ax[0].set_yscale('log')

    # Plot power spectra
    for i in range(n_samples):
        ax[1].plot(spectrum_gSNR_sample[i,:], 'k-')
    ax[1].plot(jnp.mean(spectrum_gSNR_sample, axis=0),'r-')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    fig.tight_layout()
    fig.savefig("Results_nifty/results_gSNR_%s.png" % dorm, dpi=400)

dorm = 'mock_flex10'
n_samples = 12
n_vi_iterations = 3

# Load the .npz file
data = np.load('Results_nifty/gSNR_s%d_b_10_i%d_%s.npz' % (n_samples, n_vi_iterations, dorm))

# Access arrays by their names
rG = data['rG']
gSNR_sample = data['gSNR']
spectrum_gSNR_sample = data['spectrum_gSNR']
gSNR_truth = data['gSNR_truth']
q_n_sample = data['q_n']
gamma_sample = data['gamma']
gamma_truth =data['gamma_truth']

n_samples, _ = gSNR_sample.shape 
print(n_samples)

fs=22

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=100
zeta_n=jnp.array(sp.special.jn_zeros(0, num_zeros))

# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Parameters for injection spectrum
alpha=4.23 # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs
Gam=jCR.func_Gam(alpha)

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=jnp.array([R, L, alpha, xiSNR, u0, Gam])

# Define cosmic-ray grid and diffusion coefficient
E=jnp.logspace(10.0,14.0,81) # eV 

# Define gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
Eg=jnp.logspace(np.log10(13.33521432163324),2,1) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean=jnp.mean(ngas,axis=0)[jnp.newaxis,:,:]

gamma_gSNR=jCR.func_gamma_map_gSNR((rG,jnp.mean(gSNR_sample,axis=0)),pars_prop,zeta_n,dXSdEg_Geant4,ngas_mean,drs,points_intr,E)
plot_gamma_map(gamma_gSNR,gamma_truth,dorm)
plot_gamma_l(gamma_gSNR,gamma_truth,'disk',dorm)
plot_gamma_l(gamma_gSNR,gamma_truth,'local',dorm)
plot_gSNR(rG, gSNR_truth, gSNR_sample, dorm)
print(gamma_gSNR/gamma_truth)

import LibpltCR as pCR
jE_mean, rg, zg=jCR.func_jCR_gSNR((rG,jnp.mean(gSNR_sample,axis=0)),pars_prop,zeta_n,dXSdEg_Geant4,ngas_mean,drs,points_intr,E)
jE_truth, rg, zg=jCR.func_jCR_gSNR((rG,gSNR_truth),pars_prop,zeta_n,dXSdEg_Geant4,ngas_mean,drs,points_intr,E)
NE, Nrh, Nzg = jE_truth.shape
jE_samples=np.zeros((n_samples, NE, Nrh, Nzg))
for i in range(n_samples):
    jE_sample, rg, zg=jCR.func_jCR_gSNR((rG,gSNR_sample[i]),pars_prop,zeta_n,dXSdEg_Geant4,ngas_mean,drs,points_intr,E)
    jE_samples[i,:,:,:]+=jE_sample

pCR.plot_jEp_GAL(jE_mean,rg,zg,'Results_nifty/%s' % dorm)
pCR.plot_jEp_rG(jE_truth,jE_mean,jE_samples,rg,'Results_nifty/%s' % dorm)

hdul=fits.open('NPS.fits')
print(hdul.info())
NPSmask=np.array(hdul[1].data, dtype=np.float64)

fig=plt.figure(figsize=(6, 5))

projview(
    NPSmask, 
    title=r'NPS mask',
    coord=["G"], cmap='magma',
    # min=-8.5, max=-4.5,
    # cbar_ticks=[-8.5, -6.5, -4.5],
    nest=False, 
    # unit=r'$\log_{10}\phi_{\rm fit}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=111
)

plt.savefig('Results_nifty/NPS.png', dpi=300)
plt.close()

print('Runtime: ',time.time()-start_time,'s')