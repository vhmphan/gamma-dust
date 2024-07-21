# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Demonstration of the non-parametric correlated field model in NIFTy.re

# The Model
# import jax
import os
os.environ['JAX_ENABLE_X64'] = 'True'
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

mp = 938.272e6 # eV -> Proton mass

# Load diffuse gamma-ray map from Platz et al. 2023
with h5py.File('energy_bins.hdf5', 'r') as file:
    print("Keys: %s" % file.keys())
    Eg_data=file['geom_avg_bin_energy'][:]
    Eg_data_lower=file['lower_bin_boundaries'][:]
    Eg_data_upper=file['upper_bin_boundaries'][:]
    
dEg_data=Eg_data_upper-Eg_data_lower

with h5py.File('I_dust.hdf5', 'r') as file:
    print("Keys: %s" % file['stats'].keys())
    gamma_map_mean=file['stats']['mean'][:]
    gamma_map_std=file['stats']['standard deviation'][:]

gamma_map_mean*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_mean=hp.ud_grade(gamma_map_mean[5,:], nside_out=64)
gamma_map_mean=hp.reorder(gamma_map_mean, r2n=True)
gamma_map_mean=gamma_map_mean[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1
gamma_map_mean=jnp.array(gamma_map_mean)

gamma_map_std*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_std=hp.ud_grade(gamma_map_std[5,:], nside_out=64)
gamma_map_std=hp.reorder(gamma_map_std, r2n=True)
gamma_map_std=gamma_map_std[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1
gamma_map_std=jnp.array(gamma_map_mean)

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=jnp.array(sp.special.jn_zeros(0, num_zeros))

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

# Define cosmic-ray and gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
E=jnp.logspace(10.0,14.0,81) # eV 
# Eg=jnp.logspace(1,2,2) # GeV
Eg=jnp.logspace(np.log10(13.33521432163324),2,1) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean=jnp.mean(ngas,axis=0)[jnp.newaxis,:,:]

seed = 42
key = jr.PRNGKey(seed)

R_SN = 15000.0 # pc -> SNR distribution extends up to R_SN

dims = (300, ) # Number of spatial grid points in the reconstructed gSNR 

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-6, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))

class Signal(jft.Model):
    def __init__(self, correlated_field, scaling, width=10.0):
        self.cf = correlated_field
        self.scaling = scaling
        x_ = jnp.linspace(-dims[0]//2-1, dims[0]//2, dims[0])
        self.kernel = jnp.exp(-pow(x_/width, 2)/2)
        self.kernel /= jnp.sum(self.kernel)
        self.rG = jnp.linspace(0.0, R_SN, dims[0])
        self.zeta = zeta_n
        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    def func_gSNR(self, x):
        return self.scaling(x) * jnp.exp(self.cf(x) + jnp.log(jCR.func_gSNR_CAB98(self.rG)))  

    def func_gamma_map(self, x):

        # Transport parameters
        R=pars_prop[0] # pc
        L=pars_prop[1] # pc
        alpha=pars_prop[2] 
        xiSNR=pars_prop[3]
        u0=pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

        rg=jnp.linspace(0.0,R,501)    # pc
        zg=jnp.linspace(0.0,L,41)     # pc
        p=jnp.sqrt((E+mp)**2-mp**2)  # eV
        vp=p/(E+mp)

        zeta_n = self.zeta

        # Diffusion coefficient
        pb=312.0e9
        Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1 # pc^2/yr

        # Spatial distribution of sources
        r_int=jnp.linspace(0.0,R,200000)
        fr_int=jnp.interp(r_int, self.rG, self.func_gSNR(x), right=0.0)

        j0_n=jCR.j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R)
        q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
        q_n*=(2.0/(R**2*(jCR.j1(zeta_n)**2))) # pc^-2

        # Injection spectrum of sources
        xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
        xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
        x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
        Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)

        RSNR=0.03 # yr^-1 -> SNR rate
        ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
        QE=(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)
        QE*=RSNR*vp*3.0e10

        Diff=Diff[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
        z=zg[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
        r=rg[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
        zeta_n=zeta_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
        q_n=q_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]

        Sn=jnp.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
        fEn=jCR.j0(zeta_n*r/R)*q_n*jnp.exp(u0*z/(2.0*Diff))*jnp.sinh(Sn*(L-z)/2.0)/(jnp.sinh(Sn*L/2.0)*(u0+Sn*Diff*(jnp.cosh(Sn*L/2.0)/jnp.sinh(Sn*L/2.0))))
        fE=jnp.sum(fEn,axis=0) # eV^-1 pc^-3
        fE=jnp.where(fE<0.0,0.0,fE)

        jE=fE*QE[:,jnp.newaxis,jnp.newaxis]*1.0e9/(3.086e18)**3 # GeV^-1 cm^-2 s^-1

        # Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
        qg_Geant4=1.88*jnp.trapezoid(jE[:,jnp.newaxis,:,:]*dXSdEg_Geant4[:,:,jnp.newaxis,jnp.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 -> Enhancement factor 1.88 from Kachelriess et al. 2014

        # Interpolate gamma-ray emissivity on healpix-r grid as gas
        qg_Geant4_healpixr=jCR.get_healpix_interp(qg_Geant4,rg,zg,points_intr) # GeV^-1 s^-1 -> Interpolate gamma-ray emissivity
        # points = (rg, zg)
        # N_rs, N_pix=points_intr[0].shape
        # N_E, _, _ = qg_Geant4.shape 
        # qg_Geant4_healpixr = jnp.zeros((N_E, N_rs, N_pix))
        # for j in range(N_E):
        #     interpolator = jsp.interpolate.RegularGridInterpolator(points, qg_Geant4[j, :, :], method='linear', bounds_error=False, fill_value=0.0)
        #     qg_Geant4_healpixr = qg_Geant4_healpixr.at[j, :, :].set(interpolator(points_intr))

        # Compute the diffuse emission in all gas samples
        gamma_map=jCR.func_gamma_map(ngas_mean,qg_Geant4_healpixr,drs) # GeV^-1 cm^-2 s^-1
        # gamma_map=jnp.sum(ngas_mean[:,jnp.newaxis,:,:]*qg_Geant4_healpixr[jnp.newaxis,:,:,:]*drs[jnp.newaxis,jnp.newaxis,:,jnp.newaxis],axis=2) # GeV^-1 cm^-2 s^-1

        return gamma_map

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        # return self.func_gSNR(x)[::int(dims[0]/Ndata)]
        return self.func_gamma_map(x)

signal_response = Signal(correlated_field, scaling)

# This defines the Likelihood in Bayes' law. "Amend" glues your forward model to the input
lh = jft.Gaussian(gamma_map_mean, noise_cov_inv=1.0/gamma_map_std**2).amend(signal_response)

# Now lets run the main inference scheme:
n_vi_iterations = 1
delta = 1e-4
n_samples = 5

key, k_i, k_o = jr.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="Results_nifty",
    resume=False,
)

rG=jnp.linspace(0,R_SN,dims[0])
gSNR_sample = jnp.array(tuple(signal_response.func_gSNR(s) for s in samples))
spectrum_gSNR = jnp.array(tuple(cfm.amplitude(s)[1:] for s in samples))

# Testing Bessel expansion on one of the sample
num_zeros=200
zeta_n=jnp.array(sp.special.jn_zeros(0, num_zeros))
r_int=jnp.linspace(0.0,R_SN,200000)
fr_int=jnp.interp(r_int, rG, gSNR_sample[0,:])

j0_n=jCR.j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R)
q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
q_n*=(2.0/(R**2*(jCR.j1(zeta_n)**2))) # pc^-2

gSNR=jnp.sum(q_n[:,jnp.newaxis]*jCR.j0(zeta_n[:,jnp.newaxis]*rG[jnp.newaxis,:]/R),axis=0)

# Plotting gSNR and power spectrum
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Plot mock data
# ax[0].plot(rG_data, data, c='C5', alpha=1.0, label='Case 1998')

# Plot samples and mean 
for i in range(n_samples):
    ax[0].plot(rG, gSNR_sample[i,:], c='C1', alpha=0.2, label='Samples')
ax[0].plot(rG, gSNR_sample[0,:], c='C3', alpha=1)

# Plot Bessel expansion version
ax[0].plot(rG,gSNR, 'k--')

ax[0].set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
ax[0].set_xlabel(r'$R\, {\rm (pc)}$')

# Plot power spectra
for i in range(n_samples):
    ax[1].plot(spectrum_gSNR[i,:], 'k-')
ax[1].plot(jnp.mean(spectrum_gSNR, axis=0),'r-')

ax[1].set_xscale('log')
ax[1].set_yscale('log')

fig.tight_layout()
fig.savefig("results_gSNR.png", dpi=400)

# Save the gamma-ray maps in a .npz file
np.savez('gamma_map.npz', rG=np.array(rG), gSNR=np.array(gSNR_sample))
