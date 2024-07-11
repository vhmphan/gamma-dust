import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as jnp
from jax import grad
import LibjaxCR as jCR
import LibpltCR as fCR
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from healpy.newvisufunc import projview, newprojplot

def plot_gamma_map(i, gamma_map_theta, gamma_map):
    # Plot gamma-ray sky maps
    fig=plt.figure(figsize=(18, 5))

    projview(
        np.log10(gamma_map_theta[0,0,:]), 
        title=r'Fit gamma-ray map at $E_\gamma=10$ GeV',
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
        np.log10(gamma_map[0,0,:]), 
        title=r'Mock data gamma-ray map at $E_\gamma=10$ GeV',
        coord=["G"], cmap='magma',
        min=-8.5, max=-4.5,
        cbar_ticks=[-8.5, -6.5, -4.5],
        nest=True, 
        unit=r'$\log_{10}\left[\phi_{\rm data}(E_\gamma)\right]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide",
        sub=132
    )

    projview(
        np.log10(gamma_map_theta[0,0,:]/gamma_map[0,0,:]), 
        title=r'Ratio',
        coord=["G"], cmap='magma',
        # min=-8.5, max=-4.5,
        # cbar_ticks=[-8.5, -6.5, -4.5],
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

    plt.savefig('fg_gamma-map_%d.png' % i, dpi=300)
    plt.close()

fs=22

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

# Define cosmic-ray and gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
E=jnp.logspace(10.0,14.0,81) # eV 
Eg=jnp.logspace(1,2,2) # GeV
dXSdEg_Geant4=jCR.func_dXSdEg(E*1.0e-9,Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr=jCR.load_gas('../samples_densities_hpixr.fits')

# Load data of gamma-ray map
data=np.load('gamma_map.npz')
gamma_map_data=jnp.array(data['gamma_map'])

ngas_test=ngas[0:1,:,:]
gamma_map_data_test=gamma_map_data[0:1,:,:]

r_data=jnp.linspace(0,12,100)
gSNR_data=jCR.func_gSNR_YUK04(r_data*1.0e3)

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

ax.plot(r_data,gSNR_data,label='Original',lw=2)

N=200
theta=jnp.array([1.0e-9,1.5,3.0])
lr=0.05*theta/jnp.abs(grad(jCR.loss_func_gamma_map)(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E,gamma_map_data))
for i in range(N):
    theta=jCR.update_gamma_map(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas_test,drs,points_intr,E,gamma_map_data_test,lr)
    if((i%20==0) or (i==N-1)):
        gamma_map_theta=jCR.func_gamma_map_fit(theta,pars_prop,zeta_n,dXSdEg_Geant4,ngas,drs,points_intr,E)
        plot_gamma_map(i,gamma_map_theta,gamma_map_data_test)
        print('i=%d -> A_fit=' %i,theta[0],'B_fit=',theta[1],'C_fit=',theta[2])
        ax.plot(r_data, jCR.func_gSNR_fit(theta,zeta_n,R*1.0e-3,r_data), label='i=%d' % i, linestyle='--')

print('A_org=',1.0/5.95828e+8,'B_org=',1.64,'C_org=',4.01)

ax.set_xlabel(r'$r\, ({\rm kpc})$',fontsize=fs)
ax.set_ylabel(r'$g_{\rm SNR}(r)\,({\rm pc^{-2}})$', fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig('fg_gSNR_gamma.png')
plt.close()

print(ngas[0:1,:,:].shape)

print('Runtime:',time.time()-start_time,'seconds')