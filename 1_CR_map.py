import numpy as np
import scipy as sp
import LibppGam as ppG
import LibproCR as pCR
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

mp=pCR.mp # eV -> Proton mass

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)

# Record the starting time
start_time = time.time()

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

# Open the FITS file and get info on the data
hdul=fits.open('samples_densities_hpixr.fits')
print(hdul.info())

# Access data and header of the primary HDU
rs=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins
drs=np.diff(rs)*3.086e21
rs=(hdul[1].data)['radial pixel centres'].astype(np.float64)*1.0e3 # Edges of radial bins
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()

N_sample, N_rs, N_pix=samples_HI.shape
NSIDE=int(np.sqrt(N_pix/12))

ngas=np.mean(2.0*samples_H2+samples_HI,axis=0) # cm^-3
qg_Geant4_healpixr=pCR.get_healpix_interp(qg_Geant4,Eg,rg,zg,rs,NSIDE) # GeV^-1 s^-1
gamma_map=np.sum(ngas[np.newaxis,:,:]*qg_Geant4_healpixr*drs[np.newaxis,:,np.newaxis],axis=1) # GeV^-1 cm^-2 s^-1

# # Get pixel coordinates to mask part of the map since these maps are valid only at high lattitude
# print(NSIDE)
# l, b=hp.pixelfunc.pix2ang(NSIDE, np.arange(N_pix), lonlat=True, nest=True)
# l=np.where(l<0,l+360,l)

# # Mask the disk since dust map is only up to 1.25 kpc
# mask=(np.abs(b)<=8.0) 

projview(
    np.log10(gamma_map[0,:]), 
    title=r'Gamma-ray intensity $E_\gamma=%.1f$ GeV' % Eg[0],
    coord=["G"], cmap='magma',
    min=-8.5, max=-4.5,
    nest=True, 
    unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma-map.png', dpi=150)
plt.close()

# Record the ending time
end_time=time.time()

# Calculate the elapsed time
elapsed_time=end_time-start_time

print("Elapsed time:", elapsed_time, "seconds")

