import numpy as np
import scipy as sp
import LibppGam as ppG
import LibproCR as pCR
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
import h5py

# Open the FITS file and get info on the data
hdul=fits.open('samples_densities_hpixr.fits')
print(hdul.info())

# Access data 
r=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins
dr=np.diff(r)*3.086e21 # cm -> Bin width for the integration along the line of sight
r_center=(hdul[1].data)['radial pixel centres'].astype(np.float64) # Edges of radial bins
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()

# Plot the mean column density map
NHImap=np.sum(samples_HI*dr[np.newaxis,:,np.newaxis],axis=1)
NH2map=np.sum(samples_H2*dr[np.newaxis,:,np.newaxis],axis=1)
NHmap=np.mean(2.0*NH2map+NHImap, axis=0)

# Load diffuse gamma-ray map from Platz et al. 2023
with h5py.File('JCR/energy_bins.hdf5', 'r') as file:
    print("Keys: %s" % file.keys())
    Eg_data=file['geom_avg_bin_energy'][:]
    Eg_data_lower=file['lower_bin_boundaries'][:]
    Eg_data_upper=file['upper_bin_boundaries'][:]
    
dEg_data=Eg_data_upper-Eg_data_lower

with h5py.File('JCR/I_dust.hdf5', 'r') as file:
    print("Keys: %s" % file['stats'].keys())
    gamma_map_mean=file['stats']['mean'][:]
    gamma_map_std=file['stats']['standard deviation'][:]

gamma_map_mean*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_data=np.zeros((len(Eg_data),12*64*64))
for i in range(len(Eg_data)):
    gamma_map_data[i,:]=hp.ud_grade(gamma_map_mean[i,:],nside_out=64)
    gamma_map_data[i,:]=hp.reorder(gamma_map_data[i,:],r2n=True)

l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
l=np.where(l<0,l+360,l)

mask=(np.abs(b)<=20.0) 
qg_map=gamma_map_data/NHmap[np.newaxis,:]
qg_map[:,mask]=np.nan
# qg_map_XCO=gamma_map_data/NHmap_XCO[np.newaxis,:]
# qg_map_XCO[:,mask]=np.nan

qg_loc_mean=np.nanmean(qg_map,axis=1)/(4.0*np.pi)
qg_loc_std=np.nanstd(qg_map,axis=1)/(4.0*np.pi)

# qg_loc_mean_XCO=np.nanmean(qg_map_XCO,axis=1)/(4.0*np.pi)
# qg_loc_std_XCO=np.nanstd(qg_map_XCO,axis=1)/(4.0*np.pi)

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros=150
zeta_n=sp.special.jn_zeros(0, num_zeros)

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
pars_prop=np.array([R, L, alpha, xiSNR, u0])

# Compute the coefficients
q_n=pCR.compute_coefficients(pCR.func_gSNR_YUK04,zeta_n,R)

# Define grid for cosmic-ray distribution
rg=np.linspace(0.0,R,501)    # pc
zg=np.linspace(0.0,L,41)     # pc
E=np.logspace(9.0,14.0,101) # eV 

# Compute the cross-section from Kafexhiu's code (numpy deos not work)
Eg=np.logspace(0,3,31)
dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
dXSdEg_Pythia=np.zeros((len(E),len(Eg))) 
dXSdEg_SIBYLL=np.zeros((len(E),len(Eg))) 
dXSdEg_QGSJET=np.zeros((len(E),len(Eg))) 
for i in range(len(E)):
    for j in range(len(Eg)):
        dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
        dXSdEg_Pythia[i,j]=ppG.dsigma_dEgamma_Pythia8(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
        dXSdEg_SIBYLL[i,j]=ppG.dsigma_dEgamma_SIBYLL(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV
        dXSdEg_QGSJET[i,j]=ppG.dsigma_dEgamma_QGSJET(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
jE_loc=pCR.func_jE(pars_prop,zeta_n,q_n,E,np.array([Rsol]),np.array([0.0]))/(4.0*np.pi)
qg_Geant4_loc=1.88*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
qg_Pythia_loc=1.88*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_Pythia[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
qg_SIBYLL_loc=1.88*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_SIBYLL[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1
qg_QGSJET_loc=1.88*sp.integrate.trapezoid(jE_loc[:,np.newaxis,:,:]*dXSdEg_QGSJET[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1

fs=22

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

ax.errorbar(Eg_data,Eg_data**2.8*qg_loc_mean,xerr=dEg_data/2.0,yerr=Eg_data**2.8*qg_loc_std,fmt='o',color='black',label=r'$b> 20^\circ\,,\, X_{\rm CO}=2\times 10^{20} \,{\rm K^{-1}\,(km/s)^{-1}\,cm^{-2}}$')
ax.plot(Eg,Eg**2.8*qg_Geant4_loc[:,0,0],color='orange',linewidth=3,label=r'${\rm Geant\, 4}$')
ax.plot(Eg,Eg**2.8*qg_Pythia_loc[:,0,0],'r--',linewidth=3,label=r'${\rm Pythia\, 8}$')
ax.plot(Eg,Eg**2.8*qg_SIBYLL_loc[:,0,0],'g:',linewidth=3,label=r'${\rm SIBYLL}$')
ax.plot(Eg,Eg**2.8*qg_QGSJET_loc[:,0,0],'m-.',linewidth=3,label=r'${\rm QGSJET}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$E \,{\rm (GeV)}$',fontsize=fs)
ax.set_ylabel(r'$E_\gamma^{2.8}\,q_\gamma(E_\gamma)\, {\rm (GeV^{1.8}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
for label_axd in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_axd.set_fontsize(fs)
ax.set_xlim(1.0,1.0e3)
ax.set_ylim(1.0e-28,1.0e-26)
ax.set_title(r'{\rm Local emissivity}' % rg[np.abs(rg-Rsol)==np.amin(np.abs(rg-Rsol))], fontsize=fs)
ax.legend(loc='lower left', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig("fg_emissivity.png")
plt.close()

print(qg_loc_std/qg_loc_mean)