import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import LibppGam as ppG
import scipy as sp

fs=22

# Function for interpolation
def log_interp1d(xx, yy, kind='linear'):

    logx=np.log10(xx)
    logy=np.log10(yy)
    lin_interp=sp.interpolate.interp1d(logx,logy,kind=kind)
    log_interp=lambda zz: np.power(10.0,lin_interp(np.log10(zz)))

    return log_interp

def plot_NHmap(NHmap):
    NHmap_mean=np.mean(NHmap,axis=0)
    NHmap_std=np.std(NHmap,axis=0)
    nsample=(NHmap.shape)[0]

    for i in range(nsample):
        projview(
            np.log10(NHmap[i,:]), 
            title=r'Column density sample %d' % i,
            coord=["G"], cmap='magma',
            min=19, max=23,
            nest=True, 
            unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
            graticule=True, graticule_labels=True, 
            # xlabel=r'longitude (deg)',
            # ylabel=r'latitude (deg)',
            projection_type="mollweide"
        )
        plt.savefig('fg_NH_los_%d.png' % i, dpi=150)
        plt.close()

    projview(
        np.log10(NHmap_mean), 
        title=r'Mean column density',
        coord=["G"], cmap='magma',
        min=19, max=23,
        nest=True, 
        unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide"
    )
    plt.savefig('fg_NH_los_mean.png', dpi=150)
    plt.close()

    projview(
        np.log10(NHmap_std), 
        title=r'Standard deviation column density',
        coord=["G"], cmap='magma',
        min=19, max=23,
        nest=True, 
        unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide"
    )
    plt.savefig('fg_NH_los_std.png', dpi=150)
    plt.close()

# Get the column density maps and plot these samples
NHmap=np.load('NHmap.npy')
# plot_NHmap(NHmap)

# Get the samples of gamma-ray map for plots
data=np.load('gmap.npz')
Eg=data['Eg']
gmap_Geant4=data['gmap_Geant4']
gmap_QGSJET=data['gmap_QGSJET']

nsample=(NHmap.shape)[0]
nside=int(np.sqrt((NHmap.shape)[1]/12.0))
npix=hp.nside2npix(nside)

# Get pixel coordinates to mask part of the map since these maps are valid only at high lattitude
l, b=hp.pixelfunc.pix2ang(nside, np.arange(npix), lonlat=True, nest=True)
l=np.where(l<0,l+360,l)

# Mask the disk since dust map is only up to 1.25 kpc
mask=(np.abs(b)<=8.0) 

# Illustrate for the Geant4 maps
gmap_Geant4[:,:,mask]=np.nan
gmap_mean_Geant4=np.nanmean(gmap_Geant4,axis=0)
gmap_std_Geant4=np.nanstd(gmap_Geant4,axis=0)

iEgplot=20 # Plot maps with Eg = 10 GeV

projview(
    np.log10(gmap_mean_Geant4[iEgplot,:]), 
    title=r'Mean gamma-ray map at $E_\gamma=%.2f$ GeV' % (Eg[iEgplot]),
    coord=["G"], cmap='magma',
    min=-9.5, max=-7.5,
    nest=True, 
    unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma_los_mean_Geant4.png', dpi=150)
plt.close()

projview(
    gmap_std_Geant4[iEgplot,:]/gmap_mean_Geant4[iEgplot,:], 
    title=r'Standard deviation to mean gamma-ray map at $E_\gamma=%.2f$ GeV' % (Eg[iEgplot]),
    coord=["G"], cmap='magma',
    min=0.005, max=0.2,
    nest=True, 
    unit=r'$\sigma(\phi_\gamma)/\phi_\gamma$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma_los_std_Geant4.png', dpi=150)
plt.close()

# Illutrate uncertainty of spectra in a certain region
mask=(b<=30.0) | (b>=60.0) | (l<=240.0) | (l>=270.0)
gmap_Geant4[:,:,mask]=np.nan
gmap_QGSJET[:,:,mask]=np.nan

gmap_mean_Geant4=np.nanmean(gmap_Geant4,axis=0)
gmap_mean_QGSJET=np.nanmean(gmap_QGSJET,axis=0)
gmap_std_Geant4=np.nanstd(gmap_Geant4,axis=0)
gmap_std_QGSJET=np.nanstd(gmap_QGSJET,axis=0)

projview(
    np.log10(gmap_mean_Geant4[iEgplot,:]), 
    title=r'Mean gamma-ray map at $E_\gamma=%.2f$ GeV' % (Eg[iEgplot]),
    coord=["G"], cmap='magma',
    min=-9.5, max=-7.5,
    nest=True, 
    unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma_los_mean_Geant4_zoom.png', dpi=150)
plt.close()

projview(
    gmap_std_Geant4[iEgplot,:]/gmap_mean_Geant4[iEgplot,:], 
    title=r'Standard deviation to mean gamma-ray map at $E_\gamma=%.2f$ GeV' % (Eg[iEgplot]),
    coord=["G"], cmap='magma',
    min=0.005, max=0.2,
    nest=True, 
    unit=r'$\sigma(\phi_\gamma)/\phi_\gamma$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma_los_std_Geant4_zoom.png', dpi=150)
plt.close()

phi_Geant4=np.nanmean(gmap_Geant4,axis=2)
phi_QGSJET=np.nanmean(gmap_QGSJET,axis=2)

phi_mean_Geant4=np.mean(phi_Geant4,axis=0)
phi_mean_QGSJET=np.mean(phi_QGSJET,axis=0)
phi_std_Geant4=np.std(phi_Geant4,axis=0)
phi_std_QGSJET=np.std(phi_QGSJET,axis=0)

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)
for i in range(nsample):
    ax.plot(Eg,Eg**2.8*phi_Geant4[i,:],'r--',linewidth=2.0)
    ax.plot(Eg,Eg**2.8*phi_QGSJET[i,:],'g:',linewidth=2.0)

ax.plot(Eg,Eg**2.8*phi_mean_Geant4,'r--',linewidth=2.0,label=r'${\rm Geant4}$')
ax.plot(Eg,Eg**2.8*phi_mean_QGSJET,'g:',linewidth=2.0,label=r'${\rm QGSJET}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.0e0,1.0e3)
ax.set_ylim(5.0e-8,1.0e-6)
ax.set_xlabel(r'$E_{\gamma}\, {\rm (GeV)}$',fontsize=fs)
ax.set_ylabel(r'$E_{\gamma}\phi(E_{\gamma}) \, ({\rm GeV^{1.8}\, cm^{-2}\, s^{-1}\, sr^{-1}})$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.legend(loc='lower right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig('fg_phi_gamma.png')
