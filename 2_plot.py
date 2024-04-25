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

# Filter and compute flux
# mask=(np.abs(b)<=8.0) 
mask=(b<=30.0) | (b>=60.0) | (l<=240.0) | (l>=270.0)

gmap_Geant4[:,:,mask]=np.nan
gmap_QGSJET[:,:,mask]=np.nan

gmap_mean_Geant4=np.nanmean(gmap_Geant4,axis=0)
gmap_mean_QGSJET=np.nanmean(gmap_QGSJET,axis=0)
gmap_std_Geant4=np.nanstd(gmap_Geant4,axis=0)
gmap_std_QGSJET=np.nanstd(gmap_QGSJET,axis=0)

phi_Geant4=np.nanmean(gmap_Geant4,axis=2)
phi_QGSJET=np.nanmean(gmap_QGSJET,axis=2)

phi_mean_Geant4=np.mean(phi_Geant4,axis=0)
phi_mean_QGSJET=np.mean(phi_QGSJET,axis=0)
phi_std_Geant4=np.std(phi_Geant4,axis=0)
phi_std_QGSJET=np.std(phi_QGSJET,axis=0)

iEgplot=20 # Plot maps with Eg = 10 GeV

projview(
    np.log10(gmap_mean_Geant4[iEgplot,:]), 
    title=r'Mean gamma-ray map at $E_\gamma=%.2f$ GeV' % (Eg[iEgplot]),
    coord=["G"], cmap='magma',
    # min=19, max=23,
    nest=True, 
    unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
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
    # min=19, max=23,
    nest=True, 
    unit=r'$log_{10}N_{\rm H}\, [{\rm cm}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_gamma_los_std_Geant4.png', dpi=150)
plt.close()

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

# rh=np.logspace(np.log10(68.61873),np.log10(1248.1001),517)
# print(rh[1:-1].shape)
# drh=rh[1:]-rh[0:-1]
# rh=rh[1:]

# rh1=250.0
# rh2=500.0
# rh3=750.0

# mask1=(rh<=250.0)
# mask2=(rh<=500.0)
# mask3=(rh<=750.0)

# mean=np.load('mean_cube.npy')*2.8

# map1=np.sum(mean[mask1]*drh[mask1,np.newaxis],axis=0)
# map2=np.sum(mean[mask2]*drh[mask2,np.newaxis],axis=0)
# map3=np.sum(mean[mask3]*drh[mask3,np.newaxis],axis=0)

# projview(
#     map1, 
#     title=r'Dust extinction $r_{\odot}<%d$ pc' % rh[mask1][-1],
#     coord=["G"], cmap='magma',
#     min=0, max=4,
#     nest=True, 
#     unit=r'$A(V)$',
#     graticule=True, graticule_labels=True, 
#     # xlabel=r'longitude (deg)',
#     # ylabel=r'latitude (deg)',
#     projection_type="mollweide"
# )
# plt.savefig('fg_los_500.png', dpi=150)
# plt.close()

# projview(
#     map2, 
#     title=r'Dust extinction $r_{\odot}<%d$ pc' % rh[mask2][-1],
#     coord=["G"], cmap='magma',
#     min=0, max=4,
#     nest=True, 
#     unit=r'$A(V)$',
#     graticule=True, graticule_labels=True, 
#     # xlabel=r'longitude (deg)',
#     # ylabel=r'latitude (deg)',
#     projection_type="mollweide"
# )
# plt.savefig('fg_los_500.png', dpi=150)
# plt.close()

# projview(
#     map3, 
#     title=r'Dust extinction $r_{\odot}<%d$ pc' % rh[mask3][-1],
#     coord=["G"], cmap='magma',
#     min=0, max=4,
#     nest=True, 
#     unit=r'$A(V)$',
#     graticule=True, graticule_labels=True, 
#     # xlabel=r'longitude (deg)',
#     # ylabel=r'latitude (deg)',
#     projection_type="mollweide"
# )
# plt.savefig('fg_los_750.png', dpi=150)
# plt.close()
