import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import h5py

# Open the FITS file and get info on the data
hdul=fits.open('samples_densities_hpixr.fits')
print(hdul.info())

# Access data and header of the primary HDU
r=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins

print(r)

dr=np.diff(r)*3.086e21 # cm -> Bin width for the integration along the line of sight
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()

print(samples_H2.shape)

# Plot the mean column density map
NHImap=np.sum(samples_HI*dr[np.newaxis,:,np.newaxis],axis=1)
NH2map=np.sum(samples_H2*dr[np.newaxis,:,np.newaxis],axis=1)
N_sample, N_pix=NHImap.shape

projview(
    np.log10(np.mean(NHImap,axis=0)), 
    title=r'Mean HI',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_HI_mean.png', dpi=150)
plt.close()

projview(
    np.log10(np.mean(NH2map,axis=0)), 
    title=r'Mean H2',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide"
)
plt.savefig('fg_H2_mean.png', dpi=150)
plt.close()

# # Uncomment to get column density maps from samples
# for isample in range(N_sample):
#     projview(
#         np.log10(NHImap[isample,:]), 
#         title=r'HI sample %d' % (isample),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide"
#     )
#     plt.savefig('fg_HI_sample_%d.png' % (isample), dpi=150)
#     plt.close()

#     projview(
#         np.log10(NH2map[isample,:]), 
#         title=r'H2 sample %d' % (isample),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}\phi_{\rm gamma}(E_\gamma)\, [{\rm GeV}^{-1}\, {\rm cm}^{-2}\, {\rm s}^{-2}\, {\rm sr}^{-1}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide"
#     )
#     plt.savefig('fg_H2_sample_%d.png' % (isample), dpi=150)
#     plt.close()

