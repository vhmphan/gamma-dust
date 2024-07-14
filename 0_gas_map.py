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

# Access data 
r=(hdul[2].data)['radial pixel edges'].astype(np.float64) # Edges of radial bins
dr=np.diff(r)*3.086e21 # cm -> Bin width for the integration along the line of sight
samples_HI=(hdul[3].data).T # cm^-3
samples_H2=(hdul[4].data).T # cm^-3
hdul.close()

# Plot the mean column density map
NHImap=np.sum(samples_HI*dr[np.newaxis,:,np.newaxis],axis=1)
NH2map=np.sum(samples_H2*dr[np.newaxis,:,np.newaxis],axis=1)
N_sample, N_pix=NHImap.shape
print(samples_HI.shape)

# import h5py

# # Open the HDF5 file
# with h5py.File('JCR/energy_bins.hdf5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     Eg_data=file['geom_avg_bin_energy'][:]
#     Eg_data_lower=file['lower_bin_boundaries'][:]
#     Eg_data_upper=file['upper_bin_boundaries'][:]
    
# dEg_data=Eg_data_upper-Eg_data_lower

# with h5py.File('JCR/I_dust.hdf5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     gamma_map_data=file['stats']['mean'][:]

# gamma_map_data*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
# gamma_map_data=hp.ud_grade(gamma_map_data[5,:], nside_out=64)
# gamma_map_data=hp.reorder(gamma_map_data, r2n=True)

# emi_map=gamma_map_data/(np.mean(NHImap+2.0*NH2map,axis=0)*4.0*np.pi)
# print(emi_map.shape)

# l, b=hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
# l=np.where(l<0,l+360,l)

# mask=(np.abs(b)<=8.0) 
# emi_map[mask]=np.nan

# print(np.nanmean(emi_map))

fig=plt.figure(figsize=(12, 5))

projview(
    np.log10(np.mean(NHImap,axis=0)), 
    # np.log10(emi_map), 
    title=r'Mean HI',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}N({\rm HI})\, [{\rm cm}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=121
)

projview(
    np.log10(np.mean(NH2map,axis=0)), 
    title=r'Mean H2',
    coord=["G"], cmap='viridis',
    min=20, max=23,
    nest=True, 
    unit=r'$log_{10}N({\rm H}_2)\, [{\rm cm}^{-2}]$',
    graticule=True, graticule_labels=True, 
    # xlabel=r'longitude (deg)',
    # ylabel=r'latitude (deg)',
    projection_type="mollweide",
    sub=122
)

fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

plt.savefig('fg_gas_map.png', dpi=150)
plt.close()

# Uncomment to get column density maps from samples
# for isample in range(N_sample):

#     fig=plt.figure(figsize=(12, 5))

#     projview(
#         np.log10(NHImap[isample,:]), 
#         title=r'HI sample %d' % (isample+1),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}N({\rm HI})\, [{\rm cm}^{-2}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide",
#         sub=121
#     )

#     projview(
#         np.log10(NH2map[isample,:]), 
#         title=r'H2 sample %d' % (isample+1),
#         coord=["G"], cmap='viridis',
#         min=20, max=23,
#         nest=True, 
#         unit=r'$log_{10}N({\rm H}_2)\, [{\rm cm}^{-2}]$',
#         graticule=True, graticule_labels=True, 
#         # xlabel=r'longitude (deg)',
#         # ylabel=r'latitude (deg)',
#         projection_type="mollweide",
#         sub=122
#     )

#     fig.tight_layout(pad=1.0)
#     fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)

#     plt.savefig('fg_gas_sample_%d.png' % (isample), dpi=150)
#     plt.close()

