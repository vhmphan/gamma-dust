import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# Open the FITS file
hdul=fits.open('samples_healpix.fits')

# Print the info of the FITS file to see the content and structure
hdul.info()

# Access specific HDU
# For example, to access the primary HDU (which is usually the first one):
primary_hdu=hdul[1]

# Access data and header of the primary HDU
samples=hdul[1].data
hdul.close()

rh=np.logspace(np.log10(68.61873),np.log10(1248.1001),517)
drh=rh[1:]-rh[0:-1]
rh=rh[1:]

# The dust maps are in unit of E(B-V)/pc. We need to convert to density of H atoms by multiplying with 
# ~2.22 x 10^21 (NH = A(V) x 2.22 x 10^21 cm^-2, see Guver et al. A&A 2009). 
NHmap=2.8*2.22e21*np.sum(samples*drh[np.newaxis,:,np.newaxis],axis=1)
np.save('NHmap.npy', NHmap)

# mean=np.mean(samples,axis=0)
# np.save('mean_cube.npy', mean)
