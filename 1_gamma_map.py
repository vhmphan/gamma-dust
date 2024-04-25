import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import LibppGam as ppG
import scipy as sp

# Function for interpolation
def log_interp1d(xx, yy, kind='linear'):

    logx=np.log10(xx)
    logy=np.log10(yy)
    lin_interp=sp.interpolate.interp1d(logx,logy,kind=kind)
    log_interp=lambda zz: np.power(10.0,lin_interp(np.log10(zz)))

    return log_interp

# Get the column density maps and plot these samples
NHmap=np.load('NHmap.npy')

# Compute the emissivity
# Local ISM spectra from Boschini et al. ApJS 2020
Tp_data, jTp_data=np.loadtxt('jTp_B20.txt',unpack=True,usecols=[0,1])
jTp_data*=1.0e-4 # GeV^-1 cm^-2 sr^-1 s^-1 (Tp_data in GeV)

Tp=np.logspace(-1,5,5001)
Eg=np.logspace(-1,4,51)

# Compute the cross-section from Kafexhiu's code (numpy deos not work)
dXSdEg_Geant4=np.zeros((len(Tp),len(Eg))) 
dXSdEg_QGSJET=np.zeros((len(Tp),len(Eg))) 
for i in range(len(Tp)):
    for j in range(len(Eg)):
        dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(Tp[i],Eg[j])*1.0e-27 # cm^2/GeV
        dXSdEg_QGSJET[i,j]=ppG.dsigma_dEgamma_QGSJET(Tp[i],Eg[j])*1.0e-27 # cm^2/GeV

func_jTp_B20=log_interp1d(Tp_data,jTp_data)
jTp_B20=func_jTp_B20(Tp)

# Compute gamma-ray emissivity with 2 different sets of cross sections from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
qg_Geant4=1.8*sp.integrate.trapezoid(jTp_B20[:,np.newaxis]*dXSdEg_Geant4, Tp, axis=0) # GeV^-1 s^-1 sr^-1
qg_QGSJET=1.8*sp.integrate.trapezoid(jTp_B20[:,np.newaxis]*dXSdEg_QGSJET, Tp, axis=0) # GeV^-1 s^-1 sr^-1

# Create samples of gamma-ray map 
gmap_Geant4=qg_Geant4[np.newaxis,:,np.newaxis]*NHmap[:,np.newaxis,:] # GeV^-1 cm^-2 s^-1 sr^-1
gmap_QGSJET=qg_QGSJET[np.newaxis,:,np.newaxis]*NHmap[:,np.newaxis,:] # GeV^-1 cm^-2 s^-1 sr^-1

np.savez('gmap.npz', Eg=Eg, gmap_Geant4=gmap_Geant4, gmap_QGSJET=gmap_QGSJET)

