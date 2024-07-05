import numpy as np
import scipy as sp
import LibppGam as ppG
import LibproCR as pCR
import matplotlib.pyplot as plt
import time

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
xiSNR=0.65 # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0=7.0 # km/s -> Advection speed

# Combine all parameters for proagation
pars_prop=np.array([R, L, alpha, xiSNR, u0])

# Compute the coefficients
q_n=pCR.compute_coefficients(pCR.func_gSNR_YUK04,zeta_n,R)

# Define grid for cosmic-ray distribution
r=np.linspace(0.0,R,501)
z=np.linspace(0.0,L,41)
E=np.logspace(10.0,14.0,81)
fE=pCR.func_fE(pars_prop,zeta_n,q_n,E,r,z)
fE[fE<0.0]=np.nan

vp=np.sqrt((E+mp)**2-mp**2)*3.0e10/(E+mp)
jE=fE*vp[:,np.newaxis,np.newaxis]*1.0e9 # GeV^-1 cm^-2 s^-1

# # Create an interpolator object
# fE_interp=sp.interpolate.RegularGridInterpolator((np.log10(E_old), r, z), fE)

# # # Prepare the points to interpolate
# E_new=np.logspace(10.0,14.0,401)
# E_mesh, r_mesh, z_mesh=np.meshgrid(np.log10(E_new), r, z, indexing='ij')
# points_to_interpolate=np.stack((E_mesh, r_mesh, z_mesh), axis=-1)

# # Interpolate the values
# vp_new=np.sqrt((E_new+mp)**2-mp**2)*3.0e10/(E_new+mp)
# fE_new=fE_interp(points_to_interpolate)


pCR.plot_gSNR(zeta_n,q_n,r,R)
pCR.plot_jE_p_LOC(pars_prop,zeta_n,q_n)
pCR.plot_jE_rz(fE,r,z)

# Compute the cross-section from Kafexhiu's code (numpy deos not work)
Eg=np.logspace(1,4,31)
dXSdEg_Geant4=np.zeros((len(E),len(Eg))) 
for i in range(len(E)):
    for j in range(len(Eg)):
        dXSdEg_Geant4[i,j]=ppG.dsigma_dEgamma_Geant4(E[i]*1.0e-9,Eg[j])*1.0e-27 # cm^2/GeV

# Compute gamma-ray emissivity with cross section from Kafexhiu et al. 2014 (note that 1.8 is the enhancement factor due to nuclei)
qg_Geant4=1.8*sp.integrate.trapezoid(jE[:,np.newaxis,:,:]*dXSdEg_Geant4[:,:,np.newaxis,np.newaxis], E*1.0e-9, axis=0) # GeV^-1 s^-1 sr^-1


# Record the ending time
end_time=time.time()

# Calculate the elapsed time
elapsed_time=end_time-start_time

print("Elapsed time:", elapsed_time, "seconds")

fs=22

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

ax.plot(Eg,qg_Geant4[:,r==8000.0,z==0.0]/(4.0*np.pi),'k-',linewidth=3,label=r'${\rm Local\, Emissivity}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$E \,{\rm (eV)}$',fontsize=fs)
ax.set_ylabel(r'$\varepsilon(E)\, {\rm (eV^{-1}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
for label_axd in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_axd.set_fontsize(fs)
# ax.set_xlim(1.0,100.0)
# ax.set_ylim(1.0e-37,1.0e-36)
ax.legend(loc='lower left', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig("fg_emissivity.png")