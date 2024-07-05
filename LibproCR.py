import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

mp=938.272e6 # eV -> Proton mass

#############################################################################
# Auxiliary functions
#############################################################################

# Compute the coefficients q_n
def compute_coefficients(f, zeta_n, R):
    coefficients = []
    for zeta in zeta_n:
        integrand = lambda r: r*f(r)*sp.special.j0(zeta*r/R)
        integral, _=sp.integrate.quad(integrand,0,R,epsabs=1.0e-11,epsrel=1.0e-11)
        coefficient=(2.0/(R**2*(sp.special.j1(zeta)**2)))*integral
        coefficients.append(coefficient)
    return np.array(coefficients)

# Reconstruct the function using the series expansion
def reconstruct_function(r, q_n, zeta_n, R):
    expansion=np.zeros_like(r)
    for q, zeta in zip(q_n, zeta_n):
        expansion+=q*sp.special.j0(zeta*r/R)
    return expansion

#############################################################################
# Function for transport
#############################################################################

# Surface density of SNRs
def func_gSNR_YUK04(r):
# r (pc)

    r=np.array(r)*1.0e-3 # kpc
    gSNR=np.zeros_like(r)
    mask=(r<15.0)
    gSNR[mask]=np.power((r[mask]+0.55)/9.05,1.64)*np.exp(-4.01*(r[mask]-8.5)/9.05)
    
    return gSNR/5.95828e+8 # pc^-2

# Diffusion coefficient -> Rigidity dependence (Genolini et al. 2017)
def func_D_rigid(E):
# E (eV)
    
    pb=312.0e9
    p=np.sqrt((E+mp)**2-mp**2)  # eV
    vp=p/(E+mp)

    Diff=1.1e28*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1
    Diff*=365.0*86400.0/(3.08567758e18)**2
    
    return Diff # pc^2/yr

# Normalization for the injection spectrum
def func_Gam(alpha):
    xmin=np.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=np.sqrt((1.0e14+mp)**2-mp**2)/mp
    
    Gam, _ = sp.integrate.quad(lambda x: x**(2.0-alpha)*(np.sqrt(x**2+1.0)-1.0),xmin,xmax)
    
    return Gam

# Source function from supernova remnants
def func_QSNR(alpha, xiSNR, E):
# E (eV)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    p=np.sqrt((E+mp)**2-mp**2) # eV
    vp=p/(E+mp)
    
    Gam=func_Gam(alpha)
    Q0=xiSNR*ENSR/(mp**2*vp*Gam)
    Q=Q0*(p/mp)**(2.0-alpha)
    
    return RSNR*Q # eV^-1 yr^-1

# Compute the differential number density of protons
def func_fE(pars_prop, zeta_n, q_n, E, r, z):

    R=pars_prop[0] # pc
    L=pars_prop[1] # pc
    alpha=pars_prop[2] 
    xiSNR=pars_prop[3]
    u0=pars_prop[4]*365.0*86400.0/3.086e18 # pc/yr -> Advection speed

    Diff=func_D_rigid(E)[np.newaxis,:,np.newaxis,np.newaxis] # pc^2/yr
    QE=func_QSNR(alpha,xiSNR,E)[np.newaxis,:,np.newaxis,np.newaxis] # eV^-1 yr^-1

    z=z[np.newaxis,np.newaxis,np.newaxis,:]
    r=r[np.newaxis,np.newaxis,:,np.newaxis]
    zeta_n=zeta_n[:,np.newaxis,np.newaxis,np.newaxis]
    q_n=q_n[:,np.newaxis,np.newaxis,np.newaxis]

    Sn=np.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
    fEn=sp.special.j0(zeta_n*r/R)*QE*q_n*np.exp(u0*z/(2.0*Diff))*np.sinh(Sn*(L-z)/2.0)/(np.sinh(Sn*L/2.0)*(u0+Sn*Diff*(np.cosh(Sn*L/2.0)/np.sinh(Sn*L/2.0))))
    fE=np.sum(fEn,axis=0)

    return fE/(3.086e18)**3 # eV^-1 cm^-3

#############################################################################
# Plots
#############################################################################

# Function to test the Bessel series
def plot_gSNR(zeta_n,q_n,r,R):

    fs=22

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    # Testing the Bessel series for the function of the surface density of SNRs
    gSNR_reconstructed=reconstruct_function(r,q_n,zeta_n,R)

    ax.plot(r*1.0e-3,func_gSNR_YUK04(r),label='Original',lw=2)
    ax.plot(r*1.0e-3,gSNR_reconstructed,label='Bessel series', lw=2, linestyle='--')
    ax.set_xlabel(r'$r\, ({\rm kpc})$',fontsize=fs)
    ax.set_ylabel(r'$g_{\rm SNR}(r)\,({\rm pc^{-2}})$', fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='upper right', prop={"size":fs})
    ax.grid(linestyle='--')
    
    plt.savefig('fg_gSNR.png')
    plt.close()

# Plot the local spectrum
def plot_jE_p_LOC(pars_prop, zeta_n, q_n):

    fs=22

    E=np.logspace(8.0,14.0,61)
    fE_loc=func_fE(pars_prop,zeta_n,q_n,E,np.array([8178.0]),np.array([0.0]))[:,0,0]
    vp=np.sqrt((E+mp)**2-mp**2)*3.0e10/(E+mp)

    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)

    n=2

    ax.plot(E,E**n*fE_loc*vp/(4.0*np.pi),'k-',linewidth=3,label=r'${\rm Local\, Spectrum}$')

    # Read data of CRDB file
    # AMS data
    data=np.genfromtxt('crdb_import_p_R.csv', delimiter='","', dtype=str)
    pp=(data[:,14].astype(float))*1.0e9 # eV
    pp_lower=(data[:,15].astype(float))*1.0e9 # eV
    pp_upper=(data[:,16].astype(float))*1.0e9 # eV

    Ep=np.sqrt(pp**2+mp**2)-mp # eV
    Ep_lower=np.sqrt(pp_lower**2+mp**2)-mp # eV
    Ep_upper=np.sqrt(pp_upper**2+mp**2)-mp # eV

    vpp=pp/(Ep+mp)
    jE_data=(data[:,17].astype(float))*1.0e-13/vpp # eV^-1 cm^-2 s^-1 sr^-1
    err_jE_data=(data[:,20].astype(float))*1.0e-13/vpp # eV^-1 cm^-2 s^-1 sr^-1
    ax.errorbar(Ep,Ep**n*jE_data,Ep**n*err_jE_data,[Ep-Ep_lower,Ep_upper-Ep],'^',color='green',markersize=10.0,elinewidth=2.5,label=r'${\rm AMS}$')

    # DAMPE and CALET data
    data=np.genfromtxt('crdb_import_p_Ek.csv', delimiter='","', dtype=str)
    expname=data[:,0]
    expname=np.char.lstrip(expname,'"')
    Ep=(data[:,14].astype(float))*1.0e9 # eV
    Ep_lower=(data[:,15].astype(float))*1.0e9 # eV
    Ep_upper=(data[:,16].astype(float))*1.0e9 # eV

    jE_data=(data[:,17].astype(float))*1.0e-13 # eV^-1 cm^-2 s^-1 sr^-1
    err_jE_data=(data[:,20].astype(float))*1.0e-13 # eV^-1 cm^-2 s^-1 sr^-1

    expplot=np.array(['DAMPE', 'CALET'])
    expcolr=np.array(['lightsalmon', 'brown'])
    for i in range(len(expplot)):
        mask=(expname!=expplot[i])
        ax.errorbar(Ep[expname==expplot[i]],Ep[expname==expplot[i]]**n*jE_data[expname==expplot[i]],Ep[expname==expplot[i]]**n*err_jE_data[expname==expplot[i]],[Ep[expname==expplot[i]]-Ep_lower[expname==expplot[i]],Ep_upper[expname==expplot[i]]-Ep[expname==expplot[i]]],'^',color=expcolr[i],markersize=10.0,elinewidth=2.5,label=r'${\rm %s}$' % expplot[i])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E \,{\rm (eV)}$',fontsize=fs)
    ax.set_ylabel(r'$j(E)\, {\rm (eV\,cm^{-2}\, s^{-1}\, sr^{-1})}$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.set_xlim(E[0],E[-1])
    ax.set_ylim(1.0e5,1.0e9)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig("fg_jE_p_LOC.png")
    plt.close()

def plot_jE_rz(fE, r, z):

    fs=22

    fig=plt.figure(figsize=(10, 4))
    ax=plt.subplot(111)

    # im = ax.imshow(fE[0,:,:], origin='lower', extent=[r[0]*1.0e-3, r[-1]*1.0e-3, z[0]*1.0e-3, z[-1]*1.0e-3], cmap='magma', vmin=0, vmax=20)
    im = ax.imshow(fE[0,:,:], origin='lower', extent=[r[0]*1.0e-3, r[-1]*1.0e-3, z[0]*1.0e-3, z[-1]*1.0e-3], cmap='magma')

    ## Colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel(r"$f(E) \, ({\rm eV^{-1}\, cm^{-3}})$")  

    ## Plot ranges, labels, grid
    # ax.set_xlim([x0, -x0])
    # ax.set_ylim([y0, -y0])
    ax.set_xlabel(r"$x\,{\rm [kpc]}$")
    ax.set_ylabel(r"$y\,{\rm [kpc]}$")
    # for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label_ax.set_fontsize(fs)
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.09, top=0.97)
    plt.savefig("fg_fE.png", dpi=600)
    plt.close()