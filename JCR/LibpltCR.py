import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("text",usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

# Plot the spatial cosmic-ray distribution
def plot_jEp_GAL(jE, rg, zg):

    fs=22

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Spatial distribution over the entire grid
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(jE[0,:,:], origin='lower', extent=[rg[0]*1.0e-3, rg[-1]*1.0e-3, zg[0]*1.0e-3, zg[-1]*1.0e-3], cmap='magma')

    ## Colourbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel(r'$j(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')  
    ax1.set_title(r'$E=10$\,{\rm GeV}')
    ax1.set_xlabel(r"$r_{G}\,{\rm [kpc]}$")
    ax1.set_ylabel(r"$z_{G}\,{\rm [kpc]}$")

    # Profile over r
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rg*1.0e-3, jE[0,:,0], 'r')
    ax2.set_title(r'$E=10$\,{\rm GeV}\, {\rm and}\, $z_G=0$\,{\rm kpc}')
    ax2.set_xlabel(r'$r_{G}\,{\rm [kpc]}$')
    ax2.set_ylabel(r'$j(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')
    ax2.set_xlim(rg[0]*1.0e-3,rg[-1]*1.0e-3)

    # Profile over z
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(zg*1.0e-3, jE[0,0,:], 'r')
    # ax3.plot(zg*1.0e-3, fE[0,rg==4000.0,:][0], 'g')
    # ax3.plot(zg*1.0e-3, fE[0,rg==8000.0,:][0], 'k')
    ax3.set_title(r'$E=10$\,{\rm GeV}\, {\rm and}\, $r_G=0$\,{\rm kpc}')
    ax3.set_xlabel(r'$z_{G}\,{\rm [kpc]}$')
    ax3.set_ylabel(r'$f(E) \, ({\rm GeV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}})$')
    ax3.set_xlim(zg[0]*1.0e-3,zg[-1]*1.0e-3)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.05, wspace=0.15, top=1.1, bottom=0.1, left=0.05, right=0.95)
    
    plt.savefig("fg_jEp_GAL.png", dpi=600)
    plt.close()