The intensity of cosmic rays (number of particles per unit energy per unit area per unit time and per unit solid angle) is estimated following [Maurin et al. 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...555..585M/abstract) which gives the solution to the transport equation as a sum over zeroth order Bessel function of first kind

$$ 
  j(E,r,z) = \frac{v}{4\pi}\sum_n J_0\left( \zeta_n \frac{r}{R} \right) \frac{\hat{q}_n Q(E) \exp\left( \frac{u_0 z}{2D} \right) \sinh\left( \frac{S_n (L - z)}{2} \right)}{\sinh\left( \frac{S_n L}{2} \right) \left[ u_0 + S_n D \coth\left( \frac{S_n L}{2} \right) \right]},
$$

where $\zeta_n$ are the zeros of the zeroth order Bessel function of first kind (about 150 terms are used for the estimate) and $\hat{q}_n$ is the coefficients from the Bessel expansion of the source distribution. We adopt the source distribution as in [Yusifov et al. 2004](https://ui.adsabs.harvard.edu/abs/2004A%26A...422..545Y/abstract) as shown below.

```sh
# Surface density of SNRs from Yusifov et al. 2004
def func_gSNR_YUK04(r):
# r (pc)

    r=jnp.array(r)*1.0e-3 # kpc
    gSNR=jnp.where(
        r<=15.0,
        jnp.power((r+0.55)/9.05,1.64)*jnp.exp(-4.01*(r-8.5)/9.05)/5.95828e+8,
        0.0
    )    
    return gSNR # pc^-2
```

The injection spectrum $Q(E)$ is a power-law in momentum and the diffusion coefficient is as in [Genolini et al. 2017](https://ui.adsabs.harvard.edu/abs/2017PhRvL.119x1101G/abstract). The other transport parameters are also as commonly adopted in the literature. We summarize below the injection spectrum and all the transport parameters.

```sh
# Size of the cosmic-ray halo
R=20000.0 # pc -> Radius of halo
L=4000.0  # pc -> Height of halo

# Earth's location
Rsol=8178.0 # pc 

# Parameters for injection spectrum
alpha=4.23  # -> Injection spectral index
xiSNR=0.065 # -> Fracion of SNR kinetic energy into CRs

# Advection speed
u0=7.0 # km/s 

# Combine all parameters for proagation
pars_prop=jnp.array([R, L, alpha, xiSNR, u0])

# Define cosmic-ray grid and diffusion coefficient
E=jnp.logspace(10.0,14.0,81) # eV 
p=jnp.sqrt((E+mp)**2-mp**2)  # eV
vp=p/(E+mp)
Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/312.0e9)**2)**0.1 # pc^2/yr

# Injection spectrum of sources
xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x) # -> Normalization for the injection spectrum

RSNR=0.03 # yr^-1 -> SNR rate
ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
QE=RSNR*(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha) # eV^-1 s^-1
```
