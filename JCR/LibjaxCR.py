import jax.numpy as jnp
from jax import lax
from jax import jit
from jax import grad

# Zeroth order Bessel function of first kind
@jit
def j0(x):
    def small_x(x):
        z = x * x
        num = 57568490574.0 + z * (-13362590354.0 + z * (651619640.7 +
              z * (-11214424.18 + z * (77392.33017 + z * (-184.9052456)))))
        den = 57568490411.0 + z * (1029532985.0 + z * (9494680.718 +
              z * (59272.64853 + z * (267.8532712 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (-0.1098628627e-2 + y2 * (0.2734510407e-4 +
               y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y2 * (0.1430488765e-3 +
               y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 -
               y2 * 0.934935152e-7)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 0.785398164) * ans1 - y * jnp.sin(x - 0.785398164) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))

# First order Bessel function of first kind
@jit
def j1(x):
    def small_x(x):
        z = x * x
        num = x * (72362614232.0 + z * (-7895059235.0 + z * (242396853.1 +
              z * (-2972611.439 + z * (15704.48260 + z * (-30.16036606))))))
        den = 144725228442.0 + z * (2300535178.0 + z * (18583304.74 +
              z * (99447.43394 + z * (376.9991397 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 +
               y2 * (0.2457520174e-5 - y2 * 0.240337019e-6)))
        ans2 = 0.04687499995 + y2 * (-0.2002690873e-3 +
               y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 +
               y2 * 0.105787412e-6)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 2.356194491) * ans1 - y * jnp.sin(x - 2.356194491) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))

# Surface density of SNRs from Yusifov et al. 2004
@jit
def func_gSNR_YUK04(r):
# r (pc)

    r=jnp.array(r)*1.0e-3 # kpc
    gSNR=jnp.where(
        r<15.0,
        jnp.power((r+0.55)/9.05,1.64)*jnp.exp(-4.01*(r-8.5)/9.05)/5.95828e+8,
        0.0
    )    
    return gSNR # pc^-2

# Some functions to test jax on finding best fit with gradient descent
@jit
def jcompute_coefficients(zeta_n, R):

    r=jnp.linspace(0.0,R,250000)
    fr=func_gSNR_YUK04(r)
    j0_n=j0(zeta_n[:,jnp.newaxis]*r[jnp.newaxis,:]/R)
    coefficients=jnp.trapezoid(r[jnp.newaxis,:]*fr[jnp.newaxis,:]*j0_n,r)
    coefficients*=(2.0/(R**2*(j1(zeta_n)**2)))

    return coefficients

@jit
def func_gSNR_fit(theta, zeta_n, R, r):

    A, B, C=theta
    r_int=jnp.linspace(0.0,R,200000)
    fr_int=jnp.where(
        r_int<15.0,
        A*jnp.power((r_int+0.55)/9.05,B)*jnp.exp(-C*(r_int-8.5)/9.05),
        0.0
    ) 
    j0_n=j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]/R)
    coefficients=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
    coefficients*=(2.0/(R**2*(j1(zeta_n)**2)))

    gSNR=jnp.sum(coefficients[:,jnp.newaxis]*j0(zeta_n[:,jnp.newaxis]*r[jnp.newaxis,:]/R),axis=0)

    return gSNR

@jit
def loss_func(theta, zeta_n, R, r_data, gSNR_data):
    gSNR_fit=func_gSNR_fit(theta,zeta_n,R,r_data)

    return jnp.mean((gSNR_fit-gSNR_data)**2)

@jit
def update(theta, zeta_n, R, r_data, gSNR_data, lr=jnp.array([0.01,1.0e16,0.01])):
    return theta-lr*grad(loss_func)(theta,zeta_n,R,r_data,gSNR_data)

@jit
def func_jE_YUK04(pars_prop, zeta_n, E, r, z):

    mp=938.272e6 # eV

    # Transsport parameters
    R=pars_prop[0] # pc
    L=pars_prop[1] # pc
    alpha=pars_prop[2] 
    xiSNR=pars_prop[3]
    u0=pars_prop[4]*365.0*86400.0/3.086e18 # km/s to pc/yr -> Advection speed

    p=jnp.sqrt((E+mp)**2-mp**2)  # eV
    vp=p/(E+mp)

    # Diffusion coefficient
    pb=312.0e9
    Diff=1.1e28*(365.0*86400.0/(3.08567758e18)**2)*vp*(p/1.0e9)**0.63/(1.0+(p/pb)**2)**0.1 # pc^2/yr

    # Spatial distribution of sources
    r_int=jnp.linspace(0.0,R,500000)*1.0e-3
    fr_int=jnp.where(
        r_int<15.0,
        jnp.power((r_int+0.55)/9.05,1.64)*jnp.exp(-4.01*(r_int-8.5)/9.05)/5.95828e+8,
        0.0
    ) 
    j0_n=j0(zeta_n[:,jnp.newaxis]*r_int[jnp.newaxis,:]*1.0e3/R)
    q_n=jnp.trapezoid(r_int[jnp.newaxis,:]*fr_int[jnp.newaxis,:]*j0_n,r_int)
    q_n*=1.0e6*(2.0/(R**2*(j1(zeta_n)**2))) # pc^-2

    # Injection spectrum of sources
    xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
    x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
    Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)
    # print('hj',xmin)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    QE=(xiSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha)
    QE*=RSNR*vp*3.0e10

    Diff=Diff[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    z=z[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
    r=r[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
    zeta_n=zeta_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]
    q_n=q_n[:,jnp.newaxis,jnp.newaxis,jnp.newaxis]

    Sn=jnp.sqrt((u0/Diff)**2+4.0*(zeta_n/R)**2) # pc^-2
    fEn=j0(zeta_n*r/R)*q_n*jnp.exp(u0*z/(2.0*Diff))*jnp.sinh(Sn*(L-z)/2.0)/(jnp.sinh(Sn*L/2.0)*(u0+Sn*Diff*(jnp.cosh(Sn*L/2.0)/jnp.sinh(Sn*L/2.0))))
    fE=jnp.sum(fEn,axis=0) # eV^-1 pc^-3
    fE=jnp.where(fE<0.0,0.0,fE)

    jE=fE*QE[:,jnp.newaxis,jnp.newaxis]*1.0e9/(3.086e18)**3 

    return jE # GeV^-1 cm^-2 s^-1