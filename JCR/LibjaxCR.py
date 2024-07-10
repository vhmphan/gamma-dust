import jax.numpy as jnp
from jax import lax
from jax import jit
from jax import grad

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

# # Define the log-likelihood function
# @jit
# def log_likelihood(theta, r_data, gSNR_data, zeta_n, R):
#     gSNR_fit = func_gSNR_fit(theta, zeta_n, R, r_data)
#     return -0.5 * jnp.sum((gSNR_fit - gSNR_data)**2)

# from jax import random

# # Define the MCMC sampler
# def mcmc_sampler(log_prob_fn, initial_params, step_size, num_samples, num_burnin, key):
#     num_params = len(initial_params)
#     samples = []
#     current_params = initial_params
#     current_log_prob = log_prob_fn(current_params)
    
#     for i in range(num_samples + num_burnin):
#         key, subkey = random.split(key)
#         proposal = current_params + step_size * random.normal(subkey, shape=(num_params,))
#         proposal_log_prob = log_prob_fn(proposal)
        
#         acceptance_prob = jnp.exp(proposal_log_prob - current_log_prob)
#         key, subkey = random.split(key)
#         accept = random.uniform(subkey) < acceptance_prob
        
#         current_params = lax.select(accept, proposal, current_params)
#         current_log_prob = lax.select(accept, proposal_log_prob, current_log_prob)
        
#         if i >= num_burnin:
#             samples.append(current_params)
    
#     return jnp.array(samples)