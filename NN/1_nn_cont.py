import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vphan/Minh/Code/Gas3D/gamma-dust')))

import LibjaxCR as jCR
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import healpy as hp
import time
import h5py

l, b = hp.pixelfunc.pix2ang(64, np.arange(12*64*64), lonlat=True, nest=True)
l = np.where(l < 0, l + 360, l)

mask = (np.abs(b) <= 10.0)

# Load diffuse gamma-ray map from Platz et al. 2023
with h5py.File('../JCR/energy_bins.hdf5', 'r') as file:
    print("Keys: %s" % file.keys())
    Eg_data=file['geom_avg_bin_energy'][:]
    Eg_data_lower=file['lower_bin_boundaries'][:]
    Eg_data_upper=file['upper_bin_boundaries'][:]
    
dEg_data=Eg_data_upper-Eg_data_lower

with h5py.File('../JCR/I_dust.hdf5', 'r') as file:
    print("Keys: %s" % file['stats'].keys())
    gamma_map_mean=file['stats']['mean'][:]
    gamma_map_std=file['stats']['standard deviation'][:]

gamma_map_mean*=1.0e-4*4.0*np.pi/dEg_data[:,np.newaxis]
gamma_map_mean=hp.ud_grade(gamma_map_mean[5,:], nside_out=64)
gamma_map_mean=hp.reorder(gamma_map_mean, r2n=True)
gamma_map_mean=gamma_map_mean[np.newaxis,np.newaxis,:] # GeV^-1 cm^-2 s^-1
gamma_map_mean=jnp.array(gamma_map_mean)

mp = 938.272e6  # eV

# Find the first 'num_zeros' zeros of the zeroth order Bessel function J0
num_zeros = 100
zeta_n = jnp.array(sp.special.jn_zeros(0, num_zeros))

# Size of the cosmic-ray halo
R = 20000.0  # pc -> Radius of halo
L = 4000.0  # pc -> Height of halo

# Parameters for injection spectrum
alpha = 4.23  # -> Injection spectral index
xiSNR = 0.065  # -> Fracion of SNR kinetic energy into CRs

# Transport parameter
u0 = 7.0  # km/s -> Advection speed

# Combine all parameters for propagation
pars_prop = jnp.array([R, L, alpha, xiSNR, u0])

# Define cosmic-ray grid and diffusion coefficient
E = jnp.logspace(10.0, 14.0, 81)  # eV 

# Bessel functions
r_int = jnp.linspace(0.0, R, 25000)
j0_n_int = jCR.j0(zeta_n[:, jnp.newaxis] * r_int[jnp.newaxis, :] / R)

# Define gamma-ray energy grids and compute the cross-section from Kafexhiu's code (numpy does not work)
Eg = jnp.logspace(np.log10(13.33521432163324), 2, 1)  # GeV
dXSdEg_Geant4 = jCR.func_dXSdEg(E * 1.0e-9, Eg)

# Load gas density, bin width of Heliocentric radial bin, and points for interpolating the 
ngas, drs, points_intr = jCR.load_gas('../samples_densities_hpixr.fits')
ngas_mean = jnp.mean(ngas, axis=0)[jnp.newaxis, :, :]

# Define the properties of the neural network
N_SAMPLES = 51
LAYERS = [1, 20, 20, 20, 1]
LEARNING_RATE = 0.005
N_EPOCHS = 10
epoch_print = N_EPOCHS / 10

# Load the .npz file
data_WB = np.load('nn_20_ana.npz')

# Extract weights, biases, and activation functions
weight_matrices = [data_WB[f"weight_{i}"] for i in range(len([key for key in data_WB if key.startswith("weight_")]))]
bias_vectors = [data_WB[f"bias_{i}"] for i in range(len([key for key in data_WB if key.startswith("bias_")]))]
activation_functions = []

for (fan_in, fan_out) in zip(LAYERS[:-1], LAYERS[1:]):
    activation_functions.append(jax.nn.sigmoid)

activation_functions[-1] = lambda x: x

# Function to define the network 
def network_forward(x, weights, biases, activations):
    a = x
    for W, b, f in zip(weights, biases, activations):
        a = f(a @ W + b)
    return a

# Spatial domain for the function gSNR to be inferred and renormalize for the neural network
x_samples_raw = jnp.linspace(0, 15.0, N_SAMPLES)[:, jnp.newaxis]
x_samples = x_samples_raw / x_samples_raw[-1]

# Mock data for testing
y_sol = 2.0e-9 # jCR.func_gSNR_YUK04(jnp.array([8178.0])) / jnp.exp(y_init)
gSNR_best_fit=jCR.func_gSNR_fit(jnp.array([2.1657154516413e-09,0.4440271097459036,1.3679861897204548,4.093126645363395]),zeta_n,R*1.0e-3,x_samples_raw) 
y_grtruth = jnp.log(gSNR_best_fit/y_sol) # jnp.log((gSNR_best_fit + 1.0e-9*jnp.exp(-(x_samples_raw-12.0)**2/2.0) + 1.0e-9*jnp.exp(-(x_samples_raw-6.0)**2/0.5)) / (y_sol))
y_samples_raw = y_grtruth
# y_samples = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_samples_raw) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))[0, 0, mask]
y_samples=jnp.log(gamma_map_mean)[0,0,mask]

# Loss function to compare gamma-ray maps
def loss_forward(y_guess, y_ref):
    gamma_guess = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_guess) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))
    # gamma_ref = jnp.log(jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_ref) * y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E))
    delta = gamma_guess[0, 0, mask] - y_ref
    # delta = y_guess - y_ref

    return 100.0 * jnp.mean(delta**2)

# Function to find derivatives with respect to weight matrices and bias vectors
loss_and_grad_fun = jax.value_and_grad(
    lambda Ws, bs: loss_forward(
        network_forward(
            x_samples,
            Ws,
            bs,
            activation_functions,
        ),
        y_samples,
    ),
    argnums=(0, 1),
)

# Using optax.adam to optimize the learning rate
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init((weight_matrices, bias_vectors))

# Function for updating weights and biases with gradient descent 
@jax.jit
def update(opt_state, params):
    weights, biases = params
    loss, grads = loss_and_grad_fun(weights, biases)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss

# Training loop
start_time = time.time()
print('Start training ...')
loss_history = []
gSNR_samples = []

params = (weight_matrices, bias_vectors)
# plt.scatter(x_samples_raw, jnp.exp(y_samples_raw) * y_sol, label='Samples')
plt.scatter(x_samples_raw, jnp.exp(network_forward(x_samples, weight_matrices, bias_vectors, activation_functions)) * y_sol, label='Interation 0')
plt.plot(x_samples_raw, jnp.exp(y_grtruth) * y_sol, 'k--', label='Analytic best fit')
plt.legend(loc='upper right')
plt.savefig('Results_nn/nn_epoch_fbf_0.png')
plt.close()

for epoch in range(N_EPOCHS):
    opt_state, params, loss = update(opt_state, params)

    if (epoch + 1) % epoch_print == 0:
        print(f"epoch: {epoch+1}, loss: {loss}")
        weights, biases = params
        # plt.scatter(x_samples_raw, jnp.exp(y_samples_raw) * y_sol, label='Samples')
        plt.scatter(x_samples_raw, jnp.exp(network_forward(x_samples, weights, biases, activation_functions)) * y_sol, label='Interation %d' % (epoch + 1))
        plt.plot(x_samples_raw, jnp.exp(y_grtruth) * y_sol, 'k--', label='Analytic best fit')
        plt.legend(loc='upper right')
        plt.savefig('Results_nn/nn_epoch_fbf_%d.png' % (epoch + 1))
        plt.close()
    
    loss_history.append(loss)
    gSNR_samples.append(jnp.exp(network_forward(x_samples, params[0], params[1], activation_functions)))

weights, biases = params
y_fit = network_forward(x_samples, weights, biases, activation_functions)

gamma_fit = jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_fit)*y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E)
gamma_grtruth = jCR.func_gamma_map_gSNR(((x_samples_raw.ravel()) * 1.0e3, (jnp.exp(y_grtruth)*y_sol).ravel()), pars_prop, zeta_n, dXSdEg_Geant4, ngas_mean, drs, points_intr, E)

# Save best fit and samples
np.savez('Results_nn/gSNR_nn_from_best_fit.npz', 
            rG=np.array(x_samples_raw.ravel()), 
            gSNR=np.array(jnp.exp(y_fit).ravel())*y_sol, 
            gSNR_truth=np.array(jnp.exp(y_grtruth))*y_sol, 
            gSNR_samples=np.array(gSNR_samples)*y_sol, 
            loss_history=np.array(loss_history), 
            gamma=np.array(gamma_fit), 
            gamma_truth=np.array(gamma_grtruth))

# Save weights and biases from training 
weights_dict = {f"weight_{i}": np.array(weights[i]) for i in range(len(weights))}
biases_dict = {f"bias_{i}": np.array(biases[i]) for i in range(len(biases))}
save_dict = {**weights_dict, **biases_dict}
np.savez('nn_from_best_fit.npz', **save_dict)

print('Run time: %.2f seconds' % (time.time() - start_time))
