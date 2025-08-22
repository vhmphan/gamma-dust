import h5py
import numpy as np
import torch
import torch.nn as nn
import gmapCNN 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import scipy as sp

# Plot the source distribution
def plot_gSNR(fgname, gSNR_test):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
 
    N_test = len(gSNR_test)
    nr = gSNR_test[0].shape[0] // 2

    for i in range(N_test):
        gSNR1 = gSNR_test[i][:nr]
        gSNR2 = gSNR_test[i][nr:]

        ax.plot(rSNR, gSNR1, 'r-', alpha=1)
        ax.plot(rSNR, gSNR2, 'k:', alpha=1)

    ax.set_ylabel(r'$g_{\rm SNR}\, {\rm (pc^{-2})}$')
    ax.set_xlabel(r'$R\, {\rm (pc)}$')

    ax.legend(loc='upper right', prop={"size":22})

    fig.tight_layout()
    fig.savefig("fg_gSNR_%s.png" % fgname, dpi=400)

# Load data file
with h5py.File("Data/data_gamma.h5", "r") as f:
    print("Keys:", list(f.keys()))
    keys = list(f.keys())
    rSNR = f[keys[2]][:]
    gmap = f[keys[1]][:]
    gSNR = f[keys[0]][:]

# Compute again the mean and std of the data for model prediction
mean_gmap, mean_gSNR = np.mean(np.log10(gmap), axis=0), np.mean(np.log10(gSNR), axis=0)
std_gmap, std_gSNR = np.std(np.log10(gmap), axis=0), np.std(np.log10(gSNR), axis=0)
gmap = (np.log10(gmap) - mean_gmap) / std_gmap
gSNR = (np.log10(gSNR) - mean_gSNR) / std_gSNR

# Reload the model and switch to eval mode for inference
nr = gSNR.shape[1] # -> Size of the output vector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gmapCNN.gmapCNN(nr).to(device)
model = model.double()
model.load_state_dict(torch.load("gmap_weights.pth", map_location="cpu"))
model.eval()

X_train = torch.tensor(gmap, dtype=torch.float64).unsqueeze(1).to(device)  # -> Unsqueeze to add channel dimension
Y_train = torch.tensor(gSNR, dtype=torch.float64).to(device)

N_data = gSNR.shape[0] # -> Size of data
N_test = 50 # -> Number of test samples
indices = torch.randperm(N_data)[:N_test]
gSNR_test = []

with torch.no_grad():
    for i in range(N_test):

        output = model(X_train[indices[i]].unsqueeze(0))

        gSNR1 = 10**(output.numpy()[0] * std_gSNR + mean_gSNR)
        gSNR2 = 10**(Y_train[indices[i]].numpy() * std_gSNR + mean_gSNR)

        gSNR_test.append(np.concatenate((gSNR1, gSNR2)))

plot_gSNR("nn", gSNR_test)