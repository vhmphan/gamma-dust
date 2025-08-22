import h5py
import numpy as np
import torch
import torch.nn as nn
import gmapCNN 
from torch.utils.data import TensorDataset, DataLoader
import time

start_time = time.time()

# Load data file
with h5py.File("Data/data_gamma.h5", "r") as f:
    print("Keys:", list(f.keys()))
    keys = list(f.keys())
    rSNR = f[keys[2]][:]
    gmap = f[keys[1]][:]
    gSNR = f[keys[0]][:]

# Normalize the data
mean_gmap = np.mean(np.log10(gmap), axis=0)
std_gmap = np.std(np.log10(gmap), axis=0)
gmap = (np.log10(gmap) - mean_gmap) / std_gmap

mean_gSNR = np.mean(np.log10(gSNR), axis=0)
std_gSNR = np.std(np.log10(gSNR), axis=0)
gSNR = (np.log10(gSNR) - mean_gSNR) / std_gSNR

# Size of the output vector
nr = gSNR.shape[1] 

# Call the neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gmapCNN.gmapCNN(nr).to(device)
model = model.double() # -> Convert model to double precision

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Convert data to torch tensor
X_train = torch.tensor(gmap, dtype=torch.float64).unsqueeze(1).to(device)  # -> Unsqueeze to add channel dimension
Y_train = torch.tensor(gSNR, dtype=torch.float64).to(device)
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss_history = []

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        running_loss += loss.item()

    loss_history.append(running_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {running_loss/len(train_loader):.4f}")

# Save the model parameters 
torch.save(model.state_dict(), "gmap_weights.pth")

print('Runtime: ',time.time()-start_time,'s')