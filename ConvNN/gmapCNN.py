import torch
import torch.nn as nn
import torch.nn.init as init
import h5py
import numpy as np

class gmapCNN(nn.Module):
    def __init__(self, nr: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        )
        # make spatial size stable regardless of Nx, Ny:
        self.gap = nn.AdaptiveAvgPool2d((8, 8))  # reduce to 8x8 feature map
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, nr),
        )

    def forward(self, x):
        z = self.features(x)
        z = self.gap(z)
        out = self.regressor(z)
        return out
  