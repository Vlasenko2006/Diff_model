#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:03:24 2025

@author: andrey
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from PIL import Image

def save_generated_images(samples, folder="generated_images", filename="output.png"):
    """
    Saves generated images into a specified folder.

    Args:
        samples (numpy array): Array of images to save.
        folder (str): Target folder to save images.
        filename (str): Name of the output image file.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Plot and save the figure
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")

    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory

    print(f"Image saved to {save_path}")

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 300 * 3 * 3 * 3 * 2  # Reduced number of diffusion steps
beta_start, beta_end = 1e-4, 0.02  # Noise schedule
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # Cumulative product

# FORWARD DIFFUSION PROCESS
def add_noise(x_0, t):
    """ Add noise to an image at time step t """
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    noise = torch.randn_like(x_0)
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise, noise

# SIMPLE UNET ARCHITECTURE
class SkipUNet(nn.Module):
    def __init__(self, nx=28, ny=28, dropout=0.3):
        super().__init__()

        self.dropout = dropout
        self.nx = nx
        self.ny = ny

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, nx, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nx, nx * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nx * 2, nx * 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(nx * 4, nx * 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )  # New layer

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(nx * 8, nx * 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(nx * 8, nx * 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )  # Extended Middle with additional layers

        # Decoder with Skip Connections
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(nx * 8 + nx * 8, nx * 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )  # Skip from enc4
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(nx * 4 + nx * 4, nx * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )  # Skip from enc3
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(nx * 2 + nx * 2, nx, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )  # Skip from enc2
        self.dec1 = nn.Sequential(
            nn.Conv2d(nx + nx, 1, 3, padding=1),
            nn.Dropout(self.dropout)
        )  # Skip from enc1

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)  # New layer

        # Middle layer
        m = self.middle(e4)

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([m, e4], dim=1))  # Skip connection from e4
        d3 = self.dec3(torch.cat([self.center_crop(d4, e3), e3], dim=1))  # Skip connection from e3
        d2 = self.dec2(torch.cat([self.center_crop(d3, e2), e2], dim=1))  # Skip connection from e2
        d1 = self.dec1(torch.cat([self.center_crop(d2, e1), e1], dim=1))  # Skip connection from e1

        return d1

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size.size(2)) // 2
        diff_x = (layer_width - target_size.size(3)) // 2
        return layer[:, :, diff_y: (diff_y + target_size.size(2)), diff_x: (diff_x + target_size.size(3))]

# TRAINING THE DIFFUSION MODEL
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

nx, ny = 28, 28  # Image dimensions
epochs = 300
patience = 30
dropout = 0.3

model = SkipUNet(nx=nx, ny=ny, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.SmoothL1Loss()  # Consider using SmoothL1Loss instead of MSELoss for better convergence

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)

dropout_patience_counter = 0
previous_avg_loss = float('inf')

for epoch in range(epochs):
    epoch_loss = 0  # Track total loss for the epoch
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for x_0, _ in progress_bar:
        x_0 = x_0.to(device)
        t = torch.randint(0, T, (x_0.shape[0],), device=device)  # Random time steps
        x_t, noise = add_noise(x_0, t)

        optimizer.zero_grad()
        predicted_noise = model(x_t)  # Predict noise
        loss = criterion(predicted_noise, noise)  # Loss calculation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # Update tqdm with current loss

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # Step the scheduler
    scheduler.step()
    plateau_scheduler.step(avg_loss)

    # Reduce dropout rate if learning rate is reduced
    if optimizer.param_groups[0]['lr'] < 1e-4:
        if previous_avg_loss - avg_loss < 0.03 * previous_avg_loss:
            dropout_patience_counter += 1
        else:
            dropout_patience_counter = 0
        previous_avg_loss = avg_loss

        if dropout_patience_counter >= patience:
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = max(0.001, module.p * 0.75)
            print(f"Reduced dropout rate to {model.middle[3].p}")
            dropout_patience_counter = 0

# SAMPLING FROM NOISE
@torch.no_grad()
def sample(num_samples=16):
    """ Generate images from pure noise """
    x_t = torch.randn((num_samples, 1, 28, 28)).to(device)  # Start from pure noise
    for t in reversed(range(T)):
        z = torch.randn_like(x_t) if t > 0 else 0  # No noise at step t=0
        predicted_noise = model(x_t)
        x_t = (x_t - betas[t] * predicted_noise) / torch.sqrt(alphas[t]) + torch.sqrt(betas[t]) * z
    return x_t

# Generate images
samples = sample().cpu().numpy().squeeze()

# Plot results
save_generated_images(samples, folder="generated_images", filename="output5.png")
