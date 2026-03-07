import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_embedding import SinusoidalPositionEmbeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act1(h)

        # 1. Project time embedding: [Batch, time_emb_dim] -> [Batch, out_channels]
        time_emb = self.time_mlp(t_emb)
        
        # 2. Reshape so it can broadcast across the spatial dimensions (Height, Width)
        # Turns [Batch, out_channels] into [Batch, out_channels, 1, 1]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # 3. Add the time embedding as a bias term to the intermediate activations
        h = h + time_emb 

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act2(h)

        # Add the original input (passed through the shortcut) to the final output
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        time_dim = 128
        self.time_emb = SinusoidalPositionEmbeddings(time_dim)

        self.enc1 = ResidualBlock(1, 32, time_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ResidualBlock(32, 64, time_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(64, 128, time_dim)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128, 64, time_dim) 
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, 32, time_dim)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        enc1 = self.enc1(x, t_emb)                      # Shape: [Batch, 32, 28, 28] -> SKIP 1
        p1 = self.pool1(enc1)                           # Shape: [Batch, 32, 14, 14]
        
        enc2 = self.enc2(p1, t_emb)                     # Shape: [Batch, 64, 14, 14] -> SKIP 2
        p2 = self.pool2(enc2)                           # Shape: [Batch, 64, 7, 7]

        bottleneck = self.bottleneck(p2, t_emb)         # Shape: [Batch, 128, 7, 7]

        up2 = self.upconv2(bottleneck)                  # Shape: [Batch, 64, 14, 14]
        cat2 = torch.cat((up2, enc2), dim=1)            # Concatenate 64 + 64 = 128
        dec2 = self.dec2(cat2, t_emb)                   # Shape: [Batch, 64, 14, 14]

        up1 = self.upconv1(dec2)                        # Shape: [Batch, 32, 28, 28]
        cat1 = torch.cat((up1, enc1), dim=1)            # Concatenate 32 + 32 = 64
        dec1 = self.dec1(cat1, t_emb)                   # Shape: [Batch, 32, 28, 28]

        out = self.out(dec1)                            # Shape: [Batch, 1, 28, 28]
        return out