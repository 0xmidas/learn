from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(c_in, c_out, kernel_size=4, padding=1, stride=2)
        )
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, target_size, final=False):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.final: bool = final
        if not final:
            self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        x = self.conv(x)
        if self.final:
            return torch.tanh(x)
        else:
            x = self.norm(x)
            return F.leaky_relu(x, 0.2)


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim: int = latent_dim

        # Project and reshape latent vector
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(512 * 4 * 4)

        # 4x4 -> 7x7 -> 14x14 -> 28x28
        self.up1 = UpBlock(512, 256, target_size=7)
        self.up2 = UpBlock(256, 128, target_size=14)
        self.up3 = UpBlock(128, 64, target_size=28)
        self.final = UpBlock(64, 1, target_size=28, final=True)

    def forward(self, z):
        x = self.fc(z)
        x = self.bn_fc(x)
        x = F.leaky_relu(x, 0.2)
        x = rearrange(x, "b (c h w) -> b c h w", c=512, h=4, w=4)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # No batch norm in D - can destabilize training
        # Spectral norm + dropout to prevent D from overpowering G
        self.block1 = DownBlock(1, 64)
        self.block2 = DownBlock(64, 128)
        self.block3 = DownBlock(128, 256)

        # 28 -> 14 -> 7 -> 3 (kernel=4, stride=2, pad=1)
        self.fc = nn.utils.spectral_norm(nn.Linear(256 * 3 * 3, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        return self.fc(x)
