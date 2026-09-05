"""Efficient Channel Attention (ECA) Module for NeuroSwift."""

import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Module
    Uses 1D convolution instead of two linear layers
    Reference: ECA-Net (Wang et al., 2020)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) or (B, C, H, W)
        y = x.mean(dim=-1, keepdim=True)
        y = self.conv(y.transpose(1, 2))
        y = self.sigmoid(y.transpose(1, 2))
        if x.dim() == 3:
            return x * y
        return x * y.unsqueeze(-1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (backward compatibility)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1)
        return x * weights
