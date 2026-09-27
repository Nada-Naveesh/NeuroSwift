"""
NEURALIS: Multi-Scale 1D-CNN with Efficient Channel Attention for BCI IV 2a
Adapted for 22 EEG channels and 4-class motor imagery (Left, Right, Feet, Tongue).
"""

import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) Module for 1D Temporal Features.
    Uses adaptive 1D convolution instead of FC bottleneck to preserve channel topology.
    Reference: ECA-Net (Wang et al., 2020)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = x.mean(dim=-1, keepdim=True)        # (B, C, 1)
        y = self.conv(y.transpose(1, 2))        # (B, 1, C)
        y = self.sigmoid(y.transpose(1, 2))     # (B, C, 1)
        if x.dim() == 3:
            return x * y
        return x * y.unsqueeze(-1)


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale Feature Extractor adapted for 22 EEG channels.
    Parallel 1D temporal convolutions with kernels [3, 5, 7] capture
    high-frequency beta transients and sustained mu rhythms concurrently.
    """

    def __init__(self, in_channels: int = 22, out_channels: int = 192, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_channels = out_channels // len(kernel_sizes)

        for k in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU()
                )
            )

        self.conv_combine = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn_combine = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.relu(self.bn_combine(self.conv_combine(x)))
        return x


class BCINeuroSwiftModel(nn.Module):
    """
    NEURALIS Architecture adapted for BCI Competition IV 2a (22 channels, 4 classes).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        in_channels = config.get('input_channels', 22)
        num_classes = config.get('num_classes', 4)
        kernel_sizes = config.get('kernel_sizes', [3, 5, 7])

        # Multi-scale feature extraction (22 input channels -> 192 feature channels)
        self.multiscale = MultiScaleCNN(
            in_channels=in_channels,
            out_channels=192,
            kernel_sizes=kernel_sizes
        )

        # Efficient Channel Attention
        self.attention = ECABlock(channels=192)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head (4 motor imagery classes)
        self.fc1 = nn.Linear(192, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 22, 640)
        x = self.multiscale(x)
        x = self.attention(x)
        x = self.gap(x).squeeze(-1)

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class index."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=-1)


# Neuralis alias for BCINeuroSwiftModel
BCINeuralisModel = BCINeuroSwiftModel
