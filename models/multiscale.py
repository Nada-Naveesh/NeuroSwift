"""Multi-scale 1-D convolutional branches for EEG temporal patterns."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    """
    Multi-scale feature extraction with optimal kernel sizes
    Captures EEG patterns at different time scales
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int] = [3, 5, 7]) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        branch_channels = out_channels // len(kernel_sizes)
        for k in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(),
                )
            )
        total_branch_channels = branch_channels * len(kernel_sizes)
        self.conv_combine = nn.Conv1d(total_branch_channels, out_channels, kernel_size=1)
        self.bn_combine = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.relu(self.bn_combine(self.conv_combine(x)))
        return x


class MultiScaleBranches(nn.Module):
    """Three parallel Conv1d branches (fast / medium / slow kernels)."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.fast = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.slow = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fast(x), self.medium(x), self.slow(x)], dim=1)
