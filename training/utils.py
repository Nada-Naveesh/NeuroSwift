"""Shared training helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, RANDOM_SEED, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT


class EEGTrialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rel_val = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=rel_val, random_state=seed, stratify=y_tv
    )
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(EEGTrialDataset(X, y), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def class_weights(y: Sequence[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
