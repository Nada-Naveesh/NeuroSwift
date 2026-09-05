"""Per-trial z-score and array shape helpers."""

from __future__ import annotations

import numpy as np

from src.config import N_CHANNELS, N_TIMES


def zscore_trial(trial: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    trial = np.asarray(trial, dtype=np.float32)
    mean = trial.mean(axis=-1, keepdims=True)
    std = trial.std(axis=-1, keepdims=True)
    return (trial - mean) / (std + eps)


def zscore_batch(X: np.ndarray) -> np.ndarray:
    return np.stack([zscore_trial(t) for t in X], axis=0)


def ensure_trial_shape(trial: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial, dtype=np.float32)
    if trial.ndim == 1:
        raise ValueError("EEG trial must be 2-D (channels, times)")
    if trial.ndim == 3 and trial.shape[0] == 1:
        trial = trial[0]
    if trial.shape[0] != N_CHANNELS and trial.shape[-1] == N_CHANNELS:
        trial = np.moveaxis(trial, -1, 0)
    n_ch, n_t = trial.shape[0], trial.shape[-1]
    out = np.zeros((N_CHANNELS, N_TIMES), dtype=np.float32)
    ch = min(n_ch, N_CHANNELS)
    if n_t == N_TIMES:
        out[:ch] = trial[:ch, :N_TIMES]
        return out
    old_idx = np.linspace(0, 1, n_t)
    new_idx = np.linspace(0, 1, N_TIMES)
    for i in range(ch):
        out[i] = np.interp(new_idx, old_idx, trial[i]).astype(np.float32)
    return out
