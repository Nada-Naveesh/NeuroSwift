"""Time-domain EEG augmentations and sliding window segmentation."""

from __future__ import annotations

import numpy as np

from preprocessing.augment import (
    add_gaussian_noise,
    advanced_augmentation,
    amplitude_scale,
    apply_advanced_augmentation,
    apply_window_augmentation,
    sliding_window_augmentation,
)

__all__ = [
    "sliding_window_augmentation",
    "apply_window_augmentation",
    "jitter",
    "scale",
    "time_shift",
    "augment_trial",
    "augment_dataset",
    "add_gaussian_noise",
    "amplitude_scale",
    "advanced_augmentation",
    "apply_advanced_augmentation",
]


def jitter(trial: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=trial.shape).astype(np.float32)
    return (trial + noise).astype(np.float32)


def scale(trial: np.ndarray, sigma: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    factor = rng.normal(1.0, sigma)
    return (trial * np.float32(factor)).astype(np.float32)


def time_shift(trial: np.ndarray, max_shift: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(trial, shift, axis=-1).astype(np.float32)


def augment_trial(
    trial: np.ndarray,
    rng: np.random.Generator | None = None,
    p_jitter: float = 0.8,
    p_scale: float = 0.8,
    p_shift: float = 0.5,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = trial.astype(np.float32, copy=True)
    if rng.random() < p_jitter:
        out = jitter(out, rng=rng)
    if rng.random() < p_scale:
        out = scale(out, rng=rng)
    if rng.random() < p_shift:
        out = time_shift(out, rng=rng)
    return out


def augment_dataset(X: np.ndarray, y: np.ndarray, copies: int = 1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = [X]
    ys = [y]
    for _ in range(copies):
        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
        xs.append(np.stack([augment_trial(t, rng=child) for t in X], axis=0))
        ys.append(y.copy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
