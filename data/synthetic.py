"""Class-conditional realistic EEG trial generation for demos and offline evaluation."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from data.augment import augment_trial
from src.config import CLASS_NAMES, N_CHANNELS, N_TIMES, NUM_CLASSES, RANDOM_SEED, SAMPLING_RATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOTYPES_PATH = PROJECT_ROOT / "data" / "synthetic_prototypes.npz"
PROCESSED_X_PATH = PROJECT_ROOT / "data" / "processed" / "X.npy"
PROCESSED_Y_PATH = PROJECT_ROOT / "data" / "processed" / "y.npy"


def _fallback_mu_trial(label: int, rng: np.random.Generator) -> np.ndarray:
    """Mathematical fallback if no reference data is found."""
    t = np.arange(N_TIMES) / SAMPLING_RATE
    trial = rng.normal(0, 0.35, size=(N_CHANNELS, N_TIMES)).astype(np.float32)
    C3, CZ, C4 = 8, 10, 12
    phase = rng.uniform(0, 2 * np.pi)
    mu = (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t + phase)) * np.sin(2 * np.pi * 11.0 * t + phase)
    if label == 0:  # Left Hand -> C4
        trial[C4] += 1.2 * mu
    elif label == 1:  # Right Hand -> C3
        trial[C3] += 1.2 * mu
    elif label == 2:  # Both Hands -> C3 + C4
        trial[C3] += 0.9 * mu
        trial[C4] += 0.9 * mu
    elif label == 3:  # Feet -> Cz
        trial[CZ] += 1.3 * mu
    else:  # Rest
        trial *= 0.5
    return trial.astype(np.float32)


def generate_trial(label: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate one realistic (64, 640) trial for a given motor imagery class.
    Uses class reference distributions with dynamic EEG time-domain augmentation.
    """
    rng = rng or np.random.default_rng()

    # 1. Try loading from bundled prototype archive
    if PROTOTYPES_PATH.exists():
        try:
            data = np.load(str(PROTOTYPES_PATH))
            key = f"class_{label}"
            if key in data:
                candidates = data[key]
                base_trial = candidates[rng.integers(0, len(candidates))]
                return augment_trial(base_trial, rng=rng)
        except Exception:
            pass

    # 2. Try loading from processed dataset
    if PROCESSED_X_PATH.exists() and PROCESSED_Y_PATH.exists():
        try:
            X = np.load(str(PROCESSED_X_PATH))
            y = np.load(str(PROCESSED_Y_PATH))
            indices = np.where(y == label)[0]
            if len(indices) > 0:
                base_trial = X[rng.choice(indices)]
                return augment_trial(base_trial, rng=rng)
        except Exception:
            pass

    # 3. Fallback to analytical model
    return _fallback_mu_trial(label, rng)


def generate_dataset(
    n_per_class: int = 80,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced multi-class dataset."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in range(NUM_CLASSES):
        for _ in range(n_per_class):
            xs.append(generate_trial(label, rng))
            ys.append(label)
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order]


def class_name(index: int) -> str:
    return CLASS_NAMES[index]
