"""Data Augmentation modules for NeuroSwift EEG Signals."""

import numpy as np


def add_gaussian_noise(signal, noise_level=0.05):
    """Add Gaussian noise to signal."""
    noise = np.random.randn(*signal.shape) * noise_level
    return (signal + noise).astype(np.float32)


def time_shift(signal, shift_max=10):
    """Shift signal in time along the last axis."""
    shift = np.random.randint(-shift_max, shift_max + 1)
    return np.roll(signal, shift, axis=-1).astype(np.float32)


def amplitude_scale(signal, scale_range=(0.9, 1.1)):
    """Scale signal amplitude uniformly."""
    scale = np.random.uniform(*scale_range)
    return (signal * scale).astype(np.float32)


def sliding_window_augmentation(signal, window_size=500, stride=75, sampling_rate=160):
    """Create multiple overlapping windows from a single trial."""
    window_samples = int(window_size * sampling_rate / 1000)
    stride_samples = int(stride * sampling_rate / 1000)

    windows = []
    total_len = signal.shape[-1]
    if total_len < window_samples:
        return [signal]

    for start in range(0, total_len - window_samples + 1, stride_samples):
        window = signal[:, start : start + window_samples]
        windows.append(window.astype(np.float32))

    return windows


def apply_window_augmentation(data, labels, window_size=500, stride=75, sampling_rate=160):
    """Apply sliding window augmentation across an entire dataset."""
    augmented_data = []
    augmented_labels = []

    for signal, label in zip(data, labels):
        windows = sliding_window_augmentation(
            signal, window_size=window_size, stride=stride, sampling_rate=sampling_rate
        )
        augmented_data.extend(windows)
        augmented_labels.extend([label] * len(windows))

    return np.array(augmented_data, dtype=np.float32), np.array(augmented_labels, dtype=np.int64)


def advanced_augmentation(signal, label):
    """Apply multiple augmentation techniques to generate diverse variations."""
    augmented = []
    labels = []

    # 1. Original trial
    augmented.append(signal.astype(np.float32))
    labels.append(label)

    # 2. Gaussian noise injection
    augmented.append(add_gaussian_noise(signal, noise_level=0.03))
    labels.append(label)

    # 3. Temporal shifting
    augmented.append(time_shift(signal, shift_max=8))
    labels.append(label)

    # 4. Amplitude scaling
    augmented.append(amplitude_scale(signal, scale_range=(0.92, 1.08)))
    labels.append(label)

    # 5. Combined: Temporal shift + slight jitter
    combined = add_gaussian_noise(time_shift(signal, 5), noise_level=0.02)
    augmented.append(combined)
    labels.append(label)

    return augmented, labels


def apply_advanced_augmentation(X, y):
    """Apply advanced augmentation across an entire batch or dataset."""
    augmented_X = []
    augmented_y = []

    for signal, label in zip(X, y):
        aug_signals, aug_labels = advanced_augmentation(signal, label)
        augmented_X.extend(aug_signals)
        augmented_y.extend(aug_labels)

    return np.array(augmented_X, dtype=np.float32), np.array(augmented_y, dtype=np.int64)
