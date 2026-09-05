"""Unit tests for preprocessing helpers and sliding window augmentation."""

from __future__ import annotations

import numpy as np

from preprocessing.augment import apply_window_augmentation, sliding_window_augmentation
from preprocessing.feature_engineer import ensure_trial_shape, zscore_trial
from preprocessing.signal_processor import pad_or_trim_channels
from preprocessing.trial_extractor import map_annotation_to_class
from src.config import N_CHANNELS, N_TIMES


def test_map_left_right_runs() -> None:
    assert map_annotation_to_class("T0", 4) == "Rest"
    assert map_annotation_to_class("T1", 4) == "Left Hand"
    assert map_annotation_to_class("T2", 4) == "Right Hand"
    assert map_annotation_to_class("T1", 6) == "Both Hands"
    assert map_annotation_to_class("T2", 6) == "Feet"


def test_zscore_trial() -> None:
    rng = np.random.default_rng(0)
    trial = rng.normal(4, 2, size=(N_CHANNELS, N_TIMES)).astype(np.float32)
    z = zscore_trial(trial)
    assert np.allclose(z.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(z.std(axis=-1), 1, atol=1e-4)


def test_ensure_trial_shape_resample() -> None:
    trial = np.ones((64, 320), dtype=np.float32)
    out = ensure_trial_shape(trial)
    assert out.shape == (N_CHANNELS, N_TIMES)


def test_pad_channels() -> None:
    data = np.ones((32, 100), dtype=np.float32)
    out = pad_or_trim_channels(data)
    assert out.shape == (N_CHANNELS, 100)
    assert out[:32].sum() == 32 * 100
    assert out[32:].sum() == 0


def test_sliding_window_augmentation() -> None:
    signal = np.random.randn(64, 640)
    # window_size=500ms = 80 samples at 160Hz, stride=75ms = 12 samples at 160Hz
    windows = sliding_window_augmentation(signal, window_size=500, stride=75, sampling_rate=160)
    assert len(windows) > 0
    assert windows[0].shape == (64, 80)


def test_apply_window_augmentation() -> None:
    data = np.random.randn(5, 64, 640)
    labels = np.array([0, 1, 2, 3, 4])
    aug_data, aug_labels = apply_window_augmentation(data, labels, window_size=500, stride=75, sampling_rate=160)
    assert len(aug_data) == len(aug_labels)
    assert aug_data.shape[1:] == (64, 80)
