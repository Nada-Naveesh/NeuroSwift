"""
Unit tests for NEURALIS BCI Competition IV 2a pipeline modules.
Verifies processor functions, resampling, filtering, normalization,
and model shapes with 22 channels and 4 classes.
"""

import pytest
import numpy as np
import torch
from preprocessing.bci_iv_2a_processor import BCIIV2aProcessor, generate_sample_bci_data
from models.bci_iv_2a_model import BCINeuroSwiftModel, BCINeuralisModel, ECABlock, MultiScaleCNN
from training.bci_iv_2a_config import BCI_CONFIG


def test_bci_processor_resample():
    """Verify resampling from 250 Hz to 160 Hz produces correct time sample count."""
    processor = BCIIV2aProcessor(sampling_rate=250, target_sampling_rate=160, window_duration=4)
    # 4 seconds at 250 Hz = 1000 samples
    X_raw = np.random.randn(2, 22, 1000)
    X_resampled = processor.resample_signal(X_raw, original_sr=250, target_sr=160)
    assert X_resampled.shape == (2, 22, 640), f"Expected (2, 22, 640), got {X_resampled.shape}"


def test_bci_processor_bandpass_and_zscore():
    """Verify zero-phase 7-30Hz bandpass and per-channel z-score normalization."""
    processor = BCIIV2aProcessor(sampling_rate=250, target_sampling_rate=160, window_duration=4)
    X = np.random.randn(3, 22, 640)
    X_filt = processor.bandpass_filter(X, lowcut=7, highcut=30, fs=160)
    assert X_filt.shape == (3, 22, 640)

    X_norm = processor.z_score_normalize(X_filt)
    assert X_norm.shape == (3, 22, 640)
    # Mean should be close to 0 and std close to 1 across the temporal axis
    for i in range(3):
        for ch in range(22):
            assert np.isclose(X_norm[i, ch].mean(), 0.0, atol=1e-4)
            assert np.isclose(X_norm[i, ch].std(), 1.0, atol=1e-2)


def test_bci_model_forward_shape():
    """Verify BCINeuroSwiftModel accepts (B, 22, 640) and returns (B, 4)."""
    model = BCINeuroSwiftModel(BCI_CONFIG)
    model.eval()
    x = torch.randn(4, 22, 640)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"


def test_bci_model_predict_proba():
    """Verify predict_proba outputs valid softmax probabilities."""
    model = BCINeuroSwiftModel(BCI_CONFIG)
    model.eval()
    x = torch.randn(3, 22, 640)
    proba = model.predict_proba(x)
    assert proba.shape == (3, 4)
    sums = proba.sum(dim=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_neuralis_model_alias():
    """Verify BCINeuralisModel is an alias of BCINeuroSwiftModel."""
    assert BCINeuralisModel is BCINeuroSwiftModel
