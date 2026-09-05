"""Model unit tests for NeuroSwift architecture, ECA, and MultiScaleCNN."""

from __future__ import annotations

import torch

from models.attention import ECABlock
from models.multiscale import MultiScaleCNN
from models.neuroswift import NeuroSwiftModel
from src.config import N_CHANNELS, N_TIMES, NUM_CLASSES
from training.config import CONFIG


def test_eca_block_shape() -> None:
    eca = ECABlock(channels=192, gamma=2, b=1)
    x = torch.randn(2, 192, 640)
    out = eca(x)
    assert out.shape == (2, 192, 640)


def test_multiscale_cnn_shape() -> None:
    ms = MultiScaleCNN(in_channels=64, out_channels=192, kernel_sizes=[3, 5, 7])
    x = torch.randn(2, 64, 640)
    out = ms(x)
    assert out.shape == (2, 192, 640)


def test_neuroswift_forward_shape() -> None:
    model = NeuroSwiftModel(config=CONFIG)
    model.eval()
    x = torch.randn(4, N_CHANNELS, N_TIMES)
    logits = model(x)
    assert logits.shape == (4, NUM_CLASSES)


def test_predict_proba_sums_to_one() -> None:
    model = NeuroSwiftModel()
    x = torch.randn(2, N_CHANNELS, N_TIMES)
    proba = model.predict_proba(x)
    assert proba.shape == (2, NUM_CLASSES)
    assert torch.allclose(proba.sum(dim=-1), torch.ones(2), atol=1e-5)
