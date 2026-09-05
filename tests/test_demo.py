"""Demo inference helper tests (no Streamlit runtime)."""

from __future__ import annotations

import numpy as np

from demo.utils import softmax_to_result
from src.config import CLASS_NAMES


def test_softmax_to_result() -> None:
    proba = np.array([0.05, 0.1, 0.7, 0.1, 0.05], dtype=np.float32)
    result = softmax_to_result(proba)
    assert result["label"] == CLASS_NAMES[2]
    assert abs(result["confidence"] - 0.7) < 1e-6
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
