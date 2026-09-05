"""Inference and processing utilities for NeuroSwift demo."""

from pathlib import Path
import numpy as np
import torch

from src.config import CLASS_NAMES, NUM_CLASSES


def softmax_to_result(proba: np.ndarray) -> dict:
    proba = np.asarray(proba, dtype=np.float64)
    idx = int(proba.argmax())
    return {
        "label": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown",
        "index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {CLASS_NAMES[i]: float(proba[i]) for i in range(min(len(CLASS_NAMES), len(proba)))},
    }


def predict_trial(model, trial: np.ndarray, device=None) -> dict:
    device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
    model.eval()
    if trial.ndim == 2:
        x = torch.FloatTensor(trial).unsqueeze(0).to(device)
    else:
        x = torch.FloatTensor(trial).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=-1)[0].cpu().numpy()
    return softmax_to_result(probs)
