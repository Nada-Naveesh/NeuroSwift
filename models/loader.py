"""Load NeuroSwift checkpoints for inference."""

from __future__ import annotations

from pathlib import Path
import torch

from models.neuroswift import NeuroSwiftModel
from src.config import CHECKPOINTS_DIR, DEMO_WEIGHTS_PATH, MODEL_WEIGHTS_PATH, NUM_CLASSES


def resolve_weights_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model weights not found: {candidate}")

    candidates = [
        Path("best_model_improved.pt"),
        CHECKPOINTS_DIR / "best_model_improved.pt",
        MODEL_WEIGHTS_PATH,
        DEMO_WEIGHTS_PATH,
        CHECKPOINTS_DIR / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No trained weights found. Train with `python scripts/train.py` "
        "or generate demo weights with `python scripts/train_synthetic_demo.py`."
    )


def load_model(
    path: str | Path | None = None,
    device: str | torch.device | None = None,
    num_classes: int = NUM_CLASSES,
    config: dict | None = None,
) -> tuple[NeuroSwiftModel, Path]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_path = resolve_weights_path(path)
    model = NeuroSwiftModel(config=config, num_classes=num_classes)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # Fallback to non-strict loading if minor checkpoint structure difference
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model, weights_path
