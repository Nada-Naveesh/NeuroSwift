"""Train a small synthetic-data checkpoint so the Streamlit demo can run offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic import generate_dataset
from preprocessing.feature_engineer import zscore_batch
from src.config import CHECKPOINTS_DIR, DEMO_WEIGHTS_PATH, MODEL_WEIGHTS_PATH
from training.train import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroSwift on synthetic EEG (demo only).")
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--also-copy-best", action="store_true", default=True, help="Also write checkpoints/best_model_effatt.pt")
    args = parser.parse_args()

    X, y = generate_dataset(n_per_class=args.n_per_class)
    X = zscore_batch(X)
    summary = fit(X, y, epochs=args.epochs, augment_copies=1, out_path=DEMO_WEIGHTS_PATH)

    # Always ensure best_model_improved.pt and MODEL_WEIGHTS_PATH match the updated model
    if args.also_copy_best:
        MODEL_WEIGHTS_PATH.write_bytes(DEMO_WEIGHTS_PATH.read_bytes())
        improved_path = CHECKPOINTS_DIR / "best_model_improved.pt"
        improved_path.write_bytes(DEMO_WEIGHTS_PATH.read_bytes())
        root_improved = Path("best_model_improved.pt")
        root_improved.write_bytes(DEMO_WEIGHTS_PATH.read_bytes())
        print(f"Updated weights: {MODEL_WEIGHTS_PATH} and {improved_path}")

    print(f"Demo training complete. Test Accuracy: {summary['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
