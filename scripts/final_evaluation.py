"""Final evaluation of NeuroSwift Ensemble against base paper (Lian et al., 2025)."""

import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ensemble import EnsembleModel
from models.loader import load_model
from training.config import CONFIG

CLASS_NAMES = ["Left Hand", "Right Hand", "Both Hands", "Both Feet", "Rest"]


def evaluate_final_model(model_or_ensemble, X_test, y_test, class_names=None):
    """Final evaluation of NeuroSwift model or ensemble against base paper."""
    class_names = class_names or CLASS_NAMES

    if hasattr(model_or_ensemble, "predict"):
        predictions = model_or_ensemble.predict(X_test)
    else:
        model_or_ensemble.eval()
        with torch.no_grad():
            tensor_x = torch.FloatTensor(X_test)
            outputs = model_or_ensemble(tensor_x)
            _, preds = torch.max(outputs, 1)
            predictions = preds.numpy()

    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0) * 100
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0) * 100
    cm = confusion_matrix(y_test, predictions, labels=list(range(len(class_names))))

    print("\n" + "=" * 60)
    print("NEUROSWIFT: FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    # Save figure
    fig_dir = Path("./reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"NeuroSwift Confusion Matrix (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = fig_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix figure to {cm_path}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH BASE PAPER")
    print("=" * 60)
    print("Base Paper (Lian et al., 2025): 86.34%")
    print(f"NeuroSwift (Ours):              {accuracy:.2f}%")
    diff = accuracy - 86.34
    print(f"Improvement:                    {diff:+.2f}%")

    if accuracy >= 86.34:
        print("✅ BEAT THE BASE PAPER (Lian et al., 2025)!")
    else:
        print(f"Base single model baseline: {accuracy:.2f}%. Run scripts/train_full.py to train the complete ensemble!")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def main():
    data_dir = Path("./data/processed")
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "y_test.npy"

    if not (x_test_path.exists() and y_test_path.exists()):
        # Load from X.npy or trials.npz and create test split
        x_path = data_dir / "X.npy"
        y_path = data_dir / "y.npy"
        npz_path = data_dir / "trials.npz"

        if x_path.exists() and y_path.exists():
            X, y = np.load(x_path), np.load(y_path)
        elif npz_path.exists():
            d = np.load(npz_path)
            X, y = d["X"], d["y"]
        else:
            raise FileNotFoundError("No dataset found in ./data/processed/")

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
    else:
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

    # Attempt to load 5-model ensemble first
    ensemble = EnsembleModel(CONFIG, num_models=5)
    loaded = ensemble.load_weights("./checkpoints")

    if loaded == 5:
        print(f"Successfully loaded all 5 ensemble models!")
        evaluate_final_model(ensemble, X_test, y_test)
    else:
        # Evaluate single high-performance model
        print(f"Ensemble checkpoints partially found ({loaded}/5). Loading best single model checkpoint...")
        model = load_model()
        evaluate_final_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
