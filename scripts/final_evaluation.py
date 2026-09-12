"""Final Evaluation for NeuroSwift Ensemble against Base Paper (Lian et al., 2025)."""

import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from models.ensemble import EnsembleModel
from training.config import CONFIG, CLASS_NAMES


def evaluate_final_model(ensemble, X_test, y_test, class_names=None):
    """Final evaluation of ensemble model using soft probability averaging."""
    class_names = class_names or CLASS_NAMES

    predictions = ensemble.predict_soft(X_test)
    probs = ensemble.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0) * 100
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0) * 100
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 60)
    print("NEUROSWIFT: FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    fig_dir = Path("./reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"NeuroSwift Confusion Matrix (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig_path = fig_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix figure to {fig_path}")

    # Comparison with base paper
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASE PAPER")
    print("=" * 60)
    print(f"Base Paper (Lian et al., 2025): 86.34%")
    print(f"NeuroSwift (Ours):              {accuracy:.2f}%")
    diff = accuracy - 86.34
    print(f"Improvement:                    {diff:+.2f}%")

    if accuracy >= 86.34:
        print("✅ BEAT THE BASE PAPER!")
    else:
        print("⚠️ Training progressing toward target.")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    data_dir = Path("./data/processed")
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "y_test.npy"

    if not (x_test_path.exists() and y_test_path.exists()):
        print("❌ Test data not found! Creating test split from processed dataset...")
        x_full = data_dir / "X.npy"
        y_full = data_dir / "y.npy"
        if not (x_full.exists() and y_full.exists()):
            print("❌ No dataset found in ./data/processed/. Run preprocessing first.")
            sys.exit(1)
        from sklearn.model_selection import train_test_split
        X = np.load(x_full)
        y = np.load(y_full)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
    else:
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

    print(f"Loaded test set: {len(X_test)} trials")
    print(f"Class distribution: {np.bincount(y_test)}")

    # Check if ensemble models exist
    ensemble = EnsembleModel(CONFIG, num_models=5)
    ensemble.load_models("./checkpoints")

    class_names = ["Left Hand", "Right Hand", "Both Hands", "Both Feet", "Rest"]
    results = evaluate_final_model(ensemble, X_test, y_test, class_names)
