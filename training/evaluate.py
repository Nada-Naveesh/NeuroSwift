"""Comprehensive evaluation of NeuroSwift model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.loader import load_model
from src.config import CLASS_NAMES, DATA_PROCESSED_DIR, MODEL_WEIGHTS_PATH, REPORTS_FIGURES_DIR
from training.utils import set_seed, stratified_splits


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model on test set and return all metrics.
    """
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()
    with torch.no_grad():
        tensor_x = torch.FloatTensor(X_test).to(device)
        outputs = model(tensor_x)
        _, predicted = torch.max(outputs, 1)

    predicted = predicted.cpu().numpy()
    y_true = np.asarray(y_test)

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, predicted) * 100
    precision = precision_score(y_true, predicted, average="weighted", zero_division=0) * 100
    recall = recall_score(y_true, predicted, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_true, predicted, average="weighted", zero_division=0) * 100
    cm = confusion_matrix(y_true, predicted, labels=list(range(len(class_names))))

    # Per-class metrics
    per_class_precision = precision_score(y_true, predicted, average=None, labels=list(range(len(class_names))), zero_division=0) * 100
    per_class_recall = recall_score(y_true, predicted, average=None, labels=list(range(len(class_names))), zero_division=0) * 100
    per_class_f1 = f1_score(y_true, predicted, average=None, labels=list(range(len(class_names))), zero_division=0) * 100

    print("\n" + "=" * 60)
    print("NEUROSWIFT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Precision (Weighted): {precision:.2f}%")
    print(f"Recall (Weighted): {recall:.2f}%")
    print(f"F1-Score (Weighted): {f1:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nPer-Class Performance:")
    print(f"{'Class':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 52)
    for i, name in enumerate(class_names):
        print(f"{name:<14} {per_class_precision[i]:.2f}%      {per_class_recall[i]:.2f}%      {per_class_f1[i]:.2f}%")
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix with labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — NeuroSwift")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(X: np.ndarray, y: np.ndarray, weights: Path, out_dir: Path) -> dict:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, _, X_te, y_te = stratified_splits(X, y)
    model, used_weights = load_model(weights, device=device)

    metrics = evaluate_model(model, X_te, y_te, CLASS_NAMES)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, save_path=out_dir / "confusion_matrix.png")

    report_json = {
        "weights": str(used_weights),
        "accuracy": float(metrics["accuracy"]),
        "precision_weighted": float(metrics["precision"]),
        "recall_weighted": float(metrics["recall"]),
        "f1_weighted": float(metrics["f1"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "per_class_precision": metrics["per_class_precision"].tolist(),
        "per_class_recall": metrics["per_class_recall"].tolist(),
        "per_class_f1": metrics["per_class_f1"].tolist(),
        "n_test": int(len(y_te)),
    }
    (out_dir.parent / "evaluation.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    return report_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuroSwift Model.")
    parser.add_argument("--data", type=Path, default=DATA_PROCESSED_DIR / "trials.npz")
    parser.add_argument("--weights", type=Path, default=MODEL_WEIGHTS_PATH)
    parser.add_argument("--figures", type=Path, default=REPORTS_FIGURES_DIR)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Missing {args.data}. Run preprocessing or generate demo dataset first.")
    payload = np.load(args.data)
    evaluate(payload["X"], payload["y"], args.weights, args.figures)


if __name__ == "__main__":
    main()
