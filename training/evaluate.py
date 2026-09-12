"""Evaluation for NeuroSwift."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch

from training.config import CLASS_NAMES


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate model on test set.
    Returns: (accuracy, precision, recall, f1, confusion_matrix)
    """
    class_names = class_names or CLASS_NAMES
    model.eval()

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    with torch.no_grad():
        tensor_x = torch.FloatTensor(X_test).to(device) if not isinstance(X_test, torch.Tensor) else X_test.to(device).float()
        outputs = model(tensor_x)
        _, predicted = torch.max(outputs, 1)

    predicted = predicted.cpu().numpy()
    y_true = np.asarray(y_test)

    accuracy = accuracy_score(y_true, predicted) * 100
    precision = precision_score(y_true, predicted, average="weighted", zero_division=0) * 100
    recall = recall_score(y_true, predicted, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_true, predicted, average="weighted", zero_division=0) * 100
    cm = confusion_matrix(y_true, predicted, labels=list(range(len(class_names))))

    return accuracy, precision, recall, f1, cm


def evaluate_model_dict(model, X_test, y_test, class_names=None):
    """Evaluate model and return metrics as a structured dictionary."""
    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test, class_names)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot and optionally save the confusion matrix heatmap."""
    class_names = class_names or CLASS_NAMES
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("NeuroSwift Confusion Matrix")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    from models.loader import load_model
    x_path = Path("./data/processed/X_test.npy")
    y_path = Path("./data/processed/y_test.npy")
    if not (x_path.exists() and y_path.exists()):
        x_path = Path("./data/processed/X.npy")
        y_path = Path("./data/processed/y.npy")

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        model = load_model()
        acc, prec, rec, f1, cm = evaluate_model(model, X, y)
        print(f"Test Accuracy:  {acc:.2f}%")
        print(f"Precision:      {prec:.2f}%")
        print(f"Recall:         {rec:.2f}%")
        print(f"F1-Score:       {f1:.2f}%")
        plot_confusion_matrix(cm, save_path="./reports/figures/confusion_matrix.png")
    else:
        print("No test data found in ./data/processed/")
