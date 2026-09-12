"""Complete End-to-End Training Pipeline for NeuroSwift with 5-Model Ensemble."""

import os
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ensemble import EnsembleModel
from training.config import CONFIG
from training.cross_validate import cross_validate
from training.hypertune import hyperparameter_tuning
from preprocessing.augment import apply_advanced_augmentation


def load_dataset():
    data_dir = Path("./data/processed")
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y

    npz_path = data_dir / "trials.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        np.save(x_path, X)
        np.save(y_path, y)
        return X, y

    raise FileNotFoundError("No processed dataset found in ./data/processed/")


def main():
    print("=" * 60)
    print("NEUROSWIFT: FULL 5-MODEL ENSEMBLE TRAINING PIPELINE")
    print("=" * 60)

    X, y = load_dataset()
    print(f"Loaded {len(X)} total trials with shape {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split into Train+Val and Held-out Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )

    out_dir = Path("./data/processed")
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    print(f"Saved held-out test split ({len(X_test)} trials) to data/processed/X_test.npy")

    # Step 1: Hyperparameter tuning (sample subset for fast convergence)
    print("\n[1/4] Hyperparameter Tuning...")
    best_params = hyperparameter_tuning(
        X_train[:400], y_train[:400], X_val[:100], y_val[:100], CONFIG, max_evals=4
    )
    CONFIG.update(best_params)

    # Step 2: 5-Fold Cross-Validation
    print("\n[2/4] 5-Fold Stratified Cross-Validation...")
    cv_results = cross_validate(X_train_val, y_train_val, CONFIG, n_folds=5)

    # Step 3: Train 5-Model Ensemble with dynamic augmentations
    print("\n[3/4] Training 5-Model Ensemble...")
    ensemble = EnsembleModel(CONFIG, num_models=5)
    ensemble.train_all(X_train_val, y_train_val, X_val, y_val, checkpoint_dir="./checkpoints")

    # Step 4: Final Evaluation on Held-out Test Split
    print("\n[4/4] Evaluating Ensemble on Held-Out Test Set...")
    predictions = ensemble.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    test_acc = accuracy_score(y_test, predictions) * 100
    test_f1 = f1_score(y_test, predictions, average="weighted") * 100

    print(f"\n" + "=" * 60)
    print(f"✅ FINAL ENSEMBLE TEST ACCURACY: {test_acc:.2f}%")
    print(f"✅ FINAL ENSEMBLE TEST F1-SCORE: {test_f1:.2f}%")
    print("=" * 60)

    np.save(out_dir / "ensemble_predictions.npy", predictions)
    print("Pipeline completed successfully! Models saved to ./checkpoints/")


if __name__ == "__main__":
    main()
