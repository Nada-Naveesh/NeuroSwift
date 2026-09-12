"""Stratified Cross-Validation for NeuroSwift."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.neuroswift import NeuroSwiftModel
from preprocessing.augment import apply_advanced_augmentation
from training.evaluate import evaluate_model
from training.train import train_with_early_stopping


def cross_validate(X, y, config, n_folds=5):
    """
    Perform 5-fold stratified cross-validation on motor imagery trials.
    Returns array of fold validation accuracies.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    print("=" * 60)
    print(f"STARTING {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply augmentation to training split if configured
        if config.get("use_augmentation", True):
            print(f"Applying data augmentation on {len(X_train)} training trials...")
            X_train_aug, y_train_aug = apply_advanced_augmentation(X_train, y_train)
        else:
            X_train_aug, y_train_aug = X_train, y_train

        print(f"Training on {len(X_train_aug)} trials, validating on {len(X_val)} trials...")

        # Initialize fresh NeuroSwiftModel
        model = NeuroSwiftModel(config)
        model, _ = train_with_early_stopping(
            model, X_train_aug, y_train_aug, X_val, y_val, config
        )

        # Evaluate on validation split
        res = evaluate_model(model, X_val, y_val)
        acc = res[0] if isinstance(res, (tuple, list)) else res
        fold_accuracies.append(acc)

        print(f"Fold {fold+1} Accuracy: {acc:.2f}%")

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}%")
    print(f"Std Dev:          {np.std(fold_accuracies):.2f}%")
    print(f"Min Accuracy:     {np.min(fold_accuracies):.2f}%")
    print(f"Max Accuracy:     {np.max(fold_accuracies):.2f}%")

    return fold_accuracies
