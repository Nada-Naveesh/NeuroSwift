"""Cross-validation for NeuroSwift."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from models.neuroswift import NeuroSwiftModel
from preprocessing.augment import apply_advanced_augmentation
from training.evaluate import evaluate_model
from training.train import train_with_early_stopping


def cross_validate(X, y, config, n_folds=5):
    """Perform 5-fold stratified cross-validation on motor imagery trials."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f"\n{'='*60}")
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply augmentation to training data
        if config.get("use_augmentation", True):
            X_train_aug, y_train_aug = apply_advanced_augmentation(X_train, y_train)
        else:
            X_train_aug, y_train_aug = X_train, y_train

        print(f"Training samples (after augmentation): {len(X_train_aug)}")
        print(f"Validation samples:                    {len(X_val)}")

        # Train model
        model = NeuroSwiftModel(config)
        model, _ = train_with_early_stopping(
            model, X_train_aug, y_train_aug, X_val, y_val, config
        )

        # Evaluate - evaluate_model returns tuple (acc, precision, recall, f1, cm)
        acc, precision, recall, f1, cm = evaluate_model(model, X_val, y_val)
        fold_accuracies.append(acc)

        print(f"\nFold {fold+1} Results:")
        print(f"  Accuracy:  {acc:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall:    {recall:.2f}%")
        print(f"  F1-Score:  {f1:.2f}%")

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}%")
    print(f"Std Dev:          {np.std(fold_accuracies):.2f}%")
    print(f"Min:              {np.min(fold_accuracies):.2f}%")
    print(f"Max:              {np.max(fold_accuracies):.2f}%")
    print(f"{'='*60}\n")

    return fold_accuracies
