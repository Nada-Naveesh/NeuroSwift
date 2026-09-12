"""Hyperparameter tuning for NeuroSwift."""

import itertools
import numpy as np
from models.neuroswift import NeuroSwiftModel
from training.evaluate import evaluate_model
from training.train import train_with_early_stopping


def hyperparameter_tuning(X_train, y_train, X_val, y_val, base_config):
    """
    Grid search for optimal hyperparameters across learning rates,
    batch sizes, dropout rates, and multi-scale temporal kernel sizes.
    """
    param_grid = {
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
        "dropout_rate": [0.3, 0.5],
        "kernel_sizes": [[3, 5, 7], [3, 7, 11]],
    }

    best_acc = 0.0
    best_params = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Testing {len(combinations)} hyperparameter combinations...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        config = base_config.copy()
        config.update(params)

        print(f"\nCombination {i+1}/{len(combinations)}: {params}")

        try:
            model = NeuroSwiftModel(config)
            # Use quick tuning epochs
            tune_config = config.copy()
            tune_config["epochs"] = min(config.get("epochs", 100), 20)
            tune_config["early_stopping_patience"] = 8

            model, _ = train_with_early_stopping(
                model, X_train, y_train, X_val, y_val, tune_config
            )

            # Robust extraction of accuracy
            result = evaluate_model(model, X_val, y_val)
            if isinstance(result, dict):
                acc = float(result["accuracy"])
            elif isinstance(result, (tuple, list)):
                acc = float(result[0])
            else:
                acc = float(result)

            print(f"  Validation Accuracy: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_params = params
                print(f"  ✅ New best: {acc:.2f}%")
        except Exception as e:
            print(f"  ❌ Error in combination {i+1}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best Parameters: {best_params}")
    print(f"{'='*60}")

    return best_params
