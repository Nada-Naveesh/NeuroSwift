"""Hyperparameter Tuning module for NeuroSwift."""

import itertools
import numpy as np
from models.neuroswift import NeuroSwiftModel
from training.evaluate import evaluate_model
from training.train import train_with_early_stopping


def hyperparameter_tuning(X_train, y_train, X_val, y_val, base_config, max_evals=None):
    """
    Grid search for optimal hyperparameters across kernel sizes, learning rates,
    batch sizes, and dropout rates.
    """
    param_grid = {
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64],
        "dropout_rate": [0.3, 0.5],
        "kernel_sizes": [[3, 5, 7], [3, 7, 11], [5, 9, 13]],
    }

    best_acc = 0.0
    best_params = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    if max_evals is not None and max_evals < len(combinations):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combinations), size=max_evals, replace=False)
        combinations = [combinations[i] for i in idx]

    print(f"Testing {len(combinations)} hyperparameter combinations...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        config = base_config.copy()
        config.update(params)
        # Use lower epochs during tuning search for computational efficiency
        tune_config = config.copy()
        tune_config["epochs"] = min(config.get("epochs", 100), 20)
        tune_config["early_stopping_patience"] = 5

        print(f"\nCombination {i+1}/{len(combinations)}: {params}")

        model = NeuroSwiftModel(tune_config)
        model, _ = train_with_early_stopping(
            model, X_train, y_train, X_val, y_val, tune_config
        )

        res = evaluate_model(model, X_val, y_val)
        acc = res[0] if isinstance(res, (tuple, list)) else res

        if acc > best_acc:
            best_acc = acc
            best_params = params
            print(f"✅ New best validation accuracy: {acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"Best Accuracy:   {best_acc:.2f}%")
    print(f"Best Parameters: {best_params}")
    print("=" * 60)

    return best_params
