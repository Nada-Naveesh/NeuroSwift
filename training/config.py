"""Configuration for NeuroSwift Training."""

CONFIG = {
    # Model parameters
    "input_channels": 64,
    "input_time": 640,
    "num_classes": 5,

    # Multi-scale kernels
    "kernel_sizes": [3, 5, 7],

    # Attention
    "attention_type": "ECA",
    "attention_gamma": 2,
    "attention_b": 1,

    # Training - INCREASED FOR OPTIMAL CONVERGENCE & ACCURACY
    "batch_size": 32,
    "learning_rate": 0.0005,
    "weight_decay": 1e-4,
    "epochs": 100,                  # Increased to 100 for deep convergence
    "early_stopping_patience": 30,  # Increased to 30 to prevent premature termination
    "dropout_rate": 0.5,

    # Data augmentation
    "use_augmentation": True,
    "window_size": 500,
    "window_stride": 75,

    # Cross-validation
    "cross_validation_folds": 5,
    "test_split": 0.1,
    "validation_split": 0.1,

    # Ensemble
    "num_ensemble_models": 5,
}

CLASS_NAMES = ["Left Hand", "Right Hand", "Both Hands", "Both Feet", "Rest"]
