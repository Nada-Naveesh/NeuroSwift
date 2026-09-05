"""Training and model configuration for NeuroSwift."""

from __future__ import annotations

from src.config import (  # noqa: F401
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DROPOUT,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_WEIGHTS_PATH,
    N_CHANNELS,
    N_TIMES,
    NUM_CLASSES,
    PATIENCE,
    RANDOM_SEED,
    REPORTS_FIGURES_DIR,
    SAMPLING_RATE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)

CONFIG = {
    'input_channels': 64,
    'input_time': 640,
    'num_classes': 5,
    'kernel_sizes': [3, 5, 7],
    'attention_type': 'ECA',
    'attention_gamma': 2,
    'attention_b': 1,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 15,
    'use_augmentation': True,
    'window_size': 500,
    'window_stride': 75,
    'test_split': 0.1,
    'validation_split': 0.1,
}
