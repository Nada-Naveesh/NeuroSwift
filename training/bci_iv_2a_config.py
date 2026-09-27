"""
NEURALIS: Configuration for BCI Competition IV 2a Dataset
Dedicated configuration for 22 EEG channels, 4-class motor imagery,
and cross-dataset generalization benchmarking.
"""

BCI_CONFIG = {
    # Model parameters (BCI IV 2a specific)
    'input_channels': 22,      # BCI IV 2a has 22 EEG channels
    'input_time': 640,         # 4 seconds at 160 Hz (resampled from 250 Hz)
    'num_classes': 4,          # 4 classes: Left Hand, Right Hand, Both Feet, Tongue

    # Multi-scale temporal kernels
    'kernel_sizes': [3, 5, 7],

    # Attention parameters (ECA-Net)
    'attention_type': 'ECA',
    'attention_gamma': 2,
    'attention_b': 1,

    # Optimization & Training
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 15,

    # Data augmentation
    'use_augmentation': True,
    'window_size': 500,
    'window_stride': 75,

    # Evaluation splits
    'test_split': 0.15,
    'validation_split': 0.15,
}

# 4 Motor imagery classes for BCI Competition IV 2a
BCI_CLASS_NAMES = [
    'Left Hand',
    'Right Hand',
    'Both Feet',
    'Tongue'
]

# Benchmark milestone from Base Paper (Lian et al., 2025)
BASE_PAPER_BCI_IV_2A_ACC = 83.43
