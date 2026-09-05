"""Global configuration for NeuroSwift."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Signal
SAMPLING_RATE = 160
TRIAL_DURATION_SEC = 4.0
N_TIMES = int(SAMPLING_RATE * TRIAL_DURATION_SEC)  # 640
N_CHANNELS = 64
BANDPASS_HZ = (0.5, 30.0)
NOTCH_HZ = 60.0

# PhysioNet EEGMMIDB
PHYSIONET_N_SUBJECTS = 109
EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104}
IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)
EXECUTION_RUNS = (3, 5, 7, 9, 11, 13)
LEFT_RIGHT_RUNS = (3, 4, 7, 8, 11, 12)
BOTH_FEET_RUNS = (5, 6, 9, 10, 13, 14)

CLASS_NAMES = ("Left Hand", "Right Hand", "Both Hands", "Feet", "Rest")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

CHANNEL_NAMES = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
DROPOUT = 0.5
SE_REDUCTION = 16
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# Paths
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_WEIGHTS_NAME = "best_model_effatt.pt"
MODEL_WEIGHTS_PATH = CHECKPOINTS_DIR / MODEL_WEIGHTS_NAME
DEMO_WEIGHTS_PATH = CHECKPOINTS_DIR / "demo_synthetic.pt"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
