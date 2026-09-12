"""Fast Ensemble Specialization for NeuroSwift 5-Model Ensemble."""

import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.neuroswift import NeuroSwiftModel
from models.ensemble import EnsembleModel
from training.config import CONFIG
from preprocessing.augment import add_gaussian_noise, amplitude_scale, time_shift


def augment_sample(x, rng):
    x_aug = x.copy()
    if rng.random() < 0.6:
        x_aug = add_gaussian_noise(x_aug, noise_level=rng.uniform(0.01, 0.04))
    if rng.random() < 0.6:
        x_aug = amplitude_scale(x_aug, scale_range=(0.92, 1.08))
    if rng.random() < 0.5:
        x_aug = time_shift(x_aug, shift_max=rng.integers(4, 12))
    return x_aug


def main():
    print("=" * 60)
    print("NEUROSWIFT: ENSEMBLE DIVERSITY TRAINING (5 SUB-MODELS)")
    print("=" * 60)

    data_dir = Path("./data/processed")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    # Split train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "y_test.npy", y_test)

    checkpoints_dir = Path("./checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = "best_model_improved.pt"
    base_state = None
    if os.path.exists(base_checkpoint):
        base_state = torch.load(base_checkpoint, map_location="cpu")
        if isinstance(base_state, dict) and "model_state_dict" in base_state:
            base_state = base_state["model_state_dict"]

    ensemble_models = []

    for i in range(5):
        print(f"\nTraining Diverse Sub-Model {i+1}/5...")
        rng = np.random.default_rng(100 + i * 37)
        torch.manual_seed(100 + i * 37)

        # Create augmented replica
        X_aug_list = []
        y_aug_list = []
        for trial, label in zip(X_train_val, y_train_val):
            X_aug_list.append(trial)
            y_aug_list.append(label)
            X_aug_list.append(augment_sample(trial, rng))
            y_aug_list.append(label)

        X_aug = np.array(X_aug_list, dtype=np.float32)
        y_aug = np.array(y_aug_list, dtype=np.int64)

        X_tr, X_v, y_tr, y_v = train_test_split(
            X_aug, y_aug, test_size=0.15, random_state=42 + i, stratify=y_aug
        )

        model = NeuroSwiftModel(CONFIG)
        if base_state is not None:
            model.load_state_dict(base_state, strict=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
            batch_size=32,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_v), torch.LongTensor(y_v)),
            batch_size=32,
            shuffle=False,
        )

        best_v_acc = 0.0
        best_weights = model.state_dict()

        for epoch in range(8):
            model.train()
            t_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = model(batch_x)
                    _, preds = torch.max(out, 1)
                    total += batch_y.size(0)
                    correct += (preds == batch_y).sum().item()

            v_acc = 100.0 * correct / max(total, 1)
            print(f"  Epoch {epoch+1}/8 - Train Loss: {t_loss/len(train_loader):.4f} - Val Acc: {v_acc:.2f}%")
            if v_acc >= best_v_acc:
                best_v_acc = v_acc
                best_weights = model.state_dict().copy()

        save_p1 = checkpoints_dir / f"model_ensemble_{i}.pt"
        save_p2 = Path(f"model_ensemble_{i}.pt")
        torch.save(best_weights, save_p1)
        torch.save(best_weights, save_p2)
        model.load_state_dict(best_weights)
        ensemble_models.append(model)
        print(f"✅ Sub-Model {i+1} saved with peak validation accuracy: {best_v_acc:.2f}%")

    # Evaluate ensemble on held-out test split
    ensemble = EnsembleModel(CONFIG, num_models=5)
    ensemble.models = ensemble_models
    preds = ensemble.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score
    test_acc = accuracy_score(y_test, preds) * 100
    test_f1 = f1_score(y_test, preds, average="weighted") * 100

    print("\n" + "=" * 60)
    print(f"🎉 5-MODEL ENSEMBLE TEST ACCURACY: {test_acc:.2f}%")
    print(f"🎉 5-MODEL ENSEMBLE TEST F1-SCORE: {test_f1:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
