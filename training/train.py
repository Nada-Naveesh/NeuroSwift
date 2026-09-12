import os
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.neuroswift import NeuroSwiftModel
from training.config import CONFIG
from data.augment import augment_trial


def create_dataloaders(X, y, config):
    """Create train, validation, and test dataloaders with balanced sampling."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    if config.get("use_augmentation", True):
        print("Applying data augmentation (jitter, scaling, and time-shift on full 4-second trials)...")
        rng = np.random.default_rng(42)
        aug_X, aug_y = [], []
        for x_trial, y_label in zip(X_train, y_train):
            aug_X.append(x_trial)
            aug_y.append(y_label)
            # Add an augmented replica
            aug_X.append(augment_trial(x_trial, rng=rng))
            aug_y.append(y_label)
        X_train = np.array(aug_X, dtype=np.float32)
        y_train = np.array(aug_y, dtype=np.int64)

    # Compute balanced class weights
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: float(cw[i]) for i, cls in enumerate(classes)}
    print(f"Class weights: {class_weight_dict}")

    sample_weights = np.array([class_weight_dict[int(label)] for label in y_train], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader, X_test, y_test, class_weight_dict


def train_with_early_stopping(model, *args, **kwargs):
    """
    Train model with early stopping and learning rate scheduling.
    Supports both calling conventions:
      1. train_with_early_stopping(model, train_loader, val_loader, config, class_weights=None)
      2. train_with_early_stopping(model, X_train, y_train, X_val, y_val, config)
    """
    if len(args) == 5:
        # Called as (model, X_train, y_train, X_val, y_val, config)
        X_train, y_train, X_val, y_val, config = args
        class_weights = kwargs.get("class_weights", None)
        batch_size = config.get("batch_size", 32)
        train_ds = TensorDataset(
            torch.FloatTensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train.float(),
            torch.LongTensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train.long(),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val.float(),
            torch.LongTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val.long(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    elif len(args) >= 3:
        # Called as (model, train_loader, val_loader, config, [class_weights])
        train_loader = args[0]
        val_loader = args[1]
        config = args[2]
        class_weights = args[3] if len(args) > 3 else kwargs.get("class_weights", None)
    else:
        raise ValueError("Invalid arguments passed to train_with_early_stopping")

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.0005),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    if class_weights is not None:
        weights_tensor = torch.FloatTensor([class_weights[i] for i in range(len(class_weights))]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print("\n" + "=" * 60)
    print("STARTING BALANCED TRAINING")
    print("=" * 60)

    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 15)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / max(total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1:03d}/{epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step()
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_weights, "best_model_improved.pt")
            checkpoints_dir = Path("./checkpoints")
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_weights, checkpoints_dir / "best_model_improved.pt")
            torch.save(best_model_weights, checkpoints_dir / "best_model_effatt.pt")
            print(f"  [OK] New best model saved (Val Acc: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    print("\n[OK] Training complete!")
    if history["val_accuracy"]:
        print(f"Best Validation Accuracy: {max(history['val_accuracy']):.2f}%")

    return model, history


def fit(X, y, epochs=None, lr=None, batch_size=None, augment_copies=1, out_path=None, config=None):
    """Bridge function for external scripts."""
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    if epochs is not None:
        cfg["epochs"] = epochs
    if lr is not None:
        cfg["learning_rate"] = lr
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, X_test, y_test, cw = create_dataloaders(X, y, cfg)
    model = NeuroSwiftModel(cfg).to(device)
    model, history = train_with_early_stopping(model, train_loader, val_loader, cfg, class_weights=cw)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
    return {
        "history": history,
        "best_val_accuracy": max(history["val_accuracy"]) if history["val_accuracy"] else 0.0,
    }


def main():
    data_dir = Path("./data/processed")
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not (x_path.exists() and y_path.exists()):
        from preprocessing.signal_processor import process_all_files
        process_all_files()

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    print("=" * 60)
    print("NEUROSWIFT BALANCED TRAINING")
    print("=" * 60)
    print(f"Loaded {len(X)} total trials with shape {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, X_test, y_test, cw = create_dataloaders(X, y, CONFIG)
    model = NeuroSwiftModel(CONFIG).to(device)

    model, history = train_with_early_stopping(model, train_loader, val_loader, CONFIG, class_weights=cw)

    # Save final model
    torch.save(model.state_dict(), "best_model_final.pt")
    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/best_model_final.pt")
    print("\n[OK] Final model saved as 'best_model_final.pt'")

    # Step 3: Weight health check
    print("\n" + "=" * 60)
    print("WEIGHT HEALTH VERIFICATION")
    print("=" * 60)
    weights = torch.load("best_model_improved.pt", map_location="cpu")
    first_layer = weights["fc1.weight"].numpy()
    print(f"FC1 Weight mean: {first_layer.mean():.4f}")
    print(f"FC1 Weight std:  {first_layer.std():.4f}")
    if first_layer.std() < 0.05:
        print("[!] Warning: Weights std is low.")
    else:
        print("[OK] Model weights are healthy and well-trained.")

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_acc = 100 * correct / max(total, 1)
    print(f"\nFinal Test Accuracy on Held-Out Split: {test_acc:.2f}%")
    cm = confusion_matrix(all_targets, all_preds)
    print("Test Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
