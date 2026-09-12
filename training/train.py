"""Training with early stopping and verification for NeuroSwift."""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def verify_data(X, y, name="data"):
    """Verify data is properly preprocessed and normalized."""
    print(f"\n{'='*60}")
    print(f"DATA VERIFICATION: {name}")
    print(f"{'='*60}")
    print(f"Shape: {X.shape}")
    print(f"Range: {X.min():.4f} to {X.max():.4f}")
    print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"Class distribution: {np.bincount(y)}")

    if X.max() > 100 or X.min() < -100:
        print("⚠️ WARNING: Data range is large. Verify normalization!")
    if abs(X.mean()) > 1.5:
        print("⚠️ WARNING: Data mean is not centered near 0. Verify normalization!")
    if X.std() > 5.0:
        print("⚠️ WARNING: Data std is unusually large. Verify normalization!")
    if len(np.unique(y)) < 5:
        print("⚠️ WARNING: Not all 5 classes present in dataset split!")
    else:
        print("✅ Data validation checks passed successfully!")
    print(f"{'='*60}\n")


def train_with_early_stopping(model, *args, **kwargs):
    """
    Train model with early stopping, class balancing, gradient clipping,
    and cosine annealing scheduling.
    Supports:
      - train_with_early_stopping(model, X_train, y_train, X_val, y_val, config)
      - train_with_early_stopping(model, train_loader, val_loader, config)
    """
    if len(args) == 5:
        X_train, y_train, X_val, y_val, config = args
        verify_data(X_train, y_train, "Training Data")
        verify_data(X_val, y_val, "Validation Data")

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
        y_train_arr = np.asarray(y_train)
    elif len(args) >= 3:
        train_loader, val_loader, config = args[0], args[1], args[2]
        y_train_arr = None
    else:
        raise ValueError("Invalid arguments passed to train_with_early_stopping")

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.0005),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Class-weighted loss
    if y_train_arr is not None and len(np.unique(y_train_arr)) > 1:
        class_counts = np.bincount(y_train_arr)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = model.state_dict().copy()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 30)

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs:                  {epochs}")
    print(f"Batch Size:              {config.get('batch_size', 32)}")
    print(f"Learning Rate:           {config.get('learning_rate', 0.0005)}")
    print(f"Early Stopping Patience: {patience}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation
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

        val_accuracy = 100.0 * correct / max(total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1:03d}/{epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Early stopping checkpoint check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0

            # Save locally and into checkpoints directory
            torch.save(best_model_weights, "best_model_improved.pt")
            checkpoints_dir = Path("./checkpoints")
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_weights, checkpoints_dir / "best_model_improved.pt")
            torch.save(best_model_weights, checkpoints_dir / "best_model_final.pt")
            print(f"  ✅ New best model saved (Val Acc: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_weights)
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {max(history['val_accuracy']):.2f}%")
    print(f"{'='*60}\n")

    return model, history


# Alias for backwards compatibility
fit = train_with_early_stopping
