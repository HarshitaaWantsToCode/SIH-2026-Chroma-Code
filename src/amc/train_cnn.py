"""
Training Script for 1D-CNN Automatic Modulation Classifier (AMC).

Trains a PyTorch 1D-CNN on synthetic complex I/Q baseband signals across:
- BPSK
- QPSK
- 16-QAM
- 2-FSK

Outputs:
- Trained model weights and metadata saved to `models/amc_1dcnn_weights.pt`
- Training and validation loss/accuracy curves
"""

import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.amc.dataset_generator import create_train_val_test_datasets, CNN_CLASS_LABELS, DEFAULT_WINDOW_SIZE
from src.amc.models.cnn1d_classifier import Modulation1DCNN


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cnn(
    samples_per_class: int = 600,
    window_length: int = DEFAULT_WINDOW_SIZE,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    output_path: str = "models/amc_1dcnn_weights.pt",
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING 1D-CNN AMC CLASSIFIER ===")
    print(f"Device: {device}")
    print(f"Window Length: {window_length} samples ([2, {window_length}])")
    print(f"Target Classes: {CNN_CLASS_LABELS}")
    print(f"Samples per class: {samples_per_class} (Total: {samples_per_class * len(CNN_CLASS_LABELS)})")

    # 1. Generate clean Train / Val / Test Splits
    print("\n[1/4] Generating synthetic datasets across randomized SNR / CFO / Phase...")
    train_ds, val_ds, test_ds, meta = create_train_val_test_datasets(
        total_samples_per_class=samples_per_class,
        window_length=window_length,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    print(f"  Train samples: {len(train_ds)} ({meta['samples_per_class_train']}/class)")
    print(f"  Val samples:   {len(val_ds)} ({meta['samples_per_class_val']}/class)")
    print(f"  Test samples:  {len(test_ds)} ({meta['samples_per_class_test']}/class)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model, Loss, Optimizer
    print("\n[2/4] Initializing 1D-CNN model architecture...")
    model = Modulation1DCNN(
        sequence_length=window_length,
        num_classes=len(CNN_CLASS_LABELS),
        class_labels=CNN_CLASS_LABELS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 3. Training Loop
    print(f"\n[3/4] Starting training for {epochs} epochs...")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state_dict = None
    start_time = time.time()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%"
        )

        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    elapsed_time = time.time() - start_time
    print(f"\nTraining complete in {elapsed_time:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # 4. Save Weights & Checkpoint Metadata
    print(f"\n[4/4] Saving model artifact to {output_path}...")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": best_state_dict,
        "class_labels": CNN_CLASS_LABELS,
        "window_length": window_length,
        "val_accuracy": float(best_val_acc),
        "val_loss": float(best_val_loss),
        "total_parameters": total_params,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "samples_per_class": samples_per_class,
            "seed": seed,
            "elapsed_seconds": float(elapsed_time),
        },
        "metadata": meta,
    }

    torch.save(checkpoint, out_file)
    print(f"Artifact saved successfully. Size: {os.path.getsize(out_file) / 1024:.1f} KB")

    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D-CNN AMC Classifier")
    parser.add_argument("--samples-per-class", type=int, default=600, help="Total samples per class")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/amc_1dcnn_weights.pt", help="Output weights file")
    args = parser.parse_args()

    train_cnn(
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output,
    )
