"""
TerraVision - Phase 2: V1 Baseline Training Script

This script initializes the V1 SimpleCNN model, performs a single-batch sanity check,
trains the model on the EuroSAT RGB training set, tracks epoch-by-epoch train/validation loss
and accuracy, and saves the best model checkpoint based on validation accuracy.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import SimpleCNN, count_parameters


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Trains the model for one epoch and returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates the model on validation/test set and returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_sanity_check(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Executes mandatory pre-training verification:
    1. Print model architecture and parameter count.
    2. Pass 1 real mini-batch through model.
    3. Confirm output shape is (64, 10).
    4. Confirm loss calculation.
    """
    print("=" * 65)
    print("Pre-Training Sanity Check")
    print("=" * 65)
    print("\n--- Model Architecture ---")
    print(model)

    num_params = count_parameters(model)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)

    print("\n--- Single-Batch Verification ---")
    print(f"Input batch shape:   {images.shape}")
    print(f"Output batch shape:  {outputs.shape}")
    print(f"Calculated loss:     {loss.item():.4f}")

    assert outputs.shape == (images.size(0), 10), (
        f"Expected output shape ({images.size(0)}, 10), got {outputs.shape}"
    )
    assert not torch.isnan(loss), "Calculated loss is NaN!"
    print("Sanity check PASSED: Output shape is (64, 10) and loss calculation is valid.\n")


def main():
    set_seed(42)

    # Configuration & Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 0

    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Ensure output directories exist
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = models_dir / "v1_baseline.pth"
    metrics_path = results_dir / "v1_training_metrics.json"

    # 1. Load DataLoaders
    print("\nLoading EuroSAT RGB DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 2. Initialize Model, Loss Function, and Optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Mandatory Pre-Training Sanity Check
    run_sanity_check(model, train_loader, criterion, device)

    # 4. Training Loop
    print("=" * 65)
    print(f"Starting V1 Baseline Training ({num_epochs} Epochs)")
    print("=" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Time (s)':<8}")
    print("-" * 65)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "total_time_seconds": 0.0,
    }

    best_val_acc = 0.0
    start_time_total = time.time()

    for epoch in range(1, num_epochs + 1):
        start_time_epoch = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device
        )

        elapsed_epoch = time.time() - start_time_epoch

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Checkpoint saving on best validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            history["best_val_acc"] = round(best_val_acc, 4)
            history["best_epoch"] = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            saved_str = " -> Best Model Saved"
        else:
            saved_str = ""

        print(
            f"{epoch:<6d} | {train_loss:<11.4f} | {train_acc * 100:<9.2f}% | "
            f"{val_loss:<10.4f} | {val_acc * 100:<8.2f}% | {elapsed_epoch:<8.2f}{saved_str}"
        )

    total_training_time = time.time() - start_time_total
    history["total_time_seconds"] = round(total_training_time, 2)

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 65)
    print("Training Complete!")
    print(f"Total Time:               {total_training_time:.2f} seconds")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {history['best_epoch']})")
    print(f"Checkpoint Saved To:      {checkpoint_path}")
    print(f"Metrics Saved To:         {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
