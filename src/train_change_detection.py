"""
TerraVision - Phase 4: Bi-Temporal OSCD Change Detection Siam-UNet Training Script

This script trains a 6-channel Siam-UNet model for bi-temporal Sentinel-2 satellite image
change detection on Apple Silicon (MPS GPU). It optimizes a combined BCE + Dice loss,
tracks Change IoU scores across epochs, and saves the best model checkpoint.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.dataset_change_detection import get_change_detection_dataloaders
from src.change_detection_model import SiamUNet, count_parameters


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss for Change Detection."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).squeeze(1)
        targets_float = targets.float()

        intersection = torch.sum(probs * targets_float)
        cardinality = torch.sum(probs + targets_float)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


def compute_change_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes Intersection over Union (IoU) for the changed class (1)."""
    preds = (torch.sigmoid(logits).squeeze(1) > threshold).long()
    
    inter = (preds & targets).sum().float().item()
    union = (preds | targets).sum().float().item()

    if union > 0:
        return (inter + 1e-6) / (union + 1e-6)
    return 1.0 if inter == 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    bce_fn: nn.Module,
    dice_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0

    for imgs_A, imgs_B, masks in loader:
        imgs_A, imgs_B, masks = imgs_A.to(device), imgs_B.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs_A, imgs_B)

        loss_bce = bce_fn(outputs.squeeze(1), masks.float())
        loss_dice = dice_fn(outputs, masks)
        total_loss = loss_bce + loss_dice

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_iou += compute_change_iou(outputs, masks)
        num_batches += 1

    return running_loss / num_batches, running_iou / num_batches


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    bce_fn: nn.Module,
    dice_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0

    for imgs_A, imgs_B, masks in loader:
        imgs_A, imgs_B, masks = imgs_A.to(device), imgs_B.to(device), masks.to(device)

        outputs = model(imgs_A, imgs_B)
        loss_bce = bce_fn(outputs.squeeze(1), masks.float())
        loss_dice = dice_fn(outputs, masks)
        total_loss = loss_bce + loss_dice

        running_loss += total_loss.item()
        running_iou += compute_change_iou(outputs, masks)
        num_batches += 1

    return running_loss / num_batches, running_iou / num_batches


def main():
    torch.manual_seed(42)

    patch_size = 256
    batch_size = 8
    num_epochs = 3
    learning_rate = 0.001

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training Siam-UNet OSCD Change Detection on device: {device}")

    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = models_dir / "v3_siam_unet_change_detection.pth"
    metrics_path = results_dir / "change_detection_training_metrics.json"

    print("\nLoading OSCD Bi-Temporal Change DataLoaders...")
    train_loader, val_loader, _ = get_change_detection_dataloaders(
        patch_size=patch_size,
        batch_size=batch_size,
        max_train_samples=200,
    )

    model = SiamUNet(in_channels=6, num_classes=1, init_features=32).to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = BinaryDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Siam-UNet Parameters: {count_parameters(model):,}")

    print("\n" + "=" * 70)
    print(f"Starting Siam-UNet OSCD Change Detection Training ({num_epochs} Epochs)")
    print("=" * 70)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train IoU':<11} | {'Val Loss':<10} | {'Val IoU':<10} | {'Time (s)':<8}")
    print("-" * 70)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
        "best_epoch": 0,
        "best_val_iou": 0.0,
        "total_time_seconds": 0.0,
    }

    best_val_iou = 0.0
    start_total = time.time()

    for epoch in range(1, num_epochs + 1):
        start_epoch = time.time()

        train_loss, train_iou = train_one_epoch(
            model, train_loader, bce_fn, dice_fn, optimizer, device
        )
        val_loss, val_iou = evaluate_epoch(
            model, val_loader, bce_fn, dice_fn, device
        )

        elapsed = time.time() - start_epoch

        history["train_loss"].append(round(train_loss, 4))
        history["train_iou"].append(round(train_iou, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_iou"].append(round(val_iou, 4))

        is_best = val_iou >= best_val_iou
        if is_best:
            best_val_iou = val_iou
            history["best_val_iou"] = round(best_val_iou, 4)
            history["best_epoch"] = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            saved_str = " -> Model Saved"
        else:
            saved_str = ""

        print(
            f"{epoch:<6d} | {train_loss:<11.4f} | {train_iou * 100:<10.2f}% | "
            f"{val_loss:<10.4f} | {val_iou * 100:<9.2f}% | {elapsed:<8.2f}{saved_str}"
        )

    total_time = time.time() - start_total
    history["total_time_seconds"] = round(total_time, 2)

    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Best Validation IoU: {best_val_iou * 100:.2f}% (Epoch {history['best_epoch']})")
    print(f"Checkpoint Saved To: {checkpoint_path}")
    print(f"Metrics Saved To:    {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
