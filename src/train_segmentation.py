"""
TerraVision - Phase 3: Satellite Image Segmentation U-Net Training Script

This script trains a U-Net model on the DeepGlobe 7-class satellite land cover dataset,
using a combined Cross-Entropy + Dice Loss function, tracking mIoU and Dice scores across epochs,
and saving the best model checkpoint based on validation mIoU.
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

from src.dataset_segmentation import get_segmentation_dataloaders, DEEPGLOBE_CLASSES
from src.segmentation_model import UNet, count_parameters


class DiceLoss(nn.Module):
    """Multi-class Dice Loss for Semantic Segmentation."""

    def __init__(self, num_classes: int = 7, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


def compute_miou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 7,
    smooth: float = 1e-6,
) -> float:
    """Computes Mean Intersection over Union (mIoU) across all classes."""
    preds = torch.argmax(logits, dim=1)

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()

        if union > 0:
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou)

    return sum(ious) / len(ious) if ious else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains U-Net for one epoch and returns (avg_loss, avg_miou)."""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(dataloader)

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss_ce = ce_loss_fn(outputs, masks)
        loss_dice = dice_loss_fn(outputs, masks)
        loss = loss_ce + loss_dice

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_miou += compute_miou(outputs, masks)

    return total_loss / num_batches, total_miou / num_batches


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates U-Net on validation loader and returns (avg_loss, avg_miou)."""
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(dataloader)

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss_ce = ce_loss_fn(outputs, masks)
        loss_dice = dice_loss_fn(outputs, masks)
        loss = loss_ce + loss_dice

        total_loss += loss.item()
        total_miou += compute_miou(outputs, masks)

    return total_loss / num_batches, total_miou / num_batches


def main():
    # Set seed
    torch.manual_seed(42)

    # Configuration & Hyperparameters
    patch_size = 256
    batch_size = 8
    num_epochs = 3
    learning_rate = 0.001

    # Device selection: MPS (Apple Silicon GPU), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training U-Net Segmentation Baseline on device: {device}")

    # Output paths
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = models_dir / "v2_unet_segmentation.pth"
    metrics_path = results_dir / "segmentation_training_metrics.json"

    # 1. Load DataLoaders
    print("\nLoading DeepGlobe DataLoaders...")
    train_loader, val_loader, _ = get_segmentation_dataloaders(
        patch_size=patch_size,
        batch_size=batch_size,
        max_train_samples=200,
    )

    # 2. Instantiate Model, Losses, and Optimizer
    model = UNet(in_channels=3, num_classes=len(DEEPGLOBE_CLASSES), init_features=32).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes=len(DEEPGLOBE_CLASSES))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"U-Net Parameters: {count_parameters(model):,}")

    # 3. Training Loop
    print("\n" + "=" * 70)
    print(f"Starting U-Net Segmentation Training ({num_epochs} Epochs)")
    print("=" * 70)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train mIoU':<11} | {'Val Loss':<10} | {'Val mIoU':<10} | {'Time (s)':<8}")
    print("-" * 70)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": [],
        "best_epoch": 0,
        "best_val_miou": 0.0,
        "total_time_seconds": 0.0,
    }

    best_val_miou = 0.0
    start_total = time.time()

    for epoch in range(1, num_epochs + 1):
        start_epoch = time.time()

        train_loss, train_miou = train_one_epoch(
            model, train_loader, ce_loss_fn, dice_loss_fn, optimizer, device
        )
        val_loss, val_miou = evaluate_epoch(
            model, val_loader, ce_loss_fn, dice_loss_fn, device
        )

        elapsed = time.time() - start_epoch

        history["train_loss"].append(round(train_loss, 4))
        history["train_miou"].append(round(train_miou, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_miou"].append(round(val_miou, 4))

        is_best = val_miou > best_val_miou
        if is_best:
            best_val_miou = val_miou
            history["best_val_miou"] = round(best_val_miou, 4)
            history["best_epoch"] = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            saved_str = " -> Best Model Saved"
        else:
            saved_str = ""

        print(
            f"{epoch:<6d} | {train_loss:<11.4f} | {train_miou * 100:<10.2f}% | "
            f"{val_loss:<10.4f} | {val_miou * 100:<9.2f}% | {elapsed:<8.2f}{saved_str}"
        )

    total_time = time.time() - start_total
    history["total_time_seconds"] = round(total_time, 2)

    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Best Validation mIoU: {best_val_miou * 100:.2f}% (Epoch {history['best_epoch']})")
    print(f"Checkpoint Saved To: {checkpoint_path}")
    print(f"Metrics Saved To:    {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
