"""
TerraVision - Phase 3: Satellite Image Segmentation Evaluation Script

This script loads the trained U-Net model checkpoint, evaluates strictly on the TEST split,
computes overall mIoU, Mean Dice Coefficient, Pixel Accuracy, and per-class IoU/Dice scores,
saves JSON metrics to results/, and generates a qualitative 3-panel visualization comparison
(Original Satellite Image | Ground Truth Mask | U-Net Predicted Mask).
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset_segmentation import (
    get_segmentation_dataloaders,
    DEEPGLOBE_CLASSES,
    COLOR_MAP,
)
from src.segmentation_model import UNet


def mask_indices_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    """Converts a 2D class index matrix (H, W) [0..6] to RGB image array (H, W, 3)."""
    H, W = mask_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for class_idx, color in COLOR_MAP.items():
        mask = (mask_np == class_idx)
        rgb[mask] = color

    return rgb


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 7,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates U-Net on test set, computes per-class IoU, Dice, and overall mIoU/Pixel Acc.
    """
    model.eval()

    total_correct = 0
    total_pixels = 0

    class_intersections = np.zeros(num_classes, dtype=np.float64)
    class_unions = np.zeros(num_classes, dtype=np.float64)
    class_targets_sum = np.zeros(num_classes, dtype=np.float64)
    class_preds_sum = np.zeros(num_classes, dtype=np.float64)

    sample_img, sample_target, sample_pred = None, None, None

    for idx, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # Save first batch sample for visualization
        if idx == 0:
            sample_img = images[0].cpu()
            sample_target = masks[0].cpu()
            sample_pred = preds[0].cpu()

        # Overall Pixel Accuracy
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        # Per-class Confusion Statistics
        for cls in range(num_classes):
            pred_cls = (preds == cls)
            target_cls = (masks == cls)

            inter = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()

            class_intersections[cls] += inter
            class_unions[cls] += union
            class_preds_sum[cls] += pred_cls.sum().item()
            class_targets_sum[cls] += target_cls.sum().item()

    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

    per_class_metrics = {}
    valid_ious = []
    valid_dices = []

    for cls in range(num_classes):
        class_name = DEEPGLOBE_CLASSES[cls]
        inter = class_intersections[cls]
        union = class_unions[cls]
        p_sum = class_preds_sum[cls]
        t_sum = class_targets_sum[cls]

        iou = (inter + 1e-6) / (union + 1e-6) if union > 0 else 0.0
        dice = (2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6) if (p_sum + t_sum) > 0 else 0.0

        if union > 0:
            valid_ious.append(iou)
            valid_dices.append(dice)

        per_class_metrics[class_name] = {
            "iou": round(float(iou), 4),
            "dice": round(float(dice), 4),
            "pixels_present": int(t_sum),
        }

    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0

    summary = {
        "miou": round(float(miou), 4),
        "mean_dice": round(float(mean_dice), 4),
        "pixel_accuracy": round(float(pixel_accuracy), 4),
        "per_class": per_class_metrics,
    }

    return summary, sample_img, sample_target, sample_pred


def plot_qualitative_sample(
    img_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    output_path: Path,
):
    """Generates a 3-panel visualization: Satellite Image | Ground Truth Mask | Predicted Mask."""
    # Denormalize image for display
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_np = img_tensor.numpy() * std + mean
    img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)

    target_rgb = mask_indices_to_rgb(target_tensor.numpy())
    pred_rgb = mask_indices_to_rgb(pred_tensor.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TerraVision Phase 3: U-Net Satellite Image Segmentation Result", fontsize=14, fontweight="bold")

    axes[0].imshow(img_np)
    axes[0].set_title("Input Satellite Image", fontsize=11, fontweight="semibold")
    axes[0].axis("off")

    axes[1].imshow(target_rgb)
    axes[1].set_title("Ground-Truth Mask", fontsize=11, fontweight="semibold")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("U-Net Predicted Mask", fontsize=11, fontweight="semibold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative visualization plot to '{output_path}'")


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Evaluating U-Net Segmentation on device: {device}")

    checkpoint_path = PROJECT_ROOT / "models" / "v2_unet_segmentation.pth"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    metrics_output_path = results_dir / "segmentation_evaluation_metrics.json"
    plot_output_path = results_dir / "segmentation_prediction_samples.png"

    assert checkpoint_path.exists(), f"Checkpoint not found at '{checkpoint_path}'! Train the model first."

    # 1. Load Test DataLoader
    print("\nLoading DeepGlobe Test DataLoader...")
    _, _, test_loader = get_segmentation_dataloaders(patch_size=512, batch_size=4, max_train_samples=200)

    # 2. Load Checkpoint
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, num_classes=len(DEEPGLOBE_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch", "N/A")
    best_val_miou = checkpoint.get("val_miou", 0.0)

    print(f"Checkpoint loaded (Best Val mIoU: {best_val_miou * 100:.2f}% at Epoch {best_epoch})")

    # 3. Evaluate on TEST set
    print("\nRunning inference on TEST split...")
    summary, sample_img, sample_target, sample_pred = evaluate_test_set(
        model, test_loader, device, num_classes=len(DEEPGLOBE_CLASSES)
    )

    print("\n" + "=" * 70)
    print("TerraVision Phase 3: Final Test Set Evaluation Results")
    print("=" * 70)
    print(f"Mean IoU (mIoU):     {summary['miou'] * 100:.2f}%")
    print(f"Mean Dice Score:     {summary['mean_dice'] * 100:.2f}%")
    print(f"Pixel Accuracy:      {summary['pixel_accuracy'] * 100:.2f}%")
    print("=" * 70)

    print("\n--- Per-Class IoU & Dice Breakdown ---")
    print(f"{'Class Name':<15} | {'IoU Score':<12} | {'Dice Score':<12} | {'Pixels Present':<14}")
    print("-" * 60)
    for class_name, metrics in summary["per_class"].items():
        iou_pct = metrics["iou"] * 100
        dice_pct = metrics["dice"] * 100
        pixels = metrics["pixels_present"]
        print(f"{class_name:<15} | {iou_pct:<11.2f}% | {dice_pct:<11.2f}% | {pixels:<14,d}")

    # 4. Save JSON Metrics Summary
    summary["best_epoch"] = best_epoch
    summary["best_val_miou"] = round(float(best_val_miou), 4)

    with open(metrics_output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved evaluation metrics JSON to '{metrics_output_path}'")

    # 5. Save Qualitative Visualization Plot
    plot_qualitative_sample(sample_img, sample_target, sample_pred, plot_output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
