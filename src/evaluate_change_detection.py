"""
TerraVision - Phase 4: OSCD Change Detection Evaluation & Transition Analysis Script

This script evaluates the trained Siam-UNet change detection model on the held-out OSCD TEST set.
It computes Pixel Accuracy, Change IoU, Change F1/Dice score, Changed Pixel Count,
and Change Percentage.
Outputs a 4-panel qualitative visualization plot:
[Before Image A | After Image B | Ground Truth Change Mask | Predicted Change Mask]
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

from src.dataset_change_detection import get_change_detection_dataloaders
from src.change_detection_model import SiamUNet


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    total_correct = 0
    total_pixels = 0

    total_inter = 0.0
    total_union = 0.0
    total_preds_sum = 0.0
    total_targets_sum = 0.0

    sample_img_A, sample_img_B, sample_target, sample_pred = None, None, None, None

    for idx, (imgs_A, imgs_B, masks) in enumerate(test_loader):
        imgs_A, imgs_B, masks = imgs_A.to(device), imgs_B.to(device), masks.to(device)

        outputs = model(imgs_A, imgs_B)
        preds = (torch.sigmoid(outputs).squeeze(1) > 0.5).long()

        if idx == 0:
            sample_img_A = imgs_A[0].cpu()
            sample_img_B = imgs_B[0].cpu()
            sample_target = masks[0].cpu()
            sample_pred = preds[0].cpu()

        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        inter = (preds & masks).sum().item()
        union = (preds | masks).sum().item()

        total_inter += inter
        total_union += union
        total_preds_sum += preds.sum().item()
        total_targets_sum += masks.sum().item()

    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    change_iou = (total_inter + 1e-6) / (total_union + 1e-6) if total_union > 0 else 0.0
    change_dice = (2.0 * total_inter + 1e-6) / (total_preds_sum + total_targets_sum + 1e-6)

    # Calculate Changed Pixel Count & Percentage on sample visualization
    changed_pixels = int(sample_pred.sum().item())
    total_sample_pixels = int(sample_pred.numel())
    change_percentage = (changed_pixels / total_sample_pixels) * 100.0

    summary = {
        "pixel_accuracy": round(float(pixel_accuracy), 4),
        "change_iou": round(float(change_iou), 4),
        "change_dice": round(float(change_dice), 4),
        "sample_changed_pixel_count": changed_pixels,
        "sample_total_pixel_count": total_sample_pixels,
        "sample_change_percentage": round(float(change_percentage), 2),
    }

    return summary, sample_img_A, sample_img_B, sample_target, sample_pred


def plot_qualitative_sample(
    img_A_tensor: torch.Tensor,
    img_B_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    output_path: Path,
    change_percent: float,
    changed_pixels: int,
):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_A_np = np.clip((img_A_tensor.numpy() * std + mean).transpose(1, 2, 0), 0.0, 1.0)
    img_B_np = np.clip((img_B_tensor.numpy() * std + mean).transpose(1, 2, 0), 0.0, 1.0)

    target_np = target_tensor.numpy()
    pred_np = pred_tensor.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"TerraVision Phase 4: OSCD Sentinel-2 Change Detection Result\n"
        f"Changed Pixels: {changed_pixels:,} / {pred_np.size:,} | Change Percentage: {change_percent:.2f}%",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].imshow(img_A_np)
    axes[0].set_title("Image A (Before)", fontsize=11, fontweight="semibold")
    axes[0].axis("off")

    axes[1].imshow(img_B_np)
    axes[1].set_title("Image B (After)", fontsize=11, fontweight="semibold")
    axes[1].axis("off")

    axes[2].imshow(target_np, cmap="gray")
    axes[2].set_title("Ground Truth Change Mask", fontsize=11, fontweight="semibold")
    axes[2].axis("off")

    axes[3].imshow(pred_np, cmap="magma")
    axes[3].set_title("Predicted Change Mask", fontsize=11, fontweight="semibold")
    axes[3].axis("off")

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

    print(f"Evaluating Siam-UNet OSCD Change Detection on device: {device}")

    checkpoint_path = PROJECT_ROOT / "models" / "v3_siam_unet_change_detection.pth"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    metrics_output_path = results_dir / "change_detection_evaluation_metrics.json"
    plot_output_path = results_dir / "change_detection_samples.png"

    assert checkpoint_path.exists(), f"Checkpoint not found at '{checkpoint_path}'! Train the model first."

    print("\nLoading OSCD Bi-Temporal Change Test DataLoader...")
    _, _, test_loader = get_segmentation_dataloaders = get_change_detection_dataloaders(patch_size=256, batch_size=8, max_train_samples=200)

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SiamUNet(in_channels=6, num_classes=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch", "N/A")
    best_val_iou = checkpoint.get("val_iou", 0.0)

    print(f"Checkpoint loaded (Best Val IoU: {best_val_iou * 100:.2f}% at Epoch {best_epoch})")

    print("\nRunning inference on TEST split...")
    summary, img_A, img_B, target, pred = evaluate_test_set(model, test_loader, device)

    print("\n" + "=" * 70)
    print("TerraVision Phase 4: OSCD Change Detection Test Set Evaluation Results")
    print("=" * 70)
    print(f"Change Class IoU:          {summary['change_iou'] * 100:.2f}%")
    print(f"Change F1 / Dice Score:    {summary['change_dice'] * 100:.2f}%")
    print(f"Pixel Accuracy:            {summary['pixel_accuracy'] * 100:.2f}%")
    print(f"Sample Changed Pixel Count:{summary['sample_changed_pixel_count']:,} pixels")
    print(f"Sample Change Percentage:  {summary['sample_change_percentage']:.2f}%")
    print("=" * 70)

    summary["best_epoch"] = best_epoch
    summary["best_val_iou"] = round(float(best_val_iou), 4)

    with open(metrics_output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved evaluation metrics JSON to '{metrics_output_path}'")

    plot_qualitative_sample(
        img_A, img_B, target, pred, plot_output_path,
        summary["sample_change_percentage"], summary["sample_changed_pixel_count"]
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
