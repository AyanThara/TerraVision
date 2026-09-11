"""
TerraVision - Phase 5: Satellite Image Change Quantification Script

This script loads the trained Phase 4 Siam-UNet model checkpoint and a held-out
OSCD Sentinel-2 test image pair, computes pixel-level and spatial area metrics
(m^2 and hectares based on Sentinel-2 10m GSD), assigns a TerraVision change-intensity
category (Low, Moderate, High, Very High), outputs a JSON report, and generates a
5-panel visualization summary plot.
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
from notebooks.verify_quantification import categorize_change_intensity, calculate_quantification_metrics


def plot_quantification_summary(
    img_A_tensor: torch.Tensor,
    img_B_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    metrics: Dict[str, Any],
    output_path: Path,
):
    """
    Generates a 5-panel visualization with a dedicated Quantification Summary Box.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_A_np = np.clip((img_A_tensor.numpy() * std + mean).transpose(1, 2, 0), 0.0, 1.0)
    img_B_np = np.clip((img_B_tensor.numpy() * std + mean).transpose(1, 2, 0), 0.0, 1.0)

    target_np = target_tensor.numpy()
    pred_np = pred_tensor.numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"TerraVision Phase 5: Satellite Image Change Quantification Analysis",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(img_A_np)
    axes[0].set_title("Image A (Before)", fontsize=11, fontweight="semibold")
    axes[0].axis("off")

    axes[1].imshow(img_B_np)
    axes[1].set_title("Image B (After)", fontsize=11, fontweight="semibold")
    axes[1].axis("off")

    axes[2].imshow(target_np, cmap="gray")
    axes[2].set_title("Ground Truth Change", fontsize=11, fontweight="semibold")
    axes[2].axis("off")

    axes[3].imshow(pred_np, cmap="magma")
    axes[3].set_title("Predicted Change Mask", fontsize=11, fontweight="semibold")
    axes[3].axis("off")

    # Panel 5: Text Overlay Summary Box
    axes[4].axis("off")
    summary_text = (
        f"QUANTIFICATION REPORT\n"
        f"-----------------------------------------\n"
        f"Total Pixels:       {metrics['total_pixels']:,}\n"
        f"Changed Pixels:     {metrics['changed_pixels']:,}\n"
        f"Unchanged Pixels:   {metrics['unchanged_pixels']:,}\n"
        f"Change Ratio:       {metrics['change_percentage']:.2f}%\n"
        f"Unchanged Ratio:    {metrics['unchanged_percentage']:.2f}%\n"
        f"-----------------------------------------\n"
        f"SPATIAL AREA ESTIMATE (10m GSD):\n"
        f"Changed Area:       {metrics['changed_area_m2']:,} m²\n"
        f"Changed Area:       {metrics['changed_area_hectares']:.2f} ha\n"
        f"-----------------------------------------\n"
        f"INTENSITY RATING:\n"
        f"Category:           [{metrics['change_intensity'].upper()}]\n"
        f"Note: TerraVision heuristic rating"
    )
    axes[4].text(
        0.05, 0.5, summary_text,
        transform=axes[4].transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="center",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e1e2e", edgecolor="#89b4fa", alpha=0.95),
    )

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

    print(f"Executing Change Quantification on device: {device}")

    checkpoint_path = PROJECT_ROOT / "models" / "v3_siam_unet_change_detection.pth"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    metrics_output_path = results_dir / "change_quantification.json"
    plot_output_path = results_dir / "change_quantification.png"

    assert checkpoint_path.exists(), f"Checkpoint not found at '{checkpoint_path}'! Run Phase 4 first."

    # 1. Load OSCD Test Data
    print("\nLoading OSCD Test DataLoader...")
    _, _, test_loader = get_change_detection_dataloaders(patch_size=256, batch_size=8, max_train_samples=200)

    # 2. Load Checkpoint
    print(f"Loading Phase 4 checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SiamUNet(in_channels=6, num_classes=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Run Inference on Test Pair
    print("\nRunning inference on held-out test pair...")
    with torch.no_grad():
        imgs_A, imgs_B, masks = next(iter(test_loader))
        imgs_A_dev, imgs_B_dev = imgs_A.to(device), imgs_B.to(device)

        outputs = model(imgs_A_dev, imgs_B_dev)
        preds = (torch.sigmoid(outputs).squeeze(1) > 0.5).long()

        sample_img_A = imgs_A[0].cpu()
        sample_img_B = imgs_B[0].cpu()
        sample_target = masks[0].cpu()
        sample_pred = preds[0].cpu()

    # 4. Calculate Quantification Metrics
    metrics = calculate_quantification_metrics(sample_pred.numpy(), gsd_meters=10.0)

    print("\n" + "=" * 70)
    print("TerraVision Phase 5: Satellite Change Quantification Results")
    print("=" * 70)
    print(f"Total Pixels:            {metrics['total_pixels']:,}")
    print(f"Changed Pixels:          {metrics['changed_pixels']:,}")
    print(f"Unchanged Pixels:        {metrics['unchanged_pixels']:,}")
    print(f"Change Percentage:       {metrics['change_percentage']:.2f}%")
    print(f"Unchanged Percentage:    {metrics['unchanged_percentage']:.2f}%")
    print("-" * 70)
    print(f"Estimated Changed Area:  {metrics['changed_area_m2']:,} m² ({metrics['changed_area_hectares']:.2f} hectares)")
    print(f"Spatial GSD Assumption:  10.0 meters per pixel (Sentinel-2 Band 2,3,4 standard)")
    print(f"Change Intensity Rating: [{metrics['change_intensity'].upper()}] (TerraVision Heuristic)")
    print("=" * 70)

    # 5. Save JSON Metrics Summary
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved quantification metrics JSON to '{metrics_output_path}'")

    # 6. Save Visualization Plot
    plot_quantification_summary(
        sample_img_A, sample_img_B, sample_target, sample_pred, metrics, plot_output_path
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
