"""
TerraVision - Phase 4: OSCD Satellite Image Change Detection Pipeline Verification Script

This script verifies OSCD bi-temporal data loading, Image A / Image B tensor shapes,
binary change mask shape (B, 256, 256), Siam-UNet forward pass (B, 1, 256, 256),
and loss calculation functionality.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.dataset_change_detection import get_change_detection_dataloaders
from src.change_detection_model import SiamUNet, count_parameters


def main():
    print("=" * 70)
    print("TerraVision Phase 4: OSCD Satellite Image Change Detection Verification")
    print("=" * 70)

    batch_size = 4
    patch_size = 256

    print(f"\n1. Loading OSCD Bi-Temporal Change DataLoaders (batch_size={batch_size}, patch_size={patch_size})...")
    train_loader, val_loader, test_loader = get_change_detection_dataloaders(
        patch_size=patch_size,
        batch_size=batch_size,
        max_train_samples=40,
    )

    print("\n--- Dataset Sample Counts ---")
    print(f"Train dataset samples: {len(train_loader.dataset)}")
    print(f"Val dataset samples:   {len(val_loader.dataset)}")
    print(f"Test dataset samples:  {len(test_loader.dataset)}")

    print("\n2. Fetching mini-batch from training loader...")
    imgs_A, imgs_B, masks = next(iter(train_loader))

    print("\n--- Batch Tensor Properties ---")
    print(f"Image A shape:      {imgs_A.shape}")
    print(f"Image B shape:      {imgs_B.shape}")
    print(f"Change Mask shape:  {masks.shape}")
    print(f"Mask min value:     {masks.min().item()}")
    print(f"Mask max value:     {masks.max().item()}")

    print("\n3. Instantiating Siam-UNet Model...")
    model = SiamUNet(in_channels=6, num_classes=1, init_features=32)
    num_params = count_parameters(model)
    print(f"Total Siam-UNet Parameters: {num_params:,}")

    print("\n4. Running Forward Pass through Siam-UNet...")
    model.eval()
    with torch.no_grad():
        outputs = model(imgs_A, imgs_B)

    print(f"Model Output Shape: {outputs.shape}")

    print("\n5. Testing BCEWithLogitsLoss Calculation...")
    bce_fn = nn.BCEWithLogitsLoss()
    loss = bce_fn(outputs.squeeze(1), masks.float())
    print(f"Loss value: {loss.item():.4f}")

    print("\n6. Executing Verification Checks...")
    assert imgs_A.shape == torch.Size([batch_size, 3, patch_size, patch_size]), "Image A shape mismatch!"
    assert imgs_B.shape == torch.Size([batch_size, 3, patch_size, patch_size]), "Image B shape mismatch!"
    assert masks.shape == torch.Size([batch_size, patch_size, patch_size]), "Mask shape mismatch!"
    assert outputs.shape == torch.Size([batch_size, 1, patch_size, patch_size]), "Output shape mismatch!"
    assert not torch.isnan(loss), "Loss is NaN!"

    print("  [PASS] All Phase 4 OSCD verification checks passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
