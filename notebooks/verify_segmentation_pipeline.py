"""
TerraVision - Phase 3: Satellite Image Segmentation Verification Script

This script verifies the DeepGlobe segmentation data pipeline, image and mask batch shapes,
dtype constraints, integer mask values in range [0..6], U-Net forward pass, output shape (B, 7, 512, 512),
and loss calculation functionality before full training.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def main():
    print("=" * 70)
    print("TerraVision Phase 3: Satellite Image Segmentation Verification")
    print("=" * 70)

    batch_size = 4
    patch_size = 512
    num_classes = len(DEEPGLOBE_CLASSES)

    print(f"\n1. Loading DeepGlobe DataLoaders (batch_size={batch_size}, patch_size={patch_size})...")
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        patch_size=patch_size,
        batch_size=batch_size,
        max_train_samples=40,
    )

    print("\n--- Dataset Sample Counts ---")
    print(f"Train dataset samples: {len(train_loader.dataset)}")
    print(f"Val dataset samples:   {len(val_loader.dataset)}")
    print(f"Test dataset samples:  {len(test_loader.dataset)}")

    # Fetch one batch from training loader
    print("\n2. Fetching mini-batch from training loader...")
    images, masks = next(iter(train_loader))

    print("\n--- Batch Tensor Properties ---")
    print(f"Image tensor shape: {images.shape}")
    print(f"Mask tensor shape:  {masks.shape}")
    print(f"Image tensor dtype: {images.dtype}")
    print(f"Mask tensor dtype:  {masks.dtype}")
    print(f"Mask min value:     {masks.min().item()}")
    print(f"Mask max value:     {masks.max().item()}")

    # Print class index count breakdown in sample mask
    unique_indices, counts = torch.unique(masks, return_counts=True)
    print("\n--- Class ID Pixel Counts in Mini-Batch ---")
    for idx, count in zip(unique_indices.tolist(), counts.tolist()):
        class_name = DEEPGLOBE_CLASSES[idx]
        print(f"  Class {idx} ({class_name:<12}): {count:,} pixels")

    print("\n3. Instantiating U-Net Model...")
    model = UNet(in_channels=3, num_classes=num_classes)
    num_params = count_parameters(model)
    print(f"Total U-Net Parameters: {num_params:,}")

    print("\n4. Running Forward Pass through U-Net...")
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    print("\n5. Testing Combined Cross-Entropy + Dice Loss Calculation...")
    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes=num_classes)

    ce_val = ce_loss_fn(outputs, masks)
    dice_val = dice_loss_fn(outputs, masks)
    total_loss = ce_val + dice_val

    print(f"Cross-Entropy Loss: {ce_val.item():.4f}")
    print(f"Dice Loss:          {dice_val.item():.4f}")
    print(f"Total Loss:         {total_loss.item():.4f}")

    print("\n6. Executing Verification Checks...")
    # Check 1: Image batch shape
    expected_img_shape = torch.Size([batch_size, 3, patch_size, patch_size])
    assert images.shape == expected_img_shape, f"Expected {expected_img_shape}, got {images.shape}"
    print(f"  [PASS] Image batch shape is (B, 3, 512, 512): {images.shape}")

    # Check 2: Mask batch shape
    expected_mask_shape = torch.Size([batch_size, patch_size, patch_size])
    assert masks.shape == expected_mask_shape, f"Expected {expected_mask_shape}, got {masks.shape}"
    print(f"  [PASS] Mask batch shape is (B, 512, 512): {masks.shape}")

    # Check 3: Output logits shape
    expected_out_shape = torch.Size([batch_size, num_classes, patch_size, patch_size])
    assert outputs.shape == expected_out_shape, f"Expected {expected_out_shape}, got {outputs.shape}"
    print(f"  [PASS] Output logits shape is (B, 7, 512, 512): {outputs.shape}")

    # Check 4: Valid mask range
    valid_range = all(0 <= val <= 6 for val in unique_indices.tolist())
    assert valid_range, f"Found mask index out of range [0..6]: {unique_indices.tolist()}"
    print("  [PASS] Mask class indices are valid integers in range [0..6]")

    # Check 5: Loss validity
    assert not torch.isnan(total_loss), "Loss value is NaN!"
    print("  [PASS] Loss calculation verified and non-NaN")

    print("\n" + "=" * 70)
    print("SUCCESS: All Phase 3 verification checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
