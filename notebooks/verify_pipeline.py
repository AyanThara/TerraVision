"""
TerraVision - Phase 1.4: Preprocessing & Data Pipeline Verification Script

This script verifies the EuroSAT RGB dataset preprocessing pipeline by creating
train, validation, and test DataLoaders, fetching a sample mini-batch, inspecting
tensor shapes, dtypes, value ranges, class label decodings, and validating dataset sample counts.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to allow importing from `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import torch
from src.dataset import get_dataloaders, CLASS_NAMES


def main():
    print("=" * 65)
    print("TerraVision Phase 1.4: Data Pipeline Verification")
    print("=" * 65)

    batch_size = 64
    num_workers = 0

    print(f"\n1. Loading EuroSAT RGB DataLoaders (batch_size={batch_size}, num_workers={num_workers})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Extract sample counts from datasets
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    train_count = len(train_dataset)
    val_count = len(val_dataset)
    test_count = len(test_dataset)

    print("\n--- Dataset Sample Counts ---")
    print(f"Train samples:      {train_count:,}")
    print(f"Validation samples: {val_count:,}")
    print(f"Test samples:       {test_count:,}")

    print("\n--- DataLoader Batch Counts ---")
    print(f"Train batches:      {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    print(f"Test batches:       {len(test_loader):,}")

    # Fetch one batch from training loader
    print("\n2. Fetching one mini-batch from training loader...")
    batch_images, batch_labels = next(iter(train_loader))

    min_val = float(batch_images.min())
    max_val = float(batch_images.max())

    print("\n--- Batch Inspection ---")
    print(f"Batch image shape:  {batch_images.shape}")
    print(f"Batch label shape:  {batch_labels.shape}")
    print(f"Image dtype:        {batch_images.dtype}")
    print(f"Label dtype:        {batch_labels.dtype}")
    print(f"Min tensor value:   {min_val:.4f}")
    print(f"Max tensor value:   {max_val:.4f}")

    print("\n--- Class Name Decodings (First 10 items in batch) ---")
    sample_labels = batch_labels[:10].tolist()
    for idx, label_id in enumerate(sample_labels):
        class_name = CLASS_NAMES[label_id]
        print(f"  Item {idx:2d} -> Label ID {label_id}: {class_name}")

    print("\n3. Executing Validation Checks...")

    # Check 1: Sample counts
    assert train_count == 18900, f"Expected 18,900 train samples, got {train_count}"
    assert val_count == 5400, f"Expected 5,400 val samples, got {val_count}"
    assert test_count == 2700, f"Expected 2,700 test samples, got {test_count}"
    print("  [PASS] Sample counts match required split sizes (18,900 / 5,400 / 2,700)")

    # Check 2: Image batch shape
    expected_shape = torch.Size([batch_size, 3, 64, 64])
    assert batch_images.shape == expected_shape, f"Expected shape {expected_shape}, got {batch_images.shape}"
    print(f"  [PASS] Image batch shape matches expected shape (B, 3, 64, 64): {batch_images.shape}")

    # Check 3: Valid integer labels in [0, 9]
    all_labels_valid = all(0 <= l <= 9 for l in batch_labels.tolist())
    assert all_labels_valid, "Found label out of expected range [0, 9]"
    print("  [PASS] Labels are valid integers in range [0, 9]")

    # Check 4: Image dtype float32
    assert batch_images.dtype == torch.float32, f"Expected float32, got {batch_images.dtype}"
    print("  [PASS] Image tensor dtype is float32")

    print("\n" + "=" * 65)
    print("SUCCESS: All verification checks passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
