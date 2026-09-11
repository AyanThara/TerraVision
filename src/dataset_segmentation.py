"""
TerraVision - Phase 3: Satellite Image Segmentation Dataset & Pipeline Setup

This module handles loading the DeepGlobe Land Cover Classification dataset,
mapping 7 mask classes to class index tensors (0..6),
slicing/resizing high-resolution satellite imagery into 512x512 patches,
and creating PyTorch DataLoaders for training, validation, and testing.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


# 7 DeepGlobe Land Cover Classes and RGB Color Codes
DEEPGLOBE_CLASSES: List[str] = [
    "Unknown",      # 0: Black (0, 0, 0)
    "Urban",        # 1: Cyan (0, 255, 255)
    "Agriculture",  # 2: Yellow (255, 255, 0)
    "Rangeland",    # 3: Magenta (255, 0, 255)
    "Forest",       # 4: Green (0, 255, 0)
    "Water",        # 5: Blue (0, 0, 255)
    "Barren",       # 6: White (255, 255, 255)
]

COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),       # Unknown
    1: (0, 255, 255),   # Urban
    2: (255, 255, 0),   # Agriculture
    3: (255, 0, 255),   # Rangeland
    4: (0, 255, 0),     # Forest
    5: (0, 0, 255),     # Water
    6: (255, 255, 255), # Barren
}


def parse_mask_to_indices(mask_img: Image.Image, patch_size: int = 256) -> np.ndarray:
    """
    Parses a PIL mask image into a 2D numpy array of class indices [0..6].
    Supports both single-channel pre-indexed masks (mode 'L') and RGB color-coded masks.
    """
    mask_resized = mask_img.resize((patch_size, patch_size), Image.NEAREST)
    mask_np = np.array(mask_resized)

    # Case 1: Pre-indexed single-channel mask (mode 'L' or 2D array)
    if mask_np.ndim == 2:
        # Clip values to valid class range [0..6]
        return np.clip(mask_np.astype(np.int64), 0, 6)

    # Case 2: RGB color-coded mask (H, W, 3)
    H, W, _ = mask_np.shape
    label_mask = np.zeros((H, W), dtype=np.int64)
    pixels = mask_np.reshape(-1, 3).astype(np.int32)
    min_dists = np.full(pixels.shape[0], fill_value=1e9, dtype=np.float32)

    for class_idx, color in COLOR_MAP.items():
        color_arr = np.array(color, dtype=np.int32)
        dists = np.sum((pixels - color_arr) ** 2, axis=1)
        mask = dists < min_dists
        min_dists[mask] = dists[mask]
        label_mask.flat[mask] = class_idx

    return label_mask


class DeepGlobeSegmentationDataset(Dataset):
    """
    PyTorch Dataset wrapper for DeepGlobe Land Cover Classification.
    Resizes/tiles high-res imagery into patch_size x patch_size patches on the fly.
    """

    def __init__(
        self,
        hf_dataset,
        patch_size: int = 256,
        is_train: bool = True,
    ):
        self.hf_dataset = hf_dataset
        self.patch_size = patch_size
        self.is_train = is_train

        # Standard normalization for RGB imagery
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.hf_dataset[idx]

        # Extract image and mask from sample dict
        img = item.get("pixel_values") or item.get("image") or item.get("sat_image")
        mask = item.get("mask") or item.get("label") or item.get("mask_image")

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize image to target patch size
        img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        if mask is not None:
            label_mask = parse_mask_to_indices(mask, patch_size=self.patch_size)
        else:
            label_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)

        img_tensor = self.img_transform(img)
        label_tensor = torch.from_numpy(label_mask).long()

        return img_tensor, label_tensor


def get_segmentation_dataloaders(
    dataset_name: str = "ratnaonline1/deepglobe-land-cover-classification-dataset",
    patch_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    max_train_samples: Optional[int] = 500,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads DeepGlobe dataset and returns (train_loader, val_loader, test_loader).
    """
    raw_dataset = load_dataset(dataset_name)

    train_data = raw_dataset["train"]
    if max_train_samples and len(train_data) > max_train_samples:
        train_data = train_data.select(range(max_train_samples))

    # Split train_data into 70% Train, 15% Val, 15% Test
    total = len(train_data)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)

    train_split = train_data.select(range(0, n_train))
    val_split = train_data.select(range(n_train, n_train + n_val))
    test_split = train_data.select(range(n_train + n_val, total))

    train_ds = DeepGlobeSegmentationDataset(train_split, patch_size=patch_size, is_train=True)
    val_ds = DeepGlobeSegmentationDataset(val_split, patch_size=patch_size, is_train=False)
    test_ds = DeepGlobeSegmentationDataset(test_split, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
