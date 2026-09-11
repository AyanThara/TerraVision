"""
TerraVision - Phase 4: OSCD Sentinel-2 Change Detection Dataset Setup

This module implements a PyTorch Dataset and DataLoader wrapper for the official
OGC Sentinel-2 Change Detection (OSCD) benchmark dataset (`ericyu/OSCD_Cropped_256`).
It pairs genuine Before satellite imagery (imageA) and After satellite imagery (imageB)
with expert-annotated binary change masks (0 = Unchanged, 1 = Changed).
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class OSCDChangeDetectionDataset(Dataset):
    """
    PyTorch Dataset wrapper for OSCD Bi-Temporal Sentinel-2 Change Detection.
    Returns (img_A_tensor, img_B_tensor, change_mask_tensor).
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

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.hf_dataset[idx]

        img_A = item["imageA"]
        img_B = item["imageB"]
        mask = item["label"]

        if img_A.mode != "RGB":
            img_A = img_A.convert("RGB")
        if img_B.mode != "RGB":
            img_B = img_B.convert("RGB")

        # Resize to patch_size if needed
        if img_A.size != (self.patch_size, self.patch_size):
            img_A = img_A.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            img_B = img_B.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            mask = mask.resize((self.patch_size, self.patch_size), Image.NEAREST)

        mask_np = np.array(mask)
        # Binarize mask: 0 for Unchanged, 1 for Changed (mask values can be 0 or 255)
        binary_mask = (mask_np > 128).astype(np.int64)

        tensor_A = self.img_transform(img_A)
        tensor_B = self.img_transform(img_B)
        mask_tensor = torch.from_numpy(binary_mask).long()

        return tensor_A, tensor_B, mask_tensor


def get_change_detection_dataloaders(
    dataset_name: str = "ericyu/OSCD_Cropped_256",
    patch_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    max_train_samples: Optional[int] = 200,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads OSCD benchmark dataset and creates (train_loader, val_loader, test_loader).
    """
    raw_dataset = load_dataset(dataset_name)

    train_data = raw_dataset["train"]
    test_data = raw_dataset["test"]

    if max_train_samples and len(train_data) > max_train_samples:
        train_data = train_data.select(range(max_train_samples))

    # Split train_data into 85% Train, 15% Val
    total_train = len(train_data)
    n_train = int(total_train * 0.85)

    train_split = train_data.select(range(0, n_train))
    val_split = train_data.select(range(n_train, total_train))
    test_split = test_data

    train_ds = OSCDChangeDetectionDataset(train_split, patch_size=patch_size, is_train=True)
    val_ds = OSCDChangeDetectionDataset(val_split, patch_size=patch_size, is_train=False)
    test_ds = OSCDChangeDetectionDataset(test_split, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
