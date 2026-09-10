"""
TerraVision - Phase 1.4: Preprocessing & Data Pipeline Setup

This module handles downloading/loading the EuroSAT RGB dataset from Hugging Face,
defining channel-wise normalization statistics computed strictly from the training set,
applying on-the-fly PyTorch torchvision transforms, and constructing PyTorch DataLoaders
for train, validation, and test splits.
"""

from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


# Standard class names for EuroSAT (10 classes) in numerical label order (0 to 9)
CLASS_NAMES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Normalization statistics computed strictly from the training set (18,900 images)
# Values obtained by computing per-channel mean and std on EuroSAT RGB train split
EUROSAT_MEAN: List[float] = [0.3445, 0.3803, 0.4078]
EUROSAT_STD: List[float] = [0.2031, 0.1371, 0.1157]


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset wrapper for Hugging Face EuroSAT_RGB dataset split.

    Applies on-the-fly preprocessing and image transformations without saving
    processed images to disk.
    """

    def __init__(self, hf_dataset, transform: Optional[transforms.Compose] = None):
        """
        Args:
            hf_dataset: A Hugging Face dataset split (e.g. dataset['train']).
            transform: PyTorch torchvision transform pipeline to apply to images.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply torchvision transforms if specified
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns appropriate torchvision transforms for a given dataset split.

    Args:
        split: One of 'train', 'validation', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),  # Converts PIL Image (H, W, C) [0, 255] to FloatTensor (C, H, W) [0.0, 1.0]
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])
    else:
        # Deterministic pipeline for validation and test splits
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])


def get_dataloaders(
    dataset_name: str = "giswqs/EuroSAT_RGB",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads EuroSAT_RGB dataset from Hugging Face and creates PyTorch DataLoaders.

    Args:
        dataset_name: Hugging Face dataset identifier.
        batch_size: Mini-batch size for DataLoaders.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    raw_dataset = load_dataset(dataset_name)

    train_ds = EuroSATDataset(
        raw_dataset["train"],
        transform=get_transforms("train")
    )
    val_ds = EuroSATDataset(
        raw_dataset["validation"],
        transform=get_transforms("validation")
    )
    test_ds = EuroSATDataset(
        raw_dataset["test"],
        transform=get_transforms("test")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
