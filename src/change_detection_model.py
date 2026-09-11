"""
TerraVision - Phase 4: Early-Fusion Siam-UNet Satellite Change Detection Architecture

This module implements a 6-channel Early-Fusion Siamese U-Net neural network for
bi-temporal satellite image change detection.

Input:  (B, 6, H, W) - Concatenation of Image A (Before RGB) and Image B (After RGB)
Output: (B, 1, H, W) - Binary change logit map
"""

import torch
import torch.nn as nn
from src.segmentation_model import DoubleConv, UNet, count_parameters


class SiamUNet(nn.Module):
    """
    Early-Fusion Siam-UNet for Bi-Temporal Satellite Image Change Detection.
    Takes 6-channel input (RGB_A + RGB_B) and outputs 1-channel binary logit.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 1, init_features: int = 32):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, num_classes=num_classes, init_features=init_features)

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bi-temporal image pair.
        img_A: (B, 3, H, W)
        img_B: (B, 3, H, W)
        """
        x = torch.cat([img_A, img_B], dim=1) # (B, 6, H, W)
        logits = self.unet(x)                 # (B, 1, H, W)
        return logits


if __name__ == "__main__":
    model = SiamUNet(in_channels=6, num_classes=1, init_features=32)
    print("Siam-UNet Change Detection Model:")
    print(model)
    print(f"\nTotal Trainable Parameters: {count_parameters(model):,}")

    dummy_A = torch.randn(2, 3, 256, 256)
    dummy_B = torch.randn(2, 3, 256, 256)
    dummy_output = model(dummy_A, dummy_B)
    print(f"Dummy Output Shape: {dummy_output.shape}")
