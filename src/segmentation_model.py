"""
TerraVision - Phase 3: U-Net Satellite Image Segmentation Model Architecture

This module implements a classic U-Net neural network with symmetrical contracting (encoder)
and expanding (decoder) paths with skip connections for pixel-level semantic segmentation.

Input shape:  (B, 3, H, W)
Output shape: (B, num_classes, H, W) raw logits
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Convolution Block: (Conv2d -> BatchNorm2d -> ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Symmetrical U-Net Architecture for 7-class DeepGlobe Satellite Land Cover Segmentation.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 7, init_features: int = 32):
        super().__init__()

        features = init_features

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 8, features * 16))

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features * 16, features * 8)

        self.up2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features * 8, features * 4)

        self.up3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features * 4, features * 2)

        self.up4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(features * 2, features)

        # 1x1 Conv Classifier Output Head
        self.outc = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting Path
        x1 = self.inc(x)          # (B, features, H, W)
        x2 = self.down1(x1)       # (B, features*2, H/2, W/2)
        x3 = self.down2(x2)       # (B, features*4, H/4, W/4)
        x4 = self.down3(x3)       # (B, features*8, H/8, W/8)
        x5 = self.down4(x4)       # (B, features*16, H/16, W/16)

        # Expanding Path + Skip Connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)      # (B, num_classes, H, W)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=7, init_features=32)
    print("U-Net Architecture:")
    print(model)
    print(f"\nTotal Trainable Parameters: {count_parameters(model):,}")

    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_output = model(dummy_input)
    print(f"Dummy Input Shape:  {dummy_input.shape}")
    print(f"Dummy Output Shape: {dummy_output.shape}")

