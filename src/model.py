"""
TerraVision - Phase 2: V1 Baseline CNN Model Architecture

This module defines a simple, lightweight Convolutional Neural Network (CNN) built from scratch
for EuroSAT 10-class land-use classification.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple 3-block Convolutional Neural Network for EuroSAT RGB classification.

    Input shape:  (B, 3, 64, 64)
    Output shape: (B, 10) class logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Feature Extraction Backbone
        self.features = nn.Sequential(
            # Block 1: (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling -> (B, 128, 1, 1)
            nn.Flatten(),                 # Flatten -> (B, 128)
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN()
    print("Model Architecture:")
    print(model)
    print(f"\nTotal Trainable Parameters: {count_parameters(model):,}")

    dummy_input = torch.randn(64, 3, 64, 64)
    dummy_output = model(dummy_input)
    print(f"Dummy Input Shape:  {dummy_input.shape}")
    print(f"Dummy Output Shape: {dummy_output.shape}")
