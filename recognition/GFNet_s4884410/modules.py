import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Small 4-block CNN baseline for AD vs NC (~0.39M parameters)."""

    def __init__(self, in_ch=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (in_ch, *widths)
        self.features = nn.Sequential(
            *[ConvBlock(a, b) for a, b in zip(chans, chans[1:])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], num_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
