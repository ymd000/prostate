"""A four-level U-Net trained from scratch, without external weight downloads."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "group"):
        if normalization not in ("group", "none"):
            raise ValueError("normalization must be group or none")
        layers = []
        groups = min(8, max(1, out_channels // 2))
        while out_channels % groups:
            groups -= 1
        for channels in (in_channels, out_channels):
            layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
            if normalization == "group":
                layers.append(nn.GroupNorm(groups, out_channels))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 5, base_channels: int = 32, normalization: str = "group"):
        super().__init__()
        if num_classes < 2 or base_channels < 1:
            raise ValueError("num_classes >= 2 and base_channels >= 1 are required")
        channels = [base_channels * 2**level for level in range(5)]
        self.encoders = nn.ModuleList(
            DoubleConv(in_channels, out_channels, normalization)
            for in_channels, out_channels in zip([3, *channels[:-1]], channels, strict=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(channels[level + 1], channels[level], 2, stride=2)
            for level in reversed(range(4))
        )
        self.decoders = nn.ModuleList(
            DoubleConv(channels[level] * 2, channels[level], normalization)
            for level in reversed(range(4))
        )
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        features = image
        for level, encoder in enumerate(self.encoders):
            if level:
                features = self.pool(features)
            features = encoder(features)
            skips.append(features)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips[:-1]), strict=True
        ):
            features = upsample(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            features = decoder(torch.cat((skip, features), dim=1))
        return self.head(features)
