"""Ports of the nine archived architectures, returning logits for loss and metrics."""

# Derived from __archive/models.py, Copyright (c) 2020 Ken ENDA and
# Department of Cancer Pathology Faculty of Medicine, HOKKAIDO UNIVERSITY.
# Distributed under the BSD-2-Clause terms in LICENSE.

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.vgg import cfgs, make_layers

from prostate.config import ARCHIVE_MODELS


def _vgg_features(depth: int, pretrained: bool, cache_dir: Path) -> nn.Sequential:
    features = make_layers(cfgs["A" if depth == 11 else "D"], batch_norm=True)
    if pretrained:
        weights = (
            models.VGG11_BN_Weights.DEFAULT if depth == 11 else models.VGG16_BN_Weights.DEFAULT
        )
        state = weights.get_state_dict(progress=True, check_hash=True, model_dir=str(cache_dir))
        features.load_state_dict(
            {k.removeprefix("features."): v for k, v in state.items() if k.startswith("features.")}
        )
    return features


class ConvRelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class Interpolate(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=2,
            mode=self.mode,
            align_corners=False if self.mode == "bilinear" else None,
        )


class DeconvDecoder(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu(in_size, mid_size),
            nn.BatchNorm2d(mid_size),
            nn.ConvTranspose2d(mid_size, out_size, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, in_size, mid_size, out_size, mode):
        super().__init__()
        self.block = nn.Sequential(
            Interpolate(mode),
            ConvRelu(in_size, mid_size),
            nn.BatchNorm2d(mid_size),
            ConvRelu(mid_size, out_size),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(
        self, num_classes=5, mode=None, pretrained=True, cache_dir=Path("outputs/pretrained")
    ):
        super().__init__()
        e = _vgg_features(11, pretrained, cache_dir)
        decoder = DeconvDecoder if mode is None else partial(UpsampleDecoder, mode=mode)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu)
        self.conv2 = nn.Sequential(e[4], e[5], self.relu)
        self.conv3 = nn.Sequential(e[8], e[9], self.relu, e[11], e[12])
        self.conv4 = nn.Sequential(e[15], e[16], self.relu, e[18], e[19])
        self.conv5 = nn.Sequential(e[22], e[23], self.relu, e[25], e[26])
        self.dec1 = decoder(512, 512, 256)
        self.dec2 = decoder(768, 512, 256)
        self.dec3 = decoder(768, 512, 128)
        self.dec4 = decoder(384, 256, 64)
        self.dec5 = decoder(192, 128, 32)
        self.dec6 = ConvRelu(96, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))
        d5 = self.dec1(self.pool(e5))
        d4 = self.dec2(torch.cat([d5, e5], 1))
        d3 = self.dec3(torch.cat([d4, e4], 1))
        d2 = self.dec4(torch.cat([d3, e3], 1))
        d1 = self.dec5(torch.cat([d2, e2], 1))
        return self.final(self.dec6(torch.cat([d1, e1], 1)))


class UNet16(nn.Module):
    def __init__(
        self, num_classes=5, mode=None, pretrained=True, cache_dir=Path("outputs/pretrained")
    ):
        super().__init__()
        e = _vgg_features(16, pretrained, cache_dir)
        decoder = DeconvDecoder if mode is None else partial(UpsampleDecoder, mode=mode)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu, e[3], e[4], self.relu)
        self.conv2 = nn.Sequential(e[7], e[8], self.relu, e[10], e[11], self.relu)
        self.conv3 = nn.Sequential(
            e[14], e[15], self.relu, e[17], e[18], self.relu, e[20], e[21], self.relu
        )
        self.conv4 = nn.Sequential(
            e[24], e[25], self.relu, e[27], e[28], self.relu, e[30], e[31], self.relu
        )
        self.conv5 = nn.Sequential(
            e[34], e[35], self.relu, e[37], e[38], self.relu, e[40], e[41], self.relu
        )
        self.center = decoder(512, 512, 256)
        self.dec5 = decoder(768, 512, 256)
        self.dec4 = decoder(768, 512, 256)
        self.dec3 = decoder(512, 256, 64)
        self.dec2 = decoder(192, 128, 32)
        self.dec1 = ConvRelu(96, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))
        center = self.center(self.pool(e5))
        d5 = self.dec5(torch.cat([center, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        return self.final(self.dec1(torch.cat([d2, e1], 1)))


class AlbuNet(nn.Module):
    def __init__(
        self, num_classes=5, mode=None, pretrained=True, cache_dir=Path("outputs/pretrained")
    ):
        super().__init__()
        base = models.resnet34(weights=None)
        if pretrained:
            state = models.ResNet34_Weights.DEFAULT.get_state_dict(
                progress=True, check_hash=True, model_dir=str(cache_dir)
            )
            base.load_state_dict(state)
        decoder = DeconvDecoder if mode is None else partial(UpsampleDecoder, mode=mode)
        self.pool = base.maxpool
        self.first = nn.Sequential(base.conv1, base.bn1, base.relu, self.pool)
        self.enc1, self.enc2, self.enc3, self.enc4 = (
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.dec1 = decoder(512, 512, 256)
        self.dec2 = decoder(768, 512, 256)
        self.dec3 = decoder(512, 512, 256)
        self.dec4 = decoder(384, 256, 64)
        self.dec5 = decoder(128, 128, 128)
        self.final = nn.Sequential(
            decoder(128, 128, 32), ConvRelu(32, 32), nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        e1 = self.first(x)
        e2 = self.enc1(e1)
        e3 = self.enc2(e2)
        e4 = self.enc3(e3)
        e5 = self.enc4(e4)
        d1 = self.dec1(self.pool(e5))
        d2 = self.dec2(torch.cat([d1, e5], 1))
        d3 = self.dec3(torch.cat([d2, e4], 1))
        d4 = self.dec4(torch.cat([d3, e3], 1))
        return self.final(self.dec5(torch.cat([d4, e2], 1)))


def build_archive_model(
    model_name="unet11", num_classes=5, pretrained=True, cache_dir=Path("outputs/pretrained")
) -> nn.Module:
    if model_name not in ARCHIVE_MODELS:
        raise ValueError(f"Unknown archive model: {model_name}")
    constructor = (
        UNet11
        if model_name.startswith("unet11")
        else UNet16
        if model_name.startswith("unet16")
        else AlbuNet
    )
    mode = (
        "bilinear" if model_name.endswith("b") else "nearest" if model_name.endswith("n") else None
    )
    # 'albunet' itself ends in 't'; only its _b/_n variants select interpolation.
    return constructor(
        num_classes=num_classes, mode=mode, pretrained=pretrained, cache_dir=cache_dir
    )
