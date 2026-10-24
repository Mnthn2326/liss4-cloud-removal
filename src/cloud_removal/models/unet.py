"""Conventional U-Net architecture for optical satellite cloud removal.

Implements an encoder-decoder network with skip connections, supporting
configurable base width, bilinear/transposed-conv upsampling, and exact spatial
dimension preservation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: 2x2 MaxPool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: 2x2 Transposed Convolution or Bilinear Upsampling, skip concatenation, and DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        self.up: nn.Module
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle any odd-dimension padding discrepancies
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature channels to desired output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Conventional U-Net baseline for supervised optical cloud removal.

    Architecture:
        - 4 Encoder levels with MaxPool downsampling.
        - Bottleneck feature representation.
        - 4 Decoder levels with skip connections from matching encoder stages.
        - Output 1x1 convolution with Sigmoid activation to bound surface reflectance in [0, 1].

    Args:
        in_channels: Number of input channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 3 for RGB).
        base_channels: Channel width of the initial convolutional layer (default: 32).
        bilinear: Whether to use bilinear upsampling instead of transposed conv (default: False).
        use_sigmoid: Whether to apply Sigmoid at output to ensure [0, 1] range (default: True).
        prior_reflectance: Optional target surface reflectance prior in (0, 1). If provided
            with use_sigmoid=True, initializes final conv bias to logit(prior) to accelerate
            convergence and prevent initial DC offset prediction penalty.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        bilinear: bool = False,
        use_sigmoid: bool = True,
        prior_reflectance: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.use_sigmoid = use_sigmoid
        self.prior_reflectance = prior_reflectance

        c = base_channels
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, (c * 16) // factor)

        # Decoder
        self.up1 = Up(c * 16, (c * 8) // factor, bilinear)
        self.up2 = Up(c * 8, (c * 4) // factor, bilinear)
        self.up3 = Up(c * 4, (c * 2) // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)

        # Final projection
        self.outc = OutConv(c, out_channels)
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

        if use_sigmoid and prior_reflectance is not None:
            if not (0.0 < prior_reflectance < 1.0):
                raise ValueError(
                    f"prior_reflectance must be strictly between 0 and 1, got {prior_reflectance}"
                )
            import math

            bias_val = math.log(prior_reflectance / (1.0 - prior_reflectance))
            if self.outc.conv.bias is not None:
                nn.init.constant_(self.outc.conv.bias, bias_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connection concatenation
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return self.sigmoid(logits)

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return dictionary of architecture configuration parameters."""
        return {
            "model_type": "UNet",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "bilinear": self.bilinear,
            "use_sigmoid": self.use_sigmoid,
            "prior_reflectance": self.prior_reflectance,
            "total_trainable_params": self.count_parameters(),
        }
