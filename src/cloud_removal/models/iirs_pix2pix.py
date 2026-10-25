"""IIRS/ISRO Pix2Pix U-Net Generator for False Color Composite Cloud Removal.

Re-implements the 8-stage U-Net Generator architecture originally published in:
    Maniyar, C., & Kumar, A. (2021). "Generative Adversarial Network for Cloud
    Removal from Optical Temporal Satellite Imagery."
    Soft Computing for Problem Solving, pp. 481-491. Springer, Singapore.
    DOI: 10.1007/978-981-16-2712-5_39

Trained on False Color Composite (FCC: Green, Red, NIR) satellite imagery.
Supports bidirectional reflectance normalization to integrate seamlessly with
physical surface reflectance in [0, 1] pipelines.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class IIRSPix2PixGenerator(nn.Module):
    """PyTorch implementation of the IIRS/ISRO 8-stage Pix2Pix U-Net Generator.

    Architecture:
        - 8-stage convolutional encoder (kernel size 4, stride 2, LeakyReLU 0.2).
        - 8-stage transposed convolutional decoder with skip connections,
          intermediate Dropout (0.5), and BatchNorm.
        - Terminal ConvTranspose2d with Tanh activation.

    Args:
        in_channels: Number of input channels (default 3: Green, Red, NIR).
        out_channels: Number of output channels (default 3: Green, Red, NIR).
        rescale_reflectance: If True, automatically scales input from [0, 1] to
            [-1, 1] for model forward pass and scales output back from [-1, 1]
            to [0, 1], guaranteeing compatibility with physical surface reflectance.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        rescale_reflectance: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rescale_reflectance = rescale_reflectance

        # --- Encoder ---
        # e1: 3 -> 64
        self.e1_conv = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.e1_act = nn.LeakyReLU(0.2, inplace=True)

        # e2: 64 -> 128
        self.e2_conv = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.e2_bn = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.e2_act = nn.LeakyReLU(0.2, inplace=True)

        # e3: 128 -> 256
        self.e3_conv = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.e3_bn = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.e3_act = nn.LeakyReLU(0.2, inplace=True)

        # e4: 256 -> 512
        self.e4_conv = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.e4_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.e4_act = nn.LeakyReLU(0.2, inplace=True)

        # e5: 512 -> 512
        self.e5_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.e5_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.e5_act = nn.LeakyReLU(0.2, inplace=True)

        # e6: 512 -> 512
        self.e6_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.e6_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.e6_act = nn.LeakyReLU(0.2, inplace=True)

        # e7: 512 -> 512
        self.e7_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.e7_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.e7_act = nn.LeakyReLU(0.2, inplace=True)

        # e8 (bottleneck): 512 -> 512
        self.e8_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.e8_act = nn.ReLU(inplace=True)

        # --- Decoder ---
        # d1: 512 -> 512
        self.d1_deconv = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.d1_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.d1_drop = nn.Dropout(0.5)
        self.d1_act = nn.ReLU(inplace=True)

        # d2: (512 + 512 = 1024) -> 512
        self.d2_deconv = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.d2_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.d2_drop = nn.Dropout(0.5)
        self.d2_act = nn.ReLU(inplace=True)

        # d3: (512 + 512 = 1024) -> 512
        self.d3_deconv = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.d3_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.d3_drop = nn.Dropout(0.5)
        self.d3_act = nn.ReLU(inplace=True)

        # d4: (512 + 512 = 1024) -> 512
        self.d4_deconv = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.d4_bn = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.d4_act = nn.ReLU(inplace=True)

        # d5: (512 + 512 = 1024) -> 256
        self.d5_deconv = nn.ConvTranspose2d(
            1024, 256, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.d5_bn = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.d5_act = nn.ReLU(inplace=True)

        # d6: (256 + 256 = 512) -> 128
        self.d6_deconv = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.d6_bn = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.d6_act = nn.ReLU(inplace=True)

        # d7: (128 + 128 = 256) -> 64
        self.d7_deconv = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.d7_bn = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.d7_act = nn.ReLU(inplace=True)

        # d8: (64 + 64 = 128) -> out_channels
        self.d8_deconv = nn.ConvTranspose2d(
            128, out_channels, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.d8_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 8-stage U-Net Generator.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1] (if rescale_reflectance=True)
               or [-1, 1] (if rescale_reflectance=False).

        Returns:
            Cloud-free predicted tensor [B, C, H, W] in [0, 1] or [-1, 1].
        """
        if self.rescale_reflectance:
            # Scale from [0.0, 1.0] to [-1.0, 1.0]
            x_in = 2.0 * x - 1.0
        else:
            x_in = x

        # --- Encoder Downsampling ---
        e1 = self.e1_act(self.e1_conv(x_in))
        e2 = self.e2_act(self.e2_bn(self.e2_conv(e1)))
        e3 = self.e3_act(self.e3_bn(self.e3_conv(e2)))
        e4 = self.e4_act(self.e4_bn(self.e4_conv(e3)))
        e5 = self.e5_act(self.e5_bn(self.e5_conv(e4)))
        e6 = self.e6_act(self.e6_bn(self.e6_conv(e5)))
        e7 = self.e7_act(self.e7_bn(self.e7_conv(e6)))
        e8 = self.e8_act(self.e8_conv(e7))

        # --- Decoder Upsampling with Skip Concatenation ---
        d1 = self.d1_drop(self.d1_bn(self.d1_deconv(e8)))
        c1 = self.d1_act(torch.cat([d1, e7], dim=1))

        d2 = self.d2_drop(self.d2_bn(self.d2_deconv(c1)))
        c2 = self.d2_act(torch.cat([d2, e6], dim=1))

        d3 = self.d3_drop(self.d3_bn(self.d3_deconv(c2)))
        c3 = self.d3_act(torch.cat([d3, e5], dim=1))

        d4 = self.d4_bn(self.d4_deconv(c3))
        c4 = self.d4_act(torch.cat([d4, e4], dim=1))

        d5 = self.d5_bn(self.d5_deconv(c4))
        c5 = self.d5_act(torch.cat([d5, e3], dim=1))

        d6 = self.d6_bn(self.d6_deconv(c5))
        c6 = self.d6_act(torch.cat([d6, e2], dim=1))

        d7 = self.d7_bn(self.d7_deconv(c6))
        c7 = self.d7_act(torch.cat([d7, e1], dim=1))

        out = self.d8_act(self.d8_deconv(c7))

        if self.rescale_reflectance:
            # Rescale output from [-1.0, 1.0] back to [0.0, 1.0]
            out = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)

        return out

    def get_config(self) -> Dict[str, Any]:
        """Return model metadata and configuration dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": "IIRSPix2PixGenerator",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "rescale_reflectance": self.rescale_reflectance,
            "total_params": total_params,
            "total_trainable_params": trainable_params,
            "source": "Maniyar & Kumar (IIRS ISRO 2021) Springer DOI:10.1007/978-981-16-2712-5_39",
        }
