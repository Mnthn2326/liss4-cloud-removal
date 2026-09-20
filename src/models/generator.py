"""U-Net generator for pix2pix cloud removal.

Standard pix2pix U-Net: 8 encoder (downsampling) blocks, 8 decoder
(upsampling) blocks with skip connections. Dropout(0.5) on the first 3
decoder layers. Final activation: tanh.

Architecture defined in docs/MODEL_SPEC.md.
Input:  (N, C, 256, 256)  — cloudy patch
Output: (N, C, 256, 256)  — reconstructed cloud-free patch

Channel count C is read from the dataset config, not hardcoded.
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """pix2pix U-Net generator.

    Parameters
    ----------
    in_channels : int
        Number of input bands (3 for RGB/RICE, 4 for LISS-IV with NIR).
    out_channels : int
        Number of output bands (same as in_channels for cloud removal).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # ---- Encoder (8 downsampling blocks) ----
        # Each: Conv2d(k=4, s=2, p=1) → [BN] → LeakyReLU(0.2)
        self.enc1 = self._enc_block(in_channels, 64, use_bn=False)  # 256→128
        self.enc2 = self._enc_block(64, 128)                        # 128→64
        self.enc3 = self._enc_block(128, 256)                       # 64→32
        self.enc4 = self._enc_block(256, 512)                       # 32→16
        self.enc5 = self._enc_block(512, 512)                       # 16→8
        self.enc6 = self._enc_block(512, 512)                       # 8→4
        self.enc7 = self._enc_block(512, 512)                       # 4→2
        self.bottleneck = self._enc_block(512, 512, use_bn=False)   # 2→1

        # ---- Decoder (8 upsampling blocks with skip connections) ----
        # Each: ConvTranspose2d(k=4, s=2, p=1) → BN → [Dropout(0.5)] → ReLU
        # Input channels = prev_output + skip_channels (after concat)
        self.dec7 = self._dec_block(512, 512, use_dropout=True)     # 1→2
        self.dec6 = self._dec_block(1024, 512, use_dropout=True)    # 2→4
        self.dec5 = self._dec_block(1024, 512, use_dropout=True)    # 4→8
        self.dec4 = self._dec_block(1024, 512)                      # 8→16
        self.dec3 = self._dec_block(1024, 256)                      # 16→32
        self.dec2 = self._dec_block(512, 128)                       # 32→64
        self.dec1 = self._dec_block(256, 64)                        # 64→128

        # Final upsampling to original resolution
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4,
                               stride=2, padding=1),               # 128→256
            nn.Tanh(),
        )

        # Weight initialisation (pix2pix convention)
        self.apply(self._init_weights)

    # ----- helper: encoder block -----
    @staticmethod
    def _enc_block(in_c: int, out_c: int, use_bn: bool = True) -> nn.Sequential:
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2,
                            padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    # ----- helper: decoder block -----
    @staticmethod
    def _dec_block(in_c: int, out_c: int,
                   use_dropout: bool = False) -> nn.Sequential:
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    # ----- weight init -----
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """N(0, 0.02) for conv/deconv weights, N(1, 0.02) for BN weights."""
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Encoder ----
        e1 = self.enc1(x)          # (N,  64, 128, 128)
        e2 = self.enc2(e1)         # (N, 128,  64,  64)
        e3 = self.enc3(e2)         # (N, 256,  32,  32)
        e4 = self.enc4(e3)         # (N, 512,  16,  16)
        e5 = self.enc5(e4)         # (N, 512,   8,   8)
        e6 = self.enc6(e5)         # (N, 512,   4,   4)
        e7 = self.enc7(e6)         # (N, 512,   2,   2)
        b = self.bottleneck(e7)    # (N, 512,   1,   1)

        # ---- Decoder with skip connections ----
        d7 = self.dec7(b)                        # (N, 512,  2,  2)
        d7 = torch.cat([d7, e7], dim=1)          # (N,1024,  2,  2)

        d6 = self.dec6(d7)                       # (N, 512,  4,  4)
        d6 = torch.cat([d6, e6], dim=1)          # (N,1024,  4,  4)

        d5 = self.dec5(d6)                       # (N, 512,  8,  8)
        d5 = torch.cat([d5, e5], dim=1)          # (N,1024,  8,  8)

        d4 = self.dec4(d5)                       # (N, 512, 16, 16)
        d4 = torch.cat([d4, e4], dim=1)          # (N,1024, 16, 16)

        d3 = self.dec3(d4)                       # (N, 256, 32, 32)
        d3 = torch.cat([d3, e3], dim=1)          # (N, 512, 32, 32)

        d2 = self.dec2(d3)                       # (N, 128, 64, 64)
        d2 = torch.cat([d2, e2], dim=1)          # (N, 256, 64, 64)

        d1 = self.dec1(d2)                       # (N,  64,128,128)
        d1 = torch.cat([d1, e1], dim=1)          # (N, 128,128,128)

        return self.final(d1)                    # (N, out_c, 256, 256)
