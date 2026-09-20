"""PatchGAN discriminator for pix2pix cloud removal.

70×70 receptive field PatchGAN: classifies overlapping 70×70 patches as
real/fake. Input is the concatenation of (cloudy_input, candidate_output)
along the channel dimension — conditional GAN.

Architecture defined in docs/MODEL_SPEC.md.
Input:  (N, 2*C, 256, 256)  — concat(cloudy, target_or_fake)
Output: (N, 1, 30, 30)      — per-patch real/fake probability map
"""

import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator.

    Parameters
    ----------
    in_channels : int
        Number of channels in the concatenated input.
        Default 6 = 3 (cloudy RGB) + 3 (target/fake RGB).
        For LISS-IV with 4 bands this would be 8.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: C64 — no BatchNorm (standard pix2pix convention)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256→128

            # Layer 2: C128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128→64

            # Layer 3: C256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→32

            # Layer 4: C512  — stride 1 for 70×70 receptive field
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→31

            # Layer 5: output — 1-channel per-patch prediction
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
            # 31→30
        )

        # Weight initialisation (pix2pix convention)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """N(0, 0.02) for conv weights, N(1, 0.02) for BN weights."""
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Concatenation of (cloudy_input, candidate) along channel dim.
            Shape: (N, 2*C, H, W)

        Returns
        -------
        torch.Tensor
            Per-patch real/fake probability map, shape (N, 1, 30, 30)
            for 256×256 input.
        """
        return self.model(x)
