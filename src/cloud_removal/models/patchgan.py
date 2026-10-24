"""PatchGAN Discriminator and Adversarial Loss for Conditional GANs (Pix2Pix).

Implements the standard 70x70 PatchGAN receptive field discriminator and
associated GAN loss functions for optical satellite cloud removal.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Union, cast

import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights using standard GAN initialization conventions.

    Args:
        net: Neural network module to initialize.
        init_type: Initialization type ('normal', 'xavier', 'kaiming').
        init_gain: Scaling factor for normal and xavier initializations.
    """

    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        weight = getattr(m, "weight", None)
        bias = getattr(m, "bias", None)

        if isinstance(weight, torch.Tensor) and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            w_data = weight.data
            if init_type == "normal":
                nn.init.normal_(w_data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(w_data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(w_data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(f"Initialization method [{init_type}] is not implemented")
            if isinstance(bias, torch.Tensor):
                nn.init.constant_(bias.data, 0.0)
        elif isinstance(weight, torch.Tensor) and classname.find("BatchNorm2d") != -1:
            nn.init.normal_(weight.data, 1.0, init_gain)
            if isinstance(bias, torch.Tensor):
                nn.init.constant_(bias.data, 0.0)

    net.apply(init_func)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN (N-Layer) Discriminator for Conditional Image-to-Image Translation.

    Rather than penalizing the whole image with a single scalar, PatchGAN penalizes
    high-frequency structures at the scale of local image patches ($70 \\times 70$ for
    n_layers=3), effectively modeling the image as a Markov random field.

    The discriminator accepts the concatenated condition and target image:
    `torch.cat([cloudy, target], dim=1)`, with shape `(B, in_channels, H, W)`.

    Args:
        in_channels: Total input channels = condition channels + target channels (e.g., 3 + 3 = 6).
        ndf: Number of discriminator filters in the first convolutional layer (default: 64).
        n_layers: Number of intermediate downsampling layers (default: 3 for 70x70 receptive field).
        norm_layer: Normalization layer class or string identifier ('batch', 'instance', 'none').
    """

    def __init__(
        self,
        in_channels: int = 6,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Union[str, Callable[[int], nn.Module]] = "batch",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers

        # Resolve normalization layer
        if isinstance(norm_layer, str):
            norm_str = norm_layer.lower()
            if norm_str in ("batch", "batchnorm"):
                _norm: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d
            elif norm_str in ("instance", "instancenorm"):
                _norm = functools.partial(
                    nn.InstanceNorm2d, affine=False, track_running_stats=False
                )
            elif norm_str in ("none", "identity"):
                _norm = None
            else:
                raise ValueError(f"Unsupported norm_layer string '{norm_layer}'")
        else:
            _norm = norm_layer

        use_bias = _norm is None or (
            isinstance(_norm, functools.partial) and _norm.func == nn.InstanceNorm2d
        )

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                _norm(ndf * nf_mult) if _norm is not None else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            _norm(ndf * nf_mult) if _norm is not None else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final 1-channel projection (returns raw unnormalized logits for numerical stability)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Concatenated pair of condition and candidate images, shape (B, in_channels, H, W).

        Returns:
            Logits patch map of shape (B, 1, H_out, W_out).
        """
        return self.model(x)

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return dictionary of architecture configuration parameters."""
        return {
            "model_type": "PatchGANDiscriminator",
            "in_channels": self.in_channels,
            "ndf": self.ndf,
            "n_layers": self.n_layers,
            "total_trainable_params": self.count_parameters(),
        }


class GANLoss(nn.Module):
    """Adversarial loss function supporting standard BCE with logits and LSGAN (MSE).

    Automatically handles device matching and target label tensor allocation.
    """

    real_label: torch.Tensor
    fake_label: torch.Tensor

    def __init__(
        self,
        use_lsgan: bool = False,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.use_lsgan = use_lsgan

        self.loss: nn.Module
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create label tensor with same shape and device as prediction."""
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate loss given discriminator predictions and target truth condition."""
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return cast(torch.Tensor, self.loss(prediction, target_tensor))
