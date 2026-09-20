"""Loss functions for pix2pix GAN training.

L_total = L_adversarial + lambda_L1 * L_L1

- L_adversarial: BCE loss (vanilla GAN) — or MSE (LSGAN) switchable via config
- L_L1:          Mean absolute error between generated and ground truth
- lambda_L1:     Read from configs/train_baseline.yaml (default 100)

See docs/MODEL_SPEC.md for rationale.
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Adversarial loss wrapper.

    Supports:
      - "bce": Binary cross-entropy (vanilla GAN)
      - "lsgan": MSE (least-squares GAN — more stable if BCE shows mode collapse)

    Parameters
    ----------
    loss_type : str
        "bce" or "lsgan", read from config `loss.adversarial`.
    """

    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "bce":
            self.criterion = nn.BCELoss()
        elif self.loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}. "
                             f"Use 'bce' or 'lsgan'.")

    def forward(self, prediction: torch.Tensor,
                is_real: bool) -> torch.Tensor:
        """Compute adversarial loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Discriminator output (per-patch probability map).
        is_real : bool
            If True, target is 1 (real); if False, target is 0 (fake).
        """
        target_val = 1.0 if is_real else 0.0
        target = torch.full_like(prediction, target_val)
        return self.criterion(prediction, target)


class PixelLoss(nn.Module):
    """L1 pixel loss between generated and ground truth images."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.criterion(generated, target)
