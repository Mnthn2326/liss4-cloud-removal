"""Unit tests for PatchGANDiscriminator, GANLoss, and initialization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cloud_removal.models.patchgan import (
    GANLoss,
    PatchGANDiscriminator,
    init_weights,
)
from cloud_removal.models.unet import UNet


def test_patchgan_discriminator_shapes() -> None:
    """Verify PatchGANDiscriminator output tensor dimensions for standard 256x256 inputs."""
    batch_size = 2
    in_channels = 6  # 3 cloudy + 3 clear/fake
    height, width = 256, 256

    disc = PatchGANDiscriminator(in_channels=in_channels, ndf=64, n_layers=3)
    x = torch.randn(batch_size, in_channels, height, width)

    out = disc(x)
    assert out.dim() == 4
    assert out.shape[0] == batch_size
    assert out.shape[1] == 1
    # 256 -> 128 -> 64 -> 32 -> 31 -> 30 for 3 layers + penultimate + out
    assert out.shape[2] == 30
    assert out.shape[3] == 30


def test_patchgan_discriminator_parameters_and_config() -> None:
    """Verify parameter counting and configuration dictionary generation."""
    disc = PatchGANDiscriminator(in_channels=6, ndf=32, n_layers=2)
    cfg = disc.get_config()

    assert cfg["model_type"] == "PatchGANDiscriminator"
    assert cfg["in_channels"] == 6
    assert cfg["ndf"] == 32
    assert cfg["n_layers"] == 2
    assert cfg["total_trainable_params"] > 0
    assert cfg["total_trainable_params"] == disc.count_parameters()


def test_patchgan_discriminator_normalization_options() -> None:
    """Verify PatchGANDiscriminator instantiates correctly with different norm layers."""
    for norm in ("batch", "instance", "none"):
        disc = PatchGANDiscriminator(in_channels=6, ndf=16, n_layers=2, norm_layer=norm)
        x = torch.randn(1, 6, 64, 64)
        out = disc(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1


def test_patchgan_discriminator_invalid_norm() -> None:
    """Verify unsupported normalization layer raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported norm_layer"):
        PatchGANDiscriminator(in_channels=6, norm_layer="invalid_norm")


def test_init_weights() -> None:
    """Verify standard weight initialization runs without error."""
    net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
    )
    init_weights(net, init_type="normal", init_gain=0.02)
    # Check that weights are initialized
    assert net[0].weight.data.std().item() > 0.0


def test_gan_loss_bce() -> None:
    """Verify GANLoss with standard BCE computes correct values for real and fake predictions."""
    gan_loss = GANLoss(use_lsgan=False)
    pred = torch.randn(2, 1, 30, 30, requires_grad=True)

    loss_real = gan_loss(pred, target_is_real=True)
    loss_fake = gan_loss(pred, target_is_real=False)

    assert loss_real.item() > 0.0
    assert loss_fake.item() > 0.0

    # Backpropagation check
    loss_real.backward()
    assert pred.grad is not None


def test_gan_loss_lsgan() -> None:
    """Verify GANLoss with LSGAN (MSE) operates properly."""
    gan_loss = GANLoss(use_lsgan=True)
    pred = torch.ones(2, 1, 10, 10)

    # When pred is 1.0 and target is real (1.0), MSE should be 0
    loss_real = gan_loss(pred, target_is_real=True)
    assert pytest.approx(loss_real.item(), abs=1e-5) == 0.0

    # When pred is 1.0 and target is fake (0.0), MSE should be 1.0
    loss_fake = gan_loss(pred, target_is_real=False)
    assert pytest.approx(loss_fake.item(), abs=1e-5) == 1.0


def test_cgan_adversarial_step_flow() -> None:
    """Verify a complete generator-discriminator forward and backward step executes end-to-end."""
    generator = UNet(in_channels=3, out_channels=3, base_channels=16)
    discriminator = PatchGANDiscriminator(in_channels=6, ndf=16, n_layers=2)
    gan_loss = GANLoss(use_lsgan=False)
    l1_loss = nn.L1Loss()

    cloudy = torch.randn(2, 3, 64, 64)
    clear = torch.randn(2, 3, 64, 64)

    # 1. Train Discriminator
    fake_clear = generator(cloudy)
    real_pair = torch.cat([cloudy, clear], dim=1)
    fake_pair = torch.cat([cloudy, fake_clear.detach()], dim=1)

    pred_real = discriminator(real_pair)
    pred_fake = discriminator(fake_pair)

    loss_d_real = gan_loss(pred_real, target_is_real=True)
    loss_d_fake = gan_loss(pred_fake, target_is_real=False)
    loss_d = (loss_d_real + loss_d_fake) * 0.5

    loss_d.backward()

    # 2. Train Generator
    fake_pair_for_g = torch.cat([cloudy, fake_clear], dim=1)
    pred_fake_g = discriminator(fake_pair_for_g)

    loss_g_gan = gan_loss(pred_fake_g, target_is_real=True)
    loss_g_l1 = l1_loss(fake_clear, clear)
    loss_g = loss_g_gan + 100.0 * loss_g_l1

    loss_g.backward()

    assert fake_clear.shape == clear.shape
    assert loss_g.item() > 0.0
    assert loss_d.item() > 0.0
