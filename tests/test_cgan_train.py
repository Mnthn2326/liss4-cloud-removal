"""Unit and integration tests for Pix2Pix cGAN training pipeline and artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from cloud_removal.infer import load_model_from_checkpoint
from cloud_removal.models.patchgan import GANLoss, PatchGANDiscriminator
from cloud_removal.models.unet import UNet
from cloud_removal.train_cgan import train_cgan_model, train_cgan_one_epoch


class DummyDataset(Dataset):
    """Dummy paired dataset for fast synthetic training tests."""

    def __init__(self, size: int = 4, channels: int = 3, height: int = 64, width: int = 64) -> None:
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "cloudy": torch.rand(self.channels, self.height, self.width),
            "clear": torch.rand(self.channels, self.height, self.width),
        }


def test_train_cgan_one_epoch_step() -> None:
    """Verify train_cgan_one_epoch computes losses and updates weights."""
    device = torch.device("cpu")
    generator = UNet(in_channels=3, out_channels=3, base_channels=16).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6, ndf=16, n_layers=2).to(device)

    dataset = DummyDataset(size=4, channels=3, height=64, width=64)
    loader = DataLoader(dataset, batch_size=2)

    criterion_gan = GANLoss(use_lsgan=False).to(device)
    criterion_l1 = nn.L1Loss().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    losses, mae = train_cgan_one_epoch(
        generator=generator,
        discriminator=discriminator,
        loader=loader,
        criterion_gan=criterion_gan,
        criterion_l1=criterion_l1,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        lambda_l1=100.0,
    )

    assert "loss_d" in losses
    assert "loss_d_real" in losses
    assert "loss_d_fake" in losses
    assert "loss_g" in losses
    assert "loss_g_gan" in losses
    assert "loss_g_l1" in losses
    assert losses["loss_d"] > 0.0
    assert losses["loss_g"] > 0.0
    assert mae > 0.0


def test_train_cgan_model_smoke_test(tmp_path: Path) -> None:
    """Verify train_cgan_model runs smoke test and produces compatible checkpoints."""
    config_path = Path("configs/train_cgan.yaml")
    if not config_path.is_file():
        pytest.skip("Base cGAN training config not found")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Use small compact channel width and isolated temp paths for fast execution
    cfg["model"]["base_channels"] = 16
    cfg["discriminator"]["ndf"] = 16
    cfg["discriminator"]["n_layers"] = 2
    cfg["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    cfg["training"]["epochs"] = 1

    summary = train_cgan_model(config=cfg, smoke_test=True)

    assert summary is not None
    assert len(summary["history"]) == 1
    assert "best_val_mae" in summary
    assert Path(summary["best_generator_checkpoint"]).is_file()
    assert Path(summary["best_discriminator_checkpoint"]).is_file()

    # Verify that the generated checkpoint can be loaded by our standard inference loader
    loaded_model, ckpt_dict = load_model_from_checkpoint(
        summary["best_generator_checkpoint"],
        device=torch.device("cpu"),
    )
    assert isinstance(loaded_model, UNet)
    assert "model_state_dict" in ckpt_dict
    assert "model_config" in ckpt_dict
