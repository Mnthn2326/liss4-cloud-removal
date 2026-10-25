"""Unit tests for IIRS/ISRO Pix2Pix U-Net Generator and checkpoint conversion."""

from pathlib import Path

import pytest
import torch

from cloud_removal.infer import load_model_from_checkpoint, predict_patch
from cloud_removal.models.iirs_pix2pix import IIRSPix2PixGenerator


def test_iirs_pix2pix_initialization() -> None:
    """Test model structure, parameter counting, and config metadata."""
    model = IIRSPix2PixGenerator(in_channels=3, out_channels=3, rescale_reflectance=True)
    cfg = model.get_config()

    assert cfg["model_type"] == "IIRSPix2PixGenerator"
    assert cfg["in_channels"] == 3
    assert cfg["out_channels"] == 3
    assert cfg["rescale_reflectance"] is True
    assert cfg["total_trainable_params"] > 50_000_000


def test_iirs_pix2pix_forward_rescaled() -> None:
    """Test forward pass bounds output to [0, 1] when rescale_reflectance=True."""
    model = IIRSPix2PixGenerator(in_channels=3, out_channels=3, rescale_reflectance=True)
    model.eval()

    # Input in [0, 1]
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 256, 256)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_iirs_pix2pix_forward_raw() -> None:
    """Test forward pass bounds output to [-1, 1] when rescale_reflectance=False."""
    model = IIRSPix2PixGenerator(in_channels=3, out_channels=3, rescale_reflectance=False)
    model.eval()

    # Input in [-1, 1]
    x = torch.rand(1, 3, 256, 256) * 2.0 - 1.0
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 256, 256)
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_iirs_pix2pix_checkpoint_loading_and_inference(tmp_path: Path) -> None:
    """Test saving and loading an IIRSPix2PixGenerator checkpoint via infer.py."""
    model = IIRSPix2PixGenerator(in_channels=3, out_channels=3, rescale_reflectance=True)
    ckpt_path = tmp_path / "test_iirs_checkpoint.pt"

    payload = {
        "epoch": 1600,
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
    }
    torch.save(payload, ckpt_path)

    loaded_model, ckpt_dict = load_model_from_checkpoint(ckpt_path)
    assert isinstance(loaded_model, IIRSPix2PixGenerator)
    assert ckpt_dict["epoch"] == 1600

    # Test predict_patch helper
    cloudy_patch = torch.rand(3, 256, 256)
    pred = predict_patch(loaded_model, cloudy_patch)
    assert pred.shape == (3, 256, 256)
    assert pred.min().item() >= 0.0
    assert pred.max().item() <= 1.0
