"""Unit and integration tests for False-Color Composite (FCC) U-Net configuration and dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml

from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import (
    DEFAULT_FCC_BANDS,
    DEFAULT_FCC_NAMES,
    S2_BAND_NAME_TO_INDEX,
    SEN12MSCRConfig,
    SEN12MSCRDataset,
)
from cloud_removal.train import load_config, train_model

# ==============================================================================
# FCC Band Selection & Configuration Tests (Task 1)
# ==============================================================================


def test_fcc_band_constants():
    """Verify that FCC band definitions match physical specifications."""
    assert DEFAULT_FCC_BANDS == (8, 4, 3)
    assert DEFAULT_FCC_NAMES == ("B08", "B04", "B03")
    assert S2_BAND_NAME_TO_INDEX["B08"] == 8
    assert S2_BAND_NAME_TO_INDEX["B04"] == 4
    assert S2_BAND_NAME_TO_INDEX["B03"] == 3


def test_sen12mscr_config_fcc_creation():
    """SEN12MSCRConfig must support integer and string band inputs for FCC."""
    # 1. Via factory method
    cfg_factory = SEN12MSCRConfig.create_fcc()
    assert cfg_factory.bands == (8, 4, 3)
    assert cfg_factory.band_names == ("B08", "B04", "B03")
    assert cfg_factory.composite_mode == "fcc"

    # 2. Via string band identifiers
    cfg_str = SEN12MSCRConfig(bands=("B08", "B04", "B03"))
    assert cfg_str.bands == (8, 4, 3)
    assert cfg_str.band_names == ("B08", "B04", "B03")
    assert cfg_str.composite_mode == "fcc"

    # 3. Via direct integer indices
    cfg_int = SEN12MSCRConfig(bands=(8, 4, 3), band_names=("B08", "B04", "B03"))
    assert cfg_int.bands == (8, 4, 3)
    assert cfg_int.composite_mode == "fcc"


def test_sen12mscr_config_invalid_bands():
    """Invalid band indices or names must raise ValueError."""
    with pytest.raises(ValueError, match="Unrecognized Sentinel-2 band"):
        SEN12MSCRConfig(bands=("B99", "B04", "B03"))

    with pytest.raises(ValueError, match="Invalid Sentinel-2 band index"):
        SEN12MSCRConfig(bands=(14, 4, 3))

    with pytest.raises(ValueError, match="Mismatch"):
        SEN12MSCRConfig(bands=(8, 4, 3), band_names=("B08", "B04"))


def test_fcc_synthetic_dataset_spectral_order(tmp_path: Path):
    """Verify that SEN12MSCRDataset extracts bands in the exact requested FCC order."""
    # Create a synthetic 13-band GeoTIFF where each band i has pixel value i * 100
    H, W = 16, 16
    band_count = 13
    data = np.zeros((band_count, H, W), dtype=np.uint16)
    for b in range(1, band_count + 1):
        data[b - 1, :, :] = b * 100

    cloudy_path = tmp_path / "s2_cloudy_sample.tif"
    clear_path = tmp_path / "s2_clear_sample.tif"

    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": band_count,
        "dtype": "uint16",
    }
    with rasterio.open(cloudy_path, "w", **profile) as dst:
        dst.write(data)
    with rasterio.open(clear_path, "w", **profile) as dst:
        dst.write(data)

    # Create dataset manifest
    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "cloudy_directory": "",
        "clear_directory": "",
        "paired_samples": [
            {
                "patch_id": "p_test",
                "cloudy_file": cloudy_path.name,
                "clear_file": clear_path.name,
            }
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Load dataset with FCC configuration [B08, B04, B03] -> (8, 4, 3)
    fcc_cfg = SEN12MSCRConfig.create_fcc(normalize=False)
    dataset = SEN12MSCRDataset.from_manifest(manifest_path, split="all", config=fcc_cfg)
    assert len(dataset) == 1

    sample = dataset[0]
    cloudy_t = sample["cloudy"]
    clear_t = sample["clear"]

    # Shape must be (3, 16, 16)
    assert cloudy_t.shape == (3, H, W)
    assert clear_t.shape == (3, H, W)

    # Identical band values for inputs and targets
    assert torch.all(cloudy_t == clear_t)

    # Channel 0 must correspond to Band 8 (pixel values 800)
    assert torch.all(cloudy_t[0] == 800.0)
    # Channel 1 must correspond to Band 4 (pixel values 400)
    assert torch.all(cloudy_t[1] == 400.0)
    # Channel 2 must correspond to Band 3 (pixel values 300)
    assert torch.all(cloudy_t[2] == 300.0)


def test_train_fcc_unet_config_parsing():
    """Verify train_fcc_unet.yaml loads with valid schema and FCC parameters."""
    cfg_path = Path("configs/train_fcc_unet.yaml")
    assert cfg_path.is_file(), f"Missing config file: {cfg_path}"

    cfg = load_config(cfg_path)
    assert cfg["identifier"] == "fcc_unet_baseline"
    assert cfg["model"]["in_channels"] == 3
    assert cfg["model"]["out_channels"] == 3
    assert cfg["dataset"]["bands"] == [8, 4, 3]
    assert cfg["dataset"]["band_names"] == ["B08", "B04", "B03"]
    assert cfg["dataset"]["composite_mode"] == "fcc"
    assert cfg["dataset"]["normalize"] is True
    assert cfg["training"]["loss"] == "compound_l1_ssim"
    assert cfg["training"]["scheduler"] == "cosine"


def test_fcc_unet_forward_pass_and_channels():
    """UNet configured for FCC must accept 3 channels and output 3 channels."""
    model = UNet(in_channels=3, out_channels=3, base_channels=16, use_sigmoid=True)
    model.eval()

    # FCC batch: (B=2, C=3 [B08, B04, B03], H=64, W=64)
    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 64, 64)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_fcc_smoke_training_step(tmp_path: Path):
    """Verify 1-step training execution with train_fcc_unet config."""
    cfg_path = Path("configs/train_fcc_unet.yaml")
    if not Path("data/raw/sen12mscr/manifest.json").is_file():
        pytest.skip("SEN12MS-CR dataset manifest not present")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Fast test overrides
    cfg["checkpoint_dir"] = str(tmp_path / "checkpoints_fcc")
    cfg["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    cfg["training"]["epochs"] = 1

    summary = train_model(config=cfg, smoke_test=True)
    assert summary is not None
    assert Path(summary["best_checkpoint_path"]).is_file()

    # Verify checkpoint contains FCC metadata
    ckpt = torch.load(summary["best_checkpoint_path"])
    assert ckpt["bands"] == [8, 4, 3]
    assert ckpt["band_names"] == ["B08", "B04", "B03"]
    assert ckpt["composite_mode"] == "fcc"
