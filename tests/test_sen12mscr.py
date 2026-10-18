"""Unit tests for SEN12MS-CR dataset loader and preprocessing adapter."""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from cloud_removal.sen12mscr import (
    DEFAULT_RGB_BANDS,
    PairedOpticalSample,
    SEN12MSCRConfig,
    SEN12MSCRDataset,
    validate_sen12mscr_subset,
)


@pytest.fixture
def mock_sen12mscr_fixture(tmp_path: Path) -> Path:
    """Create a lightweight synthetic SEN12MS-CR directory structure and manifest."""
    base_dir = tmp_path / "mock_sen12mscr"
    cloudy_dir = base_dir / "ROIs2017_winter_s2_cloudy" / "s2_cloudy_102"
    clear_dir = base_dir / "ROIs2017_winter_s2" / "s2_102"
    cloudy_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)

    width, height = 16, 16
    transform = from_origin(100.0, 100.0, 10.0, 10.0)
    crs = "EPSG:32630"

    paired_samples = []
    # Create 6 synthetic patch pairs
    for i in range(1, 7):
        p_name = f"p{i:03d}"
        c_fname = f"ROIs2017_winter_s2_cloudy_102_{p_name}.tif"
        clr_fname = f"ROIs2017_winter_s2_102_{p_name}.tif"

        c_path = cloudy_dir / c_fname
        clr_path = clear_dir / clr_fname

        # 13 bands of uint16 data
        # Inject deterministic values: Band 4 (Red) has value 4000, Band 3 (Green) 3000, Band 2 (Blue) 2000
        data_cloudy = np.zeros((13, height, width), dtype=np.uint16)
        data_clear = np.zeros((13, height, width), dtype=np.uint16)

        data_cloudy[3] = 4000  # B4
        data_cloudy[2] = 3000  # B3
        data_cloudy[1] = 2000  # B2

        data_clear[3] = 2000
        data_clear[2] = 1500
        data_clear[1] = 1000

        for path, data in [(c_path, data_cloudy), (clr_path, data_clear)]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=13,
                dtype=np.uint16,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(data)

        paired_samples.append({
            "patch_id": p_name,
            "cloudy_file": c_fname,
            "clear_file": clr_fname,
            "size_bytes_cloudy": c_path.stat().st_size,
            "size_bytes_clear": clr_path.stat().st_size,
        })

    manifest = {
        "dataset_name": "SEN12MS-CR",
        "official_split": "train",
        "official_split_id": 1,
        "total_paired_patches": len(paired_samples),
        "cloudy_directory": "ROIs2017_winter_s2_cloudy/s2_cloudy_102",
        "clear_directory": "ROIs2017_winter_s2/s2_102",
        "paired_samples": paired_samples,
    }

    manifest_path = base_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def test_dataset_pairing_and_tensor_shapes(mock_sen12mscr_fixture: Path):
    """Verify cloudy and clear optical patches are paired and output correct tensor shapes."""
    dataset = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="all")
    assert len(dataset) == 6

    item = dataset[0]
    assert isinstance(item["cloudy"], torch.Tensor)
    assert isinstance(item["clear"], torch.Tensor)
    assert item["cloudy"].shape == (3, 16, 16)
    assert item["clear"].shape == (3, 16, 16)
    assert item["cloudy"].dtype == torch.float32
    assert item["clear"].dtype == torch.float32


def test_reflectance_normalization(mock_sen12mscr_fixture: Path):
    """Verify digital numbers are properly scaled by 10000 to surface reflectance [0, 1]."""
    dataset = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="all")
    item = dataset[0]

    # In fixture, B4=4000, B3=3000, B2=2000 for cloudy
    # After normalization: B4=0.4, B3=0.3, B2=0.2
    cloudy_tensor = item["cloudy"]
    assert np.isclose(float(cloudy_tensor[0, 0, 0]), 0.4, atol=1e-4)  # B4
    assert np.isclose(float(cloudy_tensor[1, 0, 0]), 0.3, atol=1e-4)  # B3
    assert np.isclose(float(cloudy_tensor[2, 0, 0]), 0.2, atol=1e-4)  # B2

    clear_tensor = item["clear"]
    assert np.isclose(float(clear_tensor[0, 0, 0]), 0.2, atol=1e-4)
    assert np.isclose(float(clear_tensor[1, 0, 0]), 0.15, atol=1e-4)
    assert np.isclose(float(clear_tensor[2, 0, 0]), 0.1, atol=1e-4)


def test_deterministic_split_partitioning(mock_sen12mscr_fixture: Path):
    """Verify deterministic train/val partitioning with no shared patch IDs."""
    cfg = SEN12MSCRConfig(train_ratio=0.667, val_ratio=0.333, seed=42)
    train_ds = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="train", config=cfg)
    val_ds = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="val", config=cfg)

    assert len(train_ds) == 4
    assert len(val_ds) == 2
    assert len(train_ds) + len(val_ds) == 6

    train_ids = {train_ds[i]["patch_id"] for i in range(len(train_ds))}
    val_ids = {val_ds[i]["patch_id"] for i in range(len(val_ds))}

    assert train_ids.isdisjoint(val_ids)

    # Identical seed reproduces exact splits
    train_ds2 = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="train", config=cfg)
    train_ids2 = {train_ds2[i]["patch_id"] for i in range(len(train_ds2))}
    assert train_ids == train_ids2


def test_invalid_configurations_raise_errors(mock_sen12mscr_fixture: Path):
    """Verify that invalid parameters and missing files raise clear errors."""
    with pytest.raises(ValueError, match="Invalid Sentinel-2 band index"):
        SEN12MSCRConfig(bands=(4, 3, 99))

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SEN12MSCRConfig(train_ratio=0.5, val_ratio=0.2)

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        SEN12MSCRDataset.from_manifest("non_existent_manifest.json")


def test_real_acquired_dataset_if_present():
    """Verify real acquired SEN12MS-CR dataset integrity if present locally."""
    manifest_path = Path("data/raw/sen12mscr/manifest.json")
    if not manifest_path.is_file():
        pytest.skip("Real SEN12MS-CR dataset manifest not present")

    summary = validate_sen12mscr_subset(manifest_path)
    assert summary["status"] == "valid"
    assert summary["valid_pairs"] == 719
    assert summary["corrupt_pairs"] == 0

    dataset = SEN12MSCRDataset.from_manifest(manifest_path, split="train")
    assert len(dataset) > 500
    sample = dataset[0]
    assert sample["cloudy"].shape == (3, 256, 256)
    assert sample["clear"].shape == (3, 256, 256)
