"""Unit tests for SEN12MS-CR dataset loader, preprocessing adapter, and geometric data augmentation."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from cloud_removal.sen12mscr import (
    SEN12MSCRConfig,
    SEN12MSCRDataset,
    apply_geometric_augmentations,
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

        paired_samples.append(
            {
                "patch_id": p_name,
                "cloudy_file": c_fname,
                "clear_file": clr_fname,
                "size_bytes_cloudy": c_path.stat().st_size,
                "size_bytes_clear": clr_path.stat().st_size,
            }
        )

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


# ==============================================================================
# Geometric Data Augmentation Unit Tests (Task 2)
# ==============================================================================


def test_apply_geometric_augmentations_flips_and_rotations():
    """Verify that horizontal, vertical flips and rotations preserve values and structure."""
    # 3-channel synthetic patch with distinct pixel values across axes
    c = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    clr = c * 2.0
    mask = (c[:1] > 20).float()

    # 1. Horizontal flip
    c_hf, clr_hf, m_hf = apply_geometric_augmentations(
        c, clr, mask, hflip=True, vflip=False, rot90_k=0
    )
    assert torch.equal(c_hf, torch.flip(c, dims=[-1]))
    assert torch.equal(clr_hf, c_hf * 2.0)
    assert torch.equal(m_hf, torch.flip(mask, dims=[-1]))

    # 2. Vertical flip
    c_vf, clr_vf, m_vf = apply_geometric_augmentations(
        c, clr, mask, hflip=False, vflip=True, rot90_k=0
    )
    assert torch.equal(c_vf, torch.flip(c, dims=[-2]))
    assert torch.equal(clr_vf, c_vf * 2.0)
    assert torch.equal(m_vf, torch.flip(mask, dims=[-2]))

    # 3. 90-degree rotation (k=1)
    c_r1, clr_r1, m_r1 = apply_geometric_augmentations(
        c, clr, mask, hflip=False, vflip=False, rot90_k=1
    )
    assert torch.equal(c_r1, torch.rot90(c, k=1, dims=[-2, -1]))
    assert torch.equal(clr_r1, c_r1 * 2.0)
    assert torch.equal(m_r1, torch.rot90(mask, k=1, dims=[-2, -1]))

    # 4. 180-degree rotation (k=2)
    c_r2, clr_r2 = apply_geometric_augmentations(c, clr, None, rot90_k=2)
    assert torch.equal(c_r2, torch.rot90(c, k=2, dims=[-2, -1]))

    # 5. 270-degree rotation (k=3)
    c_r3, clr_r3 = apply_geometric_augmentations(c, clr, None, rot90_k=3)
    assert torch.equal(c_r3, torch.rot90(c, k=3, dims=[-2, -1]))

    # 6. 360-degree rotation (k=4) should be identity
    c_r4, clr_r4 = apply_geometric_augmentations(c, clr, None, rot90_k=4)
    assert torch.equal(c_r4, c)


def test_apply_geometric_augmentations_rectangular_tensors():
    """Verify that rectangular images correctly swap spatial dimensions upon 90°/270° rotation."""
    H, W = 10, 20
    c_rect = torch.arange(H * W, dtype=torch.float32).reshape(1, H, W).repeat(3, 1, 1)
    clr_rect = c_rect + 5.0
    mask_rect = torch.ones((1, H, W), dtype=torch.float32)

    c_rot90, clr_rot90, m_rot90 = apply_geometric_augmentations(
        c_rect, clr_rect, mask_rect, hflip=False, vflip=False, rot90_k=1
    )
    assert c_rot90.shape == (3, W, H)
    assert clr_rot90.shape == (3, W, H)
    assert m_rot90.shape == (1, W, H)
    # Check pixel value conservation
    assert torch.equal(torch.sort(c_rot90.flatten())[0], torch.sort(c_rect.flatten())[0])
    # Check alignment preservation
    assert torch.equal(clr_rot90, c_rot90 + 5.0)


def test_apply_geometric_augmentations_numpy():
    """Verify that NumPy arrays are handled identically without type conversion errors."""
    c_np = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    clr_np = c_np * 1.5

    c_aug, clr_aug = apply_geometric_augmentations(c_np, clr_np, hflip=True, rot90_k=1)
    assert isinstance(c_aug, np.ndarray)
    assert isinstance(clr_aug, np.ndarray)
    assert np.allclose(clr_aug, c_aug * 1.5)


def test_dataset_augmentation_training_only(mock_sen12mscr_fixture: Path):
    """Verify that augment=True enables augmentation for train split while leaving val unaugmented."""
    cfg_aug = SEN12MSCRConfig(train_ratio=0.667, val_ratio=0.333, augment=True, seed=42)
    train_ds = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="train", config=cfg_aug)
    val_ds = SEN12MSCRDataset.from_manifest(mock_sen12mscr_fixture, split="val", config=cfg_aug)

    assert train_ds.augment is True
    # Validation dataset must remain deterministic and unaugmented
    # Even when config has augment=True, val split items must not be augmented
    item_val1 = val_ds[0]
    item_val2 = val_ds[0]
    assert torch.equal(item_val1["cloudy"], item_val2["cloudy"])
    assert torch.equal(item_val1["clear"], item_val2["clear"])


def test_dataset_disabling_augmentation(mock_sen12mscr_fixture: Path):
    """Verify that when augment=False, dataset output is identical across repeated calls."""
    cfg_no_aug = SEN12MSCRConfig(augment=False)
    train_ds = SEN12MSCRDataset.from_manifest(
        mock_sen12mscr_fixture, split="train", config=cfg_no_aug
    )

    item1 = train_ds[0]
    item2 = train_ds[0]
    assert torch.equal(item1["cloudy"], item2["cloudy"])
    assert torch.equal(item1["clear"], item2["clear"])
