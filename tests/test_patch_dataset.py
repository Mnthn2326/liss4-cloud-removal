"""Unit tests for patch extraction, spatial dataset splitting, and PyTorch Dataset."""

import numpy as np
import pytest
import torch

from cloud_removal.patch_dataset import (
    DatasetSplits,
    LISS4PatchDataset,
    PatchConfig,
    assign_spatial_split,
    extract_paired_patches,
)


@pytest.fixture
def synthetic_scene_fixture():
    """Create a synthetic 3-band scene (128x128) with valid and collar regions."""
    height, width = 128, 128
    channels = 3

    # Generate synthetic clear raster
    clear = np.full((channels, height, width), 250, dtype=np.uint16)

    # Synthetic cloud mask (opacity 0.4 on bottom half)
    cloud_mask = np.zeros((height, width), dtype=np.float32)
    cloud_mask[64:, :] = 0.4

    # Synthetic cloudy raster
    cloudy = clear.copy()
    cloudy[:, 64:, :] = 550

    # Validity mask: left half (cols 0-63) is valid terrain (1), right half (cols 64-127) has collar (0)
    valid_mask = np.ones((height, width), dtype=np.uint8)
    valid_mask[:, 96:] = 0  # Rightmost 32 cols are collar padding

    return clear, cloudy, cloud_mask, valid_mask


def test_extract_patches_alignment_and_dimensions(synthetic_scene_fixture):
    """Verify extracted patches maintain consistent dimensions and cross-layer alignment."""
    clear, cloudy, cloud_mask, valid_mask = synthetic_scene_fixture
    cfg = PatchConfig(patch_size=32, stride=32, min_valid_ratio=0.80)

    splits = extract_paired_patches(
        clear_raster=clear,
        cloudy_raster=cloudy,
        cloud_mask=cloud_mask,
        valid_mask=valid_mask,
        config=cfg,
    )

    all_samples = splits.train + splits.val + splits.test
    assert len(all_samples) > 0

    for sample in all_samples:
        assert sample.clear.shape == (3, 32, 32)
        assert sample.cloudy.shape == (3, 32, 32)
        assert sample.cloud_mask.shape == (1, 32, 32)
        assert sample.valid_mask.shape == (1, 32, 32)
        assert sample.clear.dtype == np.uint16
        assert sample.cloudy.dtype == np.uint16

        # Check alignment: row_off and col_off match data extracted
        r = sample.metadata.row_off
        c = sample.metadata.col_off
        np.testing.assert_array_equal(sample.clear, clear[:, r : r + 32, c : c + 32])
        np.testing.assert_array_equal(sample.cloudy, cloudy[:, r : r + 32, c : c + 32])


def test_collar_rejection():
    """Verify that patches containing mostly collar pixels are rejected."""
    clear = np.full((3, 64, 64), 300, dtype=np.uint16)
    cloudy = clear.copy()
    cloud_mask = np.zeros((64, 64), dtype=np.float32)

    # Validity mask: top half valid (1), bottom half collar (0)
    valid_mask = np.ones((64, 64), dtype=np.uint8)
    valid_mask[32:, :] = 0

    cfg = PatchConfig(patch_size=32, stride=32, min_valid_ratio=0.95)
    splits = extract_paired_patches(clear, cloudy, cloud_mask, valid_mask, config=cfg)

    # In a 64x64 raster with patch_size 32:
    # 4 patches total: (0,0), (0,32), (32,0), (32,32).
    # The top two (rows 0-31) have 100% valid pixels -> accepted.
    # The bottom two (rows 32-63) have 0% valid pixels -> rejected.
    assert splits.total_evaluated == 4
    assert splits.total_accepted == 2
    assert splits.total_rejected == 2


def test_spatial_split_partitioning_disjoint_ids():
    """Verify that spatial regional splitting produces disjoint patch sets without ID collisions."""
    clear = np.full((3, 128, 128), 200, dtype=np.uint16)
    cloudy = clear.copy()
    cloud_mask = np.zeros((128, 128), dtype=np.float32)
    valid_mask = np.ones((128, 128), dtype=np.uint8)

    cfg = PatchConfig(patch_size=32, stride=32, train_ratio=0.50, val_ratio=0.25, test_ratio=0.25)
    splits = extract_paired_patches(clear, cloudy, cloud_mask, valid_mask, config=cfg)

    train_ids = {s.metadata.patch_id for s in splits.train}
    val_ids = {s.metadata.patch_id for s in splits.val}
    test_ids = {s.metadata.patch_id for s in splits.test}

    # Verify no overlap between splits
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Verify geographic separation along row axis
    # train (rows < 64), val (rows 64 to 95), test (rows 96 to 127)
    for s in splits.train:
        assert s.metadata.row_off < 64
        assert s.metadata.split == "train"

    for s in splits.val:
        assert 64 <= s.metadata.row_off < 96
        assert s.metadata.split == "val"

    for s in splits.test:
        assert s.metadata.row_off >= 96
        assert s.metadata.split == "test"


def test_pytorch_dataset_wrapper(synthetic_scene_fixture):
    """Verify that LISS4PatchDataset produces correct PyTorch tensors and normalized values."""
    clear, cloudy, cloud_mask, valid_mask = synthetic_scene_fixture
    cfg = PatchConfig(patch_size=32, stride=32)
    splits = extract_paired_patches(clear, cloudy, cloud_mask, valid_mask, config=cfg)

    dataset = LISS4PatchDataset(splits.train, normalize_max=1023.0)
    assert len(dataset) == len(splits.train)

    if len(dataset) > 0:
        item = dataset[0]
        assert isinstance(item["clear"], torch.Tensor)
        assert isinstance(item["cloudy"], torch.Tensor)
        assert isinstance(item["cloud_mask"], torch.Tensor)
        assert isinstance(item["valid_mask"], torch.Tensor)

        assert item["clear"].dtype == torch.float32
        assert item["clear"].shape == (3, 32, 32)
        assert item["cloudy"].shape == (3, 32, 32)
        assert item["cloud_mask"].shape == (1, 32, 32)
        assert item["valid_mask"].shape == (1, 32, 32)

        # Values scaled to [0, 1]
        assert 0.0 <= float(item["clear"].min())
        assert float(item["clear"].max()) <= 1.0


def test_invalid_patch_config_raises_errors():
    """Verify that invalid patch configurations raise informative ValueError."""
    with pytest.raises(ValueError, match="patch_size must be positive"):
        PatchConfig(patch_size=0)

    with pytest.raises(ValueError, match="stride must be positive"):
        PatchConfig(stride=-5)

    with pytest.raises(ValueError, match="min_valid_ratio must be in"):
        PatchConfig(min_valid_ratio=1.5)

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        PatchConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    with pytest.raises(ValueError, match="split_axis must be"):
        PatchConfig(split_axis="invalid_axis")
