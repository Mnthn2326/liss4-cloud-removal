"""Geospatial patch extraction and spatial dataset splitting for cloud removal.

Provides memory-efficient windowed patch extraction, spatial validity filtering,
leakage-preventing geographic train/validation/test partitioning, and a PyTorch
Dataset wrapper for supervised paired training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatchConfig:
    """Configuration for spatial patch extraction and geographic dataset splitting.

    Attributes:
        patch_size: Square tile dimension in pixels (default: 256).
        stride: Step size between adjacent patches (default: None, equal to patch_size).
        min_valid_ratio: Minimum ratio of valid non-collar pixels required to retain a patch (0.0 to 1.0).
        train_ratio: Fraction of spatial extent allocated to training split.
        val_ratio: Fraction of spatial extent allocated to validation split.
        test_ratio: Fraction of spatial extent allocated to test split.
        split_axis: Spatial axis for non-overlapping regional splitting ("spatial_y" or "spatial_x").
    """

    patch_size: int = 256
    stride: Optional[int] = None
    min_valid_ratio: float = 0.95
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_axis: str = "spatial_y"

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.stride is not None and self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if not (0.0 <= self.min_valid_ratio <= 1.0):
            raise ValueError(f"min_valid_ratio must be in [0.0, 1.0], got {self.min_valid_ratio}")
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            raise ValueError(
                f"Split ratios must sum to 1.0: train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio} (sum={total_ratio})"
            )
        if self.split_axis not in ("spatial_y", "spatial_x"):
            raise ValueError(f"split_axis must be 'spatial_y' or 'spatial_x', got '{self.split_axis}'")


@dataclass
class PatchMetadata:
    """Traceability metadata for an individual extracted patch."""

    patch_id: str
    row_off: int
    col_off: int
    height: int
    width: int
    valid_ratio: float
    mean_cloud_opacity: float
    split: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "patch_id": self.patch_id,
            "row_off": self.row_off,
            "col_off": self.col_off,
            "height": self.height,
            "width": self.width,
            "valid_ratio": float(self.valid_ratio),
            "mean_cloud_opacity": float(self.mean_cloud_opacity),
            "split": self.split,
        }


@dataclass
class PatchSample:
    """Paired multi-modal patch sample."""

    metadata: PatchMetadata
    clear: np.ndarray       # (C, H, W)
    cloudy: np.ndarray      # (C, H, W)
    cloud_mask: np.ndarray  # (1, H, W)
    valid_mask: np.ndarray  # (1, H, W)


@dataclass
class DatasetSplits:
    """Collection of extracted patches organized by spatial split."""

    train: List[PatchSample] = field(default_factory=list)
    val: List[PatchSample] = field(default_factory=list)
    test: List[PatchSample] = field(default_factory=list)
    total_evaluated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    config: Optional[PatchConfig] = None

    def summary(self) -> Dict[str, Any]:
        """Return summary dictionary of split counts and rejection statistics."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "train_count": len(self.train),
            "val_count": len(self.val),
            "test_count": len(self.test),
            "acceptance_rate": (
                float(self.total_accepted / self.total_evaluated)
                if self.total_evaluated > 0
                else 0.0
            ),
        }


def assign_spatial_split(
    row_off: int,
    col_off: int,
    patch_size: int,
    total_height: int,
    total_width: int,
    config: PatchConfig,
) -> str:
    """Assign a patch to train, val, or test split based on its spatial coordinate.

    Partitions the geographic coordinate space into contiguous regional zones
    before patch extraction, preventing spatial autocorrelation leakage between splits.

    Args:
        row_off: Row pixel offset of patch top-left corner.
        col_off: Column pixel offset of patch top-left corner.
        patch_size: Patch square dimension.
        total_height: Full raster height in pixels.
        total_width: Full raster width in pixels.
        config: PatchConfig with split ratios and split axis.

    Returns:
        One of 'train', 'val', or 'test'.
    """
    if config.split_axis == "spatial_y":
        # Split along row/latitude axis
        center_y = row_off + patch_size / 2.0
        train_bound = total_height * config.train_ratio
        val_bound = total_height * (config.train_ratio + config.val_ratio)

        if center_y < train_bound:
            return "train"
        if center_y < val_bound:
            return "val"
        return "test"
    else:
        # Split along column/longitude axis
        center_x = col_off + patch_size / 2.0
        train_bound = total_width * config.train_ratio
        val_bound = total_width * (config.train_ratio + config.val_ratio)

        if center_x < train_bound:
            return "train"
        if center_x < val_bound:
            return "val"
        return "test"


def extract_paired_patches(
    clear_raster: np.ndarray,
    cloudy_raster: np.ndarray,
    cloud_mask: np.ndarray,
    valid_mask: np.ndarray,
    config: Optional[PatchConfig] = None,
) -> DatasetSplits:
    """Extract corresponding aligned patches across all layers and assign spatial splits.

    Filters out background collar tiles and patches where valid terrain ratio falls
    below the configured min_valid_ratio threshold.

    Args:
        clear_raster: Ground-truth clear image of shape (C, H, W).
        cloudy_raster: Synthetic cloudy image of shape (C, H, W).
        cloud_mask: Synthetic cloud opacity mask of shape (H, W) or (1, H, W).
        valid_mask: Spatial validity/collar mask of shape (H, W) or (1, H, W).
        config: Optional PatchConfig instance.

    Returns:
        DatasetSplits instance containing train, val, and test patch samples.

    Raises:
        ValueError: If array shapes or dimensions are incompatible.
    """
    cfg = config if config is not None else PatchConfig()
    effective_stride = cfg.stride if cfg.stride is not None else cfg.patch_size

    if clear_raster.ndim != 3 or cloudy_raster.ndim != 3:
        raise ValueError(
            f"clear_raster and cloudy_raster must be 3D arrays (C, H, W), "
            f"got {clear_raster.shape} and {cloudy_raster.shape}"
        )
    if clear_raster.shape != cloudy_raster.shape:
        raise ValueError(
            f"clear_raster shape {clear_raster.shape} != cloudy_raster shape {cloudy_raster.shape}"
        )

    num_channels, height, width = clear_raster.shape

    # Normalize mask shapes to (1, H, W)
    if cloud_mask.ndim == 2:
        if cloud_mask.shape != (height, width):
            raise ValueError(f"cloud_mask shape {cloud_mask.shape} != spatial shape ({height}, {width})")
        c_mask_3d = cloud_mask[np.newaxis, :, :]
    elif cloud_mask.ndim == 3 and cloud_mask.shape[0] == 1:
        if cloud_mask.shape[1:] != (height, width):
            raise ValueError(f"cloud_mask shape {cloud_mask.shape} != spatial shape ({height}, {width})")
        c_mask_3d = cloud_mask
    else:
        raise ValueError(f"cloud_mask must have shape (H, W) or (1, H, W), got {cloud_mask.shape}")

    if valid_mask.ndim == 2:
        if valid_mask.shape != (height, width):
            raise ValueError(f"valid_mask shape {valid_mask.shape} != spatial shape ({height}, {width})")
        v_mask_3d = valid_mask[np.newaxis, :, :]
    elif valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        if valid_mask.shape[1:] != (height, width):
            raise ValueError(f"valid_mask shape {valid_mask.shape} != spatial shape ({height}, {width})")
        v_mask_3d = valid_mask
    else:
        raise ValueError(f"valid_mask must have shape (H, W) or (1, H, W), got {valid_mask.shape}")

    splits = DatasetSplits(config=cfg)
    patch_size = cfg.patch_size

    # Slide window across spatial grid
    # Stop before partial boundary patches to guarantee exact square dimensions
    row_starts = range(0, height - patch_size + 1, effective_stride)
    col_starts = range(0, width - patch_size + 1, effective_stride)

    for r in row_starts:
        for c in col_starts:
            splits.total_evaluated += 1

            v_sub = v_mask_3d[0, r : r + patch_size, c : c + patch_size]
            valid_pixels = int(np.count_nonzero(v_sub == 1))
            total_pixels = patch_size * patch_size
            valid_ratio = float(valid_pixels / total_pixels)

            # Filter collar background
            if valid_ratio < cfg.min_valid_ratio:
                splits.total_rejected += 1
                continue

            # Extract patches
            clear_sub = clear_raster[:, r : r + patch_size, c : c + patch_size].copy()
            cloudy_sub = cloudy_raster[:, r : r + patch_size, c : c + patch_size].copy()
            c_mask_sub = c_mask_3d[:, r : r + patch_size, c : c + patch_size].copy()
            v_mask_sub = v_mask_3d[:, r : r + patch_size, c : c + patch_size].copy()

            mean_cloud = float(np.mean(c_mask_sub))
            split_name = assign_spatial_split(
                row_off=r,
                col_off=c,
                patch_size=patch_size,
                total_height=height,
                total_width=width,
                config=cfg,
            )

            patch_id = f"patch_r{r:05d}_c{c:05d}"
            meta = PatchMetadata(
                patch_id=patch_id,
                row_off=r,
                col_off=c,
                height=patch_size,
                width=patch_size,
                valid_ratio=valid_ratio,
                mean_cloud_opacity=mean_cloud,
                split=split_name,
            )

            sample = PatchSample(
                metadata=meta,
                clear=clear_sub,
                cloudy=cloudy_sub,
                cloud_mask=c_mask_sub,
                valid_mask=v_mask_sub,
            )

            if split_name == "train":
                splits.train.append(sample)
            elif split_name == "val":
                splits.val.append(sample)
            else:
                splits.test.append(sample)

            splits.total_accepted += 1

    return splits


class LISS4PatchDataset(Dataset):
    """PyTorch Dataset for paired satellite cloud removal patches.

    Wraps extracted PatchSample instances and converts them to torch.FloatTensor
    with configurable normalization options.
    """

    def __init__(
        self,
        samples: Sequence[PatchSample],
        normalize_max: float = 1023.0,  # 10-bit dynamic range for LISS-IV
    ) -> None:
        """Initialize the PyTorch Dataset.

        Args:
            samples: Sequence of PatchSample objects.
            normalize_max: Maximum value for linear scaling to [0.0, 1.0].
                Set to 1.0 to disable scaling.
        """
        self.samples = list(samples)
        self.normalize_max = float(normalize_max) if normalize_max > 0 else 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        sample = self.samples[idx]

        # Convert to float32 tensors and scale to [0, 1]
        clear_t = torch.from_numpy(sample.clear.astype(np.float32)) / self.normalize_max
        cloudy_t = torch.from_numpy(sample.cloudy.astype(np.float32)) / self.normalize_max
        cloud_mask_t = torch.from_numpy(sample.cloud_mask.astype(np.float32))
        valid_mask_t = torch.from_numpy(sample.valid_mask.astype(np.float32))

        return {
            "clear": clear_t,
            "cloudy": cloudy_t,
            "cloud_mask": cloud_mask_t,
            "valid_mask": valid_mask_t,
            "patch_id": sample.metadata.patch_id,
            "split": sample.metadata.split,
        }
