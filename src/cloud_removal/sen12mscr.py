"""SEN12MS-CR dataset loader and preprocessing adapter.

Provides dataset abstractions, deterministic split partitioning, reflectance
normalization, geometric data augmentation, and PyTorch Dataset integration
for paired Sentinel-2 optical data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Standard Sentinel-2 Band Definitions (1-indexed for rasterio)
# 1: Aerosols, 2: Blue, 3: Green, 4: Red, 5: RE1, 6: RE2, 7: RE3, 8: NIR,
# 9: Narrow NIR, 10: Water Vapor, 11: Cirrus, 12: SWIR1, 13: SWIR2
DEFAULT_RGB_BANDS: Tuple[int, int, int] = (4, 3, 2)  # Red (B4), Green (B3), Blue (B2)
DEFAULT_RGB_NAMES: Tuple[str, str, str] = ("B04", "B03", "B02")

# False-Color Composite (FCC) for Cross-Sensor Alignment with ISRO Resourcesat-2A LISS-IV:
# Sentinel-2 B08 (NIR, ~842nm) -> LISS-IV Band 4 (NIR, 770-860nm)
# Sentinel-2 B04 (Red, ~665nm) -> LISS-IV Band 3 (Red, 620-680nm)
# Sentinel-2 B03 (Green, ~560nm) -> LISS-IV Band 2 (Green, 520-590nm)
DEFAULT_FCC_BANDS: Tuple[int, int, int] = (8, 4, 3)
DEFAULT_FCC_NAMES: Tuple[str, str, str] = ("B08", "B04", "B03")

# Mapping from standard Sentinel-2 band identifier names to 1-indexed raster band numbers
S2_BAND_NAME_TO_INDEX: Dict[str, int] = {
    "B01": 1,
    "B1": 1,
    "B02": 2,
    "B2": 2,
    "B03": 3,
    "B3": 3,
    "B04": 4,
    "B4": 4,
    "B05": 5,
    "B5": 5,
    "B06": 6,
    "B6": 6,
    "B07": 7,
    "B7": 7,
    "B08": 8,
    "B8": 8,
    "B8A": 9,
    "B08A": 9,
    "B09": 10,
    "B9": 10,
    "B10": 11,
    "B11": 12,
    "B12": 13,
}

S2_INDEX_TO_STANDARD_NAME: Dict[int, str] = {
    1: "B01",
    2: "B02",
    3: "B03",
    4: "B04",
    5: "B05",
    6: "B06",
    7: "B07",
    8: "B08",
    9: "B8A",
    10: "B09",
    11: "B10",
    12: "B11",
    13: "B12",
}

DEFAULT_MAX_REFLECTANCE: float = 10000.0


def apply_geometric_augmentations(
    cloudy: Union[np.ndarray, torch.Tensor],
    clear: Union[np.ndarray, torch.Tensor],
    cloud_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0,
) -> Tuple[Union[np.ndarray, torch.Tensor, None], ...]:
    """Apply synchronized geometric transformations to cloudy, clear, and mask components.

    Supports random or deterministic horizontal flip, vertical flip, and right-angle rotations
    (90, 180, 270 degrees). Strictly avoids interpolation to preserve exact spectral values.
    Correctly handles rectangular tensors with dimension swaps for 90° and 270° rotations.

    Args:
        cloudy: Cloudy image tensor (C, H, W) or numpy array (C, H, W).
        clear: Cloud-free reference image tensor or numpy array matching cloudy shape.
        cloud_mask: Optional cloud opacity mask (1, H, W) or (H, W).
        hflip: Whether to apply horizontal flip along width axis (-1).
        vflip: Whether to apply vertical flip along height axis (-2).
        rot90_k: Number of 90-degree orthogonal rotations in (H, W) plane.

    Returns:
        Tuple of (transformed_cloudy, transformed_clear) if cloud_mask is None, or
        (transformed_cloudy, transformed_clear, transformed_cloud_mask) if provided.
    """
    if isinstance(cloudy, torch.Tensor):
        c_t: torch.Tensor = cloudy
        clr_t: torch.Tensor = cast(torch.Tensor, clear)
        m_t: Optional[torch.Tensor] = cast(Optional[torch.Tensor], cloud_mask)

        if hflip:
            c_t = torch.flip(c_t, dims=[-1])
            clr_t = torch.flip(clr_t, dims=[-1])
            if m_t is not None:
                m_t = torch.flip(m_t, dims=[-1])

        if vflip:
            c_t = torch.flip(c_t, dims=[-2])
            clr_t = torch.flip(clr_t, dims=[-2])
            if m_t is not None:
                m_t = torch.flip(m_t, dims=[-2])

        if rot90_k % 4 != 0:
            k = rot90_k % 4
            c_t = torch.rot90(c_t, k=k, dims=[-2, -1])
            clr_t = torch.rot90(clr_t, k=k, dims=[-2, -1])
            if m_t is not None:
                m_t = torch.rot90(m_t, k=k, dims=[-2, -1])

        c_t = c_t.contiguous()
        clr_t = clr_t.contiguous()
        if m_t is not None:
            m_t = m_t.contiguous()

        if cloud_mask is None:
            return (c_t, clr_t)
        return (c_t, clr_t, m_t)
    else:
        c_np: np.ndarray = np.asarray(cloudy)
        clr_np: np.ndarray = np.asarray(clear)
        m_np: Optional[np.ndarray] = np.asarray(cloud_mask) if cloud_mask is not None else None

        if hflip:
            c_np = np.flip(c_np, axis=-1)
            clr_np = np.flip(clr_np, axis=-1)
            if m_np is not None:
                m_np = np.flip(m_np, axis=-1)

        if vflip:
            c_np = np.flip(c_np, axis=-2)
            clr_np = np.flip(clr_np, axis=-2)
            if m_np is not None:
                m_np = np.flip(m_np, axis=-2)

        if rot90_k % 4 != 0:
            k = rot90_k % 4
            c_np = np.rot90(c_np, k=k, axes=(-2, -1))
            clr_np = np.rot90(clr_np, k=k, axes=(-2, -1))
            if m_np is not None:
                m_np = np.rot90(m_np, k=k, axes=(-2, -1))

        c_np = np.ascontiguousarray(c_np)
        clr_np = np.ascontiguousarray(clr_np)
        if m_np is not None:
            m_np = np.ascontiguousarray(m_np)

        if cloud_mask is None:
            return (c_np, clr_np)
        return (c_np, clr_np, m_np)


@dataclass(frozen=True)
class SEN12MSCRConfig:
    """Configuration for SEN12MS-CR paired optical dataset.

    Attributes:
        bands: Tuple of 1-indexed band numbers or standard names to extract
            (default: (4, 3, 2) for RGB; (8, 4, 3) for FCC cross-sensor transfer).
        band_names: Corresponding names of selected bands (e.g. ("B08", "B04", "B03")).
        normalize: Whether to apply documented reflectance scaling (DN / 10000).
        clip_max: Maximum reflectance digital number before scaling (default: 10000.0).
        train_ratio: Fraction of subset allocated to local training split (default: 0.80).
        val_ratio: Fraction of subset allocated to local validation split (default: 0.20).
        seed: Random seed for deterministic local train/validation partitioning.
        augment: Whether to apply random geometric augmentations to training samples (default: False).
    """

    bands: Sequence[Union[int, str]] = DEFAULT_RGB_BANDS
    band_names: Optional[Sequence[str]] = None
    normalize: bool = True
    clip_max: float = DEFAULT_MAX_REFLECTANCE
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    seed: int = 42
    augment: bool = False

    def __post_init__(self) -> None:
        parsed_bands: List[int] = []
        for b in self.bands:
            if isinstance(b, str):
                b_clean = b.strip().upper()
                if b_clean not in S2_BAND_NAME_TO_INDEX:
                    raise ValueError(
                        f"Unrecognized Sentinel-2 band identifier: '{b}'. "
                        f"Supported names: {sorted(list(S2_BAND_NAME_TO_INDEX.keys()))}"
                    )
                parsed_bands.append(S2_BAND_NAME_TO_INDEX[b_clean])
            elif isinstance(b, (int, np.integer)):
                int_b = int(b)
                if not (1 <= int_b <= 13):
                    raise ValueError(
                        f"Invalid Sentinel-2 band index {int_b}. Must be between 1 and 13."
                    )
                parsed_bands.append(int_b)
            else:
                raise TypeError(f"Band items must be int or str, got {type(b).__name__}")

        if self.band_names is not None:
            parsed_names = tuple(str(n).strip() for n in self.band_names)
            if len(parsed_bands) != len(parsed_names):
                raise ValueError(
                    f"Mismatch: len(bands)={len(parsed_bands)} != len(band_names)={len(parsed_names)}"
                )
        else:
            parsed_names = tuple(
                S2_INDEX_TO_STANDARD_NAME.get(b, f"B{b:02d}") for b in parsed_bands
            )

        if not np.isclose(self.train_ratio + self.val_ratio, 1.0, atol=1e-3):
            raise ValueError(
                f"Split ratios must sum to 1.0, got train={self.train_ratio}, val={self.val_ratio}"
            )

        object.__setattr__(self, "bands", tuple(parsed_bands))
        object.__setattr__(self, "band_names", parsed_names)

    @property
    def composite_mode(self) -> str:
        """Return semantic composite mode description ('fcc', 'rgb', or 'custom')."""
        if self.bands == DEFAULT_FCC_BANDS:
            return "fcc"
        elif self.bands == DEFAULT_RGB_BANDS:
            return "rgb"
        return "custom"

    @classmethod
    def create_fcc(cls, **kwargs: Any) -> SEN12MSCRConfig:
        """Create False-Color Composite (FCC) configuration: B08 (NIR), B04 (Red), B03 (Green).

        Alings spectrally with ISRO Resourcesat-2A LISS-IV Bands [B4, B3, B2].
        """
        kwargs.setdefault("bands", DEFAULT_FCC_BANDS)
        kwargs.setdefault("band_names", DEFAULT_FCC_NAMES)
        return cls(**kwargs)

    @classmethod
    def create_rgb(cls, **kwargs: Any) -> SEN12MSCRConfig:
        """Create Natural-Color (RGB) configuration: B04 (Red), B03 (Green), B02 (Blue)."""
        kwargs.setdefault("bands", DEFAULT_RGB_BANDS)
        kwargs.setdefault("band_names", DEFAULT_RGB_NAMES)
        return cls(**kwargs)


@dataclass
class PairedOpticalSample:
    """Metadata and paths for a single cloudy/clear optical pair."""

    patch_id: str
    cloudy_path: Path
    clear_path: Path
    split: str


class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for paired SEN12MS-CR cloudy and cloud-free optical imagery."""

    def __init__(
        self,
        samples: Sequence[PairedOpticalSample],
        config: Optional[SEN12MSCRConfig] = None,
        augment: Optional[bool] = None,
    ) -> None:
        """Initialize dataset from a sequence of PairedOpticalSample records.

        Args:
            samples: List of validated PairedOpticalSample records.
            config: SEN12MSCRConfig specifying band selection, normalization, and defaults.
            augment: Explicit override for geometric augmentation. If None, inherits
                from config.augment. Augmentation is strictly applied only to samples
                belonging to the 'train' split.
        """
        self.samples = list(samples)
        self.config = config if config is not None else SEN12MSCRConfig()
        self.augment = bool(augment) if augment is not None else self.config.augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Read selected bands from GeoTIFFs
        with rasterio.open(sample.cloudy_path) as src_cloudy:
            cloudy_raw = src_cloudy.read(list(self.config.bands)).astype(np.float32)

        with rasterio.open(sample.clear_path) as src_clear:
            clear_raw = src_clear.read(list(self.config.bands)).astype(np.float32)

        if self.config.normalize:
            # Documented SEN12MS-CR normalization: clip to [0, 10000] and scale to [0, 1]
            cloudy_norm = np.clip(cloudy_raw, 0.0, self.config.clip_max) / self.config.clip_max
            clear_norm = np.clip(clear_raw, 0.0, self.config.clip_max) / self.config.clip_max
        else:
            cloudy_norm = cloudy_raw
            clear_norm = clear_raw

        cloudy_t = torch.from_numpy(cloudy_norm)
        clear_t = torch.from_numpy(clear_norm)

        # Apply geometric data augmentation strictly for training split
        if self.augment and sample.split == "train":
            hflip = bool(torch.rand(1).item() > 0.5)
            vflip = bool(torch.rand(1).item() > 0.5)
            rot90_k = int(torch.randint(0, 4, (1,)).item())
            aug_out = apply_geometric_augmentations(
                cloudy=cloudy_t,
                clear=clear_t,
                hflip=hflip,
                vflip=vflip,
                rot90_k=rot90_k,
            )
            cloudy_t = cast(torch.Tensor, aug_out[0])
            clear_t = cast(torch.Tensor, aug_out[1])

        return {
            "cloudy": cloudy_t,
            "clear": clear_t,
            "patch_id": sample.patch_id,
            "split": sample.split,
            "cloudy_path": str(sample.cloudy_path),
            "clear_path": str(sample.clear_path),
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        split: str = "train",
        config: Optional[SEN12MSCRConfig] = None,
        augment: Optional[bool] = None,
    ) -> SEN12MSCRDataset:
        """Instantiate dataset from an acquisition manifest.

        Args:
            manifest_path: Path to dataset manifest.json.
            split: Which partition to load ('train', 'val', or 'all').
            config: Optional dataset configuration.
            augment: Explicit override for geometric augmentation. If None, inherits
                from config.augment.

        Returns:
            SEN12MSCRDataset instance.
        """
        cfg = config if config is not None else SEN12MSCRConfig()
        m_path = Path(manifest_path)
        if not m_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        with open(m_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        base_dir = Path(manifest.get("base_directory", m_path.parent))
        cloudy_dir = base_dir / manifest["cloudy_directory"]
        clear_dir = base_dir / manifest["clear_directory"]
        if not cloudy_dir.is_dir():
            alt_dir = Path("data/raw/sen12mscr")
            if (alt_dir / manifest.get("cloudy_directory", "")).is_dir():
                base_dir = alt_dir
                cloudy_dir = base_dir / manifest["cloudy_directory"]
                clear_dir = base_dir / manifest["clear_directory"]

        paired_records: List[Tuple[str, Path, Path]] = []
        for item in manifest.get("paired_samples", []):
            p_id = item["patch_id"]
            c_file = cloudy_dir / item["cloudy_file"]
            clr_file = clear_dir / item["clear_file"]

            if c_file.is_file() and clr_file.is_file():
                paired_records.append((p_id, c_file, clr_file))

        if not paired_records:
            raise ValueError(f"No valid paired samples found in {m_path}")

        # Deterministic local split partitioning
        # Sort by patch_id for stable ordering, then shuffle with fixed seed
        paired_records.sort(key=lambda x: x[0])
        rng = np.random.default_rng(cfg.seed)
        indices = np.arange(len(paired_records))
        rng.shuffle(indices)

        n_train = int(len(paired_records) * cfg.train_ratio)
        train_indices = set(indices[:n_train])

        samples: List[PairedOpticalSample] = []
        for idx, (p_id, c_file, clr_file) in enumerate(paired_records):
            assigned_split = "train" if idx in train_indices else "val"
            if split == "all" or split == assigned_split:
                samples.append(
                    PairedOpticalSample(
                        patch_id=p_id,
                        cloudy_path=c_file,
                        clear_path=clr_file,
                        split=assigned_split,
                    )
                )

        eff_augment = augment if augment is not None else cfg.augment
        return cls(samples=samples, config=cfg, augment=eff_augment)


def validate_sen12mscr_subset(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate data integrity, alignment, and value distributions of the acquired subset."""
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    base_dir = m_path.parent
    cloudy_dir = base_dir / manifest["cloudy_directory"]
    clear_dir = base_dir / manifest["clear_directory"]

    paired_samples = manifest.get("paired_samples", [])
    total_samples = len(paired_samples)

    print(f"Validating {total_samples} paired samples from {m_path}...")
    valid_pairs = 0
    corrupt_samples = 0
    ref_shape = (13, 256, 256)
    ref_dtype = "uint16"

    # Sample a subset for in-depth raster validation
    sample_stride = max(1, total_samples // 50)  # validate ~50 rasters deeply
    min_vals, max_vals = [], []

    for idx, item in enumerate(paired_samples):
        c_path = cloudy_dir / item["cloudy_file"]
        clr_path = clear_dir / item["clear_file"]

        if not c_path.is_file() or not clr_path.is_file():
            corrupt_samples += 1
            continue

        if idx % sample_stride == 0:
            try:
                with rasterio.open(c_path) as sc, rasterio.open(clr_path) as sclr:
                    if (sc.count, sc.height, sc.width) != ref_shape:
                        corrupt_samples += 1
                        continue
                    if (sclr.count, sclr.height, sclr.width) != ref_shape:
                        corrupt_samples += 1
                        continue
                    if sc.dtypes[0] != ref_dtype or sclr.dtypes[0] != ref_dtype:
                        corrupt_samples += 1
                        continue
                    if sc.crs != sclr.crs or not np.allclose(sc.transform, sclr.transform):
                        corrupt_samples += 1
                        continue

                    # Verify reading RGB channels
                    c_rgb = sc.read([4, 3, 2])
                    sclr.read([4, 3, 2])
                    min_vals.append(float(np.min(c_rgb)))
                    max_vals.append(float(np.max(c_rgb)))
            except Exception as e:
                print(f"Error inspecting pair {item['patch_id']}: {e}")
                corrupt_samples += 1
                continue

        valid_pairs += 1

    summary = {
        "status": "valid" if corrupt_samples == 0 else "corrupted",
        "total_pairs": total_samples,
        "valid_pairs": valid_pairs,
        "corrupt_pairs": corrupt_samples,
        "raster_shape": list(ref_shape),
        "data_type": ref_dtype,
        "rgb_min_observed": float(min(min_vals)) if min_vals else 0.0,
        "rgb_max_observed": float(max(max_vals)) if max_vals else 0.0,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate SEN12MS-CR subset")
    parser.add_argument("--manifest", type=str, default="data/raw/sen12mscr/manifest.json")
    args = parser.parse_args()

    result = validate_sen12mscr_subset(args.manifest)
    print("\nValidation Summary:")
    print(json.dumps(result, indent=2))
    return 0 if result["corrupt_pairs"] == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
