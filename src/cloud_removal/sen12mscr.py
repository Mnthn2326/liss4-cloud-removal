"""SEN12MS-CR dataset loader and preprocessing adapter.

Provides dataset abstractions, deterministic split partitioning, reflectance
normalization, and PyTorch Dataset integration for paired Sentinel-2 optical data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# Standard Sentinel-2 Band Definitions (1-indexed for rasterio)
# 1: Aerosols, 2: Blue, 3: Green, 4: Red, 5: RE1, 6: RE2, 7: RE3, 8: NIR,
# 9: Narrow NIR, 10: Water Vapor, 11: Cirrus, 12: SWIR1, 13: SWIR2
DEFAULT_RGB_BANDS: Tuple[int, int, int] = (4, 3, 2)  # Red (B4), Green (B3), Blue (B2)
DEFAULT_RGB_NAMES: Tuple[str, str, str] = ("B04", "B03", "B02")
DEFAULT_MAX_REFLECTANCE: float = 10000.0


@dataclass(frozen=True)
class SEN12MSCRConfig:
    """Configuration for SEN12MS-CR paired optical dataset.

    Attributes:
        bands: Tuple of 1-indexed band numbers to extract (default: (4, 3, 2) for RGB).
        band_names: Corresponding names of selected bands.
        normalize: Whether to apply documented reflectance scaling (DN / 10000).
        clip_max: Maximum reflectance digital number before scaling (default: 10000.0).
        train_ratio: Fraction of subset allocated to local training split (default: 0.80).
        val_ratio: Fraction of subset allocated to local validation split (default: 0.20).
        seed: Random seed for deterministic local train/validation partitioning.
    """

    bands: Tuple[int, ...] = DEFAULT_RGB_BANDS
    band_names: Tuple[str, ...] = DEFAULT_RGB_NAMES
    normalize: bool = True
    clip_max: float = DEFAULT_MAX_REFLECTANCE
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.bands) != len(self.band_names):
            raise ValueError(
                f"Mismatch: len(bands)={len(self.bands)} != len(band_names)={len(self.band_names)}"
            )
        for b in self.bands:
            if not (1 <= b <= 13):
                raise ValueError(f"Invalid Sentinel-2 band index {b}. Must be between 1 and 13.")
        if not np.isclose(self.train_ratio + self.val_ratio, 1.0, atol=1e-3):
            raise ValueError(
                f"Split ratios must sum to 1.0, got train={self.train_ratio}, val={self.val_ratio}"
            )


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
    ) -> None:
        """Initialize dataset from a sequence of PairedOpticalSample records.

        Args:
            samples: List of validated PairedOpticalSample records.
            config: SEN12MSCRConfig specifying band selection and normalization.
        """
        self.samples = list(samples)
        self.config = config if config is not None else SEN12MSCRConfig()

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

        return {
            "cloudy": torch.from_numpy(cloudy_norm),
            "clear": torch.from_numpy(clear_norm),
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
    ) -> SEN12MSCRDataset:
        """Instantiate dataset from an acquisition manifest.

        Args:
            manifest_path: Path to dataset manifest.json.
            split: Which partition to load ('train', 'val', or 'all').
            config: Optional dataset configuration.

        Returns:
            SEN12MSCRDataset instance.
        """
        cfg = config if config is not None else SEN12MSCRConfig()
        m_path = Path(manifest_path)
        if not m_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        with open(m_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        base_dir = m_path.parent
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

        return cls(samples=samples, config=cfg)


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
                    clr_rgb = sclr.read([4, 3, 2])
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
