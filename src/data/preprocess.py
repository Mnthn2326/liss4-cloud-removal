"""Preprocess raw satellite imagery into tiled, normalised patch pairs.

Usage:
    python -m src.data.preprocess --config configs/dataset_rice.yaml

Steps (all parameterised by the YAML config):
  1. Read raw images from paths.raw
  2. Tile each image into patch_size x patch_size non-overlapping patches
  3. Normalise pixel values per the normalization config
  4. Save patches as .npy files under paths.processed
  5. Build a pairs manifest CSV
  6. Split at the *image* level into train/val/test to avoid data leakage
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a dataset YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image(img_array: np.ndarray, patch_size: int) -> list[np.ndarray]:
    """Tile a HxWxC image into non-overlapping patch_size x patch_size patches.

    Patches that don't fit (remainder pixels) are discarded.
    Returns a list of (patch, row_idx, col_idx) tuples.
    """
    h, w = img_array.shape[:2]
    patches = []
    for r_idx, r in enumerate(range(0, h - patch_size + 1, patch_size)):
        for c_idx, c in enumerate(range(0, w - patch_size + 1, patch_size)):
            patch = img_array[r : r + patch_size, c : c + patch_size]
            patches.append((patch, r_idx, c_idx))
    return patches


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_patch(patch: np.ndarray, cfg_norm: dict) -> np.ndarray:
    """Normalise a uint8 [0,255] patch according to the config.

    Supported methods:
      - "minmax": maps [0,255] → [min_val, max_val]  (default [-1,1])
      - "zscore": zero-mean, unit-variance (per-band or global)
    """
    patch = patch.astype(np.float32)
    method = cfg_norm.get("method", "minmax")

    if method == "minmax":
        # Default to [-1, 1] to match tanh output range
        min_val = cfg_norm.get("min_val", -1.0)
        max_val = cfg_norm.get("max_val", 1.0)
        # For the symmetric case used in the config: (pixel / 127.5) - 1
        patch = patch / 255.0  # [0, 1]
        patch = patch * (max_val - min_val) + min_val  # [min_val, max_val]
    elif method == "zscore":
        per_band = cfg_norm.get("per_band_stats", False)
        if per_band:
            for b in range(patch.shape[-1]):
                band = patch[..., b]
                patch[..., b] = (band - band.mean()) / (band.std() + 1e-8)
        else:
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return patch


# ---------------------------------------------------------------------------
# RICE-specific raw data layout detection
# ---------------------------------------------------------------------------

def discover_rice_pairs(raw_dir: str) -> list[tuple[str, str, str]]:
    """Discover cloud/ground-truth image pairs from the RICE raw directory.

    Expected layout (RICE1):
        raw_dir/
        ├── cloud/   or  RICE1/cloud/
        │   └── *.png
        └── label/   or  RICE1/label/   (ground truth)
            └── *.png

    Returns list of (image_id, cloud_path, gt_path).
    """
    raw = Path(raw_dir)

    # Try both flat and nested layouts
    for base in [raw, raw / "RICE1", raw / "RICE_DATASET" / "RICE1"]:
        cloud_dir = base / "cloud"
        # Ground truth may be called "label" or "ground_truth" or "gt"
        gt_dir = None
        for gt_name in ["label", "ground_truth", "gt"]:
            candidate = base / gt_name
            if candidate.is_dir():
                gt_dir = candidate
                break

        if cloud_dir.is_dir() and gt_dir is not None:
            break
    else:
        raise FileNotFoundError(
            f"Could not find cloud/ and label/ directories under {raw_dir}. "
            f"Expected RICE1 layout with cloud/ and label/ subdirectories."
        )

    # Match pairs by filename
    cloud_files = sorted(cloud_dir.glob("*.*"))
    pairs = []
    for cf in cloud_files:
        # Try same name in gt dir
        gt_candidates = list(gt_dir.glob(cf.stem + ".*"))
        if not gt_candidates:
            print(f"  Warning: no ground truth found for {cf.name}, skipping")
            continue
        gt_path = gt_candidates[0]
        pairs.append((cf.stem, str(cf), str(gt_path)))

    if not pairs:
        raise FileNotFoundError(
            f"No image pairs found in {cloud_dir} / {gt_dir}"
        )

    print(f"  Found {len(pairs)} image pairs in {cloud_dir.parent}")
    return pairs


# ---------------------------------------------------------------------------
# Generic pair discovery (extensible for LISS-IV later)
# ---------------------------------------------------------------------------

def discover_pairs(raw_dir: str, dataset_name: str) -> list[tuple[str, str, str]]:
    """Route to dataset-specific pair discovery based on the config name."""
    name_lower = dataset_name.lower()
    if "rice" in name_lower:
        return discover_rice_pairs(raw_dir)
    else:
        # Future: add LISS-IV, SEN12MS-CR discovery here
        raise NotImplementedError(
            f"Pair discovery not implemented for dataset '{dataset_name}'. "
            f"Add a discover_{{name}}_pairs() function in preprocess.py."
        )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_image_ids(
    image_ids: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split image IDs into train/val/test at the image level.

    This ensures all patches from one image stay in the same split,
    preventing data leakage.
    """
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(config_path: str) -> None:
    """Run the full preprocessing pipeline."""
    cfg = load_config(config_path)

    dataset_name = cfg["name"]
    bands_count = cfg["bands"]["count"]
    patch_size = cfg["patch_size"]
    raw_dir = cfg["paths"]["raw"]
    processed_dir = cfg["paths"]["processed"]
    manifest_path = cfg["paths"]["pairs_manifest"]
    cfg_norm = cfg["normalization"]
    cfg_split = cfg["split"]

    print(f"=== Preprocessing dataset: {dataset_name} ===")
    print(f"  Raw dir:       {raw_dir}")
    print(f"  Processed dir: {processed_dir}")
    print(f"  Patch size:    {patch_size}")
    print(f"  Bands:         {bands_count}")

    # --- 1. Discover pairs ---
    pairs = discover_pairs(raw_dir, dataset_name)
    image_ids = [p[0] for p in pairs]
    pairs_dict = {p[0]: (p[1], p[2]) for p in pairs}

    # --- 2. Split at image level ---
    splits = split_image_ids(
        image_ids,
        train_frac=cfg_split["train"],
        val_frac=cfg_split["val"],
        test_frac=cfg_split["test"],
        seed=cfg_split["seed"],
    )
    print(f"  Split: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

    # --- 3. Create output directories ---
    cloud_out = Path(processed_dir) / "cloud"
    gt_out = Path(processed_dir) / "gt"
    splits_dir = Path(cfg.get("paths", {}).get("splits_dir", "data/splits"))

    for d in [cloud_out, gt_out, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 4. Tile, normalise, save ---
    all_records = []  # (image_id, patch_name, cloud_npy_path, gt_npy_path)

    for img_id, (cloud_path, gt_path) in pairs_dict.items():
        # Read images
        cloud_img = np.array(Image.open(cloud_path).convert("RGB"))
        gt_img = np.array(Image.open(gt_path).convert("RGB"))

        # Validate band count
        if cloud_img.shape[-1] != bands_count:
            print(f"  Warning: {img_id} has {cloud_img.shape[-1]} bands, "
                  f"expected {bands_count}. Skipping.")
            continue

        # Tile
        cloud_patches = tile_image(cloud_img, patch_size)
        gt_patches = tile_image(gt_img, patch_size)

        if len(cloud_patches) != len(gt_patches):
            print(f"  Warning: patch count mismatch for {img_id}, skipping")
            continue

        for (c_patch, r, c), (g_patch, _, _) in zip(cloud_patches, gt_patches):
            # Normalise
            c_norm = normalize_patch(c_patch, cfg_norm)
            g_norm = normalize_patch(g_patch, cfg_norm)

            # Save
            patch_name = f"{img_id}_r{r}_c{c}"
            c_path = cloud_out / f"{patch_name}.npy"
            g_path = gt_out / f"{patch_name}.npy"

            np.save(str(c_path), c_norm)
            np.save(str(g_path), g_norm)

            all_records.append((img_id, patch_name, str(c_path), str(g_path)))

    print(f"  Total patches saved: {len(all_records)}")

    # --- 5. Build pairs manifest ---
    manifest_dir = Path(manifest_path).parent
    manifest_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "patch_name", "cloud_path", "gt_path"])
        for rec in all_records:
            writer.writerow(rec)

    print(f"  Pairs manifest: {manifest_path}")

    # --- 6. Write split CSVs ---
    for split_name, split_ids in splits.items():
        split_records = [r for r in all_records if r[0] in set(split_ids)]
        split_csv = splits_dir / f"{dataset_name}_{split_name}.csv"
        with open(str(split_csv), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "patch_name", "cloud_path", "gt_path"])
            for rec in split_records:
                writer.writerow(rec)
        print(f"  {split_name}: {len(split_records)} patches → {split_csv}")

    print("=== Preprocessing complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess satellite imagery for cloud removal"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset YAML config (e.g. configs/dataset_rice.yaml)",
    )
    args = parser.parse_args()
    preprocess(args.config)
