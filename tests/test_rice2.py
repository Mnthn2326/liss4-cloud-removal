"""Tests for RICE2 manifest construction and paired dataset loading."""

from pathlib import Path

import numpy as np
from PIL import Image

from cloud_removal.rice2 import RICE2Config, RICE2Dataset, build_rice2_manifest


def _write_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8), mode="RGB").save(path)


def test_rice2_manifest_and_loader(tmp_path: Path) -> None:
    root = tmp_path / "RICE2"
    for name in ("cloud", "label", "mask"):
        (root / name).mkdir(parents=True)
    for identifier in range(10):
        _write_rgb(root / "cloud" / f"{identifier}.png", 200)
        _write_rgb(root / "label" / f"{identifier}.png", 80)
        _write_rgb(root / "mask" / f"{identifier}.png", 255 if identifier == 0 else 0)

    manifest_path = tmp_path / "rice2_manifest.json"
    manifest = build_rice2_manifest(root, manifest_path, RICE2Config(augment=False), include_hashes=False)
    assert manifest["counts"] == {"total": 10, "train": 6, "val": 1, "test": 3}

    all_dataset = RICE2Dataset.from_manifest(manifest_path, split="all", config=RICE2Config(augment=False))
    assert len(all_dataset) == 10
    sample = next(all_dataset[index] for index in range(len(all_dataset)) if all_dataset[index]["sample_id"] == "0")
    assert sample["cloudy"].shape == (3, 16, 16)
    assert sample["clear"].shape == (3, 16, 16)
    assert sample["cloud_mask"].shape == (1, 16, 16)
    assert float(sample["cloud_mask"].mean()) == 1.0


def test_rice2_manifest_rejects_unpaired_files(tmp_path: Path) -> None:
    root = tmp_path / "RICE2"
    for name in ("cloud", "label", "mask"):
        (root / name).mkdir(parents=True)
    _write_rgb(root / "cloud" / "0.png", 100)
    _write_rgb(root / "label" / "0.png", 100)
    _write_rgb(root / "mask" / "1.png", 0)

    try:
        build_rice2_manifest(root, tmp_path / "manifest.json", include_hashes=False)
    except ValueError as error:
        assert "identical sample IDs" in str(error)
    else:
        raise AssertionError("Expected unpaired RICE2 files to be rejected")


def test_rice2_rejects_inconsistent_sizes(tmp_path: Path) -> None:
    root = tmp_path / "RICE2"
    for name in ("cloud", "label", "mask"):
        (root / name).mkdir(parents=True)
    _write_rgb(root / "cloud" / "0.png", 100)
    # Write a mismatched spatial size
    Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8), mode="RGB").save(root / "label" / "0.png")
    _write_rgb(root / "mask" / "0.png", 0)

    try:
        build_rice2_manifest(root, tmp_path / "manifest.json", include_hashes=False)
    except ValueError as error:
        assert "Inconsistent spatial dimensions" in str(error)
    else:
        raise AssertionError("Expected inconsistent image dimensions to be rejected")


def test_rice2_split_disjointness_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "RICE2"
    for name in ("cloud", "label", "mask"):
        (root / name).mkdir(parents=True)
    for identifier in range(20):
        _write_rgb(root / "cloud" / f"{identifier}.png", 200)
        _write_rgb(root / "label" / f"{identifier}.png", 80)
        _write_rgb(root / "mask" / f"{identifier}.png", 0)

    manifest_path = tmp_path / "rice2_manifest.json"
    manifest = build_rice2_manifest(root, manifest_path, RICE2Config(augment=False), include_hashes=True)

    samples = manifest["samples"]
    train_ids = {s["sample_id"] for s in samples if s["split"] == "train"}
    val_ids = {s["sample_id"] for s in samples if s["split"] == "val"}
    test_ids = {s["sample_id"] for s in samples if s["split"] == "test"}

    # Assert strictly pairwise disjoint
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 20

    # Assert hashes are valid SHA-256 strings
    for s in samples:
        assert "sha256" in s
        assert len(s["sha256"]["cloud"]) == 64
        assert len(s["sha256"]["label"]) == 64
        assert len(s["sha256"]["mask"]) == 64
