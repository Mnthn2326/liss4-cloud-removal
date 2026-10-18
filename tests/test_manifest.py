"""Tests for dataset manifest validation."""

import json
from pathlib import Path


def test_manifest_schema_and_integrity():
    """Verify that data/raw/manifest.json exists and conforms to required schema."""
    manifest_path = Path("data/raw/manifest.json")
    assert manifest_path.is_file(), "data/raw/manifest.json not found"

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    required_top_level_keys = [
        "product_id",
        "ots_product_id",
        "satellite",
        "sensor",
        "acquisition_date",
        "source_directory",
        "dvc_file",
        "total_size_bytes",
        "raster_metadata",
        "files",
    ]
    for key in required_top_level_keys:
        assert key in manifest, f"Missing required manifest key: {key}"

    assert manifest["product_id"] == "2643035131"
    assert manifest["satellite"] == "IRS-R2A"
    assert manifest["sensor"] == "L4FX"
    assert manifest["total_size_bytes"] == 1759106380

    raster_meta = manifest["raster_metadata"]
    assert raster_meta["dimensions"]["width"] == 18066
    assert raster_meta["dimensions"]["height"] == 16223
    assert raster_meta["crs"] == "EPSG:32643"
    assert raster_meta["pixel_resolution_meters"]["x"] == 5.0
    assert raster_meta["pixel_resolution_meters"]["y"] == 5.0

    files = manifest["files"]
    assert len(files) == 8, f"Expected 8 files in manifest, got {len(files)}"

    for file_entry in files:
        assert "filename" in file_entry
        assert "size_bytes" in file_entry
        assert "sha256" in file_entry
        assert len(file_entry["sha256"]) == 64  # valid SHA-256 hex string

    # Ensure associated DVC file exists
    dvc_path = Path(manifest["dvc_file"])
    assert dvc_path.is_file(), f"Associated DVC file not found: {dvc_path}"
