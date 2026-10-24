"""Unit tests for LISS-IV authentic scene ingestion and validation."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

from cloud_removal.ingest_liss4 import (
    IngestionError,
    ingest_liss4_product,
    validate_liss4_scene,
)


@pytest.fixture
def mock_liss4_scene(tmp_path: Path) -> Path:
    """Create a mock Resourcesat-2A LISS-IV scene directory with authentic metadata structure."""
    scene_dir = tmp_path / "RAF03JUN2026049233009500060SSANSTUC00GTDA"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy GeoTIFFs for BAND2, BAND3, BAND4
    H, W = 16, 16
    for b_name in ("BAND2.tif", "BAND3.tif", "BAND4.tif"):
        arr = np.full((H, W), 100, dtype=np.uint16)
        with rasterio.open(
            scene_dir / b_name,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
        ) as dst:
            dst.write(arr, 1)

    # Write BAND_META.txt
    band_meta_text = (
        "SatID= IRS-R2A\n"
        "Sensor= L4FX\n"
        "SubScene= A\n"
        "DateOfPass= 03-JUN-2026\n"
        "BitsPerPixel= 10\n"
        "SunElevationAtCenter= 70.026528\n"
        "SunAzimuthAtCenter= 74.758853\n"
        "B2_Lmin= 0.0000\n"
        "B2_Lmax= 52.0000\n"
        "B3_Lmin= 0.0000\n"
        "B3_Lmax= 47.0000\n"
        "B4_Lmin= 0.0000\n"
        "B4_Lmax= 31.5000\n"
        "CloudPercent= 1.5\n"
    )
    with open(scene_dir / "BAND_META.txt", "w", encoding="utf-8") as f:
        f.write(band_meta_text)

    return scene_dir


def test_validate_liss4_scene_success(mock_liss4_scene: Path):
    """validate_liss4_scene must succeed on an authentic product with <= 2% cloud cover."""
    res = validate_liss4_scene(mock_liss4_scene, max_cloud_percent=2.0)
    assert res["status"] == "valid"
    assert res["satellite"] == "IRS-R2A"
    assert res["sensor"] == "L4FX"
    assert res["cloud_percent"] == 1.5
    assert len(res["rasters"]) == 3


def test_validate_liss4_scene_cloud_threshold_exceeded(mock_liss4_scene: Path):
    """validate_liss4_scene must reject scenes exceeding the cloud cover limit."""
    # Set threshold to 1.0% when scene has 1.5%
    with pytest.raises(IngestionError, match="exceeds maximum allowable threshold"):
        validate_liss4_scene(mock_liss4_scene, max_cloud_percent=1.0)


def test_validate_liss4_scene_invalid_sensor(mock_liss4_scene: Path):
    """validate_liss4_scene must reject products from invalid sensors (e.g. AWiFS)."""
    meta_file = mock_liss4_scene / "BAND_META.txt"
    with open(meta_file, "r", encoding="utf-8") as f:
        text = f.read()

    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(text.replace("Sensor= L4FX", "Sensor= AWIFS"))

    with pytest.raises(IngestionError, match="Invalid sensor identity"):
        validate_liss4_scene(mock_liss4_scene)


def test_validate_liss4_scene_missing_raster(mock_liss4_scene: Path):
    """validate_liss4_scene must fail if BAND4.tif is missing."""
    (mock_liss4_scene / "BAND4.tif").unlink()
    with pytest.raises(IngestionError, match="Required band raster missing"):
        validate_liss4_scene(mock_liss4_scene)


def test_ingest_liss4_product_from_zip(tmp_path: Path, mock_liss4_scene: Path):
    """ingest_liss4_product must extract zip, validate metadata, and generate manifest."""
    # Create ZIP archive from mock scene
    zip_path = tmp_path / "RAF03JUN2026_test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in mock_liss4_scene.glob("*"):
            zf.write(f, arcname=f.name)

    dest_dir = tmp_path / "raw_destination"
    res = ingest_liss4_product(
        source_path=zip_path,
        destination_raw_dir=dest_dir,
        max_cloud_percent=2.0,
    )

    assert res["status"] == "ingested_successfully"
    assert Path(res["manifest_path"]).is_file()
    assert "dvc add" in res["dvc_command"]

    with open(res["manifest_path"], "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["satellite"] == "IRS-R2A"
    assert len(manifest["files"]) >= 4
