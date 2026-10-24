"""Tests for raster inspection utility."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from cloud_removal.inspect import inspect_raster, read_raster_window


@pytest.fixture
def synthetic_raster_fixture(tmp_path: Path) -> Path:
    """Fixture creating a synthetic 3-band GeoTIFF simulating LISS-IV imagery."""
    test_tiff = tmp_path / "sample_liss4_simulated.tif"

    width, height = 64, 64
    crs = "EPSG:32643"  # UTM Zone 43N
    transform = from_origin(300000.0, 2100000.0, 5.0, 5.0)

    np.random.seed(42)
    band_b2 = np.random.randint(100, 400, size=(height, width), dtype=np.uint16)
    band_b3 = np.random.randint(120, 500, size=(height, width), dtype=np.uint16)
    band_b4 = np.random.randint(200, 900, size=(height, width), dtype=np.uint16)

    # Set collar zeros (simulating border padding)
    band_b2[0, :] = 0
    band_b3[0, :] = 0
    band_b4[0, :] = 0

    with rasterio.open(
        test_tiff,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(band_b2, 1)
        dst.write(band_b3, 2)
        dst.write(band_b4, 3)
        dst.set_band_description(1, "BAND2")
        dst.set_band_description(2, "BAND3")
        dst.set_band_description(3, "BAND4")

    return test_tiff


def test_inspect_raster_valid(synthetic_raster_fixture: Path):
    """Test inspecting a valid 3-band GeoTIFF with simulated LISS-IV characteristics."""
    result = inspect_raster(synthetic_raster_fixture, chunk_rows=16)

    assert result["file_name"] == "sample_liss4_simulated.tif"
    assert result["width"] == 64
    assert result["height"] == 64
    assert result["count"] == 3
    assert "32643" in result["crs"]
    assert len(result["band_statistics"]) == 3

    band2_stats = result["band_statistics"][0]
    assert band2_stats["band"] == 1
    assert band2_stats["dtype"] == "uint16"
    assert band2_stats["nodata"] == 0
    # First row is 0 (nodata), so 64*63 valid pixels
    assert band2_stats["valid_pixels"] == 64 * 63
    assert 100 <= band2_stats["min"] <= 400


def test_inspect_raster_ignore_zero(synthetic_raster_fixture: Path):
    """Test inspecting raster while treating zero as collar padding."""
    result = inspect_raster(synthetic_raster_fixture, chunk_rows=32, ignore_zero=True)
    band2_stats = result["band_statistics"][0]
    assert band2_stats["valid_pixels"] == 64 * 63
    assert band2_stats["min"] >= 100


def test_read_raster_window_valid(synthetic_raster_fixture: Path):
    """Test reading a small window from a raster without loading entire image."""
    window_data = read_raster_window(
        synthetic_raster_fixture, col_off=10, row_off=10, width=16, height=16, band_idx=1
    )
    assert window_data.shape == (16, 16)
    assert window_data.dtype == np.uint16
    assert np.all(window_data >= 100)


def test_inspect_raster_file_not_found():
    """Verify that FileNotFoundError is raised for non-existent raster file."""
    with pytest.raises(FileNotFoundError):
        inspect_raster("non_existent_satellite_raster.tif")


def test_read_raster_window_file_not_found():
    """Verify that FileNotFoundError is raised for non-existent raster window read."""
    with pytest.raises(FileNotFoundError):
        read_raster_window("non_existent_satellite_raster.tif", 0, 0, 10, 10)
