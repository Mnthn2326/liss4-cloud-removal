"""Tests for raster inspection utility."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from cloud_removal.inspect import inspect_raster


def test_inspect_raster_valid(tmp_path: Path):
    """Test inspecting a valid 3-band GeoTIFF with simulated LISS-IV characteristics."""
    test_tiff = tmp_path / "sample_liss4_simulated.tif"

    # Simulate 3 bands: Green, Red, NIR (5.8m resolution, 64x64 pixels, 16-bit uint)
    width, height = 64, 64
    crs = "EPSG:32643"  # UTM Zone 43N (covering western/central India)
    transform = from_origin(300000.0, 2100000.0, 5.8, 5.8)

    # 10-bit dynamic range stored in uint16 (0 - 1023)
    np.random.seed(42)
    band_green = np.random.randint(100, 400, size=(height, width), dtype=np.uint16)
    band_red = np.random.randint(120, 500, size=(height, width), dtype=np.uint16)
    band_nir = np.random.randint(200, 900, size=(height, width), dtype=np.uint16)

    # Inject nodata in top-left pixel
    band_green[0, 0] = 0
    band_red[0, 0] = 0
    band_nir[0, 0] = 0

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
        dst.write(band_green, 1)
        dst.write(band_red, 2)
        dst.write(band_nir, 3)
        dst.set_band_description(1, "Band 2 - Green (0.52-0.59 um)")
        dst.set_band_description(2, "Band 3 - Red (0.62-0.68 um)")
        dst.set_band_description(3, "Band 4 - NIR (0.76-0.86 um)")

    result = inspect_raster(test_tiff)

    assert result["file_name"] == "sample_liss4_simulated.tif"
    assert result["width"] == 64
    assert result["height"] == 64
    assert result["count"] == 3
    assert "32643" in result["crs"]
    assert len(result["band_statistics"]) == 3

    green_stats = result["band_statistics"][0]
    assert green_stats["band"] == 1
    assert green_stats["dtype"] == "uint16"
    assert green_stats["nodata"] == 0
    assert green_stats["valid_pixels"] == (64 * 64) - 1
    assert 100 <= green_stats["min"] <= 400


def test_inspect_raster_file_not_found():
    """Verify that FileNotFoundError is raised for non-existent raster file."""
    with pytest.raises(FileNotFoundError):
        inspect_raster("non_existent_satellite_raster.tif")
