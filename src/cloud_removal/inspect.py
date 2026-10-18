"""Geospatial raster inspection utilities using Rasterio for LISS-IV imagery."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import rasterio


def inspect_raster(raster_path: str | Path) -> Dict[str, Any]:
    """Inspect a geospatial raster file and return metadata and band statistics.

    Args:
        raster_path: Path to the raster file (e.g., GeoTIFF).

    Returns:
        Dictionary containing raster dimensions, CRS, transform, band datatypes,
        nodata values, and pixel statistics (min, max, mean, std).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be read as a raster.
    """
    path = Path(raster_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        bounds = src.bounds
        crs_str = src.crs.to_string() if src.crs else "Unprojected / Unknown"

        band_stats: List[Dict[str, Any]] = []
        for band_idx in range(1, src.count + 1):
            data = src.read(band_idx)
            nodata_val = src.nodatavals[band_idx - 1]

            valid_mask = np.isfinite(data)
            if nodata_val is not None:
                valid_mask = valid_mask & (data != nodata_val)

            valid_data = data[valid_mask]

            if valid_data.size > 0:
                stats = {
                    "band": band_idx,
                    "dtype": str(src.dtypes[band_idx - 1]),
                    "nodata": nodata_val,
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "valid_pixels": int(valid_data.size),
                    "total_pixels": int(data.size),
                }
            else:
                stats = {
                    "band": band_idx,
                    "dtype": str(src.dtypes[band_idx - 1]),
                    "nodata": nodata_val,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "valid_pixels": 0,
                    "total_pixels": int(data.size),
                }
            band_stats.append(stats)

        summary: Dict[str, Any] = {
            "file_name": path.name,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": crs_str,
            "transform": [float(x) for x in src.transform],
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "driver": src.driver,
            "band_statistics": band_stats,
            "tags": dict(src.tags()),
        }

    return summary
