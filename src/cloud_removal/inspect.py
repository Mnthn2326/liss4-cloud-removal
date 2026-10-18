"""Geospatial raster inspection utilities using Rasterio for LISS-IV imagery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.windows import Window


def inspect_raster(
    raster_path: Union[str, Path],
    compute_stats: bool = True,
    chunk_rows: int = 1024,
    ignore_zero: bool = False,
) -> Dict[str, Any]:
    """Inspect a geospatial raster file and return metadata and band statistics.

    Reads data in low-memory row chunks to avoid loading large satellite scenes
    entirely into RAM.

    Args:
        raster_path: Path to the raster file (e.g., GeoTIFF).
        compute_stats: Whether to compute min, max, mean, and std per band.
        chunk_rows: Number of rows to read per streaming chunk (default 1024).
        ignore_zero: If True, treats pixel value 0 as NoData/collar padding.

    Returns:
        Dictionary containing raster dimensions, CRS, transform, band datatypes,
        nodata values, and pixel statistics.

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

        if compute_stats:
            for band_idx in range(1, src.count + 1):
                nodata_val = src.nodatavals[band_idx - 1]

                total_pixels = 0
                valid_pixels = 0
                b_min = float("inf")
                b_max = float("-inf")
                sum_val = 0.0
                sum_sq_val = 0.0

                for row_start in range(0, src.height, chunk_rows):
                    row_h = min(chunk_rows, src.height - row_start)
                    win = Window(0, row_start, src.width, row_h)
                    chunk = src.read(band_idx, window=win)

                    total_pixels += chunk.size
                    valid_mask = np.isfinite(chunk)
                    if nodata_val is not None:
                        valid_mask &= chunk != nodata_val
                    if ignore_zero:
                        valid_mask &= chunk != 0

                    valid_chunk = chunk[valid_mask]
                    if valid_chunk.size > 0:
                        valid_pixels += int(valid_chunk.size)
                        c_min = float(np.min(valid_chunk))
                        c_max = float(np.max(valid_chunk))
                        if c_min < b_min:
                            b_min = c_min
                        if c_max > b_max:
                            b_max = c_max
                        sum_val += float(np.sum(valid_chunk, dtype=np.float64))
                        sum_sq_val += float(np.sum(valid_chunk.astype(np.float64) ** 2))

                if valid_pixels > 0:
                    mean_val = sum_val / valid_pixels
                    variance = max(0.0, (sum_sq_val / valid_pixels) - (mean_val**2))
                    std_val = float(np.sqrt(variance))
                    stats = {
                        "band": band_idx,
                        "dtype": str(src.dtypes[band_idx - 1]),
                        "nodata": nodata_val,
                        "min": b_min,
                        "max": b_max,
                        "mean": mean_val,
                        "std": std_val,
                        "valid_pixels": valid_pixels,
                        "total_pixels": total_pixels,
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
                        "total_pixels": total_pixels,
                    }
                band_stats.append(stats)

        summary: Dict[str, Any] = {
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
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
            "is_tiled": bool(src.profile.get("tiled", False)),

            "block_shapes": src.block_shapes,
            "band_statistics": band_stats,
            "tags": dict(src.tags()),
        }

    return summary


def read_raster_window(
    raster_path: Union[str, Path],
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    band_idx: int = 1,
) -> np.ndarray:
    """Read a small rectangular pixel window from a raster without loading full image.

    Args:
        raster_path: Path to the raster file.
        col_off: Column offset (X start).
        row_off: Row offset (Y start).
        width: Number of columns to read.
        height: Number of rows to read.
        band_idx: 1-indexed band number to read (default 1).

    Returns:
        2D numpy array of shape (height, width).
    """
    path = Path(raster_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster file not found: {path}")

    win = Window(col_off, row_off, width, height)
    with rasterio.open(path) as src:
        data = src.read(band_idx, window=win)
    return data
