"""Lightweight, responsive Web UI and interactive endpoints for LISS-IV / Sentinel-2 cloud removal.

Served directly by the FastAPI application. Provides file upload, SEN12MS-CR benchmark
and synthetic cloud presets, side-by-side and interactive before/after split-slider views,
latency tracking, conditional MAE/PSNR evaluation, and GeoTIFF / PNG downloads.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
import torch

from cloud_removal.metrics import compute_mae, compute_psnr
from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import SEN12MSCRConfig, SEN12MSCRDataset
from cloud_removal.synthetic_clouds import CloudConfig, blend_synthetic_clouds, generate_cloud_mask

logger = logging.getLogger("cloud_removal.ui")

# In-memory prediction cache for file downloads (session_id -> data dict)
_MAX_CACHE_ENTRIES = 50
_DOWNLOAD_CACHE: Dict[str, Dict[str, Any]] = {}


def _cleanup_cache() -> None:
    """Evict expired or excess items from download cache."""
    now = time.time()
    # Remove items older than 3600 seconds
    expired = [k for k, v in _DOWNLOAD_CACHE.items() if now - v.get("timestamp", 0) > 3600]
    for k in expired:
        _DOWNLOAD_CACHE.pop(k, None)
    # If still too large, remove oldest
    if len(_DOWNLOAD_CACHE) > _MAX_CACHE_ENTRIES:
        sorted_keys = sorted(_DOWNLOAD_CACHE.keys(), key=lambda k: _DOWNLOAD_CACHE[k].get("timestamp", 0))
        for k in sorted_keys[: len(_DOWNLOAD_CACHE) - _MAX_CACHE_ENTRIES]:
            _DOWNLOAD_CACHE.pop(k, None)


def parse_uploaded_image(
    file_bytes: bytes,
    filename: str,
) -> Tuple[np.ndarray, Optional[str], Optional[List[float]], bool, Dict[str, Any]]:
    """Parse and normalize uploaded image bytes.

    Supports:
      - 13-band Sentinel-2 Level-1C/Level-2A GeoTIFFs (extracts B04, B03, B02, scales DN / 10000).
      - 3-band RGB GeoTIFFs (scales uint16 DN / 10000 or uint8 / 255 or float [0, 1]).
      - 3-band optical RGB images (.png, .jpg, .jpeg) normalized to [0, 1].

    Rejects:
      - Single-band rasters (e.g. isolated LISS-IV BAND2/3/4 rasters).
      - Unsupported channel counts (e.g. 2, 4, 8 bands).
      - Spatial dimensions outside [16, 512].
      - Non-finite pixel values (NaN / Inf).

    Returns:
        Tuple of (array [3, H, W] float32 in [0, 1], crs_str, transform_list, is_geotiff, metadata_dict).
    """
    is_tiff = (
        filename.lower().endswith((".tif", ".tiff"))
        or file_bytes.startswith(b"II*\x00")
        or file_bytes.startswith(b"MM\x00*")
    )

    if is_tiff:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    count = src.count
                    h = src.height
                    w = src.width
                    crs_str = src.crs.to_string() if src.crs else None
                    transform_list = list(src.transform) if src.transform else None
                    res_tuple = src.res if src.res else None

                    if count == 1:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "Single-band raster detected. The U-Net cloud removal model requires 3-band "
                                "RGB optical imagery (Sentinel-2 B04, B03, B02). Note: isolated LISS-IV "
                                "single-band rasters (BAND2, BAND3, BAND4) lack 3-channel alignment and "
                                "cross-sensor calibration, and cannot be evaluated individually."
                            ),
                        )
                    elif count == 13:
                        # Extract Sentinel-2 RGB: B04 (4), B03 (3), B02 (2)
                        raw = src.read([4, 3, 2]).astype(np.float32)
                        norm_arr = np.clip(raw, 0.0, 10000.0) / 10000.0
                        meta_info = {
                            "format": "13-band Sentinel-2 L1C/L2A GeoTIFF",
                            "bands_used": "B04 (Red), B03 (Green), B02 (Blue)",
                            "radiometric_scaling": "Digital Numbers (DN) / 10000.0",
                            "spatial_resolution": f"{res_tuple[0]:.1f}m x {res_tuple[1]:.1f}m" if res_tuple else "Unknown",
                            "crs": crs_str or "Unspecified",
                        }
                    elif count == 3:
                        raw = src.read([1, 2, 3]).astype(np.float32)
                        dtype_str = src.dtypes[0]
                        max_val = float(np.max(raw))
                        if max_val > 255.0 or "16" in dtype_str:
                            norm_arr = np.clip(raw, 0.0, 10000.0) / 10000.0
                            scaling_desc = "DN / 10000.0 (16-bit GeoTIFF)"
                        elif max_val > 1.0 or "8" in dtype_str:
                            norm_arr = np.clip(raw, 0.0, 255.0) / 255.0
                            scaling_desc = "DN / 255.0 (8-bit GeoTIFF)"
                        else:
                            norm_arr = np.clip(raw, 0.0, 1.0)
                            scaling_desc = "Direct Physical Surface Reflectance [0.0, 1.0]"

                        meta_info = {
                            "format": "3-band RGB GeoTIFF",
                            "bands_used": "Bands 1 (Red), 2 (Green), 3 (Blue)",
                            "radiometric_scaling": scaling_desc,
                            "spatial_resolution": f"{res_tuple[0]:.1f}m x {res_tuple[1]:.1f}m" if res_tuple else "Unknown",
                            "crs": crs_str or "Unspecified",
                        }
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"Unsupported raster channel count: {count} bands. "
                                "Supported formats are 13-band Sentinel-2 GeoTIFFs (extracts B04/B03/B02) "
                                "or 3-band RGB GeoTIFFs."
                            ),
                        )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse GeoTIFF file '{filename}': {exc}",
            )
        is_geotiff = True
    else:
        # Standard consumer optical image (PNG, JPEG)
        try:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr_hwc = np.array(img, dtype=np.float32)  # [H, W, 3]
            norm_arr = np.transpose(arr_hwc, (2, 0, 1)) / 255.0  # [3, H, W]
            crs_str = None
            transform_list = None
            is_geotiff = False
            meta_info = {
                "format": f"Standard Optical RGB Image ({img.format or 'Image'})",
                "bands_used": "Red, Green, Blue channels",
                "radiometric_scaling": "Normalized [0, 255] -> [0.0, 1.0]",
                "spatial_resolution": "Unreferenced (Pixel Space)",
                "crs": "None (Standard RGB)",
            }
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to decode image file '{filename}': {exc}",
            )

    # Validate spatial bounds
    channels, h, w = norm_arr.shape
    if channels != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly 3 channels, got {channels}.",
        )
    if h < 16 or w < 16:
        raise HTTPException(
            status_code=422,
            detail=f"Spatial dimensions ({h}x{w}) too small: minimum size is 16x16 pixels.",
        )
    if h > 512 or w > 512:
        raise HTTPException(
            status_code=422,
            detail=f"Spatial dimensions ({h}x{w}) exceed maximum allowed size of 512x512 pixels.",
        )
    if not np.all(np.isfinite(norm_arr)):
        raise HTTPException(
            status_code=422,
            detail="Input array contains non-finite values (NaN or Inf).",
        )

    return norm_arr, crs_str, transform_list, is_geotiff, meta_info


def array_to_stretched_png_base64(
    arr: np.ndarray,
    stretch_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[str, Tuple[float, float]]:
    """Convert a [3, H, W] float32 reflectance array to a base64-encoded PNG data URI.

    Applies contrast stretching so human eyes can view physical surface reflectance.
    Returns (data_uri, (p_low, p_high)).
    """
    if arr.ndim == 3 and arr.shape[0] == 3:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        hwc = arr

    hwc = np.clip(hwc, 0.0, 1.0)

    if stretch_bounds is None:
        p_low = float(np.percentile(hwc, 2.0))
        p_high = float(np.percentile(hwc, 98.0))
        if p_high <= p_low + 1e-6:
            p_low, p_high = 0.0, 1.0
    else:
        p_low, p_high = stretch_bounds

    stretched = np.clip((hwc - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
    uint8_img = (stretched * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(uint8_img, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}", (p_low, p_high)


def create_geotiff_bytes(
    pred_arr: np.ndarray,
    crs: Optional[str] = None,
    transform: Optional[List[float]] = None,
) -> bytes:
    """Export a [3, H, W] float32 prediction array to in-memory GeoTIFF bytes."""
    c, h, w = pred_arr.shape
    with MemoryFile() as memfile:
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": c,
            "dtype": "float32",
        }
        if crs:
            import re
            m = re.search(r"EPSG:\d+", str(crs), re.IGNORECASE)
            if m:
                profile["crs"] = m.group(0).upper()
            else:
                try:
                    profile["crs"] = crs
                except Exception:
                    pass
        if transform and len(transform) >= 6:
            profile["transform"] = rasterio.Affine(*transform[:6])

        with memfile.open(**profile) as dst:
            dst.write(pred_arr.astype(np.float32))
            dst.set_band_description(1, "B04_Red_Reflectance")
            dst.set_band_description(2, "B03_Green_Reflectance")
            dst.set_band_description(3, "B02_Blue_Reflectance")

        return memfile.read()


def get_available_presets() -> List[Dict[str, Any]]:
    """Return preconfigured benchmark and synthetic test presets."""
    presets = [
        {
            "id": "sen12mscr_val_0",
            "name": "Sentinel-2 Validation Patch #0 (p101)",
            "description": "Real Sentinel-2 L1C/L2A winter scene over ROI 102. Contains moderate cloud cover.",
            "category": "Real Benchmark",
            "is_synthetic": False,
            "has_ground_truth": True,
            "patch_id": "p101",
            "split": "val",
            "index": 0,
        },
        {
            "id": "sen12mscr_val_1",
            "name": "Sentinel-2 Validation Patch #1 (p104)",
            "description": "Real Sentinel-2 paired optical scene with heavy cloud cover.",
            "category": "Real Benchmark",
            "is_synthetic": False,
            "has_ground_truth": True,
            "patch_id": "p104",
            "split": "val",
            "index": 1,
        },
        {
            "id": "sen12mscr_val_2",
            "name": "Sentinel-2 Validation Patch #2 (p107)",
            "description": "Real Sentinel-2 paired scene with dense cloud core and thin cloud edges.",
            "category": "Real Benchmark",
            "is_synthetic": False,
            "has_ground_truth": True,
            "patch_id": "p107",
            "split": "val",
            "index": 2,
        },
        {
            "id": "synthetic_dense_0",
            "name": "Synthetic Cloud Simulation (40% Density)",
            "description": "Simulated multi-scale Perlin/Gaussian cloud turbulence overlaid on clear Sentinel-2 surface.",
            "category": "Synthetic Simulation",
            "is_synthetic": True,
            "has_ground_truth": True,
            "coverage": 0.40,
            "seed": 42,
        },
        {
            "id": "synthetic_haze_1",
            "name": "Synthetic Haze & Cirrus (25% Coverage)",
            "description": "Simulated semi-transparent cirrus haze overlaid on clear Sentinel-2 surface.",
            "category": "Synthetic Simulation",
            "is_synthetic": True,
            "has_ground_truth": True,
            "coverage": 0.25,
            "seed": 105,
        },
    ]
    return presets


def load_preset_data(preset_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load cloudy and clear tensors for a given preset identifier."""
    presets = {p["id"]: p for p in get_available_presets()}
    if preset_id not in presets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{preset_id}' not found. Available: {list(presets.keys())}",
        )

    preset_info = presets[preset_id]
    manifest_path = "data/raw/sen12mscr/manifest.json"

    try:
        dataset = SEN12MSCRDataset.from_manifest(manifest_path, split="val")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load benchmark dataset manifest: {exc}",
        )

    if preset_info["is_synthetic"]:
        # Use clear patch from val split, blend synthetic clouds
        sample = dataset[0]
        clear_arr = sample["clear"].numpy()  # [3, 256, 256]
        c, h, w = clear_arr.shape
        cov = preset_info.get("coverage", 0.35)
        seed = preset_info.get("seed", 42)
        cfg = CloudConfig(seed=seed, min_coverage=cov, max_coverage=cov)
        mask = generate_cloud_mask(h, w, coverage=cov, config=cfg, seed=seed)
        cloudy_arr = blend_synthetic_clouds(clear_arr, mask, config=cfg, seed=seed)
        meta = {
            "source": "Synthetic Cloud Generation (Radiative Blending on Sentinel-2 Clear Reference)",
            "is_synthetic": True,
            "crs": "EPSG:32632",
            "transform": None,
            "patch_id": f"synthetic_{preset_info['id']}",
            "spatial_resolution": "10.0m x 10.0m (Simulated)",
            "coverage": cov,
        }
    else:
        idx = preset_info.get("index", 0)
        if idx >= len(dataset):
            idx = 0
        sample = dataset[idx]
        cloudy_arr = sample["cloudy"].numpy()
        clear_arr = sample["clear"].numpy()
        meta = {
            "source": "SEN12MS-CR Paired Optical Benchmark (Winter ROI 102)",
            "is_synthetic": False,
            "crs": "EPSG:32632",
            "transform": None,
            "patch_id": sample["patch_id"],
            "spatial_resolution": "10.0m x 10.0m GSD",
        }

    return cloudy_arr, clear_arr, meta


# ============================================================================
# HTML Template (Responsive, Lightweight, Zero-CDN Dependency)
# ============================================================================

def get_ui_html() -> str:
    """Return the self-contained HTML/CSS/JS document for the Cloud Removal UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LISS-IV / Sentinel-2 Optical Cloud Removal</title>
  <style>
    :root {
      --bg: #0b1120;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #10b981;
      --warning-bg: #451a03;
      --warning-border: #d97706;
      --warning-text: #fde68a;
      --danger: #ef4444;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 1rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 1rem;
      margin-bottom: 1.5rem;
    }
    .header-title { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.6rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-primary { background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid var(--primary); }
    .badge-synth { background: rgba(217, 119, 6, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-real { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }

    /* Advisory Banner */
    .notice-banner {
      background: var(--warning-bg);
      border: 1px solid var(--warning-border);
      color: var(--warning-text);
      padding: 0.85rem 1.1rem;
      border-radius: var(--radius);
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
    }
    .notice-banner strong { color: #fff; font-weight: 600; }

    /* Layout Grid */
    .grid-2 {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 900px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1.25rem;
      margin-bottom: 1.5rem;
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 0.85rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 0.5rem;
    }

    /* Tabs */
    .tab-nav {
      display: flex;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 1rem;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 0.5rem 0.8rem;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      border-bottom: 2px solid transparent;
    }
    .tab-btn.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Form Inputs */
    .form-group { margin-bottom: 1rem; }
    label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem; }
    select, input[type="file"] {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 0.5rem;
      border-radius: 4px;
      font-size: 0.85rem;
    }
    .dropzone {
      border: 2px dashed var(--card-border);
      border-radius: var(--radius);
      padding: 1.2rem;
      text-align: center;
      cursor: pointer;
      background: #0f172a;
      transition: border-color 0.2s;
    }
    .dropzone:hover { border-color: var(--primary); }
    .dropzone p { font-size: 0.8rem; color: var(--text-muted); }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      background: var(--primary);
      color: #0b1120;
      font-weight: 600;
      font-size: 0.85rem;
      padding: 0.55rem 1rem;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      width: 100%;
      transition: background 0.2s;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover { background: #475569; }

    /* Visual Workspace */
    .view-toggle-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .toggle-group { display: flex; gap: 0.4rem; }
    .btn-toggle {
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.35rem 0.75rem;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.8rem;
    }
    .btn-toggle.active {
      background: var(--primary);
      color: #0b1120;
      font-weight: 600;
      border-color: var(--primary);
    }

    /* Comparison Split Slider */
    .slider-container {
      position: relative;
      width: 100%;
      max-width: 512px;
      aspect-ratio: 1 / 1;
      margin: 0 auto;
      overflow: hidden;
      border-radius: var(--radius);
      border: 1px solid var(--card-border);
      user-select: none;
      background: #000;
    }
    .slider-img {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      object-fit: contain;
      pointer-events: none;
    }
    .slider-img-top {
      clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
    }
    .slider-divider {
      position: absolute;
      top: 0; bottom: 0;
      left: 50%;
      width: 3px;
      background: #38bdf8;
      cursor: ew-resize;
      box-shadow: 0 0 8px rgba(0,0,0,0.8);
    }
    .slider-handle {
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      width: 32px; height: 32px;
      background: #38bdf8;
      border-radius: 50%;
      border: 2px solid #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #0b1120;
      font-size: 0.75rem;
      font-weight: bold;
      box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    }
    .slider-range {
      width: 100%;
      max-width: 512px;
      margin: 0.8rem auto 0 auto;
      display: block;
    }
    .slider-tags {
      display: flex;
      justify-content: space-between;
      max-width: 512px;
      margin: 0.3rem auto 1rem auto;
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Side-by-Side Panels */
    .panels-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }
    .panel-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 0.75rem;
      text-align: center;
    }
    .panel-box img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 4px;
      background: #000;
    }
    .panel-title {
      font-size: 0.8rem;
      font-weight: 600;
      margin-top: 0.5rem;
      color: var(--text);
    }
    .panel-subtitle {
      font-size: 0.7rem;
      color: var(--text-muted);
    }

    /* Info & Metrics Cards */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.75rem;
      margin-top: 0.5rem;
    }
    .metric-card {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 4px;
      padding: 0.6rem 0.75rem;
    }
    .metric-label { font-size: 0.7rem; color: var(--text-muted); }
    .metric-value { font-size: 1.15rem; font-weight: 700; color: #fff; }
    .metric-delta { font-size: 0.75rem; font-weight: 600; }
    .delta-pos { color: #34d399; }
    .delta-neg { color: #f87171; }

    .meta-list { font-size: 0.8rem; }
    .meta-item {
      display: flex;
      justify-content: space-between;
      padding: 0.35rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .meta-k { color: var(--text-muted); }
    .meta-v { font-weight: 500; text-align: right; word-break: break-all; }

    /* Download Row */
    .download-row {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-top: 1rem;
    }
    .btn-download {
      flex: 1;
      min-width: 150px;
    }

    /* Loading Spinner */
    .spinner {
      display: none;
      width: 20px; height: 20px;
      border: 2px solid rgba(0,0,0,0.2);
      border-top-color: #000;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .empty-state {
      padding: 3rem 1rem;
      text-align: center;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>Satellite Optical Cloud Removal</h1>
        <div>
          <span id="serviceBadge" class="badge badge-primary">Model: Compact U-Net</span>
          <span id="sourceBadge" class="badge badge-real" style="display:none;">Real S2</span>
        </div>
      </div>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
        Deterministic optical cloud removal inference service for Sentinel-2 surface reflectance (RGB: B04, B03, B02).
      </p>
    </header>

    <!-- Scientific Advisory Banner -->
    <div class="notice-banner">
      <strong>⚠️ Sensor & Domain Boundary Notice:</strong> The active model was trained exclusively on
      <strong>Sentinel-2 optical surface reflectance (SEN12MS-CR benchmark, 10m GSD, B04/B03/B02 RGB)</strong>.
      Cross-sensor inference on <strong>ISRO Resourcesat-2A LISS-IV imagery</strong> (5m GSD, Green/Red/NIR bands without Blue channel)
      remains unvalidated due to spectral band mismatch and unresolved sensor calibrations.
      Synthetic cloud examples are generated with Perlin/Gaussian turbulence simulation.
    </div>

    <div class="grid-2">
      <!-- Left Column: Controls & Upload -->
      <div>
        <div class="card">
          <div class="card-title">
            <span>Input Selection</span>
          </div>

          <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('tabPreset')">Benchmark Presets</button>
            <button class="tab-btn" onclick="switchTab('tabUpload')">Upload Image</button>
          </div>

          <!-- Tab 1: Presets -->
          <div id="tabPreset" class="tab-content active">
            <div class="form-group">
              <label for="presetSelect">Select Test Sample:</label>
              <select id="presetSelect" onchange="onPresetChange()">
                <option value="">Loading presets...</option>
              </select>
            </div>
            <p id="presetDesc" style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem; min-height: 2.5rem;"></p>
            <button id="btnRunPreset" class="btn" onclick="runPresetInference()">
              <div class="spinner" id="spinnerPreset"></div>
              <span>Run Cloud Removal</span>
            </button>
          </div>

          <!-- Tab 2: Upload -->
          <div id="tabUpload" class="tab-content">
            <div class="form-group">
              <label>Cloudy Input Image (Required):</label>
              <div class="dropzone" onclick="document.getElementById('cloudyFileInput').click()">
                <p id="cloudyFileLabel">📁 Click to choose GeoTIFF (.tif) or RGB (.png, .jpg)</p>
                <input type="file" id="cloudyFileInput" accept=".tif,.tiff,.png,.jpg,.jpeg" style="display:none;" onchange="onFileSelected(this, 'cloudyFileLabel')">
              </div>
            </div>

            <div class="form-group">
              <label>Clear Reference Scene (Optional - for MAE/PSNR evaluation):</label>
              <div class="dropzone" onclick="document.getElementById('clearFileInput').click()">
                <p id="clearFileLabel">📁 Optional ground-truth (.tif, .png)</p>
                <input type="file" id="clearFileInput" accept=".tif,.tiff,.png,.jpg,.jpeg" style="display:none;" onchange="onFileSelected(this, 'clearFileLabel')">
              </div>
            </div>

            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem;">
              • <strong>Supported formats:</strong> 13-band Sentinel-2 GeoTIFF (extracts B04/B03/B02), 3-band RGB GeoTIFF, or standard 3-channel RGB image.<br>
              • <strong>Single-band rasters</strong> (e.g. raw LISS-IV BAND2/3/4) are rejected.
            </div>

            <button id="btnRunUpload" class="btn" onclick="runUploadInference()">
              <div class="spinner" id="spinnerUpload"></div>
              <span>Upload & Run Model</span>
            </button>
          </div>
        </div>

        <!-- Metadata & Status Card -->
        <div class="card" id="metaCard" style="display:none;">
          <div class="card-title">
            <span>Inference Metadata</span>
            <span id="latencyBadge" style="font-size: 0.75rem; color: var(--primary); font-weight: bold;">-- ms</span>
          </div>
          <div class="meta-list">
            <div class="meta-item">
              <span class="meta-k">Data Source:</span>
              <span class="meta-v" id="metaSource">--</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Type / Label:</span>
              <span class="meta-v" id="metaTypeLabel">--</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Image Dimensions:</span>
              <span class="meta-v" id="metaDims">--</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">CRS (Spatial Reference):</span>
              <span class="meta-v" id="metaCrs">--</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Resolution:</span>
              <span class="meta-v" id="metaRes">--</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Model Checkpoint:</span>
              <span class="meta-v" id="metaCheckpoint">--</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Visual Workspace & Accuracy -->
      <div>
        <div class="card">
          <div class="view-toggle-bar">
            <div class="card-title" style="border:none; margin:0; padding:0;">
              <span>Reconstruction Workspace</span>
            </div>
            <div class="toggle-group">
              <button id="btnToggleSlider" class="btn-toggle active" onclick="setViewMode('slider')">⚡ Split Slider</button>
              <button id="btnTogglePanels" class="btn-toggle" onclick="setViewMode('panels')">⊞ Side-by-Side</button>
            </div>
          </div>

          <!-- Empty placeholder before inference -->
          <div id="emptyWorkspace" class="empty-state">
            <p>Select a benchmark preset or upload an optical satellite image on the left, then click <strong>Run Cloud Removal</strong>.</p>
          </div>

          <!-- Results Workspace Container -->
          <div id="activeWorkspace" style="display:none;">
            <!-- Split Slider View -->
            <div id="viewSlider">
              <div class="slider-container" id="sliderBox">
                <!-- Under image: Cloudy -->
                <img id="imgCloudySlider" class="slider-img" src="" alt="Cloudy Input">
                <!-- Top image: Predicted Clean -->
                <img id="imgPredSlider" class="slider-img slider-img-top" src="" alt="U-Net Clean Prediction">
                <!-- Vertical Divider -->
                <div class="slider-divider" id="sliderDivider">
                  <div class="slider-handle">↔</div>
                </div>
              </div>
              <input type="range" min="0" max="100" value="50" class="slider-range" id="sliderRange" oninput="updateSplit(this.value)">
              <div class="slider-tags">
                <span>◀ Clean U-Net Prediction</span>
                <span>Cloudy Input ▶</span>
              </div>
            </div>

            <!-- Side-by-Side View -->
            <div id="viewPanels" style="display:none;">
              <div class="panels-container" id="panelsBox">
                <div class="panel-box">
                  <img id="panelCloudy" src="" alt="Cloudy Input">
                  <div class="panel-title">Cloudy Input</div>
                  <div class="panel-subtitle" id="subCloudy">--</div>
                </div>
                <div class="panel-box">
                  <img id="panelPred" src="" alt="U-Net Prediction">
                  <div class="panel-title">U-Net Prediction</div>
                  <div class="panel-subtitle" id="subPred">--</div>
                </div>
                <div class="panel-box" id="boxClear" style="display:none;">
                  <img id="panelClear" src="" alt="Ground Truth">
                  <div class="panel-title">Ground Truth Clear</div>
                  <div class="panel-subtitle">Reference Target</div>
                </div>
              </div>
            </div>

            <!-- Downloads -->
            <div class="download-row">
              <button id="btnDownloadTiff" class="btn btn-secondary btn-download" onclick="downloadPrediction('geotiff')">
                💾 Download GeoTIFF (.tif)
              </button>
              <button id="btnDownloadPng" class="btn btn-secondary btn-download" onclick="downloadPrediction('png')">
                🖼️ Download PNG (.png)
              </button>
            </div>
          </div>
        </div>

        <!-- Accuracy Evaluation Card (Strictly conditional) -->
        <div class="card" id="accuracyCard" style="display:none;">
          <div class="card-title">
            <span>Quantitative Accuracy Evaluation</span>
            <span id="gtBadge" class="badge badge-real">Ground Truth Present</span>
          </div>

          <div id="gtAvailableSection">
            <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">
              Computed strictly on raw physical surface reflectance values in [0.0, 1.0]:
            </p>
            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-label">Cloudy Base MAE</div>
                <div class="metric-value" id="valBaseMae">--</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">U-Net Pred MAE</div>
                <div class="metric-value" id="valPredMae">--</div>
                <div class="metric-delta delta-pos" id="deltaMae">--</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Cloudy Base PSNR</div>
                <div class="metric-value" id="valBasePsnr">-- dB</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">U-Net Pred PSNR</div>
                <div class="metric-value" id="valPredPsnr">-- dB</div>
                <div class="metric-delta delta-pos" id="deltaPsnr">--</div>
              </div>
            </div>
          </div>

          <div id="gtUnavailableSection" style="display:none; font-size: 0.85rem; color: var(--text-muted); padding: 0.5rem 0;">
            ℹ️ <strong>No reference clear scene provided.</strong> Quantitative metrics (MAE and PSNR) are strictly omitted
            because evaluation requires a true cloud-free paired ground-truth image.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSessionId = null;
    let presetsData = [];
    let currentViewMode = "slider";

    // Initialize UI on page load
    window.addEventListener("DOMContentLoaded", async () => {
      await loadPresets();
      initSliderDrag();
    });

    function switchTab(tabId) {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      if (tabId === 'tabPreset') {
        document.querySelectorAll(".tab-btn")[0].classList.add("active");
        document.getElementById('tabPreset').classList.add("active");
      } else {
        document.querySelectorAll(".tab-btn")[1].classList.add("active");
        document.getElementById('tabUpload').classList.add("active");
      }
    }

    function setViewMode(mode) {
      currentViewMode = mode;
      document.getElementById("btnToggleSlider").classList.toggle("active", mode === "slider");
      document.getElementById("btnTogglePanels").classList.toggle("active", mode === "panels");
      document.getElementById("viewSlider").style.display = mode === "slider" ? "block" : "none";
      document.getElementById("viewPanels").style.display = mode === "panels" ? "block" : "none";
    }

    function onFileSelected(input, labelId) {
      const label = document.getElementById(labelId);
      if (input.files && input.files[0]) {
        label.innerHTML = `📄 <strong>${input.files[0].name}</strong> (${(input.files[0].size / 1024).toFixed(1)} KB)`;
      }
    }

    async function loadPresets() {
      try {
        const resp = await fetch("/api/ui/presets");
        if (!resp.ok) throw new Error("Failed to fetch presets");
        presetsData = await resp.json();
        const sel = document.getElementById("presetSelect");
        sel.innerHTML = presetsData.map(p =>
          `<option value="${p.id}">[${p.category}] ${p.name}</option>`
        ).join("");
        onPresetChange();
      } catch (err) {
        console.error(err);
        document.getElementById("presetDesc").innerText = "Failed to load presets: " + err.message;
      }
    }

    function onPresetChange() {
      const sel = document.getElementById("presetSelect");
      const selectedId = sel.value;
      const p = presetsData.find(item => item.id === selectedId);
      if (p) {
        document.getElementById("presetDesc").innerHTML =
          `<strong>${p.name}:</strong> ${p.description} <br><em>${p.is_synthetic ? '⚠️ Synthetic cloud simulation' : '✓ Real SEN12MS-CR paired data'}</em>`;
      }
    }

    async function runPresetInference() {
      const presetId = document.getElementById("presetSelect").value;
      if (!presetId) return;

      setLoading(true, "Preset");
      try {
        const formData = new FormData();
        formData.append("preset_id", presetId);

        const resp = await fetch("/api/ui/predict", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Inference failed");

        renderInferenceResult(data);
      } catch (err) {
        alert("Error: " + err.message);
      } finally {
        setLoading(false, "Preset");
      }
    }

    async function runUploadInference() {
      const cloudyInput = document.getElementById("cloudyFileInput");
      if (!cloudyInput.files || !cloudyInput.files[0]) {
        alert("Please select a cloudy input image file first.");
        return;
      }

      setLoading(true, "Upload");
      try {
        const formData = new FormData();
        formData.append("file", cloudyInput.files[0]);

        const clearInput = document.getElementById("clearFileInput");
        if (clearInput.files && clearInput.files[0]) {
          formData.append("clear_file", clearInput.files[0]);
        }

        const resp = await fetch("/api/ui/predict", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Inference failed");

        renderInferenceResult(data);
      } catch (err) {
        alert("Upload Error: " + err.message);
      } finally {
        setLoading(false, "Upload");
      }
    }

    function renderInferenceResult(data) {
      currentSessionId = data.session_id;

      // Workspace displays
      document.getElementById("emptyWorkspace").style.display = "none";
      document.getElementById("activeWorkspace").style.display = "block";
      document.getElementById("metaCard").style.display = "block";
      document.getElementById("accuracyCard").style.display = "block";

      // Images in Split Slider
      document.getElementById("imgCloudySlider").src = data.cloudy_preview;
      document.getElementById("imgPredSlider").src = data.pred_preview;
      updateSplit(50);
      document.getElementById("sliderRange").value = 50;

      // Images in Side-by-Side Panels
      document.getElementById("panelCloudy").src = data.cloudy_preview;
      document.getElementById("panelPred").src = data.pred_preview;
      document.getElementById("subCloudy").innerText = `${data.input_shape[1]}x${data.input_shape[2]} px`;
      document.getElementById("subPred").innerText = `U-Net Reconstruction`;

      if (data.clear_preview) {
        document.getElementById("boxClear").style.display = "block";
        document.getElementById("panelClear").src = data.clear_preview;
      } else {
        document.getElementById("boxClear").style.display = "none";
      }

      // Metadata
      document.getElementById("latencyBadge").innerText = `${data.latency_ms.toFixed(1)} ms CPU`;
      document.getElementById("metaSource").innerText = data.metadata.source || data.metadata.format || "Custom Upload";
      document.getElementById("metaTypeLabel").innerHTML = data.is_synthetic
        ? '<span class="badge badge-synth">Synthetic Simulation</span>'
        : '<span class="badge badge-real">Real Scene</span>';
      document.getElementById("metaDims").innerText = `${data.input_shape.join(" × ")} [Channels, H, W]`;
      document.getElementById("metaCrs").innerText = data.crs || "None (Standard Optical RGB)";
      document.getElementById("metaRes").innerText = data.metadata.spatial_resolution || "Unreferenced";
      document.getElementById("metaCheckpoint").innerText = data.checkpoint || "compact U-Net (Epoch 5)";

      // Top badges
      const srcBadge = document.getElementById("sourceBadge");
      srcBadge.style.display = "inline-block";
      if (data.is_synthetic) {
        srcBadge.className = "badge badge-synth";
        srcBadge.innerText = "Synthetic Cloud Sim";
      } else {
        srcBadge.className = "badge badge-real";
        srcBadge.innerText = "Real Optical S2";
      }

      // GeoTIFF Download availability
      const btnTiff = document.getElementById("btnDownloadTiff");
      if (data.is_geotiff) {
        btnTiff.disabled = false;
        btnTiff.title = "Download full-precision 3-band GeoTIFF with CRS and Transform";
      } else {
        btnTiff.disabled = true;
        btnTiff.title = "GeoTIFF export unavailable for standard PNG/JPG uploads without geospatial CRS metadata";
      }

      // Accuracy Metrics
      const gtAvail = document.getElementById("gtAvailableSection");
      const gtUnavail = document.getElementById("gtUnavailableSection");
      const gtBadge = document.getElementById("gtBadge");

      if (data.has_ground_truth && data.metrics) {
        gtAvail.style.display = "block";
        gtUnavail.style.display = "none";
        gtBadge.className = "badge badge-real";
        gtBadge.innerText = "Ground Truth Present";

        const m = data.metrics;
        document.getElementById("valBaseMae").innerText = m.input_mae.toFixed(4);
        document.getElementById("valPredMae").innerText = m.pred_mae.toFixed(4);
        document.getElementById("deltaMae").innerText = `${m.delta_mae >= 0 ? 'Improvement: -' : '+'}${Math.abs(m.delta_mae).toFixed(4)}`;

        document.getElementById("valBasePsnr").innerText = `${m.input_psnr.toFixed(2)} dB`;
        document.getElementById("valPredPsnr").innerText = `${m.pred_psnr.toFixed(2)} dB`;
        document.getElementById("deltaPsnr").innerText = `${m.delta_psnr >= 0 ? '+' : ''}${m.delta_psnr.toFixed(2)} dB`;
      } else {
        gtAvail.style.display = "none";
        gtUnavail.style.display = "block";
        gtBadge.className = "badge badge-synth";
        gtBadge.innerText = "Ground Truth: None";
      }
    }

    function updateSplit(pct) {
      const topImg = document.getElementById("imgPredSlider");
      const divider = document.getElementById("sliderDivider");
      topImg.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
      divider.style.left = `${pct}%`;
    }

    function initSliderDrag() {
      const box = document.getElementById("sliderBox");
      let isDragging = false;

      function onMove(clientX) {
        const rect = box.getBoundingClientRect();
        let x = clientX - rect.left;
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;
        const pct = (x / rect.width) * 100;
        updateSplit(pct);
        document.getElementById("sliderRange").value = pct;
      }

      box.addEventListener("mousedown", (e) => {
        isDragging = true;
        onMove(e.clientX);
      });
      window.addEventListener("mousemove", (e) => {
        if (isDragging) onMove(e.clientX);
      });
      window.addEventListener("mouseup", () => { isDragging = false; });

      // Touch events for mobile/tablet
      box.addEventListener("touchstart", (e) => {
        if (e.touches.length > 0) {
          isDragging = true;
          onMove(e.touches[0].clientX);
        }
      });
      window.addEventListener("touchmove", (e) => {
        if (isDragging && e.touches.length > 0) onMove(e.touches[0].clientX);
      });
      window.addEventListener("touchend", () => { isDragging = false; });
    }

    function downloadPrediction(format) {
      if (!currentSessionId) {
        alert("No active prediction to download.");
        return;
      }
      window.location.href = `/api/ui/download/${currentSessionId}?format=${format}`;
    }

    function setLoading(isLoading, type) {
      const btn = document.getElementById(type === "Preset" ? "btnRunPreset" : "btnRunUpload");
      const spinner = document.getElementById(type === "Preset" ? "spinnerPreset" : "spinnerUpload");
      btn.disabled = isLoading;
      spinner.style.display = isLoading ? "inline-block" : "none";
    }
  </script>
</body>
</html>
"""


# ============================================================================
# FastAPI UI Router Setup
# ============================================================================

def setup_ui(app: FastAPI) -> None:
    """Attach web UI endpoints and interactive prediction handlers to the FastAPI application."""

    router = APIRouter(tags=["Web UI"])

    @router.get("/", response_class=HTMLResponse, summary="Interactive Web UI")
    @router.get("/ui", response_class=HTMLResponse, summary="Interactive Web UI")
    async def get_ui():
        """Serve the self-contained, responsive Cloud Removal Web UI."""
        return HTMLResponse(content=get_ui_html(), status_code=200)

    @router.get("/api/ui/presets", summary="Get Preset Test Samples")
    async def get_presets():
        """Return available SEN12MS-CR benchmark and synthetic simulation presets."""
        return JSONResponse(content=get_available_presets())

    @router.post("/api/ui/predict", summary="Interactive Cloud Removal Inference")
    async def predict_interactive(
        request: Request,
        file: Optional[UploadFile] = File(None),
        clear_file: Optional[UploadFile] = File(None),
        preset_id: Optional[str] = Form(None),
    ):
        """Execute cloud removal inference on an uploaded image or benchmark preset.

        Reuses the existing ModelManager from app.state without code duplication.
        Supports:
          - Multi-band Sentinel-2 GeoTIFFs (13 bands)
          - 3-band RGB GeoTIFFs
          - Standard optical RGB images (.png, .jpg)
          - Pre-indexed validation and synthetic cloud presets
        """
        model_manager = getattr(request.app.state, "model_manager", None)
        if model_manager is None or not model_manager.is_ready or model_manager.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service model is not loaded or ready.",
            )

        has_ground_truth = False
        clear_arr = None
        crs_str = None
        transform_list = None
        is_geotiff = False
        is_synthetic = False
        metadata: Dict[str, Any] = {}

        if preset_id:
            # Load selected preset
            cloudy_arr, clear_arr, preset_meta = load_preset_data(preset_id)
            has_ground_truth = True
            is_synthetic = preset_meta.get("is_synthetic", False)
            crs_str = preset_meta.get("crs")
            transform_list = preset_meta.get("transform")
            is_geotiff = True
            metadata = preset_meta
        elif file is not None:
            # Handle uploaded cloudy image
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=422,
                    detail="Uploaded cloudy image file is empty.",
                )
            cloudy_arr, crs_str, transform_list, is_geotiff, metadata = parse_uploaded_image(
                content, file.filename or "uploaded_image"
            )

            # Optional clear ground truth file
            if clear_file is not None:
                clear_content = await clear_file.read()
                if clear_content:
                    c_arr, _, _, _, _ = parse_uploaded_image(clear_content, clear_file.filename or "clear_image")
                    if c_arr.shape == cloudy_arr.shape:
                        clear_arr = c_arr
                        has_ground_truth = True
                    else:
                        logger.warning(
                            "Clear image shape %s does not match cloudy image shape %s; omitting GT metrics.",
                            c_arr.shape,
                            cloudy_arr.shape,
                        )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'preset_id' or 'file' (cloudy image upload) must be provided.",
            )

        # Execute model forward pass under async lock (reusing ModelManager)
        input_tensor = torch.from_numpy(cloudy_arr)
        t0 = time.perf_counter()
        try:
            output_tensor = await model_manager.predict(input_tensor)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            logger.error("Inference execution failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model forward pass failed: {exc}",
            )

        pred_arr = output_tensor.numpy()

        # Compute quantitative metrics strictly if ground truth is present
        metrics_dict: Optional[Dict[str, float]] = None
        if has_ground_truth and clear_arr is not None:
            base_mae = float(compute_mae(cloudy_arr, clear_arr))
            pred_mae = float(compute_mae(pred_arr, clear_arr))
            delta_mae = base_mae - pred_mae

            base_psnr = float(compute_psnr(cloudy_arr, clear_arr))
            pred_psnr = float(compute_psnr(pred_arr, clear_arr))
            delta_psnr = pred_psnr - base_psnr

            metrics_dict = {
                "input_mae": round(base_mae, 5),
                "pred_mae": round(pred_mae, 5),
                "delta_mae": round(delta_mae, 5),
                "input_psnr": round(base_psnr, 2),
                "pred_psnr": round(pred_psnr, 2),
                "delta_psnr": round(delta_psnr, 2),
            }

        # Shared display contrast stretch for preview generation
        ref_for_stretch = clear_arr if clear_arr is not None else pred_arr
        cloudy_b64, stretch_bounds = array_to_stretched_png_base64(cloudy_arr)
        pred_b64, _ = array_to_stretched_png_base64(pred_arr, stretch_bounds=stretch_bounds)

        clear_b64 = None
        if has_ground_truth and clear_arr is not None:
            clear_b64, _ = array_to_stretched_png_base64(clear_arr, stretch_bounds=stretch_bounds)

        # Store in session cache for downloadable export
        _cleanup_cache()
        session_id = uuid.uuid4().hex
        _DOWNLOAD_CACHE[session_id] = {
            "pred_arr": pred_arr,
            "crs": crs_str,
            "transform": transform_list,
            "is_geotiff": is_geotiff,
            "timestamp": time.time(),
        }

        return JSONResponse(
            content={
                "session_id": session_id,
                "latency_ms": round(latency_ms, 2),
                "input_shape": list(cloudy_arr.shape),
                "output_shape": list(pred_arr.shape),
                "crs": crs_str,
                "is_geotiff": is_geotiff,
                "is_synthetic": is_synthetic,
                "has_ground_truth": has_ground_truth,
                "metrics": metrics_dict,
                "cloudy_preview": cloudy_b64,
                "pred_preview": pred_b64,
                "clear_preview": clear_b64,
                "checkpoint": model_manager.checkpoint_path,
                "metadata": metadata,
            }
        )

    @router.get("/api/ui/download/{session_id}", summary="Download Prediction")
    async def download_prediction_file(
        session_id: str,
        format: str = "png",
    ):
        """Download predicted cloud-free surface reflectance as GeoTIFF or PNG."""
        cached = _DOWNLOAD_CACHE.get(session_id)
        if not cached:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction session expired or not found. Please run inference again.",
            )

        pred_arr = cached["pred_arr"]
        crs = cached["crs"]
        transform = cached["transform"]
        is_geotiff = cached["is_geotiff"]

        fmt = format.lower()
        if fmt in ("geotiff", "tif", "tiff"):
            if not is_geotiff:
                logger.info("Exporting unreferenced GeoTIFF for non-geotiff upload.")
            tiff_bytes = create_geotiff_bytes(pred_arr, crs=crs, transform=transform)
            filename = f"cloudfree_unet_{session_id[:8]}.tif"
            return Response(
                content=tiff_bytes,
                media_type="image/tiff",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        elif fmt == "png":
            # 8-bit PNG rendered with 2%-98% stretch for natural visualization
            hwc = np.transpose(pred_arr, (1, 2, 0))
            p2, p98 = np.percentile(hwc, (2.0, 98.0))
            if p98 <= p2 + 1e-6:
                p2, p98 = 0.0, 1.0
            stretched = np.clip((hwc - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
            uint8_img = (stretched * 255.0).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(uint8_img, mode="RGB").save(buf, format="PNG")
            filename = f"cloudfree_unet_{session_id[:8]}.png"
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{format}'. Supported formats are 'geotiff' and 'png'.",
            )

    app.include_router(router)
