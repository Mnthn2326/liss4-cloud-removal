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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from rasterio.io import MemoryFile

from cloud_removal.metrics import compute_mae, compute_psnr
from cloud_removal.sen12mscr import SEN12MSCRDataset
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
        sorted_keys = sorted(
            _DOWNLOAD_CACHE.keys(), key=lambda k: _DOWNLOAD_CACHE[k].get("timestamp", 0)
        )
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
                            "spatial_resolution": (
                                f"{res_tuple[0]:.1f}m x {res_tuple[1]:.1f}m"
                                if res_tuple
                                else "Unknown"
                            ),
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
                            "spatial_resolution": (
                                f"{res_tuple[0]:.1f}m x {res_tuple[1]:.1f}m"
                                if res_tuple
                                else "Unknown"
                            ),
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
            raw_img = Image.open(io.BytesIO(file_bytes))
            img = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img
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


def get_available_models() -> List[Dict[str, Any]]:
    """Return available trained model checkpoints in the repository."""
    candidates = [
        (
            "checkpoints/best_unet.pt",
            "Full U-Net (42.13 dB - Best Model)",
            "Standard U-Net (7.76M params, compound L1+SSIM, 42.13 dB val PSNR at epoch 138).",
        ),
        (
            "checkpoints/best_compact_unet_v2.pt",
            "Compact U-Net v2 (41.09 dB - Record)",
            "Lightweight compact U-Net (1.94M params, cosine annealing schedule, 41.09 dB val PSNR).",
        ),
        (
            "checkpoints/rice2/best_rice2_compact_unet.pt",
            "RICE2 Compact U-Net (0.048 Val MAE)",
            "Lightweight U-Net trained on RICE2 paired benchmark (1.94M params, 0.048 val MAE).",
        ),
        (
            "checkpoints/compact/best_compact_unet.pt",
            "Compact U-Net (Epoch 5 Baseline)",
            "Early compact baseline model (1.94M params).",
        ),
        (
            "checkpoints/cgan/best_cgan_generator.pt",
            "Pix2Pix cGAN Generator",
            "Adversarial patch-based conditional GAN generator (11.38M params).",
        ),
        (
            "checkpoints/fcc_cgan/best_fcc_cgan_generator.pt",
            "LISS-IV FCC cGAN Generator",
            "False Color Composite (NIR-Red-Green) cGAN generator.",
        ),
    ]
    models = []
    for rel_path, name, desc in candidates:
        p = Path(rel_path)
        if p.is_file():
            models.append(
                {
                    "id": rel_path.replace("\\", "/"),
                    "name": name,
                    "description": desc,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                }
            )
    return models


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
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Remote-Sensing Optical Cloud Reconstruction — Landsat/Sentinel-2 RGB Inference</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Google Fonts: Plus Jakarta Sans (Headings/UI), Roboto (Body), Roboto Mono (Telemetry/Metrics) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Roboto+Mono:wght@400;500;600&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">

  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            display: ['"Plus Jakarta Sans"', 'Roboto', 'sans-serif'],
            sans: ['Roboto', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            mono: ['"Roboto Mono"', 'monospace'],
          },
          colors: {
            // 60% Dominant Neutral Surfaces
            canvas: '#F8F9FA',
            surface: '#FFFFFF',
            subtle: '#F1F3F4',
            borderOuter: '#DADCE0',
            borderInner: '#E8EAED',
            // 30% Typography & Ink Hierarchy
            ink: '#202124',
            inkBody: '#3C4043',
            slate: '#5F6368',
            muted: '#70757A',
            // 10% Intentional Accent (Google Blue)
            gblue: '#1A73E8',
            gblueHover: '#1557B0',
            gblueActive: '#1967D2',
            gblueTint: '#E8F0FE',
            // Semantic Warning (Understated Google Cloud Callout)
            warnBg: '#FEF7E0',
            warnBorder: '#FCE8B2',
            warnAccent: '#F9AB00',
            warnInk: '#B06000',
            // Semantic Success
            successBg: '#E6F4EA',
            successBorder: '#CEEAD6',
            successAccent: '#34A853',
            successInk: '#137333',
          }
        }
      }
    }
  </script>

  <style>
    * {
      box-sizing: border-box;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    body {
      background-color: #F8F9FA;
      color: #202124;
      font-family: 'Roboto', sans-serif;
      margin: 0;
      overflow-x: hidden;
    }

    .tabular-nums {
      font-variant-numeric: tabular-nums;
    }

    /* Clean subtle scrollbars */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: #F8F9FA;
    }
    ::-webkit-scrollbar-thumb {
      background: #DADCE0;
      border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #BDC1C6;
    }

    /* Crisp pixelated rendering for Earth Observation rasters */
    canvas.eo-canvas {
      image-rendering: pixelated;
      cursor: crosshair;
      display: block;
    }

    /* Google Material Outlined Select Focus Ring */
    .g-select:focus {
      outline: none;
      border-color: #1A73E8;
      box-shadow: 0 0 0 1px #1A73E8;
    }

    /* Subtle highlight animation when cross-sensor notice is triggered */
    @keyframes calloutHighlight {
      0%, 100% { background-color: #FEF7E0; }
      50% { background-color: #FDE293; }
    }
    .callout-flash {
      animation: calloutHighlight 0.9s ease-in-out 2;
    }

    /* Range slider modern styling */
    input[type=range] {
      accent-color: #1A73E8;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col bg-canvas text-ink selection:bg-gblueTint selection:text-gblueActive">

  <!-- ===================================================================== -->
  <!-- 1. TOP APPLICATION HEADER (Google Cloud / Earth Engine Workbench Bar) -->
  <!-- ===================================================================== -->
  <header class="sticky top-0 z-30 bg-surface border-b border-borderOuter min-h-[64px] px-6 flex items-center shadow-sm">
    <div class="max-w-[1600px] w-full mx-auto py-2.5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">

      <!-- Left: Minimal Geometric Earth Engine-inspired SVG Mark + Title & Subtitle -->
      <div class="flex items-start sm:items-center gap-3.5">
        <div class="w-9 h-9 rounded-[6px] bg-gblueTint border border-[#D2E3FC] flex items-center justify-center shrink-0 mt-0.5 sm:mt-0" aria-hidden="true">
          <!-- Minimalist Geometric Satellite / Grid SVG in #1A73E8 -->
          <svg class="w-5 h-5 text-gblue" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <path d="M10 6.5h4M6.5 10v4M17.5 10v4M10 17.5h4" />
          </svg>
        </div>
        <div>
          <h1 class="font-display text-[18px] font-semibold text-ink leading-snug tracking-[-0.01em]">
            Remote-Sensing Optical Cloud Reconstruction
          </h1>
          <p class="text-[13px] text-slate leading-normal mt-0.5">
            Deterministic optical cloud reconstruction service for RICE2 & Landsat-style RGB observations (<span class="font-mono text-[12px] text-inkBody">B04, B03, B02</span>)
          </p>
        </div>
      </div>

      <!-- Right: Active Model Status Indicators (Restrained Technical Chips) -->
      <div class="flex flex-wrap items-center gap-2 self-start lg:self-center shrink-0">
        <!-- Status Indicator 1: Model Chip -->
        <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-[4px] bg-gblueTint border border-[#D2E3FC] text-gblueActive font-mono text-[11px] font-medium tabular-nums">
          <span class="w-1.5 h-1.5 rounded-full bg-gblue animate-pulse"></span>
          <span id="serviceBadge">Model: Full U-Net (42.13 dB - Best)</span>
        </div>

        <!-- Status Indicator 2: Real / Synthetic S2 Chip -->
        <div id="sourceBadge" class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-subtle border border-borderOuter text-inkBody font-mono text-[11px] font-medium">
          <span id="sourceBadgeText">REAL OPTICAL S2</span>
        </div>
      </div>

    </div>
  </header>

  <!-- ===================================================================== -->
  <!-- 2. SENSOR & DOMAIN BOUNDARY NOTICE (Google Cloud Inline Callout)      -->
  <!-- ===================================================================== -->
  <section class="max-w-[1600px] w-full mx-auto px-6 pt-4">
    <div id="domainNoticeCallout" class="bg-warnBg border border-warnBorder border-l-[3px] border-l-warnAccent rounded-[6px] px-4 py-3 transition-colors">
      <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
        <div class="flex items-start gap-3">
          <!-- Google Material Warning Icon in #B06000 -->
          <svg class="w-4 h-4 text-warnInk shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 6zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" />
          </svg>
          <div class="text-[13px] leading-relaxed text-inkBody">
            <span class="font-medium text-warnInk mr-1">Sensor & Domain Boundary Notice:</span>
            <span>
              The production baseline is officially evaluated on the <span class="font-mono text-[12px] text-ink">RICE2</span> benchmark (Landsat-style RGB surface reflectance). Output is an <span class="font-medium text-ink">estimated surface reconstruction</span>; opaque optical cloud penetration is physically impossible. Historical ISRO LISS-IV sensor tooling is classified as <span class="font-medium text-ink">experimental</span> and unsupported for production inference (see <code class="font-mono text-[11px]">docs/future_work/liss4_roadmap.md</code>).
            </span>
          </div>
        </div>

        <button id="toggleBandSpecBtn" type="button" class="shrink-0 self-start md:self-center text-[12px] font-medium text-warnInk hover:underline inline-flex items-center gap-1">
          <span id="toggleBandSpecText">View band comparison</span>
          <svg id="toggleBandSpecIcon" class="w-3.5 h-3.5 transition-transform" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>

      <!-- Collapsible Technical Band Comparison Table -->
      <div id="bandSpecDrawer" class="hidden mt-3 pt-3 border-t border-warnBorder grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
        <div class="bg-surface border border-borderOuter rounded-[4px] p-2.5">
          <div class="font-medium text-ink mb-1 flex items-center justify-between">
            <span>Sentinel-2 MSI (Validated Training Domain)</span>
            <span class="font-mono text-[11px] text-gblueActive bg-gblueTint px-1.5 py-0.5 rounded">10m GSD • True RGB</span>
          </div>
          <p class="text-slate font-mono text-[11px] leading-relaxed">
            B04 (Red: 665 nm) • B03 (Green: 560 nm) • B02 (Blue: 490 nm) — L2A Bottom-of-Atmosphere reflectance [0.0, 1.0].
          </p>
        </div>
        <div class="bg-surface border border-borderOuter rounded-[4px] p-2.5">
          <div class="font-medium text-ink mb-1 flex items-center justify-between">
            <span>ISRO Resourcesat-2A LISS-IV (Unvalidated Domain)</span>
            <span class="font-mono text-[11px] text-warnInk bg-warnBg px-1.5 py-0.5 rounded">5.8m GSD • NIR-R-G FCC</span>
          </div>
          <p class="text-slate font-mono text-[11px] leading-relaxed">
            B2 (Green: 520–590 nm) • B3 (Red: 620–680 nm) • B4 (NIR: 770–860 nm) — Lacks optical Blue channel (B02).
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- 3. MAIN WORKSPACE LAYOUT (380px Left Inspector + Fluid Right Canvas)  -->
  <!-- ===================================================================== -->
  <main class="flex-1 max-w-[1600px] w-full mx-auto px-6 py-4 flex flex-col lg:flex-row gap-5 items-start">

    <!-- =================================================================== -->
    <!-- LEFT PANEL: MODEL AND INPUT CONFIGURATION (Fixed 380px on Desktop)  -->
    <!-- =================================================================== -->
    <aside class="w-full lg:w-[380px] shrink-0 flex flex-col gap-4">

      <!-- SECTION 1: Neural Model Architecture Card -->
      <div class="bg-surface border border-borderOuter rounded-[8px] overflow-hidden shadow-sm">
        <div class="px-5 py-4 border-b border-borderInner">
          <div class="flex items-center justify-between gap-2 mb-3">
            <h2 class="font-display text-[14px] font-semibold text-ink">
              Neural Model Architecture
            </h2>
            <span id="modelParamBadge" class="font-mono text-[11px] font-medium text-slate bg-subtle border border-borderInner px-2 py-0.5 rounded-[4px] tabular-nums">
              7.76M PARAMS (150 EP)
            </span>
          </div>

          <label for="modelSelect" class="block text-[12px] font-medium text-slate mb-1.5">
            Active checkpoint
          </label>
          <div class="relative">
            <select id="modelSelect" class="g-select w-full appearance-none bg-surface border border-borderOuter hover:border-slate rounded-[6px] pl-3 pr-8 py-2 text-[12px] font-mono text-ink transition cursor-pointer">
              <option value="">Loading available checkpoints...</option>
            </select>
            <svg class="w-4 h-4 text-slate absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
            </svg>
          </div>
          <p id="modelStatusMsg" class="text-[12px] text-slate mt-2 min-h-[1.2rem]"></p>
        </div>
      </div>

      <!-- SECTION 2: Input Selection & File Configuration Card -->
      <div class="bg-surface border border-borderOuter rounded-[8px] overflow-hidden shadow-sm">
        <div class="px-5 py-4">
          <div class="flex items-center justify-between mb-2">
            <h2 class="font-display text-[14px] font-semibold text-ink">
              Input Selection
            </h2>
            <span class="font-mono text-[11px] text-muted">Sentinel-2 RGB</span>
          </div>

          <!-- Google Material Tabs: Benchmark Presets | Upload Image -->
          <div class="flex border-b border-borderOuter mb-4 -mx-5 px-5">
            <button id="tabBtnPresets" type="button" class="pb-2.5 px-3 text-[13px] font-medium text-gblue border-b-2 border-gblue transition">
              Benchmark Presets
            </button>
            <button id="tabBtnUpload" type="button" class="pb-2.5 px-3 text-[13px] font-medium text-slate hover:text-ink border-b-2 border-transparent transition">
              Upload Image
            </button>
          </div>

          <!-- TAB PANEL A: Benchmark Presets (Active by Default) -->
          <div id="tabPanelPresets" class="space-y-3">
            <div>
              <label for="presetSelect" class="block text-[12px] font-medium text-slate mb-1.5">
                Select a paired Sentinel-2 SEN12MS-CR validation tile:
              </label>
              <div class="relative">
                <select id="presetSelect" class="g-select w-full appearance-none bg-surface border border-borderOuter hover:border-slate rounded-[6px] pl-3 pr-8 py-2 text-[12px] font-mono text-ink transition cursor-pointer">
                  <option value="">Loading presets...</option>
                </select>
                <svg class="w-4 h-4 text-slate absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
                </svg>
              </div>
            </div>

            <!-- Dynamic Preset Tile Rows -->
            <div id="presetTilesContainer" class="space-y-2">
              <!-- Populated dynamically via JS -->
            </div>

            <div id="presetDesc" class="text-[12px] text-slate bg-subtle border border-borderInner rounded-[6px] p-3 min-h-[2.8rem] leading-relaxed">
              Select a benchmark sample to see details…
            </div>

            <button id="btnRunPreset" type="button" class="w-full h-[40px] rounded-[6px] bg-gblue hover:bg-gblueHover active:bg-gblueActive text-surface font-display text-[13px] font-medium flex items-center justify-center gap-2 transition shadow-sm">
              <svg id="spinnerPreset" class="hidden w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" viewBox="0 0 20 20"></svg>
              <svg class="w-4 h-4 fill-current" viewBox="0 0 20 20">
                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
              </svg>
              <span id="runPresetBtnText">Run Cloud Removal</span>
            </button>
          </div>

          <!-- TAB PANEL B: Upload Image -->
          <div id="tabPanelUpload" class="hidden space-y-3">
            
            <!-- File Input 1: Cloudy Input Image (Required) -->
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <span class="text-[12px] font-medium text-inkBody">Cloudy Input Image</span>
                <span class="text-[11px] font-medium text-gblueActive bg-gblueTint px-1.5 py-0.5 rounded-[4px]">Required</span>
              </div>
              <div class="flex items-center gap-3 p-2.5 rounded-[6px] bg-surface border border-borderOuter hover:border-[#BDC1C6] transition">
                <canvas id="thumbCloudy" width="64" height="64" class="w-11 h-11 rounded-[4px] border border-borderOuter shrink-0 bg-subtle"></canvas>
                <div class="min-w-0 flex-1">
                  <div class="flex items-center gap-1.5">
                    <span id="cloudyFileName" class="font-mono text-[12px] font-medium text-ink truncate">Choose cloudy image...</span>
                    <span id="cloudyFileSize" class="font-mono text-[11px] text-muted tabular-nums"></span>
                  </div>
                  <div class="text-[11px] text-slate mt-0.5">
                    3-channel optical RGB • 512 × 512 px max
                  </div>
                </div>
                <label class="cursor-pointer shrink-0 text-[12px] font-medium text-gblue hover:bg-gblueTint px-2.5 py-1 rounded-[4px] transition">
                  Browse
                  <input id="cloudyFileInput" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" class="hidden" />
                </label>
              </div>
            </div>

            <!-- File Input 2: Clear Reference Scene (Optional) -->
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <span class="text-[12px] font-medium text-inkBody">Clear Reference Scene</span>
                <span class="text-[11px] text-muted">Optional</span>
              </div>
              <div class="flex items-center gap-3 p-2.5 rounded-[6px] bg-surface border border-borderOuter hover:border-[#BDC1C6] transition">
                <canvas id="thumbClear" width="64" height="64" class="w-11 h-11 rounded-[4px] border border-borderOuter shrink-0 bg-subtle"></canvas>
                <div class="min-w-0 flex-1">
                  <div class="flex items-center gap-1.5">
                    <span id="clearFileName" class="font-mono text-[12px] font-medium text-ink truncate">Optional clear target...</span>
                    <span id="clearFileSize" class="font-mono text-[11px] text-muted tabular-nums"></span>
                  </div>
                  <div class="text-[11px] text-slate mt-0.5">
                    For MAE and PSNR evaluation
                  </div>
                </div>
                <label class="cursor-pointer shrink-0 text-[12px] font-medium text-gblue hover:bg-gblueTint px-2.5 py-1 rounded-[4px] transition">
                  Browse
                  <input id="clearFileInput" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" class="hidden" />
                </label>
              </div>
            </div>

            <!-- Format Information Specification Well -->
            <div class="mt-3.5 p-3 rounded-[6px] bg-subtle border border-borderInner text-[12px] text-slate leading-relaxed">
              <div class="flex items-center justify-between mb-1">
                <span class="font-medium text-inkBody">Supported formats</span>
                <button id="testSingleBandBtn" type="button" class="text-[11px] font-medium text-gblue hover:underline">
                  Test 1-band raster rejection
                </button>
              </div>
              <p class="text-[11.5px] text-slate">
                Supports <span class="font-medium text-inkBody">13-band Sentinel-2 GeoTIFF</span>, <span class="font-medium text-inkBody">3-band RGB GeoTIFF</span>, and <span class="font-medium text-inkBody">standard 3-channel RGB images</span>. Single-band rasters are rejected.
              </p>
              <!-- Validation Callout for Single-Band Rejection Demo -->
              <div id="singleBandAlert" class="hidden mt-2 p-2 rounded-[4px] bg-[#FCE8E6] border border-[#FAD2CF] text-[#C5221F] font-mono text-[11px]">
                Rejected input <span class="font-semibold">dem_single_band.tif</span> (shape: 1 × 512 × 512). Model requires 3 optical channels (B04, B03, B02). Isolated LISS-IV single-band rasters (BAND2/3/4) lack cross-sensor calibration.
              </div>
            </div>

            <!-- Action Button: [Upload & Run Model] -->
            <button id="btnRunUpload" type="button" class="w-full h-[40px] rounded-[6px] bg-gblue hover:bg-gblueHover active:bg-gblueActive text-surface font-display text-[13px] font-medium flex items-center justify-center gap-2 transition shadow-sm">
              <svg id="spinnerUpload" class="hidden w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" viewBox="0 0 20 20"></svg>
              <svg class="w-4 h-4 fill-current" viewBox="0 0 20 20">
                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
              </svg>
              <span id="runUploadBtnText">Upload & Run Model</span>
            </button>
          </div>

        </div>
      </div>

      <!-- SECTION 3: Inference Metadata (Google Cloud Property Table) -->
      <div id="metaCard" class="bg-surface border border-borderOuter rounded-[8px] overflow-hidden shadow-sm">
        <div class="px-5 py-4">
          <div class="flex items-center justify-between mb-2.5">
            <h2 class="font-display text-[14px] font-semibold text-ink">
              Inference Metadata
            </h2>
            <span class="font-mono text-[11px] text-muted">Tensor Telemetry</span>
          </div>

          <dl class="divide-y divide-borderInner border-t border-b border-borderInner text-[12px]">
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate">Execution Time</dt>
              <dd id="metaExecTime" class="font-mono font-medium text-ink tabular-nums">-- ms CPU</dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate shrink-0">Data Source</dt>
              <dd id="metaDataSource" class="font-mono text-ink text-right truncate max-w-[210px]">--</dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate">Type / Label</dt>
              <dd id="metaTypeLabel" class="text-right">
                <span class="font-mono text-[11px] font-medium px-2 py-0.5 rounded-[4px] bg-gblueTint text-gblueActive">
                  REAL SCENE
                </span>
              </dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate shrink-0">Image Dimensions</dt>
              <dd id="metaDimensions" class="font-mono text-ink tabular-nums text-right">--</dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate">CRS</dt>
              <dd id="metaCRS" class="font-mono text-ink text-right truncate max-w-[200px]">--</dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate">Resolution</dt>
              <dd id="metaResolution" class="font-mono text-ink text-right">--</dd>
            </div>
            <div class="py-2 flex items-center justify-between gap-3">
              <dt class="text-slate shrink-0">Model Checkpoint</dt>
              <dd id="metaCheckpoint" class="font-mono text-inkBody bg-subtle px-1.5 py-0.5 rounded-[4px] truncate max-w-[205px] text-right" title="checkpoints/best_unet.pt">
                checkpoints/best_unet.pt
              </dd>
            </div>
          </dl>
        </div>
      </div>

    </aside>

    <!-- =================================================================== -->
    <!-- RIGHT PANEL: RECONSTRUCTION WORKSPACE AND EVALUATION                -->
    <!-- =================================================================== -->
    <section class="flex-1 min-w-0 w-full flex flex-col gap-5">

      <!-- WORKSPACE CARD: Controls + Visual Comparison Display -->
      <div class="bg-surface border border-borderOuter rounded-[8px] overflow-hidden shadow-sm">
        
        <!-- Workspace Controls Bar -->
        <div class="px-5 py-3.5 border-b border-borderInner flex flex-wrap items-center justify-between gap-3 bg-surface">
          
          <!-- Left: Workspace Heading & Synchronized Reflectance Readout -->
          <div class="flex flex-wrap items-center gap-3">
            <h2 class="font-display text-[15px] font-semibold text-ink">
              Reconstruction Workspace
            </h2>
            <div class="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-[4px] bg-subtle border border-borderInner font-mono text-[11px] text-slate tabular-nums">
              <span>Cursor <strong id="cursorCoord" class="text-ink font-medium">(256, 256)</strong></span>
              <span class="text-borderOuter">|</span>
              <span>Reflectance ρ: <strong id="cursorReflectance" class="text-ink font-medium">R:-- G:-- B:--</strong></span>
            </div>
          </div>

          <!-- Right: View Mode Segmented Toggle & Download Options -->
          <div class="flex flex-wrap items-center gap-2.5">
            
            <!-- Segmented View Toggle: [⚡ Split Slider] and [⊞ Side-by-Side] -->
            <div class="inline-flex rounded-[6px] border border-borderOuter bg-subtle p-0.5" role="group">
              <button id="btnViewSplit" type="button" class="px-3 py-1 rounded-[4px] text-[12px] font-medium bg-gblueTint text-gblueActive border border-[#D2E3FC] transition flex items-center gap-1.5">
                <span>⚡ Split Slider</span>
              </button>
              <button id="btnViewSideBySide" type="button" class="px-3 py-1 rounded-[4px] text-[12px] font-medium text-slate hover:text-ink transition flex items-center gap-1.5">
                <span>⊞ Side-by-Side</span>
              </button>
            </div>

            <div class="h-4 w-[1px] bg-borderOuter hidden sm:block"></div>

            <!-- Download Options -->
            <button id="btnDownloadTif" type="button" disabled class="px-3 py-1.5 rounded-[6px] bg-surface hover:bg-canvas border border-borderOuter text-[12px] font-medium text-inkBody transition inline-flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed">
              <span>💾 Download GeoTIFF (.tif)</span>
            </button>
            <button id="btnDownloadPng" type="button" disabled class="px-3 py-1.5 rounded-[6px] bg-surface hover:bg-canvas border border-borderOuter text-[12px] font-medium text-inkBody transition inline-flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed">
              <span>🖼️ Download PNG (.png)</span>
            </button>
          </div>
        </div>

        <!-- Visual Comparison Display Area -->
        <div class="p-5 bg-canvas relative min-h-[360px]">
          
          <!-- Deterministic Inference Progress Overlay -->
          <div id="inferenceLoadingOverlay" class="hidden absolute inset-0 z-20 bg-surface/85 backdrop-blur-[1px] flex flex-col items-center justify-center gap-2.5 rounded-b-[8px]">
            <div class="w-7 h-7 rounded-full border-2 border-borderOuter border-t-gblue animate-spin"></div>
            <div class="font-display text-[13px] font-medium text-ink">
              Running deterministic U-Net surface reflectance inference…
            </div>
            <div class="font-mono text-[11px] text-slate">
              Tensor [1, 3, 512, 512] • Normalizing BOA reflectance range [0.0, 1.0]
            </div>
          </div>

          <!-- VIEW 1: INTERACTIVE SPLIT SLIDER (Active by default) -->
          <div id="splitSliderView" class="flex flex-col items-center gap-3">
            
            <!-- Split Slider Controls Header -->
            <div class="w-full max-w-[560px] flex flex-wrap items-center justify-between gap-2 bg-surface border border-borderOuter rounded-[6px] px-3.5 py-2">
              <div class="flex items-center gap-1.5 text-[12px]">
                <span class="text-slate mr-1">Compare:</span>
                <button id="btnSplitPairCloudyPred" type="button" class="px-2.5 py-1 rounded-[4px] font-medium bg-gblueTint text-gblueActive">
                  Cloudy Input vs U-Net Prediction
                </button>
                <button id="btnSplitPairPredClear" type="button" class="px-2.5 py-1 rounded-[4px] font-medium text-slate hover:text-ink">
                  U-Net Prediction vs Ground Truth
                </button>
              </div>
              <span class="font-mono text-[11px] text-slate tabular-nums">
                Split: <strong id="splitPctReadout" class="text-ink">50%</strong>
              </span>
            </div>

            <!-- Interactive 512x512 Split Stage -->
            <div id="splitStage" class="relative w-full max-w-[560px] aspect-square bg-surface border border-borderOuter rounded-[6px] overflow-hidden select-none">
              <!-- Right Layer -->
              <canvas id="canvasSplitRight" width="512" height="512" class="w-full h-full block object-contain"></canvas>
              <!-- Clipped Left Layer -->
              <div id="splitLeftClip" class="absolute inset-0 overflow-hidden" style="clip-path: inset(0 50% 0 0);">
                <canvas id="canvasSplitLeft" width="512" height="512" class="w-full h-full block object-contain"></canvas>
              </div>
              <!-- Vertical Handle -->
              <div id="splitHandleLine" class="absolute top-0 bottom-0 w-[2px] bg-gblue cursor-ew-resize" style="left: 50%;">
                <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-surface border-2 border-gblue text-gblue flex items-center justify-center shadow-md">
                  <svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M13.28 7.78l3.22 2.22-3.22 2.22v-4.44zm-6.56 0v4.44L3.5 10l3.22-2.22z" clip-rule="evenodd" />
                  </svg>
                </div>
              </div>
              <!-- Corner Labels -->
              <div id="splitLeftTag" class="absolute top-3 left-3 bg-surface/95 border border-borderOuter px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium text-ink shadow-sm">
                Cloudy Input (512 × 512)
              </div>
              <div id="splitRightTag" class="absolute top-3 right-3 bg-gblueTint/95 border border-[#D2E3FC] px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium text-gblueActive shadow-sm">
                U-Net Prediction
              </div>
            </div>

            <!-- Range slider for accessibility -->
            <input type="range" min="0" max="100" value="50" id="splitRangeInput" class="w-full max-w-[560px] mx-auto block mt-1 accent-gblue cursor-pointer">
            <div class="flex justify-between max-w-[560px] w-full text-[11px] text-slate font-mono">
              <span id="sliderRangeLeftLabel">◀ Cloudy Input</span>
              <span id="sliderRangeRightLabel">U-Net Prediction ▶</span>
            </div>
          </div>

          <!-- VIEW 2: SIDE-BY-SIDE (3 Equal-Width Image Containers) -->
          <div id="sideBySideView" class="hidden grid-cols-1 md:grid-cols-3 gap-4">
            
            <!-- Image Container 1: Cloudy Input -->
            <div class="bg-surface border border-borderOuter rounded-[6px] overflow-hidden flex flex-col shadow-sm">
              <div class="px-3.5 py-2.5 bg-canvas border-b border-borderInner flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="font-display text-[13px] font-semibold text-ink">Cloudy Input</span>
                </div>
                <span id="dimsCloudy" class="font-mono text-[11px] text-muted tabular-nums">512 × 512 px</span>
              </div>
              <div class="relative aspect-square w-full bg-subtle overflow-hidden">
                <canvas id="canvasCloudy" width="512" height="512" class="eo-canvas w-full h-full object-contain"></canvas>
                <div id="reticleCloudy" class="pointer-events-none hidden absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 border-[1.5px] border-gblue rounded-full bg-white/20"></div>
              </div>
              <div class="px-3.5 py-2 bg-surface border-t border-borderInner flex items-center justify-between font-mono text-[11px] text-slate tabular-nums">
                <span id="labelCloudyFile" class="truncate max-w-[150px]">Input scene</span>
                <span id="readoutCloudy">ρ: --, --, --</span>
              </div>
            </div>

            <!-- Image Container 2: U-Net Prediction -->
            <div class="bg-surface border border-gblue rounded-[6px] overflow-hidden flex flex-col ring-1 ring-gblue/15 shadow-sm">
              <div class="px-3.5 py-2.5 bg-gblueTint/50 border-b border-[#D2E3FC] flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span id="predHeaderTitle" class="font-display text-[13px] font-semibold text-gblueActive">U-Net Prediction</span>
                </div>
                <span id="dimsPred" class="font-mono text-[11px] font-medium text-gblueActive tabular-nums">512 × 512 px</span>
              </div>
              <div class="relative aspect-square w-full bg-subtle overflow-hidden">
                <canvas id="canvasPred" width="512" height="512" class="eo-canvas w-full h-full object-contain"></canvas>
                <div id="reticlePred" class="pointer-events-none hidden absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 border-[1.5px] border-gblue rounded-full bg-white/20"></div>
              </div>
              <div class="px-3.5 py-2 bg-surface border-t border-borderInner flex items-center justify-between font-mono text-[11px] text-slate tabular-nums">
                <span id="predFooterStatus" class="text-gblueActive font-medium">Reconstructed output</span>
                <span id="readoutPred">ρ: --, --, --</span>
              </div>
            </div>

            <!-- Image Container 3: Ground Truth Clear -->
            <div id="cardClear" class="bg-surface border border-borderOuter rounded-[6px] overflow-hidden flex flex-col shadow-sm">
              <div class="px-3.5 py-2.5 bg-canvas border-b border-borderInner flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="font-display text-[13px] font-semibold text-ink">Ground Truth Clear</span>
                </div>
                <span id="dimsClear" class="font-mono text-[11px] text-muted tabular-nums">512 × 512 px</span>
              </div>
              <div class="relative aspect-square w-full bg-subtle overflow-hidden">
                <canvas id="canvasClear" width="512" height="512" class="eo-canvas w-full h-full object-contain"></canvas>
                <div id="reticleClear" class="pointer-events-none hidden absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 border-[1.5px] border-gblue rounded-full bg-white/20"></div>
              </div>
              <div class="px-3.5 py-2 bg-surface border-t border-borderInner flex items-center justify-between font-mono text-[11px] text-slate tabular-nums">
                <span>Reference target scene</span>
                <span id="readoutClear">ρ: --, --, --</span>
              </div>
            </div>

          </div>

        </div>
      </div>

      <!-- ================================================================= -->
      <!-- QUANTITATIVE ACCURACY EVALUATION CARD                             -->
      <!-- ================================================================= -->
      <div id="accuracyCard" class="bg-surface border border-borderOuter rounded-[8px] px-5 py-4 shadow-sm">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 pb-3.5 mb-4 border-b border-borderInner">
          <div>
            <h2 class="font-display text-[15px] font-semibold text-ink">
              Quantitative Accuracy Evaluation
            </h2>
            <p class="text-[12px] text-slate mt-0.5">
              Computed against raw physical surface reflectance values in the range <span class="font-mono text-ink font-medium">[0.0, 1.0]</span>
            </p>
          </div>
          <span id="gtBadge" class="font-mono text-[11px] font-medium text-successInk bg-successBg border border-successBorder px-2.5 py-1 rounded-[4px] self-start sm:self-center">
            Ground Truth Present
          </span>
        </div>

        <!-- Section A: Metrics Available (when ground truth present) -->
        <div id="gtAvailableSection" class="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          <!-- Metric Panel 1: Mean Absolute Error (MAE) -->
          <div class="bg-canvas border border-borderInner rounded-[6px] p-4 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between gap-2 mb-2">
                <span class="text-[12px] font-medium text-slate">
                  Mean Absolute Error (MAE)
                </span>
                <span class="font-mono text-[11px] text-muted">Lower is better ↓</span>
              </div>

              <!-- Metric Comparison Line -->
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-1 my-1.5 font-mono tabular-nums">
                <span class="text-[13px] text-inkBody">
                  Cloudy Base MAE: <strong id="metricCloudyMae" class="font-semibold text-ink">--</strong>
                </span>
                <span class="text-muted text-[12px]">vs</span>
                <span class="text-[13px] text-inkBody">
                  U-Net Pred MAE: <strong id="metricPredMae" class="font-semibold text-ink">--</strong>
                </span>
                <span id="metricDeltaMae" class="font-mono text-[11px] font-medium px-1.5 py-0.5 rounded-[4px] bg-subtle border border-borderOuter text-inkBody">
                  --
                </span>
              </div>
            </div>

            <!-- Comparative Scale Bars -->
            <div class="mt-3 pt-3 border-t border-borderInner space-y-2 font-mono text-[11px] tabular-nums">
              <div>
                <div class="flex justify-between text-slate mb-1">
                  <span>Cloudy Base MAE</span>
                  <span id="barTextCloudyMae" class="text-ink">--</span>
                </div>
                <div class="w-full h-1.5 bg-borderInner rounded-full overflow-hidden">
                  <div id="barFillCloudyMae" class="h-full bg-slate rounded-full transition-all duration-300" style="width: 50%;"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-slate mb-1">
                  <span>U-Net Pred MAE</span>
                  <span id="barTextPredMae" class="text-ink font-medium">--</span>
                </div>
                <div class="w-full h-1.5 bg-borderInner rounded-full overflow-hidden">
                  <div id="barFillPredMae" class="h-full bg-gblue rounded-full transition-all duration-300" style="width: 50%;"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Metric Panel 2: Peak Signal-to-Noise Ratio (PSNR) -->
          <div class="bg-canvas border border-borderInner rounded-[6px] p-4 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between gap-2 mb-2">
                <span class="text-[12px] font-medium text-slate">
                  Peak Signal-to-Noise Ratio (PSNR)
                </span>
                <span class="font-mono text-[11px] text-gblueActive">Higher is better ↑</span>
              </div>

              <!-- Metric Comparison Line -->
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-1 my-1.5 font-mono tabular-nums">
                <span class="text-[13px] text-inkBody">
                  Cloudy Base PSNR: <strong id="metricCloudyPsnr" class="font-semibold text-ink">-- dB</strong>
                </span>
                <span class="text-muted text-[12px]">vs</span>
                <span class="text-[13px] text-inkBody">
                  U-Net Pred PSNR: <strong id="metricPredPsnr" class="font-semibold text-gblueActive">-- dB</strong>
                </span>
                <span id="metricDeltaPsnr" class="font-mono text-[11px] font-medium px-1.5 py-0.5 rounded-[4px] bg-gblueTint border border-[#D2E3FC] text-gblueActive">
                  --
                </span>
              </div>
            </div>

            <!-- Comparative Scale Bars -->
            <div class="mt-3 pt-3 border-t border-borderInner space-y-2 font-mono text-[11px] tabular-nums">
              <div>
                <div class="flex justify-between text-slate mb-1">
                  <span>Cloudy Base PSNR</span>
                  <span id="barTextCloudyPsnr" class="text-ink">-- dB</span>
                </div>
                <div class="w-full h-1.5 bg-borderInner rounded-full overflow-hidden">
                  <div id="barFillCloudyPsnr" class="h-full bg-slate rounded-full transition-all duration-300" style="width: 50%;"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-slate mb-1">
                  <span>U-Net Pred PSNR</span>
                  <span id="barTextPredPsnr" class="text-gblueActive font-medium">-- dB</span>
                </div>
                <div class="w-full h-1.5 bg-borderInner rounded-full overflow-hidden">
                  <div id="barFillPredPsnr" class="h-full bg-gblue rounded-full transition-all duration-300" style="width: 65%;"></div>
                </div>
              </div>
            </div>
          </div>

        </div>

        <!-- Section B: No Ground Truth Notice (when cloudy-only upload) -->
        <div id="gtUnavailableSection" class="hidden text-[13px] text-slate py-3 px-4 bg-subtle rounded-[6px] border border-borderInner">
          <div class="flex items-start gap-2.5">
            <svg class="w-4 h-4 text-slate shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" />
            </svg>
            <span>
              <strong class="text-ink font-medium">No reference clear scene provided.</strong> Quantitative metrics (MAE and PSNR) are strictly omitted because evaluation requires a true cloud-free paired ground-truth image.
            </span>
          </div>
        </div>

      </div>

    </section>

  </main>

  <!-- Google Material Snackbar Notification -->
  <div id="snackbar" class="fixed bottom-5 right-6 z-50 hidden items-center gap-3 bg-[#323232] text-white px-4 py-2.5 rounded-[4px] shadow-lg text-[13px] transition-all">
    <span id="snackbarText">Inference completed.</span>
  </div>

  <!-- ===================================================================== -->
  <!-- APPLICATION LOGIC & FULL BACKEND API INTEGRATION                      -->
  <!-- ===================================================================== -->
  <script>
    // -----------------------------------------------------------------------
    // 1. STATE MANAGEMENT
    // -----------------------------------------------------------------------
    let currentSessionId = null;
    let presetsData = [];
    let activePresetId = 'sen12mscr_val_0';
    let splitPairMode = 'cloudy_pred'; // 'cloudy_pred' | 'pred_clear'
    let hasGroundTruth = false;

    const SIZE = 512;
    let bufCloudy = new Uint8ClampedArray(SIZE * SIZE * 4);
    let bufPred   = new Uint8ClampedArray(SIZE * SIZE * 4);
    let bufClear  = new Uint8ClampedArray(SIZE * SIZE * 4);

    // -----------------------------------------------------------------------
    // 2. INITIALIZATION ON WINDOW LOAD
    // -----------------------------------------------------------------------
    window.addEventListener('DOMContentLoaded', async () => {
      initCanvasInspection();
      initSliderDrag();
      initTabs();
      initBandDrawer();
      initSingleBandTest();
      initDownloadButtons();

      await Promise.all([loadModelsList(), loadPresets()]);

      // Automatically execute benchmark preset #0 on initial load for instant visualization
      if (presetsData.length > 0) {
        runPresetInference(presetsData[0].id);
      }
    });

    // -----------------------------------------------------------------------
    // 3. MODEL CHECKPOINTS (GET /api/ui/models & POST /api/ui/select-model)
    // -----------------------------------------------------------------------
    async function loadModelsList() {
      try {
        const resp = await fetch('/api/ui/models');
        if (!resp.ok) return;
        const data = await resp.json();
        const sel = document.getElementById('modelSelect');
        if (!sel || !data.models) return;
        sel.innerHTML = '';
        data.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = `${m.name} (${m.size_mb} MB)`;
          const activeNorm = (data.active_model || '').toLowerCase();
          const idNorm = (m.id || '').toLowerCase();
          if (activeNorm.includes(idNorm) || idNorm.includes(activeNorm)) {
            opt.selected = true;
          }
          sel.appendChild(opt);
        });
        updateModelParamBadge(data.active_model);
      } catch (err) {
        console.error('Failed to load models list', err);
      }
    }

    function updateModelParamBadge(checkpointPath) {
      const badge = document.getElementById('modelParamBadge');
      if (!badge) return;
      const chk = (checkpointPath || '').toLowerCase();
      if (chk.includes('best_unet') || chk.includes('standard_unet')) {
        badge.innerText = '7.76M PARAMS (150 EP)';
      } else if (chk.includes('best_compact_unet_v2') || chk.includes('v2')) {
        badge.innerText = '1.94M PARAMS (RECORD)';
      } else if (chk.includes('rice2')) {
        badge.innerText = '1.94M PARAMS (RICE2)';
      } else if (chk.includes('cgan') && chk.includes('fcc')) {
        badge.innerText = '11.38M GENERATOR (FCC)';
      } else if (chk.includes('cgan')) {
        badge.innerText = '11.38M GENERATOR';
      } else if (chk.includes('compact')) {
        badge.innerText = '1.94M PARAMS';
      } else {
        badge.innerText = 'ACTIVE MODEL';
      }
    }

    document.getElementById('modelSelect').addEventListener('change', onModelChange);

    async function onModelChange() {
      const sel = document.getElementById('modelSelect');
      const modelId = sel.value;
      if (!modelId) return;

      const statusMsg = document.getElementById('modelStatusMsg');
      statusMsg.style.color = '#1A73E8';
      statusMsg.innerText = '⏳ Initializing model checkpoint...';
      sel.disabled = true;

      try {
        const formData = new FormData();
        formData.append('model_id', modelId);
        const resp = await fetch('/api/ui/select-model', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to switch model');

        statusMsg.style.color = '#137333';
        statusMsg.innerText = '✓ Active model switched successfully!';
        updateModelParamBadge(data.active_checkpoint);

        let label = 'Model: U-Net';
        const chk = (data.active_checkpoint || '').toLowerCase();
        let isCrossSensor = false;
        if (chk.includes('best_unet') || chk.includes('standard_unet')) {
          label = 'Model: Full U-Net (42.13 dB - Best)';
        } else if (chk.includes('best_compact_unet_v2') || chk.includes('v2')) {
          label = 'Model: U-Net v2 (41.09 dB - Record)';
        } else if (chk.includes('rice2')) {
          label = 'Model: RICE2 Compact U-Net';
        } else if (chk.includes('cgan') && chk.includes('fcc')) {
          label = 'Model: LISS-IV FCC cGAN (Experimental)';
          isCrossSensor = true;
        } else if (chk.includes('cgan')) {
          label = 'Model: Pix2Pix cGAN';
        } else if (chk.includes('compact')) {
          label = 'Model: Compact U-Net';
        }
        document.getElementById('serviceBadge').innerText = label;

        if (isCrossSensor) {
          const callout = document.getElementById('domainNoticeCallout');
          callout.classList.remove('callout-flash');
          void callout.offsetWidth;
          callout.classList.add('callout-flash');
          document.getElementById('bandSpecDrawer').classList.remove('hidden');
        }

        showSnackbar(`Model switched to ${label.replace('Model: ', '')}`);
        setTimeout(() => { statusMsg.innerText = ''; }, 3500);

        // Re-run inference with the newly active checkpoint
        const tabPanelPresets = document.getElementById('tabPanelPresets');
        if (!tabPanelPresets.classList.contains('hidden') && activePresetId) {
          runPresetInference(activePresetId);
        }
      } catch (err) {
        statusMsg.style.color = '#C5221F';
        statusMsg.innerText = `⚠️ Error: ${err.message}`;
      } finally {
        sel.disabled = false;
      }
    }

    // -----------------------------------------------------------------------
    // 4. BENCHMARK PRESETS (GET /api/ui/presets)
    // -----------------------------------------------------------------------
    async function loadPresets() {
      try {
        const resp = await fetch('/api/ui/presets');
        if (!resp.ok) throw new Error('Failed to fetch presets');
        presetsData = await resp.json();

        // Populate dropdown
        const sel = document.getElementById('presetSelect');
        sel.innerHTML = presetsData.map(p =>
          `<option value="${p.id}">[${p.category}] ${p.name}</option>`
        ).join('');

        // Populate rich preset tile buttons
        renderPresetTiles();
        onPresetChange();
      } catch (err) {
        console.error(err);
        document.getElementById('presetDesc').innerText = 'Failed to load presets: ' + err.message;
      }
    }

    function renderPresetTiles() {
      const container = document.getElementById('presetTilesContainer');
      container.innerHTML = '';
      presetsData.slice(0, 3).forEach((p, idx) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.preset = p.id;
        const isActive = (p.id === activePresetId);
        btn.className = isActive
          ? 'preset-tile w-full text-left p-2.5 rounded-[6px] bg-gblueTint border border-gblue flex items-center justify-between transition'
          : 'preset-tile w-full text-left p-2.5 rounded-[6px] bg-surface hover:bg-canvas border border-borderOuter flex items-center justify-between transition';

        btn.innerHTML = `
          <div>
            <div class="font-mono text-[12px] font-medium ${isActive ? 'text-gblueActive' : 'text-ink'}">${p.name}</div>
            <div class="font-mono text-[11px] text-slate mt-0.5">${p.category} • Sentinel-2 10m</div>
          </div>
          <span class="text-[11px] font-medium ${isActive ? 'text-gblueActive' : 'text-slate'}">${isActive ? 'Active' : 'Load'}</span>
        `;
        btn.addEventListener('click', () => {
          activePresetId = p.id;
          document.getElementById('presetSelect').value = p.id;
          updatePresetTileHighlighting();
          onPresetChange();
          runPresetInference(p.id);
        });
        container.appendChild(btn);
      });
    }

    function updatePresetTileHighlighting() {
      document.querySelectorAll('.preset-tile').forEach(b => {
        const match = (b.dataset.preset === activePresetId);
        b.className = match
          ? 'preset-tile w-full text-left p-2.5 rounded-[6px] bg-gblueTint border border-gblue flex items-center justify-between transition'
          : 'preset-tile w-full text-left p-2.5 rounded-[6px] bg-surface hover:bg-canvas border border-borderOuter flex items-center justify-between transition';
        const titleEl = b.querySelector('.font-mono.text-\\[12px\\]');
        if (titleEl) titleEl.className = match ? 'font-mono text-[12px] font-medium text-gblueActive' : 'font-mono text-[12px] font-medium text-ink';
        const statusEl = b.querySelector('span:last-child');
        if (statusEl) {
          statusEl.textContent = match ? 'Active' : 'Load';
          statusEl.className = match ? 'text-[11px] font-medium text-gblueActive' : 'text-[11px] font-medium text-slate';
        }
      });
    }

    document.getElementById('presetSelect').addEventListener('change', () => {
      activePresetId = document.getElementById('presetSelect').value;
      updatePresetTileHighlighting();
      onPresetChange();
    });

    function onPresetChange() {
      const selectedId = document.getElementById('presetSelect').value;
      const p = presetsData.find(item => item.id === selectedId);
      if (p) {
        document.getElementById('presetDesc').innerHTML =
          `<strong>${p.name}:</strong> ${p.description} <br><span class="font-mono text-[11px]">${p.is_synthetic ? '⚠️ Simulated Gaussian/Perlin turbulence' : '✓ Real SEN12MS-CR paired optical validation scene'}</span>`;
      }
    }

    // -----------------------------------------------------------------------
    // 5. INFERENCE EXECUTION (POST /api/ui/predict)
    // -----------------------------------------------------------------------
    document.getElementById('btnRunPreset').addEventListener('click', () => {
      const selId = document.getElementById('presetSelect').value;
      if (selId) runPresetInference(selId);
    });

    async function runPresetInference(presetId) {
      if (!presetId) return;
      activePresetId = presetId;
      setLoading(true, 'Preset');

      try {
        const formData = new FormData();
        formData.append('preset_id', presetId);

        const resp = await fetch('/api/ui/predict', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Inference failed');

        renderInferenceResult(data);
      } catch (err) {
        showSnackbar(`Error: ${err.message}`);
      } finally {
        setLoading(false, 'Preset');
      }
    }

    document.getElementById('btnRunUpload').addEventListener('click', runUploadInference);

    async function runUploadInference() {
      const cloudyInput = document.getElementById('cloudyFileInput');
      if (!cloudyInput.files || !cloudyInput.files[0]) {
        showSnackbar('Please select a cloudy input image file first.');
        return;
      }

      setLoading(true, 'Upload');
      try {
        const formData = new FormData();
        formData.append('file', cloudyInput.files[0]);

        const clearInput = document.getElementById('clearFileInput');
        if (clearInput.files && clearInput.files[0]) {
          formData.append('clear_file', clearInput.files[0]);
        }

        const resp = await fetch('/api/ui/predict', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Inference failed');

        renderInferenceResult(data);
      } catch (err) {
        showSnackbar(`Upload Error: ${err.message}`);
      } finally {
        setLoading(false, 'Upload');
      }
    }

    // -----------------------------------------------------------------------
    // 6. RENDER INFERENCE RESULTS & PAINT CANVASES
    // -----------------------------------------------------------------------
    function renderInferenceResult(data) {
      currentSessionId = data.session_id;
      hasGroundTruth = Boolean(data.has_ground_truth && data.clear_preview);

      // Metadata Table Sync
      document.getElementById('metaExecTime').textContent = `${data.latency_ms.toFixed(1)} ms CPU`;
      document.getElementById('metaDataSource').textContent = data.metadata.source || data.metadata.format || 'Standard Optical RGB';
      document.getElementById('metaDimensions').textContent = `${data.input_shape.join(' × ')} [C, H, W]`;
      document.getElementById('metaCRS').textContent = data.crs || 'None (Standard Optical RGB)';
      document.getElementById('metaResolution').textContent = data.metadata.spatial_resolution || 'Unreferenced (Pixel Space)';
      document.getElementById('metaCheckpoint').textContent = data.checkpoint || 'checkpoints/best_unet.pt';

      // Type / Label Badge
      const typeLabel = document.getElementById('metaTypeLabel');
      if (data.is_synthetic) {
        typeLabel.innerHTML = '<span class="font-mono text-[11px] font-medium px-2 py-0.5 rounded-[4px] bg-warnBg border border-warnBorder text-warnInk">SYNTHETIC SIM</span>';
      } else {
        typeLabel.innerHTML = '<span class="font-mono text-[11px] font-medium px-2 py-0.5 rounded-[4px] bg-gblueTint text-gblueActive">REAL SCENE</span>';
      }

      // Top Status Indicators
      const srcBadgeText = document.getElementById('sourceBadgeText');
      if (data.is_synthetic) {
        srcBadgeText.textContent = 'SYNTHETIC CLOUD SIM';
      } else {
        srcBadgeText.textContent = 'REAL OPTICAL S2';
      }

      // Model title sync
      if (data.checkpoint) {
        let label = 'Model: U-Net';
        const chk = data.checkpoint.toLowerCase();
        if (chk.includes('best_unet') || chk.includes('standard_unet')) {
          label = 'Model: Full U-Net (42.13 dB - Best)';
        } else if (chk.includes('best_compact_unet_v2') || chk.includes('v2')) {
          label = 'Model: U-Net v2 (41.09 dB - Record)';
        } else if (chk.includes('rice2')) {
          label = 'Model: RICE2 Compact U-Net';
        } else if (chk.includes('cgan') && chk.includes('fcc')) {
          label = 'Model: LISS-IV FCC cGAN';
        } else if (chk.includes('cgan')) {
          label = 'Model: Pix2Pix cGAN';
        } else if (chk.includes('compact')) {
          label = 'Model: Compact U-Net';
        }
        document.getElementById('serviceBadge').textContent = label;
        updateModelParamBadge(data.checkpoint);
      }

      // Enable download buttons
      document.getElementById('btnDownloadPng').disabled = false;
      const btnTif = document.getElementById('btnDownloadTif');
      if (data.is_geotiff) {
        btnTif.disabled = false;
        btnTif.title = 'Download full-precision 3-band GeoTIFF with CRS and Transform';
      } else {
        btnTif.disabled = true;
        btnTif.title = 'GeoTIFF export unavailable for standard PNG/JPG uploads without geospatial metadata';
      }

      // Accuracy Metrics Card
      const gtAvail = document.getElementById('gtAvailableSection');
      const gtUnavail = document.getElementById('gtUnavailableSection');
      const gtBadge = document.getElementById('gtBadge');
      const cardClear = document.getElementById('cardClear');
      const btnSplitPairPredClear = document.getElementById('btnSplitPairPredClear');

      if (data.has_ground_truth && data.metrics) {
        gtAvail.classList.remove('hidden');
        gtUnavail.classList.add('hidden');
        gtBadge.className = 'font-mono text-[11px] font-medium text-successInk bg-successBg border border-successBorder px-2.5 py-1 rounded-[4px] self-start sm:self-center';
        gtBadge.textContent = 'Ground Truth Present';

        const m = data.metrics;
        const dMae = m.delta_mae;
        const sMae = dMae >= 0 ? '+' : '';
        const dPsnr = m.delta_psnr;
        const sPsnr = dPsnr >= 0 ? '+' : '';

        document.getElementById('metricCloudyMae').textContent = m.input_mae.toFixed(4);
        document.getElementById('metricPredMae').textContent = m.pred_mae.toFixed(4);
        document.getElementById('metricDeltaMae').textContent = `(${sMae}${dMae.toFixed(4)})`;
        document.getElementById('barTextCloudyMae').textContent = m.input_mae.toFixed(4);
        document.getElementById('barTextPredMae').textContent = `${m.pred_mae.toFixed(4)} (${sMae}${dMae.toFixed(4)})`;

        // Scale bar widths
        const cloudyMaeW = Math.min(100, Math.max(10, (m.input_mae / 0.20) * 100));
        const predMaeW = Math.min(100, Math.max(10, (m.pred_mae / 0.20) * 100));
        document.getElementById('barFillCloudyMae').style.width = `${cloudyMaeW}%`;
        document.getElementById('barFillPredMae').style.width = `${predMaeW}%`;

        document.getElementById('metricCloudyPsnr').textContent = `${m.input_psnr.toFixed(2)} dB`;
        document.getElementById('metricPredPsnr').textContent = `${m.pred_psnr.toFixed(2)} dB`;
        document.getElementById('metricDeltaPsnr').textContent = `(${sPsnr}${dPsnr.toFixed(2)} dB)`;
        document.getElementById('barTextCloudyPsnr').textContent = `${m.input_psnr.toFixed(2)} dB`;
        document.getElementById('barTextPredPsnr').textContent = `${m.pred_psnr.toFixed(2)} dB (${sPsnr}${dPsnr.toFixed(2)} dB)`;

        const cloudyPsnrW = Math.min(100, Math.max(10, (m.input_psnr / 45.0) * 100));
        const predPsnrW = Math.min(100, Math.max(10, (m.pred_psnr / 45.0) * 100));
        document.getElementById('barFillCloudyPsnr').style.width = `${cloudyPsnrW}%`;
        document.getElementById('barFillPredPsnr').style.width = `${predPsnrW}%`;

        cardClear.classList.remove('hidden');
        btnSplitPairPredClear.disabled = false;
        btnSplitPairPredClear.classList.remove('opacity-40', 'cursor-not-allowed');
      } else {
        gtAvail.classList.add('hidden');
        gtUnavail.classList.remove('hidden');
        gtBadge.className = 'font-mono text-[11px] font-medium text-warnInk bg-warnBg border border-warnBorder px-2.5 py-1 rounded-[4px] self-start sm:self-center';
        gtBadge.textContent = 'Ground Truth: None';

        cardClear.classList.add('hidden');
        btnSplitPairPredClear.disabled = true;
        btnSplitPairPredClear.classList.add('opacity-40', 'cursor-not-allowed');
        if (splitPairMode === 'pred_clear') {
          switchSplitPair('cloudy_pred');
        }
      }

      // Draw all image previews onto the canvases
      loadPreviewsToCanvases(data);
      showSnackbar(`Inference complete (${data.latency_ms.toFixed(1)} ms)`);
    }

    function loadPreviewsToCanvases(data) {
      let loadedCount = 0;
      const totalToLoad = data.clear_preview ? 3 : 2;

      function onAnyLoaded() {
        loadedCount++;
        if (loadedCount >= totalToLoad) {
          paintSplitSliderCanvases();
        }
      }

      // 1. Cloudy preview
      loadImageData(data.cloudy_preview, 'canvasCloudy', bufCloudy, (c) => {
        // Thumbnail in left panel
        const thumb = document.getElementById('thumbCloudy');
        thumb.getContext('2d').drawImage(c, 0, 0, 64, 64);
        onAnyLoaded();
      });

      // 2. Prediction preview
      loadImageData(data.pred_preview, 'canvasPred', bufPred, () => {
        onAnyLoaded();
      });

      // 3. Clear preview (if ground truth available)
      if (data.clear_preview) {
        loadImageData(data.clear_preview, 'canvasClear', bufClear, (c) => {
          const thumb = document.getElementById('thumbClear');
          thumb.getContext('2d').drawImage(c, 0, 0, 64, 64);
          onAnyLoaded();
        });
      }
    }

    function loadImageData(dataUri, canvasId, bufferArray, callback) {
      if (!dataUri) return;
      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        canvas.width = SIZE;
        canvas.height = SIZE;
        ctx.drawImage(img, 0, 0, SIZE, SIZE);
        const imgData = ctx.getImageData(0, 0, SIZE, SIZE);
        bufferArray.set(imgData.data);
        if (callback) callback(canvas);
      };
      img.src = dataUri;
    }

    function paintSplitSliderCanvases() {
      const splitL = document.getElementById('canvasSplitLeft').getContext('2d');
      const splitR = document.getElementById('canvasSplitRight').getContext('2d');

      if (splitPairMode === 'cloudy_pred') {
        splitL.putImageData(new ImageData(bufCloudy, SIZE, SIZE), 0, 0);
        splitR.putImageData(new ImageData(bufPred, SIZE, SIZE), 0, 0);
      } else {
        splitL.putImageData(new ImageData(bufPred, SIZE, SIZE), 0, 0);
        splitR.putImageData(new ImageData(bufClear, SIZE, SIZE), 0, 0);
      }
    }

    // -----------------------------------------------------------------------
    // 7. SYNCHRONIZED PIXEL REFLECTANCE INSPECTION
    // -----------------------------------------------------------------------
    function initCanvasInspection() {
      const canvases = [
        document.getElementById('canvasCloudy'),
        document.getElementById('canvasPred'),
        document.getElementById('canvasClear')
      ];
      const reticles = [
        document.getElementById('reticleCloudy'),
        document.getElementById('reticlePred'),
        document.getElementById('reticleClear')
      ];

      canvases.forEach((canvas) => {
        canvas.addEventListener('mousemove', (e) => {
          const rect = canvas.getBoundingClientRect();
          const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          const px = Math.min(SIZE - 1, Math.floor(nx * SIZE));
          const py = Math.min(SIZE - 1, Math.floor(ny * SIZE));
          const idx = (py * SIZE + px) * 4;

          reticles.forEach((r) => {
            r.classList.remove('hidden');
            r.style.left = `${nx * 100}%`;
            r.style.top  = `${ny * 100}%`;
          });

          const cR = (bufCloudy[idx] / 255).toFixed(2);
          const cG = (bufCloudy[idx + 1] / 255).toFixed(2);
          const cB = (bufCloudy[idx + 2] / 255).toFixed(2);

          const pR = (bufPred[idx] / 255).toFixed(2);
          const pG = (bufPred[idx + 1] / 255).toFixed(2);
          const pB = (bufPred[idx + 2] / 255).toFixed(2);

          const gR = (bufClear[idx] / 255).toFixed(2);
          const gG = (bufClear[idx + 1] / 255).toFixed(2);
          const gB = (bufClear[idx + 2] / 255).toFixed(2);

          document.getElementById('readoutCloudy').textContent = `ρ: ${cR}, ${cG}, ${cB}`;
          document.getElementById('readoutPred').textContent   = `ρ: ${pR}, ${pG}, ${pB}`;
          document.getElementById('readoutClear').textContent  = `ρ: ${gR}, ${gG}, ${gB}`;

          document.getElementById('cursorCoord').textContent = `(${px}, ${py})`;
          document.getElementById('cursorReflectance').textContent = `R:${pR} G:${pG} B:${pB}`;
        });

        canvas.addEventListener('mouseleave', () => {
          reticles.forEach((r) => r.classList.add('hidden'));
        });
      });
    }

    // -----------------------------------------------------------------------
    // 8. VIEW MODE TOGGLE ([⚡ Split Slider] vs [⊞ Side-by-Side])
    // -----------------------------------------------------------------------
    const btnViewSplit      = document.getElementById('btnViewSplit');
    const btnViewSideBySide = document.getElementById('btnViewSideBySide');
    const sideBySideView    = document.getElementById('sideBySideView');
    const splitSliderView   = document.getElementById('splitSliderView');

    btnViewSplit.addEventListener('click', () => {
      sideBySideView.classList.add('hidden');
      sideBySideView.classList.remove('grid');
      splitSliderView.classList.remove('hidden');
      splitSliderView.classList.add('flex');
      btnViewSplit.className      = 'px-3 py-1 rounded-[4px] text-[12px] font-medium bg-gblueTint text-gblueActive border border-[#D2E3FC] transition flex items-center gap-1.5';
      btnViewSideBySide.className = 'px-3 py-1 rounded-[4px] text-[12px] font-medium text-slate hover:text-ink transition flex items-center gap-1.5';
      paintSplitSliderCanvases();
    });

    btnViewSideBySide.addEventListener('click', () => {
      splitSliderView.classList.add('hidden');
      splitSliderView.classList.remove('flex');
      sideBySideView.classList.remove('hidden');
      sideBySideView.classList.add('grid');
      btnViewSideBySide.className = 'px-3 py-1 rounded-[4px] text-[12px] font-medium bg-gblueTint text-gblueActive border border-[#D2E3FC] transition flex items-center gap-1.5';
      btnViewSplit.className      = 'px-3 py-1 rounded-[4px] text-[12px] font-medium text-slate hover:text-ink transition flex items-center gap-1.5';
    });

    // Split Slider Pair Mode Buttons
    const btnSplitPairCloudyPred = document.getElementById('btnSplitPairCloudyPred');
    const btnSplitPairPredClear  = document.getElementById('btnSplitPairPredClear');

    function switchSplitPair(mode) {
      splitPairMode = mode;
      if (mode === 'cloudy_pred') {
        btnSplitPairCloudyPred.className = 'px-2.5 py-1 rounded-[4px] font-medium bg-gblueTint text-gblueActive';
        btnSplitPairPredClear.className  = 'px-2.5 py-1 rounded-[4px] font-medium text-slate hover:text-ink';
        document.getElementById('splitLeftTag').textContent  = 'Cloudy Input (512 × 512)';
        document.getElementById('splitRightTag').textContent = 'U-Net Prediction';
        document.getElementById('sliderRangeLeftLabel').textContent = '◀ Cloudy Input';
        document.getElementById('sliderRangeRightLabel').textContent = 'U-Net Prediction ▶';
      } else {
        btnSplitPairPredClear.className  = 'px-2.5 py-1 rounded-[4px] font-medium bg-gblueTint text-gblueActive';
        btnSplitPairCloudyPred.className = 'px-2.5 py-1 rounded-[4px] font-medium text-slate hover:text-ink';
        document.getElementById('splitLeftTag').textContent  = 'U-Net Prediction (512 × 512)';
        document.getElementById('splitRightTag').textContent = 'Ground Truth Clear';
        document.getElementById('sliderRangeLeftLabel').textContent = '◀ U-Net Prediction';
        document.getElementById('sliderRangeRightLabel').textContent = 'Ground Truth Clear ▶';
      }
      paintSplitSliderCanvases();
    }

    btnSplitPairCloudyPred.addEventListener('click', () => switchSplitPair('cloudy_pred'));
    btnSplitPairPredClear.addEventListener('click', () => switchSplitPair('pred_clear'));

    // -----------------------------------------------------------------------
    // 9. SPLIT SLIDER DRAGGING & RANGE INPUT
    // -----------------------------------------------------------------------
    function setSplitPosition(pct) {
      pct = Math.max(0, Math.min(100, pct));
      document.getElementById('splitLeftClip').style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
      document.getElementById('splitHandleLine').style.left   = `${pct}%`;
      document.getElementById('splitPctReadout').textContent  = `${Math.round(pct)}%`;
      document.getElementById('splitRangeInput').value = pct;
    }

    function initSliderDrag() {
      const stage = document.getElementById('splitStage');
      let isDragging = false;

      function onMove(clientX) {
        const rect = stage.getBoundingClientRect();
        const pct = ((clientX - rect.left) / rect.width) * 100;
        setSplitPosition(pct);
      }

      stage.addEventListener('mousedown', (e) => {
        isDragging = true;
        onMove(e.clientX);
      });
      window.addEventListener('mousemove', (e) => {
        if (isDragging) onMove(e.clientX);
      });
      window.addEventListener('mouseup', () => {
        isDragging = false;
      });

      stage.addEventListener('touchstart', (e) => {
        if (e.touches.length > 0) {
          isDragging = true;
          onMove(e.touches[0].clientX);
        }
      }, { passive: true });
      window.addEventListener('touchmove', (e) => {
        if (isDragging && e.touches.length > 0) onMove(e.touches[0].clientX);
      }, { passive: true });
      window.addEventListener('touchend', () => {
        isDragging = false;
      });

      document.getElementById('splitRangeInput').addEventListener('input', (e) => {
        setSplitPosition(parseFloat(e.target.value));
      });
    }

    // -----------------------------------------------------------------------
    // 10. INPUT TABS & FILE UPLOAD HANDLERS
    // -----------------------------------------------------------------------
    function initTabs() {
      const tabBtnPresets   = document.getElementById('tabBtnPresets');
      const tabBtnUpload    = document.getElementById('tabBtnUpload');
      const tabPanelPresets = document.getElementById('tabPanelPresets');
      const tabPanelUpload  = document.getElementById('tabPanelUpload');

      tabBtnPresets.addEventListener('click', () => {
        tabPanelPresets.classList.remove('hidden');
        tabPanelUpload.classList.add('hidden');
        tabBtnPresets.className = 'pb-2.5 px-3 text-[13px] font-medium text-gblue border-b-2 border-gblue transition';
        tabBtnUpload.className  = 'pb-2.5 px-3 text-[13px] font-medium text-slate hover:text-ink border-b-2 border-transparent transition';
      });

      tabBtnUpload.addEventListener('click', () => {
        tabPanelUpload.classList.remove('hidden');
        tabPanelPresets.classList.add('hidden');
        tabBtnUpload.className  = 'pb-2.5 px-3 text-[13px] font-medium text-gblue border-b-2 border-gblue transition';
        tabBtnPresets.className = 'pb-2.5 px-3 text-[13px] font-medium text-slate hover:text-ink border-b-2 border-transparent transition';
      });

      // Bind file pickers
      bindFileInput(document.getElementById('cloudyFileInput'), true);
      bindFileInput(document.getElementById('clearFileInput'), false);
    }

    function bindFileInput(inputEl, isCloudy) {
      inputEl.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const sizeStr = `(${(file.size / 1024).toFixed(1)} KB)`;
        if (isCloudy) {
          document.getElementById('cloudyFileName').textContent = file.name;
          document.getElementById('cloudyFileSize').textContent = sizeStr;
          document.getElementById('labelCloudyFile').textContent = file.name;
        } else {
          document.getElementById('clearFileName').textContent = file.name;
          document.getElementById('clearFileSize').textContent = sizeStr;
        }

        const reader = new FileReader();
        reader.onload = (ev) => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.getElementById(isCloudy ? 'thumbCloudy' : 'thumbClear');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, 64, 64);
            ctx.drawImage(img, 0, 0, 64, 64);
            showSnackbar(`Selected ${file.name}`);
          };
          img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
      });
    }

    // -----------------------------------------------------------------------
    // 11. AUXILIARY TOGGLES (Band Spec & Single-Band Rejection Demo)
    // -----------------------------------------------------------------------
    function initBandDrawer() {
      document.getElementById('toggleBandSpecBtn').addEventListener('click', () => {
        const drawer = document.getElementById('bandSpecDrawer');
        const icon   = document.getElementById('toggleBandSpecIcon');
        const label  = document.getElementById('toggleBandSpecText');
        const isHidden = drawer.classList.toggle('hidden');
        icon.classList.toggle('rotate-180', !isHidden);
        label.textContent = isHidden ? 'View band comparison' : 'Hide band comparison';
      });
    }

    function initSingleBandTest() {
      document.getElementById('testSingleBandBtn').addEventListener('click', () => {
        document.getElementById('singleBandAlert').classList.toggle('hidden');
      });
    }

    // -----------------------------------------------------------------------
    // 12. DOWNLOAD PREDICTIONS (GeoTIFF & PNG)
    // -----------------------------------------------------------------------
    function initDownloadButtons() {
      document.getElementById('btnDownloadPng').addEventListener('click', () => {
        if (!currentSessionId) {
          showSnackbar('No active inference prediction to download.');
          return;
        }
        window.location.href = `/api/ui/download/${currentSessionId}?format=png`;
        showSnackbar('Downloading prediction as PNG...');
      });

      document.getElementById('btnDownloadTif').addEventListener('click', () => {
        if (!currentSessionId) {
          showSnackbar('No active inference prediction to download.');
          return;
        }
        window.location.href = `/api/ui/download/${currentSessionId}?format=geotiff`;
        showSnackbar('Downloading prediction as GeoTIFF (.tif)...');
      });
    }

    // -----------------------------------------------------------------------
    // 13. UI HELPERS (Loading Spinners & Snackbar)
    // -----------------------------------------------------------------------
    function setLoading(isLoading, type) {
      const isPreset = (type === 'Preset');
      const btn = document.getElementById(isPreset ? 'btnRunPreset' : 'btnRunUpload');
      const spinner = document.getElementById(isPreset ? 'spinnerPreset' : 'spinnerUpload');
      const btnText = document.getElementById(isPreset ? 'runPresetBtnText' : 'runUploadBtnText');
      const overlay = document.getElementById('inferenceLoadingOverlay');

      btn.disabled = isLoading;
      if (isLoading) {
        spinner.classList.remove('hidden');
        overlay.classList.remove('hidden');
        btnText.textContent = isPreset ? 'Running Cloud Removal…' : 'Running inference…';
      } else {
        spinner.classList.add('hidden');
        overlay.classList.add('hidden');
        btnText.textContent = isPreset ? 'Run Cloud Removal' : 'Upload & Run Model';
      }
    }

    function showSnackbar(message) {
      const bar = document.getElementById('snackbar');
      document.getElementById('snackbarText').textContent = message;
      bar.classList.remove('hidden');
      bar.classList.add('flex');
      clearTimeout(window._snackTimer);
      window._snackTimer = setTimeout(() => {
        bar.classList.add('hidden');
        bar.classList.remove('flex');
      }, 3500);
    }
  </script>
</body>
</html>"""


# ============================================================================
# FastAPI UI Router Setup
# ============================================================================


def setup_ui(app: FastAPI) -> None:
    """Attach web UI endpoints and interactive prediction handlers to the FastAPI application."""

    router = APIRouter(tags=["Web UI"])

    @router.get("/", response_class=HTMLResponse, summary="Interactive Web UI")
    @router.get("/ui", response_class=HTMLResponse, summary="Interactive Web UI")
    async def get_ui(request: Request):
        """Serve the self-contained, responsive Cloud Removal Web UI with dynamic active model badge."""
        model_manager = getattr(request.app.state, "model_manager", None)
        badge_title = "Model: Full U-Net (42.13 dB - Best)"
        if model_manager and model_manager.checkpoint_path:
            p_str = str(model_manager.checkpoint_path).lower()
            if "best_unet" in p_str or "standard_unet" in p_str:
                badge_title = "Model: Full U-Net (42.13 dB - Best)"
            elif "best_compact_unet_v2" in p_str or "v2" in p_str:
                badge_title = "Model: U-Net v2 (41.09 dB - Record)"
            elif "rice2" in p_str:
                badge_title = "Model: RICE2 Compact U-Net"
            elif "cgan" in p_str and "fcc" in p_str:
                badge_title = "Model: LISS-IV FCC cGAN"
            elif "cgan" in p_str:
                badge_title = "Model: Pix2Pix cGAN"
            elif "fcc" in p_str:
                badge_title = "Model: FCC U-Net"
            elif "compact" in p_str:
                badge_title = "Model: Compact U-Net"
            else:
                badge_title = f"Model: {Path(model_manager.checkpoint_path).name}"

        html_content = get_ui_html().replace(
            '<span id="serviceBadge">Model: Full U-Net (42.13 dB - Best)</span>',
            f'<span id="serviceBadge">{badge_title}</span>',
        )
        return HTMLResponse(content=html_content, status_code=200)

    @router.get("/api/ui/models", summary="Get Available Model Checkpoints")
    async def get_models(request: Request):
        """Return available trained model checkpoints and currently active checkpoint."""
        model_manager = getattr(request.app.state, "model_manager", None)
        active_ckpt = (
            str(model_manager.checkpoint_path)
            if model_manager and model_manager.checkpoint_path
            else "checkpoints/best_unet.pt"
        )
        return JSONResponse(
            content={
                "models": get_available_models(),
                "active_model": active_ckpt.replace("\\", "/"),
            }
        )

    @router.post("/api/ui/select-model", summary="Switch Active Model Checkpoint")
    async def select_model(request: Request, model_id: str = Form(...)):
        """Dynamically hot-swap active model checkpoint in memory."""
        model_manager = getattr(request.app.state, "model_manager", None)
        if model_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model manager is not available.",
            )

        target_path = Path(model_id)
        if not target_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint file '{model_id}' does not exist.",
            )

        success = model_manager.load(str(target_path))
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load checkpoint '{model_id}': {model_manager.load_error}",
            )

        return JSONResponse(
            content={
                "status": "success",
                "active_checkpoint": str(model_manager.checkpoint_path).replace("\\", "/"),
                "model_config": model_manager.model_config,
            }
        )

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
                    c_arr, _, _, _, _ = parse_uploaded_image(
                        clear_content, clear_file.filename or "clear_image"
                    )
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
