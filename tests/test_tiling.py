"""Unit tests for gigapixel tiling and overlap blending inference engine (Phase 4).

Validates deterministic tile generation, seamless overlap blending, positive weight coverage,
geospatial metadata preservation, and memory-bounded raster streaming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn
from rasterio.transform import from_origin

from cloud_removal.infer import (
    generate_blending_window,
    generate_tile_coordinates,
    predict_tiled_array,
    predict_tiled_raster,
)


class IdentityModel(nn.Module):
    """Pass-through model for identity reconstruction testing."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Dummy parameter to ensure parameters() is non-empty
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScaledModel(nn.Module):
    """Model that outputs a constant or scaled function."""

    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = scale
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.scale, 0.0, 1.0)


# ============================================================================
# 1. Tile Coordinate Generation Tests
# ============================================================================


def test_tile_coordinates_coverage():
    """Verify that tile coordinates cover every single pixel in the domain."""
    h, w = 350, 480
    tile_size = 128
    overlap = 32

    tiles = generate_tile_coordinates(h, w, tile_size=tile_size, overlap=overlap)
    assert len(tiles) > 0

    # Ensure every coordinate in [0, h) x [0, w) is covered by at least one tile
    coverage = np.zeros((h, w), dtype=bool)
    for r, c, h_win, w_win in tiles:
        assert 0 <= r < h
        assert 0 <= c < w
        assert r + h_win <= h
        assert c + w_win <= w
        coverage[r : r + h_win, c : c + w_win] = True

    assert np.all(coverage), "Every pixel must be covered by generated tiles"


def test_tile_coordinates_small_image():
    """Verify handling when image is smaller than tile_size."""
    h, w = 64, 48
    tile_size = 128
    overlap = 16

    tiles = generate_tile_coordinates(h, w, tile_size=tile_size, overlap=overlap)
    assert len(tiles) == 1
    r, c, h_win, w_win = tiles[0]
    assert r == 0 and c == 0
    assert h_win == 64 and w_win == 48


def test_tile_coordinates_exact_multiple():
    """Verify coordinates when dimensions are exact multiples of stride."""
    tile_size = 64
    overlap = 16
    stride = tile_size - overlap  # 48
    h, w = stride * 3 + overlap, stride * 2 + overlap  # 160, 112

    tiles = generate_tile_coordinates(h, w, tile_size=tile_size, overlap=overlap)
    # Check boundary tiles reach exact edge
    max_r = max(t[0] + t[2] for t in tiles)
    max_c = max(t[1] + t[3] for t in tiles)
    assert max_r == h
    assert max_c == w


def test_tile_coordinates_invalid_args():
    """Verify rejection of invalid parameters."""
    with pytest.raises(ValueError, match="positive"):
        generate_tile_coordinates(0, 100)
    with pytest.raises(ValueError, match="positive"):
        generate_tile_coordinates(100, -5)
    with pytest.raises(ValueError, match="strictly greater than overlap"):
        generate_tile_coordinates(100, 100, tile_size=64, overlap=64)


# ============================================================================
# 2. Blending Window Tests
# ============================================================================


def test_blending_window_positive_floor():
    """Verify that blending window is strictly positive everywhere."""
    floor = 0.05
    w2d = generate_blending_window(64, 64, floor=floor)
    assert w2d.shape == (64, 64)
    assert np.all(np.isfinite(w2d))
    assert np.min(w2d) >= (floor * floor) - 1e-6
    assert np.max(w2d) <= 1.0 + 1e-6
    assert np.all(w2d > 0.0)


def test_blending_window_1x1():
    """Verify 1x1 window edge case."""
    w2d = generate_blending_window(1, 1)
    assert w2d.shape == (1, 1)
    assert w2d[0, 0] == 1.0


# ============================================================================
# 3. In-Memory Array Tiling Inference Tests
# ============================================================================


def test_predict_tiled_array_identity_reconstruction():
    """Verify that identity model perfectly reconstructs input with overlap blending."""
    rng = np.random.default_rng(123)
    # Rectangular image with arbitrary dimensions
    cloudy = rng.uniform(0.1, 0.9, size=(3, 217, 305)).astype(np.float32)

    model = IdentityModel()
    reconstructed = predict_tiled_array(
        model=model,
        cloudy=cloudy,
        tile_size=64,
        overlap=16,
        batch_size=4,
    )

    assert reconstructed.shape == cloudy.shape
    # Maximum absolute error should be virtually zero (< 1e-5)
    max_diff = np.max(np.abs(reconstructed - cloudy))
    assert max_diff < 1e-5, f"Identity reconstruction max diff {max_diff} exceeded tolerance"


def test_predict_tiled_array_constant_input():
    """Verify constant array reconstruction is perfectly preserved."""
    c_val = 0.42
    cloudy = np.full((3, 100, 150), c_val, dtype=np.float32)
    model = IdentityModel()

    reconstructed = predict_tiled_array(
        model=model,
        cloudy=cloudy,
        tile_size=48,
        overlap=12,
    )

    assert np.allclose(reconstructed, c_val, atol=1e-5)


# ============================================================================
# 4. Raster Tiled Inference Tests
# ============================================================================


def test_predict_tiled_raster_streaming_memoryfile():
    """Verify windowed raster-to-raster tiled inference preserves metadata and values."""
    h, w = 180, 240
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    rng = np.random.default_rng(42)
    raw_data = rng.uniform(0.1, 0.8, size=(3, h, w)).astype(np.float32)

    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }

    model = IdentityModel()

    with rasterio.MemoryFile() as in_mem:
        with in_mem.open(**meta) as src:
            src.write(raw_data)

        with in_mem.open() as src_read:
            with rasterio.MemoryFile() as out_mem:
                summary = predict_tiled_raster(
                    model=model,
                    input_source=src_read,
                    output_target=out_mem,
                    tile_size=64,
                    overlap=16,
                    batch_size=2,
                )

                assert summary["height"] == h
                assert summary["width"] == w
                assert summary["channels"] == 3
                assert summary["crs"] == crs

                with out_mem.open() as dst_read:
                    assert dst_read.height == h
                    assert dst_read.width == w
                    assert dst_read.count == 3
                    assert dst_read.crs == crs
                    assert dst_read.transform == transform

                    out_data = dst_read.read()
                    max_err = np.max(np.abs(out_data - raw_data))
                    assert max_err < 1e-5, f"Raster streaming diff {max_err} exceeded tolerance"


def test_predict_tiled_raster_file_to_file(tmp_path: Path):
    """Verify windowed raster file-to-file execution."""
    in_file = tmp_path / "cloudy_in.tif"
    out_file = tmp_path / "reconstructed_out.tif"

    h, w = 150, 150
    transform = from_origin(100.0, 200.0, 5.0, 5.0)
    crs = "EPSG:4326"

    raw_data = np.full((3, h, w), 0.35, dtype=np.float32)

    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }

    with rasterio.open(in_file, "w", **meta) as dst:
        dst.write(raw_data)

    model = ScaledModel(scale=0.5)

    summary = predict_tiled_raster(
        model=model,
        input_source=in_file,
        output_target=out_file,
        tile_size=64,
        overlap=16,
    )
    assert summary["height"] == h
    assert summary["width"] == w

    assert out_file.is_file()
    with rasterio.open(out_file) as res:
        assert res.shape == (h, w)
        assert res.crs == crs
        data = res.read()
        assert np.allclose(data, 0.35 * 0.5, atol=1e-5)
