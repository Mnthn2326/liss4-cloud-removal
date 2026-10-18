"""Unit tests for synthetic cloud generation and atmospheric blending."""

import numpy as np
import pytest

from cloud_removal.synthetic_clouds import (
    CloudConfig,
    SyntheticCloudGenerator,
    blend_synthetic_clouds,
    generate_cloud_mask,
)


def test_seed_determinism():
    """Verify that identical random seeds produce identical cloud masks."""
    mask1 = generate_cloud_mask(height=64, width=64, coverage=0.3, seed=123)
    mask2 = generate_cloud_mask(height=64, width=64, coverage=0.3, seed=123)

    np.testing.assert_array_equal(mask1, mask2)


def test_seed_divergence():
    """Verify that different random seeds produce different cloud masks."""
    mask1 = generate_cloud_mask(height=64, width=64, coverage=0.3, seed=100)
    mask2 = generate_cloud_mask(height=64, width=64, coverage=0.3, seed=200)

    assert not np.array_equal(mask1, mask2)


def test_mask_shape_and_value_range():
    """Verify cloud mask shape and opacity values stay strictly in [0.0, max_opacity]."""
    cfg = CloudConfig(min_opacity=0.25, max_opacity=0.90)
    mask = generate_cloud_mask(height=80, width=120, coverage=0.4, config=cfg, seed=42)

    assert mask.shape == (80, 120)
    assert mask.dtype == np.float32
    assert float(np.min(mask)) >= 0.0
    assert float(np.max(mask)) <= 0.90 + 1e-5

    # Pixels with cloud must be at least min_opacity
    cloudy_pixels = mask[mask > 0.0]
    assert len(cloudy_pixels) > 0
    assert np.all(cloudy_pixels >= 0.25 - 1e-5)


def test_zero_and_full_coverage_edge_cases():
    """Verify extreme coverage edge cases (0% and 100%)."""
    mask_zero = generate_cloud_mask(height=32, width=32, coverage=0.0, seed=1)
    assert np.all(mask_zero == 0.0)

    cfg = CloudConfig(max_opacity=0.95)
    mask_full = generate_cloud_mask(height=32, width=32, coverage=1.0, config=cfg, seed=1)
    assert np.allclose(mask_full, 0.95)


def test_blend_preserves_dimensions_and_channel_order_channel_first():
    """Verify that blending preserves (C, H, W) layout and data type."""
    # 3 bands (BAND2, BAND3, BAND4), 64x64, uint16
    clear = np.full((3, 64, 64), 200, dtype=np.uint16)
    mask = generate_cloud_mask(height=64, width=64, coverage=0.5, seed=42)

    cfg = CloudConfig(cloud_dn_min=800, cloud_dn_max=1000)
    cloudy = blend_synthetic_clouds(clear, mask, config=cfg, seed=42)

    assert cloudy.shape == (3, 64, 64)
    assert cloudy.dtype == np.uint16

    # Where mask is 0, cloudy image should match clear image exactly
    clear_pixels = mask == 0.0
    for c in range(3):
        np.testing.assert_array_equal(cloudy[c][clear_pixels], clear[c][clear_pixels])

    # Where mask > 0, cloudy values should increase towards cloud DN
    cloud_pixels = mask > 0.5
    for c in range(3):
        assert np.all(cloudy[c][cloud_pixels] > clear[c][cloud_pixels])


def test_blend_channel_last_and_2d():
    """Verify that (H, W, C) and 2D arrays are supported and maintain layout."""
    # (H, W, C)
    clear_hwc = np.full((48, 48, 3), 150, dtype=np.uint16)
    mask = generate_cloud_mask(height=48, width=48, coverage=0.3, seed=7)
    cloudy_hwc = blend_synthetic_clouds(clear_hwc, mask, seed=7)
    assert cloudy_hwc.shape == (48, 48, 3)
    assert cloudy_hwc.dtype == np.uint16

    # 2D (H, W)
    clear_2d = np.full((48, 48), 100, dtype=np.uint8)
    cloudy_2d = blend_synthetic_clouds(clear_2d, mask, seed=7)
    assert cloudy_2d.shape == (48, 48)
    assert cloudy_2d.dtype == np.uint8


def test_synthetic_cloud_generator_class():
    """Verify stateful SyntheticCloudGenerator helper class."""
    generator = SyntheticCloudGenerator()
    clear = np.random.randint(100, 400, size=(3, 64, 64), dtype=np.uint16)

    cloudy, mask = generator.apply(clear, coverage=0.35, seed=99)
    assert cloudy.shape == clear.shape
    assert mask.shape == (64, 64)
    assert cloudy.dtype == clear.dtype


def test_invalid_parameters_raise_clear_errors():
    """Verify that invalid configurations raise informative exceptions."""
    with pytest.raises(ValueError, match="Invalid coverage range"):
        CloudConfig(min_coverage=0.8, max_coverage=0.2)

    with pytest.raises(ValueError, match="Invalid opacity range"):
        CloudConfig(min_opacity=-0.1)

    with pytest.raises(ValueError, match="Dimensions must be strictly positive"):
        generate_cloud_mask(height=0, width=64)

    with pytest.raises(ValueError, match="Coverage must be in"):
        generate_cloud_mask(height=32, width=32, coverage=1.5)

    clear = np.zeros((3, 32, 32), dtype=np.uint16)
    mismatched_mask = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="do not match"):
        blend_synthetic_clouds(clear, mismatched_mask)
