"""Tests for evaluation metrics: MAE, PSNR, SSIM, and SAM."""

import math

import numpy as np
import pytest
import torch

from cloud_removal.metrics import (
    CompoundL1SSIMLoss,
    DifferentiableSSIM,
    compute_mae,
    compute_psnr,
    compute_regional_metrics,
    compute_sam,
    compute_ssim,
)

# ==============================================================================
# Existing Baseline Metric Tests (Regression Suite)
# ==============================================================================


def test_compute_mae_identical_arrays():
    """Identical arrays must yield exactly 0.0 MAE."""
    a = np.ones((3, 64, 64), dtype=np.float32) * 0.5
    assert compute_mae(a, a) == 0.0

    t = torch.ones((3, 64, 64), dtype=torch.float32) * 0.5
    assert compute_mae(t, t) == 0.0


def test_compute_mae_known_difference():
    """Known difference should produce exact mean absolute error."""
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.full((2, 2), 0.25, dtype=np.float32)
    assert pytest.approx(compute_mae(a, b), rel=1e-5) == 0.25

    t_a = torch.zeros((2, 2), dtype=torch.float32)
    t_b = torch.full((2, 2), 0.25, dtype=torch.float32)
    assert pytest.approx(compute_mae(t_a, t_b), rel=1e-5) == 0.25


def test_compute_psnr_identical_arrays():
    """Identical arrays (MSE = 0) must cap at 100.0 dB."""
    a = np.ones((3, 32, 32), dtype=np.float32) * 0.7
    assert compute_psnr(a, a) == 100.0

    t = torch.ones((3, 32, 32), dtype=torch.float32) * 0.7
    assert compute_psnr(t, t) == 100.0


def test_compute_psnr_known_mse():
    """Known MSE of 0.01 with data_range=1.0 must yield 20.0 dB."""
    # MSE = 0.1^2 = 0.01
    # PSNR = 10 * log10(1.0 / 0.01) = 10 * 2 = 20.0 dB
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.full((10, 10), 0.1, dtype=np.float32)

    psnr_np = compute_psnr(a, b, data_range=1.0)
    assert pytest.approx(psnr_np, abs=1e-3) == 20.0

    t_a = torch.zeros((10, 10), dtype=torch.float32)
    t_b = torch.full((10, 10), 0.1, dtype=torch.float32)
    psnr_torch = compute_psnr(t_a, t_b, data_range=1.0)
    assert pytest.approx(psnr_torch, abs=1e-3) == 20.0


def test_compute_psnr_cross_type_interoperability():
    """Metrics should gracefully accept mixed PyTorch and NumPy inputs."""
    a = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    b = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    mae = compute_mae(a, b)
    psnr = compute_psnr(a, b)

    assert mae == 0.0
    assert psnr == 100.0


# ==============================================================================
# SSIM Unit Tests (Section 6.1)
# ==============================================================================


def test_compute_ssim_identical_arrays():
    """Identical images must yield an SSIM of exactly 1.0."""
    rng = np.random.RandomState(42)
    arr = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)
    assert pytest.approx(compute_ssim(arr, arr), abs=1e-6) == 1.0

    t = torch.from_numpy(arr)
    assert pytest.approx(compute_ssim(t, t), abs=1e-6) == 1.0


def test_compute_ssim_different_images():
    """Different images must yield SSIM significantly less than 1.0."""
    rng = np.random.RandomState(42)
    img1 = rng.uniform(0.0, 0.5, size=(3, 32, 32)).astype(np.float32)
    img2 = rng.uniform(0.5, 1.0, size=(3, 32, 32)).astype(np.float32)

    ssim_val = compute_ssim(img1, img2)
    assert isinstance(ssim_val, float)
    assert ssim_val < 0.5
    assert ssim_val >= -1.0


def test_compute_ssim_constant_images():
    """Identical constant images must yield an SSIM of 1.0."""
    c_nonzero = np.full((3, 32, 32), 0.6, dtype=np.float32)
    assert pytest.approx(compute_ssim(c_nonzero, c_nonzero), abs=1e-6) == 1.0

    c_zero = np.zeros((3, 32, 32), dtype=np.float32)
    assert pytest.approx(compute_ssim(c_zero, c_zero), abs=1e-6) == 1.0

    t_nonzero = torch.full((3, 32, 32), 0.6, dtype=torch.float32)
    assert pytest.approx(compute_ssim(t_nonzero, t_nonzero), abs=1e-6) == 1.0


def test_compute_ssim_numpy_and_pytorch_return_types():
    """Return values for both NumPy and PyTorch inputs must be Python floats."""
    np_img = np.ones((3, 32, 32), dtype=np.float32) * 0.5
    torch_img = torch.ones((3, 32, 32), dtype=torch.float32) * 0.5

    res_np = compute_ssim(np_img, np_img)
    res_torch = compute_ssim(torch_img, torch_img)

    assert isinstance(res_np, float)
    assert isinstance(res_torch, float)
    assert not isinstance(res_torch, torch.Tensor)


def test_compute_ssim_single_channel():
    """Single channel 2D (H, W) images must be supported and yield 1.0 for identical."""
    rng = np.random.RandomState(42)
    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)

    assert pytest.approx(compute_ssim(img, img), abs=1e-6) == 1.0

    t_img = torch.from_numpy(img)
    assert pytest.approx(compute_ssim(t_img, t_img), abs=1e-6) == 1.0


def test_compute_ssim_multispectral():
    """Multispectral images with various channel counts (e.g. 3, 13 bands) must be supported."""
    rng = np.random.RandomState(42)

    # 3 bands (e.g. LISS-IV)
    img_3band = rng.uniform(0.0, 1.0, size=(3, 32, 32)).astype(np.float32)
    assert pytest.approx(compute_ssim(img_3band, img_3band), abs=1e-6) == 1.0

    # 13 bands (e.g. SEN12MS-CR Sentinel-2)
    img_13band = rng.uniform(0.0, 1.0, size=(13, 32, 32)).astype(np.float32)
    assert pytest.approx(compute_ssim(img_13band, img_13band), abs=1e-6) == 1.0


def test_compute_ssim_layouts_channel_first_and_last():
    """Channel-first (C, H, W) and channel-last (H, W, C) layouts must compute consistently."""
    rng = np.random.RandomState(42)
    ch_first = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)
    ch_last = np.transpose(ch_first, (1, 2, 0))  # (32, 32, 3)

    target_first = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)
    target_last = np.transpose(target_first, (1, 2, 0))

    # Auto-detected layouts
    ssim_cf = compute_ssim(ch_first, target_first)
    ssim_cl = compute_ssim(ch_last, target_last)
    assert pytest.approx(ssim_cf, abs=1e-5) == ssim_cl

    # Explicit channel_axis
    ssim_cf_explicit = compute_ssim(ch_first, target_first, channel_axis=0)
    ssim_cl_explicit = compute_ssim(ch_last, target_last, channel_axis=-1)
    assert pytest.approx(ssim_cf_explicit, abs=1e-5) == ssim_cl_explicit
    assert pytest.approx(ssim_cf, abs=1e-5) == ssim_cf_explicit


def test_compute_ssim_batched_images():
    """Batched inputs (N, C, H, W) and (N, H, W, C) must compute global mean SSIM."""
    rng = np.random.RandomState(42)
    batch_cf = rng.uniform(0.1, 0.9, size=(4, 3, 32, 32)).astype(np.float32)
    batch_cl = np.transpose(batch_cf, (0, 2, 3, 1))  # (4, 32, 32, 3)

    # Identical batch yields 1.0
    assert pytest.approx(compute_ssim(batch_cf, batch_cf), abs=1e-6) == 1.0
    assert pytest.approx(compute_ssim(batch_cl, batch_cl), abs=1e-6) == 1.0

    target_cf = rng.uniform(0.1, 0.9, size=(4, 3, 32, 32)).astype(np.float32)
    target_cl = np.transpose(target_cf, (0, 2, 3, 1))

    ssim_b_cf = compute_ssim(batch_cf, target_cf)
    ssim_b_cl = compute_ssim(batch_cl, target_cl)
    assert pytest.approx(ssim_b_cf, abs=1e-5) == ssim_b_cl


def test_compute_ssim_different_data_ranges():
    """SSIM must scale consistently with data_range (e.g. data_range=255.0 vs 1.0)."""
    rng = np.random.RandomState(42)
    img1_norm = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)
    img2_norm = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)

    img1_255 = img1_norm * 255.0
    img2_255 = img2_norm * 255.0

    ssim_norm = compute_ssim(img1_norm, img2_norm, data_range=1.0)
    ssim_255 = compute_ssim(img1_255, img2_255, data_range=255.0)

    assert pytest.approx(ssim_norm, abs=1e-5) == ssim_255


def test_compute_ssim_incompatible_shapes():
    """Mismatched shapes must raise a ValueError."""
    a = np.ones((3, 32, 32), dtype=np.float32)
    b_spatial = np.ones((3, 16, 16), dtype=np.float32)
    b_channel = np.ones((4, 32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ssim(a, b_spatial)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ssim(a, b_channel)


def test_compute_ssim_invalid_parameters():
    """Non-positive or invalid parameters must raise ValueError."""
    img = np.ones((3, 32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="data_range must be positive"):
        compute_ssim(img, img, data_range=0.0)

    with pytest.raises(ValueError, match="data_range must be positive"):
        compute_ssim(img, img, data_range=-1.0)

    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        compute_ssim(img, img, window_size=4)

    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        compute_ssim(img, img, window_size=-3)

    with pytest.raises(ValueError, match="sigma must be positive"):
        compute_ssim(img, img, sigma=0.0)

    with pytest.raises(ValueError, match="k1 must be positive"):
        compute_ssim(img, img, k1=0.0)

    with pytest.raises(ValueError, match="k2 must be positive"):
        compute_ssim(img, img, k2=-0.03)


def test_compute_ssim_small_images_and_window_sizes():
    """Small images must compute when window_size <= spatial dimensions, or raise otherwise."""
    # Small image (7x7) with compatible window_size=5
    small_img = np.ones((3, 7, 7), dtype=np.float32) * 0.5
    val = compute_ssim(small_img, small_img, window_size=5)
    assert pytest.approx(val, abs=1e-6) == 1.0

    # Image (8x8) smaller than default window_size=11 must raise ValueError
    img_8x8 = np.ones((3, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="smaller than window_size"):
        compute_ssim(img_8x8, img_8x8, window_size=11)


# ==============================================================================
# SAM Unit Tests (Section 6.2)
# ==============================================================================


def test_compute_sam_identical_nonzero_vectors():
    """Identical non-zero spectral vectors must yield an angle of 0.0 radians."""
    rng = np.random.RandomState(42)
    arr = rng.uniform(0.1, 1.0, size=(3, 32, 32)).astype(np.float32)
    assert pytest.approx(compute_sam(arr, arr), abs=1e-6) == 0.0

    t = torch.from_numpy(arr)
    assert pytest.approx(compute_sam(t, t), abs=1e-6) == 0.0


def test_compute_sam_orthogonal_vectors():
    """Orthogonal spectral vectors must yield an angle of exactly pi / 2 radians (90 deg)."""
    p = np.zeros((3, 8, 8), dtype=np.float32)
    p[0, :, :] = 1.0  # [1, 0, 0]

    t = np.zeros((3, 8, 8), dtype=np.float32)
    t[1, :, :] = 1.0  # [0, 1, 0]

    sam_rad = compute_sam(p, t)
    assert pytest.approx(sam_rad, abs=1e-5) == math.pi / 2.0

    sam_deg = compute_sam(p, t, degrees=True)
    assert pytest.approx(sam_deg, abs=1e-4) == 90.0


def test_compute_sam_opposite_vectors():
    """Opposite spectral vectors must yield an angle of exactly pi radians (180 deg)."""
    p = np.full((3, 8, 8), 1.0, dtype=np.float32)
    t = np.full((3, 8, 8), -1.0, dtype=np.float32)

    sam_rad = compute_sam(p, t)
    assert pytest.approx(sam_rad, abs=1e-5) == math.pi

    sam_deg = compute_sam(p, t, degrees=True)
    assert pytest.approx(sam_deg, abs=1e-4) == 180.0


def test_compute_sam_scalar_multiples():
    """Positive scalar multiples must yield an angle of 0.0 radians."""
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)
    p = t * 3.5

    sam_val = compute_sam(p, t)
    assert pytest.approx(sam_val, abs=1e-6) == 0.0


def test_compute_sam_known_analytical_examples():
    """Known vectors with analytically calculable angles must match exact theory."""
    # [1, 1] and [1, 0] has cos = 1 / sqrt(2) -> angle = pi / 4 = 45 degrees
    p = np.array([[[1.0]], [[1.0]]], dtype=np.float64)  # shape (2, 1, 1) -> (C, H, W)
    t = np.array([[[1.0]], [[0.0]]], dtype=np.float64)

    sam_rad = compute_sam(p, t)
    assert pytest.approx(sam_rad, abs=1e-6) == math.pi / 4.0

    sam_deg = compute_sam(p, t, degrees=True)
    assert pytest.approx(sam_deg, abs=1e-5) == 45.0

    # [1, 0, 0] and [1, 1, 0] has cos = 1 / sqrt(2) -> angle = pi / 4
    p3 = np.array([[[1.0]], [[0.0]], [[0.0]]], dtype=np.float64)
    t3 = np.array([[[1.0]], [[1.0]], [[0.0]]], dtype=np.float64)
    assert pytest.approx(compute_sam(p3, t3), abs=1e-6) == math.pi / 4.0


def test_compute_sam_numpy_and_pytorch_return_types():
    """SAM must return a standard Python float for both NumPy and PyTorch."""
    p_np = np.ones((3, 16, 16), dtype=np.float32)
    t_np = np.ones((3, 16, 16), dtype=np.float32)
    p_torch = torch.ones((3, 16, 16), dtype=torch.float32)
    t_torch = torch.ones((3, 16, 16), dtype=torch.float32)

    res_np = compute_sam(p_np, t_np)
    res_torch = compute_sam(p_torch, t_torch)

    assert isinstance(res_np, float)
    assert isinstance(res_torch, float)
    assert not isinstance(res_torch, torch.Tensor)


def test_compute_sam_channel_first_and_last():
    """Channel-first and channel-last layouts must compute identical SAM."""
    rng = np.random.RandomState(42)
    cf_p = rng.uniform(0.1, 1.0, size=(3, 32, 32)).astype(np.float32)
    cf_t = rng.uniform(0.1, 1.0, size=(3, 32, 32)).astype(np.float32)

    cl_p = np.transpose(cf_p, (1, 2, 0))  # (32, 32, 3)
    cl_t = np.transpose(cf_t, (1, 2, 0))

    sam_cf = compute_sam(cf_p, cf_t)
    sam_cl = compute_sam(cl_p, cl_t)
    assert pytest.approx(sam_cf, abs=1e-5) == sam_cl

    # Explicit channel_axis
    assert pytest.approx(compute_sam(cf_p, cf_t, channel_axis=0), abs=1e-5) == sam_cf
    assert pytest.approx(compute_sam(cl_p, cl_t, channel_axis=-1), abs=1e-5) == sam_cl


def test_compute_sam_batched_inputs():
    """Batched inputs (N, C, H, W) must compute global mean angle over valid pixels."""
    rng = np.random.RandomState(42)
    b_p = rng.uniform(0.1, 1.0, size=(4, 3, 16, 16)).astype(np.float32)
    b_t = rng.uniform(0.1, 1.0, size=(4, 3, 16, 16)).astype(np.float32)

    assert pytest.approx(compute_sam(b_p, b_p), abs=1e-6) == 0.0

    sam_batch = compute_sam(b_p, b_t)
    # Compare with manual average across batch elements
    per_img_sams = [compute_sam(b_p[i], b_t[i]) for i in range(4)]
    assert pytest.approx(sam_batch, abs=1e-5) == np.mean(per_img_sams)


def test_compute_sam_zero_magnitude_vectors():
    """Zero-magnitude vectors (collar/masked background) must be excluded from mean."""
    # 2x2 image with 2 channels:
    # Pixel (0, 0): [1, 0] vs [0, 1] -> orthogonal (pi / 2)
    # Pixel (0, 1): [1, 1] vs [1, 1] -> identical (0)
    # Pixel (1, 0): [0, 0] vs [0, 0] -> zero vector (excluded)
    # Pixel (1, 1): [0, 0] vs [0, 0] -> zero vector (excluded)
    p = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    t = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    # Valid pixels average: (pi/2 + 0) / 2 = pi/4
    sam_val = compute_sam(p, t)
    assert pytest.approx(sam_val, abs=1e-5) == math.pi / 4.0

    # All-zero identical arrays return 0.0
    all_zero = np.zeros((3, 16, 16), dtype=np.float32)
    assert compute_sam(all_zero, all_zero) == 0.0

    # Reference all-zero with non-zero prediction raises ValueError
    non_zero = np.ones((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="No valid non-zero spectral vectors"):
        compute_sam(non_zero, all_zero)


def test_compute_sam_nan_and_inf_handling():
    """Inputs containing NaN or Inf must raise ValueError."""
    valid_img = np.ones((3, 16, 16), dtype=np.float32)
    nan_img = valid_img.copy()
    nan_img[0, 5, 5] = np.nan

    inf_img = valid_img.copy()
    inf_img[1, 2, 2] = np.inf

    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_sam(nan_img, valid_img)

    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_sam(valid_img, nan_img)

    with pytest.raises(ValueError, match="must not contain Inf"):
        compute_sam(inf_img, valid_img)

    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_ssim(nan_img, valid_img)

    with pytest.raises(ValueError, match="must not contain Inf"):
        compute_ssim(inf_img, valid_img)


def test_compute_sam_incompatible_shapes_and_invalid_inputs():
    """Mismatched shapes, negative eps, and invalid axis specifications must raise ValueError."""
    a = np.ones((3, 16, 16), dtype=np.float32)
    b_spatial = np.ones((3, 8, 8), dtype=np.float32)
    b_channel = np.ones((4, 16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_sam(a, b_spatial)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_sam(a, b_channel)

    with pytest.raises(ValueError, match="eps must be non-negative"):
        compute_sam(a, a, eps=-1e-5)

    # channel_axis=0 on 4D batch is the batch dimension, not channel dimension
    batch_img = np.ones((2, 3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="batch dimension"):
        compute_sam(batch_img, batch_img, channel_axis=0)

    # channel_axis on 2D single-channel image
    img_2d = np.ones((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="channel_axis must be None for 2D"):
        compute_sam(img_2d, img_2d, channel_axis=0)


# ==============================================================================
# Cross-Backend Consistency & Interoperability Tests (Section 6.3)
# ==============================================================================


def test_ssim_cross_backend_consistency():
    """NumPy and PyTorch SSIM computations must agree within tight numerical tolerance."""
    rng = np.random.RandomState(123)
    p_np = rng.uniform(0.12, 0.88, size=(3, 32, 32)).astype(np.float64)
    t_np = rng.uniform(0.12, 0.88, size=(3, 32, 32)).astype(np.float64)

    p_torch = torch.from_numpy(p_np)
    t_torch = torch.from_numpy(t_np)

    ssim_np = compute_ssim(p_np, t_np)
    ssim_torch = compute_ssim(p_torch, t_torch)

    assert pytest.approx(ssim_np, abs=1e-6) == ssim_torch


def test_sam_cross_backend_consistency():
    """NumPy and PyTorch SAM computations must agree within tight numerical tolerance."""
    rng = np.random.RandomState(456)
    p_np = rng.uniform(0.05, 0.95, size=(4, 24, 24)).astype(np.float64)
    t_np = rng.uniform(0.05, 0.95, size=(4, 24, 24)).astype(np.float64)

    p_torch = torch.from_numpy(p_np)
    t_torch = torch.from_numpy(t_np)

    sam_np = compute_sam(p_np, t_np)
    sam_torch = compute_sam(p_torch, t_torch)

    assert pytest.approx(sam_np, abs=1e-6) == sam_torch


def test_cross_type_interoperability_ssim_and_sam():
    """Both SSIM and SAM must accept mixed NumPy array and PyTorch tensor inputs."""
    rng = np.random.RandomState(789)
    p_np = rng.uniform(0.1, 0.9, size=(3, 32, 32)).astype(np.float32)
    t_torch = torch.from_numpy(p_np.copy())

    # Mixed inputs (pred numpy, target torch)
    assert pytest.approx(compute_ssim(p_np, t_torch), abs=1e-6) == 1.0
    assert pytest.approx(compute_sam(p_np, t_torch), abs=1e-6) == 0.0

    # Mixed inputs (pred torch, target numpy)
    assert pytest.approx(compute_ssim(t_torch, p_np), abs=1e-6) == 1.0
    assert pytest.approx(compute_sam(t_torch, p_np), abs=1e-6) == 0.0


# ==============================================================================
# Regional Cloud-Masked Evaluation Tests (Task 1)
# ==============================================================================


def test_compute_regional_metrics_partition():
    """Verify separate regional metrics on known errors in cloud and clear partitions."""
    # 10x10 patch: top half (rows 0-4) is cloudy (alpha=0.8 > 0.2), bottom half clear (alpha=0.1 <= 0.2)
    H, W = 10, 10
    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[:5, :] = 0.8  # Cloud region (50 pixels)
    alpha[5:, :] = 0.1  # Clear region (50 pixels)

    target = np.zeros((1, H, W), dtype=np.float32)
    pred = np.zeros((1, H, W), dtype=np.float32)
    # Cloud region error = 0.2 -> MSE = 0.04 -> PSNR = 10 * log10(1 / 0.04) = 13.9794 dB
    pred[0, :5, :] = 0.2
    # Clear region error = 0.05 -> MSE = 0.0025 -> PSNR = 10 * log10(1 / 0.0025) = 26.0206 dB
    pred[0, 5:, :] = 0.05

    # Test with NumPy inputs
    metrics_np = compute_regional_metrics(pred, target, alpha)
    assert pytest.approx(metrics_np["MAE_cloud"], abs=1e-5) == 0.2
    assert pytest.approx(metrics_np["MAE_clear"], abs=1e-5) == 0.05
    assert pytest.approx(metrics_np["PSNR_cloud"], abs=1e-3) == 13.9794
    assert pytest.approx(metrics_np["PSNR_clear"], abs=1e-3) == 26.0206

    # Test with PyTorch inputs
    pred_t = torch.from_numpy(pred)
    target_t = torch.from_numpy(target)
    alpha_t = torch.from_numpy(alpha)
    metrics_torch = compute_regional_metrics(pred_t, target_t, alpha_t)

    assert pytest.approx(metrics_torch["MAE_cloud"], abs=1e-5) == 0.2
    assert pytest.approx(metrics_torch["MAE_clear"], abs=1e-5) == 0.05
    assert pytest.approx(metrics_torch["PSNR_cloud"], abs=1e-3) == 13.9794
    assert pytest.approx(metrics_torch["PSNR_clear"], abs=1e-3) == 26.0206


def test_compute_regional_metrics_empty_regions():
    """Empty cloud or clear region must return float('nan') rather than failing or whole-image."""
    pred = np.full((3, 16, 16), 0.4, dtype=np.float32)
    target = np.full((3, 16, 16), 0.3, dtype=np.float32)

    # 1. All clear (alpha = 0.0 everywhere <= 0.2) -> Cloud region is empty
    mask_all_clear = np.zeros((16, 16), dtype=np.float32)
    res_clear_only = compute_regional_metrics(pred, target, mask_all_clear)
    assert math.isnan(res_clear_only["MAE_cloud"])
    assert math.isnan(res_clear_only["PSNR_cloud"])
    assert pytest.approx(res_clear_only["MAE_clear"], abs=1e-5) == 0.1
    assert math.isfinite(res_clear_only["PSNR_clear"])

    # 2. All cloudy (alpha = 1.0 everywhere > 0.2) -> Clear region is empty
    mask_all_cloudy = np.ones((16, 16), dtype=np.float32)
    res_cloud_only = compute_regional_metrics(pred, target, mask_all_cloudy)
    assert pytest.approx(res_cloud_only["MAE_cloud"], abs=1e-5) == 0.1
    assert math.isfinite(res_cloud_only["PSNR_cloud"])
    assert math.isnan(res_cloud_only["MAE_clear"])
    assert math.isnan(res_cloud_only["PSNR_clear"])


def test_compute_regional_metrics_zero_mse():
    """Identical predictions in a region must cap PSNR at 100.0 dB."""
    img = np.full((3, 16, 16), 0.5, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[:8, :] = 0.9  # top cloudy, bottom clear

    res = compute_regional_metrics(img, img, mask)
    assert res["MAE_cloud"] == 0.0
    assert res["PSNR_cloud"] == 100.0
    assert res["MAE_clear"] == 0.0
    assert res["PSNR_clear"] == 100.0


def test_compute_regional_metrics_mask_broadcasting():
    """Mask broadcasting must support 2D (H,W), 3D (1,H,W), and 4D (N,1,H,W)."""
    pred_4d = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    target_4d = torch.full((2, 3, 16, 16), 0.1, dtype=torch.float32)

    # 2D mask (H, W)
    mask_2d = torch.zeros((16, 16), dtype=torch.float32)
    mask_2d[:8, :] = 1.0
    res_2d = compute_regional_metrics(pred_4d, target_4d, mask_2d)
    assert pytest.approx(res_2d["MAE_cloud"], abs=1e-5) == 0.1
    assert pytest.approx(res_2d["MAE_clear"], abs=1e-5) == 0.1

    # 3D mask (2, 16, 16) - one mask per batch element
    mask_3d = torch.zeros((2, 16, 16), dtype=torch.float32)
    mask_3d[0, :8, :] = 1.0
    mask_3d[1, :, :] = 0.5
    res_3d = compute_regional_metrics(pred_4d, target_4d, mask_3d)
    assert pytest.approx(res_3d["MAE_cloud"], abs=1e-5) == 0.1

    # 4D mask (2, 1, 16, 16)
    mask_4d = mask_3d.unsqueeze(1)
    res_4d = compute_regional_metrics(pred_4d, target_4d, mask_4d)
    assert pytest.approx(res_4d["MAE_cloud"], abs=1e-5) == 0.1


def test_compute_regional_metrics_validation():
    """Invalid shapes, NaNs, and Infs must raise ValueError."""
    pred = np.zeros((3, 16, 16), dtype=np.float32)
    target = np.zeros((3, 16, 16), dtype=np.float32)
    mismatched_target = np.zeros((3, 8, 8), dtype=np.float32)
    mismatched_mask = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_regional_metrics(pred, mismatched_target, np.zeros((16, 16)))

    with pytest.raises(ValueError, match="Cannot broadcast"):
        compute_regional_metrics(pred, target, mismatched_mask)

    nan_img = pred.copy()
    nan_img[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_regional_metrics(nan_img, target, np.zeros((16, 16)))

    nan_mask = np.zeros((16, 16), dtype=np.float32)
    nan_mask[0, 0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_regional_metrics(pred, target, nan_mask)


# ==============================================================================
# Differentiable SSIM and Compound Loss Tests (Task 4)
# ==============================================================================


def test_differentiable_ssim_identical():
    """DifferentiableSSIM forward pass must return 1.0 for identical tensors."""
    ssim_module = DifferentiableSSIM(window_size=11, sigma=1.5, data_range=1.0)
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    val = ssim_module(x, x)

    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0  # Scalar tensor
    assert pytest.approx(val.item(), abs=1e-5) == 1.0


def test_differentiable_ssim_gradient_propagation():
    """DifferentiableSSIM must propagate valid finite non-zero gradients to inputs."""
    ssim_module = DifferentiableSSIM(window_size=11, sigma=1.5, data_range=1.0)
    pred = torch.rand(2, 3, 32, 32, dtype=torch.float32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    ssim_val = ssim_module(pred, target)
    loss = 1.0 - ssim_val
    loss.backward()

    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad)), "Gradients must be finite"
    assert not torch.all(pred.grad == 0.0), "Gradients should be non-zero"
    assert pred.grad.shape == pred.shape


def test_differentiable_ssim_window_constraint():
    """DifferentiableSSIM must reject images smaller than the spatial window size."""
    ssim_module = DifferentiableSSIM(window_size=11)
    too_small = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="smaller than window_size"):
        ssim_module(too_small, too_small)


def test_compound_l1_ssim_loss_computation_and_components():
    """CompoundL1SSIMLoss must correctly compute L1 + lambda * (1 - SSIM) and return components."""
    loss_fn = CompoundL1SSIMLoss(ssim_weight=0.2, data_range=1.0)
    pred = torch.rand(2, 3, 32, 32, dtype=torch.float32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    total_loss, components = loss_fn(pred, target, return_components=True)

    assert isinstance(total_loss, torch.Tensor)
    assert "loss_total" in components
    assert "loss_l1" in components
    assert "loss_ssim" in components
    assert "ssim_val" in components

    expected_total = components["loss_l1"] + 0.2 * components["loss_ssim"]
    assert pytest.approx(components["loss_total"], abs=1e-5) == expected_total
    assert pytest.approx(total_loss.item(), abs=1e-5) == expected_total

    # Verify backpropagation through compound loss
    total_loss.backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_compound_l1_ssim_loss_identical_inputs():
    """CompoundL1SSIMLoss should be virtually 0.0 when prediction matches target."""
    loss_fn = CompoundL1SSIMLoss(ssim_weight=0.2, data_range=1.0)
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    loss = loss_fn(x, x)

    assert pytest.approx(loss.item(), abs=1e-5) == 0.0


def test_compound_l1_ssim_loss_zero_weight():
    """When ssim_weight=0.0, CompoundL1SSIMLoss reduces to standard L1 loss."""
    loss_fn = CompoundL1SSIMLoss(ssim_weight=0.0)
    l1_fn = torch.nn.L1Loss()

    pred = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    target = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    loss_compound, comps = loss_fn(pred, target, return_components=True)
    loss_pure_l1 = l1_fn(pred, target)

    assert pytest.approx(loss_compound.item(), abs=1e-6) == loss_pure_l1.item()
    assert comps["loss_ssim"] == 0.0
