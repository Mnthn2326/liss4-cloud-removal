"""Tests for evaluation metrics (MAE and PSNR)."""

import math
import numpy as np
import pytest
import torch

from cloud_removal.metrics import compute_mae, compute_psnr


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
