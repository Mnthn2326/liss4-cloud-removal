"""Evaluation metrics for optical satellite image reconstruction.

Implements Mean Absolute Error (MAE) and Peak Signal-to-Noise Ratio (PSNR)
specifically calibrated for surface reflectance normalized to [0.0, 1.0].
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch


def compute_mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute Mean Absolute Error (MAE / L1) between prediction and ground-truth target.

    Args:
        pred: Predicted tensor or numpy array of arbitrary shape.
        target: Target tensor or numpy array of identical shape.

    Returns:
        Scalar MAE value as a float.
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
        return float(torch.mean(torch.abs(pred - target)).item())
    else:
        pred_np = np.asarray(pred, dtype=np.float64)
        target_np = np.asarray(target, dtype=np.float64)
        return float(np.mean(np.abs(pred_np - target_np)))


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Convention:
        Assumes normalized imagery where maximum valid signal is data_range (default: 1.0).
        Formula:
            MSE = mean((pred - target)^2)
            PSNR = 10 * log10(data_range^2 / (MSE + eps))
        If MSE == 0 (exact pixel-for-pixel equality), capped at 100.0 dB.

    Args:
        pred: Predicted tensor or numpy array.
        target: Target ground-truth tensor or numpy array.
        data_range: Maximum possible pixel value (1.0 for normalized reflectance).
        eps: Small epsilon to avoid division by zero.

    Returns:
        Scalar PSNR value in decibels (dB).
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
        mse = float(torch.mean((pred.float() - target.float()) ** 2).item())
    else:
        pred_np = np.asarray(pred, dtype=np.float64)
        target_np = np.asarray(target, dtype=np.float64)
        mse = float(np.mean((pred_np - target_np) ** 2))

    if mse <= 0.0:
        return 100.0  # Cap perfect reconstruction at 100 dB

    psnr_val = 10.0 * math.log10((data_range ** 2) / (mse + eps))
    return float(min(psnr_val, 100.0))
