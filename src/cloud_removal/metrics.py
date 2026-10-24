"""Evaluation metrics for optical satellite image reconstruction.

Implements Mean Absolute Error (MAE), Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index Measure (SSIM), Spectral Angle Mapper (SAM),
regional cloud-masked evaluation metrics, and compound differentiable losses
specifically calibrated for surface reflectance normalized to [0.0, 1.0] and
multispectral satellite imagery.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "compute_mae",
    "compute_psnr",
    "compute_ssim",
    "compute_sam",
    "compute_regional_metrics",
    "DifferentiableSSIM",
    "CompoundL1SSIMLoss",
]


def _validate_and_canonicalize_tensors(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    channel_axis: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate shapes, values, and canonicalize input tensors to (N, C, H, W).

    Args:
        pred: Predicted tensor or numpy array.
        target: Ground-truth target tensor or numpy array.
        channel_axis: Optional channel axis index (e.g., 0, 1, or -1).

    Returns:
        Tuple of (pred_canonical, target_canonical) guaranteed to have shape
        (N, C, H, W) on the same device and with matching floating-point dtypes.

    Raises:
        ValueError: If shapes mismatch, dimensions are invalid or ambiguous,
            or inputs contain NaN or infinite values.
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
        elif pred.device != target.device:
            target = target.to(pred.device)
        pred_t = pred
        target_t = target
    elif isinstance(target, torch.Tensor):
        pred_t = torch.as_tensor(pred, dtype=target.dtype, device=target.device)
        target_t = target
    else:
        # Both are NumPy arrays or array-like collections
        pred_t = torch.as_tensor(pred, dtype=torch.float64)
        target_t = torch.as_tensor(target, dtype=torch.float64)

    if pred_t.shape != target_t.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {tuple(pred_t.shape)} != target shape {tuple(target_t.shape)}"
        )

    if torch.isnan(pred_t).any() or torch.isnan(target_t).any():
        raise ValueError("Input pred and target must not contain NaN values.")
    if torch.isinf(pred_t).any() or torch.isinf(target_t).any():
        raise ValueError("Input pred and target must not contain Inf values.")

    if not pred_t.is_floating_point():
        pred_t = pred_t.to(torch.float32)
        target_t = target_t.to(torch.float32)

    shape = tuple(pred_t.shape)
    ndim = pred_t.ndim

    if ndim == 2:
        if channel_axis is not None:
            raise ValueError(
                f"channel_axis must be None for 2D single-channel image with shape {shape}."
            )
        # (H, W) -> (1, 1, H, W)
        return pred_t.unsqueeze(0).unsqueeze(0), target_t.unsqueeze(0).unsqueeze(0)

    elif ndim == 3:
        d0, d1, d2 = shape
        if channel_axis is not None:
            norm_axis = channel_axis if channel_axis >= 0 else channel_axis + 3
            if norm_axis == 0:
                # (C, H, W) -> (1, C, H, W)
                return pred_t.unsqueeze(0), target_t.unsqueeze(0)
            elif norm_axis == 2:
                # (H, W, C) -> (1, C, H, W)
                return pred_t.permute(2, 0, 1).unsqueeze(0), target_t.permute(2, 0, 1).unsqueeze(0)
            elif norm_axis == 1:
                # (H, C, W) -> (1, C, H, W)
                return pred_t.permute(1, 0, 2).unsqueeze(0), target_t.permute(1, 0, 2).unsqueeze(0)
            else:
                raise ValueError(
                    f"Invalid channel_axis {channel_axis} for 3D image with shape {shape}."
                )
        else:
            # Repository default convention is channel-first (C, H, W).
            # Channel-last (H, W, C) is inferred when the trailing dimension is a typical channel count
            # (d2 <= 16) and spatial dimensions are distinctly larger (d0 > 16 and d1 > 16).
            if d2 <= 16 and d0 > 16 and d1 > 16:
                return pred_t.permute(2, 0, 1).unsqueeze(0), target_t.permute(2, 0, 1).unsqueeze(0)
            else:
                return pred_t.unsqueeze(0), target_t.unsqueeze(0)

    elif ndim == 4:
        N, d1, d2, d3 = shape
        if channel_axis is not None:
            norm_axis = channel_axis if channel_axis >= 0 else channel_axis + 4
            if norm_axis == 0:
                raise ValueError(
                    "channel_axis=0 is the batch dimension for 4D batched images. "
                    "Channel axis must be 1, 2, or 3 (or negative equivalents -3, -2, -1)."
                )
            elif norm_axis == 1:
                # (N, C, H, W)
                return pred_t, target_t
            elif norm_axis == 3:
                # (N, H, W, C) -> (N, C, H, W)
                return pred_t.permute(0, 3, 1, 2), target_t.permute(0, 3, 1, 2)
            elif norm_axis == 2:
                # (N, H, C, W) -> (N, C, H, W)
                return pred_t.permute(0, 2, 1, 3), target_t.permute(0, 2, 1, 3)
            else:
                raise ValueError(
                    f"Invalid channel_axis {channel_axis} for 4D image with shape {shape}."
                )
        else:
            # Repository default convention is channel-first (N, C, H, W).
            # Channel-last (N, H, W, C) is inferred when the trailing dimension is <= 16 and
            # spatial dimensions are distinctly larger (d1 > 16 and d2 > 16).
            if d3 <= 16 and d1 > 16 and d2 > 16:
                return pred_t.permute(0, 3, 1, 2), target_t.permute(0, 3, 1, 2)
            else:
                return pred_t, target_t

    else:
        raise ValueError(f"Expected 2D, 3D, or 4D array/tensor, got {ndim}D shape {shape}.")


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

    psnr_val = 10.0 * math.log10((data_range**2) / (mse + eps))
    return float(min(psnr_val, 100.0))


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    channel_axis: Optional[int] = None,
) -> float:
    """Compute Structural Similarity Index Measure (SSIM) between prediction and target.

    Implements the standard Wang et al. (2004) formulation using a local 2D Gaussian
    window applied per spectral channel and aggregated via spatial and channel mean.

    Formulation:
        SSIM(x, y) = [(2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)] /
                     [(mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2)]
        where:
            C1 = (k1 * data_range) ** 2
            C2 = (k2 * data_range) ** 2
            mu_x, mu_y: Local means computed with 2D Gaussian filter.
            sigma_x^2, sigma_y^2: Local variances.
            sigma_xy: Local covariance.

    Layout Support:
        Supports 2D (H, W), 3D (C, H, W) or (H, W, C), and 4D (N, C, H, W) or (N, H, W, C).
        By default, 3D and 4D arrays follow the repository's channel-first convention
        (C, H, W) and (N, C, H, W). Channel-last layouts are automatically detected when
        the channel dimension is unambiguously trailing, or can be explicitly designated
        via channel_axis (e.g., channel_axis=-1).

    Multispectral Handling:
        For multi-band imagery, SSIM is computed independently across the spatial extent
        of each spectral channel, and the scalar result is the unweighted mean across
        all spectral channels (and across batch elements if batched).

    Boundary Handling:
        Reflect padding of width window_size // 2 is applied prior to convolution,
        preserving image dimensions and avoiding artificial boundary luminance dropoff.

    Args:
        pred: Predicted image tensor or numpy array.
        target: Target ground-truth image tensor or numpy array of matching shape.
        data_range: Dynamic range of the image pixel values (default: 1.0 for normalized reflectance).
        window_size: Odd integer size of the square Gaussian window (default: 11).
        sigma: Standard deviation of the Gaussian window (default: 1.5).
        k1: Metric stability constant for luminance (default: 0.01).
        k2: Metric stability constant for contrast/structure (default: 0.03).
        channel_axis: Explicit axis index of the spectral channel dimension (e.g., 0, 1, or -1).

    Returns:
        Scalar SSIM value as a standard Python float in [-1.0, 1.0] (typically [0.0, 1.0]).

    Raises:
        ValueError: If shapes mismatch, parameters are non-positive or invalid, image
            dimensions are smaller than window_size, or inputs contain NaN / Inf.
    """
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    if window_size <= 0 or not isinstance(window_size, int) or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if k1 <= 0.0:
        raise ValueError(f"k1 must be positive, got {k1}")
    if k2 <= 0.0:
        raise ValueError(f"k2 must be positive, got {k2}")

    pred_t, target_t = _validate_and_canonicalize_tensors(pred, target, channel_axis=channel_axis)

    N, C, H, W = pred_t.shape
    if min(H, W) < window_size:
        raise ValueError(
            f"Image spatial extent ({H}, {W}) is smaller than window_size ({window_size}). "
            f"Specify a smaller odd window_size <= min(H, W)."
        )

    if torch.equal(pred_t, target_t):
        return 1.0

    pad = window_size // 2
    coords = (
        torch.arange(window_size, dtype=pred_t.dtype, device=pred_t.device)
        - (window_size - 1) / 2.0
    )
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # Flatten (N, C, H, W) to (N*C, 1, H, W) for efficient depthwise 2D convolution
    x = pred_t.reshape(N * C, 1, H, W)
    y = target_t.reshape(N * C, 1, H, W)

    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.conv2d(x_pad, kernel)
    mu_y = F.conv2d(y_pad, kernel)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.clamp(F.conv2d(x_pad**2, kernel) - mu_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(F.conv2d(y_pad**2, kernel) - mu_y_sq, min=0.0)
    sigma_xy = F.conv2d(x_pad * y_pad, kernel) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = num / den
    return float(ssim_map.mean().item())


def compute_sam(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    degrees: bool = False,
    eps: float = 1e-8,
    channel_axis: Optional[int] = None,
) -> float:
    """Compute Spectral Angle Mapper (SAM) between predicted and target multispectral images.

    Measures the average spectral angle between predicted and reference pixel vectors
    across all valid non-zero pixels. Invariant to absolute illumination scaling.

    Formulation:
        At each spatial pixel location p:
            theta(p) = arccos( (p . t) / (||p||_2 * ||t||_2) )
        where p is the predicted spectral vector, t is the target spectral vector,
        and ||.||_2 denotes the L2 norm across the spectral channel dimension.

    Spectral Dimension:
        Calculated across the channel dimension. In channel-first format (C, H, W) or
        (N, C, H, W), the spectral vector length is C. In channel-last format (H, W, C)
        or (N, H, W, C), the last axis is treated as the spectral dimension.

    Zero-Magnitude and Background Policy:
        Pixels where either predicted or target vector norm is <= eps (such as collar,
        masked background, or completely dark pixels) are considered invalid and are
        excluded from the mean angle calculation. If all pixels in both inputs are zero
        and identical, 0.0 is returned. If no valid pixels exist and inputs differ,
        a ValueError is raised.

    NaN / Inf Policy:
        Inputs containing NaN or infinite values raise a ValueError to ensure scientific
        validity and prevent silent corruption of benchmark metrics.

    Batch Aggregation:
        For batched inputs (4D), computes the global mean spectral angle across all valid
        pixels in the batch, consistent with MAE and PSNR batch reduction.

    Args:
        pred: Predicted multispectral tensor or numpy array.
        target: Target ground-truth tensor or numpy array of matching shape.
        degrees: If True, return mean angle in degrees [0, 180]. If False (default),
            return mean angle in radians [0, pi].
        eps: Small numerical threshold for non-zero spectral vector norm and division stability.
        channel_axis: Explicit axis index of the spectral channel dimension (e.g., 0, 1, or -1).

    Returns:
        Scalar SAM value as a standard Python float (radians by default, or degrees if degrees=True).

    Raises:
        ValueError: If shapes mismatch, dimensions are invalid, no valid non-zero pixels exist
            to compute the angle, or inputs contain NaN / Inf.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    pred_t, target_t = _validate_and_canonicalize_tensors(pred, target, channel_axis=channel_axis)

    if torch.equal(pred_t, target_t):
        return 0.0

    # Promote to float64 for intermediate dot/norm to eliminate float32 acos instability near 1.0
    p = pred_t.to(torch.float64)
    t = target_t.to(torch.float64)

    # Dot product and L2 norms across spectral channels (dim=1)
    dot = torch.sum(p * t, dim=1)  # (N, H, W)
    norm_p = torch.linalg.norm(p, dim=1)  # (N, H, W)
    norm_t = torch.linalg.norm(t, dim=1)  # (N, H, W)

    valid_mask = (norm_p > eps) & (norm_t > eps)

    if not torch.any(valid_mask):
        if torch.allclose(p, t, atol=1e-12):
            return 0.0
        raise ValueError("No valid non-zero spectral vectors found to compute SAM.")

    denom = norm_p[valid_mask] * norm_t[valid_mask]
    cos_sim = dot[valid_mask] / denom
    cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
    angles = torch.acos(cos_sim)

    mean_angle = float(torch.mean(angles).item())
    if degrees:
        mean_angle = float(mean_angle * (180.0 / math.pi))

    return mean_angle


def compute_regional_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    cloud_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.2,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """Compute separate MAE and PSNR metrics for cloud-affected and clear regions.

    Regions are partitioned using the cloud opacity map `alpha` (cloud_mask):
        - Cloud-affected region: alpha > threshold (default: 0.2)
        - Clear region: alpha <= threshold (default: 0.2)

    Formula:
        MAE_region = mean(abs(pred - target)) over valid pixels in that region
        MSE_region = mean((pred - target)^2) over valid pixels in that region
        PSNR_region = 10 * log10(data_range^2 / (MSE_region + eps))
        If MSE == 0, PSNR is capped at 100.0 dB.
        If a region contains no pixels, its MAE and PSNR are reported as NaN.

    Args:
        pred: Predicted tensor or numpy array.
        target: Target ground-truth tensor or numpy array of identical shape.
        cloud_mask: Cloud opacity map (alpha) with values in [0.0, 1.0].
            Can be shaped (H, W), (1, H, W), or (N, 1, H, W).
        threshold: Cloud opacity cutoff threshold (default: 0.2).
        data_range: Maximum pixel dynamic range (default: 1.0).
        eps: Small epsilon to prevent division by zero in log10.

    Returns:
        Dictionary containing:
            - 'MAE_cloud': Float MAE over cloud-affected pixels (or NaN if none)
            - 'PSNR_cloud': Float PSNR over cloud-affected pixels (or NaN if none)
            - 'MAE_clear': Float MAE over clear pixels (or NaN if none)
            - 'PSNR_clear': Float PSNR over clear pixels (or NaN if none)

    Raises:
        ValueError: If pred and target shapes mismatch, spatial dimensions mismatch
            the mask, or inputs contain NaN / Inf.
    """
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    is_torch = (
        isinstance(pred, torch.Tensor)
        or isinstance(target, torch.Tensor)
        or isinstance(cloud_mask, torch.Tensor)
    )

    if is_torch:
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
        elif target.device != pred.device:
            target = target.to(pred.device)
        if not isinstance(cloud_mask, torch.Tensor):
            cloud_mask = torch.as_tensor(cloud_mask, dtype=pred.dtype, device=pred.device)
        elif cloud_mask.device != pred.device:
            cloud_mask = cloud_mask.to(pred.device)

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
            )

        if torch.isnan(pred).any() or torch.isnan(target).any() or torch.isnan(cloud_mask).any():
            raise ValueError("Inputs and cloud_mask must not contain NaN values.")
        if torch.isinf(pred).any() or torch.isinf(target).any() or torch.isinf(cloud_mask).any():
            raise ValueError("Inputs and cloud_mask must not contain Inf values.")

        diff = pred.float() - target.float()
        raw_mask = cloud_mask > threshold

        # Broadcast mask to match pred shape
        m = raw_mask
        p_shape = pred.shape
        try:
            if m.ndim == 2 and len(p_shape) == 3:  # (H, W) -> (C, H, W)
                m = m.unsqueeze(0).expand(p_shape)
            elif m.ndim == 3 and len(p_shape) == 3:  # (1, H, W) -> (C, H, W)
                if m.shape[0] == 1:
                    m = m.expand(p_shape)
            elif m.ndim == 2 and len(p_shape) == 4:  # (H, W) -> (N, C, H, W)
                m = m.unsqueeze(0).unsqueeze(0).expand(p_shape)
            elif m.ndim == 3 and len(p_shape) == 4:  # (N, H, W) -> (N, C, H, W)
                m = m.unsqueeze(1).expand(p_shape)
            elif m.ndim == 4 and len(p_shape) == 4:  # (N, 1, H, W) -> (N, C, H, W)
                if m.shape[1] == 1:
                    m = m.expand(p_shape)
        except (RuntimeError, ValueError) as err:
            raise ValueError(
                f"Cannot broadcast cloud_mask of shape {tuple(cloud_mask.shape)} to {tuple(p_shape)}"
            ) from err

        if m.shape != p_shape:
            raise ValueError(
                f"Cannot broadcast cloud_mask of shape {tuple(cloud_mask.shape)} to {tuple(p_shape)}"
            )

        mask_cloud = m
        mask_clear = ~m

        # Cloud region metrics
        if mask_cloud.any():
            diff_cloud = diff[mask_cloud]
            mae_cloud = float(torch.mean(torch.abs(diff_cloud)).item())
            mse_cloud = float(torch.mean(diff_cloud**2).item())
            if mse_cloud <= 0.0:
                psnr_cloud = 100.0
            else:
                psnr_cloud = float(
                    min(10.0 * math.log10((data_range**2) / (mse_cloud + eps)), 100.0)
                )
        else:
            mae_cloud = float("nan")
            psnr_cloud = float("nan")

        # Clear region metrics
        if mask_clear.any():
            diff_clear = diff[mask_clear]
            mae_clear = float(torch.mean(torch.abs(diff_clear)).item())
            mse_clear = float(torch.mean(diff_clear**2).item())
            if mse_clear <= 0.0:
                psnr_clear = 100.0
            else:
                psnr_clear = float(
                    min(10.0 * math.log10((data_range**2) / (mse_clear + eps)), 100.0)
                )
        else:
            mae_clear = float("nan")
            psnr_clear = float("nan")

    else:
        pred_np = np.asarray(pred, dtype=np.float64)
        target_np = np.asarray(target, dtype=np.float64)
        mask_np = np.asarray(cloud_mask, dtype=np.float64)

        if pred_np.shape != target_np.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred_np.shape} != target shape {target_np.shape}"
            )

        if np.isnan(pred_np).any() or np.isnan(target_np).any() or np.isnan(mask_np).any():
            raise ValueError("Inputs and cloud_mask must not contain NaN values.")
        if np.isinf(pred_np).any() or np.isinf(target_np).any() or np.isinf(mask_np).any():
            raise ValueError("Inputs and cloud_mask must not contain Inf values.")

        diff_np = pred_np - target_np
        raw_mask_np = mask_np > threshold
        p_shape_np = pred_np.shape

        m_np = raw_mask_np
        try:
            if m_np.ndim == 2 and len(p_shape_np) == 3:
                m_np = np.broadcast_to(m_np[np.newaxis, :, :], p_shape_np)
            elif m_np.ndim == 3 and len(p_shape_np) == 3:
                if m_np.shape[0] == 1:
                    m_np = np.broadcast_to(m_np, p_shape_np)
            elif m_np.ndim == 2 and len(p_shape_np) == 4:
                m_np = np.broadcast_to(m_np[np.newaxis, np.newaxis, :, :], p_shape_np)
            elif m_np.ndim == 3 and len(p_shape_np) == 4:
                m_np = np.broadcast_to(m_np[:, np.newaxis, :, :], p_shape_np)
            elif m_np.ndim == 4 and len(p_shape_np) == 4:
                if m_np.shape[1] == 1:
                    m_np = np.broadcast_to(m_np, p_shape_np)
        except ValueError as err:
            raise ValueError(
                f"Cannot broadcast cloud_mask of shape {mask_np.shape} to {p_shape_np}"
            ) from err

        if m_np.shape != p_shape_np:
            raise ValueError(
                f"Cannot broadcast cloud_mask of shape {mask_np.shape} to {p_shape_np}"
            )

        mask_cloud_np = m_np
        mask_clear_np = ~m_np

        if np.any(mask_cloud_np):
            diff_cloud_np = diff_np[mask_cloud_np]
            mae_cloud = float(np.mean(np.abs(diff_cloud_np)))
            mse_cloud = float(np.mean(diff_cloud_np**2))
            if mse_cloud <= 0.0:
                psnr_cloud = 100.0
            else:
                psnr_cloud = float(
                    min(10.0 * math.log10((data_range**2) / (mse_cloud + eps)), 100.0)
                )
        else:
            mae_cloud = float("nan")
            psnr_cloud = float("nan")

        if np.any(mask_clear_np):
            diff_clear_np = diff_np[mask_clear_np]
            mae_clear = float(np.mean(np.abs(diff_clear_np)))
            mse_clear = float(np.mean(diff_clear_np**2))
            if mse_clear <= 0.0:
                psnr_clear = 100.0
            else:
                psnr_clear = float(
                    min(10.0 * math.log10((data_range**2) / (mse_clear + eps)), 100.0)
                )
        else:
            mae_clear = float("nan")
            psnr_clear = float("nan")

    return {
        "MAE_cloud": mae_cloud,
        "PSNR_cloud": psnr_cloud,
        "MAE_clear": mae_clear,
        "PSNR_clear": psnr_clear,
    }


class DifferentiableSSIM(nn.Module):
    """Differentiable Structural Similarity Index Measure (SSIM) for PyTorch tensors.

    Computes Wang et al. (2004) SSIM across spatial and multispectral channels
    while strictly preserving gradients with respect to predicted inputs.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if data_range <= 0.0:
            raise ValueError(f"data_range must be positive, got {data_range}")

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (k1 * data_range) ** 2
        self.C2 = (k2 * data_range) ** 2
        self.pad = window_size // 2

        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        kernel2d = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel2d)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute differentiable SSIM scalar in [-1.0, 1.0].

        Args:
            pred: Predicted tensor shaped (N, C, H, W) or (C, H, W).
            target: Ground-truth target tensor of matching shape.

        Returns:
            Scalar PyTorch float tensor representing mean SSIM.
        """
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        elif pred.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {pred.shape}")

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} != target {target.shape}")

        N, C, H, W = pred.shape
        if min(H, W) < self.window_size:
            raise ValueError(
                f"Image spatial extent ({H}, {W}) is smaller than window_size ({self.window_size})"
            )

        # Flatten (N, C, H, W) to (N*C, 1, H, W) for depthwise 2D convolution
        x = pred.reshape(N * C, 1, H, W)
        y = target.reshape(N * C, 1, H, W)
        kernel = cast(torch.Tensor, self.kernel.to(dtype=pred.dtype, device=pred.device))

        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        y_pad = F.pad(y, (self.pad, self.pad, self.pad, self.pad), mode="reflect")

        mu_x = F.conv2d(x_pad, kernel)
        mu_y = F.conv2d(y_pad, kernel)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = torch.clamp(F.conv2d(x_pad**2, kernel) - mu_x_sq, min=0.0)
        sigma_y_sq = torch.clamp(F.conv2d(y_pad**2, kernel) - mu_y_sq, min=0.0)
        sigma_xy = F.conv2d(x_pad * y_pad, kernel) - mu_xy

        num = (2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)
        den = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)

        return (num / den).mean()


class CompoundL1SSIMLoss(nn.Module):
    """Compound reconstruction loss combining L1 error with Structural Similarity.

    Formulation:
        L_total = L1(pred, target) + lambda * (1.0 - SSIM(pred, target))

    Args:
        ssim_weight: Weight lambda for the SSIM term (default: 0.2).
        data_range: Maximum pixel dynamic range (default: 1.0 for normalized reflectance).
        window_size: Window dimension for SSIM local statistics (default: 11).
        sigma: Standard deviation for Gaussian window (default: 1.5).
    """

    def __init__(
        self,
        ssim_weight: float = 0.2,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if ssim_weight < 0.0:
            raise ValueError(f"ssim_weight must be non-negative, got {ssim_weight}")
        self.ssim_weight = float(ssim_weight)
        self.l1_loss = nn.L1Loss()
        self.ssim_module = DifferentiableSSIM(
            window_size=window_size,
            sigma=sigma,
            data_range=data_range,
        )
        self.latest_components: Dict[str, float] = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute compound loss and optionally return individual components.

        Args:
            pred: Predicted tensor (N, C, H, W).
            target: Ground-truth reference tensor (N, C, H, W).
            return_components: If True, returns tuple of (total_loss, components_dict).

        Returns:
            total_loss tensor (if return_components is False), or
            (total_loss, components_dict) tuple (if return_components is True).
        """
        l1 = self.l1_loss(pred, target)
        if self.ssim_weight > 0.0:
            ssim_val = self.ssim_module(pred, target)
            loss_ssim = 1.0 - ssim_val
            total_loss = l1 + self.ssim_weight * loss_ssim
        else:
            ssim_val = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
            loss_ssim = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            total_loss = l1

        self.latest_components = {
            "loss_total": float(total_loss.detach().item()),
            "loss_l1": float(l1.detach().item()),
            "loss_ssim": float(loss_ssim.detach().item()),
            "ssim_val": float(ssim_val.detach().item()),
        }

        if return_components:
            return total_loss, self.latest_components
        return total_loss
