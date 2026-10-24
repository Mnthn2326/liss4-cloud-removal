"""Synthetic cloud generation and atmospheric overlay simulation.

Provides modular, deterministic utilities for simulating synthetic cloud
opacity masks and blending them onto clear satellite imagery with realistic
spatially-varying transmission and cloud radiance scattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class CloudConfig:
    """Configuration parameters for synthetic cloud generation and blending.

    Attributes:
        min_coverage: Minimum fractional cloud coverage in the scene (0.0 to 1.0).
        max_coverage: Maximum fractional cloud coverage in the scene (0.0 to 1.0).
        min_opacity: Minimum opacity for cloudy pixels (semi-transparent haze/cirrus).
        max_opacity: Maximum peak opacity for dense opaque cloud cores (up to 1.0).
        coarse_scale_fraction: Primary cloud macro-structure scale as fraction of min dimension.
        fine_scale_fraction: High-frequency cloud turbulence scale as fraction of min dimension.
        fine_weight: Relative weight of fine-scale turbulence in mask generation.
        cloud_dn_min: Minimum digital number for synthetic cloud radiance (e.g. 700 for 10-bit).
        cloud_dn_max: Maximum digital number for saturated cloud cores (e.g. 1023 for 10-bit).
        seed: Default deterministic random seed (None for non-deterministic).
    """

    min_coverage: float = 0.15
    max_coverage: float = 0.50
    min_opacity: float = 0.20
    max_opacity: float = 1.00
    coarse_scale_fraction: float = 0.18
    fine_scale_fraction: float = 0.05
    fine_weight: float = 0.25
    cloud_dn_min: int = 700
    cloud_dn_max: int = 1023
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_coverage <= self.max_coverage <= 1.0):
            raise ValueError(
                f"Invalid coverage range: [{self.min_coverage}, {self.max_coverage}]. "
                "Must satisfy 0.0 <= min_coverage <= max_coverage <= 1.0."
            )
        if not (0.0 <= self.min_opacity <= self.max_opacity <= 1.0):
            raise ValueError(
                f"Invalid opacity range: [{self.min_opacity}, {self.max_opacity}]. "
                "Must satisfy 0.0 <= min_opacity <= max_opacity <= 1.0."
            )
        if self.cloud_dn_min < 0 or self.cloud_dn_max < self.cloud_dn_min:
            raise ValueError(
                f"Invalid cloud DN range: [{self.cloud_dn_min}, {self.cloud_dn_max}]. "
                "Must satisfy 0 <= cloud_dn_min <= cloud_dn_max."
            )


def generate_cloud_mask(
    height: int,
    width: int,
    coverage: Optional[float] = None,
    config: Optional[CloudConfig] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a continuous 2D synthetic cloud opacity mask.

    Uses multi-scale Gaussian-smoothed stochastic fields to simulate natural
    cloud macro-shapes with feathered, semi-transparent edges and turbulent borders.

    Args:
        height: Raster height in pixels. Must be positive.
        width: Raster width in pixels. Must be positive.
        coverage: Target fractional cloud coverage in [0.0, 1.0]. If None, sampled
            uniformly from config.min_coverage to config.max_coverage.
        config: Optional CloudConfig instance. Defaults to default CloudConfig().
        seed: Random seed for deterministic generation. Overrides config.seed if provided.

    Returns:
        2D numpy array of shape (height, width) with float32 values in [0.0, max_opacity].
        0.0 represents completely clear sky; positive values represent cloud opacity.

    Raises:
        ValueError: If dimensions are non-positive or coverage is outside [0.0, 1.0].
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Dimensions must be strictly positive, received height={height}, width={width}"
        )

    cfg = config if config is not None else CloudConfig()
    effective_seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(effective_seed)

    if coverage is None:
        target_cov = float(rng.uniform(cfg.min_coverage, cfg.max_coverage))
    else:
        if not (0.0 <= coverage <= 1.0):
            raise ValueError(f"Coverage must be in [0.0, 1.0], received {coverage}")
        target_cov = coverage

    # Trivial edge cases
    if target_cov == 0.0:
        return np.zeros((height, width), dtype=np.float32)
    if target_cov == 1.0:
        return np.full((height, width), cfg.max_opacity, dtype=np.float32)

    min_dim = min(height, width)
    sigma_coarse = max(2.0, min_dim * cfg.coarse_scale_fraction)
    sigma_fine = max(1.0, min_dim * cfg.fine_scale_fraction)

    # Multi-scale smoothed noise
    noise_coarse = rng.standard_normal((height, width))
    noise_fine = rng.standard_normal((height, width))

    field_coarse = gaussian_filter(noise_coarse, sigma=sigma_coarse, mode="reflect")
    field_fine = gaussian_filter(noise_fine, sigma=sigma_fine, mode="reflect")

    combined = field_coarse + cfg.fine_weight * field_fine

    # Normalize to [0.0, 1.0]
    c_min = float(np.min(combined))
    c_max = float(np.max(combined))
    span = c_max - c_min
    if span < 1e-7:
        return np.zeros((height, width), dtype=np.float32)

    norm_field = (combined - c_min) / span

    # Calculate quantile threshold to match target coverage
    threshold = float(np.quantile(norm_field, 1.0 - target_cov))
    denom = 1.0 - threshold
    if denom < 1e-7:
        denom = 1e-7

    # Soft linear ramp from threshold to peak, scaled between min_opacity and max_opacity
    mask = np.clip((norm_field - threshold) / denom, 0.0, 1.0)
    cloudy_pixels = mask > 0.0

    # Scale non-zero cloudy pixels to [min_opacity, max_opacity]
    scaled_mask = np.zeros_like(mask, dtype=np.float32)
    opacity_range = cfg.max_opacity - cfg.min_opacity
    scaled_mask[cloudy_pixels] = cfg.min_opacity + mask[cloudy_pixels] * opacity_range

    return scaled_mask.astype(np.float32)


def blend_synthetic_clouds(
    clear_image: np.ndarray,
    cloud_mask: np.ndarray,
    config: Optional[CloudConfig] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Blend synthetic clouds onto a clear multi-spectral or single-band raster.

    Implements a simplified radiative transfer transmission model:
        I_cloudy(x) = (1 - alpha(x)) * I_clear(x) + alpha(x) * C(x)
    where alpha(x) is the cloud opacity mask in [0, 1] and C(x) is the spatially
    varying top-of-cloud scattered radiance.

    Args:
        clear_image: Source clear raster array. Shape must be (C, H, W), (H, W, C), or (H, W).
        cloud_mask: 2D cloud opacity array of shape (H, W) with values in [0.0, 1.0].
        config: Optional CloudConfig instance.
        seed: Random seed for cloud radiance field variation.

    Returns:
        Cloudy image array having identical shape, channel ordering, and data type as clear_image.

    Raises:
        ValueError: If spatial dimensions (H, W) between clear_image and cloud_mask do not match.
    """
    cfg = config if config is not None else CloudConfig()
    effective_seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(effective_seed)

    orig_dtype = clear_image.dtype
    is_channel_first = False
    is_2d = False

    if clear_image.ndim == 2:
        is_2d = True
        img_h, img_w = clear_image.shape
        clear_3d = clear_image[np.newaxis, :, :]  # (1, H, W)
        is_channel_first = True
    elif clear_image.ndim == 3:
        if (
            clear_image.shape[1] == cloud_mask.shape[0]
            and clear_image.shape[2] == cloud_mask.shape[1]
        ):
            # (C, H, W)
            is_channel_first = True
            img_h, img_w = clear_image.shape[1], clear_image.shape[2]
            clear_3d = clear_image
        elif (
            clear_image.shape[0] == cloud_mask.shape[0]
            and clear_image.shape[1] == cloud_mask.shape[1]
        ):
            # (H, W, C)
            is_channel_first = False
            img_h, img_w = clear_image.shape[0], clear_image.shape[1]
            clear_3d = np.transpose(clear_image, (2, 0, 1))  # to (C, H, W)
        else:
            raise ValueError(
                f"Spatial dimensions of clear_image {clear_image.shape} do not match "
                f"cloud_mask {cloud_mask.shape}."
            )
    else:
        raise ValueError(f"clear_image must have 2 or 3 dimensions, found {clear_image.ndim}")

    if (img_h, img_w) != cloud_mask.shape:
        raise ValueError(
            f"Image spatial shape ({img_h}, {img_w}) does not match mask shape {cloud_mask.shape}"
        )

    num_channels = clear_3d.shape[0]
    alpha = np.clip(cloud_mask, 0.0, 1.0).astype(np.float32)  # (H, W)
    transmission = 1.0 - alpha  # (H, W)

    # Determine max dynamic range for integer types
    if np.issubdtype(orig_dtype, np.integer):
        dtype_info = np.iinfo(orig_dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)
    else:
        max_val = 1.0 if float(np.max(clear_image)) <= 1.0 else float(np.max(clear_image))
        min_val = 0.0

    # Spatially varying cloud radiance field across channels
    # Thick clouds reflect high radiance across all visual/NIR bands (near-white / spectrally flat)
    cloud_radiance = np.zeros((num_channels, img_h, img_w), dtype=np.float32)
    for c in range(num_channels):
        rad_noise = rng.standard_normal((img_h, img_w))
        rad_smooth = gaussian_filter(
            rad_noise, sigma=max(2.0, min(img_h, img_w) * 0.1), mode="reflect"
        )
        rad_norm = (rad_smooth - rad_smooth.min()) / (rad_smooth.max() - rad_smooth.min() + 1e-7)

        dn_span = float(cfg.cloud_dn_max - cfg.cloud_dn_min)
        cloud_radiance[c] = cfg.cloud_dn_min + rad_norm * dn_span

    # Apply transmission and scattering blending:
    # cloudy = transmission * clear + alpha * cloud_radiance
    clear_float = clear_3d.astype(np.float32)
    cloudy_float = (
        transmission[np.newaxis, :, :] * clear_float + alpha[np.newaxis, :, :] * cloud_radiance
    )

    # Clip within valid range and cast back to original data type
    clipped = np.clip(cloudy_float, min_val, max_val)

    if np.issubdtype(orig_dtype, np.integer):
        cloudy_out = np.round(clipped).astype(orig_dtype)
    else:
        cloudy_out = clipped.astype(orig_dtype)

    # Restore original shape and layout
    if is_2d:
        return cloudy_out[0]
    if not is_channel_first:
        return np.transpose(cloudy_out, (1, 2, 0))
    return cloudy_out


class SyntheticCloudGenerator:
    """Stateful generator for reproducible synthetic cloud generation and blending."""

    def __init__(self, config: Optional[CloudConfig] = None) -> None:
        self.config = config if config is not None else CloudConfig()

    def generate_mask(
        self,
        height: int,
        width: int,
        coverage: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a synthetic cloud mask."""
        return generate_cloud_mask(
            height=height,
            width=width,
            coverage=coverage,
            config=self.config,
            seed=seed,
        )

    def apply(
        self,
        clear_image: np.ndarray,
        coverage: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply synthetic clouds to a clear image.

        Args:
            clear_image: Clear raster array of shape (C, H, W), (H, W, C), or (H, W).
            coverage: Optional fractional cloud coverage.
            seed: Optional random seed.

        Returns:
            Tuple of (cloudy_image, cloud_mask).
        """
        if clear_image.ndim == 2:
            h, w = clear_image.shape
        elif clear_image.ndim == 3:
            # Check whether (C, H, W) or (H, W, C)
            if clear_image.shape[0] in (1, 3, 4) and clear_image.shape[1] > 4:
                h, w = clear_image.shape[1], clear_image.shape[2]
            else:
                h, w = clear_image.shape[0], clear_image.shape[1]
        else:
            raise ValueError(f"clear_image must have 2 or 3 dims, got {clear_image.ndim}")

        mask = self.generate_mask(height=h, width=w, coverage=coverage, seed=seed)
        cloudy = blend_synthetic_clouds(
            clear_image=clear_image, cloud_mask=mask, config=self.config, seed=seed
        )
        return cloudy, mask
