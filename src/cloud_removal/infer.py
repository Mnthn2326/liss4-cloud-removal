"""Inference and visual evaluation utility for U-Net cloud removal baseline.

Loads trained model weights, runs evaluation on cloudy input patches, computes
quantitative metrics (MAE, PSNR, regional cloud/clear metrics), and renders
side-by-side comparison figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend suitable for headless execution
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from cloud_removal.metrics import compute_mae, compute_psnr, compute_regional_metrics
from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import SEN12MSCRDataset


def generate_tile_coordinates(
    height: int,
    width: int,
    tile_size: int = 256,
    overlap: int = 32,
) -> List[Tuple[int, int, int, int]]:
    """Generate deterministic tile coordinates covering an entire scene.

    Ensures full spatial coverage, handles rectangular dimensions, and
    guarantees that the final row and column reach the scene boundaries.

    Args:
        height: Scene height in pixels.
        width: Scene width in pixels.
        tile_size: Spatial tile size in pixels (both height and width).
        overlap: Spatial overlap in pixels between adjacent tiles.

    Returns:
        List of tuples: (row_offset, col_offset, window_height, window_width).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Scene dimensions must be positive, got ({height}, {width})")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(
            f"tile_size ({tile_size}) must be strictly greater than overlap ({overlap})"
        )

    # Generate row offsets
    if height <= tile_size:
        row_offsets = [0]
    else:
        row_offsets = []
        r = 0
        while r + tile_size < height:
            row_offsets.append(r)
            r += stride
        row_offsets.append(max(0, height - tile_size))
        row_offsets = sorted(list(set(row_offsets)))

    # Generate column offsets
    if width <= tile_size:
        col_offsets = [0]
    else:
        col_offsets = []
        c = 0
        while c + tile_size < width:
            col_offsets.append(c)
            c += stride
        col_offsets.append(max(0, width - tile_size))
        col_offsets = sorted(list(set(col_offsets)))

    tiles: List[Tuple[int, int, int, int]] = []
    for r in row_offsets:
        h_win = min(tile_size, height - r)
        for c in col_offsets:
            w_win = min(tile_size, width - c)
            tiles.append((r, c, h_win, w_win))

    return tiles


def generate_blending_window(
    height: int,
    width: int,
    floor: float = 0.05,
) -> np.ndarray:
    """Generate a 2D smooth weighting window with a strictly positive floor.

    Uses a raised cosine / Hann window formulation with a non-zero floor to
    guarantee that every valid pixel receives positive accumulated weight,
    preventing division-by-zero at image and tile boundaries.

    Args:
        height: Window height in pixels.
        width: Window width in pixels.
        floor: Minimum weight floor to guarantee strictly positive weight (default: 0.05).

    Returns:
        2D float32 array of shape (height, width) with values bounded in [floor^2, 1.0].
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Dimensions must be positive, got ({height}, {width})")

    if height == 1 and width == 1:
        return np.ones((1, 1), dtype=np.float32)

    # 1D Hann window formulation with positive floor
    if height == 1:
        y = np.ones(1, dtype=np.float32)
    else:
        y_raw = np.sin(np.pi * (np.arange(height, dtype=np.float32) + 0.5) / height) ** 2
        y = (floor + (1.0 - floor) * y_raw).astype(np.float32)

    if width == 1:
        x = np.ones(1, dtype=np.float32)
    else:
        x_raw = np.sin(np.pi * (np.arange(width, dtype=np.float32) + 0.5) / width) ** 2
        x = (floor + (1.0 - floor) * x_raw).astype(np.float32)

    w2d = np.outer(y, x).astype(np.float32)
    return w2d


def predict_tiled_array(
    model: torch.nn.Module,
    cloudy: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Execute seamless tiled inference on an in-memory 3D array [C, H, W].

    Applies overlap blending with a 2D Hann window to reduce boundary seams
    and reconstructs the entire scene at original dimensions.

    Args:
        model: Trained cloud-removal model in eval mode.
        cloudy: Input array shaped [C, H, W] in physical reflectance [0.0, 1.0].
        tile_size: Size of square tiles.
        overlap: Overlap between neighboring tiles.
        device: Execution device (CPU or CUDA).
        batch_size: Batch size for tile forward passes.

    Returns:
        Reconstructed cloud-free array [C, H, W] in range [0.0, 1.0].
    """
    if cloudy.ndim != 3:
        raise ValueError(f"Expected 3D array [C, H, W], got shape {cloudy.shape}")

    c_in, h, w = cloudy.shape
    tiles = generate_tile_coordinates(h, w, tile_size=tile_size, overlap=overlap)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    output_accum = np.zeros((c_in, h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    model.eval()

    # Pre-generate standard blending window
    std_window = generate_blending_window(tile_size, tile_size, floor=0.05)

    with torch.inference_mode():
        # Process in batches
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i : i + batch_size]
            batch_patches = []
            pad_infos = []

            for r, c, h_win, w_win in batch_tiles:
                patch = cloudy[:, r : r + h_win, c : c + w_win]
                # Pad if patch is smaller than tile_size (occurs only if scene < tile_size)
                pad_h = tile_size - h_win
                pad_w = tile_size - w_win
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                batch_patches.append(patch)
                pad_infos.append((h_win, w_win))

            input_t = torch.from_numpy(np.stack(batch_patches, axis=0)).to(device)
            out_t = model(input_t)
            out_np = out_t.cpu().numpy()

            for j, (r, c, h_win, w_win) in enumerate(batch_tiles):
                pred_patch = out_np[j, :, :h_win, :w_win]
                if h_win == tile_size and w_win == tile_size:
                    win_weight = std_window
                else:
                    win_weight = generate_blending_window(h_win, w_win, floor=0.05)

                output_accum[:, r : r + h_win, c : c + w_win] += pred_patch * win_weight
                weight_accum[r : r + h_win, c : c + w_win] += win_weight

    # Normalize by accumulated weights
    reconstructed = output_accum / np.maximum(weight_accum, 1e-7)
    return np.clip(reconstructed, 0.0, 1.0)


def predict_tiled_raster(
    model: torch.nn.Module,
    input_source: Union[str, Path, rasterio.io.DatasetReaderBase],
    output_target: Union[str, Path, rasterio.io.MemoryFile],
    tile_size: int = 256,
    overlap: int = 32,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    scale_factor: Optional[float] = None,
    bands: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Execute memory-bounded windowed tiled inference from raster to raster.

    Reads windows on-demand and streams reconstructed tiles using a sliding
    buffer to eliminate the need to hold the entire scene in RAM.

    Args:
        model: Trained model in eval mode.
        input_source: File path or open DatasetReader.
        output_target: File path or rasterio.MemoryFile to write GeoTIFF to.
        tile_size: Tile window dimension.
        overlap: Overlap between tiles.
        device: PyTorch device.
        batch_size: Batch size for model inference.
        scale_factor: Optional factor to divide input DN to obtain [0, 1] reflectance.
        bands: Explicit band indices to read (1-indexed).

    Returns:
        Metadata summary dictionary with dimensions, CRS, and processing info.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Context manager handling for input source
    if isinstance(input_source, (str, Path)):
        src_ctx = rasterio.open(input_source)
        close_src = True
    else:
        src_ctx = input_source
        close_src = False

    try:
        src = src_ctx
        h, w = src.height, src.width
        read_bands = list(bands) if bands is not None else list(range(1, src.count + 1))
        in_channels = len(read_bands)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": in_channels,
                "dtype": "float32",
                "nodata": None,
            }
        )

        tiles = generate_tile_coordinates(h, w, tile_size=tile_size, overlap=overlap)
        std_window = generate_blending_window(tile_size, tile_size, floor=0.05)

        # Distinct row offsets sorted
        unique_row_offsets = sorted(list(set(t[0] for t in tiles)))

        # Output context manager handling
        if isinstance(output_target, (str, Path)):
            dst_ctx = rasterio.open(output_target, "w", **out_meta)
            close_dst = True
        elif isinstance(output_target, rasterio.io.MemoryFile):
            dst_ctx = output_target.open(**out_meta)
            close_dst = True
        else:
            dst_ctx = output_target
            close_dst = False

        try:
            dst = dst_ctx

            # Sliding buffer covering active row span
            # Maximum rows needed is at most 2 * tile_size
            max_buf_h = min(h, 2 * tile_size + overlap)
            buf_pred = np.zeros((in_channels, max_buf_h, w), dtype=np.float32)
            buf_weight = np.zeros((max_buf_h, w), dtype=np.float32)
            buf_start_r = 0  # Global row index corresponding to buf[0]

            with torch.inference_mode():
                for row_idx, r_curr in enumerate(unique_row_offsets):
                    row_tiles = [t for t in tiles if t[0] == r_curr]

                    # Process row tiles in batches
                    for i in range(0, len(row_tiles), batch_size):
                        batch_tiles = row_tiles[i : i + batch_size]
                        batch_patches = []
                        pad_infos = []

                        for r, c, h_win, w_win in batch_tiles:
                            window = Window(c, r, w_win, h_win)
                            raw_tile = src.read(read_bands, window=window).astype(np.float32)

                            # Radiometric scaling
                            if scale_factor is not None:
                                raw_tile = raw_tile / scale_factor
                            elif src.dtypes[0] == "uint16" and float(raw_tile.max()) > 1.0:
                                raw_tile = raw_tile / 10000.0
                            elif src.dtypes[0] == "uint8":
                                raw_tile = raw_tile / 255.0

                            raw_tile = np.clip(raw_tile, 0.0, 1.0)

                            pad_h = tile_size - h_win
                            pad_w = tile_size - w_win
                            if pad_h > 0 or pad_w > 0:
                                raw_tile = np.pad(
                                    raw_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
                                )

                            batch_patches.append(raw_tile)
                            pad_infos.append((h_win, w_win))

                        input_t = torch.from_numpy(np.stack(batch_patches, axis=0)).to(device)
                        out_t = model(input_t)
                        out_np = out_t.cpu().numpy()

                        for j, (r, c, h_win, w_win) in enumerate(batch_tiles):
                            pred_patch = out_np[j, :, :h_win, :w_win]
                            if h_win == tile_size and w_win == tile_size:
                                win_weight = std_window
                            else:
                                win_weight = generate_blending_window(h_win, w_win, floor=0.05)

                            rel_r = r - buf_start_r
                            buf_pred[:, rel_r : rel_r + h_win, c : c + w_win] += (
                                pred_patch * win_weight
                            )
                            buf_weight[rel_r : rel_r + h_win, c : c + w_win] += win_weight

                    # Flush resolved rows to disk
                    is_last_row = row_idx == len(unique_row_offsets) - 1
                    if is_last_row:
                        r_flush_end = h
                    else:
                        r_flush_end = unique_row_offsets[row_idx + 1]

                    flush_len = r_flush_end - buf_start_r
                    if flush_len > 0:
                        norm_slice = buf_pred[:, :flush_len, :] / np.maximum(
                            buf_weight[:flush_len, :], 1e-7
                        )
                        norm_slice = np.clip(norm_slice, 0.0, 1.0)

                        dst.write(
                            norm_slice,
                            window=Window(0, buf_start_r, w, flush_len),
                        )

                        # Slide buffer up
                        remain_h = max_buf_h - flush_len
                        if remain_h > 0:
                            buf_pred[:, :remain_h, :] = buf_pred[:, flush_len:, :]
                            buf_pred[:, remain_h:, :] = 0.0
                            buf_weight[:remain_h, :] = buf_weight[flush_len:, :]
                            buf_weight[remain_h:, :] = 0.0
                        else:
                            buf_pred.fill(0.0)
                            buf_weight.fill(0.0)

                        buf_start_r = r_flush_end

        finally:
            if close_dst:
                dst_ctx.close()

    finally:
        if close_src:
            src_ctx.close()

    return {
        "height": h,
        "width": w,
        "channels": in_channels,
        "crs": str(out_meta.get("crs")),
        "transform": list(out_meta.get("transform", []))[:6],
        "tile_size": tile_size,
        "overlap": overlap,
        "total_tiles": len(tiles),
    }


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[Union[UNet, torch.nn.Module], Dict[str, Any]]:
    """Load U-Net or Pix2Pix architecture and trained weights from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        device: Device to place the model on (defaults to CPU).

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    m_cfg = checkpoint.get("model_config", {})
    model_type = m_cfg.get("model_type", "UNet")

    model: torch.nn.Module
    if model_type == "IIRSPix2PixGenerator":
        from cloud_removal.models.iirs_pix2pix import IIRSPix2PixGenerator

        model = IIRSPix2PixGenerator(
            in_channels=m_cfg.get("in_channels", 3),
            out_channels=m_cfg.get("out_channels", 3),
            rescale_reflectance=m_cfg.get("rescale_reflectance", True),
        )
    else:
        model = UNet(
            in_channels=m_cfg.get("in_channels", 3),
            out_channels=m_cfg.get("out_channels", 3),
            base_channels=m_cfg.get("base_channels", 32),
            bilinear=m_cfg.get("bilinear", False),
            use_sigmoid=m_cfg.get("use_sigmoid", True),
            prior_reflectance=m_cfg.get("prior_reflectance", None),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict_patch(
    model: Union[UNet, torch.nn.Module],
    cloudy_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run model inference on a single cloudy tensor [3, H, W] or batch [B, 3, H, W].

    Args:
        model: Trained UNet instance in eval mode.
        cloudy_tensor: Float tensor in range [0, 1].
        device: Device to execute on.

    Returns:
        Predicted cloud-free tensor with identical dimensions in range [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    if cloudy_tensor.ndim == 3:
        input_batch = cloudy_tensor.unsqueeze(0).to(device)
        squeeze_needed = True
    elif cloudy_tensor.ndim == 4:
        input_batch = cloudy_tensor.to(device)
        squeeze_needed = False
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {cloudy_tensor.shape}")

    with torch.no_grad():
        output = model(input_batch)

    if squeeze_needed:
        output = output.squeeze(0)

    return output.cpu()


def render_comparison_figure(
    cloudy: np.ndarray,
    pred: np.ndarray,
    clear: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "",
    stretch_percentiles: Optional[Tuple[float, float]] = (2.0, 98.0),
) -> Path:
    """Render and save side-by-side RGB comparison figure.

    Applies a shared reflectance contrast stretch for visualization across all panels
    so that natural surface features are clearly visible without altering the underlying
    physical reflectance metrics (which are computed on raw [0, 1] reflectance values).

    Arrays must be shaped [3, H, W] or [H, W, 3] with values in [0, 1].

    Args:
        cloudy: Cloudy input RGB array (physical reflectance [0, 1]).
        pred: Predicted reconstructed RGB array (physical reflectance [0, 1]).
        clear: Ground truth clear target RGB array (optional, physical reflectance [0, 1]).
        output_path: File path to save PNG figure.
        title_suffix: Additional text to append to title.
        stretch_percentiles: Tuple of (low_percentile, high_percentile) for shared display
            contrast stretch (default: (2.0, 98.0)). If None, raw unstretched values are shown.

    Returns:
        Path to the saved figure.
    """

    def to_hwc(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return np.clip(arr, 0.0, 1.0)

    cloudy_img = to_hwc(cloudy)
    pred_img = to_hwc(pred)
    clear_img = to_hwc(clear) if clear is not None else None

    # Quantitative metrics are strictly computed on raw physical reflectance values
    if clear is not None:
        mae_pred = compute_mae(pred, clear)
        psnr_pred = compute_psnr(pred, clear)
        mae_input = compute_mae(cloudy, clear)
        psnr_input = compute_psnr(cloudy, clear)

    # Shared display contrast stretch for visualization only
    if stretch_percentiles is not None:
        ref_arr = clear_img if clear_img is not None else pred_img
        p_low, p_high = np.percentile(ref_arr, stretch_percentiles)
        if p_high <= p_low + 1e-6:
            p_low, p_high = 0.0, 1.0

        def apply_stretch(img: np.ndarray) -> np.ndarray:
            return np.clip((img - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)

        cloudy_disp = apply_stretch(cloudy_img)
        pred_disp = apply_stretch(pred_img)
        clear_disp = apply_stretch(clear_img) if clear_img is not None else None
        stretch_info = f" (Display Stretch [{stretch_percentiles[0]:.0f}%-{stretch_percentiles[1]:.0f}%: {p_low:.3f}-{p_high:.3f}])"
    else:
        cloudy_disp = cloudy_img
        pred_disp = pred_img
        clear_disp = clear_img
        stretch_info = ""

    if clear_disp is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cloudy_disp)
        axes[0].set_title(f"Cloudy Input\n(Base MAE: {mae_input:.4f} | PSNR: {psnr_input:.1f} dB)")
        axes[0].axis("off")

        axes[1].imshow(pred_disp)
        axes[1].set_title(f"U-Net Prediction\n(MAE: {mae_pred:.4f} | PSNR: {psnr_pred:.1f} dB)")
        axes[1].axis("off")

        axes[2].imshow(clear_disp)
        axes[2].set_title("Clear Ground Truth Target\n(Sentinel-2 L1C / L2A)")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cloudy_disp)
        axes[0].set_title("Cloudy Input")
        axes[0].axis("off")

        axes[1].imshow(pred_disp)
        axes[1].set_title("U-Net Prediction")
        axes[1].axis("off")

    fig.suptitle(
        f"SEN12MS-CR Optical Cloud Removal {title_suffix}{stretch_info}".strip(),
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out_file = Path(output_path) if output_path else Path("reports/figures/inference_sample.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return out_file


def run_inference(
    checkpoint_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    split: str = "val",
    sample_index: int = 0,
    cloudy_path: Optional[Union[str, Path]] = None,
    clear_path: Optional[Union[str, Path]] = None,
    cloud_mask_path: Optional[Union[str, Path]] = None,
    output_figure: Optional[Union[str, Path]] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute inference workflow and generate evaluation report.

    Supports whole-image and regional cloud-masked (MAE_cloud, PSNR_cloud,
    MAE_clear, PSNR_clear) evaluation when a cloud opacity mask is available.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)

    sample = None
    cloud_mask_arr = None

    # Determine input data source
    if cloudy_path is not None:
        with rasterio.open(cloudy_path) as src:
            cloudy_arr = src.read([4, 3, 2]).astype(np.float32)
            cloudy_arr = np.clip(cloudy_arr, 0.0, 10000.0) / 10000.0
            cloudy_tensor = torch.from_numpy(cloudy_arr)
        patch_id = Path(cloudy_path).stem

        clear_arr = None
        clear_tensor = None
        if clear_path is not None:
            with rasterio.open(clear_path) as src_clr:
                clear_arr = src_clr.read([4, 3, 2]).astype(np.float32)
                clear_arr = np.clip(clear_arr, 0.0, 10000.0) / 10000.0
                clear_tensor = torch.from_numpy(clear_arr)
    else:
        m_path = Path(manifest_path) if manifest_path else Path("data/raw/sen12mscr/manifest.json")
        dataset = SEN12MSCRDataset.from_manifest(m_path, split=split)
        if sample_index >= len(dataset):
            raise IndexError(
                f"Sample index {sample_index} out of range for split {split} (len={len(dataset)})"
            )

        sample = dataset[sample_index]
        cloudy_tensor = sample["cloudy"]
        clear_tensor = sample["clear"]
        patch_id = sample["patch_id"]
        cloudy_arr = cloudy_tensor.numpy()
        clear_arr = clear_tensor.numpy()

    # Load cloud opacity mask if provided explicitly or in dataset sample
    if cloud_mask_path is not None:
        with rasterio.open(cloud_mask_path) as src_m:
            cloud_mask_arr = src_m.read(1).astype(np.float32)
            if float(cloud_mask_arr.max()) > 1.0:
                cloud_mask_arr = cloud_mask_arr / 255.0
    elif sample is not None and "cloud_mask" in sample:
        cm = sample["cloud_mask"]
        cloud_mask_arr = cm.numpy() if isinstance(cm, torch.Tensor) else np.asarray(cm)
        if cloud_mask_arr.ndim == 3 and cloud_mask_arr.shape[0] == 1:
            cloud_mask_arr = cloud_mask_arr[0]

    # Predict
    pred_tensor = predict_patch(model, cloudy_tensor, device=device)
    pred_arr = pred_tensor.numpy()

    # Compute metrics if clear target available
    metrics = {}
    if clear_arr is not None:
        metrics = {
            "baseline_input_mae": float(compute_mae(cloudy_arr, clear_arr)),
            "baseline_input_psnr": float(compute_psnr(cloudy_arr, clear_arr)),
            "unet_prediction_mae": float(compute_mae(pred_arr, clear_arr)),
            "unet_prediction_psnr": float(compute_psnr(pred_arr, clear_arr)),
        }
        # Compute regional cloud vs clear metrics if opacity mask available
        if cloud_mask_arr is not None:
            regional = compute_regional_metrics(pred_arr, clear_arr, cloud_mask_arr)
            metrics.update(regional)

    # Render figure
    fig_path = render_comparison_figure(
        cloudy=cloudy_arr,
        pred=pred_arr,
        clear=clear_arr,
        output_path=output_figure,
        title_suffix=f"- Patch {patch_id} ({split.capitalize()} #{sample_index})",
    )

    report = {
        "patch_id": patch_id,
        "input_shape": list(cloudy_tensor.shape),
        "output_shape": list(pred_tensor.shape),
        "prediction_min": float(pred_arr.min()),
        "prediction_max": float(pred_arr.max()),
        "prediction_mean": float(pred_arr.mean()),
        "has_cloud_mask": cloud_mask_arr is not None,
        "metrics": metrics,
        "comparison_figure": str(fig_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mae": checkpoint.get("val_mae"),
    }

    return report


def evaluate_full_split(
    checkpoint_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    split: str = "val",
    save_indices: Sequence[int] = (0, 1, 2, 3),
    output_dir: Optional[Union[str, Path]] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Evaluate trained model across the entire validation or test split.

    Computes both baseline input and model prediction metrics (macro and micro MAE & PSNR),
    as well as cloud-affected and clear-region metrics when opacity masks are present.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    m_path = Path(manifest_path) if manifest_path else Path("data/raw/sen12mscr/manifest.json")
    dataset = SEN12MSCRDataset.from_manifest(m_path, split=split)
    out_dir = Path(output_dir) if output_dir else Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_patch_maes, base_patch_psnrs = [], []
    unet_patch_maes, unet_patch_psnrs = [], []
    base_sq_errors, base_abs_errors = [], []
    unet_sq_errors, unet_abs_errors = [], []
    has_masks = False
    cloud_maes, cloud_psnrs = [], []
    clear_maes, clear_psnrs = [], []
    saved_figures = []

    print(f"Evaluating {len(dataset)} samples from {split} split on {device}...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cloudy_t = sample["cloudy"]
        clear_t = sample["clear"]
        patch_id = sample["patch_id"]

        cloudy_np = cloudy_t.numpy()
        clear_np = clear_t.numpy()

        pred_t = predict_patch(model, cloudy_t, device=device)
        pred_np = pred_t.numpy()

        # Baseline metrics
        b_mae = compute_mae(cloudy_np, clear_np)
        b_psnr = compute_psnr(cloudy_np, clear_np)
        base_patch_maes.append(b_mae)
        base_patch_psnrs.append(b_psnr)
        diff_base = cloudy_np - clear_np
        base_abs_errors.append(float(np.mean(np.abs(diff_base))))
        base_sq_errors.append(float(np.mean(diff_base**2)))

        # UNet metrics
        u_mae = compute_mae(pred_np, clear_np)
        u_psnr = compute_psnr(pred_np, clear_np)
        unet_patch_maes.append(u_mae)
        unet_patch_psnrs.append(u_psnr)
        diff_u = pred_np - clear_np
        unet_abs_errors.append(float(np.mean(np.abs(diff_u))))
        unet_sq_errors.append(float(np.mean(diff_u**2)))

        # Regional metrics if cloud_mask present
        if "cloud_mask" in sample:
            has_masks = True
            cm = sample["cloud_mask"]
            cm_np = cm.numpy() if isinstance(cm, torch.Tensor) else np.asarray(cm)
            reg_m = compute_regional_metrics(pred_np, clear_np, cm_np)
            if not math.isnan(reg_m["MAE_cloud"]):
                cloud_maes.append(reg_m["MAE_cloud"])
                cloud_psnrs.append(reg_m["PSNR_cloud"])
            if not math.isnan(reg_m["MAE_clear"]):
                clear_maes.append(reg_m["MAE_clear"])
                clear_psnrs.append(reg_m["PSNR_clear"])

        if idx in save_indices:
            fig_file = out_dir / f"inference_{split}_{idx}_{patch_id}.png"
            render_comparison_figure(
                cloudy=cloudy_np,
                pred=pred_np,
                clear=clear_np,
                output_path=fig_file,
                title_suffix=f"- Patch {patch_id} ({split.capitalize()} #{idx})",
            )
            saved_figures.append(str(fig_file))

    b_micro_mse = float(np.mean(base_sq_errors))
    u_micro_mse = float(np.mean(unet_sq_errors))

    report = {
        "split": split,
        "sample_count": len(dataset),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "baseline_cloudy_input": {
            "macro_mae": float(np.mean(base_patch_maes)),
            "macro_psnr": float(np.mean(base_patch_psnrs)),
            "micro_mae": float(np.mean(base_abs_errors)),
            "micro_psnr": float(10.0 * math.log10(1.0 / (b_micro_mse + 1e-10))),
        },
        "unet_prediction": {
            "macro_mae": float(np.mean(unet_patch_maes)),
            "macro_psnr": float(np.mean(unet_patch_psnrs)),
            "micro_mae": float(np.mean(unet_abs_errors)),
            "micro_psnr": float(10.0 * math.log10(1.0 / (u_micro_mse + 1e-10))),
        },
        "saved_figures": saved_figures,
    }

    if has_masks:
        report["regional_evaluation"] = {
            "MAE_cloud": float(np.mean(cloud_maes)) if cloud_maes else float("nan"),
            "PSNR_cloud": float(np.mean(cloud_psnrs)) if cloud_psnrs else float("nan"),
            "MAE_clear": float(np.mean(clear_maes)) if clear_maes else float("nan"),
            "PSNR_clear": float(np.mean(clear_psnrs)) if clear_psnrs else float("nan"),
        }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference using trained U-Net baseline")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_unet.pt", help="Path to checkpoint"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/raw/sen12mscr/manifest.json", help="Path to manifest"
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/train/all)")
    parser.add_argument("--index", type=int, default=0, help="Sample index in split")
    parser.add_argument(
        "--cloudy-path", type=str, default=None, help="Explicit path to cloudy GeoTIFF"
    )
    parser.add_argument(
        "--clear-path", type=str, default=None, help="Explicit path to clear GeoTIFF"
    )
    parser.add_argument(
        "--cloud-mask-path", type=str, default=None, help="Explicit path to cloud mask GeoTIFF"
    )
    parser.add_argument(
        "--output", type=str, default="reports/figures/inference_sample.png", help="Output PNG path"
    )
    parser.add_argument(
        "--eval-split", type=str, default=None, help="Evaluate entire split (e.g. val)"
    )
    parser.add_argument(
        "--save-indices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated indices to save figures",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Output directory for split figures",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Execution device (cpu/cuda/auto)"
    )

    args = parser.parse_args()

    if args.eval_split is not None:
        save_idx_list = [int(x.strip()) for x in args.save_indices.split(",") if x.strip()]
        report = evaluate_full_split(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            split=args.eval_split,
            save_indices=save_idx_list,
            output_dir=args.output_dir,
            device_str=args.device,
        )
    else:
        report = run_inference(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            split=args.split,
            sample_index=args.index,
            cloudy_path=args.cloudy_path,
            clear_path=args.clear_path,
            cloud_mask_path=args.cloud_mask_path,
            output_figure=args.output,
            device_str=args.device,
        )

    print("\nInference Report:")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
