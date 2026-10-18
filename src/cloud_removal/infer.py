"""Inference and visual evaluation utility for U-Net cloud removal baseline.

Loads trained model weights, runs evaluation on cloudy input patches, computes
quantitative metrics (MAE, PSNR), and renders side-by-side comparison figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless execution
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from cloud_removal.metrics import compute_mae, compute_psnr
from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import SEN12MSCRConfig, SEN12MSCRDataset


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[UNet, Dict[str, Any]]:
    """Load U-Net architecture and trained weights from a checkpoint file.

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
    model: UNet,
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

    fig.suptitle(f"SEN12MS-CR Optical Cloud Removal {title_suffix}{stretch_info}".strip(), fontsize=13, y=1.02)
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
    output_figure: Optional[Union[str, Path]] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute inference workflow and generate evaluation report."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)

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
            raise IndexError(f"Sample index {sample_index} out of range for split {split} (len={len(dataset)})")

        sample = dataset[sample_index]
        cloudy_tensor = sample["cloudy"]
        clear_tensor = sample["clear"]
        patch_id = sample["patch_id"]
        cloudy_arr = cloudy_tensor.numpy()
        clear_arr = clear_tensor.numpy()

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

    Computes both baseline input and model prediction metrics (macro and micro MAE & PSNR)
    and saves side-by-side figures for selected indices.
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
        base_sq_errors.append(float(np.mean(diff_base ** 2)))

        # UNet metrics
        u_mae = compute_mae(pred_np, clear_np)
        u_psnr = compute_psnr(pred_np, clear_np)
        unet_patch_maes.append(u_mae)
        unet_patch_psnrs.append(u_psnr)
        diff_u = pred_np - clear_np
        unet_abs_errors.append(float(np.mean(np.abs(diff_u))))
        unet_sq_errors.append(float(np.mean(diff_u ** 2)))

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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference using trained U-Net baseline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pt", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="data/raw/sen12mscr/manifest.json", help="Path to manifest")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/train/all)")
    parser.add_argument("--index", type=int, default=0, help="Sample index in split")
    parser.add_argument("--cloudy-path", type=str, default=None, help="Explicit path to cloudy GeoTIFF")
    parser.add_argument("--clear-path", type=str, default=None, help="Explicit path to clear GeoTIFF")
    parser.add_argument("--output", type=str, default="reports/figures/inference_sample.png", help="Output PNG path")
    parser.add_argument("--eval-split", type=str, default=None, help="Evaluate entire split (e.g. val)")
    parser.add_argument("--save-indices", type=str, default="0,1,2,3", help="Comma-separated indices to save figures")
    parser.add_argument("--output-dir", type=str, default="reports/figures", help="Output directory for split figures")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu/cuda/auto)")

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
            output_figure=args.output,
            device_str=args.device,
        )

    print("\nInference Report:")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
