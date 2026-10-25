"""Reproducible evaluation pipeline for trained cloud-removal models.

Evaluates model checkpoints against specified dataset partitions and generates
reports/val_metrics.json with MAE, PSNR, SSIM, SAM, and regional metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml

from cloud_removal.infer import load_model_from_checkpoint, predict_patch
from cloud_removal.metrics import (
    compute_mae,
    compute_psnr,
    compute_regional_metrics,
    compute_sam,
    compute_ssim,
)
from cloud_removal.sen12mscr import SEN12MSCRConfig, SEN12MSCRDataset


def run_evaluation(
    checkpoint_path: Union[str, Path],
    manifest_path: Union[str, Path] = "data/raw/sen12mscr/manifest.json",
    config_path: Optional[Union[str, Path]] = None,
    split: str = "val",
    max_samples: Optional[int] = None,
    output_report_path: Union[str, Path] = "reports/val_metrics.json",
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute evaluation and generate structured metrics report.

    Args:
        checkpoint_path: Path to model checkpoint file (.pt).
        manifest_path: Path to dataset manifest.json.
        config_path: Optional path to training config file.
        split: Dataset split to evaluate ('val', 'train', or 'all').
        max_samples: Optional maximum number of samples to evaluate.
        output_report_path: Path where JSON evaluation report will be saved.
        device_str: Execution device ('auto', 'cpu', 'cuda').

    Returns:
        Structured evaluation report dictionary.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    m_file = Path(manifest_path)
    if not m_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {m_file}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model and checkpoint metadata
    model, checkpoint = load_model_from_checkpoint(ckpt_file, device=device)
    model.eval()

    # Determine dataset configuration
    ds_config = None
    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg.get("dataset", {})
        ds_config = SEN12MSCRConfig(
            bands=tuple(ds_cfg.get("bands", [4, 3, 2])),
            band_names=tuple(ds_cfg.get("band_names", ["B04", "B03", "B02"])),
            normalize=ds_cfg.get("normalize", True),
            clip_max=float(ds_cfg.get("clip_max", 10000.0)),
            seed=int(ds_cfg.get("seed", 42)),
            augment=False,  # Never augment evaluation split
        )
    elif "config" in checkpoint and "dataset" in checkpoint["config"]:
        ds_cfg = checkpoint["config"]["dataset"]
        ds_config = SEN12MSCRConfig(
            bands=tuple(ds_cfg.get("bands", [4, 3, 2])),
            band_names=tuple(ds_cfg.get("band_names", ["B04", "B03", "B02"])),
            normalize=ds_cfg.get("normalize", True),
            clip_max=float(ds_cfg.get("clip_max", 10000.0)),
            seed=int(ds_cfg.get("seed", 42)),
            augment=False,
        )

    dataset = SEN12MSCRDataset.from_manifest(m_file, split=split, config=ds_config)
    n_total = len(dataset)
    if max_samples is not None and max_samples < n_total:
        dataset.samples = dataset.samples[:max_samples]

    n_eval = len(dataset)
    print(
        f"Evaluating {n_eval} samples (split='{split}') with model from {ckpt_file.name} on {device}..."
    )

    # Metric accumulators for model predictions
    unet_maes: List[float] = []
    unet_psnrs: List[float] = []
    unet_ssims: List[float] = []
    unet_sams: List[float] = []

    # Metric accumulators for input baseline (cloudy vs clear)
    base_maes: List[float] = []
    base_psnrs: List[float] = []
    base_ssims: List[float] = []
    base_sams: List[float] = []

    # Regional metric accumulators
    has_masks = False
    cloud_maes: List[float] = []
    cloud_psnrs: List[float] = []
    clear_maes: List[float] = []
    clear_psnrs: List[float] = []

    for idx in range(n_eval):
        sample = dataset[idx]
        cloudy_t = sample["cloudy"]
        clear_t = sample["clear"]

        cloudy_np = cloudy_t.numpy()
        clear_np = clear_t.numpy()

        # Model prediction
        pred_t = predict_patch(model, cloudy_t, device=device)
        pred_np = pred_t.numpy()

        # Calculate metrics for prediction vs target
        u_mae = compute_mae(pred_np, clear_np)
        u_psnr = compute_psnr(pred_np, clear_np)
        u_ssim = compute_ssim(pred_np, clear_np)
        u_sam = compute_sam(pred_np, clear_np)

        unet_maes.append(u_mae)
        unet_psnrs.append(u_psnr)
        unet_ssims.append(u_ssim)
        unet_sams.append(u_sam)

        # Calculate metrics for baseline cloudy input vs target
        b_mae = compute_mae(cloudy_np, clear_np)
        b_psnr = compute_psnr(cloudy_np, clear_np)
        b_ssim = compute_ssim(cloudy_np, clear_np)
        b_sam = compute_sam(cloudy_np, clear_np)

        base_maes.append(b_mae)
        base_psnrs.append(b_psnr)
        base_ssims.append(b_ssim)
        base_sams.append(b_sam)

        # Regional metrics if cloud_mask present
        if "cloud_mask" in sample:
            has_masks = True
            c_mask = sample["cloud_mask"]
            cm_np = c_mask.numpy() if isinstance(c_mask, torch.Tensor) else np.asarray(c_mask)
            reg = compute_regional_metrics(pred_np, clear_np, cm_np)
            if not math.isnan(reg["MAE_cloud"]):
                cloud_maes.append(reg["MAE_cloud"])
                cloud_psnrs.append(reg["PSNR_cloud"])
            if not math.isnan(reg["MAE_clear"]):
                clear_maes.append(reg["MAE_clear"])
                clear_psnrs.append(reg["PSNR_clear"])

    mean_unet_mae = float(np.mean(unet_maes)) if unet_maes else 0.0
    mean_unet_psnr = float(np.mean(unet_psnrs)) if unet_psnrs else 0.0
    mean_unet_ssim = float(np.mean(unet_ssims)) if unet_ssims else 0.0
    mean_unet_sam = float(np.mean(unet_sams)) if unet_sams else 0.0

    mean_base_mae = float(np.mean(base_maes)) if base_maes else 0.0
    mean_base_psnr = float(np.mean(base_psnrs)) if base_psnrs else 0.0
    mean_base_ssim = float(np.mean(base_ssims)) if base_ssims else 0.0
    mean_base_sam = float(np.mean(base_sams)) if base_sams else 0.0

    model_any: Any = model
    model_cfg = model_any.get_config() if hasattr(model, "get_config") else {}
    model_type = model_cfg.get("model_type", "UNet") if isinstance(model_cfg, dict) else "UNet"
    total_params = (
        model_any.count_parameters()
        if hasattr(model, "count_parameters")
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    report_payload: Dict[str, Any] = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_checkpoint": str(ckpt_file),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mae": checkpoint.get("val_mae"),
        "model_architecture": {
            "model_type": model_type,
            "in_channels": getattr(model, "in_channels", None),
            "out_channels": getattr(model, "out_channels", None),
            "base_channels": getattr(model, "base_channels", None),
            "total_trainable_params": total_params,
        },
        "dataset": {
            "manifest_path": str(m_file),
            "split": split,
            "sample_count": n_eval,
            "total_available_in_split": n_total,
            "bands": list(dataset.config.bands) if dataset.config.bands is not None else [],
            "band_names": (
                list(dataset.config.band_names) if dataset.config.band_names is not None else []
            ),
            "composite_mode": dataset.config.composite_mode,
        },
        "metrics": {
            "mae": mean_unet_mae,
            "psnr": mean_unet_psnr,
            "ssim": mean_unet_ssim,
            "sam": mean_unet_sam,
            "baseline_mae": mean_base_mae,
            "baseline_psnr": mean_base_psnr,
            "baseline_ssim": mean_base_ssim,
            "baseline_sam": mean_base_sam,
            "mae_reduction_pct": float(
                (mean_base_mae - mean_unet_mae) / (mean_base_mae + 1e-8) * 100.0
            ),
            "psnr_improvement_db": float(mean_unet_psnr - mean_base_psnr),
            "ssim_improvement": float(mean_unet_ssim - mean_base_ssim),
            "sam_reduction_rad": float(mean_base_sam - mean_unet_sam),
        },
    }

    if has_masks:
        report_payload["regional_metrics"] = {
            "MAE_cloud": float(np.mean(cloud_maes)) if cloud_maes else float("nan"),
            "PSNR_cloud": float(np.mean(cloud_psnrs)) if cloud_psnrs else float("nan"),
            "MAE_clear": float(np.mean(clear_maes)) if clear_maes else float("nan"),
            "PSNR_clear": float(np.mean(clear_psnrs)) if clear_psnrs else float("nan"),
        }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)

    print(f"Evaluation report successfully written to {out_file}")
    print(f"  Model MAE:  {mean_unet_mae:.5f} (Baseline: {mean_base_mae:.5f})")
    print(f"  Model PSNR: {mean_unet_psnr:.2f} dB (Baseline: {mean_base_psnr:.2f} dB)")
    print(f"  Model SSIM: {mean_unet_ssim:.4f} (Baseline: {mean_base_ssim:.4f})")
    print(f"  Model SAM:  {mean_unet_sam:.4f} rad (Baseline: {mean_base_sam:.4f} rad)")

    return report_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for model evaluation stage."""
    parser = argparse.ArgumentParser(description="Evaluate cloud-removal model on a dataset split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/compact/best_compact_unet.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/raw/sen12mscr/manifest.json",
        help="Path to dataset manifest.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to training configuration YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate ('val', 'train', or 'all')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of evaluation samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/val_metrics.json",
        help="Output path for JSON evaluation report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda')",
    )

    args = parser.parse_args(argv)
    try:
        run_evaluation(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            config_path=args.config,
            split=args.split,
            max_samples=args.max_samples,
            output_report_path=args.output,
            device_str=args.device,
        )
        return 0
    except Exception as err:
        print(f"Evaluation failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
