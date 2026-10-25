"""Held-out RICE2 evaluation with whole-image and cloud-region metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from cloud_removal.infer import load_model_from_checkpoint, predict_patch
from cloud_removal.metrics import compute_mae, compute_psnr, compute_ssim
from cloud_removal.rice2 import RICE2Config, RICE2Dataset


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Optional[float]:
    selected = np.broadcast_to(mask.astype(bool), prediction.shape)
    return float(np.mean(np.abs(prediction[selected] - target[selected]))) if selected.any() else None


def evaluate_rice2(
    checkpoint_path: Union[str, Path],
    manifest_path: Union[str, Path] = "data/manifests/rice2_manifest.json",
    output_path: Union[str, Path] = "reports/rice2_test_metrics.json",
    split: str = "test",
    max_samples: Optional[int] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Evaluate a checkpoint without augmentation on the selected RICE2 partition."""
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu" if device_str == "auto" else device_str)
    checkpoint_file = Path(checkpoint_path)
    model, checkpoint = load_model_from_checkpoint(checkpoint_file, device=device)
    dataset = RICE2Dataset.from_manifest(manifest_path, split=split, config=RICE2Config(augment=False))
    if max_samples is not None:
        dataset.samples = dataset.samples[:max_samples]
    result: Dict[str, List[float]] = {key: [] for key in ("mae", "psnr", "ssim", "baseline_mae", "baseline_psnr", "baseline_ssim", "cloud_mae", "baseline_cloud_mae", "clear_mae", "baseline_clear_mae")}
    for index in range(len(dataset)):
        sample = dataset[index]
        cloudy = sample["cloudy"].numpy()
        clear = sample["clear"].numpy()
        mask = sample["cloud_mask"].numpy()
        prediction = predict_patch(model, sample["cloudy"], device=device).numpy()
        result["mae"].append(compute_mae(prediction, clear))
        result["psnr"].append(compute_psnr(prediction, clear))
        result["ssim"].append(compute_ssim(prediction, clear))
        result["baseline_mae"].append(compute_mae(cloudy, clear))
        result["baseline_psnr"].append(compute_psnr(cloudy, clear))
        result["baseline_ssim"].append(compute_ssim(cloudy, clear))
        cloud = _masked_mae(prediction, clear, mask)
        baseline_cloud = _masked_mae(cloudy, clear, mask)
        clear_region = _masked_mae(prediction, clear, 1.0 - mask)
        baseline_clear = _masked_mae(cloudy, clear, 1.0 - mask)
        if cloud is not None:
            result["cloud_mae"].append(cloud)
        if baseline_cloud is not None:
            result["baseline_cloud_mae"].append(baseline_cloud)
        if clear_region is not None:
            result["clear_mae"].append(clear_region)
        if baseline_clear is not None:
            result["baseline_clear_mae"].append(baseline_clear)
    metrics = {key: _mean(value) for key, value in result.items()}
    metrics["mae_reduction_pct"] = (100.0 * (metrics["baseline_mae"] - metrics["mae"]) / metrics["baseline_mae"]) if metrics["baseline_mae"] else None
    payload = {"evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset": "RICE2", "reference_boundary": "Clear reference may be temporally separated from the cloudy observation.", "split": split, "sample_count": len(dataset), "checkpoint": str(checkpoint_file), "checkpoint_epoch": checkpoint.get("epoch"), "metrics": metrics}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a cloud-reconstruction checkpoint on RICE2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/rice2_manifest.json")
    parser.add_argument("--output", default="reports/rice2_test_metrics.json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = evaluate_rice2(args.checkpoint, args.manifest, args.output, args.split, args.max_samples, args.device)
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
