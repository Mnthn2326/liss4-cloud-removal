"""Unit tests for evaluation CLI and metrics generation."""

import json
from pathlib import Path

import torch

from cloud_removal.evaluate import run_evaluation
from cloud_removal.models.unet import UNet


def test_evaluate_run_with_checkpoint(tmp_path: Path) -> None:
    """Test run_evaluation on a small subset generating structured val_metrics.json."""
    checkpoint_path = tmp_path / "test_model.pt"
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_mae": 0.05,
            "val_psnr": 28.0,
            "bands": [4, 3, 2],
            "band_names": ["B04", "B03", "B02"],
            "composite_mode": "rgb",
            "model_config": model.get_config(),
        },
        checkpoint_path,
    )

    manifest_path = Path("data/raw/sen12mscr/manifest.json")
    if not manifest_path.is_file():
        manifest_path = Path("data/manifests/sen12mscr_manifest.json")

    output_report = tmp_path / "val_metrics.json"

    report = run_evaluation(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        split="val",
        max_samples=4,
        device_str="cpu",
        output_report_path=output_report,
    )

    metrics = report["metrics"]
    assert "mae" in metrics
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "sam" in metrics
    assert "baseline_mae" in metrics
    assert "baseline_psnr" in metrics

    assert output_report.is_file()
    with open(output_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metrics" in data
    assert "dataset" in data
    assert "model_architecture" in data
    assert data["metrics"]["mae"] == metrics["mae"]
