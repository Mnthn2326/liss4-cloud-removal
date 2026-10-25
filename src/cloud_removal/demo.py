"""Minimal reproducible cloud-removal demo using compact U-Net baseline.

Loads the trained compact U-Net checkpoint, loads a sample from the SEN12MS-CR
validation split, performs cloud removal inference, computes quantitative metrics
(MAE and PSNR), and saves a comparison figure rendered with a shared display stretch.

Usage:
    python demo.py
    python -m cloud_removal.demo --index 1 --output reports/figures/demo_sample1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Union

from cloud_removal.infer import run_inference


def run_demo(
    checkpoint_path: Union[str, Path] = "checkpoints/compact/best_compact_unet.pt",
    manifest_path: Union[str, Path] = "data/raw/sen12mscr/manifest.json",
    split: str = "val",
    sample_index: int = 0,
    output_figure: Union[str, Path] = "reports/figures/demo_comparison.png",
    device: str = "auto",
) -> Dict[str, Any]:
    """Execute the minimal end-to-end cloud-removal demo.

    Args:
        checkpoint_path: Path to trained U-Net checkpoint (.pt).
        manifest_path: Path to dataset acquisition manifest.
        split: Dataset split to evaluate ('val' or 'train').
        sample_index: Index of sample within the selected split.
        output_figure: Path to save output comparison PNG.
        device: Execution device ('cpu', 'cuda', or 'auto').

    Returns:
        Dictionary containing patch metadata, metrics, and output paths.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_p}\n"
            f"Please verify that the compact checkpoint exists at {ckpt_p}."
        )

    manifest_p = Path(manifest_path)
    if not manifest_p.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_p}\n"
            f"Please verify that the SEN12MS-CR dataset is acquired."
        )

    print("=" * 70)
    print("SEN12MS-CR Optical Cloud Removal - Minimal Demonstration")
    print("=" * 70)
    print(f"Loading checkpoint:  {ckpt_p}")
    print(f"Dataset manifest:    {manifest_p} (split='{split}', index={sample_index})")
    print(f"Target figure path:  {output_figure}")
    print(f"Requested device:    {device}")
    print("-" * 70)

    # Execute inference workflow via existing infer module
    report = run_inference(
        checkpoint_path=ckpt_p,
        manifest_path=manifest_p,
        split=split,
        sample_index=sample_index,
        output_figure=output_figure,
        device_str=device,
    )

    metrics = report.get("metrics", {})
    base_mae = metrics.get("baseline_input_mae", 0.0)
    base_psnr = metrics.get("baseline_input_psnr", 0.0)
    unet_mae = metrics.get("unet_prediction_mae", 0.0)
    unet_psnr = metrics.get("unet_prediction_psnr", 0.0)

    delta_mae = base_mae - unet_mae
    delta_psnr = unet_psnr - base_psnr

    print(f"Patch ID:            {report.get('patch_id')}")
    print(f"Input Shape:         {report.get('input_shape')}")
    print(f"Output Shape:        {report.get('output_shape')}")
    print(f"Checkpoint Epoch:    {report.get('checkpoint_epoch')}")
    print("-" * 70)
    print("Quantitative Evaluation (Physical Reflectance [0, 1]):")
    print(f"  Cloudy Input:      MAE = {base_mae:.5f}  |  PSNR = {base_psnr:.2f} dB")
    print(f"  U-Net Prediction:  MAE = {unet_mae:.5f}  |  PSNR = {unet_psnr:.2f} dB")
    print(f"  Improvement:       dMAE  = {delta_mae:+.5f} (lower is better)")
    print(f"                     dPSNR = {delta_psnr:+.2f} dB (higher is better)")
    print("-" * 70)
    print(f"Comparison figure saved: {report.get('comparison_figure')}")
    print("Shared 2%-98% contrast stretch applied for visual clarity.")
    print("=" * 70)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run minimal reproducible cloud-removal demo on compact U-Net"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/compact/best_compact_unet.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/raw/sen12mscr/manifest.json",
        help="Path to dataset manifest.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split ('val' or 'train')",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index in split (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures/demo_comparison.png",
        help="Path to save output comparison PNG figure",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('cpu', 'cuda', 'auto')",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch interactive FastAPI Web UI server on localhost:8000",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host when launching UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port when launching UI (default: 8000)",
    )

    args = parser.parse_args()

    if args.ui:
        import uvicorn

        print("=" * 70)
        print("Satellite Optical Cloud Removal - Web UI Server")
        print("=" * 70)
        print(f"Server address:   http://{args.host}:{args.port}/")
        print(f"API Docs:         http://{args.host}:{args.port}/docs")
        print("Press Ctrl+C to stop the server.")
        print("=" * 70)
        uvicorn.run("cloud_removal.api:app", host=args.host, port=args.port, reload=False)
        return 0

    try:
        run_demo(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            split=args.split,
            sample_index=args.index,
            output_figure=args.output,
            device=args.device,
        )
        return 0
    except Exception as exc:
        print(f"\n[DEMO ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
