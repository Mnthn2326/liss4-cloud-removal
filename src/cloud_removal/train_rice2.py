"""Training entry point for the RICE2 Compact U-Net benchmark baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cloud_removal.metrics import CompoundL1SSIMLoss
from cloud_removal.models.unet import UNet
from cloud_removal.rice2 import RICE2Config, RICE2Dataset
from cloud_removal.train import EarlyStopping, evaluate, set_seed, train_one_epoch


def _get_git_commit() -> str:
    try:
        import subprocess

        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _compute_file_sha256(path: Union[str, Path]) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_rice2_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def create_rice2_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    dataset_config = RICE2Config(
        train_ratio=float(dataset_cfg.get("train_ratio", 0.64)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.16)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.20)),
        seed=int(dataset_cfg.get("seed", config.get("seed", 42))),
        augment=bool(dataset_cfg.get("augment", True)),
    )
    manifest = Path(config.get("manifest_path", "data/manifests/rice2_manifest.json"))
    train_set = RICE2Dataset.from_manifest(manifest, "train", dataset_config)
    val_set = RICE2Dataset.from_manifest(
        manifest, "val", RICE2Config(**{**dataset_config.__dict__, "augment": False})
    )
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    workers = int(train_cfg.get("num_workers", 0))
    seed = int(config.get("seed", 42))

    gen = torch.Generator()
    gen.manual_seed(seed)

    def _init_fn(worker_id: int) -> None:
        import random

        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            worker_init_fn=_init_fn,
            generator=gen,
        ),
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=_init_fn,
        ),
    )


def train_rice2_model(config: Dict[str, Any], smoke_test: bool = False) -> Dict[str, Any]:
    """Train a reproducible RICE2 Compact U-Net and save the validation champion."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    training_cfg = config.get("training", {})
    requested_device = str(training_cfg.get("device", "auto"))
    device = torch.device(
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    train_loader, val_loader = create_rice2_dataloaders(config)
    train_ds = cast(RICE2Dataset, train_loader.dataset)
    val_ds = cast(RICE2Dataset, val_loader.dataset)
    if smoke_test:
        train_ds.samples = train_ds.samples[:8]
        val_ds.samples = val_ds.samples[:4]
        epochs = 1
    else:
        epochs = int(training_cfg.get("epochs", 40))
    model_cfg = config.get("model", {})
    model = UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        bilinear=bool(model_cfg.get("bilinear", False)),
        use_sigmoid=bool(model_cfg.get("use_sigmoid", True)),
        prior_reflectance=model_cfg.get("prior_reflectance"),
    ).to(device)
    criterion = CompoundL1SSIMLoss(ssim_weight=float(training_cfg.get("ssim_weight", 0.2)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/rice2"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_name = (
        "smoke_best_rice2_compact_unet.pt"
        if smoke_test
        else config.get("checkpoint_name", "best_rice2_compact_unet.pt")
    )
    latest_name = (
        "smoke_latest_rice2_compact_unet.pt"
        if smoke_test
        else config.get("latest_checkpoint_name", "latest_rice2_compact_unet.pt")
    )
    best_path = checkpoint_dir / best_name
    latest_path = checkpoint_dir / latest_name
    stopper = EarlyStopping(
        patience=int(training_cfg.get("patience", 8)),
        min_delta=float(training_cfg.get("min_delta", 1e-4)),
    )
    tracking = config.get("mlflow", {})
    mlflow.set_tracking_uri(tracking.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(tracking.get("experiment_name", "rice2-compact-unet"))
    best_mae = float("inf")
    history = []
    manifest_p = Path(config.get("manifest_path", "data/manifests/rice2_manifest.json"))
    manifest_hash = _compute_file_sha256(manifest_p)
    config_hash = hashlib.sha256(yaml.dump(config).encode("utf-8")).hexdigest()
    git_commit = _get_git_commit()

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": "RICE2",
                "seed": seed,
                "epochs": epochs,
                "device": str(device),
                "model_params": model.count_parameters(),
                "smoke_test": smoke_test,
                "git_commit": git_commit,
            }
        )
        print(
            f"Starting RICE2 training on {device} ({epochs} epochs, {len(train_ds)} train samples, {len(val_ds)} val samples)...",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            train_loss, train_mae, _ = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_mae, val_psnr, regional = evaluate(model, val_loader, criterion, device)
            metrics = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                **regional,
            }
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "validation_metrics": metrics,
                "model_config": model.get_config(),
                "config": config,
                "dataset_name": "RICE2",
                "mlflow_run_id": run.info.run_id,
                "provenance": {
                    "git_commit": git_commit,
                    "manifest_path": str(manifest_p),
                    "manifest_sha256": manifest_hash,
                    "config_sha256": config_hash,
                    "model_architecture": "Compact U-Net (16 base channels)",
                    "training_seed": seed,
                    "epoch": epoch,
                    "val_mae": val_mae,
                    "val_psnr": val_psnr,
                },
            }
            torch.save(payload, latest_path)
            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae
                torch.save(payload, best_path)
            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.4f} (MAE: {train_mae:.4f}) | "
                f"Val MAE: {val_mae:.4f} | Val PSNR: {val_psnr:.2f} dB {'[BEST]' if is_best else ''}",
                flush=True,
            )
            if stopper.step(val_mae, epoch):
                print(
                    f"Early stopping triggered at epoch {epoch} (patience={stopper.patience}).",
                    flush=True,
                )
                break
        try:
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        except OSError as artifact_exc:
            print(f"Warning: MLflow checkpoint artifact logging skipped ({artifact_exc}).")
    return {"best_checkpoint": str(best_path), "best_val_mae": best_mae, "history": history}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train Compact U-Net on RICE2")
    parser.add_argument("--config", default="configs/train_rice2_compact_unet.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    result = train_rice2_model(load_rice2_config(Path(args.config)), smoke_test=args.smoke_test)
    print(
        f"Best RICE2 checkpoint: {result['best_checkpoint']} | validation MAE: {result['best_val_mae']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
