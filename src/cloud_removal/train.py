"""Training pipeline for U-Net baseline on SEN12MS-CR with MLflow tracking.

Supports reproducible local training, smoke test runs on small subsets,
per-epoch evaluation of MAE and PSNR, checkpointing, and MLflow logging.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple

# Enable local file store tracking in modern MLflow versions
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from cloud_removal.metrics import compute_mae, compute_psnr
from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import SEN12MSCRConfig, SEN12MSCRDataset


def set_seed(seed: int = 42) -> None:
    """Set random seeds for PyTorch, NumPy, and random module for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    config: Dict[str, Any],
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train and validation splits.

    Args:
        config: Full configuration dictionary.
        max_train_samples: Optional cap on train sample count.
        max_val_samples: Optional cap on validation sample count.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    ds_cfg = config.get("dataset", {})
    sen_config = SEN12MSCRConfig(
        bands=tuple(ds_cfg.get("bands", [4, 3, 2])),
        band_names=tuple(ds_cfg.get("band_names", ["B04", "B03", "B02"])),
        normalize=ds_cfg.get("normalize", True),
        clip_max=float(ds_cfg.get("clip_max", 10000.0)),
        train_ratio=float(ds_cfg.get("train_ratio", 0.80)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.20)),
        seed=int(ds_cfg.get("seed", 42)),
    )

    manifest_path = Path(config["manifest_path"])

    # Load train and validation sets
    train_dataset = SEN12MSCRDataset.from_manifest(manifest_path, split="train", config=sen_config)
    val_dataset = SEN12MSCRDataset.from_manifest(manifest_path, split="val", config=sen_config)

    # Subsample if requested (e.g. for smoke test)
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset.samples = train_dataset.samples[:max_train_samples]

    if max_val_samples is not None and max_val_samples < len(val_dataset):
        val_dataset.samples = val_dataset.samples[:max_val_samples]

    batch_size = int(config["training"].get("batch_size", 8))
    num_workers = int(config["training"].get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch.

    Returns:
        Tuple of (average_loss, average_mae).
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    for batch in loader:
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        batch_size = cloudy.size(0)

        optimizer.zero_grad()
        pred = model(cloudy)
        loss = criterion(pred, clear)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        batch_mae = compute_mae(pred.detach(), clear.detach())
        running_mae += batch_mae * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_mae = running_mae / max(1, total_samples)
    return epoch_loss, epoch_mae


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on validation loader.

    Returns:
        Tuple of (average_loss, average_mae, average_psnr).
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            batch_size = cloudy.size(0)

            pred = model(cloudy)
            loss = criterion(pred, clear)

            running_loss += loss.item() * batch_size
            batch_mae = compute_mae(pred, clear)
            batch_psnr = compute_psnr(pred, clear)

            running_mae += batch_mae * batch_size
            running_psnr += batch_psnr * batch_size
            total_samples += batch_size

    val_loss = running_loss / max(1, total_samples)
    val_mae = running_mae / max(1, total_samples)
    val_psnr = running_psnr / max(1, total_samples)
    return val_loss, val_mae, val_psnr


def train_model(
    config: Dict[str, Any],
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute complete training workflow with MLflow tracking and checkpointing.

    Args:
        config: Training configuration dictionary.
        smoke_test: If True, execute a single epoch on a minimal subset.

    Returns:
        Summary dictionary containing training results, metrics, and checkpoint paths.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Determine execution device
    req_device = config.get("training", {}).get("device", "auto")
    if req_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(req_device)

    # Handle smoke test overrides
    max_train_samples = config["training"].get("max_train_samples")
    max_val_samples = config["training"].get("max_val_samples")
    epochs = int(config["training"].get("epochs", 10))

    if smoke_test:
        epochs = 1
        max_train_samples = 16
        max_val_samples = 8
        config["training"]["batch_size"] = min(4, int(config["training"].get("batch_size", 4)))

    train_loader, val_loader = create_dataloaders(
        config=config,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # Initialize U-Net architecture
    m_cfg = config.get("model", {})
    model = UNet(
        in_channels=int(m_cfg.get("in_channels", 3)),
        out_channels=int(m_cfg.get("out_channels", 3)),
        base_channels=int(m_cfg.get("base_channels", 32)),
        bilinear=bool(m_cfg.get("bilinear", False)),
        use_sigmoid=bool(m_cfg.get("use_sigmoid", True)),
        prior_reflectance=m_cfg.get("prior_reflectance", None),
    ).to(device)

    # Loss function and Optimizer
    criterion = nn.L1Loss()
    lr = float(config["training"].get("learning_rate", 2e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-5))
    opt_name = config["training"].get("optimizer", "adamw").lower()

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup Checkpoint directory
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_name = config.get("checkpoint_name", "best_unet.pt")
    latest_name = config.get("latest_checkpoint_name", "latest_unet.pt")
    best_checkpoint_path = checkpoint_dir / best_name
    latest_checkpoint_path = checkpoint_dir / latest_name

    # MLflow Setup
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "sen12mscr-unet-baseline")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_val_mae = float("inf")
    history = []

    print(f"==================================================")
    print(f"Starting Training: U-Net Baseline on SEN12MS-CR")
    print(f"Device: {device} | Smoke Test: {smoke_test}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Total Parameters: {model.count_parameters():,}")
    print(f"Epochs: {epochs} | Batch Size: {train_loader.batch_size} | LR: {lr}")
    print(f"==================================================")

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": "UNet",
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "base_channels": model.base_channels,
            "bilinear": model.bilinear,
            "use_sigmoid": model.use_sigmoid,
            "trainable_parameters": model.count_parameters(),
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": opt_name,
            "loss_function": "L1Loss",
            "num_train_samples": len(train_loader.dataset),
            "num_val_samples": len(val_loader.dataset),
            "device": str(device),
            "seed": seed,
            "smoke_test": smoke_test,
        })

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_mae, val_psnr = evaluate(model, val_loader, criterion, device)
            epoch_time = time.time() - t_epoch_start

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "epoch_duration_sec": epoch_time,
            }, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "epoch_duration_sec": epoch_time,
            })

            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.5f} | Train MAE: {train_mae:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f} | Val PSNR: {val_psnr:.2f} dB"
            )

            # Checkpoint payload
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "model_config": model.get_config(),
                "config": config,
            }

            # Save latest checkpoint
            torch.save(checkpoint_payload, latest_checkpoint_path)

            # Save best checkpoint
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(checkpoint_payload, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint (Val MAE: {val_mae:.5f}) to {best_checkpoint_path}")

        total_time = time.time() - start_time
        print(f"==================================================")
        print(f"Training completed in {total_time:.2f}s.")
        print(f"Best Val MAE: {best_val_mae:.5f}")
        print(f"Best Checkpoint: {best_checkpoint_path}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"==================================================")

        # Log checkpoint to MLflow artifact store
        if best_checkpoint_path.is_file():
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

    return {
        "run_id": run.info.run_id,
        "best_val_mae": best_val_mae,
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "history": history,
        "total_time_seconds": total_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train U-Net baseline on SEN12MS-CR dataset")
    parser.add_argument("--config", type=str, default="configs/train_unet.yaml", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch quick test on small sample subset")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cpu/cuda/auto)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")

    args = parser.parse_args()
    config = load_config(Path(args.config))

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir

    train_model(config=config, smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
