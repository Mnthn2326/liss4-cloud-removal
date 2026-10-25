"""Training pipeline for U-Net baseline on SEN12MS-CR with MLflow tracking.

Supports reproducible local training, smoke test runs on small subsets,
compound L1 + SSIM reconstruction loss, Cosine Annealing learning rate scheduling,
Early Stopping monitoring validation MAE, and per-epoch evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sized, Tuple, cast

# Enable local file store tracking in modern MLflow versions
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from cloud_removal.metrics import (
    CompoundL1SSIMLoss,
    compute_mae,
    compute_psnr,
    compute_regional_metrics,
)
from cloud_removal.models.unet import UNet
from cloud_removal.sen12mscr import SEN12MSCRConfig, SEN12MSCRDataset


class EarlyStopping:
    """Early stopping handler to terminate training when validation metric plateaus.

    Monitors a validation metric (default: validation MAE, where lower is better)
    and signals to stop training if no improvement greater than min_delta is observed
    for a given patience number of epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Initialize EarlyStopping handler.

        Args:
            patience: Number of epochs to wait for an improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: Metric optimization direction ('min' for loss/MAE, 'max' for PSNR/SSIM).
        """
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.early_stop = False
        self.stopping_reason: Optional[str] = None

    def step(self, metric: float, epoch: int) -> bool:
        """Update early stopping state with the latest epoch metric.

        Args:
            metric: Current validation metric value.
            epoch: Current epoch number.

        Returns:
            True if early stopping is triggered, False otherwise.
        """
        if metric is None or math.isnan(metric) or math.isinf(metric):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopping_reason = (
                    f"Early stopping triggered at epoch {epoch}: "
                    f"validation metric was non-finite or unimproved for {self.counter} consecutive epochs."
                )
            return self.early_stop

        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            return False

        if self.mode == "min":
            improvement = (self.best_score - metric) > self.min_delta
        else:
            improvement = (metric - self.best_score) > self.min_delta

        if improvement:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopping_reason = (
                    f"Early stopping triggered at epoch {epoch}: "
                    f"validation {self.mode} metric failed to improve by > {self.min_delta} "
                    f"for {self.counter} consecutive epochs (best epoch: {self.best_epoch}, best score: {self.best_score:.5f})."
                )

        return self.early_stop


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
    augment = bool(ds_cfg.get("augment", False))

    sen_config = SEN12MSCRConfig(
        bands=tuple(ds_cfg.get("bands", [4, 3, 2])),
        band_names=tuple(ds_cfg.get("band_names", ["B04", "B03", "B02"])),
        normalize=ds_cfg.get("normalize", True),
        clip_max=float(ds_cfg.get("clip_max", 10000.0)),
        train_ratio=float(ds_cfg.get("train_ratio", 0.80)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.20)),
        seed=int(ds_cfg.get("seed", 42)),
        augment=augment,
    )

    manifest_raw = config.get("manifest_path", "data/manifests/sen12mscr_manifest.json")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_file():
        fallback = (
            Path("data/manifests/sen12mscr_manifest.json")
            if "raw" in str(manifest_path)
            else Path("data/raw/sen12mscr/manifest.json")
        )
        if fallback.is_file():
            manifest_path = fallback

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
) -> Tuple[float, float, Dict[str, float]]:
    """Train model for one epoch.

    Returns:
        Tuple of (average_loss, average_mae, loss_components_dict).
    """
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_mae = 0.0
    total_samples = 0

    for batch in loader:
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        batch_size = cloudy.size(0)

        optimizer.zero_grad()
        pred = model(cloudy)

        if isinstance(criterion, CompoundL1SSIMLoss):
            loss, components = criterion(pred, clear, return_components=True)
            l1_val = components["loss_l1"]
            ssim_loss_val = components["loss_ssim"]
        else:
            loss = criterion(pred, clear)
            l1_val = float(loss.item())
            ssim_loss_val = 0.0

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * batch_size
        running_l1 += l1_val * batch_size
        running_ssim += ssim_loss_val * batch_size
        batch_mae = compute_mae(pred.detach(), clear.detach())
        running_mae += batch_mae * batch_size
        total_samples += batch_size

    n_samples = max(1, total_samples)
    epoch_loss = running_loss / n_samples
    epoch_mae = running_mae / n_samples
    components = {
        "loss_total": epoch_loss,
        "loss_l1": running_l1 / n_samples,
        "loss_ssim": running_ssim / n_samples,
    }
    return epoch_loss, epoch_mae, components


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluate model on validation loader.

    Returns:
        Tuple of (average_loss, average_mae, average_psnr, regional_metrics_dict).
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    total_samples = 0

    has_cloud_mask = False
    cloud_maes, cloud_psnrs = [], []
    clear_maes, clear_psnrs = [], []

    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            batch_size = cloudy.size(0)

            pred = model(cloudy)
            loss = criterion(pred, clear)

            running_loss += float(loss.item()) * batch_size
            batch_mae = compute_mae(pred, clear)
            batch_psnr = compute_psnr(pred, clear)

            running_mae += batch_mae * batch_size
            running_psnr += batch_psnr * batch_size
            total_samples += batch_size

            # Evaluate regional metrics if cloud opacity mask is available
            if "cloud_mask" in batch:
                has_cloud_mask = True
                c_mask = batch["cloud_mask"].to(device, non_blocking=True)
                reg_m = compute_regional_metrics(pred, clear, c_mask)
                if not math.isnan(reg_m["MAE_cloud"]):
                    cloud_maes.append(reg_m["MAE_cloud"])
                    cloud_psnrs.append(reg_m["PSNR_cloud"])
                if not math.isnan(reg_m["MAE_clear"]):
                    clear_maes.append(reg_m["MAE_clear"])
                    clear_psnrs.append(reg_m["PSNR_clear"])

    val_loss = running_loss / max(1, total_samples)
    val_mae = running_mae / max(1, total_samples)
    val_psnr = running_psnr / max(1, total_samples)

    regional_metrics: Dict[str, float] = {}
    if has_cloud_mask:
        regional_metrics = {
            "MAE_cloud": float(np.mean(cloud_maes)) if cloud_maes else float("nan"),
            "PSNR_cloud": float(np.mean(cloud_psnrs)) if cloud_psnrs else float("nan"),
            "MAE_clear": float(np.mean(clear_maes)) if clear_maes else float("nan"),
            "PSNR_clear": float(np.mean(clear_psnrs)) if clear_psnrs else float("nan"),
        }

    return val_loss, val_mae, val_psnr, regional_metrics


def train_model(
    config: Dict[str, Any],
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute complete training workflow with MLflow tracking, scheduler, and early stopping.

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

    # Loss function configuration (Compound L1 + SSIM by default)
    t_cfg = config.get("training", {})
    loss_name = str(t_cfg.get("loss", "compound_l1_ssim")).lower()
    ssim_weight = float(t_cfg.get("ssim_weight", 0.2))

    if loss_name in ("compound_l1_ssim", "compound", "l1_ssim"):
        criterion: nn.Module = CompoundL1SSIMLoss(ssim_weight=ssim_weight, data_range=1.0)
    else:
        criterion = nn.L1Loss()

    # Optimizer configuration
    lr = float(t_cfg.get("learning_rate", 2e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    opt_name = t_cfg.get("optimizer", "adamw").lower()

    if opt_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler configuration: Cosine Annealing LR
    scheduler_name = str(t_cfg.get("scheduler", "cosine")).lower()
    t_max = int(t_cfg.get("t_max", epochs))
    eta_min = float(t_cfg.get("eta_min", 1e-6))

    if scheduler_name == "cosine":
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        )
    else:
        scheduler = None

    # Early Stopping configuration
    es_enabled = bool(t_cfg.get("early_stopping", True))
    es_patience = int(t_cfg.get("patience", 5))
    es_min_delta = float(t_cfg.get("min_delta", 0.0))
    early_stopper = (
        EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode="min")
        if es_enabled
        else None
    )

    # Setup Checkpoint directory
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_name = config.get("checkpoint_name", "best_unet.pt")
    latest_name = config.get("latest_checkpoint_name", "latest_unet.pt")
    best_checkpoint_path = checkpoint_dir / best_name
    latest_checkpoint_path = checkpoint_dir / latest_name

    # MLflow Setup (supports remote tracking like DagsHub via environment variable)
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "sen12mscr-unet-baseline")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_val_mae = float("inf")
    best_epoch = 1
    early_stopped = False
    stopping_reason: Optional[str] = None
    history = []

    print("==================================================")
    print("Starting Training: U-Net Baseline on SEN12MS-CR")
    print(f"Device: {device} | Smoke Test: {smoke_test}")
    n_train = len(cast(Sized, train_loader.dataset))
    n_val = len(cast(Sized, val_loader.dataset))
    print(f"Train samples: {n_train} | Val samples: {n_val}")
    print(f"Total Parameters: {model.count_parameters():,}")
    print(f"Epochs: {epochs} | Batch Size: {train_loader.batch_size} | Initial LR: {lr}")
    print(f"Loss: {loss_name} (SSIM weight: {ssim_weight}) | Scheduler: {scheduler_name}")
    print(f"Early Stopping: {es_enabled} (Patience: {es_patience})")
    print("==================================================")

    with mlflow.start_run() as run:
        user_tag = (
            os.environ.get("MLFLOW_USER")
            or os.environ.get("DAGSHUB_USER")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
        )
        if user_tag:
            mlflow.set_tag("user", user_tag)

        # Log parameters
        mlflow.log_params(
            {
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
                "loss_function": loss_name,
                "ssim_weight": (
                    ssim_weight if loss_name in ("compound_l1_ssim", "compound", "l1_ssim") else 0.0
                ),
                "scheduler": scheduler_name,
                "scheduler_t_max": t_max,
                "scheduler_eta_min": eta_min,
                "early_stopping_enabled": es_enabled,
                "early_stopping_patience": es_patience,
                "early_stopping_min_delta": es_min_delta,
                "augmentation_enabled": bool(config.get("dataset", {}).get("augment", False)),
                "bands": str(config.get("dataset", {}).get("bands", [4, 3, 2])),
                "band_names": str(
                    config.get("dataset", {}).get("band_names", ["B04", "B03", "B02"])
                ),
                "composite_mode": str(config.get("dataset", {}).get("composite_mode", "rgb")),
                "num_train_samples": n_train,
                "num_val_samples": n_val,
                "device": str(device),
                "seed": seed,
                "smoke_test": smoke_test,
            }
        )

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            train_loss, train_mae, loss_components = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_loss, val_mae, val_psnr, val_regional = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            epoch_time = time.time() - t_epoch_start
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Step Cosine Annealing scheduler once per completed training epoch
            if scheduler is not None:
                scheduler.step()

            # Log metrics to MLflow
            metrics_dict = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "loss_total": loss_components.get("loss_total", train_loss),
                "loss_l1": loss_components.get("loss_l1", train_loss),
                "loss_ssim": loss_components.get("loss_ssim", 0.0),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "learning_rate": current_lr,
                "epoch_duration_sec": epoch_time,
            }
            if val_regional:
                for k, v in val_regional.items():
                    if not math.isnan(v):
                        metrics_dict[k] = v

            mlflow.log_metrics(metrics_dict, step=epoch)

            history_entry = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "loss_l1": loss_components.get("loss_l1", train_loss),
                "loss_ssim": loss_components.get("loss_ssim", 0.0),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "epoch_duration_sec": epoch_time,
            }
            if val_regional:
                history_entry.update(val_regional)
            history.append(history_entry)

            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:.1f}s) [LR: {current_lr:.6f}] - "
                f"Train Loss: {train_loss:.5f} (L1: {loss_components.get('loss_l1', 0.0):.4f}, SSIM: {loss_components.get('loss_ssim', 0.0):.4f}) | "
                f"Val MAE: {val_mae:.5f} | Val PSNR: {val_psnr:.2f} dB"
            )

            # Checkpoint payload
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "bands": list(config.get("dataset", {}).get("bands", [4, 3, 2])),
                "band_names": list(
                    config.get("dataset", {}).get("band_names", ["B04", "B03", "B02"])
                ),
                "composite_mode": str(config.get("dataset", {}).get("composite_mode", "rgb")),
                "model_config": model.get_config(),
                "config": config,
            }

            # Save latest checkpoint
            torch.save(checkpoint_payload, latest_checkpoint_path)

            # Save best checkpoint strictly when val_mae improves
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(checkpoint_payload, best_checkpoint_path)
                print(
                    f"  --> Saved new best checkpoint (Val MAE: {val_mae:.5f}) to {best_checkpoint_path}"
                )

            # Check Early Stopping
            if early_stopper is not None and early_stopper.step(val_mae, epoch):
                early_stopped = True
                stopping_reason = early_stopper.stopping_reason
                print(f"  [Early Stopping] {stopping_reason}")
                break

        total_time = time.time() - start_time
        print("==================================================")
        print(f"Training completed in {total_time:.2f}s.")
        print(f"Best Val MAE: {best_val_mae:.5f} (at Epoch {best_epoch})")
        print(f"Best Checkpoint: {best_checkpoint_path}")
        if early_stopped:
            print(f"Termination: Early stopped ({stopping_reason})")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("==================================================")

        # Load best checkpoint weights for final logging and registration (use CPU for safe export)
        model = model.to("cpu")
        if best_checkpoint_path.is_file():
            try:
                best_state = torch.load(best_checkpoint_path, map_location="cpu")
                if isinstance(best_state, dict) and "model_state_dict" in best_state:
                    model.load_state_dict(best_state["model_state_dict"])
            except Exception as load_exc:
                print(
                    f"Warning: Could not reload best checkpoint for MLflow model export: {load_exc}"
                )
        model.eval()

        # Build sample input and infer MLflow model signature
        in_c = model.in_channels
        input_example = np.random.uniform(0.0, 1.0, size=(1, in_c, 256, 256)).astype(np.float32)
        with torch.no_grad():
            output_example = model(torch.from_numpy(input_example)).numpy()
        model_signature = infer_signature(input_example, output_example)

        model_metadata = {
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "base_channels": model.base_channels,
            "bilinear": model.bilinear,
            "use_sigmoid": model.use_sigmoid,
            "composite_mode": str(config.get("dataset", {}).get("composite_mode", "rgb")),
            "bands": list(config.get("dataset", {}).get("bands", [4, 3, 2])),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
        }

        registered_model_name = config.get("mlflow", {}).get("registered_model_name", None)
        model_alias = config.get("mlflow", {}).get("alias", None)
        model_stage = config.get("mlflow", {}).get("stage", None)

        model_info = None
        registered_successfully = False
        try:
            if registered_model_name:
                try:
                    model_info = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        name="model",
                        signature=model_signature,
                        input_example=input_example,
                        registered_model_name=str(registered_model_name),
                        metadata=model_metadata,
                    )
                    registered_successfully = True
                    print(f"Successfully registered model '{registered_model_name}'.")
                except Exception as reg_exc:
                    print(
                        f"Warning: Model registration with name '{registered_model_name}' failed ({reg_exc}). Logging model artifact without registration."
                    )
                    model_info = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        name="model",
                        signature=model_signature,
                        input_example=input_example,
                        metadata=model_metadata,
                    )
            else:
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="model",
                    signature=model_signature,
                    input_example=input_example,
                    metadata=model_metadata,
                )
        except Exception as export_exc:
            print(f"Warning: MLflow PyTorch model package export skipped ({export_exc}). The raw checkpoint artifact remains intact.")

        # Handle alias / stage promotion if model registration succeeded
        if registered_successfully and registered_model_name and model_info is not None:
            version = getattr(model_info, "registered_model_version", None)
            if version is not None:
                client = MlflowClient(tracking_uri=tracking_uri)
                if model_alias:
                    try:
                        client.set_registered_model_alias(
                            name=str(registered_model_name),
                            alias=str(model_alias).lower(),
                            version=str(version),
                        )
                        print(
                            f"Assigned alias '{model_alias}' to model '{registered_model_name}' version {version}"
                        )
                    except Exception as a_exc:
                        print(f"Warning: Could not set alias '{model_alias}': {a_exc}")
                if model_stage:
                    try:
                        client.transition_model_version_stage(
                            name=str(registered_model_name),
                            version=str(version),
                            stage=str(model_stage),
                        )
                        print(
                            f"Transitioned model '{registered_model_name}' version {version} to stage '{model_stage}'"
                        )
                    except Exception as s_exc:
                        print(f"Warning: Could not transition stage to '{model_stage}': {s_exc}")

        # Log best checkpoint file to MLflow artifact store
        if best_checkpoint_path.is_file():
            try:
                mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
            except OSError as artifact_exc:
                # Windows temporary directories can exceed the legacy path limit in MLflow's
                # local-file artifact implementation.  The durable checkpoint remains saved.
                print(f"Warning: MLflow checkpoint artifact logging skipped ({artifact_exc}).")

    return {
        "run_id": run.info.run_id,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "early_stopped": early_stopped,
        "stopping_reason": stopping_reason,
        "history": history,
        "total_time_seconds": total_time,
        "model_uri": getattr(model_info, "model_uri", None) if model_info else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train U-Net baseline on SEN12MS-CR dataset")
    parser.add_argument(
        "--config", type=str, default="configs/train_unet.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="Override path to dataset manifest"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 1-epoch quick test on small sample subset"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cpu/cuda/auto)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Override checkpoint directory"
    )
    parser.add_argument(
        "--register-model",
        type=str,
        default=None,
        help="Register model under specified name in MLflow registry",
    )
    parser.add_argument(
        "--model-alias", type=str, default=None, help="Assign MLflow model alias (e.g. champion)"
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default=None,
        help="Transition MLflow model stage (e.g. Staging, Production)",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    # Apply CLI overrides
    if args.manifest is not None:
        config["manifest_path"] = args.manifest
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
    if args.register_model is not None:
        config.setdefault("mlflow", {})["registered_model_name"] = args.register_model
    if args.model_alias is not None:
        config.setdefault("mlflow", {})["alias"] = args.model_alias
    if args.model_stage is not None:
        config.setdefault("mlflow", {})["stage"] = args.model_stage

    train_model(config=config, smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
