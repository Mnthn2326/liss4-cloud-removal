"""Unit and integration tests for training pipeline, scheduling, early stopping, and loss components."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from cloud_removal.models.unet import UNet
from cloud_removal.train import EarlyStopping, train_model

# ==============================================================================
# Early Stopping Unit Tests (Task 3)
# ==============================================================================


def test_early_stopping_initial_state():
    """EarlyStopping must initialize with zero counter and no early stop signal."""
    es = EarlyStopping(patience=5, min_delta=0.001, mode="min")
    assert es.patience == 5
    assert es.min_delta == 0.001
    assert es.mode == "min"
    assert es.counter == 0
    assert es.best_score is None
    assert es.best_epoch is None
    assert not es.early_stop
    assert es.stopping_reason is None


def test_early_stopping_improvement_resets_counter():
    """Improving metric must update best_score and keep counter at zero."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    # Epoch 1: initial score
    assert not es.step(metric=0.50, epoch=1)
    assert es.counter == 0
    assert es.best_score == 0.50
    assert es.best_epoch == 1

    # Epoch 2: improvement greater than min_delta (0.50 - 0.45 = 0.05 > 0.01)
    assert not es.step(metric=0.45, epoch=2)
    assert es.counter == 0
    assert es.best_score == 0.45
    assert es.best_epoch == 2

    # Epoch 3: improvement greater than min_delta (0.45 - 0.40 = 0.05 > 0.01)
    assert not es.step(metric=0.40, epoch=3)
    assert es.counter == 0
    assert es.best_score == 0.40
    assert es.best_epoch == 3


def test_early_stopping_triggers_after_patience():
    """EarlyStopping must trigger exactly when unimproved epochs reach patience."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    # Epoch 1: best = 0.20
    assert not es.step(0.20, epoch=1)
    assert es.counter == 0

    # Epoch 2: degraded (0.22 > 0.20) -> counter = 1
    assert not es.step(0.22, epoch=2)
    assert es.counter == 1
    assert not es.early_stop

    # Epoch 3: minor improvement not exceeding min_delta (0.20 - 0.195 = 0.005 < 0.01) -> counter = 2
    assert not es.step(0.195, epoch=3)
    assert es.counter == 2
    assert not es.early_stop

    # Epoch 4: degraded (0.25 > 0.20) -> counter = 3 >= patience -> triggers!
    triggered = es.step(0.25, epoch=4)
    assert triggered is True
    assert es.early_stop is True
    assert es.best_epoch == 1
    assert es.best_score == 0.20
    assert "Early stopping triggered at epoch 4" in es.stopping_reason


def test_early_stopping_mode_max():
    """EarlyStopping with mode='max' must monitor metrics where higher is better (e.g. PSNR)."""
    es = EarlyStopping(patience=2, min_delta=0.5, mode="max")

    assert not es.step(metric=25.0, epoch=1)
    assert es.best_score == 25.0

    # Improved (26.0 - 25.0 = 1.0 > 0.5)
    assert not es.step(metric=26.0, epoch=2)
    assert es.best_score == 26.0
    assert es.counter == 0

    # Degraded (24.0 < 26.0) -> counter = 1
    assert not es.step(metric=24.0, epoch=3)
    assert es.counter == 1

    # Degraded (25.0 < 26.0) -> counter = 2 >= patience -> triggers!
    assert es.step(metric=25.0, epoch=4) is True
    assert es.early_stop is True
    assert es.best_epoch == 2
    assert es.best_score == 26.0


def test_early_stopping_non_finite_handling():
    """EarlyStopping must handle NaN or Inf values gracefully without crashing."""
    es = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    assert not es.step(0.30, epoch=1)
    assert not es.step(float("nan"), epoch=2)
    assert es.counter == 1

    # Second NaN triggers patience
    assert es.step(float("nan"), epoch=3) is True
    assert es.early_stop is True
    assert es.best_epoch == 1


def test_early_stopping_invalid_arguments():
    """EarlyStopping must raise ValueError for invalid arguments."""
    with pytest.raises(ValueError, match="patience must be positive"):
        EarlyStopping(patience=0)

    with pytest.raises(ValueError, match="min_delta must be non-negative"):
        EarlyStopping(patience=3, min_delta=-0.01)

    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(patience=3, mode="invalid")


# ==============================================================================
# Cosine Annealing Learning Rate Scheduler Tests (Task 3)
# ==============================================================================


def test_cosine_annealing_lr_stepping():
    """CosineAnnealingLR must step per epoch and follow the cosine decay formula."""
    model = nn.Linear(10, 2)
    initial_lr = 1e-3
    eta_min = 1e-5
    t_max = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    lrs = [optimizer.param_groups[0]["lr"]]
    for epoch in range(1, t_max + 1):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Epoch 0 (initial)
    assert pytest.approx(lrs[0], rel=1e-5) == initial_lr
    # At t_max (epoch 5), lr should reach eta_min
    assert pytest.approx(lrs[5], rel=1e-5) == eta_min

    # Verify intermediate cosine values
    for t in range(1, t_max + 1):
        expected_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * t / t_max))
        assert pytest.approx(lrs[t], rel=1e-4) == expected_lr


# ==============================================================================
# Training Pipeline & Checkpoint Protection Tests (Tasks 1, 3, 4)
# ==============================================================================


def test_checkpoint_preservation_on_early_stopping(tmp_path: Path):
    """Best checkpoint must preserve the epoch with the best metric, not the stopping epoch."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    latest_path = ckpt_dir / "latest_model.pt"

    # Simulate 5 epochs where Epoch 2 is best (MAE = 0.05) and epochs 3-5 degrade
    val_maes = [0.10, 0.05, 0.08, 0.09, 0.12]
    best_val_mae = float("inf")
    es = EarlyStopping(patience=3, min_delta=0.001, mode="min")

    model = UNet(in_channels=3, out_channels=3, base_channels=8)

    for epoch, val_mae in enumerate(val_maes, start=1):
        payload = {
            "epoch": epoch,
            "val_mae": val_mae,
            "model_state_dict": model.state_dict(),
        }
        # Save latest
        torch.save(payload, latest_path)

        # Save best strictly on improvement
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(payload, best_path)

        if es.step(val_mae, epoch):
            break

    # Early stopping should have triggered at epoch 5
    assert es.early_stop is True

    # Best checkpoint must be from epoch 2
    best_saved = torch.load(best_path)
    assert best_saved["epoch"] == 2
    assert best_saved["val_mae"] == 0.05

    # Latest checkpoint must be from epoch 5 (stopping epoch)
    latest_saved = torch.load(latest_path)
    assert latest_saved["epoch"] == 5
    assert latest_saved["val_mae"] == 0.12


def test_train_model_smoke_test(tmp_path: Path):
    """train_model smoke_test must execute end-to-end with compound loss and early stopping."""
    base_config_path = Path("configs/train_compact_unet.yaml")
    if not base_config_path.is_file():
        pytest.skip("Base training config not found")

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override for isolated fast smoke test
    cfg["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    cfg["training"]["epochs"] = 1
    cfg["training"]["loss"] = "compound_l1_ssim"
    cfg["training"]["ssim_weight"] = 0.2
    cfg["training"]["scheduler"] = "cosine"
    cfg["training"]["early_stopping"] = True
    cfg["training"]["patience"] = 2

    # Run smoke test
    summary = train_model(config=cfg, smoke_test=True)

    assert summary is not None
    assert len(summary["history"]) == 1
    assert "best_val_mae" in summary
    assert Path(summary["best_checkpoint_path"]).is_file()
    assert Path(summary["latest_checkpoint_path"]).is_file()

    # Check that training history logged the required components
    history = summary["history"]
    assert len(history) == 1
    first_epoch = history[0]
    assert "loss_l1" in first_epoch
    assert "loss_ssim" in first_epoch
    assert "train_loss" in first_epoch
    assert "learning_rate" in first_epoch
    assert first_epoch["learning_rate"] > 0.0


def test_infer_regional_metrics_with_cloud_mask(tmp_path: Path):
    """run_inference must report MAE_cloud, PSNR_cloud, MAE_clear, PSNR_clear when mask is provided."""
    import rasterio

    from cloud_removal.infer import run_inference

    ckpt_path = Path("checkpoints/compact/best_compact_unet.pt")
    manifest_path = Path("data/raw/sen12mscr/manifest.json")
    if not ckpt_path.is_file() or not manifest_path.is_file():
        pytest.skip("Compact checkpoint or dataset manifest not present")

    # Create a synthetic 256x256 GeoTIFF cloud mask: half cloudy (1.0), half clear (0.0)
    mask_file = tmp_path / "test_cloud_mask.tif"
    mask_data = np.zeros((1, 256, 256), dtype=np.float32)
    mask_data[0, :128, :] = 1.0  # Top half cloudy (> 0.2)

    with rasterio.open(
        mask_file,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype=np.float32,
    ) as dst:
        dst.write(mask_data)

    out_fig = tmp_path / "infer_regional.png"
    report = run_inference(
        checkpoint_path=ckpt_path,
        manifest_path=manifest_path,
        split="val",
        sample_index=0,
        cloud_mask_path=mask_file,
        output_figure=out_fig,
        device_str="cpu",
    )

    assert report["has_cloud_mask"] is True
    metrics = report["metrics"]
    assert "MAE_cloud" in metrics
    assert "PSNR_cloud" in metrics
    assert "MAE_clear" in metrics
    assert "PSNR_clear" in metrics
    assert not math.isnan(metrics["MAE_cloud"])
    assert not math.isnan(metrics["MAE_clear"])
    assert metrics["MAE_cloud"] >= 0.0
    assert metrics["MAE_clear"] >= 0.0
