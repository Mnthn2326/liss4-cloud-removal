"""Conditional GAN (Pix2Pix) training pipeline for optical cloud removal with MLflow tracking.

Implements paired image-to-image translation using a U-Net Generator and
a PatchGAN Discriminator, tracking generator and discriminator dynamics,
adversarial loss, L1 reconstruction loss, and validation metrics in MLflow.
"""

from __future__ import annotations

import argparse
import math
import os
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
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader

from cloud_removal.metrics import compute_mae
from cloud_removal.models.patchgan import (
    GANLoss,
    PatchGANDiscriminator,
    init_weights,
)
from cloud_removal.models.unet import UNet
from cloud_removal.train import (
    EarlyStopping,
    create_dataloaders,
    evaluate,
    load_config,
    set_seed,
)


def train_cgan_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    criterion_gan: GANLoss,
    criterion_l1: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    lambda_l1: float = 100.0,
) -> Tuple[Dict[str, float], float]:
    """Train Generator and PatchGAN Discriminator for one epoch.

    Args:
        generator: Generator network (U-Net).
        discriminator: Discriminator network (PatchGAN).
        loader: DataLoader yielding {'cloudy': Tensor, 'clear': Tensor}.
        criterion_gan: Adversarial GAN loss module.
        criterion_l1: L1 or compound reconstruction loss module.
        optimizer_g: Generator optimizer.
        optimizer_d: Discriminator optimizer.
        device: Torch device (CPU or CUDA).
        lambda_l1: Weight multiplier for L1 reconstruction loss (default: 100.0).

    Returns:
        Tuple of (losses_dict, average_train_mae).
    """
    generator.train()
    discriminator.train()

    running_loss_d = 0.0
    running_loss_d_real = 0.0
    running_loss_d_fake = 0.0
    running_loss_g = 0.0
    running_loss_g_gan = 0.0
    running_loss_g_l1 = 0.0
    running_mae = 0.0
    total_samples = 0

    for batch in loader:
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        batch_size = cloudy.size(0)

        # ---------------------------------------------------------------------
        # 1. Forward Generator to produce fake clear images
        # ---------------------------------------------------------------------
        fake_clear = generator(cloudy)

        # ---------------------------------------------------------------------
        # 2. Update Discriminator: maximize log(D(x, y)) + log(1 - D(x, G(x)))
        # ---------------------------------------------------------------------
        optimizer_d.zero_grad()

        # Real pair: (cloudy, real clear)
        real_pair = torch.cat([cloudy, clear], dim=1)
        pred_real = discriminator(real_pair)
        loss_d_real = criterion_gan(pred_real, target_is_real=True)

        # Fake pair: (cloudy, generated clear) - detach fake to prevent G gradients
        fake_pair = torch.cat([cloudy, fake_clear.detach()], dim=1)
        pred_fake = discriminator(fake_pair)
        loss_d_fake = criterion_gan(pred_fake, target_is_real=False)

        # Combined discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        optimizer_d.step()

        # ---------------------------------------------------------------------
        # 3. Update Generator: maximize log(D(x, G(x))) + lambda_l1 * L1(y, G(x))
        # ---------------------------------------------------------------------
        optimizer_g.zero_grad()

        # Discriminator prediction on fake pair with active G graph
        fake_pair_for_g = torch.cat([cloudy, fake_clear], dim=1)
        pred_fake_g = discriminator(fake_pair_for_g)
        loss_g_gan = criterion_gan(pred_fake_g, target_is_real=True)

        # L1 content loss
        loss_g_l1 = criterion_l1(fake_clear, clear)
        loss_g = loss_g_gan + (lambda_l1 * loss_g_l1)
        loss_g.backward()
        optimizer_g.step()

        # Metrics tracking
        running_loss_d += float(loss_d.item()) * batch_size
        running_loss_d_real += float(loss_d_real.item()) * batch_size
        running_loss_d_fake += float(loss_d_fake.item()) * batch_size
        running_loss_g += float(loss_g.item()) * batch_size
        running_loss_g_gan += float(loss_g_gan.item()) * batch_size
        running_loss_g_l1 += float(loss_g_l1.item()) * batch_size

        batch_mae = compute_mae(fake_clear.detach(), clear.detach())
        running_mae += batch_mae * batch_size
        total_samples += batch_size

    n = max(1, total_samples)
    losses_dict = {
        "loss_d": running_loss_d / n,
        "loss_d_real": running_loss_d_real / n,
        "loss_d_fake": running_loss_d_fake / n,
        "loss_g": running_loss_g / n,
        "loss_g_gan": running_loss_g_gan / n,
        "loss_g_l1": running_loss_g_l1 / n,
    }
    avg_mae = running_mae / n
    return losses_dict, avg_mae


def train_cgan_model(
    config: Dict[str, Any],
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute complete Pix2Pix cGAN training workflow with MLflow tracking.

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

    # Smoke test sample overrides
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

    # Initialize Generator (U-Net)
    m_cfg = config.get("model", {})
    in_channels = int(m_cfg.get("in_channels", 3))
    out_channels = int(m_cfg.get("out_channels", 3))

    generator = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=int(m_cfg.get("base_channels", 32)),
        bilinear=bool(m_cfg.get("bilinear", False)),
        use_sigmoid=bool(m_cfg.get("use_sigmoid", True)),
        prior_reflectance=m_cfg.get("prior_reflectance", None),
    ).to(device)

    # Initialize Discriminator (PatchGAN)
    d_cfg = config.get("discriminator", {})
    disc_in_channels = in_channels + out_channels
    ndf = int(d_cfg.get("ndf", 64))
    n_layers = int(d_cfg.get("n_layers", 3))
    norm_layer = str(d_cfg.get("norm_layer", "batch"))

    discriminator = PatchGANDiscriminator(
        in_channels=disc_in_channels,
        ndf=ndf,
        n_layers=n_layers,
        norm_layer=norm_layer,
    ).to(device)

    # Initialize weights if specified
    if config.get("init_weights", True):
        init_weights(discriminator, init_type="normal", init_gain=0.02)

    # Loss criteria
    t_cfg = config.get("training", {})
    lambda_l1 = float(t_cfg.get("lambda_l1", 100.0))
    use_lsgan = bool(t_cfg.get("use_lsgan", False))
    criterion_gan = GANLoss(use_lsgan=use_lsgan).to(device)
    criterion_l1 = nn.L1Loss().to(device)

    # Optimizers (Adam with beta1=0.5, beta2=0.999 is standard Pix2Pix)
    lr_g = float(t_cfg.get("learning_rate_g", t_cfg.get("learning_rate", 0.0002)))
    lr_d = float(t_cfg.get("learning_rate_d", t_cfg.get("learning_rate", 0.0002)))
    beta1 = float(t_cfg.get("beta1", 0.5))
    beta2 = float(t_cfg.get("beta2", 0.999))

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    # Early stopping
    es_enabled = bool(t_cfg.get("early_stopping", True))
    es_patience = int(t_cfg.get("patience", 5))
    es_min_delta = float(t_cfg.get("min_delta", 0.0))
    early_stopper = (
        EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode="min")
        if es_enabled
        else None
    )

    # Checkpoint paths
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/cgan"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_g_name = config.get("checkpoint_name", "best_cgan_generator.pt")
    latest_g_name = config.get("latest_checkpoint_name", "latest_cgan_generator.pt")
    best_d_name = config.get("checkpoint_d_name", "best_cgan_discriminator.pt")
    latest_d_name = config.get("latest_checkpoint_d_name", "latest_cgan_discriminator.pt")

    best_g_path = checkpoint_dir / best_g_name
    latest_g_path = checkpoint_dir / latest_g_name
    best_d_path = checkpoint_dir / best_d_name
    latest_d_path = checkpoint_dir / latest_d_name

    # MLflow Setup (supports remote tracking like DagsHub via environment variable)
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "sen12mscr-pix2pix-cgan")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_val_mae = float("inf")
    best_epoch = 1
    early_stopped = False
    stopping_reason: Optional[str] = None
    history = []

    print("==================================================")
    print("Starting Training: Pix2Pix Conditional GAN (cGAN)")
    print(f"Device: {device} | Smoke Test: {smoke_test}")
    n_train = len(cast(Sized, train_loader.dataset))
    n_val = len(cast(Sized, val_loader.dataset))
    print(f"Train samples: {n_train} | Val samples: {n_val}")
    print(f"Generator Parameters: {generator.count_parameters():,}")
    print(f"Discriminator Parameters: {discriminator.count_parameters():,}")
    print(f"Epochs: {epochs} | Batch Size: {train_loader.batch_size} | Lambda L1: {lambda_l1}")
    print(f"LR Generator: {lr_g} | LR Discriminator: {lr_d} (Betas: {beta1}, {beta2})")
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

        mlflow.log_params(
            {
                "model_type": "Pix2Pix_cGAN",
                "generator_type": "UNet",
                "discriminator_type": "PatchGANDiscriminator",
                "generator_trainable_params": generator.count_parameters(),
                "discriminator_trainable_params": discriminator.count_parameters(),
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate_g": lr_g,
                "learning_rate_d": lr_d,
                "beta1": beta1,
                "beta2": beta2,
                "lambda_l1": lambda_l1,
                "use_lsgan": use_lsgan,
                "disc_ndf": ndf,
                "disc_n_layers": n_layers,
                "disc_norm_layer": norm_layer,
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

            train_losses, train_mae = train_cgan_one_epoch(
                generator=generator,
                discriminator=discriminator,
                loader=train_loader,
                criterion_gan=criterion_gan,
                criterion_l1=criterion_l1,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                device=device,
                lambda_l1=lambda_l1,
            )

            # Evaluate generator on validation split
            val_loss, val_mae, val_psnr, val_regional = evaluate(
                model=generator,
                loader=val_loader,
                criterion=criterion_l1,
                device=device,
            )
            epoch_time = time.time() - t_epoch_start

            # Log metrics to MLflow
            metrics_dict = {
                "train_loss_d": train_losses["loss_d"],
                "train_loss_d_real": train_losses["loss_d_real"],
                "train_loss_d_fake": train_losses["loss_d_fake"],
                "train_loss_g": train_losses["loss_g"],
                "train_loss_g_gan": train_losses["loss_g_gan"],
                "train_loss_g_l1": train_losses["loss_g_l1"],
                "train_mae": train_mae,
                "val_loss_l1": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "epoch_duration_sec": epoch_time,
            }
            if val_regional:
                for k, v in val_regional.items():
                    if not math.isnan(v):
                        metrics_dict[k] = v

            mlflow.log_metrics(metrics_dict, step=epoch)

            history_entry = {"epoch": epoch, **metrics_dict}
            history.append(history_entry)

            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:.1f}s) - "
                f"Loss D: {train_losses['loss_d']:.4f} (Real: {train_losses['loss_d_real']:.4f}, Fake: {train_losses['loss_d_fake']:.4f}) | "
                f"Loss G: {train_losses['loss_g']:.4f} (Adv: {train_losses['loss_g_gan']:.4f}, L1: {train_losses['loss_g_l1']:.4f}) | "
                f"Val MAE: {val_mae:.5f} | Val PSNR: {val_psnr:.2f} dB"
            )

            # Save Generator checkpoint payload (drop-in compatible with standard UNet checkpoint)
            g_payload = {
                "epoch": epoch,
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer_g.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_psnr": val_psnr,
                "bands": list(config.get("dataset", {}).get("bands", [4, 3, 2])),
                "band_names": list(
                    config.get("dataset", {}).get("band_names", ["B04", "B03", "B02"])
                ),
                "composite_mode": str(config.get("dataset", {}).get("composite_mode", "rgb")),
                "model_config": generator.get_config(),
                "config": config,
            }
            torch.save(g_payload, latest_g_path)

            d_payload = {
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": optimizer_d.state_dict(),
                "model_config": discriminator.get_config(),
            }
            torch.save(d_payload, latest_d_path)

            # Save best checkpoint strictly when val_mae improves
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(g_payload, best_g_path)
                torch.save(d_payload, best_d_path)
                print(
                    f"  --> Saved new best cGAN checkpoint (Val MAE: {val_mae:.5f}) to {best_g_path}"
                )

            # Check Early Stopping
            if early_stopper is not None and early_stopper.step(val_mae, epoch):
                early_stopped = True
                stopping_reason = early_stopper.stopping_reason
                print(f"  [Early Stopping] {stopping_reason}")
                break

        total_time = time.time() - start_time
        print("==================================================")
        print(f"cGAN Training completed in {total_time:.2f}s.")
        print(f"Best Val MAE: {best_val_mae:.5f} (at Epoch {best_epoch})")
        print(f"Best Generator Checkpoint: {best_g_path}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("==================================================")

        # Log best generator model artifact with MLflow signature (use CPU for safe export)
        generator = generator.to("cpu")
        if best_g_path.is_file():
            try:
                best_state = torch.load(best_g_path, map_location="cpu")
                if isinstance(best_state, dict) and "model_state_dict" in best_state:
                    generator.load_state_dict(best_state["model_state_dict"])
            except Exception as load_exc:
                print(f"Warning: Could not reload best checkpoint for MLflow export: {load_exc}")

        generator.eval()
        in_c = generator.in_channels
        input_example = np.random.uniform(0.0, 1.0, size=(1, in_c, 256, 256)).astype(np.float32)
        with torch.no_grad():
            output_example = generator(torch.from_numpy(input_example)).numpy()
        model_signature = infer_signature(input_example, output_example)

        try:
            mlflow.pytorch.log_model(
                pytorch_model=generator,
                name="cgan_generator",
                signature=model_signature,
                input_example=input_example,
            )
        except Exception as mlflow_exc:
            print(f"Warning: MLflow model logging failed: {mlflow_exc}")

        # Always log best checkpoint files to MLflow artifact store
        if best_g_path.is_file():
            mlflow.log_artifact(str(best_g_path), artifact_path="checkpoints")
        if best_d_path.is_file():
            mlflow.log_artifact(str(best_d_path), artifact_path="checkpoints")

    return {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "total_time_seconds": total_time,
        "early_stopped": early_stopped,
        "stopping_reason": stopping_reason,
        "best_generator_checkpoint": str(best_g_path),
        "best_discriminator_checkpoint": str(best_d_path),
        "history": history,
    }


def main() -> None:
    """CLI entrypoint for cGAN training."""
    parser = argparse.ArgumentParser(description="Train Pix2Pix cGAN for optical cloud removal")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_cgan.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 epoch smoke test on minimal data subset",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    train_cgan_model(config=config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
