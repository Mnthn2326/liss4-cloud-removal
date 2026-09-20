"""Training loop for pix2pix cloud removal GAN.

Usage:
    python -m src.training.train --config configs/train_baseline.yaml

Reads all hyperparameters from the YAML config. Logs to MLflow (DagsHub)
per the tracking_uri in the config. Saves checkpoints to the model_dir
specified in the config.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.dataset import CloudDataset
from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.training.losses import GANLoss, PixelLoss


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_config(train_cfg: dict) -> dict:
    """Load the dataset config referenced by the training config."""
    ds_config_path = train_cfg.get("dataset", {}).get(
        "config_path", "configs/dataset_rice.yaml"
    )
    return load_config(ds_config_path)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


# ---------------------------------------------------------------------------
# Sample grid generation
# ---------------------------------------------------------------------------

def make_sample_grid(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 4,
) -> plt.Figure:
    """Generate a grid of (cloudy | generated | ground truth) samples.

    Returns a matplotlib Figure that can be saved or logged to MLflow.
    """
    generator.eval()
    clouds, fakes, gts = [], [], []

    with torch.no_grad():
        for cloud, gt in dataloader:
            cloud = cloud.to(device)
            fake = generator(cloud)
            clouds.append(cloud.cpu())
            fakes.append(fake.cpu())
            gts.append(gt)
            if sum(c.shape[0] for c in clouds) >= num_samples:
                break

    clouds = torch.cat(clouds)[:num_samples]
    fakes = torch.cat(fakes)[:num_samples]
    gts = torch.cat(gts)[:num_samples]

    # Denormalise from [-1,1] to [0,1] for display
    def denorm(t):
        return (t + 1.0) / 2.0

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        for j, (img, title) in enumerate([
            (clouds[i], "Cloudy"),
            (fakes[i], "Generated"),
            (gts[i], "Ground Truth"),
        ]):
            ax = axes[i, j]
            # (C,H,W) → (H,W,C), clamp to [0,1]
            img_np = denorm(img).permute(1, 2, 0).numpy().clip(0, 1)
            # Show only first 3 channels (RGB) even if more bands exist
            ax.imshow(img_np[..., :3])
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    generator.train()
    return fig


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    """Run the full training loop."""
    train_cfg = load_config(config_path)
    ds_cfg = load_dataset_config(train_cfg)

    # Unpack config values
    seed = train_cfg["training"].get("seed", 42)
    seed_everything(seed)

    device = get_device(train_cfg["training"].get("device", "auto"))
    epochs = train_cfg["training"]["epochs"]
    batch_size = train_cfg["training"]["batch_size"]
    lr = train_cfg["optimizer"]["lr"]
    betas = tuple(train_cfg["optimizer"]["betas"])
    lambda_l1 = train_cfg["loss"]["lambda_l1"]
    adv_loss_type = train_cfg["loss"].get("adversarial", "bce")
    log_every = train_cfg["training"].get("log_every_n_epochs", 1)
    grid_every = train_cfg["training"].get("sample_grid_every_n_epochs", 5)
    num_workers = train_cfg["training"].get("num_workers", 2)
    save_every = train_cfg["training"].get("save_every", 10)
    model_dir = train_cfg.get("experiment", {}).get(
        "model_dir", train_cfg.get("model_dir", "models/baseline")
    )
    checkpoint_dir = Path(model_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bands = ds_cfg["bands"]["count"]
    splits_dir = ds_cfg.get("paths", {}).get("splits_dir", "data/splits")
    dataset_name = ds_cfg["name"]

    print(f"=== Training config ===")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Lambda L1:  {lambda_l1}")
    print(f"  Bands:      {bands}")
    print(f"  Dataset:    {dataset_name}")

    # ---- Data ----
    train_csv = os.path.join(splits_dir, f"{dataset_name}_train.csv")
    val_csv = os.path.join(splits_dir, f"{dataset_name}_val.csv")

    train_ds = CloudDataset(train_csv)
    val_ds = CloudDataset(val_csv)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    # ---- Models ----
    gen = UNetGenerator(in_channels=bands, out_channels=bands).to(device)
    disc = PatchGANDiscriminator(in_channels=bands * 2).to(device)

    # ---- Optimisers ----
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    # ---- Losses ----
    criterion_adv = GANLoss(adv_loss_type).to(device)
    criterion_l1 = PixelLoss().to(device)

    # ---- MLflow setup ----
    mlflow_cfg = train_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    experiment_name = mlflow_cfg.get("experiment_name", "cloud-removal-baseline")

    if tracking_uri and "<your-" not in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        # For DagsHub, set env vars if needed
        os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    else:
        print("  WARNING: MLflow tracking_uri not configured. "
              "Logging locally to ./mlruns/")

    mlflow.set_experiment(experiment_name)

    # ---- Training ----
    best_val_loss = float("inf")
    metrics_log = {}

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_l1": lambda_l1,
            "adversarial_loss": adv_loss_type,
            "bands": bands,
            "dataset": dataset_name,
            "seed": seed,
        })

        for epoch in range(1, epochs + 1):
            gen.train()
            disc.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_l1_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for cloud, gt in pbar:
                cloud = cloud.to(device)
                gt = gt.to(device)

                # ---- Train Discriminator ----
                fake = gen(cloud)

                # Real pair
                real_pair = torch.cat([cloud, gt], dim=1)
                pred_real = disc(real_pair)
                loss_d_real = criterion_adv(pred_real, is_real=True)

                # Fake pair (detach generator)
                fake_pair = torch.cat([cloud, fake.detach()], dim=1)
                pred_fake = disc(fake_pair)
                loss_d_fake = criterion_adv(pred_fake, is_real=False)

                loss_d = (loss_d_real + loss_d_fake) * 0.5

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # ---- Train Generator ----
                fake_pair = torch.cat([cloud, fake], dim=1)
                pred_fake = disc(fake_pair)
                loss_g_adv = criterion_adv(pred_fake, is_real=True)
                loss_g_l1 = criterion_l1(fake, gt)
                loss_g = loss_g_adv + lambda_l1 * loss_g_l1

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                epoch_g_loss += loss_g.item()
                epoch_d_loss += loss_d.item()
                epoch_l1_loss += loss_g_l1.item()
                num_batches += 1

                pbar.set_postfix({
                    "G": f"{loss_g.item():.4f}",
                    "D": f"{loss_d.item():.4f}",
                    "L1": f"{loss_g_l1.item():.4f}",
                })

            # ---- Epoch averages ----
            avg_g = epoch_g_loss / max(num_batches, 1)
            avg_d = epoch_d_loss / max(num_batches, 1)
            avg_l1 = epoch_l1_loss / max(num_batches, 1)

            print(f"  Epoch {epoch}: G_loss={avg_g:.4f}, "
                  f"D_loss={avg_d:.4f}, L1_loss={avg_l1:.4f}")

            # ---- Log to MLflow ----
            if epoch % log_every == 0:
                mlflow.log_metrics({
                    "train/G_loss": avg_g,
                    "train/D_loss": avg_d,
                    "train/L1_loss": avg_l1,
                }, step=epoch)

            # ---- Sample grid ----
            if epoch % grid_every == 0:
                fig = make_sample_grid(gen, val_loader, device)
                grid_path = checkpoint_dir / f"grid_epoch_{epoch:03d}.png"
                fig.savefig(str(grid_path), dpi=100)
                plt.close(fig)
                mlflow.log_artifact(str(grid_path), "sample_grids")

            # ---- Checkpoint ----
            if epoch % save_every == 0 or epoch == epochs:
                ckpt = {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "config": train_cfg,
                }
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                torch.save(ckpt, str(ckpt_path))

            # ---- Track best ----
            if avg_l1 < best_val_loss:
                best_val_loss = avg_l1
                best_path = checkpoint_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                    "config": train_cfg,
                }, str(best_path))

        # ---- Final metrics ----
        metrics_log = {
            "final_epoch": epochs,
            "best_l1_loss": best_val_loss,
            "final_g_loss": avg_g,
            "final_d_loss": avg_d,
        }
        metrics_path = Path(model_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(metrics_path), "w") as f:
            json.dump(metrics_log, f, indent=2)

        mlflow.log_metrics({
            "best_l1_loss": best_val_loss,
        })

    print(f"=== Training complete ===")
    print(f"  Best L1 loss: {best_val_loss:.4f}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    print(f"  Metrics:      {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train pix2pix cloud removal GAN"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training YAML config",
    )
    args = parser.parse_args()
    train(args.config)
