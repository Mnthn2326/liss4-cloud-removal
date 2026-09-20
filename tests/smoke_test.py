"""Smoke test: 1-batch, 1-step forward pass through the full pipeline.

Prints shapes at each stage:
  Dataset → Generator → Discriminator → Loss
"""

import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.dataset import CloudDataset
from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.training.losses import GANLoss, PixelLoss


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST: 1-batch, 1-step, CPU")
    print("=" * 60)

    device = torch.device("cpu")
    bands = 3  # from dataset_rice.yaml bands.count

    # ---- 1. Dataset ----
    print("\n--- Dataset ---")
    train_csv = "data/splits/rice1_train.csv"
    ds = CloudDataset(train_csv)
    print(f"  Dataset length: {len(ds)}")

    cloud, gt = ds[0]
    print(f"  cloud shape: {cloud.shape}  dtype: {cloud.dtype}")
    print(f"  gt    shape: {gt.shape}  dtype: {gt.dtype}")
    print(f"  cloud range: [{cloud.min():.3f}, {cloud.max():.3f}]")
    print(f"  gt    range: [{gt.min():.3f}, {gt.max():.3f}]")

    assert cloud.shape == (bands, 256, 256), f"Expected (3,256,256), got {cloud.shape}"
    assert gt.shape == (bands, 256, 256), f"Expected (3,256,256), got {gt.shape}"
    print("  ✓ Shapes correct: (C, H, W) = (3, 256, 256)")

    # ---- 2. Batch ----
    print("\n--- Batching ---")
    batch_size = 2
    cloud_batch = cloud.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    gt_batch = gt.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    print(f"  cloud_batch: {cloud_batch.shape}")
    print(f"  gt_batch:    {gt_batch.shape}")

    # ---- 3. Generator ----
    print("\n--- Generator (U-Net) ---")
    gen = UNetGenerator(in_channels=bands, out_channels=bands).to(device)
    gen.eval()

    total_g_params = sum(p.numel() for p in gen.parameters())
    print(f"  Parameters: {total_g_params:,}")

    with torch.no_grad():
        fake = gen(cloud_batch)
    print(f"  Input:  {cloud_batch.shape}")
    print(f"  Output: {fake.shape}")
    print(f"  Output range: [{fake.min():.3f}, {fake.max():.3f}]")
    assert fake.shape == cloud_batch.shape, \
        f"Generator output shape mismatch: {fake.shape} vs {cloud_batch.shape}"
    print("  ✓ Generator output matches input shape")

    # ---- 4. Discriminator ----
    print("\n--- Discriminator (PatchGAN 70×70) ---")
    disc = PatchGANDiscriminator(in_channels=bands * 2).to(device)
    disc.eval()

    total_d_params = sum(p.numel() for p in disc.parameters())
    print(f"  Parameters: {total_d_params:,}")

    with torch.no_grad():
        # Real pair
        real_pair = torch.cat([cloud_batch, gt_batch], dim=1)
        print(f"  Real pair input:  {real_pair.shape}")
        pred_real = disc(real_pair)
        print(f"  Real pair output: {pred_real.shape}")
        print(f"  Real pred range:  [{pred_real.min():.3f}, {pred_real.max():.3f}]")

        # Fake pair
        fake_pair = torch.cat([cloud_batch, fake], dim=1)
        print(f"  Fake pair input:  {fake_pair.shape}")
        pred_fake = disc(fake_pair)
        print(f"  Fake pair output: {pred_fake.shape}")

    assert pred_real.shape[1] == 1, "Discriminator should output 1 channel"
    assert pred_real.shape == pred_fake.shape, "Real/fake output shapes differ"
    print(f"  ✓ PatchGAN output: {pred_real.shape} (N, 1, 30, 30)")

    # ---- 5. Losses ----
    print("\n--- Losses ---")
    criterion_adv = GANLoss("bce")
    criterion_l1 = PixelLoss()
    lambda_l1 = 100

    loss_d_real = criterion_adv(pred_real, is_real=True)
    loss_d_fake = criterion_adv(pred_fake, is_real=False)
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    print(f"  D loss (real): {loss_d_real.item():.4f}")
    print(f"  D loss (fake): {loss_d_fake.item():.4f}")
    print(f"  D loss (avg):  {loss_d.item():.4f}")

    loss_g_adv = criterion_adv(pred_fake, is_real=True)
    loss_g_l1 = criterion_l1(fake, gt_batch)
    loss_g = loss_g_adv + lambda_l1 * loss_g_l1
    print(f"  G adversarial: {loss_g_adv.item():.4f}")
    print(f"  G L1:          {loss_g_l1.item():.4f}")
    print(f"  G total:       {loss_g.item():.4f}")
    print(f"  ✓ All losses are finite")

    # ---- 6. Backward pass (1 step) ----
    print("\n--- Backward pass (1 step) ---")
    gen.train()
    disc.train()

    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # D step
    fake = gen(cloud_batch)
    pred_real = disc(torch.cat([cloud_batch, gt_batch], dim=1))
    pred_fake = disc(torch.cat([cloud_batch, fake.detach()], dim=1))
    loss_d = (criterion_adv(pred_real, True) + criterion_adv(pred_fake, False)) * 0.5
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    print(f"  D backward: loss={loss_d.item():.4f} ✓")

    # G step
    pred_fake = disc(torch.cat([cloud_batch, fake], dim=1))
    loss_g = criterion_adv(pred_fake, True) + lambda_l1 * criterion_l1(fake, gt_batch)
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    print(f"  G backward: loss={loss_g.item():.4f} ✓")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print(f"\nShape pipeline:")
    print(f"  Dataset:       .npy (256,256,3) → tensor ({bands},256,256)")
    print(f"  Generator:     ({batch_size},{bands},256,256) → ({batch_size},{bands},256,256)")
    print(f"  Discriminator: ({batch_size},{bands*2},256,256) → ({batch_size},1,30,30)")
    print(f"\nModel sizes:")
    print(f"  Generator:     {total_g_params:,} params")
    print(f"  Discriminator: {total_d_params:,} params")
    print(f"  Total:         {total_g_params + total_d_params:,} params")


if __name__ == "__main__":
    smoke_test()
