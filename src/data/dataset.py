"""PyTorch Dataset for cloud-removal patch pairs.

Reads a split CSV (e.g. data/splits/rice1_train.csv), loads the
corresponding .npy cloud/gt patches, and returns them as (C,H,W) tensors.

Usage:
    from src.data.dataset import CloudDataset
    ds = CloudDataset("data/splits/rice1_train.csv")
    cloud, gt = ds[0]  # both shape (C, H, W), dtype float32
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    """Dataset of (cloudy, ground-truth) patch pairs.

    Each .npy file stores a float32 array of shape (H, W, C) — this class
    transposes to (C, H, W) for PyTorch's Conv2d expectations.

    Parameters
    ----------
    split_csv : str
        Path to a split manifest CSV with columns:
        image_id, patch_name, cloud_path, gt_path
    """

    def __init__(self, split_csv: str):
        self.records = []
        with open(split_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(
                    (row["cloud_path"], row["gt_path"])
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cloud_path, gt_path = self.records[idx]

        # Load .npy arrays — shape (H, W, C), dtype float32
        cloud = np.load(cloud_path)
        gt = np.load(gt_path)

        # Transpose (H, W, C) → (C, H, W) for PyTorch Conv2d
        cloud = np.transpose(cloud, (2, 0, 1))  # (C, H, W)
        gt = np.transpose(gt, (2, 0, 1))          # (C, H, W)

        return torch.from_numpy(cloud), torch.from_numpy(gt)
