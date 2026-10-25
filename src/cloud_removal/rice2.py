"""RICE2 paired RGB cloud-reconstruction dataset support.

RICE2 contains matching Landsat-style RGB cloudy images, temporally nearby clear
references, and quality masks.  This module provides a deterministic data
contract for training and evaluation; it does not treat the clear reference as
same-instant physical ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test", "all"]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class RICE2Config:
    """Configuration for deterministic RICE2 loading and splitting."""

    train_ratio: float = 0.64
    val_ratio: float = 0.16
    test_ratio: float = 0.20
    seed: int = 42
    augment: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be positive")


@dataclass(frozen=True)
class RICE2Sample:
    """A single aligned RICE2 cloudy/reference/mask triplet."""

    sample_id: str
    cloudy_path: Path
    clear_path: Path
    mask_path: Path
    split: str


def _sample_sort_key(sample_id: str) -> Tuple[int, Union[int, str]]:
    """Naturally sort numeric RICE2 file stems before deterministic shuffling."""
    return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_index(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"RICE2 directory not found: {directory}")
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES}


def _validate_image_triplet(sample_id: str, cloudy: Path, clear: Path, mask: Path) -> None:
    """Validate existence, readability, non-empty content, RGB mode, and dimension match."""
    for p in (cloudy, clear, mask):
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"Corrupted or missing file for sample '{sample_id}': {p}")

    with Image.open(cloudy) as c_img, Image.open(clear) as l_img, Image.open(mask) as m_img:
        if not (c_img.size == l_img.size == m_img.size):
            raise ValueError(
                f"Inconsistent spatial dimensions for sample '{sample_id}': "
                f"cloudy={c_img.size}, clear={l_img.size}, mask={m_img.size}"
            )
        if c_img.mode not in ("RGB", "RGBA") or l_img.mode not in ("RGB", "RGBA"):
            raise ValueError(
                f"Non-RGB optical format for sample '{sample_id}': "
                f"cloudy mode={c_img.mode}, clear mode={l_img.mode}. Expected 3-band RGB."
            )


def build_rice2_manifest(
    dataset_root: Union[str, Path] = "data/raw/RICE2",
    output_path: Union[str, Path] = "data/manifests/rice2_manifest.json",
    config: Optional[RICE2Config] = None,
    include_hashes: bool = True,
    validate_integrity: bool = True,
) -> Dict[str, Any]:
    """Validate RICE2 folders and write an immutable-style paired-data manifest."""
    root = Path(dataset_root)
    cfg = config or RICE2Config()
    indexes = {
        "cloud": _image_index(root / "cloud"),
        "label": _image_index(root / "label"),
        "mask": _image_index(root / "mask"),
    }
    all_ids = set().union(*(set(index) for index in indexes.values()))
    shared_ids = set.intersection(*(set(index) for index in indexes.values()))
    if all_ids != shared_ids:
        missing = {name: sorted(all_ids - set(index)) for name, index in indexes.items()}
        raise ValueError(f"RICE2 folders do not contain identical sample IDs: {missing}")
    if not shared_ids:
        raise ValueError(f"No paired RICE2 images found under {root}")

    ids = sorted(shared_ids, key=_sample_sort_key)
    generator = np.random.default_rng(cfg.seed)
    shuffled = list(generator.permutation(ids))
    n_total = len(shuffled)
    n_train = int(n_total * cfg.train_ratio)
    n_val = int(n_total * cfg.val_ratio)
    split_map = {
        sample_id: ("train" if idx < n_train else "val" if idx < n_train + n_val else "test")
        for idx, sample_id in enumerate(shuffled)
    }

    def record(sample_id: str) -> Dict[str, Any]:
        paths = {key: indexes[key][sample_id] for key in indexes}
        if validate_integrity:
            _validate_image_triplet(sample_id, paths["cloud"], paths["label"], paths["mask"])
        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "cloudy_path": paths["cloud"].relative_to(root.parent).as_posix(),
            "clear_path": paths["label"].relative_to(root.parent).as_posix(),
            "mask_path": paths["mask"].relative_to(root.parent).as_posix(),
            "split": split_map[sample_id],
        }
        if include_hashes:
            item["sha256"] = {name: _sha256(path) for name, path in paths.items()}
        return item

    samples = [record(sample_id) for sample_id in ids]
    manifest = {
        "dataset_name": "RICE2",
        "dataset_root": root.as_posix(),
        "description": "Paired Landsat-style RGB cloud reconstruction benchmark with quality masks.",
        "reference_boundary": "Clear references may be temporally separated from cloudy images.",
        "image_layout": "RGB PNG, 512x512 pixels",
        "mask_rule": "Any non-black quality-mask pixel is treated as cloud-affected for regional metrics.",
        "split_strategy": "Deterministic shuffled sample-level split",
        "split_config": {
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "seed": cfg.seed,
        },
        "counts": {
            "total": n_total,
            "train": n_train,
            "val": n_val,
            "test": n_total - n_train - n_val,
        },
        "samples": samples,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def _load_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.any(array != 0, axis=-1, keepdims=False).astype(np.float32)[None, ...])


def _augment_pair(
    cloudy: torch.Tensor, clear: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one random geometry transform identically to image and mask tensors."""
    if torch.rand(()) > 0.5:
        cloudy, clear, mask = (torch.flip(tensor, dims=(-1,)) for tensor in (cloudy, clear, mask))
    if torch.rand(()) > 0.5:
        cloudy, clear, mask = (torch.flip(tensor, dims=(-2,)) for tensor in (cloudy, clear, mask))
    turns = int(torch.randint(0, 4, ()).item())
    if turns:
        cloudy, clear, mask = (torch.rot90(tensor, turns, dims=(-2, -1)) for tensor in (cloudy, clear, mask))
    return cloudy, clear, mask


class RICE2Dataset(Dataset[Dict[str, Any]]):
    """PyTorch dataset for aligned RGB RICE2 samples and binary quality masks."""

    def __init__(self, samples: Sequence[RICE2Sample], config: Optional[RICE2Config] = None) -> None:
        self.samples = list(samples)
        self.config = config or RICE2Config()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        cloudy, clear, mask = (
            _load_rgb(sample.cloudy_path),
            _load_rgb(sample.clear_path),
            _load_mask(sample.mask_path),
        )
        if cloudy.shape != clear.shape or cloudy.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"RICE2 sample '{sample.sample_id}' has inconsistent image or mask dimensions")
        if self.config.augment and sample.split == "train":
            cloudy, clear, mask = _augment_pair(cloudy, clear, mask)
        return {"cloudy": cloudy, "clear": clear, "cloud_mask": mask, "sample_id": sample.sample_id, "split": sample.split}

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        split: SplitName = "train",
        config: Optional[RICE2Config] = None,
    ) -> "RICE2Dataset":
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"Unsupported split '{split}'")
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = Path(payload.get("dataset_root", "data/raw/RICE2"))
        base = root.parent
        samples = []
        for item in payload.get("samples", []):
            if split != "all" and item.get("split") != split:
                continue
            samples.append(RICE2Sample(item["sample_id"], base / item["cloudy_path"], base / item["clear_path"], base / item["mask_path"], item["split"]))
        if not samples:
            raise ValueError(f"No RICE2 samples for split '{split}' in {path}")
        return cls(samples=samples, config=config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RICE2 and create a paired-data manifest")
    parser.add_argument("--dataset-root", default="data/raw/RICE2")
    parser.add_argument("--output", default="data/manifests/rice2_manifest.json")
    parser.add_argument("--no-hashes", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_rice2_manifest(args.dataset_root, args.output, include_hashes=not args.no_hashes)
    print(f"Created {args.output}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
