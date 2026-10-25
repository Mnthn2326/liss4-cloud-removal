"""Convert IIRS/ISRO Keras Pix2Pix (.h5) checkpoint to PyTorch (.pt).

Converts weights from the generator trained in:
    Maniyar & Kumar (2021). "Generative Adversarial Network for Cloud Removal
    from Optical Temporal Satellite Imagery." Springer.

Usage:
    python scripts/convert_iirs_cgan.py
    python scripts/convert_iirs_cgan.py --h5-path "path/to/model.h5" --output-path "checkpoints/cgan/iirs_isro_cgan_generator.pt"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch

from cloud_removal.infer import load_model_from_checkpoint
from cloud_removal.models.iirs_pix2pix import IIRSPix2PixGenerator


def convert_h5_to_pytorch(
    h5_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Convert Keras .h5 weights into native PyTorch model and save checkpoint."""
    h5_p = Path(h5_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not h5_p.is_file():
        raise FileNotFoundError(f"Source HDF5 model not found: {h5_p}")

    print(f"[1/4] Initializing PyTorch IIRSPix2PixGenerator...")
    model = IIRSPix2PixGenerator(in_channels=3, out_channels=3, rescale_reflectance=True)
    model.eval()

    print(f"[2/4] Reading and mapping weights from {h5_p}...")
    with h5py.File(str(h5_p), "r") as f:
        w_grp = f["model_weights"]

        def get_w(layer: str, name: str) -> np.ndarray:
            grp = w_grp[layer]
            if layer in grp:
                return np.array(grp[layer][name])
            return np.array(grp[name])

        # 1. Convolutional Encoder Layers
        conv_mappings = [
            ("conv2d_7", model.e1_conv),
            ("conv2d_8", model.e2_conv),
            ("conv2d_9", model.e3_conv),
            ("conv2d_10", model.e4_conv),
            ("conv2d_11", model.e5_conv),
            ("conv2d_12", model.e6_conv),
            ("conv2d_13", model.e7_conv),
            ("conv2d_14", model.e8_conv),
        ]
        for keras_name, pt_layer in conv_mappings:
            kernel = get_w(keras_name, "kernel:0")  # (kH, kW, in, out)
            bias = get_w(keras_name, "bias:0")
            # Keras (kH, kW, in, out) -> PyTorch (out, in, kH, kW)
            pt_layer.weight.data.copy_(torch.from_numpy(kernel).permute(3, 2, 0, 1))
            pt_layer.bias.data.copy_(torch.from_numpy(bias))

        # 2. BatchNorm Layers
        bn_mappings = [
            ("batch_normalization_5", model.e2_bn),
            ("batch_normalization_6", model.e3_bn),
            ("batch_normalization_7", model.e4_bn),
            ("batch_normalization_8", model.e5_bn),
            ("batch_normalization_9", model.e6_bn),
            ("batch_normalization_10", model.e7_bn),
            ("batch_normalization_11", model.d1_bn),
            ("batch_normalization_12", model.d2_bn),
            ("batch_normalization_13", model.d3_bn),
            ("batch_normalization_14", model.d4_bn),
            ("batch_normalization_15", model.d5_bn),
            ("batch_normalization_16", model.d6_bn),
            ("batch_normalization_17", model.d7_bn),
        ]
        for keras_name, pt_layer in bn_mappings:
            gamma = get_w(keras_name, "gamma:0")
            beta = get_w(keras_name, "beta:0")
            moving_mean = get_w(keras_name, "moving_mean:0")
            moving_var = get_w(keras_name, "moving_variance:0")
            pt_layer.weight.data.copy_(torch.from_numpy(gamma))
            pt_layer.bias.data.copy_(torch.from_numpy(beta))
            pt_layer.running_mean.data.copy_(torch.from_numpy(moving_mean))
            pt_layer.running_var.data.copy_(torch.from_numpy(moving_var))

        # 3. Transposed Convolutional Decoder Layers
        deconv_mappings = [
            ("conv2d_transpose_1", model.d1_deconv),
            ("conv2d_transpose_2", model.d2_deconv),
            ("conv2d_transpose_3", model.d3_deconv),
            ("conv2d_transpose_4", model.d4_deconv),
            ("conv2d_transpose_5", model.d5_deconv),
            ("conv2d_transpose_6", model.d6_deconv),
            ("conv2d_transpose_7", model.d7_deconv),
            ("conv2d_transpose_8", model.d8_deconv),
        ]
        for keras_name, pt_layer in deconv_mappings:
            kernel = get_w(keras_name, "kernel:0")  # (kH, kW, out, in)
            bias = get_w(keras_name, "bias:0")
            # Keras (kH, kW, out, in) -> PyTorch (in, out, kH, kW)
            pt_layer.weight.data.copy_(torch.from_numpy(kernel).permute(3, 2, 0, 1))
            pt_layer.bias.data.copy_(torch.from_numpy(bias))

    print("[3/4] Performing verification forward pass on dummy patch [1, 3, 256, 256]...")
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3, 256, 256), f"Output shape mismatch: {out.shape}"
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0, (
        f"Output range out of bounds [0, 1]: min={out.min()}, max={out.max()}"
    )
    print(f"      Verification passed! Output in [{out.min():.4f}, {out.max():.4f}]")

    print(f"[4/4] Assembling full PyTorch checkpoint dictionary and saving to {out_p}...")
    checkpoint_payload: Dict[str, Any] = {
        "epoch": 1600,
        "model_state_dict": model.state_dict(),
        "bands": [3, 4, 8],  # Green, Red, NIR (Sentinel-2 B03, B04, B08 / LISS-IV B2, B3, B4)
        "band_names": ["B03", "B04", "B08"],
        "composite_mode": "fcc",
        "val_psnr": 24.726,
        "val_ssim": 0.7104,
        "model_config": model.get_config(),
        "source": {
            "title": "Generative Adversarial Network for Cloud Removal from Optical Temporal Satellite Imagery",
            "authors": "Chintan Maniyar, Dr. Anil Kumar",
            "affiliation": "Indian Institute of Remote Sensing (IIRS), ISRO, Dehradun",
            "published": "Soft Computing for Problem Solving, pp. 481-491. Springer (2021)",
            "doi": "10.1007/978-981-16-2712-5_39",
            "original_file": h5_p.name,
        },
    }

    torch.save(checkpoint_payload, out_p)
    file_size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"Successfully saved converted checkpoint ({file_size_mb:.2f} MB) to {out_p}")

    # Double check loading with load_model_from_checkpoint
    loaded_model, loaded_meta = load_model_from_checkpoint(out_p)
    print(f"Verified round-trip loading: model_type={type(loaded_model).__name__}")
    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IIRS/ISRO Keras Pix2Pix model to PyTorch.")
    parser.add_argument(
        "--h5-path",
        type=str,
        default=r"C:\Users\MANTHAN\Downloads\model_001600_4_10.h5",
        help="Path to downloaded Keras .h5 checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/cgan/iirs_isro_cgan_generator.pt",
        help="Target PyTorch .pt destination path",
    )
    args = parser.parse_args()

    convert_h5_to_pytorch(args.h5_path, args.output_path)
