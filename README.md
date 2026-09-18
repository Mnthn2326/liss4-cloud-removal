# Cloud Removal GAN — MLOps Course Project

Conditional-GAN pipeline for removing clouds from optical satellite imagery,
evaluated on the **RICE** dataset with planned extension to **ISRO LISS-IV**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce the full pipeline
dvc repro
```

## Repo Structure

```
configs/          YAML hyper-params & dataset configs
data/             DVC-tracked data (raw, processed, splits)
docs/             Design docs & specifications
models/           DVC-tracked model checkpoints
notebooks/        EDA & demo notebooks
pipelines/        DVC pipeline definition (dvc.yaml)
src/data/         Preprocessing & PyTorch Dataset
src/models/       Generator (U-Net) & Discriminator (PatchGAN)
src/training/     Training loop, losses, model registry
src/evaluation/   Metrics (PSNR, SSIM) & sample grids
tests/            Unit tests
```

## MLOps Stack

| Layer              | Tool                      |
|--------------------|---------------------------|
| Deep learning      | PyTorch 2.x               |
| Experiment tracking| MLflow (DagsHub-hosted)    |
| Data versioning    | DVC 3.x                   |
| Remote storage     | DagsHub Storage            |
| CI/CD              | GitHub Actions             |
| Config management  | PyYAML + plain YAML        |
