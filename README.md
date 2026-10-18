# LISS4 Cloud Removal

## Overview
LISS4 Cloud Removal is a demonstration MLOps project for removing cloud cover from high-resolution satellite imagery sourced from Bhoonidhi / ISRO (LISS-IV sensor).

The primary engineering focus of this project is establishing robust MLOps practices: data versioning, reproducibility, experiment tracking, and model lineage.

> **Note:** This repository has completed **Phase 1: Environment Setup**, **Phase 2: Bhoonidhi / ISRO Data Feasibility Assessment** (see [`docs/data_feasibility.md`](docs/data_feasibility.md) and [`docs/data_inspection.md`](docs/data_inspection.md)), **Phase 3: DVC Data Versioning and Dataset Management** (see [`docs/data_versioning.md`](docs/data_versioning.md)), **Phase 4: Geospatial Preprocessing Pipeline** (see [`docs/preprocessing.md`](docs/preprocessing.md)), **Phase 5: Synthetic Cloud Generation and Paired Dataset Preparation** (see [`docs/synthetic_dataset.md`](docs/synthetic_dataset.md)), **Phase 6: SEN12MS-CR Dataset Acquisition and Integration** (see [`docs/sen12mscr_dataset.md`](docs/sen12mscr_dataset.md)), **Phase 7: U-Net Baseline Training with MLflow** (see [`docs/training_unet.md`](docs/training_unet.md)), **Phase 7.1–7.5: Compact U-Net, Shared-Stretch Rendering & Demonstration**, **Phase 7.6–7.7: LISS-IV Cross-Sensor & Synthetic Benchmark Feasibility Assessments** (see [`reports/phase7_6_liss4_feasibility.md`](reports/phase7_6_liss4_feasibility.md) and [`reports/phase7_7_benchmark_feasibility.md`](reports/phase7_7_benchmark_feasibility.md)), and **Phase 8: Prototype Packaging and Final Verification**.

## Objectives
- Establish an end-to-end MLOps pipeline for satellite imagery cloud removal.
- Track and version data and model artifacts using DVC.
- Manage experiment parameters, metrics, and models using MLflow.
- Maintain a modular, reproducible, and testable Python codebase.

## Technology Stack
- **Python** (>= 3.10)
- **Git & GitHub** for source control and collaboration
- **PyTorch** for deep learning model development
- **DVC** for data and pipeline versioning
- **MLflow** for experiment tracking and model registry
- **Rasterio** for geospatial raster data handling
- **NumPy** for numerical array operations
- **Pytest** for automated unit and integration testing

## Environment Setup

### 1. Prerequisites
- Git
- Python 3.11 (or compatible Python >= 3.10)

### 2. Clone the Repository
```bash
git clone https://github.com/Mnthn2326/liss4-cloud-removal.git
cd liss4-cloud-removal
```

### 3. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
To install the initial development dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, to install the package in editable mode:
```bash
pip install -e .
```

## Running Tests
Run the setup verification test suite using Pytest:
```bash
pytest
```

## Preprocessing Pipeline
Reproduce the geospatial 3-band stacking and spatial validity mask stage using DVC:
```bash
dvc repro preprocess
```
See [`docs/preprocessing.md`](docs/preprocessing.md) for full pipeline specifications.

## Synthetic Cloud Simulation & Patch Extraction
Reusable utilities for stochastic cloud opacity synthesis, radiative atmospheric blending, validity-based collar filtering, and geographic zero-leakage patch splitting are implemented in `src/cloud_removal/`. See [`docs/synthetic_dataset.md`](docs/synthetic_dataset.md) for specifications and clear-reference acquisition prerequisites.

## SEN12MS-CR Paired Optical Dataset
A reproducible 719-pair optical subset ($256 \times 256$, 13-band Sentinel-2 Level-1C/2A GeoTIFFs) from the official training split of the SEN12MS-CR benchmark is tracked with DVC under `data/raw/sen12mscr/`. See [`docs/sen12mscr_dataset.md`](docs/sen12mscr_dataset.md) for specifications, band definitions, and PyTorch loader usage.

## U-Net Baseline Training & MLflow Tracking
A conventional 4-stage encoder-decoder U-Net (~7.76M parameters) is trained on paired Sentinel-2 RGB patches with L1 reconstruction loss, tracked via MLflow:
```bash
# Run quick verification smoke test (1 epoch, 16 train, 8 val)
python -m cloud_removal.train --smoke-test

# Run full local training
python -m cloud_removal.train --config configs/train_unet.yaml

# Run inference and generate side-by-side comparison figure
python -m cloud_removal.infer --checkpoint checkpoints/best_unet.pt --split val --index 0

# Launch interactive MLflow UI
mlflow ui
```
See [`docs/training_unet.md`](docs/training_unet.md) for full architecture diagrams, training parameters, smoke run results, and Google Colab GPU training instructions.

## Minimal Demonstration Workflow

A minimal reproducible demonstration loads the trained compact U-Net (`checkpoints/compact/best_compact_unet.pt`, 1.94M parameters, Epoch 5) and executes end-to-end inference on a validation patch without retraining or re-downloading the dataset:

```bash
python demo.py
```

Command-line options:
```bash
python demo.py --index 0 --output reports/figures/demo_comparison.png --device auto
```

The demonstration evaluates physical surface reflectance in $[0.0, 1.0]$ (reporting MAE and PSNR improvements) and saves a 3-panel comparison figure (`[Cloudy Input | U-Net Prediction | Clear Ground Truth]`) rendered with a shared 2%–98% percentile contrast stretch.

## Scientific Boundaries and Limitations

1. **Optical Cloud Penetration Boundary**: Optical sensor wavelengths cannot physically penetrate optically thick, opaque cloud cover. The U-Net reconstructs surface features using contextual spatial priors from surrounding visible pixels and learned surface reflectance distributions. It is an inpainting / imputation model, not a physical cloud penetrator.
2. **Dataset Holdout Boundary**: The 144 validation patches are an internal holdout split from official SEN12MS-CR training ROI 102 (Winter). They monitor generalization during development and do not represent performance on external official benchmark test ROIs.
3. **Training & Demonstration Provenance**: The compact baseline was trained locally on CPU (5 epochs, Run ID `0f2e8b45bb024ec5bd6187d1d2a9d8aa`, validation MAE `0.00667`, PSNR `38.94 dB`). Google Colab provides an accelerated GPU environment for exploration and demonstration.
4. **LISS-IV Cross-Sensor Assessment**: As documented in the Phase 7.6 and Phase 7.7 feasibility reports ([`reports/phase7_6_liss4_feasibility.md`](reports/phase7_6_liss4_feasibility.md), [`reports/phase7_7_benchmark_feasibility.md`](reports/phase7_7_benchmark_feasibility.md)), the Sentinel-2 RGB model cannot be directly transferred to Resourcesat-2A LISS-IV due to spatial resolution (5 m vs 10 m), spectral bandpass (Green, Red, NIR vs RGB), and radiometric calibration gaps ($ESUN_\lambda$ omitted in product metadata). No cross-sensor accuracy is claimed.
