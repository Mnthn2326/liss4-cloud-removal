# Remote-Sensing Cloud Reconstruction

## Overview
**Remote-Sensing Cloud Reconstruction** is an engineering-first MLOps prototype for deep-learning reconstruction of surface reflectance obscured by clouds in spaceborne optical satellite observations.

The official production baseline is trained and evaluated on the **RICE2** paired Landsat-style RGB benchmark ($512 \times 512$ tiles) using a lightweight **Compact U-Net** (1.94M parameters, compound L1 + SSIM loss, 25.12 dB test PSNR).

> **Scientific Boundary & Scope Notice:**
> - **Supported Domain:** Landsat/Sentinel-2-style 3-channel RGB surface reflectance normalized strictly in $[0.0, 1.0]$.
> - **Terminology:** Model outputs are **reconstructed image estimates** derived from contextual spatial priors and learned reflectance distributions. Thick optical clouds cannot be physically penetrated.
> - **ISRO LISS-IV Status:** Historical exploration of Resourcesat-2A LISS-IV (5 m GSD, Green/Red/NIR bands, missing Blue band, uncalibrated DNs) is classified as **experimental research** and is unsupported for deployed production inference in Version 1.0. See [`docs/future_work/liss4_roadmap.md`](docs/future_work/liss4_roadmap.md) for feasibility findings and the cross-sensor roadmap.

---

## Key Documentation

| Document | Purpose |
| :--- | :--- |
| **[RICE2 Dataset Specification](docs/rice2_dataset.md)** | Details of the 736-scene RICE2 benchmark, directory structure, SHA-256 manifests, and split disjointness |
| **[Model Card (v1.0.0)](docs/model_card.md)** | Full specification of the Compact U-Net baseline, architecture, training hyperparameters, and evaluation metrics |
| **[5-Minute Defense Demo Guide](docs/demo_guide.md)** | Step-by-step presentation script for evaluators, covering UI, API, metrics, and Prometheus monitoring |
| **[Scientific & Sensor Limitations](docs/limitations.md)** | In-depth analysis of optical penetration physics, temporal reference decorrelation, and operational boundaries |
| **[RICE2 Benchmark Report](reports/rice2_benchmark_report.md)** | Full quantitative held-out test evaluation report with regional cloud-only metrics and visual comparisons |
| **[LISS-IV Research Roadmap](docs/future_work/liss4_roadmap.md)** | Technical analysis of ISRO Resourcesat-2A LISS-IV sensor gaps and future cross-sensor transfer roadmap |

---

## Benchmark Performance (RICE2 Untouched Test Split, $N=148$)

Evaluated on the independent, held-out test split of the RICE2 benchmark:

| Metric | Cloudy Input Baseline | Compact U-Net Reconstructed | Improvement |
| :--- | :--- | :--- | :--- |
| **Peak Signal-to-Noise Ratio (PSNR)** | 23.33 dB | **25.12 dB** | **+1.79 dB** |
| **Structural Similarity (SSIM)** | 0.8029 | **0.8510** | **+0.0481** |
| **Cloud-Region MAE** | 0.1294 | **0.0604** | **-53.3% Error** |
| **Overall Mean Absolute Error (MAE)** | 0.0650 | **0.0491** | **-24.4% Error** |

---

## Objectives
- Establish an end-to-end, reproducible MLOps pipeline for satellite cloud reconstruction.
- Track and version data and model artifacts using DVC.
- Manage experiment parameters, metrics, and models using MLflow.
- Serve low-latency inference via containerized FastAPI with Prometheus and Grafana observability.
- Maintain a modular, reproducible, and testable Python codebase with continuous integration.

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

## Official Production Benchmark: RICE2 Dataset

The official benchmark for Version 1.0 is the **RICE2** paired optical cloud reconstruction benchmark (736 scenes, $512 \times 512$ RGB, with paired cloudy observations, clear references, and binary cloud masks):
- **Train split:** 471 samples (64%)
- **Validation split:** 117 samples (16%)
- **Test split (Untouched):** 148 samples (20%)

```bash
# Generate deterministic RICE2 manifest with SHA-256 hashes
python -m cloud_removal.rice2 --dataset-root data/raw/RICE2 --output data/manifests/rice2_manifest.json

# Train Compact U-Net baseline on RICE2 (compound L1 + SSIM)
python -m cloud_removal.train_rice2 --config configs/train_rice2_compact_unet.yaml

# Evaluate on the untouched held-out test split
python -m cloud_removal.evaluate_rice2 --checkpoint checkpoints/rice2/best_rice2_compact_unet.pt --manifest data/manifests/rice2_manifest.json --split test --output reports/rice2_test_metrics.json
```

## Exploratory & Benchmark Datasets

### SEN12MS-CR Paired Optical Dataset
A reproducible 719-pair optical subset ($256 \times 256$, 13-band Sentinel-2 Level-1C/2A GeoTIFFs) from the official training split of the SEN12MS-CR benchmark is tracked with DVC under `data/raw/sen12mscr/`. See [`docs/sen12mscr_dataset.md`](docs/sen12mscr_dataset.md).

### Resourcesat-2A LISS-IV (Experimental)
LISS-IV ingestion and validation tooling is retained under `src/cloud_removal/ingest_liss4.py` for exploratory research. Because LISS-IV lacks a blue band and calibrated reflectance parameters in open metadata, it is **unsupported for production inference in v1.0**. See [`docs/future_work/liss4_roadmap.md`](docs/future_work/liss4_roadmap.md).

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


## Docker + FastAPI Inference Service (Phase 10)

A containerized, production-oriented FastAPI inference service exposes the trained compact U-Net model via deterministic CPU execution. See [`docs/api.md`](docs/api.md) for full endpoint specifications, request/response schemas, and architectural boundaries.

### 1. Prerequisites & Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Run the Service Locally
```bash
# Launch FastAPI service with default RICE2 compact checkpoint
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000

# Or configure an explicit checkpoint path via MODEL_CHECKPOINT:
# (PowerShell)
$env:MODEL_CHECKPOINT="checkpoints/rice2/best_rice2_compact_unet.pt"
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs` and `http://localhost:8000/openapi.json`.
The interactive Web UI is accessible at `http://localhost:8000/` (see [`docs/ui.md`](docs/ui.md)).

### 3. Build & Run with Docker
Model checkpoints are never baked into the container image. Inject trained weights dynamically via a read-only volume mount:
```bash
# Build production image
docker build -t liss4-cloud-removal-api:latest .

# Run container with read-only checkpoint mount
docker run -d \
  --name cloud-removal-service \
  -p 8000:8000 \
  -v ${PWD}/checkpoints/rice2/best_rice2_compact_unet.pt:/models/checkpoint.pt:ro \
  -e MODEL_CHECKPOINT=/models/checkpoint.pt \
  liss4-cloud-removal-api:latest
```

### 4. Example API Requests

- **Liveness probe (`/health`)**:
  ```bash
  curl -s http://localhost:8000/health
  # {"status":"healthy","service":"liss4-cloud-removal-api","version":"0.1.0","model_loaded":true,"ready":true}
  ```

- **Readiness probe (`/ready`)**:
  ```bash
  curl -s http://localhost:8000/ready
  # {"ready":true,"status":"ready","model_type":"UNet","device":"cpu","checkpoint":"...","in_channels":3,"out_channels":3}
  ```

- **Inference request (`/predict`)**:
  Accepts a 3D nested array `[3, H, W]` ($16 \le H, W \le 512$) of Sentinel-2 RGB surface reflectance values strictly bounded in $[0.0, 1.0]$.
  ```bash
  # Generate a 3x32x32 synthetic patch and send inference request:
  python -c "import json, numpy as np; json.dump({'image': np.random.uniform(0.1, 0.4, (3, 32, 32)).tolist()}, open('sample.json', 'w'))"
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @sample.json
  ```

### 5. Running API Tests
```bash
# Run unit and integration tests for API service (no real checkpoint required)
pytest tests/test_api.py -v

# Run full project test suite (81 tests)
pytest -v
```

### 6. Troubleshooting
- **`503 Service Unavailable` on `/ready` or `/predict`**: The model checkpoint is not loaded. Ensure `MODEL_CHECKPOINT` points to an existing `.pt` file, or that the default checkpoint exists at `checkpoints/compact/best_compact_unet.pt`.
- **`422 Unprocessable Content` on `/predict`**: Input validation failed. Verify that:
  - The array is 3D with exactly 3 channels (RGB: [B04, B03, B02]).
  - Spatial dimensions $H$ and $W$ are between 16 and 512 without ragged rows.
  - All values are finite numbers (no NaN or Inf).
  - Reflectance values are strictly in $[0.0, 1.0]$ (unscaled digital numbers $> 1.0$ or unphysical negative values are explicitly rejected).

## Prometheus + Grafana Monitoring (Phase 11)

A complete observability stack integrates real-time Prometheus metrics collection and Grafana dashboard visualization with the FastAPI inference service. See [`docs/monitoring.md`](docs/monitoring.md) for architecture, PromQL queries, and metric definitions.

### 1. Observability Stack Components
- **FastAPI Service (`:8000`)**: Instruments `/metrics` tracking throughput, status codes, request latency histograms, and forward-pass inference duration.
- **Prometheus (`:9090`)**: Scrapes the inference service at 5-second intervals.
- **Grafana (`:3001`)**: Pre-provisioned dashboard displaying live model readiness, request rates, error distributions, and latency quantiles ($p50$, $p95$). Default credentials: `admin` / `admin`. (Port configurable via `GRAFANA_PORT`).

### 2. Start Stack with Docker Compose
```bash
# Launch API, Prometheus, and Grafana
docker compose up -d --build

# View running services
docker compose ps
```

### 3. Run End-to-End Monitoring Demo
Generate representative traffic (liveness probes, valid inference requests, and 422 edge-case rejections) to populate live metrics:
```bash
python scripts/demo_monitoring.py --requests 15
```

### 4. Stop Stack
```bash
docker compose down
```

## Scientific Boundaries and Limitations

1. **Optical Cloud Penetration Boundary**: Optical sensor wavelengths cannot physically penetrate optically thick, opaque cloud cover. The U-Net reconstructs surface features using contextual spatial priors from surrounding visible pixels and learned surface reflectance distributions. It is an inpainting / imputation model, not a physical cloud penetrator.
2. **Dataset Holdout Boundary**: The 144 validation patches are an internal holdout split from official SEN12MS-CR training ROI 102 (Winter). They monitor generalization during development and do not represent performance on external official benchmark test ROIs.
3. **Training & Demonstration Provenance**: The compact baseline was trained locally on CPU (5 epochs, Run ID `0f2e8b45bb024ec5bd6187d1d2a9d8aa`, validation MAE `0.00667`, PSNR `38.94 dB`). Google Colab provides an accelerated GPU environment for exploration and demonstration.
4. **LISS-IV Cross-Sensor Assessment**: As documented in the Phase 7.6 and Phase 7.7 feasibility reports ([`reports/phase7_6_liss4_feasibility.md`](reports/phase7_6_liss4_feasibility.md), [`reports/phase7_7_benchmark_feasibility.md`](reports/phase7_7_benchmark_feasibility.md)), the Sentinel-2 RGB model cannot be directly transferred to Resourcesat-2A LISS-IV due to spatial resolution (5 m vs 10 m), spectral bandpass (Green, Red, NIR vs RGB), and radiometric calibration gaps ($ESUN_\lambda$ omitted in product metadata). No cross-sensor accuracy is claimed.
