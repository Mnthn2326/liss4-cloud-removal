# LISS4 Cloud Removal

## Overview
LISS4 Cloud Removal is a demonstration MLOps project for removing cloud cover from high-resolution satellite imagery sourced from Bhoonidhi / ISRO (LISS-IV sensor).

The primary engineering focus of this project is establishing robust MLOps practices: data versioning, reproducibility, experiment tracking, and model lineage.

> **Note:** This repository has completed **Phase 1: Environment Setup**, **Phase 2: Bhoonidhi / ISRO Data Feasibility Assessment** (see [`docs/data_feasibility.md`](docs/data_feasibility.md) and [`docs/data_inspection.md`](docs/data_inspection.md)), **Phase 3: DVC Data Versioning and Dataset Management** (see [`docs/data_versioning.md`](docs/data_versioning.md)), **Phase 4: Geospatial Preprocessing Pipeline** (see [`docs/preprocessing.md`](docs/preprocessing.md)), **Phase 5: Synthetic Cloud Generation and Paired Dataset Preparation** (see [`docs/synthetic_dataset.md`](docs/synthetic_dataset.md)), **Phase 6: SEN12MS-CR Dataset Acquisition and Integration** (see [`docs/sen12mscr_dataset.md`](docs/sen12mscr_dataset.md)), **Phase 7: U-Net Baseline Training with MLflow** (see [`docs/training_unet.md`](docs/training_unet.md)), **Phase 7.1–7.5: Compact U-Net, Shared-Stretch Rendering & Demonstration**, **Phase 7.6–7.7: LISS-IV Cross-Sensor & Synthetic Benchmark Feasibility Assessments** (see [`reports/phase7_6_liss4_feasibility.md`](reports/phase7_6_liss4_feasibility.md) and [`reports/phase7_7_benchmark_feasibility.md`](reports/phase7_7_benchmark_feasibility.md)), **Phase 8: Prototype Packaging and Final Verification**, **Phase 9: Continuous Integration with GitHub Actions**, **Phase 10: Docker + FastAPI Inference Service** (see [`docs/api.md`](docs/api.md)), and **Phase 11: Prometheus + Grafana Monitoring Stack** (see [`docs/monitoring.md`](docs/monitoring.md)).
>
> 📖 **Project Guide Walkthroughs:** For the comprehensive technical walkthrough and defense Q&A, see [`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md). For a quick 10-minute beginner-friendly cheat sheet, see [`docs/SIMPLE_PROJECT_WALKTHROUGH.md`](docs/SIMPLE_PROJECT_WALKTHROUGH.md).

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


## Docker + FastAPI Inference Service (Phase 10)

A containerized, production-oriented FastAPI inference service exposes the trained compact U-Net model via deterministic CPU execution. See [`docs/api.md`](docs/api.md) for full endpoint specifications, request/response schemas, and architectural boundaries.

### 1. Prerequisites & Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Run the Service Locally
```bash
# Launch FastAPI service with default compact checkpoint
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000

# Or configure an explicit checkpoint path via MODEL_CHECKPOINT:
# (PowerShell)
$env:MODEL_CHECKPOINT="checkpoints/compact/best_compact_unet.pt"
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
  -v ${PWD}/checkpoints/compact/best_compact_unet.pt:/models/checkpoint.pt:ro \
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
