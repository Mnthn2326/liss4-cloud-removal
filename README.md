# LISS4 Cloud Removal

## Overview
LISS4 Cloud Removal is a demonstration MLOps project for removing cloud cover from high-resolution satellite imagery sourced from Bhoonidhi / ISRO (LISS-IV sensor).

The primary engineering focus of this project is establishing robust MLOps practices: data versioning, reproducibility, experiment tracking, and model lineage.

> **Note:** This repository has completed **Phase 1: Environment Setup** and **Phase 2: Bhoonidhi / ISRO Data Feasibility Assessment** (see [`docs/data_feasibility.md`](docs/data_feasibility.md)). The data ingestion, preprocessing, training pipelines, experiment tracking workflows, and model deployment have not yet been implemented.

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
git clone https://github.com/<your-org>/liss4-cloud-removal.git
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
