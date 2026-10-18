# Docker + FastAPI Inference Service (Phase 10)

## Overview

The `liss4-cloud-removal` API service provides a production-oriented, containerized HTTP interface for optical satellite imagery cloud removal using the trained compact U-Net baseline architecture.

The service is built with **FastAPI** and served via **Uvicorn**, designed specifically for deterministic CPU-based inference on Sentinel-2 RGB surface reflectance patches.

---

## Scientific Boundaries & Model Limitations

> [!WARNING]
> **Cross-Sensor Inference Boundary**:
> The compact U-Net baseline (`checkpoints/compact/best_compact_unet.pt`) was trained strictly on **Sentinel-2 SEN12MS-CR** Level-1C/Level-2A imagery at **10 m spatial resolution** using RGB bands (**B04, B03, B02**).
> 
> As established in Phase 7.6 and Phase 7.7 feasibility assessments:
> 1. **Spatial Resolution**: Resourcesat-2A LISS-IV acquires at 5 m resolution, whereas the model expects 10 m.
> 2. **Spectral Bandpass**: LISS-IV acquires Green (Band 2), Red (Band 3), and NIR (Band 4); it has no optical Blue band.
> 3. **Radiometric Calibration**: Product metadata for LISS-IV omits mean exo-atmospheric solar irradiance ($ESUN_\lambda$), precluding defensible physical reflectance normalization.
>
> Successful execution of this API does **not** validate operational cloud removal on Resourcesat-2A LISS-IV imagery.

---

## Architectural Principles

1. **Separation of Liveness and Readiness**:
   - `/health` monitors basic ASGI process liveness (always 200 OK once process starts).
   - `/ready` evaluates model availability (returns 200 OK only when weights are loaded; 503 Service Unavailable when checkpoint is missing or invalid).
2. **Deterministic CPU Inference**:
   - Executes with `torch.no_grad()` in `eval()` mode.
   - An `asyncio.Lock` serializes forward passes to ensure thread safety without shared buffer corruption.
3. **Strict Numerical Validation**:
   - Rejects unnormalized integer digital numbers (DN) or out-of-bounds values with HTTP 422.
   - Rejects non-finite values (NaN, Inf, -Inf) with HTTP 422.
   - Requires exactly 3 channels shaped `[3, H, W]` with spatial dimensions $16 \le H, W \le 512$.
4. **Safe Error Handling**:
   - Internal exceptions return HTTP 500 with generic error descriptions.
   - No stack traces, Python tracebacks, or local filesystem paths are leaked in HTTP responses.
5. **Decoupled Checkpoint Lifecycle**:
   - No checkpoints are baked into the container image.
   - Weights are injected dynamically via read-only bind mounts and the `MODEL_CHECKPOINT` environment variable.

---

## API Endpoints

### 1. `GET /health`
Liveness probe to verify that the HTTP server process is running.

- **Status Code**: `200 OK`
- **Response Schema**:
```json
{
  "status": "healthy",
  "service": "liss4-cloud-removal-api",
  "version": "0.1.0",
  "model_loaded": true,
  "ready": true
}
```

### 2. `GET /ready`
Readiness probe to verify whether the model is loaded and ready for inference.

- **Status Code (Ready)**: `200 OK`
```json
{
  "ready": true,
  "status": "ready",
  "model_type": "UNet",
  "device": "cpu",
  "checkpoint": "checkpoints/compact/best_compact_unet.pt",
  "in_channels": 3,
  "out_channels": 3
}
```

- **Status Code (Unavailable)**: `503 Service Unavailable`
```json
{
  "ready": false,
  "status": "unavailable",
  "detail": "Checkpoint file not found: /models/checkpoint.pt"
}
```

### 3. `POST /predict`
Executes cloud removal inference on a single 3D patch.

- **Expected Shape**: `[3, H, W]` ($C=3$, $16 \le H, W \le 512$)
- **Channel Order**: RGB (Sentinel-2 B04, B03, B02)
- **Value Range**: Physical surface reflectance normalized strictly in $[0.0, 1.0]$
- **Request Body**:
```json
{
  "image": [
    [
      [0.21, 0.25, 0.28],
      [0.19, 0.24, 0.27]
    ],
    [
      [0.18, 0.22, 0.25],
      [0.17, 0.21, 0.24]
    ],
    [
      [0.15, 0.19, 0.22],
      [0.14, 0.18, 0.20]
    ]
  ]
}
```
- **Response Schema (`200 OK`)**:
```json
{
  "prediction": [ ... ],
  "output_shape": [3, 2, 3],
  "model_type": "UNet",
  "checkpoint": "checkpoints/compact/best_compact_unet.pt",
  "prediction_min": 0.0812,
  "prediction_max": 0.1432,
  "prediction_mean": 0.1065
}
```

### 4. Interactive OpenAPI Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Local Execution Guide

### 1. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Start Uvicorn Server
Using the default compact U-Net checkpoint:
```bash
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000 --reload
```

Using a custom checkpoint path:
```bash
# On Linux / macOS
export MODEL_CHECKPOINT="checkpoints/compact/best_compact_unet.pt"
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000

# On Windows PowerShell
$env:MODEL_CHECKPOINT="checkpoints/compact/best_compact_unet.pt"
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000
```

---

## Docker Execution Guide

### 1. Build the Docker Image
```bash
docker build -t liss4-cloud-removal-api:latest .
```

### 2. Run Container with Read-Only Checkpoint Mount
```bash
# On Linux / macOS:
docker run -d \
  --name cloud-removal-service \
  -p 8000:8000 \
  -v $(pwd)/checkpoints/compact/best_compact_unet.pt:/models/checkpoint.pt:ro \
  -e MODEL_CHECKPOINT=/models/checkpoint.pt \
  liss4-cloud-removal-api:latest

# On Windows PowerShell:
docker run -d `
  --name cloud-removal-service `
  -p 8000:8000 `
  -v ${PWD}/checkpoints/compact/best_compact_unet.pt:/models/checkpoint.pt:ro `
  -e MODEL_CHECKPOINT=/models/checkpoint.pt `
  liss4-cloud-removal-api:latest
```

### 3. Verify Container Health
```bash
docker inspect --format='{{json .State.Health.Status}}' cloud-removal-service
```

---

## Curl Request Examples

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Readiness Check
```bash
curl -X GET http://localhost:8000/ready
```

### Predict Request (Synthetic 3x32x32 Patch)
```bash
# Generate sample request payload with python and post with curl
python -c "import json, numpy as np; json.dump({'image': np.random.uniform(0.1, 0.4, (3, 32, 32)).tolist()}, open('sample.json', 'w'))"
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

---

## Automated Test Suite

Run the dedicated Phase 10 API test suite:
```bash
pytest tests/test_api.py -v
```

Run the entire project test suite:
```bash
pytest -v
```
All API tests use dependency injection with in-memory U-Net models and do not require external checkpoints, datasets, or internet connectivity.
