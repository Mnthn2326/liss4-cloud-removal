# Phase 10 — Docker + FastAPI Inference Service Report

---

## 1. Executive Summary

### Status: **`IMPLEMENTED_AND_LOCALLY_VERIFIED`**

Phase 10 establishes a containerized, production-oriented **FastAPI** inference service for the `liss4-cloud-removal` project. The service provides deterministic CPU inference for the trained compact U-Net baseline, strictly honoring existing model contracts, input normalization conventions, and spatial dimensions.

### Key Milestones Achieved:
1. **FastAPI Application Module (`src/cloud_removal/api.py`)**:
   - Implemented `/health` (process liveness probe; returns 200 OK regardless of model state).
   - Implemented `/ready` (service readiness probe; returns 200 OK when model is loaded, 503 Service Unavailable when checkpoint is missing/uninitialized).
   - Implemented `/predict` (deterministic CPU inference on 3D patches `[3, H, W]`).
   - Integrated OpenAPI Swagger documentation at `/docs` and schema at `/openapi.json`.
2. **Decoupled Model Lifecycle (`ModelManager`)**:
   - Manages model state, evaluation mode, and safe PyTorch checkpoint loading.
   - Thread-safe inference using `asyncio.Lock` to serialize execution and eliminate race conditions.
   - Checkpoint path injected dynamically via `MODEL_CHECKPOINT` environment variable with a safe repository fallback.
   - In-memory model injection for fast, deterministic unit testing without disk dependencies.
3. **Strict Validation & Security**:
   - Explicitly rejects unnormalized digital numbers (> 1.0) and negative values (< 0.0) with HTTP 422.
   - Rejects non-finite values (NaN, Inf, -Inf) with HTTP 422.
   - Enforces 3 RGB channels (Sentinel-2 B04, B03, B02) and spatial bounds $16 \le H, W \le 512$ with ragged-row checks.
   - Error masking: internal inference exceptions return generic HTTP 500 without leaking stack traces or host filesystem paths.
4. **Containerization Artifacts (`Dockerfile` & `.dockerignore`)**:
   - Multi-layer `Dockerfile` using `python:3.11-slim`.
   - Pre-installs CPU-only PyTorch to minimize image footprint.
   - Runs as non-root user (`appuser`, UID 1000).
   - Native Python `urllib` container health check against `/health`.
   - `.dockerignore` rigorously excludes checkpoints, datasets, git, virtual environments, MLflow runs, and secrets.
5. **Comprehensive Automated Tests (`tests/test_api.py`)**:
   - 25 dedicated unit and integration tests covering all 12 mandatory verification cases.
   - Total test suite now stands at **81 passing tests** (56 existing + 25 new).

---

## 2. API Contract & Endpoint Specifications

| Endpoint | Method | Status Codes | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | `200 OK` | Liveness probe; returns process status and whether model is loaded. |
| `/ready` | `GET` | `200 OK`, `503 Unavailable` | Readiness probe; returns 200 when model is loaded and ready, 503 when missing. |
| `/predict` | `POST` | `200 OK`, `422 Unprocessable`, `503 Unavailable`, `500 Error` | Executes cloud removal inference on single patch `[3, H, W]`. |
| `/docs` | `GET` | `200 OK` | Interactive Swagger UI documentation. |
| `/openapi.json` | `GET` | `200 OK` | Machine-readable OpenAPI 3.1.0 schema specification. |

### Prediction Input/Output Specification
- **Input Dimensions**: `[3, H, W]` ($C=3$ channels; $16 \le H, W \le 512$).
- **Channel Semantics**: Sentinel-2 RGB (Band 4 = Red, Band 3 = Green, Band 2 = Blue).
- **Physical Range**: Surface reflectance normalized strictly in $[0.0, 1.0]$.
- **Output Format**: Predicted cloud-free surface reflectance array `[3, H, W]` bounded in $[0.0, 1.0]$, accompanied by summary statistics (`prediction_min`, `prediction_max`, `prediction_mean`).

---

## 3. Test Suite Verification

Every test in `tests/test_api.py` operates via dependency injection using in-memory model instances or isolated configurations, ensuring zero reliance on network access or local checkpoint presence.

### Test Breakdown:
- **Health & Readiness Lifecycle**: 4 tests (uninitialized, initialized, unload transition, missing checkpoint reporting).
- **Inference Schema & Numerical Validity**: 3 tests (valid request/response, multi-resolution $32 \times 32$ and $64 \times 64$, statistics).
- **Input Validation & Guardrails**: 11 tests (2D, 4D, ragged rows, 1-channel, 4-channels, NaN, Inf, negative reflectance, unscaled DN $> 1.0$, spatial size $< 16$, spatial size $> 512$, empty body).
- **Error Handling & Security**: 2 tests (unavailable model 503, exception masking without traceback/path leakage 500).
- **Model Contract Parity**: 1 test (direct `predict_patch` vs `ModelManager.predict` parity under async lock).
- **API Documentation**: 1 test (`/docs` and `/openapi.json` schema validation).
- **Full Split / Pipeline Parity**: 3 tests (preserved from earlier phases).

```
====================== 81 passed, 16 warnings in 28.73s =======================
```

---

## 4. Docker Build, Container Smoke Test & Verification

### Status Summary Table

| Verification Step | Target / Artifact | Result | Details |
| :--- | :--- | :--- | :--- |
| **Docker Availability** | Docker Desktop 4.84.0 / Engine 29.6.2 | **PASS** | Daemon active on Windows host via WSL2 engine. |
| **Docker Image Build** | `liss4-cloud-removal-api:phase10` | **PASS** | Built cleanly from `python:3.11-slim` with `libexpat1` system dependency (image size: 598 MB content size). |
| **Container Startup** | `liss4-cloud-removal-api-phase10-test` | **PASS** | Started detached with read-only bind mount of `checkpoints/compact/best_compact_unet.pt`. |
| **Container Healthcheck** | `GET http://localhost:8000/health` | **PASS** | Built-in Docker healthcheck passed; container marked `healthy`. Process reports 200 OK. |
| **Readiness Endpoint** | `GET http://localhost:8000/ready` | **PASS** | 200 OK; confirmed `status="ready"`, `device="cpu"`, `checkpoint="/models/checkpoint.pt"`. |
| **Real Checkpoint Inference** | `POST http://localhost:8000/predict` | **PASS** | 200 OK on synthetic $3 \times 32 \times 32$ patch; returned shape `[3, 32, 32]`, all finite reflectance in $[0.0, 1.0]$. |
| **Input Validation Guardrails** | `POST http://localhost:8000/predict` | **PASS** | Verified 422 rejections on missing fields, 1ch, 4ch, $8 \times 8$ small dims, $>1.0$ reflectance, negative reflectance, and NaN. |
| **Regression Test Suite** | `pytest -q` | **PASS** | 81 of 81 tests passed in 31.70s. |
| **Container Cleanup** | `docker stop` & `docker rm` | **PASS** | Temporary test container stopped and removed cleanly. |

### Docker Build & Runtime Details
- **Build Command**: `docker build -t liss4-cloud-removal-api:phase10 .`
- **Image Identifier**: `liss4-cloud-removal-api:phase10` (digest `sha256:ae1ec4554b5b...`)
- **System Dependencies**: Added `libexpat1` to support Debian bookworm dynamic linking in `rasterio`/GDAL bindings.
- **Runtime Mount**: Read-only bind mount via `--mount "type=bind,source=$checkpoint,target=/models/checkpoint.pt,readonly"` and environment variable `MODEL_CHECKPOINT=/models/checkpoint.pt`.
- **Image Hygiene**: Confirmed `/app` in container contains only package source code and metadata; no datasets, checkpoints, git trees, or secrets are bundled.

---

## 5. Scientific Boundary Notice

> [!CAUTION]
> **Scientific Integrity & Operational Boundary**:
> The compact U-Net baseline was trained and evaluated exclusively on Sentinel-2 SEN12MS-CR Level-1C/Level-2A optical imagery.
> 
> As concluded in the Phase 7.6 and Phase 7.7 feasibility reports:
> 1. Resourcesat-2A LISS-IV imagery differs in spatial resolution (5 m vs 10 m), spectral bandpass (Green, Red, NIR vs RGB), and radiometric calibration parameters.
> 2. Exposing this inference service does **not** validate cross-sensor inference on LISS-IV imagery.
> 3. This service must **not** be presented as a validated operational LISS-IV cloud-removal system.
