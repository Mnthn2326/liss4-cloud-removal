# Phase 11 — Prometheus + Grafana Monitoring and Observability Report

---

## 1. Executive Summary

### Status: **`IMPLEMENTED_AND_VERIFIED`**

Phase 11 implements a production-grade observability stack for the `liss4-cloud-removal` inference service using **Prometheus**, **Grafana**, and **Docker Compose**. The service is instrumented to collect real-time telemetry on API throughput, HTTP status codes, request latency distributions, model inference duration, and model readiness state.

### Key Milestones Achieved:
1. **Application Metrics Instrumentation (`src/cloud_removal/api.py`)**:
   - `api_requests_total`: Tracks total HTTP requests partitioned by HTTP method, endpoint path, and status code (200, 422, 503, 500).
   - `api_request_duration_seconds`: Histogram measuring full request processing latency with fine-grained buckets.
   - `model_inference_duration_seconds`: Histogram measuring the precise forward-pass execution duration of the U-Net on CPU.
   - `model_inference_requests_total`: Counter tracking successful and failed model inferences (`status="success"` or `status="error"`).
   - `model_ready_status`: Gauge indicating whether the model is loaded and ready for inference (`1.0` = ready, `0.0` = unavailable).
   - `/metrics` endpoint exposing standard Prometheus exposition text format.
2. **Prometheus Configuration (`docker/prometheus/prometheus.yml`)**:
   - Configured with a 5-second scrape interval targeting `api:8000/metrics`.
   - Verified active scrape target status as `UP` in Prometheus TSDB.
3. **Automated Grafana Provisioning (`docker/grafana/provisioning/`)**:
   - Automated datasource provisioning connecting to Prometheus at `http://prometheus:9090`.
   - Pre-loaded dashboard: **"LISS-IV Cloud Removal MLOps Monitoring"** (`uid: liss4-cloud-removal-monitoring`).
   - Visualizes readiness status, throughput, status code distribution, inference latency percentiles ($p50, p95$), and error rates.
4. **Multi-Container Stack (`docker-compose.yml`)**:
   - Orchestrates `api` (`:8000`), `prometheus` (`:9090`), and `grafana` (`:3001`).
   - Host port 3001 mapped to container 3000 to avoid collision with existing host Grafana instances.
   - Verified end-to-end container communication and healthchecks.
5. **End-to-End Traffic Demo Script (`scripts/demo_monitoring.py`)**:
   - Exercises liveness probes, readiness checks, valid multi-scale patch predictions, and client-error edge cases (HTTP 422).
6. **Automated Test Suite Expansion (`tests/test_metrics_api.py`)**:
   - 6 new automated tests verifying metrics emission, counters, gauges, histograms, and registry isolation.
   - Full test suite now stands at **87 passing tests** (56 existing + 25 Phase 10 + 6 Phase 11).

---

## 2. Observability Architecture & Stack Status

| Service | Container Name | Host Port | Internal Port | Health / Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **API Service** | `liss4-api` | `8000` | `8000` | `Up (healthy)` | FastAPI inference service; checkpoint bind-mounted read-only. |
| **Prometheus** | `liss4-prometheus` | `9090` | `9090` | `Up` | Scrapes `api:8000/metrics` every 5s; target state: `UP`. |
| **Grafana** | `liss4-grafana` | `3001` | `3000` | `Up` | Auto-provisioned datasource & dashboard; credentials `admin` / `admin`. |

---

## 3. Telemetry Verification & Scrape Results

Execution of `python scripts/demo_monitoring.py --requests 10` produced verified metrics in Prometheus:

- **Liveness & Readiness**:
  - `api_requests_total{endpoint="/health",method="GET",status_code="200"}`: `13.0`
  - `api_requests_total{endpoint="/ready",method="GET",status_code="200"}`: `2.0`
  - `model_ready_status`: `1.0`
- **Inference Execution**:
  - `api_requests_total{endpoint="/predict",method="POST",status_code="200"}`: `10.0`
  - `model_inference_requests_total{model_type="UNet",status="success"}`: `10.0`
  - Average inference latency: `49.58 ms` ($p95$: `107.86 ms`)
- **Error Tracking**:
  - `api_requests_total{endpoint="/predict",method="POST",status_code="422"}`: `3.0`
  - Verified rejections for invalid channel counts, unscaled digital numbers, and undersized spatial dimensions.

---

## 4. Test Suite Execution Summary

```
87 passed, 16 warnings in 28.19s
```

- **Baseline Tests (Phases 1–9)**: 56 passed.
- **Phase 10 API Tests (`tests/test_api.py`)**: 25 passed.
- **Phase 11 Metrics Tests (`tests/test_metrics_api.py`)**: 6 passed:
  1. `test_metrics_endpoint_accessible` — Valid Prometheus text format.
  2. `test_api_requests_counter_increments` — Status and method label tracking.
  3. `test_model_ready_gauge_behavior` — Gauge transitions between 1.0 (ready) and 0.0 (unloaded).
  4. `test_inference_metrics_recorded` — Inference counts and duration observations.
  5. `test_invalid_request_metrics_recorded` — Client 422 errors tracked in requests counter.
  6. `test_inference_error_metrics_recorded` — Exception handling updates error counters and HTTP 500 status.

---

## 5. Scientific Boundary Notice

> [!CAUTION]
> **Scientific Integrity & Model Limitations**:
> The compact U-Net model was trained exclusively on **Sentinel-2 SEN12MS-CR** Level-1C/Level-2A optical imagery ($10\text{ m}$ RGB).
> 
> As established in Phase 7.6 and Phase 7.7 feasibility reports:
> 1. Resourcesat-2A LISS-IV differs in spatial resolution ($5\text{ m}$ vs $10\text{ m}$), spectral bandpass (Green, Red, NIR vs RGB), and lacks product exo-atmospheric solar irradiance ($ESUN_\lambda$) calibration parameters.
> 2. Operational inference on LISS-IV imagery remains unvalidated.
> 3. Production monitoring metrics (request rates, latency, errors) track operational engineering health, not cross-sensor scientific accuracy.
