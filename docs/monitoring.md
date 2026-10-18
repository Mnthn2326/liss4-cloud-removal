# Prometheus + Grafana Monitoring & Observability (Phase 11)

## Overview

Phase 11 introduces a complete observability stack for the `liss4-cloud-removal` inference service using **Prometheus** for metrics collection and **Grafana** for real-time visualization and alerting.

The service instrumentation monitors:
- HTTP request throughput, error rates, and response status codes.
- API endpoint latency distributions ($p50$, $p95$, $p99$).
- U-Net forward pass execution latency and throughput.
- Model lifecycle readiness and availability.

---

## Observability Architecture

```
                       +-----------------------+
                       |      Client / Demo    |
                       +-----------+-----------+
                                   | HTTP
                                   v
                   +-------------------------------+
                   |    FastAPI Inference Service   |
                   |          (Port 8000)          |
                   |                               |
                   |  - /health (liveness)         |
                   |  - /ready  (readiness)        |
                   |  - /predict (inference)       |
                   |  - /metrics (Prometheus)      |
                   +---------------+---------------+
                                   ^
                                   | Scrape (5s interval)
                   +---------------+---------------+
                   |       Prometheus Server       |
                   |          (Port 9090)          |
                   +---------------+---------------+
                                   ^
                                   | Query (PromQL)
                   +---------------+---------------+
                   |       Grafana Dashboard       |
                   |          (Port 3000)          |
                   |   Datasource: Prometheus      |
                   |   Auto-provisioned Dashboard  |
                   +-------------------------------+
```

---

## Metrics Catalog

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `api_requests_total` | Counter | `method`, `endpoint`, `status_code` | Total HTTP requests processed across all endpoints and status codes (200, 422, 503, 500). |
| `api_request_duration_seconds` | Histogram | `method`, `endpoint` | Total HTTP request duration from arrival to response completion. |
| `model_inference_duration_seconds` | Histogram | `model_type` | Precise forward pass execution time of the U-Net model on CPU. |
| `model_inference_requests_total` | Counter | `model_type`, `status` | Total model inference invocations labeled by `status="success"` or `status="error"`. |
| `model_ready_status` | Gauge | _None_ | Instantaneous model readiness: `1.0` if model is loaded and ready; `0.0` if uninitialized or failed. |

---

## Provisioned Dashboards

Grafana is pre-configured via automated file provisioning (`docker/grafana/provisioning/`):
- **Datasource**: Pre-configured Prometheus datasource pointing to `http://prometheus:9090`.
- **Dashboard**: **"LISS-IV Cloud Removal MLOps Monitoring"** (`uid: liss4-cloud-removal-monitoring`).

### Key Dashboard Panels:
1. **Model Inference Readiness**: Big-number stat indicator (`READY` in green or `UNAVAILABLE` in red).
2. **Total Successful Inferences**: Cumulative count of completed U-Net patch predictions.
3. **Total Inference Errors**: Cumulative count of failed inference runs.
4. **Total HTTP Requests**: Cumulative count of HTTP requests handled.
5. **HTTP Request Rate by Endpoint & Status Code**: Timeseries graph of requests/sec split by endpoint (`/predict`, `/health`, `/ready`) and HTTP status (`200`, `422`, `503`, `500`).
6. **Inference & API Latency Quantiles**: Timeseries graph tracking $p50$ and $p95$ model execution latency alongside full request latency.

---

## Running with Docker Compose

### 1. Prerequisites
- Docker & Docker Compose installed.
- Compact U-Net checkpoint present at `checkpoints/compact/best_compact_unet.pt`.

### 2. Start the Observability Stack
```bash
docker compose up -d --build
```

### 3. Service Access Points
- **FastAPI Inference Service**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics`
- **Prometheus Server**: `http://localhost:9090`
- **Grafana Dashboard**: `http://localhost:3001` (configured via `${GRAFANA_PORT:-3001}`)
  - Default credentials: `admin` / `admin`
  - Pre-loaded dashboard: **LISS-IV Cloud Removal MLOps Monitoring**

### 4. Generate Telemetry & Verify
Run the automated end-to-end traffic script to populate live metrics:
```bash
python scripts/demo_monitoring.py --requests 15
```

### 5. Stop the Stack
```bash
docker compose down
```

---

## Local Verification Without Docker

If Docker is not running, the API service and metrics can be run locally using `uvicorn`:
```bash
# Start local API
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000

# Query Prometheus metrics
curl -s http://localhost:8000/metrics

# Run test suite
pytest tests/test_metrics_api.py -v
```

---

## Scientific Boundaries & Model Limitations

> [!WARNING]
> **Cross-Sensor Inference Boundary**:
> The compact U-Net model was trained on **Sentinel-2 SEN12MS-CR** Level-1C/Level-2A optical imagery ($10\text{ m}$ RGB).
> 
> As established in Phase 7.6 and Phase 7.7 feasibility assessments:
> 1. Resourcesat-2A LISS-IV differs in spatial resolution ($5\text{ m}$ vs $10\text{ m}$), spectral bandpass (Green, Red, NIR vs RGB), and lacks product exo-atmospheric solar irradiance ($ESUN_\lambda$) calibration parameters.
> 2. Operational inference on LISS-IV imagery remains unvalidated.
> 3. Production monitoring metrics (request rates, latency, errors) track operational engineering health, not cross-sensor scientific accuracy.
