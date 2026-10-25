# 5-Minute Defense Demonstration Guide

## Overview

This guide provides a structured, minute-by-minute script for presenting the **Remote-Sensing Cloud Reconstruction** prototype during academic defense, technical review panels, or live demonstration sessions.

The presentation demonstrates an end-to-end machine learning engineering workflow: from validated benchmark dataset curation (RICE2), through reproducible deep learning baseline training (Compact U-Net), to production-grade deployment with real-time UI comparison, REST API serving, and Prometheus monitoring.

---

## Pre-Demo Setup Checklist (2 Minutes Before Demo)

Ensure all dependencies and services are active in PowerShell:

```powershell
# 1. Activate project virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Launch the inference and UI service
python -m cloud_removal.demo --ui --port 8000
```

Verify service availability:
- **Web UI**: Open [http://localhost:8000/ui](http://localhost:8000/ui) in your browser.
- **API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Metrics**: Open [http://localhost:8000/metrics](http://localhost:8000/metrics).

---

## 5-Minute Presentation Script

### Minute 1: Problem Statement, Scope & Benchmark Strategy

> *"Good morning committee members. Satellite optical sensors operating in the visible spectrum frequently suffer from cloud contamination, obscuring valuable surface data. In this project, we built a production-grade machine learning system for optical cloud reconstruction.*
>
> *Earlier exploratory iterations evaluated ISRO Resourcesat-2A LISS-IV imagery. However, through rigorous radiometric inspection, we identified that LISS-IV lacks a blue spectral band, lacks multi-temporal clear-sky pairs, and has uncalibrated digital numbers. To establish an officially verifiable scientific baseline, our Version 1.0 production pipeline is trained and evaluated on the internationally validated **RICE2 benchmark dataset** (Landsat-style $512 \times 512$ RGB imagery with paired clear-sky ground truth)."*

**Visual**: Point to the UI header showing `# Remote-Sensing Cloud Reconstruction` and the dataset notice.

---

### Minute 2: Interactive Web UI & Split Slider Demonstration

1. In the Web UI control sidebar:
   - Ensure **Active Model** is selected as `Compact U-Net (RICE2 Baseline)`.
   - Under **Benchmark Presets**, select a high-contrast sample (e.g., sample `600` or sample `700`).
2. Click **⚡ Reconstruct Image**.
3. Showcase the **Interactive Split Slider**:
   - Drag the divider handle across the satellite image.
   - Show how the cloudy layer is peeled back to reveal the reconstructed land cover beneath.
4. Switch the View Mode toggle to **⊞ Side-by-Side**:
   - Point out the 3 synchronized panels: **Cloudy Input**, **Reconstructed Image**, and **Ground Truth Reference**.

> *"Notice how the model eliminates dense cloud cover while preserving unchanged non-cloud terrain textures, roads, and field boundaries without altering the radiometric tone of clear regions."*

---

### Minute 3: Quantitative Metrics & Scientific Integrity

1. Direct the audience's attention to the **Reconstruction Quality Metrics** card below the viewer:
   - **Full MAE**: `0.0491` (-24.4% error reduction vs cloudy baseline).
   - **PSNR**: `25.12 dB` (+1.79 dB improvement).
   - **SSIM**: `0.8510` (+0.0481 gain).
   - **Cloud-Region MAE**: `0.0604` vs `0.1294` baseline (**53.3% error reduction** in cloud-covered zones).
2. Point out the **Interactive Crosshair Inspector**:
   - Hover over pixels in the image to show synchronized live $[R, G, B]$ reflectance values.
3. Highlight the **Scientific Disclaimer Banner**:
   - Point out the notice: *"Notice: Reconstructed imagery is an estimated surface approximation based on surrounding spatial context. Optical photons cannot penetrate opaque clouds."*

> *"We treat evaluation with strict scientific honesty: we report regional error metrics specifically isolated to cloud pixels, and our system explicitly labels outputs as estimates rather than claiming impossible optical penetration."*

---

### Minute 4: REST API Serving & Input Validation

Open a terminal or Postman/curl window and execute a readiness probe:

```powershell
curl -s http://localhost:8000/ready
```

Output:
```json
{
  "ready": true,
  "status": "ready",
  "model_type": "UNet",
  "model_version": "1.0.0",
  "dataset_version": "RICE2-v1",
  "device": "cpu",
  "in_channels": 3,
  "out_channels": 3
}
```

Now demonstrate defensive rejection of malformed or single-band inputs:
```powershell
# Attempting to upload a single-channel or non-RGB input returns HTTP 422
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image": [[[0.5, 0.5], [0.5, 0.5]]]}'
```

> *"The API strictly validates input dimensions, channels, and physical reflectance bounds $[0.0, 1.0]$, rejecting invalid or corrupted payloads before execution."*

---

### Minute 5: MLOps Lineage, Docker & Prometheus Observability

1. **Prometheus Metrics**:
   Open [http://localhost:8000/metrics](http://localhost:8000/metrics) in browser and highlight:
   - `model_ready_status`: `1.0` (readiness gauge).
   - `model_version_info`: tracks `model_version="1.0.0"` and `dataset_version="RICE2-v1"`.
   - `input_image_height_pixels` and `input_pixel_mean_reflectance`: real-time input drift monitoring.
   - `rejected_requests_total`: categorized failure telemetry.
2. **Reproducible Pipeline**:
   Show `dvc.yaml`:
   - Point out the 4 linked pipeline stages: `manifest` $\rightarrow$ `train` $\rightarrow$ `evaluate` $\rightarrow$ `report`.
3. **Containerized Deployment**:
   Show `Dockerfile` and `docker-compose.yml`:
   - Non-root user execution, health checks, CPU PyTorch wheels, and Prometheus/Grafana service mesh.

> *"In summary, the system satisfies production-grade standards: end-to-end data versioning, defensive API design, live Prometheus observability, and honest scientific benchmarking."*
