# Interactive Web UI & Visual Workspace

## Overview

The `liss4-cloud-removal` application features an embedded, responsive web interface served directly by the existing FastAPI application at `GET /` and `GET /ui`.

The interface provides an interactive evaluation workspace for optical satellite cloud removal without requiring separate frontend services (such as Streamlit or Gradio). It allows researchers, evaluators, and project guides to test the compact U-Net baseline interactively, compare before/after results using an interactive split-slider or side-by-side view, inspect quantitative metrics conditionally, and export predictions as full-precision GeoTIFFs or high-fidelity PNGs.

---

## Scientific Boundaries & Domain Limitations

> [!WARNING]
> **Cross-Sensor Inference Boundary**:
> The underlying compact U-Net model was trained exclusively on **Sentinel-2 SEN12MS-CR Level-1C/Level-2A optical surface reflectance at 10 m resolution (B04-Red, B03-Green, B02-Blue)**.
> 
> As established in project feasibility assessments:
> 1. **LISS-IV Sensor**: ISRO Resourcesat-2A LISS-IV acquires Green (Band 2), Red (Band 3), and NIR (Band 4) at 5 m spatial resolution. It lacks an optical Blue band.
> 2. **Cross-Sensor Validity**: Inference on LISS-IV imagery remains scientifically unvalidated. Isolated single-band rasters (such as `BAND2.tif`) are explicitly rejected by the ingestion pipeline.
> 3. **Synthetic Presets**: Simulated examples generated via Perlin/Gaussian atmospheric blending are clearly flagged with `[Synthetic Cloud Sim]` badges.

---

## Architectural Design

The UI architecture follows 6 key engineering principles:

1. **Zero External Application Server**:
   - Built with self-contained, responsive HTML5, modern CSS, and vanilla JavaScript.
   - Served directly via FastAPI's `HTMLResponse` on the existing Uvicorn server (`http://localhost:8000/`).
   - Completely offline-compatible with zero CDN dependencies.
2. **Reuse of Inference & Preprocessing Pipeline**:
   - Calls the existing `ModelManager` and async locks from `api.py`.
   - Avoids duplicating model instantiation, forward pass logic, or device management.
3. **Format Support & Radiometric Normalization**:
   - **13-band Sentinel-2 GeoTIFFs**: Automatically extracts bands 4, 3, 2 (RGB) and scales digital numbers ($DN / 10000.0$) to $[0.0, 1.0]$.
   - **3-band RGB GeoTIFFs**: Automatically detects integer DN or float surface reflectance and scales accordingly while preserving geospatial CRS and Affine Transform metadata.
   - **Standard Optical RGB (PNG/JPEG)**: Decodes 3-channel optical images normalized to $[0.0, 1.0]$. Flags geospatial metadata as unreferenced.
   - **Rejection of Incompatible Inputs**: Single-band rasters (e.g., individual LISS-IV bands), non-3/13 band rasters, non-finite pixel values, and dimensions outside $[16, 512]$ are rejected with HTTP 422.
4. **Shared Display Stretch**:
   - Displays optical surface reflectance using a shared 2%–98% percentile contrast stretch (derived from ground truth or prediction) so natural surface features are clearly visible to human observers.
5. **Strictly Conditional Quantitative Metrics**:
   - MAE and PSNR are computed **only** when a valid cloud-free ground-truth reference scene is available (via dataset preset or paired reference upload).
   - If no ground truth is provided, quantitative metric cards are omitted and replaced by an informative message.
6. **Preservation of Geospatial Metadata**:
   - When inputs are GeoTIFFs, outputs can be exported as full 3-band GeoTIFFs with the exact original CRS and Affine Transform.

---

## UI Features & User Guide

### 1. Input Modes

- **Tab 1: Benchmark Presets**:
  - Select from pre-indexed real validation samples (`SEN12MS-CR` ROI 102 Winter) or synthetic cloud simulation scenes (`40% Density`, `25% Cirrus Haze`).
  - Includes paired ground-truth clear references for immediate quantitative validation.
- **Tab 2: Image Upload**:
  - **Cloudy Image (Required)**: Drag and drop or browse for a `.tif`, `.tiff`, `.png`, or `.jpg` file.
  - **Clear Reference (Optional)**: Provide a ground-truth image to unlock real-time MAE and PSNR calculation.

### 2. Workspace View Modes

- **Interactive Split Slider (`⚡ Split Slider`)**:
  - An interactive split-view canvas with a draggable center divider.
  - Dragging the handle or moving the range slider reveals the cloud-removed prediction overlaid directly onto the cloudy input scene.
  - Works with mouse dragging, keyboard slider, and touch events on tablets and mobile devices.
- **Side-by-Side Panels (`⊞ Side-by-Side`)**:
  - Displays Cloudy Input, U-Net Reconstruction, and Ground Truth Clear (when available) in aligned comparison panels.

### 3. Quantitative Evaluation Card

When ground-truth is available, the interface displays:
- **Cloudy Baseline MAE & PSNR**: Initial distortion of the cloudy scene.
- **U-Net Prediction MAE & PSNR**: Metric after neural reconstruction.
- **Improvement ($\Delta\text{MAE}$, $\Delta\text{PSNR}$)**: Quantified delta showing reconstruction gain.

When ground-truth is absent:
- Displays `Ground Truth: None` and explains that metric calculation requires reference data.

### 4. Export & Download

- **Download GeoTIFF (`.tif`)**: Enabled when the input is a GeoTIFF; exports a 3-band Float32 raster preserving CRS, Affine transform, and surface reflectance values.
- **Download PNG (`.png`)**: Exports an 8-bit RGB image rendered with the shared contrast stretch.

---

## API Endpoints for the UI

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` or `/ui` | `GET` | Serves the interactive web interface HTML. |
| `/api/ui/presets` | `GET` | Lists available benchmark and synthetic presets. |
| `/api/ui/predict` | `POST` | Executes inference on uploaded image files or preset IDs (`multipart/form-data`). |
| `/api/ui/download/{session_id}` | `GET` | Downloads prediction as `geotiff` or `png`. |

---

## Launching the Interface

### Local Execution (PowerShell)
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Start the service
uvicorn cloud_removal.api:app --host 127.0.0.1 --port 8000
```

### Linux / macOS
```bash
source .venv/bin/activate
uvicorn cloud_removal.api:app --host 127.0.0.1 --port 8000
```

### Docker Execution
```bash
docker run -d \
  --name cloud-removal-ui \
  -p 8000:8000 \
  -v ${PWD}/checkpoints/compact/best_compact_unet.pt:/models/checkpoint.pt:ro \
  -e MODEL_CHECKPOINT=/models/checkpoint.pt \
  liss4-cloud-removal-api:latest
```

Open `http://localhost:8000/` in your web browser.
