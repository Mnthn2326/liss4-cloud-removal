# Phase 2: Bhoonidhi / ISRO Data Feasibility Assessment

## Executive Summary
This document provides a comprehensive technical feasibility assessment for sourcing satellite imagery from the **Bhoonidhi** portal (operated by the National Remote Sensing Centre / ISRO) to support the **LISS4 Cloud Removal** MLOps project.

The assessment investigates data availability, authentication procedures, sensor characteristics, product structures, and the feasibility of obtaining high-quality training pairs for supervised cloud removal.

---

## 1. Official Sources Consulted
The following authoritative sources were reviewed:
- **Bhoonidhi Data Dissemination Portal**: [bhoonidhi.nrsc.gov.in](https://bhoonidhi.nrsc.gov.in/)
- **ISRO / NRSC Data Distribution Policy**: Guidelines pursuant to the Indian Space Policy 2023 governing open data (GSD $\ge 5\text{ m}$) versus restricted/registered dissemination tiers.
- **Resourcesat-2 / Resourcesat-2A Sensor Documentation**: Official payload and calibration specifications for the Linear Imaging Self-Scanning Sensor-4 (LISS-IV).
- **Integrated Multi-mission Ground Segment for Earth Observation Satellites (IMGEOS)**: Processing level specifications and metadata standards for IRS products.
- **Bhoonidhi REST / STAC API Guidelines**: Specifications for programmatic search and catalog access.

---

## 2. Satellite Collections and Sensor Characteristics

### 2.1 Target Sensor: LISS-IV
The primary sensor of interest is **LISS-IV**, flown on **Resourcesat-2** and **Resourcesat-2A**.

| Specification | Details |
| :--- | :--- |
| **Satellites** | Resourcesat-2 (RS2), Resourcesat-2A (RS2A) |
| **Spatial Resolution** | 5.8 meters at nadir |
| **Operating Modes** | - **Multispectral (Mx)**: 3 spectral bands, swath ~23.9 km (nadir) to ~70 km<br>- **Panchromatic (Mono)**: 1 band (Red), swath ~70 km |
| **Spectral Bands (Mx)** | - **Band 2 (Green)**: $0.52 - 0.59\ \mu\text{m}$<br>- **Band 3 (Red)**: $0.62 - 0.68\ \mu\text{m}$<br>- **Band 4 (Near-Infrared / NIR)**: $0.77 - 0.86\ \mu\text{m}$ |
| **Radiometric Resolution** | 10-bit quantization (values 0–1023), distributed as 16-bit unsigned integers (`uint16`) |
| **Revisit Period** | 24 days (nadir repeat cycle); 5 days with across-track steering ($\pm 26^\circ$) |
| **Standard Processing Levels**| - **Level-1**: Radiometrically and systematically corrected<br>- **Level-2 / Orthorectified**: Precision orthorectified with DEM and GCPs (UTM / WGS-84) |

### 2.2 Product File Organization
Standard Bhoonidhi LISS-IV digital deliverables are typically packaged as:
1. **Single-band GeoTIFF files**:
   - `BAND2.tif` (Green)
   - `BAND3.tif` (Red)
   - `BAND4.tif` (NIR)
2. **Metadata manifest**: XML or structured text file (`<product_id>_meta.xml` or `metadata.txt`) detailing solar geometry (elevation, azimuth), acquisition timestamp, look angle, bounding box, calibration coefficients, and scene-level cloud percentage.

---

## 3. Data Access & Authentication Procedures

### 3.1 Portal Registration (Manual GUI)
- **Status**: Free registration is available to academic, research, and general users at [bhoonidhi.nrsc.gov.in](https://bhoonidhi.nrsc.gov.in/).
- **Workflow**:
  1. Register an account and verify email credentials.
  2. Navigate to **Browse & Order**.
  3. Define Area of Interest (AOI) via bounding box coordinates or Shapefile upload.
  4. Select **Mission**: `Resourcesat-2` or `Resourcesat-2A` and **Sensor**: `LISS-IV`.
  5. Filter by acquisition dates and cloud cover threshold.
  6. Add granules to the cart and execute download.

### 3.2 Programmatic REST / STAC API
- **Status**: Bhoonidhi supports token-based REST endpoints complying with STAC conventions, but **programmatic API access is not open by default**.
- **Requirement**: Users must request API credential authorization from NRSC support (`bhoonidhi@nrsc.gov.in`).
- **Endpoint Structure**:
  - `POST /auth/token`: Authenticates registered `userId` and `password` to generate a session Bearer token.
  - `POST /data`: Queries the catalog using STAC query parameters (bounding box, datetime, cloud cover range, sensor identifier).
  - `POST /download`: Initiates asset retrieval for an authorized product ID.

### 3.3 Current Environment Access Blocker
- **Finding**: No user credentials or API tokens are currently configured in the project environment (`.env`).
- **Conclusion**: Automated downloading cannot be performed during this feasibility phase without user credentials.
- **Remediation**: The repository owner must obtain Bhoonidhi portal credentials or manually download a sample scene for local development.

---

## 4. Inspection of Geospatial Data & Tooling Verification

To ensure environment readiness and reproducibility for Phase 3, a dedicated inspection module was implemented in [`src/cloud_removal/inspect.py`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/src/cloud_removal/inspect.py) using `rasterio`.

### Validated Capabilities
- Reading raster spatial metadata (width, height, count, CRS, Affine transform).
- Validating band data types (`uint16`) and dynamic range.
- Extracting NoData masks and calculating per-band pixel statistics (min, max, mean, standard deviation).
- Verified via unit test suite in [`tests/test_inspect.py`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/tests/test_inspect.py), passing successfully with 100% test coverage for simulated LISS-IV 3-band products.

---

## 5. Feasibility of Supervised Cloud Removal Pairs

Supervised training of deep learning cloud-removal models requires input–target pairs:
$$\{X_{\text{cloudy}}, Y_{\text{clear}}\}$$

### 5.1 Challenges with Real Multi-Temporal Pairs
Attempting to construct pairs solely by pairing real cloudy scenes with real clear scenes from different passes encounters critical technical hurdles:
1. **Long Revisit Time (24 Days)**: During a 24-day revisit window (especially in agricultural regions of India), vegetation phenology, soil moisture, crop harvests, and sun elevation change dramatically. A supervised loss function ($L_1$, MSE, SSIM) will heavily penalize the model for genuine landscape changes rather than cloud removal errors.
2. **Across-Track Steering Parallax ($\pm 26^\circ$)**: To achieve shorter revisits (5 days), LISS-IV tilts across-track. Acquisitions with differing look angles over non-flat terrain suffer from severe relief displacement and parallax. Pixel-to-pixel registration cannot be guaranteed at 5.8m resolution.
3. **Persistent Cloud Cover**: During the monsoon season (June–September), consecutive clear scenes over the same footprint are frequently unavailable for months.
4. **Absence of Pixel-Level Cloud Masks**: Bhoonidhi provides only a single scene-level cloud percentage value in the metadata, not a per-pixel cloud/shadow classification mask (such as Sentinel-2's SCL or Landsat's QA_PIXEL).

### 5.2 Recommended Solution: Synthetic Cloud Simulation over Clear LISS-IV Imagery
To establish a rigorous, reproducible supervised pipeline, we recommend a **hybrid methodology**:
1. **Target Imagery**: Source real, high-quality, cloud-free LISS-IV scenes ($\le 2\%$ cloud cover).
2. **Physically-Based Cloud Modeling**: Generate realistic synthetic thin and thick cloud layers using fractal/Perlin noise coupled with the standard atmospheric radiative transfer formulation:
   $$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$
   where $J(x)$ is the clean ground radiance, $t(x) \in [0, 1]$ represents cloud transmittance, and $A$ is atmospheric/cloud radiance.
3. **Advantages**:
   - **Zero registration error**: Input and ground truth are inherently aligned at sub-pixel precision.
   - **No phenological drift**: Surface state is completely identical.
   - **Automated ground-truth masks**: Exact transmission maps $t(x)$ serve as ground-truth cloud density masks.
   - **Controllable difficulty**: Enables systematic evaluation across varying cloud opacities.

---

## 6. Minimal Dataset Architecture & Plan

### 6.1 Raw Data Directory Layout
```text
data/
├── raw/
│   ├── manifest.csv               # Ground-truth catalog manifest
│   ├── manifest.example.csv       # Schema definition and examples
│   ├── RS2_L4_MX_098_052_.../    # Scene directory containing single-band TIFs + XML
│   │   ├── BAND2.tif              # Green band (5.8m)
│   │   ├── BAND3.tif              # Red band (5.8m)
│   │   ├── BAND4.tif              # NIR band (5.8m)
│   │   └── metadata.xml           # Product metadata
│   └── quarantine/                # Quarantined / rejected scenes
├── interim/                       # 3-band stacked, co-registered rasters
└── processed/                     # Normalized, tiled training patches (e.g. 256x256)
```

### 6.2 Data Manifest Schema
Products are cataloged in [`data/raw/manifest.csv`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/data/raw/manifest.csv) with the following core schema:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `product_id` | String | Unique ISRO granule identifier |
| `satellite` | String | `Resourcesat-2` or `Resourcesat-2A` |
| `sensor` | String | `LISS-IV` |
| `mode` | String | `Mx` (Multispectral) or `Mono` (Panchromatic) |
| `processing_level` | String | `Level-1`, `Level-2`, `Orthorectified` |
| `acquisition_date` | String | ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) |
| `path` | Integer | Satellite orbital path number |
| `row` | Integer | Satellite orbital row number |
| `sub_scene` | String | Sub-scene designation (`A`, `B`, `C`, `D`) |
| `cloud_cover_percent` | Float | Metadata-reported scene cloud cover percentage |
| `sun_elevation_deg` | Float | Solar elevation angle |
| `incidence_angle_deg` | Float | Sensor look angle ($\le 15^\circ$ preferred) |
| `crs` | String | Coordinate Reference System (e.g., `EPSG:32643`) |
| `paired_reference_id` | String | Product ID of clear pair, or `SYNTHETIC` |
| `status` | String | `ACCEPTED`, `REJECTED`, or `QUARANTINED` |
| `rejection_reason` | String | Explanatory code if not accepted |
| `local_dir` | String | Relative path to local unpacked directory |

### 6.3 Ingestion Rules
- **Deduplication**: Keyed on unique `product_id` and SHA-256 archive hash. If multiple processing levels exist for the same orbit/footprint, the higher level (`Orthorectified`) is retained.
- **Acceptance Criteria**:
  - Contains all 3 bands (`BAND2`, `BAND3`, `BAND4`) uncorrupted.
  - Spatial resolution is verified at ~5.8m with matching dimensions.
  - Off-nadir tilt $\le 15^\circ$ to minimize geometric distortion.
- **Quarantine Criteria**:
  - Missing or malformed metadata XML.
  - Scanline dropout or corrupted pixel values.
  - Extreme look angle ($> 20^\circ$) requiring manual geometric inspection.
- **Rejection Criteria**:
  - Panchromatic-only products (`Mono`).
  - Total cloud occlusion ($> 80\%$) where no ground features are detectable.

---

## 7. Version Control & DVC Compatibility
The project [`.gitignore`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/.gitignore) has been updated and verified to ensure:
- Large binary rasters (`*.tif`, `*.zip`, `*.tar.gz`) are strictly excluded from Git.
- DVC pointer files (`*.dvc`), directory keep files (`.gitkeep`), pipeline files (`dvc.yaml`, `dvc.lock`), and metadata manifests (`manifest.*`) are tracked by Git.
- Local DVC caches (`.dvc/cache/`, `.dvc/tmp/`) and secrets (`.env`) remain excluded.

---

## 8. Known Limitations & Technical Constraints
1. **Spectral Coverage**: LISS-IV lacks a Blue band (0.45–0.51 µm) and Cirrus/SWIR bands (1.37 µm, 1.6 µm, 2.2 µm). This limits the application of classical atmospheric removal algorithms (such as the Haze Optimized Transformation / HOT index) that rely heavily on Blue–Red spectral ratios.
2. **Access Automation**: Direct anonymous API access is prohibited. Initial ingestion must rely on manual ordering via Bhoonidhi GUI or authorized API keys.
3. **No Ground Truth Masks**: True cloud-free ground truth masks must either be synthetically generated or manually delineated.

---

## 9. Recommended Next Phase (Phase 3)
1. **Acquire Sample Scene**: The project owner should manually download 1–2 cloud-free LISS-IV scenes from the Bhoonidhi portal and place them in `data/raw/`.
2. **Implement Preprocessing Pipeline**:
   - Band stacking module: Combine separate `BAND2`, `BAND3`, `BAND4` GeoTIFFs into unified 3-band RGB-NIR rasters.
   - Normalization & Tiling: Tile scenes into non-overlapping patches (e.g., $256 \times 256$) suitable for neural network input.
   - Synthetic Cloud Generator: Implement the physical transmittance degradation pipeline to generate paired training data.
3. **Initialize DVC**: Track `data/raw` and `data/processed` using DVC to establish the first versioned data baseline.
