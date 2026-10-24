# Geospatial Preprocessing Pipeline: Band Stacking & Validity Masking

## 1. Overview
This document specifies the Phase 4 geospatial preprocessing pipeline for **LISS4 Cloud Removal**.

The preprocessing stage validates raw single-band LISS-IV GeoTIFFs, combines them into a unified 3-band raster, extracts an explicit spatial validity mask for the satellite scene collar, and records complete lineage metadata in a processing manifest. The entire stage is codified and tracked as a reproducible stage (`preprocess`) in `dvc.yaml`.

---

## 2. Input Product & Band Ordering

- **Product Identifier**: `2643035131`
- **Product Directory**: `data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/`
- **Satellite / Sensor**: Resourcesat-2A (`IRS-R2A`) / `L4FX`
- **Acquisition Date**: `2026-06-03`
- **Spatial CRS**: `EPSG:32643` (UTM Zone 43N / WGS 84)
- **Spatial Resolution**: $5.000000\text{ m} \times 5.000000\text{ m}$
- **Dimensions**: Width $18,066$ pixels, Height $16,223$ pixels

### Band Stacking Assignment
The 3 single-band GeoTIFFs are combined preserving exact source-channel order:

| Output Band | Source File | Source Channel Identifier | Band Description | Spectral Assignment Status |
| :---: | :--- | :--- | :--- | :--- |
| **Band 1** | `BAND2.tif` | Band 2 (`BAND2`) | `BAND2` | Unverified from metadata |
| **Band 2** | `BAND3.tif` | Band 3 (`BAND3`) | `BAND3` | Unverified from metadata |
| **Band 3** | `BAND4.tif` | Band 4 (`BAND4`) | `BAND4` | Unverified from metadata |

> **Note on Spectral Assignments:** In accordance with rigorous engineering practice, spectral wavelength labels (such as Green, Red, or NIR) are explicitly marked as unverified from the product's self-contained metadata files, which identify the channels strictly as numeric bands `2`, `3`, and `4`. No spectral assignments are assumed or guessed.


---

## 3. Output Stack Construction

- **Output File**: `data/interim/stacked_product/stacked.tif`
- **Raster Dimensions**: $18,066 \times 16,223 \times 3$ bands
- **Data Type**: `uint16` (preserving authentic 10-bit dynamic range $0 - 1023$)
- **CRS & Transform**: Identical to source (`EPSG:32643`, origin $X=279458.943$, $Y=2060222.5$, pixel size $5.0\text{ m}$)
- **Tiling & Layout**: Internally tiled (`tiled=True`, `blockxsize=512, blockysize=512`)
- **Compression**: Lossless Deflate compression (`compress=deflate`, `predictor=2`)
- **File Size**: $540,449,943\text{ bytes}$ (~$515.4\text{ MB}$, a $69.3\%$ reduction from the $1.76\text{ GB}$ uncompressed source files)
- **Memory Footprint**: Processed in block windows ($512 \times 512$ pixels), maintaining RAM consumption below $35\text{ MB}$ during execution.

---

## 4. Spatial Validity Mask

- **Output File**: `data/interim/stacked_product/valid_mask.tif`
- **Raster Dimensions**: $18,066 \times 16,223 \times 1$ band
- **Data Type**: `uint8` (`1` = valid terrain pixel, `0` = invalid collar pixel)
- **Tiling & Compression**: Tiled ($512 \times 512$), Deflate compressed
- **File Size**: $448,315\text{ bytes}$ (~$437.8\text{ KB}$)

### Validity Rule & Collar Definition
- **Exact Implemented Rule**: **Valid if at least one of the three bands is nonzero.**
  - A pixel is marked **valid** (`1`) if at least one band has a non-zero digital number: $B_2(x, y) \ne 0 \lor B_3(x, y) \ne 0 \lor B_4(x, y) \ne 0$ (`(BAND2 != 0) | (BAND3 != 0) | (BAND4 != 0)`).
  - A pixel is marked **collar / invalid** (`0`) if and only if all three bands are zero: $B_2(x, y) = 0 \land B_3(x, y) = 0 \land B_4(x, y) = 0$ (`BAND2 == 0 and BAND3 == 0 and BAND4 == 0`).

$$\text{Mask}(x, y) = \begin{cases} 1 & \text{if } B_2(x, y) \ne 0 \lor B_3(x, y) \ne 0 \lor B_4(x, y) \ne 0 \\ 0 & \text{if } B_2(x, y) = 0 \land B_3(x, y) = 0 \land B_4(x, y) = 0 \end{cases}$$

- **Pixel Counts on Scene**:
  - Valid Pixels ($1$): $230,424,446$ ($78.62\%$)
  - Collar Pixels ($0$): $62,660,272$ ($21.38\%$)
  - Total Pixels: $293,084,718$ ($100.00\%$)

### Explicit Assumptions & Limitations
1. In this standard georeferenced product, the satellite path orientation ($193.52^\circ$) leaves corner triangles outside the imaged swath. NRSC fills this unrecorded region with zeros ($DN = 0$).
2. Across the valid terrain scene, minimal ground reflectance produces DN values $> 40$. Thus, $DN = 0$ across all 3 bands unambiguously delineates the non-imaged border collar.
3. **Scene Equivalence Note**: In this specific LISS-IV product, all three bands share the identical spatial collar boundary. Valid terrain pixels have digital numbers $> 40$ across all three bands, while collar border pixels have digital number $0$ across all three bands simultaneously. Therefore, the evaluated condition *(valid if at least one band is nonzero)* and the alternative *(valid only if all three bands are nonzero)* evaluate to the exact same $230,424,446$ valid pixels ($78.62\%$) and $62,660,272$ collar pixels ($21.38\%$). However, the formal pipeline specification and code implementation strictly enforce **valid if at least one band is nonzero**, guaranteeing that valid terrain is preserved even in potential low-reflectance single-band anomalies.
4. **Important Distinction**: In some optical satellite products or extreme topographic shadow/deep clear water bodies, legitimate ground pixels might register $0$. For this specific LISS-IV product, zero-valued pixels are proven by spatial inspection to be exclusively collar padding.
5. **Data Integrity**: The validity mask is generated as an independent reference layer. Source and stacked digital numbers are **never erased, set to NaN, or modified** by the mask.

---

## 5. Digital Numbers Scale & Deferred Radiometric Calibration

Source digital numbers (DN) remain strictly **uncalibrated, unnormalized, and unscaled** in the stacked GeoTIFF.

### Rationale for Deferring Calibration
1. **Preservation of Raw Radiometry**: Writing calibrated Top-Of-Atmosphere (TOA) radiance ($mW/cm^2/sr/\mu m$) or surface reflectance ($\in [0.0, 1.0]$) requires `float32` representation. This would double storage size ($2\times$) and introduce floating-point quantization without providing downstream value during data staging.
2. **Decoupled Preprocessing**: Different cloud removal neural networks require different normalization formats (e.g. $[0, 1]$, $[-1, 1]$, or $z$-score standardization per channel). Deferring transformation allows the PyTorch `Dataset` to apply normalization on-the-fly dynamically.
3. **Lineage Safety**: The metadata coefficients ($L_{\text{min}}$, $L_{\text{max}}$) are preserved in the original product metadata and documented in the manifest for downstream conversion when training architectures are selected.

---

## 6. Execution & Pipeline Commands

### Running with DVC (Recommended)
The preprocessing pipeline is tracked by DVC in `dvc.yaml`:
```powershell
.\.venv\Scripts\dvc.exe repro preprocess
```
*When inputs and code are unchanged, DVC skips re-execution automatically:*
```text
Stage 'preprocess' didn't change, skipping
Data and pipelines are up to date.
```

### Running with Python Directly
```powershell
.\.venv\Scripts\python.exe -m cloud_removal.preprocess --input-dir data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA --output-dir data/interim/stacked_product
```

---

## 7. Output Directory Structure

```text
data/interim/
└── stacked_product/
    ├── stacked.tif           # 3-band tiled GeoTIFF (BAND2, BAND3, BAND4) [540 MB]
    ├── valid_mask.tif        # 1-band binary validity mask (uint8) [448 KB]
    └── manifest.json         # Complete provenance, hash, and parameter manifest [7 KB]
```

---

## 8. Radiometric TOA Radiance & Reflectance Calibration Pipeline (Phase 2)

Phase 2 implements the mathematically rigorous conversion from stored 10-bit integer Digital Numbers (DN) to physical Top-Of-Atmosphere (TOA) spectral radiance and dimensionless TOA reflectance.

### 8.1 The Physical Conversion Chain
The conversion avoids ad-hoc min-max scaling and follows the standard physical formulation:

$$\text{Stored DN (10-bit)} \xrightarrow{\text{Linear Sensor Gains}} L_\lambda\text{ (TOA Radiance)} \xrightarrow{\text{Solar Irradiance \& Geometry}} \rho_{\text{TOA}}\text{ (Reflectance)} \xrightarrow{\text{Permute to FCC}} \text{Model Input Tensor}$$

### 8.2 DN-to-Radiance Formulation
For standard georeferenced Resourcesat-2/2A LISS-IV products, conversion from raw digital counts to at-sensor spectral radiance is governed by the documented linear calibration equation:

$$L_\lambda = L_{\min} + \left( \frac{L_{\max} - L_{\min}}{\text{DN}_{\max} - \text{DN}_{\min}} \right) \cdot (\text{DN} - \text{DN}_{\min})$$

Where:
- $\text{DN}$: Stored integer digital number ($\text{DN}_{\min} = 0$, $\text{DN}_{\max} = 1023$ for 10-bit quantization).
- $L_{\min}, L_{\max}$: Band-specific calibration radiance limits provided in `BAND_META.txt` and `.meta`.
- $L_\lambda$: Spectral radiance in $\text{mW} / (\text{cm}^2 \cdot \text{sr} \cdot \mu\text{m})$.

**Calibration Parameters from Product Metadata (`BAND_META.txt`):**

| Band | Spectral Region | $\text{DN}_{\min}$ | $\text{DN}_{\max}$ | $L_{\min}\ (\text{mW}/\text{cm}^2/\text{sr}/\mu\text{m})$ | $L_{\max}\ (\text{mW}/\text{cm}^2/\text{sr}/\mu\text{m})$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **BAND2** | Green ($0.52 - 0.59\ \mu\text{m}$) | $0$ | $1023$ | $0.0000$ | $52.0000$ |
| **BAND3** | Red ($0.62 - 0.68\ \mu\text{m}$) | $0$ | $1023$ | $0.0000$ | $47.0000$ |
| **BAND4** | NIR ($0.77 - 0.86\ \mu\text{m}$) | $0$ | $1023$ | $0.0000$ | $31.5000$ |

### 8.3 Radiance-to-TOA Reflectance Formulation
Spectral radiance is converted to dimensionless Top-Of-Atmosphere reflectance $\rho_{\lambda, \text{TOA}}$ by accounting for solar illumination geometry and orbital Earth-Sun distance:

$$\rho_{\lambda, \text{TOA}} = \frac{\pi \cdot L_\lambda \cdot d^2}{\text{ESUN}_\lambda \cdot \cos(\theta_s)}$$

Where:
- $L_\lambda$: At-sensor spectral radiance in $\text{mW} / (\text{cm}^2 \cdot \text{sr} \cdot \mu\text{m})$.
- $d$: Earth-Sun astronomical distance in Astronomical Units (AU) on the day of acquisition:
  $$d = 1.0 - 0.01672 \cos\left( \frac{2\pi (\text{DOY} - 4)}{365.25} \right)$$
  *(For acquisition date 03-JUN-2026, $\text{DOY} = 154$, yielding $d = 1.0142\ \text{AU}$)*.
- $\theta_s$: Solar zenith angle in degrees ($\theta_s = 90^\circ - \text{SunElevationAtCenter} = 90.0^\circ - 70.026528^\circ = 19.973472^\circ$).
- $\cos(\theta_s) = \sin(70.026528^\circ) \approx 0.9398$.
- $\text{ESUN}_\lambda$: Band-averaged mean exo-atmospheric solar spectral irradiance in $\text{mW} / (\text{cm}^2 \cdot \mu\text{m})$.

### 8.4 Authoritative ESUN Parameters & Provenance
Because Bhoonidhi standard product metadata omits solar irradiance parameters, authoritative constants are sourced from official ISRO sensor calibration publications:

> **Authoritative Source:** Pandya, M. R., et al. (2013). *"At-sensor Solar Exo-atmospheric Irradiance, Rayleigh Optical Thickness and Spectral parameters of RS-2 Sensors."* Scientific Report No. `SAC/EPSA/SPSG/SR/01/2013`, Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad, India.

| Band | Spectral Range ($\mu\text{m}$) | Central Wavelength ($\mu\text{m}$) | $\text{ESUN}_\lambda\ (\text{mW}/\text{cm}^2/\mu\text{m})$ | $\text{ESUN}_\lambda\ (\text{W}/\text{m}^2/\mu\text{m})$ |
| :---: | :---: | :---: | :---: | :---: |
| **BAND2 (Green)** | $0.52 - 0.59$ | $0.555$ | **$181.89$** | $1818.9$ |
| **BAND3 (Red)** | $0.62 - 0.68$ | $0.650$ | **$156.96$** | $1569.6$ |
| **BAND4 (NIR)** | $0.77 - 0.86$ | $0.815$ | **$110.48$** | $1104.8$ |

*Units Consistency Note:* When $L_\lambda$ and $\text{ESUN}_\lambda$ are both specified in $\text{mW} / (\text{cm}^2 \cdot \mu\text{m})$, the metric units cancel out, yielding a strictly dimensionless physical reflectance $\rho_{\text{TOA}} \in [0.0, 1.0]$.

---

## 9. Cross-Sensor Scientific Alignment (Sentinel-2 to LISS-IV)

### 9.1 False-Color Composite (FCC) Spectral Correspondence
Natural-color RGB transfer from Sentinel-2 to LISS-IV is physically impossible because LISS-IV completely lacks a Blue band. In Phase 2, a 3-band False-Color Composite (FCC) configuration is established:

| Model Input Channel | Sentinel-2 Band (SEN12MS-CR) | LISS-IV Band (Resourcesat-2A) | Central Wavelengths ($\mu\text{m}$) | Spectral Role |
| :---: | :---: | :---: | :---: | :---: |
| **Channel 0** | **B08** ($10\text{ m}$) | **BAND4** ($5\text{ m}$) | $0.842\ \mu\text{m} \leftrightarrow 0.815\ \mu\text{m}$ | Near-Infrared (NIR) |
| **Channel 1** | **B04** ($10\text{ m}$) | **BAND3** ($5\text{ m}$) | $0.665\ \mu\text{m} \leftrightarrow 0.650\ \mu\text{m}$ | Red |
| **Channel 2** | **B03** ($10\text{ m}$) | **BAND2** ($5\text{ m}$) | $0.560\ \mu\text{m} \leftrightarrow 0.555\ \mu\text{m}$ | Green |

### 9.2 Critical Cross-Sensor Domain Disparities
While FCC alignment resolves channel count and primary spectral meaning, the following physical differences remain active and must be reported honestly:

1. **Spatial Resolution ($2\times$ Domain Gap):**
   - Sentinel-2 Bands 3, 4, 8 have a native ground sampling distance (GSD) of $10.0\text{ m}$.
   - Resourcesat-2A LISS-IV has a native resolution of $5.8\text{ m}$ (resampled to $5.0\text{ m}$ in standard products).
   - High-frequency edge and texture details in LISS-IV exceed the Nyquist frequency of the Sentinel-2 training data.
2. **Spectral Response Functions (SRF):**
   - Sentinel-2 B08 has a broader NIR response ($0.780 - 0.900\ \mu\text{m}$, FWHM $\sim 115\text{ nm}$) than LISS-IV Band 4 ($0.770 - 0.860\ \mu\text{m}$, FWHM $\sim 90\text{ nm}$).
   - Water vapor absorption at $0.82\ \mu\text{m}$ affects the sensors differently.
3. **Radiometric Level Disparity:**
   - SEN12MS-CR contains Sentinel-2 Level-1C / Level-2A reflectance scaled by $10,000$ ($\text{DN} / 10000.0$).
   - LISS-IV imagery is converted to Top-Of-Atmosphere (TOA) reflectance using sensor-specific calibration and geometry.
   - Atmospheric scattering (Rayleigh and aerosol optical depth) influences TOA reflectance more strongly than surface reflectance.

```powershell
# CLI verification of parsed calibration and ESUN parameters
.\.venv\Scripts\python.exe -m cloud_removal.preprocess --show-calibration
```

