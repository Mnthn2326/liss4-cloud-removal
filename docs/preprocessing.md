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
