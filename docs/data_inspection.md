# Technical Inspection: Extracted Resourcesat-2A LISS-IV Product

## 1. Product Identification
A real Resourcesat-2A LISS-IV satellite imagery product was discovered and inspected under `data/raw/`:

- **Product Identifier (ProductID)**: `2643035131`
- **OTS Product ID**: `RAF03JUN2026049233009500060SSANSTUC00GTDA`
- **Product Directory**: `data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/`
- **Satellite Mission (SatID)**: `IRS-R2A` (Resourcesat-2A)
- **Sensor**: `L4FX` (Linear Imaging Self-Scanning Sensor-4, steerable payload)
- **Sub-scene**: `A`
- **Orbit & Path / Row**: Path `095`, Row `060`, Orbit `049233`
- **Acquisition Date (DateOfPass)**: `03-JUN-2026`
- **Scene Center Time**: `05:41:22.998276 UTC`
- **Processing Level**: `STD` (Standard Georeferenced / Orthorectified, `ProdType=GEOREF`)
- **Resampling Method**: `CC` (Cubic Convolution)
- **DEM Correction Applied**: `YES`
- **Geometric Look Angles**:
  - Sun Elevation at Center: $70.03^\circ$
  - Sun Azimuth at Center: $74.76^\circ$
  - Incidence Angle: $2.98^\circ$ (Tilt angle: $-3.04^\circ$)
  - View Angle: $2.64^\circ$
- **Geographic Footprint**: Centered at Lat $18.26^\circ\text{N}$, Lon $73.34^\circ\text{E}$ (Western Maharashtra / Konkan / Western Ghats region, India).

---

## 2. Raster Metadata Table

Each band is delivered as an individual uncompressed GeoTIFF file. All three files were inspected using `rasterio`:

| Property | `BAND2.tif` | `BAND3.tif` | `BAND4.tif` |
| :--- | :--- | :--- | :--- |
| **File Path** | `data/raw/.../BAND2.tif` | `data/raw/.../BAND3.tif` | `data/raw/.../BAND4.tif` |
| **File Size** | 586,299,606 bytes (559.14 MB) | 586,299,606 bytes (559.14 MB) | 586,299,606 bytes (559.14 MB) |
| **Raster Dimensions ($W \times H$)** | $18,066 \times 16,223$ | $18,066 \times 16,223$ | $18,066 \times 16,223$ |
| **Total Raster Pixels** | 293,084,718 | 293,084,718 | 293,084,718 |
| **Number of Bands** | 1 | 1 | 1 |
| **Coordinate Reference System (CRS)** | `EPSG:32643` (UTM Zone 43N / WGS 84) | `EPSG:32643` (UTM Zone 43N / WGS 84) | `EPSG:32643` (UTM Zone 43N / WGS 84) |
| **Pixel Resolution ($X, Y$)** | $5.000000\text{ m} \times 5.000000\text{ m}$ | $5.000000\text{ m} \times 5.000000\text{ m}$ | $5.000000\text{ m} \times 5.000000\text{ m}$ |
| **Affine Geotransform** | `[5.0, 0.0, 279458.943, 0.0, -5.0, 2060222.5]` | Identical | Identical |
| **Bounding Box (UTM Easting, Northing)**| Left: `279458.94`, Right: `369788.94`<br>Bottom: `1979107.5`, Top: `2060222.5` | Identical | Identical |
| **Data Type (Storage / Quantization)** | `uint16` (10-bit DN, $0 - 1023$) | `uint16` (10-bit DN, $0 - 1023$) | `uint16` (10-bit DN, $0 - 1023$) |
| **Header NoData Value** | `None` | `None` | `None` |
| **Zero (Collar/Border) Pixels** | 62,660,272 ($21.38\%$) | 62,660,272 ($21.38\%$) | 62,660,272 ($21.38\%$) |
| **Valid Imagery (Non-zero) Pixels** | 230,424,446 ($78.62\%$) | 230,424,446 ($78.62\%$) | 230,424,446 ($78.62\%$) |
| **Valid Pixel Min** | 64 | 47 | 46 |
| **Valid Pixel Max** | 1023 (Saturation) | 1023 (Saturation) | 1023 (Saturation) |
| **Valid Pixel Mean** | 251.98 | 250.90 | 327.11 |
| **Valid Pixel Standard Deviation** | 200.35 | 205.99 | 180.58 |
| **Compression** | None (Uncompressed Strip GeoTIFF) | None (Uncompressed Strip GeoTIFF) | None (Uncompressed Strip GeoTIFF) |
| **Block Layout** | Strip: `blockxsize=18066, blockysize=1` | Strip: `blockxsize=18066, blockysize=1` | Strip: `blockxsize=18066, blockysize=1` |
| **Color Interpretation** | `ColorInterp.gray` | `ColorInterp.gray` | `ColorInterp.gray` |

---

## 3. File Integrity and Readability Results

1. **Successful Opening**: All three GeoTIFFs (`BAND2.tif`, `BAND3.tif`, `BAND4.tif`) and associated metadata text files opened cleanly without corruption or decoding errors.
2. **Windowed Read Verification**:
   - A $256 \times 256$ window centered around pixel $(X=9033, Y=8111)$ was read for each band using `rasterio.windows.Window`.
   - Window statistics:
     - `BAND2`: Min 93, Max 426, Mean 142.76, Std 41.18
     - `BAND3`: Min 75, Max 449, Mean 143.33, Std 48.08
     - `BAND4`: Min 114, Max 498, Mean 244.13, Std 54.61
   - Reads executed in under 15 milliseconds per window without loading full rasters into RAM.
3. **Full-Raster Streaming Verification**:
   - The rasters were processed in streaming chunks of 1024 rows to compute comprehensive pixel statistics. Peak memory consumption remained under 35 MB throughout the scan across 293 million pixels.
4. **Grid Alignment**:
   - The geotransform, pixel dimensions ($18,066 \times 16,223$), and geographic bounding boxes match exactly across all three bands. The zero collar mask matches to the exact pixel ($230,424,446$ non-zero pixels across all bands).

---

## 4. Metadata Discrepancies and Findings

Comparing the actual deliverables against sensor specifications and project notes revealed five notable points:

1. **Pixel Resolution ($5.0\text{ m}$ vs $5.8\text{ m}$)**:
   - *Expectation*: Nominal LISS-IV sensor ground sampling distance is $5.80\text{ m}$ at nadir.
   - *Actual*: The output rasters have a pixel size of exactly $5.000000\text{ m}$.
   - *Explanation*: In `BAND_META.txt`, lines 25–28 confirm:
     ```text
     InputResolutionAlong=   5.80
     InputResolutionAcross=  5.80
     OutputResolutionAlong=  5.00
     OutputResolutionAcross= 5.00
     ResampCode= CC
     ```
     The NRSC IMGEOS processing chain automatically resamples standard georeferenced/orthorectified LISS-IV products to a uniform $5.0\text{ m}$ grid using Cubic Convolution.

2. **Undeclared NoData Collar**:
   - The GeoTIFF headers state `nodata: None`.
   - However, because the satellite swath is inclined at heading $193.52^\circ$ relative to the north-up UTM grid, $21.38\%$ ($62,660,272$ pixels) of the rectangular raster bounding box consists of zero-valued background padding ($DN = 0$).
   - Downstream data pipelines must explicitly treat $DN = 0$ as NoData / collar padding.

3. **Missing Cloud Cover Metric (`CloudPercent = -NA-`)**:
   - In `BAND_META.txt`, line 96 reports:
     ```text
     CloudPercent= -NA-
     ```
   - While Bhoonidhi's catalog interface provides cloud cover search filtering, this specific product deliverable does not contain an evaluated scene cloud percentage in its metadata.

4. **Spectral Band Mapping Unverified Internally**:
   - The product files are named `BAND2.tif`, `BAND3.tif`, `BAND4.tif`, and the metadata reports `BandNumbers= 234`.
   - Radiometric parameters provided in `BAND_META.txt`:
     ```text
     B2_Lmin= 0.0000, B2_Lmax= 52.0000
     B3_Lmin= 0.0000, B3_Lmax= 47.0000
     B4_Lmin= 0.0000, B4_Lmax= 31.5000
     ```
   - **Important finding**: The delivered metadata files **do not** explicitly declare the central wavelengths, bandwidths, or spectral band names (e.g. Green, Red, NIR).
   - **Verification status**: From the product metadata alone, the spectral assignment is **unverified**. Based on external ISRO Resourcesat-2A specifications:
     - `BAND2` $\rightarrow$ Green ($0.52 - 0.59\ \mu\text{m}$)
     - `BAND3` $\rightarrow$ Red ($0.62 - 0.68\ \mu\text{m}$)
     - `BAND4` $\rightarrow$ Near-Infrared ($0.77 - 0.86\ \mu\text{m}$)
     This external mapping must be confirmed during band stacking.

5. **Storage Organization**:
   - The TIFF files are uncompressed strip GeoTIFFs (`blockxsize=18066, blockysize=1`). They are not tiled into blocks ($256 \times 256$ or $512 \times 512$). Random spatial window queries require reading entire rows, which should be converted to tiled Cloud-Optimized GeoTIFF (COG) in Phase 3.

---

## 5. Known Limitations

1. **Spectral Channels**: Only 3 bands are present (B2, B3, B4). No Blue band ($0.45 - 0.51\ \mu\text{m}$), Cirrus band ($1.37\ \mu\text{m}$), or SWIR bands exist, preventing the use of standard blue-based haze indices (HOT) or cirrus cloud detection.
2. **Saturation in Clouds**: Maximum pixel values reach $1023$ across all three bands, confirming radiometric saturation in dense cloud cores.
3. **No Pixel-Level Cloud Mask**: Ground truth cloud locations are not provided and must be identified synthetically or via spectral thresholding.

---

## 6. Suitability for Data-Pipeline Validation

This product is **exceptionally well-suited** for developing and validating the Phase 3 data ingestion and preprocessing pipeline:

- **Identical Spatial Grids**: All three bands share identical dimensions, CRS, affine transforms, and collar masks, eliminating the need for band-to-band spatial resampling.
- **Minimal Look Distortion**: The incidence angle is only $2.98^\circ$ (tilt $-3.04^\circ$), well below the $15^\circ$ threshold, ensuring minimal terrain relief displacement.
- **Genuine Radiometry**: Authentic 10-bit DN values with calibrated $L_{\text{min}}$ and $L_{\text{max}}$ coefficients provide realistic spectral distributions for testing normalization, tiling, and synthetic cloud degradation.

---

## 7. Outstanding Requirements Before Model Training

Before model training can begin in subsequent phases:
1. **Band Stacking**: Combine the three separate single-band rasters into a unified 3-band composite GeoTIFF (`shape = (3, 16223, 18066)`).
2. **Collar Masking**: Explicitly mask out the $DN = 0$ border collar to prevent zero padding from contaminating loss functions and training statistics.
3. **Radiometric Calibration**: Convert raw 10-bit DNs to Top-of-Atmosphere (TOA) Radiance using the metadata gains:
   $$L_\lambda = L_{\text{min}} + \left(\frac{\text{DN}}{\text{DN}_{\text{max}}}\right) \cdot (L_{\text{max}} - L_{\text{min}})$$
4. **Patch Extraction & Tiling**: Slice the valid interior into standard deep learning training patches (e.g., $256 \times 256$ or $512 \times 512$).
5. **Synthetic Cloud Pipeline**: Generate paired $\{X_{\text{cloudy}}, Y_{\text{clean}}\}$ training patches using physical transmission models on clean subsets.
6. **DVC Tracking**: Initialize DVC and track the raw product and processed dataset artifacts.
