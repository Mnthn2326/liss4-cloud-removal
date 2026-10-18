# Phase 7.6 — LISS-IV Cross-Sensor Feasibility Assessment

---

## 1. Objective and Executive Conclusion

### Objective
Perform a rigorous, conservative feasibility assessment to establish whether the existing compact U-Net baseline (`checkpoints/compact/best_compact_unet.pt`, 1.94M parameters), trained on Sentinel-2 optical imagery from **SEN12MS-CR**, can be meaningfully applied to the available **Resourcesat-2A LISS-IV** imagery (`data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/` and `data/interim/stacked_product/stacked.tif`).

### Decision: **NO-GO for Direct Inference**
**Direct inference on the current LISS-IV stack is scientifically indefensible and blocked.**

Feeding the existing LISS-IV product directly into the Sentinel-2-trained U-Net baseline would constitute an invalid, physically misleading band substitution because:
1. **LISS-IV completely lacks a Blue spectral channel.** The available LISS-IV bands are Green (B2), Red (B3), and Near-Infrared (B4). Substituting Near-Infrared into the model's Blue channel violates the fundamental physics of optical surface reflectance (vegetation reflects heavily in NIR while absorbing strongly in Blue).
2. **Radiometric Disparity:** The model expects physical Bottom-of-Atmosphere (BOA) surface reflectance in $[0.0, 1.0]$. The LISS-IV imagery contains raw, uncalibrated 10-bit quantized Digital Numbers (DN) in $[0, 1023]$.
3. **Spatial Resolution Disparity:** LISS-IV has a native ground sampling distance of $5.0\text{ m}$, whereas the U-Net was trained on $10.0\text{ m}$ Sentinel-2 rasters ($2\times$ spatial scale mismatch).
4. **Absence of Ground Truth Reference:** The LISS-IV product is a single mono-temporal pass with no paired cloud-free acquisition, precluding any legitimate quantitative evaluation.

---

## 2. Evidence Inspected

The following primary data, metadata, model configurations, and documentation artifacts were inspected:

1. **LISS-IV Product Metadata & Raw Rasters:**
   - [`data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/BAND_META.txt`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/BAND_META.txt)
   - [`data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/RAF03JUN2026049233009500060SSANSTUC00GTDA.meta`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/RAF03JUN2026049233009500060SSANSTUC00GTDA.meta)
   - Raw single-band GeoTIFFs: `BAND2.tif` (586 MB), `BAND3.tif` (586 MB), `BAND4.tif` (586 MB).
2. **Preprocessed Interim Product:**
   - [`data/interim/stacked_product/stacked.tif`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/data/interim/stacked_product/stacked.tif): 3-band composite GeoTIFF ($18,066 \times 16,223$ pixels, EPSG:32643, $5.0\text{ m}$ resolution, `dtypes=('uint16', 'uint16', 'uint16')`, descriptions `('BAND2', 'BAND3', 'BAND4')`).
3. **Model Checkpoint & Architecture Configuration:**
   - [`checkpoints/compact/best_compact_unet.pt`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/checkpoints/compact/best_compact_unet.pt): 1,942,611 parameters, 3 input channels, terminal Sigmoid bounding $[0.0, 1.0]$.
4. **Dataset Loader & Spectral Specification:**
   - [`src/cloud_removal/sen12mscr.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/sen12mscr.py): `DEFAULT_RGB_BANDS = (4, 3, 2)` (Sentinel-2 B04, B03, B02).
5. **Prior Technical Documentation:**
   - [`docs/data_inspection.md`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/docs/data_inspection.md) and [`docs/data_feasibility.md`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/docs/data_feasibility.md).

---

## 3. Verified Facts versus Unresolved Assumptions

### Verified Facts Table

| Parameter | Sentinel-2 (Model Expectation) | Resourcesat-2A LISS-IV (Available Data) | Status |
| :--- | :--- | :--- | :--- |
| **Input Channel 0** | **Band 4 (Red)**: $\lambda_c = 665\text{ nm}$ (FWHM: $38\text{ nm}$) | **Band 3 (Red)**: $620 - 680\text{ nm}$ | **Compatible** |
| **Input Channel 1** | **Band 3 (Green)**: $\lambda_c = 560\text{ nm}$ (FWHM: $45\text{ nm}$) | **Band 2 (Green)**: $520 - 590\text{ nm}$ | **Compatible** |
| **Input Channel 2** | **Band 2 (Blue)**: $\lambda_c = 490\text{ nm}$ (FWHM: $98\text{ nm}$) | **Band 4 (NIR)**: $770 - 860\text{ nm}$ | **CRITICAL MISMATCH (No Blue band)** |
| **Spatial Resolution**| **10.0 meters** | **5.0 meters** (Resampled from 5.8m via CC) | **$2\times$ Scale Difference** |
| **Radiometric Units** | **Physical Surface Reflectance** $[0.0, 1.0]$ | **10-bit Raw DN** ($0 - 1023$) stored in `uint16` | **Incompatible Scale** |
| **Calibration Parameters**| Documented fixed divisor: $\text{DN} / 10000.0$ | Band-specific $L_{\text{min}}, L_{\text{max}}$ provided; $ESUN_\lambda$ not provided | **Requires TOA Radiance & Reflectance Pipeline** |
| **Cloud-Free Reference**| Paired Sentinel-2 clear target available | **None** (Single mono-temporal acquisition) | **Evaluation Blocked** |

### Unresolved Assumptions / Blocker Analysis

1. **The "Three-Band Assumption" Trap:**
   Because both the model input shape is `[3, H, W]` and the LISS-IV stack has 3 bands, it is tempting to pass `stacked.tif` into the model. However, the three bands in LISS-IV represent **Green, Red, and Near-Infrared (NIR)**, not Red, Green, Blue.
   - In the natural world, healthy vegetation absorbs heavily in the Blue band (reflectance $\approx 2\% - 5\%$) due to chlorophyll absorption, but reflects up to $50\% - 70\%$ in the Near-Infrared band due to leaf cellular structure.
   - Feeding NIR into a convolutional filter trained to interpret Blue reflectance will cause severe artifacts: dense green vegetation will appear as an extreme positive anomaly, confusing the network's cloud/clear discrimination.
2. **The "Naive Normalization" Trap:**
   Dividing LISS-IV DN values by 1023 (or 10000) does not yield surface reflectance. LISS-IV requires sensor calibration to at-sensor radiance:
   $$L_\lambda = L_{\text{min}} + \left(\frac{\text{DN}}{\text{DN}_{\text{max}}}\right) \cdot (L_{\text{max}} - L_{\text{min}})$$
   followed by Top-of-Atmosphere (TOA) and Bottom-of-Atmosphere (BOA) atmospheric correction. Passing raw sensor counts into a reflectance-trained network destroys the numerical distribution established during training.
3. **Receptive Field Distortions:**
   Convolutional kernels trained on 10m pixels have learned spatial feature sizes (cloud boundaries, field edges) corresponding to a $2560\text{ m} \times 2560\text{ m}$ footprint per $256 \times 256$ patch. Feeding 5m data without $2\times$ spatial binning/downsampling exposes the model to features at $2\times$ physical zoom.

---

## 4. Go / No-Go Decision

### **Decision: NO-GO**
- **Exploratory inference will NOT be executed.**
- Running inference on scientifically incompatible band assignments and uncalibrated radiometry would produce uninterpretable outputs that misrepresent both the model's true capabilities and the sensor's data.
- Creating an exploratory image with false band substitution merely to claim completion violates the project's core principles of scientific integrity and reproducibility.

---

## 5. Required Next Steps to Achieve Cross-Sensor Transfer

To enable genuine, scientifically grounded cloud removal on LISS-IV in future phases, the following prerequisite engineering tasks must be completed:

1. **Sensor-Matched Architecture / Training:**
   - Train or fine-tune an explicit **False-Color Composite (FCC)** U-Net baseline on Sentinel-2 bands **(B08 NIR, B04 Red, B03 Green)** or **(B03 Green, B04 Red, B08 NIR)** from SEN12MS-CR.
   - This provides direct spectral band correspondence to LISS-IV's `(BAND2, BAND3, BAND4)` without substituting wavelengths.
2. **LISS-IV Radiometric Pipeline:**
   - Implement a radiometric calibration utility in `src/cloud_removal/preprocess.py` to convert LISS-IV 10-bit DNs to TOA Radiance and BOA surface reflectance using metadata gains ($L_{\text{min}}, L_{\text{max}}$) and solar geometry (Sun Elevation: $70.03^\circ$).
3. **Spatial Aggregation:**
   - Resample LISS-IV $5.0\text{ m}$ imagery to $10.0\text{ m}$ using area-weighted pixel averaging to match the receptive field of 10m models, or train directly on native $5.0\text{ m}$ patches.
4. **Synthetic Cloud Pairing on LISS-IV:**
   - Leverage the Phase 4 synthetic cloud simulator ([`src/cloud_removal/synthetic_clouds.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/synthetic_clouds.py)) to generate controlled, paired $\{X_{\text{cloudy}}, Y_{\text{clear}}\}$ patches directly on clean LISS-IV sub-scenes, bypassing the need for impossible multi-temporal clear pairs.

---

## 6. Formal Accuracy Disclaimers

- **Zero LISS-IV Performance Claims:** No claims of accuracy, transferability, or operational readiness are made for Resourcesat-2A LISS-IV data.
- **Scope of Existing Metrics:** The reported validation metrics (MAE `0.00667`, PSNR `40.00 dB`) apply strictly to the SEN12MS-CR Sentinel-2 validation subset and cannot be cited as evidence of LISS-IV cloud-removal capability.
