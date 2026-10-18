# Phase 7.7 — Synthetic LISS-IV Benchmark Feasibility Assessment

---

## 1. Executive Summary & Recommendation

### Recommendation: **NO-GO (BLOCKED)**
**Status:** **`BLOCKED`**

Creating a scientifically defensible synthetic cloud-removal benchmark for Resourcesat-2A LISS-IV using the current project state is **blocked**.

Although the project's synthetic cloud simulation utilities ([`src/cloud_removal/synthetic_clouds.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/synthetic_clouds.py)) are algorithmically sound and functional, the repository lacks two fundamental empirical prerequisites:
1. **No Cloud-Free LISS-IV Reference Scene:** The repository contains exactly one LISS-IV product (`RAF03JUN2026049233009500060SSANSTUC00GTDA`), which is an active monsoon acquisition featuring thick, saturated cloud cover. Generating synthetic cloudy pairs requires an authentic, clean ground reference; overlaying synthetic clouds onto an already cloud-contaminated image would produce corrupted ground truth.
2. **Incomplete Radiometric Calibration Metadata:** While minimum and maximum radiance values ($L_{\text{min}}, L_{\text{max}}$) are provided, the official metadata completely omits the Mean Solar Exo-atmospheric Spectral Irradiance ($ESUN_\lambda$) and atmospheric parameters required to convert raw Digital Numbers to physical surface reflectance.
3. **Cross-Sensor Evaluation Incompatibility:** The existing U-Net baseline was trained on Sentinel-2 RGB imagery. Because LISS-IV provides Green, Red, and Near-Infrared (NIR) channels with no Blue band, a synthetic LISS-IV benchmark cannot be evaluated against the current model checkpoint without introducing invalid spectral substitutions.

In adherence to project safeguards, **no synthetic benchmark dataset was generated, no new models were trained, and no fabricated ground truth was created.**

---

## 2. Assessment of Documented Calibration Parameters

Audited from [`data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/BAND_META.txt`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/BAND_META.txt):

### Documented Metadata Values
- **Quantization:** 10-bit unsigned integer ($DN \in [0, 1023]$) stored in 16-bit GeoTIFF container (`uint16`).
- **Radiance Gains:**
  - Band 2 (Green): $L_{\text{min}} = 0.0000$, $L_{\text{max}} = 52.0000$
  - Band 3 (Red): $L_{\text{min}} = 0.0000$, $L_{\text{max}} = 47.0000$
  - Band 4 (NIR): $L_{\text{min}} = 0.0000$, $L_{\text{max}} = 31.5000$
- **Solar Geometry:**
  - Sun Elevation at Center: $70.026528^\circ$ (Zenith Angle $\theta_s = 19.973472^\circ$)
  - Sun Azimuth at Center: $74.758853^\circ$
  - Date of Pass: `03-JUN-2026`, Center Time: `05:41:22.998276 UTC`

### Undocumented / Missing Parameters
1. **Radiance Units:** The physical units for $L_{\text{max}}$ (typically $\text{mW}/(\text{cm}^2 \cdot \text{sr} \cdot \mu\text{m})$ or $\text{W}/(\text{m}^2 \cdot \text{sr} \cdot \mu\text{m})$) are omitted from the text deliverable.
2. **Mean Solar Exo-Atmospheric Spectral Irradiance ($ESUN_\lambda$):** **Completely absent.** Conversion from Top-of-Atmosphere (TOA) Radiance $L_\lambda$ to TOA Reflectance $\rho_\lambda$ is governed by:
   $$\rho_\lambda = \frac{\pi \cdot L_\lambda \cdot d^2}{ESUN_\lambda \cdot \cos(\theta_s)}$$
   Without documented, sensor-specific $ESUN_\lambda$ values for Resourcesat-2A LISS-IV, calculating TOA reflectance requires external unverified assumptions.
3. **Atmospheric Correction Model:** No aerosol optical thickness, atmospheric transmittance, or path radiance parameters exist to translate TOA reflectance to Bottom-of-Atmosphere (BOA) surface reflectance (the domain in which the Sentinel-2 U-Net was trained).

**Verdict:** Documented calibration parameters are **insufficient** for an assumption-free conversion to physical surface reflectance.

---

## 3. Availability of Cloud-Free Reference Imagery

### Inventory of Available LISS-IV Scenes
- **Directory:** `data/raw/`
- **Total LISS-IV Products on Disk:** **1** (`RAF03JUN2026049233009500060SSANSTUC00GTDA`)

### Scene Cloud Assessment
- **Acquisition Date:** June 3, 2026 (Onset of the Indian Southwest Monsoon).
- **Location:** Western Ghats / Konkan coastal region, Maharashtra, India.
- **Observed Radiometry:**
  - Maximum pixel values reach **1023 (saturation)** across all three bands (`BAND2.tif`, `BAND3.tif`, `BAND4.tif`).
  - Saturated cores are surrounded by extensive semi-transparent cloud boundaries and haze.
  - The browse image (`RAF03JUN2026049233009500060SSANSTUC00GTDA.jpg`) confirms substantial real cloud cover over the terrain.
- **Suitability as Ground Truth Reference:** **UNSUITABLE.**
  Treating this product as a "clear reference" to overlay synthetic clouds would violate basic scientific principles: the benchmark targets ($Y_{\text{clear}}$) would already be contaminated with real clouds, shadows, and saturated pixels.

**Verdict:** **No genuinely cloud-free LISS-IV reference scene exists in the repository.**

---

## 4. Assessment of Existing Synthetic-Cloud Utilities

Inspected [`src/cloud_removal/synthetic_clouds.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/synthetic_clouds.py) and [`tests/test_synthetic_clouds.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/tests/test_synthetic_clouds.py):

### Technical Capabilities
1. **Continuous Opacity Mask Generation (`generate_cloud_mask`):**
   - Multi-scale smoothed stochastic Gaussian noise combining coarse structural blobs (`coarse_scale_fraction=0.18`) with high-frequency edge turbulence (`fine_scale_fraction=0.05`).
   - Produces continuous transmission opacity masks $\alpha(x) \in [0.0, 1.0]$ with feathered edges.
   - Fully deterministic when seeded.
2. **Atmospheric Radiative Transfer Blending (`blend_synthetic_clouds`):**
   - Implements simplified atmospheric scattering:
     $$I_{\text{cloudy}}(x) = (1 - \alpha(x)) \cdot I_{\text{clear}}(x) + \alpha(x) \cdot C(x)$$
   - Operates on arbitrary channel dimensions $(C, H, W)$ or $(H, W, C)$.
   - Specifically accommodates 10-bit integer digital numbers (`cloud_dn_min=700`, `cloud_dn_max=1023`).
   - All 8 dedicated unit tests pass in `pytest`.

### Domain Limitations
- **Clean Input Requirement:** The utility mathematically assumes that the input image $I_{\text{clear}}$ represents cloud-free surface radiance. It cannot distinguish pre-existing clouds from ground terrain.
- **Spectral Representation:** Cloud radiance $C(x)$ is simulated as near-white across channels. While physically appropriate for optical cloud scattering (Mie scattering across visible/NIR), it cannot overcome the spectral mismatch between a Green-Red-NIR scene and an RGB-trained model.

**Verdict:** The utility is **algorithmically suitable** for synthetic cloud generation, but cannot be utilized defensively until an authentic clear LISS-IV scene is provided.

---

## 5. Specific Blockers & Evidence Required to Resolve

| Blocker ID | Blocker Description | Evidence Required to Resolve |
| :--- | :--- | :--- |
| **BLOCKER-1** | **Absence of Clean Reference Scene:** The repository contains only one LISS-IV scene, which is heavily cloudy. | Acquisition of an authentic, verified cloud-free ($\le 1\%$ cloud cover) Resourcesat-2A LISS-IV scene from Bhoonidhi covering agricultural or mixed land cover. |
| **BLOCKER-2** | **Missing Solar Spectral Irradiance ($ESUN_\lambda$):** Official product metadata does not include $ESUN$, preventing calculation of TOA reflectance. | Authoritative NRSC/ISRO calibration tables specifying band-averaged solar irradiance ($ESUN$) for LISS-IV Bands 2, 3, 4. |
| **BLOCKER-3** | **Spectral Incompatibility with Existing Model:** The existing U-Net baseline expects RGB inputs. LISS-IV provides Green, Red, and NIR. | Development and training of a dedicated False-Color Composite (FCC: Green, Red, NIR) baseline model on matching SEN12MS-CR bands (B03, B04, B08). |

---

## 6. Formal Statements & Compliance

- **No Fabricated Data:** No synthetic benchmark data was generated during this phase.
- **No Accuracy Claims:** No claims of cloud removal accuracy or cross-sensor transfer are made.
- **Integrity Verified:**
  - Raw LISS-IV imagery remains unchanged (`BAND2.tif`, `BAND3.tif`, `BAND4.tif`: 586,299,606 bytes each).
  - Existing model checkpoints remain unchanged (`best_compact_unet.pt`: 23,421,227 bytes; `best_unet.pt`: 93,276,427 bytes).
  - DVC status is clean (`Data and pipelines are up to date`).
  - Git working tree remains uncommitted.
