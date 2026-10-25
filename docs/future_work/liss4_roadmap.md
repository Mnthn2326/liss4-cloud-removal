# Future Work: Resourcesat-2A LISS-IV Cross-Sensor Adaptation

> **Status:** **Experimental — Unsupported for Deployed Inference in Version 1.0**
>
> The Version 1.0 production baseline is trained and officially evaluated on the **RICE2** paired RGB benchmark. The LISS-IV tooling in this repository is retained strictly for exploratory cross-sensor feasibility research and is not supported for production inference.

---

## 1. Physical Sensor Discrepancies

As established in the Phase 7.6 and Phase 7.7 feasibility audits, direct transfer of an RGB-trained deep neural network to Resourcesat-2A LISS-IV imagery is blocked by three fundamental domain discrepancies:

| Domain Factor | RICE2 / Sentinel-2 Production Baseline | ISRO Resourcesat-2A LISS-IV |
| :--- | :--- | :--- |
| **Spectral Channels** | Standard RGB (Red, Green, Blue) | False Color Composite (Band 2: Green, Band 3: Red, Band 4: NIR) — **No Blue band** |
| **Spatial Resolution** | $10\text{ m}$ (Sentinel-2) / $30\text{ m}$ (Landsat) | $5.8\text{ m}$ (Resampled to $5.0\text{ m}$ in Level-1 products) |
| **Radiometric Units** | Bottom-of-Atmosphere (BOA) Reflectance $[0.0, 1.0]$ | Uncalibrated Digital Numbers ($DN \in [0, 1023]$, 10-bit or 7-bit quantized) |
| **Solar Irradiance** | Standard $ESUN_\lambda$ tables provided in metadata | **Omitted** from official Bhoonidhi `.xml` metadata files |

---

## 2. Dataset Constraints

* **Mono-temporal Monsoon Scene**: The repository's LISS-IV dataset (`data/raw/liss4/`) consists of a single acquisition (`RAF03JUN...`) taken during the active monsoon season with extensive cumulus cloud cover.
* **Absence of Clear-Sky Ground Truth**: There is no temporally adjacent, co-registered cloud-free LISS-IV scene of the same geographic footprint. Without ground truth, quantitative validation (MAE, PSNR, SSIM) on real LISS-IV scenes is impossible without synthetic simulation.
* **Uncertain Calibration**: Calculating Top-of-Atmosphere (TOA) reflectance requires:
  $$\rho_\lambda = \frac{\pi \cdot L_\lambda \cdot d^2}{ESUN_\lambda \cdot \cos(\theta_s)}$$
  Because $ESUN_\lambda$ values for Resourcesat-2A LISS-IV are omitted from open Bhoonidhi product metadata and vary with sensor response functions, converting DNs to physical reflectance requires external calibration tables from the National Remote Sensing Centre (NRSC).

---

## 3. Retained Exploratory Modules

The following source modules are maintained for ongoing academic exploration but are flagged as experimental:

* `src/cloud_removal/ingest_liss4.py`: Unzips, checks checksums, and parses Band 2/3/4 GeoTIFFs from Bhoonidhi products.
* `src/cloud_removal/inspect.py`: Summarizes raster metadata, coordinate reference systems, and radiometric ranges.
* `src/cloud_removal/models/patchgan.py` & `src/cloud_removal/train_cgan.py`: Experimental Conditional GAN pipeline for adversarial domain transfer.

---

## 4. Roadmap to Version 2.0 LISS-IV Support

To responsibly graduate LISS-IV from experimental research to an operational model:

1. **Multi-temporal Acquisition**: Retrieve 2–3 authentic LISS-IV scenes over identical Path/Row coordinates with one acquisition having $< 1\%$ cloud cover to establish a ground-truth pair.
2. **Calibration Standard**: Secure official NRSC Bandpass solar irradiance values to convert 10-bit raw digital numbers directly to physical BOA or TOA reflectance.
3. **Dedicated FCC Architecture**: Train a model natively on False Color Composite bands (NIR, Red, Green) rather than attempting zero-shot inference with RGB weights.
4. **Unsupervised Domain Adaptation**: Explore CycleGAN or contrastive unpaired domain translation between Sentinel-2 and LISS-IV to bridge the $5\text{ m} \leftrightarrow 10\text{ m}$ resolution gap.
