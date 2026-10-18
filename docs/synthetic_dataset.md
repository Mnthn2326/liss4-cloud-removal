# Synthetic Cloud Generation & Paired Dataset Preparation

## 1. Executive Status & Reference-Data Strategy

### Verified Clear Reference Status: **UNAVAILABLE (BLOCKED)**
- **Current Local Holdings**: The repository currently holds a single raw LISS-IV scene under `data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/` (acquired on 2026-06-03 during the Indian southwest monsoon season).
- **Physical Characteristics**: The scene features extensive real cloud formations with dense cores saturating 10-bit digital numbers up to $1023$.
- **Defensible Ground-Truth Principle**:
  - In supervised deep learning (such as a U-Net baseline), the target tensor must represent true cloud-free surface reflectance.
  - **The existing cloudy scene cannot serve as clear ground truth**, and synthesizing fake "clear" ground truth by artificially masking or interpolating cloud pixels from the same image is scientifically indefensible.
  - **No external data was silently downloaded**, and no synthetic pairs were generated from the cloudy scene.
  - Consequently, **production paired-dataset generation is formally BLOCKED** until a genuine cloud-free reference scene of the identical geographic footprint is acquired.
- **Implemented Deliverables**:
  - Reusable, deterministic synthetic cloud generation module: [`src/cloud_removal/synthetic_clouds.py`](../src/cloud_removal/synthetic_clouds.py).
  - Reusable, leakage-preventing patch extraction and geographic splitting module: [`src/cloud_removal/patch_dataset.py`](../src/cloud_removal/patch_dataset.py).
  - Comprehensive unit test suites operating on synthetic fixtures: [`tests/test_synthetic_clouds.py`](../tests/test_synthetic_clouds.py) and [`tests/test_patch_dataset.py`](../tests/test_patch_dataset.py).

---

## 2. Smallest Viable Path to Obtain a Clear Reference

To unblock supervised paired dataset generation, a single cloud-free companion scene must be retrieved from the Bhoonidhi / ISRO open catalog with the following search criteria:

| Parameter | Specification | Rationale |
| :--- | :--- | :--- |
| **Satellite / Sensor** | Resourcesat-2A (`IRS-R2A`) / `L4FX` | Ensures identical $5.0\text{ m}$ spatial resolution and radiometric characteristics |
| **Nominal Path / Row** | Path `049`, Row `233`, Sub-scene `A` | Matches the exact geographic footprint ($18.26^\circ\text{N}, 73.34^\circ\text{E}$) |
| **Acquisition Season** | Post-monsoon winter / dry season (December – March) | Minimizes meteorological cloud likelihood and haze |
| **Catalog Cloud Threshold**| $< 5.0\%$ cloud cover | Guarantees clear terrain target across the full swath |
| **Processing Level** | Standard Georeferenced (`SSANSTUC00GTDA`) | Native projection in `EPSG:32643` (UTM Zone 43N) matching `stacked.tif` |

### Registration & Alignment Requirements Before Ingestion
Upon download:
1. Verify dimensions ($18,066 \times 16,223$) and coordinate reference system (`EPSG:32643`).
2. Verify sub-pixel or pixel-level geometric alignment with `data/interim/stacked_product/stacked.tif`. If sub-pixel shifts exist due to orthorectification terrain displacement, co-registration or spatial cropping must be applied and logged in the lineage manifest.
3. Track the new product independently in DVC under `data/raw/` with a separate `.dvc` tracking file.

---

## 3. Synthetic Cloud Generation Engine

The module [`src/cloud_removal/synthetic_clouds.py`](../src/cloud_removal/synthetic_clouds.py) implements a physics-inspired, stochastic transmission and scattering model.

### 3.1 Mathematical Formulation
Natural cloud formations exhibit continuous, non-uniform spatial opacity: dense, saturated cloud centers feathered by semi-transparent cirrus and haze borders.

1. **Multi-Scale Cloud Opacity Field ($\alpha$)**:
   Stochastic fields are synthesized across two spatial frequency scales:
   $$\Phi(x, y) = G_{\sigma_{\text{coarse}}}(N_1) + w_{\text{fine}} \cdot G_{\sigma_{\text{fine}}}(N_2)$$
   where $N_1, N_2 \sim \mathcal{N}(0, 1)$ are white Gaussian noise arrays, $G_\sigma$ denotes a 2D Gaussian filter, $\sigma_{\text{coarse}} = 0.18 \times \min(H, W)$, $\sigma_{\text{fine}} = 0.05 \times \min(H, W)$, and $w_{\text{fine}} = 0.25$.

2. **Coverage Quantile Thresholding**:
   To enforce exact target cloud coverage $c \in [0.0, 1.0]$, the normalized field $\tilde{\Phi} \in [0, 1]$ is thresholded at quantile $\tau = Q_{1 - c}(\tilde{\Phi})$. The opacity map is smoothly scaled:
   $$\alpha(x, y) = \begin{cases} 0 & \text{if } \tilde{\Phi}(x, y) \le \tau \\ \alpha_{\text{min}} + \frac{\tilde{\Phi}(x, y) - \tau}{1 - \tau} (\alpha_{\text{max}} - \alpha_{\text{min}}) & \text{if } \tilde{\Phi}(x, y) > \tau \end{cases}$$
   where $\alpha_{\text{min}} = 0.20$ represents thin haze, and $\alpha_{\text{max}} = 1.00$ represents opaque cloud cores.

3. **Atmospheric Radiative Transmission & Scattering**:
   The cloudy pixel value $I_{\text{cloudy}}$ for channel $b$ is computed via:
   $$I_{\text{cloudy}}(b, x, y) = (1 - \alpha(x, y)) \cdot I_{\text{clear}}(b, x, y) + \alpha(x, y) \cdot C(b, x, y)$$
   where $C(b, x, y) \in [\text{DN}_{\text{min}}, \text{DN}_{\text{max}}]$ simulates top-of-cloud scattered solar radiance (defaulting to $[700, 1023]$ for 10-bit dynamic range).

4. **Data Type Preservation**:
   Calculations are performed in `float32`. Output digital numbers are clipped to $[0, 65535]$ and rounded back to `uint16`, strictly preserving dimensions, channel ordering (`BAND2`, `BAND3`, `BAND4`), and array memory layout.

### 3.2 Known Limitations of Synthetic Clouds
- **Absence of 3D Cloud Shadows**: Real cloud decks cast directional surface shadows based on solar azimuth ($74.76^\circ$) and elevation. This 2D overlay does not currently model shadow casting.
- **Spectral Uniformity**: Real atmospheric scattering exhibits wavelength dependence (Rayleigh scattering $\propto \lambda^{-4}$ and Mie scattering). The simplified model assumes uniform high reflectance across VIS/NIR.
- **Microphysics & BRDF**: Simplified Gaussian synthesis does not model droplet condensation, turbulence thermodynamics, or anisotropic surface reflectance.
- **Evaluation Constraint**: Performance evaluated on synthetic clouds **must never be claimed** as real-world cloud removal accuracy.

---

## 4. Patch Extraction & Geographic Splitting Strategy

The module [`src/cloud_removal/patch_dataset.py`](../src/cloud_removal/patch_dataset.py) manages tiled extraction and spatial partitioning.

### 4.1 Patch Specifications
- **Patch Dimension**: Default $256 \times 256$ pixels (covering $1.28\text{ km} \times 1.28\text{ km}$ at $5\text{ m}$ GSD).
- **Stride**: Configurable (default $256$, producing non-overlapping tiles).
- **Collar Rejection Threshold**: `min_valid_ratio = 0.95`. Any candidate tile where valid scene pixels comprise $< 95\%$ of the window area is rejected. This guarantees that background collar padding ($DN = 0$) does not contaminate neural network loss functions or training batches.

### 4.2 Geographic Spatial Splitting (Zero-Leakage Policy)
Satellite imagery exhibits high spatial autocorrelation: adjacent pixels share terrain, land cover, and atmospheric characteristics. Randomly distributing adjacent patches between train and validation sets introduces severe data leakage.

To ensure rigorous evaluation:
- The coordinate extent is partitioned along the **spatial $Y$-axis (latitude)** into contiguous regional zones before extraction:
  - **Training Zone**: Top $70\%$ of the spatial extent ($Y \in [0, 0.70 \times H)$)
  - **Validation Zone**: Middle $15\%$ of the spatial extent ($Y \in [0.70 \times H, 0.85 \times H)$)
  - **Test Zone**: Bottom $15\%$ of the spatial extent ($Y \in [0.85 \times H, H]$)
- Each patch is assigned strictly according to its spatial center coordinates.
- **Zero Spatial Overlap**: Train, validation, and test patches come from physically distinct geographic regions. No patch identifiers or spatial extents cross split boundaries.

```
+-------------------------------------------------------+  Row 0
|                                                       |
|                     TRAINING ZONE                     |
|                         (70%)                         |
|                                                       |
+-------------------------------------------------------+  Row ~11,356
|                   VALIDATION ZONE                     |
|                         (15%)                         |
+-------------------------------------------------------+  Row ~13,789
|                      TEST ZONE                        |
|                         (15%)                         |
+-------------------------------------------------------+  Row 16,223
```

### 4.3 PyTorch Dataset Wrapper (`LISS4PatchDataset`)
- Implements standard `torch.utils.data.Dataset`.
- Converts raw `uint16` patches to `torch.FloatTensor` scaled to $[0.0, 1.0]$ using nominal 10-bit dynamic range ($1023.0$).
- Yields a structured sample dictionary:
  ```python
  {
      "clear": torch.FloatTensor,       # Shape: (3, 256, 256)
      "cloudy": torch.FloatTensor,      # Shape: (3, 256, 256)
      "cloud_mask": torch.FloatTensor,  # Shape: (1, 256, 256)
      "valid_mask": torch.FloatTensor,  # Shape: (1, 256, 256)
      "patch_id": str,                  # e.g. "patch_r02048_c04096"
      "split": str                      # "train" | "val" | "test"
  }
  ```

---

## 5. Dataset Manifest Schema (Specification for Unblocking)

When a verified clear reference raster is integrated, the dataset pipeline will generate `data/processed/patches/dataset_manifest.json` conforming to this schema:

```json
{
  "pipeline_stage": "synthetic_patch_dataset",
  "provenance": {
    "clear_reference_product": "data/raw/<CLEAR_PRODUCT_DIR>",
    "clear_reference_sha256": "...",
    "stacked_cloudy_product": "data/interim/stacked_product/stacked.tif",
    "validity_mask": "data/interim/stacked_product/valid_mask.tif"
  },
  "parameters": {
    "patch_size": 256,
    "stride": 256,
    "min_valid_ratio": 0.95,
    "cloud_generation": {
      "min_coverage": 0.15,
      "max_coverage": 0.50,
      "min_opacity": 0.20,
      "max_opacity": 1.00,
      "seed": 42
    },
    "spatial_split": {
      "policy": "contiguous_spatial_y",
      "train_ratio": 0.70,
      "val_ratio": 0.15,
      "test_ratio": 0.15
    }
  },
  "split_statistics": {
    "total_evaluated_windows": 0,
    "total_accepted_patches": 0,
    "total_rejected_collar_patches": 0,
    "train_patch_count": 0,
    "val_patch_count": 0,
    "test_patch_count": 0
  }
}
```

---

## 6. Verification & Automated Testing

All Phase 5 utilities are thoroughly validated by automated tests using lightweight in-memory synthetic arrays:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_clouds.py tests/test_patch_dataset.py
```

Test coverage includes:
- **Seed Determinism**: Confirms identical random seeds produce bit-for-bit identical opacity masks.
- **Seed Divergence**: Confirms differing seeds generate distinct cloud configurations.
- **Range & Shape Invariants**: Confirms opacity values stay strictly in $[0, \text{max\_opacity}]$ and dimensions match inputs.
- **Channel & Layout Fidelity**: Confirms $(C, H, W)$, $(H, W, C)$, and 2D arrays maintain layout and data type without clipping or overflow.
- **Collar Rejection**: Confirms boundary and collar tiles below threshold are rejected.
- **Spatial Separation**: Confirms train, validation, and test splits contain zero overlapping patch IDs and conform to spatial regional bounds.
- **PyTorch Dataset Compatibility**: Confirms tensors are correctly instantiated and normalized.
