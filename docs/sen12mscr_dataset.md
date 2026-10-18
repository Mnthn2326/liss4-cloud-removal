# SEN12MS-CR Dataset: Acquisition, Preprocessing & Data Loader

## 1. Overview & Dataset Provenance

**SEN12MS-CR** is a global, multi-modal, all-season benchmark dataset curated for supervised cloud removal in spaceborne Earth observation. It provides co-registered triplets of space-borne radar (Sentinel-1 dual-pol SAR) alongside cloud-covered and cloud-free multi-spectral optical observations (Sentinel-2 Level-1C/Level-2A).

- **Official Project Page**: [https://patricktum.github.io/cloud_removal/sen12mscr/](https://patricktum.github.io/cloud_removal/sen12mscr/)
- **Repository Hosting**: MediaTUM (Technical University of Munich) repository ID `1554803` (`dataserv.ub.tum.de`).
- **Citation & Authors**:
  > P. Ebel, A. Meraner, M. Schmitt and X. X. Zhu, *"Multisensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery,"* IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS), 2020. DOI: [10.1109/TGRS.2020.3024744](https://doi.org/10.1109/TGRS.2020.3024744).
- **License & Terms of Use**:
  - Openly licensed under **CC-BY**.
  - Contains modified Copernicus Sentinel data (2017–2018).

---

## 2. Acquired Subset & Storage Footprint

Rather than downloading the entire 291.9 GiB collection, a targeted, reproducible subset of **719 paired optical patches** was acquired using custom streaming decompression directly from the MediaTUM FTP repository.

### Subset Specifications
- **Region of Interest (ROI)**: `ROIs2017_winter_102` (ROI `102` from meteorological winter).
- **Official Split**: **Official Training Split (Split 1)**.
- **Number of Paired Patches**: **719** ($1,438$ GeoTIFFs total: 719 cloudy optical + 719 clear optical).
- **Storage Footprint**:
  - Total Disk Footprint: $2,454,017,838\text{ bytes}$ (~$2.29\text{ GiB}$).
  - File Size per Patch: Exactly $1,706,432\text{ bytes}$ uncompressed ($256 \times 256 \times 13 \text{ bands} \times 2\text{ bytes/pixel}$).
  - Network Stream Consumed: ~800 MB total compressed payload.
- **Directory Structure**:
  ```text
  data/raw/sen12mscr/
  ├── manifest.json
  ├── ROIs2017_winter_s2/
  │   └── s2_102/
  │       ├── ROIs2017_winter_s2_102_p40.tif
  │       └── ... (719 clear GeoTIFFs)
  └── ROIs2017_winter_s2_cloudy/
      └── s2_cloudy_102/
          ├── ROIs2017_winter_s2_cloudy_102_p40.tif
          └── ... (719 cloudy GeoTIFFs)
  ```

---

## 3. Band Selection & Physical Normalization

### 3.1 Sentinel-2 Spectral Channels
Each raw SEN12MS-CR optical patch contains 13 spectral bands stored as `uint16` GeoTIFFs ($256 \times 256$ pixels, $10\text{ m}$ GSD for visual bands):

| GeoTIFF Band (1-indexed) | Channel Name | Central Wavelength | Spatial Resolution | Selected for 3-Channel RGB |
| :---: | :---: | :---: | :---: | :---: |
| Band 1 | `B01` (Aerosols) | $443\text{ nm}$ | $60\text{ m}$ | No |
| **Band 2** | `B02` (Blue) | $490\text{ nm}$ | $10\text{ m}$ | **Yes (Channel 3 / Blue)** |
| **Band 3** | `B03` (Green) | $560\text{ nm}$ | $10\text{ m}$ | **Yes (Channel 2 / Green)** |
| **Band 4** | `B04` (Red) | $665\text{ nm}$ | $10\text{ m}$ | **Yes (Channel 1 / Red)** |
| Band 5 | `B05` (Red Edge 1) | $705\text{ nm}$ | $20\text{ m}$ | No |
| Band 6 | `B06` (Red Edge 2) | $740\text{ nm}$ | $20\text{ m}$ | No |
| Band 7 | `B07` (Red Edge 3) | $783\text{ nm}$ | $20\text{ m}$ | No |
| Band 8 | `B08` (NIR 1) | $842\text{ nm}$ | $10\text{ m}$ | No |
| Band 9 | `B08A` (Narrow NIR) | $865\text{ nm}$ | $20\text{ m}$ | No |
| Band 10 | `B09` (Water Vapor) | $940\text{ nm}$ | $60\text{ m}$ | No |
| Band 11 | `B10` (Cirrus) | $1375\text{ nm}$ | $60\text{ m}$ | No |
| Band 12 | `B11` (SWIR 1) | $1610\text{ nm}$ | $20\text{ m}$ | No |
| Band 13 | `B12` (SWIR 2) | $2190\text{ nm}$ | $20\text{ m}$ | No |

### 3.2 Distinction from LISS-IV Band Identifiers
> **Critical Distinction**: In LISS-IV (Resourcesat-2A), the channels are tracked as source bands `BAND2`, `BAND3`, and `BAND4`. In SEN12MS-CR (Sentinel-2), the channels are standard European Space Agency Copernicus bands, where true-color RGB corresponds to **Band 4 (Red), Band 3 (Green), and Band 2 (Blue)**. The data loader enforces `bands=(4, 3, 2)` to deliver standard RGB tensors.

### 3.3 Reflectance Scaling
In Sentinel-2 Level-1C/Level-2A products, surface reflectance values are scaled by a factor of $10,000$. Following the official SEN12MS-CR preprocessing specification:
$$\text{Reflectance} = \frac{\text{clip}(\text{DN}, 0, 10000)}{10000.0}$$
This projects digital numbers linearly into true surface reflectance $\in [0.0, 1.0]$. Dense cloud cores naturally saturate near $\text{DN} \ge 10,000$ (reflectance $1.0$), while clear land/water reflectance ranges from $0.02$ to $0.85$.

---

## 4. Official Splits & Split Partitioning Policy

The official SEN12MS-CR benchmark defines 3 disjoint splits distributed across all four seasons (`splits.csv`):
- **Official Train Split (`split = 1`)**: $107,143$ patches across 118 ROIs.
- **Official Validation Split (`split = 2`)**: $7,176$ patches across 10 ROIs.
- **Official Test Split (`split = 3`)**: $7,899$ patches across 10 ROIs (`ROIs1158_spring_106`, `ROIs1158_spring_123`, `ROIs1158_spring_140`, `ROIs1158_spring_31`, `ROIs1158_spring_44`, `ROIs1868_summer_119`, `ROIs1868_summer_73`, `ROIs1970_fall_139`, `ROIs2017_winter_108`, `ROIs2017_winter_63`).

### Zero-Leakage Policy for This Subset
- The acquired subset (ROI `102`, Winter) belongs **strictly to the official training split** (`split 1`).
- For local model prototyping, the 719 samples are deterministically partitioned into:
  - **Local Train**: $575$ samples ($80\%$)
  - **Local Validation**: $144$ samples ($20\%$)
- **Official Test Split Preserved**: None of the 10 official test ROIs were downloaded or accessed, preserving the official test benchmark completely unobserved for final external evaluation.

---

## 5. PyTorch Data Loader Usage

The module [`src/cloud_removal/sen12mscr.py`](../src/cloud_removal/sen12mscr.py) provides a high-level `torch.utils.data.Dataset` interface:

```python
from cloud_removal.sen12mscr import SEN12MSCRDataset, SEN12MSCRConfig
from torch.utils.data import DataLoader

# Load local train and val splits from the manifest
train_dataset = SEN12MSCRDataset.from_manifest("data/raw/sen12mscr/manifest.json", split="train")
val_dataset = SEN12MSCRDataset.from_manifest("data/raw/sen12mscr/manifest.json", split="val")

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
# Output: Train samples: 575, Val samples: 144

# Instantiate DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Fetch a batch
batch = next(iter(train_loader))
cloudy_tensors = batch["cloudy"]  # Shape: (16, 3, 256, 256), dtype: torch.float32, range: [0, 1]
clear_tensors = batch["clear"]    # Shape: (16, 3, 256, 256), dtype: torch.float32, range: [0, 1]
patch_ids = batch["patch_id"]     # List of strings e.g. ['p100', 'p101', ...]
```

---

## 6. DVC Data Versioning & Reproducibility

The acquired raw dataset is tracked by DVC independently of Git:
- **Tracking File**: `data/raw/sen12mscr.dvc` (tracked in Git; raw GeoTIFFs ignored).
- **Reproduction Command**:
  To reproduce or re-download the exact same 719-patch subset from MediaTUM:
  ```powershell
  .\.venv\Scripts\python.exe -m cloud_removal.acquire_sen12mscr --output-dir data/raw/sen12mscr --season winter --roi 102
  ```
- **Validation Command**:
  ```powershell
  .\.venv\Scripts\python.exe -m cloud_removal.sen12mscr --manifest data/raw/sen12mscr/manifest.json
  ```

---

## 7. Known Limitations
1. **Mono-Temporal Pairs**: In SEN12MS-CR, cloudy and clear optical observations are acquired on different dates (within a seasonal window). Slight seasonal vegetation changes or sun angle differences may exist between clear targets and cloudy inputs.
2. **Sentinel-2 vs LISS-IV Transferability**: SEN12MS-CR utilizes Sentinel-2 MSI data ($10\text{ m}$ GSD, 13 bands). The target Indian satellite LISS-IV operates at $5\text{ m}$ GSD across 3 bands. While the cloud-removal U-Net architecture trained on SEN12MS-CR learns realistic cloud-penetration features, fine-tuning or domain adaptation may be required when transferring back to LISS-IV.
