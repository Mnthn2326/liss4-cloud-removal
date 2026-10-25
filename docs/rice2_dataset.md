# RICE2 Dataset Specification and Verification

## 1. Overview & Dataset Provenance

The **RICE2** (Remote Sensing Image Cloud Removing) benchmark dataset is an established academic dataset curated by Lin et al. (Wuhan University) designed specifically for supervised evaluation of optical cloud removal algorithms. 

The dataset contains natural, non-synthetic paired optical observations extracted from **Landsat-8 OLI** and high-resolution Google Earth optical imagery across diverse terrain types, urban centers, agricultural regions, coastal areas, and mountainous landscapes.

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | RICE2 (Remote Sensing Image Cloud Removing - Set 2) |
| **Primary Reference** | Lin et al., *"RICE: A Remote Sensing Image Cloud Removing Dataset"*, 2019 |
| **Sensor Type** | Landsat-style Optical RGB (Red, Green, Blue) |
| **Total Triplets** | 736 paired scenes (2,208 image files) |
| **Spatial Dimensions** | $512 \times 512$ pixels per tile |
| **Spectral Channels** | 3 channels (RGB: Red, Green, Blue) |
| **Pixel Bit-Depth** | 8-bit unsigned integer ($[0, 255]$) normalized to float32 ($[0.0, 1.0]$) |
| **Storage Footprint** | ~262.7 MB uncompressed |
| **DVC Tracking** | `data/raw/RICE2.dvc` (Directory hash: `5c4149ba58c148e678bfbb70868f00db.dir`) |

---

## 2. Directory Hierarchy & Triplet Pairing

The dataset is organized into three strictly synchronized subdirectories within `data/raw/RICE2/`:

```text
data/raw/RICE2/
├── cloud/           # 736 cloudy observation tiles (0.png to 735.png)
├── label/           # 736 clear ground-truth reference tiles (0.png to 735.png)
└── mask/            # 736 binary cloud mask tiles (0.png to 735.png)
```

Each observation consists of an immutable triplet referenced by an integer `sample_id` ($0 \le \text{sample\_id} \le 735$):
1. **Cloudy Input Image (`cloud/<id>.png`)**: Optical observation obscured by heterogeneous cloud formations (thin cirrus, dense cumulus, and associated cloud shadows).
2. **Clear Reference Image (`label/<id>.png`)**: Cloud-free optical observation of the identical spatial footprint captured under clear atmospheric conditions at an adjacent temporal revisit.
3. **Cloud Mask (`mask/<id>.png`)**: Binary cloud boundary mask where:
   - $\text{DN} = 255$: Cloud-obscured pixel ($\alpha = 1.0$).
   - $\text{DN} = 0$: Clear, unobscured surface pixel ($\alpha = 0.0$).

---

## 3. Partitioning Strategy & Split Disjointness

To guarantee rigorous evaluation without data leakage, the 736 samples are partitioned into deterministic `train`, `val`, and `test` splits using a fixed pseudorandom seed ($42$):

| Split | Sample Count | Ratio | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | 471 samples | 64.0% | Model parameter optimization via gradient descent |
| **Val** | 117 samples | 15.9% | Checkpoint selection, early stopping, and hyperparameter tuning |
| **Test** | 148 samples | 20.1% | **Held-out, untouched benchmark evaluation only** |
| **Total** | **736 samples** | **100.0%** | Complete RICE2 benchmark corpus |

> [!IMPORTANT]
> **Strict Evaluation Integrity**:
> The 148 test samples are permanently sealed during training. No test sample is ever seen by the network during forward training passes, learning rate scheduling, or early-stopping trigger checks. The test set was evaluated exactly once to generate the final performance metrics in `reports/rice2_benchmark_report.md`.
>
> All splits are strictly pairwise disjoint:
> $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 4. Manifest Schema & Checksums

The full dataset catalog is indexed in [`data/manifests/rice2_manifest.json`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/data/manifests/rice2_manifest.json). Each sample record stores relative file paths and cryptographic SHA-256 checksums:

```json
{
  "sample_id": "0",
  "split": "train",
  "cloud": "data/raw/RICE2/cloud/0.png",
  "label": "data/raw/RICE2/label/0.png",
  "mask": "data/raw/RICE2/mask/0.png",
  "sha256": {
    "cloud": "218bc16db90e2b3be8952ae99026da68ee8a58f4a3e2a05cf6cb5b161df1ff84",
    "label": "39234851dfefc82bbbe5aa48259e8f6e729aa181515222fb20f18aa006c09b55",
    "mask": "f03bda636ee2268aeec936d5b03513b190f845a706da620c3547eb19c28905fe"
  }
}
```

Manifest generation is automated by `src/cloud_removal/rice2.py`:
```bash
python -m cloud_removal.rice2 --data-dir data/raw/RICE2 --output data/manifests/rice2_manifest.json
```

---

## 5. Radiometric Normalization & Preprocessing

1. **Range Normalization**:
   Raw 8-bit unsigned integer pixel values $p \in [0, 255]$ are converted to 32-bit floating point and mapped to physical surface reflectance:
   $$x = \frac{p}{255.0} \in [0.0, 1.0]$$
2. **Tensor Layout**:
   Images are loaded as PyTorch tensors shaped $[C, H, W] = [3, 512, 512]$ with channel ordering corresponding to Red, Green, and Blue.
3. **Synchronous Data Augmentation**:
   During training (`split="train"`), geometric transformations (random horizontal and vertical flips) are applied synchronously across all three triplet tensors using identical random seeds to preserve spatial alignment. Augmentation is disabled for validation and testing.

---

## 6. Pre-flight Data Integrity Validation

The loader enforces defensive integrity checks via `_validate_image_triplet()` in `src/cloud_removal/rice2.py`:
- **File Existence**: Verifies all three paths exist and contain non-zero byte size.
- **Image Decodability**: Verifies PIL can open and parse image headers without corruption.
- **Channel Format**: Verifies cloudy and label images are strictly 3-channel RGB.
- **Spatial Alignment**: Validates that all three files share identical dimensions ($512 \times 512$). Mismatched geometries raise `ValueError` immediately.
