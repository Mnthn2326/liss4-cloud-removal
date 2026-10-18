# DVC Data Versioning and Dataset Management

## 1. Overview
This document defines the data versioning architecture established for the **LISS4 Cloud Removal** project in Phase 3. 

The goal of this setup is to ensure complete dataset reproducibility, tracking satellite imagery independently of Git while maintaining strict lineage between code revisions and data versions.

---

## 2. DVC Configuration

- **DVC Version**: `3.67.1`
- **Configuration File**: [`.dvc/config`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/.dvc/config)
- **Cache Linking Strategy**: `hardlink,symlink,copy`
  - On the local NTFS volume (`E:`), DVC links files between `.dvc/cache` and the workspace via hard links.
  - This avoids duplicating the 1.76 GB of raster data on disk while preserving source file immutability.
- **Ignore Rules**: [`.dvcignore`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/.dvcignore) and [`.gitignore`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/.gitignore) ensure that raw imagery is excluded from Git while `.dvc` pointers and metadata manifests are tracked.

---

## 3. Tracked Dataset Specification

The extracted Resourcesat-2A LISS-IV product is tracked as a single cohesive dataset unit:

- **Product Identifier**: `2643035131`
- **OTS Product ID**: `RAF03JUN2026049233009500060SSANSTUC00GTDA`
- **Source Directory**: `data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA/`
- **Total Dataset Size**: `1,759,106,380 bytes` (~1.64 GiB uncompressed)
- **Constituent Files (8)**:
  - `BAND2.tif`: Green band raster ($586,299,606\text{ bytes}$)
  - `BAND3.tif`: Red band raster ($586,299,606\text{ bytes}$)
  - `BAND4.tif`: NIR band raster ($586,299,606\text{ bytes}$)
  - `BAND_META.txt`: Acquisition, sensor angles, and radiometric calibration metadata
  - `ACC_REP.txt`: Geometric accuracy report
  - `RAF03JUN2026049233009500060SSANSTUC00GTDA.meta`: Detailed product metadata
  - `RAF03JUN2026049233009500060SSANSTUC00GTDA.jpg`: Browse thumbnail
  - `RAF03JUN2026049233009500060SSANSTUC00GTDA.jpg.aux.xml`: Browse statistics
- **DVC Pointer File**: [`data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA.dvc`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA.dvc)
  - Directory Hash: `021d8ec95da816d7ee19b07d0880c273.dir`
  - Committed to Git: **Yes**
- **Dataset Manifest**: [`data/raw/manifest.json`](file:///e:/Third%20Year%20%2826-27%29%20%28Semester%201%29/liss4-cloud-removal/data/raw/manifest.json)
  - Contains complete file sizes, SHA-256 hashes, coordinate systems, and dimensions.
  - Committed to Git: **Yes**

---

## 4. DVC Remote Configuration

- **Remote Name**: `local_storage` (Set as default remote)
- **Storage Type**: Local Filesystem Storage
- **Remote Location**: `E:\dvc-storage\liss4-cloud-removal`
- **Rationale**:
  - Located completely outside the Git repository and virtual environment.
  - Sited on a drive volume (`E:`) with ample free space (> 360 GB).
  - Persists independently of workspace checkouts or branch switching.

---

## 5. Workflows & Command Reference (PowerShell)

All commands are executed inside the project root using the project virtual environment.

### 5.1 Checking Dataset Status
To inspect the status of the tracked dataset relative to the local cache:
```powershell
.\.venv\Scripts\dvc.exe data status
```
*Expected Output:* `No changes.` (indicates the workspace data matches the `.dvc` specification).

To compare the local cache against the remote storage:
```powershell
.\.venv\Scripts\dvc.exe status data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA.dvc -c
```

### 5.2 Pushing Dataset to Remote Storage
To upload the tracked dataset from the local cache to the configured remote:
```powershell
.\.venv\Scripts\dvc.exe push data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA.dvc
```
*Result:* Pushes all 9 objects (the `.dir` manifest and all 8 constituent files) to `E:\dvc-storage\liss4-cloud-removal\files\md5\`.

### 5.3 Pulling / Restoring Dataset
If the local workspace data is missing or needs to be synchronized:
```powershell
.\.venv\Scripts\dvc.exe pull data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA.dvc
```

### 5.4 Exporting / Retrieving Data to an External Path
To retrieve the dataset from the repository into another directory without checking out:
```powershell
.\.venv\Scripts\dvc.exe get . data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA -o E:\destination_path
```

---

## 6. Git vs. DVC Responsibility Separation

| Concern | Managed by Git | Managed by DVC |
| :--- | :--- | :--- |
| **Source Code & Scripts** | `src/`, `tests/` | Ignored |
| **Configuration** | `pyproject.toml`, `.dvc/config` | Configures storage endpoints |
| **Data Pointers** | `*.dvc`, `dvc.yaml`, `dvc.lock` | Generates hashes & verifies integrity |
| **Catalog Manifests** | `data/raw/manifest.json` | Validated against stored files |
| **Binary Imagery & Rasters** | Strictly ignored via `.gitignore` | Versioned in `.dvc/cache` & remotes |
| **Repository Size** | Stays small (< 5 MB) | Handles gigabyte-scale satellite rasters |

---

## 7. Restoration Verification & Checksum Evidence

To verify dataset restorability without altering the primary workspace:
1. The dataset was retrieved from the remote into an isolated directory `E:\dvc_restore_test` using `dvc get`.
2. All 8 restored files were verified against the original files using SHA-256 cryptographic hashes:

| File Name | Size (Bytes) | SHA-256 Checksum | Match Status |
| :--- | :--- | :--- | :--- |
| `ACC_REP.txt` | 713 | `ad61a7f98c80daedc7c09235a57493285de35309f5867de8b43209fac7b5da8c` | **MATCH** |
| `BAND2.tif` | 586,299,606 | `4af49a3c102451e350eee8b38400f68f62f536faf9b4c93895c7bf804f5042bb` | **MATCH** |
| `BAND3.tif` | 586,299,606 | `ed01776cccb8a742d8c7217bb5229303df971582ce793212e6312086f748b6c5` | **MATCH** |
| `BAND4.tif` | 586,299,606 | `d451632c77dfe1a1df19eca237e55d22532a03096880a0dc4e4096b917d2db9b` | **MATCH** |
| `BAND_META.txt` | 2,616 | `8b8c6da9175133185f94d363f3a3aac8da919f186fcca95fed7173a6db54fe78` | **MATCH** |
| `RAF03JUN...jpg` | 200,101 | `bc77ef167ebf576d1f24ff00a8eb3c8bb2ba48c3d6e6eb9510cc4bd0339c62aa` | **MATCH** |
| `RAF03JUN...aux.xml` | 1,481 | `5bb4be40a3eebdc1bd588fc6e57fa7e67bf60cce2bd119ca551b897902c90c5c` | **MATCH** |
| `RAF03JUN...meta` | 2,651 | `90fc5647e614a9935ce408248bc683df9b5ac55ff28ed7560809e8bd57a8cf44` | **MATCH** |

**Result**: 100% byte-for-byte fidelity confirmed. The test restore directory was subsequently removed.

---

## 8. Known Limitations of Local Remote

- **Single-Machine Bound**: The local remote (`E:\dvc-storage\liss4-cloud-removal`) is accessible only from this workstation.
- **Collaborative Work / Cloud GPU Execution**: When training models in cloud environments (e.g., Google Colab or GCP Vertex AI), an external cloud bucket remote (e.g., Google Cloud Storage or AWS S3) must be configured:
  ```powershell
  dvc remote add gcs_remote gs://<bucket-name>/liss4-cloud-removal
  dvc remote default gcs_remote
  ```
- **Read-Only Hardlink Protection**: Because DVC links cache files with hardlinks, modifying tracked files in-place is prevented by file permissions to protect cache integrity.
