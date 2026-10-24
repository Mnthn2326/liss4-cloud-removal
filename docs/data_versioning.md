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
  - `BAND2.tif`: Source band 2 raster ($586,299,606\text{ bytes}$)
  - `BAND3.tif`: Source band 3 raster ($586,299,606\text{ bytes}$)
  - `BAND4.tif`: Source band 4 raster ($586,299,606\text{ bytes}$)

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

## 8. Multi-Cloud Remote Storage Architecture & Authentication

### 8.1 Zero-Secrets Architecture
To support reproducible collaboration across research teams and automated CI/CD runners without risking credential leakage:
- **No secrets in repository**: Credentials, access keys, tokens, and storage passwords are **never** committed to Git, `.dvc/config`, or tracked files.
- **Environment-driven authentication**: Cloud authentication relies strictly on standard cloud provider environment variables or instance metadata services (IAM roles, Workload Identity, Managed Identities).
- **Endpoint modularity**: Storage endpoints (S3 bucket URLs, GCS buckets, Azure containers) are declared in `.dvc/config` as uncredentialed endpoint templates.

### 8.2 Supported Cloud Providers & Credentials Setup

#### 1. Amazon Web Services (AWS S3)
Configured remote: `s3_remote` (`s3://liss4-cloud-removal-dvc`)
```powershell
# Set standard AWS credentials in environment
$env:AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_ID"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY"
$env:AWS_DEFAULT_REGION = "ap-south-1"

# Switch default remote or push directly
.\.venv\Scripts\dvc.exe push -r s3_remote
```

#### 2. MinIO (S3-Compatible Self-Hosted Object Storage)
Configured remote: `minio_remote` (`s3://liss4-dvc-bucket` with `endpointurl = http://localhost:9000`)
```powershell
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
$env:AWS_ENDPOINT_URL = "http://localhost:9000"

.\.venv\Scripts\dvc.exe push -r minio_remote
```

#### 3. Google Cloud Storage (GCS)
Configured remote: `gcs_remote` (`gs://liss4-cloud-removal-dvc`)
```powershell
# Service account key path
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account-key.json"

.\.venv\Scripts\dvc.exe push -r gcs_remote
```

#### 4. Microsoft Azure Blob Storage
Configured remote: `azure_remote` (`azure://liss4-cloud-removal-dvc`)
```powershell
$env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"

.\.venv\Scripts\dvc.exe push -r azure_remote
```

### 8.3 Remote Verification & Status Monitoring
To audit synchronization between local cache and configured remotes:
```powershell
# Check status against active default remote (local_storage)
.\.venv\Scripts\dvc.exe status -c

# Check status against specific cloud remote
.\.venv\Scripts\dvc.exe status -r s3_remote

# Test push and pull operations
.\.venv\Scripts\dvc.exe push -r local_storage
.\.venv\Scripts\dvc.exe pull -r local_storage
```

---

## 9. Phase 2: Authentic Cloud-Free LISS-IV Reference Scene Acquisition

### 9.1 Objective & Criteria
To evaluate cloud-removal transfer performance on authentic Resourcesat-2A LISS-IV imagery without synthetic cloud simulation, a genuine cloud-free reference scene must be acquired from ISRO's official Bhoonidhi data distribution portal.

| Parameter | Selection Specification |
| :--- | :--- |
| **Data Source** | ISRO Bhoonidhi Portal ([https://bhoonidhi.nrsc.gov.in/](https://bhoonidhi.nrsc.gov.in/)) |
| **Satellite** | Resourcesat-2A (`IRS-R2A`) |
| **Sensor** | LISS-IV (`L4FX`, Mono / Multispectral mode) |
| **Scene Footprint** | Path `095`, Row `060`, Subscene `A` (Western Maharashtra / Pune corridor) |
| **Spatial Overlap** | Co-located with cloudy scene `RAF03JUN2026049233009500060SSANSTUC00GTDA` |
| **Cloud Cover Requirement** | $\le 2.0\%$ at scene and evaluation patch level |
| **Optimal Temporal Window** | Post-monsoon / Winter dry season (November 2025 – February 2026) |
| **Processing Level** | Standard Georeferenced (`STD`, Map projection UTM Zone 43N, WGS 84) |
| **Required Files** | `BAND2.tif` (Green), `BAND3.tif` (Red), `BAND4.tif` (NIR), `BAND_META.txt`, `.meta` |

### 9.2 Official Acquisition Protocol (Bhoonidhi)
1. **User Authentication**: Log in to [https://bhoonidhi.nrsc.gov.in/](https://bhoonidhi.nrsc.gov.in/) with registered user credentials. Access to raw Resourcesat-2A standard products requires manual OTP authentication; direct automated downloading without user session cookies is prevented by ISRO security policies.
2. **Search Query**:
   - Platform: `RESOURCESAT-2A`
   - Sensor: `LISS-4`
   - Path / Row: `095` / `060`
   - Subscene: `A`
   - Date Range: `01-NOV-2025` to `28-FEB-2026`
   - Cloud Cover Filter: `0 - 2%`
3. **Cart & Download**: Add the qualified clear product to the cart, submit the order request, and download the full ZIP product archive to the local workstation.

### 9.3 Ingestion & Verification Workflow
Once downloaded, run the verified automated ingestion utility:
```powershell
# Validate and ingest downloaded scene ZIP into data/raw/
.\.venv\Scripts\python.exe -m cloud_removal.ingest_liss4 --source path/to/downloaded_scene.zip --target-dir data/raw --max-cloud-percent 2.0
```

The tool:
- Confirms `SatID` is `IRS-R2A` and `Sensor` is `L4FX`.
- Validates that `CloudPercent` $\le 2.0\%$.
- Confirms all 3 spectral bands (`BAND2.tif`, `BAND3.tif`, `BAND4.tif`) exist with identical 10-bit integer format and UTM Zone 43N georeferencing.
- Computes SHA-256 hashes for all files and generates an authentic manifest.
- Outputs exact DVC tracking commands.

### 9.4 DVC Versioning Commands
Track the ingested scene following established repository standards:
```powershell
# 1. Add product to DVC tracking
.\.venv\Scripts\dvc.exe add data/raw/<OTS_PRODUCT_ID>

# 2. Stage DVC pointer and manifest in Git
git add data/raw/<OTS_PRODUCT_ID>.dvc data/raw/<OTS_PRODUCT_ID>/manifest.json

# 3. Commit metadata to Git
git commit -m "feat(data): version authentic clear Resourcesat-2A LISS-IV reference scene"

# 4. Push raster blobs to local storage remote
.\.venv\Scripts\dvc.exe push data/raw/<OTS_PRODUCT_ID>.dvc
```

### 9.5 Current Acquisition Status
- **Status:** **Blocked by missing external authenticated data from Bhoonidhi.**
- **Rationale:** Downloading authentic Level-1/Level-2 satellite scenes from Bhoonidhi requires an active authenticated session and OTP authorization from a registered Indian citizen/researcher account. No synthetic or substitute data has been fabricated. The ingestion pipeline (`src/cloud_removal/ingest_liss4.py`) is fully implemented, tested, and ready for immediate ingestion once the archive is placed on disk.

