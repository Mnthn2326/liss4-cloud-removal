# Phase 7: U-Net Baseline Training with MLflow

## Overview

Phase 7 implements the first supervised deep learning model for optical satellite cloud removal in `liss4-cloud-removal`. A conventional **U-Net** architecture is trained on the paired Sentinel-2 optical imagery acquired in Phase 6 from the **SEN12MS-CR** dataset.

The training pipeline integrates **MLflow** experiment tracking to record hyperparameters, per-epoch loss and evaluation metrics (MAE and PSNR), and model checkpoint artifacts. A dedicated inference utility enables loading trained weights, evaluating unseen validation patches, and rendering side-by-side visual comparisons.

---

## Model Architecture

### Conventional U-Net Specification

The baseline model implements a standard 4-level encoder-decoder U-Net with skip connections:

```
Input (3 x 256 x 256)
  │
  ▼
[DoubleConv: 3 -> 32] ──────────── Skip 1 ────────────► [Up 4: 64 -> 32]
  │ (MaxPool 2x2)                                              │
  ▼                                                            ▼
[DoubleConv: 32 -> 64] ─────────── Skip 2 ────────────► [Up 3: 128 -> 64]
  │ (MaxPool 2x2)                                              │
  ▼                                                            ▼
[DoubleConv: 64 -> 128] ────────── Skip 3 ────────────► [Up 2: 256 -> 128]
  │ (MaxPool 2x2)                                              │
  ▼                                                            ▼
[DoubleConv: 128 -> 256] ───────── Skip 4 ────────────► [Up 1: 512 -> 256]
  │ (MaxPool 2x2)                                              │
  ▼                                                            │
[Bottleneck: 256 -> 512] ──────────────────────────────────────┘
                                                               │
                                                               ▼
                                                      [OutConv: 32 -> 3]
                                                               │
                                                               ▼
                                                      [Sigmoid Activation]
                                                               │
                                                               ▼
                                                    Output (3 x 256 x 256)
```

### Key Architectural Characteristics

1. **DoubleConv Blocks**: Two consecutive $3 \times 3$ 2D convolutions, each followed by Batch Normalization and ReLU activation (`padding=1`, preserving spatial dimensions).
2. **Encoder Downsampling**: 4 stages using $2 \times 2$ Max Pooling with stride 2.
3. **Decoder Upsampling**: 4 stages using $2 \times 2$ Transposed Convolutions (or configurable bilinear upsampling), followed by concatenation with skip features from the matching encoder stage and DoubleConv refinement.
4. **Output Projection & Bounding**: A final $1 \times 1$ convolution projects 32 feature channels to 3 output RGB channels. A terminal **Sigmoid** activation function bounds predicted surface reflectance strictly to $[0.0, 1.0]$, preventing unphysical negative reflectance or overflow values.
5. **Parameter Count**:
   - `base_channels = 32`: **7,763,107** trainable parameters (~7.76M).
   - `base_channels = 16`: **1,942,675** trainable parameters (~1.94M).

---

## Dataset and Preprocessing Pipeline

| Property | Value |
| :--- | :--- |
| **Source Dataset** | SEN12MS-CR (Official Training ROI 102, Winter) |
| **Total Available Pairs** | 719 cloudy and clear optical GeoTIFF pairs |
| **Local Training Split** | 575 pairs (80% deterministic split, seed=42) |
| **Local Validation Split** | 144 pairs (20% internal holdout, seed=42) |
| **Patch Dimensions** | $256 \times 256$ pixels |
| **Optical Channels** | Sentinel-2 bands 4, 3, 2 (RGB: Red, Green, Blue) |
| **Tensor Dimensions** | $[3, 256, 256]$ (Channel-first PyTorch float32) |
| **Reflectance Normalization** | Clipped to $[0, 10000]$ and scaled by $10000.0 \to [0.0, 1.0]$ |

> [!IMPORTANT]
> **Evaluation Boundary Disclosure**: The local validation set of 144 samples is an internal holdout split extracted from official training ROI 102. It serves to monitor generalization and prevent overfitting during development. It does **not** represent performance on the official SEN12MS-CR benchmark test ROIs, which have not been acquired or evaluated.

---

## Training Hyperparameters & Configuration

Default training parameters are declaratively defined in `configs/train_unet.yaml`:

```yaml
seed: 42
manifest_path: "data/raw/sen12mscr/manifest.json"
checkpoint_dir: "checkpoints"

mlflow:
  experiment_name: "sen12mscr-unet-baseline"
  tracking_uri: "mlruns"

model:
  in_channels: 3
  out_channels: 3
  base_channels: 32
  bilinear: false
  use_sigmoid: true

training:
  epochs: 10
  batch_size: 8
  learning_rate: 0.0002
  weight_decay: 0.00001
  optimizer: "adamw"
  loss: "l1"
  num_workers: 0
  device: "auto"
```

- **Objective Loss**: L1 / Mean Absolute Error ($\text{MAE} = \frac{1}{N} \sum |y_{\text{pred}} - y_{\text{true}}|$).
- **Optimizer**: AdamW with learning rate $2 \times 10^{-4}$ and weight decay $1 \times 10^{-5}$.
- **Evaluation Metrics**:
  - **MAE**: Mean Absolute Error on normalized surface reflectance $[0.0, 1.0]$.
  - **PSNR**: Peak Signal-to-Noise Ratio calibrated for $\text{data\_range}=1.0$, capped at $100.0\text{ dB}$ for exact equality:
    $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0^2}{\text{MSE} + \epsilon}\right)$$

---

## MLflow Experiment Tracking

The training engine (`src/cloud_removal/train.py`) logs full lineage to the local file store at `mlruns/`:

- **Experiment Name**: `sen12mscr-unet-baseline`
- **Logged Parameters**:
  - `model_type`, `in_channels`, `out_channels`, `base_channels`, `bilinear`, `use_sigmoid`
  - `trainable_parameters`
  - `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `optimizer`, `loss_function`
  - `num_train_samples`, `num_val_samples`, `seed`, `device`, `smoke_test`
- **Logged Per-Epoch Metrics**:
  - `train_loss`, `train_mae`, `val_loss`, `val_mae`, `val_psnr`, `epoch_duration_sec`
- **Artifacts**:
  - Best model checkpoint (`checkpoints/best_unet.pt`) automatically logged to MLflow artifacts store upon run completion.

To view the interactive MLflow UI:
```bash
.venv\Scripts\mlflow ui
# Open http://127.0.0.1:5000 in your browser
```

---

---

## Verification & Quality Audit (Phase 7.1)

### 1. Diagnosis of the Initial Smoke Run Quality Concern
During initial smoke testing with default zero-bias initialization (`nn.Conv2d` default `bias=0`), the network outputs $\text{Sigmoid}(0) \approx 0.50$ (observed: $0.535$). However, natural Sentinel-2 optical surface reflectance across SEN12MS-CR averages **0.1073** (with individual patches between $0.096$ and $0.116$).
Because of this large DC offset, an untrained 1-epoch model outputting $\approx 0.535$ exhibited an initial MAE of $|0.535 - 0.107| = 0.428 \approx 0.435$, whereas raw cloudy input vs clear ground truth averages only **0.0754** MAE across the 144 validation patches.

### 2. No-Training Baseline on Full Local Validation Subset (144 samples)
- **Macro-average MAE**: `0.07536`
- **Macro-average PSNR**: `25.95 dB`
- **Micro-average MAE**: `0.07536`
- **Micro-average PSNR**: `18.92 dB`
- **Per-patch MAE Range**: `[0.00579, 0.28678]` (Median: `0.04383`)
- **Per-patch PSNR Range**: `[10.46 dB, 42.01 dB]` (Median: `26.33 dB`)

### 3. Controlled Overfit Diagnostic
- Training on 8 fixed patches for 35 epochs confirmed the model overfits properly:
  - With default initialization, loss steadily decreased from $0.440$ to $0.319$.
  - With calibrated prior reflectance bias ($b = \text{logit}(0.107) = -2.12$), Epoch 1 MAE dropped immediately to `0.02425` (`31.38 dB`), reaching `0.00696` MAE (`39.60 dB`) by Epoch 35.

### 4. Calibrated Smoke Run Results
Configuring `prior_reflectance: 0.107` in `configs/train_unet.yaml`:

```bash
.venv\Scripts\python -m cloud_removal.train --smoke-test
```

```
==================================================
Starting Training: U-Net Baseline on SEN12MS-CR
Device: cpu | Smoke Test: True
Train samples: 16 | Val samples: 8
Total Parameters: 7,763,107
Epochs: 1 | Batch Size: 4 | LR: 0.0002
==================================================
Epoch [ 1/ 1] (6.3s) - Train Loss: 0.03634 | Train MAE: 0.03634 | Val Loss: 0.02644 | Val MAE: 0.02644 | Val PSNR: 30.59 dB
  --> Saved new best checkpoint (Val MAE: 0.02644) to checkpoints\best_unet.pt
==================================================
Training completed in 6.45s.
Best Val MAE: 0.02644
Best Checkpoint: checkpoints\best_unet.pt
MLflow Run ID: 47a1947a6a4940219c50494089d3528c
==================================================
```

---

## Inference and Qualitative Evaluation

The inference module (`src/cloud_removal/infer.py`) evaluates trained checkpoints on unseen patches:

```bash
.venv\Scripts\python -m cloud_removal.infer \
    --checkpoint checkpoints/best_unet.pt \
    --split val \
    --index 0 \
    --output reports/figures/inference_val_0.png
```

### Inference Output

```json
{
  "patch_id": "p101",
  "input_shape": [3, 256, 256],
  "output_shape": [3, 256, 256],
  "prediction_min": 0.1056,
  "prediction_max": 0.1102,
  "prediction_mean": 0.1084,
  "metrics": {
    "baseline_input_mae": 0.0908,
    "baseline_input_psnr": 17.68,
    "unet_prediction_mae": 0.0269,
    "unet_prediction_psnr": 30.70
  },
  "comparison_figure": "reports\\figures\\inference_val_0.png",
  "checkpoint_epoch": 1,
  "checkpoint_val_mae": 0.0264
}
```

The script renders a 3-panel horizontal comparison figure (`[Cloudy Input | U-Net Prediction | Clear Ground Truth]`) saved at `reports/figures/inference_val_0.png`.

---

## Instructions for Full Training on Google Colab (GPU)

Because local development runs on CPU (`torch+cpu`), full 10–20 epoch training across all 575 training patches is best performed on an accelerated GPU runtime:

### Step 1: Open Google Colab and Select GPU
1. Create a new notebook in Google Colab.
2. Go to **Runtime** $\to$ **Change runtime type** $\to$ Select **T4 GPU** (or A100/V100).

### Step 2: Clone and Setup Environment
```bash
!git clone https://github.com/<your-username>/liss4-cloud-removal.git
%cd liss4-cloud-removal

# Install dependencies
!pip install -q rasterio mlflow torch torchvision pyyaml matplotlib pillow
```

### Step 3: Fetch Dataset from DVC Remote (or download subset)
If using a remote DVC storage:
```bash
!dvc pull data/raw/sen12mscr.dvc
```
Or execute the automated acquisition script directly:
```bash
!python -m cloud_removal.acquire_sen12mscr --samples 719 --output-dir data/raw/sen12mscr
```

### Step 4: Run Full Training
```bash
!python -m cloud_removal.train \
    --config configs/train_unet.yaml \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.0002 \
    --device cuda
```

### Step 5: Evaluate Best Checkpoint
```bash
!python -m cloud_removal.infer \
    --checkpoint checkpoints/best_unet.pt \
    --split val \
    --index 0 \
    --output reports/figures/colab_val_0.png
```

---

## Phase 7.3 & 7.4: Reproducible Demonstration & Shared Display Stretch

### Shared Display Contrast Stretch
Sentinel-2 optical digital numbers scaled by $1/10000$ represent physical surface reflectance. Winter land surface reflectance is naturally low ($\sim 0.06 - 0.14$). Directly rendering raw floating-point reflectance in sRGB $[0.0, 1.0]$ produces visually dark panels (~9% display luminance) for both clear targets and model predictions.

To solve this while preserving scientific integrity:
- `render_comparison_figure` in `src/cloud_removal/infer.py` applies a shared $2\% - 98\%$ percentile contrast stretch derived from the clear ground reference across all three panels (Cloudy, Prediction, Clear).
- Numerical metrics (`compute_mae`, `compute_psnr`) remain strictly computed on raw physical reflectance values in $[0.0, 1.0]$.

### Minimal Demonstration (`demo.py`)
A minimal reproducible demonstration loads the compact U-Net (`checkpoints/compact/best_compact_unet.pt`, 1.94M parameters) and runs inference end-to-end on validation sample 0:

```bash
python demo.py
```
Output:
```
======================================================================
SEN12MS-CR Optical Cloud Removal - Minimal Demonstration
======================================================================
Loading checkpoint:  checkpoints\compact\best_compact_unet.pt
Dataset manifest:    data\raw\sen12mscr\manifest.json (split='val', index=0)
Target figure path:  reports/figures/demo_comparison.png
Requested device:    auto
----------------------------------------------------------------------
Patch ID:            p101
Input Shape:         [3, 256, 256]
Output Shape:        [3, 256, 256]
Checkpoint Epoch:    5
----------------------------------------------------------------------
Quantitative Evaluation (Physical Reflectance [0, 1]):
  Cloudy Input:      MAE = 0.09086  |  PSNR = 17.68 dB
  U-Net Prediction:  MAE = 0.00586  |  PSNR = 41.59 dB
  Improvement:       dMAE  = +0.08500 (lower is better)
                     dPSNR = +23.91 dB (higher is better)
----------------------------------------------------------------------
Comparison figure saved: reports\figures\demo_comparison.png
Shared 2%-98% contrast stretch applied for visual clarity.
======================================================================
```

---

## Unit Testing

55 comprehensive tests verify model architecture, shape preservation, Sigmoid output bounding, parameter counting, checkpoint reload equivalence, MAE accuracy, PSNR calibration, shared display stretch, and the minimal demo workflow:

```bash
.venv\Scripts\pytest
```
Result: `55 passed in 30.31s`.

---

## Scientific Boundaries and Evaluation Limitations

1. **Optical Cloud Penetration Boundary**: Optical sensor channels (RGB) cannot physically penetrate optically thick, opaque clouds. The U-Net reconstructs surface reflectance using spatial context from surrounding unclouded pixels and statistical priors learned during training. It acts as an optical inpainting / imputation model, not a physical cloud-penetrating sensor.
2. **Dataset Holdout vs Benchmark Test ROIs**: The 144 validation samples are an internal holdout split from official SEN12MS-CR training ROI 102 (Winter). They prevent overfitting during development but do not represent performance on the official SEN12MS-CR benchmark test ROIs.
3. **Training & Demonstration Provenance**: The compact baseline checkpoint (`checkpoints/compact/best_compact_unet.pt`) was trained locally on CPU across 5 epochs (AdamW lr=2e-4, batch=8, Run ID `0f2e8b45bb024ec5bd6187d1d2a9d8aa`, validation MAE `0.00667`, PSNR `38.94 dB`). Google Colab provides an accelerated GPU environment for exploration and demonstration.
4. **Physical Reflectance vs Display Luminance**: Surface reflectance across winter scenes naturally clusters at low values ($\sim 0.06 - 0.14$). An MAE of $0.00667$ represents close quantitative fidelity in physical reflectance space $[0.0, 1.0]$. For human visual evaluation, a shared 2%–98% percentile contrast stretch is applied across panels.
5. **Cross-Sensor Non-Transferability (LISS-IV)**: As established in Phase 7.6 and Phase 7.7, the Sentinel-2 trained U-Net cannot be directly deployed or evaluated against Resourcesat-2A LISS-IV data due to resolution divergence (5 m vs 10 m), spectral misalignment (Green-Red-NIR vs RGB; absence of Blue channel), and radiometric calibration gaps ($ESUN_\lambda$ omitted in product deliverables). No cross-sensor transfer claims are made.

