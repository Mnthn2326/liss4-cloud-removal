# 🚀 LISS-IV Cloud Removal — Collaborative Team Training Guide

Welcome to the collaborative training pipeline! This guide provides step-by-step instructions for our 4-person team to train different satellite cloud-removal architectures in **Google Colab** on free **T4 GPUs**, with all experiment parameters, loss curves, and model checkpoints automatically streaming into our centralized **DagsHub MLflow dashboard**.

---

## 📌 Step 0: Accept DagsHub Invitation (First Time Only)

1. Log in to your account at [dagshub.com](https://dagshub.com) (sign up with GitHub if you don't have one).
2. Click this official repository invite link to join:
   👉 **[Click Here to Accept the DagsHub Collaboration Invite](https://dagshub.com/invitation/accept/202610241443010080a32925b165f57aee393c9eb568def378fc934aa9)**
3. Once accepted, you will have collaborator access to [`https://dagshub.com/Mnthn2326/liss4-cloud-removal`](https://dagshub.com/Mnthn2326/liss4-cloud-removal).

---

## ⚠️ Crucial Team Execution Rules in Google Colab

Open the collaborative notebook in Colab:
👉 [`notebooks/train_collaborative_colab.ipynb`](https://github.com/Mnthn2326/liss4-cloud-removal/blob/master/notebooks/train_collaborative_colab.ipynb)

> [!CAUTION]
> **DO NOT RUN THE ENTIRE NOTEBOOK FROM TOP TO BOTTOM!**
> Running every cell will overwrite other models and take hours. Follow this strict execution flow:
>
> 1. **EVERYONE RUNS COMMON SETUP:**
>    * **Cell 2 (Step 1):** GPU Hardware Check
>    * **Cell 4 (Step 2):** Clone Repository & Install Dependencies
>    * **Cell 6 (Step 3):** Connect to DagsHub MLflow & **Set Your Name**
>    * **Cell 8 (Step 4):** Acquire SEN12MS-CR Dataset
>
> 2. **IN STEP 5, RUN ONLY YOUR ASSIGNED TRACK:**
>    * **Teammate 1:** Run **Cell 11 ONLY** (Track A: Standard U-Net)
>    * **Teammate 2:** Run **Cell 13 ONLY** (Track B: Compact U-Net)
>    * **Teammate 3:** Run **Cell 15 ONLY** (Track C: Pix2Pix cGAN)
>    * **Teammate 4:** Run **Cell 17 ONLY** (Track D: LISS-IV FCC cGAN)
>
> 3. **EVERYONE RUNS POST-TRAINING VERIFICATION:**
>    * **Cell 19 (Step 6):** Run visual inference on validation sample `0`
>    * **Cell 21 (Step 7):** Download your trained `.pt` checkpoint file

> [!IMPORTANT]
> **🚨 DO NOT FORGET TO CHANGE YOUR NAME IN CELL 6 (Step 3):**
> ```python
> # In Cell 6, change "YOUR_NAME_HERE" to your actual name:
> TEAMMATE_NAME = "Manthan"  # e.g., "Manthan", "Teammate_2", etc.
> os.environ["MLFLOW_USER"] = TEAMMATE_NAME
> ```
> This tags your runs in DagsHub so everyone can see who trained what!

---

## 🛠️ How to Modify Config Files Directly in Colab

To train with different hyperparameters, you do **not** need to fork or clone locally. Modify the YAML file directly in Google Colab:

### Method A: Using the Colab File Editor (Recommended)
1. On the left sidebar of Google Colab, click the **Folder icon** 📁.
2. Navigate to `liss4-cloud-removal` $\to$ `configs/`.
3. **Double-click** on your assigned config file (e.g. `train_unet.yaml` or `train_cgan.yaml`).
4. An editor opens on the right side of Colab.
5. Edit the values (e.g., change `learning_rate` or `lambda_l1`), press `Ctrl + S` to save, and run your training cell!

```
Google Colab File Browser:
📁 liss4-cloud-removal/
   ├── 📁 configs/
   │      ├── train_unet.yaml         <-- Teammate 1 (Double click to edit)
   │      ├── train_compact_unet.yaml <-- Teammate 2 (Double click to edit)
   │      ├── train_cgan.yaml         <-- Teammate 3 (Double click to edit)
   │      └── train_fcc_cgan.yaml     <-- Teammate 4 (Double click to edit)
```

### Method B: Using a Single-Line Bash Command
You can also insert a code cell right above your training cell to update parameters quickly:
```bash
# Example: Change learning rate to 0.0001
!sed -i 's/learning_rate: 0.0002/learning_rate: 0.0001/' configs/train_unet.yaml

# Example: Change lambda_l1 to 50.0
!sed -i 's/lambda_l1: 100.0/lambda_l1: 50.0/' configs/train_cgan.yaml
```

---

## 👥 Teammate Assignments & 5 Hyperparameter Sets Each

---

### 👤 Teammate 1: Standard Full U-Net (Benchmark & Loss Tuning)
* **Assigned Model:** 4-Stage Full U-Net (~7.76M Parameters)
* **Assigned Config File:** `configs/train_unet.yaml`
* **Colab Training Cell:** **Cell 11** (`!python -m cloud_removal.train --config configs/train_unet.yaml`)

| Run ID | Experiment Focus | Exact Parameters to Set in `configs/train_unet.yaml` |
| :--- | :--- | :--- |
| **Run 1.1** | **Default Compound Baseline** | `training.loss: "compound_l1_ssim"`, `training.ssim_weight: 0.2`, `training.learning_rate: 0.0002`, `training.epochs: 10`, `training.batch_size: 8` |
| **Run 1.2** | **High Structural Fidelity** | `training.loss: "compound_l1_ssim"`, `training.ssim_weight: 0.5`, `training.learning_rate: 0.0002`, `training.epochs: 10` |
| **Run 1.3** | **Pure L1 Loss Benchmark** | `training.loss: "l1"`, `training.learning_rate: 0.0002`, `training.epochs: 10` |
| **Run 1.4** | **Conservative Learning Rate** | `training.loss: "compound_l1_ssim"`, `training.ssim_weight: 0.2`, `training.learning_rate: 0.00005`, `training.epochs: 10` |
| **Run 1.5** | **Large Batch Scale** | `training.batch_size: 16`, `training.learning_rate: 0.0004`, `training.epochs: 10` |

---

### 👤 Teammate 2: Compact Mobile U-Net (Efficiency & API Deployment)
* **Assigned Model:** Slimmed 4-Stage U-Net (~1.94M Parameters, 3x Faster Inference)
* **Assigned Config File:** `configs/train_compact_unet.yaml`
* **Colab Training Cell:** **Cell 13** (`!python -m cloud_removal.train --config configs/train_compact_unet.yaml`)

| Run ID | Experiment Focus | Exact Parameters to Set in `configs/train_compact_unet.yaml` |
| :--- | :--- | :--- |
| **Run 2.1** | **Default Compact Baseline** | `model.base_channels: 16`, `training.learning_rate: 0.0003`, `training.batch_size: 8`, `training.epochs: 5`, `training.loss: "compound_l1_ssim"` |
| **Run 2.2** | **High-Throughput Pure L1** | `training.loss: "l1"`, `training.batch_size: 16`, `training.learning_rate: 0.0004`, `training.epochs: 5` |
| **Run 2.3** | **Extended Training Epochs** | `training.epochs: 10`, `training.learning_rate: 0.0002`, `training.patience: 4` |
| **Run 2.4** | **AdamW Weight Decay** | `training.optimizer: "adamw"`, `training.weight_decay: 0.0001`, `training.learning_rate: 0.0003`, `training.epochs: 5` |
| **Run 2.5** | **Ultra-Compact Channel Width** | `model.base_channels: 8`, `training.learning_rate: 0.0005`, `training.batch_size: 8`, `training.epochs: 5` |

---

### 👤 Teammate 3: Pix2Pix Conditional GAN (Texture & Boundary Realism)
* **Assigned Model:** U-Net Generator + $70 \times 70$ PatchGAN Discriminator (~10.5M Parameters)
* **Assigned Config File:** `configs/train_cgan.yaml`
* **Colab Training Cell:** **Cell 15** (`!python -m cloud_removal.train_cgan --config configs/train_cgan.yaml`)

| Run ID | Experiment Focus | Exact Parameters to Set in `configs/train_cgan.yaml` |
| :--- | :--- | :--- |
| **Run 3.1** | **Standard Pix2Pix Baseline** | `training.lambda_l1: 100.0`, `training.learning_rate_g: 0.0002`, `training.learning_rate_d: 0.0002`, `discriminator.norm_layer: "batch"`, `training.epochs: 10` |
| **Run 3.2** | **High Edge Sharpness** | `training.lambda_l1: 50.0`, `training.learning_rate_g: 0.0001`, `training.learning_rate_d: 0.0003` (TTUR: D learns 3x faster) |
| **Run 3.3** | **High Content Preservation** | `training.lambda_l1: 200.0`, `training.learning_rate_g: 0.00015`, `training.learning_rate_d: 0.00015` |
| **Run 3.4** | **Least Squares GAN (LSGAN)** | `training.use_lsgan: true`, `training.lambda_l1: 100.0`, `training.learning_rate_g: 0.0002`, `training.learning_rate_d: 0.0002` |
| **Run 3.5** | **Instance Normalization** | `discriminator.norm_layer: "instance"`, `training.lambda_l1: 100.0`, `training.epochs: 10` |

---

### 👤 Teammate 4: ISRO LISS-IV False-Color Composite (Sensor Alignment)
* **Assigned Model:** Spectral-Aligned cGAN on Sentinel-2 NIR (B08), Red (B04), and Green (B03)
* **Assigned Config File:** `configs/train_fcc_cgan.yaml`
* **Colab Training Cell:** **Cell 17** (`!python -m cloud_removal.train_cgan --config configs/train_fcc_cgan.yaml`)

| Run ID | Experiment Focus | Exact Parameters to Set in `configs/train_fcc_cgan.yaml` |
| :--- | :--- | :--- |
| **Run 4.1** | **Default FCC Baseline** | `model.prior_reflectance: 0.107`, `training.lambda_l1: 100.0`, `training.learning_rate_g: 0.0002`, `training.epochs: 5`, `training.batch_size: 4` |
| **Run 4.2** | **Vegetation-Tuned Prior** | `model.prior_reflectance: 0.20` (compensates for strong NIR vegetative reflection), `training.lambda_l1: 120.0`, `training.epochs: 8` |
| **Run 4.3** | **Deep Receptive Field Discriminator** | `discriminator.n_layers: 4` ($142 \times 142$ patch size), `discriminator.ndf: 64`, `training.epochs: 6` |
| **Run 4.4** | **Smooth LSGAN FCC Dynamics** | `training.use_lsgan: true`, `training.lambda_l1: 150.0`, `training.learning_rate_g: 0.00015`, `training.epochs: 8` |
| **Run 4.5** | **FCC U-Net Comparison** | Run `!python -m cloud_removal.train --config configs/train_fcc_unet.yaml` (Trains non-adversarial U-Net on FCC for direct comparison) |

---

## 📊 Viewing and Comparing Runs in DagsHub

As soon as a training run finishes:

1. Open your dashboard at: **`https://dagshub.com/Mnthn2326/liss4-cloud-removal`**
2. Click on the **Experiments** tab (MLflow icon).
3. **Filter by Teammate:** Click on the `user` column or filter by tag (`user = "YourName"`).
4. **Compare Any 2 or More Models:**
   * Select the checkboxes next to Teammate 1's U-Net and Teammate 3's cGAN.
   * Click **Compare** to view interactive overlay plots of **Generator Loss**, **Discriminator Loss**, **Validation MAE**, and **Validation PSNR**.
5. **Download Best Model Checkpoint:**
   * Click on the winning run $\to$ **Artifacts** tab.
   * Download `best_unet.pt` or `best_cgan_generator.pt` for local evaluation or serving!
