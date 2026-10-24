# Conditional GAN (Pix2Pix) Cloud Removal Pipeline

## Overview

This pipeline introduces a Conditional Generative Adversarial Network (**Pix2Pix cGAN**) for optical satellite cloud removal, inspired by the high-frequency perceptual sharpening demonstrated in recent remote sensing literature while adhering to strict MLOps principles, scientific data splitting, and reproducible tracking.

```
                            ┌────────────────────────┐
                            │ Cloudy Satellite Image │
                            └──────────┬─────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             ┌─────────────────┐               ┌─────────────────┐
             │ Real Clear (Y)  │               │ U-Net Generator │
             └────────┬────────┘               └────────┬────────┘
                      │                                 │ Generated (G(X))
                      │                                 ▼
                      │                        ┌─────────────────┐
                      │                        │ L1 Content Loss │◄───── (Pixel Fidelity)
                      │                        └─────────────────┘
                      ▼                                 ▼
              ┌──────────────────────────────────────────────────┐
              │            PatchGAN Discriminator (D)            │◄───── (High-Frequency Realism)
              └──────────────────────────────────────────────────┘
```

---

## 1. Mathematical Formulation

### Generator Objective
The generator minimizes a compound loss comprising the standard adversarial loss plus an $L_1$ reconstruction loss with weight multiplier $\lambda_{L1} = 100$:

$$\mathcal{L}_{G} = \mathbb{E}_{x \sim p(x)} \left[ \log(1 - D(x, G(x))) \right] + \lambda_{L1} \cdot \mathbb{E}_{x, y \sim p(x, y)} \left[ \| y - G(x) \|_1 \right]$$

### Discriminator Objective
The PatchGAN discriminator maximizes the probability of assigning the correct binary label to both real paired scenes and generated fake scenes:

$$\mathcal{L}_{D} = \frac{1}{2} \left( \mathbb{E}_{x, y \sim p(x, y)} \left[ \log D(x, y) \right] + \mathbb{E}_{x \sim p(x)} \left[ \log (1 - D(x, G(x))) \right] \right)$$

---

## 2. Implemented Modules

| Module / File | Description |
| :--- | :--- |
| [`src/cloud_removal/models/patchgan.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/models/patchgan.py) | $70 \times 70$ PatchGAN discriminator, `GANLoss` supporting BCE with logits & LSGAN, and Gaussian weight initialization. |
| [`src/cloud_removal/models/unet.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/models/unet.py) | Generator architecture (compatible with existing checkpoint format). |
| [`src/cloud_removal/train_cgan.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/src/cloud_removal/train_cgan.py) | Modular, MLflow-tracked dual-optimizer training loop with Early Stopping. |
| [`configs/train_cgan.yaml`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/configs/train_cgan.yaml) | Training configuration for standard RGB (Sentinel-2 B04, B03, B02). |
| [`configs/train_fcc_cgan.yaml`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/configs/train_fcc_cgan.yaml) | False-Color Composite configuration matching Resourcesat-2A LISS-IV bands (B08 NIR, B04 Red, B03 Green). |
| [`tests/test_patchgan.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/tests/test_patchgan.py) | Unit tests verifying discriminator shapes, weight initializations, and loss flows. |
| [`tests/test_cgan_train.py`](file:///e:/Third%20Year%20(26-27)%20(Semester%201)/liss4-cloud-removal/tests/test_cgan_train.py) | End-to-end integration tests confirming checkpoint drop-in compatibility with the inference pipeline. |

---

## 3. Usage & CLI Execution

### Smoke Test (Verification)
Run a quick 1-epoch smoke test across train and validation splits:
```bash
python -m cloud_removal.train_cgan --smoke-test
```

### Full Training (RGB Baseline)
```bash
python -m cloud_removal.train_cgan --config configs/train_cgan.yaml
```

### Full Training (LISS-IV Aligned FCC Baseline)
```bash
python -m cloud_removal.train_cgan --config configs/train_fcc_cgan.yaml
```

### Inference & Serving
Because the generator checkpoint payload matches the standard U-Net signature, trained cGAN generator checkpoints can be directly passed into our existing inference engine and API:
```bash
# Geospatial Tiled Inference
python -m cloud_removal.infer --checkpoint checkpoints/cgan/best_cgan_generator.pt --split val --index 0

# Fast Demonstration
python demo.py --checkpoint checkpoints/cgan/best_cgan_generator.pt

# FastAPI Docker Service
$env:MODEL_CHECKPOINT="checkpoints/cgan/best_cgan_generator.pt"
uvicorn cloud_removal.api:app --host 0.0.0.0 --port 8000
```
