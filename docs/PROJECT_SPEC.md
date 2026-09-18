# Project Spec: Generative AI-Based Cloud Removal & Reconstruction (LISS-IV)

## 1. Problem statement
Persistent cloud cover degrades the usability of optical satellite imagery (LISS-IV) over
tropical/mountainous regions (NER India), limiting land-use mapping, disaster monitoring,
environmental assessment, and infrastructure analysis. Traditional cloud masking discards data;
this project instead **reconstructs** the cloud-covered regions using generative deep learning.

## 2. Course framing (read this before writing any code)
This project is graded for an **MLOps** course. The generative model is a *component*, not the
deliverable. Priority order for engineering effort:

1. A working, versioned, reproducible **end-to-end pipeline** (data → train → eval → register → serve → monitor)
2. A generative model that is *good enough to demonstrate the pipeline*, not state-of-the-art
3. Only after (1) and (2) are solid: model quality improvements, auxiliary SAR fusion, etc.

Do not over-invest in model architecture novelty. A working pix2pix-style conditional GAN,
correctly wrapped in MLOps tooling, scores higher here than an unwrapped SOTA transformer.

## 3. Scope decisions (locked — do not deviate without updating this file)
- **Primary dataset for early development:** RICE (public, small, matches CloudGAN precedent).
  Used as a stand-in for LISS-IV until Bhoonidhi access is granted.
- **LISS-IV integration:** deferred until data access confirmed. The pipeline MUST be
  dataset-agnostic (config-driven — see `configs/dataset_*.yaml`) so swapping in LISS-IV is a
  config change, not a rewrite.
- **Auxiliary SAR fusion (Sentinel-1via SEN12MS-CR):** stretch goal, phase 2. Do not block
  phase 1 on this.
- **Infrastructure constraint:** free-tier only (Google Colab / Kaggle for compute; DagsHub for
  hosted DVC + MLflow; GitHub Actions for CI; Hugging Face Spaces for serving). No paid cloud,
  no self-hosted servers, no long-running daemons (Colab sessions are ephemeral).

## 4. Objectives (from problem statement)
- Automated cloud removal framework for LISS-IV imagery
- Reconstruct cloud-covered regions preserving spatial structure + spectral consistency
- Quantitative (PSNR/SSIM) + qualitative evaluation
- Scalable, reproducible workflow suitable for operational use
- Comparative note on generative architectures considered (pix2pix baseline vs. transformer-based
  alternatives such as DVPNet — documented, not necessarily implemented)

## 5. Milestone: Week-1 review target (~60% complete)
End-to-end loop closing once on RICE, with visible artifacts at every MLOps stage, even if each
stage is minimal. Deferred explicitly and stated as such in the review: LISS-IV data, SAR fusion,
full monitoring/alerting, hyperparameter tuning.

Definition of done for the review:
- [ ] `dvc repro` runs the full preprocess → train → eval pipeline on RICE
- [ ] Training run logged to MLflow (DagsHub) with loss curves + sample images
- [ ] Best checkpoint registered in MLflow Model Registry
- [ ] GitHub Actions CI passes (lint + smoke test)
- [ ] FastAPI `/predict` endpoint runs locally (Docker optional if time-constrained) and returns a
      cloud-removed image for a sample input

## 6. Non-goals (explicitly out of scope, all phases)
- Real-time / streaming inference
- Multi-region generalization beyond NER India
- Training from scratch without any transfer learning basis
- Building custom orchestration infra (Airflow/Kubeflow) — DVC pipelines are sufficient at this scale

## 7. Reference implementations consulted
| Repo | Relevance | What we take from it |
|---|---|---|
| JerrySchonenberg/CloudGAN | Two-stage detect+inpaint (AE + SN-PatchGAN), RICE dataset | Mask/inpaint decomposition idea (not adopted directly — see MODEL_SPEC.md) |
| Chintan2108/Cloud-Removal-...-cGAN | pix2pix-style conditional GAN, single-stage | **This is our baseline architecture pattern** |
| huangwenwenlili/DVPNet | Transformer, spatial+frequency prompting, SEN12MS-CR (SAR+optical) | Reference for phase-2 SAR fusion; too heavy for phase-1 scope |
