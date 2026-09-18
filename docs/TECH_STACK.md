# Tech Stack

Every choice below is constrained by: **free-tier infra only**, **ephemeral compute (Colab/Kaggle)**,
**one-week-to-first-review timeline**. Do not substitute a tool that requires a persistent server
or a paid tier without updating this file and PROJECT_SPEC.md.

## Core ML
| Purpose | Tool | Version | Notes |
|---|---|---|---|
| Language | Python | 3.10 | Colab default-compatible |
| DL framework | PyTorch | 2.x + torchvision | Not TensorFlow — CloudGAN used TF1.15 which is EOL; PyTorch has better ecosystem support for the tooling below |
| GAN losses | Standard adversarial (BCE or LSGAN) + L1 pixel loss | — | Keep it simple; do not add perceptual/frequency losses until baseline works |
| Image I/O | Rasterio, GDAL | — | Required for multi-band satellite imagery (LISS-IV, Sentinel), not plain PNG/JPEG libraries |
| Augmentation | Albumentations | — | Handles multi-channel geospatial tiles correctly |
| Array/tensor ops | NumPy, OpenCV, scikit-image | — | scikit-image for PSNR/SSIM metrics specifically |

## Data & experiment versioning
| Purpose | Tool | Notes |
|---|---|---|
| Data versioning | DVC | Tracks large image datasets outside git |
| Remote storage | DagsHub-hosted DVC remote | Free tier, no S3/GCS bill |
| Pipeline definition | DVC pipelines (`dvc.yaml`, `dvc repro`) | Declarative, stateless, reruns only stale stages — no orchestrator daemon needed |
| Experiment tracking | MLflow, hosted via DagsHub | Free hosted tracking server; survives Colab session death |
| Model registry | MLflow Model Registry (DagsHub) | Staging → Production promotion |

## CI/CD
| Purpose | Tool | Notes |
|---|---|---|
| CI | GitHub Actions | Free for public repos; lint + unit tests + CPU-only smoke train on every PR |
| Linting | ruff (or flake8) + black | Fast, single-tool preference: ruff |
| Testing | pytest | Data shape tests, loss sanity checks, tiny smoke-train test |
| Containerization | Docker | Single Dockerfile for the serving image |
| CD / hosting | Hugging Face Spaces (Docker SDK) | Free, public inference endpoint for demos |

## Serving & monitoring
| Purpose | Tool | Notes |
|---|---|---|
| Inference API | FastAPI + Uvicorn | `/predict` endpoint: image in, cloud-removed image out |
| Monitoring | Scheduled GitHub Action (cron) + custom script | Pings deployed endpoint with held-out test images, logs PSNR/SSIM + input band stats to MLflow; no persistent monitoring server |
| Drift detection (stretch) | Evidently AI | Only if time allows post-review |

## Explicitly rejected / deferred
- **Airflow / Kubeflow / Prefect** — require a persistent scheduler; overkill at this data/team scale and incompatible with ephemeral Colab compute. DVC pipelines cover the same need.
- **Self-hosted MLflow server** — DagsHub's hosted instance removes this entirely.
- **TensorFlow** — used by CloudGAN reference repo, but PyTorch chosen for this project for ecosystem/tooling reasons above.
- **Kubernetes** — no justification at this scale; Docker + HF Spaces is sufficient.

## Compute
| Use | Environment |
|---|---|
| Training | Google Colab (free GPU tier) or Kaggle Notebooks |
| CI smoke tests | GitHub Actions runner (CPU only, synthetic tiny data) |
| Inference demo | Hugging Face Spaces (CPU, small enough model to run without GPU) |
