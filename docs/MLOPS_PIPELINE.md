# MLOps Pipeline Architecture

## The loop this project implements
```
versioned data -> tracked experiment -> registered model -> containerized serving -> scheduled monitoring
```
Every arrow above must be a real, runnable artifact by the review, not a diagram-only concept.

## 1. Data & pipeline stages (DVC)
`pipelines/dvc.yaml` defines these stages, each with explicit deps/outs so `dvc repro` only
reruns what's stale:

| Stage | Command (conceptual) | Depends on | Produces |
|---|---|---|---|
| `preprocess` | `src/data/preprocess.py --config configs/dataset_rice.yaml` | raw data, dataset config | tiled/normalized data + pairs manifest |
| `train` | `src/training/train.py --config configs/dataset_rice.yaml --train-config configs/train_baseline.yaml` | processed data | model checkpoint, MLflow run |
| `eval` | `src/evaluation/evaluate.py --checkpoint <path> --config ...` | checkpoint, test split | PSNR/SSIM report, sample grid, logged to MLflow |
| `register` | `src/training/register_model.py --run-id <mlflow_run>` | eval report | model version in MLflow Registry (stage=Staging) |

Run the whole thing with `dvc repro`. Each stage is a plain Python script — no orchestrator
daemon required, which matters because Colab sessions don't persist.

## 2. Experiment tracking (MLflow via DagsHub)
Every training run logs:
- Hyperparameters (from `train_baseline.yaml`)
- Per-epoch generator loss, discriminator loss, L1 loss
- PSNR/SSIM on validation split, per epoch or every N epochs
- Sample image grids (input / generated / ground truth) as artifacts
- Dataset config used (tag the run with dataset name + git commit hash for reproducibility)

## 3. Model registry
- Promote a run's checkpoint to the registry only after it clears a minimum bar (e.g. SSIM above
  a threshold you set once you see baseline numbers — don't hardcode a number before training once)
- Stages: `Staging` (just trained, passed eval) → `Production` (manually promoted after review)
- The serving layer always loads whatever is tagged `Production` — never a raw checkpoint path

## 4. CI (GitHub Actions, `.github/workflows/ci.yaml`)
Runs on every PR:
- Lint (ruff/black check)
- Unit tests (pytest):
  - Data shape tests: tile output matches `patch_size` and `bands.count` from config
  - Loss sanity test: loss is finite and decreases over a handful of steps on synthetic data
  - Smoke train: 1 batch, 1 step, tiny synthetic tensors, CPU-only, asserts it runs without error
    in under ~30s — this validates the training loop's plumbing without needing real data or GPU
- This is cheap to build and is exactly the kind of check an MLOps reviewer wants to see

## 5. Serving (FastAPI + Docker)
- `src/serving/app.py`: loads the `Production`-tagged model from the registry at startup
- `POST /predict`: accepts an image (or raster tile), returns the cloud-removed reconstruction
- `GET /health`: basic liveness check
- `Dockerfile`: single-stage build, CPU inference (model is small enough — no GPU needed to serve)
- Deploy target: Hugging Face Spaces (Docker SDK), free tier

## 6. Monitoring (scheduled GitHub Action, no persistent server)
`.github/workflows/monitor.yaml`, cron-scheduled (e.g. daily):
- Calls the deployed `/predict` endpoint with a fixed held-out set of test images
- Computes PSNR/SSIM against known ground truth, logs to MLflow as a `monitoring` run
- Computes basic input-side stats (band means/stds) as a stand-in for drift detection
- If metrics fall below a threshold: fail the workflow (visible red X) — this is your "alerting"
  for a free-tier setup; a real alerting channel (Slack/email webhook) is a fast follow-on if
  time allows, not required for the review

## 7. What "done" looks like per stage, for the review
- DVC: `dvc repro` runs clean, `dvc dag` shows the 4-stage graph
- MLflow: at least one full training run visible on the DagsHub dashboard with loss curves
- Registry: at least one model version registered
- CI: green checkmark on the repo, visible in a PR
- Serving: `curl localhost:8000/predict -F file=@sample.png` returns an image locally (Docker/HF
  deploy is a bonus if time allows, not blocking)
- Monitoring: script exists and runs manually at minimum; scheduled workflow is a bonus if time allows
