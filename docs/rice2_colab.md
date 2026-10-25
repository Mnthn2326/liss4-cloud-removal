# RICE2 GPU training in Google Colab

Use [the Colab notebook](../notebooks/train_rice2_colab.ipynb) with a T4 GPU.

Before opening Colab, create two archives in the project root. Do not include `.venv`, `.git`, `data`, `checkpoints`, `mlruns`, or `scratch` in the source archive.

```powershell
tar -a -c -f project_source.zip README.md pyproject.toml requirements.txt src configs tests notebooks docs Dockerfile docker-compose.yml dvc.yaml .gitignore .dockerignore
tar -a -c -f RICE2.zip data\raw\RICE2
```

Upload both files to `Google Drive/MyDrive/rice2-prototype/`. In Colab select **Runtime → Change runtime type → T4 GPU**, open the notebook, and run its cells in order.

The final outputs are copied to `Google Drive/MyDrive/rice2-prototype/outputs/`:

- `best_rice2_compact_unet.pt`
- `latest_rice2_compact_unet.pt`
- `rice2_manifest.json`
- `rice2_test_metrics.json`

Only report the held-out `rice2_test_metrics.json` in the final presentation. The clear RICE2 reference is a temporally nearby observation, so call the result *cloud reconstruction* rather than same-instant cloud-free truth.

## Kaggle alternative

Create a private Kaggle Dataset containing the same two ZIP files, attach it to a GPU notebook, and replace `<your-dataset-slug>` below. Run these cells in order:

```python
from pathlib import Path
import os, shutil, zipfile

INPUT = Path('/kaggle/input/<your-dataset-slug>')
WORKSPACE = Path('/kaggle/working/liss4-cloud-removal')
PROJECT_ARCHIVE = next(INPUT.rglob('project_source.zip'))
RICE2_ARCHIVE = next(INPUT.rglob('RICE2.zip'))
assert os.system('nvidia-smi') == 0

shutil.rmtree(WORKSPACE, ignore_errors=True)
WORKSPACE.mkdir(parents=True)
with zipfile.ZipFile(PROJECT_ARCHIVE) as archive:
    archive.extractall(WORKSPACE)
with zipfile.ZipFile(RICE2_ARCHIVE) as archive:
    archive.extractall(WORKSPACE)
os.chdir(WORKSPACE)
assert (WORKSPACE / 'data/raw/RICE2/cloud').is_dir()
```

```python
!pip -q install mlflow rasterio pyyaml pillow pytest ruff
!pip -q install -e . --no-deps
!python -m cloud_removal.rice2 --dataset-root data/raw/RICE2 --output data/manifests/rice2_manifest.json
!python -m pytest tests/test_rice2.py tests/test_train.py -q
!python -m cloud_removal.train_rice2 --config configs/train_rice2_compact_unet.yaml --device cuda
!python -m cloud_removal.evaluate_rice2 --checkpoint checkpoints/rice2/best_rice2_compact_unet.pt --manifest data/manifests/rice2_manifest.json --split test --output reports/rice2_test_metrics.json --device cuda
```

Download `checkpoints/rice2/best_rice2_compact_unet.pt` and `reports/rice2_test_metrics.json` from Kaggle **Output** after the run completes.
