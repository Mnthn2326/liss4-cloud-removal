# Data Spec

## Principle
The pipeline must never hardcode a dataset's paths, band count, or resolution. Every dataset is
described by one YAML config under `configs/dataset_*.yaml` (schema below), and all
preprocessing/training/eval code reads from that config. This is what lets us prototype on RICE
today and swap to LISS-IV later without rewriting the pipeline — see `configs/dataset_rice.yaml`
for a working example.

## Dataset config schema
```yaml
name: str                    # dataset identifier, used in MLflow run tags
bands:
  count: int                 # number of channels the model expects
  names: [str, ...]          # e.g. ["R","G","B"] or ["B2","B3","B4","NIR"]
patch_size: int              # tile size in pixels, e.g. 256
paths:
  raw: str                   # path to raw downloaded data (DVC-tracked)
  processed: str             # path to tiled/normalized output (DVC-tracked)
  pairs_manifest: str        # CSV/JSON mapping cloudy_tile -> clear_tile (-> mask, if available)
split:
  train: float               # fraction, e.g. 0.8
  val: float
  test: float
  seed: int                  # for reproducibility
normalization:
  method: str                # "minmax" | "zscore"
  per_band_stats: bool       # whether stats are computed per-band or globally
auxiliary:
  sar_available: bool        # true only for SEN12MS-CR / future LISS-IV+Sentinel-1 pairing
  sar_path: str | null
```

## Datasets in use

### RICE (phase 1, active now)
- Source: https://github.com/BUPTLdy/RICE_DATASET
- RICE-I: 500 pairs, filmy/thin cloud + cloud-free, Google Earth RGB
- RICE-II: 736 pairs, Landsat 8 OLI/TIRS, includes cloud mask derived from QA band
- Bands: RGB (3), matches `bands.count: 3`
- Use RICE-I for the fastest baseline; RICE-II if mask-aware evaluation is wanted

### SEN12MS-CR (phase 2 stretch)
- ~110,000 samples, paired Sentinel-2 (cloudy/clear) + Sentinel-1 SAR, global coverage
- Source: https://mediatum.ub.tum.de/1554803
- Only pull a small regional subset (NER-adjacent latitude/climate if possible) — full dataset is
  far larger than needed and will blow past free-tier storage/compute budgets

### LISS-IV (target, pending access)
- Source: Bhoonidhi (ISRO)
- Bands: B2 (Green), B3 (Red), B4 (NIR) — **3 bands, not 4** (LISS-IV standard product is
  multispectral 3-band + a separate panchromatic product; confirm exact band set on data receipt
  and update `configs/dataset_liss4.yaml` accordingly — do not assume until verified)
- Resolution: ~5.8m spatial resolution
- Action item once access is granted: fill in `configs/dataset_liss4.yaml`, run the same
  preprocessing DVC stage with `--dataset liss4`, confirm tile counts and pair alignment before
  attempting any training

## Preprocessing steps (dataset-agnostic, parameterized by config above)
1. Read raw scene (Rasterio) → reproject/co-register if auxiliary data involved
2. Tile into `patch_size x patch_size` patches with configurable overlap (default: no overlap for
   training tiles, small overlap for inference to avoid seam artifacts)
3. Normalize per `normalization` config
4. Build cloudy/clear pair manifest (and mask manifest, where available)
5. Split into train/val/test per `split` config, stratified by scene if possible to avoid leakage
   (never split individual tiles from the same scene across train and test)
6. Write processed tiles + manifest to `paths.processed` — this becomes the DVC-tracked pipeline
   output
