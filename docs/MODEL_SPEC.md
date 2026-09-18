# Model Spec

## Phase 1 (required for review): pix2pix-style conditional GAN

Single-stage image-to-image translation. Input: cloudy image tile. Output: cloud-free
reconstruction. This mirrors Chintan2108's cGAN pattern, not CloudGAN's two-stage
detect-then-inpaint pattern — a single network is simpler to train, test, and wrap in MLOps
tooling, which matters more than architectural sophistication at this stage.

### Generator: U-Net
- Encoder-decoder with skip connections (standard pix2pix U-Net)
- Input: `H x W x C` tile, `C` = 3 (RGB) for RICE, configurable per dataset (see DATA_SPEC.md —
  LISS-IV will have 4 bands: B2/B3/B4/NIR)
- Output: same shape as input, reconstructed cloud-free tile
- Encoder: 8 downsampling conv blocks (Conv-BatchNorm-LeakyReLU), 4→512 channels
- Decoder: 8 upsampling conv blocks (ConvTranspose-BatchNorm-ReLU) with skip connections from
  matching encoder layer, dropout (0.5) on first 3 decoder layers during training
- Final activation: `tanh` (inputs/targets normalized to [-1, 1])

### Discriminator: PatchGAN
- Classifies overlapping N×N patches as real/fake rather than the whole image (standard pix2pix
  choice — better texture detail than a full-image discriminator)
- Input: concatenation of (cloudy input, candidate output) along channel dim — conditional GAN
- 5 conv layers, stride 2, BatchNorm + LeakyReLU, sigmoid output patch map
- Patch size target: 70×70 receptive field (standard pix2pix default)

### Loss
```
L_total = L_adversarial + lambda_L1 * L_L1
```
- `L_adversarial`: standard GAN BCE loss (or LSGAN MSE variant if training is unstable —
  try BCE first, switch only if mode collapse observed)
- `L_L1`: mean absolute error between generated and ground-truth cloud-free image (encourages
  low-frequency correctness; the discriminator handles high-frequency realism)
- `lambda_L1 = 100` (standard pix2pix default — do not tune this until the baseline trains cleanly)

### Training config (baseline defaults — override via config, not hardcoded)
- Optimizer: Adam, lr=2e-4, betas=(0.5, 0.999) for both G and D
- Batch size: 4–8 (constrained by Colab free-tier GPU memory)
- Epochs: start with 50–100 on RICE (small dataset, converges fast); log every epoch to MLflow
- Image size: 256×256 patches
- LR schedule: constant for first half of training, linear decay to 0 for second half (pix2pix
  convention) — implement only if time allows; constant LR is an acceptable v1

### Evaluation metrics
- PSNR (peak signal-to-noise ratio) — scikit-image
- SSIM (structural similarity) — scikit-image
- Qualitative: side-by-side grid (input / generated / ground truth) logged as an MLflow artifact
  every N epochs

## Phase 2 (stretch, post-review): SAR-fusion extension
- Add Sentinel-1 SAR as an additional input channel (early fusion: concatenate to generator input
  before the first conv layer)
- Dataset: SEN12MS-CR (already paired optical+SAR+cloud-free — see DVPNet repo for reference)
- Rationale: SAR penetrates cloud cover, giving the generator real structural signal instead of
  purely hallucinating from context — directly addresses the "auxiliary data fusion" line in the
  original problem statement
- This is a **new model version in the registry**, not a replacement — phase 1 model stays as the
  baseline comparison point

## Explicitly not doing in phase 1
- Transformer/attention-based architectures (DVPNet-style) — too slow to train from scratch on
  Colab free tier within the timeline; revisit only if phase-1 GAN quality is insufficient and
  time allows
- Diffusion models — training cost prohibitive at this compute budget
- Two-stage detect+inpaint (CloudGAN pattern) — adds a second model to version/serve/monitor for
  marginal benefit at this project stage
