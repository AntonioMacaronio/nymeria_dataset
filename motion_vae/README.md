# Motion VAE

A Continuous Motion VAE for the Nymeria dataset, based on the [EgoTwin paper](https://arxiv.org/abs/2508.13013). The VAE encodes full-body motion sequences into a compressed temporal latent space using head-centric representation and 1D causal convolutions.

## Quick Start

```python
from motion_vae import MotionVAE, MotionVAEConfig
import torch

config = MotionVAEConfig()
model = MotionVAE(config)

# Forward pass with a raw tensor (B, T, 294)
x = torch.randn(4, 152, 294)
output = model(x, preprocess=False)
# output['recon']  -> (4, 152, 294)
# output['mean']   -> (4, 38, 768)
# output['z']      -> (4, 38, 768)

# Or with a Nymeria batch (runs preprocessing automatically)
# output = model(batch)  # batch is a BatchedNymeriaTrainingSeq
```

### Training

```bash
python -m motion_vae.train --data_dir /path/to/nymeria/hdf5 --batch_size 32
```

> **Note:** This module uses relative imports and must be run as a package (`python -m motion_vae.train`), not as a script (`python motion_vae/train.py`).

## Architecture

The VAE compresses motion sequences with a 4x temporal downsampling factor using 1D causal convolutions:

```
Input (B, T, 294) ─── Encoder ──→ Latent (B, T/4, 768) ─── Decoder ──→ Output (B, T, 294)
```

**Encoder:**
```
Conv1d(294 → 512)                    # input projection
2x ResNetBlock1D(512)                # stage 1
DownsampleBlock(512 → 512, stride=2) # 2x downsample
2x ResNetBlock1D(512)                # stage 2
DownsampleBlock(512 → 512, stride=2) # 2x downsample (total 4x)
2x ResNetBlock1D(512)                # final stage
Conv1d(512 → 1536) → split → mean(768), logvar(768)
```

**Decoder:** Symmetric, replacing `DownsampleBlock` with `UpsampleBlock` (transposed convolutions).

All convolutions are **causal** (left-padded), so each output timestep only depends on current and past inputs. ~25.7M parameters total.

## Head-Centric Representation (294D)

Nymeria provides world-coordinate motion (23 XSens joints). The `NymeriaMotionPreprocessor` converts this into a head-centric representation that's better suited for learning:

| Feature  | Shape       | Dim | Description |
|----------|-------------|-----|-------------|
| `hr`     | (B, T, 6)     | 6   | Absolute head rotation (6D) |
| `hr_dot` | (B, T, 6)     | 6   | Frame-to-frame head rotation change (6D) |
| `hp`     | (B, T, 3)     | 3   | Absolute head position |
| `hp_dot` | (B, T, 3)     | 3   | Frame-to-frame head position change |
| `jp`     | (B, T, 23, 3) | 69  | Joint positions in head coordinate frame |
| `jv`     | (B, T, 23, 3) | 69  | Joint velocities in head coordinate frame |
| `jr`     | (B, T, 23, 6) | 138 | Joint rotations (6D) |
| **Total** |             | **294** | |

The 6D rotation representation (Zhou et al., "On the Continuity of Rotation Representations in Neural Networks") uses the first two columns of the rotation matrix, avoiding the discontinuities of Euler angles or quaternions.

## Loss Function

Following EgoTwin, the loss uses **grouped reconstruction** plus KL divergence:

```
L_VAE = (1/4) * Σ_c L_rec^c  +  λ_KL * L_KL
```

where *c* ranges over four feature groups, each computed as MSE:

| Group       | Indices   | Features |
|-------------|-----------|----------|
| `head_6D`   | 0–12      | `hr`, `hr_dot` |
| `head_3D`   | 12–18     | `hp`, `hp_dot` |
| `joint_3D`  | 18–156    | `jp`, `jv` |
| `joint_6D`  | 156–294   | `jr` |

Default λ_KL = 1e-4.

## File Structure

```
motion_vae/
├── __init__.py              # Package exports
├── config.py                # MotionVAEConfig, MotionVAETrainingConfig
├── rotation_utils.py        # matrix_to_rotation_6d, rotation_6d_to_matrix
├── motion_preprocessing.py  # HeadCentricMotion, NymeriaMotionPreprocessor
├── vae_model.py             # CausalConv1d, ResNetBlock1D, MotionEncoder, MotionDecoder, MotionVAE
└── train.py                 # Loss functions, MotionVAETrainer, CLI entry point
```

### Dependencies on the rest of the repo

- `transforms/` — `SO3` is available but not directly imported; `rotation_utils.py` implements 6D conversions with pure PyTorch for simplicity.
- `nymeria/nymeria_dataloader.py` — `NymeriaDataset`, `BatchedNymeriaTrainingSeq`, `nymeria_collate_fn` are used in `train.py`. This dependency requires `hdf5plugin` and `decord`, so `train.py` is **not** imported at the package level (to allow using the model code without those libraries installed).

## Training Stability Notes

Two measures prevent KL divergence from exploding during training:

1. **Logvar clamping:** The encoder clamps logvar to [-10, 10] before returning it.
2. **Output projection initialization:** Both the encoder's final projection (which produces mean/logvar) and the decoder's output projection are initialized with small weights (std=0.01, zero bias), so the model starts near zero output / zero KL.

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `input_dim` | 294 | Flattened head-centric features |
| `hidden_dim` | 512 | Conv channel width |
| `latent_dim` | 768 | Per-timestep latent dimension |
| `num_resnet_blocks` | 2 | ResNet blocks per stage |
| `sequence_length` | 152 | Must be divisible by 4 |
| `lambda_kl` | 1e-4 | KL weight |
| `learning_rate` | 1e-4 | AdamW with cosine decay |
| `gradient_clip` | 1.0 | Max gradient norm |
| `batch_size` | 32 | Suggested starting point |
