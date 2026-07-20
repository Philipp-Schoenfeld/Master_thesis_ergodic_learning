# Flow Matching Trajectory Generation

A modular PyTorch + JAX pipeline for learning and generating ergodic trajectories using **Conditional Flow Matching** with a **Patchified 1D U-Net** architecture.

## Overview

The core idea: learn a flow from Gaussian noise to structured trajectories (B-spline control points), then evaluate the generated control points as smooth B-spline curves using the `bsplinax` library.

### Architecture — `PatchUNetFlowNetwork`

```
Input x (B, nξ, nd)
  → ① Patchify (1D-CNN, kernel=1)   — nd → D per control point
  → ② Sinusoidal Time Embedding     — t ∈ [0,1] → R^D, broadcast added
  → ③ 1D U-Net Backbone             — Enc1/2/3 + Bottleneck + Dec1/2 with skip connections
  → ④ MLP Output Head               — D → nd velocity per control point
  → vθ (B, nξ, nd)
```

### Conditional Extension — `CondPatchUNetFlowNetwork`

A **shape-conditioned** variant that takes reference control points as additional input:

```
ref_cps (B, nξ, nd)
  → ShapeEncoder (Patchify + GlobalAvgPool + MLP) → R^D
  → combined conditioning: tokens += (TimeEmb(t) + ShapeEnc(ref_cps))
```

Train **once** on all shapes simultaneously; at inference, condition on any shape's reference to generate trajectories without retraining.

## Trajectory Datasets

| Dataset | Description |
|---|---|
| `polynomial_N` | N-shape from polynomial fit of 4 waypoints |
| `h_shape` | H-shape via piecewise-linear arc-length interpolation |
| `N_and_H` | N (left half) + H (right half) side-by-side |
| `ergodic_db` | Ergodic trajectories from Stein Variational Gradient Descent |
| `all_shapes` | Mixed batch of all above (for conditional training) |

## Usage

```bash
# Unconditional — single shape overfit
python flow_matching_runner.py --dataset polynomial_N --epochs 3000

# Unconditional — train and visualise all 4 shapes separately
python flow_matching_runner.py --run_all --epochs 3000

# Conditional — single model trained on all shapes, shape-conditioned at inference
python flow_matching_runner.py --model cond_patch_unet --epochs 5000
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `patch_unet` | `patch_unet` or `cond_patch_unet` |
| `--dataset` | `polynomial_N` | see table above |
| `--nxi` | `20` | number of B-spline control points |
| `--D` | `128` | token/embedding dimension |
| `--epochs` | `3000` | training epochs |
| `--bspline_deg` | `5` | B-spline degree for rendering |
| `--bspline_pts` | `512` | dense curve resolution |

## File Structure

```
thesis_architecture/
├── flow_matching_runner.py              # Main entrypoint (modular, argparse)
├── flow_matching_patch_unet.py          # PatchUNet architecture + unconditional CFM
├── flow_matching_cond_patch_unet.py     # Conditional model (ShapeEncoder + CondPatchUNet)
├── flow_matching_unet.py                # Earlier CNN-Attention UNet variant
├── flow_matching_trajectory_generation.py  # Simple MLP/CNN baselines
├── flow_matching_test.py                # Unit tests
├── data_loader.py                       # SQLite ergodic trajectory loader
└── Trajectory_data_generator/           # Ergodic DB generation scripts
```

## Dependencies

- **PyTorch** — flow matching training and model
- **JAX** — B-spline basis evaluation via `bsplinax`
- `bsplinax` (included as `../bsplinax-main/`)
- `matplotlib`, `numpy`, `pandas`, `sqlite3`

## Method

Conditional Flow Matching (CFM) with straight OT paths:

```
x_t = (1-t)·x_0 + t·x_1       (interpolated path)
u_t = x_1 - x_0                (target velocity)
L = E[||v_θ(x_t, t) - u_t||²] (MSE loss)
```

At inference, Euler integration from `x_0 ~ N(0,I)` to `x_1` (learned trajectory CPs).
