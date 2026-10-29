# Active sensing: new models next to the earlier runs

Same 25 shapes, UCB (kappa 1.0 -> 0.05), full execution, <= 5 rounds, TSVEC-SVGD. Every planner is compared to the linear rays of its own run (paired over shapes). E = ergodic energy against the true density (lower is better). Equal-path column: E at cumulative path length 15, interpolated between rounds. Δ = planner − linear rays (95 % bootstrap CI over shapes); negative = better than the rays.

## 50 it · rbf  (iterations 50, candidates 5, belief rbf)

| condition | method | E final | Δ final vs rays | E at L=15 | Δ at L=15 vs rays | n (equal path) |
|---|---|---|---|---|---|---|
| no prior samples | old start model (default ckpt) | 11.59 ± 1.86 | -0.14 [-0.89, +0.62] | 14.26 | +0.95 [+0.28, +1.64] | 22 |
| no prior samples | EBM init | 11.74 ± 1.70 | +0.01 [-0.98, +0.94] | 13.95 | +0.38 [-0.87, +1.53] | 22 |
| no prior samples | heuristic | 12.48 ± 1.87 | +0.75 [-0.14, +1.59] | 17.41 | +3.52 [+1.92, +5.28] | 6 |
| no prior samples | linear rays | 11.74 ± 1.74 | – | 11.69 | – | 22 |
| no prior samples | random walk | 10.19 ± 1.55 | -1.55 [-2.48, -0.57] | 11.46 | -1.97 [-3.38, -0.61] | 22 |
| no prior samples | self-supervised init | 12.87 ± 2.06 | +1.13 [+0.16, +2.24] | 15.82 | +2.09 [+1.29, +2.91] | 22 |
| ten known samples | old start model (default ckpt) | 10.91 ± 1.64 | -0.90 [-1.58, -0.26] | 13.48 | -0.73 [-1.58, +0.11] | 14 |
| ten known samples | EBM init | 11.47 ± 1.62 | -0.34 [-1.31, +0.60] | 12.94 | -0.93 [-2.88, +0.85] | 16 |
| ten known samples | heuristic | 11.93 ± 1.94 | +0.12 [-0.60, +0.89] | 14.23 | +0.00 [+0.00, +0.00] | 1 |
| ten known samples | linear rays | 11.81 ± 1.86 | – | 13.79 | – | 17 |
| ten known samples | random walk | 10.38 ± 1.59 | -1.42 [-2.39, -0.42] | 12.28 | -1.55 [-2.56, -0.48] | 17 |
| ten known samples | self-supervised init | 12.44 ± 1.88 | +0.63 [-0.20, +1.36] | 15.75 | +2.54 [+0.98, +4.10] | 17 |

## 1500 it · rbf  (iterations 1500, candidates 30, belief rbf)

| condition | method | E final | Δ final vs rays | E at L=15 | Δ at L=15 vs rays | n (equal path) |
|---|---|---|---|---|---|---|
| no prior samples | old start model (default ckpt) | 10.30 ± 1.46 | -1.38 [-2.33, -0.52] | 13.40 | +0.18 [-1.31, +1.55] | 25 |
| no prior samples | EBM init | 11.34 ± 1.61 | -0.34 [-1.26, +0.49] | 13.40 | +0.18 [-1.00, +1.30] | 25 |
| no prior samples | heuristic | 11.21 ± 1.70 | -0.47 [-1.17, +0.27] | 12.93 | -0.29 [-1.35, +0.74] | 25 |
| no prior samples | linear rays | 11.68 ± 1.79 | – | 13.22 | – | 25 |
| no prior samples | random walk | 10.86 ± 1.61 | -0.82 [-1.87, +0.07] | 12.93 | -0.29 [-1.49, +0.92] | 25 |
| no prior samples | self-supervised init | 10.79 ± 1.89 | -0.89 [-2.21, +0.39] | 12.36 | -0.87 [-2.20, +0.37] | 25 |
| ten known samples | old start model (default ckpt) | 10.82 ± 1.80 | -0.53 [-1.33, +0.28] | 12.85 | -0.30 [-1.41, +0.82] | 25 |
| ten known samples | EBM init | 11.38 ± 1.80 | +0.03 [-0.63, +0.74] | 13.48 | +0.33 [-0.66, +1.32] | 25 |
| ten known samples | heuristic | 10.60 ± 1.94 | -0.75 [-1.66, +0.21] | 13.21 | +0.06 [-0.68, +0.84] | 25 |
| ten known samples | linear rays | 11.35 ± 1.78 | – | 13.15 | – | 25 |
| ten known samples | random walk | 11.03 ± 1.96 | -0.32 [-1.11, +0.57] | 13.03 | -0.12 [-1.03, +0.85] | 25 |
| ten known samples | self-supervised init | 11.35 ± 1.74 | +0.00 [-0.79, +0.80] | 13.33 | +0.18 [-0.81, +1.15] | 25 |

## 1500 it · rbf (23.10.)  (iterations 1500, candidates 30, belief rbf)

| condition | method | E final | Δ final vs rays | E at L=15 | Δ at L=15 vs rays | n (equal path) |
|---|---|---|---|---|---|---|
| no prior samples | old start model (default ckpt) | 10.26 ± 1.72 | -0.57 [-1.58, +0.15] | 12.15 | -0.66 [-1.98, +0.44] | 25 |
| no prior samples | linear rays | 10.83 ± 2.04 | – | 12.81 | – | 25 |
| no prior samples | self-supervised init | 10.60 ± 1.72 | -0.23 [-1.27, +0.65] | 12.79 | -0.01 [-0.75, +0.70] | 25 |
| ten known samples | old start model (default ckpt) | 10.20 ± 1.65 | +0.47 [-0.32, +1.30] | 11.59 | +0.28 [-0.51, +1.11] | 25 |
| ten known samples | linear rays | 9.73 ± 1.50 | – | 11.31 | – | 25 |
| ten known samples | self-supervised init | 10.10 ± 1.62 | +0.37 [-0.31, +1.08] | 11.94 | +0.63 [-0.59, +1.69] | 25 |

## 1500 it · ak  (iterations 1500, candidates 30, belief ak)

| condition | method | E final | Δ final vs rays | E at L=15 | Δ at L=15 vs rays | n (equal path) |
|---|---|---|---|---|---|---|
| no prior samples | old start model (default ckpt) | 7.78 ± 1.18 | -0.53 [-1.34, +0.19] | 10.77 | -0.48 [-1.16, +0.21] | 25 |
| no prior samples | linear rays | 8.31 ± 1.33 | – | 11.25 | – | 25 |
| no prior samples | self-supervised init | 7.95 ± 1.43 | -0.36 [-1.54, +0.89] | 11.04 | -0.21 [-1.12, +0.75] | 25 |
| ten known samples | old start model (default ckpt) | 6.81 ± 1.13 | +0.00 [-0.55, +0.61] | 8.34 | +0.13 [-0.73, +1.13] | 24 |
| ten known samples | linear rays | 6.81 ± 1.11 | – | 8.42 | – | 25 |
| ten known samples | self-supervised init | 6.62 ± 1.10 | -0.19 [-0.78, +0.45] | 9.34 | +0.92 [+0.01, +1.89] | 25 |

## 100 it · ak (new)  (iterations 100, candidates 30, belief ak)

| condition | method | E final | Δ final vs rays | E at L=15 | Δ at L=15 vs rays | n (equal path) |
|---|---|---|---|---|---|---|
| no prior samples | Diffusion U-Net λ=630 | 9.92 ± 1.45 | +1.91 [+0.96, +2.90] | 11.77 | +1.21 [+0.38, +2.03] | 25 |
| no prior samples | DiT λ=630 | 10.13 ± 1.55 | +2.13 [+1.01, +3.66] | 11.82 | +1.26 [+0.42, +2.21] | 25 |
| no prior samples | linear rays (no network) | 8.01 ± 1.14 | – | 10.55 | – | 25 |
| no prior samples | U-Net λ=1300 (600 ep.) | 9.87 ± 1.38 | +1.86 [+0.93, +2.80] | 12.16 | +1.60 [+0.82, +2.36] | 25 |
| no prior samples | old start model (λ=500) | 8.78 ± 1.38 | +0.78 [-0.57, +2.05] | 11.09 | +0.54 [-0.39, +1.43] | 25 |
| no prior samples | U-Net, no ErgLoss | 9.05 ± 1.46 | +1.05 [-0.07, +2.36] | 10.17 | +0.29 [-0.35, +0.86] | 24 |
| no prior samples | U-Net λ=1890 | 9.24 ± 1.40 | +1.24 [+0.08, +2.50] | 11.66 | +1.11 [+0.13, +2.09] | 25 |
| no prior samples | U-Net λ=630 | 9.69 ± 1.55 | +1.69 [+0.55, +3.09] | 11.65 | +1.10 [+0.15, +2.06] | 25 |
| ten known samples | Diffusion U-Net λ=630 | 9.44 ± 1.47 | +2.03 [+1.38, +2.72] | 10.82 | +2.38 [+1.47, +3.31] | 23 |
| ten known samples | DiT λ=630 | 9.82 ± 1.46 | +2.41 [+1.75, +3.10] | 11.12 | +2.55 [+1.68, +3.47] | 23 |
| ten known samples | linear rays (no network) | 7.41 ± 1.20 | – | 8.20 | – | 23 |
| ten known samples | U-Net λ=1300 (600 ep.) | 9.22 ± 1.33 | +1.80 [+1.09, +2.63] | 11.03 | +2.53 [+1.66, +3.56] | 23 |
| ten known samples | old start model (λ=500) | 7.62 ± 1.30 | +0.21 [-0.49, +1.02] | 8.50 | +0.79 [-0.00, +1.66] | 22 |
| ten known samples | U-Net, no ErgLoss | 8.55 ± 1.26 | +1.14 [+0.57, +1.70] | 10.05 | +1.70 [+0.81, +2.60] | 23 |
| ten known samples | U-Net λ=1890 | 9.03 ± 1.32 | +1.61 [+1.15, +2.12] | 11.08 | +2.60 [+1.67, +3.52] | 23 |
| ten known samples | U-Net λ=630 | 8.68 ± 1.23 | +1.26 [+0.83, +1.74] | 10.34 | +1.99 [+1.32, +2.69] | 23 |

Cells with n < 12 shapes in the equal-path column are not plotted (they cover too few shapes).
