# Active sensing with different start-conditioned planners (as_models_20261028)

UCB target, belief ak, 100 TSVEC-SVGD iterations per round, <= 5 rounds, 25 shapes per cell. E = ergodic energy against the true density (lower is better); mean ± SEM over shapes; Δ = paired difference to **linear rays (no network)** (95 % bootstrap CI over shapes).

## no prior samples

| model | E round 0 | E round 1 | E round 2 | E final | Δ final vs ref | swept mass | path length | reached 99 % | rounds to 99 % |
|---|---|---|---|---|---|---|---|---|---|
| DiT λ=630 | 16.23 ± 1.93 | 12.80 ± 1.68 | 11.28 ± 1.52 | 10.13 ± 1.55 | +2.13 [+1.01, +3.66] | 0.995 | 23.5 | 84% | 3.95 |
| U-Net λ=630 | 15.90 ± 2.24 | 12.64 ± 1.84 | 10.96 ± 1.61 | 9.69 ± 1.55 | +1.69 [+0.55, +3.09] | 0.992 | 24.5 | 80% | 4.10 |
| U-Net λ=1300 (600 ep.) | 16.52 ± 2.16 | 13.09 ± 1.64 | 11.39 ± 1.46 | 9.87 ± 1.38 | +1.86 [+0.93, +2.80] | 0.994 | 25.2 | 88% | 4.18 |
| U-Net λ=1890 | 17.06 ± 2.52 | 12.71 ± 1.70 | 10.60 ± 1.46 | 9.24 ± 1.40 | +1.24 [+0.08, +2.50] | 0.996 | 25.2 | 92% | 4.13 |
| U-Net, no ErgLoss | 16.05 ± 2.16 | 12.05 ± 1.58 | 9.82 ± 1.30 | 9.05 ± 1.46 | +1.05 [-0.07, +2.36] | 0.993 | 23.0 | 76% | 4.11 |
| Diffusion U-Net λ=630 | 16.27 ± 2.12 | 12.52 ± 1.57 | 11.07 ± 1.46 | 9.92 ± 1.45 | +1.91 [+0.96, +2.90] | 0.993 | 24.7 | 84% | 4.05 |
| old start model (λ=500) | 16.22 ± 2.02 | 12.09 ± 1.69 | 10.14 ± 1.39 | 8.78 ± 1.38 | +0.78 [-0.57, +2.05] | 0.991 | 24.6 | 68% | 3.88 |
| linear rays (no network) | 14.82 ± 1.92 | 10.37 ± 1.44 | 8.48 ± 1.24 | 8.01 ± 1.14 | – | 0.992 | 25.4 | 76% | 3.21 |

## ten known samples

| model | E round 0 | E round 1 | E round 2 | E final | Δ final vs ref | swept mass | path length | reached 99 % | rounds to 99 % |
|---|---|---|---|---|---|---|---|---|---|
| DiT λ=630 | 15.19 ± 2.18 | 11.89 ± 1.64 | 10.66 ± 1.42 | 9.82 ± 1.46 | +2.41 [+1.75, +3.10] | 0.993 | 22.6 | 84% | 3.86 |
| U-Net λ=630 | 14.99 ± 2.10 | 11.30 ± 1.55 | 9.83 ± 1.29 | 8.68 ± 1.23 | +1.26 [+0.83, +1.74] | 0.993 | 24.0 | 76% | 4.00 |
| U-Net λ=1300 (600 ep.) | 14.76 ± 1.73 | 11.84 ± 1.48 | 10.25 ± 1.34 | 9.22 ± 1.33 | +1.80 [+1.09, +2.63] | 0.995 | 24.5 | 100% | 4.12 |
| U-Net λ=1890 | 14.61 ± 1.50 | 12.13 ± 1.54 | 9.96 ± 1.30 | 9.03 ± 1.32 | +1.61 [+1.15, +2.12] | 0.993 | 24.9 | 76% | 3.84 |
| U-Net, no ErgLoss | 13.36 ± 1.69 | 11.14 ± 1.45 | 9.58 ± 1.24 | 8.55 ± 1.26 | +1.14 [+0.57, +1.70] | 0.993 | 21.9 | 80% | 3.80 |
| Diffusion U-Net λ=630 | 15.05 ± 2.02 | 11.63 ± 1.62 | 10.27 ± 1.46 | 9.44 ± 1.47 | +2.03 [+1.38, +2.72] | 0.993 | 23.4 | 84% | 3.86 |
| old start model (λ=500) | 13.77 ± 1.80 | 10.26 ± 1.38 | 8.09 ± 1.06 | 7.62 ± 1.30 | +0.21 [-0.49, +1.02] | 0.992 | 24.9 | 72% | 4.17 |
| linear rays (no network) | 9.41 ± 1.39 | 8.34 ± 1.19 | 6.65 ± 1.25 | 7.41 ± 1.20 | – | 0.993 | 24.7 | 76% | 3.00 |

## Energy at equal cumulative path length

Per shape linear interpolation of E over the cumulative path length (rounds 0-2). n = shapes whose path covers the length in both models. Δ = paired difference to **linear rays (no network)**.

| path length | model | E | n | Δ vs ref (95 % CI) |
|---|---|---|---|---|
| no prior samples, L=9 | DiT λ=630 | 14.42 ± 1.78 | 25 | +0.28 [-0.63, +1.15] |
| no prior samples, L=9 | U-Net λ=630 | 14.23 ± 2.02 | 25 | +0.10 [-1.11, +1.33] |
| no prior samples, L=9 | U-Net λ=1300 (600 ep.) | 14.87 ± 1.89 | 25 | +0.74 [-0.27, +1.74] |
| no prior samples, L=9 | U-Net λ=1890 | 14.89 ± 2.05 | 25 | +0.76 [-0.56, +2.10] |
| no prior samples, L=9 | U-Net, no ErgLoss | 13.81 ± 1.83 | 25 | -0.33 [-1.04, +0.38] |
| no prior samples, L=9 | Diffusion U-Net λ=630 | 14.52 ± 1.88 | 25 | +0.39 [-0.84, +1.62] |
| no prior samples, L=9 | old start model (λ=500) | 14.36 ± 1.81 | 25 | +0.23 [-0.65, +1.17] |
| no prior samples, L=9 | linear rays (no network) | 14.13 ± 1.84 | 25 | – |
| no prior samples, L=12 | DiT λ=630 | 12.69 ± 1.67 | 25 | +0.38 [-0.55, +1.25] |
| no prior samples, L=12 | U-Net λ=630 | 12.56 ± 1.82 | 25 | +0.25 [-0.71, +1.19] |
| no prior samples, L=12 | U-Net λ=1300 (600 ep.) | 13.09 ± 1.63 | 25 | +0.78 [-0.01, +1.55] |
| no prior samples, L=12 | U-Net λ=1890 | 12.79 ± 1.69 | 25 | +0.47 [-0.58, +1.42] |
| no prior samples, L=12 | U-Net, no ErgLoss | 11.04 ± 1.41 | 24 | -0.43 [-1.14, +0.30] |
| no prior samples, L=12 | Diffusion U-Net λ=630 | 12.66 ± 1.64 | 25 | +0.34 [-0.67, +1.27] |
| no prior samples, L=12 | old start model (λ=500) | 12.21 ± 1.68 | 25 | -0.11 [-0.94, +0.70] |
| no prior samples, L=12 | linear rays (no network) | 12.31 ± 1.63 | 25 | – |
| no prior samples, L=15 | DiT λ=630 | 11.82 ± 1.57 | 25 | +1.26 [+0.42, +2.21] |
| no prior samples, L=15 | U-Net λ=630 | 11.65 ± 1.70 | 25 | +1.10 [+0.15, +2.06] |
| no prior samples, L=15 | U-Net λ=1300 (600 ep.) | 12.16 ± 1.53 | 25 | +1.60 [+0.82, +2.36] |
| no prior samples, L=15 | U-Net λ=1890 | 11.66 ± 1.57 | 25 | +1.11 [+0.13, +2.09] |
| no prior samples, L=15 | U-Net, no ErgLoss | 10.17 ± 1.32 | 24 | +0.29 [-0.35, +0.86] |
| no prior samples, L=15 | Diffusion U-Net λ=630 | 11.77 ± 1.52 | 25 | +1.21 [+0.38, +2.03] |
| no prior samples, L=15 | old start model (λ=500) | 11.09 ± 1.54 | 25 | +0.54 [-0.39, +1.43] |
| no prior samples, L=15 | linear rays (no network) | 10.55 ± 1.45 | 25 | – |
| ten known samples, L=9 | DiT λ=630 | 13.33 ± 1.84 | 18 | +4.04 [+2.54, +5.74] |
| ten known samples, L=9 | U-Net λ=630 | 12.91 ± 1.74 | 18 | +3.59 [+2.25, +5.05] |
| ten known samples, L=9 | U-Net λ=1300 (600 ep.) | 13.38 ± 1.59 | 18 | +4.64 [+3.06, +6.28] |
| ten known samples, L=9 | U-Net λ=1890 | 13.50 ± 1.50 | 18 | +4.64 [+3.17, +6.11] |
| ten known samples, L=9 | U-Net, no ErgLoss | 12.21 ± 1.54 | 18 | +2.90 [+1.20, +4.67] |
| ten known samples, L=9 | Diffusion U-Net λ=630 | 13.29 ± 1.83 | 18 | +4.54 [+2.48, +6.85] |
| ten known samples, L=9 | old start model (λ=500) | 11.47 ± 1.49 | 18 | +2.47 [+1.00, +4.13] |
| ten known samples, L=9 | linear rays (no network) | 8.65 ± 1.16 | 18 | – |
| ten known samples, L=12 | DiT λ=630 | 11.80 ± 1.63 | 25 | +2.93 [+1.89, +4.09] |
| ten known samples, L=12 | U-Net λ=630 | 11.16 ± 1.50 | 25 | +2.30 [+1.52, +3.10] |
| ten known samples, L=12 | U-Net λ=1300 (600 ep.) | 11.93 ± 1.46 | 25 | +3.06 [+2.00, +4.11] |
| ten known samples, L=12 | U-Net λ=1890 | 12.28 ± 1.52 | 25 | +3.42 [+2.38, +4.44] |
| ten known samples, L=12 | U-Net, no ErgLoss | 10.96 ± 1.40 | 25 | +2.09 [+0.86, +3.45] |
| ten known samples, L=12 | Diffusion U-Net λ=630 | 11.60 ± 1.61 | 25 | +2.73 [+1.69, +3.81] |
| ten known samples, L=12 | old start model (λ=500) | 9.23 ± 1.15 | 24 | +0.85 [-0.01, +1.74] |
| ten known samples, L=12 | linear rays (no network) | 8.87 ± 1.27 | 25 | – |
| ten known samples, L=15 | DiT λ=630 | 11.12 ± 1.50 | 23 | +2.55 [+1.68, +3.47] |
| ten known samples, L=15 | U-Net λ=630 | 10.34 ± 1.35 | 23 | +1.99 [+1.32, +2.69] |
| ten known samples, L=15 | U-Net λ=1300 (600 ep.) | 11.03 ± 1.39 | 23 | +2.53 [+1.66, +3.56] |
| ten known samples, L=15 | U-Net λ=1890 | 11.08 ± 1.40 | 23 | +2.60 [+1.67, +3.52] |
| ten known samples, L=15 | U-Net, no ErgLoss | 10.05 ± 1.29 | 23 | +1.70 [+0.81, +2.60] |
| ten known samples, L=15 | Diffusion U-Net λ=630 | 10.82 ± 1.52 | 23 | +2.38 [+1.47, +3.31] |
| ten known samples, L=15 | old start model (λ=500) | 8.50 ± 1.09 | 22 | +0.79 [-0.00, +1.66] |
| ten known samples, L=15 | linear rays (no network) | 8.20 ± 1.26 | 23 | – |
