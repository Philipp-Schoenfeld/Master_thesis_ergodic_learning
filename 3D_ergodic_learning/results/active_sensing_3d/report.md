# 3D active sensing: start-conditioned planners vs linear rays

UCB target (kappa 1.0 -> 0.05), RBF-GP belief over the particles, 150 refinement iterations per round, <= 5 rounds, 30 candidates, sensing/coverage radius 0.06. E = Sinkhorn footprint error of the whole driven path against the TRUE density (lower is better); mean ± SEM over missions; Δ = paired difference to **linear rays** (95 % bootstrap CI).


## no prior samples

| method | n | E round 0 | E round 1 | E round 2 | E final | Δ final vs linear | swept final | path length | reached 99 % | rounds to 99 % |
|---|---|---|---|---|---|---|---|---|---|---|
| DiT-B 750 ep old DB | 27 | 3.92e-03 ± 7.6e-04 | 2.00e-03 ± 2.9e-04 | 1.89e-03 ± 2.9e-04 | 1.87e-03 ± 2.9e-04 | -3.50e-04 [-8.05e-04, +1.03e-04] | 0.995 | 11.3 | 93% | 2.74 |
| DiT-S old DB | 27 | 3.96e-03 ± 7.7e-04 | 1.94e-03 ± 2.9e-04 | 1.54e-03 ± 2.0e-04 | 1.49e-03 ± 1.8e-04 | -7.25e-04 [-1.20e-03, -2.86e-04] | 0.994 | 11.9 | 93% | 2.81 |
| DiT-S refined DB | 27 | 3.82e-03 ± 7.5e-04 | 2.33e-03 ± 4.6e-04 | 1.97e-03 ± 4.4e-04 | 1.94e-03 ± 4.4e-04 | -2.83e-04 [-9.54e-04, +6.00e-04] | 0.995 | 11.1 | 93% | 2.30 |
| U-Net D512 750 ep old DB | 27 | 3.96e-03 ± 7.6e-04 | 2.02e-03 ± 2.9e-04 | 1.60e-03 ± 2.0e-04 | 1.49e-03 ± 2.0e-04 | -7.26e-04 [-1.14e-03, -3.00e-04] | 0.996 | 11.6 | 96% | 2.81 |
| linear | 27 | 4.77e-03 ± 8.0e-04 | 3.40e-03 ± 4.1e-04 | 2.70e-03 ± 3.4e-04 | 2.22e-03 ± 2.6e-04 | – | 0.993 | 14.6 | 81% | 3.63 |

## ten known samples

| method | n | E round 0 | E round 1 | E round 2 | E final | Δ final vs linear | swept final | path length | reached 99 % | rounds to 99 % |
|---|---|---|---|---|---|---|---|---|---|---|
| DiT-B 750 ep old DB | 27 | 3.41e-03 ± 6.1e-04 | 2.26e-03 ± 3.7e-04 | 1.70e-03 ± 2.2e-04 | 1.59e-03 ± 2.1e-04 | -5.08e-04 [-9.42e-04, -7.70e-05] | 0.995 | 11.6 | 85% | 3.04 |
| DiT-S old DB | 27 | 3.39e-03 ± 6.1e-04 | 2.11e-03 ± 3.8e-04 | 1.43e-03 ± 2.2e-04 | 1.39e-03 ± 2.2e-04 | -7.05e-04 [-1.17e-03, -2.31e-04] | 0.995 | 11.9 | 96% | 2.85 |
| DiT-S refined DB | 27 | 3.23e-03 ± 6.0e-04 | 2.33e-03 ± 4.7e-04 | 2.09e-03 ± 4.4e-04 | 2.11e-03 ± 4.4e-04 | +7.73e-06 [-5.98e-04, +8.32e-04] | 0.997 | 11.5 | 100% | 2.30 |
| U-Net D512 750 ep old DB | 27 | 3.32e-03 ± 5.9e-04 | 1.74e-03 ± 2.4e-04 | 1.44e-03 ± 1.9e-04 | 1.48e-03 ± 2.0e-04 | -6.22e-04 [-1.02e-03, -2.43e-04] | 0.994 | 12.2 | 89% | 3.07 |
| linear | 27 | 4.34e-03 ± 6.9e-04 | 2.75e-03 ± 3.5e-04 | 2.22e-03 ± 2.5e-04 | 2.10e-03 ± 2.4e-04 | – | 0.993 | 13.7 | 78% | 3.44 |
