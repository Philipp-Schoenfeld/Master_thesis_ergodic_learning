# Start-conditioned CFM comparison `cfm_models_20261028`

144 holdout cases x 30 samples, 100 Euler steps; warm start: 90 initial curves per density; same particles and flow noise for every model.

## Models

| label | backbone | lambda_erg | epochs | params | eval GPU | t per plan() [s] |
|---|---|---|---|---|---|---|
| dit-l630 | dit | 630 | 179 | 40.0M | NVIDIA RTX A5000 | 2.376 |
| unet-l0 | unet | 0 | 179 | 87.5M | NVIDIA RTX A5000 | 7.460 |
| unet-l210 | unet | 210 | 179 | 87.5M | NVIDIA GeForce RTX 3090 | 6.434 |
| unet-l630 | unet | 630 | 179 | 87.5M | NVIDIA GeForce RTX 3090 | 6.453 |
| unet-l1300-600ep | unet | 1300 | 599 | 87.5M | NVIDIA GeForce RTX 3090 | 6.465 |
| unet-l1890 | unet | 1890 | 179 | 87.5M | NVIDIA RTX A5000 | 7.459 |
| diff-l630 | unet-diff | 630 | 179 | 87.5M | NVIDIA RTX A5000 | 7.607 |

Plan time = median wall-clock of one call generating all samples of a case; only comparable between models evaluated on the same GPU type.

## A: generation quality (mean over samples, then cases)

| group | method | cases | Hellinger | best-of-N Hell. | K=10 error | coverage | smooth | jerk RMS | outside | start err |
|---|---|---|---|---|---|---|---|---|---|---|
| all | ground_truth | 144 | 0.268 | 0.268 | 0.105 | 0.812 | 0.102 | 4.5e+03 | 0.0000 | 1.5e-08 |
| all | dit-l630 | 144 | 0.298 | 0.269 | 0.558 | 0.786 | 0.121 | 5.11e+03 | 0.0017 | 1.5e-08 |
| all | unet-l0 | 144 | 0.359 | 0.331 | 1.11 | 0.720 | 0.099 | 4.52e+03 | 0.0012 | 1.5e-08 |
| all | unet-l210 | 144 | 0.340 | 0.312 | 0.817 | 0.741 | 0.106 | 4.72e+03 | 0.0022 | 1.5e-08 |
| all | unet-l630 | 144 | 0.335 | 0.297 | 0.914 | 0.748 | 0.124 | 5.44e+03 | 0.0025 | 1.5e-08 |
| all | unet-l1300-600ep | 144 | 0.333 | 0.294 | 1 | 0.756 | 0.139 | 6.19e+03 | 0.0047 | 1.5e-08 |
| all | unet-l1890 | 144 | 0.344 | 0.302 | 1.19 | 0.750 | 0.166 | 7.09e+03 | 0.0099 | 1.5e-08 |
| all | diff-l630 | 144 | 0.336 | 0.297 | 0.982 | 0.748 | 0.125 | 5.17e+03 | 0.0025 | 1.5e-08 |
| all | chord | 144 | 0.829 | 0.737 | 96.4 | 0.093 | 0.000 | 4.39e-10 | 0.0000 | 0.0e+00 |
| split=val | ground_truth | 48 | 0.268 | 0.268 | 0.107 | 0.813 | 0.102 | 4.37e+03 | 0.0000 | 1.6e-08 |
| split=val | dit-l630 | 48 | 0.300 | 0.272 | 0.643 | 0.783 | 0.118 | 5.01e+03 | 0.0017 | 1.6e-08 |
| split=val | unet-l0 | 48 | 0.360 | 0.332 | 1.13 | 0.719 | 0.100 | 4.5e+03 | 0.0008 | 1.6e-08 |
| split=val | unet-l210 | 48 | 0.340 | 0.313 | 0.848 | 0.740 | 0.103 | 4.66e+03 | 0.0016 | 1.6e-08 |
| split=val | unet-l630 | 48 | 0.337 | 0.299 | 0.99 | 0.745 | 0.122 | 5.39e+03 | 0.0025 | 1.6e-08 |
| split=val | unet-l1300-600ep | 48 | 0.333 | 0.296 | 1.09 | 0.755 | 0.137 | 6.12e+03 | 0.0047 | 1.6e-08 |
| split=val | unet-l1890 | 48 | 0.346 | 0.304 | 1.28 | 0.747 | 0.162 | 6.96e+03 | 0.0098 | 1.6e-08 |
| split=val | diff-l630 | 48 | 0.339 | 0.301 | 1.07 | 0.746 | 0.126 | 5.06e+03 | 0.0028 | 1.6e-08 |
| split=val | chord | 48 | 0.827 | 0.731 | 98.7 | 0.095 | 0.000 | 4.48e-10 | 0.0000 | 0.0e+00 |
| split=val_start | ground_truth | 96 | 0.268 | 0.268 | 0.104 | 0.812 | 0.102 | 4.56e+03 | 0.0000 | 1.4e-08 |
| split=val_start | dit-l630 | 96 | 0.297 | 0.268 | 0.515 | 0.787 | 0.122 | 5.16e+03 | 0.0017 | 1.4e-08 |
| split=val_start | unet-l0 | 96 | 0.358 | 0.330 | 1.09 | 0.721 | 0.099 | 4.53e+03 | 0.0015 | 1.4e-08 |
| split=val_start | unet-l210 | 96 | 0.340 | 0.312 | 0.802 | 0.741 | 0.108 | 4.74e+03 | 0.0025 | 1.4e-08 |
| split=val_start | unet-l630 | 96 | 0.334 | 0.295 | 0.877 | 0.749 | 0.126 | 5.47e+03 | 0.0025 | 1.4e-08 |
| split=val_start | unet-l1300-600ep | 96 | 0.332 | 0.293 | 0.957 | 0.756 | 0.140 | 6.22e+03 | 0.0048 | 1.4e-08 |
| split=val_start | unet-l1890 | 96 | 0.343 | 0.301 | 1.14 | 0.751 | 0.167 | 7.16e+03 | 0.0099 | 1.4e-08 |
| split=val_start | diff-l630 | 96 | 0.335 | 0.295 | 0.935 | 0.750 | 0.125 | 5.23e+03 | 0.0024 | 1.4e-08 |
| split=val_start | chord | 96 | 0.830 | 0.741 | 95.3 | 0.092 | 0.000 | 4.35e-10 | 0.0000 | 0.0e+00 |
| kind=base | ground_truth | 72 | 0.239 | 0.239 | 0.0839 | 0.866 | 0.090 | 4.18e+03 | 0.0000 | 1.6e-08 |
| kind=base | dit-l630 | 72 | 0.276 | 0.246 | 0.578 | 0.835 | 0.116 | 4.88e+03 | 0.0008 | 1.6e-08 |
| kind=base | unet-l0 | 72 | 0.362 | 0.328 | 1.34 | 0.750 | 0.091 | 4.25e+03 | 0.0007 | 1.6e-08 |
| kind=base | unet-l210 | 72 | 0.331 | 0.299 | 0.935 | 0.780 | 0.102 | 4.48e+03 | 0.0012 | 1.6e-08 |
| kind=base | unet-l630 | 72 | 0.316 | 0.276 | 0.925 | 0.796 | 0.122 | 5.16e+03 | 0.0017 | 1.6e-08 |
| kind=base | unet-l1300-600ep | 72 | 0.311 | 0.270 | 0.991 | 0.805 | 0.134 | 5.8e+03 | 0.0025 | 1.6e-08 |
| kind=base | unet-l1890 | 72 | 0.321 | 0.277 | 1.14 | 0.803 | 0.162 | 6.82e+03 | 0.0058 | 1.6e-08 |
| kind=base | diff-l630 | 72 | 0.318 | 0.276 | 1 | 0.796 | 0.120 | 4.92e+03 | 0.0014 | 1.6e-08 |
| kind=base | chord | 72 | 0.832 | 0.727 | 92.6 | 0.098 | 0.000 | 4.4e-10 | 0.0000 | 0.0e+00 |
| kind=unk0 | ground_truth | 72 | 0.297 | 0.297 | 0.125 | 0.759 | 0.114 | 4.81e+03 | 0.0000 | 1.4e-08 |
| kind=unk0 | dit-l630 | 72 | 0.320 | 0.293 | 0.537 | 0.736 | 0.126 | 5.34e+03 | 0.0026 | 1.4e-08 |
| kind=unk0 | unet-l0 | 72 | 0.356 | 0.333 | 0.874 | 0.690 | 0.107 | 4.79e+03 | 0.0018 | 1.4e-08 |
| kind=unk0 | unet-l210 | 72 | 0.348 | 0.325 | 0.7 | 0.701 | 0.111 | 4.95e+03 | 0.0031 | 1.4e-08 |
| kind=unk0 | unet-l630 | 72 | 0.354 | 0.318 | 0.904 | 0.700 | 0.127 | 5.72e+03 | 0.0033 | 1.4e-08 |
| kind=unk0 | unet-l1300-600ep | 72 | 0.354 | 0.318 | 1.01 | 0.706 | 0.144 | 6.57e+03 | 0.0070 | 1.4e-08 |
| kind=unk0 | unet-l1890 | 72 | 0.368 | 0.327 | 1.24 | 0.697 | 0.170 | 7.36e+03 | 0.0140 | 1.4e-08 |
| kind=unk0 | diff-l630 | 72 | 0.354 | 0.317 | 0.961 | 0.701 | 0.131 | 5.42e+03 | 0.0036 | 1.4e-08 |
| kind=unk0 | chord | 72 | 0.826 | 0.747 | 100 | 0.088 | 0.000 | 4.38e-10 | 0.0000 | 0.0e+00 |

## Paired against `unet-l630` (model minus reference, mean over cases, 95% cluster-bootstrap CI over densities; negative = better)

| model | metric | difference | 95% CI | model better in |
|---|---|---|---|---|
| dit-l630 | hellinger | -0.03705 | [-0.03955, -0.03469] | 100% |
| dit-l630 | ergodic_k10 | -0.3566 | [-0.3879, -0.3281] | 100% |
| dit-l630 | smooth | -0.003794 | [-0.007295, -0.0005514] | 60% |
| dit-l630 | outside | -0.0007831 | [-0.001351, -0.0002984] | 41% |
| unet-l0 | hellinger | +0.02408 | [+0.01245, +0.0375] | 27% |
| unet-l0 | ergodic_k10 | +0.1911 | [+0.03503, +0.3552] | 41% |
| unet-l0 | smooth | -0.02554 | [-0.03159, -0.01932] | 79% |
| unet-l0 | outside | -0.001266 | [-0.002215, -0.0003707] | 53% |
| unet-l210 | hellinger | +0.00486 | [-0.0001748, +0.01055] | 49% |
| unet-l210 | ergodic_k10 | -0.09702 | [-0.1622, -0.02584] | 65% |
| unet-l210 | smooth | -0.01801 | [-0.02379, -0.01225] | 78% |
| unet-l210 | outside | -0.0003273 | [-0.001031, +0.0003382] | 42% |
| unet-l1300-600ep | hellinger | -0.0025 | [-0.004172, -0.0007293] | 60% |
| unet-l1300-600ep | ergodic_k10 | +0.08577 | [+0.06464, +0.1066] | 24% |
| unet-l1300-600ep | smooth | +0.01434 | [+0.01132, +0.01726] | 16% |
| unet-l1300-600ep | outside | +0.002248 | [+0.001237, +0.003498] | 18% |
| unet-l1890 | hellinger | +0.009143 | [+0.006557, +0.01189] | 20% |
| unet-l1890 | ergodic_k10 | +0.2719 | [+0.234, +0.3117] | 6% |
| unet-l1890 | smooth | +0.04115 | [+0.03382, +0.04868] | 4% |
| unet-l1890 | outside | +0.00738 | [+0.004805, +0.01022] | 6% |
| diff-l630 | hellinger | +0.001129 | [-0.0006606, +0.003147] | 44% |
| diff-l630 | ergodic_k10 | +0.06713 | [+0.04268, +0.09079] | 30% |
| diff-l630 | smooth | +0.0009207 | [-0.001754, +0.003513] | 42% |
| diff-l630 | outside | +3.617e-06 | [-0.0005226, +0.0006311] | 33% |

## B: warm start (Levenberg-Marquardt, K=16, w=15), Hellinger after n iterations

| init | 0 | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| dit-l630 | 0.297 | 0.236 | 0.233 | 0.231 | 0.231 | 0.230 | 0.229 |
| unet-l0 | 0.357 | 0.246 | 0.240 | 0.238 | 0.237 | 0.236 | 0.235 |
| unet-l210 | 0.339 | 0.243 | 0.237 | 0.235 | 0.234 | 0.234 | 0.232 |
| unet-l630 | 0.334 | 0.240 | 0.235 | 0.233 | 0.232 | 0.231 | 0.230 |
| unet-l1300-600ep | 0.331 | 0.235 | 0.230 | 0.229 | 0.228 | 0.227 | 0.226 |
| unet-l1890 | 0.342 | 0.232 | 0.228 | 0.226 | 0.225 | 0.225 | 0.224 |
| diff-l630 | 0.336 | 0.237 | 0.232 | 0.230 | 0.229 | 0.228 | 0.227 |
| chord | 0.827 | 0.334 | 0.223 | 0.213 | 0.213 | 0.212 | 0.212 |

LM iterations until the Hellinger distance chords reach after 100 iterations (per density):

| init | median iterations | reached within 200 | already at 0 |
|---|---|---|---|
| dit-l630 | inf | 12% | 0% |
| unet-l0 | inf | 15% | 0% |
| unet-l210 | inf | 15% | 0% |
| unet-l630 | inf | 10% | 0% |
| unet-l1300-600ep | inf | 15% | 0% |
| unet-l1890 | inf | 25% | 0% |
| diff-l630 | inf | 17% | 0% |
| chord | 100 | 100% | 0% |

## C: start conditioning

```
{
 "dit-l630": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.23842072486877441,
  "route_dist_between_starts": 0.36728668212890625,
  "t_sample_median_s": 2.375709746964276,
  "gpu": "NVIDIA RTX A5000",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.0005615234375,
  "grid_hellinger_mean": 0.26269592505486833
 },
 "unet-l0": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.0948953777551651,
  "route_dist_between_starts": 0.3603924512863159,
  "t_sample_median_s": 7.460441988077946,
  "gpu": "NVIDIA RTX A5000",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.0009033203125,
  "grid_hellinger_mean": 0.3460177938074495
 },
 "unet-l210": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.11240044981241226,
  "route_dist_between_starts": 0.36215659976005554,
  "t_sample_median_s": 6.434358158498071,
  "gpu": "NVIDIA GeForce RTX 3090",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.0012451171875,
  "grid_hellinger_mean": 0.3178175420792134
 },
 "unet-l630": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.2736441493034363,
  "route_dist_between_starts": 0.3698614537715912,
  "t_sample_median_s": 6.452701004498522,
  "gpu": "NVIDIA GeForce RTX 3090",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.001123046875,
  "grid_hellinger_mean": 0.30640705056410406
 },
 "unet-l1300-600ep": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.28938257694244385,
  "route_dist_between_starts": 0.3658512830734253,
  "t_sample_median_s": 6.465182949999871,
  "gpu": "NVIDIA GeForce RTX 3090",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.001318359375,
  "grid_hellinger_mean": 0.3028962993073569
 },
 "unet-l1890": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.301162987947464,
  "route_dist_between_starts": 0.3575078248977661,
  "t_sample_median_s": 7.458933673973661,
  "gpu": "NVIDIA RTX A5000",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.003603515625,
  "grid_hellinger_mean": 0.3123064468179704
 },
 "diff-l630": {
  "start_err_max": 3.925959994697482e-08,
  "route_dist_within_case": 0.30672505497932434,
  "route_dist_between_starts": 0.3682861328125,
  "t_sample_median_s": 7.60690028994577,
  "gpu": "NVIDIA RTX A5000",
  "grid_start_err_max": 3.371747884011707e-08,
  "grid_outside_mean": 0.000966796875,
  "grid_hellinger_mean": 0.3091231643802582
 }
}
```
