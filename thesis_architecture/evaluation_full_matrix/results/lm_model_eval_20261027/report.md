# Start-conditioned CFM evaluation `lm_model_eval_20261027`

144 holdout cases, 10 samples each, 15.8 min.

## A: generation quality (mean over samples, then cases)

| group | method | cases | Hellinger | best-of-N Hellinger | K=10 error | smooth | outside | start err |
|---|---|---|---|---|---|---|---|---|
| split=val | chord | 48 | 0.828 | 0.738 | 9.87e+01 | 0.000 | 0.000 | 0.0e+00 |
| split=val | ground_truth | 48 | 0.268 | 0.268 | 1.07e-01 | 0.102 | 0.000 | 1.6e-08 |
| split=val | new | 48 | 0.334 | 0.305 | 9.95e-01 | 0.136 | 0.005 | 1.6e-08 |
| split=val | old | 48 | 0.409 | 0.375 | 2.35e+00 | 0.128 | 0.003 | 1.6e-08 |
| split=val_start | chord | 96 | 0.830 | 0.745 | 9.53e+01 | 0.000 | 0.000 | 0.0e+00 |
| split=val_start | ground_truth | 96 | 0.268 | 0.268 | 1.04e-01 | 0.102 | 0.000 | 1.4e-08 |
| split=val_start | new | 96 | 0.333 | 0.302 | 1.16e+00 | 0.141 | 0.006 | 1.4e-08 |
| kind=base | chord | 72 | 0.832 | 0.736 | 9.26e+01 | 0.000 | 0.000 | 0.0e+00 |
| kind=base | ground_truth | 72 | 0.239 | 0.239 | 8.39e-02 | 0.090 | 0.000 | 1.6e-08 |
| kind=base | new | 72 | 0.313 | 0.282 | 1.18e+00 | 0.135 | 0.002 | 1.6e-08 |
| kind=base | old | 24 | 0.383 | 0.354 | 1.94e+00 | 0.109 | 0.002 | 1.6e-08 |
| kind=unk0 | chord | 72 | 0.826 | 0.749 | 1.00e+02 | 0.000 | 0.000 | 0.0e+00 |
| kind=unk0 | ground_truth | 72 | 0.297 | 0.297 | 1.25e-01 | 0.114 | 0.000 | 1.4e-08 |
| kind=unk0 | new | 72 | 0.353 | 0.325 | 1.02e+00 | 0.144 | 0.009 | 1.4e-08 |
| kind=unk0 | old | 24 | 0.434 | 0.396 | 2.76e+00 | 0.146 | 0.005 | 1.6e-08 |
| start=new | chord | 96 | 0.830 | 0.745 | 9.53e+01 | 0.000 | 0.000 | 0.0e+00 |
| start=new | ground_truth | 96 | 0.268 | 0.268 | 1.04e-01 | 0.102 | 0.000 | 1.4e-08 |
| start=new | new | 96 | 0.333 | 0.302 | 1.16e+00 | 0.141 | 0.006 | 1.4e-08 |
| start=stored | chord | 48 | 0.828 | 0.738 | 9.87e+01 | 0.000 | 0.000 | 0.0e+00 |
| start=stored | ground_truth | 48 | 0.268 | 0.268 | 1.07e-01 | 0.102 | 0.000 | 1.6e-08 |
| start=stored | new | 48 | 0.334 | 0.305 | 9.95e-01 | 0.136 | 0.005 | 1.6e-08 |
| start=stored | old | 48 | 0.409 | 0.375 | 2.35e+00 | 0.128 | 0.003 | 1.6e-08 |

## New vs old model, paired on the stored starts

| metric | new | old | new better in |
|---|---|---|---|
| hellinger | 0.334 | 0.4089 | 100% of 48 |
| ergodic_k10 | 0.9947 | 2.349 | 92% of 48 |
| smooth | 0.1362 | 0.1277 | 46% of 48 |
| outside | 0.005273 | 0.003402 | 19% of 48 |

## C: start conditioning

```
{
 "start_err_max": 3.925959994697482e-08,
 "start_err_mean": 1.5084193885962145e-08,
 "route_dist_between_starts": 0.37820949272086807,
 "route_dist_within_case": 0.286481058981804,
 "grid_start_err_max": 3.371747884011707e-08,
 "grid_outside_mean": 0.00126953125
}
```

## B: warm start (LM, K=16, w=15)

| init | H@0 | H@5 | H@10 | H@25 | H@50 | H@100 | H@200 |
|---|---|---|---|---|---|---|---|
| new | 0.316 | 0.204 | 0.201 | 0.201 | 0.201 | 0.200 | 0.200 |
| old | 0.387 | 0.210 | 0.204 | 0.203 | 0.203 | 0.202 | 0.201 |
| chord | 0.830 | 0.344 | 0.207 | 0.194 | 0.194 | 0.193 | 0.193 |

Iterations to match the Hellinger distance chords reach after 100 iterations:

```
{
 "new": {
  "median_iters": Infinity,
  "share_reached_within_budget": 0.25,
  "share_at_iter0": 0.0
 },
 "old": {
  "median_iters": Infinity,
  "share_reached_within_budget": 0.3333333333333333,
  "share_at_iter0": 0.0
 },
 "chord": {
  "median_iters": 100.0,
  "share_reached_within_budget": 1.0,
  "share_at_iter0": 0.0
 }
}
```