# Ergodicity validation

Median over 30 candidates at the final iteration. hell = Hellinger distance of the time-averaged visitation density to the target (floor = i.i.d. samples), tv4 = time vs. mass fraction on a 4x4 partition, low = extra time outside the support, cov = target mass within 0.05 of the path. FAIL lists the failed checks.

## A (hellinger floor 0.20)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.34 -> 0.19 | 0.05 | -0.00 | 0.93 | 0.02 | 3.5 | PASS |
| linear_adam_bspline | 0.77 -> 0.21 | 0.07 | 0.01 | 0.90 | 0.04 | 3.2 | PASS |
| linear_adam_waypoint | 0.77 -> 0.13 | 0.04 | 0.01 | 0.99 | 0.00 | 10.4 | PASS |
| linear_adam_waypoint_smooth | 0.77 -> 0.18 | 0.06 | 0.01 | 0.94 | 0.02 | 3.8 | PASS |

## blobs_75_25 (hellinger floor 0.18)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.34 -> 0.18 | 0.06 | 0.01 | 0.95 | 0.05 | 3.5 | PASS |
| linear_adam_bspline | 0.84 -> 0.20 | 0.07 | -0.01 | 0.92 | 0.04 | 2.2 | PASS |
| linear_adam_waypoint | 0.84 -> 0.15 | 0.07 | 0.01 | 0.98 | 0.00 | 9.1 | PASS |
| linear_adam_waypoint_smooth | 0.84 -> 0.19 | 0.06 | -0.02 | 0.92 | 0.03 | 2.5 | PASS |

## korean_5 (hellinger floor 0.23)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.42 -> 0.25 | 0.07 | 0.05 | 0.86 | 0.07 | 3.9 | PASS |
| linear_adam_bspline | 0.77 -> 0.27 | 0.07 | 0.06 | 0.85 | 0.12 | 4.3 | PASS |
| linear_adam_waypoint | 0.77 -> 0.15 | 0.07 | 0.03 | 0.99 | 0.00 | 14.5 | PASS |
| linear_adam_waypoint_smooth | 0.77 -> 0.23 | 0.07 | 0.06 | 0.91 | 0.06 | 5.2 | PASS |

## organic_30 (hellinger floor 0.14)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.23 -> 0.12 | 0.04 | -0.00 | 0.98 | 0.00 | 2.0 | PASS |
| linear_adam_bspline | 0.75 -> 0.15 | 0.04 | -0.00 | 0.98 | 0.01 | 2.0 | PASS |
| linear_adam_waypoint | 0.75 -> 0.11 | 0.04 | 0.01 | 1.00 | 0.00 | 6.0 | PASS |
| linear_adam_waypoint_smooth | 0.75 -> 0.13 | 0.04 | -0.00 | 0.98 | 0.01 | 2.4 | PASS |

## ring (hellinger floor 0.21)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.33 -> 0.19 | 0.06 | -0.01 | 0.93 | 0.02 | 3.5 | PASS |
| linear_adam_bspline | 0.84 -> 0.20 | 0.07 | 0.02 | 0.94 | 0.06 | 4.9 | PASS |
| linear_adam_waypoint | 0.84 -> 0.12 | 0.05 | -0.01 | 0.99 | 0.00 | 13.3 | PASS |
| linear_adam_waypoint_smooth | 0.84 -> 0.21 | 0.07 | 0.03 | 0.95 | 0.06 | 5.7 | PASS |

## sun_gmm3 (hellinger floor 0.22)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.38 -> 0.25 | 0.10 | 0.02 | 0.89 | 0.07 | 3.8 | PASS |
| linear_adam_bspline | 0.73 -> 0.26 | 0.09 | -0.00 | 0.86 | 0.08 | 3.2 | PASS |
| linear_adam_waypoint | 0.73 -> 0.15 | 0.05 | -0.00 | 0.97 | 0.00 | 10.4 | PASS |
| linear_adam_waypoint_smooth | 0.73 -> 0.23 | 0.09 | 0.01 | 0.91 | 0.05 | 4.1 | PASS |

## uniform (hellinger floor 0.33)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_adam_bspline | 0.48 -> 0.35 | 0.08 | 0.00 | 0.62 | 0.13 | 5.6 | FAIL: C7 coverage |
| linear_adam_bspline | 0.77 -> 0.38 | 0.09 | 0.00 | 0.61 | 0.27 | 5.8 | FAIL: C7 coverage |
| linear_adam_waypoint | 0.77 -> 0.20 | 0.05 | 0.00 | 0.95 | 0.01 | 17.0 | PASS |
| linear_adam_waypoint_smooth | 0.77 -> 0.31 | 0.06 | 0.00 | 0.72 | 0.12 | 7.4 | FAIL: C7 coverage |
