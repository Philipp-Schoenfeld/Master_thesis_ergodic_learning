# Ergodicity validation

Median over 30 candidates at the final iteration. hell = Hellinger distance of the time-averaged visitation density to the target (floor = i.i.d. samples), tv4 = time vs. mass fraction on a 4x4 partition, low = extra time outside the support, cov = target mass within 0.05 of the path. FAIL lists the failed checks.

## A (hellinger floor 0.20)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.34 -> 0.34 | 0.14 | -0.00 | 0.79 | 2.02 | 3.6 | PASS |
| linear_sun_pointmass | 0.77 -> 0.26 | 0.14 | 0.02 | 0.90 | 4.18 | 3.8 | PASS |
| linear_sun_jerk | 0.77 -> 0.36 | 0.18 | 0.05 | 0.78 | 5.53 | 2.8 | PASS |
| cfm_adam_bspline | 0.34 -> 0.24 | 0.07 | 0.03 | 0.91 | 0.12 | 4.5 | PASS |
| linear_adam_bspline | 0.77 -> 0.27 | 0.07 | 0.04 | 0.87 | 0.16 | 3.9 | PASS |
| linear_adam_waypoint | 0.77 -> 0.21 | 0.07 | 0.01 | 0.93 | 0.02 | 5.1 | PASS |
| linear_adam_waypoint_smooth | 0.77 -> 0.23 | 0.06 | 0.02 | 0.92 | 0.12 | 4.5 | PASS |
| cfm_bspline_svgd | 0.34 -> 0.20 | 0.05 | 0.01 | 0.92 | 0.05 | 3.6 | PASS |
| linear_bspline_svgd | 0.77 -> 0.16 | 0.04 | 0.01 | 0.97 | 0.06 | 6.0 | PASS |
| linear_waypoint_svgd | 0.77 -> 0.13 | 0.04 | 0.02 | 0.99 | 0.01 | 10.2 | PASS |
| linear_waypoint_svgd_smooth | 0.77 -> 0.17 | 0.05 | 0.01 | 0.97 | 0.06 | 5.6 | PASS |

## blobs_75_25 (hellinger floor 0.18)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.34 -> 0.34 | 0.28 | 0.03 | 0.84 | 4.19 | 3.1 | FAIL: C2 tv_4, C6 regions |
| linear_sun_pointmass | 0.84 -> 0.20 | 0.15 | -0.00 | 0.95 | 4.32 | 2.9 | FAIL: C6 regions |
| linear_sun_jerk | 0.84 -> 0.28 | 0.16 | 0.04 | 0.90 | 4.76 | 2.1 | FAIL: C6 regions |
| cfm_adam_bspline | 0.34 -> 0.22 | 0.06 | 0.03 | 0.94 | 0.14 | 4.2 | PASS |
| linear_adam_bspline | 0.84 -> 0.25 | 0.08 | 0.04 | 0.90 | 0.17 | 3.1 | PASS |
| linear_adam_waypoint | 0.84 -> 0.19 | 0.06 | 0.01 | 0.94 | 0.03 | 3.8 | PASS |
| linear_adam_waypoint_smooth | 0.84 -> 0.23 | 0.07 | 0.02 | 0.93 | 0.14 | 3.5 | PASS |
| cfm_bspline_svgd | 0.34 -> 0.21 | 0.08 | 0.02 | 0.94 | 0.29 | 4.2 | PASS |
| linear_bspline_svgd | 0.84 -> 0.19 | 0.06 | 0.00 | 0.94 | 0.05 | 3.4 | PASS |
| linear_waypoint_svgd | 0.84 -> 0.15 | 0.07 | 0.01 | 0.98 | 0.02 | 10.2 | PASS |
| linear_waypoint_svgd_smooth | 0.84 -> 0.19 | 0.05 | 0.01 | 0.95 | 0.14 | 3.5 | PASS |

## korean_5 (hellinger floor 0.23)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.42 -> 0.42 | 0.15 | 0.07 | 0.68 | 2.30 | 3.9 | FAIL: C7 coverage |
| linear_sun_pointmass | 0.77 -> 0.36 | 0.23 | 0.02 | 0.76 | 10.69 | 3.8 | PASS |
| linear_sun_jerk | 0.77 -> 0.57 | 0.41 | 0.07 | 0.47 | 27.60 | 2.3 | FAIL: C1 hellinger, C2 tv_4, C7 coverage |
| cfm_adam_bspline | 0.42 -> 0.29 | 0.07 | 0.07 | 0.85 | 0.18 | 4.9 | PASS |
| linear_adam_bspline | 0.77 -> 0.31 | 0.08 | 0.07 | 0.82 | 0.22 | 4.5 | PASS |
| linear_adam_waypoint | 0.77 -> 0.25 | 0.07 | 0.05 | 0.92 | 0.07 | 6.9 | PASS |
| linear_adam_waypoint_smooth | 0.77 -> 0.27 | 0.07 | 0.07 | 0.86 | 0.16 | 5.0 | PASS |
| cfm_bspline_svgd | 0.42 -> 0.27 | 0.08 | 0.06 | 0.85 | 0.12 | 4.1 | PASS |
| linear_bspline_svgd | 0.77 -> 0.21 | 0.07 | 0.05 | 0.93 | 0.09 | 5.9 | PASS |
| linear_waypoint_svgd | 0.77 -> 0.16 | 0.08 | 0.06 | 0.99 | 0.01 | 11.5 | PASS |
| linear_waypoint_svgd_smooth | 0.77 -> 0.25 | 0.07 | 0.06 | 0.90 | 0.12 | 5.6 | PASS |

## organic_30 (hellinger floor 0.14)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.23 -> 0.23 | 0.10 | -0.00 | 0.92 | 1.25 | 2.0 | PASS |
| linear_sun_pointmass | 0.75 -> 0.15 | 0.07 | 0.03 | 0.99 | 0.48 | 3.1 | PASS |
| linear_sun_jerk | 0.75 -> 0.23 | 0.10 | 0.07 | 0.97 | 1.90 | 2.2 | PASS |
| cfm_adam_bspline | 0.23 -> 0.19 | 0.05 | 0.02 | 0.98 | 0.07 | 3.8 | PASS |
| linear_adam_bspline | 0.75 -> 0.24 | 0.07 | 0.04 | 0.95 | 0.17 | 3.0 | PASS |
| linear_adam_waypoint | 0.75 -> 0.16 | 0.05 | 0.00 | 0.98 | 0.01 | 2.8 | PASS |
| linear_adam_waypoint_smooth | 0.75 -> 0.21 | 0.06 | 0.03 | 0.98 | 0.11 | 3.5 | PASS |
| cfm_bspline_svgd | 0.23 -> 0.14 | 0.04 | -0.00 | 0.98 | 0.02 | 2.3 | PASS |
| linear_bspline_svgd | 0.75 -> 0.11 | 0.03 | -0.00 | 0.99 | 0.02 | 4.3 | PASS |
| linear_waypoint_svgd | 0.75 -> 0.12 | 0.04 | 0.02 | 1.00 | 0.01 | 6.3 | PASS |
| linear_waypoint_svgd_smooth | 0.75 -> 0.12 | 0.04 | -0.00 | 0.99 | 0.02 | 3.8 | PASS |

## ring (hellinger floor 0.21)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.33 -> 0.33 | 0.16 | 0.02 | 0.79 | 1.14 | 3.4 | PASS |
| linear_sun_pointmass | 0.84 -> 0.15 | 0.06 | 0.01 | 0.96 | 0.20 | 4.5 | PASS |
| linear_sun_jerk | 0.84 -> 0.29 | 0.16 | 0.04 | 0.87 | 3.94 | 3.3 | PASS |
| cfm_adam_bspline | 0.33 -> 0.25 | 0.08 | 0.02 | 0.90 | 0.12 | 4.2 | PASS |
| linear_adam_bspline | 0.84 -> 0.27 | 0.08 | 0.05 | 0.90 | 0.15 | 4.3 | PASS |
| linear_adam_waypoint | 0.84 -> 0.22 | 0.10 | -0.01 | 0.93 | 0.02 | 6.9 | PASS |
| linear_adam_waypoint_smooth | 0.84 -> 0.21 | 0.07 | 0.03 | 0.95 | 0.09 | 5.1 | PASS |
| cfm_bspline_svgd | 0.33 -> 0.22 | 0.07 | 0.01 | 0.89 | 0.05 | 3.4 | PASS |
| linear_bspline_svgd | 0.84 -> 0.15 | 0.05 | -0.01 | 0.96 | 0.04 | 5.8 | PASS |
| linear_waypoint_svgd | 0.84 -> 0.14 | 0.06 | 0.01 | 0.99 | 0.01 | 12.5 | PASS |
| linear_waypoint_svgd_smooth | 0.84 -> 0.20 | 0.06 | 0.03 | 0.95 | 0.08 | 5.7 | PASS |

## sun_gmm3 (hellinger floor 0.22)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.38 -> 0.38 | 0.16 | 0.05 | 0.79 | 1.86 | 3.9 | PASS |
| linear_sun_pointmass | 0.73 -> 0.21 | 0.07 | 0.00 | 0.91 | 0.28 | 3.7 | PASS |
| linear_sun_jerk | 0.73 -> 0.33 | 0.18 | 0.04 | 0.78 | 2.13 | 2.4 | PASS |
| cfm_adam_bspline | 0.38 -> 0.27 | 0.09 | 0.04 | 0.89 | 0.15 | 4.8 | PASS |
| linear_adam_bspline | 0.73 -> 0.30 | 0.10 | 0.04 | 0.85 | 0.18 | 4.0 | PASS |
| linear_adam_waypoint | 0.73 -> 0.26 | 0.08 | -0.01 | 0.90 | 0.06 | 5.1 | PASS |
| linear_adam_waypoint_smooth | 0.73 -> 0.27 | 0.12 | 0.01 | 0.88 | 0.14 | 4.5 | PASS |
| cfm_bspline_svgd | 0.38 -> 0.25 | 0.10 | 0.03 | 0.89 | 0.08 | 4.1 | PASS |
| linear_bspline_svgd | 0.73 -> 0.27 | 0.10 | 0.04 | 0.89 | 0.27 | 6.0 | PASS |
| linear_waypoint_svgd | 0.73 -> 0.16 | 0.06 | 0.02 | 0.97 | 0.02 | 10.2 | PASS |
| linear_waypoint_svgd_smooth | 0.73 -> 0.25 | 0.09 | 0.04 | 0.90 | 0.11 | 5.3 | PASS |

## uniform (hellinger floor 0.33)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| cfm_only | 0.48 -> 0.48 | 0.17 | 0.00 | 0.53 | 2.56 | 5.8 | FAIL: C7 coverage |
| linear_sun_pointmass | 0.77 -> 0.27 | 0.07 | 0.00 | 0.78 | 0.60 | 7.8 | PASS |
| linear_sun_jerk | 0.77 -> 0.53 | 0.26 | 0.00 | 0.43 | 13.56 | 4.1 | FAIL: C1 hellinger, C2 tv_4, C7 coverage |
| cfm_adam_bspline | 0.48 -> 0.37 | 0.07 | 0.00 | 0.63 | 0.25 | 6.1 | FAIL: C7 coverage |
| linear_adam_bspline | 0.77 -> 0.37 | 0.09 | 0.00 | 0.62 | 0.26 | 5.8 | FAIL: C7 coverage |
| linear_adam_waypoint | 0.77 -> 0.37 | 0.06 | 0.00 | 0.71 | 0.19 | 9.7 | FAIL: C7 coverage |
| linear_adam_waypoint_smooth | 0.77 -> 0.33 | 0.07 | 0.00 | 0.68 | 0.17 | 6.6 | FAIL: C7 coverage |
| cfm_bspline_svgd | 0.48 -> 0.37 | 0.10 | 0.00 | 0.62 | 0.51 | 6.4 | FAIL: C7 coverage |
| linear_bspline_svgd | 0.77 -> 0.31 | 0.05 | 0.00 | 0.73 | 0.39 | 8.2 | FAIL: C7 coverage |
| linear_waypoint_svgd | 0.77 -> 0.19 | 0.05 | 0.00 | 0.96 | 0.01 | 15.6 | PASS |
| linear_waypoint_svgd_smooth | 0.77 -> 0.28 | 0.06 | 0.00 | 0.77 | 0.13 | 8.8 | PASS |
