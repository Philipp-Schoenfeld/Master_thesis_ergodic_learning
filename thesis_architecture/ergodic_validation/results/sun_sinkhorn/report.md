# Ergodicity validation

Median over 30 candidates at the final iteration. hell = Hellinger distance of the time-averaged visitation density to the target (floor = i.i.d. samples), tv4 = time vs. mass fraction on a 4x4 partition, low = extra time outside the support, cov = target mass within 0.05 of the path. FAIL lists the failed checks.

## A (hellinger floor 0.20)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.77 -> 0.12 | 0.04 | 0.01 | 0.99 | 0.04 | 5.3 | PASS |
| linear_sun_jerk | 0.77 -> 0.27 | 0.10 | 0.05 | 0.89 | 0.44 | 3.4 | PASS |

## blobs_75_25 (hellinger floor 0.18)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.84 -> 0.14 | 0.04 | -0.00 | 0.97 | 0.05 | 4.1 | PASS |
| linear_sun_jerk | 0.84 -> 0.25 | 0.06 | 0.04 | 0.92 | 0.51 | 2.7 | PASS |

## korean_5 (hellinger floor 0.23)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.77 -> 0.15 | 0.04 | 0.04 | 0.98 | 0.06 | 6.5 | PASS |
| linear_sun_jerk | 0.77 -> 0.33 | 0.11 | 0.09 | 0.81 | 0.55 | 4.0 | PASS |

## organic_30 (hellinger floor 0.14)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.75 -> 0.11 | 0.03 | 0.01 | 0.99 | 0.06 | 3.8 | PASS |
| linear_sun_jerk | 0.75 -> 0.22 | 0.08 | 0.06 | 0.97 | 0.74 | 2.7 | PASS |

## ring (hellinger floor 0.21)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.84 -> 0.13 | 0.04 | 0.01 | 0.97 | 0.03 | 6.0 | PASS |
| linear_sun_jerk | 0.84 -> 0.71 | 0.47 | 0.50 | 0.35 | 18.42 | 3.7 | FAIL: C1 hellinger, C2 tv_4, C3 low_excess, C5 no late degradation |

## sun_gmm3 (hellinger floor 0.22)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.73 -> 0.16 | 0.07 | 0.00 | 0.95 | 0.04 | 5.2 | PASS |
| linear_sun_jerk | 0.73 -> 0.29 | 0.13 | 0.04 | 0.87 | 0.36 | 3.3 | PASS |

## uniform (hellinger floor 0.33)

| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |
|---|---|---|---|---|---|---|---|
| linear_sun_pointmass | 0.77 -> 0.19 | 0.05 | 0.00 | 0.90 | 0.06 | 9.0 | PASS |
| linear_sun_jerk | 0.77 -> 0.62 | 0.36 | 0.00 | 0.35 | 16.12 | 4.6 | FAIL: C1 hellinger, C2 tv_4 |
