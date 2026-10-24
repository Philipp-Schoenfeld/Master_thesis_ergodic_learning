# Teacher study `full_20261024`

25 shapes; selection rule: smoothest candidate with Hellinger <= best + 0.05 (per shape, pool stages ['bs25', 'bs25_pol100', 'bs25_pol300']); smoothness = `smooth`; candidates with more than 0.0 of their points outside [0.03, 0.97] excluded.

## Selected teacher vs. current DB teacher (B-spline, 25 CP)

* Hellinger change (selected - DB): mean -0.023
* smoothness energy ratio: median 0.35 (mean 0.47)
* acceleration RMS ratio: median 0.50
* jerk RMS ratio: median 0.26

DB teacher (bs25): Hellinger 0.248, smooth 0.329, acc RMS 221.3, jerk RMS 16602

## Ranking (hit rate, then smoothness of the best-of-N pick)

| config | stage | hit rate | Hellinger | smooth | acc RMS | jerk RMS | path |
|---|---|---|---|---|---|---|---|
| lm_K16_o2_w15 | bs25 | 1.00 | 0.207 | 0.237 | 141.5 | 6714 | 4.59 |
| lm_o2_w5 | bs25 | 1.00 | 0.210 | 0.327 | 163.7 | 7294 | 5.13 |
| adam_o2_w5 | bs25 | 1.00 | 0.221 | 0.369 | 183.3 | 9558 | 4.74 |
| lm_K16_cont_o2_w15 | bs25 | 0.96 | 0.215 | 0.177 | 120.7 | 6329 | 4.00 |
| lm_o2_w15 | bs25 | 0.96 | 0.216 | 0.206 | 127.1 | 5310 | 4.54 |
| lm_o1_w3000 | bs25 | 0.96 | 0.213 | 0.224 | 202.4 | 13365 | 4.35 |
| lm_K16_o2_w15 | bs25_pol300 | 0.96 | 0.215 | 0.246 | 139.5 | 5188 | 4.70 |
| lm_K16_o2_w15 | bs25_pol100 | 0.96 | 0.215 | 0.261 | 141.0 | 5602 | 4.71 |
| gui_lr8e-3_n3000 | bs25_pol300 | 0.92 | 0.219 | 0.159 | 113.1 | 3946 | 4.39 |
| lm_o2_w15 | bs25_pol300 | 0.92 | 0.217 | 0.202 | 123.1 | 4185 | 4.53 |
| lm_o2_w5 | bs25_pol300 | 0.92 | 0.206 | 0.323 | 163.1 | 6458 | 5.01 |
| lm_o3_w0.01 | bs25 | 0.92 | 0.220 | 0.368 | 165.1 | 5260 | 4.86 |
| sun_sinkhorn_o2_w3000 | bs25 | 0.88 | 0.216 | 0.162 | 115.1 | 5983 | 3.96 |
| gui_lr8e-3_n3000 | bs25_pol100 | 0.84 | 0.212 | 0.168 | 115.2 | 4027 | 4.39 |
| lm_o2_w15 | bs25_pol100 | 0.84 | 0.213 | 0.198 | 121.8 | 4327 | 4.40 |
| lm_o2_w5 | bs25_pol100 | 0.84 | 0.213 | 0.321 | 160.0 | 6573 | 4.91 |
| lm_o3_w0.01 | bs25_pol300 | 0.84 | 0.218 | 0.327 | 157.7 | 4634 | 4.73 |
| lm_K16_cont_o2_w15 | bs25_pol300 | 0.80 | 0.214 | 0.117 | 103.4 | 3920 | 3.81 |
| lm_K16_o2_w50 | bs25 | 0.80 | 0.207 | 0.129 | 99.1 | 3926 | 3.97 |
| lm_o1_w3000 | bs25_pol300 | 0.80 | 0.197 | 0.172 | 204.2 | 14257 | 3.91 |
| adam_o2_w15 | bs25 | 0.80 | 0.216 | 0.222 | 136.6 | 6233 | 4.17 |
| gui_lr2e-3_n3000 | bs25_pol300 | 0.80 | 0.210 | 0.223 | 135.0 | 4965 | 4.62 |
| lm_K16_cont_o2_w15 | bs25_pol100 | 0.76 | 0.209 | 0.136 | 104.8 | 4369 | 3.75 |
| sun_sinkhorn_o3_w3 | bs25 | 0.76 | 0.205 | 0.159 | 100.6 | 4074 | 3.87 |
| tsvec_o2_w5 | bs25 | 0.76 | 0.228 | 0.607 | 232.8 | 13759 | 5.53 |

## Which configuration produced the chosen teacher

* lm_K16_cont_o2_w15 / bs25: 4 shapes
* gui_lr8e-3_n3000 / bs25_pol300: 3 shapes
* lm_K16_cont_o2_w15 / bs25_pol100: 3 shapes
* lm_K16_cont_o2_w15 / bs25_pol300: 3 shapes
* sun_sinkhorn_o2_w3000 / bs25: 2 shapes
* gui_lr8e-3_n3000 / bs25_pol100: 2 shapes
* lm_o1_w3000 / bs25_pol100: 1 shapes
* lm_cont_o2_w15 / bs25_pol300: 1 shapes
* lm_o1_w3000 / bs25: 1 shapes
* lm_K16_o2_w50 / bs25_pol100: 1 shapes
* lm_o2_w50 / bs25_pol100: 1 shapes
* lm_o2_w50 / bs25: 1 shapes
* lm_K16_o2_w50 / bs25: 1 shapes
* lm_K16_o2_w15 / bs25: 1 shapes
