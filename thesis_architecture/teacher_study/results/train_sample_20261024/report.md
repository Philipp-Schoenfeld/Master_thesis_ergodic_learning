# Teacher study `train_sample_20261024`

40 shapes; selection rule: smoothest candidate with Hellinger <= best + 0.05 (per shape, pool stages ['bs25', 'bs25_pol100', 'bs25_pol300']); smoothness = `smooth`; candidates with more than 0.0 of their points outside [0.03, 0.97] excluded.

## Selected teacher vs. current DB teacher (B-spline, 25 CP)

* Hellinger change (selected - DB): mean -0.021
* smoothness energy ratio: median 0.32 (mean 0.44)
* acceleration RMS ratio: median 0.46
* jerk RMS ratio: median 0.25

DB teacher (bs25): Hellinger 0.246, smooth 0.336, acc RMS 240.1, jerk RMS 17664

## Ranking (hit rate, then smoothness of the best-of-N pick)

| config | stage | hit rate | Hellinger | smooth | acc RMS | jerk RMS | path |
|---|---|---|---|---|---|---|---|
| lm_K16_cont_o2_w15 | bs25 | 1.00 | 0.219 | 0.114 | 112.5 | 5544 | 3.80 |
| lm_K16_cont_o2_w15 | bs25_pol300 | 1.00 | 0.225 | 0.134 | 108.4 | 4311 | 3.81 |
| lm_o2_w15 | bs25 | 1.00 | 0.224 | 0.201 | 129.8 | 5394 | 4.41 |
| lm_K16_o2_w15 | bs25 | 1.00 | 0.214 | 0.228 | 143.0 | 6658 | 4.28 |
| lm_o2_w15 | bs25_pol300 | 1.00 | 0.224 | 0.229 | 128.2 | 4493 | 4.43 |
| lm_o2_w15 | bs25_pol100 | 1.00 | 0.225 | 0.234 | 129.0 | 4780 | 4.47 |
| lm_K16_o2_w15 | bs25_pol300 | 1.00 | 0.224 | 0.258 | 136.0 | 5099 | 4.40 |
| gui_lr8e-3_n3000 | bs25_pol300 | 0.97 | 0.225 | 0.140 | 111.5 | 3815 | 4.16 |
| lm_K16_o2_w15 | bs25_pol100 | 0.95 | 0.217 | 0.243 | 136.0 | 5328 | 4.38 |
| lm_K16_cont_o2_w15 | bs25_pol100 | 0.93 | 0.220 | 0.111 | 105.3 | 4401 | 3.67 |
| gui_lr8e-3_n3000 | bs25_pol100 | 0.93 | 0.222 | 0.149 | 114.7 | 3943 | 4.26 |
| gui_lr8e-3_n3000 | bs25 | 0.47 | 0.233 | 0.186 | 123.3 | 4467 | 4.58 |

## Which configuration produced the chosen teacher

* lm_K16_cont_o2_w15 / bs25_pol300: 14 shapes
* lm_K16_cont_o2_w15 / bs25_pol100: 9 shapes
* lm_K16_cont_o2_w15 / bs25: 8 shapes
* gui_lr8e-3_n3000 / bs25_pol300: 3 shapes
* lm_K16_o2_w15 / bs25: 2 shapes
* gui_lr8e-3_n3000 / bs25_pol100: 1 shapes
* lm_o2_w15 / bs25_pol300: 1 shapes
* lm_K16_o2_w15 / bs25_pol300: 1 shapes
* lm_o2_w15 / bs25: 1 shapes
