# Teacher study `pilot_20261024`

3 shapes; selection rule: smoothest candidate with Hellinger <= best + 0.05 (per shape, pool stages ['bs25', 'bs25_pol100', 'bs25_pol300']); smoothness = `smooth`.

## Selected teacher vs. current DB teacher (B-spline, 25 CP)

* Hellinger change (selected - DB): mean -0.022
* smoothness energy ratio: median 0.32 (mean 0.34)
* acceleration RMS ratio: median 0.60
* jerk RMS ratio: median 0.16

DB teacher (bs25): Hellinger 0.207, smooth 0.367, acc RMS 245.5, jerk RMS 19780

## Ranking (hit rate, then smoothness of the best-of-N pick)

| config | stage | hit rate | Hellinger | smooth | acc RMS | jerk RMS | path |
|---|---|---|---|---|---|---|---|
| lm_K16_o2_w50 | bs25 | 1.00 | 0.189 | 0.112 | 92.2 | 3435 | 3.63 |
| lm_o1_w3000 | bs25 | 1.00 | 0.182 | 0.146 | 233.5 | 15679 | 3.43 |
| lm_o2_w15 | bs25 | 1.00 | 0.185 | 0.158 | 116.2 | 4233 | 4.26 |
| lm_o2_w15 | bs25_pol100 | 1.00 | 0.187 | 0.158 | 114.2 | 3685 | 4.26 |
| lm_o2_w15 | bs25_pol300 | 1.00 | 0.188 | 0.167 | 114.0 | 3502 | 4.27 |
| sun_sinkhorn_o2_w3000 | bs25 | 1.00 | 0.189 | 0.169 | 116.8 | 6817 | 3.69 |
| lm_K16_o2_w15 | bs25_pol300 | 1.00 | 0.182 | 0.174 | 123.9 | 4225 | 4.35 |
| lm_K16_o2_w15 | bs25 | 1.00 | 0.180 | 0.175 | 128.4 | 6171 | 3.93 |
| lm_o1_w3000 | bs25_pol100 | 1.00 | 0.183 | 0.176 | 238.7 | 17901 | 3.85 |
| lm_K16_o2_w15 | bs25_pol100 | 1.00 | 0.186 | 0.177 | 128.1 | 4740 | 4.35 |
| lm_o1_w3000 | bs25_pol300 | 1.00 | 0.175 | 0.184 | 255.1 | 18839 | 3.77 |
| sun_sinkhorn_o3_w1 | bs25 | 1.00 | 0.182 | 0.185 | 118.3 | 6058 | 3.77 |
| sun_sinkhorn_o3_w3 | bs25 | 1.00 | 0.192 | 0.185 | 109.7 | 5063 | 3.74 |
| lm_o1_w1000 | bs25 | 1.00 | 0.186 | 0.193 | 256.1 | 19913 | 3.94 |
| lm_o3_w0.01 | bs25_pol300 | 1.00 | 0.189 | 0.196 | 127.0 | 3501 | 3.91 |
| lm_o1_w1000 | bs25_pol300 | 1.00 | 0.178 | 0.200 | 291.7 | 22284 | 4.00 |
| adam_o2_w15 | bs25_pol300 | 1.00 | 0.190 | 0.207 | 131.4 | 4820 | 3.73 |
| adam_o2_w15 | bs25 | 1.00 | 0.189 | 0.212 | 137.6 | 6388 | 3.78 |
| lm_o3_w0.01 | bs25 | 1.00 | 0.191 | 0.215 | 138.4 | 4486 | 4.00 |
| lm_o2_w5 | bs25 | 1.00 | 0.180 | 0.230 | 145.8 | 5698 | 4.56 |
| sun_sinkhorn_o3_w0.3 | bs25 | 1.00 | 0.188 | 0.244 | 139.4 | 7737 | 3.86 |
| adam_o2_w5 | bs25 | 1.00 | 0.187 | 0.244 | 164.4 | 7923 | 3.89 |
| sun_sinkhorn_o2_w0 | bs25 | 1.00 | 0.180 | 0.285 | 175.8 | 12540 | 4.18 |
| lm_o2_w5 | bs25_pol300 | 1.00 | 0.176 | 0.326 | 157.1 | 5194 | 4.44 |
| sun_sinkhorn_o2_w0 | bs25_pol300 | 1.00 | 0.181 | 0.335 | 182.8 | 12287 | 4.20 |

## Which configuration produced the chosen teacher

* lm_o1_w3000 / bs25: 1 shapes
* lm_K16_o2_w50 / bs25: 1 shapes
* adam_o3_w0.03 / bs25_pol300: 1 shapes
