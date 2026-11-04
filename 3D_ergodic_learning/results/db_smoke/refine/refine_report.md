# DB refinement report

Medians. erg = Sinkhorn footprint error; so / pt = standoff / pointing against a local PCA plane (surface_metrics.py); len = control-polygon length. 0 = projected teacher (on the surface), off = moved 0.12 along the normal, 1 = refined.

| subset | n | erg0 | erg_off | erg1 | so0 | so1 | pt0 [deg] | pt1 [deg] | len0 | len1 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 62 | 2.848e-03 | 3.168e-03 | 1.404e-03 | 0.007 | 0.102 | 9.1 | 13.6 | 6.82 | 7.86 |
| train | 14 | 3.615e-03 | 3.611e-03 | 1.528e-03 | 0.007 | 0.100 | 11.1 | 13.3 | 6.97 | 7.83 |
| val_beides | 16 | 2.405e-03 | 3.125e-03 | 1.293e-03 | 0.007 | 0.099 | 8.5 | 13.0 | 6.76 | 7.51 |
| val_flaeche | 16 | 1.960e-03 | 1.923e-03 | 9.232e-04 | 0.005 | 0.103 | 4.9 | 9.5 | 5.44 | 6.44 |
| val_form | 16 | 4.358e-03 | 4.006e-03 | 1.895e-03 | 0.008 | 0.099 | 12.9 | 15.0 | 7.46 | 8.36 |
| group baugruppe | 8 | 2.258e-03 | 2.314e-03 | 1.080e-03 | 0.008 | 0.098 | 10.1 | 15.2 | 7.85 | 9.27 |
| group buchstabe | 9 | 4.144e-03 | 4.021e-03 | 2.375e-03 | 0.014 | 0.082 | 18.4 | 22.0 | 9.95 | 10.83 |
| group extern | 1 | 5.323e-03 | 8.111e-03 | 4.647e-03 | 0.016 | 0.061 | 32.4 | 36.6 | 8.87 | 9.88 |
| group organismus | 11 | 4.571e-03 | 4.083e-03 | 1.827e-03 | 0.006 | 0.105 | 4.2 | 9.4 | 6.59 | 8.04 |
| group primitiv | 17 | 3.750e-03 | 2.599e-03 | 1.746e-03 | 0.004 | 0.106 | 5.7 | 9.6 | 5.33 | 6.49 |
| group superquadrik | 6 | 1.182e-03 | 1.479e-03 | 6.561e-04 | 0.005 | 0.100 | 9.2 | 12.8 | 5.33 | 6.21 |
| group wort | 10 | 1.416e-03 | 2.264e-03 | 9.825e-04 | 0.008 | 0.072 | 30.7 | 31.0 | 6.24 | 7.01 |

Filter: `{"filter_pct": 98.0, "erg_threshold": 0.020548564381897446, "max_so_dev": 0.06, "dropped": 2, "train_before": 16, "train_after": 14}`
