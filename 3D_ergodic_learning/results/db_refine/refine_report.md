# DB refinement report

Medians. erg = Sinkhorn footprint error; so / pt = standoff / pointing against a local PCA plane (surface_metrics.py); len = control-polygon length. 0 = projected teacher (on the surface), off = moved 0.12 along the normal, 1 = refined.

| subset | n | erg0 | erg_off | erg1 | so0 | so1 | pt0 [deg] | pt1 [deg] | len0 | len1 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 90565 | 3.579e-03 | 3.504e-03 | 2.917e-04 | 0.006 | 0.106 | 7.8 | 25.2 | 6.68 | 9.21 |
| train | 81135 | 3.607e-03 | 3.517e-03 | 2.946e-04 | 0.006 | 0.106 | 7.8 | 25.2 | 6.67 | 9.20 |
| val_beides | 506 | 2.414e-03 | 2.599e-03 | 1.386e-04 | 0.006 | 0.110 | 8.1 | 21.3 | 6.32 | 8.82 |
| val_flaeche | 508 | 2.608e-03 | 2.582e-03 | 1.418e-04 | 0.006 | 0.110 | 7.8 | 22.5 | 6.13 | 8.39 |
| val_form | 8416 | 3.403e-03 | 3.449e-03 | 2.857e-04 | 0.006 | 0.107 | 8.1 | 24.8 | 6.78 | 9.35 |
| group baugruppe | 10295 | 3.985e-03 | 4.121e-03 | 3.593e-04 | 0.009 | 0.099 | 12.5 | 28.8 | 7.91 | 10.62 |
| group buchstabe | 12614 | 4.313e-03 | 4.820e-03 | 5.008e-04 | 0.012 | 0.091 | 16.0 | 32.1 | 9.74 | 12.44 |
| group ebene | 5280 | 1.668e-03 | 5.393e-04 | 1.011e-04 | 0.000 | 0.136 | 0.0 | 15.5 | 4.60 | 5.16 |
| group extern | 2628 | 4.210e-03 | 4.883e-03 | 4.735e-04 | 0.011 | 0.087 | 18.6 | 33.7 | 7.90 | 10.85 |
| group organismus | 19591 | 4.115e-03 | 3.775e-03 | 4.170e-04 | 0.005 | 0.116 | 4.1 | 23.6 | 6.37 | 9.05 |
| group primitiv | 17212 | 4.091e-03 | 3.516e-03 | 3.114e-04 | 0.006 | 0.113 | 7.6 | 23.5 | 6.99 | 9.45 |
| group superquadrik | 12563 | 3.327e-03 | 2.961e-03 | 2.175e-04 | 0.005 | 0.115 | 6.7 | 22.1 | 6.12 | 8.30 |
| group wort | 10382 | 1.077e-03 | 2.297e-03 | 7.955e-05 | 0.008 | 0.091 | 33.5 | 25.5 | 5.89 | 8.13 |

Filter: `{"filter_pct": 98.0, "erg_threshold": 0.0015122296568006296, "max_so_dev": 0.06, "dropped": 2654, "train_before": 83789, "train_after": 81135}`
