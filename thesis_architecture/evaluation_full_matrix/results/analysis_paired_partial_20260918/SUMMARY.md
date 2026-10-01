# Intensive-Auswertung: best_of_30_paired_20260918 (PARTIAL)

Datenbasis: 8 von 25 Formen vollstaendig (A, a_lc, digit_5, greek_upper_0, korean_5, rand_ana_poly_10, rand_gmm_10, rand_gmm_20), 1888 Zeilen.
Kein weiterer Job seit 2026-09-20 -- dies ist ein Zwischenstand, kein Endergebnis.

## 1. Paired-SVGD-Effekt (svgd0 -> svgd25, no_replan, derselbe Kandidat)
representation    family  E_ergodic_total_svgd0  E_ergodic_total_svgd25     delta  pct_improvement  n
     particles    niveau              16.548250               13.273128 -3.275122        19.791351 32
     particles       ucb              22.067732               16.170297 -5.897435        26.724244 32
     particles      mass              13.504202               10.868026 -2.636176        19.521156 32
     particles eid_tuned              16.146266               14.337278 -1.808987        11.203751 32
      spectral    niveau              14.527880               12.019370 -2.508511        17.266874 32
      spectral       ucb              15.899760               13.334929 -2.564830        16.131253 32
      spectral      mass              12.644739               10.550103 -2.094637        16.565282 32
      spectral eid_tuned              19.694201               15.620702 -4.073499        20.683748 32

## 2. Random Walk: svgd0 vs. svgd500 (derselbe Best-of-30-Kandidat)
WICHTIGE EINORDNUNG: svgd500 verfeinert direkt gegen die EXAKTE Ground-Truth-Dichte (random_walk hat kein Wissensstufen-Konzept, nutzt immer die volle Wahrheit), mit 20x mehr Iterationen als jede andere Strategie hier verwendet. Der SVGD-Optimierer minimiert direkt dieselbe Energie, die E_ergodic_total misst -- bei genug Iterationen gegen die exakte Wahrheit konvergiert JEDE Startbahn nahe ans Optimum, unabhaengig von ihrer Qualitaet. Das testet "kann der Solver selbst, mit genug Zeit und vollem Wissen, nahe loesen", NICHT "ist eine Zufallsbahn als Prior so gut wie ein trainiertes Netz". Nicht fair vergleichbar mit den wissensstufen-beschraenkten CFM-/Heuristik-Zahlen unten.
            E_ergodic_total  coverage   path_len  smoothness_energy
subvariant                                                         
svgd0             14.910614  0.046554  10.707504          11.176618
svgd500            0.472540  0.039316   4.408305           0.367250

## 3. Heuristik/Linear-Waypoints: SVGD-Sweep (0/500/1000 Iterationen)
                method family  svgd_iters  E_ergodic_total
       heuristic_tuned    eid           0        55.317401
       heuristic_tuned    eid         500        15.679127
       heuristic_tuned    eid        1000        15.533139
       heuristic_tuned    lse           0        56.891419
       heuristic_tuned    lse         500        13.457460
       heuristic_tuned    lse        1000        13.203055
       heuristic_tuned   mass           0        59.142431
       heuristic_tuned   mass         500        11.392162
       heuristic_tuned   mass        1000        11.108362
       heuristic_tuned    ucb           0        52.993857
       heuristic_tuned    ucb         500        14.822315
       heuristic_tuned    ucb        1000        14.533234
linear_waypoints_tuned    eid         500        14.370015
linear_waypoints_tuned    eid        1000        15.401878
linear_waypoints_tuned    lse         500        11.967276
linear_waypoints_tuned    lse        1000        13.267107
linear_waypoints_tuned   mass         500        10.001251
linear_waypoints_tuned   mass        1000        11.154805
linear_waypoints_tuned    ucb         500        13.355109
linear_waypoints_tuned    ucb        1000        14.579977

## 4. Repraesentationsvergleich: particles vs. spectral (alle CFM-Strategien)
                E_ergodic_total  coverage  smoothness_energy  path_len
representation                                                        
particles             16.107545  0.043931           1.179554  6.794945
spectral              22.479506  0.067497           1.556117  6.020104

ACHTUNG: `spectral` lief hier (wie im ganzen Projekt) auf einem kleineren, andersartigen Trainings-Formensatz als `particles` -- dieser Vergleich testet auch Out-of-Distribution-Generalisierung, nicht nur Repraesentation an sich (siehe spectral_planner.py Modul-Docstring).

## 5. Replan-Schema: no_replan vs. replan_1_6
               E_ergodic_total  coverage  path_len
replan_scheme                                     
no_replan            15.037859  0.056249  4.377899
replan_1_6           23.549192  0.055179  8.437149

## 6. Wissensstufen: ground_truth -> half_known -> ten_samples -> none_known
                     E_ergodic_total  coverage
knowledge_condition                           
ground_truth               10.086437  0.049468
half_known                 18.635206  0.056013
ten_samples                23.101221  0.058292
none_known                 25.351238  0.059083

## 7. Methodenvergleich (Kernergebnis)
Fairness-Hinweis: heuristic_tuned/linear_waypoints_tuned sind deterministisch/Einmal-Planung (kein Replan-Konzept) -- CFM wird hier deshalb auf `no_replan` beschraenkt, fuer einen Einmal-Planung-gegen-Einmal-Planung-Vergleich. CFMs replan_1_6-Option ist separat in Abschnitt 5 zu sehen (deutlich schlechter: 23.5 vs. 15.0) -- in dieser Tabelle NICHT mitgemittelt, sonst wuerde CFM unfair schlechter aussehen.
                                             method  E_ergodic_total
           random_walk (svgd500) *see caveat above*         0.472540
            linear_waypoints_tuned (best: mass@500)        10.001251
 CFM (spectral, no_replan, best: mass_tuned_svgd25)        10.550103
CFM (particles, no_replan, best: mass_tuned_svgd25)        10.868026
                  heuristic_tuned (best: mass@1000)        11.108362
                                          lawnmower        18.780876

## 8. Pro-Form-Aufschluesselung (CFM particles, bestes Strategie)
shape
A                    9.190910
a_lc                11.169997
digit_5              8.306963
greek_upper_0       10.077203
korean_5            11.575870
rand_ana_poly_10    25.502301
rand_gmm_10          4.013901
rand_gmm_20          7.107062

## 9. 99%-Coverage-Erreichungsrate je Methode
                method  reach_rate
             particles    0.251736
       heuristic_tuned    0.250000
linear_waypoints_tuned    0.250000
              spectral    0.250000
             lawnmower    0.000000
           random_walk    0.000000
Hinweis: bei diesem strikten 99%-Schwellwert erreichen die meisten Varianten ihn fast nie (siehe caveat in CLUSTER_CONTEXT_best_of_n.md) -- die Rate selbst ist aussagekraeftiger als die Schrittzahl.

## 10. Candidate-pool coverage: mean +/- std across the 30 raw candidates, per knowledge condition
knowledge_condition               group  coverage_mean  coverage_std  n
       ground_truth    particles/niveau       0.052599      0.003575  8
       ground_truth       particles/ucb       0.043634      0.002833  8
       ground_truth      particles/mass       0.045280      0.003898  8
       ground_truth particles/eid_tuned       0.048022      0.005310  8
       ground_truth     spectral/niveau       0.058896      0.002407  8
       ground_truth        spectral/ucb       0.053228      0.002702  8
       ground_truth       spectral/mass       0.052516      0.002569  8
       ground_truth  spectral/eid_tuned       0.060986      0.010199  8
       ground_truth         random_walk       0.096562      0.037966  8
         half_known    particles/niveau       0.064420      0.007331  8
         half_known       particles/ucb       0.063173      0.006642  8
         half_known      particles/mass       0.055617      0.006650  8
         half_known particles/eid_tuned       0.080589      0.008084  8
         half_known     spectral/niveau       0.074011      0.005884  8
         half_known        spectral/ucb       0.080409      0.005354  8
         half_known       spectral/mass       0.070376      0.004359  8
         half_known  spectral/eid_tuned       0.115894      0.003489  8
         half_known         random_walk       0.096562      0.037966  8
        ten_samples    particles/niveau       0.067399      0.008561  8
        ten_samples       particles/ucb       0.071200      0.009785  8
        ten_samples      particles/mass       0.070525      0.009319  8
        ten_samples particles/eid_tuned       0.062391      0.007260  8
        ten_samples     spectral/niveau       0.066268      0.001247  8
        ten_samples        spectral/ucb       0.065886      0.001141  8
        ten_samples       spectral/mass       0.063165      0.005833  8
        ten_samples  spectral/eid_tuned       0.076029      0.005513  8
        ten_samples         random_walk       0.096562      0.037966  8
         none_known    particles/niveau       0.075215      0.012164  8
         none_known       particles/ucb       0.073670      0.010568  8
         none_known      particles/mass       0.070502      0.009483  8
         none_known particles/eid_tuned       0.074811      0.011344  8
         none_known     spectral/niveau       0.067583      0.000872  8
         none_known        spectral/ucb       0.067668      0.000850  8
         none_known       spectral/mass       0.067615      0.000822  8
         none_known  spectral/eid_tuned       0.067542      0.000858  8
         none_known         random_walk       0.096562      0.037966  8
coverage_mean/_std are themselves averages over the 8 shapes within each (knowledge_condition, group) cell -- n=8 per row, not the spread BETWEEN shapes. random_walk's value is IDENTICAL across all 4 panels (0.097): each shape's random_walk row is computed once (method='shared', no belief involved at all) and the exact same row is replicated with a different knowledge_condition label for all 4 conditions -- not a real condition-dependence, just the label reused four times.

## 11. Path length and smoothness/energy per method
                                                     path_len  smoothness_energy
method                                                                          
random_walk (svgd500) *see caveat above*             4.408305           0.367250
linear_waypoints_tuned (best: mass@500)              5.410750           0.589565
CFM (spectral, no_replan, best: mass_tuned_svgd25)   3.844642           0.166365
CFM (particles, no_replan, best: mass_tuned_svgd25)  5.213950           0.248646
heuristic_tuned (best: mass@1000)                    4.450089           0.197129
lawnmower                                            8.799999           1.519248
smoothness_energy = w * sum(acceleration^2), 3-point finite difference, resampled to 128 points -- this IS the robotics energy/control-effort proxy, not a separate quantity from smoothness.

## 12. Steps (normalized arclength) to first reach 99% coverage, all 4 conditions
                                             method knowledge_condition  steps_to_full_coverage  reached  n
CFM (particles, no_replan, best: mass_tuned_svgd25)        ground_truth                     0.0      1.0  8
CFM (particles, no_replan, best: mass_tuned_svgd25)          half_known                     1.0      0.0  8
CFM (particles, no_replan, best: mass_tuned_svgd25)         ten_samples                     1.0      0.0  8
CFM (particles, no_replan, best: mass_tuned_svgd25)          none_known                     1.0      0.0  8
                  heuristic_tuned (best: mass@1000)        ground_truth                     0.0      1.0  8
                  heuristic_tuned (best: mass@1000)          half_known                     1.0      0.0  8
                  heuristic_tuned (best: mass@1000)         ten_samples                     1.0      0.0  8
                  heuristic_tuned (best: mass@1000)          none_known                     1.0      0.0  8
           random_walk (svgd500) *see caveat above*        ground_truth                     1.0      0.0  8
           random_walk (svgd500) *see caveat above*          half_known                     1.0      0.0  8
           random_walk (svgd500) *see caveat above*         ten_samples                     1.0      0.0  8
           random_walk (svgd500) *see caveat above*          none_known                     1.0      0.0  8
                                          lawnmower        ground_truth                     1.0      0.0  8
                                          lawnmower          half_known                     1.0      0.0  8
                                          lawnmower         ten_samples                     1.0      0.0  8
                                          lawnmower          none_known                     1.0      0.0  8
This metric is essentially BINARY at the 99% threshold in this data: ~0 only for ground_truth (forced by definition, not earned), ~1 (never reached) for every other condition, with one marginal exception (particles/none_known: 0.69% of cases reached it). A graded answer to "how many steps to recover" would need a looser threshold (e.g. 80-90%) than the project's current 99% default.

## 13. Candidate-pool coverage distributions (Gaussian approximation from mean+-std)
## 14. Explore vs. exploit balance per method
                                                     E_ergodic_explore  E_ergodic_exploit
method                                                                                   
random_walk (svgd500) *see caveat above*                      0.333149           0.139391
linear_waypoints_tuned (best: mass@500)                       9.853261           0.147990
CFM (spectral, no_replan, best: mass_tuned_svgd25)           10.383818           0.166285
CFM (particles, no_replan, best: mass_tuned_svgd25)          10.754409           0.113617
heuristic_tuned (best: mass@1000)                            10.957908           0.150454
lawnmower                                                    18.743589           0.037287

## 15. Knowledge-condition degradation per method
                                             method knowledge_condition  E_ergodic_total
CFM (particles, no_replan, best: mass_tuned_svgd25)        ground_truth         0.928124
CFM (particles, no_replan, best: mass_tuned_svgd25)          half_known        12.077864
CFM (particles, no_replan, best: mass_tuned_svgd25)         ten_samples         9.265489
CFM (particles, no_replan, best: mass_tuned_svgd25)          none_known        21.200627
 CFM (spectral, no_replan, best: mass_tuned_svgd25)        ground_truth         1.219544
 CFM (spectral, no_replan, best: mass_tuned_svgd25)          half_known        12.410256
 CFM (spectral, no_replan, best: mass_tuned_svgd25)         ten_samples        10.515613
 CFM (spectral, no_replan, best: mass_tuned_svgd25)          none_known        18.054997
                  heuristic_tuned (best: mass@1000)        ground_truth         0.191969
                  heuristic_tuned (best: mass@1000)          half_known        13.207155
                  heuristic_tuned (best: mass@1000)         ten_samples        11.182062
                  heuristic_tuned (best: mass@1000)          none_known        19.852262
            linear_waypoints_tuned (best: mass@500)        ground_truth         0.225768
            linear_waypoints_tuned (best: mass@500)          half_known        12.813870
            linear_waypoints_tuned (best: mass@500)         ten_samples        10.743119
            linear_waypoints_tuned (best: mass@500)          none_known        16.222247
           random_walk (svgd500) *see caveat above*        ground_truth         0.472540
           random_walk (svgd500) *see caveat above*          half_known         0.472540
           random_walk (svgd500) *see caveat above*         ten_samples         0.472540
           random_walk (svgd500) *see caveat above*          none_known         0.472540
                                          lawnmower        ground_truth        18.780876
                                          lawnmower          half_known        18.780876
                                          lawnmower         ten_samples        18.780876
                                          lawnmower          none_known        18.780876

## 16. Representation x knowledge-condition heatmap (E_ergodic_total)
knowledge_condition  ground_truth  half_known  ten_samples  none_known
representation                                                        
particles                7.182109   18.744559    17.682251   20.821260
spectral                12.990766   18.525854    28.520191   29.881215

## 17. Per-shape breakdown, all top methods (random_walk excluded, see caveat)
                                             method            shape  E_ergodic_total
            linear_waypoints_tuned (best: mass@500)                A         7.130560
            linear_waypoints_tuned (best: mass@500)             a_lc         9.583120
            linear_waypoints_tuned (best: mass@500)          digit_5         6.946669
            linear_waypoints_tuned (best: mass@500)    greek_upper_0         6.903885
            linear_waypoints_tuned (best: mass@500)         korean_5         8.387215
            linear_waypoints_tuned (best: mass@500) rand_ana_poly_10        28.856352
            linear_waypoints_tuned (best: mass@500)      rand_gmm_10         3.407484
            linear_waypoints_tuned (best: mass@500)      rand_gmm_20         8.794724
 CFM (spectral, no_replan, best: mass_tuned_svgd25)                A         7.010389
 CFM (spectral, no_replan, best: mass_tuned_svgd25)             a_lc         9.105075
 CFM (spectral, no_replan, best: mass_tuned_svgd25)          digit_5         7.986795
 CFM (spectral, no_replan, best: mass_tuned_svgd25)    greek_upper_0         6.907504
 CFM (spectral, no_replan, best: mass_tuned_svgd25)         korean_5        13.486910
 CFM (spectral, no_replan, best: mass_tuned_svgd25) rand_ana_poly_10        29.008377
 CFM (spectral, no_replan, best: mass_tuned_svgd25)      rand_gmm_10         3.584139
 CFM (spectral, no_replan, best: mass_tuned_svgd25)      rand_gmm_20         7.311631
CFM (particles, no_replan, best: mass_tuned_svgd25)                A         9.190910
CFM (particles, no_replan, best: mass_tuned_svgd25)             a_lc        11.169997
CFM (particles, no_replan, best: mass_tuned_svgd25)          digit_5         8.306963
CFM (particles, no_replan, best: mass_tuned_svgd25)    greek_upper_0        10.077203
CFM (particles, no_replan, best: mass_tuned_svgd25)         korean_5        11.575870
CFM (particles, no_replan, best: mass_tuned_svgd25) rand_ana_poly_10        25.502301
CFM (particles, no_replan, best: mass_tuned_svgd25)      rand_gmm_10         4.013901
CFM (particles, no_replan, best: mass_tuned_svgd25)      rand_gmm_20         7.107062
                  heuristic_tuned (best: mass@1000)                A         8.652281
                  heuristic_tuned (best: mass@1000)             a_lc        10.586519
                  heuristic_tuned (best: mass@1000)          digit_5         7.749237
                  heuristic_tuned (best: mass@1000)    greek_upper_0         9.152311
                  heuristic_tuned (best: mass@1000)         korean_5        10.811587
                  heuristic_tuned (best: mass@1000) rand_ana_poly_10        30.745271
                  heuristic_tuned (best: mass@1000)      rand_gmm_10         3.976599
                  heuristic_tuned (best: mass@1000)      rand_gmm_20         7.193092
                                          lawnmower                A        16.421095
                                          lawnmower             a_lc        16.642576
                                          lawnmower          digit_5        17.018491
                                          lawnmower    greek_upper_0        16.421095
                                          lawnmower         korean_5        15.577491
                                          lawnmower rand_ana_poly_10        49.645777
                                          lawnmower      rand_gmm_10         5.458813
                                          lawnmower      rand_gmm_20        13.061671

## 18. Coverage-reach rate at looser thresholds (80% / 90% / 99%)
                        reached_80  reached_90  reached_99
method                                                    
CFM particles             0.083333         0.0         0.0
CFM spectral              0.000000         0.0         0.0
heuristic_tuned           0.000000         0.0         0.0
linear_waypoints_tuned    0.125000         0.0         0.0
lawnmower                 1.000000         1.0         0.0
Averaged over half_known/ten_samples/none_known only (ground_truth excluded -- forced to reached=1 regardless of threshold, would dilute the signal). lawnmower stands out: 100% reach rate at 80%, vs. near-0% for the ergodic-optimizing methods -- systematic sweeping guarantees eventual area coverage, something none of the density-matching methods are explicitly optimizing for.


## 19. svgd0 vs. svgd25 vs. svgd500 (deeper SVGD budget)
representation    family     svgd0    svgd25   svgd500
     particles    niveau 20.411442 17.030101 17.058103
     particles       ucb 28.562223 21.339871 19.422133
     particles      mass 16.823915 14.181327 14.763187
     particles eid_tuned 20.206616 18.369428 19.494821
      spectral    niveau 17.319336 15.080242 17.037874
      spectral       ucb 19.862937 17.380796 19.048973
      spectral      mass 15.545293 13.660289 14.960852
      spectral eid_tuned 24.475045 19.688237 19.748528
ground_truth excluded above (collapses toward 0 like random_walk's svgd500, mean E_ergodic_total=1.021) -- shown separately, not averaged in.

