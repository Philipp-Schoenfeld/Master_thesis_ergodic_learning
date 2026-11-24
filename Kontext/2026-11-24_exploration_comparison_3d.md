# Kontext: Explorationsvergleich in 3D auf einer Ebene (Sessions 2026-11-18 bis 11-24)

Fortsetzung der 2D-Pipeline `plots/exploration_comparison_combined` (`run_exploration_comparison.py` +
`plot_exploration_summary.py`). Auftrag Philipp: "genau diese Evaluationpipeline fuer 3D; geplant wird auf einer
Ebene (sieht in der Simulation aus wie 2D), rechnet aber mit echten 3D-Koordinaten, nicht projiziert; nutze die
3D-trainierten Checkpoints; alle Baseline-Vergleiche in 3D; neue Baseline: die x hoechsten Punkte des GP-Belief per
Traveling Salesman verbinden." **Status: fertig gerechnet (4200 Missionen), ausgewertet, gepusht.**

## 1. Entscheidungen (von Philipp bestaetigt)
- TSP wird **bis zum Budget aufgefuellt** (`tsp_fill`): reine 12-Punkte-Tour ist nur ~1,8 lang bei Budget ~5-6;
  `tsp_fill` nimmt per Bisektion so viele Maxima, bis die Tour das Budget erreicht (Obergrenze 200 Punkte).
  Die unaufgefuellte Variante `tsp` ist implementiert, wurde aber nicht im Hauptlauf gerechnet.
- `walk_len` 5.2 (Ebenen-Einheiten) fuer modellfreie Baselines und Strahl-Inits (Jobs b/c): die Netze fahren
  ~5,0-5,5 je Runde; Netze ignorieren `walk_len`. Default im Code bleibt 6.1 (opt-in, alter Wert unveraendert).
- Genehmigungen: Testjob, 3 Hauptjobs und Replot-Job jeweils einzeln bestaetigt.

## 2. Aufbau (alles neu, in `thesis_architecture/evaluation_full_matrix/`)
- `plane3d.py`: `Plane` (flat / tilted 50 Grad / diagonal; Einheitsquadrat in [0.14,0.86], Zentrum (0.5,0.5,0.5),
  Skala s=0.72, Rahmen wie `surfaces.ebene_flach`), 3D-Ergodik `Ergodic3D` (Kosinus-Fourier-Metrik, K=10 je Achse,
  Lambda=(1+|k|^2)^-2), `TsvecSvgd3D` (3D-Portierung des TSVEC-SVGD, analytischer Jacobi im Kontrollpunktraum,
  Ebenen-Residuum w=1000, Randterm in (u,v)), `AdamSvgd3D` (beta 100, lr 2e-3), `SolverBatch3D`.
- `mission_plane3d.py`: Belief (GP auf 64x64 3D-Ebenenpunkten via `belief3d.SurfaceBelief` / `AttentiveSurfaceBelief`,
  Sensor 4-Punkt-Ring, Rauschen 0,05, max. 64 Beobachtungen), Debt-Density-UCB-Ziel, Netz-Planer (3D-
  `generate_particle_trajectories` mit Ebenenkraft w_surface 50, w_align 1, force_weight 20), Baselines
  (rw, lawn, gpgoal, linray, tsp, tsp_fill, linadam, lintsvec), `run_mission`, `Evaluator3D`.
- `run_exploration_comparison_3d.py`: Runner (Flags `--plane --walk_len --K --w_plane --tsp_points --tsp_min_dist
  --tsp_source --ckpt method=path --merge_from --plot_only --smoke --dry_run --max_minutes --out_dir --db`),
  Resume-faehige DB je TAG, Plots (`rbf/`, `ak/`, `variants/`, `world3d/` = 3D-Ansicht), Aggregation.
- `test_plane3d.py`: 8 Tests (Chart-Roundtrip, flache Ebene = Trainingsebene, Jacobi vs. Finite Differenzen,
  Ergodik-Koeffizienten, Held-Karp vs. Brute Force, TSP auf Ebene/im Budget, 2-opt <= NN, tsp_fill nutzt Budget), alle gruen.
- `run_job_exploration_comparison_3d{,_test,_replot}.bash`; `plot_exploration_summary.py` additiv um
  tsp, tsp_fill, nooffset200, dits_old erweitert.
- Alle Laengen/Metriken in "Ebenen-Einheiten" (Welt/s) -> mit 2D vergleichbar; die 3D-Ergodik-Fehler selbst
  sind NICHT mit den 2D-Werten vergleichbar (andere Metrik).
- TSP: exakt (Held-Karp) bis 12 Punkte, darueber Nearest Neighbour + Delta-2-opt; Non-Maximum-Suppression min_dist 0,10.

Netze/Checkpoints (`find_checkpoint3d`): unet630 (`lm630unet`, ep750), dit630 (`lm630dit`, ep750), dits_old
(`dbcmpold`, ep150), nooffset200 (`nooffset200_flow3d`, ep300), selfsup (`selfsup_3d_..._final`).
Importkollision 2D/3D: `mission_plane3d` ruft `paths3d.setup()` zuerst und prueft per Assert, dass das 3D-Modul
geladen ist; `common.data` wird bewusst nicht importiert (eigene `load_truth`-Kopie).

## 3. Cluster-Laeufe
- Testjob 215476 (debug, A5000): Missionszeit Netze 11-31 s, Pfad-Baselines bis 16 s.
- Hauptjobs 218736 (unet630, dits_old), 218737 (dit630, selfsup, lintsvec), 218738 (nooffset200, linadam, rw,
  lawn, gpgoal, linray, tsp_fill), je ~4h15-4h25, COMPLETED; DBs lokal per `--merge_from` zusammengefuehrt (4200 Runs, 12 Methoden).
- Replot-Job 227239 (CPU, 1h48): 150 Einzelabbildungen + Tabellen + `summary/`.
- Gepusht auf den Cluster: `exploration_3d/{paths3d,net_zoo3d,belief3d}.py`, selfsup-Ckpt, und
  `3D_ergodic_learning/flow_matching_cond_particles_crossattn.py` (lokal mit optionalen `force`-Kwargs, Default None = unveraendert).

## 4. Ergebnisse (letzte Runde, E_truth, 175 Missionen je Methode/Kernel; kleiner = besser)
| Methode | E RBF | E AK |
|---|---|---|
| gpgoal | 3,64 | 2,23 |
| dit630 | 4,54 | 2,67 |
| tsp_fill | 4,56 | 3,07 |
| nooffset200 | 4,58 | 2,76 |
| lintsvec | 4,64 | 2,67 |
| unet630 | 4,76 | 2,94 |
| dits_old | 4,76 | 2,86 |
| linadam | 4,90 | 2,98 |
| selfsup | 5,06 | 3,33 |
| linray | 6,67 | 6,19 |
| lawn | 7,98 | 7,98 |
| rw | 19,5 | 19,5 |

- Gepaarter Bootstrap gegen lintsvec: **kein Netz ist signifikant besser in E**; unet630 (+0,27) und dits_old
  (+0,19) sind mit AK signifikant schlechter. Netze sind etwas besser im Belief-RMSE (~-0,004).
- gpgoal hat das niedrigste E, aber den schlechtesten Belief (RMSE 0,217) und geringere Abdeckung (0,90): E allein
  belohnt hier das Ansteuern der Dichte, nicht die Belief-Qualitaet.
- Budget: Baselines ~5,1, Netze 5,3-5,5 (unet630 5,50, dit630 5,27, dits_old 5,49, nooffset200 4,83), selfsup 3,97
  -> selfsup faehrt kuerzer, Vergleich zu seinen Gunsten/Ungunsten beachten.
- Mittlere Zeit je Runde (RBF): Pfad-Baselines 0,01-0,2 s, lintsvec 1,0 s, selfsup 1,0 s, dits_old 2,2 s,
  nooffset200 3,1 s, dit630 4,5 s, unet630 5,1 s (Netz-Init ~1,2-4,1 s, Verfeinerung ~1 s). Mit AK kommt ~1,2-1,3 s
  Belief-Fit je Runde dazu.
- Ebenentreue: groesste Abweichung von der Ebene vor dem Einrasten 0,043 (dit630), selfsup 0,040, sonst < 0,006.

## 5. Dateien / Ablage
- Ergebnisse (Plots, CSVs): `thesis_architecture/evaluation_full_matrix/plots/exploration_comparison_3d_flat/`
  (`summary/` mit 7 Gesamtplots + `ranking.csv`, `metrics_all.csv`, `summary_by_round.csv`,
  `summary_by_shape_final.csv`, `timing_by_method.csv`, `seg_len_by_method.csv`, `paired_vs_unet630_final.csv`)
  sowie Teilordner `_a/_b/_c` je Job. Logs in `cluster_logs/expl_cmp_3d-*`.
- **Nicht im Git** (zu gross, `.gitignore`): die vier `exploration_comparison.db` (100-600 MB). Kopien in
  `transfer/2026-11-24_exploration_3d/` (`exploration_comparison_3d_flat{,_a,_b,_c}.db`), originale lokal in den plots-Ordnern.
- Nicht eingecheckt: lokale Testordner `_smoke`, `_timingtest`.
- Lokal ohne conda: `PYTHONPATH=<repo>/bsplinax-main` setzen (bsplinax fehlt im venv).

## 6. Offene Punkte / Hinweise
- `tsp` (ohne Fuellen) ist nicht im Hauptlauf; bei Bedarf als eigener Job nachziehbar (`METHODS=tsp`).
- Nur Ebene `flat` gerechnet; `tilted` und `diagonal` sind implementiert und getestet, aber nicht als Benchmark gelaufen.
- 'A' und 'greek_upper_0' liefern identische Ergebnisse (gleiche Dichte im Datensatz, kein Bug).
