# Kontext: Active-Sensing-Benchmark MIT Hindernissen (Sessions 2026-11-16/17)

Fortsetzung von `2026-11-12_active_sensing_eval_cluster.md` (Benchmark ohne Hindernisse). Auftrag Philipp:
"genau dieses Experiment nochmal, aber mit Hindernissen: 6 Hindernisse wie beim Wuerfel, Hindernisvermeidung
in der Inferenz, in den SVGD-Solvern und in Lawnmower/Linear/allen Baselines; erst ausreichend testen, dann die
Bench am Cluster." **Status: fertig gerechnet und ausgewertet, alle 4550 Missionen kollisionsfrei.**

## 1. Entscheidungen (von Philipp bestaetigt)
- Layout `dice6`: 6 Kreise, Radius 0,08, Zentren x in {0,27; 0,73}, y in {0,2; 0,5; 0,8} (Wuerfel-Sechs),
  plus 0,01 Sicherheitsabstand (Planer halten 0,09 Abstand ein, gemessen wird gegen 0,08). ~12 % der Flaeche blockiert.
  Startpunkt (0,5; 0,5) und Lawnmower-Ecke (0,06; 0,06) sind frei.
- Kein RRT (war nur Aufzaehlung; im Benchmark gab es nie ein RRT).
- Planungsdichte Phi ist in den Hindernissen 0 (fuer alle Methoden gleich); Ground Truth bleibt unveraendert.
- Alle 13 Methoden (die 10 vom letzten Lauf + unet0, unet1890, lm1300), beide Kernel, 25 Formen x 7 Varianten,
  sonst identisches Setup und identische Seeds wie ohne Hindernisse (5 Runden, 8 Kandidaten, 500 SVGD-It., kappa 1->0,05).

## 2. Implementierung (alles opt-in; ohne `--obstacles` bitgleich wie vorher)
Neue Dateien in `thesis_architecture/evaluation_full_matrix/`:
- `workspace_obstacles.py`: Layout, Freiraum-Maske, Kollisionsstatistik, hindernisfaehige Baselines,
  Kandidatenwahl, Freiraeum-Schritt, Rueckfallebene.
- `test_workspace_obstacles.py` (12 Unit-Tests, ~3 min CPU), `check_obstacle_runs.py` (unabhaengige
  Kollisionspruefung jeder gespeicherten Bahn, Exit 1 bei Kollision), `compare_obstacle_benchmark.py`
  (gepaarter Vergleich mit/ohne Hindernisse), `run_job_exploration_comparison_obstacles[_test].bash`.
Geaendert: `run_exploration_comparison.py` (`--obstacles dice6`, `--obstacle_radius/_margin`, `--w_obstacle` 1000,
`--obstacle_guidance` 20; eigene DB/Ordner mit Suffix `obstacles_dice6_r0.08`; neue Metriken), `sample_annealed_mission.py`
(`run_mission(obstacles=...)`), `exploration/common/tsvec_svgd.py` + `adam_svgd.py` (Hindernis-Hinge-Residuum, analytischer
Jacobi = Autodiff bis 1e-15), `svgd_batched.py`, `selfsup_planner.py`, Plot-Funktionen (zeichnen Hindernisse),
`run_convergence_tests.py` / `run_timing_tests.py` (+ Jobskripte via `OBST=dice6`, vorbereitet, NICHT gerechnet).

Vermeidung je Methode:
| Methode | Hindernisvermeidung |
|---|---|
| CFM-Netze (unet630, dit630, old500, unet0, unet1890, lm1300) | Abstossung waehrend der Flow-Integration (Gewicht 20 ab t=0,3) + Polish |
| selfsup | Polish nach der Generierung (Single-Pass, kein Flow) |
| TSVEC-SVGD / Adam-SVGD | Hinge-Term w=1000 in der Energie (hoeher brachte nichts, nur schlechteres E; SVGD laesst thermischen Rest) |
| alle SVGD-Methoden danach | bester kollisionsfreier Kandidat; Freiraeum-Schritt (gewichteter Gauss-Newton auf 4096 Kurvenpunkten, Start fest); falls noch nicht frei: Rueckfallebene `guarantee_clear` (Kreisboegen) |
| linray, lintsvec/linadam-Init | Strahlen reflektieren an Waenden UND Kreisen (Billard) |
| rw | Schritt, der ein Hindernis trifft, wird neu gezogen; lange Sehnen nach dem Bogenlaengen-Resampling auf Kreisboegen |
| lawn, gpgoal | Bahnen/Linien auf dem kuerzeren Kreisbogen um Hindernisse; gpgoal-Ziele nur aus dem Freiraum |

Neue Metriken (DB `rounds`, NULL ohne Hindernisse): `obs_pen_max`, `obs_in_frac`, `seg_obs_pen_max`,
`seg_obs_len_in_frac`, `obs_viol_ref` (Verletzung des gewaehlten SVGD-Kandidaten VOR dem Freiraeumen),
`obs_fallback` (1 = Rueckfallebene noetig), `E_truth_free`, `swept06_free` (GT ausserhalb der Hindernisse).

## 3. Testverlauf (wichtig fuer die Arbeit)
- Lokal: Unit-Tests, Smoke aller 13 Methoden, Konvergenz- und Zeittest-Smoke mit Hindernissen.
- Cluster-Preflight 211217 (52 Missionen): 2 Kollisionen (lintsvec 0,0015; old500 0,014) -> alter Freiraeum-Schritt
  (Gradientenabstieg) konvergierte zu langsam. Ersetzt durch Gauss-Newton + dichte Pruefung + Rueckfallebene.
- 211242 (abgebrochen, lief auf der dgx-station V100 ~7x langsamer -> seitdem `-C rtx` in den Jobskripten).
- 211343 (78 Missionen, RTX A5000): kollisionsfrei, Rueckfallebene 3/270 SVGD-Runden.
- Hauptlauf 211389-211393 (Jobs a-e): 12 Methoden kollisionsfrei, **rw** in 31/175 Missionen je Kernel bis 0,016
  im Hindernis (Sehnen nach dem Resampling, nicht durch die Tests abgedeckt). Fix + Unit-Test, rw neu gerechnet in
  Job 212063 (TAG f): kollisionsfrei. In der Gesamt-DB sind die alten rw-Zeilen durch f ersetzt.
  **Achtung: die Job-DB `_c` enthaelt noch die alten (kollidierenden) rw-Zeilen -- nicht erneut mergen.**
- Endpruefung Gesamt-DB (`collision_check.txt`): 4550/4550 Missionen kollisionsfrei. Rueckfallebene in 0,4-1,2 %
  der SVGD-Runden (je 7-21 von 1750 Runden).

## 4. Ergebnisse (Gesamt-DB, 13 Methoden)
Gesamtrang (mittlerer Rang, 1 = best):
- RBF: lintsvec 4,92 | linadam 5,04 | lm1300 5,50 | unet630 5,51 | unet1890 5,57 | old500 6,42 | dit630 6,58 |
  unet0 7,15 | gpgoal 7,17 | linray 7,52 | selfsup 8,48 | lawn 8,93 | rw 12,22
- AK: lm1300 5,04 | lintsvec 5,16 | linadam 5,28 | unet1890 5,29 | unet630 5,41 | old500 6,07 | dit630 6,34 |
  gpgoal 6,66 | unet0 6,77 | selfsup 7,47 | lawn 9,46 | linray 9,57 | rw 12,47

Mit vs. ohne Hindernisse (End-E vs. GT, gepaart, Differenz mit-ohne, 95 %-CI ueber Formen):
- Abdeckung (swept06) sinkt bei allen Suchmethoden um 4-6 Prozentpunkte (erwartbar: ~12 % Flaeche blockiert).
- End-E sinkt bei den meisten Methoden leicht (z. B. unet630 RBF 9,38 -> 8,25, AK 6,30 -> 5,50; lintsvec RBF 8,02 -> 7,46),
  steigt bei selfsup (RBF +1,0) und gpgoal (RBF +1,2) und stark bei rw (+11,4).
  Vermutete Ursache des Rueckgangs: maskiertes Phi buendelt die Planung auf den erreichbaren Teil; noch nicht geprueft.
- linray profitiert stark (E RBF 12,4 -> 8,9, swept +24 %): ohne Hindernisse pendelten achsparallele Strahlen zwischen
  den Waenden, die Kreise brechen diese Symmetrie.
- Ranking-Bild wie ohne Hindernisse: ergodische Suche ohne Prior (lintsvec/linadam) gleichauf/vor den Netzen;
  lm1300 (ohne Hindernisse nicht im Vergleich) ist mit AK das beste Netz.

## 5. Wo liegt was
- Git: Code (oben), `plots/exploration_comparison_obstacles_dice6_r0.08/{collision_check.txt, extended/}` (Tabellen,
  `metrics_all.csv`, `obstacle_vs_free_final.{csv,png}`, `summary/01-07_*.png`, `ranking.csv`), `cluster_logs/expl_*obst*`,
  `results/timing_tests_smoke_obstacles_dice6_r0.08/`.
- `transfer/2026-11-17_obstacle_benchmark/` (1,9 GB, NICHT im Git, Struktur = Repo-Pfade ab
  `thesis_architecture/evaluation_full_matrix/`): Gesamt-DB `plots/exploration_comparison_obstacles_dice6_r0.08/
  exploration_comparison_obstacles_dice6_r0.08.db` (4550 Laeufe), Job-DBs + Einzelplots je Form `_a`..`_f`,
  Preflight-DBs `_preflight`, `_preflight_2`, `_preflight_3`, Smoke-DB, Konvergenz-Smoke `sweep.db`.
  Fuer erneute Auswertung an die Repo-Pfade zurueckkopieren.
- Am Cluster liegen dieselben Job-Ordner unter `~/Master_thesis/thesis_architecture/evaluation_full_matrix/plots/`.

## 6. Reproduzieren / Weiterarbeit
```bash
cd thesis_architecture/evaluation_full_matrix
python test_workspace_obstacles.py                       # 12 Tests
O=plots/exploration_comparison_obstacles_dice6_r0.08
python check_obstacle_runs.py --db $O/exploration_comparison_obstacles_dice6_r0.08.db
python -c "import run_exploration_comparison as r; r.aggregate(r.ResultDB('$O/exploration_comparison_obstacles_dice6_r0.08.db'), '$O/extended', list(r.METHODS))"
python plot_exploration_summary.py --dir $O/extended --db $O/exploration_comparison_obstacles_dice6_r0.08.db
python compare_obstacle_benchmark.py --obst_csv $O/extended/metrics_all.csv --free_csv plots/exploration_comparison_combined/metrics_all.csv
```
Offen:
- Einzelplots je Form mit ALLEN 13 Methoden fehlen (nur je Job-Ordner fuer dessen Methoden); `run_exploration_comparison.py
  --obstacles dice6 --plot_only --db <Gesamt-DB> --methods <13>` ist schwer -> als CPU-Job am Cluster.
- Konvergenz- und Zeittests mit Hindernissen vorbereitet (`OBST=dice6 METHODS=... sbatch run_job_convergence_tests.bash`,
  `OBST=dice6 sbatch run_job_timing_tests.bash`), noch nicht gerechnet.
- Pruefen, warum End-E mit Hindernissen bei den Suchmethoden sinkt (siehe 4).
- Commits ohne KI-Hinweis.
