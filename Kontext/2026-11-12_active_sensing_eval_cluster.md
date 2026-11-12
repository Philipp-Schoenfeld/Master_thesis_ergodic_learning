# Kontext: Active-Sensing-Evaluation am Cluster starten (Session 2026-11-12)

Uebergabe an den Rechner MIT Cluster-Zugang. Alles Noetige steht hier; Rueckfragen an Philipp sollten
nicht noetig sein (ausser den Freigaben, die CLAUDE.md ohnehin vorschreibt, siehe 0). Auf dem Quellrechner
wurde NICHTS davon gerechnet (nur Smoke-Tests mit Fake-Gewichten, ein abgebrochener Lauf, siehe 2).

## 0. Regeln, die auch dort gelten (CLAUDE.md hat Vorrang)
- Jobs nur per `sbatch`, nie `srun bash ...`. Vor JEDEM Cluster-Kommando (auch `python` auf dem Login-Knoten,
  `sbatch`) den genauen Befehl zeigen und bestaetigen lassen; Philipp hat den Start der unten stehenden Jobs
  ausdruecklich gewollt ("das am Cluster starten"), die Bestaetigung vor dem Abschicken bleibt trotzdem Pflicht.
- Code/Kommentare/Plot-Beschriftungen Englisch, mit Philipp Deutsch. Commits ohne KI-Hinweis/Co-Authored-By.
- Keine Partition angeben (`main` ist Default), immer `-t` setzen (alle Skripte tun das). Max. 3 GPUs gleichzeitig;
  darueber `--qos=scavenger` (preemptible; alle Laeufe sind zeilenweise resumierbar, bei Abbruch dasselbe Kommando
  nochmal abschicken).
- Datei-Sync (rsync/git) ist ohne Rueckfrage erlaubt.

## 1. Ziel der Aufgabe
Active-Sensing-Evaluationsmatrix (`thesis_architecture/evaluation_full_matrix`): das beste DiT, das beste
U-Net (Flow Matching) und das Self-supervised-Modell gegen Explorations-Baselines (Lawnmower, Random Walk,
Gerade zu einem aus der GP-Schaetzung gezogenen Punkt, Gerade-Initialisierung) vs. "unsere Methode" =
ergodische Suche mit und ohne Prior. Schon vorhandene Ergebnisse (DB) NICHT neu rechnen, nur Fehlendes
ergaenzen. Zusaetzlich: systematische Konvergenztests und Rechenzeittests fuer alle Varianten der Pipeline.

## 2. Stand
- Fertige DB (liegt nur auf dem Quellrechner, git-ignoriert, 254 MB):
  `plots/exploration_comparison/exploration_comparison.db` mit `unet630, dit630, rw, lawn, linadam`
  (je 2 Kernel x 25 Holdout-Formen x 7 Varianten = 350 Missionen je Methode).
- Neu vorbereitet und lokal im Smoke-Test geprueft (Mechanik, nicht Zahlen): fuenf neue Methoden, Timing-Test,
  Konvergenztests, Merge-/Auswerte-Skripte, Plot-Skript fuer 13 Methoden.
- Auf dem Quellrechner lief kurz der echte Vergleichslauf und wurde abgebrochen: die lokale DB enthaelt deshalb
  33 vollstaendige `selfsup`-Missionen. Das ist unkritisch (gleiche Seeds, der Merge ueberspringt vorhandene
  Zeilen; Clusterergebnisse und lokale Zeilen duerfen sich ueberlappen).
- Auf dem Cluster ist noch nichts gelaufen.

## 3. Entscheidungen (Philipp hat sie bestaetigt; nicht neu diskutieren)
- Bestes U-Net = `old500` (`transfer/netz2d_startpunkt.pt`, lambda=500, End-E 8,20 laut
  `results/as_models_20261028/summary.json`; unet0 8,80, unet630 9,19, unet1890 9,13, lm1300 9,54).
  Bestes DiT = `dit630` (einziges 2D-DiT). Self-supervised = `selfsup`. Siehe `best_models.json`.
- Mit Prior = `dit630`, `old500`, `selfsup` (+ `unet630` als Paar-Referenz zu `dit630`).
- Ohne Prior (ergodische Suche) = `lintsvec`: dieselben 8 Geraden wie `linadam`, verfeinert mit demselben
  TSVEC-SVGD wie die Netze.
- Baselines: `rw` (Random Walk), `lawn` (Lawnmower), `gpgoal` (Gerade zu Ziel ~ geklemmtes GP-Posterior-Mittel,
  Runde 0 uniform, bei erreichtem Ziel neues Ziel bis Wegbudget 6,1/Runde), `linray` (beste der 8 Geraden gegen
  das Rundenziel, unverfeinert gefahren = "Gerade-Initialisierung").
- Setup aller Missionen identisch zum ersten Lauf: 5 Runden, 8 Kandidaten, 500 SVGD-Iterationen, kappa 1->0,05,
  volle Ausfuehrung, Seed = Formindex*100 + Variante, Kernel `rbf` und `ak`.

## 4. Dateien (alle in `thesis_architecture/evaluation_full_matrix/`)
| Datei | Zweck |
|---|---|
| `run_exploration_comparison.py` | Vergleichslauf (resumierbar; `--dry_run` zeigt offene Missionen; `--methods`, `--db`, `--out_dir`, `--max_minutes`) |
| `mission_baselines.py` | `gpgoal` / `linray` |
| `selfsup_planner.py` | Selfsup-Wrapper; Default-Checkpoint wird in `checkpoints/` ODER `<repo>/transfer/` gesucht |
| `select_best_models.py`, `best_models.json` | Modellwahl (nur lesend, fuer die Doku) |
| `merge_exploration_dbs.py` | Fuegt eine Extra-DB add-only in die Haupt-DB ein (`--src`, `--dst`, `--dry_run`) |
| `plot_exploration_summary.py` | Gesamt-Plots/Tabellen fuer alle 13 Methoden (`--dir`, `--db`, `--methods`) |
| `run_timing_tests.py` | Rechenzeittests (Teile planner, init, refiner, belief, missions) |
| `run_convergence_tests.py` | Konvergenztests (Teile sweep, trace, analysis) |
| `check_active_sensing_setup.py` | Pre-Flight-Check (Checkpoints, Daten, bsplinax, CUDA), `--link` verlinkt fehlende Checkpoints |
| `run_job_exploration_comparison_extra.bash` | sbatch: fehlende Vergleichsmethoden (`METHODS=`, `TAG=`, `MAX_MIN=`) |
| `run_job_convergence_tests.bash` | sbatch: Konvergenztests ein Job je Methode (`METHODS=`, `PARTS=`) |
| `run_job_timing_tests.bash` | sbatch: Zeittests, GPU-Typ gepinnt (`-C rtx3090`) |
| `ACTIVE_SENSING_EVAL_PLAN.md` | ausfuehrlicher Plan (Design der Konvergenz-/Zeittests, Schaetzungen, Vorbehalte) |

Geaenderte bestehende Dateien: `sample_annealed_mission.py` (Timing-Felder, `conv_iters`, `path_fn` mit Belief),
`sample_annealed_mission_overview.py` (neue Methoden, `find_checkpoint` ignoriert `*flow3d*`-Checkpoints).

## 5. Vorbereitung auf dem Cluster
Der Cluster-Pfad der Job-Skripte ist `~/Master_thesis/thesis_architecture/evaluation_full_matrix`
(`cd` steht in den Skripten; Conda-Env `thesis`).

1. Code holen: `cd ~/Master_thesis && git pull` (oder vom Cluster-Laptop per rsync nach
   `Master_thesis/thesis_architecture/` pushen).
2. Checkpoints. Die Suche (`find_checkpoint`) liest `<repo>/transfer/*.pt` und `<repo>/transfer/*/*.pt`.
   Noetig (Dateinamen siehe Ausgabe des Checks): `unet630`, `dit630`, `old500`, `selfsup`; optional `unet0`,
   `unet1890`, `lm1300` (nur Zeittest). Die Netze wurden groesstenteils auf dem Cluster trainiert, liegen also dort
   evtl. schon in `thesis_architecture/checkpoints/`. Pruefen und verlinken:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate thesis
   cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix
   python check_active_sensing_setup.py --link      # CPU-only, kurz; zeigt MISSING-Zeilen
   ```
   Was danach noch `MISSING` ist, muss vom Quellrechner (`<repo>/transfer/`, `transfer/2026-10-28/`) per
   Drive/USB/scp nach `~/Master_thesis/transfer/` kommen (jeweils ~350 MB, DiT 160 MB):
   - `transfer/netz2d_startpunkt.pt` (old500)
   - `transfer/selfsup_selfsup_particles_date_08_12_10h13min_nxi25_D384_N256_K8_div10_final.pt`
   - `transfer/2026-10-28/cond_particles_..._LAMERG50_ERGLOSS-SINKHORN-w630-..._final.pt` (unet630, Datum 10_27_10h43min)
   - `transfer/2026-10-28/cond_particles_..._DIT-d12-h6_LAMERG50_..._final.pt` (dit630, Datum 10_28_08h23min)
   - optional: `..._LAMERG0_final.pt` (unet0), `..._LAMERG75_...w1890..._final.pt` (unet1890), `..._LMTEACHER_..._final.pt` (lm1300)
   Der Check muss mit `SETUP OK` enden, bevor Jobs abgeschickt werden.
3. Datensaetze/Formen (`ergodic_dataset_generator/*.db`) liegen im Git; `bsplinax` kommt aus der Conda-Umgebung
   (der Check meldet, falls nicht). Optionaler NP-Belief-Test: `transfer/2026-10-28/belief_np_..._final.pt`.
4. Trockenlauf (CPU, Sekunden): `python run_exploration_comparison.py --dry_run` muss fuer `selfsup, old500,
   lintsvec, gpgoal, linray` je 350 Missionen melden (auf dem Cluster ohne Haupt-DB: 350 je Methode auch fuer
   die fuenf alten, diese werden mit `--methods` ausgeschlossen, siehe unten).

## 6. Jobs (alle per `sbatch`, im Ordner `evaluation_full_matrix`)
Die Zeiten sind Schaetzungen aus den Missionszeiten des ersten Laufs (RTX 2070) und noch nicht am Cluster gemessen.

Welle 1 - fehlende Vergleichsmethoden, 3 parallele Jobs (1750 Missionen gesamt, je ~1-3 h; lokal waren es 10-25 s je Mission), jeder mit eigener DB:
```bash
TAG=a METHODS=selfsup,old500 sbatch run_job_exploration_comparison_extra.bash   # 700 Missionen
TAG=b METHODS=lintsvec       sbatch run_job_exploration_comparison_extra.bash   # 350
TAG=c METHODS=gpgoal,linray  sbatch run_job_exploration_comparison_extra.bash   # 700, billig (kein SVGD)
```
Ausgabe: `plots/exploration_comparison_extra_<TAG>/exploration_comparison_extra.db` (+ Plots in `.../extended/`, fuer
die Auswertung irrelevant). Bei Zeitlimit/Abbruch: dasselbe Kommando erneut abschicken (setzt fort).

Welle 2 - Konvergenztests, ein Job je Methode (je ~1,5-3 h; sweep = Iterationsbudget 0..1000 auf Missionsebene,
600 Missionen = 25 Formen x 3 Varianten x 8 Niveaus, AK; trace = ein 1000-Iterationen-Lauf je Form). Die Jobs rechnen
nur (`--no_tables`), Tabellen/Plots entstehen einmal nach dem Zusammenfuehren (Abschnitt 7). Erst starten, wenn Welle 1 fertig ist
(GPU-Limit 3), oder die ueberzaehligen Jobs mit `--qos=scavenger` abschicken:
```bash
METHODS=lintsvec sbatch run_job_convergence_tests.bash
METHODS=old500   sbatch run_job_convergence_tests.bash
METHODS=dit630   sbatch run_job_convergence_tests.bash
METHODS=selfsup  sbatch --qos=scavenger run_job_convergence_tests.bash     # oder per --dependency=afterany:<ID>
METHODS=linadam  sbatch --qos=scavenger run_job_convergence_tests.bash
```
Ausgabe je Job: `results/convergence_tests_main_<method>/` (`sweep.db`, `trace/*.npz`, CSVs/PNGs). Hinweis: Ohne
Haupt-DB wird das 500-Iterationen-Niveau mitgerechnet (+1/8 Aufwand); das ist gewollt und unkritisch.

Welle 3 - Zeittests (messen nur auf EINEM GPU-Typ; waehrend des Laufs moeglichst keine eigenen Jobs auf demselben
Knoten, sonst Zeitrauschen; ~1,5-2 h):
```bash
sbatch run_job_timing_tests.bash        # pinnt -C rtx3090; Ausgabe results/timing_tests_cluster_rtx3090/ inkl. hardware.json
```
Zuletzt: Zusammenfuehren und Auswerten der Konvergenzjobs (Abschnitt 7).

## 7. Ergebnisse zurueckholen und zusammenfuehren
Die Haupt-DB (254 MB) ist git-ignoriert und liegt nur auf dem Quellrechner; die Extra-DBs ebenso nicht im Git (zu
gross). Rueckweg:
1. Cluster -> Cluster-Laptop (rsync-Pull, erlaubt): `plots/exploration_comparison_extra_*/`,
   `results/convergence_tests_main_*/`, `results/timing_tests_cluster_rtx3090/`.
2. Auf den Rechner mit der Haupt-DB bringen (Drive/USB), oder die Haupt-DB zum Cluster-Laptop kopieren. Dort:
   ```bash
   cd thesis_architecture/evaluation_full_matrix
   for t in a b c; do python merge_exploration_dbs.py --src plots/exploration_comparison_extra_$t/exploration_comparison_extra.db; done
   python run_exploration_comparison.py --plot_only --methods unet630,dit630,rw,lawn,linadam,selfsup,old500,lintsvec,gpgoal,linray
   python plot_exploration_summary.py --dir plots/exploration_comparison/extended \
       --methods rw,lawn,gpgoal,linray,linadam,lintsvec,dit630,old500,selfsup,unet630
   python run_convergence_tests.py --parts analysis
   ```
   `merge_exploration_dbs.py --dry_run` zeigt vorher, was ergaenzt wuerde.
3. Konvergenz-Ergebnisse zusammenfuehren (sweep.db hat dasselbe Schema, der Merge ist getestet; die Methodennamen
   `<method>@<n_iters>` sind eindeutig) und einmal auswerten:
   ```bash
   mkdir -p results/convergence_tests_main/trace
   for m in lintsvec old500 dit630 selfsup linadam; do
     python merge_exploration_dbs.py --src results/convergence_tests_main_$m/sweep.db --dst results/convergence_tests_main/sweep.db
     cp results/convergence_tests_main_$m/trace/*.npz results/convergence_tests_main/trace/
   done
   python run_convergence_tests.py --parts sweep,trace,analysis --tag main        --methods lintsvec,old500,dit630,selfsup,linadam --main_db plots/exploration_comparison/exploration_comparison.db
   ```
   Der letzte Aufruf rechnet nichts mehr (alle Eintraege vorhanden; `--methods` MUSS explizit gesetzt sein, sonst
   werden zusaetzlich `unet630` u. a. berechnet) und erzeugt `sweep_by_iters.csv`, `sweep_saturation.csv`,
   `sweep_convergence.png`, `sweep_cost_vs_quality.png`, `trace_*`, `variant_convergence.*`,
   `belief_convergence.*`, `coverage_reach.csv`. Die Teile variants/reach/belief brauchen die Haupt-DB (wird sonst
   uebersprungen). Das 500er-Niveau von `dit630`/`linadam` liest er aus der Haupt-DB, wenn vorhanden.
4. Ergebnisse fuer die Arbeit: Gesamttabelle/Ranking unter `plots/exploration_comparison/extended/` und
   `.../summary/`, Konvergenz- und Zeitplots unter den `results/*`-Ordnern.

## 8. Bekannte Punkte
- `linray` ist in Runde 0 deterministisch (flaches Ziel); Varianten unterscheiden sich nur durch Messrauschen,
  daher kleine Varianz - Eigenschaft der Baseline.
- Der Selfsup-Checkpoint hat geringe Kandidatendiversitaet (diversity ~0,08); er bekommt wie die Netze den
  Cloud-Cutoff 0,2.
- `linadam` (Adam-SVGD, lr 2e-3) konvergiert in 500 Iterationen langsam; das ist der alte Befund, keine Fehlfunktion.
- Zeitmessung: lokal gespeicherte `runs.seconds` der alten Laeufe (RTX 2070, inkl. Metriken) sind NICHT mit
  Clusterzeiten vergleichbar; fuer Zeitvergleiche nur `timing_tests_*` verwenden.
- NP-Belief-Zeit im Zeittest ist lokal nicht getestet (in try/except gekapselt).
- Aus Versehen geloeschte und per `git checkout` wiederhergestellte `-wal/-shm`-Dateien unter
  `results/as_models_20261028_*` sind unveraendert (git status dort sauber); nicht erneut aufraeumen.
- Smoke-Test (lokal, Minuten, ohne Cluster-Last pruefbar): `run_exploration_comparison.py --smoke`,
  `run_timing_tests.py --smoke`, `run_convergence_tests.py --smoke`.
