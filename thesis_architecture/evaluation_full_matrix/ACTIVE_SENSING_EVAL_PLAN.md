# Active-Sensing-Evaluation: Ergänzungen (Stand 2026-11-12, nichts davon ausgeführt)

Geschrieben für Philipp (Ablaufplan, Entscheidungen, Aufwand).

## 1. Was schon existiert (wird nicht neu gerechnet)

`plots/exploration_comparison/exploration_comparison.db`: 5 Methoden × 2 Kernel (RBF, AK) × 25 Holdout-Formen × 7 Varianten,
5 Runden, 8 Kandidaten, 500 SVGD-Iterationen, κ 1→0,05, volle Ausführung, Seed = Formindex·100 + Variante.
Methoden: `unet630`, `dit630`, `rw` (Random Walk), `lawn` (Lawnmower), `linadam` (Geraden + Adam-SVGD).
Plus Metriken in `metrics_all.csv`, Tabellen, `summary/` (Rank-Scorecard usw.). Die zweite Pipeline
(`run_mission_eval.py`, Ergebnisse `results/as_models_*`, `active_sensing_20261023_*`) bleibt unverändert.

## 2. Was fehlt und neu vorbereitet ist

| Methode | Schlüssel | Bedeutung |
|---|---|---|
| Self-supervised + TSVEC | `selfsup` | Netz ohne Startkonditionierung, danach derselbe TSVEC-SVGD |
| bestes U-Net | `old500` | laut `select_best_models.py` (niedrigster End-E, Mittel beider Wissensstufen aus `as_models_20261028`): 8,20 vs. `unet0` 8,80, `unet630` 9,19. Unterschiede liegen innerhalb ~1 SEM |
| bestes DiT | `dit630` | einziges 2D-DiT, bereits in der DB |
| Ergodic Search **ohne** Prior | `lintsvec` | dieselben 8 Geraden wie `linadam`, verfeinert vom selben TSVEC-SVGD wie die Netze |
| Gerade zu GP-Punkt | `gpgoal` | Ziel ∝ geklemmtem GP-Posterior-Mittel (Runde 0 ohne Messung: uniform); bei erreichtem Ziel neues Ziel, bis das Wegbudget (6,1/Runde) aufgebraucht ist |
| Gerade-Init ohne Suche | `linray` | beste der 8 Geraden gegen das Rundenziel, unverfeinert gefahren |

„Mit Prior“ = `dit630`, `old500`, `selfsup` (+ `unet630` als Paar-Referenz). Alle neuen Methoden teilen Formen, Seeds, Belief und
Budget mit den vorhandenen, sind also paarweise vergleichbar. Zusätzlich vorhanden, aber nicht im Default: `unet0`, `unet1890`, `lm1300`.

## 3. Reihenfolge

1. (lokal, 1 Min.) `python select_best_models.py` → `best_models.json` (Modellwahl prüfen/ändern: `--methods` im Runner).
2. Selfsup-Checkpoint lokal holen (rsync-Pull, liegt nur auf dem Cluster) **oder** den Cluster-Job nutzen.
3. Fehlende Vergleichsläufe: `python run_exploration_comparison.py --dry_run` zeigt die Zahl der Missionen je Methode.
   Lokal: `python run_exploration_comparison.py` (resumierbar, `--max_minutes`). Cluster: `sbatch run_job_exploration_comparison_extra.bash`,
   danach `python merge_exploration_dbs.py --src plots/exploration_comparison_extra/exploration_comparison_extra.db`.
   Ergebnisse (Tabellen/Plots der erweiterten Menge) landen in `plots/exploration_comparison/extended/`, die ersten Plots bleiben unberührt.
4. Zusammenfassung: `python plot_exploration_summary.py --dir plots/exploration_comparison/extended`
   (`--methods rw,lawn,gpgoal,linray,linadam,lintsvec,dit630,old500,selfsup,unet630` für die Kernauswahl).
5. Zeittests: `sbatch run_job_timing_tests.bash` (ein GPU-Typ, `hardware.json` wird mitgeschrieben).
6. Konvergenztests: ein Job je Methode (`run_job_convergence_tests.bash`, Kommentar im Skript), zum Schluss `--parts analysis` lokal.

## 4. Konvergenztests (`run_convergence_tests.py`)

- **sweep** (Mission-Ebene): n_iters ∈ {0,10,25,50,100,250,500,1000}, 25 Formen × 3 Varianten, AK, Methoden `dit630`, `old500`, `selfsup`,
  `lintsvec`, `linadam`. Das 500er-Niveau wird aus der Vergleichs-DB gelesen, wenn vorhanden. Ausgabe: End-E vs. Iterationen,
  gepaarte Differenz zu `lintsvec` (95 %-Bootstrap über Formen), Sättigungsbudget (5 %/10 % vom Wert bei 1000), Kosten-Qualitäts-Plot.
- **trace** (Solver-Ebene): ein 1000-Iterationen-Lauf je Methode und Form, E(Iteration) aller Kandidaten gegen das Rundenziel; n90/n99,
  Restgewinn der zweiten Hälfte, „wie viele Iterationen spart der Prior“ (`trace_prior_savings.csv`).
- **analysis** (ohne Rechnung): Varianten-Konvergenz (reichen 7 Seeds? CI-Halbbreite und Ranking-Stabilität vs. Variantenzahl;
  auf der vorhandenen DB getestet), Coverage-Erreichen, Belief-Konvergenz über die Runden, Solver-Konvergenz der `run_mission_eval`-Shards.

## 5. Zeittests (`run_timing_tests.py`)

Teile: `planner` (Netz-Forward vs. n_init), `init` (netzfreie Kandidatenquellen), `refiner` (TSVEC/Adam, s/Iteration vs. n_init und
Iterationen), `belief` (RBF/AK/NP Update + Posterior vs. gespeicherte Messungen), `missions` (komplette Mission je Methode × Kernel,
Stufen `t_target/t_init/t_refine/t_belief`). Die Vergleichsläufe schreiben die Stufenzeiten ab sofort in die DB (`t_*`, `timing_by_method.csv`);
ältere Läufe haben dort NULL und nur `runs.seconds` (inkl. Metriken, lokal auf der RTX 2070).

## 6. Geschätzter Aufwand (Schätzungen, noch nicht gemessen)

| Schritt | Missionen | Zeit |
|---|---|---|
| fehlende Vergleichsmethoden (5 × 350) | 1750 | ~4–5 h lokal (aus den Zeiten des ersten Laufs), Cluster schneller |
| Sweep je Methode | 600 (8 Iterationsniveaus x 75) | ~1,5–3 h (Summe der Niveaus ≈ 3,9 x eine 500er-Mission) |
| Trace je Methode | 25 | ~0,25–0,5 h |
| Zeittests | – | ~1,5–2 h |

## 7. Bekannte Punkte / Vorbehalte

- `old500` ist ein älteres, anders trainiertes Modell (λ=500, nicht dieselbe Datenbasis wie die 2026-10-28-Serie); die Wahl ist
  entschieden (bestes U-Net laut `as_models_20261028`). Das Paar `unet630`/`dit630` bleibt für den Backbone-Vergleich bei gleichem Training.
- `linray` ist in Runde 0 deterministisch (flaches Ziel), Varianten unterscheiden sich nur über das Messrauschen → kleinere Varianz als bei
  den anderen Methoden, das ist ein Merkmal der Baseline.
- Selfsup bekommt wie die Netze den Cloud-Cutoff 0,2; der Checkpoint hat laut `selfsup_planner.py` geringe Kandidatendiversität.
- `find_checkpoint` schließt jetzt `*flow3d*`-Checkpoints aus (lokal hätte `dit630` sonst auf ein 3D-DiT gezeigt).
- Smoke-Tests nur mit Fake-Selfsup-Gewichten; die Zahlen daraus sind bedeutungslos, geprüft wurde die Mechanik.
