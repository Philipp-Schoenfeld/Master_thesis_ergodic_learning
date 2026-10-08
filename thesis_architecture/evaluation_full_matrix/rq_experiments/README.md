# rq_experiments — Sun-Vergleich, RQ1–RQ6

Experimente zur Thesis-Doc ["Flow Matching Ergodic Coverage als
Hauptreferenz"](../../../). Jedes Skript ist lokal lauffähig (Smoke-Test in
Sekunden bis Minuten) und cluster-tauglich (resumable wo sinnvoll, `--out_tag`,
`sbatch`-fähig) — siehe `../run_job_mission_eval.bash` für das Muster.

**Sprachregel:** Dieser Ordner entstand über mehrere Sessions. Alles ab dem
2026-10-06-Update (RQ3/RQ4/RQ6, `sinkhorn_jax.py`, `alm_solver.py`, alle neuen
Funktionen in `exp_common.py`/`paper_style.py`) ist durchgehend Englisch
(Code, Kommentare, Docstrings) gemäß der projektweiten Sprachregel in
`CLAUDE.md`. Die älteren RQ1/RQ2/RQ5-Dateien sind größtenteils Deutsch
kommentiert und wurden nicht rückwirkend übersetzt — nur Textstellen, die bei
der Erweiterung ohnehin angefasst wurden.

## Stand je Forschungsfrage

| RQ | Skript(e) | Status | Lokal getestet |
| --- | --- | --- | --- |
| RQ1 Laufzeit | `rq1_rq2_runtime.py` + `rq1_plot.py` | **fertig** | ja, 5 echte Trials, CPU |
| RQ2 Robustheit bei Budget | `rq1_rq2_runtime.py` + `rq2_plot.py` (gleiche CSV) | **fertig** | ja, dieselben 5 Trials |
| RQ3 GP-Ziele | `rq3_gp_targets.py` | **fertig** | ja, 2 Trials, `half_known` |
| RQ4 Samples/Sinkhorn | `sinkhorn_jax.py` + `rq4_sinkhorn_samples.py` | **fertig** | ja, 2 Icons × 2 Starts |
| RQ5 Dynamiken | `dynamics_zoo.py` + `rq5_dynamics.py` | **fertig** | ja, echte CFM-Kurve, 3 Trials |
| RQ6 λ₀/ALM | `alm_solver.py` + `rq6_alm_lambda0.py` | **fertig, kleinerer Scope als urspruenglich geplant — siehe unten** | ja, 10 Trials |

Alle sechs Forschungsfragen haben jetzt ein lauffähiges Experiment. "Fertig"
heißt: End-to-End getestet mit echten (nicht simulierten) Berechnungen auf
kleinem Umfang — keines ist in großem Maßstab (100+ Trials, volle
Shape-/Icon-Menge) gelaufen; das ist der nächste Schritt, nicht mehr diese
Session.

## Gemeinsamer Kern

- `exp_common.py`, `dynamics_zoo.py`: siehe deren eigene Docstrings (RQ1/RQ2/RQ5,
  größtenteils unverändert aus der vorigen Session).
- `exp_common.py` wurde für RQ4 erweitert (`timed_sinkhorn_solve`,
  `load_icon`/`icon_density_grid`/`coverage_error`,
  `cfm_raw_curve_from_particles`) — der gemeinsame Schleifenkörper von
  `timed_solve` und `timed_sinkhorn_solve` ist jetzt in `_timed_loop`
  faktorisiert, damit die beiden Referenzflüsse (Stein, Sinkhorn) nicht
  auseinanderdriften können. Verifiziert: `timed_sun_solve` liefert nach
  diesem Refactor exakt dieselben Werte wie vorher (siehe Kommentar im
  Git-Verlauf dieser Datei).
- `exploration/common/sun_refine.py` wurde um `make_grid_score` erweitert
  (ein eigenständiger JAX-Score aus einem beliebigen Dichtegitter, für RQ3s
  GP-Belief — nicht nur für `ergodic_solver`s GMM-Ziele).

## Wichtige Lektionen aus den Testläufen

### RQ1/RQ2/RQ5 (vorige Session)

1. **CPU-Zeiten sind für RQ1 nicht aussagekräftig.** Ein CFM-Vorwärtsdurchlauf
   kostet hier (keine GPU) ~13-17s, auf einer RTX 2070 Super laut
   `run_mission_eval.py`s eigener Doku ~0.23s. **Vor dem echten RQ1-Lauf:
   `--device cuda`.**
2. Finite-Differenzen zur Steuerungsrekonstruktion sind fragil (Point-Mass
   2. Ordnung) — behoben durch Wiederverwendung von `ergodic_solver._pid_track`.
3. Dubins' Geschwindigkeit muss aus der Kurve kommen (Bogenlänge ÷
   Trajektoriendauer), nicht aus einer Konstante — sonst schießt die Bahn
   schon bei 0 Iterationen über das Einheitsquadrat hinaus.
4. `paper_style.grouped_bars`/`violin` zeichnen "nicht erreicht" (NaN)
   explizit schraffiert bzw. mit Fußzeile, nie als Balken der Höhe 0.

### RQ3 (neu)

- Kein Wrapper um `run_mission_eval.py`: diese Datei ist CFM-only und
  batcht ganze Shape-Sets auf der GPU. RQ3 braucht Sun als gleichwertige
  Alternative im selben Schleifenslot, also eine eigene, einfachere
  Pro-Trial-Schleife (kein Batching über Formen).
- "Sun pro Runde" ist ein lokaler Plan (Gerade zum Dichteschwerpunkt +
  FM-Stein), kein globaler mehrteiliger Pfad wie beim Datengenerator — es
  gibt keine natürliche GMM-Komponentenstruktur, durch die ein
  Rundenplan routen könnte.
- `metrics_explore_exploit.swept_mass_fraction` ist die hier verwendete
  Übersetzung von Suns "gesammelte positive Lebenssignale" (kontinuierliche
  Dichte statt binärem Signal) — explizit benannt, nicht stillschweigend
  gleichgesetzt.

### RQ4 (neu)

- **`ott-jax` importiert in diesem Environment nicht**
  (`jax==0.4.35`, `ott-jax==0.6.0`: `TypeError: register_dataclass() missing
  2 required positional arguments` — `ott` ruft die alte Zwei-Positionsargument-
  Signatur auf, `jax` verlangt die neue; keine neuere `ott`-Version behebt das).
  Gelöst durch eine eigene, ~80-zeilige log-domain-stabilisierte
  Sinkhorn-Divergenz in reinem JAX (`sinkhorn_jax.py`), statt das projektweite
  `jax` zurückzustufen (Risiko für `lqrax`/`ergodic_solver.py`).
- **Suns Kostenskalierung (`* 1e3`) ist kein Detail.** Ohne sie ist der
  Sinkhorn-Gradient bei derselben Schrittweite wie Stein (0.01) um den Faktor
  ~2700 zu klein — 300 Iterationen bewegen die Bahn dann kaum (gemessen:
  Coverage-Fehler 0.029 → 0.027 statt → 0.0004). Erst `cost_scale=1e3` +
  Suns eigenes `step_size=0.005` bringen die erwartete schnelle Konvergenz.
- Der Sinkhorn-Referenzfluss **muss JIT-kompiliert werden**
  (`jax.jit`, wie `es._make_stein_grad` es für Stein schon tut) — ohne das
  dauert ein 300-Iterationen-Lauf mehrere Minuten statt ~10 Sekunden (jede
  äußere Iteration würde die 50er-Sinkhorn-Innenschleife im Eager-Modus neu
  ausführen statt eine kompilierte Fassung wiederzuverwenden).
- **Suns Coverage-Fehler IST die Fourier-Ergodik-Metrik von Mathew und Mezić**
  (2011, bestätigt per Websuche während dieser Session) — exakt
  `exp_common.ergodic_error`/`target_coeffs`, bereits aus RQ1/RQ2/RQ5
  vorhanden. Keine zweite Metrik nötig; `coverage_error` in `exp_common.py`
  ist nur ein benannter Alias dafür.
- Icon-Punktwolken direkt aus
  [MurpheyLab/lqr-flow-matching](https://github.com/MurpheyLab/lqr-flow-matching)
  (`tutorials/test_objects/2d/*.txt`) geholt und unter `sun_icons/` abgelegt
  (10 Dateien, ~1.1 MB) — kein Netzwerkzugriff zur Laufzeit nötig.
- `load_icon` samplet standardmäßig auf 256 Partikel herunter (Projekt-
  Konvention `N_PARTICLES=256`), sowohl aus Geschwindigkeitsgründen (Sinkhorn-
  Kostenmatrix) als auch damit CFMs Partikel-Konditionierung (über
  `cfm_raw_curve_from_particles`, NEU — konditioniert direkt auf Samples,
  ganz ohne Dichtegitter) in derselben Größenordnung arbeitet wie beim
  Training.

### RQ6 (neu, kleinerer Scope als ursprünglich gedacht)

- **TSVEC (Li et al.) selbst hat keine Lagrange-Multiplikatoren** — Zitat aus
  dem Paper: *"the current framework can be readily extended to an Augmented
  Lagrangian constrained optimization approach; however, for simplicity, we
  leave this for future work."* "SE(3)-Constraints" im Titel meint dort
  Optimierung auf der Lie-Gruppe (Retraktion, Paralleltransport,
  SE(3)-Kernel), nicht ein Lagrange-beschränktes Problem.
- **Flow-Opts λ ist ein anderes λ als im CLAUDE.md-Konzept angenommen:**
  die Dualvariable eines ADMM-artigen Fixpunkt-Projektionslösers
  (`A ξ=b`, `Gξ≤h`, `g(ξ)≤0`) für Multi-Roboter-Kollisionsvermeidung, mit
  einem SEPARAT trainierten, selbstüberwacht (Fixpunkt-Residuum-Loss) auf
  diesem Löser trainierten Initialisierungsnetz — nicht der im Code
  existierende `LambdaHead` (ein einzelner MLP-Kopf auf gepoolten Features
  des Hauptnetzes).
- **Was `alm_solver.py` deshalb ist:** der tatsächlich fehlende Baustein (ein
  echter Augmented-Lagrangian-Löser mit expliziten Multiplikatoren für
  Startpunkt-Gleichheit, Workspace-Ungleichheit und eine quadratische
  Hindernis-Ungleichheit auf B-Spline-Kontrollpunkten) — **ohne** das
  trainierte Prädiktor-Netz, das den Umfang dieser Session gesprengt hätte
  (neue Architektur, neue selbstüberwachte Trainingsschleife, GPU-Budget).
  `rq6_alm_lambda0.py` testet stattdessen die engere, trotzdem aussagekräftige
  Frage: hilft ein mit-Null-Kosten warmgestartetes λ/μ (aus der vorigen
  Runde einer Zwei-Runden-Neuplanung) überhaupt, bevor man in einen
  Prädiktor investiert?
- **Zwei Implementierungsfehler dabei gefunden und behoben** (siehe
  Git-Verlauf von `alm_solver.py`): (1) reines Gradientenabstiegs-Innenloop
  bei `step_size=0.05` divergiert auf dieser Zielfunktionsskala (W_ERGODIC=600)
  binnen weniger äußerer Iterationen zu NaN — behoben durch Adam mit
  denselben Hyperparametern wie `svgd_batched.py` (`ADAM_LR=2e-3`). (2) eine
  Zeile, die den Startpunkt-Kontrollpunkt in JEDEM inneren Schritt auf seinen
  Ausgangswert zurücksetzte, verhinderte, dass die Gleichheitsbedingung
  überhaupt vom Löser erfüllt werden konnte (die Verletzung blieb exakt auf
  ihrem Anfangswert stehen) — ersatzlos entfernt, der Lagrange-Term selbst
  übernimmt das Heranziehen.
- **Vorläufiges Ergebnis (10 Trials, Smoke-Umfang):** kein klarer Gewinn durch
  Multiplikator-Warmstart (Median der Differenz = 0 Iterationen, 4 von 10
  Trials schneller warm, 4 schneller kalt, 2 gleich). Das ist ein ehrlicher,
  vorläufiger Nullbefund bei kleiner Stichprobe — keine Aussage darüber, ob
  ein trainierter Prädiktor (der eine informiertere Schätzung als "vorige
  Runde" liefern könnte) ebenfalls keinen Effekt hätte.
- `tol=5e-3` statt `1e-3`: bei `1e-3` plateaut die Hindernis-Ungleichheit oft
  knapp über der Schwelle (ein Kontrollpunkt bleibt ~2-3% zu nah am
  Hindernis) und erreicht `n_outer=80` nie — ein genuin aktiver Constraint
  nahe dem Optimum, keine fehlerhafte Abbruchbedingung.

## Usage

```bash
# RQ1 + RQ2 (gleiche CSV, zwei Auswertungen)
python rq1_rq2_runtime.py --n_trials 100 --out_tag rq1_rq2_full --device cuda
python rq1_plot.py --out_tag rq1_rq2_full
python rq2_plot.py --out_tag rq1_rq2_full --budgets 0.1,0.5,2.0

# RQ3 (Replanning unter GP-Belief)
python rq3_gp_targets.py --n_trials 20 --conditions half_known,ten_samples,none_known \
    --out_tag rq3_full --device cuda

# RQ4 (Sinkhorn + Sun-Icons + eigene Samples)
python rq4_sinkhorn_samples.py --n_starts 10 --out_tag rq4_full --device cuda

# RQ5 (qualitatives Raster + quantitativer Balken)
python rq5_dynamics.py --n_trials 30 --out_tag rq5_full --device cuda

# RQ6 (Augmented-Lagrangian-Löser, Multiplikator-Warmstart)
python rq6_alm_lambda0.py --n_trials 50 --out_tag rq6_full   # kein GPU/CFM noetig

# Schnelle lokale Smoke-Tests (Sekunden bis wenige Minuten)
python rq1_rq2_runtime.py --n_trials 3 --checkpoints 0,10,25,50 --no_cfm --out_tag smoke
python rq3_gp_targets.py --n_trials 2 --methods sun,cfm_raw --max_rounds 5 --out_tag smoke
python rq4_sinkhorn_samples.py --icons star,heart --n_starts 2 --out_tag smoke
python rq5_dynamics.py --n_trials 2 --skip_quant --out_tag smoke
python rq6_alm_lambda0.py --n_trials 10 --out_tag smoke
```

Alle Ausgaben landen unter `../results/<out_tag>/` (CSV + `plots/`), wie bei
den anderen `evaluation_full_matrix`-Runnern. Cluster-Jobs ausschließlich per
`sbatch` einreichen (Projektregel) — keines dieser Skripte wurde bislang auf
dem Cluster ausgeführt.

## Was für größere Läufe noch sinnvoll wäre

- RQ1/RQ2: `--device cuda`, deutlich mehr Trials (Sun selbst nutzt 100).
- RQ3: mehr Trials, und ein Vergleich gegen `ground_truth` als Kontrollbedingung.
- RQ4: alle 10 Icons, mehr Startpunkte, zusätzlich eigene (Buchstaben-)Formen
  als zweite Samplequelle neben Suns Icons.
- RQ5: mehr Trials für stabilere Mediane im Balkendiagramm.
- RQ6: deutlich mehr Trials (10 ist zu wenig für eine belastbare Aussage),
  und — falls gewünscht — der nächste echte Schritt wäre tatsächlich ein
  trainiertes Prädiktor-Netz nach Flow-Opts Vorbild, nicht mehr nur dieser
  Löser.
