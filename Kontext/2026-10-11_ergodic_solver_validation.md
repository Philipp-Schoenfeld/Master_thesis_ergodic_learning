# Kontext: Validierung und Neuimplementierung der drei Ergodik-Löser (Session 2026-10-11/12)

Fortsetzung von `2026-10-08_smoothness_comparison.md`. Auslöser: Philipps visuelle Prüfung von
`results/smoothness_comparison_v4test_20261011/plots/per_shape/*_dist.png` ("some of them are in no
way ergodic", z. B. `A_linear_adam_bspline_dist.png`, `A_cfm_bspline_svgd_dist.png`,
`A_linear_sun_jerk_dist.png`). Auftrag: alle drei Löser (Sun, Adam-SVGD, TSVEC) gegen bewährte
Referenzimplementierungen prüfen/neu implementieren, eine intensive Ergodizitäts-Validierung bauen,
die Visualisierung überarbeiten, die Smoothness-Pipeline für eine Beispielform neu rechnen, danach
PC herunterfahren.

(Endergebnisse: siehe Abschnitt 8 "Ergebnis" am Ende.)

## 1. Befund zur Darstellung

Die `*_dist.png` zeigten die *punktweise mittlere Trajektorie* von 30 Kandidaten plus
Kovarianz-Ellipsen. Der Mittelwert 30 verschiedener Bahnen (lineare Sehnen in 30 Winkeln!) ist
keine Bahn und kollabiert zur Mitte; die Ellipsen übermalen die Dichte. Das war ein großer Teil des
"nicht ergodisch"-Eindrucks. Ersetzt (gleicher Dateiname, `plot_smoothness_comparison.
plot_trajectory_distribution`): oben alle 30 Bahnen + der *Median-Kandidat* (nach Hellinger, kein
Cherry-Picking) fett, unten dessen zeitgemittelte Aufenthaltsdichte mit der Zielkontur.

## 2. Unabhängige Ergodizitäts-Validierung (`thesis_architecture/ergodic_validation/`)

Ergodizität = Zeitmittel = Raummittel. Die Löser optimieren eine abgeschnittene Fourier-Metrik
(K = 8..10 Moden) -- die kann ~0 sein, ohne dass die Bahn sichtbar ergodisch ist. Deshalb Maße, die
NICHT von dieser Metrik abhängen (`ergodicity_metrics.py`):
- `hellinger`: Hellinger-Abstand zwischen geglätteter Aufenthaltsdichte (Histogramm der
  zeitgleichverteilten Bahnpunkte, Gauss 0.03) und gleich geglätteter Zieldichte.
- `tv_c`: Zeitanteil vs. Massenanteil auf c x c Zellen (c = 2, 4, 8).
- `low_excess`: Zusatzzeit, wo das Ziel ~0 ist (< 2 % des Maximums).
- `coverage` (nur Info), `spec_rel_20` (400 Moden), `ergodic_k10` (Projektmaß).
- Untergrenze: i.i.d.-Stichproben aus dem Ziel mit gleich vielen Punkten (T = 128): Hellinger
  ~0.14-0.33 je nach Ziel.

Kanonische Ziele (`targets.py`): `sun_gmm3` (offizielles 3-GMM aus Suns Tutorial), `blobs_75_25`
(Zeitanteil links muss 0.75 sein), `uniform`, `ring` (Loch darf nicht besucht werden), Holdout-Formen
`A`, `organic_30`, `korean_5`.

Prüfstand `validate_solvers.py` ruft exakt `run_smoothness_comparison_v2.run_cell` auf (also den
Pipeline-Code), 30 Kandidaten, 2000 Iterationen, Checkpoints 0/25/100/300/1000/2000. Kriterien auf
dem Median: C1 Hellinger <= Untergrenze + 0.2, C2 tv_4 <= 0.25, C3 low_excess <= 0.1, C4 nicht
schlechter als Init, C5 keine späte Verschlechterung (> 0.05 über dem besten Checkpoint), C6
Regionen (Blob 75 %, Loch) +-0.1. (Ein erstes C7 Coverage >= 0.75 wurde zur Info herabgestuft:
eine Bahn der Länge L deckt höchstens ~2*0.05*L ab -- auf `uniform` scheiterten daran alle Familien.)
Ausgaben: `report.md`, `report.json`, `summary.png` (Variante x Ziel), `fig_<ziel>.png`.

Tests: `test_ergodicity_metrics.py` (exakte Fälle, i.i.d. schlägt Sehnen, Konsistenz), alle
Einzeltests gesammelt in `run_all_tests.py`.

## 3. Bestandsaufnahme (vorheriger Code, `results/baseline_v4`)

64/77 Zellen bestanden. Auf allen glatten GMM-Zielen und auf A/organic_30 waren TSVEC (mit
beta = 100) und Adam ergodisch -- auf A z. B. Hellinger 0.13-0.27 bei Untergrenze 0.20. Echte
Fehlschläge: Sun-Jerk (korean_5 0.57, uniform 0.53) und Sun auf zwei Blobs (61 % statt 75 % Zeit
links, beide Sun-Varianten).

## 4. Sun et al. (Flow Matching Ergodic Coverage, RSS 2025)

- Offizielles Tutorial MurpheyLab/lqr-flow-matching (`stein_flow_coverage.ipynb`) transkribiert
  (`reference_sun_check.py`): unsere Portierung (`sun_refine.py`) reproduziert es auf dem
  offiziellen Benchmark (mittlerer Punktabstand 0.015, Hellinger 0.273 vs. 0.270). Kern korrekt.
- Grundsätzliche Schwäche des Stein-Flusses: innerhalb eines Modus hängt grad log p nicht vom
  Modusgewicht ab -> Masse zwischen getrennten Modi wird nicht umverteilt (Blobs 61/39).
- Neu: `exploration/common/sun_sinkhorn.py`, die zweite offizielle Variante (Sinkhorn-Divergenz-
  Fluss, `sinkhorn_flow_coverage.ipynb`: L1-Kosten, eps 0.01, Skala 1e3, Schritt 0.005). Eigene
  Log-Domain-Sinkhorn-Implementierung (ott-jax nicht installiert; Installation hätte die JAX-
  Version verschoben), Gradient per Envelope-Theorem, Warmstart der Potentiale, GPU-Zwilling in
  torch (JAX hat hier kein CUDA). LQR-Update in einer fusionierten jit-Funktion (separat übergebene
  Jacobis machten den Riccati-Solve 27x langsamer). Tests `test_sun_sinkhorn.py`: Gradient ==
  Autodiff (1e-15), Warmstart cos 1.0, Partikelfluss 75 % im 75 %-Blob, Bahn 0.745 (Stein 0.604),
  torch == JAX (bis Iteration 10 auf 1e-6; danach Rundungs-Chaos durch Vorzeichensprung der
  L1-Kosten, betrifft auch torch-CPU vs. torch-GPU -- Ergodizitätsmaße gleich).
- Weicher Randterm wie im Stein-Port ergänzt (W_BOUNDARY_FLOW, wirkt nur außerhalb 3 % Rand).
- Pipeline: `SUN_FLOW` (Default 'sinkhorn', Env `THESIS_SUN_FLOW`), Labels jetzt "Sun flow matching".
- Punktmasse + Sinkhorn: auf allen 7 Zielen bestanden und besser als Stein (korean_5 0.36 -> 0.15,
  A 0.26 -> 0.12), dafür längere, weniger glatte Bahnen (accel ~3-5x).
- Ruck-Dynamik (eigene Erweiterung, `JerkPenalizedLQR`): mit Sinkhorn besser auf 5 Zielen, aber
  ring/uniform scheitern (Bahnende verlässt das Gebiet). Kleinere Schritte (0.0025, 0.00125) und
  der Randterm helfen nicht; kleineres R (0.01) wird instabil. Ursache: der Sinkhorn-Gradient ist
  pro Zeitpunkt unabhängig (zeitlich hochfrequent), ein Dreifach-Integrator kann ihn nicht
  nachführen. (Entscheidung zur Ruck-Variante: siehe Ergebnis.)

## 5. Adam-SVGD

Neu: `exploration/common/adam_svgd.py` nach Liu & Wang (SVGD-Referenzcode DartML, Median-Heuristik
h = med^2 / log(N+1)) und Lee et al. "Stein Variational Ergodic Search" (RSS 2024: gemeinsamer
Schwarm über ganze Trajektorien, Temperatur). Energie/Jacobi aus `tsvec_svgd` (FD-geprüft),
K = 10, beta = 100, Adam lr 2e-3. Ersetzt in der Pipeline den GUI-Refiner (`ADAM_IMPL = 'gui'`
stellt ihn wieder her; GUI selbst unverändert). Tests `test_adam_svgd.py`: SVGD trifft Mittelwert/
Kovarianz einer Gauss-Verteilung, Abstoßung und Energiegradient == Autograd, kein Kollaps.
Gegen den GUI-Refiner: auf 26/28 Zellen ergodischer (organic_30 0.24 -> 0.15), B-Splines glatter;
Waypoints ohne Smoothness zackiger (Pfad 5 -> 10, inhärent: Ergodik kennt keine Reihenfolge).

## 6. TSVEC

Kein öffentlicher Code (arXiv 2603.09458). Implementierung unverändert bis auf beta = 100 (vorige
Session) und einen Robustheitsfix (nicht-zusammenhängende Basismatrix brach `torch.kron`).
Fourier-Moden-Sweep (`sweep_modes.py`): K = 16/24 bringt auf A kaum etwas -> K = 10 bleibt.

## 7. Dateien

Neu: `ergodic_validation/` (metrics, targets, validate_solvers, reference_sun_check, sweep_modes,
sweep_jerk, sweep_jerk_step, run_all_tests, test_ergodicity_metrics), `exploration/common/
adam_svgd.py`, `sun_sinkhorn.py`, `test_adam_svgd.py`, `test_sun_sinkhorn.py`,
`evaluation_full_matrix/plot_smoothness_overview.py`, `compute_convergence.py`,
`merge_smoothness_dbs.py`, `seed_smoothness_db.py`, `qualitative_smoothness_fig.py`.
Geändert: `run_smoothness_comparison_v2.py` (ADAM_IMPL/K/BETA/LR, SUN_FLOW, `_init_params`,
`_rows_from_log`), `plot_smoothness_comparison.py` (neue *_dist.png, Labels), `tsvec_svgd.py`
(beta, contiguous), `svgd_batched.py` (smoothness_weight, jitter_scale).
Ergebnisse: `ergodic_validation/results/` (baseline_v4, adam_ref, sun_sinkhorn, final_*).

## 8. Ergebnis (2026-10-12)

**Entscheidung Ruck-Variante:** bleibt beim Stein-Fluss (`SUN_SPEC['linear_sun_jerk']['flow'] =
'stein'`). Mit Sinkhorn verlassen die Bahnenden auf ring/uniform das Gebiet (pathologisch); mit
Stein bleiben sie drin und konvergieren nur langsam. Ursache des Sinkhorn-Problems bei Ruck-Dynamik
nicht geklärt (Schrittweite, Randterm, zeitliche Glättung, kleineres R ausgeschlossen).

**Abschluss-Prüfstand** (`ergodic_validation/results/final_v5`, finaler Code, 7 Ziele x 11 Varianten,
30 Kandidaten, 2000 Iterationen): 73/77 Zellen bestanden. Sun-Punktmasse (Sinkhorn), alle vier
Adam-Varianten und alle vier TSVEC-Varianten bestehen auf allen 7 Zielen ALLE Kriterien.
Fehlschläge: `cfm_only` auf blobs (reine Netzausgabe, Massenanteil 0.63 statt 0.75) und
`linear_sun_jerk` auf blobs (Stein-Massenfehler), korean_5 (Hellinger 0.57) und uniform (0.53).

Hellinger (Median, final) -- Untergrenze i.i.d. 0.14-0.33:
| Variante | A | blobs | korean_5 | organic_30 | ring | sun_gmm3 | uniform |
|---|---|---|---|---|---|---|---|
| sun pointmass (Sinkhorn) | 0.12 | 0.14 | 0.15 | 0.11 | 0.13 | 0.16 | 0.19 |
| sun jerk (Stein) | 0.36 | 0.28! | 0.57! | 0.23 | 0.29 | 0.33 | 0.53! |
| adam cfm bspline | 0.19 | 0.18 | 0.25 | 0.12 | 0.19 | 0.25 | 0.35 |
| tsvec cfm bspline | 0.20 | 0.21 | 0.27 | 0.14 | 0.22 | 0.25 | 0.37 |
| tsvec linear waypoint (+smooth) | 0.13 (0.17) | 0.15 (0.19) | 0.16 (0.25) | 0.12 (0.12) | 0.14 (0.20) | 0.16 (0.25) | 0.19 (0.28) |

**Tests:** `ergodic_validation/run_all_tests.py` -> 6/6 Testdateien bestanden (Metriken,
Sun-Referenzabgleich, TSVEC, Adam, Sinkhorn, Pipeline-Selbsttest).

**Smoothness-Pipeline, Beispielform A** (finaler Code): `evaluation_full_matrix/results/
smoothness_comparison_v5_A_20261012/` (DB mit allen Zwischenständen, summary.csv, plots/ inkl. neuer
*_dist.png, convergence.npz, overview.png, overview_convergence.png, qualitative_final.png).
Laufzeit 6.9 min (TSVEC/Adam ~17-22 s je Zelle auf der RTX 2070 SUPER, Sun-Sinkhorn 132 s, Sun-Stein
128 s). Mittlerer ergodischer Fehler (K=10) auf A: TSVEC 0.02-0.06, Adam 0.00-0.04, Sun-Punktmasse
0.04, Sun-Ruck 6.69, CFM allein 1.95.

**Offen:**
- Sun-Ruck: Sinkhorn-Instabilität verstehen oder Ruck-Variante anders formulieren.
- Volllauf über alle 25 Formen mit dem finalen Code (lokal ~2 h; die v3-DB ist veraltet).
- Cluster-Sync weiterhin offen (Host-Key-Verifikation; nicht eigenmächtig akzeptiert).
- Sun-Varianten starten fest am Rand (Sehnenende): Teil der Bahn ist Anlauf.
