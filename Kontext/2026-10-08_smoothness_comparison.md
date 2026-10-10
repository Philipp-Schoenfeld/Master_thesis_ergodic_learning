# Kontext: Smoothness-Vergleich, Runde 1 + 2, TSVEC-SVGD-Reimplementierung (Session 2026-10-08 bis 2026-10-10)

Zusammenfassung der gesamten Konversation zu diesem Thema, als Gedächtnisstütze für
spätere Sessions. Chronologisch. Begleitdateien (Skripte aus dem temporären
Scratchpad, die sonst verloren wären, und die wichtigsten Abbildungen) liegen in
`Kontext/2026-10-08_smoothness_comparison_dateien/`.

**Stand am Ende der Session (2026-10-10):** Der neue TSVEC-Löser (`tsvec_svgd.py`) ist
repariert, getestet und auf den Cluster gepusht. Die Runde-2-Ergebnisse der vier
TSVEC-Varianten (`results/smoothness_comparison_v2_20261009`) sind **ungültig** und
müssen neu gerechnet werden. Ein Re-Run ist noch NICHT abgeschickt.

---

## 1. Ausgangsfrage (2026-10-08)

Philipp wollte einen großen Smoothness-Vergleich, direkt auf allen Holdout-Shapes ohne
Unknown-Regions. Pro Shape eine Verteilung von 30 Trajektorien, davon Mittelwert und
Kovarianz der Smoothness, und JEDER Zwischenschritt jeder SVGD-Iteration in einer DB
gespeichert. Fünf Varianten:

1. CFM-Inferenz allein (B-Spline)
2. CFM + reguläres SVGD bis Konvergenz (B-Spline)
3. nur SVGD, lineare Initialisierung, B-Spline
4. nur SVGD, lineare Initialisierung, ohne B-Spline (nur Waypoints)
5. wie 4, plus Smoothness-Kraft

Das ist fast wörtlich Philipps eigene Notiz aus `meeting_notes.odt`
("wie sehen Mean und Covariance aus bei einer Verteilung von 30 Trajektorien ... bei
den verschiedensten SVGD-Zwischenschritten", "Svdg direkt auf einer Trajektorie ohne
B-Spline mit Smoothing-Cost", "füge für die SVGD-Optimierung noch Smoothness hinzu").

## 2. Runde 1: Design-Entscheidungen

- **Shapes:** alle 25 `val`-Shapes aus `ergodic_dataset_775.db` -- diese DB enthält
  keine Unknown-Region-Shapes (die liegen nur in `ergodic_dataset_improved_final.db`,
  Suffix `#unk0/#unk1`), damit ist "ohne Unknown-Regions" automatisch erfüllt.
- **Löser für Varianten 2-5:** einheitlich der "sun"-Löser (Sun et al. FM-Stein,
  `exploration/common/sun_refine.py`, Default-Backend seit 2026-10-06, derselbe Kern wie
  die Datengenerierung). Nur so sind 2 vs. 3 (Prior hilft?) und 3 vs. 4 (B-Spline hilft?)
  kontrollierte Vergleiche.
- `n_iters = 600` (Ground-Truth-Konvention), NXI = 25 Kontrollpunkte, T = 128 Punkte.
- Lineare Init: `init_baselines.linear_angle_path`, 30 Winkel gleichverteilt über
  [0°, 180°). CFM: `transfer/netz2d_startpunkt.pt` via `apply_cfm_belief.CfmPlanner`,
  Partikelwolke `phi_particles(truth, 256, mode='uniform')`, 30 Kandidaten.
- **Smoothness-Metrik:** die Projekt-Konvention `metrics_explore_exploit.smoothness_energy`
  (Bogenlängen-Resampling auf 128 Punkte, dann `W_SMOOTH * sum(accel²)`), dazu Pfadlänge.

### Opt-in-Erweiterungen an `sun_refine.py` (Default = altes Verhalten, bit-identisch)

- `log_space='raw'`: Bisher liefen die geloggten ZWISCHENzustände immer durch einen
  B-Spline-Fit (`_fit_matrix(nxi)`), selbst bei `nxi == T`; "ohne B-Spline" galt also nur
  für die Endkurve. `'raw'` loggt jetzt linear auf `nxi` Punkte resamplete Rohpositionen
  (`_linear_resample_matrix`, exakt dieselbe Interpolation wie `_resample`).
- `smoothness_weight` (Default 0): zusätzliche Kraft `-w * grad(sum accel²)` auf den
  Positionsanteil des Stein-Gradienten `dx`, bevor er in den LQR-Solve geht.
  Runde 1 nutzte provisorisch `w = 200`.
- `svgd_batched.BatchedSunTorch.run` reicht beides durch und gibt zusätzlich `final_pos` zurück.

### Neue Dateien (Runde 1)

`evaluation_full_matrix/`: `smoothness_db.py` (SQLite, Zustände per `state_codec.py`
komprimiert, eine Zeile pro (shape, variant, cand_idx)), `run_smoothness_comparison.py`,
`plot_smoothness_comparison.py`, `test_smoothness_comparison.py`,
`run_job_smoothness_comparison_test.bash`, `run_job_smoothness_comparison.bash`,
`run_job_smoothness_comparison_replot.bash`.

## 3. Runde 1: Läufe und Ergebnisse

| Job | Zweck | Ergebnis |
|---|---|---|
| 164887 | Test, 2 Shapes | ok, 103 s Rechnen, ~125 s Plots |
| 164902 | voller Lauf, 25 Shapes | 46:46 (39:39 Rechnen, ~7 min Plots), 236 MB |
| 164948 | Replot mit Ergodizitäts-Panel | 13:28 |
| 164952 | Replot mit Strichmustern | 13:14 |

Ergebnisse: `results/smoothness_comparison_20261009/` (lokal und auf dem Cluster).

Mittelwerte über 25 Shapes bei Iteration 600:

| Variante | Smoothness | Pfadlänge | Ergodischer Fehler |
|---|---|---|---|
| cfm_only | 0.145 | 3.69 | 3.13 |
| cfm_svgd | 0.245 (schlechteste) | 4.23 | **1.35 (beste)** |
| linear_svgd_bspline | 0.152 | 3.40 | 6.56 |
| linear_svgd_raw | 0.162 | 3.48 | 6.54 |
| linear_svgd_raw_smooth | **0.116 (beste)** | 3.23 | 6.71 |

Interpretation (nach Philipps Rückfragen korrigiert):
- Der eigentliche Trade-off hängt an der **Initialisierung**, nicht an B-Spline vs. Waypoints:
  CFM+SVGD hat die mit Abstand beste Abdeckung, zahlt dafür mit Glattheit. Alle drei
  linearen Varianten liegen beim ergodischen Fehler fast gleich (6.5-6.7).
- Die Smoothness-Kraft ist bei linearer Init praktisch "gratis" (glatter, kaum
  Abdeckungsverlust). Meine erste Formulierung, sie koste Abdeckung, war überinterpretiert.
- `cfm_only` (null SVGD-Iterationen) schlägt alle linearen Varianten nach 600 Iterationen.
- **Warum ist cfm_svgd so unglatt, obwohl B-Spline?** Beim sun-Löser ist der B-Spline
  KEIN Optimierungsraum: optimiert wird die volle dynamik-simulierte 201-Punkte-Bahn,
  `nxi` ist nur die Projektion für Ausgabe/Log (milder Tiefpass, erklärt den kleinen
  Unterschied bspline 0.152 vs. raw 0.162). cfm_svgd ist unglatt, weil es am tiefsten in
  Richtung feiner Dichtestruktur konvergiert. Meine erste Erklärung ("25 Kontrollpunkte
  können keine hohen Frequenzen") stimmte nur für den alten tsvec-Löser.

### Plot-Änderungen (Runde 1)

- Drittes Panel **ergodischer Fehler**, post hoc aus den gespeicherten Zuständen berechnet
  (`ergodic_energy_torch.ergodic_term`, K = 10, W = 600), nicht in der DB gespeichert.
  Erster lokaler Versuch wurde OOM-gekillt (Rechner hat nur ~1.2 GB frei, `fourier_basis`
  baut ~1.8 GB Zwischentensor) -> Berechnung in 2500er-Chunks, wie
  `run_mission_eval.py::ergodic_E_batch`. **Philipps Regel daraus: große Ausführungen
  nur auf dem Cluster.**
- Unterschiedliche Strichmuster + Linienbreiten pro Variante, weil übereinanderliegende
  Linien sonst wie "fehlend" aussehen.

## 4. Runde 2: Neue Anfrage (2026-10-09)

Dieselben Plots, aber sieben Varianten, zwei Löserfamilien, **2000 Iterationen**, alle
Iterationen gespeichert, 30 Kandidaten:

- **sun_svgd** = bestehender Dynamik-Löser (`sun_refine.py`)
- **bspline_svgd / waypoint_svgd** = NEU: direkte Punktoptimierung ohne Dynamikmodell, nach
  arXiv:2603.09458 (Li, Jin, Teng, Gong, Chalvatzaki, "Stein Variational Ergodic Surface
  Coverage with SE(3) Constraints") -- das ist genau das TSVEC-Paper aus CLAUDE.md. Das
  Paper selbst nutzt keine B-Splines, nur SE(3)-Via-Points.

| Variante | Löser | Init | Parametrisierung | Smoothness |
|---|---|---|---|---|
| cfm_only | -- | CFM | B-Spline | -- |
| cfm_bspline_svgd | TSVEC | CFM | B-Spline | ja |
| linear_bspline_svgd | TSVEC | linear | B-Spline | ja |
| linear_waypoint_svgd | TSVEC | linear | Waypoints | nein |
| linear_waypoint_svgd_smooth | TSVEC | linear | Waypoints | ja |
| linear_sun_pointmass | sun | linear | Waypoints | -- (Punktmasse) |
| linear_sun_jerk | sun | linear | Waypoints | -- (glattere Dynamik) |

**Philipps Entscheidungen** (per Rückfrage):
- Die 30 Kandidaten sind **ein gemeinsamer, interagierender SVGD-Schwarm** (wie im Paper),
  nicht 30 unabhängige Läufe.
- "Glattere Dynamik" = **Doppelintegrator mit Ruck-Strafe**.

### Umsetzung

- `JerkPenalizedLQR` in `sun_refine.py`: Zustand `[px,py,vx,vy,ax,ay]`, Steuerung = Ruck,
  `Q = diag(1,1,0.001,0.001,0.01,0.01)`, `R = diag(0.05,0.05)` (provisorisch, nicht
  gesweept). Als paralleler Satz jit-Funktionen (`_iteration2`, `_run_batch_logged2`, ...),
  damit der bestehende Punktmasse-Pfad unangetastet bleibt; neuer Parameter
  `dynamics='pointmass'|'jerk'` auf `run_batch`, `prepare`, `BatchedSunTorch.run`.
  Test: Beschleunigungskosten 0.131 -> 0.002.
- `exploration/common/tsvec_svgd.py` (neu): 2D-Reduktion des Papers, gegen das
  heruntergeladene PDF geschrieben (die HTML-Fassung verstümmelt Gleichungen). Energie
  Eq. 20a (Smoothness) + 20d (Ergodik); V_a/V_f (SE(3)-Oberfläche) entfallen; eigener
  Rand-Term ergänzt (Paper hat keinen). Kernel Eq. 22 (pro Zeitschritt, Bandbreite 0.05),
  Gauss-Newton-Präkonditionierer Eq. 21.
- `evaluation_full_matrix/run_smoothness_comparison_v2.py`, `test_smoothness_comparison_v2.py`,
  `run_job_smoothness_comparison_v2_test.bash`, `run_job_smoothness_comparison_v2.bash`,
  `run_job_smoothness_comparison_v2_replot.bash`; `exploration/common/test_tsvec_svgd.py`.
- `plot_smoothness_comparison.py`: Farben/Labels/Strichmuster für die neuen Varianten,
  Verteilungsplots jetzt für jede Variante außer `cfm_only` (statt fester Runde-1-Liste),
  Smoothness-Achse **symlog** (`linthresh=0.3`), damit die Varianten nahe 0 neben der bis
  ~21 wachsenden sichtbar bleiben.

### Bugs auf dem Weg (alle behoben, mit Regressionstest)

1. **Autodiff-Jacobian zu langsam:** `vmap(jacrev(...))` = 8.7 s/Iteration (CPU) ->
   ~4.9 h pro Zelle. Ersetzt durch analytischen Jacobian (Smoothness linear, Ergodik
   geschlossene Sinus-Ableitung, Rand Indikator), ~110x schneller.
2. **Falsche Ableitung der Fourier-Basis:** `coeffs_from_points` ist ein separables
   Produkt `cos(πk₁x)·cos(πk₂y)`, nicht `cos(π k·x)`. Per Finite-Differenzen-Test gefunden.
3. **CUDA-Tensor durch `np.asarray`** in `TsvecSvgd.run()` (nur auf der Cluster-GPU
   sichtbar, Job 164971) -> `_to_tensor`.

### Läufe Runde 2

| Job | Zweck | Ergebnis |
|---|---|---|
| 164971 | Test | FAILED (CUDA-Tensor-Bug) |
| 164977 | Test | ok, aber w_smooth=5-"Freeze" entdeckt |
| 164983 | Test mit Workaround w_smooth=1e-4 | ok |
| 164986 | voller Lauf, 25 Shapes x 7 Varianten | 2:51:53, 175 Zellen, 820 MB DB |
| 165104 | Replot symlog | 39:01 |

Ergebnisse: `results/smoothness_comparison_v2_20261009/` -- **TSVEC-Varianten ungültig**
(siehe Abschnitt 5); `cfm_only` und die beiden sun-Varianten sind davon nicht betroffen.

Beobachtungen am (ungültigen) Lauf: sun-Varianten konvergieren über 2000 Iterationen
sauber (ergodischer Fehler ~3.6 Punktmasse, ~6.3 Ruck-Dynamik); die Ruck-Dynamik ist
deutlich glatter. GPU: die torch-basierte TSVEC-Rechnung nutzt die Cluster-GPU
(~25-45 s pro Zelle bei 2000 Iterationen), die JAX-basierte sun-Rechnung fällt weiter auf
CPU zurück (~72-78 s pro Zelle).

## 5. Der eigentliche Fehler im TSVEC-Löser (2026-10-10)

Philipp: "fast keine der Trajektorien sieht ergodisch aus, prüfe die bspline_svgd-
Implementierung". Richtig: der ergodische Fehler bewegte sich über 2000 Iterationen nur
von 0.0379 auf 0.0357 (ungewichtet), die Pfade wurden lang und zittrig.

Ursachen (Diagnose auf Shape A, 30 lineare Sehnen, Waypoints):
1. **Energieskala (Hauptursache).** p ∝ exp(−V); mit den Paper-Gewichten (ergodisch 3.0)
   ist V auf unserer Fourier-Metrik nur ~0.1, die Zielverteilung also fast flach. Die
   Kernel-Abstoßung überwiegt den Score, SVGD diffundiert statt abzudecken. Die absoluten
   Paper-Gewichte gehören zu dessen Punktwolken/LBO-Energie und sind nicht übertragbar.
2. **Fehlendes 1/N** im Eq.-21-System (Detommaso et al. SVN, das das Paper zitiert,
   mittelt den Präkonditionierer): jeder Schritt war N = 30-fach zu klein.
3. **Kernel summiert statt gemittelt** über T: k bis 128, k² bis 16384; Eq. 21 gewichtet
   Hessians mit k², den Score mit k -> Schritt zusätzlich um 1/T geschrumpft.
4. **Absolute Dämpfung** (1e-3): H = JᵀJ hat bei Waypoints ohne Smoothness Rang ≤ 100
   (Fourier-Moden) in 256 Dimensionen; im Rest regelte nur die Dämpfung, die Abstoßung
   machte dort riesige, zittrige Schritte (Pfadlänge 1 -> 40 ohne Abdeckungsgewinn).

Der w_smooth=5-"Freeze" aus Runde 2 war ein Symptom derselben Skalen-Schieflage, kein
eigenständiges Smoothness-Problem.

**Fix in `tsvec_svgd.py`:**
- Gewichte = Projekt-Konvention aus `tsvec_2d.py` / `ergodic_energy_torch.py`:
  **ergodisch 600, Smoothness 15, Rand 30** (= Paper-Ergodikgewicht x200). Paper-Werte
  bleiben nur als Referenzkonstanten.
- Kernel über t gemittelt (`_kernel_and_grad`, k ∈ [0,1]).
- Eq.-21-System durch N geteilt.
- Levenberg-Marquardt-Dämpfung relativ zur mittleren Diagonale (`LM_DAMPING = 0.1`).
- `TAU = 0.1` (Paper-Wert, jetzt ein echter 10%-gedämpfter Newton-Schritt).
- In `run_smoothness_comparison_v2.py`: Workaround `W_SMOOTH_ACTIVE = 1e-4` entfernt;
  `w_smooth = 15` für die Varianten mit Smoothness, 0 für `linear_waypoint_svgd`.
- Tests: Jacobian-Check jetzt relativ, Kernel-Gradient direkt gegen `_kernel_and_grad`,
  neuer Regressionstest "ergodischer Fehler fällt um > 50 % in 100 Iterationen"
  (32.4 -> 0.96). Alles grün.

Sweep vor dem Fix (Shape A, 400 Iterationen, Plot-Skala wie in den Grafiken; Skript
`tsvec_fix_sweep.py` in den Begleitdateien): nur Update-Regel-Fixes bei Paper-Gewichten
23 -> 35 (schlechter); Energieskala x100 -> 0.35; Projektgewichte:

| Konfiguration | Ergodischer Fehler | Pfadlänge | Smoothness |
|---|---|---|---|
| Waypoints, w_s = 15 | 0.24 | 5.6 | 0.42 |
| B-Spline, w_s = 15 | 0.44 | 5.6 | 0.29 |
| B-Spline, w_s = 0 | 0.72 | 6.4 | 1.19 |
| Waypoints, w_s = 0 | 0.33 | 44 | 865 (extrem zittrig) |

Zum Vergleich: kaputte Version ~23, sun-Löser ~3.6, CFM allein ~3.1. Konvergenz in
~100 Iterationen.

### Qualitative Prüfung (lokal, 3 Shapes, 5 Varianten, 400 Iterationen)

Lauf `results/_qual_fix/`, Abbildungen in den Begleitdateien
(`qualitative_final.png`, `qualitative_steps_A_linear_waypoint_svgd_smooth.png`,
erzeugt mit `qualitative_fig.py`).

| Variante | Ergodischer Fehler (A / rand_gmm_10 / organic_0) | Bild |
|---|---|---|
| CFM allein | 1.9 / 2.7 / 1.5 | folgt der Kontur |
| TSVEC B-Spline (CFM- oder lineare Init) | 0.4-1.2 | Schleifen über die ganze Dichte |
| TSVEC Waypoints + Smoothness | 0.24-0.28 | glatte Schleifen, sieht am besten aus |
| TSVEC Waypoints ohne Smoothness | 0.1-0.3 | Gekritzel über das ganze Quadrat |

Schrittfolge (A, Waypoints + Smoothness): Iteration 0 -> 5 -> 25 -> 100 -> 400 mit
ergodischem Fehler 22.7 -> 18.9 -> 11.7 -> 1.1 -> 0.24.

Der gefixte Code ist auf den Cluster gepusht. **Kein Re-Run abgeschickt** -- meine Frage
(voller Re-Run in neues out_tag ~3 h, oder nur die vier TSVEC-Varianten neu) hat
Philipp abgebrochen.

## 6. Offene Punkte

- **Runde 2 neu rechnen** mit dem gefixten Löser (`run_job_smoothness_comparison_v2.bash`,
  sinnvollerweise neues `OUT_TAG`). Entscheidung voll vs. nur TSVEC-Varianten offen.
- **TSVEC-Bahnen kreuzen leere Bereiche** (z. B. außerhalb des Buchstabens A): die Metrik
  nutzt nur 100 tieffrequente Fourier-Moden (K = 10) und bestraft das Verlassen der
  scharfen Form nicht. sun steuert über den Dichtegradienten und bleibt auf der Form.
  Option: mehr Moden für den TSVEC-Löser. Nicht geändert.
- **Waypoints ohne Smoothness** ist ein pathologisches Gekritzel -- ehrliches
  Ablationsergebnis, kein Bug.
- Shapes **A und greek_upper_0** haben in `ergodic_dataset_775.db` bit-identische
  Dichtegitter (24 von 25 verschieden). Nicht angefasst.
- **JAX im `thesis`-Env auf dem Cluster hat kein CUDA** (fällt auf CPU zurück), torch schon.
- **Verteilungsplots** (`*_dist.png`): die großen farbigen Flächen sind die
  Kovarianz-Ellipsen (2σ, x2 skaliert, alpha 0.25) an ~10 Punkten; bei weit gestreuten
  Partikeln verschmelzen sie zu großen Flächen. Philipp fand die Bilder "komisch";
  erklärt, nicht geändert.
- Provisorische, nicht gesweepte Parameter: `smoothness_weight = 200` (Runde 1, sun),
  `JerkPenalizedLQR`-Gewichte, `LM_DAMPING = 0.1`.

## 7. Lokale Pfade

- Repo: `/media/philipp/storage/Dokumente/Uni/Master_thesis` (= Symlink
  `/home/philipp/Documents/Uni/Master_thesis`)
- Runde 1: `thesis_architecture/evaluation_full_matrix/results/smoothness_comparison_20261009/`
- Runde-1-Test: `.../results/smoothness_comparison_test_20261008/`
- Runde 2 (TSVEC ungültig): `.../results/smoothness_comparison_v2_20261009/`
- Runde-2-Test: auf dem Cluster `.../results/smoothness_comparison_v2_test_20261009b/`
- Qualitative Prüfung nach dem Fix: `.../results/_qual_fix/`
- Paper-PDF lag nur temporär in `/tmp`; Quelle: arXiv:2603.09458v3.
