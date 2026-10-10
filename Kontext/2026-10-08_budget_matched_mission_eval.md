# Kontext: Budget-Matched Replanning-Mission-Evaluation (Session 2026-10-08 bis 2026-10-10)

Zusammenfassung der gesamten Konversation zu diesem Thema, als Gedächtnisstütze für
spätere Sessions. Chronologisch.

## 1. Ausgangsfrage

Ausgangspunkt war die bestehende Replanning-Missions-Auswertung
`results/mission_eval_20261006/` (erzeugt von `run_mission_eval.py`): cfm, random_walk
und linear bekommen dort pro Replanning-Runde alle gleich viele SVGD-Iterationen
(fix 1500). Philipps Idee: stattdessen die CFM-Warmstart-Variante pro Runde so lange
SVGD rechnen lassen, bis sie "gut genug" ist (ergodische Konvergenz/Plateau), diese
Iterationszahl messen, und random_walk/linear in derselben Runde auf exakt diese
Zahl deckeln. Ziel: dieselben fünf Plots wie im Original neu erzeugen, aber unter
dieser Budget-Matching-Logik:
- `executions_to_95_box.png` (+ Nachbarschwellen)
- `overview_{E,J}_truth_continuous_8units_svgd_conv_{plateau,target}.png`

Zentrale Vorfrage: kann man das aus der bestehenden DB post-hoc herleiten, oder muss
neu gerechnet werden?　**Antwort:** nur Runde 0 ist "gratis" rekonstruierbar (gleiche
Seeds, gleiche Startbedingung, unabhängig von späterer Ausführung); ab Runde 1 hängt
Belief/Position vom tatsächlich gefahrenen (jetzt kürzer optimierten) Pfad ab, d. h.
jede Folgerunde braucht echtes Neu-Rechnen (CFM-Inferenz + SVGD) für alle drei Methoden.

## 2. Technische Umsetzung

Neues Skript: `thesis_architecture/evaluation_full_matrix/run_mission_eval_budget_matched.py`
- `BudgetGroup` pro (knowledge_condition, strategy): koppelt cfm/random_walk/linear
  rundenweise. Pro Runde: erst cfm's eigene 30 Kandidaten (`cfm_run_iters`, Default 1500
  bzw. 600 je nach Job), Plateau-Kriterium (gleiche Regel wie
  `plot_mission_continuous.py --svgd_conv`: erste Iteration, ab der sich der Fehler über
  die nächsten `conv_window`=100 Iterationen um weniger als `conv_tol`=5 % relativ
  verbessert) pro Shape ermittelt (Median über die 30 cfm-Kandidaten, auf das
  `metric_stride`-Raster von 10 gerundet) -> das ist das Budget dieser Runde für dieses
  Shape. Danach random_walk/linear mit genau diesem Budget gedeckelt (gemeinsamer
  SVGD-Batch-Call bis `max(use_iters)`, danach pro Shape auf den eigenen Wert
  zurückgeschnitten). Fällt ein Shape für cfm schon raus (99 % erreicht), wird der
  zuletzt bekannte Budget-Wert fortgeschrieben (`last_budget`-Dict, auch beim Resume
  aus der DB rekonstruiert).
- Nutzt dieselbe `mission_db.py`-Schema wie das Original -> bestehende Plot-Skripte
  brauchen keine Änderung.
- Läuft NICHT mit dem `--parallel_sets`-Overlap-Scheduler des Originals (Gruppen laufen
  strikt sequentiell ab) -- bewusste Vereinfachung, siehe Abschnitt 4 zur Laufzeit.

Zwei kleine, rückwärtskompatible Änderungen an `run_mission_eval.py` (Default-Verhalten
unverändert, per `test_mission_eval.py` nach der Änderung erneut grün):
- `MissionSet.run_svgd(..., n_iters=None)`: optionaler Override statt immer `a.n_iters`.
- `MissionSet.execute(..., n_iters_used=None)`: das tatsächlich verwendete Iterationsbudget
  wird in die `candidates`-Tabelle geschrieben statt pauschal `ctx.args.n_iters`.
- `_pool_task`'s `pack`-Branch: optionales `wide_range`-Flag fürs State-Packing (s. u.).

**Gefundener und gefixter Bug (echtes Edge Case, kein Implementierungsfehler):**
`state_codec.py` quantisiert Kontrollpunkte verlustbehaftet auf 16 Bit über
`[Q_LO, Q_HI] = [-0.5, 1.5]` -- kalibriert für KONVERGIERTE (iteration ~1500) Zustände.
Unteroptimierte, früh gestoppte Kandidaten (besonders random_walk bei sehr kleinem
Budget) können Kontrollpunkte weit außerhalb dieses Bereichs haben (beobachtet:
Sättigung bei genau -0.5, echter Wert vermutlich deutlich darunter) -> beim Pur-Dekodieren
Rekonstruktionsfehler bis 7e-2 statt der erwarteten ~1e-4 (Toleranzcheck in
`plot_mission_continuous.py` schlug fehl). Fix: `state_codec.py` kodiert jetzt optional
einen zweiten, weiteren Quantisierungsbereich (`Q_LO_WIDE, Q_HI_WIDE = -3.0, 4.0`),
rückwärtskompatibel über das ohnehin vorhandene Header-Byte (Bit 2 = "wide range");
alte Blobs dekodieren weiterhin exakt wie vorher. `run_mission_eval_budget_matched.py`
packt grundsätzlich mit `wide_range=True`. Nach dem Fix: Konsistenzcheck bestanden
(max. Abweichung 1,9e-4 im echten Cluster-Lauf).

Neue Job-Skripte (sbatch, nicht per `srun` direkt gestartet, wie vom Projekt gefordert):
- `run_job_mission_eval_budget_matched_smoke.bash` (kleiner Cluster-Smoketest, echter
  CFM-Checkpoint + GPU, 4 Shapes, 1 Bedingung, 1 Strategie, max_rounds=6, 2 h Limit)
- `run_job_mission_eval_budget_matched.bash` (voller Lauf: 25 Shapes x 3 Bedingungen x
  3 Strategien, `cfm_run_iters=600`, `--time_budget_h 22`)

Wichtiger Stolperstein, der im Smoketest gefunden wurde: `plot_mission_eval.py` ohne
`--box_only` versucht zusätzlich `svgd_convergence_round*.png` zu schreiben, was gleich
lange `E_series` für alle Kandidaten voraussetzt -- bricht bei variablen
Pro-Shape-Budgets mit `ValueError: all input arrays must have the same shape`. Fix:
`--box_only` in beiden Job-Skripten ergänzt (die gewünschten Plots waren zu dem
Zeitpunkt im Smoketest bereits geschrieben, der Crash kam erst danach bei einem nicht
angefragten Plot).

## 3. Validierung

1. Lokaler Dry-Run-Smoketest (CPU, `DummyPlanner`, kein Checkpoint nötig): 2 Shapes,
   3 Runden -- bestätigt, dass das Budget pro Shape/Runde exakt gleich für alle drei
   Methoden ist, Konsistenzcheck bestanden, alle 5 angefragten Plot-Typen erzeugt.
2. `test_mission_eval.py` (komplette bestehende Selbsttest-Suite) läuft nach den beiden
   Edits weiterhin vollständig grün (inkl. State-Codec-Roundtrip-Test).
3. Cluster-Smoketest (Job 164250, echter CFM-Checkpoint + GPU, 4 Shapes, `none_known`/
   `eid`, max_rounds=6, `cfm_run_iters=600`): erfolgreich, Konsistenzcheck bestanden
   (1,7e-4), alle Plots geschrieben (nach dem `--box_only`-Fix). Lief in 4,1 min für
   6 Runden x 4 Shapes x 1 Gruppe (JAX/Sun-Refiner lief auf CPU, nicht GPU -- bestätigt
   als dieselbe Umgebungs-Eigenart wie schon im Original-Lauf, kein neues Problem).

## 4. Voller Lauf (Cluster-Job 164886)

Submitted und durchgelaufen ohne Rückfrage, nachdem Philipp "start that yourself" sagte
(vorher wurde die Smoketest- und Vollauf-Freigabe jeweils einzeln eingeholt, wie von den
Cluster-Sicherheitsregeln verlangt).

- Laufzeit: 08:43:15 -- 22:36:18 Uhr am 2026-10-09, also **~13 h 53 min**
  (Schätzung vorher: 9--18 h, lag also im erwarteten Rahmen, eher mittig/oberes Drittel).
- Exit-Code 0, alle 9 (Bedingung, Strategie)-Gruppen fertig:
  - lse/half_known: 106,0 min, 49/75 Missionen erreichten 99 %
  - ucb/half_known: 118,6 min, 55/75
  - eid/half_known: 65,8 min, 72/75
  - lse/ten_samples: 99,1 min, 60/75
  - ucb/ten_samples: 108,0 min, 66/75
  - eid/ten_samples: 68,1 min, 75/75
  - lse/none_known: 100,5 min, 59/75
  - ucb/none_known: 102,8 min, 70/75
  - eid/none_known: 63,5 min, 75/75
- Konsistenzcheck in `plot_mission_continuous.py`: max. Abweichung 1,9e-4 (Segment),
  7,8e-4 (swept_mass) -- unauffällig.
- Alle 5 angefragten Plots erfolgreich erzeugt.

**Ergebnisse liegen lokal unter:**
`/media/philipp/storage/Dokumente/Uni/Master_thesis/thesis_architecture/evaluation_full_matrix/results/mission_eval_budget_matched_20261008_sun/`
(per rsync vom Cluster gezogen, ~2,3 GB: `plots/`, `shards/*.db` (27 SQLite-Shards),
`config.json`, `analysis/continuous_8units_svgd_conv.npz`). Dieselbe Struktur liegt auch
noch auf dem Cluster unter
`~/Master_thesis/thesis_architecture/evaluation_full_matrix/results/mission_eval_budget_matched_20261008_sun/`.

Die 5 angefragten Plots konkret:
- `plots/executions_to_threshold/executions_to_{75,80,90,95,99}_box.png`
- `plots/overview_E_truth_continuous_8units_svgd_conv_plateau.png`
- `plots/overview_E_truth_continuous_8units_svgd_conv_target.png`
- `plots/overview_J_truth_continuous_8units_svgd_conv_plateau.png`
- `plots/overview_J_truth_continuous_8units_svgd_conv_target.png`

## 5. Zentraler inhaltlicher Befund (wichtig für die Thesis-Diskussion!)

Philipps Beobachtung beim Betrachten der Ergebnisse: entgegen der Erwartung ("CFM
konvergiert am schnellsten, sollte also klar am besten sein") gleichen sich die drei
Methoden unter Budget-Matching stark an -- vor allem linear kommt nah an cfm heran.

**Mechanismus (gemeinsam erarbeitet):**

1. Das Plateau-Kriterium ist RELATIV, nicht absolut (< 5 % Verbesserung über die
   nächsten 100 Iterationen). CFMs Warmstart ist bereits dichte-konditioniert, also nah
   an einer guten Lösung -> die Verbesserungskurve flacht schnell ab, einfach weil
   wenig Spielraum bleibt, nicht weil absolut "genug" optimiert wurde. Im Vollauf:
   CFM-Plateau-Median kollabierte auf ~20 Iterationen (statt ~340 im ungedeckelten
   Original-Lauf).
2. Rückkopplungseffekt: dasselbe (CFM-abgeleitete) kleine Budget gilt für alle drei
   Methoden in derselben Runde, inklusive CFMs eigener Ausführung -> kaum echte
   Verfeinerung irgendwo -> wenig präzises Belief-Update -> diffusere Zieldichte in der
   nächsten Runde -> noch weniger Verfeinerung nötig, um "Plateau" zu erreichen. Ein
   sich selbst verstärkender Kollaps Richtung Minimalbudget, sichtbar in den
   Rundenprotokollen (z. B. Budgets ~150--300 in Runde 1 mancher Gruppen, ~10--40 ab
   Runde 4--6).
3. Warum trotzdem CFM >~ linear >> random_walk (nicht alle drei gleich schlecht):
   **Glattheit der Rohinitialisierung entscheidet, wie viel ein kleines Budget bringt.**
   - CFM: glatt UND dichte-bewusst -> braucht am wenigsten.
   - linear: glatt, aber dichte-BLIND -> SVGD muss nur biegen, nicht erst Unebenheiten
     glätten -> mit wenigen Dutzend Iterationen oft fast so gut wie CFM. Zusätzlich:
     30 Kandidaten über 360° verteilt, die Selektion pickt ohnehin die am besten
     ausgerichtete Linie VOR jeder Optimierung.
   - random_walk: weder glatt noch dichte-bewusst -> ein Teil des (ohnehin knappen)
     Budgets geht für reines Glätten drauf, bevor überhaupt Fortschritt Richtung
     Zieldichte möglich ist -> bleibt klar zurück.

   Qualitativ an zwei selbst erzeugten Abbildungen bestätigt (siehe Abschnitt 6).

**Offene Konsequenz / Vorschlag (noch nicht umgesetzt, Philipp muss entscheiden):**
Das aktuelle Budget-Matching testet dadurch nicht sauber "wie viel Rechenzeit braucht
CFM wirklich", sondern rutscht in ein entartetes Regime mit praktisch keiner Verfeinerung
für irgendeine Methode. Mögliche Fixes, die besprochen, aber NICHT implementiert wurden:
- Eine Mindest-Iterationsschwelle (Floor) fürs Budget einführen, damit das Plateau-
  Kriterium nicht gegen nahe Null kollabieren kann.
- Budget einmalig/referenziell bestimmen (z. B. aus dem ungedeckelten Original-Lauf)
  statt es rundenweise im selben, sich selbst degradierenden Lauf neu herzuleiten --
  entkoppelt die Budget-Messung vom Rückkopplungseffekt.
- Ggf. ein absolutes statt relatives Verbesserungskriterium.

## 6. Qualitative Visualisierungen

Zwei Abbildungen erzeugt (Skript: nur im Scratchpad der Session, NICHT ins Repo
übernommen -- bei Bedarf in `evaluation_full_matrix/` als Tool nachziehen):
- **Figur A** (`qual_A_driven_paths.png`): volle gefahrene Pfade, 3 Methoden x 3
  Beispiel-Shapes (`A`, `organic_10`, `cjk_0`, Bedingung `none_known`/`eid`) auf der
  WHITE_INFERNO-Zieldichte. Zeigt direkt: cfm braucht am wenigsten Einheiten (9/14/18),
  linear ist nah dran (14/13/19), random_walk klar mehr (13/18/24).
- **Figur B** (`qual_B_raw_vs_refined.png`): für `organic_10`, Runden 1--4, pro Methode
  Rohinitialisierung (gestrichelt, Iteration 0) vs. tatsächlich gefahrene,
  budget-gedeckelte Kurve (durchgezogen) -- visuelle Bestätigung des
  Glattheits-Arguments aus Abschnitt 5.

Diese beiden PNGs lagen nur im Session-Scratchpad (`/tmp/claude-1000/.../scratchpad/`),
sind flüchtig und nach Sessionende vermutlich nicht mehr vorhanden -- bei Bedarf das
Erzeugungsskript neu schreiben (Logik: siehe Abschnitt 5, nutzt
`mission_db.iter_rounds`/`iter_candidates(with_states=False)`, `init_curve`-Feld der
`candidates`-Tabelle für die Rohkurve, `rounds.segment` fürs tatsächlich Gefahrene).

## 7. Offene nächste Schritte (Stand Ende dieser Session)

- Entscheidung aussteht: Budget-Floor-Fix implementieren und neu rechnen, oder die
  aktuellen Ergebnisse so stehen lassen (mit der Kollaps-Erklärung als eigenes Finding
  in der Thesis)?
- Qualitative Visualisierungen ggf. als festes Tool ins Repo übernehmen.
- Unabhängig davon laufen auf dem Cluster parallel noch andere, NICHT von mir gestartete
  Jobs (z. B. `smooth_c...`/`svgd_con...`, zu einem separaten Smoothness-Feature in
  `exploration/common/sun_refine.py` + `svgd_batched.py`, das bereits vor dieser Session
  uncommitted im Arbeitsverzeichnis lag) -- siehe `git status`, falls das relevant wird.

## 8. Relevante Dateipfade (Kurzreferenz)

Code (lokal + auf Cluster synchronisiert):
- `thesis_architecture/evaluation_full_matrix/run_mission_eval_budget_matched.py`
- `thesis_architecture/evaluation_full_matrix/run_mission_eval.py` (editiert)
- `thesis_architecture/evaluation_full_matrix/state_codec.py` (editiert)
- `thesis_architecture/evaluation_full_matrix/run_job_mission_eval_budget_matched_smoke.bash`
- `thesis_architecture/evaluation_full_matrix/run_job_mission_eval_budget_matched.bash`

Ergebnisse:
- `thesis_architecture/evaluation_full_matrix/results/mission_eval_budget_matched_20261008_sun/`
  (lokal und auf dem Cluster)
- Vergleich/Referenz: `thesis_architecture/evaluation_full_matrix/results/mission_eval_20261006/`
  (Original, fixes 1500-Iterationen-Budget)

Cluster-Job-IDs: Smoketest 164250, Vollauf 164886 (beide `stud_schonfeld@mn.ias.informatik.tu-darmstadt.de`).
