# Kontext: Teacher-Studie (Session 2026-10-24) -- beste Lehrer-Trajektorien fuer die DB

Auftrag (Philipp): umfangreiche Studie, wie sich mit unseren drei SVGD-Loesern (Sun FM, Adam-SVGD,
TSVEC-SVGD, gern weitere Optionen) und verschiedenen Smoothness-Kraeften moeglichst ergodische UND sehr
glatte Trajektorien fuer die Lehrer-DB erzeugen lassen. Idee: Waypoints optimieren, dann auf B-Splines
projizieren (zusaetzlicher Glaettungseffekt). Vorher Git-Stand holen und mergen (Fast-forward auf 0b32274e,
lokale Aenderungen unberuehrt; vor dem Push erneut gefetcht: nichts Neues auf origin/main).
Zusatzauftrag: besonders gegen Philipps Setup vergleichen (siehe 4).

## 1. Entscheidungen
- Auswahlregel: pro Form der glatteste Kandidat, dessen Hellinger hoechstens 0.05 ueber dem Bestwert der Form
  liegt. Pool = 25-CP-B-Spline und seine Polish-Stufen (`--pool db`). Smoothness = `smooth` (Beschleunigungs-
  energie, Projektkonvention, W_SMOOTH * sum accel^2 auf der bogenlaengen-resampelten Bahn).
- Randfilter (neu, Standard): Kandidaten mit Punkten ausserhalb [0.03, 0.97] werden bei der Auswahl
  ausgeschlossen (`--max_outside 0`). Klemmen der Kontrollpunkte statt Ausschluss kostet +0.02 Hellinger (zu
  teuer); der Ausschluss kostet fast nichts (Auswahl: Hellinger -0.023 statt -0.026 gegenueber DB).
- Lange Laeufe nur nach Freigabe; Pilot und Vollstudie liefen lokal auf Philipps ausdruecklichen Auftrag.

## 2. Aufbau (`thesis_architecture/teacher_study/`)
- `pipeline.py`: eine Zelle = (Form, Konfiguration): Init (30 Sehnen von x0 zu Punkten eines Rings, oder DB-Lehrer
  + glattes Rauschen) -> Loeser -> Stufen `wp` (128 Waypoints), `bs<nxi>[d<grad>]` (LS-Projektion auf geklemmte
  B-Spline-Basis, CP 0 = x0 exakt), `bs25_pol<k>` (k Adam-Schritte im CP-Raum, ohne Schwarm). Metriken pro Stufe:
  Hellinger, tv_2/4/8, low_excess, Coverage, spec_rel_20, Fourier-Fehler K=10, smooth, Pfadlaenge, Zeit-RMS von
  Geschwindigkeit/Beschleunigung/Ruck (Gesamtzeit = 1), Krümmung p95/max, Randanteil, Startfehler, `diverged`.
- `configs.py`: Gitter `smoke` (10), `pilot` (49), `full` (18). `run_teacher_study.py` (resumierbar, `--split`,
  `--n_shapes`, `--configs`, `--time_budget_h`), `analyze_teacher_study.py` (Auswahl, Ranking, report.md, vier Plots),
  `plot_best.py` (Gesamtuebersicht), `plot_variants.py` (Varianten x Formen), `test_teacher_study.py` (10 Tests,
  ~30 s, gruen).
- Erweiterungen bestehender Module (Defaults bit-identisch): `tsvec_svgd.TsvecSvgd(smooth_order=1|2|3)` mit
  `_diff_matrix` (Residuum h^2 x^(order), h = 1/(T-1), Order 2 == `_accel_matrix`), `adam_svgd.AdamSvgd(smooth_order)`
  + `run(start=)` (harter Pin), `sun_sinkhorn.run_batch(smoothness_weight, smooth_order)` (nur Torch-Pfad),
  `svgd_batched.BatchedSvgdTorch(lr=)` (Default 2e-3).
- Loeser in der Studie: `tsvec`, `adam` (gemeinsamer Schwarm, Liu&Wang/Lee), `lm`, `gui_adam` (Philipps Setup),
  `sun_sinkhorn`, `sun_stein`. `lm` = eigener batchweiser Levenberg-Marquardt auf der TSVEC-Residuenenergie, KEIN
  Schwarm, deterministisch, lokales Minimum pro Startkurve (Dämpfung relativ zur mittleren Diagonale von J^T J,
  lambda /3 bei Erfolg, x4 bei Misserfolg, Klemmen auf [0.02,0.98] und Start-Pin nach dem Schritt). Optionen: `K`
  (Fourier-Moden der Solver-Energie, gemeldete Metriken bleiben K=10), `schedule` (Gewichts-Kontinuation).
- `bsplinax` ist lokal nur ueber `../bsplinax-main` importierbar (pipeline.py haengt den Pfad an).

## 3. Kalibrierung (Form A) und Pilot
- Iterationen: LM 100-300, TSVEC > 600 (2000 gewaehlt), Adam > 2500 (5000 gewaehlt, noch nicht ganz konvergiert).
- Gewichtsbereiche: Ruck 0.01-0.3 (w=2 bleibt auf Sehnen stecken; TSVEC-Ruck >= 0.03 kollabiert, Hellinger 0.57-0.81),
  Beschleunigung 5-150 (>= 150 verliert Ergodik), Sinkhorn-Beschleunigung bis 3000 ok, Sinkhorn-Ruck ab 30 kollabiert,
  Geschwindigkeitskraft 1000-3000 sehr ergodisch, aber kaum glatter. Sun-Stein konvergiert bei 1000 Its. nicht gut
  (Hellinger 0.35, mit Ruck-Dynamik 0.46).
- Pilot (3 Formen A, korean_5, organic_30; 49 Konfigurationen; 33.7 min): `results/pilot_20261024/`.

## 4. Vollstudie `results/full_20261024/` (25 Holdout-Formen, 18 Konfigurationen + DB, 475 Zellen, ~80 min)
Philipps Referenz-Setup `gui_*`: Adam-SVGD des GUI-Refiners (`BatchedSvgdTorch`, 8 Partikel je Bahn, K=8,
600*Ergodik + 15*Beschl.^2 + Rand, lr 8e-3, Startpin W_START=500 + harter Snap am Ende, bester Partikel nach Energie)
auf 128 Waypoints; Iterationszahl war nicht vorgegeben -> 1000 und 3000 (+ lr 2e-3 als Kontrolle).

Mittel ueber Kandidaten und Formen, Stufe bs25 (Hellinger | smooth | Beschl. RMS | Ruck RMS | Kandidaten mit Randpunkten):
| Konfiguration | H | smooth | acc | jerk | outside>0 |
|---|---|---|---|---|---|
| teacher_db | 0.248 | 0.329 | 221 | 16600 | 0.16 |
| gui_lr8e-3_n3000 (Referenz) | 0.272 | 0.215 | 128 | 4821 | 0.47 |
| gui_lr8e-3_n1000 / lr2e-3_n3000 | 0.283 / 0.286 | 0.361 / 0.331 | 166 / 159 | 7173 / 6860 | - |
| lm_o2_w15 | 0.231 | 0.382 | 162 | 7144 | 0.15 |
| lm_K16_o2_w15 | 0.217 | 0.463 | 180 | 8376 | 0.16 |
| lm_K16_cont_o2_w15 (empfohlen) | 0.234 | 0.213 | 134 | 7172 | 0.09 |
| lm_cont_o2_w15 | 0.253 | 0.195 | 127 | 6442 | - |
| sun_sinkhorn_o2_w3000 / o3_w3 | 0.235 / 0.244 | 0.261 / 0.230 | 136 / 118 | 6999 / 4644 | - |
| adam_o2_w15 | 0.243 | 0.304 | 155 | 7569 | - |
| tsvec_o2_w15 | 0.252 | 0.427 | 178 | 8836 | - |
- `lm_K16_cont_o2_w15` hat dieselbe Glattheit wie die Referenz (0.213 vs 0.215) bei Hellinger 0.234 statt 0.272
  und ist auf 25/25 Formen ergodischer (gepaart). Die Referenz ist bei 3000 Iterationen noch nicht konvergiert
  (1000 -> 3000: smooth 0.361 -> 0.215); ein Lauf mit ~10000 Its. wurde NICHT gemacht.
- Auswahl (Regel + Randfilter): Hellinger im Mittel -0.023 gegenueber DB; Median-Verhaeltnisse zur DB: smooth 0.35,
  Beschleunigung 0.50, Ruck 0.26. Die DB-Bahn verliert allein durch die B-Spline-Projektion ~0.045 Hellinger
  (Waypoints 0.154 -> bs25 0.189 auf A); bei glatten Waypoints kostet die Projektion nur 0.005-0.01.
- Polish (100/300 Schritte) bringt kaum etwas. K=16 senkt Hellinger (0.231 -> 0.217), Kontinuation macht glatter
  (smooth 0.463 -> 0.213) bei ~0.017 mehr Hellinger.

## 5. Stichprobe Trainingsformen `results/train_sample_20261024/` (40 zufaellige Formen, seed 0, 5 Min./Konfig.)
Bestaetigt das Holdout-Bild: Auswahl Hellinger -0.021 gegenueber DB, smooth 0.32x, Beschl. 0.46x, Ruck 0.25x
(Median); `lm_K16_cont_o2_w15` liefert 31 von 40 gewaehlten Teachern. Mittel bs25: DB H 0.246 / smooth 0.336;
Referenz 0.265 / 0.208; lm_K16_cont 0.226 / 0.202. Randanteil: DB 30 % der Kandidaten, Referenz 50 %, lm_K16_cont 8 %.

## 6. Einordnung / Vorbehalte (wichtig fuer die Arbeit)
- LM ist KEIN State of the Art der ergodischen Trajektorienoptimierung, sondern ein klassisches Verfahren
  (Levenberg 1944, Marquardt 1963; aus dem Gedaechtnis zitiert, nicht nachgeschlagen) mit eigener, einfacher
  Implementierung. Ergodik-Literatur (ebenfalls aus dem Gedaechtnis): Mathew & Mezic 2011 (Metrik), Miller & Murphey
  2013, Lee et al. 2024 (Stein Variational Ergodic Search), Sun et al. 2025, Li et al. 2026 (TSVEC).
- Der Vergleich ist fair fuer "beste Einzelbahn in unserer Pipeline", NICHT fuer "Verfahren X grundsaetzlich besser":
  LM hat keine Schwarm-Abstossung, nutzt Gauss-Newton (konvergiert in ~100-300 Its.), Adam/TSVEC bekamen weniger
  Abstimmung/Budget (TSVEC TAU = 0.1), LM hatte 9 von 18 Konfigurationen (Mehrfachvergleich), die Sun-Loeser loesen ein
  dynamisch gebundenes Problem. Unterschiede in Hellinger sind klein gegen die Untergrenze 0.14-0.33.
- Nicht gemacht: Abgleich gegen `scipy.optimize.least_squares`; Gegenproben (Adam ohne Kernel, TSVEC mit groesserem TAU,
  Adam 15000 Its.); Trainingsvergleich des Flow-Matching-Netzes mit den neuen Teachern (das eigentliche Kriterium).

## 7. Empfehlung und naechste Schritte
Neu generieren mit `lm_K16_cont_o2_w15` (LM, K=16, Kontinuation 150>50>15, 128 Waypoints, Projektion auf 25 CP
Grad 5 mit festem Start, 30 Kandidaten je Form, Auswahlregel + Randfilter). Alternativen: `lm_K16_o2_w15` (mehr
Ergodik, weniger glatt), `gui_lr8e-3_n3000` (SVGD-Linie, schlechtere Ergodik). Laufzeit LM ~3 s je Form fuer 30
Kandidaten (750 Formen < 1 h, Schaetzung). Offen: DB-Export-Skript (Auswahl -> `ergodic_pairs`-Format, Bahn als
Float32 (200,2) bzw. CPs), Gegenproben aus 6, Trainingsvergleich.

## 8. Befehle
```
cd thesis_architecture/teacher_study
python run_teacher_study.py --grid full --out_tag full_20261024            # 25 Holdout-Formen, ~80 min
python run_teacher_study.py --grid full --split train --n_shapes 40 --configs lm_K16_cont_o2_w15,lm_K16_o2_w15,lm_o2_w15,gui_lr8e-3_n3000 --out_tag train_sample_20261024
python analyze_teacher_study.py --out_tag full_20261024                    # report.md, selection.csv, Plots
python plot_best.py --out_tag full_20261024 --best lm_K16_cont_o2_w15 --ref gui_lr8e-3_n3000
python plot_variants.py --out_tag full_20261024
python test_teacher_study.py
```
Ergebnisordner: `report.md`, `summary_config_stage.csv`, `selection.csv`, `plots/` (pareto, force_sweeps,
projection_effect, selection_gallery, best_overview, variants_grid), `finals/*.npz` (Waypoints bzw. Kontrollpunkte je
Stufe), `metrics.csv.gz` (die unkomprimierte CSV ist per .gitignore ausgeschlossen; `analyze` liest die .gz).
