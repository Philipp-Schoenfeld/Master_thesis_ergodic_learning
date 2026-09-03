# Handoff: offsetfreie 3D-Datenbank + Feinjustierung

Kontext fuer die Fortsetzung auf einem anderen PC / dem Cluster. Stand: 2026-09-03.

## Was passiert ist

`3D_ergodic_learning/project_db_3d.py` projiziert die 2D-Ground-Truth-Bahnen aus
`ergodic_dataset_775.db` auf 3D-Oberflaechen. Dabei wurde jeder Bahnpunkt per
`pos = hitp + standoff * hitn` um `standoff=0.12` entlang der Flaechennormale
nach aussen verschoben — die Zielbahn lag also nicht auf der Zielverteilung,
sondern 0.12 Einheiten davor.

**Fix:** Default von `--standoff` (CLI und `projiziere()`) auf `0.0` gesetzt.
Lokal neu gebaut und verifiziert:

```
7750 Eintraege, standoff_mean = 0.0 / standoff_sd = 0.0 fuer alle Zeilen
```

Ergebnisdatei: `3D_ergodic_learning/ergodic_dataset_3d_no_offset.db` (177 MB —
ueber GitHubs 100-MB-Limit, deshalb in `.gitignore` und **nicht** im Commit;
das ist der Teil, der manuell per Drive uebertragen werden muss).

Die alte Datei `ergodic_dataset_3d.db` (standoff=0.12) liegt unveraendert auf
dem Cluster und wird von den bisherigen `surfB_lang`-Checkpoints referenziert —
zum Vergleich nicht ueberschreiben, sondern die neue Datei unter dem neuen
Namen daneben legen.

## Was mit git/rsync auf den anderen PC kommt

Commit enthaelt:
- `3D_ergodic_learning/project_db_3d.py` (standoff-Fix)
- `3D_ergodic_learning/run_job_3d_finetune_no_offset.bash` (neuer Feinjustierungs-Job)
- `3D_ergodic_learning/HANDOFF_no_offset_db.md` (diese Datei)
- `.gitignore` (neue DB-Datei ergaenzt)

Nicht im Commit, manuell per Drive:
- `3D_ergodic_learning/ergodic_dataset_3d_no_offset.db` → auf den Cluster nach
  `~/Master_thesis/3D_ergodic_learning/ergodic_dataset_3d_no_offset.db`

## Ausfuehrung auf dem Cluster

Nach dem Kopieren der DB, **nur per `sbatch`** (nie `srun` direkt, siehe
CLAUDE.md-Sicherheitsregeln):

```bash
cd ~/Master_thesis/3D_ergodic_learning
sbatch run_job_3d_finetune_no_offset.bash
```

Das Skript:
1. Setzt fort, falls schon ein `surfB_nooffset_ft`-Checkpoint existiert
   (`--resume`, inkl. Optimizer/Scheduler/Epoche).
2. Sonst laedt es per `--load_model` **nur die Gewichte** des letzten
   `surfB_lang`-Checkpoints (`ep1750`, trainiert mit standoff=0.12) — das ist
   der Warm-Start. Optimizer/Scheduler/Epochenzaehler starten frisch, damit
   der Kosinus-Lernratenplan nicht auf Epoche ~1750 landet und sofort bei
   Lernrate ~0 haengt.
3. 400 Epochen bei `lr=3e-5` (niedriger als die 1e-4 des Grundlaufs, weil
   Feinjustierung, nicht Neutraining), sonst identische Konfiguration
   (`--orientation --rot_full --lambda_erg 100 --erg_K 6 --erg_on position`).

`--load_model` laedt bewusst nur `model_state_dict` (siehe
`flow_matching_runner_particles.py:438-441`) — im Unterschied zu `--resume`,
das den kompletten Trainingszustand uebernimmt und daher fuer einen echten
Warm-Start auf neuen Daten ungeeignet waere.

## Ist Warm-Start hier realistisch?

Ja, mit Einschraenkung. Architektur, Task-Formulierung und alle uebrigen
Hyperparameter (D=384, nxi=25, n_particles=512, SE(3)-Rahmen via `lookat`,
ergodischer Term) bleiben identisch — nur die Zielposition jedes Bahnpunkts
verschiebt sich um bis zu 0.12 Einheiten senkrecht zur Flaeche (bei Formen mit
Ausdehnung ~1 also ein spuerbarer, aber kein dominanter Anteil der Bahnlaenge).
Das Netz muss keine neue Aufgabe lernen, sondern eine bestehende Loesung an
einen verschobenen Zielraum anpassen — der klassische Fall, in dem Warm-Start
gegenueber Neutraining Zeit spart.

Zwei Punkte, die das Ergebnis trotzdem nicht "kostenlos" machen:
- Der ergodische Term (`--lambda_erg 100`, `--erg_on position`) misst
  Abdeckung relativ zur tatsaechlichen Bahnposition. Mit standoff=0.12 lag die
  Bahn systematisch ausserhalb der Dichte auf der Flaeche; das Netz hat also
  gelernt, mit diesem Versatz einen brauchbaren Kompromiss zu finden. Dieser
  Kompromiss ist nach der Umstellung nicht mehr optimal und muss neu gelernt
  werden — daher 400 Epochen Feinjustierung statt nur weniger Dutzend.
- Die SE(3)-Rahmen (`frame_mode=lookat`) haengen nur von der Flaechennormale
  `hitn` ab, nicht vom `standoff` — die Orientierungs-Labels aendern sich durch
  den Fix nicht. Das Risiko liegt also ausschliesslich auf der Positions-,
  nicht der Rotationsseite.

400 Epochen bei `lr=3e-5` ist eine Ausgangsschaetzung, keine feste Vorgabe:
`--run_tag surfB_nooffset_ft` erlaubt beliebiges Nachlegen per erneutem
`sbatch`, falls die Holdout-Metriken danach noch klar hinter dem alten Lauf
liegen.

## Danach: Auswertung nicht vergessen

Projektstandard (siehe Pinned Memory "evaluation-standard-holdout"): Vergleich
alt (standoff=0.12) vs. neu (standoff=0.0, feinjustiert) **ueber die gesamte
Holdout-Menge**, mit Bildern **und** Metriken — nicht an einer einzelnen Form.
`run_surface_eval.py` / `plot_surface_metrics.py` / `auswertung_3d.py` in
diesem Ordner sind dafuer der bestehende Unterbau.
