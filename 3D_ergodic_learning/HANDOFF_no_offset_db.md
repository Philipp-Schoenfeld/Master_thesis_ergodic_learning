# Handoff: 3D-Datenbank ohne Offset/Sprung-Bug + Feinjustierung

Kontext fuer die Fortsetzung auf einem anderen PC / dem Cluster. Stand: 2026-09-04.

## Was passiert ist

`3D_ergodic_learning/project_db_3d.py` projiziert die 2D-Ground-Truth-Bahnen aus
`ergodic_dataset_775.db` auf 3D-Oberflaechen. Zwei Defekte wurden gefunden und
behoben.

**1. Standoff-Offset.** Jeder Bahnpunkt wurde per `pos = hitp + standoff * hitn`
um `standoff=0.12` entlang der Flaechennormale nach aussen verschoben — die
Zielbahn lag also nicht auf der Zielverteilung, sondern 0.12 Einheiten davor.
Fix: Default von `--standoff` (CLI und `projiziere()`) auf `0.0` gesetzt.

**2. Fehlschuss-Fallback riss Spruenge in die Bahn.** Wenn ein Bahnpunkt beim
Raycasting die Flaeche verfehlte (2D-Kurve lief ueber die projizierte
Silhouette hinaus), wurde er durch den *naechstgelegenen* Punkt auf der
gesamten Flaeche ersetzt. Bei konvexen Formen (Kugel, Wuerfel) liegt dieser
Punkt zwar auf der Flaeche, aber wegen der Kruemmung oft weit entfernt von der
eigentlichen Bahn — sichtbar als lange, gerade "Ausreisser" quer durch die
Szene (siehe Chat vom 2026-09-04, Screenshot Stanford-Bunny). Ueberpruefung:
selbst bei der Kugel (Fehlschuss nur ~1 % im Median) lag der Median-Sprung
`jump_max` ueber alle Eintraege bei 0.39 — fast so gross wie die Bahnlaenge
selbst. Fix: Fehlschuss-Rohpunkte werden verworfen statt ersetzt; `n_points`
ist dadurch je Eintrag variabel (bereits vorher so im Schema vorgesehen), die
bestehende Downstream-Auswahl der `nxi=25` Kontrollpunkte
(`np.linspace(0, n_points-1, nxi)` in `data_surfaces.py`) verteilt sich einfach
ueber die kuerzere, saubere Restbahn — keine Aenderung dort noetig.

**3. Zufaelliger Startpunkt ergaenzt.** `ergodic_pairs.x0` (der 2D-Startpunkt,
von dem aus der SVGD/CE-Solver die Bahn ueberhaupt erst geplant hat — nicht
identisch mit `trajectory[0]`, das schon einen Zeitschritt Dynamik zurueckgelegt
hat) wurde bisher beim 3D-Projizieren gar nicht mit uebernommen. Er wird jetzt
durch denselben Projektor geworfen und als `start_pos`/`start_rot6` (plus
`start_xy` zur Nachvollziehbarkeit) in `ergodic_pairs_3d` gespeichert. Aktuell
nur gespeichert, noch **nicht** als Konditionierung ins Netz verdrahtet — das
waere eine eigene Architektur-Entscheidung (neuer Eingabekanal), hier bewusst
nicht mitgemacht.

Vorher/Nachher ueber alle 775 Eintraege je Oberflaeche (`AVG(jump_max)`, aus
der `--build`-Konsolenausgabe, jeweils schon mit standoff=0.0 — reiner
Effekt des Fehlschuss-Fixes):

| Oberflaeche      | Sprung vorher | Sprung nachher |
|------------------|--------------:|---------------:|
| wuerfel          |         0.586 |          0.193 |
| bunny            |         0.391 |          0.230 |
| ei               |         0.237 |          0.100 |
| kugel            |         0.290 |          0.106 |
| prisma           |         0.278 |          0.211 |
| torus            |         0.227 |          0.241¹|
| kegel            |         0.109 |          0.120¹|
| ebene (3×)       |    0.057–0.064|     0.057–0.064|

¹ Torus und Kegel minimal gestiegen statt gefallen. Nachvollziehbar: bei einer
Kugel oder einem Wuerfel liegt der naechstgelegene Punkt fuer einen
Fehlschuss-Rand unter Umstaenden auf einer ganz anderen Stelle der Flaeche
(grosser Fehler durch die alte Fallback-Logik). Bei Torus (Loch in der Mitte)
und Kegel (Spitze am Silhouettenrand) grenzt dagegen fast ueberall Flaeche an
die Fehlschuss-Zone an, der alte Fallback-Punkt lag also meist schon zufaellig
nicht allzu weit daneben. Wird der Rohpunkt stattdessen entfernt, ist die
direkte Verbindung der beiden echten Nachbarpunkte ueber die (reale, nicht
mehr kaschierte) Luecke in seltenen Faellen minimal weiter als der alte,
zufaellig brauchbare Fallback-Punkt war. Fehlschuss-Anteil je Oberflaeche ist
durch den Fix ohnehin unveraendert, weil dieselben Rohpunkte wie vorher als
Fehlschuss erkannt werden — nur ihre Behandlung hat sich geaendert.

Ergebnisdatei: `3D_ergodic_learning/ergodic_dataset_3d_no_offset.db` (~166 MB —
ueber GitHubs 100-MB-Limit, deshalb in `.gitignore` und **nicht** im Commit;
das ist der Teil, der manuell per Drive uebertragen werden muss).

Die alte Datei `ergodic_dataset_3d.db` (standoff=0.12, Fallback-Bug) liegt
unveraendert auf dem Cluster und wird von den bisherigen `surfB_lang`-
Checkpoints referenziert — zum Vergleich nicht ueberschreiben, sondern die
neue Datei unter dem neuen Namen daneben legen.

Interaktive Vorher/Nachher-Ansicht (alle 10 Oberflaechen, 4 Formen je
Oberflaeche, inkl. Startpunkt-Marker): Artifact "Flächenkontakt"
(<https://claude.ai/code/artifact/165cbe93-1a30-4e0c-8648-b0f961a37eb1>) —
Export dafuer in `3D_ergodic_learning/export_offset_check_viz.py` (kein Teil
der Pipeline, nur Debug-Tool).

## Was mit git/rsync auf den anderen PC kommt

Commit enthaelt:

- `3D_ergodic_learning/project_db_3d.py` (Standoff-, Fehlschuss- und Startpunkt-Fix)
- `3D_ergodic_learning/data_surfaces.py` (Docstring an neues Fehlschuss-Verhalten angepasst)
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
  den Standoff-Fix nicht. Das Risiko liegt also ausschliesslich auf der
  Positions-, nicht der Rotationsseite.

Der zweite Fix (Fehlschuss-Fallback → Sprung-Bug, oben) aendert die
Einschaetzung eher zum Besseren: er entfernt Label-Rauschen (die langen
Ausreisser), stoert die gelernte Loesung also nicht in eine neue Richtung,
sondern macht die Zielverteilung nur sauberer. Falls die Feinjustierung
schneller konvergiert als erwartet, ist das ein plausibles Zeichen dafuer,
nicht eines fuer einen Fehler.

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
