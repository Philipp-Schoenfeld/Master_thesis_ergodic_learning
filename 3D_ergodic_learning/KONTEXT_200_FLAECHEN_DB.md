# Wie `ergodic_dataset_3d.db` entsteht

`ergodic_dataset_3d.db` (91.950 Einträge über 200 Trägerflächen, 1.307
Formen) ist selbst **nicht** im Repository — über GitHubs 100-MB-Limit je
Datei, siehe `.gitignore`. Diese Datei dokumentiert stattdessen den Code, der
sie erzeugt, damit sie aus einem sauberen Checkout heraus reproduzierbar
bleibt.

## Zwei-Stufen-Pipeline

1. **2D-Grundlage** (`thesis_architecture/ergodic_dataset_generator/`):
   der SVGD-Löser erzeugt für jede Zieldichte eine ergodische 2D-Bahn. Diese
   Datenbank ist `DB_IN` in `project_db_3d.py` (Voreinstellung
   `thesis_architecture/ergodic_dataset_generator/ergodic_dataset_775.db`
   oder eine neuere Variante — siehe `--db_in`).
2. **3D-Projektion** (`project_db_3d.py`): Dichte *und* Bahn werden durch
   denselben Strahlprojektor auf eine gekrümmte 3D-Trägerfläche geworfen
   (siehe Modulkopf dort für das Verfahren). Ergebnis ist die SE(3)-Bahn
   `ergodic_dataset_3d.db`.

```
python project_db_3d.py --build
```

baut mit den Voreinstellungen alle Flächen aus `surfaces.alle_keys()`. Für
einen schnellen Rauchtest vorher: `python project_db_3d.py --probe`
(10 Trainings- + 4 Holdout-Paare je Blickwinkel, eigene Ausgabedatei
`ergodic_dataset_3d_probe.db`).

## Woher die 200 Flächen kommen

`surfaces.py` registriert acht Gruppen (`GRUPPEN`), die sich auf mehrere
Module verteilen:

| Gruppe | Anzahl | Modul | Verfahren |
|---|---|---|---|
| primitiv | — | `surfaces.py` / `koerper.py` | analytische Grundkörper |
| ebene | — | `surfaces.py` | ebene Referenzflächen |
| buchstabe | — | `surfaces.py` (DejaVu Sans) | Einzelbuchstaben als Volumenkörper |
| extern | — | `externe_netze.py` | fremde Netze, siehe `LIZENZEN.md` |
| organismus | 45 | `organismen.py` | Metaball-Skalarfeld + `skimage.measure.marching_cubes` |
| baugruppe | 25 | `baugruppen.py` | boolesche Verknüpfung eigener `koerper.py`-Grundkörper via `manifold3d` |
| superquadrik | 28 | `superquadriken.py` | geschlossene Formel (`e1`, `e2`) auf Ikosphären-Topologie |
| wort | 25 | `woerter.py` | `text_volumen.baue()` auf kurze Zeichenketten |

Die letzten vier Gruppen (123 Formen) sind die Erweiterung von 77 auf 200
Flächen — vollständig selbst erzeugt, keine externe Quelle (siehe
`LIZENZEN.md`, Abschnitt "Eigene prozedurale Formen"). Die `extern`-Gruppe
dagegen bezieht echte fremde Netze (Open3D-Testdaten, Khronos-glTF-Samples);
Herkunft, Lizenz und Zitierhinweis je Modell stehen ebenfalls in
`LIZENZEN.md`.

## Zusätzliche Abhängigkeiten

Für die 200-Flächen-Erweiterung neu in `requirements.txt`: `trimesh`,
`open3d`, `scipy`, `shapely`, `mapbox-earcut` (Triangulierung von Buchstaben
mit Löchern wie O/A/B/R/0), `scikit-image` (Marching Cubes) und
`manifold3d` (boolesche Verknüpfung der Baugruppen — ohne diese Bibliothek
fällt `trimesh.boolean` still auf ein nicht mehr wasserdichtes Ergebnis
zurück, siehe Notiz in `koerper.halbkugel()`).

## Wo sie gebaut wurde

Der Bau ist reine CPU-Arbeit (Strahlprojektion, kein Netztraining) und lief
lokal, nicht als Cluster-Job — es gibt kein `run_job_3d_build*.bash` und
keine passenden Cluster-Logs. Die Kopie auf dem Cluster
(`~/Master_thesis/3D_ergodic_learning/ergodic_dataset_3d.db`) ist identisch
zur lokalen (gleiche Größe, gleicher Zeitstempel 29.08., 16:37) und kam per
`rsync` dorthin, wie im Cluster-Workflow in der Projekt-`CLAUDE.md`
beschrieben.
