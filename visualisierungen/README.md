# Visualisierungen

Sammelordner für alle Ergebnisse, die im Rahmen dieser Session erzeugt wurden.
Jeder Unterordner entspricht einer Aufgabe; die Nummerierung ist die
Bearbeitungsreihenfolge.

```
01_phi_kreuz_laenge/       Neues Φ-Kreuz mit dem Längen+Startpunkt-Checkpoint
  trajektorien/               je Zelle (Muster_Φ-Modell) ein Unterordner mit
                               bahnen.json, metriken.csv, anytime.json, PNGs
  metriken/                   zusammengefasste Auswertung über alle Zellen
  seite/                      die fertige interaktive Seite (HTML)

02_holdout_vergleich/      Holdout-Generierung + Metriken für alle drei Checkpoints
  2d_startpunkt/               netz2d_startpunkt.pt — Visualisierungsgitter
  2d_laenge/                   netz2d_laenge.pt — Visualisierungsgitter
  3d_flaechen/                 netz3d_flaechen.pt — Visualisierungsgitter
  metriken/                    summary.csv/per_shape.csv/results.json je Checkpoint
  seite/                       die fertige interaktive Seite (HTML)

logs/                      Rohe Konsolen-Logs der Läufe, ein Log pro Ordner oben,
                            zum Nachvollziehen falls eine Zahl überrascht.
```

**Checkpoint-Herkunft:** alle drei Checkpoints liegen unter
`thesis_architecture/checkpoints/` (2D) bzw. `3D_ergodic_learning/checkpoints/`
(3D), kopiert aus `thesis_architecture/transfer/`.

**Wichtiger Befund vorab:** `netz2d_laenge.pt` ist mit nur 39 Trainings-Epochen
deutlich weniger ausgereift als die anderen beiden (500 bzw. 1519 Epochen) —
seine Zahlen hier sind ein früher Zwischenstand, kein Endergebnis. Die
Längenkonditionierung selbst reagiert bei diesem Trainingsstand noch kaum auf
die angeforderte Ziellänge (siehe `logs/`) — das Φ-Kreuz in `01_.../` läuft
deshalb ohne `--target_length`, als reiner Qualitätstest der belief-getriebenen
Exploration mit dem neuen, größeren Netz (nxi=64 statt 25).
