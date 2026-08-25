# 3D Ergodic Learning

3D-Kopie der Partikel-Linie aus `thesis_architecture/`. Gleiche Architektur,
gleiche Verlustfunktionen, gleiche Konventionen — aber auf $\mathbb{R}^3$ statt
$\mathbb{R}^2$.

**Arbeitsannahme:** Die Datenbank enthält 3D-Verteilungen und 3D-Trajektorien,
die auf einer Ebene liegen. Der Code ist durchgehend 3D; solange die vorhandene
Datenbank 2D ist, wird sie beim Laden auf die Ebene $z = 0{,}5$ gehoben. Ein
echter 3D-Datensatz ersetzt sie später ohne Änderung an den Aufrufern.

---

## Schnellstart

```bash
conda activate thesis
cd 3D_ergodic_learning

python test_3d_port.py                    # 24 Prüfungen, ~5 s, ohne Datenbank

# Kurzer Machbarkeitslauf (CPU, wenige Minuten)
python flow_matching_runner_particles_selfsupervised.py \
    --shapes A,organic_0,digit_5 --epochs 60 --D 128 \
    --n_particles 128 --grid_res 48 --erg_K 6 --mini_batch 3
```

Auf dem Cluster: `sbatch run_job_selfsupervised_3d.bash` (gestaffelt mit
Machbarkeits-Torwächter), `run_job_particles_3d.bash`, `run_job_eval_3d.bash`.

---

## Was sich gegenüber 2D geändert hat

### Unverändert übernommen

Das U-Net arbeitet entlang der **Trajektorienachse**, nicht im Raum. Faltungen,
Self-Attention, Cross-Attention, Skip-Verbindungen, FiLM-Zeitkonditionierung und
CFG sind daher wörtlich identisch. Ebenso die B-Spline-Basis: `curve = B @ cps`
bildet Kontrollpunkte auf Kurvenpunkte entlang des *Parameters* ab und kennt die
Koordinatendimension gar nicht. Auch Glattheits-, Rand- und Hindernisterm waren
bereits über `sum(-1)` geschrieben und funktionieren unverändert.

### Echte Änderungen

| Was | 2D | 3D | Grund |
|---|---|---|---|
| Fourier-Basis | $\cos(\pi k_1x)\cos(\pi k_2y)$ | $\;\cdot\cos(\pi k_3z)$ | — |
| Moden | $K^2$ = 100 bei K=10 | $K^3$ = **1000** | kubisches Wachstum |
| $\Lambda_k$-Exponent | $-3/2$ | $-2$ | $-(d+1)/2$, Sobolev-Gewicht nach Mathew & Mezić |
| Dichte | Gitter $R^2$ | Volumen $R^3$ | — |
| Gitterauflösung | 128 | **64** | 128³ × 750 Formen ≈ 6 GB |
| Partikel | $(x,y,\mu)$ | $(x,y,z,\mu)$ | — |
| Partikelzahl | 256 | **512** | ein Volumen braucht mehr Stichproben als eine Fläche |
| Hindernis | Kreis | Kugel | — |
| Rotation | $SO(2)$ | $SO(2)$ um $z$, optional $SO(3)$ | siehe unten |
| Visualisierung | `imshow` | 3D-Achse + Bodenprojektion | — |

### Drei Stellen, an denen es nicht beim Verbreitern blieb

**Symmetriebruch beim Hindernis.** In 2D ist die Senkrechte auf eine Tangente
eindeutig (Drehung um 90°). In 3D spannt sie eine Ebene auf. Die gewählte
Richtung ist die Komponente des Mittelpunktsversatzes orthogonal zur Tangente —
genau die Richtung, die ein rein radiales Potential nicht liefern kann. Läuft die
Kurve exakt durch den Mittelpunkt, verschwindet auch die, und der Rückfall ist
das Kreuzprodukt mit derjenigen Koordinatenachse, die am wenigsten zur Tangente
ausgerichtet ist. `test_3d_port.py` prüft alle drei entarteten Fälle.

**Speicher bei $\varphi_k$.** Die volle Basismatrix wäre $(R^3, K^3)$ — bei
R=64, K=10 rund 1 GB pro Aufruf. `target_coeffs_from_grid` rechnet daher in
Blöcken über die Gitterpunkte; das Ergebnis ist identisch, der Spitzenverbrauch
gedeckelt. Dasselbe gilt für `coverage_distance`.

**Die Ebene darf nicht unendlich dünn sein.** Eine exakt flache Zielverteilung
ist mit einer abgeschnittenen Kosinusbasis nicht darstellbar — eine Dirac-Schicht
in $z$ hat Energie bei jeder Frequenz. Die Dichte wird deshalb als Gaußscher
*Slab* der Breite `Z_SIGMA = 0.05` um die Ebene gelegt. Das ist keine Ungenauigkeit,
sondern die Auflösungsgrenze: bei K=10 ist die kürzeste darstellbare Wellenlänge
$2/(K-1) \approx 0{,}22$. Ein dünnerer Slab würde aliasen statt beschrieben zu
werden.

---

## Warum die Rotation standardmäßig nur um $z$ geht

Bei planaren Daten hält eine Drehung um die Ebenennormale die Ebene eine Ebene.
Das ist die Voraussetzung dafür, einen 3D-Lauf gegen die 2D-Referenz zu prüfen.
`--rot_full` schaltet auf allgemeines $SO(3)$ um und kippt die Daten aus ihrer
Ebene — sinnvoll, sobald die Datenbank wirklich 3D ist, irreführend davor.

---

## Die Planaritätsmetrik

Neu gegenüber 2D und der wichtigste Selbsttest dieses Ports:

```python
planarity(curves)   # RMS-Abstand der Kurve von ihrer eigenen Ausgleichsebene
```

Bei planaren Zielen sollte eine gesunde Bahn nahe 0 liegen. Die Metrik
unterscheidet zwei Fehlerbilder, die sonst gleich aussehen:

- **≈ 0 und niedrige Energie** — korrekt: die 3D-Pipeline reproduziert das 2D-Verhalten.
- **≈ 0, aber die Ausgabe ignoriert $z$ völlig** — würde auffallen, weil der
  Solver-Referenzwert exakt 0,00000 ist und ein degeneriertes Modell das nicht
  gleichzeitig mit guter Energie erreicht.
- **deutlich > 0** — das Modell nutzt die dritte Dimension, obwohl das Ziel flach
  ist. Bei planaren Daten ist das ein Fehler, bei echten 3D-Daten das Ziel.

`evaluate_models.py` gibt sie als eigene Spalte aus.

---

## Verifikation

`python test_3d_port.py` — 24 Prüfungen, alle bestehen:

```
[ok] 3D basis on z=0 equals 2D basis (k3=0 slice)     max |diff| = 0.00e+00
[ok] same at z=0.5 (the plane the DB is lifted to)    max |diff| = 0.00e+00
[ok] analytic grad matches autograd                   max |diff| = 1.07e-13
[ok] chain rule B^T g equals autograd                 max |diff| = 3.16e-13
[ok] polish clears: x-axis through centre             0.1170 -> 0.00e+00
[ok] polish clears: space diagonal                    0.1149 -> 0.00e+00
[ok] z-rotation keeps planar data planar              max z spread = 0.00e+00
[ok] --rot_full does tilt out of plane                max z spread = 0.235
```

Die erste Prüfung ist die tragende: sie zeigt, dass die 3D-Basis auf die
2D-Basis **zurückfällt** statt still etwas anderes zu rechnen.

End-to-End auf der echten Datenbank getestet: Datenladen (775 Formen, 2D→3D
gehoben), Volumenbau, Partikelziehung, beide Runner, Auswertung und
Visualisierung laufen durch. Der Solver-Referenzwert hat Planarität exakt
0,00000 — die gehobenen Bahnen liegen also perfekt in der Ebene, wie erwartet.

---

## Orientierung (Stufe 0 + 1 + 2)

Vollständig implementiert und opt-in. Ohne `--orientation` ist jeder Lauf
zeichengleich mit einem reinen Positionslauf — kein Kopf wird gebaut, kein Feld
berechnet, kein Term addiert.

```bash
# Stufe 0 allein: Frames aus einer vorhandenen Kurve, ohne Modell
python -c "
from orientation import SurfaceField, frames_for_curve
R = frames_for_curve(curve, SurfaceField(volume), mode='lookat')"

# Stufe 1+2: selbstüberwacht Position und Orientierung gemeinsam lernen
python flow_matching_runner_particles_selfsupervised.py \
    --orientation --ergodic_on footprint --standoff_target 0.12

# Stufe 1 überwacht: Labels kommen aus Stufe 0
python flow_matching_runner_particles.py --orientation --frame_mode lookat
```

### Stufe 0 — Frames aus der Kurve

`orientation.py` bietet zwei Konstruktionen, weil sie verschiedene Fragen
beantworten:

- **`lookat`** — die Sensorachse zeigt auf die Fläche, der verbleibende
  Freiheitsgrad folgt der Fahrtrichtung. Aufgabenbewusst, Standard.
- **`rmf`** — Rotation-Minimizing Frame per Doppelreflexion (Wang et al. 2008):
  folgt der Kurve mit minimaler Verdrillung, braucht kein Ziel. **Frenet-Serret
  ist bewusst nicht der Standard** — es ist bei Krümmung null undefiniert und
  kippt an Wendepunkten, und generierte Bahnen enthalten regelmäßig fast gerade
  Abschnitte.

Stufe 0 ist zugleich der **Label-Lieferant für den überwachten Zweig**: die
Datenbank hat keine Orientierung, also erzeugt `orientation_targets()` sie aus
der gespeicherten Bahn. Ein so trainiertes Netz kann Stufe 0 höchstens
erreichen — weshalb der selbstüberwachte Zweig, der Orientierung aus der
Zielfunktion lernt, der interessantere ist.

### Stufe 1 — der Orientierungskopf

6D-Darstellung nach Zhou et al. (2019): zwei 3-Vektoren, per Gram-Schmidt zur
Rotationsmatrix. Der Grund ist nicht Bequemlichkeit — **jede Darstellung von
SO(3) mit weniger als fünf Dimensionen ist notwendig unstetig**, ein Netz auf
Eulerwinkeln oder Quaternionen muss also um eine Unstetigkeit herumlernen, die
es nie treffen kann.

Der zweite, wichtigere Grund: ℝ⁶ ist ein **Vektorraum**. Die CFM-Interpolation
`x_t = (1−t)x₀ + t·x₁` und der MSE auf der Geschwindigkeit sind nur dort gültig.
6D vorhersagen und *danach* auf SO(3) projizieren hält die gesamte bestehende
Flow-Matching-Maschinerie gültig; Quaternionen oder Matrizen direkt vorherzusagen
täte das nicht. Dasselbe gilt für den B-Spline: die Basis wirkt auf die
6D-Kontrollwerte, projiziert wird pro Kurvenpunkt.

**Das ist ein bewusster Kompromiss:** das resultierende R(t) ist glatt im
umgebenden 6D-Raum, nicht eine Geodäte konstanter Geschwindigkeit auf SO(3).
Falls intrinsische Glattheit relevant wird, ist der saubere Weg ein kumulativer
B-Spline auf der Lie-Gruppe (Sommer et al. 2020) — eine deutlich größere
Änderung, hier bewusst nicht gegangen.

Der Kopf ist **null-initialisiert plus Identitäts-Offset**: ein frisch gebautes
Modell sagt exakt die Identitätsrotation vorher, dieselbe Logik wie bei den
null-initialisierten FiLM-Projektionen.

### Stufe 2 — die Orientierungsterme

| Term | Form | Gewicht |
|---|---|---|
| Zeigen | `Σ (1 − ⟨R·e_z, d⟩)` | `W_POINT = 0.1` |
| Standoff | Hinge-Band um den Sollabstand | `W_STANDOFF = 300` |
| Winkelglattheit | geodätische zweite Differenz auf SO(3) | `W_ANGSMOOTH = 2.0` |

**Diese drei Gewichte stammen nicht aus dem Solver.** `W_ERGODIC`, `W_SMOOTH`
und `W_BOUNDARY` sind bewusst wörtlich übernommen, damit Ergebnisse vergleichbar
bleiben; diese drei sind neu und aus Größenordnungen hergeleitet:

- Zeigen liegt pro Punkt in [0, 2], über T=100 also O(10) → mit 0,1 ein Beitrag O(1), passend zu einem ergodischen Term von 1–5.
- Standoff: bei Anfangsabweichung 0,09 ergibt `300 · 0,5 · 0,09² · 100 ≈ 121` — **anfangs dominant**. Das ist gewollt (die Bahn muss erst von der Fläche abheben) und verschwindet bei Konvergenz, weil der Term dann null ist. Wenn dir die frühe Dominanz zu stark ist: `--w_standoff 50`.

Der Runner loggt jeden Term getrennt nach W&B, damit das Verhältnis sichtbar
bleibt.

### `--ergodic_on footprint` — der Punkt, an dem Orientierung zählt

Bei aktivem Standoff ist der Roboter absichtlich **nicht** auf der Fläche. Die
Abdeckung an seiner eigenen Position zu messen, belohnt dann das Falsche.
`footprint` misst dort, wo der Sensorstrahl landet (`p + dist(p)·a`) — damit
hängt die Abdeckung von der Blickrichtung ab, und Orientierung wird Teil der
Aufgabe statt Dekoration. Das ist der Default, sobald `--orientation` gesetzt
ist.

### Neue Metriken

`evaluate_models.py` gibt zusätzlich aus: mittlerer Zeigefehler in Grad, Anteil
des Pfades innerhalb 30° Einfallswinkel, Standoff-Abweichung und rotatorische
Weglänge. Mit `--orientation_ref` bekommt auch die Solver-Referenzzeile
Stufe-0-Frames, sodass der Vergleich like-for-like ist: die Solver-Bahn mit der
bestmöglich *abgeleiteten* Orientierung gegen ein Modell, das eine *lernt*.

---

## Dateien

| Datei | Rolle |
|---|---|
| `obstacles.py` | Kugelhindernis, Abstoßungsgradient, Politur mit 3D-Symmetriebruch |
| `ergodic_energy_torch.py` | Solver-Energie, 3D-Fourier-Basis, Planarität, Diversität |
| `ergodic_metric.py` | Ergodischer Zusatzterm für den CFM-Verlust |
| `orientation.py` | 6D-Darstellung, RMF/Look-at-Frames, Oberflächenfeld (Stufe 0/1) |
| `orientation_energy.py` | Zeige-, Standoff-, Winkelglattheitsterm, `SE3Energy` (Stufe 2) |
| `flow_matching_cond_particles_crossattn.py` | CFM-Netz, 3D-Fourier-Features, Orientierungskopf |
| `flow_matching_particles_selfsupervised.py` | Einpass-Generator, optional mit Orientierung |
| `data_3d.py` | DB-Zugriff, 2D→3D-Anhebung, Volumenbau, Partikel, Augmentierung |
| `viz_3d.py` | 3D-Panels mit Bodenprojektion und Sensorachsen-Pfeilen |
| `flow_matching_runner_particles.py` | Überwachtes Training |
| `flow_matching_runner_particles_selfsupervised.py` | Selbstüberwachtes Training |
| `model_zoo.py` | Checkpoint-Laden mit Dimensionsprüfung |
| `evaluate_models.py` | Metriken inkl. Planarität und Orientierung |
| `test_3d_port.py` | 41 Sanity-Prüfungen |

---

## Bekannte Einschränkungen

**Die Dimension der DB-Blobs ist nicht getaggt.** `_looks_3d` rät anhand der
Länge: durch 3 teilbar und nicht durch 2. Ein Blob der Länge 6·n passt auf beide
Lesarten; in dem Fall gewinnt die 2D-Lesart, was für die jetzige Datenbank richtig
ist. Ein echter 3D-Datensatz sollte eine explizite Spalte mitführen.

**Es gibt keine 3D-Ground-Truth.** Der überwachte Runner imitiert die in die
Ebene gehobenen 2D-Bahnen. Für echtes 3D-Training müsste der SVGD-Solver auf
3D-Zielen neu laufen. Der selbstüberwachte Runner braucht das nicht — seine
Energie ist analytisch in jeder Dimension definiert. Das ist der stärkste
praktische Vorteil dieses Zweigs und der Grund, ihn hier zuerst zu fahren.

**Die Solver-Gewichte sind ungeprüft für 3D.** `W_ERGODIC=600`, `W_SMOOTH=15`,
`W_BOUNDARY=30` wurden aus dem 2D-Solver wörtlich übernommen, damit ein Lauf auf
planaren Daten vergleichbar bleibt. Ob dieselbe Balance auch bei echten
3D-Zielen sinnvoll ist, ist offen — `W_SMOOTH` ist auf die Punktabstände von
T=100 kalibriert, und die ändern sich, wenn Bahnen durch ein Volumen statt durch
eine Fläche laufen.

**Die Orientierung ist ambient-glatt, nicht geodätisch.** Siehe den Abschnitt zu
Stufe 1: die B-Spline-Interpolation läuft im 6D-Raum, die Projektion auf SO(3)
danach. Für große Rotationen zwischen benachbarten Kontrollpunkten ist das
merklich anders als eine Geodäte. Bei den erzeugten Bahnen sind die Sprünge
klein, aber die Grenze sollte man kennen.

**Der Standoff-Term dominiert früh.** Bei Start auf der Fläche ist er rund
zwanzigmal größer als der ergodische Term. Das wirkt wie ein Curriculum (erst
abheben, dann abdecken) und verschwindet bei Konvergenz — aber falls die
Abdeckung darunter leidet, ist `--w_standoff` der erste Regler.
