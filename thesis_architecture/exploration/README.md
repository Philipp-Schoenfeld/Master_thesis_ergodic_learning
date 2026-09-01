# Exploration: einen Pfad planen, dessen Ziel man noch nicht kennt

Fünf Varianten (A–E) für die Frage: **Wie deckt man eine Informationsdichte ab,
die man erst erkunden muss?**

Alle fünf teilen sich einen Unterbau in `common/` und unterscheiden sich nur im
Planungsansatz. Sie laufen ohne trainiertes Netz — dann optimiert ein
Gradientenplaner die Kontrollpunkte direkt — und lassen sich mit `--checkpoint`
auf den amortisierten Generator umstellen, ohne dass sich sonst etwas ändert.

## Die Kernidee

Die Zielverteilung ist keine Eingabe mehr, sondern wird aus dem aktuellen
Wissensstand abgeleitet:

    Φ(x) = μ(x) + κ · σ(x)

μ ist der Posterior-Mittelwert eines Gauß-Prozesses über dem unbekannten Feld,
σ seine Unsicherheit. Unerkundete Gebiete haben hohes σ, ziehen also Masse an
und werden angefahren; nach der Messung fällt σ dort und die Masse wandert
weiter. **Erkundung entsteht als Abdeckung eines sich verändernden Ziels** — es
braucht keinen Umschalter zwischen „erkunden" und „abdecken".

Das ist die UCB-Akquisefunktion aus GP-UCB (Srinivas et al.), und genau diese
Funktion benutzt KL-E³ (Abraham et al.) als ergodische Zielverteilung.

### Warum *eine* kombinierte Dichte und nicht zwei Verluste

Der naheliegende Entwurf — ein ergodischer Verlust auf das Bekannte, ein zweiter
auf die Unsicherheit, beide addiert — ist schlecht gestellt. Ergodizität heißt,
dass die Aufenthaltsverteilung *einer* Zielverteilung gleicht. Zwei gleichzeitig
sind im Allgemeinen unerfüllbar; die Bahn findet einen Kompromiss, der keine von
beiden trifft, und das Gewichtsverhältnis entscheidet willkürlich, welche
stärker verfehlt wird. Werden die Dichten dagegen *vor* dem Verlust verrechnet,
ist es wieder ein wohlgestelltes Problem, und κ ist ein interpretierbarer
Parameter statt eines Verlustgewicht-Verhältnisses.

## Die fünf Varianten

| | Idee | Planungen/Mission | Kernannahme |
|---|---|---|---|
| **A** | Φ = μ + κσ, einmal planen | 1 | Unsicherheit ist statisch |
| **B** | differenzierbare Vorausschau statt Belohnung | 1 | Varianz hängt nur von Messorten ab |
| **C** | Receding Horizon mit Abdeckungsgedächtnis | n_segments | Glaube wächst zwischen Abschnitten |
| **D** | glaubenskonditioniert, lang planen kurz fahren | n_rounds | Glaube trägt die Historie |
| **E** | erst σ, dann μ (Baseline) | 2 | Phasen sind trennbar |

**A** ist der Referenzpunkt. Seine Grenze: κ ist fest, und die Unsicherheitskarte
ist zum Planungszeitpunkt eingefroren. Beim Abfahren fällt sie aber dort, wo die
Bahn gerade war — der Planer zählt also doppelt.

**B** ist die einzige Variante, in der die Bahn dafür optimiert wird, was sie
*lernen* wird. Möglich durch eine GP-Eigenschaft: **die Posterior-Varianz hängt
nur von den Messorten ab, nicht von den gemessenen Werten.** Man kann also exakt
ausrechnen, wie viel Unsicherheit eine geplante Bahn auflösen wird, bevor man sie
abfährt — und das ist differenzierbar in der Bahn. Damit braucht es kein
Reinforcement Learning: statt eines terminalen Skalars mit hoher Varianz gibt es
einen exakten Gradienten.

**C** behebt A's Konstruktionsfehler durch Neuplanen mit fortgeschriebenem
Glauben, und führt die bereits abgefahrene Bahn in der Abdeckungsrechnung mit.

**D** unterscheidet sich von C darin, dass jede Runde eine *vollständige* Bahn
neu erzeugt und nur deren Anfang abgefahren wird. Kein Abdeckungsgedächtnis —
die Information über Gesehenes steckt vollständig im Glauben. **Für die
Thesis-Architektur die interessanteste Variante: der Eingang des Netzes ändert
sich gar nicht**, es konditioniert ohnehin auf Partikelwolken.

**E** ist die Baseline, die eine Verteidigung verlangt. Sie fährt zwei volle
Trajektorien und muss die anderen deshalb deutlich schlagen, nicht knapp.

## Benutzung

```bash
# Einzelne Variante
python variant_a_combined/run_a.py --shapes 4 --kappa 0 2 6
python variant_b_diffsim/run_b.py  --shapes 3 --lambda_cov 0 20000
python variant_c_receding/run_c.py --shapes 3 --segments 4 --ablate_history
python variant_d_belief_cond/run_d.py --shapes 3 --rounds 4
python variant_e_two_stage/run_e.py --shapes 4

# Alle fünf im direkten Vergleich
python run_all.py --shapes 4 --out results/vergleich.csv

# Mit trainiertem Netz statt Gradientenplaner
python run_all.py --shapes 4 --checkpoint ../checkpoints/<datei>.pt

python test_exploration.py
```

## Metriken

Eine Mission ist nur gut, wenn **alle drei** stimmen — ein einzelner Skalar
verdeckt genau den Zielkonflikt:

- `coverage` — Abdeckung der **wahren** Dichte, die der Planer nie gesehen hat
- `info_gain` — abgebaute Gesamtunsicherheit
- `belief_rmse` — stimmt das Gelernte auch?

Dazu zwei Kostenspalten: `path_len` (E fährt doppelt so weit) und `plan_s` (C
und D planen mehrfach pro Mission — **das ist die Größe, gegen die ein
amortisierter Generator antritt**).

## Offene Punkte und Fallstricke

**Der flache Prior ist entartet.** Ohne Vormessungen ist μ = 0 und σ überall
gleich, also ist Φ für *jedes* κ gleichverteilt. Erkundung ist in diesem Zustand
nicht definiert, weil kein Ort informativer ist als ein anderer. `--n_prior > 0`
verwenden; `is_degenerate()` meldet den Fall, statt ihn stumm passieren zu
lassen.

**Die Gedächtnis-Ablation in C ist nicht längenkontrolliert.** Ohne Historie
fällt die Abdeckung *besser* aus, bei rund 40 % längerem Pfad — jeder Abschnitt
versucht dann die ganze Verteilung zu treffen und fährt weit aus. Der Vergleich
sagt über das Gedächtnis für sich genommen also nichts. Sauber wird er erst mit
einer Längenstrafe oder Geschwindigkeitsgrenze pro Abschnitt.

**Skalen klaffen auseinander.** In Variante B liegt der Unsicherheitsterm bei
mehreren hundert, der ergodische Fehler bei ~1e-2. Ohne ein `lambda_cov` in der
Größenordnung 1e4 ist die Abdeckung im Gesamtziel praktisch nicht vertreten.

**Die Vorausschau muss den Messprozess abbilden.** Ein erster Lauf war um ~40 %
zu pessimistisch, weil sie nur die Bahnpunkte modellierte, gemessen aber ein
Sensorring um jeden Punkt wird. Mit Ring liegt die Abweichung bei ~16 %. Ein
Test hält das fest.

**Die partielle Beobachtbarkeit ist konstruiert, nicht gegeben.** Die Datenbank
enthält die fertige Dichte; `common/observation.py` macht daraus einen
Messprozess. Wie realistisch das Ganze ist, entscheidet sich dort mehr als an
jeder Modellwahl — deshalb stehen die Annahmen explizit in dieser Datei und
nicht verstreut in den Runnern.

## Aufbau

```
common/
  belief.py       GP-Posterior, uncertainty_after() für Variante B
  observation.py  Sensormodell: was eine Bahn über das wahre Feld verrät
  acquisition.py  Φ = μ + κσ, Partikelziehung, κ-Zeitplan
  metrics.py      Bewertung gegen die Wahrheit
  data.py         wahre Dichten aus ergodic_dataset_775.db, Ausgangsglaube
  planner.py      GradientPlanner (ohne Netz) | ModelPlanner (amortisiert)
```

Der Wechsel zwischen den beiden Planern ist eine Zeile — dieselbe Mission,
derselbe Glaube, derselbe Messprozess. Genau dort wird messbar, was
Amortisierung bringt.

## MuJoCo-Roboterarm (`mujoco_sim/`)

Ein Franka Panda mit Radiergummi am Endeffektor fährt vor einem aufrecht
stehenden Writeboard. Zwei Betriebsarten:

```bash
# Standalone: die gespeicherte ergodische Bahn einer Form abfahren
python -m mujoco_sim.run_mujoco --shape A
python -m mujoco_sim.run_mujoco --shape A --erase --speed 0.5
python -m mujoco_sim.run_mujoco --list-shapes

# Live, von der 2D-Schaltzentrale getrieben (drei Prozesse)
python main.py
```

Die Bahn kommt aus `3D_ergodic_learning/ergodic_dataset_robot.db`, die
Zieldichte aus `ergodic_dataset_775.db` (in reinem NumPy gerastert, kein JAX
nötig). Beim ersten Aufruf einer Form wird die Gelenkbahn geplant (~20 s) und
in `mujoco_sim/cache/` abgelegt; danach startet sie sofort.

```
mujoco_sim/
  board.py       Board-Geometrie (eine Quelle der Wahrheit), (u,v)<->Welt,
                 Laden von Bahn und Zieldichte
  ik.py          Aufgabenprioritäts-IK für die Radiergummi-Spitze
  run_mujoco.py  Szene, Playback, Live-Modus, CLI
```

### Warum die IK so aussieht

Der Radiergummi ist ein Zylinder — die Drehung um seine eigene Achse ist
bedeutungslos. Die IK stellt deshalb eine strenge Prioritätenkette auf die
sieben Armgelenke:

1. **Position** der Spitze (hart, 3 DOF)
2. **Werkzeugachse** gegen die Brettnormale (im Nullraum von 1)
3. **Abstand zu den Gelenkgrenzen** (im verbleibenden Nullraum)

Zwei Punkte entscheiden darüber, ob das überhaupt konvergiert:

- Der Orientierungsfehler muss im **Welt**-Frame stehen, weil die Rotationszeilen
  von `mj_jacSite` Welt-Frame sind (`mju_subQuat` liefert ihn im lokalen Frame).
  Diese Verwechslung war die Ursache dafür, dass das Handgelenk bei ~180° hängen
  blieb und ein Zufalls-Jitter als Notausstieg nötig war.
- Gesättigte Gelenke müssen **gesperrt und der Schritt neu gelöst** werden. Ein
  blosses `clip(q + dq)` verfälscht die Schrittrichtung und lässt die Lösung
  mehrere Millimeter neben einem gut erreichbaren Ziel stehenbleiben.

Zufalls-Startwerte landen auf sehr verschiedenen IK-Zweigen, von denen nur
manche das ganze Brett bedienen können. Der Planer siebt deshalb alle Startwerte
auf einer ausgedünnten Bahn vor und verfolgt nur die aussichtsreichsten in voller
Auflösung; gewertet wird zuerst „erreicht die Bahn", dann flache Werkzeuglage,
dann Gelenkglätte.

### Warum ein Vorhalt auf den Sollwert

Die Panda-Aktuatoren sind PD-Stellglieder (`gainprm=kp`, `biasprm=[0,-kp,-kd]`),
ein Gelenk hinkt seinem Kommando also um `kd/kp · q̇` hinterher — hier 100 ms.
Da die Gelenkbahn vorab geplant ist, wird einfach `q_soll(t + kd/kp)` kommandiert.
Das senkt den Nachlauf der Spitze auf der 'A'-Bahn von 28 mm auf 6 mm; im
Live-Modus (Vorhalt aus der Cursor-Geschwindigkeit) von 16 mm auf 2 mm.

### Agententempo in der Schaltzentrale

Der Agent im 2D-Panel und der Arm in MuJoCo laufen jetzt mit **derselben
physikalischen Geschwindigkeit**. Der Tempo-Regler heisst deshalb `Speed cm/s`
und meint Zentimeter pro Sekunde auf dem Brett; sein Maximum ist
`MAX_TIP_SPEED` aus `mujoco_sim/board.py` — das Tempo, das der Arm noch sauber
abfaehrt. Beide Seiten lesen dieselbe Konstante.

Vorher wurde die Bahn nach *Index* getaktet
(`step = len(seg) / (14 * speed)` Stuetzpunkte pro 55-ms-Tick). Das hatte zwei
Fehler:

- Das Tempo hing an der Laenge der geplanten Runde, nicht an der Strecke. Eine
  lange und eine kurze Bahn wurden in gleich vielen Ticks abgefahren; auf der
  'A'-Bahn (301 cm) kam der Agent auf ~90 cm/s, also das Dreifache dessen, was
  der Arm halten kann.
- Der Regler war invertiert: `speed` steht im *Nenner*, ein hoeherer Wert machte
  den Agenten langsamer.

Jetzt wird nach Bogenlaenge getaktet, mit einem Fliesskomma-Wegzaehler
(`play_arc`). Der Zaehler ist noetig, weil ein erzwungener Mindestschritt von
einem Stuetzpunkt pro Tick eine grob abgetastete Bahn wieder zu schnell laufen
liesse — bei 120 Punkten waeren aus 20 cm/s rund 32 cm/s geworden.

Gemessen auf der 'A'-Bahn, Abstand Arm zu Agent:

| Taktung | Tempo | mittlerer Abstand | max |
|---|---|---|---|
| alt, Regler 3 | 90 cm/s | 44 mm | 143 mm |
| alt, Regler 8 | 55 cm/s | 87 mm | 259 mm |
| neu, 20 cm/s | 20 cm/s | 11 mm | 44 mm |
| neu, 30 cm/s | 30 cm/s | 22 mm | 54 mm |

Dazu wird das Gelenkkommando im Live-Modus auf die Panda-Datenblattgrenzen
(`JOINT_VMAX`) begrenzt. Die Online-IK muss gelegentlich auf einen anderen
Loesungszweig wechseln — am zuverlaessigsten unten in der Brettmitte — und
kommandierte diesen Sprung sonst als einzelnen ~46 rad/s-Schritt, dem der Arm
nicht folgen kann. Mit der Begrenzung sinkt der schlimmste Ausreisser von 95 mm
auf 26 mm, und die Spitze bleibt durchgehend hinter der Brettflaeche.
