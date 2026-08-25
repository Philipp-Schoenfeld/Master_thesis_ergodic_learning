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
