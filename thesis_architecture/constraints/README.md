# Inference-Time Constraints

Fünf Constraints, die dem **bereits trainierten** Flow-Matching-Modell zur
Inferenzzeit auferlegt werden. Das Modell wird nie nachtrainiert und sieht den
Constraint nie — exakt das Muster, das `obstacles.py` schon für die
Hindernisvermeidung benutzt.

## Das gemeinsame Rezept

Jeder Constraint liefert einen Gradienten `dE/d(Kontrollpunkte)`, der

1. **während der ODE** mit quadratischer Rampe von der Geschwindigkeit
   abgezogen wird (bei kleinem `t` ist der Zustand noch Rauschen, ihn dort zu
   beschränken verzerrt nur den Flow), und
2. **danach im Polish** in reinem Gradientenabstieg die Restverletzung auf
   ~0 drückt.

Beides steckt in `common.guided_generate` / `common.polish`.

## Versuchsaufbau: ein Modell, ein Modus, beide Arme

Beide verglichenen Kurven stammen aus **demselben Checkpoint im selben Modus** —
gleicher Seed, gleiche Konditionierungs-Partikel, gleiches `cfg_weight`, gleiche
ODE-Schritte. Der einzige Unterschied ist der addierte Kraft-Term.

Wichtig für die Einordnung: Der verwendete Checkpoint ist
**längenkonditioniert** (`length_cond=True`), aber `common.guided_generate`
übergibt `length` nie. Das Modell erhält also durchgehend seinen
`null_length_token`, sein gelernter Längenkanal ist in allen Läufen
abgeschaltet. Für die Constraints 1–3 und 5 ist das ohne Belang; für
Constraint 4 heißt es, dass hier die *Kraft* gegen *keine Vorgabe* gemessen
wird — **nicht** gegen die gelernte Längenkonditionierung. Dieser dritte Arm
wäre der eigentlich interessante Vergleich und ist bislang nicht gelaufen.

## Zwei Gradienten-Routen — die Wahl ist nicht kosmetisch

| Route | Wann | Umsetzung |
|---|---|---|
| **Punktweise** | Die Penalty zerfällt über einzelne Kurvenpunkte (Hindernis, Keep-in, Flächen-Attraktion) | `obstacles.curve_repulsion_grad`: die Kettenregel durch die lineare Basis kollabiert zu `B.T @ grad`, kein Autograd nötig |
| **Gekoppelt** | Die Penalty verbindet Nachbarpunkte (Krümmung, Bogenlänge, Tangentenausrichtung) | `common.curve_energy_grad`: Autograd durch dieselbe Basisabbildung |

Die punktweise Abkürzung ist für gekoppelte Penalties schlicht falsch.

## Die fünf Constraints

| # | Ordner | Constraint | Route |
|---|---|---|---|
| 1 | `01_keepin_workspace/` | Keep-in-Region (Arbeitsraum/Korridor) — Spiegelbild der Hindernisvermeidung | punktweise |
| 2 | `02_waypoint_anchor/` | Waypoint-Pinning (Start, Via, Ende) | gekoppelt |
| 3 | `03_max_curvature/` | Maximalkrümmung (Mindest-Wendekreis) | gekoppelt |
| 4 | `04_path_length/` | Ziel-Bogenlänge | gekoppelt |
| 5 | `05_se3_alignment/` | SE(3)-Tangentenausrichtung auf gelifteter 3D-Fläche | gekoppelt |

## Ausführen

```bash
python run_all.py                      # alle Constraints, volle Holdout-Menge
python run_all.py --shapes A digit_5    # schneller Smoke-Test
python 03_max_curvature/run_curvature.py --quantile 0.7   # einzeln, mit Optionen
```

Jeder Runner schreibt in sein `results/`: eine Panel-Grid-Abbildung über alle
Holdout-Formen und eine `*_metrics.csv`. `run_all.py` aggregiert zusätzlich
nach `results/summary.csv`.

## Warum immer *paarweise* Metriken

Jede Zeile berichtet neben der Constraint-eigenen Verletzung auch die
Solver-Metriken (`E_ergodic`, `coverage`, Länge) **vorher und nachher**. Ein
Constraint, der sich selbst erfüllt, indem er die Abdeckung zerstört, ist ein
gescheiterter Constraint — und nur das Zahlenpaar zeigt das. Die Regionen,
Pins, Schranken und Ziellängen werden dabei **pro Form** aus der jeweils
ungeführten Kurve bzw. dem Dichte-Träger kalibriert, damit die Aufgabe für
jede Form gleich schwer ist.

## Gemessene Ergebnisse (Holdout n=25, Seed 0)

Der Constraint wird pro Form kalibriert; `coverage` ist eine Distanz, kleiner
ist besser.

| Constraint | Eigene Metrik | vorher → nachher | E_ergodic | coverage |
|---|---|---|---|---|
| 1 Keep-in **Box** | max Austritt | 0.144 → **0.000** | 2.54 → 3.55 | 0.0411 → 0.0536 |
| 1 Keep-in **Kreis** | max Austritt | 0.225 → **0.000** | 2.54 → 5.22 | 0.0411 → 0.0643 |
| 2 Waypoints | max Pin-Fehler | 0.517 → **0.000** | 2.54 → 3.00 | 0.0411 → 0.0406 |
| 3 Krümmung | κ_peak / κ_p99 | 3166 → **25.5** / 483 → **24.7** | 2.54 → 2.93 | 0.0411 → 0.0414 |
| 4 Länge 0.7× | \|rel. Fehler\| | 0.429 → **0.007** | 2.54 → 2.65 | 0.0411 → 0.0436 |
| 4 Länge 1.3× | \|rel. Fehler\| | 0.231 → **0.032** | 2.54 → 2.48 | 0.0411 → 0.0400 |
| 5 SE(3) (n=75) | \|cos(t,n)\| | 0.030 → **0.016** | — | 0.0406 → 0.0405 |

Ablesbar daraus:

* **Waypoints und Krümmung sind fast gratis.** Beide erfüllen ihre Vorgabe
  hart, ohne die Abdeckung messbar zu verschlechtern.
* **Die Keep-in-Region ist der teuerste Constraint**, und der Kreis kostet
  doppelt so viel wie die Box (E_ergodic mehr als verdoppelt). Der Kreisradius
  ist die größte Halbausdehnung der freien Kurve, was die Ecken hart
  beschneidet — die Zieldichte liegt dort aber weiterhin.
* **Verlängern ist schwerer als Verkürzen** (rel. Fehler 3.2 % gegen 0.7 %):
  der Spline lässt sich leichter zusammenziehen als aufspannen.
* **Constraint 5 verbessert die Tangentialität bei 25/25 Formen auf allen drei
  Körpern** (Kugel −40 %, Pyramide −49 %, Torus −43 %) und kostet keine
  Footprint-Abdeckung. Der Preis steht woanders: `max |SDF|` verschlechtert
  sich (Kugel 0.0020 → 0.0050, Torus 0.0018 → 0.0079) — Positions- und
  Richtungsterm konkurrieren. Die verbleibenden Verletzungen sammeln sich
  sichtbar auf den **Pyramidenkanten**, also genau dort, wo das Normalenfeld
  springt.

### Kraft gegen gelernte Konditionierung (Constraint 4, dritter Arm)

`04_path_length/run_compare_conditioning.py` holt den Vergleich nach, den die
Haupttabelle **nicht** enthält: dieselbe Ziellänge einmal per Inferenz-Kraft
und einmal über den trainierten FiLM-Längenkanal des Checkpoints. Der Ladder
ist absolut statt relativ, weil die Einbettung als
`u = (log1p(L) − log1p(log_ref)) / log_scale` normiert und `log_ref = 11.05`
ist — das Trainingszentrum liegt also bei L ≈ 11, während das Modell
unkonditioniert nur L ≈ 6.3 erzeugt.

Ergebnis über 150 Kombinationen (25 Formen × 6 Ziellängen):

| angeforderte Länge | 4 | 6 | 8 | 11 | 14 | 18 |
|---|---|---|---|---|---|---|
| erreicht, **Konditionierung** | 6.37 | 6.34 | 6.38 | 6.33 | 6.36 | 6.32 |
| erreicht, **Kraft** | 4.02 | 6.02 | 7.86 | 9.85 | 10.82 | 11.17 |

**Die Konditionierung hat null Längenautorität.** Über einen 4.5-fachen
Anforderungsbereich schwankt die erreichte Länge um 0.06; das Modell gibt
unabhängig von der Vorgabe die natürliche Länge der jeweiligen Form aus. Die
Kraft folgt der Vorgabe bis ca. 8 nahezu exakt und sättigt dann bei ~11 — der
Spline mit 64 Kontrollpunkten nimmt nicht mehr Länge auf.

Das liegt nicht an einem zu leisen Signal. Im Sampler verwirft `cond_drop_mask`
nur die *Dichte*-Konditionierung, beide CFG-Zweige sehen dieselbe Länge, sie
kürzt sich in der Differenz weitgehend weg — dafür existiert der separate
`length_cfg_weight`. Aufdrehen hilft aber nicht: bei `length_cfg=6` wird die
Antwort gegenläufig (Ziel 4 → L 6.94, Ziel 18 → L 5.17) und die Coverage
verschlechtert sich von 0.0415 auf 0.0705.

Offener Verdacht, noch nicht geprüft: `log_ref = 11.05` gegen tatsächlich
erzeugte 6.3 ist rund Faktor 2. Das könnte ein **Definitions-Mismatch der
Länge** sein — Trainingslabels auf der rohen Wegpunkt-Polylinie gegen die hier
gemessene Bogenlänge der 256-Punkt-B-Spline. Nachprüfbar, indem man die
Längenverteilung in `ergodic_dataset_length.db` direkt gegen die gerenderte
Bahnlänge derselben Einträge hält.

### Eine Metrik, die *nicht* funktioniert

`anteil_ueber` (Anteil der Kurvenpunkte über κ_max) taugt bei Constraint 3
nicht als Erfolgsmaß: Da κ_max als 80 %-Quantil der ungeführten Kurve
definiert ist, startet dieser Anteil bei **jeder** Form tautologisch bei
exakt 0.201, und das Erfüllen der Schranke drückt die Ausreißer *auf* den
Schwellwert, wo sie ihn weiter umspielen. Ein Lauf kann κ_peak von 26773 auf
40 zusammenstauchen, ohne dass sich dieser Anteil bewegt. Berichtet werden
deshalb κ_peak und κ_p99.

## Zwei numerische Fallen (beide teuer gelernt)

* **Krümmung ist nicht skaleninvariant.** Eine reine Krümmungs-Penalty ist nach
  unten unbeschränkt: Kurve aufblähen senkt κ = 1/Radius überall. Ungebremst
  hat der Abstieg genau das getan und eine Bahn der Länge 4.55 im Einheits-
  quadrat auf Länge 240 gestreckt. `MaxCurvature` braucht deshalb zwingend den
  einseitigen Längen-Guard.
* **Force-Clipping ist Pflicht, nicht Feintuning.** Der rohe Krümmungs-Gradient
  erreicht Spitzen um 3e2, während die Kontrollpunkte auf Skala ~1 leben — mit
  festem `dt` divergiert explizites Euler, statt zu steuern. `max_force` in
  `guided_generate`/`polish` kappt das.
