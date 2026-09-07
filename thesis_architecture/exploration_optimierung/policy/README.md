# policy — gelernte Regler fuer die Laengeneinheit-Mission

Die Rastersuche in `exploration_optimierung/` hat **eine feste** Einstellung
gesucht und gefunden: `niveau`, `tau = 0,6067`, 25 SVGD-Iterationen, `n = 6`,
J = 0,258 ueber die volle Holdout-Menge. Fest heisst: dieselbe Einstellung in
jeder Runde und fuer jede Form.

Dieses Paket beantwortet die Anschlussfrage: **bringt eine Einstellung, die
sich nach dem Glaubenszustand richtet, mehr?** Zwei Wege, beide umgesetzt:

| | Option A — `train.py` | Option B — `ppo.py` |
|---|---|---|
| Lernverfahren | ueberwacht (Wertmodell ueber Kandidaten) | PPO im Missionsloop |
| Ziel | die kurzsichtig beste Runde treffen | die Rueckkehr der ganzen Episode |
| Trainingsdaten | Orakel-Entscheidungen (`oracle.py`) | eigene Rollouts, gestartet aus A's Daten (Verhaltensklonen) |
| Kosten | Minuten (Datensatz liegt vor) | Stunden (jede Iteration faehrt echte Missionen) |
| Kann finden | bessere Einstellung *jetzt* | Strategien ueber mehrere Runden (erst erkunden, dann ausbeuten) |

## Der gemeinsame Unterbau

Alle Regler fahren **denselben** Simulator. `mission.LaengenMission` hat dafuer
einen Hook bekommen:

```python
m = LaengenMission(planner, truths, names, args, policy=regler)
# regler(r, zustaende) -> [(phi_modell, param, svgd_iters), ...]  je Form
```

`policy=None` faehrt exakt den bisherigen Weg (gleiche Reihenfolge der
Zufallszahlen), die Zahlen der bereits gerechneten Studien bleiben also
gueltig. Ein `Zustand` (siehe `features.py`) enthaelt GP-Mittelwert,
GP-Unsicherheit, Besuchsdichte, gefahrene Bahn und Rundennummer — **nicht** die
wahre Dichte. Das ist die zentrale Trennung des ganzen Pakets: der Regler sieht
nur, was ein echter Roboter auch saehe.

## Ablauf

```bash
# 1. Orakel: Obergrenze + Trainingsdatensatz (lokal, ~3 h auf einer RTX 2070 S)
python -m exploration_optimierung.policy.oracle \
    --n_shapes 25 --n_max 10 --seeds 2 --param_punkte 4 --svgd_buckets 0 25 100

# 2. Option A: Wertmodell, formweise kreuzvalidiert (Minuten)
python -m exploration_optimierung.policy.train --folds 5

# 3. Option B: Verhaltensklonen + PPO (Stunden; --nur_bc fuer den Zwischenstand)
python -m exploration_optimierung.policy.ppo --iterationen 40 --n_envs 8

# 4. Auswertung: alle Regler, volle Holdout-Menge, Bilder und Metriken
python -m exploration_optimierung.policy.evaluate --seeds 2 --mit_orakel
```

Schnelldurchlauf zum Pruefen der Kette (Minuten, keine belastbaren Zahlen):

```bash
python -m exploration_optimierung.policy.oracle --n_shapes 3 --n_max 3 --seeds 1 --schnell
python -m exploration_optimierung.policy.train --folds 3 --epochen 50
python -m exploration_optimierung.policy.ppo --nur_bc --bc_epochen 50
python -m exploration_optimierung.policy.evaluate --n_shapes 3 --n_max 3
```

## Was das Orakel ist — und was nicht

`oracle.py` probiert je Runde **jeden** Kandidaten wirklich aus und faehrt den
besten. Das ist kein einsetzbares Verfahren (K-mal so teuer wie eine Mission),
sondern zweierlei:

1. Die **Obergrenze** des kurzsichtigen Waehlens. Ohne sie ist "gelernt schlaegt
   fest" nicht einzuordnen — der Abstand koennte fast alles Erreichbare sein
   oder ein Zehntel davon.
2. Der **Trainingsdatensatz**. Jede Kandidatenbewertung ist eine Zeile
   `(Zustand, Aktion) -> Guete`; aus einer Entscheidung fallen K Zeilen ab
   statt einer. Genau daraus lernt Option A, und daraus wird Option B
   vorbereitet.

Ausgewaehlt wird **gegen den Glauben** (`coverage_vs_truth(bahn, mu_hat)`),
nicht gegen die Wahrheit. Nach der Wahrheit auszuwaehlen waere ein Blick auf
das Ergebnis und wuerde beide Zwecke zerstoeren.

Die Zeitspalte des Orakels enthaelt nur den Aufwand des *gefahrenen*
Kandidaten (Gesamtzeit / K); seine echten Suchkosten stehen als `wallclock_s`
in `results/policy_orakel.json`.

## Warum ein Wertmodell (Option A) und keine direkte Aktionsvorhersage

Der beste Kandidat ist oft nicht eindeutig — die J-Landschaft ist flach, und
welcher von mehreren fast gleich guten Kandidaten gerade das Minimum haelt, ist
zu einem guten Teil Rauschen. Eine Klassifikation auf den Argmin lernt dieses
Rauschen mit. Das Wertmodell lernt stattdessen die ganze Bewertungskurve ueber
die Kandidaten und waehlt zur Laufzeit deren Minimum — flache Bereiche bleiben
flach, und aus einer Entscheidung werden K Trainingszeilen.

Gemessen wird deshalb nicht der MSE, sondern das **Bedauern**: `score_norm` des
gewaehlten Kandidaten, also 0 fuer den besten und 1 fuer den schlechtesten der
jeweiligen Entscheidung. Daneben stehen immer drei Vergleichswerte — Orakel
(0,0), Zufall (~0,5) und die feste Betriebseinstellung. Der letzte ist die
Messlatte.

## Aufteilung der Formen — der methodisch heikle Punkt

Die 25 Formen sind bereits die Holdout-Menge des Flow-Matching-Netzes. Wer die
Richtlinie auf denselben 25 Formen trainiert *und* testet, berichtet eine zu
gute Zahl. Zwei Auswege, beide vorgesehen:

* **Kreuzvalidierung ueber Formgruppen** (Voreinstellung, `train.py --folds 5`):
  jede Falte haelt ganze Formen zurueck; die berichteten Zahlen gelten fuer nie
  gesehene Formen. Bequem, aber jede Form liefert nur einmal einen Testwert.
* **Getrennte Splits** (`--split train` in `oracle.py`/`ppo.py`/`evaluate.py`):
  die Richtlinie lernt auf den Trainingsformen des Planernetzes, die 25
  Validierungsformen bleiben ausschliesslich Test. Sauberer; kostet einen
  zweiten Orakel-Lauf. Dass das *Planernetz* die Trainingsformen kennt, stoert
  nicht — gelernt wird die Wahl der Einstellung, nicht die Bahn.

Welcher Weg in der Thesis steht, ist eine Setzung und sollte dort begruendet
werden.

## Belohnung und Beobachtung bei Option B

```
r_t = -(q_t - q_{t-1}) - lambda_len - lambda_time * dt(svgd_iters)
```

Summiert ueber die Episode ist das bis auf eine Konstante genau `-J` aus
`objective.py` — der Agent optimiert die Zielfunktion der Studie selbst und
nicht etwas Aehnliches. Die Belohnung benutzt die wahre Dichte (ueber
`cov_norm`), die **Beobachtung nicht**: der uebliche asymmetrische Aufbau, bei
dem der Simulator im Training mehr weiss als der Agent, der Agent zur Laufzeit
aber ohne dieses Wissen auskommt.

Die Zeitstrafe ist modelliert (`T_PLAN + T_ITER * iters`, angepasst an die
gemessene SVGD-Kurve der Studie) statt gemessen: die Wanduhr haengt an der
Auslastung des Rechners und waere fuer den Agenten reines Belohnungsrauschen.
In der Auswertung wird dagegen echte Zeit gemessen.

Festgelegt sind bewusst: feste Rundenzahl (kein Abbruch als Aktion), dichte
Belohnung je Runde. Das haelt den Vergleich mit A und der festen Einstellung
sauber; eine Abbruch-Aktion beantwortet zusaetzlich die n*-Frage, ist aber
schwerer zu interpretieren und waere ein eigener Schritt.

## Ausgabedateien

| Datei | Inhalt |
|---|---|
| `results/policy_datensatz.csv` | jede Kandidatenbewertung: Zustand, Aktion, Guete |
| `results/policy_orakel.json` | Obergrenze (J, q, n) und was das Orakel je Runde waehlte |
| `results/policy_a_training.json` | Faltenzahlen, Bedauern, Merkmalswichtigkeiten |
| `results/policy_b_training.json` | BC- und PPO-Verlauf |
| `results/policy_metriken.csv` | Auswertung: jede Form, jede Rundenzahl, jeder Regler |
| `results/policy_vergleich.json` | Aggregat je Regler + paarweiser Vergleich gegen fest |
| `results/policy_panel.png` | 25 Formen, alle Regler uebereinander |
| `results/policy_kurven.png` | q(n) und J(n) je Regler |
| `results/policy_aktionen.png` | was die Regler tatsaechlich waehlen |
| `policy/ablage/policy_a.pt`, `policy_b.pt` | die trainierten Regler |

## Auf dem Cluster

`run_job_policy.bash` faehrt die ganze Kette (Orakel, A, B, Auswertung) als ein
SLURM-Job. Wie alle Jobs dieses Projekts wird er **ausschliesslich per
`sbatch`** eingereiht, nie per `srun` gestartet:

```bash
sbatch run_job_policy.bash
```
