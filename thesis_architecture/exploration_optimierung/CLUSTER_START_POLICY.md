# Start auf dem Cluster — gelernte Regler (Option A und B)

Diese Datei ist der ganze Kontext, den du auf dem anderen Rechner brauchst, um
die Kette zu starten. Die Begründungen stehen in
[`policy/README.md`](policy/README.md), die Vorgängerstudie in
[`CLUSTER_START.md`](CLUSTER_START.md).

## Worum es geht

Die Rastersuche hat **eine feste** Betriebseinstellung gefunden: `niveau`,
`tau = 0,6067`, 25 SVGD-Iterationen, `n = 6`, `J = 0,258` über die volle
Holdout-Menge. Fest heißt: dieselbe Einstellung in jeder Runde und für jede
Form. Dieser Lauf beantwortet die Anschlussfrage — **bringt eine Einstellung,
die sich nach dem Glaubenszustand richtet, mehr?**

Vier Stufen, ein Job:

| Stufe | Was | Ergebnis |
|---|---|---|
| 1 · Orakel | probiert je Runde **alle** Kandidaten wirklich aus, fährt den besten | Obergrenze + Trainingsdatensatz (`policy_datensatz.csv`) |
| 2 · Option A | Wertmodell über (Zustand, Aktion), formweise kreuzvalidiert | `policy_a.pt`, Bedauern gegen Orakel/Zufall/fest |
| 3 · Option B | Verhaltensklonen aus Stufe 1, danach PPO im Missionsloop | `policy_b.pt`, Lernkurve |
| 4 · Auswertung | alle Regler über die volle Holdout-Menge | Metrik-CSV, Vergleichs-JSON, drei Abbildungen |

Alle Regler fahren **denselben** Simulator (`mission.LaengenMission` mit dem
neuen `policy=`-Hook) und werden mit **derselben** Zielfunktion `J` gemessen —
Unterschiede sind Unterschiede der Entscheidung, nicht des Aufbaus.

## Voraussetzungen

| Was | Wo | Im Git? |
|---|---|---|
| Code (inkl. `policy/`) | `~/Master_thesis/thesis_architecture/` | ✅ ja |
| Formen-Datenbank `ergodic_dataset_775.db` | `.../ergodic_dataset_generator/` | ✅ ja |
| **Checkpoint `netz2d_startpunkt.pt`** | `~/Master_thesis/transfer/` | ❌ **nein** (`*.pt` ist gitignored) |
| Conda-Umgebung `thesis` | — | — |

Der Checkpoint ist die einzige Lücke — dieselbe wie bei der Vorgängerstudie.
Prüfen, notfalls hochladen:

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
    'ls -la ~/Master_thesis/transfer/netz2d_startpunkt.pt'

rsync -av --progress transfer/netz2d_startpunkt.pt \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/transfer/
```

## Ablauf

### 1. Code holen

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de
cd ~/Master_thesis && git pull
```

### 2. Vorabprüfung auf dem Login-Knoten (30 s, ohne GPU, ohne Netz)

```bash
cd ~/Master_thesis/thesis_architecture
conda activate thesis
python -m exploration_optimierung.policy.test_policy
python -m exploration_optimierung.test_smoke --schnell
```

Erwartet: `Alles in Ordnung.` (71 Prüfungen) und `27 bestanden, 0
fehlgeschlagen`. Der Selbsttest fährt die ganze Kette mit einem Ersatzplaner
durch — er prüft die Mechanik (Zeilenzuordnung, SVGD-Gruppierung,
Modellablage, `Belohnung = 1 − J`), nicht die Güte. Läuft er durch, ist der
Job selbst nur noch eine Frage der Rechenzeit.

### 3. Job einreihen

```bash
cd ~/Master_thesis/thesis_architecture/exploration_optimierung
sbatch run_job_policy.bash
```

**Nie direkt starten** (kein `srun bash run_job_policy.bash`) — ausschließlich
über `sbatch`, wie alle Jobs dieses Projekts.

Das Skript verteilt ein gemeinsames Budget von 11,5 h innerhalb des 12-h-Limits
dynamisch auf die vier Stufen: wird das Orakel früher fertig, bekommt PPO die
Zeit. Jede Stufe bekommt ihr Budget als `--max_minuten` und **hört von selbst
geordnet auf** — sie schreibt, was sie hat, statt vom Zeitlimit abgeschnitten
zu werden. `--signal=SIGTERM@120` wirkt zusätzlich.

Grobe Erwartung (lokal auf einer RTX 2070 SUPER gemessen, Cluster-GPU eher
schneller): Orakel ≈ 2,5–4 h (24 000 geplante Wolken bei 0,38 s je Wolke),
Option A wenige Minuten, PPO füllt den Rest, Auswertung mit Orakelspalte
≈ 1–1,5 h.

### 4. Zusehen

```bash
squeue -u $USER
tail -f policy-<JOBID>.out
```

Fortschrittszeilen sehen so aus:

```
[1/4] Orakel — Budget 330 min  (14:02)
  Seed 0: Runde 3/10  (412.7s/Runde, noch 48.1 min)
  Seed 0: J* = 0.2431 bei n = 8  (q = 0.0912, 68.8 min)
[2/4] Option A — Wertmodell, 5 Falten
  Bedauern  Modell 0.31   Fest 0.44   Zufall 0.50   (Orakel 0,0)   Treffer 22%
[3/4] Option B — Verhaltensklonen + PPO, Budget 240 min
  Iter  12  Rueckkehr +0.7412  J~0.2588  q(n) 0.1104  Entropie 1.832
```

### 5. Ergebnisse zurückholen

**Achtung:** `exploration_optimierung/results/` und `*.pt` sind **gitignored** —
die Ergebnisse kommen *nicht* per `git pull` zurück. Entweder per `rsync`:

```bash
rsync -av \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/results/policy_* \
    thesis_architecture/exploration_optimierung/results/
rsync -av \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/policy/ablage/ \
    thesis_architecture/exploration_optimierung/policy/ablage/
```

oder auf dem Cluster bewusst gegen die Ignore-Regel committen (so ist die
Vorgängerstudie in `origin/main` gelandet):

```bash
git add -f exploration_optimierung/results/policy_*
git commit -m "Ergebnisse der gelernten Regler" && git push
```

### 6. Was man anschaut

| Datei | Frage, die sie beantwortet |
|---|---|
| `results/policy_vergleich.json` | **die Kernzahl**: `J*` je Regler und der paarweise Vergleich gegen fest (Mittel, Median, auf wie vielen der 25 Formen besser) |
| `results/policy_panel.png` | Sind gute Zahlen auch gute Bahnen? Alle 25 Formen, alle Regler übereinander |
| `results/policy_kurven.png` | `q(n)` und `J(n)` je Regler — wo liegt das jeweilige `n*` |
| `results/policy_aktionen.png` | **Worin besteht die gelernte Regel?** Wählt der Regler je nach Runde verschieden, oder immer dasselbe (dann ist er keine kontextabhängige Richtlinie, egal wie gut die Zahl ist) |
| `results/policy_a_training.json` | Bedauern je Falte + Merkmalswichtigkeiten (welche Zustandsgröße die Wahl trägt) |
| `results/policy_b_training.json` | PPO-Lernkurve — steigt die Rückkehr überhaupt über den BC-Startpunkt? |

**Zwei Orakelzahlen, nicht eine.** `policy_orakel.json` ist der eigenständige
Rollout: er committet genau die Bahn, die er geprobt hat — die schärfere
Obergrenze. Die Zeile `orakel` in `policy_vergleich.json` wählt dagegen nur die
*Einstellung* und lässt die Mission neu planen; weil die Flow-ODE stochastisch
ist, fährt sie nicht die geprobte Bahn. Beide gehören nebeneinander: die erste
sagt, was ein reaktiver Regler im besten Fall erreichen könnte, die zweite ist
der faire Vergleich mit A und B.

## Stellschrauben

Alles per Umgebungsvariable, ohne die Datei zu ändern:

```bash
sbatch --export=ALL,PARAM_PUNKTE=3,SEEDS=1 run_job_policy.bash   # billiger
sbatch --export=ALL,SPLIT=train run_job_policy.bash              # strenge Trennung
sbatch --export=ALL,FORCE=1 run_job_policy.bash                  # alles neu rechnen
```

| Variable | Vorgabe | Wirkung |
|---|---|---|
| `GESAMT_MIN` | 690 | gemeinsames Budget aller Stufen (min) |
| `N_SHAPES` / `N_MAX` / `SEEDS` | 25 / 10 / 2 | Umfang des Orakel-Laufs |
| `PARAM_PUNKTE` / `SVGD_BUCKETS` | 4 / `0 25 100` | Kandidatenraum (4 Modelle × 4 × 3 = 48) |
| `SPLIT` | `val` | `train` = auf den Trainingsformen lernen, die 25 Validierungsformen nur zum Testen |
| `PPO_ITER` / `PPO_ENVS` / `PPO_NMAX` | 200 / 8 / 8 | PPO; die Iterationszahl ist eine Obergrenze, das Zeitbudget bremst früher |
| `FORCE` | 0 | 1 = fertige Stufen nicht überspringen |

### Fortsetzen nach dem Zeitlimit

Fertige Stufen werden übersprungen, ein Folgejob setzt dort fort, wo die Kette
stand:

```bash
JID=$(sbatch --parsable run_job_policy.bash)
sbatch --dependency=afterany:$JID run_job_policy.bash
```

## Wenn etwas schiefgeht

| Symptom | Ursache | Abhilfe |
|---|---|---|
| `... ist nicht startpunkt-konditioniert` | falscher oder fehlender Checkpoint | `netz2d_startpunkt.pt` nach `~/Master_thesis/transfer/` |
| `CUDA out of memory` in Stufe 1 | `--plan_batch 128` zu groß für die zugeteilte Karte | im Skript auf 64 setzen |
| Stufe 2 sagt „kein Datensatz vorhanden“ | Stufe 1 hat nichts geschrieben | `policy-<JOBID>.err` ansehen; mit `FORCE=1` neu |
| `.out` bleibt leer, Job läuft aber | SLURM-Pufferung | normal — das Skript ruft `srun --unbuffered` auf; mit `nvidia-smi` prüfen, nicht am leeren Log |
| PPO-Rückkehr sinkt über die Iterationen | RL ist instabil, ein Seed reicht nicht | `--seed` ändern, `PPO_ENVS` erhöhen; Rückfall ist Option A |
| `UnicodeEncodeError` | nur auf Windows-Konsolen (cp1252) | auf dem Cluster irrelevant |

## Was der Lauf **nicht** entscheidet

Ob eine gelernte Richtlinie in die Thesis gehört, entscheidet nicht `J` allein.
Zwei Setzungen bleiben bei dir:

1. **Die Aufteilung der Formen.** Die 25 Formen sind zugleich der Holdout des
   Planernetzes. Voreinstellung ist die Kreuzvalidierung über Formgruppen
   (`--folds 5`); `SPLIT=train` ist der strengere Weg und kostet einen zweiten
   Orakel-Lauf. Welcher Weg in der Thesis steht, gehört dort begründet.
2. **Was ein Ergebnis bedeutet.** Schlägt A die feste Einstellung nicht, ist
   das selbst ein sauberes Ergebnis („ein fester Wert liegt bereits nah am
   Erreichbaren“) — und die begründete Absage an den viel teureren Weg B.
