# Start auf dem Cluster — Kurzanleitung für den anderen Rechner

Diese Datei ist der ganze Kontext, den du auf dem anderen PC brauchst. Alles
Weitere steht in `README.md` daneben.

## Worum es geht

Gesucht ist die beste Betriebseinstellung der **Längeneinheit-Mission**:
ergodische Bahn planen → genau *eine* Längeneinheit fahren (= die Diagonale der
Zieldomäne, √2) → Glaube aktualisieren → neu planen. Start **ohne jedes
Vorwissen** über die Zielverteilung.

Optimiert wird, getrennt je Zieldichte-Modell:

| Modell | Regler | | Modell | Regler |
|---|---|---|---|---|
| `ucb` | `kappa` | | `mass` | `w` |
| `eid` | `kappa` | | `niveau` | `tau` |

dazu die **Zahl der Ausführungen** `n` und die **Zahl der SVGD-Iterationen**.
Zielfunktion `J = q + 0.02·n + 0.004·t` (`q` = Restanteil des
Abdeckungsfehlers, `n` = gefahrene Strecke in Längeneinheiten, `t` = Rechenzeit).

Der Zielraum ist **16 Parameterpunkte × 6 SVGD-Werte × 4 Modelle × 2 Startwerte
= 768 Auswertungen**, jede über alle 25 Holdout-Formen und 12 Runden.

## Voraussetzungen auf dem Cluster

| Was | Wo | Im Git? |
|---|---|---|
| Code | `~/Master_thesis/thesis_architecture/` | ✅ ja |
| Formen-Datenbank `ergodic_dataset_775.db` | `.../ergodic_dataset_generator/` | ✅ ja (7 MB) |
| **Checkpoint `netz2d_startpunkt.pt`** | `~/Master_thesis/transfer/` | ❌ **nein** (350 MB, gitignored) |
| Conda-Umgebung `thesis` | — | — |

Der Checkpoint ist die einzige Lücke. Prüfen und notfalls hochladen:

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
    'ls -la ~/Master_thesis/transfer/netz2d_startpunkt.pt'

# fehlt er:
rsync -av --progress transfer/netz2d_startpunkt.pt \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/transfer/
```

## Ablauf

**1. Code auf den Cluster.** Entweder dort `git pull`, oder von hier:

```bash
rsync -av --exclude '__pycache__' --exclude 'results/cache' \
    thesis_architecture/exploration_optimierung/ \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/
rsync -av thesis_architecture/exploration/common/svgd_refine.py \
    thesis_architecture/run_job_erkundung_opt.bash \
    thesis_architecture/submit_erkundung_kette.sh \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/
```

**2. Aufwand schätzen** (rechnet nichts, kalibriert nur — zeigt die echte
Cluster-GPU-Geschwindigkeit):

```bash
cd ~/Master_thesis/thesis_architecture/
python -m exploration_optimierung.optimize --mode beide --search voll \
    --param_points 16 --seeds 2 --estimate_only
```

**3. Job-Kette einreihen.** Der volle Suchraum braucht mehr als die 24 h, die
ein `stud`-Job laufen darf. Jedes Glied rechnet 23 h, beendet sich sauber und
das nächste setzt über den Zwischenspeicher fort:

```bash
cd ~/Master_thesis/thesis_architecture/
./submit_erkundung_kette.sh 3          # drei Glieder ≈ 69 h Rechenzeit
```

> Jobs **ausschließlich** per `sbatch` einreihen, nie `srun bash script.bash`.
> Das erledigt `submit_erkundung_kette.sh` korrekt.

**4. Am nächsten Morgen den Zwischenstand holen.** Der laufende Job schreibt
alle 16 Auswertungen die Tabellen neu — es liegt also jederzeit ein
vollständiger, abholbarer Stand da:

```bash
# Stand ansehen, ohne irgendetwas zu holen
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
    'cd ~/Master_thesis/thesis_architecture && \
     python -m exploration_optimierung.optimize --mode beide --search voll \
        --param_points 16 --seeds 2 --status'

# Ergebnisse herunterladen
rsync -av \
    stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/results/ \
    thesis_architecture/exploration_optimierung/results/
```

**5. Lokal auswerten und zeichnen** (braucht keine erneute Suche):

```bash
cd thesis_architecture
python -m exploration_optimierung.optimize --mode beide --search voll \
    --param_points 16 --seeds 2 --reweight
python -m exploration_optimierung.plots
```

`--reweight` baut alle Tabellen allein aus `results/cache/` — ohne GPU, ohne
Neurechnung. Danach liegen die Abbildungen in `results/`.

## Was nach der ersten Nacht fertig ist

Die Arbeitsliste ist **nach SVGD-Iterationen sortiert**, nicht nach Modell.
Deshalb ist jedes Präfix eine in sich abgeschlossene Studie: nach dem ersten
Durchgang liegt die *vollständige* Untersuchung ohne SVGD vor (alle 4 Modelle,
alle 16 Parameterwerte), jeder weitere Durchgang füllt eine ganze Zeile der
Wärmekarte nach.

Grobe Projektion (die echte Zahl steht nach `--estimate_only` fest, sie hängt
an der zugeteilten GPU):

| nach ~ | fertig |
|---|---|
| 1,5 h | `svgd=0` — komplette Studie ohne Verfeinerung |
| 3 h | `+ svgd=25` |
| 5 h | `+ svgd=50` |
| 7 h | `+ svgd=100` |
| 10 h | `+ svgd=200` |
| 15 h | `+ svgd=400` — Startwert 0 komplett |
| 30 h | Startwert 1 komplett, alles fertig |

Ein Abbruch kostet höchstens die gerade laufende Auswertung — jede fertige
liegt als JSON in `results/cache/`.

## Ergebnisdateien

| Datei | Inhalt |
|---|---|
| `alle_laeufe_<tag>.csv` | **jede einzelne Messung**: eine Zeile je (Einstellung, Form, Rundenzahl) |
| `kurven_<tag>.csv` | die `(n, q)`-Kurven, über die Formen gemittelt |
| `suche_<tag>.csv` | Rangfolge je Einstellung |
| `lauf_<tag>.json` | die vollständige Konfiguration des Laufs |
| `waermekarte_<tag>.png` | `J` über (Parameter × SVGD-Iterationen) |
| `formen_<tag>.png` | `q` je einzelner Holdout-Form |
| `panel_<tag>.png` | die beste Einstellung auf allen 25 Formen |

`<tag>` ist `ohne_svgd` bzw. `mit_svgd`.

## Selbsttest

Wenn irgendetwas komisch aussieht — 67 Prüfungen, rund 4 Minuten, braucht GPU
und Checkpoint:

```bash
python -m exploration_optimierung.test_smoke
```
