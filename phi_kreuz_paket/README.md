# Φ-Kreuz mit echtem Vorwissen — alles an einem Ort

Was hier steht, reicht aus, um das Experiment auf einem frischen Rechner zu
fahren. Stand: Dienstag, 25.08.2026.

```
01_preflight.sh     prueft, ob alles da ist (aendert nichts)
02_phi_kreuz.sh     faehrt das Kreuz
03_auswerten.py     fasst die Zellen zu Tabellen zusammen
README.md           diese Datei
```

---

## Warum drüben Code fehlte

Der Commit `c620598` enthält den gesamten `exploration`-Ordner samt
GP-Wissensmaske, den sieben Φ-Modellen und Variante D. Er wurde committet,
aber **nie erfolgreich gepusht** — GitHub weist jede Datei über 100 MB ab, und
in dem Commit steckte `3D_ergodic_learning/ergodic_dataset_3d.db` mit
**118,5 MB**. Der Push brach ab, der Code blieb hier liegen.

Die Reparatur steht unten unter „Den Push zum Laufen bringen".

---

## Was das Experiment ist

Ein Roboter kennt einen Teil eines Dichtefeldes **exakt** und den Rest **gar
nicht**. Aus diesem Wissensstand wird eine Zieldichte Φ gebildet, das
CFM+ErgLoss-Netz plant darauf eine ergodische Bahn, der Roboter fährt sie ab
und misst dabei. Gefragt ist, welche Modellierung von Φ damit am besten
zurechtkommt — und ob es einen Unterschied macht, *wo* die Wissenslücke sitzt.

**Drei Muster von Vorwissen**

| Muster | bekannt | Anteil |
|---|---|---|
| `haelfte` | linke Hälfte | 50,0 % |
| `quadranten` | zwei diagonal gegenüberliegende Quadranten | 50,0 % |
| `loch` | alles außer einem Kreis in der Mitte | 75,9 % |

Im bekannten Gebiet gilt μ = Wahrheit und σ = 0, außerhalb μ = 0 und σ = 1.
Verifiziert: innen RMSE **0,00e+00**.

**Sieben Zieldichten** — `ucb` (μ+κσ), `stretch`, `mass`, `ei`, `lse`
(Niveaumenge), `mi` (wechselseitige Information), `eid` (Informationsdichte).

**Sieben Missionen** — `orakel` (kennt die Wahrheit, Obergrenze), `glaube-1`
(einmal planen), `glaube-R` (drei Runden), `zweistufig`, **`glaube-D`**
(Abdeckungsschuld, dreißig Runden zu je einem Zehntel), `B-warm`, `maeher`
(festes Raster, Grundlinie).

Macht **21 Zellen** zu je 12 Formen und 7 Missionen.

---

## Variante D — der Kern dieses Durchlaufs

Plane eine ganze Bahn, fahre ein Zehntel davon, buche um, plane neu — dreißig
Mal. Die Umbuchung senkt die Anziehung besuchter Gebiete, aber **nicht auf
null**:

```
sigma_eff = sigma · (1 − v)                    Unsicherheit dort geloescht, wo gefahren wurde
mu_eff    = mu · (1 − w · v · (1 − sigma))     Anziehung faellt proportional zu
                     ↑   ↑        ↑             Gewicht, verbrachter Zeit, Wissen
```

Ein Gebiet, das besucht **und dabei gut vermessen** wurde, verliert seine
Anziehung. Eines, das besucht, aber weiterhin unsicher ist, behält sie.

**Erst mit dem startpunkt-konditionierten Netz ergibt das Sinn.** Vorher begann
jede neu geplante Bahn irgendwo, und der Roboter musste erst hinfahren — bei
dreißig Runden eine erhebliche Strecke. Der Schalter `--d_join netz` gibt dem
Netz die aktuelle Position als FiLM-Konditionierung. Gemessen: **0,0 %
Anfahrtsanteil** statt bis zu 170 %.

---

## Was über Git kommt und was nicht

**Über Git:** der gesamte Code — `exploration/` mit `apply_cfm_belief.py`,
`common/belief.py` (GP und Wissensmaske), `common/acquisition.py` (die sieben
Φ-Modelle), die Netzarchitekturen, `bsplinax-main/`, diese Skripte.

**Über Google Drive**, weil zu groß für GitHub:

| Datei | Größe | wohin | wofür |
|---|---|---|---|
| `netz2d_startpunkt.pt` | 334 MB | `thesis_architecture/checkpoints/` | **nötig** |
| `ergodic_dataset_775.db` | 7 MB | `thesis_architecture/ergodic_dataset_generator/` | **nötig** |
| `ergodic_dataset_start.db` | 10 MB | dieselbe | nur fürs Training |
| `netz3d_flaechen.pt` | 334 MB | `3D_ergodic_learning/checkpoints/` | nur für 3D |
| `gross/` | 429 MB | an ihre Originalpfade | Rohdaten, PDFs — optional |

Für das Φ-Kreuz reichen die **ersten beiden, 341 MB**.

Die Datenbankfrage ist eine Stolperstelle: das Kreuz liest seine zwölf
Holdout-Formen über `load_truth` aus `common/data.py`, und dessen `DEFAULT_DB`
ist **`ergodic_dataset_775.db`**. Ein `--db`-Schalter existiert dort nicht.
`ergodic_dataset_start.db` mit den 1187 Formen wird nur fürs Training gebraucht.

---

## Der Checkpoint

**`netz2d_startpunkt.pt`** — Lauf `start_lang`, Epoche 159 von 500,
`start_cond: True`, `n_flat: 400`, D 384, nxi 25, 256 Partikel. Trainiert mit
CFM plus ergodischem Zusatzterm (Sinkhorn, w = 1300, blur 0,05, t_power 2) auf
1187 Formen mit gleichverteilten Startpunkten.

Er ist der einzige mit Startpunkt-Konditionierung und der einzige, der die 400
flachen Formen gesehen hat. Der Lauf ist nicht zu Ende — die Lernrate steht bei
7,89e-05 statt am Boden.

**Für Inferenz reicht er, zum Weitertrainieren nicht:** der
`optimizer_state_dict` (667 MB) ist entfernt.

---

## Loslegen

```bash
./phi_kreuz_paket/01_preflight.sh     # prueft Code, Daten, Checkpoint, Pakete, CUDA
./phi_kreuz_paket/02_phi_kreuz.sh 1   # erst eine Form, zum Ausprobieren
./phi_kreuz_paket/02_phi_kreuz.sh     # dann alle zwoelf
python phi_kreuz_paket/03_auswerten.py
```

Pakete, falls die Umgebung neu ist:
```bash
pip install torch numpy matplotlib scipy geomloss tqdm
# fuer die RTX 2070 Super:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Erprobt mit torch 2.12.0, numpy 1.26.4, matplotlib 3.10.6, scipy 1.16.1,
geomloss 0.3.1. **JAX wird nicht gebraucht** — nur der Datensatzgenerator nutzt es.

**Laufzeit:** auf der GPU rund 0,3 s je Planung, das ganze Kreuz etwa eine
halbe Stunde. Auf der CPU 13,3 s je Planung, also rund drei Stunden. Variante D
ist der Brocken: dreißig der rund achtunddreißig Planungen je Form und Zelle.

---

## Den Push zum Laufen bringen

Auf dem alten Rechner, in dieser Reihenfolge:

```bash
cd /media/philipp/storage/Dokumente/Uni/Master_thesis

cat >> .gitignore <<'ENDE'

# Zu gross fuer GitHub (Limit 100 MB je Datei) — per Drive uebertragen.
*.db
*.npz
bahnen.json
transfer/
__pycache__/
*.pyc
ENDE

# Die grossen Dateien aus der Verfolgung nehmen. Sie bleiben auf der Platte
# liegen; --cached loescht nur den Git-Eintrag.
git rm --cached -r --quiet 3D_ergodic_learning/ergodic_dataset_3d.db \
    3D_ergodic_learning/cache \
    thesis_architecture/exploration/results/*/*/bahnen.json \
    thesis_architecture/exploration/results/*/bahnen.json 2>/dev/null

git add .gitignore .
git commit --amend --no-edit          # haengt es an c620598 an, das ist noch unversandt
git push
```

`--amend` ist hier zulaessig, weil `c620598` noch nirgends veroeffentlicht ist.
Danach pruefen, dass nichts Grosses mehr drinsteckt:

```bash
git ls-tree -r -l HEAD | awk '$4>104857600 {print $4/1048576" MB", $5}'
```

Leer heisst: der Push geht durch.

---

## Fallstricke

- **`--device cuda` nicht vergessen**, sonst rechnet alles auf der CPU.
- **Zwei Planerklassen.** Benutzt wird `CfmPlanner` in `apply_cfm_belief.py`,
  nicht `ModelPlanner` in `common/planner.py`. Beide sind erweitert, aber nur
  die erste läuft.
- **`--max_obs 96`** dünnt die Messpunkte aus. Ohne das wüchse die Gram-Matrix
  des GP über dreißig Runden auf mehrere tausend Punkte.
- **`--prior_mode` steht auf `messungen`**, wenn man nichts angibt. Das ist der
  alte Modus mit 60 Punktmessungen, absichtlich als Vorgabe, damit frühere
  Läufe reproduzierbar bleiben. Für dieses Experiment muss `wahrheit` gesetzt
  sein — `02_phi_kreuz.sh` tut das.
- **8 GB VRAM** reichen für die Inferenz reichlich. Fürs 3D-Training eng: der
  Cluster fuhr Batch 64 auf Karten mit 11–12 GB, rechne mit `--mini_batch 32`.

---

## Was noch offen ist

1. **Schlägt Variante D die anderen Missionen?** Ein Zwischenstand auf Form „A"
   sagt: Abdeckungsfehler 0,0264 bei Weglänge 18,23, gegen 0,0358 bei 13,82 für
   `glaube-R`. Besser, aber auf längerem Weg — der Vergleich bei *gleichem*
   Budget steht noch aus, dafür sind die Anytime-Kurven da.
2. **Hält der Robustheitsbefund?** Mit dem alten Modus `messungen` war die
   Niveaumenge am unempfindlichsten dagegen, wo das Vorwissen liegt (Spanne
   0,0015 gegen 0,0125 bei UCB). Mit exakter Grundwahrheit ist die Kante viel
   härter — ob das Bild gleich bleibt, ist die eigentliche Frage.
3. **Interaktive Auswertung** aus den `bahnen.json`, durchklickbar nach Muster,
   Zieldichte, Form und Mission.
