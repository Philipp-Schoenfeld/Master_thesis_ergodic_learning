# Kontext für den Wechsel auf den Rechner mit RTX 2070 Super

Stand: Mittwoch, 26.08.2026, 12:30.

Diese Datei ist die Übergabe. Sie sagt, welche Checkpoints es gibt und wofür,
was auf dem Cluster gelaufen ist, was zuletzt am Code geändert wurde und welche
Befehle drüben laufen sollen.

---

## 1. Die Checkpoints

Alle liegen lokal unter `thesis_architecture/checkpoints/` bzw.
`3D_ergodic_learning/checkpoints/`. Über Git kommen sie **nicht** mit
(`.gitignore` fängt `*.pt` ab, GitHub weist über 100 MB ohnehin zurück) —
für den Wechsel steht das Nötige in `transfer/`.

| Datei in `transfer/` | Ursprung | Stand | wofür |
|---|---|---|---|
| `netz2d_startpunkt.pt` | `start_lang` | **Epoche 500 / 500, fertig** | Φ-Kreuz, Variante D |
| `netz2d_laenge.pt` | `len_test` | Epoche 40 (Vorabtest) | erste Längen-Experimente |
| `netz3d_flaechen.pt` | `surfB_lang` | Epoche 1520 / 1750 | 3D-Flächenzweig |

### Der wichtigste: `netz2d_startpunkt.pt`

| | |
|---|---|
| Architektur | `flow_matching_cond_particles_start.py` — Partikel-Cross-Attention **plus Startpunkt-Konditionierung** |
| Training | CFM + ergodischer Zusatzterm, Sinkhorn, w = 1300, blur 0,05, t_power 2 |
| Datensatz | `ergodic_dataset_start.db`, 1187 Formen, Startpunkte gleichverteilt |
| Kennzeichen | `start_cond: True`, `n_flat: 400`, `D: 384`, `nxi: 25`, `n_particles: 256` |
| Größe | 334 MB (Optimiererzustand entfernt) |

**Dieser Lauf ist durch** — alle 500 Epochen, Lernrate bis zum Boden. Er ist der
einzige mit Startpunkt-Konditionierung und der einzige, der die 400 flachen
Formen gesehen hat.

Ohne Startpunkt-Konditionierung kann Variante D nicht dort weiterplanen, wo der
Roboter steht; der frühere Behelf (`--d_join nearest` plus Anfahrt) kostete bis
zu 170 % zusätzliche Weglänge, mit ihr sind es **0,0 %**.

### `netz2d_laenge.pt` — mit Vorbehalt

Der Testlauf vom 25.08. lief auf einer **unvollständigen** Datenbank (2820 von
21.234 Zeilen), weil damals sieben von acht Array-Aufgaben an fehlenden
CJK-Schriften abstürzten. Er zeigt, dass die Längen-Konditionierung technisch
funktioniert, ist aber kein brauchbares Modell. Das richtige Längen-Training
läuft gerade (siehe Abschnitt 5).

### Für Inferenz reicht das, zum Weitertrainieren nicht

In allen `transfer/`-Dateien fehlt der `optimizer_state_dict` (667 MB je Datei,
die zwei AdamW-Momente). Wer drüben weitertrainieren will, braucht die
ungekürzten Dateien aus `thesis_architecture/checkpoints/`.

---

## 2. Einrichten

```bash
git clone git@github.com:Philipp-Schoenfeld/Master_thesis_ergodic_learning.git
cd Master_thesis_ergodic_learning
pip install torch numpy matplotlib scipy geomloss tqdm
# fuer die RTX 2070 Super:
pip install torch --index-url https://download.pytorch.org/whl/cu124

mkdir -p thesis_architecture/checkpoints 3D_ergodic_learning/checkpoints
# aus Google Drive:
mv ~/Downloads/netz2d_startpunkt.pt  thesis_architecture/checkpoints/
mv ~/Downloads/netz2d_laenge.pt      thesis_architecture/checkpoints/
mv ~/Downloads/netz3d_flaechen.pt    3D_ergodic_learning/checkpoints/
```

Erprobt mit torch 2.12.0, numpy 1.26.4, matplotlib 3.10.6, scipy 1.16.1,
geomloss 0.3.1. **JAX wird für die Inferenz nicht gebraucht** — nur der
Datensatzgenerator nutzt es.

### Welche Datenbank wofür

| Datenbank | Größe | über Git? | wofür |
|---|---|---|---|
| `ergodic_dataset_775.db` | 6,9 MB | **ja** | Φ-Kreuz — die zwölf Holdout-Formen |
| `ergodic_dataset_start.db` | 10,3 MB | **ja** | Startpunkt-Training |
| `ergodic_dataset_length.db` | 201 MB | nein, per Drive | Längen-Training |
| `ergodic_dataset_3d.db` | 118 MB | nein, per Drive | 3D-Zweig |

Das Φ-Kreuz liest seine Holdout-Formen über `load_truth` aus `common/data.py`,
dessen `DEFAULT_DB` ist **`ergodic_dataset_775.db`**; ein `--db`-Schalter
existiert in `apply_cfm_belief.py` nicht.

Prüfen:

```bash
python -c "
import torch
c = torch.load('thesis_architecture/checkpoints/netz2d_startpunkt.pt',
               map_location='cpu', weights_only=False)
print('epoch', c['epoch'], '| start_cond', c.get('start_cond'), '| n_flat', c.get('n_flat'))
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Erwartet: `epoch 499 | start_cond True | n_flat 400` und die RTX 2070 Super.

---

## 3. Das Φ-Kreuz — das Experiment für drüben

Fertig vorbereitet unter `phi_kreuz_paket/`:

```bash
./phi_kreuz_paket/01_preflight.sh     # prueft Code, Daten, Checkpoint, Pakete, CUDA
./phi_kreuz_paket/02_phi_kreuz.sh 1   # erst eine Form, ~2 min
./phi_kreuz_paket/02_phi_kreuz.sh     # alle zwoelf, ~30 min auf der GPU
python phi_kreuz_paket/03_auswerten.py
```

Der Preflight prüft nicht bloß Dateinamen, sondern ob `MaskiertesWissen`,
`--prior_mode wahrheit`, `--d_join netz`, die sieben Φ-Modelle und `start_cond`
im Planer wirklich im Code stehen.

**Was es misst.** Drei Muster von Vorwissen (`haelfte`, `quadranten`, `loch`)
mal sieben Zieldichten mal zwölf Formen mal sieben Missionen. Im bekannten
Gebiet liegt die Grundwahrheit exakt vor (σ = 0), außerhalb gar kein Wissen
(μ = 0, σ = 1) — verifiziert: innen RMSE 0,00e+00.

Auf der CPU kostet eine Planung 13,3 s, auf der GPU rund 0,3 s. Variante D ist
der Brocken: 30 der etwa 38 Planungen je Form und Zelle.

---

## 4. Was zuletzt am Code geändert wurde

**`exploration/common/belief.py`** — `MaskiertesWissen` und `muster_maske()`.
Bewusst *kein* dicht konditionierter GP: ein Wahrheitsgitter im Abstand 0,04 ist
bei Korrelationslänge 0,08 zu 97 % korreliert, und die Cholesky-Zerlegung bricht
in float32 ab 205 Punkten zusammen. Ein GP würde außerdem Information über die
Grenze hinweg tragen, was hier nicht gewollt ist.

**`exploration/apply_cfm_belief.py`** — `--prior_mode {messungen,wahrheit}`
(Vorgabe `messungen`, alte Läufe bleiben reproduzierbar), `--d_join netz` für
Variante D mit Startpunkt, `CfmPlanner` erkennt `start_cond` im Checkpoint.
**Achtung:** es gibt zwei Planerklassen; benutzt wird `CfmPlanner` in dieser
Datei, nicht `ModelPlanner` in `common/planner.py`.

**`flow_matching_cond_particles_length.py`** (neu) — Längen-Konditionierung.
`LengthEmbedding` normiert logarithmisch gegen Median und Standardabweichung des
Datensatzes, dann Fourier-Merkmale und MLP, addiert auf `time_cond`. Dazu
`null_length_token` und ein **eigener** 10 %-Dropout: nicht die Partikelmaske,
sonst sähe das Netz nie den Fall „Dichte bekannt, Länge frei". Am Ende der
Generierung passt `_resample_kontrollpunkte` die Zahl der Kontrollpunkte an die
tatsächliche Bogenlänge an.

Gemessen nach 60 Optimierungsschritten: Längenwirkung 2,07e-02 gegen 1,41e-03
für die Zeit, freier Zweig unterscheidbar bei 2,26e-02.

**`flow_matching_runner_length.py`** (neu) — liest `length` und `n_iters`,
schlüsselt Varianten als `label#i{n_iters}`, bestimmt `log_ref` und `log_scale`
aus dem Datensatz und legt beides ins Checkpoint. `--nxi` steht auf 64.

**`ergodic_dataset_generator/ergodic_solver.py`** — `checkpoints` und
`konvergenz_tol`. Vorgabe `None` reproduziert das alte Verhalten exakt.

**`ergodic_dataset_generator/shape_library.py`** — Sockel-Unterstützung, vier
flache Formfamilien, und `_finde_font()`: sucht CJK-Schriften der Reihe nach in
`$ERGODIC_FONT_DIR`, `<Projektwurzel>/fonts/`, `~/Master_thesis/fonts/` und den
Systempfaden — **und prüft, dass die Datei existiert**. Das alte `try/except`
löste nie aus, weil `FontProperties(fname=...)` beim Anlegen nichts prüft.
Die Schriften liegen unter `fonts/` (45 MB, über Drive zu übertragen).

---

## 5. Stand der Läufe auf dem Cluster

### Fertig

| Lauf | Stand | Ergebnis |
|---|---|---|
| **Längen-Datenbank** | ✅ | 21.234 Zeilen, 1187/1187 Formen, Länge 2,14–45,67 |
| `start_lang` (2D Startpunkt) | ✅ | **500/500 Epochen** |
| `surf_kurz` (3D) | ✅ | 350/350, loss 0,735 |
| Visualisierung der Längen-DB | ✅ | 11 Bilder unter `ergodic_dataset_generator/visualizations/laengen/` |

### Läuft

| Lauf | Stand | fertig |
|---|---|---|
| `len_test` (136800) | 40 Epochen, Rauchtest | ~14:00 |
| `surf_lang` (135768) | 353/586, loss **0,352** | ~18:00 |
| Längen-Haupttraining (136801–136803) | wartet auf `len_test` (`afterok`) | Do–Fr |

`surf_lang` liegt mit 0,352 klar unter den 0,735 des einzigen fertigen 3D-Laufs.

### Abgebrochen

`start_kurz` (200-Epochen-Zeitplan) — überflüssig, seit `start_lang` durch ist.
`surf_10h` — bei 22 von 58 Epochen; er war mit `copies_per_char 30` dreimal so
groß angelegt wie `surf_lang` (142.680 gegen 48.052 Schritte) und hätte noch
21,7 h gebraucht. Sein Checkpoint bei Epoche 40 liegt in
`3D_ergodic_learning/checkpoints/`.

---

## 6. Die Längen-Datenbank

19 Iterationsstände je Form: 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250,
1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 10000.

Die vier mittleren Stützstellen (250, 400, 4000, 6000) kamen nachträglich dazu,
weil die Bahn zwischen 300→500 (18,3 %), 3000→5000 (16,3 %) und 500→750 (14,3 %)
am stärksten wuchs. Damit bleibt jeder Abschnitt unter rund zehn Prozent.

**Oberhalb von 10000 wurde nicht erweitert.** Der Zuwachs fällt dort auf 0,64
Längeneinheiten je 1000 Iterationen gegen 11,6 am Anfang; 25000 brächten bei
2,5-facher Rechenzeit rund 20 % mehr Länge, und die Bahnen unterschieden sich um
3 bis 4 % — zu wenig, um als eigenes Trainingsbeispiel zu taugen.

97 % der Formen laufen bis 10000 durch, die Konvergenzprüfung greift also selten.
Der eigentliche Gewinn ist ein anderer: 19 Varianten aus **einem** Solver-Lauf
statt 19 Neustarts.

Spreizung je Form: Median 3,31×, größte 11,59×.

---

## 7. Cluster-Ressourcen — gemessen, nicht geschätzt

Der Cluster wertet die Effizienz aus und meldet sie. Der Bericht über vierzehn
Tage lautete: **CPU-Effizienz 24,07 %**, **RAM-Effizienz 10,64 %** — 49 Tage
CPU-Zeit und 9406 GB-Stunden unnötig blockiert.

Nachgemessen mit `sacct` über die Läufe vom 24. bis 26.08.:

| Job | CPU-Eff. | RAM-Eff. | RAM tatsächlich |
|---|---|---|---|
| `surf_lang` | 25,0 % | 4,8 % | 1,54 GB |
| `start_kurz` | 25,2 % | 10,1 % | 3,25 GB |
| `len_test` | 26,4 % | 8,1 % | 2,59 GB |
| Datensatz-Array | 18,5 % | 21,7 % | 1,74 GB |

**Jeder Job nutzt effektiv einen Kern.** Die Arbeit liegt auf der GPU, oder sie
ist in JAX sequentiell über die Zeitschritte. Lokal gemessen: das Netz braucht
13,47 s je Planung mit zwölf Threads und 13,24 s mit sechs — Threads bringen
nichts.

Daraufhin angepasst:

| Skript | vorher | jetzt |
|---|---|---|
| `run_job_length_test.bash` | `-c 4 --mem=32G` | `-c 2 --mem=8G` |
| `run_job_length_hp.bash` | `-c 4 --mem=32G` | `-c 2 --mem=8G` |
| `run_data_gen.bash` | `-c 6 --mem=8G` | `-c 2 --mem=3G` |
| `run_viz_laengen.bash` | `-c 4 --mem=8G` | `-c 2 --mem=4G` |

Der Array belegte mit `-c 6` über acht Aufgaben **48 der 50 CPUs** des
Kontingents und blockierte damit die eigenen GPU-Trainings, die auf
`AssocGrpCpuLimit` warteten, obwohl eine Karte frei war. Mit `-c 2` sind es 16.

**Regel für künftige Jobs:** vor dem Einreichen messen. Für gelaufene Jobs gibt

    sacct -j <ID> -o JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem

die tatsächliche CPU-Zeit und den Spitzenspeicher. `TotalCPU / (Elapsed ×
AllocCPUS)` ist die CPU-Effizienz. Kontingent: `cpu=50, gres/gpu=3, mem=150G`.

---

## 8. Fallstricke

- **`--device cuda` nicht vergessen**, sonst rechnet alles auf der CPU (Faktor 40).
- **Zwei Planerklassen** — benutzt wird `CfmPlanner` in `apply_cfm_belief.py`.
- **`--max_obs 96`** dünnt die Messpunkte aus; ohne das wüchse die Gram-Matrix
  des GP über dreißig Runden auf mehrere tausend Punkte.
- **`--prior_mode` steht auf `messungen`**, wenn man nichts angibt. Für das
  Kreuz muss `wahrheit` gesetzt sein; `02_phi_kreuz.sh` tut das.
- **8 GB VRAM** reichen für Inferenz. Fürs 3D-Training eng: der Cluster fuhr
  Batch 64 auf Karten mit 11–12 GB, rechne mit `--mini_batch 32`.
- **CJK-Schriften** braucht nur der Datensatzgenerator, nicht die Inferenz.

---

## 9. Offene Fragen

1. **Schlägt Variante D die anderen Missionen?** Ein Zwischenstand auf Form „A"
   sagt: Abdeckungsfehler 0,0264 bei Weglänge 18,23 gegen 0,0358 bei 13,82 für
   `glaube-R`. Besser, aber auf längerem Weg — der Vergleich bei *gleichem*
   Budget steht aus, dafür sind die Anytime-Kurven da.
2. **Hält der Robustheitsbefund bei echtem Vorwissen?** Mit dem alten Modus
   `messungen` war die Niveaumenge am unempfindlichsten dagegen, wo das
   Vorwissen liegt (Spanne 0,0015 gegen 0,0125 bei UCB).
3. **Greift die Längen-Konditionierung auf dem vollen Datensatz?** Sichtbar an
   den Holdout-Bildern: dieselbe Zieldichte, verschiedene Längenvorgaben.
4. **Interaktive Auswertung** aus den `bahnen.json`, durchklickbar nach Muster,
   Zieldichte, Form und Mission.
