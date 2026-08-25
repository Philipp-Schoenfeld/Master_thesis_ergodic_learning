# Kontext für den Wechsel auf den Rechner mit RTX 2070 Super

Stand: Dienstag, 25.08.2026, 13:00.

Diese Datei ist die Übergabe. Sie sagt, mit welchem Checkpoint zu starten ist,
was gerade wo läuft, was zuletzt am Code geändert wurde und welche Befehle
drüben laufen sollen.

---

## 1. Der Checkpoint, mit dem du startest

**`thesis_architecture/checkpoints/netz2d_startpunkt.pt`**

| | |
|---|---|
| Ursprung | Lauf `start_lang`, Job 135443, Epoche 159 von 500 |
| Architektur | `flow_matching_cond_particles_start.py` — Partikel-Cross-Attention **plus Startpunkt-Konditionierung** |
| Training | CFM + ergodischer Zusatzterm, Sinkhorn, w = 1300, blur 0,05, t_power 2 |
| Datensatz | `ergodic_dataset_start.db`, 1187 Formen, Startpunkte gleichverteilt über die Fläche |
| Kennzeichen im Checkpoint | `start_cond: True`, `n_flat: 400`, `D: 384`, `nxi: 25`, `n_particles: 256` |
| Größe | 334 MB (Optimiererzustand entfernt, für Inferenz nicht nötig) |

**Warum dieser und kein anderer.** Er ist der einzige Checkpoint mit
Startpunkt-Konditionierung. Ohne die kann Variante D nicht dort weiterplanen,
wo der Roboter gerade steht — der bisherige Behelf (`--d_join nearest` plus
Anfahrt) kostete bis zu 170 % zusätzliche Weglänge. Mit ihm sind es **0,0 %**.

Er ist außerdem der einzige, der die 400 flachen Formen gesehen hat (Sockel,
weichgezeichnet, breite Moden, Konturen). Die führen die Trainingsverteilung an
Glaubensdichten heran: gemessene Massenkonzentration 0,471 gegen 0,853 bei den
alten Trainingsdichten, das Φ-Band liegt bei 0,337–0,481.

**Der Lauf ist nicht zu Ende.** Epoche 159 von 500, die Lernrate steht noch bei
7,89e-05 statt am Boden. Er läuft auf dem Cluster weiter und schreibt alle zehn
Epochen einen neuen Stand — mit Rotation, es bleibt immer nur der neueste liegen.
Wenn du später einen frischeren willst, ist es der einzige Treffer von

    ls -t checkpoints/*START_FLAT400_L500*_ep*.pt | head -1

**Für Inferenz reicht dieser Stand. Zum Fortsetzen des Trainings nicht:** in
`netz2d_startpunkt.pt` fehlt der `optimizer_state_dict` (667 MB, die zwei
AdamW-Momente). Wer drüben weitertrainieren will, braucht die ungekürzte
1002-MB-Datei vom Cluster.

`3D_ergodic_learning/checkpoints/netz3d_flaechen.pt` (Epoche 423) ist das
3D-Gegenstück, für den Flächenzweig. Nur nötig, wenn du dort weiterarbeitest.

---

## 2. Einrichten

```bash
git clone git@github.com:Philipp-Schoenfeld/Master_thesis_ergodic_learning.git
cd Master_thesis_ergodic_learning
conda env create -f environment.yml 2>/dev/null || conda activate thesis

mkdir -p thesis_architecture/checkpoints 3D_ergodic_learning/checkpoints
# Aus Google Drive holen und einsortieren:
mv ~/Downloads/netz2d_startpunkt.pt      thesis_architecture/checkpoints/
mv ~/Downloads/netz3d_flaechen.pt        3D_ergodic_learning/checkpoints/
mv ~/Downloads/ergodic_dataset_775.db    thesis_architecture/ergodic_dataset_generator/
mv ~/Downloads/ergodic_dataset_start.db  thesis_architecture/ergodic_dataset_generator/
```

**Welche Datenbank wofür.** Das Φ-Kreuz liest seine zwölf Holdout-Formen über
`load_truth` aus `common/data.py`, und dessen `DEFAULT_DB` ist
**`ergodic_dataset_775.db`** — ein `--db`-Schalter existiert in
`apply_cfm_belief.py` nicht. Ohne diese Datei bricht der Lauf sofort ab.
`ergodic_dataset_start.db` (1187 Formen) wird nur fürs **Training** gebraucht,
nicht für das Kreuz.

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

Erwartet: `epoch 159 | start_cond True | n_flat 400` und die RTX 2070 Super.

---

## 3. Das Experiment, das drüben laufen soll

Das Φ-Kreuz mit **echtem Vorwissen** und **Variante D**. Auf der GPU kostet eine
Planung rund 0,3 s statt 13,3 s auf der CPU — das ganze Kreuz über alle zwölf
Formen läuft in etwa einer halben Stunde.

```bash
cd thesis_architecture/exploration
CK=../checkpoints/netz2d_startpunkt.pt

for MUSTER in haelfte quadranten loch; do
  for PHI in ucb stretch mass ei lse mi eid; do
    OUT="results/phi_wahrheit/${MUSTER}_${PHI}"
    [ -f "$OUT/metriken.csv" ] && continue
    python -u apply_cfm_belief.py --ckpt "$CK" \
      --shapes 12 --rounds 3 \
      --missions orakel glaube-1 glaube-R zweistufig glaube-D B-warm maeher \
      --phi_model "$PHI" --prior_pattern "$MUSTER" --prior_mode wahrheit \
      --gp_noise 0.05 \
      --d_rounds 30 --d_execute_frac 0.10 --d_join netz \
      --save_paths --out_dir "$OUT" --device cuda
  done
done
```

Die Schleife überspringt fertige Zellen, ein Abbruch ist also unkritisch.

### Was die Schalter bedeuten

- `--prior_mode wahrheit` — im bekannten Gebiet liegt die **Grundwahrheit exakt**
  vor (σ = 0), außerhalb **gar kein Wissen** (μ = 0, σ = 1). Der alte Modus
  `messungen` zog dort nur 60 Punktmessungen und ließ den GP interpolieren.
- `--prior_pattern` — wo das Wissen liegt: `haelfte` (linke Hälfte bekannt),
  `quadranten` (zwei diagonale Quadranten), `loch` (alles außer einem Fleck).
- `--d_join netz` — Variante D plant vom aktuellen Standort aus, per
  Startpunkt-Konditionierung. Nur mit einem `start_cond`-Checkpoint sinnvoll.
- `--d_rounds 30 --d_execute_frac 0.10` — dreißig Runden zu je einem Zehntel der
  geplanten Bahn. Ergibt rund drei Bahnlängen und ist damit mit `glaube-R`
  (drei volle Runden) vergleichbar.

---

## 4. Was zuletzt am Code geändert wurde

Alles davon ist im Repo, nichts steht nur auf dem Cluster.

**`exploration/common/belief.py`** — neue Klasse `MaskiertesWissen` und
`muster_maske()`. Grundwahrheit im bekannten Gebiet, gar kein Wissen außerhalb.
Bewusst *kein* dicht konditionierter GP: ein Wahrheitsgitter im Abstand 0,04 ist
bei Korrelationslänge 0,08 zu 97 % korreliert, und die Cholesky-Zerlegung bricht
in float32 ab 205 Punkten zusammen. Außerdem würde ein GP Information über die
Grenze hinweg tragen, was hier nicht gewollt ist.

**`exploration/apply_cfm_belief.py`**
- `--prior_mode {messungen,wahrheit}`, Default `messungen` (alte Läufe bleiben
  reproduzierbar), plus `--sigma_bekannt`.
- `CfmPlanner` erkennt `start_cond` im Checkpoint und lädt dann die andere
  Architektur. **Achtung:** es gibt zwei Planerklassen; benutzt wird
  `CfmPlanner` in dieser Datei, nicht `ModelPlanner` in `common/planner.py`.
  Beide sind erweitert, aber nur die erste läuft.
- `run_variant_d` kennt `--d_join netz`. Kein `nearest`-Einstieg, keine Anfahrt.
- Die Maske wandert nach `bahnen.json` und ins Formen-Bild.

**`thesis_architecture/flow_matching_cond_particles_start.py`** — Kopie der
Partikel-Architektur mit `StartEmbedding`: Fourier-Merkmale des Startpunkts,
MLP auf Breite D, **addiert auf die Zeit-Einbettung**. Läuft damit durch dieselbe
`film_proj` in jedem `ConvResBlock` wie die Flusszeit. Nach der Integration wird
der erste Kontrollpunkt zusätzlich hart gesetzt — Konditionierung macht den
Startpunkt wahrscheinlich, nicht exakt.

**`thesis_architecture/flow_matching_runner_start.py`** — `--db`, `--tag`,
`--keep_checkpoints` (Rotation: beim Speichern eines neuen Standes werden ältere
desselben Laufs gelöscht). Der Laufname trägt `_START`, `_FLAT400` und den Tag.

**`ergodic_dataset_generator/shape_library.py`** — Sockel-Unterstützung
(`mit_sockel`, `_vielleicht_sockel`) und vier neue Formfamilien
(`flat_ped`, `flat_blur`, `flat_broad`, `flat_ring`), zusammen 400 Trainings-
und 12 Holdout-Formen. `all_dataset_shapes()` und `train_shape_names()` sind
bewusst **nicht** angefasst — deren Reihenfolge hängt an einem Shuffle mit
Keim 42, jede Ergänzung hätte die bestehenden 750 Trainingsformen ausgetauscht.

**`ergodic_dataset_generator/dichte_numpy.py`** — Dichteauswertung ohne JAX, für
Übersichtsgrafiken. Gegen die JAX-Fassung geprüft: maximale Abweichung 1,2e-06.

---

## 5. Was gerade läuft und wo

### Auf dem Cluster (Kontingent: `gres/gpu=3`, alle drei belegt)

| Job | ID | Stand | Ende |
|---|---|---|---|
| `start_lang` | 135443 | Ep. 139/500, cn09, 178 s/Ep | Mi ~04:29 |
| `surf_lang` (3D) | 135086 | Ep. 741/1750, dgx-station | Di ~20:15 |
| `surf_10h` (3D) | 135475 | Ep. 4/75, cn01 (RTX 2080 Ti) | Di ~19:36 |
| `phi_wahrheit` | 135774 | wartet auf GPU-Platz | ab ~19:36, +2 h |
| Folgejobs | 135765–135770 | verkettet über `afterany` | bis Fr/Sa |

`phi_wahrheit` rechnet dasselbe Experiment wie oben, aber über alle zwölf
Formen. Wenn du es drüben auf der GPU fährst, kannst du den Job stornieren:
`ssh <cluster> scancel 135774`.

### Lokal auf dem alten Rechner

Dasselbe Kreuz, aber nur auf Form „A", zwei Prozesse à sechs Threads.
Ergebnisse in `thesis_architecture/exploration/results/phi_wahrheit_A/`.
Fertig gegen 13:57.

---

## 6. Fallstricke

- **`--device cuda` nicht vergessen.** Ohne das läuft alles auf der CPU, und
  eine Planung kostet dort 13,3 s statt 0,3 s.
- **8 GB VRAM.** Für Inferenz reichlich. Fürs 3D-Training eng: der Cluster
  fuhr `D=384`, 512 Partikel, Batch 64 auf Karten mit 11–12 GB. Rechne mit
  `--mini_batch 32`. Für 2D (256 Partikel) sollte es passen.
- **Modelle nicht über Git.** GitHub weist Dateien über 100 MB ab, die
  Checkpoints sind 334 MB. `.gitignore` fängt `*.pt` bereits ab. Auch `*.db`
  gehört dort hinein: `ergodic_dataset_3d.db` ist 118,5 MB und ließe den Push
  scheitern.
- **Zwei Planerklassen.** Siehe oben — `ModelPlanner` in `common/planner.py`
  wird nicht benutzt. Wer dort etwas ändert, ändert nichts am Verhalten.
- **`--max_obs 96`** dünnt die Messpunkte aus. Ohne das würde die Gram-Matrix
  des GP über dreißig Runden auf mehrere tausend Punkte wachsen.
- **float16-Fassung ungetestet.** `netz2d_startpunkt_fp16.pt` (167 MB) existiert,
  aber ob die erzeugten Bahnen identisch bleiben, ist nicht geprüft. Nimm
  float32, bis das gemessen ist.

---

## 7. Offene Fragen, an denen es weitergeht

1. **Schlägt Variante D die anderen Missionen?** Ein erster Probelauf auf einer
   Form mit acht Runden sagte nein (0,0751 gegen 0,0596 bei gleichem Budget).
   Mit dreißig Runden und zwölf Formen steht es noch aus.
2. **Hält der Befund aus dem Messungs-Kreuz auch bei echtem Vorwissen?** Dort
   war die Niveaumenge am robustesten (Spanne 0,0015 über die drei Muster,
   gegen 0,0125 bei UCB). Mit exakter Grundwahrheit ist die Kante viel härter.
3. **Interaktive Auswertung.** Aus den `bahnen.json` soll eine Grafik entstehen,
   durch die sich nach Muster, Zieldichte, Form und Mission klicken lässt.
