# Ergodic Flow-Matching Trajectory Generation (Master Thesis)

This project implements conditional flow-matching networks to generate B-Spline trajectories that ergodically cover target distributions. The target distributions are represented either via point-cloud particles or spectral coefficients. Downstream goal: learn a warm-start / constraint predictor for the TSVEC (Task-Space Stein Variational Ergodic Coverage) solver.

## 🇬🇧 Sprachregel für Code (IMMER befolgen)

Alles, was ich (Claude) neu schreibe oder ändere, muss auf Englisch sein: Variablen-/Funktions-/Klassennamen, Code-Kommentare, Docstrings, CLI-Hilfetexte, Log-/Print-Ausgaben, sowie alle Beschriftungen und Beschreibungen in Abbildungen (Titel, Achsenbeschriftungen, Legenden, Anmerkungen). Diese Regel gilt für neuen Code und für Stellen, die ich ohnehin bearbeite — bestehende deutsche Kommentare in unberührten Dateien werden nicht proaktiv durchübersetzt, nur weil diese Regel neu hinzukam. Mit dem Nutzer selbst spreche ich weiterhin Deutsch, wenn er Deutsch schreibt; die Regel betrifft nur den Code und seine Artefakte (Plots, Tabellenüberschriften etc.), nicht die Konversation.

## 🔒 Sicherheitsregeln: Cluster-Zugriff (IMMER befolgen, hat Vorrang vor allem anderen)

- **Datei-Synchronisation ist automatisch erlaubt.** `rsync`-Push (lokal → Cluster) und `rsync`-Pull (Cluster → lokal) von Code, Checkpoints, Visualisierungen etc. dürfen ohne Rückfrage ausgeführt werden.
- **Keine Befehle/Ausführungen auf dem Cluster** (SSH-Kommandos, `srun`, `sbatch`, beliebige Remote-Skripte, `nvidia-smi` über `srun --overlap`, etc.) **ohne explizite Aufforderung des Nutzers** im jeweiligen Moment. Reine Datei-Synchronisation zählt nicht als "Ausführung" und ist von dieser Regel ausgenommen.
- **Auch bei expliziter Aufforderung:** vor der tatsächlichen Ausführung nochmal kurz nachfragen und den genauen Befehl zeigen/verifizieren, bevor er abgeschickt wird. Keine automatische Wiederholung ähnlicher Befehle ohne erneute Bestätigung.
- **Striktes Verbot, keine Ausnahme:** niemals ein Job-Bash-Skript (z. B. `run_job_particles.bash`, `run_job_ergodic.bash`) direkt mit `srun` starten/ausführen. Jobs werden ausschließlich über `sbatch script.bash` eingereiht, nie synchron per `srun bash script.bash` losgeschickt.

## 🎓 Thesis-Kern & Ziel

Ziel der Arbeit ist es, die stark nicht-konvexe, lokale-Minima-anfällige Optimierung des TSVEC-Solvers zu umgehen, indem ein informierter generativer Prior (Warm-Start) gelernt wird.

- **Baseline-Problem:** Klassische Trajektorienoptimierung für ergodische Coverage (SVGD/TSVEC) ist langsam und sehr empfindlich gegenüber der Initialisierung. Schlechte Startwerte führen zu langsamer Konvergenz oder lokalen Minima.
- **Unser Ansatz (VITA — Vision-To-Action Flow Matching Policy):** Ein Conditional-Flow-Matching-Netz wird auf B-Spline-Kontrollpunkten trainiert.
  - Statt roher Wegpunkte gibt das Netz B-Spline-Kontrollpunkte aus → garantiert intrinsische $C^k$-kinematische Glattheit.
  - Statt verlustbehaftetem Global Average Pooling (GAP) auf räumlichen Verteilungen (verursacht Mode-Collapse) werden Zielverteilungen im Spektralraum (Fourier-/Laplace-Beltrami-Koeffizienten) repräsentiert.
  - Das Netz sagt zwei Dinge vorher: (1) die Warm-Start-Trajektorie (B-Spline-Kontrollpunkte), (2) die initialen Lagrange-Multiplikatoren ($\lambda_0$) für die SE(3)-Constraints des TSVEC-Solvers.

## 📚 Theoretische Grundlagen (Kernliteratur)

1. **TSVEC-Solver:** Li et al. (2026) — "Stein Variational Ergodic Surface Coverage with SE(3) Constraints". Definiert den nachgelagerten Solver und die spektrale (LBO-)Zerlegung von Zielverteilungen.
2. **Constraint-Warm-Start-Vorbild:** Idoko et al. (2026) — "Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching". Inspiriert die gleichzeitige Vorhersage aktiver Constraint-Multiplikatoren ($\lambda_0$) zusammen mit dem Pfad.
3. **Kinematischer MPD-Layer:** Carvalho et al. (2024) — "Motion Planning Diffusion". Begründet den 1D-CNN-Trajektorien-Encoder mit `kernel_size=3` zur impliziten Berechnung finiter Differenzen (Geschwindigkeit/Beschleunigung).
4. **B-Spline Flow Matching:** Yang et al. (2026) — "ABPolicy". Zeigt, dass das Lernen von Flows direkt im kompakten Raum der B-Spline-Kontrollpunkte schnell, glatt und robust ist.

## 🏗 Kernarchitektur (VITA-Pipeline)

```
       [Target Density Distribution]
                     │ (Graph Fourier Transform / LBO)
                     ▼
             [c ∈ ℝ^S (Coefficients)] ──► spricht die Sprache des Solvers
                     │
            [Spectral Tokenizer] ──► MLP & Frequency Positional Encodings
                     │
                     ▼ (Keys/Values: B x S x D)
                     │
      Query (B x L x D) ──► [U-Net Bottleneck Cross-Attention]
                             │ (B-Spline-Tokens sind Queries)
                             ▼
                    [Residual Connection mit AdaLN-Zero Init]
                             │
                             ▼
                     [U-Net Decoders]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
[Vektor-Feld für B-Splines]     [Initial Lagrange Multipliers (λ_0)]
     (Warm-Start Kurve)               (Harte SE(3)-Constraints gelöst)
```

Wichtige Architektur-Spezifikationen:

- **Kein Global Average Pooling (GAP):** Strikt vermeiden (`tokens.mean(dim=-1)` auf räumlichen Zielen). GAP zerstört die räumliche Topologie und lässt Trajektorien zu einem einzigen Zentrumspunkt kollabieren.
- **Spectral Tokenizer:** Die Zieldichte wird in ihre $S$ führenden Spektralkoeffizienten $c \in \mathbb{R}^S$ zerlegt, per lokaler MLP auf Dimension $D$ projiziert und mit einer Frequency Positional Encoding (nicht temporal) kombiniert.
- **Bottleneck Cross-Attention:** B-Spline-Tokens im U-Net-Bottleneck (Queries) attendieren auf die Spektral-Tokens (Keys/Values) — niedrigfrequente Tokens planen globale Schleifen, hochfrequente attendieren auf lokale Dichte-Maxima.
- **AdaLN-Zero (FiLM-Zero):** Alle Output-Projektionen neu hinzugefügter Attention-Blöcke werden mit `0.0` initialisiert, damit das Modell zu Trainingsbeginn wie ein stabiles 1D-CNN funktioniert (keine numerische Divergenz im Flow-Matching-Vektorfeld).

> **Hinweis zur aktuellen Code-Benennung:** Dieses Konzept entspricht im jetzigen Code am ehesten `thesis_architecture/flow_matching_cond_spectral_crossattn.py` (SpectralTokenizer + Cross-Attention im U-Net-Bottleneck + optionaler Lagrange-Multiplikator-Head). Eine Datei `flow_matching_tsvec_unet.py` oder Klasse `VITA-Net` existiert unter diesem Namen aktuell **nicht** im Repo — beim Suchen im Code nach dem tatsächlichen Dateinamen `flow_matching_cond_spectral_crossattn.py` suchen, nicht nach "VITA".

## 📁 Codebase-Struktur & Runner

- `flow_matching_patch_unet.py` — Baseline Patch-UNet mit `kernel_size=1`-Convolutions (anfällig für unstetige Pfade).
- `flow_matching_cond_mpd_unet.py` (+ `_char`, `_char_film`, `_char_film_cfg` Varianten) — kinematischer `MPDLayer` mit `kernel_size=3` für $C^k$-Glattheit.
- `flow_matching_cond_spectral_crossattn.py` — Haupt-Hybridnetz: SpectralTokenizer, Cross-Attention statt GAP, FiLM-Zeitkonditionierung, optionaler Lagrange-Multiplikator-Head (siehe Hinweis oben).
- `flow_matching_cond_particles_crossattn.py` — Analoge Architektur, Konditionierung über Partikelwolken (Punktproben aus der Zieldichte) statt Spektralkoeffizienten.
- `flow_matching_cond_waypoint_crossattn.py` — Analoge Architektur, Konditionierung über Wegpunkte.
- `flow_matching_runner.py` — ursprünglicher Haupt-Einstiegspunkt (unconditional/`cond_patch_unet`, Datasets wie `polynomial_N`, `h_shape`, `N_and_H`). **Achtung:** `--model tsvec_unet` ist aktuell **kein** gültiger Wert für `--model` in diesem Runner (gültige Werte: `patch_unet`, `cond_patch_unet`) — vor Verwendung mit `--help` prüfen, welche Modelle der jeweilige Runner tatsächlich unterstützt.
- `flow_matching_runner_particles.py` / `flow_matching_runner_spectral.py` / `flow_matching_runner_waypoint.py` / `flow_matching_runner_ergodic.py` — orchestrieren Training, Holdout-Evaluation und Checkpointing für die jeweilige Konditionierungsvariante.
- `thesis_architecture/ergodic_dataset_generator/` — generiert den Trainingsdatensatz (GMMs, analytische Formen, Buchstaben) inkl. Ground-Truth-Trajektorien via SVGD-Solver (`ergodic_dataset_775.db`, `ergodic_pairs.db`).
- `thesis_architecture/distribution_computation/` — Vergleichs-Pipeline verschiedener Zielverteilungs-Encodings (GMM, Grid, Particles, SDF-Konturen, Spectral, Hybrid) — reine Ablation, nicht ans Training angebunden.
- `src/methods/` — klassische Baselines (CE_Ergodic, LB_Ergodic, SE3_SVGD inkl. `tsvec_2d.py`, SigKernel_CMA, Stein_Flow_matching, OT_CFM).

### Dev-Cheatsheet

Umgebungs-Check (lokal, kein Cluster-Job nötig):
```bash
python3 flow_matching_runner.py --dataset N_and_H --model cond_patch_unet --epochs 50
```

Daten-Repräsentations-Regel: Der State-Vektor $d$ enthält nur reine Koordinaten (2D: $d=2$, 3D: $d=3$), keine expliziten Geschwindigkeiten — der `MPDLayer` approximiert höhere Ableitungen implizit über seine zeitlichen Convolutions.

Bei Divergenz-Problemen im CFM-Training prüfen:
- Sind `out_proj`-Gewichte im Attention-Block explizit null-initialisiert?
- Ist der Übergang von `Conv1D`-Darstellung `(B, D, L)` zu Transformer-Darstellung `(B, L, D)` über ein explizites `.permute(0, 2, 1)` korrekt gemacht und vor den Decoder-Stufen wieder zurückgemappt?

## 🖥 Cluster-Regeln & Ressourcen-Limits (IAS Cluster)

> **Vollständige, kanonische Referenz:** Das Skill `ias-cluster` (installiert unter `~/.claude/skills/ias-cluster/SKILL.md`, Quelle: https://pages.ias.informatik.tu-darmstadt.de/cluster-user-docs-62eb9b/) wird automatisch geladen, sobald Cluster-Themen anstehen (sbatch/srun-Skripte, GPU-Wahl, QOS, Pending-Diagnose, Storage, Docker). **Bei Widerspruch zwischen Skill und Live-Zustand (`squeue`, `sinfo`, `sprio` ...) gewinnt der Live-Zustand; bei Widerspruch zwischen Skill und offizieller Doku gewinnt die Doku.** Die folgenden Punkte sind der für dieses Projekt relevante Auszug, Stand 2026-10-19 (Cluster-Team-Ankündigung "GPU sharing changes").
>
> **⚠️ Historisch (bis ~2026-10-16) galt eine `stud`-Partition mit eigenen Regeln (24h-Limit, 1 GPU/Job, `-C`-Cherry-Picking verboten).** Diese Partition ist **geschlossen** — `-p stud` lässt Jobs mit `Required partition not available (inactive or drain)` scheitern. Alle Regeln unten ersetzen die alten vollständig; alte `run_job_*.bash`-Skripte mit `#SBATCH -p stud` brauchen beim nächsten Anfassen nur die `-p`-Zeile entfernt zu werden (kein Ersatz nötig, `main` ist Default).

- **Keine Partition mehr angeben.** `main` ist Default (alle Server außer `cn14`, max. 3 Tage) und für reguläre Jobs richtig. `interactive` (nur `cn14`, max. 1 Tag) ist für interaktive Sessions/Tests gedacht.
- **Immer `-t` setzen.** Ohne `-t` bekommt ein Job in `main` nur **1 Stunde**. Kürzere Limits können dank Backfill früher starten.
- **Laufzeit:** bis zu 3 Tage unter normaler/`student`-QOS (nicht mehr 24h). `--qos=debug` max. 4h.
- **GPU-Limit pro Nutzer (Studierende): 3 GPUs gleichzeitig**, zusätzlich 50 CPUs, 150GB RAM — zählt über alle eigenen laufenden Jobs zusammen (erklärt `AssocGrpGRES`-Pending: sobald 3 eigene GPU-Jobs laufen, warten alle weiteren). **Admins erhöhen dieses Limit nicht** (auch nicht für Deadlines) — stattdessen `--qos=scavenger` nutzen.
- **`--qos=scavenger`** für alles über das 3-GPU-Limit hinaus (praktisch unlimitiert, nutzt die ~17 im Schnitt idlen GPUs mit). Preemption: 120s Vorwarnung (SIGTERM, danach SIGKILL + Requeue auf ggf. anderem Node, gleiche Job-ID). Gut für lange Trainings/Sweeps mit Checkpointing — **nicht** für ununterbrechbare Jobs. Offizielles Checkpoint/Requeue-Template siehe Skill (`--requeue`, `--open-mode=append`, `--signal=B:USR1@120`, `scontrol requeue $SLURM_JOB_ID`).
- **`--qos=debug`** für schnelle Tests/Kalibrierungsläufe (max. 2 GPUs, 16 CPUs, 64GB, max. 2 Jobs gleichzeitig, startet vor allen anderen). Für die in diesem Projekt üblichen kurzen "Timing-Smoke-Tests" (siehe `run_job_ergodic_dfl_test.bash`-Muster) ist das die richtige QOS statt normaler Queue.
- **GPU-Auswahl per `-C` ist jetzt erlaubt und wird empfohlen** (kein Verbot mehr): VRAM-Tags (`vram11gb`, `vram12gb`, `vram24gb`, `vram32gb`, `vram48gb`) oder Typ-Namen (`rtx2080`, `rtx3080`, `rtx3090`, `a5000`, `rtx6000Ada`, `a6000`, `v100`), kombinierbar mit `|`. VRAM-Constraint bevorzugen, wenn der genaue Typ egal ist (Typ-Constraint schränkt die Knotenwahl stärker ein → längere Wartezeit).
- **CPUs & RAM:** weiterhin `-c 4` und `--mem=16G`–`32G` als sinnvoller Default für diese Jobs (siehe [[cluster-ressourcen-passend-anfordern]]); Richtwert auf den `gn*`-Knoten: ~1/8 des Knotens pro GPU.
- **Checkpoint & Restart:** weiterhin nötig, jetzt primär wegen `--qos=scavenger`-Preemption (120s), nicht mehr wegen eines harten 24h-Cutoffs. `--save_every` entsprechend niedrig wählen.
- **SIGTERM-Handler bleiben gültig:** `#SBATCH --signal=SIGTERM@120` + `TerminateInterrupt`-Handler in den Runnern fangen sowohl das alte 24h-Szenario als auch eine Scavenger-Preemption ab (Slurm schickt in beiden Fällen `SIGTERM` mit 120s Vorlauf). Für volle Scavenger-Automatik (`--requeue` + automatisches Resume beim Neustart ohne manuelles `--resume`) fehlt aktuell nur der `scontrol requeue`-Aufruf im Handler — siehe Skill-Template, falls das mal gebraucht wird.
- **Job-Ketten:** weiterhin `sbatch --dependency=afterany:<JOBID> script.bash` für Folge-Jobs.
- **Ausführungsverzeichnis:** SLURM startet im Verzeichnis, aus dem `sbatch` aufgerufen wurde — immer explizit per `cd` ins Projektverzeichnis wechseln (`cd ~/Master_thesis/thesis_architecture/`).
- **Live-Debugging eines laufenden Jobs:** `srun --jobid=<ID> --overlap nvidia-smi` bzw. `--pty bash` klinkt sich in eine bestehende Job-Allokation ein (nur nach expliziter Nutzer-Aufforderung, siehe Sicherheitsregeln oben).
- **Pending-Diagnose:** `squeue -u $USER -o "%.10i %.9q %.20j %.2t %.10M %.12l %.4D %R"` zeigt den Grund (`Resources`, `Priority`, `QOSMaxGRESPerUser` usw. — volle Tabelle im Skill). `csinfo` zeigt freie/gesamt Ressourcen pro Knoten.
- **Wenn etwas am Cluster selbst kaputt wirkt** (Node down, Dateisystemfehler, fehlende GPU, Partition unerwartet im Drain) **nicht selbst umgehen/fixen** — dem Nutzer melden und auf den `ias-cluster`-Discord-Channel bzw. cluster@ias.informatik.tu-darmstadt.de verweisen.

## 🛠 Tech Stack & Environments

- **Core:** Python, PyTorch (Deep Learning), JAX (nur für B-Spline-Koordinatengenerierung via `bsplinax`).
- **Data:** SQLite3-Datenbanken (`ergodic_dataset_775.db`, `ergodic_pairs.db`), JSON für Dichte-Parameter.
- **Environments:**
  - Conda-Environment: `thesis` (lokal und auf dem Cluster identisch benannt).
  - Cluster-OS: Linux. ~20 GPU-Server (`cn01`–`cn14`, `gn01`–`gn11`, `dgx-station`), volle Hardware-/GPU-Tabelle im `ias-cluster`-Skill. Seit 2026-10-16 für Studierende alle Server nutzbar (vorher nur die `stud`-Partition), weiterhin max. 3 GPUs gleichzeitig.

## 🔄 Cluster & Development Workflow

1. **Lokale Entwicklung:** Code wird lokal editiert und getestet.
2. **Sync zum Cluster:**
   ```bash
   rsync -av --progress file.py stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/
   ```
3. **Ausführung:** Jobs werden über SLURM-Skripte eingereicht (`run_job_particles.bash`, `run_job_ergodic.bash`, ...), ausschließlich per `sbatch` (siehe Sicherheitsregeln oben).
   - Conda vor dem Training aktivieren: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate thesis`.
   - `srun` in diesen Skripten mit `--unbuffered` aufrufen (`srun --unbuffered python ...`), sonst bleiben `tqdm`-Fortschrittsbalken (und andere häufige Prints) in der `.err`-Datei stecken — Ursache ist SLURMs eigene Pipe-Pufferung zwischen `slurmstepd` und der Applikation, die auch bei explizitem Python-seitigem Flush (oder `-u`/`PYTHONUNBUFFERED=1` allein) bestehen bleibt.
4. **Monitoring:** Training wird nach Weights & Biases (W&B) geloggt — zuverlässiger als das Mitlesen von `.out`/`.err`, weil unabhängig von SLURM-I/O-Pufferung. Stdout/Stderr zusätzlich in `slurm-JOBID.out` bzw. `flow_particles-JOBID.err`.
5. **Ergebnisse holen:** Checkpoints/Visualisierungen per `rsync` zurück auf den lokalen Rechner.

## 🎨 Visualization Aesthetics (Strict Guidelines)

All trajectory plotting must adhere to this unified, clean style:
- **Background:** White (`facecolor='white'`).
- **Axes & Grid:** Light grey grid (`alpha=0.2`), grey spines (`#ccc`), dark text (`#1A1A2E`, `#555`).
- **Target Density (Heatmap):**
  - Use custom `WHITE_INFERNO` colormap (white fading into standard inferno).
  - Overlay with `alpha=0.55` so it remains pale and background-like.
- **Ground Truth Trajectories:** Deep Blue (`#1565C0`), `lw=2.5`, `alpha=0.9`.
- **Generated Trajectories:** Bright Neon Green (`#00C853`).
  - First trajectory: `lw=2.2`, `alpha=0.95`.
  - Subsequent trajectories (if multiple): `alpha=0.3`.
- **Conditioning Particles:** Dark dots (`#444444`), `s=6`, `alpha=0.3`.

## ⚠️ Important Pitfalls & Conventions

- **Device Management:** Always explicitly cast tensors (like holdout particles/spectrals) to the correct GPU `device` before passing to the model. `RuntimeError: Expected all tensors to be on the same device` is a common risk.
- **Dimensionality:** Pay attention to batch dimensions during CFG (Classifier-Free Guidance) generation. Trajectories usually have shape `[Batch, Nxi, Nd]`.
- **`generate_particle_trajectories`-Batch-Mismatch (bereits gefixt):** Wird ein bereits 3D-vorgebatchter `particles`-Tensor übergeben (z. B. via `.unsqueeze(0)`), griff die alte `if particles.ndim == 2`-Expansion auf `num_samples` nicht → Batch-Size-Mismatch zwischen Partikel- und Masken-Tensor (`RuntimeError: size of tensor a (2) must match size of tensor b (10)`). Fix in `flow_matching_cond_particles_crossattn.py` prüft zusätzlich `particles.shape[0] == 1`. Bei ähnlichen Fehlern in den Waypoint-/Spectral-Varianten zuerst hier nachsehen.
- **SLURM-Ausgabepufferung:** siehe Workflow-Abschnitt oben — `srun --unbuffered` nicht vergessen, sonst wirkt ein Job "hängend", obwohl er auf der GPU aktiv rechnet (immer erst mit `nvidia-smi`/W&B verifizieren, bevor man einen Job als hängend einstuft).
- **Checkpoint Resilience:** Runner scripts are designed to automatically detect and resume from the latest checkpoint in `checkpoints/` to handle cluster job time limits (e.g., 12h or 24h limits).
- **Nachverfeinerung = Suns FM-Stein (seit 2026-10-06):** `SvgdRefiner` (`exploration/common/svgd_refine.py`) nutzt standardmäßig `backend='sun'` = derselbe Kern wie die Datengenerierung (`ergodic_solver.py`), implementiert in `exploration/common/sun_refine.py` (Batch: `svgd_batched.BatchedSunTorch`). Der alte TSVEC-artige SVGD bleibt über `--refiner tsvec` bzw. `THESIS_REFINER=tsvec` erreichbar; Sun-Läufe bekommen den Ausgabe-Zusatz `_sun`, alte Ordner behalten ihre Namen. `n_iters` zählt bei Sun FM-Stein-Iterationen (Ground Truth: 600).
- **Standalone Visualization:** Use `visualize_checkpoint.py` to regenerate holdout plots from any `.pt` file without needing to spin up the full training loop.
