# Ergodic Flow-Matching Trajectory Generation (Master Thesis)

This project implements conditional flow-matching networks to generate B-Spline trajectories that ergodically cover target distributions. The target distributions are represented either via point-cloud particles or spectral coefficients. Downstream goal: learn a warm-start / constraint predictor for the TSVEC (Task-Space Stein Variational Ergodic Coverage) solver.

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

## 🖥 Cluster-Regeln & Ressourcen-Limits (IAS Cluster, `stud`-Partition)

- **Maximale Laufzeit:** 24 Stunden (`#SBATCH --time=24:00:00`). Längere Jobs werden abgelehnt.
- **GPUs:** maximal 1 GPU pro Job (`#SBATCH --gres=gpu:1`).
- **CPUs & RAM:** standardmäßig `-c 4` und `--mem=16G` bis `--mem=32G`.
- **Kein Cherry-Picking von GPU-Modellen:** `#SBATCH -C 'rtx3080|rtx3090|...'` ist auf `stud` **verboten** und lässt den Job nicht starten. Der `-C`-Parameter darf in keinem Skript aktiv vorkommen (in den bestehenden `run_job_*.bash`-Skripten ist die Zeile bewusst mit `##SBATCH` auskommentiert — so muss es bleiben).
- **Checkpoint & Restart:** Da Jobs maximal 24h laufen, muss regelmäßig gecheckpointet werden (`--save_every` niedrig genug wählen, damit mindestens ein Checkpoint sicher vor Ablauf des Zeitlimits entsteht).
- **SIGTERM nutzen:** `#SBATCH --signal=SIGTERM@120` warnt das Python-Skript 120s vor dem harten 24h-Limit, um einen finalen Notfall-Checkpoint auszulösen (siehe `TerminateInterrupt`-Handler in den Runnern).
- **Job-Ketten:** Folge-Jobs nach Ablauf des Zeitlimits werden mit `sbatch --dependency=afterany:<JOBID> script.bash` eingereiht.
- **Ausführungsverzeichnis:** SLURM startet im Verzeichnis, aus dem `sbatch` aufgerufen wurde — immer explizit per `cd` ins Projektverzeichnis wechseln (`cd ~/Master_thesis/thesis_architecture/`).
- **Live-Debugging eines laufenden Jobs:** `srun --jobid=<ID> --overlap nvidia-smi` bzw. `--pty bash` klinkt sich in eine bestehende Job-Allokation ein (nur nach expliziter Nutzer-Aufforderung, siehe Sicherheitsregeln oben).

## 🛠 Tech Stack & Environments

- **Core:** Python, PyTorch (Deep Learning), JAX (nur für B-Spline-Koordinatengenerierung via `bsplinax`).
- **Data:** SQLite3-Datenbanken (`ergodic_dataset_775.db`, `ergodic_pairs.db`), JSON für Dichte-Parameter.
- **Environments:**
  - Conda-Environment: `thesis` (lokal und auf dem Cluster identisch benannt).
  - Cluster-OS: Linux. Bekannte Nodes: `cn01`, `cn03`, `dgx-station`.

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
- **Standalone Visualization:** Use `visualize_checkpoint.py` to regenerate holdout plots from any `.pt` file without needing to spin up the full training loop.
