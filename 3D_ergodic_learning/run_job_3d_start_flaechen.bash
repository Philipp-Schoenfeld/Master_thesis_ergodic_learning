#!/bin/bash
#SBATCH -J flow3d_startfl
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 22:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Feinabgleich: 79 Zielflaechen + Startpunkt-Konditionierung, warmgestartet
# aus dem Flaechen-Checkpoint (Epoche 1750).
#
# ── Ressourcen: gemessen, nicht geschaetzt ─────────────────────────────────
# --mem=12G:  Der Datenlader wurde lokal vermessen — 34 693 Trainingseintraege
#             brauchen 1,21 GB Spitzenspeicher, davon 284 MB der Partikelstapel
#             und 31 MB der Zieltensor. Dazu kommen Torch samt CUDA-Kontext
#             (~2,5 GB) und die Bilderzeugung. Rund 4 GB im Betrieb, 12 als
#             Puffer. Die 32G des Vorgaengerjobs waren nie belegt.
# -c 2:       Die Arbeit liegt auf der GPU. Auf der CPU bleibt der Schnitt des
#             Minibatches aus dem angehefteten Partikelstapel (1 MB je Schritt,
#             ueber DMA) und die Hauptschleife. Mehr Kerne beschleunigen davon
#             nichts und blockieren nur das eigene Kontingent.
#
# NACH DEM ERSTEN LAUF nachmessen und gegebenenfalls nachziehen:
#   sacct -j <ID> -o JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem
# CPU-Effizienz = TotalCPU / (Elapsed x AllocCPUS), Speicher = MaxRSS / ReqMem.
#
# ── Warum --init_model und nicht --resume ──────────────────────────────────
# --resume setzt einen Lauf fort und laedt Modell, Optimierer und Scheduler;
# daran haengt die Job-Kette weiter unten, und es bleibt unangetastet.
# --init_model faengt einen *neuen* Lauf an und nimmt nur die Gewichte mit,
# mit frischem AdamW und frischem Lernplan. Der Runner prueft dabei, dass
# ausschliesslich start_emb.* und null_start_token fehlen, und bricht sonst ab:
# ein stillschweigend halb geladenes 87,5-M-Netz waere der teuerste Fehler,
# den dieser Lauf machen kann.
#
# ── Wie lang eine Epoche ist, und warum das hier neu entschieden wurde ─────
# `--copies_per_char` hat seine urspruengliche Aufgabe verloren. Frueher lag
# jeder Eintrag so oft im Zieltensor, damit eine Epoche mehr Augmentierungen
# sah — bei 775 Formen billig, bei 34 693 Eintraegen eine halbe Million Zeilen
# im Speicher. Der Runner vervielfacht nichts mehr; die Augmentierung ist
# ohnehin bei jedem Zug neu, und der Wert legt jetzt allein die Epochenlaenge
# fest.
#
# Damit ist er frei waehlbar, und die sinnvolle Wahl ist 1: eine Epoche ist ein
# gewichteter Durchlauf durch die Daten, 34 693 Beispiele. Bei den gemessenen
# ~15 ms je Beispiel sind das 8,7 Minuten — 152 Epochen passen in ein
# 22-h-Glied, die 250 Epochen also in zwei. Mit `copies_per_char 3` waere eine
# Epoche 26 Minuten lang gewesen, und die 50 Warmup-Epochen haetten den ganzen
# ersten Job aufgebraucht.
#
# Die 250 Epochen sind an der Sache begruendet, nicht an der Wanduhr: ein
# Feinabgleich soll die 79 neuen Flaechen sehen, nicht das Training von vorn
# machen. 250 Durchlaeufe sind rund 8,7 Millionen Beispiele — in derselben
# Groessenordnung wie die 9,2 Millionen des gesamten ep1750-Laufs, und der
# fing bei zufaelligen Gewichten an.
#
# ── Warum Warmup ───────────────────────────────────────────────────────────
# Das Netz kommt aus 1750 Epochen auf sieben Flaechen und sieht jetzt 79, von
# denen es die meisten nie gesehen hat. Ohne Anlauf bekaeme es im ersten
# Schritt die volle Lernrate auf einen Gradienten, der mit dem Gelernten wenig
# zu tun hat. 50 Epochen linear hoch auf 2e-5, danach Cosine auf 1e-6.
#
# ── Warum --erg_on footprint Pflicht ist ───────────────────────────────────
# Sobald ein Standoff und der Orientierungsterm aktiv sind, verlangt ein auf
# der Position gemessener ergodischer Term, in der Zielflaeche zu bleiben,
# waehrend der Standoff verlangt, sie zu verlassen. Am Fussabdruck gemessen
# wollen beide dasselbe. Der Runner bricht ohne --erg_on footprint mit einer
# Erklaerung ab, und das zu Recht.
#
# Folgejob anhaengen (so oft wie noetig):
#   sbatch --dependency=afterany:<JOBID> run_job_3d_start_flaechen.bash
# Das Folgeglied findet seinen Checkpoint ueber den Run-String selbst und
# setzt dann mit --resume fort, nicht mehr mit --init_model.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

RUN_TAG=${RUN_TAG:-startfl22}
DB3D=${DB3D:-ergodic_dataset_3d.db}
INIT=${INIT:-checkpoints/surf_flow3d_particle_ergodic_surfB_lang_nxi25_D384_N512_R64_C1_flip0.0_SURF_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2_ep1750.pt}

# Der Run-String, den der Runner selbst bildet — daran haengt das Wiederfinden
# des eigenen Checkpoints in der Kette.
PAT="checkpoints/startfl_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

if [ ! -f "$LATEST" ]; then
    echo "=== Kaltstart: Testsuite ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

    echo
    echo "=== Kaltstart: Warmstart-Pruefung ==="
    # Belegt, dass die 87,5 M Gewichte wirklich geladen werden, bevor 22 h
    # darauf gesetzt werden. Geprueft wird unter der Zielfunktion des
    # geladenen Laufs auf dessen eigener Datenbank — mit der Zielfunktion des
    # Feinabgleichs waere der Vergleich sinnlos, weil es eine andere ist.
    srun --unbuffered python pruefe_warmstart.py \
        --init_model "$INIT" --db3d ergodic_dataset_3d_alt.db \
        --batches 40 --mini_batch 32 \
        || { echo "Warmstart-Pruefung fehlgeschlagen — Abbruch."; exit 1; }
fi

if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    START_ARG="--resume $LATEST"
else
    echo "Kaltstart, Warmstart aus: $INIT"
    START_ARG="--init_model $INIT"
fi

echo
echo "=== Training: 79 Flaechen + Startpunkt, 22 h ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

srun --unbuffered python flow_matching_runner_particles.py \
    $START_ARG \
    --db3d "$DB3D" \
    --run_tag "$RUN_TAG" \
    --save_model checkpoints/startfl.pt \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0.1 \
    --mix ebene=0.25 --ebene_flach_anteil 0.5 \
    --erg_on footprint \
    --lambda_erg 100 --erg_K 6 --erg_pts 128 --erg_t_power 2.0 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 \
    --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --D 384 --n_particles 512 --grid_res 64 \
    --copies_per_char 1 --p_flip 0.0 \
    --lr 2e-5 --warmup_epochs 50 --lr_min 1e-6 \
    --epochs 250 --mini_batch 128 \
    --save_every 10 --viz_every 20 \
    --n_holdout_viz 10 \
    --use_wandb --wandb_project flow3d-surfaces
