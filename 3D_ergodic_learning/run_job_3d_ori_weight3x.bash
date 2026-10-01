#!/bin/bash
#SBATCH -J 3d_ori_w3x
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Feinjustierung der aktuellen 200-Flaechen-Version (run_job_3d_no_offset_200.bash,
# RUN_TAG=nooffset200) mit auf das Dreifache angehobenem Orientation-Objective-
# Gewicht (--lambda_ori 0.012 -> 0.036). Alles andere bleibt exakt wie im
# Basislauf: dieselbe Datenbank, dieselbe Architektur, dieselben inneren
# Gewichte von point/standoff/angsmooth (--w_point/--w_standoff/--w_angsmooth) -
# angehoben wird ausschliesslich das Gesamtgewicht des Objective-Terms selbst.
#
# --init_model statt --resume: das ist ein neuer, eigenstaendiger Lauf mit
# geaenderter Zielfunktion, kein Kettenglied des Basislaufs. Mit --resume
# wuerden Optimierer- und Scheduler-Zustand des Basislaufs mitkommen -
# Adams Momentum/Varianz sind dann auf die alte (position-lastige)
# Loss-Landschaft eingeschwungen, und CosineAnnealingLR.load_state_dict()
# ueberschreibt T_max wieder mit dem alten Wert, wodurch ein hoeheres
# --epochs die Lernrate NICHT neu anheben wuerde (sie bliebe bei eta_min
# haengen). --init_model startet dagegen mit frischem Optimierer/Scheduler
# und uebernimmt nur die Gewichte.
#
# Der Init-Checkpoint wird NICHT fest verdrahtet, sondern beim Kaltstart per
# ls -t aus den nooffset200-Checkpoints ermittelt (== "die aktuelle
# 3D-Version" zum Zeitpunkt, an dem dieser Job zum ersten Mal laeuft).
#
# Ressourcen: gleiche Modellgroesse (D=384, 512 Partikel) wie der Basislauf,
# deshalb dieselbe Anforderung (16G/-c2) uebernommen statt neu zu schaetzen -
# nach dem ersten Kettenglied mit
#   sacct -j <ID> -o JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem
# pruefen und bei Bedarf fuer Folgeglieder nachziehen.
#
# Zeitbudget: 100 Epochen sind ein Drittel des Basislaufs (300 Epochen) -
# selbst auf dem langsamsten beobachteten Knoten (dgx-station/V100, ~111
# Epochen/24h beim vergleichbaren flaechen200-Lauf) passt das in ein
# einzelnes 24h-Glied. Die Resume-Kette unten bleibt trotzdem stehen, falls
# der Job auf einem langsameren Knoten landet oder ein Zwischenstopp noetig
# wird.
#
# Folgejob anhaengen:
#   sbatch --dependency=afterany:<JOBID> run_job_3d_ori_weight3x.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

RUN_TAG=${RUN_TAG:-oriw3x}
DB3D=${DB3D:-ergodic_dataset_3d_no_offset_200.db}

# ── Quelle: der aktuelle nooffset200-Checkpoint (Basislauf, lambda_ori=0.012) ─
BASE_PAT="checkpoints/nooffset200_flow3d_particle_ergodic_nooffset200_*.pt"
INIT=$(ls -t $BASE_PAT 2>/dev/null | head -1)

if [ -z "$INIT" ] || [ ! -f "$INIT" ]; then
    echo "Kein nooffset200-Checkpoint gefunden (Muster: $BASE_PAT) - Abbruch."
    echo "Lief run_job_3d_no_offset_200.bash bereits auf diesem Cluster-Konto?"
    exit 1
fi
echo "Init-Checkpoint (aktuelle 3D-Version): $INIT"

# ── Eigene Kette: ab dem zweiten Glied wird das HIER begonnene Fine-Tuning
# fortgesetzt (--resume), nicht wieder vom Basislauf aus warmgestartet.
PAT="checkpoints/${RUN_TAG}_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

if [ ! -f "$LATEST" ]; then
    echo "=== Kaltstart: Testsuite ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

    echo
    echo "=== Kaltstart: Warmstart-Pruefung ==="
    # Kein --erwartet: pruefe_warmstart.py vergleicht dann gegen den im
    # Checkpoint selbst gespeicherten Verlust (ckpt['loss']) statt gegen eine
    # hier hart hinterlegte Zahl, die zum jeweils aktuellen Checkpoint passen
    # muesste. Zielfunktion und Datenbank entsprechen exakt dem Basislauf
    # (run_job_3d_no_offset_200.bash), NICHT der neuen 3x-Gewichtung - die
    # Pruefung stellt sicher, dass der Checkpoint unveraendert das liefert,
    # womit er zuletzt trainiert wurde, bevor mit neuer Zielfunktion
    # weitertrainiert wird.
    srun --unbuffered python pruefe_warmstart.py \
        --init_model "$INIT" --db3d "$DB3D" \
        --db_splits train \
        --batches 80 --mini_batch 32 \
        --erg_on footprint --lambda_erg 100 --erg_K 6 --erg_pts 128 \
        --erg_t_power 2.0 --w_cfm_rot 0.5 \
        --orientation --lambda_ori 0.012 --w_point 0.1 --w_standoff 300 \
        --w_angsmooth 2.0 --standoff_target 0.12 --standoff_band 0.03 \
        || { echo "Warmstart-Pruefung fehlgeschlagen — Abbruch."; exit 1; }
fi

if [ -f "$LATEST" ]; then
    echo "Setze eigenes Fine-Tuning fort von: $LATEST"
    START_ARG="--resume $LATEST"
else
    echo "Kaltstart, Warmstart aus: $INIT"
    START_ARG="--init_model $INIT"
fi

echo
echo "=== Training: Orientation-Objective-Gewicht x3 (0.012 -> 0.036), 100 Epochen ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

srun --unbuffered python flow_matching_runner_particles.py \
    $START_ARG \
    --db3d "$DB3D" \
    --run_tag "$RUN_TAG" \
    --save_model "checkpoints/${RUN_TAG}.pt" \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0.1 \
    --erg_on footprint \
    --lambda_erg 100 --erg_K 6 --erg_pts 128 --erg_t_power 2.0 \
    --lambda_ori 0.036 --w_cfm_rot 0.5 \
    --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --D 384 --n_particles 512 --grid_res 64 \
    --copies_per_char 1 --p_flip 0.0 \
    --lr 2e-5 --warmup_epochs 10 --lr_min 1e-6 \
    --epochs 100 --mini_batch 128 \
    --save_every 10 --viz_every 20 \
    --n_holdout_viz 10 \
    --use_wandb --wandb_project flow3d-surfaces
