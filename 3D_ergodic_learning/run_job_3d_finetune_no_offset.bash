#!/bin/bash
#SBATCH -J surf_ft_nooff
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 08:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Feinjustierung des surfB_lang-Checkpoints (trainiert mit standoff=0.12) auf der
# offsetfreien Datenbank (standoff=0.0, siehe project_db_3d.py). Warm-Start statt
# Neutraining: gleiche Architektur/Konfiguration wie run_job_3d_lang.bash, nur die
# Zielpositionen liegen jetzt exakt auf der Flaeche statt 0.12 davor.
#
# Ablauf:
#   1. Eigener Fortschritt hat Vorrang: existiert schon ein Checkpoint dieses
#      Laufs (TAG), wird per --resume fortgesetzt (Optimizer/Scheduler/Epoche
#      inklusive).
#   2. Sonst: der letzte surfB_lang-Checkpoint wird per --load_model geladen —
#      nur die Gewichte, mit frischem Optimizer/Scheduler und Epoche 0. Genau
#      das ist der Warm-Start.
#   3. Kein Checkpoint gefunden: Kaltstart (sollte hier nicht vorkommen).
#
# Deutlich weniger Epochen als der 1750-Epochen-Lauf, weil das Netz nur die neue
# Standoff-Verschiebung lernen muss, nicht die Aufgabe von Grund auf. Falls die
# Holdout-Metriken (siehe HANDOFF_no_offset_db.md) nach 400 Epochen noch klar
# unterlegen sind: mit --run_tag "$TAG" nachlegen, laeuft automatisch weiter.
#
#   sbatch run_job_3d_finetune_no_offset.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning

TAG="surfB_nooffset_ft"
WARM_TAG="surfB_lang"
CKPT_DIR="checkpoints"
DB="ergodic_dataset_3d_no_offset.db"
mkdir -p "$CKPT_DIR"

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$DB" ]; then
    echo "FEHLER: $DB fehlt — erst manuell ueber den persoenlichen Drive hierher kopieren."
    exit 1
fi

RES=$(ls -t "$CKPT_DIR"/*_${TAG}_*_ep*.pt 2>/dev/null | head -1)
RES_ARG=""
LOAD_ARG=""
if [ -n "$RES" ]; then
    echo "Setze eigenen Lauf fort ab: $(basename "$RES")"
    RES_ARG="--resume $RES"
else
    WARM=$(ls -t "$CKPT_DIR"/*_${WARM_TAG}_*_ep*.pt 2>/dev/null | head -1)
    if [ -n "$WARM" ]; then
        echo "Warm-Start von: $(basename "$WARM")"
        LOAD_ARG="--load_model $WARM"
    else
        echo "Kein surfB_lang-Checkpoint gefunden — Kaltstart."
    fi
fi

srun --unbuffered python -u flow_matching_runner_particles.py \
    --db3d "$DB" \
    --run_tag "$TAG" \
    $RES_ARG $LOAD_ARG \
    --orientation --rot_full \
    --lambda_erg 100 --erg_K 6 --erg_on position --erg_t_power 2.0 \
    --lambda_ori 0.0 --w_cfm_rot 1.0 \
    --D 384 --n_particles 512 --nxi 25 \
    --copies_per_char 1 --mini_batch 64 --lr 3e-5 \
    --save_every 20 --keep_checkpoints 1 \
    --viz_every 100 \
    --save_model "$CKPT_DIR/surf_nooffset_ft.pt" \
    --use_wandb \
    --epochs 400

echo "Endzeit:    $(date)"
