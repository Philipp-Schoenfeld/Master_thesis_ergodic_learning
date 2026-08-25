#!/bin/bash
#SBATCH -J surf_kurz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 05:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Kurzer Lauf auf der projizierten 3D-Datenbank — Frist Dienstag 11 Uhr.
#
# Die Epochenzahl ist an die *tatsaechlich* verfuegbare Hardware angepasst, nicht
# an die schnellste im Cluster. Der Scheduler sagt den Start fuer Dienstag 05:36
# auf cn02 voraus, und cn02 traegt eine RTX 2080. Gemessen wurde auf einer
# RTX 3080 Ti: 13,2 s je Epoche (Job 134978). Fuer die 2080 ist mit rund dem
# Dreifachen zu rechnen, also etwa 45 s.
#
# Fenster: 05:36 bis 11:00 sind 5,4 Stunden; abzueglich Anlauf und
# Checkpoint-Schreibzeit bleiben knapp 4,8. Bei 45 s je Epoche sind das 384 —
# gesetzt sind 350, damit der Kosinus-Plan auch dann auslaeuft, wenn die
# Schaetzung um ein Fuenftel danebenliegt.
#
# Landet der Job auf einer schnelleren Karte, ist er frueher fertig. Das ist der
# bewusste Handel: ein Lauf mit ausgelaufenem Lernratenplan ist mehr wert als
# einer, den das Zeitlimit mittendrin abschneidet.
#
#   sbatch run_job_3d_kurz.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning

TAG="surfA_kurz"
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Falls ein frueherer Versuch abgebrochen ist: den letzten Stand aufnehmen.
RES=$(ls -t "$CKPT_DIR"/*_${TAG}_*_ep*.pt 2>/dev/null | head -1)
RES_ARG=""
if [ -n "$RES" ]; then
    echo "Fortsetzung ab: $(basename "$RES")"
    RES_ARG="--resume $RES"
fi

srun --unbuffered python -u flow_matching_runner_particles.py \
    --db3d ergodic_dataset_3d.db \
    --run_tag "$TAG" \
    $RES_ARG \
    --orientation --rot_full \
    --lambda_erg 100 --erg_K 6 --erg_on position --erg_t_power 2.0 \
    --lambda_ori 0.0 --w_cfm_rot 1.0 \
    --D 384 --n_particles 512 --nxi 25 \
    --copies_per_char 1 --mini_batch 64 --lr 1e-4 \
    --save_every 20 --keep_checkpoints 1 \
    --viz_every 100 \
    --save_model "$CKPT_DIR/surf.pt" \
    --use_wandb \
    --epochs 350

echo "Endzeit:    $(date)"
