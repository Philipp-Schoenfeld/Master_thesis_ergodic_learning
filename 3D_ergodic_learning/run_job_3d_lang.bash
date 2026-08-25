#!/bin/bash
#SBATCH -J surf_lang
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 23:30:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Langer Lauf, in 11-Stunden-Bloecke zerlegt — Frist Mittwoch 14 Uhr.
#
# Beide Bloecke teilen sich die Laufkennung `surfB_lang` und die Gesamtzahl von
# 1750 Epochen. Die Zahl ist die *Gesamt*zahl, nicht die des Blocks: der
# Kosinus-Lernratenplan wird darueber gelegt, und der zweite Block setzt anhand
# des gespeicherten Epochenzaehlers dort fort, wo der erste aufhoerte. Stuende
# in jedem Block die halbe Zahl, fiele die Lernrate in der Mitte des Trainings
# einmal auf ihr Minimum und spraenge dann wieder hoch.
#
# Die gemeinsame Kennung sorgt zugleich dafuer, dass die Checkpoint-Rotation
# den Stand des ersten Blocks ersetzt statt ihn liegen zu lassen.
#
# **Was die Zerlegung bringt und was nicht.** Der Scheduler sagt fuer jede
# angeforderte Laufzeit denselben Start voraus — kuerzere Bloecke verschaffen
# also keinen frueheren Beginn. Was sie bringen: sie passen in Luecken, die der
# Backfill spaeter oeffnet, und je Block steht weniger auf dem Spiel, falls ein
# Knoten ausfaellt.
#
#   sbatch run_job_3d_lang1.bash
#   sbatch --dependency=afterany:<ID> run_job_3d_lang2.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning

TAG="surfB_lang"
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
    --epochs 1750

echo "Endzeit:    $(date)"
