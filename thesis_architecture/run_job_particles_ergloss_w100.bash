#!/bin/bash
#SBATCH -J flow_erg_w100
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Ablation Fortsetzung: w=100 — doppelt so hoch wie das bisherige Optimum
# Die Reihe w=2..50 war monoton fallend ohne Saettigung (E_erg 4.25 -> 2.80),
# w=50 lag praktisch auf Solver-Niveau. Dieser Lauf sucht den Kipppunkt.
# Hyperparameter identisch zur ersten Reihe, nur Zeitlimit 18h -> 20h,
# damit die 500 Epochen erreicht werden (erste Reihe endete bei ~486).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

WEIGHT=100
LATEST_CKPT=$(ls -t checkpoints/*flow_matching_particle_ergodic_date_*_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w${WEIGHT}-K8-tp2_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Setze Training fort von: $LATEST_CKPT"
    srun --unbuffered python flow_matching_runner_particles.py \
        --resume "$LATEST_CKPT" \
        --D 384 \
        --n_particles 256 \
        --copies_per_char 15 \
        --p_flip 0.0 \
        --epochs 500 \
        --mini_batch 256 \
        --save_every 20 \
        --viz_every 20 \
        --lambda_erg $WEIGHT \
        --erg_K 8 \
        --erg_pts 128 \
        --erg_t_power 2.0 \
        --use_wandb
else
    echo "Starte neues Training..."
    srun --unbuffered python flow_matching_runner_particles.py \
        --D 384 \
        --n_particles 256 \
        --copies_per_char 15 \
        --p_flip 0.0 \
        --epochs 500 \
        --mini_batch 256 \
        --save_every 20 \
        --viz_every 20 \
        --lambda_erg $WEIGHT \
        --erg_K 8 \
        --erg_pts 128 \
        --erg_t_power 2.0 \
        --use_wandb
fi
