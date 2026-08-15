#!/bin/bash
#SBATCH -J particles_12h
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@60

# Config: C8 (copies_per_char=8), N64 (n_particles=64) -> ~10x fewer compute per epoch
# Expected: ~500 epochs in 12h

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

LATEST_CKPT=$(ls -t checkpoints/*flow_matching_particle_ergodic_date_*_nxi25_D384_N64_C8_flip0.0_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Setze Training fort von: $LATEST_CKPT"
    srun --unbuffered python flow_matching_runner_particles.py \
        --resume "$LATEST_CKPT" \
        --D 384 \
        --n_particles 64 \
        --copies_per_char 8 \
        --p_flip 0.0 \
        --epochs 500 \
        --mini_batch 256 \
        --save_every 50 \
        --viz_every 50 \
        --use_wandb
else
    echo "Starte neues Training..."
    srun --unbuffered python flow_matching_runner_particles.py \
        --D 384 \
        --n_particles 64 \
        --copies_per_char 8 \
        --p_flip 0.0 \
        --epochs 500 \
        --mini_batch 256 \
        --save_every 50 \
        --viz_every 50 \
        --use_wandb
fi
