#!/bin/bash
#SBATCH -J selfsup
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

# Stage 1: feasibility gate. One shape, one candidate, no diversity — a few
# minutes. Exits non-zero if the energy did not fall, so a broken objective
# never reaches the full run.
echo "=== Stage 1: feasibility ==="
srun --unbuffered python -u flow_matching_runner_particles_selfsupervised.py \
    --shapes A \
    --n_candidates 1 \
    --diversity_weight 0.0 \
    --epochs 200 \
    --D 128 \
    --n_particles 64 \
    --save_every 0 \
    --viz_every 0 \
    --assert_energy_drops || { echo "[ABORT] Feasibility failed."; exit 1; }

# Stage 2: full training over the 750-shape split.
# diversity_weight is set against the measured energy scale: the energy settles
# around 4-15 (the solver itself reaches 1.5-4 on these shapes) and the
# diversity reward saturates near 0.8, so weight 10 keeps the two terms in the
# same order of magnitude. At 100 the repulsion outweighs the energy 6:1.
echo "=== Stage 2: full training ==="
LATEST_CKPT=$(ls -t checkpoints/selfsup_selfsup_particles_*_K8_div10_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Setze Training fort von: $LATEST_CKPT"
    srun --unbuffered python -u flow_matching_runner_particles_selfsupervised.py \
        --resume "$LATEST_CKPT" \
        --D 384 \
        --n_particles 256 \
        --n_candidates 8 \
        --diversity_weight 10 \
        --epochs 500 \
        --mini_batch 32 \
        --save_every 20 \
        --viz_every 20 \
        --use_wandb
else
    echo "Starte neues Training..."
    srun --unbuffered python -u flow_matching_runner_particles_selfsupervised.py \
        --D 384 \
        --n_particles 256 \
        --n_candidates 8 \
        --diversity_weight 10 \
        --epochs 500 \
        --mini_batch 32 \
        --save_every 20 \
        --viz_every 20 \
        --use_wandb
fi
