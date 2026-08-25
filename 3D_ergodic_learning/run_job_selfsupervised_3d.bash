#!/bin/bash
#SBATCH -J selfsup3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Selbstueberwachtes 3D-Training gegen die Solver-Energie.
# Gestaffelt: erst ein kurzer Machbarkeitstest als Torwaechter, dann das
# volle Training. Faellt die Energie im Test nicht, bricht das Skript ab,
# statt 20 GPU-Stunden zu verbrennen.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

echo "=== Phase 1: Machbarkeitstest (wenige Formen, wenige Epochen) ==="
srun --unbuffered python flow_matching_runner_particles_selfsupervised.py \
    --shapes A,organic_0,digit_5 \
    --n_candidates 1 --diversity_weight 0.0 \
    --epochs 60 --gate_epochs 40 --assert_energy_drops --min_drop_ratio 0.5 \
    --D 128 --n_particles 128 --grid_res 48 --erg_K 6 \
    --mini_batch 3 --save_every 0 --viz_every 60 \
    || { echo "Machbarkeitstest fehlgeschlagen — Abbruch."; exit 1; }

echo
echo "=== Phase 2: volles Training ==="
LATEST=$(ls -t checkpoints/selfsup_3d_selfsup3d_*_ep*.pt 2>/dev/null | head -1)
RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
fi

srun --unbuffered python flow_matching_runner_particles_selfsupervised.py \
    $RESUME \
    --n_train_shapes 750 \
    --n_candidates 8 --diversity_weight 10 \
    --epochs 500 \
    --D 384 --n_particles 512 --grid_res 64 --erg_K 8 \
    --mini_batch 32 \
    --save_every 20 --viz_every 20 \
    --use_wandb
