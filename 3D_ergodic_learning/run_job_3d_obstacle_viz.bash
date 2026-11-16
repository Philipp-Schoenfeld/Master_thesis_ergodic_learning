#!/bin/bash
#SBATCH -J 3d_obs_viz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:40:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=16G
#SBATCH -c 2

# Exports the interactive obstacle viewer (viz/obstacle_viewer.html): holdout shapes,
# paths without obstacle / with trained conditioning / with the inference-time
# constraint / both, for three sphere positions each.
#
#   sbatch run_job_3d_obstacle_viz.bash                       # newest obs630unetft checkpoint
#   CKPT=checkpoints/xyz.pt N_ITEMS=8 sbatch run_job_3d_obstacle_viz.bash

N_ITEMS=${N_ITEMS:-12}
DB3D=${DB3D:-ergodic_dataset_3d_lm.db}
OUT=${OUT:-viz/obstacle_viewer.html}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg
CKPT=${CKPT:-$(ls -t checkpoints/obs3d_*obs630unetft*_ep*.pt | head -1)}
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  ckpt: $CKPT  start: $(date)"

srun --unbuffered python -u export_obstacle_viz.py --ckpt "$CKPT" --db3d "$DB3D" \
    --n_items "$N_ITEMS" --steps 100 --n_gen 4 --out "$OUT"
echo "done: $(date)"
