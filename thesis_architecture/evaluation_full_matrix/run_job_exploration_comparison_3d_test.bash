#!/bin/bash
#SBATCH -J expl_cmp_3d_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=16G
#SBATCH -c 4

# Timing / sanity test of the 3D exploration comparison at the REAL budget (5 rounds, 8 candidates,
# 500 SVGD iterations) on one shape, one variant, both kernels and every method -> seconds per
# mission per method (runs.seconds, timing_by_method.csv) and the networks' mean path length per
# round (seg_len_by_method.csv, to budget-match --walk_len). Own folder, deleted results do not
# touch the real DB.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
srun --unbuffered python -u run_exploration_comparison_3d.py \
    --n_shapes 1 --n_variants 1 \
    --out_dir plots/exploration_comparison_3d_flat_timingtest \
    --max_minutes 80 || exit 1
echo "[job] $(date) finished"
