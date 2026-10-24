#!/bin/bash
#SBATCH -J eval_erg_probe
#SBATCH -t 00:20:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Timing probe only, NOT the real evaluation: 2 shapes x 2 trajectories,
# n_iters=50, to measure real ms/candidate-iter for BatchedSunTorch's
# run_batch (pure JAX, CPU-only in this env -- confirmed repeatedly in
# earlier job logs: "CUDA-enabled jaxlib is not installed" even on GPU
# nodes) before sizing the real job's time limit and CPU count.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
CKPT="checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_10_16_14h49min_S256_nxi25_D384_C75_flip0.0_final.pt"
srun --unbuffered python evaluation_full_matrix/eval_dfl_ergodicity_svgd.py \
    --checkpoint "$CKPT" --n_per_shape 2 --n_iters 50 --shapes_limit 2 \
    --out eval_results/probe.json
