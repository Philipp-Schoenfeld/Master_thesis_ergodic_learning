#!/bin/bash
#SBATCH -J eval_erg_full
#SBATCH -t 01:00:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Full ergodicity + SVGD-convergence validation of the DFL-trained final
# checkpoint: 25 holdout shapes x 30 generated trajectories each, SVGD
# warm-start (generated) vs. cold-start (random walk) over 600 iterations.
# Time budget sized from two timing probes (jobs 174545, 174546): run_batch
# scales from 8.4ms/candidate-iter at C=8 down to 1.2ms/candidate-iter at
# C=50 (JAX vmap amortizes), so C=1500 x 600 iters should take roughly
# 20-30 min; 1h leaves a real but not wasteful margin for generation +
# overhead, not a blind 24h default.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
CKPT="checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_10_16_14h49min_S256_nxi25_D384_C75_flip0.0_final.pt"
srun --unbuffered python evaluation_full_matrix/eval_dfl_ergodicity_svgd.py \
    --checkpoint "$CKPT" --n_per_shape 30 --n_iters 600 \
    --out eval_results/dfl_final_eval.json
