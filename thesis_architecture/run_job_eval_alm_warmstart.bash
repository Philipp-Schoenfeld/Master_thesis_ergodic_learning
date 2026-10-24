#!/bin/bash
#SBATCH -J eval_alm_warmstart
#SBATCH -t 00:20:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# The actual Decision-Focused-Learning validation: ALM-solver outer_iters
# to feasibility, learned lambda0 vs. zero-init, in the style of
# rq6_alm_lambda0.py. Pure PyTorch (no JAX), so much cheaper than the SVGD
# eval -- should finish in well under 20 minutes.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
CKPT="checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_10_16_14h49min_S256_nxi25_D384_C75_flip0.0_final.pt"
srun --unbuffered python evaluation_full_matrix/eval_alm_warmstart_convergence.py \
    --checkpoint "$CKPT" --n_per_shape 30 --n_outer 30 \
    --out eval_results/alm_warmstart.json
