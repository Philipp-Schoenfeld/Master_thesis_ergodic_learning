#!/bin/bash
#SBATCH -J eval_dfl_ckpt
#SBATCH -t 00:45:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p interactive
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'

# Holdout evaluation on the current checkpoint (epoch 433/500), independent
# of whether/when the remaining epochs finish -- generation-only, no
# training. 45 min budget: the in-training holdout viz pass alone was
# observed to take ~10-15 min on top of a normal epoch, generous margin here
# since this job does nothing else.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

CKPT=$(ls -t checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_*_S256_nxi25_D384_C75_flip0.0_ep*.pt 2>/dev/null | head -1)
echo "Evaluating checkpoint: $CKPT"
srun --unbuffered python eval_dfl_checkpoint.py --checkpoint "$CKPT" --tag mid_run_eval
