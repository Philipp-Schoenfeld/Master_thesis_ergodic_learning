#!/bin/bash
#SBATCH -J ebm_holdout_eval
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

# Holdout evaluation of the trained EBM checkpoint on the 49 val-split
# shapes: ranking accuracy, Langevin/CEM refinement quality, and a
# qualitative figure. CPU-only on purpose -- this is a handful of forward
# passes over 49 shapes, not training, so it needs no GPU and no QOS beyond
# the default (no -p needed either: `main` is the cluster-wide default
# partition since the `stud` partition closed on 2026-10-16).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/src/methods/EBM_RankingNCE
mkdir -p results

CKPT=$(ls -t checkpoints/*_final.pt | grep -v calib | head -1)
echo "Evaluating checkpoint: $CKPT"

srun --unbuffered python3 -u evaluate_holdout.py --checkpoint "$CKPT"
