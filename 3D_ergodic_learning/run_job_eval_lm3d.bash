#!/bin/bash
#SBATCH -J eval_lm3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=16G
#SBATCH -c 2

# Holdout comparison of the two finished 3D lambda-630 runs (DiT-B, U-Net D512)
# against the previous production 3D model (nooffset200, U-Net D384, Fourier
# ErgLoss lambda 100) on the val_form / val_flaeche / val_beides splits of the
# LM DB. A 3-entry smoke pass runs first so a broken checkpoint/label fails in
# a minute instead of at the end of the full pass.
#
#   sbatch run_job_eval_lm3d.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

DIT=$(ls checkpoints/lm3d_*lm630dit_*_ep0750.pt | head -1)
UNET=$(ls checkpoints/lm3d_*lm630unet_*_ep0750.pt | head -1)
OLD=$(ls checkpoints/nooffset200_*_ep0300.pt | head -1)
echo "DiT:  $DIT"; echo "UNet: $UNET"; echo "Old:  $OLD"
[ -f "$DIT" ] && [ -f "$UNET" ] && [ -f "$OLD" ] || { echo "checkpoint missing"; exit 1; }

MODELS=("DiT-B (lambda 630)=$DIT" "U-Net D512 (lambda 630)=$UNET" "U-Net D384 nooffset200 (lambda 100)=$OLD")

echo "=== smoke pass ==="
srun --unbuffered python eval_lm3d_compare.py --models "${MODELS[@]}" \
    --per_split 3 --n_gen 2 --steps 10 --n_examples 1 \
    --out_dir results/lm3d_compare_smoke || exit 1

echo "=== full pass ==="
srun --unbuffered python eval_lm3d_compare.py --models "${MODELS[@]}" \
    --per_split 100 --n_gen 4 --steps 100 --n_examples 3 \
    --out_dir results/lm3d_compare
echo "end: $(date)"
