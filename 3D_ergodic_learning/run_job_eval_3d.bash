#!/bin/bash
#SBATCH -J eval3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Auswertung aller 3D-Checkpoints auf den Holdout-Formen.
# Checkpoint-Pfade unten anpassen, bevor der Job eingereiht wird.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

srun --unbuffered python evaluate_models.py \
    --checkpoints \
        checkpoints/selfsup_3d_selfsup3d_XXX_final.pt \
        checkpoints/cond_particles_3d_flow3d_XXX_final.pt \
    --labels "Selbstueberwacht 3D" "CFM 3D" \
    --n_samples 8 --steps 100 \
    --grid_res 64 --erg_K 8 \
    --out metrics_3d \
    --visualize
