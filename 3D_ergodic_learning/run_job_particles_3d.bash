#!/bin/bash
#SBATCH -J flow3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Ueberwachtes 3D-Flow-Matching auf den (in die Ebene gehobenen) DB-Trajektorien.
# Achtung: solange die Datenbank planar ist, imitiert dieses Modell planare
# Bahnen. Das ist Absicht — es ist die Kontrollgruppe, gegen die sich das
# selbstueberwachte 3D-Modell messen laesst.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

LATEST=$(ls -t checkpoints/cond_particles_3d_flow3d_*_ep*.pt 2>/dev/null | head -1)
RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
fi

srun --unbuffered python flow_matching_runner_particles.py \
    $RESUME \
    --D 384 --n_particles 512 --grid_res 64 \
    --copies_per_char 15 --p_flip 0.0 \
    --epochs 500 --mini_batch 128 \
    --save_every 20 --viz_every 20 \
    --lambda_erg 0.0 \
    --use_wandb
