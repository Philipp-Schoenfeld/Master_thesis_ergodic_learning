#!/bin/bash
#SBATCH -J ergodic_crossattn_50c
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL

# Conda-Umgebung laden
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

LATEST_CKPT=$(ls -t checkpoints/*flow_matching_ergodic_date_*_S256_nxi25_D384_C50_flip0.0_ep*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Kein Checkpoint gefunden. Starte komplett neues Training..."
    python flow_matching_runner_ergodic.py --S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 50 --epochs 750 --lr 2e-4 --use_wandb --wandb_project "flow-matching"
else
    echo "Setze Training fort von: $LATEST_CKPT"
    python flow_matching_runner_ergodic.py --S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 50 --epochs 750 --lr 2e-4 --use_wandb --wandb_project "flow-matching" --resume "$LATEST_CKPT"
fi
