#!/bin/bash
#SBATCH -J waypoint_crossattn
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

LATEST_CKPT=$(ls -t checkpoints/*flow_matching_waypoint_date_*_ep*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Kein Checkpoint gefunden. Starte komplett neues Training..."
    python flow_matching_runner_waypoint.py --use_wandb --wandb_project "flow-matching"
else
    echo "Setze Training fort von: $LATEST_CKPT"
    python flow_matching_runner_waypoint.py --use_wandb --wandb_project "flow-matching" --resume "$LATEST_CKPT"
fi
