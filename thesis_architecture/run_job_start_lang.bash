#!/bin/bash
#SBATCH -J start_lang
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Langer Lauf: dieselbe Konfiguration, volle 500 Epochen.
# 500 ist die Epochenzahl der bisher besten Laeufe, nicht auf die Wanduhr
# gerechnet. Reicht das 24-Stunden-Limit nicht, setzt der Folgejob fort.
#
# Daten: ergodic_dataset_start.db, 1187 Formen (775 Bestand + 400 flache
# + 12 flache Holdouts im eigenen Split val_flat). Startpunkte gleichverteilt
# ueber die Flaeche statt in der Ecke.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

ARGS="--D 384 --n_particles 256 --copies_per_char 75 --p_flip 0.0 \
      --epochs 500 --mini_batch 256 --save_every 10 --viz_every 10 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --tag L500 --keep_checkpoints 1 --db ergodic_dataset_start.db --use_wandb"

LATEST=$(ls -t checkpoints/*_START_FLAT*_L500_*_ep*.pt 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starte langen Lauf (Startpunkt-Konditionierung, CFM+ErgLoss, 500 Epochen)"
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
