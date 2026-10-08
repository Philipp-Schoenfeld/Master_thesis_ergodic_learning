#!/bin/bash
#SBATCH -J start_ft_short
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Kurze erste Finetune-Phase, 200 Epochen -- fuer ein erstes Ergebnis, bevor
# spaeter der volle Lauf (copies_per_char=25, 500 Epochen, eigener Tag)
# folgt. Dieselbe Basis wie run_job_start_finetune_improved.bash, siehe
# dessen Kommentare fuer --finetune_from vs. --resume und die
# "bereits fertig"-Sperre.
#
# copies_per_char=5 statt der anfangs geplanten 15: gemessen skaliert die
# Epochenzeit NICHT proportional mit der Sample-Zahl (15 war mit ~41-50
# Min/Epoche kaum schneller als 75 mit ~49 Min/Epoche -- der Flaschenhals
# liegt woanders, vermutlich an der V100 auf dgx-station ohne BF16-Tensor-
# Cores). Eine weitere Senkung auf 5 kostet also vermutlich kaum zusaetzliche
# Zeit, liefert aber mehr Formenvielfalt je Gradientenschritt.
#
# --tag bleibt bewusst FTSHORT15x200 (nicht umbenannt): --resume findet den
# naechsten Checkpoint ueber das Tag-Glob, eine Umbenennung wuerde die
# Fortsetzung von Epoche 50 abreissen. Der Name ist ab jetzt nur noch
# historisch (erste 50 Epochen liefen mit 15, ab hier mit 5).
#
#   sbatch run_job_start_finetune_short.bash
#   sbatch --dependency=afterany:<job_id> run_job_start_finetune_short.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

FINETUNE_FROM="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"

ARGS="--D 384 --n_particles 256 --copies_per_char 5 --p_flip 0.0 \
      --epochs 200 --mini_batch 256 --save_every 10 --viz_every 10 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --tag FTSHORT15x200 --keep_checkpoints 1 \
      --db ergodic_dataset_improved_final.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_FLAT*_FTSHORT15x200_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_FLAT*_FTSHORT15x200_*_ep*.pt 2>/dev/null | head -1)

if [ -f "$FINAL_DONE" ]; then
    echo "Lauf bereits abgeschlossen ($FINAL_DONE) -- ueberzaehliger Kettenglied-Job, nichts zu tun."
elif [ -f "$LATEST" ]; then
    echo "Setze Finetune-Lauf fort von: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starte Finetuning von: $FINETUNE_FROM"
    srun --unbuffered python flow_matching_runner_start.py --finetune_from "$FINETUNE_FROM" $ARGS
fi
