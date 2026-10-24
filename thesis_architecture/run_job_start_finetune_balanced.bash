#!/bin/bash
#SBATCH -J start_ft_bal
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Pilot-Finetune auf ergodic_dataset_improved_final.db mit den zwei Fixes
# gegen die im FTSHORT15x200-Lauf gemessene Plain-Shape-Regression (siehe
# compare_unknown_structured_results.json: dort old=0.0044 vs. new=0.0067 auf
# den 24 Plain-Zieldichten, +54.4%, waehrend die 48 Unknown-Varianten nur
# +4.3% zeigten):
#
#   1. --lr 1e-5 statt der vollen Pretraining-LR 1e-4 -- CosineAnnealingLR
#      wird bei --finetune_from frisch aufgesetzt (Optimizer/Scheduler-State
#      geht NICHT vom Quell-Checkpoint mit), volle LR auf einem bereits
#      konvergierten Checkpoint ist ein Rezept fuer katastrophisches
#      Vergessen.
#   2. --plain_oversample_factor 2.0 -- gleicht das 1:2-Verhaeltnis
#      Plain:Unknown-Varianten aus (n_variants=2 in generate_dataset_unknown.py)
#      wieder auf ~1:1 Gradientenmasse pro Epoche aus (siehe
#      _build_dataset in flow_matching_runner_start.py).
#
# Nur 30 Epochen (statt sofort 500/FTIMPROVED): bei ~45-50 Min/Epoche (siehe
# run_job_start_finetune_short.bash) ca. 22-25h, passt in ein 24h-Fenster.
# Ausgangspunkt ist bewusst wieder der ORIGINALE L500-Checkpoint, nicht der
# bereits in die falsche Richtung gelaufene FTSHORT15x200-Stand.
#
# Nach Abschluss Validierungs-Gate fahren, BEVOR ein teurer 500-Epochen-Lauf
# committed wird:
#   python compare_unknown_structured_cluster.py \
#       --new_ckpt checkpoints/..._FTBAL30_..._final.pt --tag pilot_ftbal30
# Kriterium: Plain-Regression muss deutlich kleiner als die +54.4% von
# FTSHORT15x200 sein, Unknown-Varianten mindestens auf altem Niveau.
#
#   sbatch run_job_start_finetune_balanced.bash
#   sbatch --dependency=afterany:<job_id> run_job_start_finetune_balanced.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

FINETUNE_FROM="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"

ARGS="--D 384 --n_particles 256 --copies_per_char 5 --p_flip 0.0 \
      --epochs 30 --mini_batch 256 --save_every 10 --viz_every 10 \
      --lr 1e-5 --plain_oversample_factor 2.0 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --tag FTBAL30 --keep_checkpoints 1 \
      --db ergodic_dataset_improved_final.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_FLAT*_FTBAL30_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_FLAT*_FTBAL30_*_ep*.pt 2>/dev/null | head -1)

if [ -f "$FINAL_DONE" ]; then
    echo "Lauf bereits abgeschlossen ($FINAL_DONE) -- ueberzaehliger Kettenglied-Job, nichts zu tun."
elif [ -f "$LATEST" ]; then
    echo "Setze Finetune-Lauf fort von: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starte Finetuning von: $FINETUNE_FROM"
    srun --unbuffered python flow_matching_runner_start.py --finetune_from "$FINETUNE_FROM" $ARGS
fi
