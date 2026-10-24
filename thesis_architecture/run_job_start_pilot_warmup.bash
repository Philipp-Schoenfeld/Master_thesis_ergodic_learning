#!/bin/bash
#SBATCH -J pilot_warm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 05:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Nachfolger von pilot_both (Job 167936, PILOTBOTH5): gleiche Fixes (lr=1e-5,
# plain_oversample_factor=2.0), zusaetzlich --lr_warmup_epochs 3 gegen den
# vermuteten AdamW-Cold-Start (PILOTBOTH5 hatte frischen Optimizer-State und
# wurde bei nur 5 Epochen ungewoehnlich gleichmaessig schlechter: ALL +32.7%,
# PLAIN +23.2%, UNKNOWN +35.8% -- siehe
# compare_unknown_structured_results_PILOTBOTH5.json).
#
# 50 statt 5 Epochen: PILOTBOTH5 brauchte auf einem schnellen Knoten (cn14)
# nur ~45-50s/Epoche, also ist mehr Budget jetzt billig -- und 5 Epochen
# waren zu wenig fuer ein belastbares Signal. Faellt der Job auf einen
# langsamen Knoten (dgx-station/V100, ~45-50 Min/Epoche), greift das
# uebliche Resume-ueber-Checkpoint-Schema automatisch beim naechsten
# sbatch-Aufruf dieses Skripts.
#
#   sbatch run_job_start_pilot_warmup.bash
#   sbatch --dependency=afterany:<job_id> run_job_start_pilot_warmup.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

TAG="PILOTWARM50"
FINETUNE_FROM="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"

ARGS="--D 384 --n_particles 256 --copies_per_char 5 --p_flip 0.0 \
      --epochs 50 --mini_batch 256 --save_every 10 --viz_every 10 \
      --lr 1e-5 --plain_oversample_factor 2.0 --lr_warmup_epochs 3 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --tag ${TAG} --keep_checkpoints 1 \
      --db ergodic_dataset_improved_final.db --use_wandb"

echo "Knoten: $(hostname)   Start: $(date)"

FINAL_DONE=$(ls checkpoints/*_START_FLAT*_${TAG}_*_final.pt 2>/dev/null | head -1)
if [ -f "$FINAL_DONE" ]; then
    echo "Training bereits abgeschlossen: $FINAL_DONE"
else
    LATEST=$(ls -t checkpoints/*_START_FLAT*_${TAG}_*_ep*.pt 2>/dev/null | head -1)
    if [ -f "$LATEST" ]; then
        echo "Setze Finetune-Lauf fort von: $LATEST"
        srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
    else
        echo "Starte Finetuning von: $FINETUNE_FROM"
        srun --unbuffered python flow_matching_runner_start.py --finetune_from "$FINETUNE_FROM" $ARGS
    fi
fi

FINAL_DONE=$(ls checkpoints/*_START_FLAT*_${TAG}_*_final.pt 2>/dev/null | head -1)
if [ -f "$FINAL_DONE" ]; then
    echo "Validierungs-Gate: vergleiche $FINAL_DONE gegen Baseline..."
    srun --unbuffered python compare_unknown_structured_cluster.py --new_ckpt "$FINAL_DONE" --tag ${TAG}
else
    echo "Kein finaler Checkpoint vorhanden -- Vergleich uebersprungen (Zeitlimit?)."
fi

echo "Ende: $(date)"
