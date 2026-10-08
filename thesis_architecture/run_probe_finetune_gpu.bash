#!/bin/bash
#SBATCH -J ft_probe
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Reiner Zeit-Probelauf: 5 Epochen mit der echten Produktionskonfiguration
# (D=384, nxi=25, copies_per_char=75, ErgLoss-Sinkhorn w=1300) auf der NEUEN,
# bereits gemergten Datenbank (ergodic_dataset_improved_final.db, 3516 Formen).
# Zweck ist ausschliesslich, echte Sekunden/Epoche auf einer Cluster-GPU mit
# der tatsaechlichen Ziel-DB zu messen, um daraus hochzurechnen, wie viele
# 24h-Jobs die 500 Epochen brauchen werden. Kein --use_wandb,
# Trainingsergebnis wird verworfen (eigener --tag GPUPROBE, kollidiert nicht
# mit dem echten Finetune-Lauf).
#
# 3h statt der urspruenglichen 30 Minuten: ein erster Versuch mit 30 Minuten
# lief ins Zeitlimit, OHNE dass auch nur eine einzige Epoche fertig wurde --
# allein das Vorberechnen der 128x128-Dichtegitter fuer 3480 Formen (JAX,
# jeder Aufruf kompiliert frisch) fraesst sich fast durch das ganze Budget.
# Fuer die echten 20h-Trainingsjobs ist das vernachlässigbar, fuer einen
# knappen Probelauf war es toedlich.
#
#   sbatch run_probe_finetune_gpu.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

echo "Knoten: $(hostname)   Start: $(date)"

FINETUNE_FROM="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"

srun --unbuffered python flow_matching_runner_start.py \
    --finetune_from "$FINETUNE_FROM" \
    --D 384 --n_particles 256 --nxi 25 --copies_per_char 75 --p_flip 0.0 \
    --epochs 5 --mini_batch 256 --save_every 1000 --viz_every 1000 \
    --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
    --tag GPUPROBE --keep_checkpoints 1 \
    --db ergodic_dataset_improved_final.db

echo "Ende: $(date)"
