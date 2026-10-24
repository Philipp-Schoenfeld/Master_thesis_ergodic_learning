#!/bin/bash
#SBATCH -J ebm_rankingnce_train
#SBATCH -t 1-00:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
##SBATCH -C 'vram24gb|vram32gb|vram48gb'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# Full training run of the Ranking-NCE energy-based policy (particle-cloud +
# start-point conditioning) on ergodic_dataset_start_plus.db -- the second
# generative head in the methods comparison (src/methods/), alongside CFM.
# No -p needed (`main` is the default partition since `stud` closed on
# 2026-10-16); stays on default QOS (not --qos=scavenger) since this is one
# real run under the 3-GPU/user limit, not a sweep -- see CLAUDE.md. Default
# QOS now allows up to 3 days, but in practice 500 epochs took ~1h46 in the
# calibrated run (~12s/epoch), so 24h remains a generous margin.
# Checkpointed every --save_every epochs (see checkpoint_rotation.py: exactly
# one checkpoint per run segment survives) so a timeout/preemption is
# recoverable via `sbatch --dependency=afterany:<JOBID>
# run_job_ebm_particles_start.bash --resume <path_to_latest_ep####.pt>`.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/src/methods/EBM_RankingNCE
mkdir -p checkpoints

srun --unbuffered python3 -u train_ebm_particles_start.py \
    --epochs 500 --mini_batch 256 --save_every 20 \
    --use_wandb --wandb_project flow-matching \
    "$@"
