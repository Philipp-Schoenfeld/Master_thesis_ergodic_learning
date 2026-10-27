#!/bin/bash
#SBATCH -J eval_lm_model
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:15:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Structured ~45 min test of the CFM trained on the teacher-study DB
# (run_job_start_lm.bash, run name ..._CPS_INPAINT_LMTEACHER_...), see the module
# docstring of eval_start_cfm_model.py:
#   A  generation quality on the holdout densities + their large-area variants,
#      stored AND unseen start points: new model vs. old L500 start model vs.
#      ground truth vs. chords (no solver), 10 samples per case
#   B  warm start: Levenberg-Marquardt (the teacher solver) from 30 initial curves
#      of new model / old model / chords, quality after 0..200 iterations
#   C  start conditioning: start error, effect of the start, 5x5 grid of starts
#
# The script keeps its own clock: part A may use 40% of --budget_min, part B stops
# starting new densities at 92%, so the run ends after ~45 min by itself whatever the
# GPU is (the sizes below fit an RTX-class card; -t has 30 min of margin for startup,
# JAX density grids and plotting). Debug QOS: short, starts first, max 4 h.
#
#   sbatch run_job_eval_lm_model.bash

CKPT_NEW=$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_10_24_21h47min_nxi25_D384_N256_C25_flip0.0_START_FLAT2400_CPS_INPAINT_LMTEACHER_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt
CKPT_OLD=$HOME/Master_thesis/transfer/netz2d_startpunkt.pt
DB=$HOME/Master_thesis/thesis_architecture/ergodic_dataset_generator/ergodic_dataset_lm_nxi25.db

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
ls -la "$CKPT_NEW" "$CKPT_OLD" "$DB" || exit 1
srun --unbuffered python -u eval_start_cfm_model.py \
    --ckpt "$CKPT_NEW" --old_ckpt "$CKPT_OLD" --db "$DB" \
    --out_tag lm_model_eval_20261027 \
    --n_samples 10 --steps 100 --n_warm 24 --n_cand 30 --budget_min 45
echo "exit code $?  end: $(date)"
