#!/bin/bash
#SBATCH -J start_adam_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Timing test for the 3-day training (run_job_start_adam_svgd.bash): the exact
# training config (D384, 256 particles, batch 256, CFM + Sinkhorn ErgLoss
# w1300, --target_cps --start_inpaint), but only 3 epochs at 10 copies per
# density, with one checkpoint save and one holdout visualisation at the end,
# so all three costs are measured.
#
# Data: ergodic_dataset_adam_svgd_timing.db from run_job_dataset_adam_svgd_test.bash
# (2344 densities x 3 starts = 7032 trajectories, nxi 25, only 200 solver iterations -- the trajectories
# are not converged, which does not matter for timing: same rows, same shapes,
# same control-point column). Submit with a dependency on that job.
#
# Read off: tqdm s/ep in the .err file -> s/step = (s/ep) / ceil(N_train*10/256),
# the startup time ("Precomputing ... density grids") and the viz time from the
# timestamps below. Checkpoints go to checkpoints/test_timing/ and are deleted
# at the end; the holdout PNGs (Trajectory_data_generator/*_TIMING_*) stay as a
# check that sampling with --start_inpaint works end to end.
#
#   sbatch --dependency=afterok:<DS_TEST_JOBID> run_job_start_adam_svgd_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
T0=$(date +%s)
srun --unbuffered python -u flow_matching_runner_start.py \
    --nxi 25 --D 384 --n_particles 256 --copies_per_char 10 --p_flip 0.0 \
    --epochs 3 --mini_batch 256 --save_every 3 --viz_every 3 \
    --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
    --target_cps --start_inpaint --tag TIMING --keep_checkpoints 1 \
    --db ergodic_dataset_adam_svgd_timing.db \
    --save_model checkpoints/test_timing/cond_particles_crossattn.pt
echo "[train test] $(( $(date +%s) - T0 )) s total  end: $(date)"
rm -rf checkpoints/test_timing
