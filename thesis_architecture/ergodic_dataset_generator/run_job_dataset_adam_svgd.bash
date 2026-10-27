#!/bin/bash
#SBATCH -J ds_adam
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Full Adam-SVGD training DB (generate_dataset_adam_svgd.py): the 1172 base
# densities of ergodic_dataset_improved_final.db + their #unk0 large-area
# variant = 2344 densities, 3 start points each (source x0 + 2 uniform random,
# --n_starts 3) = 7032 trajectories, linear init from x0 through the centre,
# Adam-SVGD on 128 raw waypoints with a hard start pin, stop when the ergodic
# error plateaus (<1% over 1000 it), at most 10000 iterations; the waypoint
# curve is then projected onto a B-spline with NXI control points (control
# point 0 = x0), which is what the DB stores and the CFM trains on.
#
# Set from run_job_dataset_adam_svgd_test.bash (job 181119, RTX A6000):
#   waypoints -> nxi 40 (study_waypoint_projection.py, same 48 densities):
#           ergodic error (K=10) 2.66e-4, bending 443, curvature p95 22.7 --
#           smoother AND at least as ergodic as every direct B-spline solve
#           (best direct: nxi 50, 2.83e-4 / 699 / 24.9; nxi 25: 3.48e-4 / 1852).
#           The projection reproduces the waypoint curve to ~6e-4 at nxi 40
#           (nxi 50 adds nothing, nxi 32 starts to filter: 2.85e-4).
#   --lr 8e-3 (Adam step, default 2e-3), study_waypoint_options.py on the same
#           48 densities: ergodic error 2.41e-4 (lower for 60% of them),
#           bending 304 (-31%, smoother for 67%), cusp-like hairpins in 6%
#           instead of 15% of the curves, ~30% fewer iterations. A stronger
#           smoothness force (x2..x8, polish phase) only trades ergodicity for
#           smoothness, an extra jerk force made both worse.
#   float64: float32 gave identical results, the time difference is negligible.
#   -t: density grids ~9 min; 200 iterations over all 7032 trajectories took
#       0.6 min -> mean ~4000 iterations ~15-35 min; validation (grids again,
#       runner check, 32-density sensitivity re-solve) ~30 min. ~1.5 h, 4 h
#       leaves room for a slower card.
#
# Resumable: rows already in the output DB are skipped, so a timeout or
# preemption only needs the same sbatch again. Same resources as the test job.
#
#   sbatch run_job_dataset_adam_svgd.bash

NXI=40
DTYPE=float64

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
srun --unbuffered python -u generate_dataset_adam_svgd.py \
    --solve_space waypoints --T 128 --nxi ${NXI} --lr 8e-3 --compute_dtype ${DTYPE} \
    --out ergodic_dataset_adam_svgd_nxi${NXI}.db
ls -la ergodic_dataset_adam_svgd_nxi${NXI}.db

# Gate for the afterok-chained training: validate_adam_svgd_db.py exits 1 on any
# FAIL (schema, completeness, splits, leakage, start pin, recomputed ergodic
# error, runner compatibility), so a broken DB never reaches the 3-day run.
srun --unbuffered python -u validate_adam_svgd_db.py \
    --db ergodic_dataset_adam_svgd_nxi${NXI}.db \
    --out results_adam_svgd_validation_nxi${NXI} --sensitivity 32
RC=$?
echo "validation exit code ${RC}  end: $(date)"
exit ${RC}
