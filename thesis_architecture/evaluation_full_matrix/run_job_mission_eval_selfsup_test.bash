#!/bin/bash
#SBATCH -J mission_selfsup_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Test run: adds 'selfsup' (the self-supervised single-pass generator,
# selfsup_planner.SelfsupPlanner) as a fourth init method for the replanning
# missions (run_mission_eval.py), ADDITIVE to the existing mission_eval_20261006
# run (cfm/random_walk/linear, 27 sets, ~22h, see run_job_mission_eval.bash).
#
# --refiner tsvec is REQUIRED here: the project default switched to 'sun' on
# 2026-10-06 (output dir gets a '_sun' suffix), but mission_eval_20261006 is an
# "old", unsuffixed tsvec run. Without --refiner tsvec this job would create a
# *new* directory mission_eval_20261006_sun instead of adding to the existing
# one -- see the memory note on refiner naming / CLAUDE.md.
#
# Small/local self-tests already passed before this job exists (CPU, dry_run
# and the real checkpoint, see test_mission_eval.py::test_mission_real_selfsup):
# at n_iters=100 the selected candidate's start_gap was ~0.17-0.38 (selfsup has
# no start conditioning, unlike cfm, so it needs refinement iterations to close
# the gap -- see selfsup_planner.py's docstring), down to ~0.01-0.08 at 500.
# This run uses the production n_iters=1500, well past where that was still
# shrinking, so start_gap is expected to be small; worth a quick look at the
# resulting table/plot regardless.
#
# Scope: 1 knowledge state x 1 strategy x --methods selfsup only, 3 rounds --
# gives seconds/round for selfsup (compare against the ~6.7 s/shape/round CFM
# planning bottleneck documented in run_job_mission_eval.bash: selfsup replaces
# that forward pass with a single cheap batched call, so it should be faster,
# but SVGD/Sun refinement itself -- the actual bottleneck for every method --
# costs the same regardless of init). Writes into the EXISTING mission_eval_20261006
# shards directory (resumable), only the new selfsup shard is touched.

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[test] $(date) selfsup mini-run (half_known/lse, 3 rounds)"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec \
    --methods selfsup \
    --conditions half_known \
    --strategies lse \
    --max_rounds 3 \
    --workers 3 \
    --parallel_sets 2
rc=$?
echo "[test] $(date) mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/shards/half_known__lse__selfsup.db
