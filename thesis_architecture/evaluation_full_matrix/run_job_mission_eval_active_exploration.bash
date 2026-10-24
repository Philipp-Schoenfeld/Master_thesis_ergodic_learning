#!/bin/bash
#SBATCH -J mission_ae_full
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Full active-exploration run (Philipp's request, 2026-10-20): ONE solver per
# submission (pass it as $1 -- `sbatch -J mission_ae_full_sun run_job_
# mission_eval_active_exploration.bash sun`), so the 3 solvers (sun, tsvec=
# Adam-SVGD, tsvec_svgd=new paper-faithful solver) can run as up to 3
# concurrent cluster jobs under Philipp's 3-GPU limit instead of serially --
# roughly a 3x wall-clock cut over one combined job. All 25 holdout shapes x
# 4 knowledge conditions (ground_truth -- the oracle reference the gap-to-
# oracle metric needs, plus half_known/ten_samples/none_known) x 2
# strategies (ucb, eid) x 6 init methods (cfm, selfsup, ebm, linear,
# random_walk, heuristic -- dfl excluded, see the plan) = 48 mission sets
# for this solver.
#
# -t sized from two real timing probes (jobs 174766 toy-scale correctness,
# 174835 real n_init=30/n_iters=1500 scale): at production particle count,
# tsvec_svgd turned out to cost the SAME as tsvec/Adam-SVGD (77s vs 77s for
# 6 sets x 2 shapes x 1 round) -- the feared O(n_init^2) joint-swarm /
# sequential-per-shape penalty never materialises at n_init=30, both are
# ~3.2x faster than `sun` (244s for the same workload, CPU/JAX-bound as
# historically known). Extrapolated from the existing mission_eval_20261006
# production baseline (36 sets, sun only, full 25-shape replanning: 9.1h
# measured) scaled to this run's 48 sets: sun ~11-13h, tsvec/tsvec_svgd
# ~3.5-4h each. Submit with an explicit `sbatch -t HH:MM:SS -J <name>` CLI
# override per solver (takes precedence over the placeholder below) --
# sun: -t 16:00:00, tsvec/tsvec_svgd: -t 06:00:00 each.
#
# Resumable per (shape, round) like every other run in this harness --
# `--time_budget_h` stops OPENING NEW ROUNDS a safety margin before the wall
# clock limit, so the job exits cleanly instead of a mid-round SIGTERM kill
# (which is still data-safe -- every round commits its own SQLite
# transaction -- just wastes whatever partial round was in flight). A
# follow-up job via `sbatch --dependency=afterany:<JOBID> run_job_mission_
# eval_active_exploration.bash <refiner> [time_budget_h]` (same OUT_TAG)
# picks up exactly where this one stopped.
#
# BUG FIXED 2026-10-21 (found after the first submission, jobs 175376-78):
# this script used to not propagate `srun`'s exit code, so a job genuinely
# killed by its own walltime (exit 143/SIGTERM) still showed as SLURM
# "COMPLETED" with code 0:0 -- indistinguishable from an actually-finished
# run in `sacct` without reading the .out tail. `--time_budget_h` was also
# left at a stale placeholder (46h) irrespective of the job's real `-t`, so
# it never triggered before the hard kill either. Both jobs 175377/175378
# (tsvec/tsvec_svgd, -t 6h) were in fact only ~12/48 sets in when SIGTERM'd;
# `$2` now lets a resubmission set a `--time_budget_h` that actually matches
# whatever `-t` is passed on the `sbatch` command line.

REFINER="${1:?usage: sbatch run_job_mission_eval_active_exploration.bash sun-or-tsvec-or-tsvec_svgd [time_budget_h]}"
TIME_BUDGET_H="${2:-22}"
OUT_TAG=mission_active_exploration_20261020

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) refiner=${REFINER} time_budget_h=${TIME_BUDGET_H} -> results/${OUT_TAG} (+ run_suffix)"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner ${REFINER} \
    --conditions ground_truth,half_known,ten_samples,none_known \
    --strategies ucb,eid \
    --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
    --workers 6 \
    --parallel_sets 4 \
    --time_budget_h ${TIME_BUDGET_H}
rc=$?
echo "[job] $(date) refiner=${REFINER} run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"
exit ${rc}
