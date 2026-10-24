#!/bin/bash
#SBATCH -J mission_kappa_full
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Full run of the kappa-annealed, full-execution replanning test (Philipp's
# request, 2026-10-21) -- see run_job_mission_eval_kappa_schedule_test.bash's
# header for the full scope rationale (6 methods x 1 solver [tsvec_svgd] x
# 3 knowledge conditions x 'ucb' only x --max_rounds 5 x --kappas
# 2,1,0.5,0.1,0.05 x --full_execution = 18 mission sets, all 25 holdout
# shapes). DO NOT SUBMIT until the test job's measured per-round cost has
# been used to confirm/adjust -t below (8h is sized from the earlier 77s/
# 6-sets/2-shapes/1-round tsvec_svgd probe extrapolated to 18 sets x 25
# shapes x 5 (capped) rounds, not yet re-measured at THIS run's actual
# scope).
#
# Resumable per (shape, round) like every other run in this harness --
# `--time_budget_h` stops opening new rounds a safety margin before the wall
# clock limit. $1 optionally overrides it (default 7.5h).

TIME_BUDGET_H="${1:-7.5}"
OUT_TAG=mission_kappa_schedule_20261021

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) time_budget_h=${TIME_BUDGET_H} -> results/${OUT_TAG}_tsvec_svgd"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec_svgd \
    --conditions none_known,half_known,ten_samples \
    --strategies ucb \
    --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
    --max_rounds 5 \
    --kappas 2,1,0.5,0.1,0.05 \
    --full_execution \
    --workers 6 \
    --parallel_sets 4 \
    --time_budget_h ${TIME_BUDGET_H}
rc=$?
echo "[job] $(date) run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"
exit ${rc}
