#!/bin/bash
#SBATCH -J mission_ae_fullexec
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# --full_execution variant of the active-exploration run (Philipp's request,
# 2026-10-21, as a SEPARATE run alongside the one-length-unit-per-round runs
# in mission_active_exploration_20261020[*] -- both are kept, not replaced,
# as two different driving conventions to compare): drives each refined
# candidate in FULL (whatever length it naturally is, typically ~2-7 length
# units) instead of trimming to exactly one length unit before replanning.
# Same 48 mission sets/solver (4 knowledge conditions incl. the ground_truth
# oracle reference x 2 strategies x 6 methods), same resumability, same
# --refiner/--time_budget_h arguments as run_job_mission_eval_active_
# exploration.bash -- see that script's header for the method/solver/
# scope rationale, identical here except for `--full_execution`.
#
# Expected to need noticeably FEWER replanning rounds to reach 99% coverage
# per shape (each driven segment covers several length units instead of
# one), which is also expected to reduce wall-clock relative to the one-
# unit-per-round runs -- not yet measured at the time this was written, so
# -t is set to the same 24h default as those runs' continuation jobs rather
# than assuming the speedup; adjust after reading this run's own first
# hour of "SVGD batch Xs" / round-count lines.

REFINER="${1:?usage: sbatch run_job_mission_eval_active_exploration_fullexec.bash sun-or-tsvec-or-tsvec_svgd [time_budget_h]}"
TIME_BUDGET_H="${2:-22}"
OUT_TAG=mission_active_exploration_fullexec_20261021

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) refiner=${REFINER} time_budget_h=${TIME_BUDGET_H} --full_execution -> results/${OUT_TAG} (+ run_suffix)"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner ${REFINER} \
    --conditions ground_truth,half_known,ten_samples,none_known \
    --strategies ucb,eid \
    --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
    --workers 6 \
    --parallel_sets 4 \
    --time_budget_h ${TIME_BUDGET_H} \
    --full_execution
rc=$?
echo "[job] $(date) refiner=${REFINER} run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"
exit ${rc}
