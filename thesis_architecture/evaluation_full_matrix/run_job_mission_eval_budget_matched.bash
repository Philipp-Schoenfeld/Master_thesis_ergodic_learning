#!/bin/bash
#SBATCH -J mission_budget
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Full-matrix budget-matched replanning-mission run (run_mission_eval_budget_matched.py):
# same scope as the original mission_eval_20261006 (run_job_mission_eval.bash)
# -- 25 holdout shapes x 3 knowledge states x 3 target-density strategies --
# except the SVGD budget per replanning round is not a fixed 1500 iterations
# for every method, but set each round by how many iterations CFM itself
# needs to plateau (see run_mission_eval_budget_matched.py docstring);
# random_walk and linear get capped to that same number in that round.
#
# cfm_run_iters=600 (not the original 1500): generous relative to the
# conv_window=100 look-ahead and the ~310-340 median plateau iteration
# measured on the fixed-1500 run (mission_eval_20261006/analysis/
# analyze_stdout.txt / mission_continuous-*.out), carried over from the
# smoke-test run (run_job_mission_eval_budget_matched_smoke.bash) -- revisit
# this value if that smoke test's own printed "conv plateau" stats land
# close to 600 (not enough headroom) or far below it (wasted budget).
#
# UNLIKE run_job_mission_eval.bash, this run does not overlap mission groups
# on the GPU (no --parallel_sets equivalent: BudgetGroup.run() couples cfm's
# own round to the baselines' capped round within one group, which the
# original's cooperative round-robin scheduler was not built for) -- it
# processes the 9 (condition, strategy) groups strictly one after another.
# Expect meaningfully fewer total SVGD iterations than the original run
# (roughly 600 + two sub-1500 capped budgets per candidate instead of
# 3 x 1500) but less GPU/CPU overlap; net wall-clock vs. the original ~22 h
# is uncertain until calibrated against the smoke test's measured per-round
# timing -- if one job does not finish within 24 h, resume with
# `sbatch --dependency=afterany:<JOBID> run_job_mission_eval_budget_matched.bash`
# (same --out_tag, --time_budget_h stops opening new rounds before the limit
# so the job ends cleanly; each (shape, round) is resumable per the shard DB,
# same as run_mission_eval.py).

OUT_TAG=mission_eval_budget_matched_20261008
PLOT_TAG=${OUT_TAG}_sun     # run_suffix(--refiner sun) default, see the smoke job script

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full budget-matched run -> results/${PLOT_TAG}"
srun --unbuffered python -u run_mission_eval_budget_matched.py \
    --out_tag ${OUT_TAG} \
    --conditions half_known,ten_samples,none_known \
    --strategies lse,ucb,eid \
    --cfm_run_iters 600 \
    --conv_window 100 \
    --conv_tol 0.05 \
    --workers 3 \
    --time_budget_h 22
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots/tables"
# --box_only: without it plot_mission_eval.py also writes svgd_convergence_round*.png,
# which assumes every candidate's E_series has the SAME length (true only for a fixed
# iteration budget) -- crashes on this run's per-shape/per-round variable budgets. Not
# one of the requested plots anyway, so just skip it (confirmed on the smoke test).
srun --unbuffered python -u plot_mission_eval.py --out_tag ${PLOT_TAG} --box_only \
    --box_thresholds 0.75,0.8,0.9,0.95,0.99
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} \
    --max_units 8 --workers 4 --svgd_conv --conv_window 100 --conv_tol 0.05 --n_iters 600
