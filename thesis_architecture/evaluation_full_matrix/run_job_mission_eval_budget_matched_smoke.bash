#!/bin/bash
#SBATCH -J mission_budget_smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Cluster smoke test of run_mission_eval_budget_matched.py (coupled cfm /
# random_walk / linear missions: the SVGD budget each round is set by how
# many iterations CFM itself needs to plateau, and random_walk/linear are
# capped to that same number in that same round -- see the script's
# docstring). This is the REAL-checkpoint / GPU counterpart of the local
# `--dry_run` smoke test already validated (DummyPlanner, CPU, 2 shapes);
# this run exercises the actual CFM network and the real 'sun' SVGD solver.
#
# Small scope on purpose: 4 holdout shapes, 1 knowledge condition, 1
# strategy, max 6 rounds. cfm_run_iters=600 (not the full 1500) is still
# generous relative to the conv_window=100 look-ahead and the ~310-340
# median plateau iteration measured on the fixed-1500-iteration run
# mission_eval_20261006 (see analysis/analyze_stdout.txt there), so plateau
# detection has room without paying for the full 1500 the original run used.
#
# Resources: same shape as run_job_mission_eval.bash (CFM planning is the
# same per-candidate cost; SVGD itself should be cheaper here since random_walk
# and linear only run up to the CFM-derived cap, not a fixed 1500).

OUT_TAG=mission_eval_budget_matched_smoke_20261008
# run_mission_eval_budget_matched.py appends run_suffix(--refiner) (default
# 'sun' -> '_sun') to the output directory; the plot scripts take --out_tag
# literally (no suffix logic), so they need the already-suffixed name.
PLOT_TAG=${OUT_TAG}_sun

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) budget-matched smoke run -> results/${PLOT_TAG}"
srun --unbuffered python -u run_mission_eval_budget_matched.py \
    --out_tag ${OUT_TAG} \
    --shapes A,organic_10,digit_5,rand_gmm_10 \
    --conditions none_known \
    --strategies eid \
    --max_rounds 6 \
    --cfm_run_iters 600 \
    --conv_window 100 \
    --conv_tol 0.05 \
    --workers 3 \
    --time_budget_h 1.5
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots"
# --box_only: without it plot_mission_eval.py also writes svgd_convergence_round*.png,
# which assumes every candidate's E_series has the SAME length (true only for a fixed
# iteration budget) -- crashes on this run's per-shape/per-round variable budgets. Not
# one of the requested plots anyway, so just skip it.
srun --unbuffered python -u plot_mission_eval.py --out_tag ${PLOT_TAG} --box_only \
    --box_thresholds 0.75,0.9,0.95,0.99
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} --max_units 6 --workers 3 \
    --svgd_conv --conv_window 100 --conv_tol 0.05 --n_iters 600
