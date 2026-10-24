#!/bin/bash
#SBATCH -J mission_budget_fx_smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Cluster smoke test of the FULL-EXECUTION variant of run_mission_eval_budget_matched.py
# (--full_execution is on by default there): each replanning round now drives the
# WHOLE refined candidate (CFM ~3.6, linear ~2.0, random_walk ~7.2 length units
# median, measured on the fixed-1500 run) instead of trimming to exactly one length
# unit before replanning -- "immer eine gesamte Ausfuehrung ... danach update und
# replan" (Philipp, 2026-10-11). Budget matching itself (CFM's own plateau sets the
# round's SVGD cap for all three methods) is unchanged.
#
# Already validated locally (CPU, --dry_run): missions now reach 99% in far fewer
# ROUNDS (~6 instead of ~15+), since each round covers more ground; the
# executions-to-threshold box plots switch automatically (config.json's
# full_execution flag) to measuring DRIVEN DISTANCE instead of round count, since
# round count is no longer a fair cross-method comparison once round widths differ
# by method. Also validated: plot_mission_continuous.py's round windows and the
# svgd_conv right axis now use each round's actual [start, end) distance, not its
# round index.
#
# Small scope on purpose, mirrors run_job_mission_eval_budget_matched_smoke.bash:
# 4 shapes, 1 condition, 1 strategy, real CFM checkpoint + GPU, 2h cap.

OUT_TAG=mission_eval_budget_matched_fullexec_smoke_20261011
PLOT_TAG=${OUT_TAG}_sun     # run_suffix(--refiner sun) default, see the smoke job script

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) budget-matched full-execution smoke run -> results/${PLOT_TAG}"
srun --unbuffered python -u run_mission_eval_budget_matched.py \
    --out_tag ${OUT_TAG} \
    --shapes A,organic_10,digit_5,rand_gmm_10 \
    --conditions none_known \
    --strategies eid \
    --max_rounds 15 \
    --cfm_run_iters 600 \
    --conv_window 100 \
    --conv_tol 0.05 \
    --workers 3 \
    --time_budget_h 1.5
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${PLOT_TAG} --box_only \
    --box_thresholds 0.75,0.9,0.95,0.99
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} --max_units 20 --workers 3 \
    --svgd_conv --conv_window 100 --conv_tol 0.05 --n_iters 600
