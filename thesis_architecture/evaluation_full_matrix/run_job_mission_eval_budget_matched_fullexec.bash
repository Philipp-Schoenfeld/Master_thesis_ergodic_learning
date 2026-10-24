#!/bin/bash
#SBATCH -J mission_budget_fx
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Full-matrix budget-matched FULL-EXECUTION replanning-mission run
# (run_mission_eval_budget_matched.py, --full_execution on by default): same
# scope as run_job_mission_eval_budget_matched.bash (25 holdout shapes x 3
# knowledge states x 3 target-density strategies, CFM's own SVGD plateau sets
# the round's iteration budget for all three methods) except each replanning
# round now drives the WHOLE refined candidate instead of trimming to exactly
# one length unit -- "immer eine gesamte Ausfuehrung ... danach update und
# replan" (Philipp, 2026-10-11).
#
# Validated on a 4-shape/1-group cluster smoke test (job 166727, 2026-10-11):
# exit 0, no errors, all three consistency checks < 2e-4, all plots written.
# That smoke test needed only 8 rounds (vs ~30+ under the one-length-unit
# regime) and finished in 6.8 min; scaling shapes 4 -> 25 (~linear, same as
# before) gives ~42 min for that one (eid/none_known) group -- of the 9
# groups, 'lse'/'ucb' were ~1.5-1.9x slower than 'eid' in the previous
# (non-full-execution) budget-matched run (mission_eval_budget_matched_20261008_sun,
# 13h53m total), so expect roughly 6-11 h total here, well inside the 22h
# budget and the 24h hard cluster limit -- almost certainly one job, no
# --dependency=afterany chaining needed, but nothing stops a resume (same
# --out_tag) if it does.
#
# --max_units 28 (not 8): the smoke test's worst mission needed ~24 length
# units of DRIVEN DISTANCE to reach 99% (full_execution rounds are longer, so
# missions finish in far fewer ROUNDS but the total distance covered is
# similar to before) -- 8 units would cut most missions off mid-plot.

OUT_TAG=mission_eval_budget_matched_fullexec_20261011
PLOT_TAG=${OUT_TAG}_sun     # run_suffix(--refiner sun) default, see the smoke job script

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full budget-matched full-execution run -> results/${PLOT_TAG}"
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
srun --unbuffered python -u plot_mission_eval.py --out_tag ${PLOT_TAG} --box_only \
    --box_thresholds 0.75,0.8,0.9,0.95,0.99
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} \
    --max_units 28 --workers 4 --svgd_conv --conv_window 100 --conv_tol 0.05 --n_iters 600
