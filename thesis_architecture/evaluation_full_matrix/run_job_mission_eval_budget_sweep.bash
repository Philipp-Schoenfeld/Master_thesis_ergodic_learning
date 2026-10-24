#!/bin/bash
#SBATCH -J mission_bsweep
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120
#SBATCH --array=0-5

# Budget-sweep: the SAME experiment as the original mission_eval_20261006 run
# (full 25-shape x 3-condition x 3-strategy x 3-method matrix, standard
# one-length-unit-per-round replanning, unmodified run_mission_eval.py) but
# at a FIXED global SVGD budget per task instead of 1500 -- answers "how fast
# does CFM's head-start over the baselines shrink as the budget grows?"
# (Empfehlung B, Warmstart-Diagnose-Artefakt). Each array task is its own
# independent SLURM job instance with its own GPU (respects the 1-GPU-per-job
# rule); all 6 can run concurrently if the cluster has capacity.
#
# --svgd_conv is NOT meaningful here (low budgets like 10 iterations leave no
# room for its 100-iteration look-ahead window) -- only the standard
# executions-to-threshold box plots and the plain continuous E/J/swept plots
# are generated. A separate small script can later plot "median executions
# vs. budget" across all 6 (+ the existing 1500-iteration run) once this data
# exists -- that cross-budget curve is the actual payoff figure; this job
# only produces the per-budget raw data and the standard per-run plots.
#
# Timing: UNKNOWN for budget <1500 -- not yet measured. CFM planning
# (~6.7s/shape/round) is per-round, not per-iteration, so it does not shrink
# with the budget; SVGD cost scales with n_iters so should shrink a lot at
# the low end. Whether fewer SVGD iterations needs MORE replanning rounds to
# reach 99% (worse-aimed but still exactly 1-unit-long segments) is also
# unmeasured. 20h cap is a safe upper bound reusing the original run's
# --time_budget_h 22 margin, not a real estimate -- recommend a small smoke
# test (e.g. --shapes A,organic_10 --n_shapes 2) before trusting this for
# the full matrix.

NITERS=(10 50 100 250 500 1000)
N=${NITERS[$SLURM_ARRAY_TASK_ID]}
OUT_TAG=mission_eval_budget_sweep_${N}iters_20261012
PLOT_TAG=${OUT_TAG}_sun     # run_suffix(--refiner sun) default

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) budget sweep n_iters=${N} -> results/${PLOT_TAG}"
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --n_iters ${N} \
    --methods cfm,random_walk,linear \
    --workers 2 \
    --parallel_sets 4 \
    --time_budget_h 19
rc=$?
echo "[job] $(date) run (n_iters=${N}) finished with exit code ${rc}"

echo "[job] $(date) plots"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${PLOT_TAG} --box_only \
    --box_thresholds 0.75,0.8,0.9,0.95,0.99
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} \
    --max_units 8 --workers 4
