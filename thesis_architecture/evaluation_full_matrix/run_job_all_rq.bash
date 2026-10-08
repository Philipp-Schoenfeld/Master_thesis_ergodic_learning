#!/bin/bash
#SBATCH -J all_rq
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120
##SBATCH -C 'rtx3080|rtx3090|a5000'
# Combined RQ1-RQ6 job (Sun comparison, see rq_experiments/README.md): runs
# all six research-question experiments as sequential steps in ONE job,
# sharing the single allocated GPU. RQ6 needs no GPU at all (pure JAX
# optimisation on 25x2 control points, ~1.3s/trial measured locally) but
# runs here too so the whole comparison is one job to submit and monitor.
#
# Scope per step (see the comment above each step for the local CPU-only
# measurement it is based on): sized so the WORST CASE -- no GPU speedup at
# all for the CFM-dependent steps (RQ1-5), i.e. the same per-trial cost
# measured locally without a GPU -- still finishes in ~5h, leaving large
# margin under the 24h limit. On the actual GPU node this should run
# considerably faster, since only the CFM forward pass benefits from the
# GPU: the LQR/Riccati solve inside every "Sun" method (`ergodic_solver
# ._build_lqr`, `dynamics_zoo.build`) is explicitly pinned to the CPU in all
# five rqN_*.py scripts, matching Sun's own "LQR solving on CPU is faster".
#
# Resumable: every rqN_*.py script now writes its CSV row-by-row with a
# flush after each trial and skips trials/combinations already on disk (see
# README.md's "Checkpoint & Restart" section) -- re-submitting this same
# script with the same OUT_TAG continues exactly where a killed/timed-out
# run left off, no manual bookkeeping needed. `--time_budget_h` on each step
# stops that step from starting a new trial once its share of the job's time
# budget is used up, so a slow step cannot starve the steps after it.
#
# Follow-up after a 24h kill: sbatch --dependency=afterany:<JOBID> run_job_all_rq.bash
#
# Resources measured locally (CPU only, see above): the heaviest step
# (RQ4, Sinkhorn divergence over a 128x256 cost matrix) used < 2GB RSS;
# -c 4 / --mem=24G leaves comfortable headroom for JAX's own overhead and the
# CFM network on top.
OUT_TAG=all_rq_20261007
TIME_BUDGET_PER_STEP_H=3.5    # 5 steps x 3.5h = 17.5h, leaves >6h margin under the 24h limit

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix/rq_experiments

echo "[job] $(date) starting combined RQ1-RQ6 run, OUT_TAG=${OUT_TAG}"

# RQ1+RQ2 -- runtime/budget. Sun's own Q1 protocol uses 100 trials. Measured
# locally (CPU, no GPU): ~148s for 5 trials up to checkpoint 200 -> ~30s/trial
# worst case; 100 trials ~= 50min.
echo "[job] $(date) RQ1+RQ2 (rq1_rq2_runtime.py)"
srun --unbuffered python -u rq1_rq2_runtime.py --n_trials 100 --out_tag "${OUT_TAG}" \
    --time_budget_h ${TIME_BUDGET_PER_STEP_H} \
    || echo "[job] RQ1+RQ2 data step failed (exit $?), continuing with the rest"
# rq1_rq2_runtime.py only writes the CSV; the plots are separate scripts
# (both read-only over that CSV, seconds to run, no GPU/CFM needed).
srun --unbuffered python -u rq1_plot.py --out_tag "${OUT_TAG}" \
    || echo "[job] RQ1 plot step failed (exit $?), continuing with the rest"
srun --unbuffered python -u rq2_plot.py --out_tag "${OUT_TAG}" \
    || echo "[job] RQ2 plot step failed (exit $?), continuing with the rest"

# RQ3 -- GP targets. Measured locally: 1 trial x 1 condition x 1 method x 3
# rounds = 8s; full scope (3 conditions x 3 methods, 15 rounds) is ~45x more
# work ~= 360s/trial worst case; 15 trials ~= 90min.
echo "[job] $(date) RQ3 (rq3_gp_targets.py)"
srun --unbuffered python -u rq3_gp_targets.py --n_trials 15 --out_tag "${OUT_TAG}" \
    --time_budget_h ${TIME_BUDGET_PER_STEP_H} \
    || echo "[job] RQ3 step failed (exit $?), continuing with the rest"

# RQ4 -- Sinkhorn/samples. Measured locally: 1 icon x 1 start ~= 44s/
# combination worst case; all 10 icons x 10 starts ~= 100 combinations ~= 73min.
echo "[job] $(date) RQ4 (rq4_sinkhorn_samples.py)"
srun --unbuffered python -u rq4_sinkhorn_samples.py --n_starts 10 --out_tag "${OUT_TAG}" \
    --time_budget_h ${TIME_BUDGET_PER_STEP_H} \
    || echo "[job] RQ4 step failed (exit $?), continuing with the rest"

# RQ5 -- dynamics. Measured locally: the qualitative grid once ~46s, then
# ~80s/trial for the quantitative sweep (6 dynamics x 2 methods x 7
# checkpoints); 30 trials ~= 41min.
echo "[job] $(date) RQ5 (rq5_dynamics.py)"
srun --unbuffered python -u rq5_dynamics.py --n_trials 30 --out_tag "${OUT_TAG}" \
    --time_budget_h ${TIME_BUDGET_PER_STEP_H} \
    || echo "[job] RQ5 step failed (exit $?), continuing with the rest"

# RQ6 -- ALM solver. No GPU/CFM, pure JAX optimisation: 10 trials = 13s
# measured locally -> 200 trials ~= 4min, no budget risk.
echo "[job] $(date) RQ6 (rq6_alm_lambda0.py)"
srun --unbuffered python -u rq6_alm_lambda0.py --n_trials 200 --out_tag "${OUT_TAG}" \
    || echo "[job] RQ6 step failed (exit $?)"

echo "[job] $(date) all steps attempted -> results/${OUT_TAG}/ (one subfolder per rqN_*.py's own CSV/plots)"
