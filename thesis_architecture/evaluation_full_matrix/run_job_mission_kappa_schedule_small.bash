#!/bin/bash
#SBATCH -J mission_kappa_small
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Same evaluation as mission_kappa_schedule_20261021 (6 methods x tsvec_svgd
# x 3 knowledge conditions x ucb-only x --max_rounds 5 x --kappas
# 2,1,0.5,0.1,0.05 x --full_execution), but with a much smaller candidate
# distribution and iteration budget (Philipp's request, 2026-10-22):
# --n_init 5 (was 30) and --n_iters 50 (was 1500). Expect noticeably worse
# per-round SVGD convergence for methods whose median convergence needs more
# than 50 iterations in the full-budget run (random_walk: median 140 iters,
# linear: 50, ebm: 40 -- see mission_kappa_schedule_20261021's own "conv
# target" log lines) -- that is an expected consequence of the smaller
# budget Philipp asked for, not a bug to fix here.
#
# Generation + both plot scripts combined into ONE job this time (lesson
# from the previous run: running plotting directly via ssh on the login
# node hit that node's RLIMIT_NPROC and crashed on `import numpy`; a proper
# sbatch job runs on a compute node instead, same convention every other
# step in this project already follows). Given the much smaller n_init/
# n_iters, generation + plotting together should comfortably fit the 2h
# limit (the full-budget run took ~1h49 generation + ~28 min plotting
# combined; this one should be well under that).

OUT_TAG=mission_kappa_schedule_small_20261022

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) generation: n_init=5, n_iters=50"
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
    --n_init 5 \
    --n_iters 50 \
    --workers 6 \
    --parallel_sets 4
rc=$?
echo "[job] $(date) generation exit code ${rc} after $(( $(date +%s) - T0 )) s"
if [ ${rc} -ne 0 ]; then echo "[job] generation failed, skipping plots"; exit ${rc}; fi

echo "[job] $(date) plot_mission_eval.py"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${OUT_TAG}_tsvec_svgd
rc1=$?

echo "[job] $(date) plot_mission_continuous.py --svgd_conv"
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG}_tsvec_svgd --svgd_conv
rc2=$?

echo "[job] $(date) done"
exit $(( rc1 != 0 ? rc1 : rc2 ))
