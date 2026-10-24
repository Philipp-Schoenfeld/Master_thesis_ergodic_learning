#!/bin/bash
#SBATCH -J mission_ae_test2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# Second, REALISTICALLY-SCALED timing probe (Philipp's request, 2026-10-20),
# follow-up to run_job_mission_eval_active_exploration_test.bash (job 174766,
# COMPLETED -- confirmed correctness for all 6 methods x 3 solvers, but at
# toy scale n_init=3/n_iters=5, which specifically underestimates
# `tsvec_svgd`'s cost: its joint-swarm kernel term is O(n_init^2), so the
# toy probe's n_init=3 hides ~100x of the cost jump to the real n_init=30,
# on top of the ~300x from n_iters=5 -> 1500 that applies to every solver.
#
# Scope, deliberately narrow to stay cheap while using REAL n_init/n_iters
# (left at their --n_iters 1500 / --n_init 30 defaults, no override): 2
# shapes, none_known only (the condition with the most active shapes for
# the longest, i.e. the realistic per-round cost case), ucb only,
# max_rounds=1 (one round is enough to read a real per-round cost; no
# accumulation risk since there is nothing to accumulate over), all 6
# methods, 3 solvers as sequential sub-steps.
#
# This is NOT a full correctness re-test (job 174766 already covered that) --
# only to read real "SVGD batch Xs" numbers per (method, solver) at
# production scale before sizing the full 25-shape x 4-condition run.

OUT_TAG=mission_ae_test2_20261020

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for REF in sun tsvec tsvec_svgd; do
    echo "[test2] $(date) refiner=${REF}: 2 shapes, none_known, ucb, 6 methods, REAL n_init=30/n_iters=1500, max_rounds=1"
    T0=$(date +%s)
    srun --unbuffered python -u run_mission_eval.py \
        --out_tag ${OUT_TAG} \
        --refiner ${REF} \
        --n_shapes 2 \
        --conditions none_known \
        --strategies ucb \
        --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
        --max_rounds 1 \
        --workers 6 \
        --parallel_sets 3
    rc=$?
    echo "[test2] $(date) refiner=${REF} exit code ${rc} after $(( $(date +%s) - T0 )) s"
done

echo "[test2] $(date) done -- read each refiner's per-(method) 'SVGD batch Xs' line above."
