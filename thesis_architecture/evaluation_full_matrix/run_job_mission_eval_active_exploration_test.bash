#!/bin/bash
#SBATCH -J mission_ae_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# Timing/correctness probe for the active-exploration comparison (Philipp's
# request, 2026-10-20): 6 init methods (cfm, selfsup, ebm, linear,
# random_walk, heuristic -- dfl excluded, see the plan) x 2 strategies
# (ucb, eid) x 3 solvers (sun, tsvec=Adam-SVGD, tsvec_svgd=new paper-faithful
# solver) x 4 knowledge conditions (ground_truth as the oracle reference for
# the gap-to-oracle metric, plus half_known/ten_samples/none_known).
#
# This job does NOT run the full matrix -- only 2 shapes, max_rounds=3, the
# 2 cheapest-to-reason-about conditions (ground_truth + none_known, the two
# endpoints), BOTH strategies, ALL 6 methods, as 3 sequential sub-steps (one
# per solver) -- enough to read real per-(method,solver) per-round cost and
# extrapolate the full 25-shape x 4-condition run BEFORE sizing/submitting
# it. --qos=debug for fast scheduling (CLAUDE.md: the right QOS for exactly
# this kind of short timing probe).
#
# Local --dry_run smoke tests (CPU, no checkpoints, tiny n_init/n_iters/
# max_rounds) already passed for all 6 methods x 3 solvers x {ground_truth,
# none_known} before this job was written -- this is the first real-
# checkpoint, real-GPU run of the new `ebm`/`heuristic` mission-harness
# wiring and the new `BatchedTsvecSvgdTorch` adapter (svgd_batched.py).
#
# Flagged risk this probe exists to resolve: `tsvec_svgd` cannot batch
# different shapes into one GPU call the way sun/tsvec do (TsvecSvgd's joint
# SVGD swarm needs one shared target per call) -- BatchedTsvecSvgdTorch
# instead loops one call per active shape per round. At full scale (n_init=
# 30, n_iters=1500, up to 25 active shapes/round) this could be the
# dominant cost; read this job's per-round "SVGD batch Xs" lines for
# tsvec_svgd specifically before assuming it scales like the other two.

OUT_TAG=mission_ae_test_20261020

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch cuda available:', torch.cuda.is_available())"

echo "[test] $(date) local dry_run self-test (CPU, tiny scale, all 6 methods x 3 solvers)"
for REF in sun tsvec tsvec_svgd; do
    srun --unbuffered python -u run_mission_eval.py --dry_run \
        --out_tag smoke_ae_cluster --n_shapes 1 --conditions ground_truth \
        --strategies ucb --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
        --n_init 3 --n_iters 5 --max_rounds 2 --workers 1 --parallel_sets 1 \
        --refiner ${REF} || { echo "[test] dry_run FAILED for refiner=${REF}"; exit 1; }
done

for REF in sun tsvec tsvec_svgd; do
    echo "[test] $(date) refiner=${REF}: 2 shapes, 2 conditions, 2 strategies, 6 methods, max_rounds=3"
    T0=$(date +%s)
    srun --unbuffered python -u run_mission_eval.py \
        --out_tag ${OUT_TAG} \
        --refiner ${REF} \
        --n_shapes 2 \
        --conditions ground_truth,none_known \
        --strategies ucb,eid \
        --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
        --max_rounds 3 \
        --workers 6 \
        --parallel_sets 3
    rc=$?
    echo "[test] $(date) refiner=${REF} mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"
    du -sh results/${OUT_TAG}$([ "${REF}" = sun ] && echo _sun || [ "${REF}" = tsvec_svgd ] && echo _tsvec_svgd)
done

echo "[test] $(date) gap-to-oracle analysis (sanity check, tiny scale)"
srun --unbuffered python -u analyze_gap_to_oracle.py \
    --out_tag ${OUT_TAG} --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
    --strategies ucb,eid --refiners sun,tsvec,tsvec_svgd

echo "[test] $(date) done -- read the per-round 'SVGD batch Xs' lines above (or"
echo "  %x-%j.out) to extrapolate the full 25-shape x 4-condition run's time budget."
