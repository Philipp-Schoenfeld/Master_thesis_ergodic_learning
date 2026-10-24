#!/bin/bash
#SBATCH -J mission_kappa_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# Timing/correctness probe for the kappa-annealed, full-execution replanning
# test (Philipp's request, 2026-10-21): a NEW mission protocol, not a bigger
# version of the earlier 48-set/solver active-exploration runs --
#   * 6 init methods (cfm, selfsup, ebm, linear, random_walk, heuristic),
#     ONE solver only (tsvec_svgd, the static comparison's winner),
#   * 3 knowledge conditions (none_known, half_known, ten_samples -- no
#     ground_truth: kappa-annealed exploration is specifically a partial-
#     knowledge question),
#   * ONLY the 'ucb' strategy ('eid' dropped -- kappa governs 'ucb' only,
#     'eid' has its own tuned parameter a kappa list says nothing about),
#   * `--full_execution` (drive the WHOLE refined candidate every round, not
#     one length unit) + hard `--max_rounds 5`,
#   * `--kappas 2,1,0.5,0.1,0.05` (common.acquisition.kappa_schedule's
#     exploration-to-exploitation idea, applied per round via a new flag in
#     run_mission_eval.py -- see its MissionSet.__init__/refresh() diff,
#     2026-10-21).
# = 6 x 1 x 3 x 1 = 18 mission sets, capped at 5 rounds each (not "until 99%
# coverage, up to 40 rounds" like before) -- a much smaller run than the
# previous 48-sets-per-solver active-exploration matrix.
#
# Local --dry_run + a standalone kappa-schedule unit check (both exact
# values 2.0/1.0/0.5/0.1/0.05 per round, end-to-end through MissionSet
# .refresh()) already passed before this job was written. This cluster job
# is the first REAL-checkpoint, real-GPU run of the new --kappas wiring --
# only 2 shapes, --qos=debug for fast turnaround, to catch anything the
# local CPU dry_run couldn't (real CFM/selfsup/EBM network behaviour, real
# TsvecSvgd solver) before sizing/submitting the full 25-shape run.

OUT_TAG=mission_kappa_test_20261021

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[test] $(date) 2 shapes, 3 conditions, ucb only, 6 methods, tsvec_svgd, max_rounds=5, kappa 2->0.05"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec_svgd \
    --n_shapes 2 \
    --conditions none_known,half_known,ten_samples \
    --strategies ucb \
    --methods cfm,selfsup,ebm,linear,random_walk,heuristic \
    --max_rounds 5 \
    --kappas 2,1,0.5,0.1,0.05 \
    --full_execution \
    --workers 6 \
    --parallel_sets 4
rc=$?
echo "[test] $(date) exit code ${rc} after $(( $(date +%s) - T0 )) s"
du -sh results/${OUT_TAG}_tsvec_svgd 2>/dev/null

echo "[test] $(date) done -- read the per-round 'SVGD batch Xs' lines above to"
echo "  extrapolate the full 25-shape run's time budget."
exit ${rc}
