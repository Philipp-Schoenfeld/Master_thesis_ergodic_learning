#!/bin/bash
#SBATCH -J smooth_v2_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Test run for round 2 of the smoothness comparison (Philipp's request,
# 2026-10-09): 7 variants contrasting the dynamics-based "sun" solver
# (sun_refine.py, point-mass and new jerk-penalised dynamics) against a fresh
# reimplementation of the TSVEC paper's own preconditioned SVGD
# (tsvec_svgd.py, arXiv:2603.09458 -- see its module docstring), B-spline and
# raw-waypoint parameterisations, all at --n_iters 2000 (vs. round 1's 600),
# 30 particles as ONE joint, mutually-interacting SVGD swarm (not 30
# independent runs, per Philipp's explicit choice).
#
# This job does NOT run the full 25-shape matrix -- only
#   1. self-test (test_smoothness_comparison_v2.py, which itself runs
#      exploration/common/test_tsvec_svgd.py first -- gradient checks of the
#      new solver's analytic Jacobian, timing, jerk-dynamics sanity, the
#      full dry-run pipeline + shared plotting script),
#   2. the REAL per-cell workload (n_init=30, n_iters=2000, all 7 variants)
#      on 2 shapes, to read off real wall-clock/disk numbers instead of
#      guessing (round 1's workflow: job 164887/164902/164948).
# Local CPU numbers already gathered before this job (no local GPU):
#   - tsvec_svgd 2000 iterations, N=30, raw waypoints (T=128): 90 ms/iter
#     -> ~180 s/cell; B-spline (nxi=25) similar (~79 ms/iter in a smaller
#     local probe). ~3 min/cell x 4 tsvec_svgd variants x 25 shapes ~ 5 h.
#   - sun_refine (point-mass / jerk), 2000 iters, raw waypoints: round 1
#     measured ~21 s/cell at 600 iters on this cluster -> ~70 s/cell
#     extrapolated at 2000 -> x 2 variants x 25 shapes ~ 1 h.
#   -> rough CPU-only estimate ~6 h total; this job's GPU request is there to
#      see whether torch (unlike JAX, which fell back to CPU for "sun" in
#      round 1 -- see that round's .err) actually gets real CUDA accel here,
#      which would cut this down. Read this job's per-cell "done in Xs"
#      lines to get the real number before sizing the full run.
#   - disk: a 2000-iteration tsvec_svgd cell logs (2001, 30, <=256, 2)
#     float32 -> packed via the same state_codec.py as round 1, ~3.3x round
#     1's per-cell size (2001 vs 601 states); round 1's full 25-shape DB was
#     236 MB at 601 states x 5 variants -- expect roughly comparable or
#     somewhat larger here (fewer variants needing full logging offsets more
#     states/variant), read off `du -sh` below rather than guessing further.
#
# New code exercised here for the first time against the real GPU + the real
# CFM checkpoint (only CPU-tested locally): tsvec_svgd.py in full, sun_refine.
# py's `dynamics='jerk'` path, run_smoothness_comparison_v2.py's orchestration.

OUT_TAG=smoothness_comparison_v2_test_20261009b
N_TEST_SHAPES=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch cuda available:', torch.cuda.is_available())"

echo "[test] $(date) self-test"
srun --unbuffered python -u test_smoothness_comparison_v2.py || { echo "[test] self-test FAILED"; exit 1; }

echo "[test] $(date) mini-run: ${N_TEST_SHAPES} shapes, all 7 variants, n_init=30, n_iters=2000 (real scale)"
T0=$(date +%s)
srun --unbuffered python -u run_smoothness_comparison_v2.py \
    --out_tag ${OUT_TAG} \
    --n_shapes ${N_TEST_SHAPES} \
    --n_init 30 \
    --n_iters 2000
rc=$?
echo "[test] $(date) mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/smoothness_comparison.db

echo "[test] $(date) plots/summary"
T1=$(date +%s)
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
echo "[test] $(date) plotting took $(( $(date +%s) - T1 )) s"

echo "[test] $(date) done -- read the per-cell 'done in Xs' lines above (or"
echo "  %x-%j.out) to extrapolate the full 25-shape run's time/disk budget."
cat results/${OUT_TAG}/summary.csv
