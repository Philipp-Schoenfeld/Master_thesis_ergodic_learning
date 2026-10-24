#!/bin/bash
#SBATCH -J init_cmp_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
##SBATCH -C 'vram24gb|vram32gb|vram48gb'
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Timing/correctness probe for the 7-init-method x 2-solver comparison
# (Philipp's request, 2026-10-19; see thesis_architecture/evaluation_full_
# matrix/run_init_comparison.py's module docstring for the full variant
# table). Real scale (n_init=10, n_iters=1500, all 7x2=14 cells) on only
# N_TEST_SHAPES=2 holdout shapes -- same convention `run_job_smoothness_
# comparison_v2_test.bash` already used: read the per-cell "done in Xs"
# lines in this job's .out to extrapolate the full 25-shape run's time
# budget BEFORE sizing/submitting run_job_init_comparison.bash.
#
# No -p (main is default since the stud partition closed 2026-10-16).
# Checkpoints used (all confirmed present on the cluster via `rsync
# --list-only` before this job was written):
#   cfm      transfer/netz2d_startpunkt.pt
#   selfsup  checkpoints/selfsup_selfsup_particles_date_08_12_10h13min_..._final.pt
#   ebm      src/methods/EBM_RankingNCE/checkpoints/ebm_..._C100_final.pt
#   dfl      checkpoints/cond_spectral_crossattn_ergodic_dfl_..._final.pt

OUT_TAG=init_comparison_test_20261019
N_TEST_SHAPES=2
DFL_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_10_16_14h49min_S256_nxi25_D384_C75_flip0.0_final.pt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch cuda available:', torch.cuda.is_available())"

echo "[test] $(date) self-test (--dry_run, CPU, tiny scale, all 7 methods x 2 solvers)"
srun --unbuffered python -u run_init_comparison.py --dry_run --shapes A \
    --n_init 3 --n_iters 10 --out_tag smoke_dry_cluster || { echo "[test] self-test FAILED"; exit 1; }

echo "[test] $(date) mini-run: ${N_TEST_SHAPES} shapes, all 14 variants, n_init=10, n_iters=1500 (real scale)"
T0=$(date +%s)
srun --unbuffered python -u run_init_comparison.py \
    --out_tag ${OUT_TAG} \
    --n_shapes ${N_TEST_SHAPES} \
    --n_init 10 \
    --n_iters 1500 \
    --dfl_ckpt "${DFL_CKPT}"
rc=$?
echo "[test] $(date) mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/init_comparison.db

echo "[test] $(date) plots/summary"
T1=$(date +%s)
srun --unbuffered python -u plot_init_comparison.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_init_comparison_overview.py --out_tag ${OUT_TAG}
echo "[test] $(date) plotting took $(( $(date +%s) - T1 )) s"

echo "[test] $(date) done -- read the per-cell 'done in Xs' lines above (or"
echo "  %x-%j.out) to extrapolate the full 25-shape run's time/disk budget."
cat results/${OUT_TAG}/summary.csv
