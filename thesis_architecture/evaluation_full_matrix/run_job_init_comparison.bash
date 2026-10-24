#!/bin/bash
#SBATCH -J init_cmp_full
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
##SBATCH -C 'vram24gb|vram32gb|vram48gb'
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Full 25-holdout-shape run of the 7-init-method x 2-solver comparison
# (Philipp's request, 2026-10-19; see run_init_comparison.py's module
# docstring for the variant table and metric definitions).
#
# -t sized from the real smoke test (job 174567, 2 shapes, n_init=10,
# n_iters=1500, real checkpoints, dgx-station/V100-32GB): ~11s/cell solver
# phase (28 cells in 270-310s) + ~8.5s/(shape,variant) plot phase (237s for
# 2x14) -> 25 shapes extrapolates to ~65 min solver + ~45-50 min plotting,
# ~1h50-2h total. 4h leaves a real margin without being a wasteful blind
# default (shorter requests also get better backfill scheduling).
#
# No -p (main is default since the stud partition closed 2026-10-16).
# `run_init_comparison.py` is resumable per (shape, method, solver, cand_idx)
# -- a timeout/preemption just needs the same `sbatch` command re-run.

OUT_TAG=init_comparison_20261019
DFL_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_10_16_14h49min_S256_nxi25_D384_C75_flip0.0_final.pt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full run -> results/${OUT_TAG}"
T0=$(date +%s)
srun --unbuffered python -u run_init_comparison.py \
    --out_tag ${OUT_TAG} \
    --n_init 10 \
    --n_iters 1500 \
    --dfl_ckpt "${DFL_CKPT}"
rc=$?
echo "[job] $(date) run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/init_comparison.db

echo "[job] $(date) plots/summary"
srun --unbuffered python -u plot_init_comparison.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_init_comparison_overview.py --out_tag ${OUT_TAG}
echo "[job] $(date) done"
