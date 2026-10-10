#!/bin/bash
#SBATCH -J svgd_conv_selfsup_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Test run: adds 'selfsup' as a fourth initialisation family to the SVGD
# convergence benchmark (run_svgd_convergence.py), ADDITIVE to the existing
# results/svgd_convergence_20261001/svgd_convergence.db (cfm/random_walk/linear).
#
# --refiner tsvec is REQUIRED: the project default switched to 'sun' on
# 2026-10-06 (output dir gets a '_sun' suffix appended), but
# svgd_convergence_20261001 is an "old", unsuffixed tsvec run -- without this
# flag the job would create a separate svgd_convergence_20261001_sun directory
# instead of extending the existing one.
#
# --n_iters 3000 directly (not the original's two-stage 1000-then-extend via
# extend_svgd_convergence.py): the referenced plots are the _3000iters ones,
# and a fresh method doesn't need the historical two-step path.
#
# Scope: 1 shape x 1 knowledge state x 1 strategy x --methods selfsup only --
# gives seconds/(shape,cond,strategy) block for selfsup at 3000 iterations,
# the basis for the full run's time estimate (run_job_svgd_convergence_selfsup.bash).
# Reference point: the fixed-start variant of this benchmark measured ~4-16
# min/shape depending on extra pinning overhead at 1000 iterations
# (run_job_svgd_convergence_fixedstart.bash); this test's 3000-iteration,
# single-method, single-block number is not directly comparable, hence a
# dedicated small measurement here rather than reusing that figure.

OUT_TAG=svgd_convergence_20261001

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[test] $(date) selfsup mini-run (1 shape, half_known/lse, 3000 iters)"
T0=$(date +%s)
srun --unbuffered python -u run_svgd_convergence.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec \
    --methods selfsup \
    --n_shapes 1 \
    --conditions half_known \
    --strategies lse \
    --n_iters 3000 \
    --workers 6
rc=$?
echo "[test] $(date) mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/svgd_convergence.db
