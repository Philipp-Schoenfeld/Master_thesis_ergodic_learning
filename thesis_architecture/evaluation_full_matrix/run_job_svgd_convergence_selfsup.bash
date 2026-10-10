#!/bin/bash
#SBATCH -J svgd_conv_selfsup
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=24G
#SBATCH -c 12
#SBATCH --signal=SIGTERM@120

# Full additive run: 'selfsup' as a fourth initialisation family for the SVGD
# convergence benchmark, written into the EXISTING
# results/svgd_convergence_20261001/svgd_convergence.db (cfm/random_walk/linear
# rows untouched). 12 shapes x 4 knowledge states x 3 strategies x 30 inits,
# 3000 iterations each (matching the _3000iters plots), resumable.
#
# --refiner tsvec is REQUIRED (see run_job_svgd_convergence_selfsup_test.bash
# for why).
#
# Time estimate (measured, job 164951, test job: 1 shape x 1 condition x 1
# strategy x 30 inits x 3000 iters, --workers 6, dgx-station V100): 8.3 min
# wall for that one (shape,cond,strategy) block == ~99.6 s per individual
# 3000-iteration CPU SVGD refinement (this pipeline refines on CPU worker
# processes, not the GPU -- GPU is only used for the cfm/selfsup forward
# pass). Full scope is 144 such blocks (12 shapes x 4 conditions x 3
# strategies) = 4320 individual refinements. At the test's --workers 6 that
# is ~20h, right at the 24h SBATCH limit with no safety margin -- raised to
# --workers 12 / -c 12 here (dgx-station has 20 CPUs; 12 leaves headroom for
# other jobs on the shared node) for an estimated ~10h. Resumable either way
# (`existing_keys` skips finished (shape,cond,strategy,method,init) rows), so
# even a worse-than-expected run just needs a follow-up
# `sbatch --dependency=afterany:<JOBID> run_job_svgd_convergence_selfsup.bash`,
# no recomputation or lost work.

OUT_TAG=svgd_convergence_20261001

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) selfsup additive run -> results/${OUT_TAG}"
srun --unbuffered python -u run_svgd_convergence.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec \
    --methods selfsup \
    --n_iters 3000 \
    --workers 12 \
    --time_budget_h 22
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots/tables (now include selfsup)"
srun --unbuffered python -u plot_svgd_convergence.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_svgd_convergence.py --out_tag ${OUT_TAG} --metric J
