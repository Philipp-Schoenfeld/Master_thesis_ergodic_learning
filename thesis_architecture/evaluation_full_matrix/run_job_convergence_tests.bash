#!/bin/bash
#SBATCH -J convergence_tests
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Convergence tests of the active-sensing pipeline (run_convergence_tests.py).
#   sweep  : missions with n_iters in {0,10,25,50,100,250,500,1000}, 25 shapes x 3 variants, AK belief,
#            per method (the 500-iteration level is read from the finished comparison DB, which must
#            then contain that method -- merge the extra comparison runs first, or it is recomputed).
#   trace  : one 1000-iteration run per method and shape with E(iteration) of all candidates.
#   analysis: from stored results only (no compute).
#
# One job per method keeps jobs short and parallel (<= 3 GPUs at a time, see CLAUDE.md):
#   METHODS=lintsvec sbatch run_job_convergence_tests.bash
#   METHODS=old500   sbatch run_job_convergence_tests.bash
#   METHODS=dit630   sbatch run_job_convergence_tests.bash
#   METHODS=selfsup  sbatch run_job_convergence_tests.bash
#   METHODS=linadam  sbatch run_job_convergence_tests.bash
#   PARTS=analysis   sbatch run_job_convergence_tests.bash      # last, after pulling the results
# Estimate per method (sweep + trace; the 8 iteration levels sum to ~3.9x a 500-iteration mission, x 75
# missions, 10-25 s per 500-iteration mission locally): ~1.5-3 h;
# --max_minutes stops cleanly, re-submitting continues (resumable). Results go to
# results/convergence_tests_main/ (sweep.db, trace/*.npz) -- parallel jobs write the same sweep.db:
# SQLite WAL is unreliable on network file systems, so every job gets its OWN output folder (OUT_TAG default
# main_<method>) and the results are merged afterwards (see Kontext/2026-11-12_active_sensing_eval_cluster.md,
# section 7). Jobs run with --no_tables: tables/figures need all methods together and are made once after merging.

PARTS=${PARTS:-sweep,trace}
METHODS=${METHODS:-lintsvec}
MAX_MIN=${MAX_MIN:-690}
OUT_TAG=${OUT_TAG:-main_${METHODS//,/_}}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) parts=${PARTS} methods=${METHODS} tag=${OUT_TAG}"
srun --unbuffered python -u run_convergence_tests.py \
    --parts ${PARTS} --methods ${METHODS} --tag ${OUT_TAG} \
    --max_minutes ${MAX_MIN} --no_tables || exit 1
echo "[job] $(date) finished"
