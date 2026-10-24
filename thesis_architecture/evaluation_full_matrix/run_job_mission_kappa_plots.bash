#!/bin/bash
#SBATCH -J mission_kappa_plots
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --mem=12G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Plots for the kappa-annealed/full-execution replanning run (job 176368,
# COMPLETED, 18 shards under results/mission_kappa_schedule_20261021_tsvec_svgd).
# CPU-only, no GPU requested -- pure plotting from the already-written DB.
#
# Why this is a proper sbatch job and not run directly via ssh (as the first,
# mistaken attempt was, 2026-10-22): running real numpy/JAX/OpenBLAS compute
# directly on the shared login node hit that node's per-user RLIMIT_NPROC
# (thread/process) ceiling and crashed on `import numpy` itself -- login
# nodes are for light tasks (file management, job submission), not actual
# compute, same rule every other plotting step in this project already
# followed via sbatch. A `kill -9` on that direct-ssh attempt's main process
# also left its multiprocessing pool workers orphaned on the login node
# (reparented, not cascaded by the kill), contributing to the exhaustion --
# cleaned up separately before this job was written.
#
# plot_mission_eval.py / plot_mission_continuous.py's METHOD_STYLE dict was
# missing 'ebm'/'heuristic' entries (silently dropped from every figure, not
# an error) -- fixed in plot_mission_eval.py before this job; plot_mission_
# continuous.py reuses it via `pme.METHOD_STYLE`, no separate fix needed.

OUT_TAG=mission_kappa_schedule_20261021_tsvec_svgd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) plot_mission_eval.py"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${OUT_TAG}
rc1=$?
echo "[job] $(date) plot_mission_eval.py exit code ${rc1}"

echo "[job] $(date) plot_mission_continuous.py --svgd_conv"
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG} --svgd_conv
rc2=$?
echo "[job] $(date) plot_mission_continuous.py exit code ${rc2}"

echo "[job] $(date) done"
exit $(( rc1 != 0 ? rc1 : rc2 ))
