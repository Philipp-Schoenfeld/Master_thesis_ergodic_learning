#!/bin/bash
#SBATCH -J smooth_cmp_replot
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:40:00
#SBATCH -p stud
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Re-plot only (Philipp's request, 2026-10-09): add a third panel, ergodic
# error against the true density, computed from the ALREADY-stored SVGD
# states of the full run (job 164902, results/smoothness_comparison_20261009)
# -- no SVGD re-run needed, so no GPU requested here, just CPU/sqlite/
# matplotlib. See plot_smoothness_comparison.py's module docstring for how
# the ergodic series is computed (ergodic_energy_torch.ergodic_term, chunked
# to bound memory -- same (batch, T, 100-modes, 2) intermediate concern
# run_mission_eval.py::ergodic_E_batch already chunks for; ~1.8 GB per
# (shape, variant) cell unchunked, confirmed by an OOM-kill when this was
# first tried on a local machine with only ~1.2 GB free -- not an issue with
# this job's --mem=8G).

OUT_TAG=smoothness_comparison_20261009

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) re-plotting results/${OUT_TAG}"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
rc=$?
echo "[job] $(date) done, exit code ${rc}"
