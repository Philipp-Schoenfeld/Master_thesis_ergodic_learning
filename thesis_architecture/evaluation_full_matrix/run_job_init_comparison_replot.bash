#!/bin/bash
#SBATCH -J init_cmp_replot
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Re-plot only (Philipp's request, 2026-10-19/20): validated categorical
# palette (dataviz skill), a zoomed convergence sub-panel (the full-range log
# plot compressed the converged/plateau region into a sliver), and a new
# best_method_ranking.png (per-metric rank heatmap + composite avg-rank bar
# chart -- "an denen man die beste Methode am besten ablesen kann"). No SVGD
# re-run needed, reuses the already-computed DB (job 174569, results/
# init_comparison_20261019). No GPU requested, pure CPU/sqlite/matplotlib.
#
# This explicitly belongs on the cluster, not local (CLAUDE.md: "Große
# Laeufe/Re-Plots nur auf dem Cluster, lokal ~1.2 GB frei, keine GPU") --
# the first attempt at this re-plot was run locally and had to be aborted
# after 18+ min with no completion; `ErgodicSeriesCache.dense_states` also
# now caches its decoded (n_states, T, 2) arrays per (shape, variant) instead
# of `load()` and `coverage_final()` each decoding independently, which the
# new ranking chart's added `coverage_final` calls made newly expensive to
# skip -- --mem=24G sized for holding all 350 cells' decoded states at once
# (roughly 5 GB) plus normal overhead.

OUT_TAG=init_comparison_20261019

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) re-plotting results/${OUT_TAG}"
T0=$(date +%s)
srun --unbuffered python -u plot_init_comparison.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_init_comparison_overview.py --out_tag ${OUT_TAG}
rc=$?
echo "[job] $(date) done, exit code ${rc}, after $(( $(date +%s) - T0 )) s"
