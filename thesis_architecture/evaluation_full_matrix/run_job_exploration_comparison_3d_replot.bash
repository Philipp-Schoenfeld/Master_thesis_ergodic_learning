#!/bin/bash
#SBATCH -J expl_cmp_3d_replot
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH --mem=16G
#SBATCH -c 4

# Per-shape figures (round overview, variants, 3D world view; 25 shapes x 2 kernels) and the tables
# of the merged 3D comparison DB, CPU only. The merged DB is built locally with
#   python run_exploration_comparison_3d.py --merge_from plots/exploration_comparison_3d_flat_<TAG>/exploration_comparison.db
# and pushed to plots/exploration_comparison_3d_flat/ before this job runs.

PLANE=${PLANE:-flat}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) plane=${PLANE}"
srun --unbuffered python -u run_exploration_comparison_3d.py --plane ${PLANE} --plot_only --device cpu || exit 1
srun --unbuffered python -u plot_exploration_summary.py --dir plots/exploration_comparison_3d_${PLANE} || exit 1
echo "[job] $(date) finished"
