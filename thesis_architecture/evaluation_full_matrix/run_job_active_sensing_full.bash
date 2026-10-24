#!/bin/bash
#SBATCH -J active_sensing_full
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Active-sensing evaluation, full run: which belief model (RBF-GP,
# Attentive-Kernel GP, pretrained neural process) works best together with
# which initialiser (start-conditioned CFM, linear rays, self-supervised
# generator)? Replanning missions of run_mission_eval.py on all 25 holdout
# shapes, knowledge conditions none_known and ten_samples, UCB target with
# kappa annealed 1.0 -> 0.05 over 5 rounds, full execution of every refined
# candidate, TSVEC-SVGD refinement (30 candidates x 1500 iterations).
# 18 mission sets; results per belief in results/<tag>_tsvec_svgd/belief_*/.
#
# Resumable: --time_budget_h stops opening new rounds before the time limit;
# re-submitting this script continues with the same OUT_TAG, and mission sets
# with an identical fingerprint in other runs under results/ are reused.
#
# Estimate from test job 180646 (RTX 3090, 18 sets x 4 shapes x 2 rounds in
# 13.1 min, ~1 CPU core, 2.2 GB RSS, 131 MB results): at most ~3.4 h if
# linear in shape-rounds, expected ~2-3.5 h since most missions reach 99 %
# before round 5; results <= ~2 GB.

TIME_BUDGET_H="${1:-5.5}"
OUT_TAG=active_sensing_20261023
RUN_DIR=${OUT_TAG}_tsvec_svgd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) time_budget_h=${TIME_BUDGET_H} -> results/${RUN_DIR}"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec_svgd \
    --conditions none_known,ten_samples \
    --strategies ucb \
    --methods cfm,linear,selfsup \
    --beliefs rbf,ak,np \
    --max_rounds 5 \
    --kappa0 1.0 --kappa1 0.05 \
    --full_execution \
    --workers 2 \
    --parallel_sets 4 \
    --time_budget_h ${TIME_BUDGET_H} || exit 1
echo "[job] $(date) missions done after $(( $(date +%s) - T0 )) s"

srun --unbuffered python -u plot_active_sensing_eval.py --out_tag ${RUN_DIR}
for B in rbf ak np; do
    srun --unbuffered python -u plot_mission_eval.py --out_root results/${RUN_DIR} \
        --out_tag belief_${B} --no_per_shape \
        || echo "[job] plot_mission_eval.py failed for belief_${B} (standard plots only)"
done
echo "[job] $(date) finished after $(( $(date +%s) - T0 )) s"
