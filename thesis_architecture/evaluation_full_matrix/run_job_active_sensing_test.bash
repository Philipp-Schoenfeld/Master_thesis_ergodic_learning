#!/bin/bash
#SBATCH -J active_sensing_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 3

# GPU test of the active-sensing evaluation before the full run: the real
# networks and the full per-round budget (30 candidates x 1500 TSVEC-SVGD
# iterations), all 18 mission sets (2 knowledge conditions x 3 initialisers x
# 3 beliefs), but only the first 4 holdout shapes and 2 rounds. Checks the whole
# chain incl. the AK fit / NP forward on the GPU and gives per-set-round timings
# for the estimate of the full run, then writes the evaluation plots.
#
# Resources from the comparable full mission run 176368 (18 sets x 25 shapes x
# 5 rounds, same solver/budget): ~1 CPU core on average, 4.1 GB peak RSS.

OUT_TAG=active_sensing_test_20261023

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) -> results/${OUT_TAG}_tsvec_svgd"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec_svgd \
    --conditions none_known,ten_samples \
    --strategies ucb \
    --methods cfm,linear,selfsup \
    --beliefs rbf,ak,np \
    --n_shapes 4 \
    --max_rounds 2 \
    --kappa0 1.0 --kappa1 0.05 \
    --full_execution \
    --workers 2 \
    --parallel_sets 4 || exit 1
echo "[job] $(date) missions done after $(( $(date +%s) - T0 )) s"

srun --unbuffered python -u plot_active_sensing_eval.py --out_tag ${OUT_TAG}_tsvec_svgd
echo "[job] $(date) finished after $(( $(date +%s) - T0 )) s"
