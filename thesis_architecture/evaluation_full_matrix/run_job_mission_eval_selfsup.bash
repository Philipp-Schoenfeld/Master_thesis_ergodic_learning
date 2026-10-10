#!/bin/bash
#SBATCH -J mission_selfsup
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=10G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Full additive run: 'selfsup' as a fourth init method for the replanning
# missions, 9 NEW sets (3 knowledge states x 3 strategies) written into the
# EXISTING results/mission_eval_20261006/shards/ directory, alongside the
# already-finished cfm/random_walk/linear shards (27 sets, untouched).
#
# --refiner tsvec is REQUIRED (see run_job_mission_eval_selfsup_test.bash for
# why -- otherwise this writes to a new mission_eval_20261006_sun directory
# instead of extending the existing one).
#
# Time estimate (measured, job 164950, test job: half_known/lse/selfsup, all
# 25 shapes batched on the GPU, 3 rounds): round times 15 s, 19 s, 22 s (grows
# with the driven path so far), isolated on one V100 with no other set
# competing for the GPU. The shard grew 47 MB over those 3 rounds (~15.7
# MB/round, same order for every method since state storage size is
# independent of the init). Cross-checked against the original 27-set run:
# 13 GB / 27 sets / 15.7 MB/round implies ~30 rounds/set on average, and at
# 22h/27 sets = ~49 min/set that run averaged ~98 s/round/set -- ~4-5x this
# test's isolated per-round time, consistent with --parallel_sets 4 meaning up
# to 4 sets share the one GPU concurrently (this job keeps the same
# --parallel_sets 4). Using that empirical 49 min/set (a conservative basis,
# since it also contains cfm's extra ODE-planning cost, which selfsup doesn't
# pay): 9 sets x ~49 min ~= 7.4h worst case, likely less once fewer than 4 of
# the (now only 9) sets remain active near the end. --time_budget_h 10 below
# leaves comfortable margin under the 24h SBATCH limit either way.

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) selfsup additive run (9 sets) -> results/${OUT_TAG}"
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --refiner tsvec \
    --methods selfsup \
    --workers 2 \
    --parallel_sets 4 \
    --time_budget_h 10
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots/tables (now include selfsup in every panel)"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG} --max_units 8 --workers 6
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG} --max_units 8 --workers 6 --svgd_conv
