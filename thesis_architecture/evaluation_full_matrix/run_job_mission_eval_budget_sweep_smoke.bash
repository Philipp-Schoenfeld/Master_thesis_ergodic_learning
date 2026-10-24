#!/bin/bash
#SBATCH -J mission_bsweep_smoke
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -t 00:45:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --array=0-5

# Smoke test / timing calibration for run_job_mission_eval_budget_sweep.bash:
# same 6 fixed SVGD budgets, but only 2 shapes, 1 knowledge condition, 1
# strategy, small round cap -- enough to measure real per-round wall-clock at
# each budget level before committing cluster time to the full 25x3x3 matrix
# x 6 budgets.

NITERS=(10 50 100 250 500 1000)
N=${NITERS[$SLURM_ARRAY_TASK_ID]}
OUT_TAG=mission_eval_budget_sweep_smoke_${N}iters_20261012

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) budget sweep smoke n_iters=${N}"
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --n_iters ${N} \
    --shapes A,organic_10 \
    --conditions none_known \
    --strategies eid \
    --max_rounds 10 \
    --workers 2 \
    --parallel_sets 2 \
    --time_budget_h 0.6
rc=$?
echo "[job] $(date) smoke run (n_iters=${N}) finished with exit code ${rc}"
