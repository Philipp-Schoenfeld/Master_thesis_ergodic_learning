#!/bin/bash
#SBATCH -J ds_lm
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --array=0-2
#SBATCH --signal=SIGTERM@120

# Training DB with the teacher-study method (generate_dataset_lm.py,
# lm_K16_cont_o2_w15, Kontext/2026-10-24_teacher_study.md): 1172 base densities
# of ergodic_dataset_improved_final.db + their #unk0 large-area variant, 3 start
# points each = 7032 trajectories; per trajectory 30 chords -> Levenberg-
# Marquardt (K=16, continuation 150 > 50 > 15) on 128 waypoints -> 25 control
# points + polish stages -> the study's selection rule (+ control points in
# [0, 1]).
#
# Three array tasks = three GPUs (the student limit; the training only starts
# after the merge), every 3rd trajectory each, own part DB (SQLite writers do
# not share a file). -t: local CPU 45 s/trajectory, GPU expected well below
# 10 s -> 2344 per task < 7 h; 12 h is a margin. Resumable: a re-run skips the
# rows already in the part DB. No GPU type constraint: no bf16 here.
#
#   sbatch run_job_dataset_lm.bash      (merge + validation: run_job_dataset_lm_merge.bash)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Task ${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
srun --unbuffered python -u generate_dataset_lm.py --out ergodic_dataset_lm_nxi25.db \
    --part ${SLURM_ARRAY_TASK_ID} --n_parts 3
echo "end: $(date)"
