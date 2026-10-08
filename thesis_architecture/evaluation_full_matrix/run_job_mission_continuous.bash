#!/bin/bash
#SBATCH -J mission_continuous
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:15:00
#SBATCH -p stud
#SBATCH --mem=4G
#SBATCH -c 4

# E_truth / J_truth of the driven path as a continuous function of the distance
# driven (0-8 length units, 50 points per unit), with a band over the 30
# alternative candidates of every planning round, for the mission run
# results/mission_eval_20261006 (plot_mission_continuous.py).
#
# No GPU requested: the script only reads the shard DBs, decodes the final SVGD
# state of ~161k candidates and evaluates the ergodic metric on the CPU.
#
# Timing: measured locally (6 workers) 3 min 22 s wall, ~11.5 min CPU time in
# total. With 4 workers that is ~3-4 min of compute plus reading ~5 GB of
# candidate blobs from /home (beegfs), so ~5-10 min expected; 10 min reserved.
# Memory: measured MaxRSS 1.6 GB in job 164027; --svgd_conv additionally keeps the
# SVGD metric series of all ~161k candidates (~0.2 GB, plus pickling copies) -> 4G.
#
# --svgd_conv adds a right axis with the SVGD iterations to convergence at every
# replanning point, for two criteria (plateau and time-to-target), written as
# *_svgd_conv_plateau.png and *_svgd_conv_target.png; the plain figures stay untouched.
# Extra cost per candidate: rendering 151 logged plans for the plan length, a
# measured 9 min 27 s in job 164055 (vs. 3 min 13 s without) -- reserve >= 15 min.

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python -u plot_mission_continuous.py \
    --out_tag ${OUT_TAG} \
    --max_units 8 \
    --workers 4 \
    --svgd_conv
