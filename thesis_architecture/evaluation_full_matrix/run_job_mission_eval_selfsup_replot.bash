#!/bin/bash
#SBATCH -J mission_selfsup_replot2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -p stud
#SBATCH --mem=6G
#SBATCH -c 6
#SBATCH --signal=SIGTERM@120

# Second attempt, narrowed to just the actually-requested output:
# plot_mission_continuous.py (overview_{E,J}_truth_continuous_8units*.png,
# overview_swept_truth_continuous_8units.png). Job 165099 (first attempt,
# 30 min limit) got through plot_mission_eval.py's overview_*.png fine (fresh,
# selfsup included -- already pulled locally) but then stalled somewhere after
# svgd_convergence_round6.png (per_shape/ or write_tables) and never reached
# the plot_mission_continuous.py calls at all -- likely filesystem contention
# from the concurrently-running svgd_conv_selfsup job (164989, 12 CPU workers
# hammering the same network home dir), which has since finished. 90 min this
# time, generous margin over the ~9-13 min measured for the 3-method data in
# run_job_mission_continuous.bash (now 4 methods, ~33% more candidates).

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) plot_mission_continuous.py (plain)"
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG} --max_units 8 --workers 6
echo "[job] $(date) plot_mission_continuous.py (--svgd_conv)"
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${OUT_TAG} --max_units 8 --workers 6 --svgd_conv
echo "[job] $(date) done"
