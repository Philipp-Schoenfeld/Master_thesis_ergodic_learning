#!/bin/bash
#SBATCH -J surf_timing
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:12:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=24G
#SBATCH -c 4

# Wie lange dauert eine Epoche auf der projizierten 3D-Datenbank?
#
# Drei Epochen mit den Einstellungen, die auch die langen Laeufe benutzen. Die
# erste Epoche ist wegen cuDNN-Autotuning und Kernel-Kompilierung langsamer und
# taugt nicht als Mass; deshalb drei, und gerechnet wird mit der letzten.
#
#   sbatch run_job_3d_timing.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning

echo "Knoten:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Startzeit: $(date +%H:%M:%S)"

srun --unbuffered python -u flow_matching_runner_particles.py \
    --db3d ergodic_dataset_3d.db \
    --orientation --rot_full \
    --lambda_erg 100 --erg_K 6 --erg_on position --erg_t_power 2.0 \
    --lambda_ori 0.0 --w_cfm_rot 1.0 \
    --D 384 --n_particles 512 --nxi 25 \
    --copies_per_char 1 --mini_batch 64 --lr 1e-4 \
    --epochs 3 --save_every 0 --viz_every 0 \
    --save_model checkpoints/timing.pt

echo "Endzeit:   $(date +%H:%M:%S)"
