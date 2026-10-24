#!/bin/bash
#SBATCH -J ebm_rankingnce_test
#SBATCH -t 00:10:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=2G
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=END,FAIL

# Sanity-check run for the Ranking-NCE energy-based policy (second generative
# head alongside CFM). CPU-only smoke test, no GPU needed: shapes/NaN checks,
# Ranking-NCE loss learnability, rank_candidates, CEM and Langevin refinement,
# particle-conditioning mode, noise-schedule bounds.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/src/methods/EBM_RankingNCE

srun --unbuffered python3 -u test_ebm_ranking_nce.py
