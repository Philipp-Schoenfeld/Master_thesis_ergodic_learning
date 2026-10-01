#!/bin/bash
#SBATCH -J best_of_n_full_buf8h
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 08:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# 8h-Sicherheitspuffer-Job, gleiche Sorte wie run_job_best_of_n_full.bash
# (identischer Umfang, identisches --out_tag), nur mit kuerzerem Zeitlimit.
# Haengt ans Ende der 5x24h-Kette (siehe CLUSTER_CONTEXT_best_of_n.md: ~75-77h
# Bedarf vs. 5x24h=120h Kette -- dieser 6. Job ist reine Sicherheit, falls die
# Kette aus irgendeinem Grund (Warteschlangen-Unterbrechung, langsamerer Node)
# nicht ausgereicht hat. Da jede Zeile einzeln gecacht ist (row_done-Check),
# ist ein Job, der nichts mehr zu tun findet, ein schneller No-Op.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python run_best_of_n_matrix.py \
    --representations particles,spectral \
    --strategies niveau_svgd0,niveau_svgd25,ucb_tuned_svgd0,ucb_tuned_svgd25,mass_tuned_svgd0,mass_tuned_svgd25,eid_tuned_svgd0,eid_tuned_svgd25,eid_optuna_ideal_v2 \
    --replan_schemes no_replan,replan_1_6 \
    --n_candidates 30 \
    --selection post_svgd \
    --heuristic_families lse,ucb,mass,eid \
    --spectral_ckpt ~/Master_thesis/thesis_architecture/checkpoints/cond_spectral_crossattn_ergodic_S256_nxi25_D384_flip0.0_final.pt \
    --out_tag best_of_30_full_20260918
