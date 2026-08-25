#!/bin/bash
#SBATCH -J eval_ergloss
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Auswertung der ERGLOSS-Gewichtsablation (w = 2, 5, 15, 25, 50).
# Gleiche Einstellungen wie die Auswertung vom 14.08. (n_samples=8, steps=100),
# damit die Zahlen direkt mit der dortigen Tabelle vergleichbar sind.
# Die Solver-Referenzzeile fuegt evaluate_models.py selbst hinzu.
#
# Hinweis: Labels enthalten bewusst keine Kommas — die CSV-Ausgabe von
# evaluate_models.py maskiert nicht, ein Komma im Label wuerde die Spalten
# verschieben.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

P=checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date

srun --unbuffered python evaluate_models.py \
    --checkpoints \
        "${P}_08_15_13h17min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w2-K8-tp2_ep0492.pt" \
        "${P}_08_11_17h16min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w5-K8-tp2_ep0500.pt" \
        "${P}_08_15_13h23min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w15-K8-tp2_ep0486.pt" \
        "${P}_08_15_13h44min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w25-K8-tp2_ep0485.pt" \
        "${P}_08_16_07h17min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w50-K8-tp2_ep0486.pt" \
    --labels \
        "ErgLoss w=2 (ep492)" \
        "ErgLoss w=5 (ep500)" \
        "ErgLoss w=15 (ep486)" \
        "ErgLoss w=25 (ep485)" \
        "ErgLoss w=50 (ep486)" \
    --n_samples 8 \
    --steps 100 \
    --out metrics_ergloss_ablation \
    --visualize \
    --viz_n_gen 5 \
    --viz_obstacle_mode both \
    --viz_dir Trajectory_data_generator/viz_ergloss_ablation
