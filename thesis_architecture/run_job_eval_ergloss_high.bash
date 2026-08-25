#!/bin/bash
#SBATCH -J eval_erg_high
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Auswertung der zweiten ERGLOSS-Reihe (w = 100, 150, 200, 300, 400).
# Alle fuenf Laeufe haben die vollen 500 Epochen erreicht.
# Gleiche Einstellungen wie die Auswertungen vom 14.08. und 17.08.
# (n_samples=8, steps=100), damit die Zahlen direkt an die Reihe
# w = 2..50 anschliessen. Ohne --visualize: die Endzustandsbilder
# sind bereits heruntergeladen, hier zaehlen nur die Metriken.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

P=checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date

srun --unbuffered python evaluate_models.py \
    --checkpoints \
        "${P}_08_17_10h49min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w100-K8-tp2_ep0500.pt" \
        "${P}_08_17_10h49min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w150-K8-tp2_ep0500.pt" \
        "${P}_08_17_11h18min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w200-K8-tp2_ep0500.pt" \
        "${P}_08_17_14h58min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w300-K8-tp2_ep0500.pt" \
        "${P}_08_17_19h06min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w400-K8-tp2_ep0500.pt" \
    --labels \
        "ErgLoss w=100 (ep500)" \
        "ErgLoss w=150 (ep500)" \
        "ErgLoss w=200 (ep500)" \
        "ErgLoss w=300 (ep500)" \
        "ErgLoss w=400 (ep500)" \
    --n_samples 8 \
    --steps 100 \
    --out metrics_ergloss_high
