#!/bin/bash
#SBATCH -J cfm_prior
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:15:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Sechs Missionen auf allen zwoelf Holdout-Formen, gefahren mit drei
# verschiedenen Mengen an Vorwissen: 12, 30 und 60 Vorabmessungen.
#
# Alles andere bleibt gleich — derselbe Checkpoint, derselbe Zufallskeim,
# dieselben Formen, dieselben Parameter. Der einzige Unterschied zwischen den
# drei Durchlaeufen ist `--n_prior`. Damit ist jede Abweichung in den Zahlen
# dem Vorwissen zuzuschreiben und nichts sonst.
#
# Jeder Durchlauf schreibt in ein eigenes Verzeichnis `results/cfm_prior/nNN/`,
# damit die drei Ergebnismengen nebeneinander bestehen bleiben.
#
# Zeitbudget: Job 134118 brauchte 2:50 fuer neun Missionen auf zwoelf Formen.
# Hier sind es sechs Missionen mit knapp der halben Zahl an Netz-Planungen, also
# rund 100 s je Durchlauf plus etwa 30 s Anlauf — dreimal gerechnet gut sieben
# Minuten. Reserviert sind 15, also etwa das Doppelte; kuerzere Laufzeiten
# werden vom Scheduler ausserdem frueher eingeplant.
#
#   sbatch run_job_cfm_prior.bash
#   sbatch --export=ALL,CKPT=/pfad/zum/checkpoint.pt run_job_cfm_prior.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint fehlt: $CKPT" >&2
    exit 1
fi

echo "Knoten:      $(hostname)"
echo "Startzeit:   $(date)"
echo "Checkpoint:  $(basename "$CKPT")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for NPRIOR in 12 30 60; do
    echo
    echo "=============================================================="
    echo "  $NPRIOR Vorabmessungen   ($(date +%H:%M:%S))"
    echo "=============================================================="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --missions orakel glaube-1 glaube-R zweistufig B-warm maeher \
        --save_paths \
        --shapes 12 \
        --n_prior $NPRIOR \
        --rounds 3 \
        --kappa0 3.0 \
        --kappa1 0.3 \
        --n_particles 256 \
        --truth_res 128 \
        --gp_res 64 \
        --gp_noise 0.05 \
        --sensor_radius 0.06 \
        --phi_mode uniform \
        --grad_steps 200 \
        --refine_steps 100 \
        --refine_lr 0.03 \
        --n_probe 40 \
        --lambda_unc 1.0 \
        --lambda_cov 20000 \
        --anytime_points 12 \
        --viz_shapes 4 \
        --seed 0 \
        --out_dir "results/cfm_prior/n${NPRIOR}" || exit 1
done

echo
echo "Endzeit:     $(date)"
