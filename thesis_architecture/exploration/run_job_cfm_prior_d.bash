#!/bin/bash
#SBATCH -J cfm_priorD
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:35:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Derselbe Sweep wie run_job_cfm_prior.bash, aber mit Variante D dazu:
# lang planen, zehn Prozent fahren, Abdeckungsschuld nachziehen, neu planen.
#
# Sieben Missionen, zwoelf Formen, drei Mengen an Vorwissen = 252 Bahnen.
#
# Danach eine Ablation: dieselbe Variante D, aber mit `--d_join start`. Sie
# steigt stur am Anfang der neu geplanten Bahn ein statt an deren
# naechstgelegenem Punkt. Der Unterschied ist genau die Strecke, die der
# fehlende Start-Eingang des Netzes kostet.
#
# Zeitbudget: der Sechs-Missionen-Sweep lief in wenigen Minuten je Durchlauf.
# Variante D bringt zwanzig Planungen je Form statt dreien, also grob eine
# zusaetzliche Minute je Durchlauf; die Ablation noch einmal rund fuenf.
# Reserviert sind 35 Minuten fuer geschaetzte rund 20.
#
#   sbatch run_job_cfm_prior_d.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"
[ -f "$CKPT" ] || { echo "Checkpoint fehlt: $CKPT" >&2; exit 1; }

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
        --missions orakel glaube-1 glaube-R zweistufig glaube-D B-warm maeher \
        --save_paths \
        --shapes 12 \
        --n_prior $NPRIOR \
        --rounds 3 \
        --d_rounds 20 \
        --d_execute_frac 0.10 \
        --d_join nearest \
        --visit_sat 0.25 \
        --debt_weight 1.0 \
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
        --out_dir "results/cfm_prior_d/n${NPRIOR}" || exit 1
done

for NPRIOR in 12 30 60; do
    echo
    echo "=============================================================="
    echo "  Ablation: Einstieg am Bahnanfang, $NPRIOR Vorabmessungen"
    echo "=============================================================="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --missions glaube-D \
        --d_join start \
        --save_paths \
        --shapes 12 \
        --n_prior $NPRIOR \
        --d_rounds 20 \
        --d_execute_frac 0.10 \
        --visit_sat 0.25 \
        --debt_weight 1.0 \
        --kappa0 3.0 \
        --kappa1 0.3 \
        --n_particles 256 \
        --truth_res 128 \
        --gp_res 64 \
        --gp_noise 0.05 \
        --sensor_radius 0.06 \
        --phi_mode uniform \
        --no_grad_baseline \
        --anytime_points 12 \
        --viz_shapes 4 \
        --seed 0 \
        --out_dir "results/cfm_prior_d/n${NPRIOR}_joinstart" || exit 1
done

echo
echo "Endzeit:     $(date)"
