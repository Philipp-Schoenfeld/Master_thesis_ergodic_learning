#!/bin/bash
#SBATCH -J phi_kreuz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:05:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Das vollstaendige Kreuz: jede Modellierung der Zieldichte mit jeder Mission,
# bei drei Mengen an Vorwissen.
#
#   7 Phi-Modelle  ×  3 Messmengen  =  21 Durchlaeufe
#   je Durchlauf: orakel, glaube-1, glaube-R, zweistufig, B-warm, maeher
#                 auf allen 12 Holdout-Formen
#
# `glaube-D` fehlt hier mit Absicht: es plant zwanzigmal je Form statt drei- bis
# viermal und macht damit rund siebzig Prozent der Rechenzeit aus. Es laeuft im
# Anschlussjob `run_job_phi_kreuz_d.bash`, damit beide Jobs ein enges Zeitfenster
# bekommen und nicht einer mit einer Stunde Luft in der Warteschlange steht.
#
# Zeitbudget: der Sechs-Missionen-Sweep (Job 134776) brauchte rund zwei Minuten
# je Durchlauf. 21 × 2 = 42 Minuten, reserviert sind 65.
#
#   sbatch run_job_phi_kreuz.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/exploration

DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"
[ -f "$CKPT" ] || { echo "Checkpoint fehlt: $CKPT" >&2; exit 1; }

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MODEL in ucb stretch mass mi ei lse eid; do
  for NPRIOR in 12 30 60; do
    echo
    echo "=== $MODEL / $NPRIOR Messungen   ($(date +%H:%M:%S)) ==="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --phi_model $MODEL \
        --missions orakel glaube-1 glaube-R zweistufig B-warm maeher \
        --save_paths \
        --shapes 12 --n_prior $NPRIOR --rounds 3 \
        --kappa0 3.0 --kappa1 0.3 \
        --n_particles 256 --truth_res 128 --gp_res 64 --gp_noise 0.05 \
        --sensor_radius 0.06 --phi_mode uniform \
        --refine_steps 100 --refine_lr 0.03 --n_probe 40 \
        --lambda_unc 1.0 --lambda_cov 20000 \
        --anytime_points 12 --viz_shapes 2 --seed 0 \
        --out_dir "results/phi_kreuz/${MODEL}_n${NPRIOR}" || exit 1
  done
done
echo; echo "Endzeit:    $(date)"
