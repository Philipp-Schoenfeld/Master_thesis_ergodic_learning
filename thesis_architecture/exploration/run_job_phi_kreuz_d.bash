#!/bin/bash
#SBATCH -J phi_kreuzD
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:20:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Das vollstaendige Kreuz: jede Modellierung der Zieldichte mit jeder Mission,
# bei drei Mengen an Vorwissen.
#
#   7 Phi-Modelle  ×  3 Messmengen  =  21 Durchlaeufe
#   je Durchlauf: nur `glaube-D` auf allen 12 Holdout-Formen
#
# Der teure Teil des Kreuzes, deshalb ein eigener Job: D plant zwanzigmal je
# Form. Wird mit einer Abhaengigkeit auf den Hauptjob eingereiht, damit beide
# nicht gleichzeitig um dieselbe GPU konkurrieren:
#
#   sbatch --dependency=afterok:<ID> run_job_phi_kreuz_d.bash
#
# Zeitbudget: aus Job 134780 abgeleitet, dort kostete ein D-Durchlauf ueber
# 12 Formen rund 2,5 Minuten. 21 × 2,5 = 53 Minuten, reserviert sind 80.

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
        --missions glaube-D \
        --save_paths \
        --shapes 12 --n_prior $NPRIOR --rounds 3 \
        --kappa0 3.0 --kappa1 0.3 \
        --n_particles 256 --truth_res 128 --gp_res 64 --gp_noise 0.05 \
        --sensor_radius 0.06 --phi_mode uniform \
        --no_grad_baseline \
        --anytime_points 12 --viz_shapes 2 --seed 0 \
        --out_dir "results/phi_kreuz_d/${MODEL}_n${NPRIOR}" || exit 1
  done
done
echo; echo "Endzeit:    $(date)"
