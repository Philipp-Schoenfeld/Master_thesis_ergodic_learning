#!/bin/bash
#SBATCH -J cfm_phi
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:20:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Vier Modellierungen der Zieldichte, jeweils bei wenig und viel Vorwissen.
#
# Die Vorabrechnung ohne Netz (`vergleich_phi_modelle.py`) hat gezeigt, dass die
# bisherige additive Form von einer Gleichverteilung kaum zu unterscheiden ist:
# der Anteil der Zielmasse auf dem wahren Traeger liegt bei 0,352 gegen 0,355
# fuer eine gleichverteilte Dichte, und er waechst zwischen 12 und 60 Messungen
# nur um zwei Prozent. Hier wird geprueft, ob sich das auch in der gefahrenen
# Bahn niederschlaegt.
#
# Mit dabei ist `eid`, die erwartete Informationsdichte nach Miller et al.
# (2016) — theoretisch die am besten begruendete Alternative, weil sie die
# Zieldichte gar nicht als Feld, sondern als Fisher-Information einer Messung
# ansetzt.
#
# Bewusst schmal gehalten: drei Missionen statt sieben, zwei Messmengen statt
# drei. Zehn Durchlaeufe zu je rund einer Minute; reserviert sind 20 Minuten.
#
#   sbatch run_job_phi_modelle.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"
[ -f "$CKPT" ] || { echo "Checkpoint fehlt: $CKPT" >&2; exit 1; }

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Erst die reine Rechnung ohne Netz — sie braucht keine GPU, kostet Sekunden
# und liefert die Kennzahlen, gegen die das Gefahrene gelesen wird.
srun --unbuffered python -u vergleich_phi_modelle.py --shapes 12 \
     --out_dir results/phi_modelle || exit 1

for MODEL in ucb mass lse mi eid; do
  for NPRIOR in 12 60; do
    echo
    echo "=============================================================="
    echo "  Modell $MODEL, $NPRIOR Vorabmessungen   ($(date +%H:%M:%S))"
    echo "=============================================================="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --phi_model $MODEL \
        --missions glaube-1 glaube-R maeher \
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
        --no_grad_baseline \
        --anytime_points 12 \
        --viz_shapes 4 \
        --seed 0 \
        --out_dir "results/phi_lauf/${MODEL}_n${NPRIOR}" || exit 1
  done
done

echo
echo "Endzeit:    $(date)"
