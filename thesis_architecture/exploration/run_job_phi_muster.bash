#!/bin/bash
#SBATCH -J phi_muster
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:10:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Das Phi-Kreuz noch einmal, aber mit *gezieltem* statt zufaelligem Vorwissen.
#
# Zufaellig gestreute Vormessungen erzeugen einen Glauben, der ueberall gleich
# lueckenhaft ist. Der Vergleich der Zieldichten misst dann nur, wie sie mit
# gleichmaessigem Halbwissen umgehen — nicht, ob sie das Unbekannte finden.
#
# Diese drei Muster erzeugen eine Kante zwischen bekannt und unbekannt:
#
#   haelfte      die linke Haelfte ist bekannt. Der einfachste Fall: eine
#                gerade Kante, das Unbekannte grenzt an den Rand.
#   quadranten   zwei diagonal gegenueberliegende Viertel sind bekannt. Der
#                bekannte Teil ist unzusammenhaengend, eine Bahn muss also
#                zweimal ueber unbekanntes Gebiet.
#   loch         alles ausser einer Scheibe in der Mitte ist bekannt. Das
#                Unbekannte ist von Wissen umschlossen — es gibt keine Kante
#                zum Rand, an der man sich entlanghangeln koennte. Der
#                schwierigste der drei.
#
# 60 Messungen statt 12: das bekannte Gebiet soll wirklich bekannt sein. Bei
# 60 Punkten auf der halben Flaeche liegt der mittlere Abstand bei etwa 0,09
# und damit in der Groessenordnung der GP-Korrelationslaenge von 0,08.
#
# 7 Zieldichten × 3 Muster × 6 Missionen × 12 Formen = 1512 Trajektorien.
# Job 134916 brauchte fuer denselben Umfang 38:50; reserviert sind 70 Minuten.
#
#   sbatch run_job_phi_muster.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/exploration

DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"
[ -f "$CKPT" ] || { echo "Checkpoint fehlt: $CKPT" >&2; exit 1; }

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MUSTER in haelfte quadranten loch; do
  for MODEL in ucb stretch mass mi ei lse eid; do
    echo
    echo "=== $MUSTER / $MODEL   ($(date +%H:%M:%S)) ==="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --phi_model $MODEL \
        --prior_pattern $MUSTER \
        --n_prior 60 \
        --missions orakel glaube-1 glaube-R zweistufig B-warm maeher \
        --save_paths \
        --shapes 12 --rounds 3 \
        --kappa0 3.0 --kappa1 0.3 \
        --n_particles 256 --truth_res 128 --gp_res 64 --gp_noise 0.05 \
        --sensor_radius 0.06 --phi_mode uniform \
        --refine_steps 100 --refine_lr 0.03 --n_probe 40 \
        --lambda_unc 1.0 --lambda_cov 20000 \
        --anytime_points 12 --viz_shapes 2 --seed 0 \
        --out_dir "results/phi_muster/${MUSTER}_${MODEL}" || exit 1
  done
done
echo; echo "Endzeit:    $(date)"
