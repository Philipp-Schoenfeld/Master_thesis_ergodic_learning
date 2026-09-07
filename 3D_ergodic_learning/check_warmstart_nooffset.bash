#!/bin/bash
#SBATCH -J check_warmstart2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:15:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=6G
#SBATCH -c 2

# Nur-Diagnose: die tatsaechliche Pruefzeile aus run_job_3d_no_offset.bash,
# jetzt mit --mix ebene=0.25 (Fix nach der Falschmeldung vom 04.09.). Muss
# vor der echten 24h-Kette gruen sein.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

srun --unbuffered python pruefe_warmstart.py \
    --init_model "checkpoints/flaechen200_flow3d_particle_ergodic_flaechen200_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_MIX-ebene=0.25_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0111.pt" \
    --db3d ergodic_dataset_3d.db \
    --batches 80 --mini_batch 32 \
    --mix ebene=0.25 --ebene_flach_anteil 0.5 \
    --erg_on footprint --lambda_erg 100 --erg_K 6 --erg_pts 128 \
    --erg_t_power 2.0 --w_cfm_rot 0.5 \
    --orientation --lambda_ori 0.012 --w_point 0.1 --w_standoff 300 \
    --w_angsmooth 2.0 --standoff_target 0.12 --standoff_band 0.03
