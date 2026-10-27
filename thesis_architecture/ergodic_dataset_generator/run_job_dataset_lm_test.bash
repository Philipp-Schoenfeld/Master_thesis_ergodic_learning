#!/bin/bash
#SBATCH -J ds_lm_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 2

# Test of the teacher-study DB generator (generate_dataset_lm.py, method
# lm_K16_cont_o2_w15) on the cluster: 7 base densities + their #unk0 variants,
# 3 starts each = 42 trajectories (real settings), then the full validation
# incl. an 8-density sensitivity re-solve. Gives seconds per trajectory on a
# cluster GPU (local CPU: ~45 s) for the -t of the array job, and checks the
# environment (teacher_study, ergodic_validation, geomloss-free metrics).
#
#   sbatch run_job_dataset_lm_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

NAMES="greek_lower_6,rand_gmm_164,rand_gmm_complex_64,rand_ana_poly_15,flat_ped_54,cjk_17,korean_5"
NAMES="${NAMES},$(echo ${NAMES} | sed 's/,/#unk0,/g')#unk0"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
rm -f ergodic_dataset_lm_test.db
T0=$(date +%s)
srun --unbuffered python -u generate_dataset_lm.py --out ergodic_dataset_lm_test.db --names "${NAMES}"
echo "[generate] $(( $(date +%s) - T0 )) s for 42 trajectories"

srun --unbuffered python -u validate_adam_svgd_db.py --db ergodic_dataset_lm_test.db --subset \
    --out results_lm_validation_test --sensitivity 8
RC=$?
echo "validation exit code ${RC}  end: $(date)"
exit ${RC}
