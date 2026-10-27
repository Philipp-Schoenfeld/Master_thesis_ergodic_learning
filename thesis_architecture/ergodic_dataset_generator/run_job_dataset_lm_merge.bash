#!/bin/bash
#SBATCH -J ds_lm_merge
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -c 2

# Joins the three part DBs of run_job_dataset_lm.bash into
# ergodic_dataset_lm_nxi25.db and validates it (validate_adam_svgd_db.py, incl.
# a 32-density sensitivity re-solve with the same LM pipeline -> GPU). Exit
# code 1 on any FAIL, so the afterok-chained training does not start on a
# broken DB.
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_job_dataset_lm_merge.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Node: $(hostname)  start: $(date)"
python -u generate_dataset_lm.py --out ergodic_dataset_lm_nxi25.db --merge || exit 1
ls -la ergodic_dataset_lm_nxi25.db

srun --unbuffered python -u validate_adam_svgd_db.py --db ergodic_dataset_lm_nxi25.db \
    --out results_lm_validation_nxi25 --sensitivity 32
RC=$?
echo "validation exit code ${RC}  end: $(date)"
exit ${RC}
