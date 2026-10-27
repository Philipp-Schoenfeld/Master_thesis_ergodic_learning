#!/bin/bash
#SBATCH -J ds_adam_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=10G
#SBATCH -c 2

# Test job for the Adam-SVGD training DB (generate_dataset_adam_svgd.py):
#
#   1. nxi study (study_nxi_adam_svgd.py): 24 base shapes + their #unk0
#      variants, the real solver config (linear init, hard start pin, plateau
#      stop 1% / 1000 it, max 10000 it), nxi = 12 16 20 25 32 40 50, float64.
#      Again for nxi 20 25 32 in float32 compute: consumer RTX cards run
#      float64 at 1/64 of the float32 rate, so float32 is the speed option IF
#      it gives the same trajectories.
#   2. Timing of the full generator: all 2344 densities x 3 starts = 7032
#      trajectories (4096 on the GPU at once, the rest refilled), exactly 200
#      iterations (no early stop), float64 and float32. Gives the seconds per
#      iteration at full batch and the density-grid time of the full set. The
#      float64 DB of this step (nxi 25, 200 iterations, NOT a training DB)
#      is kept as ergodic_dataset_adam_svgd_timing.db: the training test job
#      uses it to time the real training config.
#
# -C rtx: same GPU class as the full run (and the training, see
# gpu-constraint notes); debug QOS because this is a short calibration run.
# One GPU-side process, JAX grids on one core -> -c 2. RAM: 2344 grids at
# 128^2 in float64 are ~0.3 GB, plus JAX/torch.
#
#   sbatch run_job_dataset_adam_svgd_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
OUT=results_adam_svgd_test

T0=$(date +%s)
srun --unbuffered python -u study_nxi_adam_svgd.py --out ${OUT}/nxi_f64 --n_shapes 24 \
    --nxi 12 16 20 25 32 40 50 --compute_dtype float64
echo "[study f64] $(( $(date +%s) - T0 )) s"

T0=$(date +%s)
srun --unbuffered python -u study_nxi_adam_svgd.py --out ${OUT}/nxi_f32 --n_shapes 24 \
    --nxi 20 25 32 --compute_dtype float32
echo "[study f32] $(( $(date +%s) - T0 )) s"

for DT in float64 float32; do
    DB=ergodic_dataset_adam_svgd_timing_${DT}.db
    rm -f ${DB}
    T0=$(date +%s)
    srun --unbuffered python -u generate_dataset_adam_svgd.py --out ${DB} --nxi 25 \
        --max_iters 200 --min_iters 200 --tol -1 --check_every 50 --compute_dtype ${DT}
    echo "[timing ${DT}] $(( $(date +%s) - T0 )) s for 7032 trajectories x 200 iterations (incl. grids)"
done
mv ergodic_dataset_adam_svgd_timing_float64.db ergodic_dataset_adam_svgd_timing.db
rm -f ergodic_dataset_adam_svgd_timing_float32.db
echo "end: $(date)"
