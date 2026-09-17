#!/bin/bash
#SBATCH -J policy_probe
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@60

# ===========================================================================
# Disposable timing probe for the oracle stage's real per-round cost with
# actual parallelism (--workers 7), after job 154992 stalled at 150 shapes
# under --workers 0 (all SVGD refinement serialized onto one of 8 allocated
# CPUs). Writes to its own throwaway files (policy_datensatz_probe.csv /
# policy_orakel_probe.json), never the real dataset the full run reads/writes.
#
# Queue only via sbatch, never srun directly:
#
#     sbatch run_job_probe.bash
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
cd ~/Master_thesis/thesis_architecture || exit 1

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if [ -n "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1; then
  RUN="srun --unbuffered"
else
  RUN=""
fi

echo "=========================================================="
echo "Zeit-Sondierung: 10 Formen, 2 Runden, 1 Seed, --workers 7"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$RUN python -m exploration_optimierung.policy.oracle \
    --n_shapes 10 --n_max 2 --seeds 1 --split train --workers 7 \
    --zufallsmaske \
    --out exploration_optimierung/results/policy_datensatz_probe.csv \
    --bericht exploration_optimierung/results/policy_orakel_probe.json \
    --max_minuten 30

echo "Fertig."
