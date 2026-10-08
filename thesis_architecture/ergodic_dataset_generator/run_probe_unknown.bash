#!/bin/bash
#SBATCH -J unk_probe
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:20:00
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=3G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Probelauf fuer generate_dataset_unknown.py: nur 4 Zieldichten x 2 Varianten
# (8 Loeser-Laeufe), kurzes Zeitlimit. Dient dazu, die Umgebung auf dem
# Cluster-Knoten zu pruefen (Pakete, Pfade, JAX-CPU-Betrieb) und echte
# Pro-Form-Zeiten auf einem echten Rechenknoten zu messen, BEVOR der grosse
# Array-Job (run_data_gen_unknown.bash) eingereicht wird.
#
# Ressourcen wie `run_data_gen.bash`: bewusst ohne GPU (JAX laeuft auf der
# CPU, effektiv auf einem Kern, siehe dortiger Kommentar), 2 Kerne, 3G reichen
# erfahrungsgemaess.
#
#   sbatch run_probe_unknown.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Knoten: $(hostname)   Start: $(date)"

python -u generate_dataset_unknown.py \
    --src_db ergodic_dataset_improved.db \
    --out ergodic_dataset_improved_unknown_probe.db \
    --shapes_from 0 --shapes_to 4

echo "Ende: $(date)"
