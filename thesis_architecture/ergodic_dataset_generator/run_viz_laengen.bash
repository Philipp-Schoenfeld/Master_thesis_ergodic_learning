#!/bin/bash
#SBATCH -J laengen_viz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=4G
#SBATCH -c 2

# Ressourcen sind gemessen, nicht geschaetzt. Aus `sacct` ueber die Laeufe
# vom 24. bis 26.08.:
#
#     Job                CPU-Eff   RAM-Eff   RAM genutzt
#     surf_lang           25.0 %     4.8 %      1.54 GB
#     start_kurz          25.2 %    10.1 %      3.25 GB
#     len_test            26.4 %     8.1 %      2.59 GB
#     Datensatz-Array     18.5 %    21.7 %      1.74 GB
#
# Jeder Job nutzt effektiv EINEN Kern — die Arbeit liegt auf der GPU
# beziehungsweise ist in JAX sequentiell ueber die Zeitschritte. Vier Kerne
# und 32 G anzufordern blockiert andere Nutzer und die eigenen Jobs, die sonst
# am Kontingent (cpu=50, gres/gpu=3, mem=150G) warten.
# Visualisierung der fertigen Laengen-Datenbank. Ohne GPU.
# Fuehrt vorher zusammen, damit die Bilder den vollstaendigen Stand zeigen.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

python -u merge_length_db.py --out ergodic_dataset_length.db
python -u viz_laengen_db.py --db ergodic_dataset_length.db --out visualizations/laengen --formen 10
