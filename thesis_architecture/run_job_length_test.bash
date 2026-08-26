#!/bin/bash
#SBATCH -J len_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

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
# Testlauf der Laengen-Konditionierung.
#
# Zweck ist nicht ein gutes Modell, sondern zwei Fragen: faellt der Loss, und
# greift die klassifikatorfreie Fuehrung? Letzteres ist am Laufende an den
# Holdout-Bildern zu sehen — dieselbe Zieldichte, verschiedene Laengenvorgaben.
#
# 40 Epochen. Der Datensatz hat rund 17.800 Zeilen (1187 Formen mal bis zu
# fuenfzehn Varianten); mit copies_per_char 5 sind das etwa 350 Schritte je
# Epoche, also dieselbe Groessenordnung wie der Startpunkt-Lauf. nxi 64 statt
# 25 kostet grob das Zweieinhalbfache je Schritt.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

DB=ergodic_dataset_generator/ergodic_dataset_length.db

# Die Array-Aufgaben schreiben je in eine eigene Datei. Das Zusammenfuehren
# steht hier und nicht in einem eigenen Job: so ist die Kette geschlossen und
# ein vergessener Zwischenschritt kann den Trainingslauf nicht auf einer
# halben Datenbank starten lassen.
# IMMER zusammenfuehren, nicht nur wenn die Datei fehlt. Beim ersten Versuch
# stand hier ein `if [ ! -f ]`; nachdem sieben von acht Array-Aufgaben an
# fehlenden Schriften gescheitert waren, existierte die Datei mit einem
# Sechstel der Daten — und der Trainingslauf nahm sie, ohne die inzwischen
# nachgelieferten Zeilen zu sehen. Das Zusammenfuehren ueberspringt vorhandene
# (Form, Iterationszahl)-Paare, ist also gefahrlos wiederholbar.
echo "Fuehre die Teil-Datenbanken zusammen..."
( cd ergodic_dataset_generator && python -u merge_length_db.py \
    --out ergodic_dataset_length.db ) || exit 1
if [ ! -f "$DB" ]; then echo "Datenbank fehlt: $DB"; exit 1; fi

N_FORMEN=$(python -c "import sqlite3;print(sqlite3.connect('$DB').execute('select count(distinct shape_name) from ergodic_pairs').fetchone()[0])")
echo "Formen in der Datenbank: $N_FORMEN"
if [ "$N_FORMEN" -lt 1000 ]; then
    echo "ABBRUCH: nur $N_FORMEN von 1187 Formen. Der Datensatz ist unvollstaendig."
    exit 1
fi
echo "Zeilen in der Datenbank: $(python -c "import sqlite3;print(sqlite3.connect('$DB').execute('select count(*) from ergodic_pairs').fetchone()[0])")"

ARGS="--D 384 --n_particles 256 --nxi 64 --copies_per_char 2 --p_flip 0.0 \
      --epochs 40 --mini_batch 128 --lr 1e-4 \
      --save_every 5 --viz_every 5 --keep_checkpoints 1 \
      --p_drop_length 0.1 --cfg_weight 2.0 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --db ergodic_dataset_length.db --tag TEST --use_wandb"

LATEST=$(ls -t checkpoints/*_LEN-*_TEST_*_ep*.pt 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    python -u flow_matching_runner_length.py --resume "$LATEST" $ARGS
else
    echo "Starte Testlauf der Laengen-Konditionierung"
    python -u flow_matching_runner_length.py $ARGS
fi
