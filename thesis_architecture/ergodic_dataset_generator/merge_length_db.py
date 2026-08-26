r"""
merge_length_db.py
==================
Die Teil-Datenbanken des Array-Jobs zu einer zusammenfuehren.

Acht Prozesse, die gleichzeitig in dieselbe SQLite-Datei schreiben, blockieren
sich gegenseitig — deshalb schreibt jede Array-Aufgabe in ihre eigene und wird
hinterher eingesammelt.

    python merge_length_db.py --out ergodic_dataset_length.db \
        ergodic_dataset_length_part*.db
"""
import argparse
import glob
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset_length import init_db, CHECKPOINTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('teile', nargs='*', default=None)
    p.add_argument('--out', default='ergodic_dataset_length.db')
    a = p.parse_args()

    teile = a.teile or sorted(glob.glob('ergodic_dataset_length_part*.db'))
    if not teile:
        sys.exit('Keine Teil-Datenbanken gefunden.')

    ziel = init_db(a.out)
    da = {(r[0], r[1]) for r in
          ziel.execute("SELECT shape_name, n_iters FROM ergodic_pairs")}

    ges = 0
    for t in teile:
        if os.path.abspath(t) == os.path.abspath(a.out):
            continue
        q = sqlite3.connect(t)
        n = 0
        for r in q.execute(
                "SELECT shape_name, split, density_params, trajectory, x0, dt,"
                " tsteps, n_iters, length, generated_at FROM ergodic_pairs"):
            if (r[0], r[7]) in da:
                continue
            ziel.execute(
                "INSERT INTO ergodic_pairs (shape_name, split, density_params,"
                " trajectory, x0, dt, tsteps, n_iters, length, generated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)", r)
            da.add((r[0], r[7]))
            n += 1
        ziel.commit()
        q.close()
        print('  %-42s %6d Zeilen' % (os.path.basename(t), n))
        ges += n

    print('\n  %d Zeilen in %s' % (ges, a.out))
    print('  Formen: %d' % ziel.execute(
        "SELECT count(DISTINCT shape_name) FROM ergodic_pairs").fetchone()[0])
    print('  Varianten je Form:')
    for n_iters, k in ziel.execute(
            "SELECT n_iters, count(*) FROM ergodic_pairs GROUP BY n_iters"
            " ORDER BY n_iters"):
        print('    %6d Iterationen  %5d Formen' % (n_iters, k))
    lo, hi, mit = ziel.execute(
        "SELECT min(length), max(length), avg(length) FROM ergodic_pairs").fetchone()
    print('  Laenge: %.2f bis %.2f, im Mittel %.2f' % (lo, hi, mit))
    ziel.close()


if __name__ == '__main__':
    main()
