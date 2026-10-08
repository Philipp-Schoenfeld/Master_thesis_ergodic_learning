r"""
merge_unknown_db.py
====================
Fuehrt die Teil-Datenbanken des Explorations-Array-Jobs
(`generate_dataset_unknown.py`, Dateien `*_part*.db`) zusammen und haengt sie
an eine Kopie von `ergodic_dataset_improved.db` an, statt diese direkt zu
ueberschreiben -- wer das Ergebnis pruefen will, hat damit weiterhin das
unveraenderte Original.

    python merge_unknown_db.py --base ergodic_dataset_improved.db \
        --out ergodic_dataset_improved_final.db \
        ergodic_dataset_improved_unknown_part*.db
"""
import argparse
import glob
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset_length import init_db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('teile', nargs='*', default=None)
    p.add_argument('--base', default='ergodic_dataset_improved.db',
                   help='Basis-DB mit den unveraenderten Original-Zeilen.')
    p.add_argument('--out', default='ergodic_dataset_improved_final.db')
    a = p.parse_args()

    teile = a.teile or sorted(glob.glob('ergodic_dataset_improved_unknown_part*.db'))
    if not teile:
        sys.exit('Keine Teil-Datenbanken gefunden.')
    if os.path.exists(a.out):
        sys.exit(f'Zieldatei existiert bereits, wird nicht ueberschrieben: {a.out}')

    shutil.copy2(a.base, a.out)
    ziel = sqlite3.connect(a.out)
    da = {r[0] for r in ziel.execute("SELECT shape_name FROM ergodic_pairs")}

    cols = ("shape_name, split, density_params, trajectory, x0, dt,"
            " tsteps, n_iters, length, generated_at")
    ges = 0
    for t in teile:
        q = sqlite3.connect(t)
        n = 0
        for r in q.execute(f"SELECT {cols} FROM ergodic_pairs"):
            if r[0] in da:
                continue
            ziel.execute(
                f"INSERT INTO ergodic_pairs ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", r)
            da.add(r[0])
            n += 1
        ziel.commit()
        q.close()
        print('  %-48s %6d Zeilen' % (os.path.basename(t), n))
        ges += n

    total = ziel.execute("SELECT count(*) FROM ergodic_pairs").fetchone()[0]
    original = ziel.execute(
        "SELECT count(*) FROM ergodic_pairs WHERE shape_name NOT LIKE '%#unk%'"
    ).fetchone()[0]
    unknown = total - original
    print(f'\n  {ges} neue Zeilen aus {len(teile)} Teil-DBs in {a.out}')
    print(f'  gesamt: {total}  (original: {original}, unbekannt-Varianten: {unknown})')
    ziel.close()


if __name__ == '__main__':
    main()
