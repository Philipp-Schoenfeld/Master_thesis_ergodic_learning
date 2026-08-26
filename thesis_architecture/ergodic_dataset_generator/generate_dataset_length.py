r"""
generate_dataset_length.py
==========================
Laengen-Varianten fuer die Trajektorien-Datenbank.

Fuer jede Zielverteilung wird der SVGD-Loeser **einmal** bis zur groessten
Iterationszahl gefahren und an festen Zwischenpunkten die Bahn mitgeschrieben.
Fuenfzehn getrennte Laeufe waeren die naheliegende Variante — sie wuerden aber
fuenfzehnmal durch dieselben ersten hundert Iterationen laufen und kosteten
rund das Achtfache.

Alle Varianten einer Form teilen sich **denselben** zufaelligen Startpunkt.
Damit ist die Pfadlaenge die einzige Groesse, die sich zwischen ihnen
unterscheidet — sonst waere nicht zu trennen, ob das Netz auf die Laenge oder
auf den Startpunkt reagiert.

An jedem Checkpoint wird die physikalische Pfadlaenge berechnet. Waechst sie
ueber die letzten zwei Abstaende jeweils um weniger als `--konvergenz` (1 %),
bricht der Lauf fuer diese Form ab: weitere Varianten waeren Kopien.

    python generate_dataset_length.py --db ergodic_dataset_length.db \
        --tsteps 400 --shapes_from 0 --shapes_to 150
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np
import zlib

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)

# Die Vorgabe aus der Aufgabenstellung, ergaenzt um vier Stuetzstellen dort,
# wo die Bahn zwischen zwei Checkpoints am staerksten wuchs. Gemessen an 196
# fertigen Formen betrug der Laengenzuwachs je Abschnitt:
#
#     300 ->  500   18.3 %      3000 -> 5000   16.3 %
#     500 ->  750   14.3 %      5000 -> 7500   12.6 %
#     200 ->  300   13.5 %      7500 -> 10000   8.1 %
#
# Ergaenzt sind 250, 400, 4000 und 6000; damit bleibt jeder Abschnitt unter
# rund zehn Prozent. Das kostet praktisch nichts: der Loeser kommt ohnehin
# dort vorbei, es sind vier zusaetzliche `traj_sim`-Aufrufe im selben Lauf.
#
# Nicht ergaenzt wurde oberhalb von 10000. Der Zuwachs faellt dort auf 0,64
# Laengeneinheiten je 1000 Iterationen (gegen 11,6 am Anfang); 25000 braechten
# bei 2,5-facher Rechenzeit rund 20 % mehr Laenge, und die Bahnen unterschieden
# sich um 3 bis 4 % — zu wenig, um als eigenes Trainingsbeispiel zu taugen.
CHECKPOINTS = [100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500,
               2000, 2500, 3000, 4000, 5000, 6000, 7500, 10000]


def pfadlaenge(tr):
    return float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum())


def init_db(pfad):
    conn = sqlite3.connect(pfad)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ergodic_pairs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_name     TEXT    NOT NULL,
            split          TEXT    NOT NULL DEFAULT 'train',
            density_params TEXT    NOT NULL,
            trajectory     BLOB    NOT NULL,
            x0             TEXT    NOT NULL,
            dt             REAL    NOT NULL,
            tsteps         INTEGER NOT NULL,
            n_iters        INTEGER NOT NULL,
            length         REAL    NOT NULL,
            generated_at   TEXT
        )""")
    # Eine Form kommt mehrfach vor, einmal je Iterationszahl. Der Index macht
    # das Ueberspringen bereits erzeugter Varianten beim Fortsetzen billig.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_form_iter "
                 "ON ergodic_pairs(shape_name, n_iters)")
    conn.commit()
    return conn


def speichern(conn, name, split, shape_def, traj, x0, dt, tsteps, n_iters, laenge):
    if shape_def.get('type') == 'analytical':
        params = {'type': 'analytical', 'segments': shape_def['segments'],
                  'sigma': shape_def.get('sigma', 0.025)}
    else:
        params = {'means': shape_def['means'], 'covs': shape_def['covs'],
                  'weights': [float(w) for w in shape_def['weights']]}
    if shape_def.get('pedestal'):
        params['pedestal'] = shape_def['pedestal']
    conn.execute(
        "INSERT INTO ergodic_pairs (shape_name, split, density_params, trajectory,"
        " x0, dt, tsteps, n_iters, length, generated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (name, split, json.dumps(params),
         np.asarray(traj, dtype=np.float32).tobytes(),
         json.dumps([float(v) for v in x0]), dt, tsteps, int(n_iters),
         float(laenge), time.strftime('%Y-%m-%dT%H:%M:%S')))
    conn.commit()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default='ergodic_dataset_length.db')
    p.add_argument('--tsteps', type=int, default=400,
                   help='Punkte je Bahn. Hoeher als die 200 der bisherigen '
                        'Datenbank: die Kontrollpunkte im Runner sind ein '
                        'Teilraster dieser Punkte, und bei nxi=64 oder 128 '
                        'waeren 201 Punkte die eigentliche Grenze.')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--step_size', type=float, default=0.01)
    p.add_argument('--h', type=float, default=0.01)
    p.add_argument('--score_scale', type=float, default=1.0)
    p.add_argument('--konvergenz', type=float, default=0.01,
                   help='Relatives Laengenwachstum, unter dem abgebrochen wird.')
    p.add_argument('--x0_margin', type=float, default=0.03)
    p.add_argument('--seed', type=int, default=20260825)
    p.add_argument('--shapes_from', type=int, default=None)
    p.add_argument('--shapes_to', type=int, default=None)
    p.add_argument('--val_only', action='store_true',
                   help='Nur die Holdout-Formen erzeugen.')
    a = p.parse_args()

    from shape_library import get_shape, make_pdf_and_score, VALIDATION_SHAPES, \
        train_shape_names, flat_shape_names
    from ergodic_solver import run_ergodic_coverage

    namen = [(n, 'val') for n in VALIDATION_SHAPES]
    if not a.val_only:
        namen += [(n, 'train') for n in train_shape_names(750)]
        namen += [(n, 'train') for n in flat_shape_names('train')]
        namen += [(n, 'val_flat') for n in flat_shape_names('val')]
    if a.shapes_from is not None or a.shapes_to is not None:
        namen = namen[a.shapes_from or 0:a.shapes_to]

    conn = init_db(a.db)
    da = {(r[0], r[1]) for r in
          conn.execute("SELECT shape_name, n_iters FROM ergodic_pairs")}
    erledigt = {n for n, _ in namen
                if all((n, c) in da for c in CHECKPOINTS)}

    print(f"  Datenbank : {a.db}")
    print(f"  Formen    : {len(namen)}  (fertig: {len(erledigt)})")
    print(f"  Checkpoints: {CHECKPOINTS}")
    print(f"  tsteps={a.tsteps}  Konvergenz bei <{a.konvergenz:.1%} Wachstum",
          flush=True)

    t_start = time.time()
    n_zeilen = 0
    for k, (name, split) in enumerate(namen):
        if name in erledigt:
            continue
        # Ein Keim je Form: alle Varianten teilen sich denselben Startpunkt,
        # und ein Neustart des Jobs zieht denselben wieder.
        #
        # crc32 statt hash(): Pythons Zeichenketten-Hash ist je Prozess
        # zufaellig gesalzen. Mit hash() bekaeme dieselbe Form nach einem
        # Jobneustart einen anderen Startpunkt — und da teilweise erzeugte
        # Formen fortgesetzt werden, laegen dann Varianten mit
        # verschiedenen Startpunkten unter demselben Namen. Genau das, was
        # der feste Startpunkt verhindern soll.
        keim = zlib.crc32(f"{a.seed}:{name}".encode()) & 0x7FFFFFFF
        rng = np.random.default_rng(keim)
        x0 = tuple(rng.uniform(a.x0_margin, 1.0 - a.x0_margin, size=2))

        shape_def = get_shape(name)
        _, score_fn = make_pdf_and_score(shape_def)

        t0 = time.perf_counter()
        zwischen, _init, laengen = run_ergodic_coverage(
            score_fn, x0=x0, shape_def=shape_def, dt=a.dt, tsteps=a.tsteps,
            num_iters=max(CHECKPOINTS), step_size=a.step_size, h=a.h,
            score_scale=a.score_scale, checkpoints=CHECKPOINTS,
            konvergenz_tol=a.konvergenz)
        dauer = time.perf_counter() - t0

        for n_iters, L in laengen:
            if (name, n_iters) in da:
                continue
            speichern(conn, name, split, shape_def, zwischen[n_iters], x0,
                      a.dt, a.tsteps, n_iters, L)
            n_zeilen += 1

        rest = (len(namen) - k - 1) * dauer
        print(f"  [{k+1}/{len(namen)}] {name:<22} {len(laengen):2d} Varianten  "
              f"L {laengen[0][1]:.2f} -> {laengen[-1][1]:.2f}  "
              f"{dauer:5.1f}s  (Rest ~{rest/3600:.1f} h)", flush=True)

    conn.close()
    print(f"\n  {n_zeilen} Zeilen ergaenzt in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
