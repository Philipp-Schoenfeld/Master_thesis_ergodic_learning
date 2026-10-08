r"""
generate_dataset_unknown.py
============================
Explorations-Varianten fuer `ergodic_dataset_improved.db`.

Liest die bestehende, auf eine Trajektorie je Zieldichte gefilterte Datenbank
(siehe `filter_best_ergodicity.py`) und erzeugt je Zeile `--n_variants`
(Standard 2) zusaetzliche Zeilen, in denen ein zufaelliger Flaechenbereich als
"unbekannt" gilt und mit entsprechend hoher Dichte in die Zieldichte gemischt
wird (`unknown_region.build_unknown_variant`). Fuer jede Variante laeuft
dieselbe Loeser-Pipeline wie fuer die uebrigen Trajektorien dieser Datenbank:
heuristische Initialisierung (hier: bestehender Form-Pfad + Maeander-Sweep
ueber den unbekannten Bereich) gefolgt vom SVGD-Loeser, maximal
`--max_iters` Iterationen, mit vorzeitigem Abbruch, sobald der ergodische
Fehler konvergiert (`konvergenz_metric='ergodic'` in `ergodic_solver.py`).

Start- und Basis-Zieldichte bleiben je Variante identisch zur Quellzeile --
nur der unbekannte Bereich (Lage, Form, Flaechenanteil, Dichtegewicht) ist
zufaellig und je Variante verschieden.

Jede Aufgabe schreibt in eine EIGENE Datenbank (Array-Job-Konvention wie in
`run_data_gen.bash`); `merge_unknown_db.py` fuehrt die Teile zusammen.

    python generate_dataset_unknown.py \
        --src_db ergodic_dataset_improved.db \
        --out ergodic_dataset_improved_unknown_part0.db \
        --shapes_from 0 --shapes_to 150
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import zlib

import numpy as np

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)

from generate_dataset_length import init_db, CHECKPOINTS
from shape_library import make_pdf_and_score, pdf_on_grid
from ergodic_solver import run_ergodic_coverage
from unknown_region import build_unknown_variant

K = 10


def _k_grid(K=K):
    k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                     dtype=np.float64)
    Lambda = (1.0 + (k_idx ** 2).sum(axis=1)) ** -1.5
    return k_idx, Lambda


def _phi_k_from_grid(shape_def, k_idx, grid_res=128):
    d_map, _, _ = pdf_on_grid(shape_def, resolution=grid_res)
    w = d_map.reshape(-1).astype(np.float64)
    w = w / w.sum()
    xs = np.linspace(0.0, 1.0, grid_res)
    gy, gx = np.meshgrid(xs, xs, indexing='ij')
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    Fk = np.cos(np.pi * grid[:, None, :] * k_idx[None, :, :]).prod(axis=-1)
    return w @ Fk


def speichern(conn, name, split, shape_def, traj, x0, dt, tsteps, n_iters, laenge):
    params = dict(shape_def)  # type/segments/means/covs/weights/pedestal/unknown_region
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
    p.add_argument('--src_db', default='ergodic_dataset_improved.db')
    p.add_argument('--out', default='ergodic_dataset_improved_unknown.db')
    p.add_argument('--n_variants', type=int, default=2)
    p.add_argument('--max_iters', type=int, default=10000)
    p.add_argument('--step_size', type=float, default=0.01)
    p.add_argument('--h', type=float, default=0.01)
    p.add_argument('--score_scale', type=float, default=1.0)
    p.add_argument('--konvergenz', type=float, default=0.01,
                   help='Relative Aenderung des ergodischen Fehlers, unter '
                        'der der Loeser vorzeitig abbricht.')
    p.add_argument('--area_lo', type=float, default=0.40)
    p.add_argument('--area_hi', type=float, default=0.75)
    p.add_argument('--a_lo', type=float, default=0.60)
    p.add_argument('--a_hi', type=float, default=0.80)
    p.add_argument('--grid_res', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--shapes_from', type=int, default=None)
    p.add_argument('--shapes_to', type=int, default=None)
    a = p.parse_args()

    src = sqlite3.connect(a.src_db)
    rows = src.execute(
        "SELECT shape_name, split, density_params, x0, dt, tsteps"
        " FROM ergodic_pairs ORDER BY id ASC").fetchall()
    src.close()
    if a.shapes_from is not None or a.shapes_to is not None:
        rows = rows[a.shapes_from or 0:a.shapes_to]

    conn = init_db(a.out)
    erledigt = {r[0] for r in conn.execute("SELECT shape_name FROM ergodic_pairs")}

    k_idx, Lambda = _k_grid()

    print(f"  Quelle    : {a.src_db}  ({len(rows)} Basis-Zieldichten in diesem Bereich)")
    print(f"  Ziel      : {a.out}  (fertig: {len(erledigt)})")
    print(f"  Varianten : {a.n_variants} je Zieldichte, Flaeche "
          f"[{a.area_lo:.0%}, {a.area_hi:.0%}], a in [{a.a_lo}, {a.a_hi}]")
    print(f"  max_iters={a.max_iters}  Konvergenz bei <{a.konvergenz:.1%} "
          f"Aenderung des ergodischen Fehlers", flush=True)

    t_start = time.time()
    n_zeilen = 0
    n_total = len(rows) * a.n_variants
    k = 0
    for shape_name, split, dp_str, x0_str, dt, tsteps in rows:
        base_def = json.loads(dp_str)
        x0 = tuple(json.loads(x0_str))

        for v in range(a.n_variants):
            k += 1
            variant_name = f"{shape_name}#unk{v}"
            if variant_name in erledigt:
                continue

            # Ein Keim je (Form, Variante): reproduzierbar und, wichtiger,
            # stabil ueber einen Jobneustart hinweg -- derselbe Keim liefert
            # wieder denselben unbekannten Bereich, nicht einen neuen.
            keim = zlib.crc32(f"{a.seed}:{shape_name}:{v}".encode()) & 0x7FFFFFFF
            rng = np.random.default_rng(keim)

            t0 = time.perf_counter()
            new_def, p_traj_init, info = build_unknown_variant(
                base_def, x0, rng, tsteps=tsteps, dt=dt,
                area_lo=a.area_lo, area_hi=a.area_hi, a_range=(a.a_lo, a.a_hi))

            phi_k = _phi_k_from_grid(new_def, k_idx, grid_res=a.grid_res)
            _, score_fn = make_pdf_and_score(new_def)

            zwischen, _init_traj, laengen = run_ergodic_coverage(
                score_fn, x0=x0, custom_p_traj=p_traj_init, dt=dt, tsteps=tsteps,
                num_iters=a.max_iters, step_size=a.step_size, h=a.h,
                score_scale=a.score_scale, checkpoints=CHECKPOINTS,
                konvergenz_tol=a.konvergenz, konvergenz_metric='ergodic',
                phi_k=phi_k, k_idx=k_idx, Lambda=Lambda)
            n_iters_final, e_final = laengen[-1]
            traj = zwischen[n_iters_final]
            dauer = time.perf_counter() - t0
            # `length`-Spalte bleibt projektweit die tatsächliche Pfadlänge
            # (z.B. von `flow_matching_runner_length.py` so gelesen) -- der
            # ergodische Fehler ist nur ein Lauf-Diagnosewert, keine Pfadlänge.
            pfadlaenge = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

            speichern(conn, variant_name, split, new_def, traj, x0, dt, tsteps,
                      n_iters_final, pfadlaenge)
            n_zeilen += 1

            rest = (n_total - k) * dauer
            print(f"  [{k}/{n_total}] {variant_name:<26} area={info['area_frac']:.2f} "
                  f"a={info['a']:.2f}  n_iters={n_iters_final:5d}  "
                  f"E={laengen[0][1]:.4f}->{e_final:.4f}  {dauer:5.1f}s  "
                  f"(Rest ~{rest/3600:.2f} h)", flush=True)

    conn.close()
    print(f"\n  {n_zeilen} Zeilen ergaenzt in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
