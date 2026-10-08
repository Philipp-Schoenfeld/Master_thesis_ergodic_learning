r"""
filter_best_ergodicity.py
==========================
Reduziert `ergodic_dataset_length.db` auf eine Zeile je Zieldichte.

Je Dichte liegen dort bis zu 19 Varianten vor, die sich nur in der Anzahl
SVGD-Solver-Iterationen (`n_iters`) unterscheiden. Dieses Skript waehlt je
Dichte die Variante mit dem niedrigsten ergodischen Fehler zur Zieldichte aus
und schreibt genau eine Zeile pro Dichte in eine neue Datenbank.

Gruppierung ueber den exakten `density_params`-String (nicht `shape_name`) --
dieselbe Konvention wie in `flow_matching_runner_length.py`, siehe dortiger
Kommentar: "Geschluesselt wird ueber die Dichteparameter selbst, nicht ueber
den Formnamen -- das ist exakt und nicht bloss plausibel."

Metrik: dieselbe gewichtete Fourier-Spektraldistanz, die der TSVEC/SVGD-Solver
selbst minimiert (`ergodic_energy_torch.py`, K=10, 100 Moden):

    E = 0.5 * sum_k  Lambda_k * (c_k - phi_k)^2

ausgewertet auf den rohen, vom Solver erzeugten Trajektorienpunkten (keine
B-Spline-Rendering noetig, die DB speichert bereits dichte Punkte), gegen
phi_k aus dem 128x128-Dichtegitter der jeweiligen Zieldichte.

Usage:
    python filter_best_ergodicity.py \
        --db ergodic_dataset_length.db --out ergodic_dataset_improved.db
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset_length import init_db
from shape_library import pdf_on_grid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ergodic_energy_torch import make_k_grid, target_coeffs_from_grid, coeffs_from_points

K = 10
GRID_RES = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='ergodic_dataset_length.db')
    p.add_argument('--out', default='ergodic_dataset_improved.db')
    p.add_argument('--grid_res', type=int, default=GRID_RES)
    args = p.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f'Datenbank nicht gefunden: {args.db}')
    if os.path.exists(args.out):
        sys.exit(f'Zieldatei existiert bereits, wird nicht ueberschrieben: {args.out}')

    quelle = sqlite3.connect(args.db)
    rows = quelle.execute(
        "SELECT shape_name, split, density_params, trajectory, x0, dt,"
        " tsteps, n_iters, length, generated_at FROM ergodic_pairs").fetchall()
    quelle.close()
    print(f'  {len(rows)} Zeilen aus {args.db} gelesen')

    gruppen = defaultdict(list)
    for r in rows:
        gruppen[r[2]].append(r)  # density_params-String als Schluessel

    print(f'  {len(gruppen)} distinkte Zieldichten')

    k_idx, Lambda = make_k_grid(K)
    k_idx_t = torch.from_numpy(k_idx).float()
    Lambda_t = torch.from_numpy(Lambda).float()

    ziel = init_db(args.out)
    t_start = time.time()
    gewonnene_n_iters = []

    for i, (dp_str, varianten) in enumerate(gruppen.items()):
        params = json.loads(dp_str)
        d_map, _, _ = pdf_on_grid(params, resolution=args.grid_res)
        phi_k = target_coeffs_from_grid(torch.from_numpy(np.asarray(d_map)).float(),
                                        k_idx_t)

        trajs = np.stack([np.frombuffer(v[3], dtype=np.float32).reshape(-1, 2)
                          for v in varianten])
        c_k = coeffs_from_points(torch.from_numpy(trajs), k_idx_t)
        diff = c_k - phi_k.unsqueeze(0)
        err = 0.5 * (Lambda_t * diff.pow(2)).sum(dim=-1)

        best = int(torch.argmin(err).item())
        r = varianten[best]
        gewonnene_n_iters.append(r[7])

        ziel.execute(
            "INSERT INTO ergodic_pairs (shape_name, split, density_params,"
            " trajectory, x0, dt, tsteps, n_iters, length, generated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)", r)

        if (i + 1) % 100 == 0 or (i + 1) == len(gruppen):
            dt = time.time() - t_start
            eta = dt / (i + 1) * (len(gruppen) - i - 1)
            print(f'  [{i+1}/{len(gruppen)}]  {dt:6.1f}s vergangen, '
                  f'ETA {eta:5.1f}s', flush=True)

    ziel.commit()
    print(f'\n  {len(gruppen)} Zeilen in {args.out} geschrieben '
          f'({time.time()-t_start:.1f}s fuer Dichte+Metrik-Berechnung)')

    n_iters_arr = np.array(gewonnene_n_iters)
    print('  Verteilung der gewaehlten n_iters:')
    for val, cnt in zip(*np.unique(n_iters_arr, return_counts=True)):
        print(f'    {val:6d} Iterationen: {cnt:4d}x gewaehlt')

    for split, cnt in ziel.execute(
            "SELECT split, count(*) FROM ergodic_pairs GROUP BY split"):
        print(f'  split={split}: {cnt}')

    ziel.close()


if __name__ == '__main__':
    main()
