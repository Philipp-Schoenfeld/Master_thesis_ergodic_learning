#!/usr/bin/env python3
r"""
run_all.py
==========
Alle fuenf Varianten auf denselben Formen, mit demselben Glauben, demselben
Messprozess und denselben Zufallszahlen. Nur der Planungsansatz wechselt.

Die Tabelle ist bewusst mehrspaltig: eine Mission ist nur dann gut, wenn sie
die wahre Dichte abdeckt (`coverage`), dabei etwas lernt (`info_gain`) und das
Gelernte auch stimmt (`belief_rmse`). Ein einzelner Skalar verdeckt genau den
Zielkonflikt, um den es geht.

`path_len` ist die wichtigste Nebenspalte: Variante E faehrt zwei volle
Trajektorien und muss die anderen deshalb deutlich schlagen, nicht knapp.
`plan_s` ist die zweite: bei C und D wird mehrfach pro Mission geplant, und
genau das ist die Rechnung, die ein amortisierter Generator unterbietet.

    python run_all.py --shapes 4
"""

import argparse
import importlib.util
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from common.data import load_truth          # noqa: E402
from common.planner import GradientPlanner, ModelPlanner   # noqa: E402
from common import metrics                  # noqa: E402


def _load(variant_dir, module):
    path = os.path.join(_here, variant_dir, module + '.py')
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=3)
    p.add_argument('--grid_res', type=int, default=32)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--lambda_cov', type=float, default=20000.0)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--checkpoint', default=None,
                   help='Trainiertes Netz statt Gradientenplaner — der '
                        'Vergleich, um den es in der Arbeit geht.')
    p.add_argument('--out', default=None, help='CSV-Ausgabe.')
    a = p.parse_args()

    A = _load('variant_a_combined', 'run_a')
    B = _load('variant_b_diffsim', 'run_b')
    C = _load('variant_c_receding', 'run_c')
    D = _load('variant_d_belief_cond', 'run_d')
    E = _load('variant_e_two_stage', 'run_e')

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print(f"\n=== Alle Varianten, {len(names)} Formen: {', '.join(names)} ===")
    print(f"  Planer: {'Netz ' + os.path.basename(a.checkpoint) if a.checkpoint else 'Gradient'}"
          f"  |  Metrik: {a.metric}  |  n_prior={a.n_prior}\n")

    def mk(seed, nxi=25):
        if a.checkpoint:
            return ModelPlanner(a.checkpoint, nxi=nxi)
        return GradientPlanner(nxi=nxi, metric=a.metric, steps=a.steps, seed=seed)

    common = dict(n_prior=a.n_prior, grid_res=a.grid_res,
                  n_particles=a.n_particles)
    per_variant = {k: [] for k in 'ABCDE'}

    for i in range(len(names)):
        truth = T[i]
        per_variant['A'].append(A.run_mission(
            truth, mk(i), kappa=a.kappa, seed=i, label='A', **common)[1])
        per_variant['B'].append(B.run_mission(
            truth, GradientPlanner(metric=a.metric, steps=1, seed=i),
            lambda_cov=a.lambda_cov, steps=a.steps, seed=i, label='B',
            **common)[1])
        per_variant['C'].append(C.run_mission(
            truth, mk, n_segments=a.segments, seed=i, label='C', **common)[1])
        per_variant['D'].append(D.run_mission(
            truth, mk, rounds=a.rounds, seed=i, label='D', **common)[1])
        per_variant['E'].append(E.run_mission(
            truth, mk, seed=i, label='E', **common)[1])

    titles = {
        'A': 'A  kombinierte Dichte',
        'B': 'B  diff. Vorausschau',
        'C': 'C  receding horizon',
        'D': 'D  glaubenskonditioniert',
        'E': 'E  zweistufig (Baseline)',
    }
    rows = []
    for k in 'ABCDE':
        per = per_variant[k]
        keys = ['coverage', 'info_gain', 'belief_rmse', 'path_len', 'n_obs',
                'plan_s']
        m = {kk: sum(r.get(kk, 0.0) for r in per) / len(per) for kk in keys}
        m['variant'] = titles[k]
        rows.append(m)

    metrics.print_table(rows, "Mittel ueber die Formen")
    print("\n  coverage: kleiner besser · info_gain: groesser besser")
    print("  belief_rmse: kleiner besser · path_len und plan_s: Kostenspalten\n")

    if a.out:
        import csv
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  geschrieben nach {a.out}\n")


if __name__ == '__main__':
    main()
