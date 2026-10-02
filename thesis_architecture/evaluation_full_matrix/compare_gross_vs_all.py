#!/usr/bin/env python3
r"""
compare_gross_vs_all.py
=========================
Stellt den besten *bewerteten* Fund der Cluster-Studie `ideal_v2_gross_cluster`
(`mi_optuna_gross`, siehe `variant_runner.STRATEGIES`) den 9 bisherigen
Strategien aus `results/full_run_with_optuna_ideal_v2_20260916/` gegenueber —
ohne diese neu zu rechnen. Gleiches Muster wie `compare_optuna_ideal_v2.py`:
Zeilen zusammenfuehren, dieselbe Zusammenfassungs-/Abbildungspipeline aus
`run_eval_matrix.py` fahren.

    python compare_gross_vs_all.py
"""

import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from run_eval_matrix import summarise, plot_metric_bars, plot_tradeoff, NUMERIC_METRIC_KEYS  # noqa: E402
from metrics_explore_exploit import add_J  # noqa: E402

BASE_RUN = os.path.join(_here, 'results', 'full_run_with_optuna_ideal_v2_20260916', 'tables', 'all_runs.csv')
NEW_RUN = os.path.join(_here, 'results', 'mi_optuna_gross_run_20260917', 'tables', 'all_runs.csv')
OUT_DIR = os.path.join(_here, 'results', 'gross_vs_alle')


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC_METRIC_KEYS:
            if r.get(k) not in (None, ''):
                r[k] = float(r[k])
        if r.get('svgd_iters') not in (None, ''):
            r['svgd_iters'] = int(float(r['svgd_iters']))
        if r.get('J') in (None, ''):
            add_J(r)
    return rows


def main():
    base_rows = load_rows(BASE_RUN)
    new_rows = load_rows(NEW_RUN)
    rows = base_rows + new_rows

    tables_dir = os.path.join(OUT_DIR, 'tables')
    plots_dir = os.path.join(OUT_DIR, 'plots')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    n_variants = len({r['variant_id'] for r in rows})
    print(f"{len(rows)} Zeilen, {n_variants} Varianten "
         f"({len(base_rows)} aus full_run_with_optuna_ideal_v2_20260916, "
         f"{len(new_rows)} aus mi_optuna_gross_run_20260917) -> {OUT_DIR}")


if __name__ == '__main__':
    main()
