#!/usr/bin/env python3
r"""
regen_metric_bars.py
=====================
Regeneriert nur die `metric_bars`-Abbildungen eines bereits gelaufenen
`run_eval_matrix.py`/`run_ideal_matrix.py`-Durchlaufs aus der abgelegten
`tables/all_runs.csv` -- ohne irgendetwas neu zu rechnen. Gedacht fuer
Aenderungen an `plot_metric_bars` selbst (neue Metrik, neue Formel-
Beschriftung), die auf bereits vorhandene Ergebnisse angewendet werden sollen.

    python regen_metric_bars.py full_run_with_optuna_ideal_v2_20260916
    python regen_metric_bars.py ideal_run_20260912
"""

import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from run_eval_matrix import plot_metric_bars, NUMERIC_METRIC_KEYS  # noqa: E402
from metrics_explore_exploit import add_J  # noqa: E402


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
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out_tag = sys.argv[1]
    run_dir = os.path.join(_here, 'results', out_tag)
    csv_path = os.path.join(run_dir, 'tables', 'all_runs.csv')
    plots_dir = os.path.join(run_dir, 'plots')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = load_rows(csv_path)
    plot_metric_bars(rows, plots_dir)
    print(f"{len(rows)} Zeilen aus {csv_path} -> {os.path.join(plots_dir, 'metric_bars')}")


if __name__ == '__main__':
    main()
