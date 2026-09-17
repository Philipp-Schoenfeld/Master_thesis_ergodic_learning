#!/usr/bin/env python3
r"""
compare_optuna_ideal_v2.py
===========================
Stellt den neuen Optuna-Fund (`eid_optuna_ideal_v2`, siehe
`variant_runner.STRATEGIES`) den acht bisherigen, von Hand getunten
Strategien aus `results/ideal_run_20260912/` gegenueber — ohne diese neu zu
rechnen, denn ihre Parameter haben sich durch den Umbau in `variant_runner.py`
nachweislich nicht geaendert (siehe Regressionscheck in der Sitzung).

Liest beide `all_runs.csv`, haengt sie zusammen und fuehrt danach exakt
dieselbe Zusammenfassungs-/Abbildungspipeline wie `run_eval_matrix.py`/
`run_ideal_matrix.py` (`summarise`, `plot_metric_bars`, `plot_tradeoff`) —
keine neue Logik, nur ein dritter Aufruf derselben Funktionen auf den
zusammengefuehrten Zeilen.

Baseline-Zeilen (Maeander/Irrfahrt) stehen in beiden Quell-CSVs, mit
identischen Werten (gleicher Seed, gleiche Formreihenfolge) — sie werden aus
der alten Quelle beim Zusammenfuehren entfernt, damit sie nicht doppelt in
die Mittelwerte eingehen.

    python compare_optuna_ideal_v2.py
"""

import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from run_eval_matrix import summarise, plot_metric_bars, plot_tradeoff, NUMERIC_METRIC_KEYS  # noqa: E402

OLD_RUN = os.path.join(_here, 'results', 'ideal_run_20260912', 'tables', 'all_runs.csv')
NEW_RUN = os.path.join(_here, 'results', 'optuna_ideal_v2_run_20260916', 'tables', 'all_runs.csv')
OUT_DIR = os.path.join(_here, 'results', 'optuna_ideal_v2_vs_bisherige')

BASELINE_METHODS = {'lawnmower', 'random_walk'}


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC_METRIC_KEYS:
            if r.get(k) not in (None, ''):
                r[k] = float(r[k])
        if r.get('svgd_iters') not in (None, ''):
            r['svgd_iters'] = int(float(r['svgd_iters']))
    return rows


def main():
    old_rows = [r for r in load_rows(OLD_RUN) if r.get('method') not in BASELINE_METHODS]
    new_rows = load_rows(NEW_RUN)
    rows = old_rows + new_rows

    tables_dir = os.path.join(OUT_DIR, 'tables')
    plots_dir = os.path.join(OUT_DIR, 'plots')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    n_variants = len({r['variant_id'] for r in rows})
    print(f"{len(rows)} Zeilen, {n_variants} Varianten "
         f"({len(old_rows)} aus ideal_run_20260912 ohne Baselines, "
         f"{len(new_rows)} aus optuna_ideal_v2_run_20260916) -> {OUT_DIR}")


if __name__ == '__main__':
    main()
