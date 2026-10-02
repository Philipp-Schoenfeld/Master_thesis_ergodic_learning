#!/usr/bin/env python3
r"""
merge_mlp_into_run.py
======================
Mischt die MLP-Policy-Zeilen aus `results/<mlp_tag>/tables/all_runs.csv`
in die grosse CSV eines bestehenden Laufs (`results/<base_tag>/tables/
all_runs.csv`) und erzeugt eine neue Kopie der metric_bars-Plots mit der
MLP-Variante darin.

Die Ausgabe-Plots landen in einem eigenen Unterverzeichnis:
    results/<base_tag>/plots/metric_bars_with_mlp/

So bleibt der urspruengliche Lauf unveraendert, und der direkte Vergleich
existiert als eigene Plot-Kopie.

Beispiel
--------
    python merge_mlp_into_run.py \
        --base_tag full_run_with_optuna_ideal_v2_20260916 \
        --mlp_tag  mlp_policy_eval_20260917

    # Nur die Plots neu erzeugen (CSV bereits gemergt):
    python merge_mlp_into_run.py \
        --base_tag full_run_with_optuna_ideal_v2_20260916 \
        --mlp_tag  mlp_policy_eval_20260917 \
        --plots_only
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from run_eval_matrix import (plot_metric_bars, NUMERIC_METRIC_KEYS,  # noqa: E402
                             plot_tradeoff)
from metrics_explore_exploit import add_J                            # noqa: E402


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_tag', type=str, required=True,
                    help='Unterordner in results/ mit dem bestehenden Lauf.')
    ap.add_argument('--mlp_tag', type=str, required=True,
                    help='Unterordner in results/ mit dem mlp_policy_eval-Lauf.')
    ap.add_argument('--plots_only', action='store_true',
                    help='Nur Plots neu erzeugen, CSV nicht erneut schreiben.')
    ap.add_argument('--out_subdir', type=str, default='metric_bars_with_mlp',
                    help='Name des Plot-Unterordners in plots/ des base-Laufs.')
    args = ap.parse_args()

    base_dir = os.path.join(_here, 'results', args.base_tag)
    mlp_dir = os.path.join(_here, 'results', args.mlp_tag)
    base_csv = os.path.join(base_dir, 'tables', 'all_runs.csv')
    mlp_csv = os.path.join(mlp_dir, 'tables', 'all_runs.csv')

    if not os.path.isfile(base_csv):
        raise FileNotFoundError(f"Basis-CSV nicht gefunden: {base_csv}")
    if not os.path.isfile(mlp_csv):
        raise FileNotFoundError(f"MLP-CSV nicht gefunden: {mlp_csv}")

    base_rows = load_rows(base_csv)
    mlp_rows = load_rows(mlp_csv)

    # Duplikat-Check: keine MLP-variant_ids, die bereits in der Basis sind
    base_vids = {r['variant_id'] for r in base_rows}
    mlp_vids = {r['variant_id'] for r in mlp_rows}
    new_vids = mlp_vids - base_vids
    skipped = mlp_vids & base_vids
    if skipped:
        print(f"[merge] Uebersprungen (bereits in Basis): {sorted(skipped)}")
    if not new_vids:
        print("[merge] Keine neuen MLP-Varianten -- nur Plots werden neu erzeugt.")
    else:
        print(f"[merge] Neue MLP-Varianten: {sorted(new_vids)}")

    # Alle Spalten der kombinierten Tabelle bestimmen
    all_rows = base_rows + [r for r in mlp_rows if r['variant_id'] in new_vids]

    merged_csv = os.path.join(base_dir, 'tables', 'all_runs_with_mlp.csv')
    if not args.plots_only:
        keys = sorted({k for r in all_rows for k in r.keys()})
        os.makedirs(os.path.dirname(merged_csv), exist_ok=True)
        with open(merged_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"[merge] {len(all_rows)} Zeilen -> {merged_csv}")

    # Plots in Kopie erzeugen
    plots_dir_base = os.path.join(base_dir, 'plots')
    plots_out = os.path.join(plots_dir_base, args.out_subdir)
    # metric_bars landen in plots_out/metric_bars/
    plot_metric_bars(all_rows, plots_out)
    plot_tradeoff(all_rows, plots_out)
    print(f"[merge] Plots -> {plots_out}")


if __name__ == '__main__':
    main()
