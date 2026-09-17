#!/usr/bin/env python3
r"""
recompute_extra_metrics.py
============================
Adds `smoothness_energy` and `steps_to_full_coverage(_reached)` to an
already finished `run_eval_matrix.py` / `run_ideal_matrix.py` run, computed
from the cached `raw/**/trajectory.npy` files -- no trajectory is
regenerated. Purely local post-processing (minutes, not hours; no GPU, no
cluster job).

Starts from `tables/all_runs.csv` (not `raw/**/metrics.json`): the two
representation-comparison columns `representation`/`strategy`/
`replan_scheme` that `run_ideal_matrix.py` adds are only present in the CSV
rows, not in the individual `metrics.json` files (those are written by
`score_and_save` before that extra `row.update(...)` happens) -- starting
from the JSON files would silently lose those three columns.

    python recompute_extra_metrics.py full_run_with_optuna_ideal_v2_20260916
    python recompute_extra_metrics.py ideal_run_20260912 --truth_res 96
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common.data import load_truth                                # noqa: E402
import variant_runner as vr                                        # noqa: E402
from metrics_explore_exploit import (smoothness_energy,            # noqa: E402
                                     steps_to_full_coverage, add_J)
from run_eval_matrix import (summarise, plot_metric_bars,          # noqa: E402
                             NUMERIC_METRIC_KEYS)


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


def raw_curve_path(raw_dir, row):
    """Reconstructs the `trajectory.npy` path `score_and_save` wrote for
    this row. Knowledge-independent baselines (lawnmower/random_walk) are
    saved once under the literal directory name 'shared', not under their
    (duplicated) `knowledge_condition` value -- see
    `variant_runner.KNOWLEDGE_DEPENDENT` and the `fixed = {...}` loops in
    `run_eval_matrix.py`/`run_ideal_matrix.py`.
    """
    key = (row['method'], row['subvariant'])
    cond_dir = row['knowledge_condition']
    if not vr.KNOWLEDGE_DEPENDENT.get(key, True):
        cond_dir = 'shared'
    return os.path.join(raw_dir, cond_dir, row['method'], row['variant_id'],
                        row['shape'], 'trajectory.npy')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_tag', type=str)
    ap.add_argument('--truth_res', type=int, default=96,
                    help='Must match the resolution the original run used. '
                         'Default 96 is the project default in both '
                         'run_eval_matrix.py and run_ideal_matrix.py -- '
                         'neither run records an override in config.json if '
                         'one was passed, so pass it explicitly if you know '
                         'the run used something else.')
    args = ap.parse_args()

    run_dir = os.path.join(_here, 'results', args.out_tag)
    raw_dir = os.path.join(run_dir, 'raw')
    tables_dir = os.path.join(run_dir, 'tables')
    plots_dir = os.path.join(run_dir, 'plots')
    csv_path = os.path.join(tables_dir, 'all_runs.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = load_rows(csv_path)
    print(f"[recompute_extra_metrics] {len(rows)} Zeilen aus {csv_path}")

    truth_cache = {}

    def _truth(shape_name):
        if shape_name not in truth_cache:
            _names, truths = load_truth(labels=[shape_name], n=999, split='val',
                                        resolution=args.truth_res, device='cpu')
            truth_cache[shape_name] = truths[0]
        return truth_cache[shape_name]

    missing = 0
    for i, row in enumerate(rows):
        p = raw_curve_path(raw_dir, row)
        if not os.path.isfile(p):
            missing += 1
            row['smoothness_energy'] = float('nan')
            row['steps_to_full_coverage'] = float('nan')
            row['steps_to_full_coverage_reached'] = float('nan')
            continue
        curve = torch.from_numpy(np.load(p)).float()
        truth = _truth(row['shape'])

        row['smoothness_energy'] = smoothness_energy(curve)
        steps_frac, steps_reached = steps_to_full_coverage(
            curve, truth, knowledge_condition=row['knowledge_condition'])
        row['steps_to_full_coverage'] = steps_frac
        row['steps_to_full_coverage_reached'] = float(steps_reached)

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(rows)}]")

    if missing:
        print(f"[recompute_extra_metrics] [warn] {missing} Zeilen ohne "
             f"trajectory.npy -- neue Metriken dort NaN, unveraendert sonst.")

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    print(f"[recompute_extra_metrics] fertig -> {tables_dir}, {plots_dir}")


if __name__ == '__main__':
    main()
