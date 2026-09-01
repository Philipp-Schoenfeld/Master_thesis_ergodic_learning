#!/usr/bin/env python3
r"""
run_all.py
==========
Run all five inference-time constraints over the full holdout split and
collect one summary table.

Each constraint owns its folder and its own runner; this only sequences them
and aggregates the per-constraint CSVs into `results/summary.csv`, so a single
command reproduces every figure and every number in one go.

    python run_all.py                # full holdout, all constraints
    python run_all.py --shapes A digit_5     # quick smoke run
"""
import argparse
import csv
import os
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from common import summarise  # noqa: E402

# (folder, runner, extra args, the pair of columns that says "did it work")
JOBS = [
    ('01_keepin_workspace', 'run_keepin.py', ['--region', 'box'],
     ('austritt_frei', 'austritt_constr')),
    ('01_keepin_workspace', 'run_keepin.py', ['--region', 'circle'],
     ('austritt_frei', 'austritt_constr')),
    ('02_waypoint_anchor', 'run_waypoints.py', ['--pins', 'start_via_end'],
     ('pin_fehler_frei', 'pin_fehler_constr')),
    ('03_max_curvature', 'run_curvature.py', [],
     ('kappa_peak_frei', 'kappa_peak_constr')),
    ('04_path_length', 'run_length.py', ['--ratio', '0.7'],
     ('rel_fehler_frei', 'rel_fehler_constr')),
    ('04_path_length', 'run_length.py', ['--ratio', '1.3'],
     ('rel_fehler_frei', 'rel_fehler_constr')),
    ('05_se3_alignment', 'run_alignment.py', [],
     ('cos_tn_basis', 'cos_tn_align')),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--python', default=sys.executable)
    args = p.parse_args()

    extra = (['--shapes'] + args.shapes) if args.shapes else []
    out_dir = os.path.join(_here, 'results')
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for folder, runner, job_args, (col_before, col_after) in JOBS:
        cmd = [args.python, os.path.join(_here, folder, runner)] + job_args + extra
        label = f"{folder} {' '.join(job_args)}".strip()
        print(f"\n{'=' * 78}\n>>> {label}\n{'=' * 78}")
        r = subprocess.run(cmd, cwd=os.path.join(_here, folder))
        if r.returncode != 0:
            print(f"!!! {label} failed with exit code {r.returncode}")
            continue

        res = os.path.join(_here, folder, 'results')
        seen = {(s['constraint'], s['variante']) for s in summary}
        for fn in sorted(os.listdir(res)):
            # A folder accumulates one CSV per variant (e.g. length 0.7 and
            # 1.3), and every job re-scans the whole folder -- skip the ones a
            # previous job already recorded instead of double-counting them.
            if not fn.endswith('_metrics.csv'):
                continue
            if (folder, fn.replace('_metrics.csv', '')) in seen:
                continue
            with open(os.path.join(res, fn), encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            if not rows or col_before not in rows[0]:
                continue
            before = [float(x[col_before]) for x in rows]
            after = [float(x[col_after]) for x in rows]
            summary.append({
                'constraint': folder, 'variante': fn.replace('_metrics.csv', ''),
                'n': len(rows), 'metrik': col_before.rsplit('_', 1)[0],
                'vorher': sum(before) / len(before),
                'nachher': sum(after) / len(after),
            })

    if summary:
        path = os.path.join(out_dir, 'summary.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\n{'=' * 78}\nGesamtuebersicht  ->  {path}\n{'=' * 78}")
        print(f"{'Constraint':<22}{'Variante':<26}{'n':>4}{'vorher':>12}{'nachher':>12}")
        for s in summary:
            print(f"{s['constraint']:<22}{s['variante']:<26}{s['n']:>4}"
                  f"{s['vorher']:>12.4f}{s['nachher']:>12.4f}")


if __name__ == '__main__':
    main()
