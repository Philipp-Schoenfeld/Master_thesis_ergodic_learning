"""Backfill summary.csv / convergence.npz into the DB's `summary` / `convergence` tables for a
run that was computed before those tables existed (`smoothness_db.save_summary` /
`save_convergence`, added 2026-10-12). New runs write both the files and the DB tables directly
(`plot_smoothness_comparison.write_summary_csv`, `compute_convergence.py`); this is only for
results produced before that.

Usage:  python backfill_results_to_db.py --out_tag smoothness_comparison_v6_20261012
"""
import argparse
import csv
import os

import numpy as np

import smoothness_db as sdb

_here = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    out_dir = os.path.join(_here, 'results', args.out_tag)
    conn = sdb.open_db(os.path.join(out_dir, 'smoothness_comparison.db'))

    csv_path = os.path.join(out_dir, 'summary.csv')
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        sdb.save_summary(conn, rows)
        print(f"[backfill] summary: {len(rows)} rows -> DB")
    else:
        print(f"[backfill] no summary.csv in {out_dir}, skipped")

    npz_path = os.path.join(out_dir, 'convergence.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        its, shapes, variants, curves, n90 = d['its'], d['shapes'], d['variants'], d['curves'], d['n90']
        n = 0
        for si, shape in enumerate(shapes):
            for vi, variant in enumerate(variants):
                curve = curves[si, vi]
                if np.isnan(curve).all():
                    continue
                sdb.save_convergence(conn, str(shape), str(variant), its, curve, n90[si, vi])
                n += 1
        print(f"[backfill] convergence: {n} (shape, variant) rows -> DB")
    else:
        print(f"[backfill] no convergence.npz in {out_dir}, skipped")


if __name__ == '__main__':
    main()
