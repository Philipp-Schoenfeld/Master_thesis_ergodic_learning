r"""
combine_projection_results.py -- merge a base run's and a projection run's derived results
==========================================================================================
Combines the plot inputs of two result folders WITHOUT touching or copying the (large) run
tables: `summary.csv` and `convergence.npz` of `--base_tag` (the existing comparison, e.g. v6)
and `--proj_tag` (`project_waypoint_variants.py`, after `compute_convergence.py` and
`plot_smoothness_comparison.py` ran on it) are concatenated along the variant axis into
`results/<out_tag>/`, together with a light `smoothness_comparison.db` that holds only the config
and the `convergence` table (what `plot_convergence_linear.py` reads). Afterwards
`plot_smoothness_overview.py` and `plot_convergence_linear.py` work on `--out_tag` as usual.

Usage:
  python combine_projection_results.py --base_tag smoothness_comparison_v6_20261012 \
      --proj_tag smoothness_comparison_v6_wpproj_20261026 --out_tag smoothness_comparison_v6_with_proj_20261026
"""
import argparse
import csv
import os

import numpy as np

import smoothness_db as sdb

_here = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base_tag', required=True)
    ap.add_argument('--proj_tag', required=True)
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    res = lambda tag, f: os.path.join(_here, 'results', tag, f)               # noqa: E731
    os.makedirs(res(args.out_tag, ''), exist_ok=True)

    # summary.csv: base rows, then the projected rows
    with open(res(args.base_tag, 'summary.csv')) as f:
        base_rows = list(csv.DictReader(f))
        fields = list(base_rows[0].keys())
    with open(res(args.proj_tag, 'summary.csv')) as f:
        proj_rows = list(csv.DictReader(f))
    with open(res(args.out_tag, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(base_rows + proj_rows)

    # convergence.npz: concatenate along the variant axis (same shapes and checkpoints required)
    a, b = np.load(res(args.base_tag, 'convergence.npz')), np.load(res(args.proj_tag, 'convergence.npz'))
    assert list(a['shapes']) == list(b['shapes']), 'shape lists differ'
    assert np.array_equal(a['its'], b['its']), 'iteration checkpoints differ (different n_iters?)'
    dup = set(a['variants']) & set(b['variants'])
    assert not dup, f'variants in both runs: {dup}'
    variants = list(a['variants']) + list(b['variants'])
    curves = np.concatenate([a['curves'], b['curves']], axis=1)
    n90 = np.concatenate([a['n90'], b['n90']], axis=1)
    np.savez(res(args.out_tag, 'convergence.npz'), its=a['its'], shapes=a['shapes'],
             variants=np.array(variants), curves=curves, n90=n90)

    # light DB: config + convergence table
    conn = sdb.open_db(res(args.out_tag, 'smoothness_comparison.db'))
    sdb.set_meta(conn, 'config', dict(shapes=[str(s) for s in a['shapes']],
                                      variants=[str(v) for v in variants],
                                      n_iters=int(a['its'][-1]), base_tag=args.base_tag,
                                      proj_tag=args.proj_tag))
    for si, shape in enumerate(a['shapes']):
        for vi, v in enumerate(variants):
            sdb.save_convergence(conn, str(shape), str(v), a['its'], curves[si, vi], n90[si, vi])
    conn.close()
    print(f'[combine] {len(variants)} variants x {len(a["shapes"])} shapes -> {res(args.out_tag, "")}')


if __name__ == '__main__':
    main()
