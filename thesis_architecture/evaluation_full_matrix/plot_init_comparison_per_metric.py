r"""
plot_init_comparison_per_metric.py
=====================================
One standalone, high-information plot PER METRIC (Philipp's request,
2026-10-20: "für jede Metrik einen eigenen, übersichtlichen Plot, mit
höchster Aussagekraft") for `run_init_comparison.py`'s 7-method x 2-solver
comparison.

Form choice: a horizontal box plot (median + IQR + whiskers) of the 25
per-shape values, ONE row per (method, solver), sorted best-to-worst (all 6
metrics here are "lower is better"), with the raw 25 shape values overlaid
as a jittered strip -- more informative than a bare mean+-std bar (which
assumes a symmetric spread and hides outlier shapes/multimodality) while
staying a single, standard, publication-ready form per metric, as asked.
Boxes reuse the same per-shape loaders as `plot_init_comparison_overview.py`
(`ErgodicSeriesCache.load`/`coverage_final`, DB series, `n90_from_series`) --
no new aggregation logic, just a different, more informative presentation of
the same per-shape values already computed for the dashboard/ranking plots.

Axis scale per metric is set from the ACTUAL data spread (not guessed):
ergodic error (~39x), smoothness (~11x) and n90 (~1560x) get a log x-axis;
coverage (~2.7x), path length (~2.8x) and outside-fraction (has exact 0.0
entries, so log is invalid anyway) get a linear one.

Usage
-----
    python plot_init_comparison_per_metric.py --out_tag init_comparison_YYYYMMDD
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

import init_comparison_db as idb                                    # noqa: E402
from plot_init_comparison import (                                  # noqa: E402
    METHODS, SOLVERS, METHOD_COLORS, METHOD_LABELS, SOLVER_LABELS, ErgodicSeriesCache, _style,
)
from plot_init_comparison_overview import _per_shape_final, _n90_per_shape  # noqa: E402

#: (key, title, x-label, log-x, loader). `loader(ergo, conn, shapes, variant)
#: -> (n_shapes,) float.
METRICS = (
    ('ergodic', 'Ergodic error', 'Ergodic error vs. true density (lower = better coverage)',
     True, lambda ergo, conn, shapes, v: _per_shape_final(ergo, conn, shapes, v, 'ergodic')),
    ('coverage', 'Coverage', 'Coverage distance: density-weighted mean distance to path '
     '(lower = better)', False,
     lambda ergo, conn, shapes, v: _per_shape_final(ergo, conn, shapes, v, 'coverage')),
    ('smooth', 'Smoothness', 'Smoothness cost (lower = smoother)', True,
     lambda ergo, conn, shapes, v: _per_shape_final(ergo, conn, shapes, v, 'smooth')),
    ('path_len', 'Path length', 'Path length (domain widths)', False,
     lambda ergo, conn, shapes, v: _per_shape_final(ergo, conn, shapes, v, 'path_len')),
    ('outside', 'Outside fraction', 'Fraction of steps outside the target support '
     '(lower = better)', False,
     lambda ergo, conn, shapes, v: _per_shape_final(ergo, conn, shapes, v, 'outside')),
    ('n90', 'Time to convergence (n90)', 'SVGD iterations to 90% of final improvement '
     '(lower = faster)', True,
     lambda ergo, conn, shapes, v: _n90_per_shape(ergo, shapes, v)),
)


def plot_metric(key, title, xlabel, log_x, loader, ergo, conn, shapes, variants, out_path):
    entries = []
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            vals = loader(ergo, conn, shapes, variant)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            entries.append(dict(method=method, solver=solver, vals=vals, median=np.median(vals)))
    if not entries:
        return
    entries.sort(key=lambda e: e['median'])

    n = len(entries)
    fig, ax = plt.subplots(figsize=(9.5, 0.52 * n + 1.6), facecolor='white')
    ys = np.arange(n)[::-1]

    # jittered strip of the raw per-shape values, BEHIND the boxes -- shows
    # outlier shapes and multimodality a mean+-std bar would hide.
    rng = np.random.default_rng(0)
    for y, e in zip(ys, entries):
        jitter = rng.uniform(-0.28, 0.28, size=e['vals'].shape)
        ax.scatter(e['vals'], y + jitter, color=METHOD_COLORS[e['method']], s=10, alpha=0.35,
                  zorder=2, edgecolors='none')

    bp = ax.boxplot([e['vals'] for e in entries], positions=ys, vert=False, widths=0.6,
                    patch_artist=True, showfliers=False, zorder=3,
                    medianprops=dict(color='#1A1A2E', linewidth=1.8),
                    whiskerprops=dict(color='#1A1A2E', linewidth=1.0),
                    capprops=dict(color='#1A1A2E', linewidth=1.0))
    for box, e in zip(bp['boxes'], entries):
        box.set_facecolor(METHOD_COLORS[e['method']])
        box.set_alpha(0.85)
        box.set_edgecolor('#1A1A2E')
        box.set_linewidth(1.0 if e is entries[0] else 0.6)
        if e['solver'] == 'adam_svgd':
            box.set_hatch('///')

    best = entries[0]
    for y, e in zip(ys, entries):
        weight = 'bold' if e is best else 'normal'
        label = f"{METHOD_LABELS[e['method']]} + {SOLVER_LABELS[e['solver']]}"
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center',
               fontsize=8, color='#1A1A2E', fontweight=weight)

    ax.set_yticks([])
    ax.set_ylim(-0.7, n - 0.3)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_title(f'{title} -- {len(shapes)} holdout shapes, final iteration', fontsize=11,
                color='#1A1A2E')
    fig.text(0.99, 0.005, 'box = median/IQR over shapes  |  dots = individual shapes  |  '
            'hatched = Adam-SVGD, solid = TSVEC-SVGD', ha='right', va='bottom', fontsize=7,
            color='#555')
    _style(ax)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'init_comparison.db')
    conn = idb.open_db(db_path)
    cfg = idb.get_meta(conn, 'config', {})
    shapes = cfg.get('shapes') or sorted({r[0] for r in conn.execute("SELECT DISTINCT shape FROM runs")})
    variants = sorted({r[0] for r in conn.execute("SELECT DISTINCT variant FROM runs")})
    plot_dir = os.path.join(out_dir, 'plots', 'per_metric')
    os.makedirs(plot_dir, exist_ok=True)

    ergo = ErgodicSeriesCache(conn, cache_path=os.path.join(out_dir, 'metric_cache.pkl'))

    for key, title, xlabel, log_x, loader in METRICS:
        out_path = os.path.join(plot_dir, f'metric_{key}.png')
        plot_metric(key, title, xlabel, log_x, loader, ergo, conn, shapes, variants, out_path)
        print(f"[plot_per_metric] {out_path} written")

    ergo.save()


if __name__ == '__main__':
    main()
