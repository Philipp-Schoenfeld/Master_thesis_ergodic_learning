r"""
plot_init_comparison_insights.py
===================================
Three further plots (Philipp's request, 2026-10-20: "denke dir noch weitere
aussagekräftige Plots aus"), each answering a question the dashboard/ranking/
per-metric plots don't directly show:

  learned_vs_classical.png   Does the thesis's central hypothesis -- a
                              learned generative prior beats classical
                              initialization heuristics -- actually hold?
                              Diverging bar: %% improvement of each learned
                              method (cfm/selfsup/ebm/dfl) over the BEST
                              classical baseline (linear/random_walk/
                              heuristic, same solver, same metric), per
                              metric. Positive = learned wins.
  speed_vs_quality.png       Pareto scatter, n90 (iterations to converge) vs.
                              final ergodic error -- is a fast method also a
                              good one, or is there a speed/quality
                              trade-off? (Both axes "lower = better", so the
                              bottom-left corner is the ideal.)
  per_shape_heatmap.png      Per-shape robustness: ergodic-error RANK (1 =
                              best of 14) for every (shape, method, solver)
                              cell, shapes grouped by family (letters/GMM/
                              polynomial/organic/CJK, the DB's own order) --
                              reveals whether a method is uniformly good or
                              only wins on easy shapes.

Reuses `plot_init_comparison_overview.py`'s loaders/cache (no reimplemented
aggregation).

Usage
-----
    python plot_init_comparison_insights.py --out_tag init_comparison_YYYYMMDD
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
from plot_init_comparison_overview import (                         # noqa: E402
    _per_shape_final, _n90_per_shape, SOLVER_MARKERS,
)

LEARNED = ('cfm', 'selfsup', 'ebm', 'dfl')
CLASSICAL = ('linear', 'random_walk', 'heuristic')

#: diverging pair (dataviz skill: blue<->red, gray neutral midpoint) -- a
#: DIFFERENT color job from the methods' categorical identity colors above
#: (this chart's color encodes sign of improvement, not which method it is).
DIVERGING_POS, DIVERGING_NEG, DIVERGING_MID = '#2a78d6', '#e34948', '#9b9890'

METRIC_DEFS = (
    ('ergodic', 'Ergodic error'), ('coverage', 'Coverage'), ('smooth', 'Smoothness'),
    ('path_len', 'Path length'), ('outside', 'Outside fraction'), ('n90', 'n90 (convergence)'),
)


def _median_value(ergo, conn, shapes, variant, key):
    vals = (_n90_per_shape(ergo, shapes, variant) if key == 'n90'
           else _per_shape_final(ergo, conn, shapes, variant, key))
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def plot_learned_vs_classical(conn, shapes, variants, out_path, ergo):
    """Small multiples, one diverging-bar panel per metric -- clearer than
    stacking all 6 metrics into one row per method (which needs a legend
    just to say which bar-slice is which metric)."""
    methods_order = [(m, s) for s in SOLVERS for m in LEARNED if f"{m}__{s}" in variants]
    n_rows = len(methods_order)

    classical_best = {}
    for solver in SOLVERS:
        for key, _label in METRIC_DEFS:
            vals = [_median_value(ergo, conn, shapes, f"{m}__{solver}", key) for m in CLASSICAL
                   if f"{m}__{solver}" in variants]
            vals = [v for v in vals if np.isfinite(v)]
            classical_best[(solver, key)] = min(vals) if vals else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(14, 0.42 * n_rows + 3.5), facecolor='white')
    for ax, (key, label) in zip(axes.flat, METRIC_DEFS):
        ys, pcts, colors = [], [], []
        for ri, (method, solver) in enumerate(methods_order):
            cb = classical_best[(solver, key)]
            lv = _median_value(ergo, conn, shapes, f"{method}__{solver}", key)
            if not (np.isfinite(cb) and np.isfinite(lv)) or cb == 0:
                continue
            pct = (cb - lv) / cb * 100.0         # positive = learned better
            ys.append(ri)
            pcts.append(pct)
            colors.append(DIVERGING_POS if pct >= 0 else DIVERGING_NEG)
        ax.barh(ys, pcts, color=colors, alpha=0.9, edgecolor='#1A1A2E', linewidth=0.4, height=0.7)
        ax.axvline(0, color=DIVERGING_MID, lw=1.1)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{METHOD_LABELS[m]} + {SOLVER_LABELS[s]}" for m, s in methods_order]
                           if ax is axes.flat[0] or ax is axes.flat[3] else [], fontsize=7.5)
        ax.set_title(label, fontsize=9, color='#1A1A2E')
        ax.set_xlabel('%% improvement', fontsize=8)
        ax.invert_yaxis()
        _style(ax)

    handles = [plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=DIVERGING_POS,
                          markersize=9, label='Learned method better'),
              plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=DIVERGING_NEG,
                        markersize=9, label='Classical baseline better')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Does the learned prior beat classical initialization? -- '
                f'{len(shapes)} holdout shapes, median per metric, %% improvement over '
                'the best of (linear / random walk / heuristic), same solver',
                fontsize=11, color='#1A1A2E')
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)


def plot_speed_vs_quality(conn, shapes, variants, out_path, ergo):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            ergodic = _median_value(ergo, conn, shapes, variant, 'ergodic')
            n90 = _median_value(ergo, conn, shapes, variant, 'n90')
            if not (np.isfinite(ergodic) and np.isfinite(n90)):
                continue
            color = METHOD_COLORS[method]
            ax.scatter([n90], [ergodic], color=color, marker=SOLVER_MARKERS[solver], s=90,
                      edgecolors='#1A1A2E', linewidths=0.7, zorder=3)
            ax.annotate(f"{METHOD_LABELS[method]} + {SOLVER_LABELS[solver]}", (n90, ergodic),
                       fontsize=6.5, color='#1A1A2E', xytext=(5, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n90 -- SVGD iterations to 90%% of final improvement (lower = faster)')
    ax.set_ylabel('Final ergodic error (lower = better coverage)')
    ax.set_title(f'Speed vs. quality trade-off -- {len(shapes)} holdout shapes, median per variant\n'
                '(bottom-left = ideal: fast AND accurate)', fontsize=10, color='#1A1A2E')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)


def plot_per_shape_heatmap(conn, shapes, variants, out_path, ergo):
    """Ergodic-error RANK (1=best of the 14 variants) per shape -- robustness
    across the holdout set's shape families, not just the aggregate."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#0d366b'])

    method_solver = [(m, s) for s in SOLVERS for m in METHODS if f"{m}__{s}" in variants]
    rank_matrix = np.full((len(shapes), len(method_solver)), np.nan)
    for si, shape in enumerate(shapes):
        vals = []
        for method, solver in method_solver:
            v = _per_shape_final(ergo, conn, [shape], f"{method}__{solver}", 'ergodic')
            vals.append(float(v[0]) if v.size else np.nan)
        order = np.argsort(vals)
        ranks = np.empty(len(vals))
        ranks[order] = np.arange(1, len(vals) + 1)
        ranks[np.isnan(vals)] = np.nan
        rank_matrix[si] = ranks

    n_shapes, n_cols = rank_matrix.shape
    fig, ax = plt.subplots(figsize=(0.55 * n_cols + 2, 0.33 * n_shapes + 1.5), facecolor='white')
    norm = 1 - (rank_matrix - 1) / max(n_cols - 1, 1)
    ax.imshow(norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    for i in range(n_shapes):
        for j in range(n_cols):
            if np.isfinite(rank_matrix[i, j]):
                txt_color = '#1A1A2E' if norm[i, j] < 0.55 else 'white'
                ax.text(j, i, f"{rank_matrix[i, j]:.0f}", ha='center', va='center', fontsize=6,
                       color=txt_color)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{METHOD_LABELS[m]}\n{SOLVER_LABELS[s]}" for m, s in method_solver],
                       fontsize=6.5, rotation=60, ha='right')
    ax.set_yticks(range(n_shapes))
    ax.set_yticklabels(shapes, fontsize=7)
    ax.set_title('Per-shape ergodic-error rank (1 = best of 14) -- robustness across '
                'the holdout set', fontsize=10, color='#1A1A2E')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_shapes + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    fig.tight_layout()
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
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    ergo = ErgodicSeriesCache(conn, cache_path=os.path.join(out_dir, 'metric_cache.pkl'))

    plot_learned_vs_classical(conn, shapes, variants, os.path.join(plot_dir, 'learned_vs_classical.png'), ergo)
    print("[plot_insights] learned_vs_classical.png written")

    plot_speed_vs_quality(conn, shapes, variants, os.path.join(plot_dir, 'speed_vs_quality.png'), ergo)
    print("[plot_insights] speed_vs_quality.png written")

    plot_per_shape_heatmap(conn, shapes, variants, os.path.join(plot_dir, 'per_shape_heatmap.png'), ergo)
    print("[plot_insights] per_shape_heatmap.png written")

    ergo.save()


if __name__ == '__main__':
    main()
