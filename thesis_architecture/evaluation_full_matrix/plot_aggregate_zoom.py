"""Zoomed version of plots/aggregate.png (smoothness / path length / ergodic error vs. SVGD iteration).

Same quantities as aggregate.png (mean over the holdout shapes of the per-shape mean over the
candidates), but readable: log-x iteration axis, y windows narrowed to where the variants actually
differ, no std bands (they overlap into one blob), one shared legend underneath. Curves leaving a
window (e.g. the raw-waypoint variants in the smoothness panel) are simply clipped.

Smoothness and path length come from the DB; the ergodic error comes from convergence.npz
(`compute_convergence.py`), so nothing is recomputed from the stored states.

Usage:  python plot_aggregate_zoom.py --out_tag smoothness_comparison_v6_20261012
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

import smoothness_db as sdb
from plot_smoothness_comparison import (VARIANT_COLORS, VARIANT_LABELS, VARIANT_LINESTYLES, VARIANT_LW,
                                        load_series, _style)

_here = os.path.dirname(os.path.abspath(__file__))
INK = '#1A1A2E'
CFM_VARIANTS = {'cfm_bspline_svgd', 'cfm_adam_bspline'}


def db_mean_curve(conn, shapes, variant, series):
    """Mean over shapes of the per-shape candidate mean -> (steps, curve)."""
    per_shape = []
    for shape in shapes:
        steps, mat = load_series(conn, shape, variant, series)
        if steps is None:
            continue
        valid = ~np.all(np.isnan(mat), axis=0)
        per_shape.append(np.nanmean(mat[:, valid], axis=0))
    n = min(len(m) for m in per_shape)
    return np.arange(n), np.stack([m[:n] for m in per_shape]).mean(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--smooth_ylim', type=float, nargs=2, default=(0.03, 3.0))
    ap.add_argument('--path_ylim', type=float, nargs=2, default=(2.0, 7.0))
    ap.add_argument('--ergodic_ylim', type=float, nargs=2, default=(0.01, 30.0))
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    conn = sdb.open_db(os.path.join(out_dir, 'smoothness_comparison.db'))
    shapes = sdb.get_meta(conn, 'config')['shapes']
    conv = np.load(os.path.join(out_dir, 'convergence.npz'))
    c_variants, its = list(conv['variants']), conv['its']
    ergodic = {v: conv['curves'][:, c_variants.index(v)].mean(axis=0) for v in c_variants}

    panels = [('smooth_series', 'Smoothness energy (lower = smoother)', args.smooth_ylim),
              ('path_len_series', 'Path length', args.path_ylim),
              ('ergodic', 'Ergodic error vs. true density (lower = better coverage)', args.ergodic_ylim)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), facecolor='white')
    plain = FuncFormatter(lambda v, _: f'{v:g}')
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        style_ax = _style
        style_ax(ax)
        for variant in c_variants:
            color, cfm = VARIANT_COLORS[variant], variant in CFM_VARIANTS
            if variant == 'cfm_only':
                if key == 'ergodic':
                    y0 = float(ergodic[variant][0])
                else:
                    y0 = float(db_mean_curve(conn, shapes, variant, key)[1][0])
                ax.axhline(y0, color=color, lw=1.8, ls='--', label=VARIANT_LABELS[variant])
                continue
            if key == 'ergodic':
                x, y = its[1:], ergodic[variant][1:]
            else:
                x, y = db_mean_curve(conn, shapes, variant, key)
                x, y = x[1:], y[1:]
            ax.plot(x, y, color=color, lw=VARIANT_LW[variant] + (0.8 if cfm else 0),
                    ls=VARIANT_LINESTYLES[variant], label=VARIANT_LABELS[variant], zorder=4 if cfm else 3)
        ax.set_xscale('log')
        if key != 'path_len_series':   # path length spans less than one decade: linear axis reads better
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(LogLocator(subs=(2, 5), numticks=20))
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
            ax.tick_params(which='minor', labelsize=7, colors='#555')
        ax.set_ylim(*ylim)
        ax.grid(True, which='minor', alpha=0.1)
        ax.set_xlabel('SVGD iteration')
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f'Smoothness / path length / ergodic error, zoomed -- mean over {len(shapes)} holdout shapes '
                 '(log x, no std bands; curves leaving a window are clipped)', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    out = os.path.join(out_dir, 'plots', 'aggregate_zoom.png')
    fig.savefig(out, dpi=140, facecolor='white')
    print(out)


if __name__ == '__main__':
    main()
