"""Linear-scale alternative to overview_convergence.png's log-log convergence panel.

The log-log view (plot_smoothness_overview.py) is good for comparing RATES across many
decades of iterations, but it visually compresses how far apart the variants actually are in
absolute terms. This script plots the same median-over-shapes ergodic-error curves
(`smoothness_db`'s `convergence` table, written by compute_convergence.py) with LINEAR axes,
as two panels sharing one legend:

  left   full range -- shows the true absolute gap to the slow/non-ergodic variants
         (the Sun jerk-penalised flow dominates the y-range here).
  right  zoomed to the lower cluster (variants whose final error is small) -- so the variants
         that matter for the thesis comparison stay legible instead of being flattened near 0
         by the slow outlier.

Usage:  python plot_convergence_linear.py --out_tag smoothness_comparison_v6cont2_20261012
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import smoothness_db as sdb
from plot_smoothness_comparison import VARIANT_COLORS, VARIANT_LINESTYLES, VARIANT_LW
from plot_smoothness_overview import CFM_VARIANTS, GROUPS

_here = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = '#1A1A2E', '#555555', '#CCCCCC'
#: y-limit of the zoomed right panel -- variants whose final error sits clearly below this stay
#: readable; chosen so every TSVEC/Adam/CFM-only variant fits, only the Sun variants (often >>1)
#: are cut off there (they are what the left, full-range panel is for).
ZOOM_YMAX = 1.3


def style(ax):
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    conn = sdb.open_db(os.path.join(_here, 'results', args.out_tag, 'smoothness_comparison.db'))
    cfg = sdb.get_meta(conn, 'config')
    shapes, variants = cfg['shapes'], cfg['variants']

    flat = [(g, v, lab, mk) for g, items in GROUPS for v, lab, mk in items if v in variants]
    prefix = {'CFM': '', 'TSVEC-SVGD': 'TSVEC: ', 'Adam-SVGD': 'Adam: ', 'Sun flow matching (dynamics)': 'Sun: ',
              'Waypoint solve + B-spline projection': 'Proj. '}

    curves = {}
    for g, v, lab, mk in flat:
        per_shape = [sdb.load_convergence(conn, sh, v) for sh in shapes]
        per_shape = [c for c in per_shape if c is not None]
        if not per_shape:
            continue
        ck = per_shape[0][0]
        mat = np.stack([c[1] for c in per_shape])            # (n_shapes, n_checkpoints)
        curves[v] = (ck, np.nanmedian(mat, axis=0))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.4), facecolor='white')
    for ax, ymax, title in ((axL, None, 'Full range'),
                            (axR, ZOOM_YMAX, f'Zoomed (y ≤ {ZOOM_YMAX})')):
        style(ax)
        for g, v, lab, mk in flat:
            if v not in curves:
                continue
            ck, curve = curves[v]
            c = VARIANT_COLORS[v]
            cfm = v in CFM_VARIANTS
            label = prefix[g] + lab
            if v == 'cfm_only':
                ax.axhline(curve[0], color=c, lw=3.0 if cfm else 2.0, ls=(0, (6, 3)), label=label, zorder=3)
                continue
            ls = VARIANT_LINESTYLES.get(v, '-')
            lw = (VARIANT_LW.get(v, 2.0) + 1.0) if cfm else VARIANT_LW.get(v, 2.0)
            ax.plot(ck, curve, color=c, lw=lw, ls=ls, label=label, zorder=4 if cfm else 3)
            if cfm:
                ax.plot(ck, curve, color=INK, lw=lw + 1.4, alpha=0.3, zorder=2)
        ax.set_xlabel('SVGD iteration', color=INK, fontsize=10)
        ax.set_ylabel('Ergodic error (median over shapes)', color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=10.5, loc='left')
        if ymax is not None:
            ax.set_ylim(0, ymax)
    axL.legend(fontsize=7.5, loc='upper right', frameon=False, ncol=1)
    fig.suptitle(f'Convergence, linear scale -- {len(shapes)} holdout shapes, bold + dark outline = '
                'CFM-initialised variants (compare with the log-log view in overview_convergence.png)',
                color=INK, fontsize=11, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(_here, 'results', args.out_tag, 'convergence_linear.png')
    fig.savefig(out, dpi=130, facecolor='white')
    print(out)


if __name__ == '__main__':
    main()
