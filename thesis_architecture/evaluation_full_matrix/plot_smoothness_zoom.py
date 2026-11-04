"""Zoomed-in version of the coverage-vs-smoothness scatter from plot_smoothness_overview.py.

Same data (summary.csv), same styling; only the axis window is narrowed so the crowded cluster of
variants can be told apart. Variants outside the window are listed in a note instead of being drawn.

Usage:  python plot_smoothness_zoom.py --out_tag smoothness_comparison_v6_20261012
        python plot_smoothness_zoom.py --out_tag ... --xlim 0.02 0.3 --ylim 0.07 1.3
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from plot_smoothness_comparison import VARIANT_COLORS
from plot_smoothness_overview import CFM_VARIANTS, GROUPS, INK, MUTED, load, pareto, stats, style

_here = os.path.dirname(os.path.abspath(__file__))
PREFIX = {'CFM': '', 'TSVEC-SVGD': 'TSVEC: ', 'Adam-SVGD': 'Adam: ', 'Sun flow matching (dynamics)': 'Sun: '}
#: label offset in points per variant, hand-tuned for the default zoom window
OFFSETS = {
    'cfm_bspline_svgd': (14, -16), 'linear_bspline_svgd': (14, -3), 'linear_waypoint_svgd_smooth': (14, 6),
    'cfm_adam_bspline': (16, -20), 'linear_adam_bspline': (-14, -16), 'linear_adam_waypoint_smooth': (-14, 8),
    'linear_sun_pointmass': (14, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--xlim', type=float, nargs=2, default=(0.02, 0.3), help='ergodic-error window')
    ap.add_argument('--ylim', type=float, nargs=2, default=(0.07, 1.3), help='smoothness-cost window')
    args = ap.parse_args()
    vals, n_shapes = load(args.out_tag)

    flat = [(g, v, lab, mk) for g, items in GROUPS for v, lab, mk in items]
    st = {v: (stats(vals[(v, 'ergodic_final_mean')]), stats(vals[(v, 'smooth_final_mean')])) for _, v, _, _ in flat}
    pts = [(st[v][0][0], st[v][1][0]) for _, v, _, _ in flat]

    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor='white')
    style(ax)
    hidden = []
    for (g, v, lab, mk), (x, y) in zip(flat, pts):
        if not (args.xlim[0] <= x <= args.xlim[1] and args.ylim[0] <= y <= args.ylim[1]):
            hidden.append(f'{PREFIX[g]}{lab} ({x:.3g}, {y:.3g})')
            continue
        (_, xlo, xhi), (_, ylo, yhi) = st[v]
        c, cfm = VARIANT_COLORS[v], v in CFM_VARIANTS
        ax.plot([xlo, xhi], [y, y], color=c, lw=1.4, alpha=0.5, zorder=2)
        ax.plot([x, x], [ylo, yhi], color=c, lw=1.4, alpha=0.5, zorder=2)
        ax.scatter([x], [y], s=260 if cfm else 190, marker=mk, color=c,
                   edgecolor=INK if cfm else 'white', linewidth=2.4 if cfm else 1.5, zorder=4)
        dx, dy = OFFSETS.get(v, (14, 6))
        ax.annotate(f'{PREFIX[g]}{lab}  ({x:.3f}, {y:.2f})', (x, y), xytext=(dx, dy), textcoords='offset points',
                    fontsize=9.5 if cfm else 9, color=INK, fontweight='bold' if cfm else 'normal',
                    ha='left' if dx > 0 else 'right', zorder=5)
    front = pareto(pts)
    ax.step([pts[i][0] for i in front], [pts[i][1] for i in front], where='post',
            color=INK, lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.grid(True, which='minor', alpha=0.1)
    plain = FuncFormatter(lambda v, _: f'{v:g}')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(plain)
    ax.tick_params(which='minor', labelsize=7.5, colors=MUTED)
    ax.set_xlabel('Ergodic error  (lower = better coverage)', color=INK, fontsize=11)
    ax.set_ylabel('Smoothness cost  (lower = smoother)', color=INK, fontsize=11)
    ax.set_title(f'Coverage vs. smoothness, zoomed (median over {n_shapes} shapes; cross = interquartile range)',
                 color=INK, fontsize=11.5, loc='left')
    ax.text(0.012, 0.07, 'best', transform=ax.transAxes, color=MUTED, fontsize=10, style='italic')
    if hidden:
        ax.text(0.985, 0.015, 'outside window: ' + '; '.join(hidden), transform=ax.transAxes, color=MUTED,
                fontsize=7, ha='right', va='bottom', wrap=True)
    handles = [plt.Line2D([], [], marker=m, ls='', color='#888888', markersize=9, label=n)
               for n, m in (('CFM only', 'D'), ('TSVEC-SVGD', 'o'), ('Adam-SVGD', 's'), ('Sun flow matching', '^'))]
    ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=False, title='Solver family', title_fontsize=9)
    fig.tight_layout()
    out = os.path.join(_here, 'results', args.out_tag, 'overview_zoom.png')
    fig.savefig(out, dpi=150, facecolor='white')
    print(out)
    print('hidden:', hidden)


if __name__ == '__main__':
    main()
