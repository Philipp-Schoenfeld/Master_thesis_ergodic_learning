"""One-figure overview of the smoothness comparison: coverage vs. smoothness trade-off.

Reads `results/<out_tag>/summary.csv` (written by plot_smoothness_comparison.py; one row
per shape x variant) and writes two versions next to it:

  overview.png              coverage/smoothness scatter + three dot plots (no convergence)
  overview_convergence.png  the same plus a convergence panel and convergence-sized markers
                            (needs convergence.npz, see compute_convergence.py)

Layout of both:

  left   scatter: ergodic error (x) vs. smoothness cost (y), both log scale; the lower
         left corner is best. Marker = solver family, colour = variant, cross = IQR over
         the shapes, dashed line = Pareto front (no other variant is better in both).
  right  three dot plots with the same variant order: ergodic error, smoothness, path
         length; dot = median over the shapes, bar = interquartile range.

Only in overview_convergence.png:
  left   (below the scatter) convergence: ergodic error vs. SVGD iteration, median over shapes.
         Dot on a curve = n90 (see compute_convergence.py).
  right  marker SIZE additionally encodes convergence: iterations needed to reach 90 % of the
         final improvement (n90, median over shapes; larger = slower). Needs convergence.npz
         from `compute_convergence.py`.

Usage:  python plot_smoothness_overview.py --out_tag smoothness_comparison_v3_20261011
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_smoothness_comparison import VARIANT_COLORS, VARIANT_LINESTYLES

_here = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = '#1A1A2E', '#555555', '#CCCCCC'

#: (variant, short label, marker); grouped by solver family, in display order
GROUPS = [
    ('CFM', [('cfm_only', 'CFM only', 'D')]),
    ('TSVEC-SVGD', [
        ('cfm_bspline_svgd', 'CFM init, B-spline', 'o'),
        ('linear_bspline_svgd', 'Linear init, B-spline', 'o'),
        ('linear_waypoint_svgd', 'Linear init, waypoints', 'o'),
        ('linear_waypoint_svgd_smooth', 'Linear init, waypoints + smoothness', 'o')]),
    ('Adam-SVGD', [
        ('cfm_adam_bspline', 'CFM init, B-spline', 's'),
        ('linear_adam_bspline', 'Linear init, B-spline', 's'),
        ('linear_adam_waypoint', 'Linear init, waypoints', 's'),
        ('linear_adam_waypoint_smooth', 'Linear init, waypoints + smoothness', 's')]),
    ('Sun flow matching (dynamics)', [
        ('linear_sun_pointmass', 'Linear init, point mass', '^'),
        ('linear_sun_jerk', 'Linear init, jerk-penalised', '^')]),
    ('Waypoint solve + B-spline projection', [
        ('linear_waypoint_svgd_proj', 'TSVEC, waypoints', 'X'),
        ('linear_waypoint_svgd_smooth_proj', 'TSVEC, waypoints + smoothness', 'X'),
        ('linear_adam_waypoint_proj', 'Adam, waypoints', 'X'),
        ('linear_adam_waypoint_smooth_proj', 'Adam, waypoints + smoothness', 'X'),
        ('linear_sun_pointmass_proj', 'Sun, point mass', 'X'),
        ('linear_sun_jerk_proj', 'Sun, jerk-penalised', 'X')]),
]
#: variants whose initialisation comes from the CFM network: drawn bold everywhere
CFM_VARIANTS = {'cfm_only', 'cfm_bspline_svgd', 'cfm_adam_bspline'}
METRICS = [('ergodic_final_mean', 'Ergodic error', True),
           ('smooth_final_mean', 'Smoothness cost', True),
           ('path_len_final_mean', 'Path length', False)]


def load(out_tag):
    path = os.path.join(_here, 'results', out_tag, 'summary.csv')
    vals = {}
    for r in csv.DictReader(open(path)):
        for key, _, _ in METRICS:
            vals.setdefault((r['variant'], key), []).append(float(r[key]))
    n_shapes = len({r['shape'] for r in csv.DictReader(open(path))})
    return vals, n_shapes


def stats(a):
    a = np.asarray(a)
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


def pareto(points):
    """Indices of points not dominated in (x, y), both to be minimised."""
    keep = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y)
                   for j, (x2, y2) in enumerate(points) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def style(ax):
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, alpha=0.2)


def render(out_tag, vals, n_shapes, with_conv):
    """Draw one overview version and return its path."""
    # only variants present in this run (older runs lack the projection group)
    flat = [(g, v, lab, mk) for g, items in GROUPS for v, lab, mk in items
            if (v, 'ergodic_final_mean') in vals]
    if with_conv:
        conv = np.load(os.path.join(_here, 'results', out_tag, 'convergence.npz'))
        c_variants = list(conv['variants'])
        flat = [f for f in flat if f[1] in c_variants]
        n90 = {v: np.nanmedian(conv['n90'][:, c_variants.index(v)]) if not np.isnan(
            conv['n90'][:, c_variants.index(v)]).all() else np.nan for v in c_variants}

    def marker_size(v):
        """Right-panel marker size. With convergence: from n90 on a log scale (10 -> 40,
        2000 -> 300; no iterations -> 18); without: constant, CFM variants slightly larger."""
        if not with_conv:
            return 105 if v in CFM_VARIANTS else 70
        if np.isnan(n90[v]):
            return 18
        t = (np.log10(max(n90[v], 10)) - 1.0) / (np.log10(2000) - 1.0)
        return 40 + 260 * float(np.clip(t, 0, 1))

    if with_conv:
        fig = plt.figure(figsize=(19, 11.5), facecolor='white')
        gs = fig.add_gridspec(2, 5, width_ratios=[3.0, 0.78, 1.25, 1.0, 1.0], height_ratios=[1.0, 0.72],
                              wspace=0.08, hspace=0.2, left=0.045, right=0.99, top=0.915, bottom=0.06)
    else:
        fig = plt.figure(figsize=(19, 8.2), facecolor='white')
        gs = fig.add_gridspec(1, 5, width_ratios=[3.0, 0.78, 1.25, 1.0, 1.0], wspace=0.08,
                              left=0.045, right=0.99, top=0.88, bottom=0.09)

    # ---- left: trade-off scatter ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    pts = []
    for g, v, lab, mk in flat:
        ex, ey = stats(vals[(v, 'ergodic_final_mean')]), stats(vals[(v, 'smooth_final_mean')])
        pts.append((ex[0], ey[0]))
        c = VARIANT_COLORS[v]
        ax.plot([ex[1], ex[2]], [ey[0], ey[0]], color=c, lw=1.2, alpha=0.45, zorder=2)
        ax.plot([ex[0], ex[0]], [ey[1], ey[2]], color=c, lw=1.2, alpha=0.45, zorder=2)
        cfm = v in CFM_VARIANTS
        ax.scatter([ex[0]], [ey[0]], s=150 if with_conv else (190 if cfm else 140), marker=mk, color=c,
                   edgecolor=INK if cfm else 'white', linewidth=2.4 if cfm else 1.5, zorder=4)
    front = pareto(pts)
    ax.step([pts[i][0] for i in front], [pts[i][1] for i in front], where='post',
            color=INK, lw=1.0, ls='--', alpha=0.5, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.006, 30)
    ax.set_xlabel('Ergodic error  (lower = better coverage)', color=INK, fontsize=10)
    ax.set_ylabel('Smoothness cost  (lower = smoother)', color=INK, fontsize=10)
    ax.set_title('Coverage vs. smoothness (median over shapes; cross = interquartile range)',
                 color=INK, fontsize=10.5, loc='left')
    ax.text(0.015, 0.02, 'best', transform=ax.transAxes, color=MUTED, fontsize=9, style='italic')
    ax.text(0.5, 0.012, 'dashed: Pareto front', transform=ax.transAxes, color=MUTED, fontsize=8, ha='center')
    offsets = {  # label offset in points, per variant, hand-tuned to avoid collisions
        'cfm_only': (10, 8), 'cfm_bspline_svgd': (11, -14), 'linear_bspline_svgd': (11, 9),
        'linear_waypoint_svgd': (11, 9), 'linear_waypoint_svgd_smooth': (11, 6),
        'cfm_adam_bspline': (-11, 7), 'linear_adam_bspline': (11, -13),
        'linear_adam_waypoint': (11, 6), 'linear_adam_waypoint_smooth': (-11, -12),
        'linear_sun_pointmass': (11, 6), 'linear_sun_jerk': (-11, 10)}
    prefix = {'CFM': '', 'TSVEC-SVGD': 'TSVEC: ', 'Adam-SVGD': 'Adam: ', 'Sun flow matching (dynamics)': 'Sun: ',
              'Waypoint solve + B-spline projection': 'Proj. '}
    for (g, v, lab, mk), (x, y) in zip(flat, pts):
        dx, dy = offsets.get(v, (8, 4))
        ax.annotate(prefix[g] + lab, (x, y), xytext=(dx, dy), textcoords='offset points',
                    fontsize=8.2 if v in CFM_VARIANTS else 7.6, color=INK,
                    fontweight='bold' if v in CFM_VARIANTS else 'normal',
                    ha='left' if dx > 0 else 'right', zorder=5)
    handles = [plt.Line2D([], [], marker=m, ls='', color='#888888', markersize=8, label=n)
               for n, m in (('CFM only', 'D'), ('TSVEC-SVGD', 'o'), ('Adam-SVGD', 's'),
                            ('Sun flow matching', '^'))]
    if any(g == 'Waypoint solve + B-spline projection' for g, _, _, _ in flat):
        handles.append(plt.Line2D([], [], marker='X', ls='', color='#888888', markersize=8,
                                  label='Waypoint solve + B-spline projection'))
    ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=False, title='Solver family',
              title_fontsize=8)

    # ---- left bottom: convergence curves ------------------------------------------
    if with_conv:
        cx = fig.add_subplot(gs[1, 0])
        style(cx)
        its = conv['its']
        for g, v, lab, mk in flat:
            curve = np.nanmedian(conv['curves'][:, c_variants.index(v)], axis=0)
            c, cfm = VARIANT_COLORS[v], v in CFM_VARIANTS
            if v == 'cfm_only':
                cx.axhline(curve[0], color=c, lw=3.0, ls=(0, (6, 3)), zorder=3)
                cx.text(1.1, curve[0] * 1.12, 'CFM only (no iterations)', color=INK, fontsize=8,
                        fontweight='bold')
                continue
            ls = VARIANT_LINESTYLES.get(v, '-')
            cx.plot(its[1:], curve[1:], color=c, lw=3.2 if cfm else 1.9, ls=ls,
                    zorder=4 if cfm else 3, label=prefix[g] + lab)
            if cfm:  # dark under-stroke so CFM curves read as bold
                cx.plot(its[1:], curve[1:], color=INK, lw=4.6, alpha=0.35, zorder=2)
            if not np.isnan(n90[v]):
                k = int(np.argmin(np.abs(its - n90[v])))
                cx.scatter([its[k]], [curve[k]], s=55, marker=mk, color=c,
                           edgecolor=INK if cfm else 'white', linewidth=1.6 if cfm else 1.0, zorder=5)
        cx.set_xscale('log')
        cx.set_yscale('log')
        cx.set_xlabel('SVGD iteration', color=INK, fontsize=10)
        cx.set_ylabel('Ergodic error (median over shapes)', color=INK, fontsize=10)
        cx.set_title('Convergence (marker = n90: 90 % of the final improvement reached)',
                     color=INK, fontsize=10.5, loc='left')
        cx.legend(fontsize=7, loc='lower left', frameon=False, ncol=2)

    # ---- right: dot plots in a shared variant order -----------------------------
    ypos, gaps, y = {}, [], 0.0
    for gi, (g, items) in enumerate(GROUPS):
        for v, lab, mk in items:
            ypos[v] = y
            y += 1.0
        gaps.append((g, ypos[items[0][0]], ypos[items[-1][0]]))
        y += 0.7
    first = None
    for k, (key, title, logx) in enumerate(METRICS):
        a = fig.add_subplot(gs[:, 2 + k], sharey=first)
        first = first or a
        style(a)
        stat = {f[1]: stats(vals[(f[1], key)]) for f in flat}
        best = min(stat, key=lambda v: stat[v][0])
        for g, v, lab, mk in flat:
            m, lo, hi = stat[v]
            c = VARIANT_COLORS[v]
            cfm = v in CFM_VARIANTS
            a.plot([lo, hi], [ypos[v]] * 2, color=c, lw=4.5 if cfm else 3.2, alpha=0.6 if cfm else 0.4,
                   solid_capstyle='round', zorder=2)
            a.scatter([m], [ypos[v]], s=marker_size(v), marker=mk, color=c,
                      edgecolor=INK if cfm else 'white', linewidth=2.0 if cfm else 1.2, zorder=4)
            if cfm:
                a.axhspan(ypos[v] - 0.5, ypos[v] + 0.5, color=c, alpha=0.07, zorder=0, lw=0)
            a.annotate(f'{m:.2f}' if m < 100 else f'{m:.0f}', (m, ypos[v]), xytext=(0, 5 + marker_size(v) ** 0.5 / 2),
                       textcoords='offset points', ha='center', fontsize=7,
                       color=INK, fontweight='bold' if v == best else 'normal')
        if logx:
            a.set_xscale('log')
        a.set_title(title, color=INK, fontsize=10, loc='left')
        a.set_ylim(y - 0.2, -1.1)
        for g, y0, y1 in gaps[1:]:
            a.axhline(y0 - 0.95, color=GRID, lw=0.8)
        if k == 0:
            a.set_yticks([ypos[v] for _, v, _, _ in flat])
            a.set_yticklabels([f'{lab}' for _, _, lab, _ in flat], fontsize=8, color=INK)
            for tl, (_, v, _, _) in zip(a.get_yticklabels(), flat):
                if v in CFM_VARIANTS:
                    tl.set_fontweight('bold')
                    tl.set_fontsize(8.8)
            for g, y0, y1 in gaps:
                a.text(0.015, y0 - 0.5, g, transform=a.get_yaxis_transform(), ha='left',
                       va='center', fontsize=8.5, color=MUTED, fontweight='bold')
        else:
            a.tick_params(labelleft=False)

    if with_conv:
        size_handles = [plt.scatter([], [], s=sz, color='#999999', edgecolor='white')
                        for sz in (40 + 260 * (np.log10(n) - 1.0) / (np.log10(2000) - 1.0)
                                   for n in (10, 100, 1000))]
        size_handles.insert(0, plt.scatter([], [], s=18, color='#999999', edgecolor='white'))
        fig.legend(size_handles, ['no iterations', 'n90 = 10', '100', '1000 iterations'],
                   loc='lower right', ncol=4, fontsize=8.5, frameon=False, title='Marker size (right panels): '
                   'iterations to 90 % of the improvement, larger = slower', title_fontsize=8.5,
                   bbox_to_anchor=(0.99, 0.0))
        subtitle = ('Bold label + dark outline = CFM-initialised variants; right-panel marker size = '
                    'convergence (iterations to 90 % of the improvement).')
        y_title = 0.985
    else:
        subtitle = 'Bold label + dark outline = CFM-initialised variants.'
        y_title = 0.975
    fig.suptitle(f'Smoothness comparison: 11 variants, {n_shapes} holdout shapes, final iteration. '
                 'Dot = median, bar = interquartile range over the shapes; bold number = best.\n' + subtitle,
                 color=INK, fontsize=11.5, x=0.045, ha='left', y=y_title)
    out = os.path.join(_here, 'results', out_tag, 'overview_convergence.png' if with_conv else 'overview.png')
    fig.savefig(out, dpi=130, facecolor='white')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    vals, n_shapes = load(args.out_tag)
    print(render(args.out_tag, vals, n_shapes, with_conv=False))
    if os.path.exists(os.path.join(_here, 'results', args.out_tag, 'convergence.npz')):
        print(render(args.out_tag, vals, n_shapes, with_conv=True))
    else:
        print('[overview] convergence.npz missing -> run compute_convergence.py for the second version')


if __name__ == '__main__':
    main()
