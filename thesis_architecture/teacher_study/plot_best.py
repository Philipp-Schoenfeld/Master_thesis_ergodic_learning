r"""
plot_best.py -- one figure that shows the best teacher configuration of a study run
====================================================================================
Left: ergodicity (Hellinger) vs smoothness for every configuration after the B-spline projection
(mean over candidates and shapes), the Pareto front, the highlighted configuration, the current DB
teacher and the reference setup. Right: the highlighted configuration, the reference setup and the DB
teacher on a few holdout shapes. Within a configuration the candidate shown is chosen by the study's
rule (smoothest candidate with Hellinger <= the configuration's best on that shape + eps), not by eye.

    python plot_best.py --out_tag full_20261024 --best lm_K16_cont_o2_w15 --ref gui_lr8e-3_n3000
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.gridspec import GridSpec                                   # noqa: E402

import analyze_teacher_study as an                                         # noqa: E402
from analyze_teacher_study import C_DARK, C_GEN, C_GREY, C_GT, WHITE_INFERNO   # noqa: E402

REF_COLOR = '#EF6C00'


def config_means(d, stage):
    out = {}
    for cfg in np.unique(d['config']):
        m = (d['config'] == cfg) & (d['stage'] == stage) & (d['diverged'] == 0)
        if m.any():
            per_shape = [(d['hellinger'][m & (d['shape'] == s)].mean(), d['smooth'][m & (d['shape'] == s)].mean())
                         for s in np.unique(d['shape'][m])]
            out[cfg] = np.mean(per_shape, axis=0)
    return out


def pareto_front(pts):
    """pts: {name: (h, s)} -> names not dominated in (h, s), both minimised."""
    return [a for a, (h, s) in pts.items()
            if not any(h2 <= h and s2 <= s and (h2 < h or s2 < s) for b, (h2, s2) in pts.items() if b != a)]


def pick(d, shape, cfg, stage, eps):
    m = np.flatnonzero((d['shape'] == shape) & (d['config'] == cfg) & (d['stage'] == stage) & (d['diverged'] == 0))
    ok = m[d['hellinger'][m] <= d['hellinger'][m].min() + eps]
    return ok[np.argmin(d['smooth'][ok])]


def draw(ax, sd, X, color, lw, title, sub):
    ax.imshow(sd.density, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
              alpha=0.55, aspect='equal', zorder=0)
    ax.plot(X[:, 0], X[:, 1], color=color, lw=lw, alpha=0.95, zorder=3)
    ax.plot(*sd.x0, 'o', color=C_DARK, ms=4, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#cccccc')
    if title:
        ax.set_title(title, fontsize=9, color=C_DARK)
    ax.set_xlabel(sub, fontsize=8, color=C_GREY)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--best', required=True, help='configuration to highlight')
    ap.add_argument('--ref', default=None, help='reference configuration (e.g. the GUI Adam setup)')
    ap.add_argument('--shapes', default='A,korean_5,organic_0,rand_gmm_complex_10')
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--stage', default='bs25')
    args = ap.parse_args()

    import pipeline as pl
    run_dir = os.path.join(_here, 'results', args.out_tag)
    d = an.apply_boundary_filter(an.load(os.path.join(run_dir, 'metrics.csv')), 0.0)
    shapes = [s.strip() for s in args.shapes.split(',')]
    sds = {s.name: s for s in pl.load_shapes(shapes)}

    means = config_means(d, args.stage)
    front = set(pareto_front({k: v for k, v in means.items() if k != 'teacher_db'}))

    fig = plt.figure(figsize=(17, 8.2), facecolor='white')
    gs = GridSpec(len(shapes), 5, figure=fig, width_ratios=[2.6, 0.12, 1, 1, 1], wspace=0.12, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    an.style_axes(ax)
    fx = sorted((means[c][0], means[c][1]) for c in front)
    ax.step([p[0] for p in fx], [p[1] for p in fx], where='post', color=C_GREY, lw=1, ls='--', alpha=0.6, zorder=1)
    for cfg, (h, s) in means.items():
        if cfg == 'teacher_db':
            continue
        fam = d['solver'][d['config'] == cfg][0]
        ax.scatter(h, s, s=70 if cfg in front else 28, c=an.SOLVER_COLOR.get(fam, C_GREY),
                   alpha=0.95 if cfg in front else 0.35, edgecolors=C_DARK if cfg in front else 'none',
                   linewidths=0.6, zorder=2)
        if cfg in front and cfg not in (args.best, args.ref):
            ax.annotate(cfg, (h, s), xytext=(5, 4), textcoords='offset points', fontsize=7, color=C_GREY)
    ax.scatter(*means['teacher_db'], marker='*', s=330, c=C_GT, edgecolors=C_DARK, zorder=5)
    ax.annotate('current DB teacher', means['teacher_db'], xytext=(8, 6), textcoords='offset points',
                fontsize=9, color=C_GT, fontweight='bold')
    hb, sb = means[args.best]
    ax.scatter(hb, sb, marker='o', s=260, facecolors='none', edgecolors=C_GEN, linewidths=3, zorder=6)
    ax.annotate(f'best: {args.best}\nH {hb:.3f}, smooth {sb:.3f}', (hb, sb), xytext=(-10, -46),
                textcoords='offset points', fontsize=9, color='#007A33', fontweight='bold', ha='right')
    if args.ref:
        hr, sr = means[args.ref]
        ax.scatter(hr, sr, marker='D', s=110, c=REF_COLOR, edgecolors=C_DARK, zorder=6)
        ax.annotate(f'reference: {args.ref}\nH {hr:.3f}, smooth {sr:.3f}', (hr, sr), xytext=(8, 8),
                    textcoords='offset points', fontsize=9, color=REF_COLOR, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlabel('Hellinger distance to target (lower = more ergodic)   [mean over 25 shapes and 30 candidates]')
    ax.set_ylabel('acceleration energy (log scale, lower = smoother)')
    ax.set_title(f'All configurations after projection onto {args.stage} (25 control points); '
                 f'big markers = Pareto front', color=C_DARK)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in an.SOLVER_COLOR.items()
               if k != 'db']
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper right', labelcolor=C_DARK)

    cols = [('teacher_db', 'DB teacher (blue)', C_GT, 2.5),
            (args.ref, 'reference setup (orange)', REF_COLOR, 2.4),
            (args.best, 'best (green)', C_GEN, 2.4)]
    for r, shape in enumerate(shapes):
        sd = sds[shape]
        for c, (cfg, label, color, lw) in enumerate(cols):
            a = fig.add_subplot(gs[r, 2 + c])
            if cfg is None:
                a.set_axis_off()
                continue
            if cfg == 'teacher_db':
                j = np.flatnonzero((d['shape'] == shape) & (d['config'] == cfg) & (d['stage'] == args.stage))[0]
            else:
                j = pick(d, shape, cfg, args.stage, args.eps)
            f = np.load(os.path.join(run_dir, 'finals', f'{shape}__{cfg}.npz'))
            X = an._render_stage(f[args.stage][int(d['cand'][j])], args.stage)
            sub = f'H {d["hellinger"][j]:.3f} | smooth {d["smooth"][j]:.2f} | acc {d["acc_rms"][j]:.0f}'
            draw(a, sd, X, color, lw, label if r == 0 else '', sub)
            if c == 0:
                a.set_ylabel(shape, fontsize=10, color=C_DARK)
    fig.suptitle('Best teacher configuration: more ergodic than the reference at the same smoothness, '
                 'and much smoother than the DB teacher', color=C_DARK, fontsize=13)
    out = os.path.join(run_dir, 'plots', 'best_overview.png')
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    print(f'[plot_best] front: {sorted(front)}')
    print(f'[plot_best] wrote {out}')


if __name__ == '__main__':
    main()
